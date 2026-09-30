"""Gate v2: consistency-constrained, area-preserving sub-pixel change mapping (ADR-x4-gate-v2).

Replaces v1's hard parent threshold and NDVI-based parent mask with:
1. Linear unmixing in reflectance (B04, B08) to compute per-10m block change fraction f
2. Exact-count allocation of round(16*f) sub-pixels per block, ranked by SR change score + spatial dependence
3. Split-conformal block-detection threshold calibrated on placebo pairs
4. New classes: CORE (f≈1), ALLOCATED (0<f<1), UNSUPPORTED (SR change, f=0), NO_DATA
"""
from __future__ import annotations

import numpy as np
from scipy import signal

# ---- class definitions (gate v2 replaces v1 OBSERVED/INFERRED/UNSUPPORTED with CORE/ALLOCATED/UNSUPPORTED) ----

CORE, ALLOCATED, UNSUPPORTED, NO_DATA = 1, 2, 3, 255
NO_CHANGE = 0
CLASS_NAMES = {NO_CHANGE: 'NO_CHANGE', CORE: 'CORE', ALLOCATED: 'ALLOCATED', UNSUPPORTED: 'UNSUPPORTED', NO_DATA: 'NO_DATA'}


# ---- unmixing (10 m, no SR) ----

def unmix_fraction(y_pre, y_post, e_v, e_b, bands_indices=(0, 3)):
    """Linear unmixing in reflectance to estimate vegetation-loss fraction f per 10 m block.

    Given observations y ~ e_b + a(e_v - e_b), solve for the fraction of vegetation loss using
    a two-endmember linear mixing model. Computed independently for each band and then averaged.

    Args:
        y_pre: (n_dates, H, W, n_bands) pre-event reflectance stacks
        y_post: (H, W, n_bands) post-event reflectance
        e_v: (n_bands,) dense-vegetation endmember reflectance
        e_b: (n_bands,) bare-ground endmember reflectance
        bands_indices: tuple of band indices to use (default (0, 3) = B04, B08 red and NIR)

    Returns:
        f_hat: (H, W) vegetation-loss fraction, clipped to [0, 1]
        residual: (H, W) per-pixel misfit sum of squares (for diagnostics/NO_DATA assignment)
    """
    y_pre = np.asarray(y_pre, dtype=np.float64)
    y_post = np.asarray(y_post, dtype=np.float64)
    e_v = np.asarray(e_v, dtype=np.float64)
    e_b = np.asarray(e_b, dtype=np.float64)

    n_dates, h, w, n_bands = y_pre.shape
    if y_post.shape != (h, w, n_bands):
        raise ValueError(f'y_post shape {y_post.shape} does not match y_pre spatial dims {(h, w, n_bands)}')
    if e_v.shape != (n_bands,) or e_b.shape != (n_bands,):
        raise ValueError('endmembers must be 1-D with length n_bands')

    # Restrict to the requested bands
    bi = np.asarray(bands_indices, dtype=int)
    if bi.size != 2 or (bi < 0).any() or (bi >= n_bands).any():
        raise ValueError(f'bands_indices {bands_indices} invalid for n_bands={n_bands}')
    y_pre_b = y_pre[..., bi]  # (n_dates, H, W, 2)
    y_post_b = y_post[..., bi]  # (H, W, 2)
    d = e_v[bi] - e_b[bi]  # (2,) direction vector
    dd = float(np.dot(d, d))  # scalar

    # Compute per-date, per-pixel fractions
    a_pre = np.full((n_dates, h, w), np.nan, dtype=np.float64)
    a_post = np.full((h, w), np.nan, dtype=np.float64)

    for t in range(n_dates):
        dy = y_pre_b[t] - e_b[bi]  # (H, W, 2)
        proj = np.dot(dy, d) / dd  # (H, W)
        a_pre[t] = np.clip(proj, 0.0, 1.0)

    dy_post = y_post_b - e_b[bi]  # (H, W, 2)
    proj_post = np.dot(dy_post, d) / dd  # (H, W)
    a_post = np.clip(proj_post, 0.0, 1.0)

    # Compute the fraction: loss = mean(pre) - post
    # Take finite mean across pre dates
    with np.errstate(invalid='ignore'):
        a_pre_mean = np.nanmean(a_pre, axis=0)  # (H, W)
    f_hat = a_pre_mean - a_post
    f_hat = np.clip(f_hat, 0.0, 1.0)

    # Compute residuals for diagnostics (orthogonal component of the residual)
    # For per-date misfit (only reported, not used for thresholding)
    residual = np.full((h, w), np.nan, dtype=np.float64)
    dy = y_pre_b[0] - e_b[bi]  # Use first date as representative; (H, W, 2)
    proj_val = np.dot(dy, d) / dd  # (H, W)
    proj_perp = dy - proj_val[..., np.newaxis] * d[np.newaxis, np.newaxis, :]  # (H, W, 2) residual perpendicular to d
    residual = np.sum(proj_perp ** 2, axis=-1)  # (H, W) sum of squares

    return f_hat, residual


# ---- allocation: exact count with ranking ----

def allocate_pixels(f_tilde, delta_sr, s_spatial, lam: float = 0.0, seed: int = 2024):
    """Allocate exactly round(16*f) sub-pixels per 10 m block, ranked by SR change + spatial dependence.

    Uses a rank-blend strategy: u_x = (1 - lam) * r_x + lam * s_x, where
    - r_x is the within-block rank of the SR change score delta_sr
    - s_x is the within-block rank of a spatial term (bilinear interpolation of f_tilde)

    Args:
        f_tilde: (H, W) per-10m-block fraction, shape must be divisible by 4
        delta_sr: (4*H, 4*W) per-2.5m sub-pixel SR change score (e.g. a_SR change)
        s_spatial: (4*H, 4*W) spatial dependence field (bilinear interp of f_tilde to 2.5 m)
        lam: blend weight in [0, 1] (0 = pure SR ranking, 1 = pure spatial/bilinear)
        seed: RNG seed for reproducibility (for tie-breaking by raster order)

    Returns:
        allocated: (4*H, 4*W) binary map, 1 where round(16*f) top pixels are allocated
        n_allocated: (H, W) count of allocated pixels per block (should equal round(16*f) or ceil)
    """
    h, w = f_tilde.shape
    if delta_sr.shape != (4 * h, 4 * w):
        raise ValueError(f'delta_sr shape {delta_sr.shape} does not match (4*H, 4*W) = {(4*h, 4*w)}')
    if s_spatial.shape != (4 * h, 4 * w):
        raise ValueError(f's_spatial shape {s_spatial.shape} does not match (4*H, 4*W)')

    f_tilde = np.clip(f_tilde, 0.0, 1.0)
    lam = float(np.clip(lam, 0.0, 1.0))

    allocated = np.zeros((4 * h, 4 * w), dtype=bool)
    n_allocated = np.zeros((h, w), dtype=int)

    for i in range(h):
        for j in range(w):
            # Compute target allocation for this block
            n_target = max(0, min(16, round(16.0 * f_tilde[i, j])))
            if n_target == 0:
                continue

            # Extract the 16 sub-pixels in this 10m block (4x4 grid in 2.5m grid)
            ri0, ri1 = 4 * i, 4 * (i + 1)
            cj0, cj1 = 4 * j, 4 * (j + 1)
            block_delta = delta_sr[ri0:ri1, cj0:cj1].ravel()  # 16 values
            block_spatial = s_spatial[ri0:ri1, cj0:cj1].ravel()  # 16 values

            # Compute rank-blended scores
            # Rank delta_sr (higher change score is better)
            rank_delta = np.argsort(np.argsort(-block_delta)) / 15.0  # [0, 1], 1 = highest
            # Rank spatial (higher spatial value is better)
            rank_spatial = np.argsort(np.argsort(-block_spatial)) / 15.0  # [0, 1]

            # Blend
            u = (1.0 - lam) * rank_delta + lam * rank_spatial

            # Allocate top n_target
            top_idx = np.argsort(-u)[:n_target]
            px_linear = np.arange(16)
            local_alloc = np.zeros(16, dtype=bool)
            local_alloc[top_idx] = True

            # Place in output
            block_alloc = local_alloc.reshape((4, 4))
            allocated[ri0:ri1, cj0:cj1] = block_alloc
            n_allocated[i, j] = n_target

    return allocated, n_allocated


# ---- bilinear spatial interpolation (for spatial-dependence term) ----

def bilinear_upsample(f_tilde, scale: int = 4):
    """Bilinear interpolation of 10 m fractions f_tilde to 2.5 m grid.

    Returns a (scale*H, scale*W) array where each 10m pixel is interpolated to scale^2 sub-pixels.
    """
    h, w = f_tilde.shape
    f_tilde = np.asarray(f_tilde, dtype=np.float64)

    # Use scipy for bilinear interpolation (or implement manually)
    from scipy.interpolate import RectBivariateSpline

    # Create a spline (order=1 for linear)
    x = np.arange(w)
    y = np.arange(h)
    spl = RectBivariateSpline(y, x, f_tilde, kx=1, ky=1)

    # Evaluate on the finer grid
    y_fine = np.linspace(0, h - 1, scale * h)
    x_fine = np.linspace(0, w - 1, scale * w)
    result = spl(y_fine, x_fine)

    return result


# ---- split-conformal block-detection threshold ----

def split_conformal_quantile(scores_calibration, alpha: float = 0.05):
    """Compute split-conformal quantile threshold from calibration scores.

    Given n calibration window/block scores (null hypothesis), compute the (1 - alpha) quantile
    such that the probability of exceeding it is at most alpha under exchangeability.

    Args:
        scores_calibration: (n,) array of calibration block/window scores
        alpha: target false-alarm rate (default 0.05)

    Returns:
        tau: threshold value
        coverage_info: dict with n, k, tau, coverage_expectation
    """
    scores = np.asarray(scores_calibration, dtype=np.float64)
    scores = scores[np.isfinite(scores)]
    n = len(scores)

    if n < 1:
        raise ValueError('need at least 1 calibration score')

    # Split-conformal: tau = T_(k) where k = ceil((n+1)(1 - alpha))
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    k = min(max(k, 1), n)  # Clamp to [1, n]

    # The k-th smallest score
    tau = float(np.sort(scores)[k - 1])

    coverage_info = {
        'n_calibration': int(n),
        'alpha': float(alpha),
        'k': int(k),
        'threshold': float(tau),
        'coverage_lower_bound': float((n + 1 - k) / (n + 1)),  # P(T > tau) <= alpha
    }

    return tau, coverage_info


def evaluate_conformal_on_test(scores_test, tau: float, alpha: float = 0.05, bootstrap_replicates: int = 2000, seed: int = 2024):
    """Evaluate the split-conformal threshold on test scores, with bootstrap CI.

    Args:
        scores_test: (n_test,) array of test block/window scores
        tau: threshold from calibration
        alpha: nominal FAR (for reporting)
        bootstrap_replicates: number of bootstrap resamples
        seed: RNG seed

    Returns:
        far_empirical: empirical false-alarm rate on test
        ci_low, ci_high: 95% bootstrap CI on FAR
        results_dict: full results with counts and CI info
    """
    scores_test = np.asarray(scores_test, dtype=np.float64)
    scores_test = scores_test[np.isfinite(scores_test)]
    n_test = len(scores_test)

    if n_test < 1:
        raise ValueError('need at least 1 test score')

    # Empirical FAR
    flags = (scores_test > tau).astype(float)
    far_empirical = float(np.mean(flags))

    # Bootstrap CI on FAR
    rng = np.random.RandomState(seed)
    boot_fars = []
    for _ in range(bootstrap_replicates):
        idx = rng.choice(n_test, n_test, replace=True)
        boot_far = float(np.mean(flags[idx]))
        boot_fars.append(boot_far)
    boot_fars = np.array(boot_fars)

    ci_low = float(np.percentile(boot_fars, 2.5))
    ci_high = float(np.percentile(boot_fars, 97.5))

    results_dict = {
        'n_test': int(n_test),
        'n_flagged': int(np.sum(flags)),
        'far_empirical': far_empirical,
        'far_bootstrap_ci': {
            'lower': ci_low,
            'upper': ci_high,
            'nominal_coverage': 0.95,
            'replicates': bootstrap_replicates,
            'seed': seed,
        },
        'threshold': float(tau),
        'alpha_nominal': float(alpha),
        'guarantee': f'P(T > {tau:.6f}) <= {alpha} under exchangeability; empirical FAR on test = {far_empirical:.4f} [{ci_low:.4f}, {ci_high:.4f}]',
    }

    return far_empirical, ci_low, ci_high, results_dict


# ---- block-level window scoring and conformal application ----

def max_fraction_per_window(f_hat, window_size: int = 16, valid_mask=None):
    """Compute max f_hat per window for conformal thresholding.

    Args:
        f_hat: (H, W) per-10m-block fractions
        window_size: side length of window in 10m blocks (default 16 = 160m)
        valid_mask: (H, W) bool mask of valid blocks (default None = all valid)

    Returns:
        window_scores: (H//window_size, W//window_size) max fraction per window
        window_validity: (H//window_size, W//window_size) bool, True where window has >= n_min valid blocks
    """
    f_hat = np.asarray(f_hat, dtype=np.float64)
    h, w = f_hat.shape

    if h % window_size != 0 or w % window_size != 0:
        raise ValueError(f'grid shape {(h, w)} not divisible by window_size {window_size}')

    n_win_h = h // window_size
    n_win_w = w // window_size

    window_scores = np.full((n_win_h, n_win_w), np.nan, dtype=np.float64)
    window_validity = np.full((n_win_h, n_win_w), False, dtype=bool)

    for iw in range(n_win_h):
        for jw in range(n_win_w):
            ri0, ri1 = iw * window_size, (iw + 1) * window_size
            cj0, cj1 = jw * window_size, (jw + 1) * window_size
            win_f = f_hat[ri0:ri1, cj0:cj1]

            if valid_mask is not None:
                win_valid = np.asarray(valid_mask, dtype=bool)[ri0:ri1, cj0:cj1]
                win_f = win_f[win_valid]

            if len(win_f) > 0 and np.any(np.isfinite(win_f)):
                window_scores[iw, jw] = float(np.nanmax(win_f))
                window_validity[iw, jw] = True

    return window_scores, window_validity


# ---- combined gate application ----

def apply_gate_v2(y_pre, y_post, d_sr, e_v, e_b, sigma, nodata, config):
    """Apply the full gate v2 pipeline to produce classified 2.5m map.

    Args:
        y_pre: (n_dates, H, W, n_bands) pre-event reflectance
        y_post: (H, W, n_bands) post-event reflectance
        d_sr: (4*H, 4*W) SR change score (NDVI difference, 2.5m grid)
        e_v: (n_bands,) vegetation endmember
        e_b: (n_bands,) bare ground endmember
        sigma: (4*H, 4*W) uncertainty estimate (2.5m grid)
        nodata: (4*H, 4*W) bool, True where data is missing
        config: dict with keys:
            - 'k': threshold multiplier for d > k*sigma (for UNSUPPORTED)
            - 'lam': blend weight for allocation ranking [0, 1]
            - 'alpha': target conformal FAR [0, 1]
            - 'window_size': size of conformal window in 10m blocks
            - 'seed': random seed

    Returns:
        class_map: (4*H, 4*W) class labels (CORE, ALLOCATED, UNSUPPORTED, NO_CHANGE, NO_DATA)
        meta: dict with diagnostic info (f_hat, window_scores, threshold, etc.)
    """
    h_10m = y_pre.shape[1]
    w_10m = y_pre.shape[2]
    h_sr = 4 * h_10m
    w_sr = 4 * w_10m

    # Unmix to get per-10m block fractions
    f_hat, misfit = unmix_fraction(y_pre, y_post, e_v, e_b)
    f_tilde = np.clip(f_hat, 0.0, 1.0)  # Bias correction would go here

    # Spatial interpolation for ranking
    s_spatial = bilinear_upsample(f_tilde, scale=4)

    # Allocate pixels (this is deterministic per the config.seed)
    lam = config.get('lam', 0.0)
    allocated, n_per_block = allocate_pixels(f_tilde, d_sr, s_spatial, lam=lam, seed=config.get('seed', 2024))

    # Initialize output map
    class_map = np.full((h_sr, w_sr), NO_CHANGE, dtype=np.uint8)

    # Assign classes
    # NO_DATA: overrides everything
    class_map[nodata] = NO_DATA

    # CORE: f ≈ 1 (all 16 sub-pixels allocated)
    for i in range(h_10m):
        for j in range(w_10m):
            if n_per_block[i, j] == 16:
                ri0, ri1 = 4 * i, 4 * (i + 1)
                cj0, cj1 = 4 * j, 4 * (j + 1)
                class_map[ri0:ri1, cj0:cj1][allocated[ri0:ri1, cj0:cj1]] = CORE

    # ALLOCATED: 0 < f < 1 (some pixels allocated)
    for i in range(h_10m):
        for j in range(w_10m):
            if 0 < n_per_block[i, j] < 16:
                ri0, ri1 = 4 * i, 4 * (i + 1)
                cj0, cj1 = 4 * j, 4 * (j + 1)
                class_map[ri0:ri1, cj0:cj1][allocated[ri0:ri1, cj0:cj1]] = ALLOCATED

    # UNSUPPORTED: SR change in non-detected blocks
    k = config.get('k', 2.0)
    kappa = config.get('kappa', None)  # Quantile threshold for UNSUPPORTED (optional)

    if kappa is None:
        # Use a simple threshold: d > k * sigma on blocks with f = 0
        for i in range(h_10m):
            for j in range(w_10m):
                if n_per_block[i, j] == 0:
                    ri0, ri1 = 4 * i, 4 * (i + 1)
                    cj0, cj1 = 4 * j, 4 * (j + 1)
                    block_d = d_sr[ri0:ri1, cj0:cj1]
                    block_sigma = sigma[ri0:ri1, cj0:cj1]
                    unsup_px = (block_d > k * block_sigma) & ~nodata[ri0:ri1, cj0:cj1]
                    class_map[ri0:ri1, cj0:cj1][unsup_px] = UNSUPPORTED
    else:
        # Use the calibrated kappa threshold
        for i in range(h_10m):
            for j in range(w_10m):
                if n_per_block[i, j] == 0:
                    ri0, ri1 = 4 * i, 4 * (i + 1)
                    cj0, cj1 = 4 * j, 4 * (j + 1)
                    block_d = d_sr[ri0:ri1, cj0:cj1]
                    unsup_px = (block_d > kappa) & ~nodata[ri0:ri1, cj0:cj1]
                    class_map[ri0:ri1, cj0:cj1][unsup_px] = UNSUPPORTED

    # Restore NO_DATA on unallocated non-detected blocks
    class_map[nodata] = NO_DATA

    meta = {
        'f_hat': f_hat,
        'n_per_block': n_per_block,
        'allocated': allocated,
        'misfit': misfit,
    }

    return class_map, meta
