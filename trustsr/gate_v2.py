"""Gate v2 (F2 rebuild): consistency-constrained, area-preserving sub-pixel change mapping.

This module replaces the gate-v2 implementation that experiments/x4_calibration.py and
experiments/x9_v2_production.py CALLED. Those two callers were invalidated by the A10 audit
(experiments/results/INVALIDATED.md); the ADR for this rebuild is docs/adr-f2-gate-v2.md. Every
change below is pinned to the B-ID it fixes:

  B6  Detection is calibrated on PLACEBO (null-event) window scores only -- see
      `split_conformal_quantile`, fed by F1's calibration (even-tile) split. x4 calibrated on
      `pre_mean - post_mean` of the REAL event pair with `valid = ones`, so its "guarantee" was
      computed on data containing the landslide. Nothing in this module ever reads a post-event date;
      the caller (experiments/f2_gate_v2.py) obtains tau exclusively from pre-event placebo folds.
  B7  `block_sum_deviation` compares two INDEPENDENTLY derived quantities: the number of 2.5 m pixels
      labelled CORE/ALLOCATED inside a block's 4x4 footprint of the OUTPUT class map, versus
      round(16*f) recomputed from the unmixing fraction. It never receives `n_per_block` from the
      allocator. x4's `check_block_sum_consistency` compared h*w*100 m^2 with itself.
  B8  This module carries no config hash at all; the caller hashes configs/fix.yaml itself. The
      literal string 'pending-a3-output' appears nowhere.
  B9  Allocation is ranked by a TRUE 2.5 m SR change score supplied by the caller (`d_sr`), which must
      vary inside a 10 m block. `assert_sr_score_is_not_replicated` raises if `d_sr` is constant within
      every 4x4 block -- i.e. if somebody passes `np.repeat(np.repeat(d_10m, 4), 4)` as x9 did.
  B10 `tau` is a REQUIRED positional argument of `apply_gate_v2`. There is no default, and a non-finite
      tau raises. Changing tau changes the output map (tests/test_gate_v2.py::test_tau_mutation_*).
      x9 loaded tau_win from x4.json and never passed it in, so every block with f>0 was ALLOCATED.
  B11 NO_DATA is built from the SCL validity mask on EVERY date (`nodata_from_scl`), propagated
      nearest-neighbour to 2.5 m. x9 derived NO_DATA from NaN pixels alone (512 px vs v1's 205,956).
  B12 Endmembers are ESTIMATED FROM DATA by `estimate_endmembers`, from stable/high-NDVI and low-NDVI
      pre-event pixels outside the 1500 m plausibility disks and the landslide footprint. They are
      never hard-coded and never come from a label; the function has no access to any label, to any
      class map or to any post-event array (tests/test_gate_v2.py::TestEndmembersNeverFromLabels).
"""
from __future__ import annotations

import inspect

import numpy as np
from scipy import signal

# ---- class definitions (gate v2 replaces v1 OBSERVED/INFERRED/UNSUPPORTED with CORE/ALLOCATED/UNSUPPORTED) ----

CORE, ALLOCATED, UNSUPPORTED, NO_DATA = 1, 2, 3, 255
NO_CHANGE = 0
CLASS_NAMES = {NO_CHANGE: 'NO_CHANGE', CORE: 'CORE', ALLOCATED: 'ALLOCATED', UNSUPPORTED: 'UNSUPPORTED', NO_DATA: 'NO_DATA'}

SCALE = 4                 # 10 m -> 2.5 m
SUBPX = SCALE * SCALE     # 16 sub-pixels per 10 m block


def flagged_v2(class_map) -> np.ndarray:
    """configs/fix.yaml units.flagged_definition.v2 = CORE union ALLOCATED. UNSUPPORTED is NEVER flagged."""
    cm = np.asarray(class_map)
    return (cm == CORE) | (cm == ALLOCATED)


# ---------------------------------------------------------------- endmembers (B12) ----------------------------------------------------------------

def robust_sd(x, axis=None) -> np.ndarray:
    """1.4826 * median absolute deviation: the robust SD used for the endmember sensitivity check."""
    x = np.asarray(x, dtype=np.float64)
    med = np.nanmedian(x, axis=axis, keepdims=True)
    return 1.4826 * np.nanmedian(np.abs(x - med), axis=axis)


def estimate_endmembers(refl_pre, valid_pre, ndvi_pre, exclude, *,
                        ndvi_high: float = 0.80, ndvi_low: float = 0.20, stable_std_max: float = 0.05,
                        min_count: int = 100):
    """Estimate the two mixing endmembers FROM PRE-EVENT DATA ONLY (B12's direct fix).

    There is deliberately no `labels`, no `class_map`, no `footprint_is_change` and no post-event
    argument in this signature. `exclude` is an EXCLUSION geometry (the 1500 m plausibility disks union
    the landslide footprint, reused verbatim from configs/wayanad_evidence.yaml via
    trustsr.placebo_v2.exclusion_mask_10m): pixels inside it are REMOVED from the estimation sample, so
    the event area can never contribute an endmember. Removing pixels is not labelling them; the test
    `test_endmembers_ignore_everything_inside_the_exclusion` proves the excluded pixels' VALUES have
    zero influence on the result.

    Selection rules (fixed before the run; see docs/adr-f2-gate-v2.md):
      e_v : valid on every pre date, outside `exclude`, mean pre NDVI >= `ndvi_high` AND per-pixel NDVI
            standard deviation across the pre dates <= `stable_std_max` ("stable, high-NDVI"). Both
            numbers are the published E6 stable-pixel statistics already cited in
            configs/wayanad_evidence.yaml `change` (date-to-date NDVI std <= 0.05; dense-canopy mean
            NDVI ~0.81); neither was chosen after seeing an F2 result.
      e_b : valid on every pre date, outside `exclude`, NDVI <= `ndvi_low` on EVERY pre date
            (configs/fix.yaml f2_gate_v2.endmembers.e_b).

    Args:
        refl_pre: (T, H, W, B) pre-event reflectance, model band order [B04, B03, B02, B08]
        valid_pre: (T, H, W) bool, SCL validity per date
        ndvi_pre: (T, H, W) NDVI per date
        exclude: (H, W) bool, True = excluded from the estimation sample
        min_count: below this many pixels an endmember is refused (raises), never silently widened

    Returns:
        dict with 'e_v', 'e_b' (each (B,) float64), per-band robust SDs, pixel counts and the rules used.
    """
    refl_pre = np.asarray(refl_pre, dtype=np.float64)
    valid_pre = np.asarray(valid_pre, dtype=bool)
    ndvi_pre = np.asarray(ndvi_pre, dtype=np.float64)
    exclude = np.asarray(exclude, dtype=bool)
    if refl_pre.ndim != 4:
        raise ValueError(f'refl_pre must be (T,H,W,B), got {refl_pre.shape}')
    t, h, w, nb = refl_pre.shape
    if valid_pre.shape != (t, h, w) or ndvi_pre.shape != (t, h, w) or exclude.shape != (h, w):
        raise ValueError('refl_pre / valid_pre / ndvi_pre / exclude shapes disagree')
    if t < 2:
        raise ValueError('need >= 2 pre dates to measure temporal stability')

    valid_all = valid_pre.all(axis=0) & ~exclude & np.isfinite(ndvi_pre).all(axis=0) \
        & np.isfinite(refl_pre).all(axis=(0, 3))
    ndvi_mean = ndvi_pre.mean(axis=0)
    ndvi_std = ndvi_pre.std(axis=0, ddof=1)

    veg = valid_all & (ndvi_mean >= ndvi_high) & (ndvi_std <= stable_std_max)
    bare = valid_all & (ndvi_pre <= ndvi_low).all(axis=0)

    refl_mean = refl_pre.mean(axis=0)                       # (H, W, B) temporal mean reflectance
    out = {}
    for name, sel in (('e_v', veg), ('e_b', bare)):
        n = int(sel.sum())
        if n < min_count:
            raise ValueError(f'{name}: only {n} qualifying pixels (< min_count={min_count}); '
                             f'refusing to estimate an endmember from so few pixels rather than widening the rule')
        sample = refl_mean[sel]                              # (n, B)
        out[name] = np.median(sample, axis=0)
        out[f'{name}_robust_sd'] = robust_sd(sample, axis=0)
        out[f'{name}_count'] = n

    out['rules'] = {
        'e_v': f'valid on all {t} pre dates, outside exclusion, mean pre NDVI >= {ndvi_high}, '
               f'per-pixel NDVI std across pre dates <= {stable_std_max}',
        'e_b': f'valid on all {t} pre dates, outside exclusion, NDVI <= {ndvi_low} on EVERY pre date',
        'statistic': 'per-band median of the temporal-mean reflectance over the selected pixels; '
                     'robust SD = 1.4826 * MAD',
        'never_from_labels': 'no label, class map or post-event array is an argument of estimate_endmembers',
    }
    out['n_pre_dates'] = int(t)
    out['n_valid_outside_exclusion'] = int(valid_all.sum())
    return out


def endmember_sensitivity(fn, em: dict, bands_indices=(0, 3)) -> dict:
    """Recompute `fn(e_v, e_b)` with each endmember perturbed by +/- 1 robust SD (configs/fix.yaml
    f2_gate_v2.endmembers.record). `fn` must return a scalar summary of the result (e.g. mean f, or the
    flagged-pixel count). Returns the baseline and the four perturbed values with their deltas.
    """
    e_v, e_b = np.asarray(em['e_v'], float), np.asarray(em['e_b'], float)
    sv, sb = np.asarray(em['e_v_robust_sd'], float), np.asarray(em['e_b_robust_sd'], float)
    base = float(fn(e_v, e_b))
    out = {'baseline': base, 'perturbations': {}}
    for name, ev, eb in (('e_v_plus_1sd', e_v + sv, e_b), ('e_v_minus_1sd', e_v - sv, e_b),
                         ('e_b_plus_1sd', e_v, e_b + sb), ('e_b_minus_1sd', e_v, e_b - sb)):
        val = float(fn(ev, eb))
        out['perturbations'][name] = {'value': val, 'delta': val - base,
                                      'relative_delta': (val - base) / base if base else None}
    deltas = [abs(v['delta']) for v in out['perturbations'].values()]
    out['max_abs_delta'] = max(deltas)
    out['max_relative_delta'] = max(abs(v['relative_delta']) for v in out['perturbations'].values()) if base else None
    out['bands_indices'] = list(bands_indices)
    return out


# ---------------------------------------------------------------- unmixing (10 m, no SR) ----------------------------------------------------------------

def unmix_fraction(y_pre, y_post, e_v, e_b, bands_indices=(0, 3)):
    """Two-endmember linear unmixing in reflectance -> per-10 m-block vegetation-loss fraction f.

    UNCHANGED from the pre-F2 module apart from documentation: this function was already correct. It
    takes the endmembers as arguments (it never hard-codes them -- that was x9's bug, B12, in the
    CALLER), it reads no labels, and it works in reflectance on (B04, B08) as configs/fix.yaml
    f2_gate_v2.unmixing requires.

    y ~ e_b + a (e_v - e_b), so a = (y - e_b) . d / |d|^2 with d = e_v - e_b restricted to the two
    bands. f = mean_t a_pre(t) - a_post, clipped to [0, 1] (one-sided: greening is not change).

    Args:
        y_pre: (n_dates, H, W, n_bands) pre-event reflectance stacks
        y_post: (H, W, n_bands) post-event (or held-out placebo) reflectance
        e_v, e_b: (n_bands,) endmembers from `estimate_endmembers`
        bands_indices: band indices to unmix on; default (0, 3) = B04 (red), B08 (NIR)

    Returns:
        f_hat: (H, W) fraction in [0, 1]
        residual: (H, W) misfit perpendicular to the mixing line on the FIRST pre date (diagnostic only;
                  never used for thresholding or for NO_DATA)
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

    bi = np.asarray(bands_indices, dtype=int)
    if bi.size != 2 or (bi < 0).any() or (bi >= n_bands).any():
        raise ValueError(f'bands_indices {bands_indices} invalid for n_bands={n_bands}')
    y_pre_b = y_pre[..., bi]
    y_post_b = y_post[..., bi]
    d = e_v[bi] - e_b[bi]
    dd = float(np.dot(d, d))
    if dd <= 0:
        raise ValueError('e_v and e_b coincide on the unmixing bands; the mixing line is degenerate')

    a_pre = np.clip(np.dot(y_pre_b - e_b[bi], d) / dd, 0.0, 1.0)      # (n_dates, H, W)
    a_post = np.clip(np.dot(y_post_b - e_b[bi], d) / dd, 0.0, 1.0)    # (H, W)

    with np.errstate(invalid='ignore'):
        a_pre_mean = np.nanmean(a_pre, axis=0)
    f_hat = np.clip(a_pre_mean - a_post, 0.0, 1.0)

    dy = y_pre_b[0] - e_b[bi]
    proj_val = np.dot(dy, d) / dd
    proj_perp = dy - proj_val[..., None] * d[None, None, :]
    residual = np.sum(proj_perp ** 2, axis=-1)
    return f_hat, residual


def band_sigma_from_pre(y_pre, valid=None, bands_indices=(0, 3), floor_quantile: float = 0.5):
    """Per-pixel, per-band reflectance noise sigma, measured as the temporal standard deviation across
    the pre dates (ddof=1), floored at the scene `floor_quantile` quantile of that same statistic.

    The floor is a variance floor, not a tuned parameter: without it a pixel whose 2-3 pre dates happen
    to agree to ~1e-6 produces a score of order 1e5 and a single such pixel would set tau. (F1's exported
    d/sigma_v1 window scores run up to 255 for exactly this reason.) The floor is stated here and
    reported in experiments/results/f2.json; it is fixed at the median and was not varied.

    Returns (H, W, 2) sigma for the two unmixing bands.
    """
    y = np.asarray(y_pre, dtype=np.float64)[..., np.asarray(bands_indices, int)]     # (T, H, W, 2)
    if y.shape[0] < 2:
        raise ValueError('need >= 2 pre dates to measure a temporal sigma')
    sig = y.std(axis=0, ddof=1)                                                       # (H, W, 2)
    sel = np.isfinite(sig).all(axis=-1)
    if valid is not None:
        sel &= np.asarray(valid, bool)
    if not sel.any():
        raise ValueError('no valid pixels to set the sigma floor')
    floor = np.quantile(sig[sel], floor_quantile, axis=0)                             # (2,)
    return np.maximum(sig, floor[None, None, :]), floor


def fraction_sigma(sigma_bands, e_v, e_b, n_pre: int, bands_indices=(0, 3)) -> np.ndarray:
    """sigma_f by first-order error propagation through the unmixing (the documented choice; the
    bootstrap alternative was not used -- see docs/adr-f2-gate-v2.md).

    a = (y - e_b).d / |d|^2 with d = e_v - e_b on the two bands, so for independent per-band noise
        Var(a) = (sigma_R^2 d_R^2 + sigma_N^2 d_N^2) / |d|^4.
    f = mean_t a_pre(t) - a_post, and the pre dates are independent of the post date, so
        Var(f) = Var(a) * (1/n_pre + 1).

    Endmember uncertainty is deliberately NOT folded in here; it is reported separately as the
    +/- 1 robust SD sensitivity check (`endmember_sensitivity`), because a scene-wide endmember shift is
    a systematic, not an independent per-pixel, error.
    """
    sigma_bands = np.asarray(sigma_bands, dtype=np.float64)
    bi = np.asarray(bands_indices, int)
    d = np.asarray(e_v, float)[bi] - np.asarray(e_b, float)[bi]
    dd = float(np.dot(d, d))
    if dd <= 0:
        raise ValueError('degenerate mixing line')
    if n_pre < 1:
        raise ValueError('n_pre must be >= 1')
    var_a = (sigma_bands[..., 0] ** 2 * d[0] ** 2 + sigma_bands[..., 1] ** 2 * d[1] ** 2) / dd ** 2
    return np.sqrt(var_a * (1.0 / n_pre + 1.0))


# ---------------------------------------------------------------- windows and detection (B6, B10) ----------------------------------------------------------------

def window_max_score(s_block, window_size: int = 16, valid_block=None):
    """Per-window max of the per-block score (the same window-max convention F1 used for its exported
    calibration scores). Windows are `window_size` x `window_size` 10 m blocks (16 -> 160 m).

    Returns (window_scores, window_valid); a window with no valid block is NaN / False.
    """
    s = np.asarray(s_block, dtype=np.float64)
    h, w = s.shape
    if h % window_size or w % window_size:
        raise ValueError(f'block grid {(h, w)} is not a whole number of {window_size}-block windows')
    if valid_block is not None:
        s = np.where(np.asarray(valid_block, bool), s, np.nan)
    tiles = s.reshape(h // window_size, window_size, w // window_size, window_size)
    with np.errstate(invalid='ignore'):
        finite = np.isfinite(tiles)
        scores = np.where(finite.any(axis=(1, 3)),
                          np.nanmax(np.where(finite, tiles, -np.inf), axis=(1, 3)), np.nan)
    return scores, finite.any(axis=(1, 3))


# kept for the old public name used by tests/x4-era code; delegates to window_max_score
def max_fraction_per_window(f_hat, window_size: int = 16, valid_mask=None):
    """Per-window max of `f_hat` (kept for backward compatibility; detection uses `window_max_score`
    on s = f / sigma_f, not on f)."""
    return window_max_score(f_hat, window_size, valid_mask)


def split_conformal_quantile(scores_calibration, alpha: float = 0.05):
    """tau = the ceil((n+1)(1-alpha))-th order statistic of the CALIBRATION scores.

    B6's fix is in WHAT is passed here: F1's placebo (null-event) calibration-split window scores, in
    which every positive is a false alarm by construction. x4 passed scores computed from the real event
    pair, so its "P(T > tau) <= alpha" was a statement about data containing the landslide.
    """
    scores = np.asarray(scores_calibration, dtype=np.float64)
    scores = scores[np.isfinite(scores)]
    n = len(scores)
    if n < 1:
        raise ValueError('need at least 1 calibration score')
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        raise ValueError(f'ceil((n+1)(1-alpha)) = {k} > n = {n}: the calibration split is too small to '
                         f'support alpha = {alpha} without extrapolation; refusing to clamp silently')
    tau = float(np.sort(scores)[k - 1])
    info = {'n_calibration': int(n), 'alpha': float(alpha), 'k': int(k), 'tau': tau,
            'exceedance_upper_bound': float((n + 1 - k) / (n + 1)),   # P(new score > tau) <= this, under exchangeability
            'formula': 'tau = T_(k), k = ceil((n+1)(1-alpha)) (split conformal, one-sided)'}
    return tau, info


def evaluate_conformal_on_test(scores_test, tau: float, alpha: float = 0.05, bootstrap_replicates: int = 2000, seed: int = 2024):
    """Empirical exceedance rate of `tau` on held-out scores, with a percentile bootstrap CI."""
    scores_test = np.asarray(scores_test, dtype=np.float64)
    scores_test = scores_test[np.isfinite(scores_test)]
    n_test = len(scores_test)
    if n_test < 1:
        raise ValueError('need at least 1 test score')
    flags = (scores_test > tau).astype(float)
    far_empirical = float(np.mean(flags))
    rng = np.random.RandomState(seed)
    boot = np.array([float(np.mean(flags[rng.choice(n_test, n_test, replace=True)]))
                     for _ in range(bootstrap_replicates)])
    ci_low, ci_high = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    results = {
        'n_test': int(n_test), 'n_flagged': int(flags.sum()), 'far_empirical': far_empirical,
        'far_bootstrap_ci': {'lower': ci_low, 'upper': ci_high, 'nominal_coverage': 0.95,
                             'replicates': bootstrap_replicates, 'seed': seed},
        'threshold': float(tau), 'alpha_nominal': float(alpha),
        'guarantee': f'P(T > {tau:.6f}) <= {alpha} under exchangeability of windows (assumed, not proven); '
                     f'empirical exceedance on test = {far_empirical:.4f} [{ci_low:.4f}, {ci_high:.4f}]',
    }
    return far_empirical, ci_low, ci_high, results


# ---------------------------------------------------------------- allocation (B9) ----------------------------------------------------------------

def assert_sr_score_is_not_replicated(d_sr, scale: int = SCALE, name: str = 'd_sr'):
    """B9's guard. Raises if `d_sr` is constant inside EVERY scale x scale block, which is exactly what
    `np.repeat(np.repeat(d_10m, 4, 0), 4, 1)` (x9's allocation ranking) produces. A genuine 2.5 m SR
    change field varies within a block; a nearest-neighbour replicate of a 10 m field never does.

    Returns the fraction of blocks with non-zero within-block variation.
    """
    a = np.asarray(d_sr, dtype=np.float64)
    h, w = a.shape[-2:]
    if h % scale or w % scale:
        raise ValueError(f'{name} shape {a.shape} is not a whole number of {scale}x{scale} blocks')
    blocks = a.reshape(h // scale, scale, w // scale, scale)
    with np.errstate(invalid='ignore'):
        spread = np.nanmax(blocks, axis=(1, 3)) - np.nanmin(blocks, axis=(1, 3))
    varying = np.isfinite(spread) & (spread > 0)
    frac = float(varying.mean())
    if frac == 0.0:
        raise ValueError(
            f'{name} is constant inside every {scale}x{scale} block: this is a nearest-neighbour '
            f'replicate of a 10 m field, not a 2.5 m SR change score (B9). Refusing to allocate.')
    return frac


def _within_block(a, scale: int = SCALE):
    """(4H, 4W) -> (H, W, 16), the 16 sub-pixels of each 10 m block in raster order."""
    a = np.asarray(a)
    h4, w4 = a.shape
    return a.reshape(h4 // scale, scale, w4 // scale, scale).transpose(0, 2, 1, 3).reshape(
        h4 // scale, w4 // scale, scale * scale)


def _from_within_block(b, scale: int = SCALE):
    """(H, W, 16) -> (4H, 4W)."""
    h, w, _ = b.shape
    return b.reshape(h, w, scale, scale).transpose(0, 2, 1, 3).reshape(h * scale, w * scale)


def _descending_rank01(x):
    """Within-block rank in [0, 1], 1 = largest value. Ties are broken by raster order (stable argsort)."""
    order = np.argsort(-x, axis=-1, kind='stable')
    pos = np.empty_like(order)
    np.put_along_axis(pos, order, np.broadcast_to(np.arange(x.shape[-1]), x.shape).copy(), axis=-1)
    return 1.0 - pos / (x.shape[-1] - 1.0)


def neighbour_agreement(alloc_bool) -> np.ndarray:
    """Neighbour agreement, defined precisely: for every 2.5 m pixel, the FRACTION OF ITS 8 NEIGHBOURS
    (8-connectivity on the 2.5 m grid, zero-padded outside the array) that are allocated in the
    provisional pass-1 allocation. Range [0, 1]. The pixel itself is excluded, so the term is genuinely
    a statement about the neighbourhood and not a restatement of the pixel's own score.
    """
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.float64)
    return signal.convolve2d(np.asarray(alloc_bool, dtype=np.float64), kernel, mode='same', boundary='fill') / 8.0


def allocate_pixels(f_tilde, d_sr, lam: float = 0.0, scale: int = SCALE, check_sr: bool = True):
    """Allocate EXACTLY round(16*f) sub-pixels per 10 m block, ranked by the true 2.5 m SR change score
    plus lam * neighbour agreement (configs/fix.yaml f2_gate_v2.allocation).

    Two passes, so the neighbour-agreement term is well defined and non-circular:
      pass 1: rank by `d_sr` alone -> provisional allocation A0
      pass 2: u = (1 - lam) * rank(d_sr) + lam * neighbour_agreement(A0), re-rank, allocate.
    Both ranks are within-block ranks in [0, 1], so lam <= 0.5 keeps the SR term dominant.

    Args:
        f_tilde: (H, W) per-block fraction in [0, 1]
        d_sr:    (4H, 4W) per-2.5 m SR change score. MUST vary within blocks (B9 guard).
        lam:     neighbour-agreement weight in [0, 1]
        check_sr: run `assert_sr_score_is_not_replicated` (disable only in fixtures that deliberately
                  test the blocky case)

    Returns (allocated, n_per_block).
    """
    f_tilde = np.clip(np.asarray(f_tilde, dtype=np.float64), 0.0, 1.0)
    d_sr = np.asarray(d_sr, dtype=np.float64)
    h, w = f_tilde.shape
    if d_sr.shape != (scale * h, scale * w):
        raise ValueError(f'd_sr shape {d_sr.shape} does not match ({scale}H, {scale}W) = {(scale * h, scale * w)}')
    lam = float(lam)
    if not 0.0 <= lam <= 1.0:
        raise ValueError('lam must be in [0, 1]')
    if check_sr:
        assert_sr_score_is_not_replicated(d_sr, scale)

    n_target = np.clip(np.round(scale * scale * f_tilde).astype(int), 0, scale * scale)   # (H, W)

    dblk = _within_block(np.nan_to_num(d_sr, nan=-np.inf), scale)                         # (H, W, 16)
    r_sr = _descending_rank01(dblk)

    def _alloc(u):
        order = np.argsort(-u, axis=-1, kind='stable')
        pos = np.empty_like(order)
        np.put_along_axis(pos, order, np.broadcast_to(np.arange(u.shape[-1]), u.shape).copy(), axis=-1)
        return pos < n_target[..., None]

    a0 = _alloc(r_sr)                                                                      # pass 1
    if lam == 0.0:
        allocated = _from_within_block(a0, scale)
        return allocated, n_target

    na = neighbour_agreement(_from_within_block(a0, scale))
    u = (1.0 - lam) * r_sr + lam * _within_block(na, scale)
    allocated = _from_within_block(_alloc(u), scale)
    return allocated, n_target


def bilinear_upsample(f_tilde, scale: int = SCALE):
    """Bilinear interpolation of the 10 m fractions to 2.5 m. NOT used for allocation ranking any more
    (that is `d_sr`); kept because the SR-vs-bilinear divergence regression test needs a bilinear
    comparator to rank against.
    """
    from scipy.interpolate import RectBivariateSpline
    f_tilde = np.asarray(f_tilde, dtype=np.float64)
    h, w = f_tilde.shape
    spl = RectBivariateSpline(np.arange(h), np.arange(w), f_tilde, kx=1, ky=1)
    return spl(np.linspace(0, h - 1, scale * h), np.linspace(0, w - 1, scale * w))


# ---------------------------------------------------------------- NO_DATA (B11) ----------------------------------------------------------------

def nodata_from_scl(valid_per_date_10m, scale: int = SCALE, extra_invalid_10m=None) -> np.ndarray:
    """NO_DATA at 2.5 m, built from the SCL validity mask on EVERY date (B11's direct fix).

    `valid_per_date_10m` is (T, H, W) bool from experiments.wayanad_evidence.data.scl_valid (the v1
    logic, reused verbatim, not reinvented): a pixel is usable only if it is not cloud, cloud-shadow/dark
    or invalid in SCL on that date, and was actually acquired. A 10 m pixel is NO_DATA unless it is valid
    on ALL dates; the mask is then propagated NEAREST-NEIGHBOUR to the 2.5 m grid, so one masked 10 m
    pixel yields exactly 16 NO_DATA sub-pixels REGARDLESS of whether its numeric value is finite.

    x9 instead used `np.isnan(d_sr) | np.isnan(sigma_sr)` and reported 512 NO_DATA px where v1 reported
    205,956: cloud- and shadow-contaminated pixels were silently mapped as change.
    """
    v = np.asarray(valid_per_date_10m, dtype=bool)
    if v.ndim != 3:
        raise ValueError(f'valid_per_date_10m must be (T,H,W); got {v.shape}')
    bad_10m = ~v.all(axis=0)
    if extra_invalid_10m is not None:
        bad_10m = bad_10m | np.asarray(extra_invalid_10m, bool)
    return np.repeat(np.repeat(bad_10m, scale, axis=0), scale, axis=1)


# ---------------------------------------------------------------- block-sum test (B7) ----------------------------------------------------------------

def block_sum_deviation(class_map, f_effective, nodata=None, scale: int = SCALE) -> dict:
    """The REAL per-block sum test (B7's direct fix).

    Two independently derived quantities:
      observed = number of 2.5 m pixels labelled CORE or ALLOCATED inside the block's scale x scale
                 footprint of the OUTPUT class map (counted from the output array itself);
      expected = round(16 * f) recomputed here from the unmixing fraction.
    The allocator's own `n_per_block` is deliberately NOT an argument, so nothing is compared with
    itself. (x4's check_block_sum_consistency compared h*w*100 m^2 against h*w*100 m^2.)

    Returns max/mean absolute deviation over all blocks and over detected blocks (expected > 0), plus
    the block indices of the worst offenders.
    """
    cm = np.asarray(class_map)
    f = np.clip(np.asarray(f_effective, dtype=np.float64), 0.0, 1.0)
    h, w = f.shape
    if cm.shape != (scale * h, scale * w):
        raise ValueError(f'class_map {cm.shape} does not match {scale}x the fraction grid {f.shape}')
    observed = _within_block(flagged_v2(cm).astype(np.int64), scale).sum(axis=-1)
    expected = np.round(scale * scale * f).astype(np.int64)
    dev = observed - expected
    detected = expected > 0
    nodata_blocks = (_within_block(np.asarray(nodata, bool).astype(np.int64), scale).sum(axis=-1) > 0) \
        if nodata is not None else np.zeros((h, w), bool)
    worst = np.unravel_index(int(np.argmax(np.abs(dev))), dev.shape)
    return {
        'max_abs_deviation_all_blocks': int(np.abs(dev).max()),
        'max_abs_deviation_detected_blocks': int(np.abs(dev[detected]).max()) if detected.any() else 0,
        'mean_abs_deviation_all_blocks': float(np.abs(dev).mean()),
        'n_blocks': int(dev.size), 'n_detected_blocks': int(detected.sum()),
        'n_blocks_touching_nodata': int(nodata_blocks.sum()),
        'n_blocks_with_nonzero_deviation': int((dev != 0).sum()),
        'worst_block_rc': [int(worst[0]), int(worst[1])],
        'worst_block_observed': int(observed[worst]), 'worst_block_expected': int(expected[worst]),
        'pass': bool(np.abs(dev).max() == 0),
        'definition': 'observed = count of CORE|ALLOCATED px in the block 4x4 footprint of the OUTPUT class '
                      'map; expected = round(16*f) recomputed from the unmixing fraction. Two independent '
                      'paths; the allocator\'s n_per_block is not used (B7).',
    }


# ---------------------------------------------------------------- full gate ----------------------------------------------------------------

def apply_gate_v2(y_pre, y_post, d_sr, e_v, e_b, sigma_bands, nodata, tau, *,
                  lam: float = 0.0, window_size: int = 16, k_unsupported: float = 2.0,
                  sigma_sr=None, bands_indices=(0, 3), scale: int = SCALE, check_sr: bool = True):
    """Full gate v2: unmix -> score -> conformal window detection -> exact-count SR-ranked allocation.

    `tau` is a REQUIRED positional argument (B10). There is no default and no silent fallback: a None or
    non-finite tau raises, so the detection step can never be skipped the way x9 skipped it.

    Args:
        y_pre:  (T, H, W, B) pre-event reflectance on the 10 m grid
        y_post: (H, W, B) post-event / held-out reflectance
        d_sr:   (4H, 4W) TRUE 2.5 m SR change score (per-pixel SR NDVI difference from the tiled
                E1/E8 dihedral SR stack). Must vary within 10 m blocks (B9 guard).
        e_v, e_b: endmembers from `estimate_endmembers` (never hard-coded, B12)
        sigma_bands: (H, W, 2) per-band reflectance sigma from `band_sigma_from_pre`
        nodata: (4H, 4W) bool from `nodata_from_scl` (B11)
        tau:    conformal window threshold from `split_conformal_quantile` on F1's CALIBRATION placebo
                window scores (B6)
        sigma_sr: optional (4H, 4W) 2.5 m sigma; if given, UNSUPPORTED = d_sr > k_unsupported * sigma_sr
                  inside blocks that allocated nothing. UNSUPPORTED is never counted as flagged.

    Returns (class_map, meta).
    """
    if tau is None or not np.isfinite(tau):
        raise ValueError('apply_gate_v2 requires an explicit, finite tau (B10): detection may never be '
                         'silently disabled by a missing threshold')
    tau = float(tau)

    y_pre = np.asarray(y_pre, dtype=np.float64)
    t, h, w, _ = y_pre.shape
    nodata = np.asarray(nodata, dtype=bool)
    if nodata.shape != (scale * h, scale * w):
        raise ValueError(f'nodata {nodata.shape} must be on the 2.5 m grid {(scale * h, scale * w)}')

    f_hat, misfit = unmix_fraction(y_pre, y_post, e_v, e_b, bands_indices)
    sigma_f = fraction_sigma(sigma_bands, e_v, e_b, n_pre=t, bands_indices=bands_indices)
    with np.errstate(invalid='ignore', divide='ignore'):
        s_block = np.where(sigma_f > 0, f_hat / sigma_f, np.nan)

    # a 10 m block is invalid if ANY of its 16 sub-pixels is NO_DATA -> it never scores and never allocates
    block_invalid = _within_block(nodata.astype(np.int64), scale).sum(axis=-1) > 0
    block_invalid |= ~np.isfinite(s_block)
    valid_block = ~block_invalid

    win_scores, win_valid = window_max_score(s_block, window_size, valid_block)
    with np.errstate(invalid='ignore'):
        win_detected = win_valid & (win_scores > tau)
    detected_block = np.repeat(np.repeat(win_detected, window_size, axis=0), window_size, axis=1) & valid_block

    f_eff = np.where(detected_block, np.nan_to_num(f_hat, nan=0.0), 0.0)
    allocated, n_per_block = allocate_pixels(f_eff, d_sr, lam=lam, scale=scale, check_sr=check_sr)

    class_map = np.full((scale * h, scale * w), NO_CHANGE, dtype=np.uint8)
    core_block = np.repeat(np.repeat(n_per_block == scale * scale, scale, 0), scale, 1)
    alloc_block = np.repeat(np.repeat((n_per_block > 0) & (n_per_block < scale * scale), scale, 0), scale, 1)
    class_map[allocated & core_block] = CORE
    class_map[allocated & alloc_block] = ALLOCATED

    if sigma_sr is not None:
        empty_block = np.repeat(np.repeat(n_per_block == 0, scale, 0), scale, 1)
        with np.errstate(invalid='ignore'):
            unsup = empty_block & (np.asarray(d_sr, float) > k_unsupported * np.asarray(sigma_sr, float))
        class_map[unsup & ~nodata] = UNSUPPORTED

    class_map[nodata] = NO_DATA        # NO_DATA overrides everything (B11)

    meta = {
        'f_hat': f_hat, 'f_effective': f_eff, 'sigma_f': sigma_f, 's_block': s_block,
        'valid_block': valid_block, 'window_scores': win_scores, 'window_valid': win_valid,
        'window_detected': win_detected, 'detected_block': detected_block,
        'n_per_block': n_per_block, 'allocated': allocated, 'misfit': misfit,
        'tau': tau, 'lam': float(lam), 'window_size_10m_px': int(window_size),
        'n_windows_valid': int(win_valid.sum()), 'n_windows_detected': int(win_detected.sum()),
        'block_sum': block_sum_deviation(class_map, f_eff, nodata, scale),
        'class_counts': {name: int((class_map == code).sum()) for code, name in CLASS_NAMES.items()},
    }
    return class_map, meta


# ---------------------------------------------------------------- self-audit helper ----------------------------------------------------------------

def gate_v2_signature_audit() -> dict:
    """Machine-checkable facts asserted by tests: tau is a required positional parameter of
    apply_gate_v2, and estimate_endmembers takes no label-like argument."""
    ag = inspect.signature(apply_gate_v2).parameters
    em = inspect.signature(estimate_endmembers).parameters
    banned = ('label', 'labels', 'truth', 'target', 'post', 'y_post', 'class_map', 'footprint_is_change')
    return {
        'apply_gate_v2_params': list(ag),
        'tau_is_required_positional': ('tau' in ag and ag['tau'].default is inspect.Parameter.empty
                                       and ag['tau'].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD),
        'estimate_endmembers_params': list(em),
        'estimate_endmembers_has_no_label_argument': not any(p in em for p in banned),
    }
