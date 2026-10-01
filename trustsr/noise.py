"""Noise model for the before/after gate (A5): a predictive SD for ONE new observation against the pre-event mean.

Why this exists. Gate v1 used sigma = sqrt(sigma_pre^2 + sigma_post^2) where sigma_pre is the plain std of all
(pre date x dihedral run) samples and sigma_post is the std of the eight dihedral runs of the single post date. That is
neither the spread of a *new date* around the pre mean nor is it inflated for the uncertainty of the pre mean itself:

    v1 (n_pre dates, R runs each):   sigma_pre^2 ~ [R (n-1) / (nR - 1)] s2_dates      (n = 3, R = 8: 16/23 = 0.70 s2_dates)
    predictive variance of d:        s2_dates (1 + 1/n)                                (n = 3: 1.33 s2_dates)

so v1 sigma is a fixed ~28 % below the predictive SD at n_pre = 3 (the new SD is ~38 % above v1), independent of the data;
sigma_post (dihedral only, ~0.002 NDVI) adds nothing. See docs/adr-x5-noise-model.md.

New model (configs/exceptional.yaml `a5.formula`):

    sigma^2 = s2_dates * (1 + 1/n_pre) + s2_dihedral_pre / n_pre + s2_dihedral_post

* s2_dates: variance of the per-date means (each the mean of that date's dihedral runs) ACROSS dates, never of pooled runs.
* s2_dihedral_*: mean over dates of the within-date dihedral variance (pre), dihedral variance of the post date (post).
* With n_pre = 3 the per-pixel s2_dates has 2 degrees of freedom, so `estimate_s2_dates` shrinks it (empirical Bayes toward a
  stratum or a local window). The shrinkage returns E[sigma^2 | data], the posterior mean, because that is the variance of the
  predictive distribution; the estimator is unbiased for the variance, not for the SD (Jensen).
* `effective_k` is the multiplier k_eff with P(|d| <= k_eff * sigma) = nominal (0.9545 for a Gaussian 2 sigma).

Everything is numpy; no raster or torch dependency. Arrays are (T, H, W) date stacks or (T, N) for tests.
"""
from __future__ import annotations

import math

import numpy as np

from experiments.wayanad_evidence.stats import Welford

NOMINAL_2SIGMA = math.erf(2.0 / math.sqrt(2.0))          # 0.9545 for a Gaussian, the pre-registered nominal coverage
METHODS = ('raw', 'window', 'eb_stratified', 'eb_window')


def nominal_coverage(k: float) -> float:
    """P(|Z| <= k) for a standard normal."""
    return math.erf(k / math.sqrt(2.0))


# ---- special functions (no scipy in the pinned environment) -----------------------------------------------------------

def trigamma(x):
    """psi'(x) for x > 0: upward recurrence to x >= 10, then the asymptotic series (error < 1e-13)."""
    x = np.asarray(x, dtype=np.float64)
    if np.any(x <= 0):
        raise ValueError('trigamma is implemented for x > 0 only')
    acc = np.zeros_like(x)
    y = x.copy()
    for _ in range(10):
        low = y < 10.0
        acc = acc + np.where(low, 1.0 / (y * y), 0.0)
        y = np.where(low, y + 1.0, y)
    inv = 1.0 / y
    inv2 = inv * inv
    series = inv + 0.5 * inv2 + inv * inv2 * (1 / 6 - inv2 * (1 / 30 - inv2 * (1 / 42 - inv2 * (1 / 30))))
    out = acc + series
    return float(out) if out.ndim == 0 else out


def trigamma_inverse(y: float, lo: float = 1e-6, hi: float = 1e9) -> float:
    """x with trigamma(x) = y, by bisection in log x (trigamma is strictly decreasing). Clamped to [lo, hi]."""
    if not y > 0:
        raise ValueError('trigamma_inverse needs y > 0')
    if y >= trigamma(lo):
        return lo
    if y <= trigamma(hi):
        return hi
    a, b = math.log(lo), math.log(hi)
    for _ in range(200):
        m = 0.5 * (a + b)
        if trigamma(math.exp(m)) > y:
            a = m
        else:
            b = m
    return math.exp(0.5 * (a + b))


# ---- streaming per-date dihedral moments ------------------------------------------------------------------------------

class DateMoments:
    """One Welford accumulator per date: the dihedral runs of a date are accumulated separately, never pooled across dates.

    `update(date, x)` takes one run's NDVI; NaN runs are skipped per pixel. Memory is 3 float64 arrays per date.
    """

    def __init__(self, shape, dates):
        self.dates = [str(d) for d in dates]
        if len(set(self.dates)) != len(self.dates):
            raise ValueError('dates must be unique')
        self.shape = tuple(shape)
        self.acc = {d: Welford(self.shape) for d in self.dates}

    def update(self, date, x, where=None):
        self.acc[str(date)].update(x, where)

    def update_at(self, date, region, x, where=None):
        self.acc[str(date)].update_at(region, x, where)

    def count_stack(self):
        return np.stack([self.acc[d].count for d in self.dates])

    def mean_stack(self, min_runs: int = 1):
        """(T, ...) per-date mean over that date's runs; NaN where fewer than `min_runs` valid runs."""
        return np.stack([self.acc[d].valid_mean(min_runs) for d in self.dates])

    def var_stack(self, ddof: int = 1):
        """(T, ...) per-date dihedral variance; NaN where a date has <= ddof valid runs."""
        return np.stack([self.acc[d].variance(ddof) for d in self.dates])


def pool_moments(means, variances, counts, ddof: int = 1):
    """Exact merge of per-date moments into the moments of ALL runs pooled (Chan et al.): (mean, variance).

    This reproduces the v1 sigma_pre (a single Welford over every pre date x run). A date with zero runs is ignored;
    with one run it contributes no within-date sum of squares.
    """
    means, variances, counts = (np.asarray(a, dtype=np.float64) for a in (means, variances, counts))
    ok = counts > 0
    n = np.where(ok, counts, 0.0)
    total = n.sum(axis=0)
    m = np.where(ok, means, 0.0)
    mean = np.divide((n * m).sum(axis=0), total, out=np.full(total.shape, np.nan), where=total > 0)
    within = np.where(counts > 1, (counts - 1.0) * np.nan_to_num(variances), 0.0).sum(axis=0)
    between = (n * np.where(ok, (m - np.where(total > 0, mean, 0.0)) ** 2, 0.0)).sum(axis=0)
    out = np.full(total.shape, np.nan)
    np.divide(within + between, total - ddof, out=out, where=total > ddof)
    return mean, out


def sigma_v1(pre_std, post_std):
    """The OLD (gate v1) sigma, kept for comparison: sqrt(sigma_pre_pooled^2 + sigma_post^2)."""
    return np.sqrt(np.asarray(pre_std, dtype=np.float64) ** 2 + np.asarray(post_std, dtype=np.float64) ** 2)


# ---- variance across dates and shrinkage --------------------------------------------------------------------------------

def date_variance(date_means):
    """Sample variance (ddof 1) across the date axis of (T, ...) per-date means, NaN-aware per pixel.

    Returns (s2, dof) with dof = valid dates - 1 (0 and NaN s2 where fewer than 2 dates are valid). Needs T >= 2.
    """
    a = np.asarray(date_means, dtype=np.float64)
    if a.shape[0] < 2:
        raise ValueError(f'need at least 2 dates to estimate a variance across dates, got {a.shape[0]} (n_pre < 2)')
    ok = np.isfinite(a)
    n = ok.sum(axis=0)
    a0 = np.where(ok, a, 0.0)
    mean = np.divide(a0.sum(axis=0), n, out=np.zeros(n.shape), where=n > 0)
    ss = (np.where(ok, a - mean, 0.0) ** 2).sum(axis=0)
    s2 = np.full(n.shape, np.nan)
    np.divide(ss, n - 1, out=s2, where=n > 1)
    return s2, np.maximum(n - 1, 0)


def box_mean(a, window: int, valid=None):
    """NaN-aware local mean over a (window x window) box, shrinking at the borders (integral image, O(HW))."""
    if window < 1 or window % 2 == 0:
        raise ValueError('window must be a positive odd integer')
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError('box_mean needs a 2-D array')
    ok = np.isfinite(a) if valid is None else (np.isfinite(a) & np.asarray(valid, bool))
    r = window // 2
    v = np.where(ok, a, 0.0)

    def boxsum(x):
        p = np.pad(x, ((r + 1, r), (r + 1, r)))
        c = p.cumsum(axis=0).cumsum(axis=1)
        h, w = x.shape
        return c[2 * r + 1:2 * r + 1 + h, 2 * r + 1:2 * r + 1 + w] - c[:h, 2 * r + 1:2 * r + 1 + w] \
            - c[2 * r + 1:2 * r + 1 + h, :w] + c[:h, :w]

    s, n = boxsum(v), boxsum(ok.astype(np.float64))
    return np.divide(s, n, out=np.full(a.shape, np.nan), where=n > 0.5)


def quantile_strata(level, n_strata: int, valid=None):
    """Equal-count strata of an observable `level` (e.g. the pre-mean NDVI): (labels int, -1 where invalid; edges)."""
    level = np.asarray(level, dtype=np.float64)
    ok = np.isfinite(level) if valid is None else (np.isfinite(level) & np.asarray(valid, bool))
    if n_strata < 1 or not ok.any():
        raise ValueError('need n_strata >= 1 and at least one valid pixel')
    edges = np.unique(np.quantile(level[ok], np.linspace(0, 1, n_strata + 1)[1:-1])) if n_strata > 1 else np.zeros(0)
    labels = np.full(level.shape, -1, dtype=np.int32)
    labels[ok] = np.searchsorted(edges, level[ok], side='right')
    return labels, edges


def _stratum_means(s2, labels, ok):
    n = int(labels[ok].max()) + 1 if ok.any() else 1
    sums = np.bincount(labels[ok], weights=s2[ok], minlength=n)
    cnt = np.bincount(labels[ok], minlength=n)
    means = np.divide(sums, cnt, out=np.full(n, np.nan), where=cnt > 0)
    return means, cnt


def _fit_nu0(s2, prior, dof, ok, floor_rel_median: float, nu0_max: float) -> float:
    """Prior degrees of freedom by the log-variance moment estimator (Smyth 2004 style).

    If s2_i ~ sigma_i^2 chi2_dof/dof and sigma_i^2 ~ scaled-inverse-chi2(nu0, s0^2), then
    var(log s2_i) = trigamma(dof/2) + trigamma(nu0/2)  (spread of the ratio around the prior location).
    """
    sel = ok & np.isfinite(prior) & (prior > 0)
    if sel.sum() < 10:
        return 0.0
    floor = floor_rel_median * float(np.median(s2[sel]))
    z = np.log(np.maximum(s2[sel], max(floor, 1e-300)) / prior[sel])
    excess = float(np.var(z, ddof=1)) - float(np.mean(trigamma(np.maximum(dof[sel], 1) / 2.0)))
    if excess <= trigamma(nu0_max / 2.0):
        return float(nu0_max)                     # no detectable heterogeneity beyond sampling noise: shrink fully
    return float(min(2.0 * trigamma_inverse(excess), nu0_max))


def _blend(s2, prior, dof, nu0):
    """Posterior mean of sigma^2: ((nu0-2) prior + dof s2) / (nu0-2 + dof). prior weight 0 when nu0 <= 2 (heavy prior tail)."""
    w0 = max(nu0 - 2.0, 0.0)
    out = np.full(s2.shape, np.nan)
    ok = np.isfinite(s2) & np.isfinite(prior) & (dof > 0)
    out[ok] = (w0 * prior[ok] + dof[ok] * s2[ok]) / (w0 + dof[ok])
    return out, w0


def estimate_s2_dates(date_means, method: str = 'eb_stratified', *, strata_labels=None, window=None, valid=None,
                      floor_rel_median: float = 1e-5, nu0_max: float = 1e4):
    """Per-pixel variance ACROSS dates of the per-date means, with the small-n shrinkage selected by `method`.

    raw            s2 from the dates themselves (n_pre - 1 degrees of freedom: 2 at n_pre = 3). Unbiased, very noisy.
    window         local pooled mean of raw s2 over a (window x window) box (needs 2-D pixels). Biased where variance
                   changes within the window; degrees of freedom ~ (n_pre - 1) x window^2 (correlated pixels: fewer).
    eb_stratified  posterior mean shrinking raw s2 toward the mean s2 of its stratum (`strata_labels`, e.g. quantile
                   strata of the pre-mean NDVI); prior dof nu0 by moments.
    eb_window      the same, shrinking toward the local window mean.

    Returns (s2_hat, info). NaN where fewer than two dates are valid. `valid` restricts the pixels used to fit the prior.
    """
    if method not in METHODS:
        raise ValueError(f'unknown method {method!r}; choose from {METHODS}')
    s2, dof = date_variance(date_means)
    dof = dof.astype(np.float64)
    ok = np.isfinite(s2) & (dof > 0)
    fit_ok = ok if valid is None else (ok & np.asarray(valid, bool))
    info = {'method': method, 'n_dates': int(np.asarray(date_means).shape[0]), 'dof_max': float(dof.max()),
            'valid_px': int(ok.sum()), 'nu0': None, 'shrink_weight': 0.0}
    if method == 'raw':
        return s2, info
    if method in ('window', 'eb_window'):
        if window is None or s2.ndim != 2:
            raise ValueError(f'method {method!r} needs a 2-D array and an odd `window`')
        prior = box_mean(s2, window, fit_ok)
        info['window'] = int(window)
    else:
        if strata_labels is None:
            raise ValueError("method 'eb_stratified' needs `strata_labels` (see quantile_strata)")
        labels = np.asarray(strata_labels)
        if labels.shape != s2.shape:
            raise ValueError('strata_labels must match the pixel grid')
        good = fit_ok & (labels >= 0)
        means, cnt = _stratum_means(s2, labels, good)
        fallback = float(s2[good].mean()) if good.any() else np.nan
        means = np.where(np.isfinite(means), means, fallback)
        prior = np.where(labels >= 0, means[np.clip(labels, 0, len(means) - 1)], fallback)
        info['n_strata'] = int(len(means))
    if method == 'window':
        out = np.where(ok, prior, np.nan)
        return out, info
    nu0 = _fit_nu0(s2, prior, dof, fit_ok, floor_rel_median, nu0_max)
    out, w0 = _blend(s2, prior, dof, nu0)
    info['nu0'] = nu0
    info['shrink_weight'] = float(np.mean(w0 / (w0 + dof[ok]))) if ok.any() else 0.0
    return out, info


# ---- the predictive sigma -----------------------------------------------------------------------------------------------

def predictive_sigma2(s2_dates, n_pre: int, s2_dihedral_pre=0.0, s2_dihedral_post=0.0, floor: float = 0.0):
    """sigma^2 = s2_dates (1 + 1/n_pre) + s2_dihedral_pre / n_pre + s2_dihedral_post, floored at `floor` (a variance).

    NaN inputs give NaN (NO_DATA); negative variances are an upstream bug and raise. n_pre < 2 raises: with a single
    pre date there is no across-date variance to borrow.
    """
    if int(n_pre) != n_pre or n_pre < 2:
        raise ValueError(f'n_pre must be an integer >= 2, got {n_pre}')
    s2d, dp, dq = (np.asarray(a, dtype=np.float64) for a in (s2_dates, s2_dihedral_pre, s2_dihedral_post))
    for name, a in (('s2_dates', s2d), ('s2_dihedral_pre', dp), ('s2_dihedral_post', dq)):
        finite = a[np.isfinite(a)]
        if finite.size and finite.min() < 0:
            raise ValueError(f'{name} has negative variances (min {finite.min():.3g})')
    out = s2d * (1.0 + 1.0 / n_pre) + dp / n_pre + dq
    return np.maximum(out, floor)


def predictive_sigma(s2_dates, n_pre: int, s2_dihedral_pre=0.0, s2_dihedral_post=0.0, floor: float = 0.0):
    """sqrt of `predictive_sigma2`; `floor` is a VARIANCE (sigma_floor^2)."""
    return np.sqrt(predictive_sigma2(s2_dates, n_pre, s2_dihedral_pre, s2_dihedral_post, floor))


def predictive_sigma_from_dates(pre_date_means, pre_dihedral_var, post_dihedral_var, *, method: str = 'eb_stratified',
                                strata_labels=None, n_strata: int = 20, window=None, valid=None, floor: float = 0.0,
                                floor_rel_median: float = 1e-5, nu0_max: float = 1e4):
    """Wave-2 entry point: per-pixel predictive SD of (new date - pre mean).

    pre_date_means  (n_pre, H, W) per-date mean NDVI (mean of that date's dihedral runs)
    pre_dihedral_var(n_pre, H, W) per-date dihedral variance (ddof 1); post_dihedral_var (H, W)
    floor           a VARIANCE floor (sigma_floor^2)
    For 'eb_stratified' without `strata_labels`, equal-count strata of the pre-mean NDVI (`n_strata`) are used.
    Returns (sigma, info); NaN where undefined.
    """
    means = np.asarray(pre_date_means, dtype=np.float64)
    n_pre = means.shape[0]
    if n_pre < 2:
        raise ValueError(f'need n_pre >= 2 pre dates, got {n_pre}')
    if method == 'eb_stratified' and strata_labels is None:
        with np.errstate(invalid='ignore'):
            level = np.nanmean(means, axis=0)
        strata_labels, _ = quantile_strata(level, n_strata, valid)
    s2, info = estimate_s2_dates(means, method, strata_labels=strata_labels, window=window, valid=valid,
                                 floor_rel_median=floor_rel_median, nu0_max=nu0_max)
    with np.errstate(invalid='ignore'):
        dih_pre = np.nanmean(np.asarray(pre_dihedral_var, dtype=np.float64), axis=0)
    sigma = predictive_sigma(s2, n_pre, dih_pre, post_dihedral_var, floor)
    info['n_pre'] = int(n_pre)
    return sigma, info


# ---- calibration of k -----------------------------------------------------------------------------------------------------

def _usable(d, sigma, mask):
    d, sigma = np.asarray(d, dtype=np.float64), np.asarray(sigma, dtype=np.float64)
    ok = np.isfinite(d) & np.isfinite(sigma) & (sigma > 0)
    return d, sigma, ok if mask is None else ok & np.asarray(mask, bool)


def coverage(d, sigma, k: float, mask=None) -> float:
    """Fraction of usable pixels (finite d, finite sigma > 0, inside `mask`) with |d| <= k sigma."""
    d, sigma, ok = _usable(d, sigma, mask)
    if not ok.any():
        raise ValueError('no usable pixels')
    return float(np.mean(np.abs(d[ok]) <= k * sigma[ok]))


def effective_k(d, sigma, nominal: float = NOMINAL_2SIGMA, mask=None) -> float:
    """k_eff with P(|d| <= k_eff sigma) = `nominal` on the usable calibration pixels (the `nominal` quantile of |d|/sigma)."""
    if not 0.0 < nominal < 1.0:
        raise ValueError('nominal must be in (0, 1)')
    d, sigma, ok = _usable(d, sigma, mask)
    if not ok.any():
        raise ValueError('no usable calibration pixels for k_eff')
    return float(np.quantile(np.abs(d[ok]) / sigma[ok], nominal))


# ---- synthetic generator (tests and the script's labelled-synthetic bias table) --------------------------------------------

def simulate_dates(n_px: int = 120_000, n_pre: int = 3, runs: int = 8, seed: int = 0, nu_true: float = 10.0,
                   delta_ratio: float = 0.07, n_levels: int = 8):
    """SYNTHETIC per-run NDVI with a known truth, for bias checks (never real-data evidence).

    Pixel i has date-effect variance tau_i^2 = level_scale(u_i) * g_i, g_i ~ scaled-inverse-chi2(nu_true) with mean 1, where the
    observable u_i sets the level (8 levels, 0.5e-3 .. 3.5e-3). Run j of date t is a_t + eps_tj with a_t ~ N(0, tau_i^2) shared by
    all runs of the date and eps ~ N(0, delta^2) independent (delta = delta_ratio x mean tau). The LAST of the n_pre + 1 dates is the
    new date. True predictive variance of d = mean_runs(new) - mean_dates(mean_runs(pre)) is (tau^2 + delta^2/runs)(1 + 1/n_pre).
    """
    rng = np.random.default_rng(seed)
    u = rng.random(n_px)
    level_scale = (0.5 + 3.0 * np.floor(u * n_levels) / (n_levels - 1)) * 1e-3
    g = (nu_true - 2.0) / rng.chisquare(nu_true, n_px)
    tau2 = level_scale * g
    delta = delta_ratio * math.sqrt(tau2.mean())
    a = rng.standard_normal((n_pre + 1, n_px)) * np.sqrt(tau2)
    eps = rng.standard_normal((n_pre + 1, runs, n_px)) * delta
    x = a[:, None, :] + eps
    true_var = (tau2 + delta ** 2 / runs) * (1.0 + 1.0 / n_pre)
    return {'u': u, 'x': x, 'tau2': tau2, 'delta2': delta ** 2, 'true_var': true_var, 'n_pre': n_pre, 'runs': runs}


def moments_from_runs(x):
    """Per-date mean, dihedral variance and the accumulator from (T, R, N) run values via `DateMoments`."""
    T, R, n = x.shape
    dm = DateMoments((n,), [str(t) for t in range(T)])
    for t in range(T):
        for r in range(R):
            dm.update(str(t), x[t, r])
    return dm.mean_stack(), dm.var_stack(), dm
