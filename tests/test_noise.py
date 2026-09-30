"""A5 noise model: synthetic tests. Every claim here is EVIDENCE: synthetic (a simulation with a known truth), not real data.

The simulation generates, per pixel i, date effects a_t ~ N(0, tau_i^2) (seasonal/atmospheric/registration variability that all
dihedral runs of one date share) plus independent dihedral run noise eps ~ N(0, delta^2). The per-date NDVI of run j is
mu + a_t + eps_tj. The gate compares one NEW date against the pre mean, so the TRUE predictive variance of
d = mean_runs(new) - mean_dates(mean_runs(pre)) is (tau_i^2 + delta^2/R) (1 + 1/n_pre).
"""
import math

import numpy as np
import pytest

from experiments.wayanad_evidence.stats import Welford
from trustsr import noise as N

N_PRE, RUNS = 3, 8


def simulate(n_px=120_000, n_pre=N_PRE, runs=RUNS, seed=0, nu_true=10.0, delta_ratio=0.07, n_levels=8):
    """Heterogeneous tau_i^2 = level_scale(u_i) * g_i with g_i ~ scaled-inverse-chi2(nu_true), mean 1; u observable."""
    rng = np.random.default_rng(seed)
    u = rng.random(n_px)
    level_scale = (0.5 + 3.0 * np.floor(u * n_levels) / (n_levels - 1)) * 1e-3          # 5e-4 .. 3.5e-3 in variance
    g = (nu_true - 2.0) / rng.chisquare(nu_true, n_px)                                  # E[g] = 1
    tau2 = level_scale * g
    delta = delta_ratio * math.sqrt(tau2.mean())
    a = rng.standard_normal((n_pre + 1, n_px)) * np.sqrt(tau2)                          # date effects; last = the new date
    eps = rng.standard_normal((n_pre + 1, runs, n_px)) * delta
    x = a[:, None, :] + eps                                                             # (T, R, N) NDVI of every run
    true_var = (tau2 + delta ** 2 / runs) * (1.0 + 1.0 / n_pre)
    return {'u': u, 'x': x, 'tau2': tau2, 'delta2': delta ** 2, 'true_var': true_var, 'n_pre': n_pre, 'runs': runs}


def moments_from_runs(x):
    """Per-date mean and dihedral variance (ddof 1) from (T, R, N) runs via the streaming accumulators."""
    T, _, n = x.shape
    dm = N.DateMoments((n,), [str(t) for t in range(T)])
    for t in range(T):
        for r in range(x.shape[1]):
            dm.update(str(t), x[t, r])
    return dm.mean_stack(), dm.var_stack(), dm


@pytest.fixture(scope='module')
def sim():
    s = simulate()
    means, dvar, _ = moments_from_runs(s['x'])
    s.update(means=means, dvar=dvar)
    s['d'] = means[-1] - means[:-1].mean(axis=0)
    s['level'] = means[:-1].mean(axis=0) * 0 + s['u']                                  # observable covariate for strata
    return s


# ---- special functions and small helpers --------------------------------------------------------------------------------

def test_trigamma_known_values_and_inverse():
    assert N.trigamma(1.0) == pytest.approx(math.pi ** 2 / 6, rel=1e-12)
    assert N.trigamma(0.5) == pytest.approx(math.pi ** 2 / 2, rel=1e-12)
    for x in (0.25, 0.5, 1.0, 3.0, 12.5, 400.0):
        assert N.trigamma_inverse(N.trigamma(x)) == pytest.approx(x, rel=1e-6)


def test_nominal_coverage_two_sigma():
    assert N.nominal_coverage(2.0) == pytest.approx(0.9545, abs=5e-5)
    assert N.NOMINAL_2SIGMA == pytest.approx(N.nominal_coverage(2.0))


def test_box_mean_matches_brute_force_with_nan_and_edges():
    rng = np.random.default_rng(1)
    a = rng.random((13, 17))
    a[3, 4] = np.nan
    valid = np.ones_like(a, bool)
    valid[0, :3] = False
    got = N.box_mean(a, 5, valid)
    ok = valid & np.isfinite(a)
    for r, c in [(0, 0), (3, 4), (6, 8), (12, 16), (1, 1)]:
        win = (slice(max(r - 2, 0), r + 3), slice(max(c - 2, 0), c + 3))
        sel = ok[win]
        assert got[r, c] == pytest.approx(a[win][sel].mean() if sel.any() else np.nan, nan_ok=True)
    with pytest.raises(ValueError):
        N.box_mean(a, 4, valid)


# ---- streaming per-date accumulators ------------------------------------------------------------------------------------

def test_date_moments_match_numpy_and_skip_nan():
    rng = np.random.default_rng(2)
    x = rng.random((3, 8, 6, 7))
    x[1, 2, 0, 0] = np.nan
    x[2, :, 5, 5] = np.nan
    dm = N.DateMoments((6, 7), ['a', 'b', 'c'])
    for t, name in enumerate('abc'):
        for r in range(8):
            dm.update(name, x[t, r])
    assert np.allclose(dm.mean_stack(), np.nanmean(x, axis=1), equal_nan=True)
    assert np.allclose(dm.var_stack(), np.nanvar(x, axis=1, ddof=1), equal_nan=True)
    assert dm.count_stack()[1, 0, 0] == 7 and dm.count_stack()[2, 5, 5] == 0
    assert np.isnan(dm.mean_stack()[2, 5, 5])                      # zero valid runs -> NaN, not 0
    with pytest.raises(KeyError):
        dm.update('nope', x[0, 0])


def test_date_moments_update_at_region_only():
    dm = N.DateMoments((4, 4), ['a'])
    dm.update_at('a', (slice(0, 2), slice(0, 2)), np.ones((2, 2)))
    assert dm.count_stack()[0, :2, :2].min() == 1 and dm.count_stack()[0, 2:, :].max() == 0


def test_pool_moments_equals_single_welford_over_all_runs(sim):
    """v1 sigma_pre is a Welford over all runs of all pre dates; the per-date merge must reproduce it exactly."""
    x = sim['x'][:N_PRE, :, :2000]
    w = Welford((2000,))
    for t in range(N_PRE):
        for r in range(RUNS):
            w.update(x[t, r])
    means, dvar, dm = moments_from_runs(x)
    pooled_mean, pooled_var = N.pool_moments(means, dvar, dm.count_stack()[:N_PRE], ddof=1)
    assert np.allclose(pooled_mean, w.mean, atol=1e-12)
    assert np.allclose(pooled_var, w.variance(1), rtol=1e-10)
    sigma_old = N.sigma_v1(np.sqrt(pooled_var), np.sqrt(dvar[0]))
    assert np.allclose(sigma_old, np.sqrt(w.std(1) ** 2 + dvar[0]), rtol=1e-10)


# ---- the predictive formula and its edge cases --------------------------------------------------------------------------

def test_predictive_sigma2_formula_and_errors():
    s2d, dp, dq = np.array([4.0e-4]), np.array([1e-6]), np.array([2e-6])
    got = N.predictive_sigma2(s2d, 3, dp, dq)
    assert got == pytest.approx(4.0e-4 * (1 + 1 / 3) + 1e-6 / 3 + 2e-6)
    assert N.predictive_sigma(s2d, 3, dp, dq) == pytest.approx(np.sqrt(got))
    for bad in (1, 0, -3):
        with pytest.raises(ValueError):
            N.predictive_sigma2(s2d, bad, dp, dq)
    with pytest.raises(ValueError):
        N.predictive_sigma2(np.array([-1e-4]), 3, dp, dq)          # negative variance is a bug upstream, not a NaN to hide
    with pytest.raises(ValueError):
        N.predictive_sigma2(s2d, 3, np.array([-1e-6]), dq)


def test_predictive_sigma2_nan_propagates_and_floor_applies():
    s2d = np.array([np.nan, 0.0, 1e-4])
    out = N.predictive_sigma2(s2d, 3, np.zeros(3), np.zeros(3), floor=1e-6)
    assert np.isnan(out[0])                                        # NaN in -> NaN out (becomes NO_DATA), never a silent floor
    assert out[1] == pytest.approx(1e-6) and out[2] == pytest.approx(1e-4 * 4 / 3)
    assert np.all(N.predictive_sigma(s2d, 3, np.zeros(3), np.zeros(3), floor=1e-6)[1:] > 0)


def test_date_variance_needs_two_dates_and_handles_nan():
    with pytest.raises(ValueError):
        N.date_variance(np.zeros((1, 3, 3)))                       # n_pre < 2: no variance across dates exists
    m = np.array([[[1.0, 1.0]], [[2.0, np.nan]], [[4.0, 3.0]]])    # (3 dates, 1, 2 px)
    s2, dof = N.date_variance(m)
    assert s2[0, 0] == pytest.approx(np.var([1, 2, 4], ddof=1)) and dof[0, 0] == 2
    assert s2[0, 1] == pytest.approx(np.var([1, 3], ddof=1)) and dof[0, 1] == 1
    only_one = np.array([[[1.0]], [[np.nan]], [[np.nan]]])
    s2, dof = N.date_variance(only_one)
    assert np.isnan(s2[0, 0]) and dof[0, 0] == 0


def test_old_sigma_is_v1_formula():
    pre_std, post_std = np.array([0.03, 0.04]), np.array([0.002, 0.003])
    assert np.allclose(N.sigma_v1(pre_std, post_std), np.sqrt(pre_std ** 2 + post_std ** 2))


# ---- simulation: old sigma is too small, new sigma is unbiased ------------------------------------------------------

def test_old_sigma_underestimates_true_predictive_sd(sim):
    means, dvar = sim['means'], sim['dvar']
    counts = np.full(means[:N_PRE].shape, float(RUNS))
    _, pooled_var = N.pool_moments(means[:N_PRE], dvar[:N_PRE], counts, ddof=1)
    old2 = N.sigma_v1(np.sqrt(pooled_var), np.sqrt(dvar[-1])) ** 2
    ratio_var = old2.mean() / sim['true_var'].mean()
    # analytic expectation with n_pre = 3, R = 8: pooled variance ~ (16/23) s^2 versus (4/3) s^2  ->  ratio 0.52
    assert 0.47 < ratio_var < 0.58
    assert 0.68 < math.sqrt(ratio_var) < 0.76                      # v1 sigma is ~28 % below the predictive SD (new/old ~ +38 %)


@pytest.mark.parametrize('method', ['raw', 'eb_stratified'])
def test_new_sigma_variance_is_unbiased_within_5_percent(sim, method):
    kwargs = {'strata_labels': N.quantile_strata(sim['u'], 8)[0]} if method == 'eb_stratified' else {}
    s2, info = N.estimate_s2_dates(sim['means'][:N_PRE], method=method, **kwargs)
    new2 = N.predictive_sigma2(s2, N_PRE, sim['dvar'][:N_PRE].mean(axis=0), sim['dvar'][-1])
    assert new2.mean() / sim['true_var'].mean() == pytest.approx(1.0, abs=0.05)
    if method == 'eb_stratified':
        assert 6.0 < info['nu0'] < 16.0                            # true prior dof is 10


def test_shrinkage_reduces_variance_of_the_variance_estimate(sim):
    labels = N.quantile_strata(sim['u'], 8)[0]
    raw, _ = N.estimate_s2_dates(sim['means'][:N_PRE], method='raw')
    eb, _ = N.estimate_s2_dates(sim['means'][:N_PRE], method='eb_stratified', strata_labels=labels)
    target = sim['tau2'] + sim['delta2'] / RUNS
    mse = lambda est: np.mean((np.log(np.maximum(est, 1e-12)) - np.log(target)) ** 2)
    assert mse(eb) < 0.75 * mse(raw)                               # log-variance error falls by a wide margin
    assert np.mean((eb / target - 1) ** 2) < np.mean((raw / target - 1) ** 2)
    assert np.var(eb) < np.var(raw)


def test_window_estimator_unbiased_on_smooth_field():
    rng = np.random.default_rng(5)
    h = w = 240
    yy, xx = np.mgrid[:h, :w]
    tau2 = (1.0 + 0.8 * np.sin(xx / 40.0) * np.cos(yy / 55.0)) * 1e-3          # smooth spatial variance field
    a = rng.standard_normal((N_PRE + 1, h, w)) * np.sqrt(tau2)
    s2_raw, _ = N.estimate_s2_dates(a[:N_PRE], method='raw')
    s2_win, info = N.estimate_s2_dates(a[:N_PRE], method='window', window=15)
    assert s2_win.mean() / tau2.mean() == pytest.approx(1.0, abs=0.05)
    assert np.mean((s2_win / tau2 - 1) ** 2) < np.mean((s2_raw / tau2 - 1) ** 2)
    s2_ebw, info = N.estimate_s2_dates(a[:N_PRE], method='eb_window', window=15)
    assert s2_ebw.mean() / tau2.mean() == pytest.approx(1.0, abs=0.05)
    assert np.isfinite(info['nu0'])


def test_eb_falls_back_to_raw_when_no_prior_information():
    """Pixels with wildly heterogeneous variance and no informative stratum: nu0 <= 2 -> no shrinkage, weight 0 (no crash)."""
    rng = np.random.default_rng(6)
    tau2 = np.exp(rng.normal(0, 3.0, 50_000))                       # log-sd 3: prior nearly flat
    a = rng.standard_normal((3, 50_000)) * np.sqrt(tau2)
    raw, _ = N.estimate_s2_dates(a, method='raw')
    eb, info = N.estimate_s2_dates(a, method='eb_stratified', strata_labels=np.zeros(50_000, int))
    assert info['shrink_weight'] < 0.3
    assert np.all(np.isfinite(eb))
    assert np.corrcoef(np.log(raw), np.log(eb))[0, 1] > 0.98


# ---- effective k ----------------------------------------------------------------------------------------------------------

def test_effective_k_recovers_nominal_coverage_out_of_sample(sim):
    labels = N.quantile_strata(sim['u'], 8)[0]
    s2, _ = N.estimate_s2_dates(sim['means'][:N_PRE], method='raw')                 # 2-dof: heavy-tailed z
    sigma = N.predictive_sigma(s2, N_PRE, sim['dvar'][:N_PRE].mean(axis=0), sim['dvar'][-1], floor=1e-8)
    n = sigma.size // 2
    cal, val = slice(0, n), slice(n, None)
    naive = N.coverage(sim['d'][val], sigma[val], 2.0)
    assert naive < 0.90                                            # raw 2-dof sigma at k = 2 badly under-covers (t_2: 0.82)
    k_eff = N.effective_k(sim['d'][cal], sigma[cal])
    assert k_eff > 3.0
    assert N.coverage(sim['d'][val], sigma[val], k_eff) == pytest.approx(N.NOMINAL_2SIGMA, abs=0.005)
    # with the shrunk sigma the naive k = 2 is already near nominal and k_eff stays near 2
    s2b, _ = N.estimate_s2_dates(sim['means'][:N_PRE], method='eb_stratified', strata_labels=labels)
    sig_b = N.predictive_sigma(s2b, N_PRE, sim['dvar'][:N_PRE].mean(axis=0), sim['dvar'][-1], floor=1e-8)
    assert abs(N.coverage(sim['d'][val], sig_b[val], 2.0) - N.NOMINAL_2SIGMA) < 0.02
    assert 1.8 < N.effective_k(sim['d'][cal], sig_b[cal]) < 2.3


def test_effective_k_mask_nan_and_errors():
    d = np.array([0.1, -0.2, 5.0, np.nan, 0.05])
    s = np.array([0.1, 0.1, 0.1, 0.1, 0.0])
    mask = np.array([True, True, False, True, True])
    k = N.effective_k(d, s, nominal=0.5, mask=mask)                 # z = 1, 2 (5.0 masked, NaN and sigma 0 dropped)
    assert 1.0 <= k <= 2.0
    assert N.coverage(d, s, 1.5, mask=mask) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        N.effective_k(np.array([np.nan]), np.array([1.0]))
    with pytest.raises(ValueError):
        N.effective_k(d, s, nominal=1.0)


def test_top_level_wrapper_matches_pieces(sim):
    sigma, info = N.predictive_sigma_from_dates(sim['means'][:N_PRE], sim['dvar'][:N_PRE], sim['dvar'][-1], method='raw', floor=0.0)
    s2, _ = N.estimate_s2_dates(sim['means'][:N_PRE], method='raw')
    want = N.predictive_sigma(s2, N_PRE, sim['dvar'][:N_PRE].mean(axis=0), sim['dvar'][-1])
    assert np.allclose(sigma, want, equal_nan=True)
    assert info['n_pre'] == N_PRE and info['method'] == 'raw'
    with pytest.raises(ValueError):
        N.predictive_sigma_from_dates(sim['means'][:1], sim['dvar'][:1], sim['dvar'][-1])
    with pytest.raises(ValueError):
        N.estimate_s2_dates(sim['means'][:N_PRE], method='nonsense')
