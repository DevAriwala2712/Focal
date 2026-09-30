import numpy as np
import pytest

from trustsr.bootstrap import block_sums, paired_bootstrap_ci, ratio_bootstrap_ci


def test_block_sums_partial_edge_blocks_and_total():
    a = np.arange(35, dtype=float).reshape(5, 7)
    s = block_sums(a, 3)
    assert s.shape == (2, 3)                       # ceil(5/3), ceil(7/3): edge blocks are kept, not cropped
    assert s.sum() == a.sum()
    assert s[0, 0] == a[:3, :3].sum() and s[1, 2] == a[3:, 6:].sum()


def test_block_sums_nan_counts_as_zero_and_bool_input():
    a = np.ones((4, 4))
    a[0, 0] = np.nan
    assert block_sums(a, 2)[0, 0] == 3.0
    assert block_sums(np.ones((4, 4), bool), 2).tolist() == [[4, 4], [4, 4]]


def test_paired_ci_covers_true_mean_and_is_deterministic():
    rng = np.random.default_rng(0)
    d = rng.normal(0.5, 1.0, 200)
    r1 = paired_bootstrap_ci(d, replicates=1000, ci=0.95, seed=2024)
    r2 = paired_bootstrap_ci(d, replicates=1000, ci=0.95, seed=2024)
    assert r1 == r2
    assert r1['lo'] < 0.5 < r1['hi'] and r1['n'] == 200
    assert r1['mean'] == pytest.approx(d.mean())


def test_paired_ci_excludes_zero_for_clear_effect_not_for_null():
    rng = np.random.default_rng(1)
    assert paired_bootstrap_ci(rng.normal(1, 0.2, 100), 1000, 0.95, 1)['lo'] > 0
    null = paired_bootstrap_ci(rng.normal(0, 1, 100), 1000, 0.95, 1)
    assert null['lo'] < 0 < null['hi']


def test_ratio_ci_matches_point_estimate_and_block_resampling_widens_correlated_ci():
    rng = np.random.default_rng(2)
    n_blocks = 400
    den = np.full(n_blocks, 1000.0)
    # spatially clustered rate: some blocks carry almost all events
    rate = np.where(rng.random(n_blocks) < 0.1, 0.5, 0.005)
    num = rate * den
    r = ratio_bootstrap_ci(num, den, replicates=1000, ci=0.95, seed=3)
    assert r['estimate'] == pytest.approx(num.sum() / den.sum())
    iid_se = np.sqrt(r['estimate'] * (1 - r['estimate']) / den.sum())
    assert (r['hi'] - r['lo']) / (2 * 1.96) > 3 * iid_se     # block CI is far wider than the naive pixel-iid CI
    assert r['denominator'] == den.sum()


def test_ratio_ci_difference_of_two_numerators_shares_blocks():
    den = np.full(50, 100.0)
    a = np.full(50, 30.0)
    b = np.full(50, 10.0)
    r = ratio_bootstrap_ci(a, den, 500, 0.95, 4, num_b=b)
    assert r['estimate'] == pytest.approx(0.2) and r['lo'] == pytest.approx(0.2) and r['hi'] == pytest.approx(0.2)


def test_empty_or_zero_denominator_is_an_error_not_nan():
    with pytest.raises(ValueError):
        ratio_bootstrap_ci(np.zeros(3), np.zeros(3), 10, 0.95, 0)
    with pytest.raises(ValueError):
        paired_bootstrap_ci(np.array([]), 10, 0.95, 0)
