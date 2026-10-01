"""F3 noise model v2: unit tests. Real-data heavy stages (E6 selection, Wayanad LODO) are exercised end-to-end by
running `python -m experiments.f3_noise_v2` (see experiments/results/f3.json); these tests cover the estimator and
the change-preservation check with fast synthetic/small fixtures, per the task's requirement to test:
  (1) the offset estimator recovers a known synthetic offset,
  (2) the coverage computation matches A5's definition (shared with `trustsr.noise.coverage`) on a fixture,
  (3) the change-preservation test itself is executable and asserted.
"""
from __future__ import annotations

import numpy as np
import pytest

from experiments.f3_noise_v2 import change_preservation_test, normalise_dates, robust_offsets
from trustsr import noise as N


# ---- (1) the offset estimator recovers a known synthetic offset ---------------------------------------------------------

def test_robust_offsets_recovers_known_constant_shift():
    rng = np.random.default_rng(0)
    h, w = 60, 60
    base = 0.5 + 0.05 * rng.standard_normal((h, w))
    mask = np.ones((h, w), bool)
    mask[:5] = False                                    # a border region excluded, like disks/footprint
    true_offsets = np.array([0.0, 0.03, -0.07])
    stack = base[None] + true_offsets[:, None, None]
    ref = np.nanmedian(stack, axis=0)                    # same reference construction as normalise_dates
    got = robust_offsets(stack, ref, mask)
    # offsets are only identified up to their own median (a global additive constant cancels), so compare
    # after centring both the true and the fitted offsets
    assert np.allclose(got - got.mean(), true_offsets - true_offsets.mean(), atol=1e-8)


def test_robust_offsets_is_robust_to_a_minority_of_outlier_pixels():
    rng = np.random.default_rng(1)
    h, w = 80, 80
    base = 0.4 + 0.02 * rng.standard_normal((h, w))
    mask = np.ones((h, w), bool)
    contaminated = base.copy()
    contaminated[:8, :8] -= 5.0                          # ~1 % of pixels are wild outliers (e.g. a real local change)
    stack = np.stack([base, contaminated])
    ref = np.nanmedian(stack, axis=0)
    off = robust_offsets(stack, ref, mask)
    # the median offset of the contaminated date must stay close to 0 (untouched by the outlier patch), not -5
    assert abs(off[1]) < 0.05


def test_normalise_dates_removes_a_known_per_date_offset_and_leaves_held_out_evaluable():
    rng = np.random.default_rng(2)
    h, w = 50, 50
    field = 0.6 + 0.03 * rng.standard_normal((h, w))
    mask = np.ones((h, w), bool)
    pool = np.stack([field + 0.02, field - 0.02])         # two pool dates, offset +/- 0.02 around `field`
    held = field + 0.10                                    # held-out date carries a +0.10 offset
    norm_pool, norm_held, off_pool, off_held, ref = normalise_dates(pool, mask, held, mask)
    assert norm_pool.shape == pool.shape and norm_held.shape == held.shape
    # after normalisation the pool dates should agree with each other (their +/- 0.02 offset removed)
    assert np.std(norm_pool[0][mask] - norm_pool[1][mask]) < 1e-6
    assert abs(off_held - 0.10) < 0.02                     # held-out offset recovered close to the injected 0.10
    d = norm_held - norm_pool.mean(axis=0)
    assert abs(np.median(d[mask])) < 1e-9                  # exact by construction (see f3_noise_v2.py docstring)


def test_normalise_dates_single_pool_date_and_shape_errors():
    mask = np.ones((10, 10), bool)
    pool = np.full((1, 10, 10), 0.5)
    norm_pool, norm_held, off_pool, off_held, ref = normalise_dates(pool, mask, np.full((10, 10), 0.7), mask)
    assert off_held == pytest.approx(0.2, abs=1e-9)
    with pytest.raises(ValueError):
        normalise_dates(np.zeros((0, 5, 5)), np.ones((5, 5), bool))
    with pytest.raises(ValueError):
        robust_offsets(np.zeros((2, 5, 5)), np.zeros((5, 5)), np.zeros((5, 5), bool))   # empty mask


# ---- (2) coverage computation matches A5's definition (shared code: trustsr.noise.coverage) -----------------------------

def test_coverage_definition_matches_a5_shared_fixture():
    """`configs/exceptional.yaml` a5.coverage_metric: fraction of stable pixels with |d| <= 2*sigma. F3 reuses
    `trustsr.noise.coverage` verbatim (imported, not reimplemented) for exactly this reason: a shared fixture must
    give the identical number whichever caller computes it."""
    d = np.array([0.1, -0.3, 0.05, 2.5, -0.02, 4.0])
    sigma = np.full(6, 1.0)
    mask = np.array([True, True, True, True, True, False])       # last pixel excluded (e.g. outside disk)
    # by hand: |d|<=2 on the first 5 masked-in pixels -> 0.1,0.3,0.05,2.5(excluded>2),0.02 => 4 of 5 covered
    expected = 4 / 5
    assert N.coverage(d, sigma, 2.0, mask=mask) == pytest.approx(expected)
    # end-to-end through the normalisation pipeline: d built the same way run_wayanad builds it
    field = np.full((20, 20), 0.5)
    stable_mask = np.ones((20, 20), bool)
    pool = np.stack([field, field])                                # two identical "remaining" pre dates: zero spread
    held = field.copy()
    held[0, 0] += 10.0                                              # one wild pixel: must NOT be covered at k=2, sigma~0
    norm_pool, norm_held, *_ = normalise_dates(pool, stable_mask, held, stable_mask)
    d2 = norm_held - norm_pool.mean(axis=0)
    sigma2 = np.full((20, 20), 1e-6)
    cov = N.coverage(d2, sigma2, 2.0, mask=stable_mask)
    assert cov == pytest.approx((stable_mask.sum() - 1) / stable_mask.sum())


# ---- (3) the change-preservation test is executable and asserted --------------------------------------------------------

def test_change_preservation_test_runs_and_preserves_an_injected_drop():
    rng = np.random.default_rng(3)
    h, w = 100, 100
    base = 0.6 + 0.02 * rng.standard_normal((h, w))
    means = {'2024-01-16': base + 0.01, '2024-01-21': base - 0.01, '2024-01-26': base.copy()}
    mask = np.ones((h, w), bool)
    mask[:10] = False
    out = change_preservation_test(means, mask, rng_seed=7, drop=0.30, patch_px=15)
    assert out['evidence'].startswith('synthetic')
    assert out['injected_patch_px'] > 0
    # the core claim under test: a small local drop is not absorbed by the scene-wide median offset
    assert out['recovery_ratio'] == pytest.approx(1.0, abs=0.02)
    assert abs(out['offset_shift_as_fraction_of_injected_drop']) < 0.02


def test_change_preservation_test_flags_absorption_when_the_patch_dominates_the_mask():
    """Sanity check on the test's own power: if the 'stable' mask is mostly the injected patch (so the offset
    estimator is no longer robust to it), the recovery ratio must clearly depart from 1 -- otherwise the
    change-preservation check would be tautological (always passing) and should be dropped, not trusted.

    The patch anchor is chosen from a random stable pixel inside `change_preservation_test`; to make the patch
    dominate the mask DETERMINISTICALLY (independent of which stable pixel the RNG lands on), the mask is reduced
    to a single stable pixel -- the estimator is then fit on n=1, so its "robust median" is that one pixel's value
    and the injected drop moves it completely (100 % absorption is the expected, correct behaviour here).
    """
    h, w = 20, 20
    base = np.full((h, w), 0.5)
    means = {'a': base.copy(), 'b': base.copy(), 'c': base.copy()}
    mask = np.zeros((h, w), bool)
    mask[5, 5] = True                                     # exactly one stable pixel: the median offset IS that pixel
    out = change_preservation_test(means, mask, rng_seed=0, drop=0.30, patch_px=3)
    assert out['stable_px_used_to_fit_offset'] == 1
    assert out['injected_patch_px'] == 1
    assert out['recovery_ratio'] < 0.1                    # offset absorption is (correctly) total when n = 1


def test_change_preservation_test_raises_on_no_stable_pixels():
    means = {'a': np.zeros((10, 10)), 'b': np.zeros((10, 10))}
    with pytest.raises(ValueError):
        change_preservation_test(means, np.zeros((10, 10), bool), rng_seed=0, drop=0.3, patch_px=5)
