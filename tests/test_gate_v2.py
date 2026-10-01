"""Tests for F2's rebuilt gate v2 (trustsr/gate_v2.py). Each class is pinned to the B-ID it proves fixed.

  B6/B10 TestTauMutation, TestConformalQuantile   B7  TestBlockSum
  B9     TestSrVsBilinearRanking                  B11 TestNoDataFromScl
  B12    TestEndmembersNeverFromLabels
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from trustsr.gate_v2 import (
    ALLOCATED, CORE, NO_CHANGE, NO_DATA, allocate_pixels, apply_gate_v2,
    assert_sr_score_is_not_replicated, band_sigma_from_pre, bilinear_upsample, block_sum_deviation,
    estimate_endmembers, flagged_v2, fraction_sigma, gate_v2_signature_audit, neighbour_agreement,
    nodata_from_scl, split_conformal_quantile, unmix_fraction, window_max_score,
)

E_V = np.array([0.03, 0.05, 0.04, 0.42])       # B04, B03, B02, B08 (unmixing uses indices 0 and 3)
E_B = np.array([0.22, 0.20, 0.18, 0.26])


def _scene(n_blocks=32, n_pre=3, seed=0, f_true=None):
    """Synthetic reflectance built by MIXING the two endmembers, so the true fraction is known."""
    rng = np.random.RandomState(seed)
    h = w = n_blocks
    a_pre = np.full((h, w), 1.0)
    a_post = a_pre - (np.zeros((h, w)) if f_true is None else f_true)
    mix = lambda a: E_B[None, None, :] + a[..., None] * (E_V - E_B)[None, None, :]
    y_pre = np.stack([mix(a_pre) + rng.normal(0, 2e-4, (h, w, 4)) for _ in range(n_pre)])
    y_post = mix(a_post) + rng.normal(0, 2e-4, (h, w, 4))
    d_sr = rng.normal(0, 1.0, (4 * h, 4 * w))                       # varies WITHIN every 4x4 block
    nodata = np.zeros((4 * h, 4 * w), bool)
    sigma_bands, _ = band_sigma_from_pre(y_pre)
    return y_pre, y_post, d_sr, sigma_bands, nodata


# ---------------------------------------------------------------- unmixing ----------------------------------------------------------------

class TestUnmixing:
    def test_pure_vegetation_on_both_sides_gives_zero_fraction(self):
        y_pre, y_post, *_ = _scene(n_blocks=8, f_true=np.zeros((8, 8)))
        f, _ = unmix_fraction(y_pre, y_post, E_V, E_B)
        assert np.abs(f).max() < 0.01

    def test_recovers_a_known_mixing_fraction(self):
        f_true = np.full((8, 8), 0.5)
        y_pre, y_post, *_ = _scene(n_blocks=8, f_true=f_true)
        f, _ = unmix_fraction(y_pre, y_post, E_V, E_B)
        assert np.allclose(f, 0.5, atol=0.01)

    def test_fraction_is_clipped_to_unit_interval(self):
        f_true = np.full((8, 8), 2.0)
        y_pre, y_post, *_ = _scene(n_blocks=8, f_true=f_true)
        f, _ = unmix_fraction(y_pre, y_post, E_V, E_B)
        assert f.min() >= 0.0 and f.max() <= 1.0

    def test_degenerate_endmembers_raise(self):
        y_pre, y_post, *_ = _scene(n_blocks=8)
        with pytest.raises(ValueError):
            unmix_fraction(y_pre, y_post, E_V, E_V)

    def test_sigma_f_scales_with_band_noise(self):
        s1 = fraction_sigma(np.full((4, 4, 2), 0.001), E_V, E_B, n_pre=3)
        s2 = fraction_sigma(np.full((4, 4, 2), 0.002), E_V, E_B, n_pre=3)
        assert np.allclose(s2, 2 * s1)


# ---------------------------------------------------------------- B12 ----------------------------------------------------------------

class TestEndmembersNeverFromLabels:
    def _inputs(self, seed=0):
        rng = np.random.RandomState(seed)
        h = w = 40
        ndvi = np.full((3, h, w), 0.85) + rng.normal(0, 0.005, (3, h, w))
        ndvi[:, 20:, :] = 0.10 + rng.normal(0, 0.005, (3, 20, w))       # a low-NDVI half for e_b
        refl = np.zeros((3, h, w, 4))
        refl[:] = E_V
        refl[:, 20:, :, :] = E_B
        refl += rng.normal(0, 1e-4, refl.shape)
        valid = np.ones((3, h, w), bool)
        exclude = np.zeros((h, w), bool)
        exclude[:8, :8] = True
        return refl, valid, ndvi, exclude

    def test_signature_carries_no_label_argument(self):
        audit = gate_v2_signature_audit()
        assert audit['estimate_endmembers_has_no_label_argument']
        params = inspect.signature(estimate_endmembers).parameters
        assert 'y_post' not in params and 'labels' not in params and 'class_map' not in params

    def test_endmembers_ignore_everything_inside_the_exclusion(self):
        """The VALUES inside the exclusion geometry must have zero influence: the event area can never
        contribute an endmember, so the estimate cannot be a disguised label."""
        refl, valid, ndvi, exclude = self._inputs()
        base = estimate_endmembers(refl, valid, ndvi, exclude, min_count=10)
        refl2, ndvi2 = refl.copy(), ndvi.copy()
        refl2[:, :8, :8, :] = 99.0                                       # absurd values, inside the exclusion
        ndvi2[:, :8, :8] = -5.0
        other = estimate_endmembers(refl2, valid, ndvi2, exclude, min_count=10)
        assert np.array_equal(base['e_v'], other['e_v'])
        assert np.array_equal(base['e_b'], other['e_b'])
        assert base['e_v_count'] == other['e_v_count'] and base['e_b_count'] == other['e_b_count']

    def test_endmembers_recover_the_planted_spectra_and_report_counts(self):
        refl, valid, ndvi, exclude = self._inputs()
        em = estimate_endmembers(refl, valid, ndvi, exclude, min_count=10)
        assert np.allclose(em['e_v'], E_V, atol=1e-3)
        assert np.allclose(em['e_b'], E_B, atol=1e-3)
        assert em['e_v_count'] > 0 and em['e_b_count'] > 0
        assert em['e_v_count'] + em['e_b_count'] <= em['n_valid_outside_exclusion']
        assert len(em['e_v_robust_sd']) == 4 and len(em['e_b_robust_sd']) == 4

    def test_refuses_rather_than_widening_when_too_few_pixels(self):
        refl, valid, ndvi, exclude = self._inputs()
        with pytest.raises(ValueError, match='min_count'):
            estimate_endmembers(refl, valid, ndvi, exclude, min_count=10 ** 7)

    def test_an_invalid_date_removes_the_pixel_from_the_sample(self):
        refl, valid, ndvi, exclude = self._inputs()
        base = estimate_endmembers(refl, valid, ndvi, exclude, min_count=10)
        valid2 = valid.copy()
        valid2[1, :10, :] = False
        other = estimate_endmembers(refl, valid2, ndvi, exclude, min_count=10)
        assert other['e_v_count'] < base['e_v_count']


# ---------------------------------------------------------------- B6/B10 ----------------------------------------------------------------

class TestConformalQuantile:
    def test_order_statistic_index(self):
        scores = np.arange(1, 101, dtype=float)
        tau, info = split_conformal_quantile(scores, alpha=0.05)
        assert info['k'] == int(np.ceil(101 * 0.95)) == 96
        assert tau == 96.0

    def test_refuses_to_clamp_when_calibration_is_too_small(self):
        with pytest.raises(ValueError, match='too small'):
            split_conformal_quantile(np.arange(5.0), alpha=0.05)


class TestTauMutation:
    """B10's direct regression: x9 loaded tau_win and never passed it, so detection never happened."""

    def _case(self):
        f_true = np.zeros((32, 32))
        f_true[4:12, 4:12] = 0.6
        return _scene(n_blocks=32, f_true=f_true, seed=3)

    def test_tau_is_a_required_positional_argument(self):
        audit = gate_v2_signature_audit()
        assert audit['tau_is_required_positional']
        y_pre, y_post, d_sr, sigma_bands, nodata = self._case()
        with pytest.raises(TypeError):
            apply_gate_v2(y_pre, y_post, d_sr, E_V, E_B, sigma_bands, nodata)      # no tau -> TypeError

    def test_non_finite_tau_raises(self):
        y_pre, y_post, d_sr, sigma_bands, nodata = self._case()
        for bad in (None, np.nan, np.inf):
            with pytest.raises(ValueError, match='explicit, finite tau'):
                apply_gate_v2(y_pre, y_post, d_sr, E_V, E_B, sigma_bands, nodata, bad)

    def test_changing_tau_changes_the_output_map(self):
        y_pre, y_post, d_sr, sigma_bands, nodata = self._case()
        low, _ = apply_gate_v2(y_pre, y_post, d_sr, E_V, E_B, sigma_bands, nodata, 1.0)
        high, meta_h = apply_gate_v2(y_pre, y_post, d_sr, E_V, E_B, sigma_bands, nodata, 1e9)
        assert not np.array_equal(low, high)
        assert flagged_v2(low).sum() > 0
        assert flagged_v2(high).sum() == 0                     # nothing detected at an unreachable tau
        assert meta_h['n_windows_detected'] == 0

    def test_detection_is_monotone_in_tau(self):
        y_pre, y_post, d_sr, sigma_bands, nodata = self._case()
        counts = [flagged_v2(apply_gate_v2(y_pre, y_post, d_sr, E_V, E_B, sigma_bands, nodata, t)[0]).sum()
                  for t in (1.0, 50.0, 1e9)]
        assert counts[0] >= counts[1] >= counts[2]
        assert counts[0] > counts[2]


# ---------------------------------------------------------------- B9 ----------------------------------------------------------------

class TestSrVsBilinearRanking:
    """B9's direct regression: x9 ranked sub-pixels by np.repeat(np.repeat(d_10m,4),4), so no SR signal
    ever entered allocation. These tests fail if the allocator stops using the 2.5 m SR field."""

    def _fixture(self):
        f = np.array([[0.25, 0.50], [0.75, 0.25]])
        bil = bilinear_upsample(f, 4)                       # the naive spatial ranking field
        rng = np.random.RandomState(7)
        sr = -bil + rng.normal(0, 0.3, bil.shape)           # deliberately (anti-)ordered vs the bilinear field
        return f, sr, bil

    def test_sr_and_bilinear_rankings_select_different_sub_pixels(self):
        f, sr, bil = self._fixture()
        a_sr, n_sr = allocate_pixels(f, sr, lam=0.0)
        a_bil, n_bil = allocate_pixels(f, bil, lam=0.0)
        assert np.array_equal(n_sr, n_bil)                  # same COUNTS ...
        assert not np.array_equal(a_sr, a_bil)              # ... different SUB-PIXELS

    def test_allocation_matches_the_sr_ranking_not_the_bilinear_one(self):
        f, sr, bil = self._fixture()
        a_sr, _ = allocate_pixels(f, sr, lam=0.0)
        for i in range(f.shape[0]):
            for j in range(f.shape[1]):
                blk_sr = sr[4 * i:4 * i + 4, 4 * j:4 * j + 4].ravel()
                chosen = a_sr[4 * i:4 * i + 4, 4 * j:4 * j + 4].ravel()
                n = int(round(16 * f[i, j]))
                expected = set(np.argsort(-blk_sr, kind='stable')[:n].tolist())
                assert set(np.flatnonzero(chosen).tolist()) == expected

    def test_a_blocky_replicated_field_is_rejected(self):
        """The exact x9 construction must raise instead of being silently accepted."""
        f = np.array([[0.5, 0.5], [0.5, 0.5]])
        d_10m = np.array([[0.4, 0.1], [0.2, 0.3]])
        blocky = np.repeat(np.repeat(d_10m, 4, axis=0), 4, axis=1)
        with pytest.raises(ValueError, match='nearest-neighbour replicate'):
            allocate_pixels(f, blocky, lam=0.0)
        with pytest.raises(ValueError, match='nearest-neighbour replicate'):
            assert_sr_score_is_not_replicated(blocky)

    def test_a_real_sr_field_passes_the_guard(self):
        rng = np.random.RandomState(1)
        assert assert_sr_score_is_not_replicated(rng.normal(size=(16, 16))) == 1.0

    def test_neighbour_agreement_is_about_neighbours_only(self):
        a = np.zeros((5, 5), bool)
        a[2, 2] = True
        na = neighbour_agreement(a)
        assert na[2, 2] == 0.0                              # the pixel itself is excluded
        assert na[1, 1] == pytest.approx(1 / 8)
        assert na[0, 0] == 0.0

    def test_lambda_changes_which_sub_pixels_are_chosen_but_never_how_many(self):
        f, sr, _ = self._fixture()
        a0, n0 = allocate_pixels(f, sr, lam=0.0)
        a5, n5 = allocate_pixels(f, sr, lam=0.5)
        assert np.array_equal(n0, n5)
        assert not np.array_equal(a0, a5)


# ---------------------------------------------------------------- B7 ----------------------------------------------------------------

class TestBlockSum:
    """B7's direct regression: x4 compared h*w*100 m^2 with h*w*100 m^2. These tests compare the OUTPUT
    class map with round(16*f) and, crucially, include a case where the check MUST fail."""

    def test_exact_on_a_real_gate_run(self):
        f_true = np.zeros((32, 32))
        f_true[8:20, 8:20] = np.linspace(0.1, 0.95, 12)[None, :]
        y_pre, y_post, d_sr, sigma_bands, nodata = _scene(n_blocks=32, f_true=f_true, seed=11)
        cm, meta = apply_gate_v2(y_pre, y_post, d_sr, E_V, E_B, sigma_bands, nodata, 1.0)
        assert meta['block_sum']['pass']
        assert meta['block_sum']['max_abs_deviation_all_blocks'] == 0
        assert meta['block_sum']['n_detected_blocks'] > 0

    def test_the_check_is_not_tautological_it_can_fail(self):
        """Corrupt ONE allocated pixel in the output map and the deviation must become 1. A test that
        compares a quantity with itself cannot do this."""
        f_true = np.zeros((32, 32))
        f_true[8:20, 8:20] = 0.5
        y_pre, y_post, d_sr, sigma_bands, nodata = _scene(n_blocks=32, f_true=f_true, seed=12)
        cm, meta = apply_gate_v2(y_pre, y_post, d_sr, E_V, E_B, sigma_bands, nodata, 1.0)
        assert meta['block_sum']['max_abs_deviation_all_blocks'] == 0
        rr, cc = np.nonzero(flagged_v2(cm))
        corrupted = cm.copy()
        corrupted[rr[0], cc[0]] = NO_CHANGE
        bad = block_sum_deviation(corrupted, meta['f_effective'], nodata)
        assert bad['max_abs_deviation_all_blocks'] == 1
        assert not bad['pass']

    def test_block_sum_does_not_receive_the_allocator_count(self):
        params = inspect.signature(block_sum_deviation).parameters
        assert 'n_per_block' not in params and 'allocated' not in params
        assert list(params)[:3] == ['class_map', 'f_effective', 'nodata']

    def test_allocated_count_equals_round_16f_per_block(self):
        f = np.array([[0.0, 0.1, 0.3], [0.5, 0.7, 1.0]])
        rng = np.random.RandomState(5)
        alloc, n = allocate_pixels(f, rng.normal(size=(8, 12)), lam=0.0)
        for i in range(f.shape[0]):
            for j in range(f.shape[1]):
                assert alloc[4 * i:4 * i + 4, 4 * j:4 * j + 4].sum() == int(round(16 * f[i, j]))
                assert n[i, j] == int(round(16 * f[i, j]))


# ---------------------------------------------------------------- B11 ----------------------------------------------------------------

class TestNoDataFromScl:
    """B11's direct regression: x9 built NO_DATA from NaN pixels only and reported 512 NO_DATA px where
    v1 reported 205,956, so cloud/shadow pixels were mapped as change."""

    def test_one_scl_masked_10m_pixel_yields_exactly_16_nodata_sub_pixels(self):
        valid = np.ones((3, 8, 8), bool)
        valid[1, 5, 6] = False                            # masked on ONE date only
        nodata = nodata_from_scl(valid, 4)
        assert nodata.shape == (32, 32)
        assert nodata.sum() == 16
        assert nodata[20:24, 24:28].all()

    def test_masked_pixel_is_nodata_even_though_its_value_is_finite_and_looks_like_change(self):
        f_true = np.zeros((32, 32))
        f_true[5, 6] = 0.95                                # a big, perfectly finite "change" ...
        y_pre, y_post, d_sr, sigma_bands, _ = _scene(n_blocks=32, f_true=f_true, seed=21)
        assert np.isfinite(y_post[5, 6]).all()             # ... with no NaN anywhere, so x9 would map it
        valid = np.ones((3, 32, 32), bool)
        valid[2, 5, 6] = False                             # ... but SCL says the pixel is cloud/shadow
        nodata = nodata_from_scl(valid, 4)
        cm, meta = apply_gate_v2(y_pre, y_post, d_sr, E_V, E_B, sigma_bands, nodata, 1.0)
        block = cm[20:24, 24:28]
        assert (block == NO_DATA).all() and block.size == 16
        assert not flagged_v2(cm)[20:24, 24:28].any()
        assert meta['class_counts']['NO_DATA'] == 16

    def test_nan_only_logic_would_have_missed_it(self):
        """Explicitly contrasts the two derivations, so the regression is visible in the test itself."""
        valid = np.ones((2, 4, 4), bool)
        valid[0, 1, 1] = False
        values = np.ones((4, 4))                           # numerically fine everywhere
        nan_based = np.repeat(np.repeat(~np.isfinite(values), 4, 0), 4, 1)
        scl_based = nodata_from_scl(valid, 4)
        assert nan_based.sum() == 0
        assert scl_based.sum() == 16

    def test_nodata_blocks_never_allocate(self):
        f_true = np.full((32, 32), 0.75)
        y_pre, y_post, d_sr, sigma_bands, _ = _scene(n_blocks=32, f_true=f_true, seed=22)
        valid = np.ones((3, 32, 32), bool)
        valid[0, :4, :] = False
        nodata = nodata_from_scl(valid, 4)
        cm, meta = apply_gate_v2(y_pre, y_post, d_sr, E_V, E_B, sigma_bands, nodata, 1.0)
        assert (cm[:16, :] == NO_DATA).all()
        assert meta['f_effective'][:4, :].max() == 0.0
        assert meta['block_sum']['max_abs_deviation_all_blocks'] == 0


# ---------------------------------------------------------------- misc ----------------------------------------------------------------

class TestWindowsAndClasses:
    def test_window_max_score_shape_and_values(self):
        s = np.arange(32 * 32, dtype=float).reshape(32, 32)
        scores, valid = window_max_score(s, 16)
        assert scores.shape == (2, 2) and valid.all()
        assert scores[0, 0] == s[:16, :16].max()

    def test_window_with_no_valid_block_is_nan(self):
        s = np.ones((16, 16))
        scores, valid = window_max_score(s, 16, valid_block=np.zeros((16, 16), bool))
        assert not valid[0, 0] and np.isnan(scores[0, 0])

    def test_core_is_a_full_block_and_allocated_is_partial(self):
        f_true = np.zeros((32, 32))
        f_true[4, 4] = 1.0                                  # -> 16/16 sub-px = CORE
        f_true[4, 5] = 0.5                                  # -> 8/16 sub-px = ALLOCATED
        y_pre, y_post, d_sr, sigma_bands, nodata = _scene(n_blocks=32, f_true=f_true, seed=31)
        cm, _ = apply_gate_v2(y_pre, y_post, d_sr, E_V, E_B, sigma_bands, nodata, 1.0)
        assert (cm[16:20, 16:20] == CORE).all()
        assert (cm[16:20, 20:24] == ALLOCATED).sum() == 8

    def test_flagged_v2_excludes_unsupported(self):
        cm = np.array([[NO_CHANGE, CORE], [ALLOCATED, 3]], dtype=np.uint8)   # 3 = UNSUPPORTED
        assert flagged_v2(cm).tolist() == [[False, True], [True, False]]
