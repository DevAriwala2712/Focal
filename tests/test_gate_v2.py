"""Tests for gate v2: unmixing, allocation, conformal threshold, and end-to-end gate."""
import pytest
import numpy as np

from trustsr.gate_v2 import (
    unmix_fraction, allocate_pixels, bilinear_upsample, split_conformal_quantile,
    evaluate_conformal_on_test, max_fraction_per_window, apply_gate_v2,
    CORE, ALLOCATED, UNSUPPORTED, NO_CHANGE, NO_DATA
)


class TestUnmixFraction:
    """Test linear unmixing in reflectance."""

    def test_unmix_pure_vegetation(self):
        """Pure vegetation should give f = 0 (no loss)."""
        h, w, n_bands = 10, 10, 4
        e_v = np.array([0.03, 0.04, 0.05, 0.40])  # Red, NIR-like endmembers
        e_b = np.array([0.15, 0.10, 0.08, 0.20])

        y_pre = np.tile(e_v[np.newaxis, np.newaxis, :], (2, h, w, 1))  # Pure veg
        y_post = np.tile(e_v[np.newaxis, np.newaxis, :], (h, w, 1))     # No change

        f_hat, _ = unmix_fraction(y_pre, y_post, e_v, e_b)

        assert f_hat.shape == (h, w)
        assert np.allclose(f_hat, 0.0, atol=0.01)

    def test_unmix_pure_bare(self):
        """Pure bare ground should give f = 0 (no vegetation to lose)."""
        h, w, n_bands = 10, 10, 4
        e_v = np.array([0.03, 0.04, 0.05, 0.40])
        e_b = np.array([0.15, 0.10, 0.08, 0.20])

        y_pre = np.tile(e_b[np.newaxis, np.newaxis, :], (2, h, w, 1))  # Pure bare
        y_post = np.tile(e_b[np.newaxis, np.newaxis, :], (h, w, 1))     # No change

        f_hat, _ = unmix_fraction(y_pre, y_post, e_v, e_b)

        assert f_hat.shape == (h, w)
        assert np.allclose(f_hat, 0.0, atol=0.01)

    def test_unmix_mixed_pixel(self):
        """Mixed pixel: 50% veg + 50% bare in pre, all bare in post -> f = 0.5."""
        h, w, n_bands = 10, 10, 4
        e_v = np.array([0.03, 0.04, 0.05, 0.40])
        e_b = np.array([0.15, 0.10, 0.08, 0.20])

        # Pre: 50% veg + 50% bare
        y_pre_mixed = 0.5 * e_v + 0.5 * e_b
        y_pre = np.tile(y_pre_mixed[np.newaxis, np.newaxis, :], (2, h, w, 1))

        # Post: 100% bare
        y_post = np.tile(e_b[np.newaxis, np.newaxis, :], (h, w, 1))

        f_hat, _ = unmix_fraction(y_pre, y_post, e_v, e_b)

        assert f_hat.shape == (h, w)
        # Should be close to 0.5
        assert np.allclose(f_hat, 0.5, atol=0.05)

    def test_unmix_clipped_to_01(self):
        """Fractions should be clipped to [0, 1]."""
        h, w, n_bands = 5, 5, 4
        e_v = np.array([0.03, 0.04, 0.05, 0.40])
        e_b = np.array([0.15, 0.10, 0.08, 0.20])

        # Create outlier reflectance that would give f > 1
        y_pre = np.tile(e_v[np.newaxis, np.newaxis, :] * 2, (1, h, w, 1))
        y_post = np.tile(e_b[np.newaxis, np.newaxis, :], (h, w, 1))

        f_hat, _ = unmix_fraction(y_pre, y_post, e_v, e_b)

        assert np.all(f_hat >= 0.0) and np.all(f_hat <= 1.0)


class TestAllocation:
    """Test sub-pixel allocation."""

    def test_allocate_all_pixels(self):
        """When f = 1, all 16 pixels should be allocated."""
        h, w = 5, 5
        f_tilde = np.ones((h, w))
        delta_sr = np.random.RandomState(2024).randn(4 * h, 4 * w)
        s_spatial = np.random.RandomState(2024).randn(4 * h, 4 * w)

        allocated, n_per_block = allocate_pixels(f_tilde, delta_sr, s_spatial, lam=0.5)

        assert allocated.shape == (4 * h, 4 * w)
        assert np.all(n_per_block == 16)
        assert np.sum(allocated) == h * w * 16

    def test_allocate_no_pixels(self):
        """When f = 0, no pixels should be allocated."""
        h, w = 5, 5
        f_tilde = np.zeros((h, w))
        delta_sr = np.random.RandomState(2024).randn(4 * h, 4 * w)
        s_spatial = np.random.RandomState(2024).randn(4 * h, 4 * w)

        allocated, n_per_block = allocate_pixels(f_tilde, delta_sr, s_spatial, lam=0.5)

        assert np.all(n_per_block == 0)
        assert np.sum(allocated) == 0

    def test_allocate_half_block(self):
        """When f = 0.5, 8 pixels per block should be allocated."""
        h, w = 5, 5
        f_tilde = 0.5 * np.ones((h, w))
        delta_sr = np.random.RandomState(2024).randn(4 * h, 4 * w)
        s_spatial = np.random.RandomState(2024).randn(4 * h, 4 * w)

        allocated, n_per_block = allocate_pixels(f_tilde, delta_sr, s_spatial, lam=0.5)

        assert np.all(n_per_block == 8)
        assert np.sum(allocated) == h * w * 8

    def test_allocate_ranking(self):
        """Top-ranked pixels should be allocated."""
        h, w = 2, 2
        f_tilde = 0.5 * np.ones((h, w))

        # Create a known pattern: highest SR in top-left corner
        delta_sr = np.zeros((4 * h, 4 * w))
        delta_sr[0:2, 0:2] = 10.0  # Top-left: high score
        delta_sr[2:4, 2:4] = 1.0   # Bottom-right: low score

        s_spatial = np.ones((4 * h, 4 * w)) * 0.5

        allocated, n_per_block = allocate_pixels(f_tilde, delta_sr, s_spatial, lam=0.0)  # Pure SR ranking

        # Top-left should have most of the allocated pixels
        assert np.sum(allocated[0:2, 0:2]) >= np.sum(allocated[2:4, 2:4])


class TestBilinearUpsample:
    """Test spatial interpolation."""

    def test_upsample_shape(self):
        """Output should be 4x larger."""
        h, w = 5, 5
        f_tilde = np.random.randn(h, w)
        s_spatial = bilinear_upsample(f_tilde, scale=4)

        assert s_spatial.shape == (4 * h, 4 * w)

    def test_upsample_constant(self):
        """Constant field should remain constant."""
        h, w = 5, 5
        f_tilde = np.ones((h, w)) * 0.5
        s_spatial = bilinear_upsample(f_tilde, scale=4)

        assert np.allclose(s_spatial, 0.5, atol=0.01)

    def test_upsample_linear(self):
        """Linear ramp should be interpolated smoothly."""
        h, w = 5, 5
        x = np.linspace(0, 1, w)
        f_tilde = np.tile(x[np.newaxis, :], (h, 1))
        s_spatial = bilinear_upsample(f_tilde, scale=4)

        # Check that the fine grid has intermediate values
        assert s_spatial.min() >= 0.0
        assert s_spatial.max() <= 1.0


class TestSplitConformal:
    """Test split-conformal threshold."""

    def test_conformal_quantile_alpha_0p05(self):
        """k should correspond to alpha = 0.05 for n >= 10."""
        n = 100
        scores_cal = np.random.RandomState(2024).randn(n)

        tau, info = split_conformal_quantile(scores_cal, alpha=0.05)

        # k = ceil((100+1)(1-0.05)) = ceil(95.95) = 96
        # So tau is the 96th smallest (top 5 scores)
        assert info['k'] == 96
        assert info['n_calibration'] == n
        assert info['coverage_lower_bound'] <= 0.05

    def test_conformal_test_evaluation(self):
        """Test FAR evaluation."""
        tau = 0.0
        scores_test = np.random.RandomState(2024).randn(100)

        far, ci_lo, ci_hi, results = evaluate_conformal_on_test(scores_test, tau, alpha=0.05)

        # Scores are symmetric, so FAR should be ~0.5
        assert 0.4 < far < 0.6
        assert ci_lo < far < ci_hi
        assert results['n_test'] == 100


class TestMaxFractionPerWindow:
    """Test window scoring."""

    def test_window_shape(self):
        """Window grid should be H/window x W/window."""
        h, w = 32, 32  # 2x2 windows
        f_hat = np.random.RandomState(2024).randn(h, w)

        window_scores, window_validity = max_fraction_per_window(f_hat, window_size=16)

        assert window_scores.shape == (2, 2)
        assert window_validity.shape == (2, 2)

    def test_window_validity(self):
        """All windows should be valid by default."""
        h, w = 32, 32
        f_hat = np.random.RandomState(2024).randn(h, w)

        window_scores, window_validity = max_fraction_per_window(f_hat, window_size=16)

        assert np.all(window_validity)

    def test_window_values_are_maxima(self):
        """Window scores should be the max within each window."""
        h, w = 32, 32
        f_hat = np.random.RandomState(2024).randn(h, w)

        window_scores, _ = max_fraction_per_window(f_hat, window_size=16)

        # Check a specific window
        iw, jw = 0, 0
        expected_max = np.max(f_hat[0:16, 0:16])
        assert np.isclose(window_scores[iw, jw], expected_max)


class TestApplyGateV2EndToEnd:
    """Test full gate v2 pipeline."""

    def test_gate_v2_output_shape(self):
        """Output class map should match input 2.5m grid."""
        h_10m, w_10m = 10, 10
        n_bands = 4
        n_dates = 3

        y_pre = np.random.RandomState(2024).randn(n_dates, h_10m, w_10m, n_bands) * 0.1 + 0.3
        y_post = np.random.RandomState(2024).randn(h_10m, w_10m, n_bands) * 0.1 + 0.3
        d_sr = np.random.RandomState(2024).randn(4 * h_10m, 4 * w_10m) * 0.05
        e_v = np.array([0.03, 0.04, 0.05, 0.40])
        e_b = np.array([0.15, 0.10, 0.08, 0.20])
        sigma = np.ones((4 * h_10m, 4 * w_10m)) * 0.05
        nodata = np.zeros((4 * h_10m, 4 * w_10m), dtype=bool)

        config = {
            'k': 2.0,
            'lam': 0.5,
            'alpha': 0.05,
            'window_size': 16,
            'seed': 2024,
        }

        class_map, meta = apply_gate_v2(y_pre, y_post, d_sr, e_v, e_b, sigma, nodata, config)

        assert class_map.shape == (4 * h_10m, 4 * w_10m)
        assert meta['f_hat'].shape == (h_10m, w_10m)
        assert meta['n_per_block'].shape == (h_10m, w_10m)

    def test_gate_v2_no_data_preserved(self):
        """NO_DATA pixels should remain NO_DATA."""
        h_10m, w_10m = 5, 5
        n_bands = 4
        n_dates = 2

        y_pre = np.ones((n_dates, h_10m, w_10m, n_bands)) * 0.3
        y_post = np.ones((h_10m, w_10m, n_bands)) * 0.3
        d_sr = np.zeros((4 * h_10m, 4 * w_10m))
        e_v = np.array([0.03, 0.04, 0.05, 0.40])
        e_b = np.array([0.15, 0.10, 0.08, 0.20])
        sigma = np.ones((4 * h_10m, 4 * w_10m)) * 0.05

        nodata = np.zeros((4 * h_10m, 4 * w_10m), dtype=bool)
        nodata[0:8, 0:8] = True  # Mark some pixels as NO_DATA

        config = {'k': 2.0, 'lam': 0.5, 'alpha': 0.05, 'window_size': 4, 'seed': 2024}

        class_map, _ = apply_gate_v2(y_pre, y_post, d_sr, e_v, e_b, sigma, nodata, config)

        assert np.all(class_map[nodata] == NO_DATA)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
