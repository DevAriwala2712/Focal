"""Tests for the placebo (null-event) test harness (A3)."""
from __future__ import annotations

import numpy as np
import pytest

from trustsr.placebo import (
    NO_CHANGE, NO_DATA, OBSERVED, UNSUPPORTED,
    checkerboard_split, compute_far_pixel, compute_far_window, expand_checkerboard,
)


class TestCheckerboard:
    """Test spatial checkerboard split."""

    def test_checkerboard_split_shape(self):
        """Test that checkerboard split has the right shape."""
        split = checkerboard_split((256, 512), tile_px=128)
        assert split.shape == (2, 4)

    def test_checkerboard_parity(self):
        """Test that checkerboard alternates parity."""
        split = checkerboard_split((256, 512), tile_px=128)
        # (0,0) is even (False), (0,1) is odd (True), etc.
        assert split[0, 0] == False  # even parity
        assert split[0, 1] == True  # odd parity
        assert split[1, 0] == True  # odd parity
        assert split[1, 1] == False  # even parity

    def test_expand_checkerboard_shape(self):
        """Test that expansion to pixel level has the right shape."""
        split = checkerboard_split((256, 512), tile_px=128)
        pixel_split = expand_checkerboard(split, 128, (256, 512))
        assert pixel_split.shape == (256, 512)

    def test_expand_checkerboard_consistency(self):
        """Test that expanded checkerboard is consistent within tiles."""
        split = checkerboard_split((256, 512), tile_px=128)
        pixel_split = expand_checkerboard(split, 128, (256, 512))
        # All pixels in tile (0, 0) should be False
        assert np.all(pixel_split[0:128, 0:128] == False)
        # All pixels in tile (0, 1) should be True
        assert np.all(pixel_split[0:128, 128:256] == True)


class TestFARPixel:
    """Test pixel-level FAR computation."""

    def test_far_pixel_basic(self):
        """Test pixel FAR with synthetic data."""
        flagged = np.zeros((256, 256), dtype=bool)
        flagged[0:128, 0:128] = True  # 50% of pixels in first tile
        valid = np.ones((256, 256), dtype=bool)
        tile_split = np.array([[False, True], [True, False]])

        result = compute_far_pixel(flagged, valid, tile_split, 128, replicates=100, seed=42)

        # Test split has 1 True tile (50% FAR) and 1 False tile (0% FAR)
        # Expected: 50% of valid pixels in test split are flagged
        assert 'estimate' in result
        assert 'lo' in result
        assert 'hi' in result
        assert result['estimate'] > 0.4  # Rough check

    def test_far_pixel_zero(self):
        """Test pixel FAR with no flags."""
        flagged = np.zeros((256, 256), dtype=bool)
        valid = np.ones((256, 256), dtype=bool)
        tile_split = np.array([[False, True], [True, False]])

        result = compute_far_pixel(flagged, valid, tile_split, 128, replicates=100, seed=42)

        assert result['estimate'] == 0.0

    def test_far_pixel_with_invalid(self):
        """Test pixel FAR with invalid pixels."""
        flagged = np.zeros((256, 256), dtype=bool)
        flagged[0:128, 0:128] = True
        valid = np.ones((256, 256), dtype=bool)
        valid[0:64, 0:64] = False
        tile_split = np.array([[False, True], [True, False]])

        result = compute_far_pixel(flagged, valid, tile_split, 128, replicates=100, seed=42)

        # Only valid pixels count
        assert 'estimate' in result
        assert result['denominator'] > 0


class TestFARWindow:
    """Test window-level FAR computation."""

    def test_far_window_basic(self):
        """Test window FAR with synthetic data."""
        parent_10m = np.zeros((16, 16), dtype=bool)
        parent_10m[0:8, 0:8] = True
        valid = np.ones((16, 16), dtype=bool)
        tile_split = np.array([[False, True], [True, False]])

        result = compute_far_window(
            parent_10m, valid, tile_split, tile_px=128, window_10m_px=16,
            replicates=100, seed=42
        )

        assert 'estimate' in result
        assert 'lo' in result
        assert 'hi' in result

    def test_far_window_zero(self):
        """Test window FAR with no flags."""
        parent_10m = np.zeros((16, 16), dtype=bool)
        valid = np.ones((16, 16), dtype=bool)
        tile_split = np.array([[False, True], [True, False]])

        result = compute_far_window(
            parent_10m, valid, tile_split, tile_px=128, window_10m_px=16,
            replicates=100, seed=42
        )

        assert result['estimate'] == 0.0


class TestDeterminism:
    """Test that the harness is deterministic."""

    def test_checkerboard_deterministic(self):
        """Test that checkerboard split is deterministic."""
        split1 = checkerboard_split((512, 512), tile_px=128, seed=2024)
        split2 = checkerboard_split((512, 512), tile_px=128, seed=2024)
        assert np.array_equal(split1, split2)

    def test_bootstrap_seed_consistency(self):
        """Test that bootstrap with same seed gives same results."""
        flagged = np.random.RandomState(42).rand(256, 256) > 0.9
        valid = np.ones((256, 256), dtype=bool)
        tile_split = checkerboard_split((256, 256), 128, seed=2024)

        result1 = compute_far_pixel(flagged, valid, tile_split, 128, replicates=50, seed=42)
        result2 = compute_far_pixel(flagged, valid, tile_split, 128, replicates=50, seed=42)

        assert result1['estimate'] == result2['estimate']
        assert result1['lo'] == result2['lo']
        assert result1['hi'] == result2['hi']
