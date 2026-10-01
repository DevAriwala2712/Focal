"""Placebo (null-event) test harness for the TrustSR gate (A3).

Run the full pipeline on pre-vs-pre image pairs where no event occurred (leave-one-out among
the clear Jan–May 2024 Sentinel-2 dates on Wayanad). Every flagged change is a false alarm by
construction. Score the false-alarm rate (pixel and object level) for: 10 m rule, ungated SR,
gate v1, gate v1 with A5's new σ, and gate v2 (when ready).

This is a reusable harness that takes a gate-scoring function and outputs per-gate FAR estimates
with block-bootstrap CIs.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, TypeAlias

import numpy as np

from trustsr.bootstrap import block_sums, ratio_bootstrap_ci

logger = logging.getLogger(__name__)

# Type aliases
GateScorer: TypeAlias = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float], np.ndarray]
"""Gate scorer: (d, sigma, parent_hr, nodata, k) -> class array"""


# Class constants (from gate.py)
NO_CHANGE, OBSERVED, INFERRED, UNSUPPORTED, NO_DATA = 0, 1, 2, 3, 255
CLASS_NAMES = {
    NO_CHANGE: 'NO_CHANGE',
    OBSERVED: 'OBSERVED',
    INFERRED: 'INFERRED',
    UNSUPPORTED: 'UNSUPPORTED',
    NO_DATA: 'NO_DATA',
}


def checkerboard_split(shape: tuple[int, int], tile_px: int, seed: int = 2024) -> np.ndarray:
    """Spatial checkerboard split of (H, W) into 128 px tiles: even parity = calibration, odd = test.

    Returns (H_tiles, W_tiles) array of bool, True = test (odd parity), False = calibration (even).
    Tile assignment is based on grid position (ti, tj) in the tile coordinate system.
    """
    h, w = shape
    h_tiles = -(-h // tile_px)
    w_tiles = -(-w // tile_px)
    split = np.zeros((h_tiles, w_tiles), dtype=bool)
    for ti in range(h_tiles):
        for tj in range(w_tiles):
            parity = (ti + tj) % 2
            split[ti, tj] = bool(parity)
    return split


def expand_checkerboard(split: np.ndarray, tile_px: int, shape: tuple[int, int]) -> np.ndarray:
    """Expand tile-level checkerboard split to per-pixel split.

    Returns (H, W) bool array, True = test split.
    """
    h, w = shape
    h_tiles, w_tiles = split.shape
    pixel_split = np.zeros((h, w), dtype=bool)
    for ti in range(h_tiles):
        for tj in range(w_tiles):
            r0 = ti * tile_px
            r1 = min((ti + 1) * tile_px, h)
            c0 = tj * tile_px
            c1 = min((tj + 1) * tile_px, w)
            pixel_split[r0:r1, c0:c1] = split[ti, tj]
    return pixel_split


def compute_far_pixel(
    flagged: np.ndarray, valid: np.ndarray, tile_split: np.ndarray, tile_px: int,
    replicates: int = 2000, ci: float = 0.95, seed: int = 2024,
) -> dict:
    """Compute pixel-level FAR with block bootstrap.

    Per-block sums are computed over 128 px (10 m) tiles of the flagged/valid pixels.

    Args:
        flagged: (H, W) bool array, pixels flagged as change
        valid: (H, W) bool array, valid pixels (not cloud/nodata)
        tile_split: (H_tiles, W_tiles) bool array, test split at tile level (True = test)
        tile_px: tile size in pixels (128)
    """
    flagged = np.asarray(flagged, dtype=float)
    valid = np.asarray(valid, dtype=float)

    # Expand tile split to pixel level
    pixel_split = expand_checkerboard(tile_split, tile_px, flagged.shape)

    # Compute per-tile sums for test split only
    num = block_sums(np.where(pixel_split, flagged, 0.0), tile_px)
    den = block_sums(np.where(pixel_split, valid, 0.0), tile_px)
    result = ratio_bootstrap_ci(num, den, replicates, ci, seed)
    result['unit'] = 'pixel'
    result['definition'] = 'flagged valid 2.5 m px / valid 2.5 m px, area weighted (px = 6.25 m²), test split only'
    return result


def compute_far_window(
    flagged: np.ndarray, valid: np.ndarray, tile_split: np.ndarray, tile_px: int,
    window_10m_px: int = 16,
    replicates: int = 2000, ci: float = 0.95, seed: int = 2024,
) -> dict:
    """Compute window-level FAR with block bootstrap.

    Windows are 16×16 10 m pixels (160 m). FAR = fraction of windows with >= 1 flagged pixel.
    Block bootstrap is over 128 px (10 m) tiles.
    """
    flagged = np.asarray(flagged, dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    h, w = flagged.shape
    h_windows = -(-h // window_10m_px)
    w_windows = -(-w // window_10m_px)

    # Upsample the 10 m parent to 2.5 m and compute window indicators
    window_flagged = np.zeros((h_windows, w_windows), dtype=bool)
    window_valid = np.zeros((h_windows, w_windows), dtype=bool)

    for wi in range(h_windows):
        for wj in range(w_windows):
            r0 = wi * window_10m_px
            r1 = min((wi + 1) * window_10m_px, h)
            c0 = wj * window_10m_px
            c1 = min((wj + 1) * window_10m_px, w)
            window = flagged[r0:r1, c0:c1]
            valid_window = valid[r0:r1, c0:c1]
            window_flagged[wi, wj] = window.any()
            window_valid[wi, wj] = valid_window.any()

    # Map windows to tiles (128 px 10 m = 32×32 2.5 m pixels)
    # A window starts at 10 m indices (wi, wj); its tile is (ti, tj) where ti = wi * 16 // 128, etc.
    # Actually, simpler: we have window array that's already 10 m indexed.
    # Each 128 px 10 m tile contains 8×8 windows (128 / 16 = 8).
    # Let's remap windows to tiles.
    h_tiles = -(-h // tile_px)
    w_tiles = -(-w // tile_px)
    tile_window_flagged = np.zeros((h_tiles, w_tiles), dtype=float)
    tile_window_valid = np.zeros((h_tiles, w_tiles), dtype=float)

    for wi in range(h_windows):
        for wj in range(w_windows):
            ti = wi // (tile_px // window_10m_px)
            tj = wj // (tile_px // window_10m_px)
            if ti < h_tiles and tj < w_tiles:
                # Count windows in this tile
                tile_window_flagged[ti, tj] += float(window_flagged[wi, wj])
                tile_window_valid[ti, tj] += float(window_valid[wi, wj])

    # Bootstrap on test split: count total flagged / total valid windows per tile
    num = np.where(tile_split, tile_window_flagged, 0.0).ravel()
    den = np.where(tile_split, tile_window_valid, 0.0).ravel()
    result = ratio_bootstrap_ci(num, den, replicates, ci, seed)
    result['unit'] = 'window'
    result['definition'] = 'fraction of 16×16 10 m px windows (160 m) with ≥1 flagged pixel, test split only'
    result['window_size_10m_px'] = window_10m_px
    return result


@dataclass
class PlaceboResult:
    """Results from placebo test for one gate and one date pair."""
    evidence: str = 'real'
    gate_name: str = ''
    pre_dates: list[str] = field(default_factory=list)
    post_date: str = ''
    pixel_far: dict = field(default_factory=dict)
    window_far: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'gate': self.gate_name,
            'pre_dates': self.pre_dates,
            'post_date': self.post_date,
            'pixel_far': self.pixel_far,
            'window_far': self.window_far,
        }


def run_placebo(
    data_dict: dict,
    date_pairs: list[tuple[str, str, str]],
    gate_scorers: dict[str, GateScorer],
    exclude_region: np.ndarray | None = None,
    tile_px: int = 128,
    k: float = 2.0,
    replicates: int = 2000,
    ci: float = 0.95,
    seed: int = 2024,
) -> dict:
    """Run placebo test on pre-vs-pre pairs.

    Args:
        data_dict: dict with keys like 'ndvi_2024-01-16', 'dihedral_pre_means' (per date), 'valid_all', etc.
        date_pairs: list of (pre_date1, pre_date2, post_date) tuples; each pair is scored
        gate_scorers: dict {gate_name: scorer_function}
        exclude_region: (H, W) bool mask, True = exclude from FAR (e.g., slide footprint)
        tile_px: tile size for checkerboard split and block bootstrap (default 128)
        k: threshold multiplier for sigma (default 2.0)
        replicates: bootstrap replicates (default 2000)
        ci: confidence interval (default 0.95)
        seed: random seed (default 2024)

    Returns:
        dict with per-gate results, effective dates, pairs scored, etc.
    """
    results = {}
    all_pairs = []

    for pre1, pre2, post in date_pairs:
        logger.info(f"Processing pair {pre1} vs {pre2} (vs post {post})")

        # Load NDVI means and stds for the pre dates (pre-event pool)
        # Assume data_dict has 'dihedral_pre_means', 'dihedral_pre_stds' keyed by date
        pre_means = np.stack([data_dict[f'dihedral_pre_means_{d}'] for d in [pre1, pre2]])
        pre_stds = np.stack([data_dict[f'dihedral_pre_stds_{d}'] for d in [pre1, pre2]])
        post_mean = data_dict[f'dihedral_post_mean_{post}']
        post_std = data_dict[f'dihedral_post_std_{post}']
        valid_all = data_dict['valid_all']
        parent_10m = data_dict['parent_10m']

        # Compute means and stds
        pre_mean = pre_means.mean(axis=0)
        pre_std_pooled = np.sqrt(np.mean(pre_stds ** 2, axis=0))  # Mean of per-date stds

        # Upsample parent to 2.5 m (scale = 4)
        scale = 4
        parent_hr = np.repeat(np.repeat(parent_10m, scale, axis=0), scale, axis=1)
        valid_hr = np.repeat(np.repeat(valid_all, scale, axis=0), scale, axis=1)

        # Compute change
        d = pre_mean - post_mean
        sigma = np.sqrt(pre_std_pooled ** 2 + post_std ** 2)

        # Exclude region
        if exclude_region is not None:
            nodata = ~valid_hr | exclude_region
        else:
            nodata = ~valid_hr

        # Checkerboard split
        tile_split_2d = checkerboard_split(d.shape, tile_px, seed)
        tile_split = expand_checkerboard(tile_split_2d, tile_px, d.shape)

        # Limit to test split
        test_mask = tile_split & ~nodata

        # Score each gate
        pair_key = f"{pre1}_{pre2}_vs_{post}"
        if pair_key not in results:
            results[pair_key] = {}

        for gate_name, scorer in gate_scorers.items():
            cls = scorer(d, sigma, parent_hr, nodata, k)

            # Gated pixels: not NO_CHANGE and not NO_DATA
            gated = (cls != NO_CHANGE) & (cls != NO_DATA)

            # FAR: fraction of flagged pixels in the test split
            flagged = gated & test_mask
            valid = test_mask

            pixel_far = compute_far_pixel(flagged, valid, tile_split_2d, tile_px, replicates, ci, seed)
            window_far = compute_far_window(
                parent_10m, valid_all, tile_split_2d, tile_px,
                window_10m_px=16, replicates=replicates, ci=ci, seed=seed
            )

            result = PlaceboResult(
                gate_name=gate_name,
                pre_dates=[pre1, pre2],
                post_date=post,
                pixel_far=pixel_far,
                window_far=window_far,
            )
            results[pair_key][gate_name] = result.to_dict()
            all_pairs.append((pre1, pre2, post))

    # Compute effective independent dates
    unique_dates = set()
    for pre1, pre2, post in all_pairs:
        unique_dates.add(pre1)
        unique_dates.add(pre2)
        unique_dates.add(post)

    return {
        'results_by_pair': results,
        'pairs_scored': all_pairs,
        'effective_unique_dates': len(unique_dates),
        'date_list': sorted(unique_dates),
    }
