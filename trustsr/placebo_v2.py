"""F1: the real placebo (null-event) false-alarm-rate harness, replacing trustsr/placebo.py's `run_placebo`
(quarantined together with experiments/x3_placebo.py; see experiments/results/INVALIDATED.md).

Every fix below is pinned to a B-ID from the A10 audit and to configs/fix.yaml's `f1_placebo` block:

  B1  Every placebo pair's reference is the MEAN OF THE OTHER PRE-EVENT DATES in the pool. The post-event
      image (2024-12-06, after the 2024-07-30 event) never enters a pair. `guarded_load_date` enforces this
      as a hard, unit-tested guard: it raises for any date >= EVENT_DATE, so a placebo code path literally
      cannot load the post image.
  B2  `GATES` below are five distinct functions with different bodies. None aliases another (test:
      `test_gates_differ_on_fixture`).
  B3  `compute_far_window_v2` is given each gate's OWN flagged array. Swapping the flag map changes the
      window FAR (test: `test_window_far_changes_with_flag_map`). Contrast with trustsr.placebo.run_placebo,
      which always called compute_far_window on the raw `parent_10m` regardless of which gate produced it.
  B4  `pool_fold_blocks` concatenates the per-tile (numerator, denominator) blocks of every fold BEFORE calling
      `ratio_bootstrap_ci` once, so the reported CI is the CI of the pooled point estimate, not one fold's CI
      sitting next to a different pooled number.
  B5  `flagged_v1` = OBSERVED union INFERRED (configs/fix.yaml `units.flagged_definition.v1`), never "every
      class that isn't NO_CHANGE/NO_DATA". UNSUPPORTED is reported separately, never counted as flagged.

Scope (stated plainly, not smoothed over): the pretrained model's 2.5 m SR dihedral-mean/variance products
(`data/experiments-cache/wayanad_evidence/per_date_ndvi/*.npy`) exist, from the already-audited v1 pipeline,
only for the three originally-cached January 2024 dates (2024-01-16, 2024-01-21, 2024-01-26) on the
corrected AOI crop (rows 256:896, cols 128:640 of the 1024x1024 10 m grid). Extending the SR-dependent gates
(ungated_S, gate_v1, gate_v1_with_a5_sigma) to new dates would require re-running the E1 tiler + E8 dihedral
SR pipeline per date, which was out of the time budget for this run. What WAS extended with real, newly
fetched Sentinel-2 L2A data (Planetary Computer STAC, same pipeline as experiments/wayanad_evidence/fetch.py)
are three more clear dates found by a STAC/SCL audit (2024-02-05, 2024-02-10, 2024-02-15) used for the
`rule_10m` gate only, since that gate needs no SR product. See `experiments/f1_placebo.py` for the STAC audit
log and `experiments/results/f1.json` for exactly which gate ran on which pool.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from experiments.wayanad_evidence.gate import (
    INFERRED, NO_CHANGE, NO_DATA, OBSERVED, UNSUPPORTED, classify, parent_mask, upsample,
)
from trustsr.bootstrap import block_sums, ratio_bootstrap_ci
from trustsr.noise import predictive_sigma_from_dates, sigma_v1
from trustsr.placebo import checkerboard_split, expand_checkerboard

EVENT_DATE = '2024-07-30'       # the landslide; the only post-event image is 2024-12-06

GATE_NAMES = ('rule_10m', 'ungated_S_v1_sigma', 'gate_v1', 'gate_v1_with_a5_sigma', 'gate_v2')


class PostEventDateBlocked(ValueError):
    """Raised by `guarded_load_date` for any date on/after EVENT_DATE (B1's hard guard)."""


def guarded_load_date(cfg, cache_dir, date: str):
    """The ONLY sanctioned entry point for placebo-v2 code to load a Sentinel-2 date.

    Raises PostEventDateBlocked for date >= EVENT_DATE ('2024-07-30'), so no placebo pair can ever contain
    the real post-event image. Delegates to experiments.wayanad_evidence.data.load_date (unchanged, v1 code)
    for anything earlier.
    """
    if str(date) >= EVENT_DATE:
        raise PostEventDateBlocked(f'{date} is on or after the event date {EVENT_DATE}; a placebo pair may never load it')
    from experiments.wayanad_evidence.data import load_date
    return load_date(cfg, cache_dir, date)


# ---------------------------------------------------------------- fold data ----------------------------------------------------------------

@dataclass
class Fold:
    """One leave-one-out placebo fold: `held_out` stands in for a 'post' image; `pre_dates` are pooled as the
    reference. Every field the gates need is precomputed here so gate functions are pure (d, sigma, mask) -> flagged.
    """
    held_out: str
    pre_dates: list[str]
    has_sr: bool                       # 2.5 m dihedral SR product available for every date in this fold
    parent_hr: np.ndarray = None       # (H,W) bool at 2.5 m, upsampled 10 m parent-drop mask
    nodata: np.ndarray = None          # (H,W) bool at 2.5 m
    d: np.ndarray = None               # (H,W) float, SR NDVI drop (mean pre - held_out), 2.5 m; None if not has_sr
    sigma_v1: np.ndarray = None        # (H,W) float; None if not has_sr
    sigma_a5: np.ndarray = None        # (H,W) float; None if not has_sr
    n_pre: int = 0


def _crop_10m(a, crop):
    r0, r1, c0, c1 = crop
    return a[r0:r1, c0:c1]


def _load_dihedral(cache_root: Path, date: str) -> np.ndarray:
    """(8, H, W) float32 dihedral-run SR NDVI for one date, from the v1 pipeline's per_date_ndvi cache."""
    path = cache_root / 'per_date_ndvi' / f'{date}_dihedral_means.npy'
    if not path.is_file():
        raise FileNotFoundError(f'{path} not cached; SR-dependent gates cannot score this date')
    return np.load(path)


def build_sr_fold(cfg, cache_dir: Path, cache_root: Path, crop, held_out: str, pre_dates: list[str], threshold: float) -> Fold:
    """A fold with a real 2.5 m SR product on both sides (all `pre_dates` and `held_out`)."""
    from experiments.wayanad_evidence.data import scl_valid

    all_dates = pre_dates + [held_out]
    ten_m = {d: guarded_load_date(cfg, cache_dir, d) for d in all_dates}
    valid_all_10 = np.logical_and.reduce([ten_m[d]['valid'] for d in all_dates])
    ndvi_pre_mean_10 = np.mean([ten_m[d]['ndvi'] for d in pre_dates], axis=0)
    drop_10 = np.where(valid_all_10, ndvi_pre_mean_10 - ten_m[held_out]['ndvi'], np.nan).astype('float32')
    parent_10 = parent_mask(drop_10, valid_all_10, threshold)
    parent_10c = _crop_10m(parent_10, crop)
    valid_10c = _crop_10m(valid_all_10, crop)
    parent_hr = upsample(parent_10c, 4)
    valid_hr_from_10m = upsample(valid_10c, 4)

    dih = {d: _load_dihedral(cache_root, d) for d in all_dates}
    full = np.ones(dih[all_dates[0]].shape[1:], bool)
    for d in all_dates:
        full &= np.isfinite(dih[d]).sum(axis=0) == dih[d].shape[0]
    nodata = ~valid_hr_from_10m | ~full

    with np.errstate(invalid='ignore'):
        pre_means = np.stack([np.nanmean(dih[d], axis=0) for d in pre_dates])
        pre_vars = np.stack([np.nanvar(dih[d], axis=0, ddof=1) for d in pre_dates])
        post_mean = np.nanmean(dih[held_out], axis=0)
        post_var = np.nanvar(dih[held_out], axis=0, ddof=1)

    pre_mean = pre_means.mean(axis=0)
    d = pre_mean - post_mean
    pooled_pre_var = np.mean(pre_vars, axis=0)               # mean of per-date dihedral variances (v1's sigma_pre^2 term)
    sig_v1 = sigma_v1(np.sqrt(pooled_pre_var), np.sqrt(post_var))

    valid_fit = ~nodata
    sig_a5, a5_info = predictive_sigma_from_dates(
        pre_means, pre_vars, post_var, method='raw', valid=valid_fit)

    return Fold(held_out=held_out, pre_dates=pre_dates, has_sr=True, parent_hr=parent_hr, nodata=nodata,
                d=d.astype('float64'), sigma_v1=sig_v1, sigma_a5=sig_a5, n_pre=len(pre_dates))


def build_10m_fold(cfg, cache_dir: Path, crop, held_out: str, pre_dates: list[str], threshold: float) -> Fold:
    """A fold usable ONLY by rule_10m (no SR product required): parent mask at 10 m, upsampled to 2.5 m."""
    all_dates = pre_dates + [held_out]
    ten_m = {d: guarded_load_date(cfg, cache_dir, d) for d in all_dates}
    valid_all_10 = np.logical_and.reduce([ten_m[d]['valid'] for d in all_dates])
    ndvi_pre_mean_10 = np.mean([ten_m[d]['ndvi'] for d in pre_dates], axis=0)
    drop_10 = np.where(valid_all_10, ndvi_pre_mean_10 - ten_m[held_out]['ndvi'], np.nan).astype('float32')
    parent_10 = parent_mask(drop_10, valid_all_10, threshold)
    parent_10c = _crop_10m(parent_10, crop)
    valid_10c = _crop_10m(valid_all_10, crop)
    return Fold(held_out=held_out, pre_dates=pre_dates, has_sr=False,
                parent_hr=upsample(parent_10c, 4), nodata=~upsample(valid_10c, 4), n_pre=len(pre_dates))


# ---------------------------------------------------------------- exclusions ----------------------------------------------------------------

def exclusion_mask_10m(cfg, transform, crop, footprint_full_aoi: np.ndarray) -> np.ndarray:
    """1500 m disks around the published plausibility points, union the (already-dilated) landslide footprint,
    both reused verbatim from configs/wayanad_evidence.yaml / experiments/wayanad_evidence outputs (never redefined).
    Returns a (H,W) bool at 10 m on the crop grid, True = excluded.
    """
    from pyproj import Transformer
    r0, r1, c0, c1 = crop
    fp = footprint_full_aoi[r0:r1, c0:c1]
    shape = fp.shape
    to_metric = Transformer.from_crs('EPSG:4326', cfg['aoi']['crs'], always_xy=True)
    radius = cfg['aoi']['plausibility_radius_m']
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    px_x = transform.c + (c0 + xx + 0.5) * transform.a
    px_y = transform.f + (r0 + yy + 0.5) * transform.e
    disks = np.zeros(shape, bool)
    for pt in cfg['aoi']['plausibility_points'].values():
        x, y = to_metric.transform(pt['lon'], pt['lat'])
        disks |= (np.hypot(px_x - x, px_y - y) <= radius)
    return fp | disks


# ---------------------------------------------------------------- gates (B2: five distinct functions) ----------------------------------------------------------------

def flagged_v1(cls: np.ndarray) -> np.ndarray:
    """B5: the ONLY definition of 'flagged' for a v1-style class map. UNSUPPORTED is deliberately excluded."""
    return (cls == OBSERVED) | (cls == INFERRED)


def gate_rule_10m(fold: Fold, k: float) -> np.ndarray:
    """The 10 m parent rule alone: flagged wherever the upsampled 10 m NDVI-drop mask is positive. No SR, no sigma."""
    return fold.parent_hr & ~fold.nodata


def gate_ungated_s_v1_sigma(fold: Fold, k: float) -> np.ndarray:
    """S alone (v1 sigma), NEVER gated by the parent mask: the 'ungated SR' baseline the keep rule compares against."""
    with np.errstate(invalid='ignore'):
        s = fold.d > k * fold.sigma_v1
    return s & ~fold.nodata


def gate_v1(fold: Fold, k: float) -> np.ndarray:
    """OBSERVED union INFERRED from experiments.wayanad_evidence.gate.classify with the v1 sigma."""
    cls = classify(fold.d, fold.sigma_v1, fold.parent_hr, fold.nodata, k)
    return flagged_v1(cls)


def gate_v1_with_a5_sigma(fold: Fold, k: float) -> np.ndarray:
    """OBSERVED union INFERRED from the same classify(), but with the A5 predictive sigma (trustsr.noise)."""
    cls = classify(fold.d, fold.sigma_a5, fold.parent_hr, fold.nodata, k)
    return flagged_v1(cls)


def gate_v2_blocked(fold: Fold, k: float) -> np.ndarray:
    """gate_v2 does not exist yet (lands in F2). NEVER a copy of gate_v1's output — raises."""
    raise NotImplementedError('gate_v2 is BLOCKED until F2 lands (configs/fix.yaml f2_gate_v2); '
                              'this function must never be aliased to gate_v1')


GATES = {
    'rule_10m': gate_rule_10m,
    'ungated_S_v1_sigma': gate_ungated_s_v1_sigma,
    'gate_v1': gate_v1,
    'gate_v1_with_a5_sigma': gate_v1_with_a5_sigma,
    'gate_v2': gate_v2_blocked,
}
SR_REQUIRED_GATES = ('ungated_S_v1_sigma', 'gate_v1', 'gate_v1_with_a5_sigma', 'gate_v2')


# ---------------------------------------------------------------- FAR (B3, B4) ----------------------------------------------------------------

def compute_far_pixel_v2(flagged, valid, tile_split_2d, tile_px, replicates, ci, seed):
    """Pixel FAR, block-bootstrapped over `tile_px`-px tiles, on exactly the split passed in (`tile_split_2d`
    True = included). Uses THIS gate's own `flagged`/`valid` arrays (never another gate's).
    """
    pixel_split = expand_checkerboard(tile_split_2d, tile_px, flagged.shape)
    num = block_sums(np.where(pixel_split, flagged, 0.0).astype(np.float64), tile_px)
    den = block_sums(np.where(pixel_split, valid, 0.0).astype(np.float64), tile_px)
    out = ratio_bootstrap_ci(num, den, replicates, ci, seed)
    out.update(unit='pixel', definition='flagged valid 2.5 m px / valid 2.5 m px, area weighted (px = 6.25 m^2)')
    return out, num, den


def window_indicators(flagged, valid, window_px: int):
    """(H/window_px, W/window_px) bool: a window is flagged/valid if >=1 of its pixels is flagged/valid."""
    return block_sums(flagged.astype(np.float64), window_px) > 0, block_sums(valid.astype(np.float64), window_px) > 0


def compute_far_window_v2(flagged, valid, tile_split_2d, tile_px, window_px, replicates, ci, seed):
    """B3's direct fix: window FAR computed from THIS gate's own `flagged` array, never a shared `parent_10m`.

    Windows are `window_px`-px squares (16x16 10 m px = 64x64 2.5 m px -> window_px=64). Block bootstrap units
    are the `tile_px`-px tiles (128 px at 2.5 m); each tile holds (tile_px // window_px)^2 windows.
    """
    if tile_px % window_px:
        raise ValueError('tile_px must be a whole multiple of window_px')
    ratio = tile_px // window_px
    win_flag, win_valid = window_indicators(flagged, valid, window_px)
    tile_split_windows = np.repeat(np.repeat(tile_split_2d, ratio, axis=0), ratio, axis=1)
    if tile_split_windows.shape != win_flag.shape:
        raise ValueError(f'grid mismatch: {tile_split_windows.shape} vs {win_flag.shape}; crop dims must divide tile_px')
    num = block_sums(np.where(tile_split_windows, win_flag, 0.0).astype(np.float64), ratio)
    den = block_sums(np.where(tile_split_windows, win_valid, 0.0).astype(np.float64), ratio)
    out = ratio_bootstrap_ci(num, den, replicates, ci, seed)
    out.update(unit='window', definition='fraction of 16x16 10 m px windows (160 m) with >=1 flagged px, this gate\'s own flag map',
              window_px_2p5m=window_px)
    return out, num, den


def pool_fold_blocks(per_fold_num_den: list[tuple[np.ndarray, np.ndarray]], replicates, ci, seed):
    """B4's direct fix: concatenate every fold's per-tile (num, den) blocks and bootstrap ONCE, so the CI is
    of the SAME pooled point estimate that is reported, not a single fold's CI shown beside a pooled number.
    """
    num = np.concatenate([n.ravel() for n, _ in per_fold_num_den])
    den = np.concatenate([d.ravel() for _, d in per_fold_num_den])
    return ratio_bootstrap_ci(num, den, replicates, ci, seed)
