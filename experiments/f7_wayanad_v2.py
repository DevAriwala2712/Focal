"""F7: the Wayanad v2 production rerun. Replaces the INVALIDATED experiments/x9_v2_production.py.

Pre-registered in `configs/fix.yaml` `f7_wayanad_v2` (Wave 0, before this script existed). This is the
real-event map: 3 pre dates + the single post-event date, the corrected AOI (11.490 N, 76.160 E), the
real tiled SEN2SR-lite SR with 8 dihedral runs, F3's per-date normalisation, and F2's gate v2.

  ############################################################################################
  #  READ THIS FIRST.  The gate this script runs (F2's gate v2) FAILED its own detection      #
  #  keep rule.  On F1's placebo test split its window false-alarm rate is                    #
  #      0.0652  [0.0478, 0.0850]   against a pre-registered requirement of <= 0.05.          #
  #  Everything below therefore describes WHAT THIS GATE CONFIGURATION PRODUCES on the real   #
  #  landslide.  It is NOT a validated detector and none of these numbers may be presented    #
  #  as a validated detection of the Wayanad scar.  F7's status is FAIL for that reason.      #
  #  Related, independent (F5/`experiments/results/f5.json`): under REALISTIC (unmixing-based, #
  #  non-oracle) fraction estimation, SR-ranked sub-pixel allocation was substantially WORSE   #
  #  than naive blocky assignment (paired IoU -0.313).  F7 uses realistic fractions.           #
  ############################################################################################

What this script actually runs (no cached SR is taken on trust):
  1. REAL tiled SR.  The pinned SEN2SR-lite is loaded and run over the crop for all four dates through
     the v1 tiler (`experiments/wayanad_evidence/tiler.py`, native 128 px tiles, 16 px crop margin,
     stride 96 -- the E1 lattice, ADR-001) with all 8 dihedral variants per tile
     (`experiments/wayanad_evidence/sr.py::sr_variants`, E8's transform set).  35 tiles x 8 runs x
     4 dates = 1120 real forward passes.  The result is then compared against the cached product and
     against v1's own Welford moments as an independent reproduction check.
  2. tau is RECOMPUTED (not read verbatim) with gate_v2's own score on exactly F1's placebo folds,
     splits, window grid and alpha, following docs/adr-f2-gate-v2.md section 3, and cross-checked
     against experiments/results/f2.json.  F1's raw export is consumed and its sha256 verified.
  3. Endmembers are estimated from pre-event data outside the 1500 m disks + footprint (B12), with the
     same deterministic NDVI-ceiling escalation F2 recorded as a pre-registration deviation.
  4. F3's per-date robust median-offset normalisation is applied per band before sigma/gate inputs
     (configs/fix.yaml f3_noise_v2.interaction_test: F3 was KEPT).
  5. NO_DATA from the SCL validity mask on every date (B11).  The exclusion geometry is NOT part of
     NO_DATA here -- excluding the scar would delete the very thing this map is for; it is used only to
     keep the event out of the endmember estimation sample.
  6. The real block-sum test on THIS production class map (B7), recomputed independently of the
     allocator.
  7. v1-vs-v2 comparison ONLY on quantities with identical definitions on both sides, enforced in code
     by `assert_well_defined_comparison` (configs/fix.yaml f7_wayanad_v2.no_cross_class_comparison).
  8. The E5 parent-first cascade's tile selection on the real parent mask: forward passes saved, and
     what the cascade would have missed.

Run: `python -m experiments.f7_wayanad_v2` from the repo root.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from experiments.e5_parent_cascade import select_tiles
from experiments.f3_noise_v2 import normalise_dates
from experiments.wayanad_evidence import config as C
from experiments.wayanad_evidence.data import load_date
from experiments.wayanad_evidence.step3_sr import fill_invalid
from experiments.wayanad_evidence.stats import Welford, ndvi as sr_ndvi
from experiments.wayanad_evidence.tiler import tile_plan
from risk.common import digest, write_json
from trustsr import gate_v2 as G
from trustsr import placebo_v2 as P

ROOT = Path(__file__).resolve().parent.parent
FIX_CONFIG = ROOT / 'configs' / 'fix.yaml'
CALIB_SCORES = ROOT / 'data/experiments-cache/f1_calibration_scores.npz'
CALIB_SCORES_SHA256_EXPECTED = '53a2fe8a3a7852d7648d6d8d17abaf982b1435d23ffb0afcd9fda8effd2679e4'
F2_JSON = ROOT / 'experiments/results/f2.json'
F1_JSON = ROOT / 'experiments/results/f1.json'
F5_JSON = ROOT / 'experiments/results/f5.json'
V1_STEP4 = ROOT / 'experiments/wayanad_evidence/outputs/step4.json'

CROP = (256, 896, 128, 640)                                  # rows, cols on the 1024 10 m AOI grid; v1's crop
PRE_DATES = ['2024-01-16', '2024-01-21', '2024-01-26']       # configs/fix.yaml f7_wayanad_v2.dates.pre
POST_DATE = '2024-12-06'                                     # configs/fix.yaml f7_wayanad_v2.dates.post
PRE_4TH_CANDIDATE = '2023-12-27'                             # pre-registered as ALLOWED, only if obtainable
WINDOW_10M = 16
TILE_PX = 128                                                # 2.5 m px per block-bootstrap tile (F1's split)
WINDOW_PX = 64
ALPHA = 0.05
LAM = 0.0                                                    # F2's fitted value (f2.json lambda.value); re-read below
M2_PER_PX = 6.25                                             # (10 m / 4)^2, configs/fix.yaml f7_wayanad_v2.report_per_class

# ---- the FAILED keep rule this run carries forward, stated once, in code -----------------------------
GATE_V2_FAR_FAILURE = {
    'what': "gate v2's window false-alarm rate on F1's placebo TEST split",
    'estimate': 0.0652, 'ci': [0.0478, 0.0850], 'alpha': ALPHA, 'keep_rule': 'window FAR <= 0.05',
    'verdict': 'NOT MET',
    'meaning': 'The detector whose output this map is was measured, on null-event data, to raise false '
               'alarms in about 6.5 % of 160 m windows where the pre-registered ceiling was 5 %. F7 is '
               'therefore reported as FAIL. These class counts describe what THIS gate configuration '
               'produces on the real landslide; they are not a validated detection.',
    'source': 'experiments/results/f2.json keep_rule.window_far_test; docs/adr-f2-gate-v2.md',
}
F5_CONTEXT = (
    'F5 (experiments/results/f5.json) found that under REALISTIC (unmixing-based, non-oracle) fraction '
    'estimation, SR-ranked sub-pixel allocation was substantially WORSE than naive blocky assignment '
    '(paired IoU -0.313). F7 allocates with realistic, unmixing-derived fractions, so F5 is a direct '
    'prior signal that the sub-pixel allocation quality in this map is not to be relied on.'
)


# ---------------------------------------------------------------- v1-vs-v2 comparison guard ----------------------------------------------------------------

# Only these pairs have IDENTICAL definitions on both sides, so only these may be compared
# (configs/fix.yaml f7_wayanad_v2.compare_v1_vs_v2_only_on / .no_cross_class_comparison).
COMPARABLE_PAIRS = {
    ('v1:OBSERVED|INFERRED', 'v2:CORE|ALLOCATED'):
        'mapped change area: the set of 2.5 m pixels the product maps as change. v1 flagged = OBSERVED '
        'union INFERRED; v2 flagged = CORE union ALLOCATED (configs/fix.yaml units.flagged_definition). '
        'Same grid, same crop, same dates, same denominator.',
    ('v1:NO_DATA', 'v2:NO_DATA'):
        'NO_DATA: 10 m pixels invalid in SCL (cloud / shadow / invalid / not acquired) or with undefined '
        'NDVI on ANY of the four dates, replicated to 2.5 m, union pixels where the SR product is '
        'non-finite. Identical construction on both sides (v1 step4_gate.py; v2 gate_v2.nodata_from_scl).',
}
# Pairs that LOOK comparable and are not. Each entry says why, and the guard refuses them.
INCOMPARABLE_PAIRS = {
    ('v1:UNSUPPORTED', 'v2:UNSUPPORTED'):
        "DIFFERENT DEFINITIONS -- not compared. v1 UNSUPPORTED = (d_SR > k*sigma_SR) AND NOT parent, where "
        "'parent' is a fixed 0.30 NDVI-drop threshold on the 10 m image. v2 UNSUPPORTED = (d_SR > "
        "k*sigma_SR) AND the 10 m block allocated zero sub-pixels, where 'allocated zero' comes from a "
        "conformal window detection on the unmixing score plus round(16f) = 0. The gating predicate is a "
        "different function of different inputs, so the two counts are not two measurements of one "
        "quantity. Both are reported separately, neither is differenced against the other.",
    ('v1:OBSERVED', 'v2:CORE'):
        'DIFFERENT DEFINITIONS -- not compared. v1 OBSERVED = SR-significant AND parent-positive; v2 CORE '
        '= a fully allocated block (round(16f) = 16). Neither is a refinement of the other.',
    ('v1:INFERRED', 'v2:ALLOCATED'):
        'DIFFERENT DEFINITIONS -- not compared. v1 INFERRED = parent-positive but NOT SR-significant; v2 '
        'ALLOCATED = a partially allocated block (0 < round(16f) < 16). Different predicates entirely.',
}


def assert_well_defined_comparison(name_a: str, name_b: str) -> str:
    """Refuse any v1-vs-v2 comparison whose two sides are not the same quantity.

    configs/fix.yaml `f7_wayanad_v2.no_cross_class_comparison: true` and `f9_checks`
    'no_row_compares_classes_with_different_definitions'. This is the code that enforces it: every number
    in this script that puts a v1 quantity next to a v2 quantity goes through `compare_sets`, which calls
    this first. Comparing v1 UNSUPPORTED with v2 UNSUPPORTED, or v1 OBSERVED with v2 CORE, RAISES.
    """
    key = (name_a, name_b)
    if key in COMPARABLE_PAIRS:
        return COMPARABLE_PAIRS[key]
    if key in INCOMPARABLE_PAIRS:
        raise ValueError(f'refusing to compare {name_a} with {name_b}: {INCOMPARABLE_PAIRS[key]}')
    raise ValueError(f'refusing to compare {name_a} with {name_b}: this pair is not registered in '
                     f'COMPARABLE_PAIRS, so its two sides have not been shown to share a definition')


def compare_sets(name_a: str, mask_a, name_b: str, mask_b) -> dict:
    """Compare two 2.5 m pixel sets, but only after the guard accepts the pair."""
    definition = assert_well_defined_comparison(name_a, name_b)
    a, b = np.asarray(mask_a, bool), np.asarray(mask_b, bool)
    if a.shape != b.shape:
        raise ValueError(f'{name_a} {a.shape} and {name_b} {b.shape} are not on the same grid')
    inter, union = int((a & b).sum()), int((a | b).sum())
    return {
        'quantity': definition, 'grid_2p5m': list(a.shape), 'm2_per_px': M2_PER_PX,
        'a_name': name_a, 'a_px': int(a.sum()), 'a_m2': int(a.sum()) * M2_PER_PX,
        'a_km2': int(a.sum()) * M2_PER_PX / 1e6,
        'b_name': name_b, 'b_px': int(b.sum()), 'b_m2': int(b.sum()) * M2_PER_PX,
        'b_km2': int(b.sum()) * M2_PER_PX / 1e6,
        'b_minus_a_px': int(b.sum()) - int(a.sum()),
        'b_minus_a_km2': (int(b.sum()) - int(a.sum())) * M2_PER_PX / 1e6,
        'b_over_a': (int(b.sum()) / int(a.sum())) if a.any() else None,
        'intersection_px': inter, 'union_px': union,
        'iou': (inter / union) if union else None,
        'in_a_not_b_px': int((a & ~b).sum()), 'in_b_not_a_px': int((b & ~a).sum()),
    }


# ---------------------------------------------------------------- stage 1: the real SR run ----------------------------------------------------------------

def run_real_sr(cfg, root, cache, dates, crop, verbose=True) -> dict:
    """Run the pinned SEN2SR-lite over the crop for every date: real tiles, real 8-dihedral runs.

    This is not a cache read. `Runner` loads the hash-verified pinned weights, `tile_plan` is the v1 /
    E1 tile lattice (128 px native input, 16 px crop margin, stride 96) and `sr_variants` performs the
    8 dihedral transforms per tile, inverting each back to the original orientation before NDVI is taken
    (E8's convention). Returns per-date (8, 4H, 4W) SR NDVI plus v1's pooled Welford moments.
    """
    from experiments.wayanad_evidence.sr import Runner
    scale = cfg['tiling']['scale']
    tile, margin, stride = cfg['tiling']['tile'], cfg['tiling']['crop_margin_px'], cfg['tiling']['stride']
    floor = cfg['radiometry']['min_denominator']
    r0c, r1c, c0c, c1c = crop
    hc, wc = r1c - r0c, c1c - c0c
    plan = tile_plan((hc, wc), tile, margin, stride)

    runner = Runner(cfg, root)
    arrays = {d: load_date(cfg, cache, d) for d in dates}
    valid_10m = {d: arrays[d]['valid'][r0c:r1c, c0c:c1c] for d in dates}
    valid_all_10m = np.logical_and.reduce([valid_10m[d] for d in dates])
    valid_hr = np.repeat(np.repeat(valid_all_10m, scale, 0), scale, 1)

    pre_dates = [d for d in dates if d != POST_DATE]
    acc = {'pre': Welford((hc * scale, wc * scale)), 'post': Welford((hc * scale, wc * scale))}
    per_date, passes, seconds = {}, 0, {}
    for date in dates:
        a = arrays[date]
        refl = a['refl'][:, r0c:r1c, c0c:c1c]
        model_in = fill_invalid(refl, a['valid_scl'][r0c:r1c, c0c:c1c])
        nd_runs = np.zeros((8, hc * scale, wc * scale), dtype=np.float32)
        target = acc['post'] if date == POST_DATE else acc['pre']
        t0 = time.perf_counter()
        for n, t in enumerate(plan):
            r0, c0 = t['r0'], t['c0']
            (rl, rh), (cl, ch) = t['keep_r'], t['keep_c']
            from experiments.wayanad_evidence.sr import sr_variants
            v = sr_variants(runner, model_in[:, r0:r0 + tile, c0:c0 + tile], f'{date} tile#{n}',
                            runner.oom_type, runner.cleanup, runner.memory)
            passes += 8
            hr = (slice((rl - r0) * scale, (rh - r0) * scale), slice((cl - c0) * scale, (ch - c0) * scale))
            vv = v[:, :, hr[0], hr[1]]
            out_region = (slice(rl * scale, rh * scale), slice(cl * scale, ch * scale))
            nd = sr_ndvi(vv[:, 0], vv[:, 3], floor)
            for k in range(8):
                target.update_at(out_region, nd[k], where=valid_hr[out_region])
                nd_runs[k, out_region[0], out_region[1]] = nd[k]
        seconds[date] = time.perf_counter() - t0
        per_date[date] = nd_runs
        if verbose:
            print(f'  SR {date}: {len(plan)} tiles x 8 runs in {seconds[date]:.1f}s', flush=True)

    ddof = cfg['change']['ddof']
    moments = {'pre_mean': acc['pre'].mean, 'pre_std': acc['pre'].std(ddof), 'pre_count': acc['pre'].count,
               'post_mean': acc['post'].mean, 'post_std': acc['post'].std(ddof), 'post_count': acc['post'].count}
    return {'per_date_ndvi': per_date, 'moments': moments, 'tiles_per_date': len(plan),
            'tile_origins': [(t['r0'], t['c0']) for t in plan], 'tile_plan': plan,
            'forward_passes': passes, 'seconds_per_date': seconds,
            'valid_10m': valid_10m, 'valid_all_10m': valid_all_10m,
            'arrays': arrays, 'pre_dates': pre_dates, 'crop_2p5m': (hc * scale, wc * scale),
            'device': cfg['model']['device'],
            'method': f'pinned SEN2SR-lite, {len(plan)} native {tile} px tiles per date (crop margin '
                      f'{margin} px, stride {stride}), 8 dihedral variants per tile, NDVI per run after '
                      f'inverting the transform'}


def reproduction_check(sr: dict, cache: Path) -> dict:
    """Does this run reproduce the cached real SR product and v1's own moments? Independent evidence
    that the SR above really is the same tiled dihedral SR the rest of the project used."""
    out = {'cached_per_date_ndvi': {}, 'v1_step3_moments': None}
    for date, runs in sr['per_date_ndvi'].items():
        p = cache / 'per_date_ndvi' / f'{date}_dihedral_means.npy'
        if not p.is_file():
            out['cached_per_date_ndvi'][date] = {'cached_file_present': False}
            continue
        ref = np.load(p)
        both = np.isfinite(ref) & np.isfinite(runs)
        out['cached_per_date_ndvi'][date] = {
            'cached_file_present': True, 'shape': list(ref.shape),
            'identical_including_nan_placement': bool(np.array_equal(ref, runs, equal_nan=True)),
            'bytes_identical_strict_nan_ne_nan': bool(np.array_equal(ref, runs)),
            'max_abs_diff_where_both_finite': float(np.abs(ref[both] - runs[both]).max()) if both.any() else None,
            'n_finite_disagreements': int((np.isfinite(ref) != np.isfinite(runs)).sum())}
    sp = cache / 'step3_state.npz'
    if sp.is_file():
        with np.load(sp) as z:
            ref = {k: z[k] for k in z.files}
        out['v1_step3_moments'] = {
            k: float(np.nanmax(np.abs(np.where(np.isfinite(ref[k]) & np.isfinite(sr['moments'][k]),
                                               ref[k] - sr['moments'][k], 0.0))))
            for k in ('pre_mean', 'pre_std', 'post_mean', 'post_std')}
        out['v1_step3_moments']['counts_identical'] = bool(
            np.array_equal(ref['pre_count'], sr['moments']['pre_count'])
            and np.array_equal(ref['post_count'], sr['moments']['post_count']))
        out['v1_step3_moments']['note'] = ('max |this run - cached v1 step3_state.npz| per moment field, over '
                                           'pixels finite in both')
    return out


# ---------------------------------------------------------------- stage 2: tau, recomputed ----------------------------------------------------------------

def recompute_tau(cfg_ev, cache, seed, verbose=True) -> dict:
    """tau with gate_v2's OWN score on EXACTLY F1's placebo folds, splits, window grid and alpha.

    docs/adr-f2-gate-v2.md section 3: F1 exported max(d/sigma_v1), a different score function, so the
    verbatim number is statistically invalid for gate v2 (split conformal requires one score function on
    both halves) and would make the gate flag nothing -- a vacuous FAR of 0. F7 follows the same
    reasoning rather than diverging, and cross-checks against f2.json.

    Every date here is loaded through `P.guarded_load_date`, so the post-event image cannot enter the
    calibration path (F1's B1 guard). The production map below loads the post date deliberately and by a
    different route; tau never sees it.
    """
    from experiments.f2_gate_v2 import build, endmember_ceiling_table, estimate_endmembers_at, \
        fold_score, normalise_fold, window_split_masks
    folds, dih, refl, valid, ndvi, excl_10, excl_hr, aoi = build(cfg_ev, cache)
    shape_2p5m = folds[0].nodata.shape
    calib_w, test_w, tile_split = window_split_masks(shape_2p5m, seed)
    stack = lambda key, dates: np.stack([aoi[key][d] for d in dates])
    pool = [f.held_out for f in folds]
    ndvi_low, _em_all, _table = endmember_ceiling_table(
        stack('refl', pool), stack('valid', pool), stack('ndvi', pool), aoi['exclude'])
    nd_stack = np.stack([ndvi[d] for d in pool])
    stable_mask = (np.stack([valid[d] for d in pool]).all(axis=0) & ~excl_10
                   & (nd_stack.std(axis=0, ddof=1) <= 0.05) & np.isfinite(nd_stack).all(axis=0))
    calib_scores = []
    for fold in folds:
        em = estimate_endmembers_at(stack('refl', fold.pre_dates), stack('valid', fold.pre_dates),
                                    stack('ndvi', fold.pre_dates), aoi['exclude'], ndvi_low)
        rp, rq, _off = normalise_fold(fold, stable_mask)
        s, vb, _sb, _fl, _f = fold_score(fold, em['e_v'], em['e_b'], rp, rq)
        win_s, win_v = G.window_max_score(s, WINDOW_10M, vb)
        calib_scores.append(win_s[calib_w & win_v & np.isfinite(win_s)])
    calib_scores = np.concatenate(calib_scores)
    tau, info = G.split_conformal_quantile(calib_scores, ALPHA)
    if verbose:
        print(f'  tau recomputed = {tau:.6f} (n = {info["n_calibration"]})', flush=True)
    f1_raw = np.load(CALIB_SCORES)['calibration_window_scores'].astype(np.float64)
    return {
        'value': tau, **info,
        'n_pre_dates_in_calibration': int(folds[0].refl_pre.shape[0]),
        'ndvi_low_used': ndvi_low,
        'provenance': 'ceil((n+1)(1-alpha))-th order statistic of the per-160 m-window max of gate_v2\'s '
                      'OWN score s = f/sigma_f, over F1\'s three leave-one-date-out PLACEBO folds, '
                      'restricted to the CALIBRATION (even-tile) half of F1\'s checkerboard split. '
                      'Null-event data only (P.guarded_load_date blocks the post-event date in this path).',
        'f1_export_consumed': {
            'path': str(CALIB_SCORES.relative_to(ROOT)), 'sha256': digest(CALIB_SCORES),
            'sha256_matches_f1_json': digest(CALIB_SCORES) == CALIB_SCORES_SHA256_EXPECTED,
            'n_scores': int(f1_raw.size),
            'tau_if_taken_verbatim': float(G.split_conformal_quantile(f1_raw, ALPHA)[0]),
            'why_not_verbatim': 'F1 exported max(d/sigma_v1), an SR-NDVI z-score on a different scale '
                                f'({float(f1_raw.min()):.2f} to {float(f1_raw.max()):.2f}) from gate_v2\'s '
                                'unmixing score f/sigma_f. Split conformal is valid only when calibration '
                                'and test scores come from the same score function. F7 follows F2\'s '
                                'reasoning (docs/adr-f2-gate-v2.md section 3) rather than diverging; the '
                                'verbatim number is reported for transparency and would flag nothing.'},
    }


# ---------------------------------------------------------------- stage 3: the production gate ----------------------------------------------------------------

def production_inputs(cfg_ev, cache, sr, crop):
    """The 10 m reflectance / validity / NDVI the gate needs, plus the exclusion geometry (endmembers
    only) and the 2.5 m SR change score and sigma."""
    import rasterio
    r0, r1, c0, c1 = crop
    with rasterio.open(ROOT / 'experiments/wayanad_evidence/outputs/footprint.tif') as src:
        footprint_full, transform = src.read(1).astype(bool), src.transform
    excl_10_crop = P.exclusion_mask_10m(cfg_ev, transform, crop, footprint_full)
    excl_10_full = P.exclusion_mask_10m(cfg_ev, transform, (0, 1024, 0, 1024), footprint_full)

    arrays = sr['arrays']
    dates = PRE_DATES + [POST_DATE]
    refl = {d: np.moveaxis(arrays[d]['refl'][:, r0:r1, c0:c1], 0, -1).astype(np.float64) for d in dates}
    ndvi10 = {d: arrays[d]['ndvi'][r0:r1, c0:c1].astype(np.float64) for d in dates}
    valid = sr['valid_10m']

    # endmembers are estimated over the WHOLE AOI (F2's choice, restated): bare ground is very rare here
    aoi = {'refl': {d: np.moveaxis(arrays[d]['refl'], 0, -1).astype(np.float64) for d in PRE_DATES},
           'valid': {d: arrays[d]['valid'] for d in PRE_DATES},
           'ndvi': {d: arrays[d]['ndvi'].astype(np.float64) for d in PRE_DATES},
           'exclude': excl_10_full}
    return refl, ndvi10, valid, excl_10_crop, excl_10_full, aoi


def sr_change_and_sigma(sr):
    """d_SR = mean over pre dates of the 8-run dihedral mean NDVI, minus the post date's 8-run mean;
    sigma_SR = v1's dihedral dispersion sqrt(sigma_pre^2 + sigma_post^2). Both from THIS run's SR."""
    m = sr['moments']
    d_sr = m['pre_mean'] - m['post_mean']
    sigma_sr = np.sqrt(m['pre_std'] ** 2 + m['post_std'] ** 2)
    return d_sr, sigma_sr


# ---------------------------------------------------------------- stage 4: E5 cascade ----------------------------------------------------------------

def cascade_full_aoi(exp_cfg, seed, cfg_ev, parent10_full) -> dict:
    """The same E5 selection on the FULL 1024x1024 AOI parent mask and the full AOI tile grid.

    SR was NOT run over the full AOI in F7 (sr_run.ran_on says so), so this is a tile-selection and
    forward-pass COUNT on a real parent mask, not a measured saving. It is reported because F7's own crop
    is a 6.4 x 5.1 km box centred on the scar, where by construction almost every tile fires -- which
    makes the crop-level saving (0) a fact about the crop, not about the cascade.
    """
    s = exp_cfg['e5']
    tile, margin, stride = cfg_ev['tiling']['tile'], cfg_ev['tiling']['crop_margin_px'], cfg_ev['tiling']['stride']
    plan = tile_plan(parent10_full.shape, tile, margin, stride)
    origins = [(t['r0'], t['c0']) for t in plan]
    sel = select_tiles(origins, parent10_full, tile, s['min_parent_pixels'], s['rings'],
                       s['audit_fraction'], seed)
    n_dates = len(PRE_DATES) + 1
    full, casc = len(origins) * 8 * n_dates, len(sel['processed']) * 8 * n_dates
    return {
        'scope': 'the full 1024x1024 10 m AOI parent mask (experiments/wayanad_evidence step1_state.npz), '
                 'same E5 parameters, same 8 runs x 4 dates cost model',
        'sr_was_not_run_at_this_scale_in_f7': True,
        'tiles': {'total': len(origins), 'fired': len(sel['fired']), 'ring': len(sel['ring']),
                  'audit': len(sel['audit']), 'processed': len(sel['processed']),
                  'skipped': len(sel['skipped'])},
        'forward_passes': {'full_rerun_baseline': full, 'cascade': casc, 'saved': full - casc,
                           'saved_fraction': (full - casc) / full},
        'parent_positive_10m_px_in_aoi': int(parent10_full.sum()),
    }


def cascade_report(exp_cfg, seed, sr, parent10_crop, flagged_v2_mask, nodata) -> dict:
    """E5's parent-first tile selection on the REAL parent mask, over the real tile grid used above.

    Forward passes are counted, not modelled: every processed tile costs exactly 8 passes per date.
    What the cascade would MISS is measured against the full map this script just produced.
    """
    s = exp_cfg['e5']
    origins = sr['tile_origins']
    sel = select_tiles(origins, parent10_crop, TILE_PX, s['min_parent_pixels'], s['rings'],
                       s['audit_fraction'], seed)
    n_dates = len(PRE_DATES) + 1
    full = len(origins) * 8 * n_dates
    casc = len(sel['processed']) * 8 * n_dates
    # v1's stitch assigns each output pixel to exactly one tile's keep-window, so a skipped tile leaves
    # its keep-window unreconstructed. Measure exactly that region.
    scale = 4
    unreconstructed = np.zeros(sr['crop_2p5m'], bool)
    for t in sr['tile_plan']:
        if (t['r0'], t['c0']) in sel['processed']:
            continue
        (rl, rh), (cl, ch) = t['keep_r'], t['keep_c']
        unreconstructed[rl * scale:rh * scale, cl * scale:ch * scale] = True
    missed = flagged_v2_mask & unreconstructed
    return {
        'source': 'experiments/e5_parent_cascade.py::select_tiles, applied to the REAL 10 m parent mask of '
                  'this crop over the real tile grid of the SR run above',
        'parameters': {'min_parent_pixels': s['min_parent_pixels'], 'rings': s['rings'],
                       'audit_fraction': s['audit_fraction'], 'seed': seed,
                       'dihedral_runs_per_tile': 8, 'dates': n_dates},
        'tiles': {'total': len(origins), 'fired': len(sel['fired']), 'ring': len(sel['ring']),
                  'audit': len(sel['audit']), 'processed': len(sel['processed']),
                  'skipped': len(sel['skipped'])},
        'forward_passes': {'full_rerun_baseline': full, 'cascade': casc, 'saved': full - casc,
                           'saved_fraction': (full - casc) / full, 'ratio': casc / full,
                           'definition': 'tiles x 8 dihedral runs x 4 dates; the full baseline is exactly '
                                         'what this script ran'},
        'cost_of_the_saving': {
            'unreconstructed_2p5m_px': int(unreconstructed.sum()),
            'unreconstructed_fraction_of_crop': float(unreconstructed.mean()),
            'v2_flagged_px_inside_skipped_tiles': int(missed.sum()),
            'v2_flagged_px_total': int(flagged_v2_mask.sum()),
            'fraction_of_v2_flagged_lost': (float(missed.sum() / flagged_v2_mask.sum())
                                            if flagged_v2_mask.any() else None),
            'nodata_px_inside_skipped_tiles': int((np.asarray(nodata, bool) & unreconstructed).sum()),
            'note': 'v1 stitching gives every 2.5 m pixel exactly one owning tile, so a skipped tile leaves '
                    'its keep-window with no SR at all. The cascade\'s recall is bounded by the 10 m parent '
                    'mask by construction (E5\'s own stated limitation).'},
    }


# ---------------------------------------------------------------- stage 5: the wow figure ----------------------------------------------------------------

def wow_figure(path: Path, sr, v1_flagged, v2_class, d_sr, parent10, nodata, dates_label) -> dict:
    """Comparison panel at the scar margin: blocky 10 m parent vs v2's 2.5 m map, with the FAILED FAR on it."""
    scale = 4
    # The panel is about the SCAR MARGIN, so pick the 160 m box containing the most 10 m parent-mask
    # BOUNDARY blocks (blocks whose 4-neighbourhood is not constant in the parent mask). That is where a
    # blocky 10 m product and a 2.5 m allocation can visibly disagree; a window deep inside the scar is
    # all-parent and shows nothing. Ties break on the number of ALLOCATED (partial) pixels.
    p = np.asarray(parent10, bool)
    edge = np.zeros_like(p)
    edge[1:, :] |= p[:-1, :] != p[1:, :]
    edge[:-1, :] |= p[:-1, :] != p[1:, :]
    edge[:, 1:] |= p[:, :-1] != p[:, 1:]
    edge[:, :-1] |= p[:, :-1] != p[:, 1:]
    win, win10 = 64, 16                                            # 64 px at 2.5 m = 16 blocks = 160 m
    eh, ew = edge.shape[0] // win10, edge.shape[1] // win10
    edge_per_win = edge[:eh * win10, :ew * win10].reshape(eh, win10, ew, win10).sum(axis=(1, 3))
    alloc = (np.asarray(v2_class) == G.ALLOCATED)
    h, w = alloc.shape
    alloc_per_win = alloc[:h // win * win, :w // win * win].reshape(
        h // win, win, w // win, win).sum(axis=(1, 3))
    key = edge_per_win.astype(np.float64) + 1e-9 * alloc_per_win[:eh, :ew]
    br, bc = np.unravel_index(int(np.argmax(key)), key.shape)
    r0, c0 = br * win, bc * win
    sl = (slice(r0, r0 + win), slice(c0, c0 + win))
    info = {'window_2p5m_rowcol': [int(r0), int(c0)], 'window_px': win, 'window_m': win * 2.5,
            'window_selected_by': 'the 160 m window containing the most 10 m parent-mask boundary blocks '
                                  '(the scar margin), ties broken on ALLOCATED pixel count',
            'parent_boundary_blocks_in_window': int(edge_per_win[br, bc]),
            'allocated_px_in_window': int(alloc_per_win[br, bc])}
    blocky = np.repeat(np.repeat(parent10, scale, 0), scale, 1)
    arrays = {'d_sr_2p5m': d_sr[sl], 'blocky_parent_10m_replicated': blocky[sl].astype(np.uint8),
              'v2_class_2p5m': np.asarray(v2_class)[sl], 'v1_flagged_2p5m': np.asarray(v1_flagged)[sl],
              'nodata_2p5m': np.asarray(nodata)[sl]}
    np.savez_compressed(path.with_suffix('.npz'), **arrays)
    caption = (
        f"Wayanad, corrected AOI 11.490 N 76.160 E, AT THE SCAR MARGIN. "
        f"Pre {', '.join(PRE_DATES)}  ->  post {POST_DATE}. "
        f"MIDDLE: the 10 m parent decision replicated 4x4 (blocky) -- the slider's 'before'. "
        f"RIGHT: gate v2's 2.5 m map -- the slider's 'after'. "
        f"pretrained SEN2SR-lite, NOT fine-tuned (F4 has not shipped). Model reconstruction, not "
        f"observation. "
        f"NOT A VALIDATED DETECTOR: this gate's window false-alarm rate on F1's placebo test split is "
        f"{GATE_V2_FAR_FAILURE['estimate']:.4f} [{GATE_V2_FAR_FAILURE['ci'][0]:.4f}, "
        f"{GATE_V2_FAR_FAILURE['ci'][1]:.4f}], above the pre-registered ceiling of "
        f"{GATE_V2_FAR_FAILURE['alpha']} -- the keep rule FAILED.")
    info['caption'] = caption
    info['arrays_npz'] = str(path.with_suffix('.npz').relative_to(ROOT))
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap, BoundaryNorm
        cmap = ListedColormap(['#1b1b1b', '#e8462c', '#f2a93b', '#4c6fa5', '#000000'])
        norm = BoundaryNorm([-.5, .5, 1.5, 2.5, 3.5, 4.5], 5)
        code = np.full(arrays['v2_class_2p5m'].shape, 0, np.int64)
        code[arrays['v2_class_2p5m'] == G.CORE] = 1
        code[arrays['v2_class_2p5m'] == G.ALLOCATED] = 2
        code[arrays['v2_class_2p5m'] == G.UNSUPPORTED] = 3
        code[arrays['v2_class_2p5m'] == G.NO_DATA] = 4
        fig, ax = plt.subplots(1, 3, figsize=(16.5, 6.4))
        ax[0].imshow(arrays['d_sr_2p5m'], cmap='magma')
        ax[0].set_title('SR NDVI drop d (2.5 m)\n8-dihedral mean, real tiled SR', fontsize=10)
        ax[1].imshow(arrays['blocky_parent_10m_replicated'], cmap=ListedColormap(['#1b1b1b', '#e8462c']),
                     vmin=0, vmax=1, interpolation='nearest')
        ax[1].set_title('10 m parent, replicated 4x4 (blocky)', fontsize=10)
        ax[2].imshow(code, cmap=cmap, norm=norm, interpolation='nearest')
        ax[2].set_title('gate v2, 2.5 m\nred CORE / orange ALLOCATED / blue UNSUPPORTED', fontsize=10)
        for a in ax:
            a.set_xticks([]); a.set_yticks([])
        fig.suptitle(f'Wayanad landslide  |  pre {", ".join(PRE_DATES)}  ->  post {POST_DATE}  |  '
                     f'{win * 2.5:.0f} m window', fontsize=12)
        fig.text(0.5, 0.015, caption, ha='center', va='bottom', fontsize=8, wrap=True)
        fig.subplots_adjust(bottom=0.20, top=0.88)
        fig.savefig(path, dpi=140)
        plt.close(fig)
        info['png'] = str(path.relative_to(ROOT))
        info['rendered'] = True
    except Exception as exc:                                        # pragma: no cover - environment dependent
        info['rendered'] = False
        info['render_error'] = f'{type(exc).__name__}: {exc}'
        info['how_to_render'] = ('the three 64x64 2.5 m arrays in the .npz are the full content of the '
                                 'panel; render them side by side with the caption verbatim')
    return info


# ---------------------------------------------------------------- main ----------------------------------------------------------------

def fourth_pre_date_audit(cache: Path) -> dict:
    """configs/fix.yaml pre-registers 2023-12-27 as a 4th pre date IF obtainable. Report, never fabricate."""
    checked = {
        '10m_bands_npz': cache / f'{PRE_4TH_CANDIDATE}.npz',
        'sr_dihedral_ndvi': cache / 'per_date_ndvi' / f'{PRE_4TH_CANDIDATE}_dihedral_means.npy',
        'sr_identity': cache / f'sr_identity_{PRE_4TH_CANDIDATE}.npy',
    }
    scl_dir = cache / 'scl20'
    scl_hits = sorted(p.name for p in scl_dir.glob(f'*_{PRE_4TH_CANDIDATE.replace("-", "")}T*')) \
        if scl_dir.is_dir() else []
    present = {k: v.is_file() for k, v in checked.items()}
    return {
        'candidate': PRE_4TH_CANDIDATE,
        'preregistered_as_allowed': True,
        'paths_checked': {k: str(v.relative_to(ROOT)) for k, v in checked.items()},
        'present': present, 'scl20_files_matching_this_date': scl_hits,
        'available': bool(all(present.values())),
        'decision': f'{PRE_4TH_CANDIDATE} is NOT cached for the corrected AOI (no 10 m bands, no SCL, no SR '
                    f'product) and this run does not fetch imagery. F7 therefore uses the 3 pre-registered '
                    f'pre dates only. No substitute date was used and none was invented. This matches F1\'s '
                    f'report, which recorded the same absence.'
        if not all(present.values()) else
        f'{PRE_4TH_CANDIDATE} is cached and was used as a 4th pre date.',
    }


def main(verbose=True):
    started = datetime.now(timezone.utc).isoformat()
    fix_cfg = yaml.safe_load(FIX_CONFIG.read_text(encoding='utf-8'))
    fix_hash = digest(FIX_CONFIG)
    seed = fix_cfg['seed']
    exp_cfg = yaml.safe_load((ROOT / 'configs/experiments.yaml').read_text(encoding='utf-8'))
    cfg_ev, ev_root, ev_hash = C.load()
    cache = C.cache(cfg_ev, ev_root)
    k_v1 = cfg_ev['change']['k']
    f2 = json.loads(F2_JSON.read_text(encoding='utf-8'))
    lam = float(f2['lambda']['value'])

    dates = PRE_DATES + [POST_DATE]
    fourth = fourth_pre_date_audit(cache)

    # ---- 1. the real SR run -------------------------------------------------------------------------
    if verbose:
        print('F7: real tiled SR (pinned SEN2SR-lite, 8 dihedral runs per tile)...', flush=True)
    sr = run_real_sr(cfg_ev, ev_root, cache, dates, CROP, verbose)
    repro = reproduction_check(sr, cache)
    d_sr, sigma_sr = sr_change_and_sigma(sr)
    sr_varying = G.assert_sr_score_is_not_replicated(d_sr, 4)

    # ---- 2. tau, recomputed on F1's placebo folds ---------------------------------------------------
    if verbose:
        print('F7: recomputing tau on F1\'s placebo folds...', flush=True)
    tau_info = recompute_tau(cfg_ev, cache, seed, verbose)
    tau = tau_info['value']
    tau_info['f2_json_value'] = f2['tau']['value']
    tau_info['matches_f2'] = bool(abs(tau - f2['tau']['value']) < 1e-9)

    # ---- 3. endmembers ------------------------------------------------------------------------------
    from experiments.f2_gate_v2 import endmember_ceiling_table, estimate_endmembers_at
    refl, ndvi10, valid, excl_crop, excl_full, aoi = production_inputs(cfg_ev, cache, sr, CROP)
    st = lambda key: np.stack([aoi[key][d] for d in PRE_DATES])
    ndvi_low, em, ceiling_table = endmember_ceiling_table(st('refl'), st('valid'), st('ndvi'), aoi['exclude'])

    # ---- 4. F3's per-date normalisation -------------------------------------------------------------
    nd_pre = np.stack([ndvi10[d] for d in PRE_DATES])
    stable_mask = (np.stack([valid[d] for d in PRE_DATES]).all(axis=0) & ~excl_crop
                   & (nd_pre.std(axis=0, ddof=1) <= 0.05) & np.isfinite(nd_pre).all(axis=0))
    pre_stack = np.stack([refl[d] for d in PRE_DATES])                   # (3, h, w, B)
    post_stack = refl[POST_DATE]                                         # (h, w, B)
    n_pre_raw, n_post_raw = np.empty_like(pre_stack), np.empty_like(post_stack)
    offsets = {}
    for b in range(pre_stack.shape[-1]):
        p, q, op, oq, _ = normalise_dates(pre_stack[..., b], stable_mask, post_stack[..., b], stable_mask)
        n_pre_raw[..., b], n_post_raw[..., b] = p, q
        offsets[f'band{b}'] = {'pre_pool': [float(v) for v in op], 'post': float(oq)}
    y_pre, y_post = n_pre_raw, n_post_raw

    # ---- 5. NO_DATA (B11). The exclusion geometry is deliberately NOT part of it --------------------
    valid_stack_10m = np.stack([valid[d] for d in dates])
    nodata = G.nodata_from_scl(valid_stack_10m, 4)
    sr_finite = np.isfinite(d_sr) & np.isfinite(sigma_sr)
    for date in dates:
        sr_finite &= np.isfinite(sr['per_date_ndvi'][date]).all(axis=0)
    nodata = nodata | ~sr_finite

    # ---- 6. run the gate ----------------------------------------------------------------------------
    sigma_bands, sigma_floor = G.band_sigma_from_pre(y_pre, valid=None)
    class_map, meta = G.apply_gate_v2(y_pre, y_post, d_sr, em['e_v'], em['e_b'], sigma_bands, nodata, tau,
                                      lam=lam, window_size=WINDOW_10M, k_unsupported=k_v1,
                                      sigma_sr=sigma_sr)
    # independent recomputation of the block-sum test on the production map (B7)
    block_sum = G.block_sum_deviation(class_map, meta['f_effective'], nodata, 4)
    if block_sum['max_abs_deviation_all_blocks'] != meta['block_sum']['max_abs_deviation_all_blocks']:
        raise AssertionError('block-sum recomputation disagrees with the in-gate computation')

    # persist the production map so tests/test_f7_wayanad_v2.py can re-run the block-sum test on the REAL
    # output rather than on a fixture (the arrays are large; the cache directory is gitignored)
    np.savez_compressed(cache.parent / 'f7_production_state.npz', class_map=class_map,
                        f_effective=meta['f_effective'], nodata=nodata)

    v2_flagged = G.flagged_v2(class_map)
    counts = {name: int((class_map == code).sum()) for code, name in G.CLASS_NAMES.items()}
    per_class = {n: {'pixels_2p5m': c, 'm2': c * M2_PER_PX, 'km2': c * M2_PER_PX / 1e6}
                 for n, c in counts.items()}

    # ---- tau mutation on the REAL map: the loaded threshold is used (B10) ---------------------------
    mutation = {}
    for name, t in (('tau_times_0p5', tau * 0.5), ('tau_fitted', tau), ('tau_times_2', tau * 2.0)):
        cm, _ = G.apply_gate_v2(y_pre, y_post, d_sr, em['e_v'], em['e_b'], sigma_bands, nodata, t,
                                lam=lam, window_size=WINDOW_10M, k_unsupported=k_v1, sigma_sr=sigma_sr)
        mutation[name] = {'tau': float(t), 'flagged_px': int(G.flagged_v2(cm).sum())}
    tau_changes_map = len({v['flagged_px'] for v in mutation.values()}) > 1

    # ---- the n_pre mismatch between calibration and production, measured not hidden ----------------
    sigma_f_prod = G.fraction_sigma(sigma_bands, em['e_v'], em['e_b'], n_pre=len(PRE_DATES))
    sigma_f_calib = G.fraction_sigma(sigma_bands, em['e_v'], em['e_b'], n_pre=2)
    scale_2_over_3 = float(np.nanmedian(sigma_f_calib / sigma_f_prod))
    cm_calibmatch, _ = G.apply_gate_v2(y_pre, y_post, d_sr, em['e_v'], em['e_b'], sigma_bands, nodata,
                                       tau * scale_2_over_3, lam=lam, window_size=WINDOW_10M,
                                       k_unsupported=k_v1, sigma_sr=sigma_sr)
    n_pre_mismatch = {
        'calibration_n_pre': tau_info['n_pre_dates_in_calibration'], 'production_n_pre': len(PRE_DATES),
        'why_it_matters': 'sigma_f = sqrt(Var(a)(1/n_pre + 1)), so with 3 pre dates sigma_f is SMALLER than '
                          'in the 2-pre-date placebo folds tau was calibrated on. The same physical change '
                          'therefore scores HIGHER here than it would have in calibration: the production '
                          'gate is slightly MORE sensitive than the threshold was calibrated for, which '
                          'pushes the true false-alarm rate ABOVE the already-failing 0.0652.',
        'median_sigma_f_ratio_2pre_over_3pre': scale_2_over_3,
        'flagged_px_at_tau': int(v2_flagged.sum()),
        'flagged_px_at_tau_rescaled_to_match_calibration_sensitivity':
            int(G.flagged_v2(cm_calibmatch).sum()),
        'headline_uses': 'tau as calibrated (no rescaling); the rescaled count is reported as a sensitivity, '
                         'not substituted for the headline',
    }

    # ---- endmember sensitivity on the REAL map -----------------------------------------------------
    def flagged_count(ev, eb):
        cm, _ = G.apply_gate_v2(y_pre, y_post, d_sr, ev, eb, sigma_bands, nodata, tau, lam=lam,
                                window_size=WINDOW_10M, k_unsupported=k_v1, sigma_sr=sigma_sr)
        return int(G.flagged_v2(cm).sum())
    sensitivity = G.endmember_sensitivity(flagged_count, em)
    sensitivity['note'] = ('flagged (CORE|ALLOCATED) 2.5 m px on THIS production map with tau and lambda '
                           'fixed, recomputed with each endmember shifted by +/- 1 robust SD in every band')

    # ---- 7. v1 vs v2, only on shared definitions ----------------------------------------------------
    import rasterio
    with rasterio.open(ROOT / 'experiments/wayanad_evidence/outputs/class_map_2p5m.tif') as src:
        v1_class = src.read(1)
    V1 = {'NO_CHANGE': 0, 'OBSERVED': 1, 'INFERRED': 2, 'UNSUPPORTED': 3, 'NO_DATA': 255}
    if v1_class.shape != class_map.shape:
        raise AssertionError(f'v1 map {v1_class.shape} and v2 map {class_map.shape} are not on one grid')
    v1_flagged = (v1_class == V1['OBSERVED']) | (v1_class == V1['INFERRED'])
    v1_nodata = v1_class == V1['NO_DATA']

    comparisons = {
        'mapped_change_area': compare_sets('v1:OBSERVED|INFERRED', v1_flagged, 'v2:CORE|ALLOCATED', v2_flagged),
        'no_data': compare_sets('v1:NO_DATA', v1_nodata, 'v2:NO_DATA', class_map == G.NO_DATA),
    }

    # boundary blocks: where does v2's 2.5 m allocation diverge from a naive 4x4 replication of the 10 m parent?
    r0, r1, c0, c1 = CROP
    with np.load(cache / 'step1_state.npz') as z:
        parent10_full = z['parent']
    parent10 = parent10_full[r0:r1, c0:c1]
    blocky = np.repeat(np.repeat(parent10, 4, 0), 4, 1)
    block_nodata = G._within_block(nodata.astype(np.int64), 4).sum(axis=-1) > 0
    differs_blk = G._within_block((v2_flagged != blocky).astype(np.int64), 4).sum(axis=-1) > 0
    nb = np.zeros_like(parent10)
    nb[1:, :] |= parent10[:-1, :] != parent10[1:, :]
    nb[:-1, :] |= parent10[:-1, :] != parent10[1:, :]
    nb[:, 1:] |= parent10[:, :-1] != parent10[:, 1:]
    nb[:, :-1] |= parent10[:, :-1] != parent10[:, 1:]
    boundary = nb & ~block_nodata
    scored = ~block_nodata
    not_a_replication = G._within_block(v2_flagged.astype(np.int64), 4).sum(axis=-1)
    not_a_replication = (not_a_replication > 0) & (not_a_replication < 16) & scored
    comparisons['boundary_allocation_divergence'] = {
        'definition': 'A 10 m pixel is a BOUNDARY pixel if its 4-neighbourhood is not constant in v1\'s 10 m '
                      'parent mask. The "blocky parent" is that 10 m parent decision replicated 4x4 to 2.5 m '
                      '-- exactly what a no-SR product would emit. A boundary pixel COUNTS if v2\'s 16 '
                      'sub-pixel flagged labels are not all equal to the blocky parent value there. Blocks '
                      'touching NO_DATA are excluded from both sides.',
        'n_10m_blocks_scored': int(scored.sum()),
        'n_10m_boundary_blocks': int(boundary.sum()),
        'n_10m_boundary_blocks_whose_2p5m_allocation_differs_from_blocky_parent':
            int((boundary & differs_blk).sum()),
        'fraction_of_boundary_blocks_differing':
            float((boundary & differs_blk).sum() / boundary.sum()) if boundary.any() else None,
        'n_10m_blocks_anywhere_differing_from_blocky_parent': int((scored & differs_blk).sum()),
        'n_10m_blocks_with_genuine_subpixel_structure_0_lt_n_lt_16': int(not_a_replication.sum()),
        'caveat': F5_CONTEXT,
    }

    # UNSUPPORTED: definitions differ -> reported, explicitly NOT compared
    unsupported_report = {
        'compared': False,
        'reason': INCOMPARABLE_PAIRS[('v1:UNSUPPORTED', 'v2:UNSUPPORTED')],
        'v1': {'name': 'v1:UNSUPPORTED', 'px': int((v1_class == V1['UNSUPPORTED']).sum()),
               'km2': int((v1_class == V1['UNSUPPORTED']).sum()) * M2_PER_PX / 1e6,
               'definition': 'd_SR > k*sigma_SR AND NOT parent-positive (parent = 10 m NDVI drop > 0.30)'},
        'v2': {'name': 'v2:UNSUPPORTED', 'px': counts['UNSUPPORTED'],
               'km2': counts['UNSUPPORTED'] * M2_PER_PX / 1e6,
               'definition': 'd_SR > k*sigma_SR AND the 10 m block allocated zero sub-pixels (no conformal '
                             'window detection, or round(16f) = 0)'},
        'enforced_by': 'experiments.f7_wayanad_v2.assert_well_defined_comparison raises on this pair',
    }

    # ---- 8. E5 cascade ------------------------------------------------------------------------------
    cascade = cascade_report(exp_cfg, seed, sr, parent10, v2_flagged, nodata)
    cascade['crop_note'] = (
        'This crop is a 6.4 x 5.1 km box CENTRED ON THE SCAR, so nearly every tile is parent-positive or '
        'in the buffer ring: the cascade processes all of them and saves nothing here. That is a fact '
        'about the crop, not about the cascade. The full-AOI selection below is the honest scale at which '
        'the cascade has anything to skip.')
    cascade['full_aoi_selection_for_context'] = cascade_full_aoi(exp_cfg, seed, cfg_ev, parent10_full)

    # ---- 9. the wow figure --------------------------------------------------------------------------
    fig = wow_figure(ROOT / 'experiments/results/f7_wow_figure.png', sr, v1_flagged, class_map, d_sr,
                     parent10, nodata, dates)

    # ---- result -------------------------------------------------------------------------------------
    result = {
        'status': 'FAIL',
        'caveat': (
            'THE EXPERIMENT RAN CORRECTLY; THE DETECTOR IT MEASURES DID NOT PASS ITS OWN KEEP RULE. '
            'F7 executed end to end on real data -- real tiled SR (1120 real forward passes), a tau '
            'recomputed from placebo folds only, a block-sum test that passes on this production map -- '
            'but the gate it runs is F2\'s gate v2, whose pre-registered detection keep rule FAILED: '
            f"window FAR {GATE_V2_FAR_FAILURE['estimate']:.4f} "
            f"[{GATE_V2_FAR_FAILURE['ci'][0]:.4f}, {GATE_V2_FAR_FAILURE['ci'][1]:.4f}] against a required "
            f"<= {ALPHA}. Every class count, area and comparison in this report describes WHAT THIS GATE "
            'CONFIGURATION PRODUCES on the real Wayanad scene. None of it is a validated detection, and '
            'none of it may be quoted as one. Status is FAIL for exactly this reason. ' + F5_CONTEXT),
        'evidence': 'real', 'experiment': 'f7_wayanad_v2',
        'config_sha256': fix_hash, 'wayanad_evidence_config_sha256': ev_hash,
        'replaces': {'invalidated': 'experiments/x9_v2_production.py, experiments/results/x9.json, '
                                    'experiments/x9_REPORT.md',
                     'see': 'experiments/results/INVALIDATED.md (B9-B12); x9 files are untouched by F7'},
        'gate_v2_keep_rule_failure': GATE_V2_FAR_FAILURE,
        'f5_realistic_fraction_finding': F5_CONTEXT,
        'aoi_center': fix_cfg['aoi_center'],
        'crop_10m_rows_cols': list(CROP),
        'grid': {'blocks_10m': list(meta['f_hat'].shape), 'px_2p5m': list(class_map.shape),
                 'window_10m_px': WINDOW_10M, 'm2_per_2p5m_px': M2_PER_PX,
                 'crop_area_km2': class_map.size * M2_PER_PX / 1e6},
        'dates': {'pre': PRE_DATES, 'post': POST_DATE, 'event': P.EVENT_DATE,
                  'fourth_pre_date': fourth,
                  'post_event_date_is_read_here_deliberately':
                      'F7 is the production map, so the post-event image IS an input. F1\'s '
                      'P.guarded_load_date guard remains active in the tau path (recompute_tau), which is '
                      'the only place a threshold is fitted; the production path loads the post date '
                      'through experiments.wayanad_evidence.data.load_date instead.'},
        'sr_run': {
            'is_real_inference': True,
            'method': sr['method'], 'device': sr['device'],
            'tiles_per_date': sr['tiles_per_date'], 'dates_run': dates,
            'forward_passes_total': sr['forward_passes'],
            'seconds_per_date': {k: round(v, 2) for k, v in sr['seconds_per_date'].items()},
            'ran_on': f'the real cached crop rows {CROP[0]}:{CROP[1]}, cols {CROP[2]}:{CROP[3]} of the '
                      f'1024x1024 corrected-AOI 10 m grid = {CROP[1]-CROP[0]}x{CROP[3]-CROP[2]} at 10 m -> '
                      f'{class_map.shape[0]}x{class_map.shape[1]} at 2.5 m. This is exactly the region v1, '
                      f'F1 and F2 use, which is what makes the v1-vs-v2 comparison below well defined. The '
                      f'full 1024x1024 AOI was NOT super-resolved in this run.',
            'reproduction_check': repro,
            'sr_score_blocks_with_within_block_variation_fraction': sr_varying,
            'b9_guard': 'trustsr.gate_v2.assert_sr_score_is_not_replicated ran on the production d_SR and '
                        'passed; x9\'s np.repeat(np.repeat(d_10m,4),4) would raise',
        },
        'tau': tau_info,
        'n_pre_mismatch_between_calibration_and_production': n_pre_mismatch,
        'lambda': {'value': lam, 'source': 'experiments/results/f2.json lambda.value, fitted on F1\'s '
                                           'calibration split by split-half dihedral reproducibility; not '
                                           'refitted here (refitting on production data would use the event)'},
        'endmembers': {
            'e_v': [float(v) for v in em['e_v']], 'e_b': [float(v) for v in em['e_b']],
            'e_v_robust_sd': [float(v) for v in em['e_v_robust_sd']],
            'e_b_robust_sd': [float(v) for v in em['e_b_robust_sd']],
            'e_v_count': em['e_v_count'], 'e_b_count': em['e_b_count'],
            'n_valid_outside_exclusion': em['n_valid_outside_exclusion'], 'rules': em['rules'],
            'ndvi_low_used': ndvi_low, 'ndvi_low_ceiling_table': ceiling_table,
            'estimated_on': 'the whole 1024x1024 10 m AOI, PRE-EVENT dates only, outside the 1500 m '
                            'plausibility disks and the landslide footprint',
            'exclusion_is_not_a_label': G.gate_v2_signature_audit(),
            'sensitivity_plus_minus_1_robust_sd': sensitivity,
            'preregistration_deviation_inherited_from_f2':
                f'configs/fix.yaml pre-registers e_b as NDVI <= 0.20; that selects 8 px over the whole '
                f'evergreen AOI. F2 escalated the ceiling deterministically in 0.05 steps to {ndvi_low}; F7 '
                f'reuses the same rule and the same table. e_b is therefore the darkest land cover present, '
                f'not true bare ground, so f is upper-leaning and is NOT a calibrated area.',
        },
        'normalisation_f3': {
            'applied': True,
            'why': 'configs/fix.yaml f3_noise_v2.interaction_test -- F3 PASSED and was kept, so the same '
                   'per-date robust median-offset normalisation is applied here',
            'method': 'experiments.f3_noise_v2.normalise_dates, per band, reference = per-pixel median over '
                      'the 3 pre dates, offsets fit on stable pixels (NDVI std <= 0.05 across pre dates) '
                      'outside the 1500 m disks + footprint; the post date\'s offset is fit exactly as a '
                      'deployed date\'s would be',
            'stable_px_10m': int(stable_mask.sum()), 'offsets': offsets,
            'band_order': ['B04', 'B03', 'B02', 'B08'],
        },
        'no_data': {
            'source': 'experiments.wayanad_evidence.data.scl_valid on EVERY one of the 4 dates, AND-ed, '
                      'nearest-neighbour to 2.5 m, union pixels where the SR product is non-finite',
            'exclusion_geometry_is_NOT_included':
                'the 1500 m disks and the landslide footprint are used ONLY to keep the event out of the '
                'endmember sample. Making them NO_DATA would delete the scar this map exists to show.',
            'px': counts['NO_DATA'], 'km2': counts['NO_DATA'] * M2_PER_PX / 1e6,
            'valid_px': int((~nodata).sum()),
        },
        'class_report': per_class,
        'class_report_denominator': {
            'total_2p5m_px': int(class_map.size),
            'total_km2': class_map.size * M2_PER_PX / 1e6,
            'valid_2p5m_px': int((~nodata).sum()),
            'note': 'percentages of a class should be taken against valid_2p5m_px, not the total',
        },
        'mapped_change_v2': {
            'flagged_definition': 'CORE union ALLOCATED (configs/fix.yaml units.flagged_definition.v2); '
                                  'UNSUPPORTED is never counted as flagged',
            'px': int(v2_flagged.sum()), 'm2': int(v2_flagged.sum()) * M2_PER_PX,
            'km2': int(v2_flagged.sum()) * M2_PER_PX / 1e6},
        'block_sum_test': {**block_sum,
                           'recomputed_independently_of_the_gate': True,
                           'scope': 'THIS production map on the real AOI crop -- not F2\'s placebo folds'},
        'detection': {'tau': tau, 'alpha': ALPHA,
                      'n_windows_valid': meta['n_windows_valid'],
                      'n_windows_detected': meta['n_windows_detected'],
                      'window_size_10m_px': WINDOW_10M,
                      'n_blocks_detected': int(meta['detected_block'].sum()),
                      'tau_mutation_on_the_real_map': mutation,
                      'tau_changes_the_output_map': bool(tau_changes_map)},
        'sigma_f': {'method': 'first-order error propagation; Var(f) = Var(a)(1/n_pre + 1)',
                    'sigma_band_floor_per_band': [float(v) for v in sigma_floor],
                    'n_pre': len(PRE_DATES)},
        'v1_vs_v2': {
            'preamble': 'Only quantities with IDENTICAL definitions on both sides are compared. The guard '
                        '`assert_well_defined_comparison` raises on any other pair; see '
                        'INCOMPARABLE_PAIRS and tests/test_f7_wayanad_v2.py.',
            'v1_source': 'experiments/wayanad_evidence/outputs/class_map_2p5m.tif (v1 pipeline, SOUND, '
                         'unmodified), same crop, same 4 dates',
            'comparisons': comparisons,
            'unsupported_not_compared': unsupported_report,
            'v1_class_counts_px': {n: int((v1_class == c).sum()) for n, c in V1.items()},
            'v2_class_counts_px': counts,
        },
        'e5_cascade': cascade,
        'wow_figure': fig,
        'limitations': [
            'THE GATE FAILED ITS OWN FAR KEEP RULE (0.0652 [0.0478, 0.0850] vs <= 0.05). This map is what '
            'that gate produces, not a validated detection.',
            F5_CONTEXT,
            f'Only 3 pre dates. {PRE_4TH_CANDIDATE} is pre-registered as an allowed 4th pre date but is not '
            f'cached for the corrected AOI and this run fetches no imagery; see dates.fourth_pre_date for '
            f'every path checked. No substitute was used.',
            'One post-event date (2024-12-06), 4 months after the 2024-07-30 event: regrowth, seasonality '
            'and illumination differences between a January pre-pool and a December post date are '
            'confounded with the landslide. There is no second post date to separate them.',
            'tau was calibrated on 2-pre-date placebo folds and applied to a 3-pre-date production score. '
            'sigma_f shrinks with n_pre, so the production gate is slightly MORE sensitive than the '
            'threshold was calibrated for -- pushing the true FAR further above the already-failing value. '
            'Measured and reported in n_pre_mismatch_between_calibration_and_production, not corrected.',
            'SR ran on the 640x512 (10 m) crop, not the full 1024x1024 AOI. Stated exactly in sr_run.ran_on.',
            'e_b is the darkest land cover in the AOI, not true bare ground (the pre-registered NDVI <= 0.20 '
            'rule selects 8 px). f is upper-leaning and is NOT a calibrated area.',
            'The gate remains highly sensitive to the endmembers; see '
            'endmembers.sensitivity_plus_minus_1_robust_sd for the effect on THIS map.',
            'v2 UNSUPPORTED and v1 UNSUPPORTED have different definitions and are NOT differenced anywhere; '
            'both are reported with their definitions.',
            'The E5 cascade numbers are a tile-selection and forward-pass count on the real parent mask, '
            'plus a measurement of what the skipped tiles would have cost. The cascade\'s recall is bounded '
            'by the 10 m parent mask by construction.',
            'No labels were used and no 2.5 m ground truth exists, so no accuracy, F1 or IoU-against-truth '
            'is reported anywhere. The IoU in v1_vs_v2 is agreement between two products, not accuracy.',
            cfg_ev['labels']['model'], cfg_ev['labels']['k'], cfg_ev['labels']['comparison'],
            fix_cfg['units']['window_caveat'],
        ],
    }
    result.update(started_utc=started, finished_utc=datetime.now(timezone.utc).isoformat())
    write_json(ROOT / 'experiments/results/f7.json', result)
    write_report(result, ROOT / 'experiments/results/f7_REPORT.md')
    if verbose:
        print(json.dumps({'status': result['status'],
                          'forward_passes': sr['forward_passes'],
                          'block_sum_max_deviation': block_sum['max_abs_deviation_all_blocks'],
                          'v2_flagged_km2': result['mapped_change_v2']['km2'],
                          'v1_flagged_km2': comparisons['mapped_change_area']['a_km2'],
                          'e5_passes_saved': cascade['forward_passes']['saved']}, indent=2))
    return result


def write_report(r, path: Path):
    c = r['v1_vs_v2']['comparisons']
    ma, bd = c['mapped_change_area'], c['boundary_allocation_divergence']
    u = r['v1_vs_v2']['unsupported_not_compared']
    fp = r['e5_cascade']['forward_passes']
    fail = r['gate_v2_keep_rule_failure']
    L = [
        '# F7 — Wayanad v2 production rerun (replaces the invalidated x9)', '',
        f"**Status: {r['status']}** · evidence: real · `config_sha256` "
        f"`{r['config_sha256']}` (sha256 of `configs/fix.yaml`)", '',
        '> ## ⚠ THE DETECTOR THIS MAP COMES FROM FAILED ITS OWN KEEP RULE', '>',
        f"> gate v2's window false-alarm rate on F1's placebo **test** split is "
        f"**{fail['estimate']:.4f} [{fail['ci'][0]:.4f}, {fail['ci'][1]:.4f}]** against a pre-registered "
        f"requirement of **≤ {fail['alpha']}**. The keep rule is **{fail['verdict']}**.", '>',
        f"> {fail['meaning']}", '>',
        f"> **F5, independently:** {r['f5_realistic_fraction_finding']}", '',
        '## What was actually run', '',
        f"- **Real SR inference, not a cache read.** {r['sr_run']['method']}. "
        f"**{r['sr_run']['forward_passes_total']:,} real forward passes** through the pinned "
        f"SEN2SR-lite on {r['sr_run']['device']} "
        f"({', '.join(f'{k}: {v}s' for k, v in r['sr_run']['seconds_per_date'].items())}).",
        f"- Ran on: {r['sr_run']['ran_on']}",
        f"- Dates: pre {', '.join(r['dates']['pre'])} → post {r['dates']['post']} "
        f"(event {r['dates']['event']}).",
        f"- 4th pre date `{r['dates']['fourth_pre_date']['candidate']}`: "
        f"{r['dates']['fourth_pre_date']['decision']}",
        f"- τ = **{r['tau']['value']:.6f}** = T_({r['tau']['k']}) of n = {r['tau']['n_calibration']} "
        f"placebo calibration window scores (α = {r['tau']['alpha']}); matches `f2.json`: "
        f"{r['tau']['matches_f2']}. Verbatim-from-F1 τ would be "
        f"{r['tau']['f1_export_consumed']['tau_if_taken_verbatim']:.2f} and would flag nothing.",
        f"- F3 normalisation applied: {r['normalisation_f3']['applied']} "
        f"({r['normalisation_f3']['stable_px_10m']:,} stable 10 m px used to fit the offsets).",
        f"- λ = {r['lambda']['value']} ({r['lambda']['source']}).", '',
        '### Reproduction check on the SR', '',
        '| date | cached product present | identical (NaN-aware) | max abs diff |', '|---|---|---|---|',
    ] + [f"| {d} | {v.get('cached_file_present')} | {v.get('identical_including_nan_placement')} | "
         f"{v.get('max_abs_diff_where_both_finite')} |"
         for d, v in r['sr_run']['reproduction_check']['cached_per_date_ndvi'].items()] + [
        '',
        f"v1 step3 moments, max |this run − cached|: "
        f"{json.dumps({k: v for k, v in (r['sr_run']['reproduction_check']['v1_step3_moments'] or {}).items() if k.endswith('mean') or k.endswith('std')})}",
        '',
        '## Per-class report (2.5 m, 6.25 m² per pixel)', '',
        '| class | pixels | m² | km² |', '|---|---:|---:|---:|',
    ] + [f"| {n} | {v['pixels_2p5m']:,} | {v['m2']:,.2f} | {v['km2']:.6f} |"
         for n, v in r['class_report'].items()] + [
        '',
        f"Denominator: {r['class_report_denominator']['total_2p5m_px']:,} px "
        f"({r['class_report_denominator']['total_km2']:.4f} km²) in the crop, of which "
        f"{r['class_report_denominator']['valid_2p5m_px']:,} are valid. "
        f"{r['class_report_denominator']['note']}", '',
        f"**Mapped change (v2)** = CORE ∪ ALLOCATED = {r['mapped_change_v2']['px']:,} px = "
        f"**{r['mapped_change_v2']['km2']:.6f} km²**. UNSUPPORTED is never counted as flagged.", '',
        '## Block-sum test on this production map (B7)', '',
        f"- **max |deviation| = {r['block_sum_test']['max_abs_deviation_all_blocks']}** over "
        f"{r['block_sum_test']['n_blocks']:,} 10 m blocks "
        f"({r['block_sum_test']['n_detected_blocks']:,} with round(16f) > 0 — note this is a different "
        f"count from the {r['detection']['n_blocks_detected']:,} blocks inside a conformally detected "
        f"window, many of which round to zero sub-pixels), "
        f"{'PASS' if r['block_sum_test']['pass'] else 'FAIL'}.",
        f"- {r['block_sum_test']['definition']}",
        f"- Recomputed independently of the gate's own call: "
        f"{r['block_sum_test']['recomputed_independently_of_the_gate']}. Scope: "
        f"{r['block_sum_test']['scope']}.", '',
        '## Detection', '',
        f"- τ = {r['detection']['tau']:.6f}; {r['detection']['n_windows_detected']} of "
        f"{r['detection']['n_windows_valid']} valid 160 m windows detected; "
        f"{r['detection']['n_blocks_detected']:,} 10 m blocks detected.",
        f"- τ is used, not merely loaded (x9's B10 bug): "
        + ', '.join(f"{k} → {v['flagged_px']:,} flagged px"
                    for k, v in r['detection']['tau_mutation_on_the_real_map'].items()) + '.', '',
        f"- **n_pre mismatch.** {r['n_pre_mismatch_between_calibration_and_production']['why_it_matters']} "
        f"At a threshold rescaled to match calibration sensitivity the flagged count would be "
        f"{r['n_pre_mismatch_between_calibration_and_production']['flagged_px_at_tau_rescaled_to_match_calibration_sensitivity']:,} "
        f"instead of {r['n_pre_mismatch_between_calibration_and_production']['flagged_px_at_tau']:,}; the "
        f"headline uses τ as calibrated.", '',
        '## v1 vs v2 — only on quantities with identical definitions', '',
        f"{r['v1_vs_v2']['preamble']}", '',
        '| quantity | v1 | v2 | v2 − v1 | IoU |', '|---|---:|---:|---:|---:|',
        f"| mapped change area (km²) | {ma['a_km2']:.6f} | {ma['b_km2']:.6f} | "
        f"{ma['b_minus_a_km2']:+.6f} | {ma['iou']:.4f} |",
        f"| mapped change area (2.5 m px) | {ma['a_px']:,} | {ma['b_px']:,} | {ma['b_minus_a_px']:+,} | — |",
        f"| NO_DATA (2.5 m px) | {c['no_data']['a_px']:,} | {c['no_data']['b_px']:,} | "
        f"{c['no_data']['b_minus_a_px']:+,} | {c['no_data']['iou']:.4f} |", '',
        f"- v1 flagged = `{ma['a_name']}`, v2 flagged = `{ma['b_name']}`. {ma['quantity']}",
        f"- In v1 but not v2: {ma['in_a_not_b_px']:,} px. In v2 but not v1: {ma['in_b_not_a_px']:,} px.", '',
        '### Sub-pixel divergence from a blocky 10 m parent', '',
        f"{bd['definition']}", '',
        f"- 10 m blocks scored: **{bd['n_10m_blocks_scored']:,}**; boundary blocks: "
        f"**{bd['n_10m_boundary_blocks']:,}**.",
        f"- **Boundary 10 m pixels whose 2.5 m allocation differs from the blocky parent: "
        f"{bd['n_10m_boundary_blocks_whose_2p5m_allocation_differs_from_blocky_parent']:,}** "
        f"({100 * (bd['fraction_of_boundary_blocks_differing'] or 0):.1f} % of boundary blocks).",
        f"- Anywhere in the crop: {bd['n_10m_blocks_anywhere_differing_from_blocky_parent']:,} blocks differ; "
        f"{bd['n_10m_blocks_with_genuine_subpixel_structure_0_lt_n_lt_16']:,} blocks carry genuine "
        f"sub-pixel structure (0 < allocated < 16).",
        f"- ⚠ {bd['caveat']}", '',
        '### UNSUPPORTED — NOT compared', '',
        f"{u['reason']}", '',
        f"- v1 UNSUPPORTED: {u['v1']['px']:,} px ({u['v1']['km2']:.6f} km²) — {u['v1']['definition']}",
        f"- v2 UNSUPPORTED: {u['v2']['px']:,} px ({u['v2']['km2']:.6f} km²) — {u['v2']['definition']}",
        f"- No difference, ratio or IoU is computed between them. Enforced by {u['enforced_by']}.", '',
        '## E5 parent-first cascade — forward passes saved', '',
        f"- Tiles: {r['e5_cascade']['tiles']['total']} total, {r['e5_cascade']['tiles']['fired']} fired, "
        f"{r['e5_cascade']['tiles']['ring']} ring, {r['e5_cascade']['tiles']['audit']} audit, "
        f"**{r['e5_cascade']['tiles']['skipped']} skipped**.",
        f"- Forward passes: full re-run baseline **{fp['full_rerun_baseline']:,}** "
        f"(exactly what this script ran) vs cascade **{fp['cascade']:,}** → "
        f"**{fp['saved']:,} saved ({100 * fp['saved_fraction']:.1f} %)**. {fp['definition']}.",
        f"- Cost: {r['e5_cascade']['cost_of_the_saving']['unreconstructed_2p5m_px']:,} 2.5 m px would have "
        f"no SR at all, containing "
        f"{r['e5_cascade']['cost_of_the_saving']['v2_flagged_px_inside_skipped_tiles']:,} of v2's "
        f"{r['e5_cascade']['cost_of_the_saving']['v2_flagged_px_total']:,} flagged px "
        f"({100 * (r['e5_cascade']['cost_of_the_saving']['fraction_of_v2_flagged_lost'] or 0):.2f} %). "
        f"{r['e5_cascade']['cost_of_the_saving']['note']}",
        f"- ⚠ {r['e5_cascade']['crop_note']}",
        f"- **Full AOI for context** ({r['e5_cascade']['full_aoi_selection_for_context']['scope']}; SR was "
        f"NOT run at this scale in F7): "
        f"{r['e5_cascade']['full_aoi_selection_for_context']['tiles']['total']} tiles, "
        f"{r['e5_cascade']['full_aoi_selection_for_context']['tiles']['skipped']} skipped → "
        f"{r['e5_cascade']['full_aoi_selection_for_context']['forward_passes']['cascade']:,} vs "
        f"{r['e5_cascade']['full_aoi_selection_for_context']['forward_passes']['full_rerun_baseline']:,} "
        f"passes, **{r['e5_cascade']['full_aoi_selection_for_context']['forward_passes']['saved']:,} saved "
        f"({100 * r['e5_cascade']['full_aoi_selection_for_context']['forward_passes']['saved_fraction']:.1f} %)**.",
        '',
        '## The figure', '',
        f"- Rendered: {r['wow_figure'].get('rendered')}"
        + (f" → `{r['wow_figure']['png']}`" if r['wow_figure'].get('png') else ''),
        f"- Underlying arrays: `{r['wow_figure']['arrays_npz']}` (window at 2.5 m row "
        f"{r['wow_figure']['window_2p5m_rowcol'][0]}, col {r['wow_figure']['window_2p5m_rowcol'][1]}, "
        f"{r['wow_figure']['window_px']}×{r['wow_figure']['window_px']} px = "
        f"{r['wow_figure']['window_m']:.0f} m).",
        '', f"> {r['wow_figure']['caption']}", '',
        '## Limitations', '',
    ] + [f'- {x}' for x in r['limitations']] + [
        '', '## Verdict', '', r['caveat'], '',
    ]
    path.write_text('\n'.join(L) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
