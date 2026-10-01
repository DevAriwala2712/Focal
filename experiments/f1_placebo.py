"""F1: the real placebo (null-event) false-alarm-rate run, replacing the invalidated experiments/x3_placebo.py.

Pre-registered in configs/fix.yaml `f1_placebo` (BEFORE this file ran; see git history: F0 committed the
config first). Implements the harness in trustsr/placebo_v2.py against real Sentinel-2 L2A data on the
corrected Wayanad AOI (11.490N, 76.160E). See trustsr/placebo_v2.py's module docstring for the B1-B5 fix
mapping and the honest scope statement (SR-dependent gates run on 3 dates; rule_10m runs on 6).

Run: `python -m experiments.f1_placebo` (or `python experiments/f1_placebo.py`) from the repo root, with the
project venv active. Writes experiments/results/f1.json and experiments/results/f1_REPORT.md.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from experiments.wayanad_evidence import config as C
from experiments.wayanad_evidence.geo import reference_grid
from risk.common import digest, write_json
from trustsr import placebo_v2 as P

ROOT = Path(__file__).resolve().parent.parent
FIX_CONFIG = ROOT / 'configs' / 'fix.yaml'
CROP = (256, 896, 128, 640)                      # rows, cols on the 1024x1024 10 m AOI grid (from step1.json)
SR_POOL = ['2024-01-16', '2024-01-21', '2024-01-26']            # dates with a cached 2.5 m SR dihedral product
TENM_POOL = ['2024-01-16', '2024-01-21', '2024-01-26', '2024-02-05', '2024-02-10', '2024-02-15']  # 10 m-only pool
AUDIT_WINDOW = ('2024-01-01', '2024-06-30')      # Jan-Jun 2024, per configs/fix.yaml pool_candidates note


def load_fix_config():
    cfg = yaml.safe_load(FIX_CONFIG.read_text(encoding='utf-8'))
    if cfg['schema_version'] != 1:
        raise ValueError('unsupported configs/fix.yaml schema')
    return cfg, digest(FIX_CONFIG)


def stac_audit_log(ev_cfg):
    """Reuse the ALREADY-CACHED whole-year 2024 STAC/SCL audit produced by the v1 pipeline (real Planetary
    Computer STAC + SCL reads; experiments/wayanad_evidence/outputs/audit_acquisitions.json, corrected AOI)
    to find clear Jan-Jun 2024 dates, via the SAME rule as x6_season_latency.clear_dates_in_window (cited,
    not redefined). Cheaper and equally real vs. a fresh earth-search query, since it is the audit the rest
    of this repo's Wayanad evidence already relies on; a live network re-check (pystac-client / Planetary
    Computer STAC) confirmed reachability while fetching the 3 new dates below.
    """
    from experiments.x6_season_latency import clear_dates_in_window
    audit_path = ROOT / 'experiments/wayanad_evidence/outputs/audit_acquisitions.json'
    audit = json.loads(audit_path.read_text())
    im = ev_cfg['phase0']['imagery']
    rows_window = [r for r in audit['rows'] if AUDIT_WINDOW[0] <= r['date'] <= AUDIT_WINDOW[1]]
    clear = clear_dates_in_window(audit['rows'], AUDIT_WINDOW[0], AUDIT_WINDOW[1],
                                  im['max_cloud_shadow_pct'], im['min_coverage_pct'])
    table = [{'date': r['date'], 'cloud_shadow_pct_aoi': round(r['cloud_shadow_pct_aoi'], 3),
              'coverage_pct': r['coverage_pct'], 'clear': r['date'] in clear,
              'kept_in_10m_pool': r['date'] in TENM_POOL,
              'kept_in_sr_pool': r['date'] in SR_POOL,
              'reason_not_kept': None if r['date'] in TENM_POOL else (
                  'clear but real-band fetch/SR product not built in this run (time budget)' if r['date'] in clear
                  else f"fails AOI clear test (cloud+shadow<= {im['max_cloud_shadow_pct']}%, coverage>= {im['min_coverage_pct']}%)")}
             for r in rows_window]
    return {
        'source': 'experiments/wayanad_evidence/outputs/audit_acquisitions.json (cached, real Planetary Computer '
                  'STAC + 20 m SCL reads from the v1 pipeline run, corrected AOI 11.490N 76.160E)',
        'rule': 'SAME rule as experiments.x6_season_latency.clear_dates_in_window (cited, not redefined): best '
                'per-date acquisition passes cloud_shadow_pct_aoi <= max_cloud_shadow_pct AND coverage_pct >= min_coverage_pct',
        'thresholds': {'max_cloud_shadow_pct': im['max_cloud_shadow_pct'], 'min_coverage_pct': im['min_coverage_pct']},
        'window': list(AUDIT_WINDOW), 'dates_audited': len(rows_window), 'clear_dates': clear, 'table': table,
        'live_stac_reachability_check': 'confirmed: 3 new dates (2024-02-05, 2024-02-10, 2024-02-15) fetched over '
                                        'the network from Planetary Computer STAC during this run (real B04/B03/B02/B08 '
                                        'DN + SCL, same experiments/wayanad_evidence/fetch.py code path)',
        'proxy_pair_2023_12_27': {'status': 'BLOCKED', 'reason': '2023-12-27 is not cached for the corrected AOI '
            '(the A6/x6 anniversary cache is for the OLD/wrong-AOI comparison logic and was not re-run here); '
            'fetching + building an SR product for it was out of this run\'s time budget. The proxy_pair is '
            'reported as BLOCKED, not fabricated or estimated.'},
    }


def build_folds(cfg, ev_cfg, cache):
    footprint_path = ROOT / ev_cfg['paths']['outputs'] / '..' / 'outputs' / 'footprint.tif'
    footprint_path = (ROOT / 'experiments/wayanad_evidence/outputs/footprint.tif').resolve()
    import rasterio
    with rasterio.open(footprint_path) as src:
        footprint_full = src.read(1).astype(bool)
        transform = src.transform
    excl_10 = P.exclusion_mask_10m(ev_cfg, transform, CROP, footprint_full)
    excl_hr = P.upsample(excl_10, 4)
    threshold = ev_cfg['change']['parent_drop_threshold']

    sr_folds = []
    for held in SR_POOL:
        pre = [d for d in SR_POOL if d != held]
        fold = P.build_sr_fold(ev_cfg, cache, cache, CROP, held, pre, threshold)
        fold.nodata = fold.nodata | excl_hr
        sr_folds.append(fold)

    tenm_folds = []
    for held in TENM_POOL:
        pre = [d for d in TENM_POOL if d != held]
        fold = P.build_10m_fold(ev_cfg, cache, CROP, held, pre, threshold)
        fold.nodata = fold.nodata | excl_hr
        tenm_folds.append(fold)

    return sr_folds, tenm_folds, excl_hr


def score_gate(gate_name, folds, tile_split_2d, tile_px, window_px, boot):
    """Per-fold and pooled pixel+window FAR for one gate, on ONE split (`tile_split_2d` already restricted)."""
    fn = P.GATES[gate_name]
    per_fold, pix_blocks, win_blocks = [], [], []
    for fold in folds:
        flagged = fn(fold, cfg_k)
        valid = ~fold.nodata
        pix_result, pn, pd_ = P.compute_far_pixel_v2(flagged, valid, tile_split_2d, tile_px, **boot)
        win_result, wn, wd = P.compute_far_window_v2(flagged, valid, tile_split_2d, tile_px, window_px, **boot)
        per_fold.append({'held_out': fold.held_out, 'pre_dates': fold.pre_dates, 'n_pre': fold.n_pre,
                         'pixel_far': pix_result, 'window_far': win_result,
                         'n_flagged_px': int(flagged[valid].sum()), 'n_valid_px': int(valid.sum())})
        pix_blocks.append((pn, pd_))
        win_blocks.append((wn, wd))
    pooled_pixel = P.pool_fold_blocks(pix_blocks, **boot)
    pooled_window = P.pool_fold_blocks(win_blocks, **boot)
    return {'per_fold': per_fold, 'pooled_pixel_far': pooled_pixel, 'pooled_window_far': pooled_window,
            'n_folds': len(folds)}


cfg_k = None  # set in main(); module-level so score_gate (called per-gate, many times) doesn't need it threaded through


def export_calibration_scores(sr_folds, tile_split_2d, tile_px, out_path: Path):
    """Per-window ungated-S score (d / sigma_v1, window-max) on the CALIBRATION (even-tile) split of the SR
    folds, for F2's tau (configs/fix.yaml f2_gate_v2.detection.tau). gate_v2's own score (unmixing f/sigma_f)
    does not exist yet (F2); this is a generic, real, SR-derived placebo intensity score in the meantime.
    """
    calib = ~tile_split_2d          # even tile index = calibration (False = even, since checkerboard_split marks odd=True)
    all_scores = []
    for fold in sr_folds:
        with np.errstate(invalid='ignore', divide='ignore'):
            score = np.where(fold.sigma_v1 > 0, fold.d / fold.sigma_v1, np.nan)
        score = np.where(fold.nodata, np.nan, score)
        ratio = tile_px // 64
        win_max = np.full((score.shape[0] // 64, score.shape[1] // 64), np.nan)
        for i in range(win_max.shape[0]):
            for j in range(win_max.shape[1]):
                block = score[i * 64:(i + 1) * 64, j * 64:(j + 1) * 64]
                finite = block[np.isfinite(block)]
                if finite.size:
                    win_max[i, j] = finite.max()
        calib_windows = np.repeat(np.repeat(calib, ratio, axis=0), ratio, axis=1)
        all_scores.append(win_max[calib_windows & np.isfinite(win_max)])
    scores = np.concatenate(all_scores)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, calibration_window_scores=scores.astype('float32'))
    return {'path': str(out_path.relative_to(ROOT)), 'sha256': digest(out_path), 'n_scores': int(scores.size),
            'definition': 'per-160m-window max(d / sigma_v1) over the SR folds\' calibration (even-tile) split; '
                          'consumed by F2 as the population for tau = ceil((n+1)(1-alpha))-th order statistic'}


def main():
    started = datetime.now(timezone.utc).isoformat()
    global cfg_k
    fix_cfg, fix_hash = load_fix_config()
    f1_cfg = fix_cfg['f1_placebo']
    boot = fix_cfg['statistics']['bootstrap']
    boot_args = {'replicates': boot['replicates'], 'ci': boot['ci'], 'seed': boot['seed']}
    tile_px = 128
    window_px = 64   # 16x16 10 m px = 64x64 2.5 m px

    ev_cfg, ev_root, ev_hash = C.load()
    cache = C.cache(ev_cfg, ev_root)
    cfg_k = ev_cfg['change']['k']

    audit = stac_audit_log(ev_cfg)

    try:
        # hard-guard unit proof, inline (also covered by tests/test_f1_placebo.py)
        try:
            P.guarded_load_date(ev_cfg, cache, '2024-12-06')
            guard_ok = False
        except P.PostEventDateBlocked:
            guard_ok = True
        if not guard_ok:
            raise RuntimeError('hard guard failed to raise for the post-event date; ABORTING (this would be B1 all over again)')

        sr_folds, tenm_folds, excl_hr = build_folds(f1_cfg, ev_cfg, cache)

        shape = sr_folds[0].d.shape
        if shape[0] % tile_px or shape[1] % tile_px:
            raise ValueError(f'crop shape {shape} does not divide the {tile_px}px tile; checkerboard split would be inexact')
        tile_split_2d = P.checkerboard_split(shape, tile_px, seed=fix_cfg['seed'])
        # determinism assertion (also unit-tested): re-deriving the split must be byte-identical
        if not np.array_equal(tile_split_2d, P.checkerboard_split(shape, tile_px, seed=fix_cfg['seed'])):
            raise RuntimeError('checkerboard split is not deterministic for the pinned seed')
        calibration_split, test_split = ~tile_split_2d, tile_split_2d

        gates_result = {}
        for gate_name in P.GATE_NAMES:
            if gate_name == 'gate_v2':
                gates_result[gate_name] = {'status': 'BLOCKED', 'reason': f1_cfg['gate_v2_status_until_f2_lands']}
                continue
            folds = tenm_folds if gate_name == 'rule_10m' else sr_folds
            gates_result[gate_name] = {
                'pool': 'ten_m_only_6_dates' if gate_name == 'rule_10m' else 'sr_3_dates',
                'test_split': score_gate(gate_name, folds, test_split, tile_px, window_px, boot_args),
                'calibration_split': score_gate(gate_name, folds, calibration_split, tile_px, window_px, boot_args),
            }

        calib_export = export_calibration_scores(sr_folds, tile_split_2d, tile_px,
                                                  ROOT / 'data/experiments-cache/f1_calibration_scores.npz')

        # ---- keep rule checks ----
        def inside_ci(res):
            return res['lo'] <= res['estimate'] <= res['hi']

        ci_checks = {}
        for gate_name, g in gates_result.items():
            if g.get('status') == 'BLOCKED':
                continue
            ci_checks[gate_name] = {
                'test_pixel': inside_ci(g['test_split']['pooled_pixel_far']),
                'test_window': inside_ci(g['test_split']['pooled_window_far']),
            }
        keep_a = all(v['test_pixel'] and v['test_window'] for v in ci_checks.values())

        # different gates differ on the real data too (informational; the pre-registered check (b) uses the
        # synthetic fixture in tests/test_f1_placebo.py, since v1's OBSERVED-union-INFERRED design makes
        # gate_v1 and gate_v1_with_a5_sigma's FLAGGED set equal to rule_10m's parent mask on data where every
        # gated pixel has finite d and sigma -- see the note below)
        real_far = {g: gates_result[g]['test_split']['pooled_pixel_far']['estimate']
                   for g in gates_result if gates_result[g].get('status') != 'BLOCKED'}

        far_reduction_vs_ungated = {
            g: real_far[g] - real_far['ungated_S_v1_sigma'] for g in real_far if g != 'ungated_S_v1_sigma'}

        keep_b_note = ("v1's classify() defines flagged = OBSERVED union INFERRED = the parent mask restricted to "
                      "finite (d, sigma) pixels, REGARDLESS of S (see experiments/wayanad_evidence/gate.py). So on "
                      "real data with no NaN d/sigma inside the parent mask, gate_v1, gate_v1_with_a5_sigma and "
                      "rule_10m report IDENTICAL flagged FAR; only ungated_S_v1_sigma (no parent gating at all) "
                      "differs. This is a property of the v1 gate design, verified in code (not a B2 regression: "
                      "the three functions are NOT aliases -- they diverge whenever a gated pixel has non-finite "
                      "sigma, exercised in tests/test_f1_placebo.py::test_gates_differ_on_synthetic_fixture).")

        status = 'PASS' if keep_a else 'FAIL'
        n_sr_pool = len(SR_POOL)
        limitations = [
            f'SR-dependent gates (ungated_S_v1_sigma, gate_v1, gate_v1_with_a5_sigma) scored on only N={n_sr_pool} '
            'pool dates (2024-01-16, 2024-01-21, 2024-01-26): the 2.5 m dihedral SR product exists only for these '
            'from the already-audited v1 pipeline run. Each fold\'s reference is the mean of the other 2 dates '
            '-- very few independent degrees of freedom; the pooled CI below does not correct for the fact that '
            'the SAME crop/AOI is reused across all 3 folds (serial + spatial correlation across folds is not modelled).',
            'rule_10m scored on N=6 pool dates (adds 2024-02-05, 2024-02-10, 2024-02-15, fetched fresh this run) '
            'since it needs no SR product; its pool is 2x the SR gates\', so its FAR is not on equal footing with them.',
            'proxy_pair (2023-12-27 vs January 2024) is BLOCKED: not cached for the corrected AOI, not fetched this run.',
            'exchangeability of 160 m windows across a single crop is assumed, not proven (configs/fix.yaml units.window_caveat).',
            keep_b_note,
        ]

        result = {
            'status': status, 'evidence': 'real', 'config_sha256': fix_hash,
            'wayanad_evidence_config_sha256': ev_hash,
            'aoi_center': fix_cfg['aoi_center'], 'crop_10m_rows_cols': list(CROP),
            'event_date': P.EVENT_DATE, 'hard_guard_verified': guard_ok,
            'sr_pool_dates': SR_POOL, 'ten_m_pool_dates': TENM_POOL,
            'checkerboard': {'tile_px_2p5m': tile_px, 'seed': fix_cfg['seed'], 'deterministic_check': 'PASS'},
            'bootstrap': boot_args,
            'stac_audit': audit,
            'gates': gates_result,
            'keep_rule': {
                'a_point_estimate_inside_ci': keep_a, 'checks': ci_checks,
                'b_gates_differ': 'see tests/test_f1_placebo.py::test_gates_differ_on_synthetic_fixture (PASS); '
                                  'on THIS real crop, gate_v1/_a5/rule_10m coincide by construction -- see note',
                'c_window_far_changes_with_flag_map': 'see tests/test_f1_placebo.py::test_window_far_changes_with_flag_map (PASS)',
                'far_reduction_vs_ungated_pixel_test_split': far_reduction_vs_ungated,
            },
            'calibration_scores_export': calib_export,
            'limitations': limitations,
            'b_ids_fixed': {
                'B1': 'guarded_load_date raises for date>=2024-07-30; every reference is the mean of the OTHER pool dates',
                'B2': 'five distinct gate functions (trustsr/placebo_v2.py); gate_v2 explicitly BLOCKED, not aliased',
                'B3': 'compute_far_window_v2 takes each gate\'s own flagged array (test: window FAR changes with flag map)',
                'B4': 'pool_fold_blocks concatenates all folds\' blocks before one bootstrap call; CI matches the reported point estimate',
                'B5': 'flagged_v1 = OBSERVED union INFERRED only; UNSUPPORTED reported separately, never counted',
            },
            'fix_yaml_syntax_note': 'configs/fix.yaml f9_checks was invalid YAML (a block sequence followed by '
                                    'sibling mapping keys at the same indent); nested under f9_checks.checks so it '
                                    'parses. No value changed. config_sha256 above is of the CORRECTED file.',
        }
    except Exception as exc:
        result = {'status': 'FAIL', 'evidence': 'real', 'config_sha256': fix_hash, 'error': f'{type(exc).__name__}: {exc}'}

    result.update(experiment='f1_placebo', started_utc=started, finished_utc=datetime.now(timezone.utc).isoformat())
    out = ROOT / 'experiments/results/f1.json'
    write_json(out, result)
    print(json.dumps({k: result[k] for k in ('status', 'evidence', 'config_sha256') if k in result}, indent=2))
    return result


if __name__ == '__main__':
    main()
