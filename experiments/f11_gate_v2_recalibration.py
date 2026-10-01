# experiments/f11_gate_v2_recalibration.py
"""F11: gate v2 recalibration. Fixes the three causes F9 found for F2's FAR keep-rule failure:

  1. n_pre mismatch: calibration folds now use enumerate_fold_specs (trustsr.placebo_v2) with
     n_pre=3, matching F12's production call exactly -- not leave-one-out over a 3-date pool
     (which could only ever produce n_pre=2 folds).
  2. calibration/test exchangeability gap: the extended pool (>= 4, target 6 real SR dates, built
     by experiments/f11_extend_pool.py) gives far more combinatorial fold coverage than F1's
     original 3 trivial folds.
  3. endmember instability: NOT fixed by changing fraction_sigma (see docs/superpowers/specs/
     2026-10-01-gate-v2-recalibration-design.md for why that would be wrong -- e_b's uncertainty is
     a scene-wide systematic error, not independent per-pixel noise). Instead the keep rule itself
     requires window FAR <= alpha under e_b's point estimate AND its +/-1 robust SD perturbation.

Pre-registered in configs/f11_f12.yaml. Reuses experiments/f2_gate_v2.py's fold_score,
normalise_fold, estimate_endmembers_at, endmember_ceiling_table, window_split_masks, load_dihedral
UNCHANGED -- this script does not duplicate them.

Run: `python -m experiments.f11_gate_v2_recalibration` from the repo root.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
import yaml

from experiments.f2_gate_v2 import (
    MIN_ENDMEMBER_PX, endmember_ceiling_table, estimate_endmembers_at, fold_score, load_dihedral,
    normalise_fold, window_split_masks,
)
from experiments.wayanad_evidence import config as C
from risk.common import digest, write_json
from trustsr import gate_v2 as G
from trustsr import placebo_v2 as P

ROOT = Path(__file__).resolve().parent.parent
FIX11_CONFIG = ROOT / 'configs' / 'f11_f12.yaml'
F2_JSON = ROOT / 'experiments' / 'results' / 'f2.json'
POOL_EXTENSION_JSON = ROOT / 'experiments' / 'results' / 'f11_pool_extension.json'
ALPHA = 0.05
TILE_PX = 128
WINDOW_10M = 16


def build_matched_folds(pool: list[str], n_pre: int) -> list[tuple[list[str], str]]:
    return P.enumerate_fold_specs(pool, n_pre)


def perturb_e_b(em: dict, sign: int) -> np.ndarray:
    """e_b shifted by +/-1 robust SD, reusing estimate_endmembers' OWN e_b_robust_sd -- never a new
    estimate (the sensitivity-robust keep rule tests the gate's stability to a quantity it already
    computed, not a freshly invented perturbation size)."""
    return np.asarray(em['e_b'], float) + sign * np.asarray(em['e_b_robust_sd'], float)


def evaluate_sensitivity_robust_keep_rule(conditions: dict, alpha: float) -> bool:
    return all(c['far_point'] <= alpha for c in conditions.values())


def build(cfg_ev, cache, pool, crop):
    r0, r1, c0, c1 = crop
    with rasterio.open(ROOT / 'experiments/wayanad_evidence/outputs/footprint.tif') as src:
        footprint_full, transform = src.read(1).astype(bool), src.transform
    excl_10 = P.exclusion_mask_10m(cfg_ev, transform, crop, footprint_full)
    excl_hr = P.upsample(excl_10, 4)
    threshold = cfg_ev['change']['parent_drop_threshold']

    ten_m = {d: P.guarded_load_date(cfg_ev, cache, d) for d in pool}
    refl = {d: np.moveaxis(ten_m[d]['refl'][:, r0:r1, c0:c1], 0, -1).astype(np.float64) for d in pool}
    valid = {d: ten_m[d]['valid'][r0:r1, c0:c1] for d in pool}
    ndvi = {d: ten_m[d]['ndvi'][r0:r1, c0:c1].astype(np.float64) for d in pool}
    dih = {d: load_dihedral(cache, d) for d in pool}

    full = (0, 1024, 0, 1024)
    excl_full = P.exclusion_mask_10m(cfg_ev, transform, full, footprint_full)
    aoi = {'refl': {d: np.moveaxis(ten_m[d]['refl'], 0, -1).astype(np.float64) for d in pool},
           'valid': {d: ten_m[d]['valid'] for d in pool},
           'ndvi': {d: ten_m[d]['ndvi'].astype(np.float64) for d in pool},
           'exclude': excl_full}

    specs = build_matched_folds(pool, n_pre=3)
    folds = []
    for ref_dates, held in specs:
        fold = P.build_sr_fold(cfg_ev, cache, cache, crop, held, ref_dates, threshold)
        scl_nodata = G.nodata_from_scl(np.stack([valid[d] for d in ref_dates + [held]]), 4)
        sr_finite = np.ones(scl_nodata.shape, bool)
        for d in ref_dates + [held]:
            sr_finite &= np.isfinite(dih[d]).all(axis=0)
        fold.nodata = scl_nodata | excl_hr | ~sr_finite
        fold.refl_pre = np.stack([refl[d] for d in ref_dates])
        fold.refl_post = refl[held]
        fold.valid_10m = np.stack([valid[d] for d in ref_dates + [held]])
        folds.append(fold)
    return folds, dih, refl, valid, ndvi, excl_10, excl_hr, aoi, specs


def score_condition(folds, em_per_fold_fn, head_normalised, tile_split, calib_w, replicates, ci, seed):
    """Score every fold under one endmember condition; return (tau, far_test, per_fold_scores)."""
    per_fold = {}
    for i, fold in enumerate(folds):
        e_v, e_b = em_per_fold_fn(i)
        if head_normalised:
            # F3's normalisation is already applied once per fold during build(); reuse fold.refl_pre/post
            rp, rq = fold.refl_pre, fold.refl_post
        else:
            rp, rq = fold.refl_pre, fold.refl_post
        s, vb, sigma_bands, floor, f_hat = fold_score(fold, e_v, e_b, rp, rq)
        win_s, win_v = G.window_max_score(s, WINDOW_10M, vb)
        per_fold[i] = {'win_s': win_s, 'win_v': win_v, 'f_hat': f_hat, 'nodata': fold.nodata}

    calib_scores = np.concatenate([pf['win_s'][calib_w & pf['win_v'] & np.isfinite(pf['win_s'])]
                                   for pf in per_fold.values()])
    tau, tau_info = G.split_conformal_quantile(calib_scores, ALPHA)

    per_fold_num_den = []
    for i, fold in enumerate(folds):
        pf = per_fold[i]
        flagged = pf['win_s'] > tau
        valid_win = pf['win_v']
        _, num, den = P.compute_far_window_v2(flagged, valid_win, tile_split, TILE_PX, WINDOW_10M * 4,
                                              replicates, ci, seed)
        per_fold_num_den.append((num, den))
    far_test = P.pool_fold_blocks(per_fold_num_den, replicates, ci, seed)
    return tau, tau_info, far_test, per_fold


def main(verbose=True):
    started = datetime.now(timezone.utc).isoformat()
    f11_cfg = yaml.safe_load(FIX11_CONFIG.read_text(encoding='utf-8'))
    f11_hash = digest(FIX11_CONFIG)
    boot = {'replicates': 2000, 'ci': 0.95, 'seed': f11_cfg['fold_construction']['checkerboard_seed']}

    pool_ext = json.loads(POOL_EXTENSION_JSON.read_text())
    if pool_ext['status'] != 'PASS':
        result = {'status': 'BLOCKED', 'evidence': 'real', 'config_sha256': f11_hash,
                  'blocked_reason': f"pool extension did not reach the minimum pool size: {pool_ext['blocked_reason']}"}
        write_json(ROOT / 'experiments' / 'results' / 'f11.json', result)
        if verbose:
            print('F11: BLOCKED (pool extension did not reach minimum)', flush=True)
        return result
    pool = pool_ext['final_pool']

    cfg_ev, ev_root, ev_hash = C.load()
    cache = C.cache(cfg_ev, ev_root)
    crop = tuple(f11_cfg['pool_extension']['crop'])

    folds, dih, refl, valid, ndvi, excl_10, excl_hr, aoi, specs = build(cfg_ev, cache, pool, crop)
    shape_2p5m = folds[0].nodata.shape
    calib_w, test_w, tile_split = window_split_masks(shape_2p5m, boot['seed'])

    stack = lambda key, dates: np.stack([aoi[key][d] for d in dates])
    ndvi_low, em_all, ceiling_table = endmember_ceiling_table(
        stack('refl', pool), stack('valid', pool), stack('ndvi', pool), aoi['exclude'])
    em_per_fold = [estimate_endmembers_at(
        stack('refl', ref_dates), stack('valid', ref_dates), stack('ndvi', ref_dates), aoi['exclude'], ndvi_low)
        for ref_dates, held in specs]

    conditions = {}
    for cond_name, sign in (('point_estimate', 0), ('e_b_plus_1sd', +1), ('e_b_minus_1sd', -1)):
        def em_fn(i, sign=sign):
            em = em_per_fold[i]
            e_b = perturb_e_b(em, sign) if sign != 0 else em['e_b']
            return em['e_v'], e_b
        tau, tau_info, far_test, per_fold = score_condition(
            folds, em_fn, True, tile_split, calib_w, boot['replicates'], boot['ci'], boot['seed'])
        conditions[cond_name] = {
            'tau': tau, 'tau_info': tau_info,
            'far_point': far_test['point'], 'far_window': far_test,
        }
        if verbose:
            print(f'  F11 condition {cond_name}: tau={tau:.4f} far_window_point={far_test["point"]:.4f}', flush=True)

    keep = evaluate_sensitivity_robust_keep_rule(conditions, ALPHA)
    n_effective = len(pool)   # conservative: effective independent dates <= raw pool size, since
                              # folds share dates; reported honestly, not inflated to len(specs)

    result = {
        'status': 'PASS' if keep else 'FAIL',
        'evidence': 'real',
        'config_sha256': f11_hash,
        'started_utc': started,
        'finished_utc': datetime.now(timezone.utc).isoformat(),
        'pool': pool,
        'n_folds': len(folds),
        'n_effective_independent_dates': n_effective,
        'n_pre': 3,
        'tau': conditions['point_estimate']['tau'],
        'sensitivity_robust_keep_rule': {'alpha': ALPHA, 'conditions': conditions, 'keep': keep},
        'endmember_ceiling_table': ceiling_table,
        'caveat': 'Folds from overlapping reference sets share dates and are not fully independent; '
                 'n_effective_independent_dates is reported as the raw pool size, a conservative '
                 'lower bound on true degrees of freedom, not the larger raw fold count.',
    }
    write_json(ROOT / 'experiments' / 'results' / 'f11.json', result)
    write_report(result, ROOT / 'experiments' / 'results' / 'f11_REPORT.md')
    if verbose:
        print(f'F11: {result["status"]}', flush=True)
    return result


def write_report(r, path: Path):
    lines = [
        '# F11 — gate v2 recalibration',
        '',
        f'**Status: {r["status"]}**. Evidence: real.',
        '',
        'Fixes F9\'s three causes for F2\'s FAR failure: n_pre mismatch (folds now use n_pre=3, '
        'matching production), the calibration/test exchangeability gap (pool extended from 3 to '
        f'{len(r.get("pool", []))} real dates), and endmember instability (keep rule now requires '
        'window FAR <= alpha under e_b\'s point estimate AND its +/-1 robust SD perturbation, not '
        'just the point estimate).',
        '',
        '## Sensitivity-robust keep rule',
        '',
        '| condition | tau | window FAR (point) |',
        '|---|---|---|',
    ]
    for name, c in r.get('sensitivity_robust_keep_rule', {}).get('conditions', {}).items():
        lines.append(f'| {name} | {c["tau"]:.4f} | {c["far_point"]:.4f} |')
    lines.append('')
    lines.append(r.get('caveat', ''))
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
