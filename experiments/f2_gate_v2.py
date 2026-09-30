"""F2: the gate-v2 rebuild run. Replaces the invalidated experiments/x4_calibration.py +
experiments/x9_v2_production.py gate-v2 path (experiments/results/INVALIDATED.md, B6-B12).

Pre-registered in configs/fix.yaml `f2_gate_v2` (committed in Wave 0, before this file ran). The
implementation lives in trustsr/gate_v2.py; the rationale and the B-ID mapping are in
docs/adr-f2-gate-v2.md. This script:

  1. estimates the two mixing endmembers FROM DATA on the pre-event dates, outside the 1500 m
     plausibility disks and the landslide footprint, and records their counts and a +/- 1 robust-SD
     sensitivity check                                                                          (B12)
  2. computes the per-10 m-block change fraction f by two-endmember unmixing in reflectance and the
     block score s = f / sigma_f by error propagation
  3. derives tau as the ceil((n+1)(1-alpha))-th order statistic of the CALIBRATION (even-tile)
     PLACEBO window scores of F1's folds -- null-event data only                              (B6, B10)
  4. allocates exactly round(16*f) sub-pixels per detected block, ranked by the TRUE 2.5 m SR NDVI
     difference from the cached real E1-tiled / E8-dihedral SEN2SR-lite stack, plus lam * neighbour
     agreement, with lam fitted on the calibration split only                                    (B9)
  5. builds NO_DATA from the SCL validity mask on every date                                    (B11)
  6. runs the real per-block sum test (two independent paths)                                    (B7)
  7. scores gate_v2 in F1's harness on the TEST (odd-tile) split and writes f2.json / f2_REPORT.md
     with configs/fix.yaml's own sha256                                                          (B8)

Run: `python -m experiments.f2_gate_v2` from the repo root.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from experiments.f3_noise_v2 import normalise_dates
from experiments.wayanad_evidence import config as C
from risk.common import digest, write_json
from trustsr import gate_v2 as G
from trustsr import placebo_v2 as P

ROOT = Path(__file__).resolve().parent.parent
FIX_CONFIG = ROOT / 'configs' / 'fix.yaml'
CALIB_SCORES = ROOT / 'data/experiments-cache/f1_calibration_scores.npz'
CALIB_SCORES_SHA256_EXPECTED = '53a2fe8a3a7852d7648d6d8d17abaf982b1435d23ffb0afcd9fda8effd2679e4'

CROP = (256, 896, 128, 640)                                # same crop as F1 (rows, cols on the 1024 10 m grid)
SR_POOL = ['2024-01-16', '2024-01-21', '2024-01-26']       # the only dates with a cached real 2.5 m SR product
TILE_PX = 128                                              # 2.5 m px per block-bootstrap tile
WINDOW_PX = 64                                             # 2.5 m px per 160 m window = 16x16 10 m blocks
WINDOW_10M = 16
ALPHA = 0.05
LAM_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]                  # capped at 0.5 so the SR rank term always dominates


# ---------------------------------------------------------------- data ----------------------------------------------------------------

def load_dihedral(cache_root: Path, date: str) -> np.ndarray:
    return np.load(cache_root / 'per_date_ndvi' / f'{date}_dihedral_means.npy')


def build(cfg_ev, cache):
    """Folds (leave-one-date-out over the 3 SR pool dates) plus the 10 m reflectance / SCL validity that
    gate_v2 needs. Every date is loaded through P.guarded_load_date, so the post-event image can never
    enter (B1's guard, inherited from F1)."""
    import rasterio
    r0, r1, c0, c1 = CROP
    with rasterio.open(ROOT / 'experiments/wayanad_evidence/outputs/footprint.tif') as src:
        footprint_full, transform = src.read(1).astype(bool), src.transform
    excl_10 = P.exclusion_mask_10m(cfg_ev, transform, CROP, footprint_full)          # (h,w) bool, True = excluded
    excl_hr = P.upsample(excl_10, 4)
    threshold = cfg_ev['change']['parent_drop_threshold']

    ten_m = {d: P.guarded_load_date(cfg_ev, cache, d) for d in SR_POOL}
    refl = {d: np.moveaxis(ten_m[d]['refl'][:, r0:r1, c0:c1], 0, -1).astype(np.float64) for d in SR_POOL}
    valid = {d: ten_m[d]['valid'][r0:r1, c0:c1] for d in SR_POOL}                    # SCL validity & finite NDVI
    ndvi = {d: ten_m[d]['ndvi'][r0:r1, c0:c1].astype(np.float64) for d in SR_POOL}
    dih = {d: load_dihedral(cache, d) for d in SR_POOL}                              # (8,H,W) real SR NDVI, 2.5 m

    # Endmembers are estimated over the WHOLE 1024x1024 AOI, not only F1's 640x512 crop: configs/fix.yaml
    # says "pre-event pixels ... outside disks+footprint" and says nothing about the crop, and the wider
    # sample matters because bare ground is extremely rare here (see the deviation note in f2.json).
    full = (0, 1024, 0, 1024)
    excl_full = P.exclusion_mask_10m(cfg_ev, transform, full, footprint_full)
    aoi = {'refl': {d: np.moveaxis(ten_m[d]['refl'], 0, -1).astype(np.float64) for d in SR_POOL},
           'valid': {d: ten_m[d]['valid'] for d in SR_POOL},
           'ndvi': {d: ten_m[d]['ndvi'].astype(np.float64) for d in SR_POOL},
           'exclude': excl_full}

    folds = []
    for held in SR_POOL:
        pre = [d for d in SR_POOL if d != held]
        fold = P.build_sr_fold(cfg_ev, cache, cache, CROP, held, pre, threshold)
        # B11: NO_DATA from the SCL validity mask on EVERY date, nearest-neighbour to 2.5 m, union the
        # exclusion geometry and the pixels where the SR product itself is non-finite.
        scl_nodata = G.nodata_from_scl(np.stack([valid[d] for d in pre + [held]]), 4)
        sr_finite = np.ones(scl_nodata.shape, bool)
        for d in pre + [held]:
            sr_finite &= np.isfinite(dih[d]).all(axis=0)
        fold.nodata = scl_nodata | excl_hr | ~sr_finite
        fold.refl_pre = np.stack([refl[d] for d in pre])
        fold.refl_post = refl[held]
        fold.valid_10m = np.stack([valid[d] for d in pre + [held]])
        folds.append(fold)
    return folds, dih, refl, valid, ndvi, excl_10, excl_hr, aoi


# ---- the e_b pre-registration deviation, handled in the open ----------------------------------------
NDVI_LOW_GRID = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]      # 0.20 is the pre-registered rule
MIN_ENDMEMBER_PX = 100


def endmember_ceiling_table(refl_s, valid_s, ndvi_s, exclude):
    """How many pixels satisfy the e_b rule at each NDVI ceiling, and the resulting e_b.

    configs/fix.yaml f2_gate_v2.endmembers.e_b pre-registers 'NDVI <= 0.2'. On this AOI in January that
    rule selects essentially nothing (the Western Ghats crop is evergreen; there is no bare ground
    outside the landslide footprint, which is excluded by construction). The escalation below is
    deterministic, is applied BEFORE any FAR or block-sum number is computed, and the entire table is
    reported so the reader can see exactly what the pre-registered ceiling would have given.
    """
    rows, chosen = [], None
    for ceil_ in NDVI_LOW_GRID:
        try:
            em = estimate_endmembers_at(refl_s, valid_s, ndvi_s, exclude, ceil_)
            rows.append({'ndvi_low': ceil_, 'e_b_count': em['e_b_count'],
                         'e_b': [float(v) for v in em['e_b']],
                         'meets_min_count': em['e_b_count'] >= MIN_ENDMEMBER_PX})
            if chosen is None:
                chosen = (ceil_, em)
        except ValueError:
            n = int((np.asarray(valid_s).all(axis=0) & ~np.asarray(exclude)
                     & (np.asarray(ndvi_s) <= ceil_).all(axis=0)).sum())
            rows.append({'ndvi_low': ceil_, 'e_b_count': n, 'e_b': None, 'meets_min_count': False})
    if chosen is None:
        raise ValueError('no NDVI ceiling in the grid yields enough bare-ground pixels')
    return chosen[0], chosen[1], rows


def estimate_endmembers_at(refl_s, valid_s, ndvi_s, exclude, ndvi_low):
    return G.estimate_endmembers(refl_s, valid_s, ndvi_s, exclude,
                                 ndvi_low=ndvi_low, min_count=MIN_ENDMEMBER_PX)


def half_ensemble_d(dih, pre, held, halves=((0, 1, 2, 3), (4, 5, 6, 7))):
    """Split-half SR change fields: mean over dihedral runs in each half. Used ONLY to fit lam."""
    out = []
    for hs in halves:
        pre_mean = np.mean([dih[d][list(hs)].mean(axis=0) for d in pre], axis=0)
        out.append(pre_mean - dih[held][list(hs)].mean(axis=0))
    return out


# ---------------------------------------------------------------- pipeline pieces ----------------------------------------------------------------

def normalise_fold(fold, stable_mask):
    """F3's per-date robust median-offset normalisation (configs/fix.yaml f3_noise_v2.interaction_test:
    'if kept, apply the same normalisation in F2'), applied per band in reflectance. The offsets are fit
    on stable pixels outside disks + footprint; the held-out date's offset is fit against the pool
    reference exactly as a deployed date's would be."""
    pre = np.asarray(fold.refl_pre)                                # (T,h,w,B)
    post = np.asarray(fold.refl_post)                              # (h,w,B)
    npre, npost, offs = np.empty_like(pre), np.empty_like(post), {}
    for b in range(pre.shape[-1]):
        p, q, op, oq, _ = normalise_dates(pre[..., b], stable_mask, post[..., b], stable_mask)
        npre[..., b], npost[..., b] = p, q
        offs[f'band{b}'] = {'pool': [float(v) for v in op], 'held_out': float(oq)}
    return npre, npost, offs


def fold_score(fold, e_v, e_b, refl_pre, refl_post):
    """(s_block, valid_block, sigma_bands, f_hat) for one fold -- gate_v2's OWN score, no SR involved."""
    sigma_bands, floor = G.band_sigma_from_pre(refl_pre, valid=None)
    f_hat, _ = G.unmix_fraction(refl_pre, refl_post, e_v, e_b)
    sigma_f = G.fraction_sigma(sigma_bands, e_v, e_b, n_pre=refl_pre.shape[0])
    with np.errstate(invalid='ignore', divide='ignore'):
        s = np.where(sigma_f > 0, f_hat / sigma_f, np.nan)
    block_invalid = G._within_block(fold.nodata.astype(np.int64)).sum(axis=-1) > 0
    return s, ~block_invalid & np.isfinite(s), sigma_bands, floor, f_hat


def window_split_masks(shape_2p5m, seed):
    """(calibration, test) boolean masks on the 40x32 window grid, derived from F1's checkerboard split."""
    tile_split = P.checkerboard_split(shape_2p5m, TILE_PX, seed=seed)     # True = test (odd tiles)
    ratio = TILE_PX // WINDOW_PX
    test_w = np.repeat(np.repeat(tile_split, ratio, axis=0), ratio, axis=1)
    return ~test_w, test_w, tile_split


# ---------------------------------------------------------------- main ----------------------------------------------------------------

def main():
    started = datetime.now(timezone.utc).isoformat()
    fix_cfg = yaml.safe_load(FIX_CONFIG.read_text(encoding='utf-8'))
    fix_hash = digest(FIX_CONFIG)
    boot = fix_cfg['statistics']['bootstrap']
    boot_args = {'replicates': boot['replicates'], 'ci': boot['ci'], 'seed': boot['seed']}
    seed = fix_cfg['seed']

    cfg_ev, ev_root, ev_hash = C.load()
    cache = C.cache(cfg_ev, ev_root)
    k_v1 = cfg_ev['change']['k']

    calib_sha = digest(CALIB_SCORES)
    f1_scores = np.load(CALIB_SCORES)['calibration_window_scores'].astype(np.float64)

    folds, dih, refl, valid, ndvi, excl_10, excl_hr, aoi = build(cfg_ev, cache)
    shape_2p5m = folds[0].nodata.shape
    calib_w, test_w, tile_split = window_split_masks(shape_2p5m, seed)

    # ---- (1) endmembers, estimated from data, never from labels (B12) ------------------------------
    stack = lambda key, dates: np.stack([aoi[key][d] for d in dates])
    ndvi_low, em_all, ceiling_table = endmember_ceiling_table(
        stack('refl', SR_POOL), stack('valid', SR_POOL), stack('ndvi', SR_POOL), aoi['exclude'])
    em_per_fold = {fold.held_out: estimate_endmembers_at(
        stack('refl', fold.pre_dates), stack('valid', fold.pre_dates), stack('ndvi', fold.pre_dates),
        aoi['exclude'], ndvi_low) for fold in folds}
    em_sensitivity_by_ceiling = ceiling_table

    # stable-pixel mask for F3's normalisation fit: valid everywhere, outside disks + footprint, and
    # temporally stable in NDVI (the same <= 0.05 date-to-date std rule cited in configs/wayanad_evidence.yaml)
    nd_stack = np.stack([ndvi[d] for d in SR_POOL])
    stable_mask = (np.stack([valid[d] for d in SR_POOL]).all(axis=0) & ~excl_10
                   & (nd_stack.std(axis=0, ddof=1) <= 0.05) & np.isfinite(nd_stack).all(axis=0))

    # ---- (2,3) per-fold scores and tau, for the normalised and un-normalised tracks -----------------
    tracks = {}
    for track in ('normalised', 'no_normalisation'):
        per_fold, offsets = {}, {}
        for fold in folds:
            em = em_per_fold[fold.held_out]
            if track == 'normalised':
                rp, rq, off = normalise_fold(fold, stable_mask)
                offsets[fold.held_out] = off
            else:
                rp, rq = fold.refl_pre, fold.refl_post
            s, vb, sigma_bands, floor, f_hat = fold_score(fold, em['e_v'], em['e_b'], rp, rq)
            win_s, win_v = G.window_max_score(s, WINDOW_10M, vb)
            per_fold[fold.held_out] = {'s': s, 'valid_block': vb, 'sigma_bands': sigma_bands,
                                       'sigma_floor': floor, 'f_hat': f_hat, 'refl_pre': rp, 'refl_post': rq,
                                       'win_s': win_s, 'win_v': win_v}
        calib_scores = np.concatenate([pf['win_s'][calib_w & pf['win_v'] & np.isfinite(pf['win_s'])]
                                       for pf in per_fold.values()])
        tau, tau_info = G.split_conformal_quantile(calib_scores, ALPHA)
        tracks[track] = {'per_fold': per_fold, 'tau': tau, 'tau_info': tau_info, 'offsets': offsets,
                         'calib_scores': calib_scores}

    head = tracks['normalised']

    # ---- (4) lam fitted on the CALIBRATION split only, by split-half dihedral reproducibility --------
    calib_px = P.expand_checkerboard(~tile_split, TILE_PX, shape_2p5m)
    lam_rows = []
    for lam in LAM_GRID:
        jac = []
        for fold in folds:
            pf = head['per_fold'][fold.held_out]
            f_eff = np.where(pf['valid_block'], np.nan_to_num(pf['f_hat'], nan=0.0), 0.0)
            da, db = half_ensemble_d(dih, fold.pre_dates, fold.held_out)
            aa, _ = G.allocate_pixels(f_eff, da, lam=lam)
            ab, _ = G.allocate_pixels(f_eff, db, lam=lam)
            aa, ab = aa & calib_px, ab & calib_px
            union = (aa | ab).sum()
            jac.append(float((aa & ab).sum() / union) if union else np.nan)
        lam_rows.append({'lam': lam, 'split_half_jaccard_per_fold': jac,
                         'mean_jaccard': float(np.nanmean(jac))})
    lam_star = float(max(lam_rows, key=lambda r: r['mean_jaccard'])['lam'])

    # ---- SR reality check (B9): the ranking field must vary inside 10 m blocks ----------------------
    sr_varying = {fold.held_out: G.assert_sr_score_is_not_replicated(fold.d, 4) for fold in folds}

    # ---- (5,6,7) run the gate and score it in F1's harness -----------------------------------------
    def run_track(track, tau, lam):
        gates, blocks_pix, blocks_win, blocks_pix_c, blocks_win_c, metas = [], [], [], [], [], {}
        for fold in folds:
            pf = tracks[track]['per_fold'][fold.held_out]
            em = em_per_fold[fold.held_out]
            fold.gate_v2_params = {'e_v': em['e_v'], 'e_b': em['e_b'], 'sigma_bands': pf['sigma_bands'],
                                   'tau': tau, 'lam': lam, 'window_size': WINDOW_10M,
                                   'k_unsupported': k_v1, 'sigma_sr': fold.sigma_v1}
            fold.refl_pre, fold.refl_post = pf['refl_pre'], pf['refl_post']   # this track's reflectance
            flagged = P.gate_v2(fold, k_v1)
            meta = fold.gate_v2_params.pop('last_meta')
            v = ~fold.nodata
            pr, pn, pd_ = P.compute_far_pixel_v2(flagged, v, tile_split, TILE_PX, **boot_args)
            wr, wn, wd = P.compute_far_window_v2(flagged, v, tile_split, TILE_PX, WINDOW_PX, **boot_args)
            prc, pnc, pdc = P.compute_far_pixel_v2(flagged, v, ~tile_split, TILE_PX, **boot_args)
            wrc, wnc, wdc = P.compute_far_window_v2(flagged, v, ~tile_split, TILE_PX, WINDOW_PX, **boot_args)
            blocks_pix.append((pn, pd_)); blocks_win.append((wn, wd))
            blocks_pix_c.append((pnc, pdc)); blocks_win_c.append((wnc, wdc))
            metas[fold.held_out] = meta
            gates.append({'held_out': fold.held_out, 'pre_dates': fold.pre_dates,
                          'pixel_far_test': pr, 'window_far_test': wr,
                          'pixel_far_calibration': prc, 'window_far_calibration': wrc,
                          'n_flagged_px': int(flagged[v].sum()), 'n_valid_px': int(v.sum()),
                          'n_windows_valid': meta['n_windows_valid'],
                          'n_windows_detected': meta['n_windows_detected'],
                          'class_counts': meta['class_counts'], 'block_sum': meta['block_sum'],
                          'sr_score_blocks_varying_fraction': sr_varying[fold.held_out]})
        return {
            'per_fold': gates,
            'pooled_pixel_far_test': P.pool_fold_blocks(blocks_pix, **boot_args),
            'pooled_window_far_test': P.pool_fold_blocks(blocks_win, **boot_args),
            'pooled_pixel_far_calibration': P.pool_fold_blocks(blocks_pix_c, **boot_args),
            'pooled_window_far_calibration': P.pool_fold_blocks(blocks_win_c, **boot_args),
            'block_sum_max_deviation': max(g['block_sum']['max_abs_deviation_all_blocks'] for g in gates),
            'block_sum_max_deviation_detected': max(g['block_sum']['max_abs_deviation_detected_blocks'] for g in gates),
        }, metas

    normalised_result, metas = run_track('normalised', head['tau'], lam_star)
    unnorm_result, _ = run_track('no_normalisation', tracks['no_normalisation']['tau'], lam_star)

    block_sum_max = max(normalised_result['block_sum_max_deviation'], unnorm_result['block_sum_max_deviation'])
    block_sum_pass = block_sum_max == 0

    # ---- tau mutation evidence on real data (B10): a different tau must move the map ----------------
    f0 = folds[0]
    pf0 = head['per_fold'][f0.held_out]
    em0 = em_per_fold[f0.held_out]
    mut = {}
    for name, t in (('tau_fitted', head['tau']), ('tau_times_0p5', head['tau'] * 0.5), ('tau_times_2', head['tau'] * 2.0)):
        cm, _ = G.apply_gate_v2(pf0['refl_pre'], pf0['refl_post'], f0.d, em0['e_v'], em0['e_b'],
                                pf0['sigma_bands'], f0.nodata, t, lam=lam_star, window_size=WINDOW_10M)
        mut[name] = {'tau': float(t), 'n_flagged_px': int(G.flagged_v2(cm).sum())}
    tau_mutation_changes_map = len({v['n_flagged_px'] for v in mut.values()}) > 1

    # ---- endmember +/- 1 robust SD sensitivity (fold 0, tau and lam held fixed) ---------------------
    def flagged_count(ev, eb):
        cm, _ = G.apply_gate_v2(pf0['refl_pre'], pf0['refl_post'], f0.d, ev, eb, pf0['sigma_bands'],
                                f0.nodata, head['tau'], lam=lam_star, window_size=WINDOW_10M)
        return int(G.flagged_v2(cm).sum())
    sensitivity = G.endmember_sensitivity(flagged_count, em0)
    sensitivity['note'] = ('flagged (CORE|ALLOCATED) 2.5 m pixel count on the 2024-01-16 placebo fold with tau '
                           'and lam held fixed, recomputed with each endmember shifted by +/- 1 robust SD '
                           '(1.4826*MAD) in every band')

    # ---- keep rules --------------------------------------------------------------------------------
    win_test = normalised_result['pooled_window_far_test']
    pix_test = normalised_result['pooled_pixel_far_test']
    keep_window_far = bool(win_test['estimate'] <= ALPHA)
    status = 'PASS' if (block_sum_pass and keep_window_far) else 'FAIL'

    def em_json(em):
        return {'e_v': [float(v) for v in em['e_v']], 'e_b': [float(v) for v in em['e_b']],
                'e_v_robust_sd': [float(v) for v in em['e_v_robust_sd']],
                'e_b_robust_sd': [float(v) for v in em['e_b_robust_sd']],
                'e_v_count': em['e_v_count'], 'e_b_count': em['e_b_count'],
                'n_valid_outside_exclusion': em['n_valid_outside_exclusion'],
                'n_pre_dates': em['n_pre_dates'], 'rules': em['rules']}

    result = {
        'status': status, 'evidence': 'real', 'experiment': 'f2_gate_v2',
        'config_sha256': fix_hash, 'wayanad_evidence_config_sha256': ev_hash,
        'aoi_center': fix_cfg['aoi_center'], 'crop_10m_rows_cols': list(CROP),
        'sr_pool_dates': SR_POOL, 'event_date': P.EVENT_DATE,
        'bands_unmixed': ['B04', 'B08'], 'band_order': ['B04', 'B03', 'B02', 'B08'],
        'grid': {'blocks_10m': list(head['per_fold'][SR_POOL[0]]['f_hat'].shape),
                 'px_2p5m': list(shape_2p5m), 'window_10m_px': WINDOW_10M,
                 'tile_px_2p5m': TILE_PX, 'window_px_2p5m': WINDOW_PX,
                 'n_windows_total': int(head['per_fold'][SR_POOL[0]]['win_s'].size),
                 'n_windows_calibration': int(calib_w.sum()), 'n_windows_test': int(test_w.sum())},
        'endmembers': {
            'headline_all_3_pre_dates': em_json(em_all),
            'per_fold_used_for_scoring': {k: em_json(v) for k, v in em_per_fold.items()},
            'exclusion_geometry': '1500 m disks around configs/wayanad_evidence.yaml plausibility_points '
                                  'union experiments/wayanad_evidence/outputs/footprint.tif, via '
                                  'trustsr.placebo_v2.exclusion_mask_10m (reused, not redefined)',
            'n_excluded_10m_px_crop': int(excl_10.sum()),
            'n_excluded_10m_px_full_aoi': int(aoi['exclude'].sum()),
            'estimated_on': 'the whole 1024x1024 10 m AOI (not only F1\'s 640x512 crop), pre-event dates only',
            'sensitivity_plus_minus_1_robust_sd': sensitivity,
            'never_from_labels': G.gate_v2_signature_audit(),
            'ndvi_low_used': ndvi_low,
            'ndvi_low_ceiling_table': em_sensitivity_by_ceiling,
        },
        'preregistration_deviations': [{
            'item': "configs/fix.yaml f2_gate_v2.endmembers.e_b: 'pre-event pixels with NDVI <= 0.2, valid on "
                    "all pre dates, outside disks+footprint'",
            'problem': 'On this AOI in January the rule selects almost nothing. Over the whole 1024x1024 AOI, '
                       'outside the disks + footprint and valid on all three pre dates, only 8 pixels have '
                       'NDVI <= 0.2 on every date (0 pixels inside F1\'s 640x512 crop). The Western Ghats AOI '
                       'is evergreen; the only large bare surface is the landslide scar itself, which the '
                       'exclusion geometry removes by construction -- and which must be removed, because using '
                       'it would make e_b a label (B12).',
            'resolution': f'The NDVI ceiling was escalated deterministically in 0.05 steps from the '
                          f'pre-registered 0.20 until at least {MIN_ENDMEMBER_PX} qualifying pixels existed. '
                          f'The first ceiling that qualified is {ndvi_low}. The full count/endmember table at '
                          'every ceiling is in endmembers.ndvi_low_ceiling_table.',
            'when': 'decided and executed BEFORE any tau, block-sum or FAR number was computed in this run; '
                    'no threshold was moved after seeing a result',
            'consequence_stated_plainly': f'e_b at NDVI <= {ndvi_low} is the darkest, least-vegetated land in '
                                          'the AOI, NOT true bare ground. It sits closer to e_v than a real '
                                          'soil/rock endmember would, which SHORTENS the mixing line and '
                                          'therefore INFLATES f for a given spectral change. The fraction f '
                                          'reported by this gate is an upper-leaning estimate of vegetation-loss '
                                          'fraction on this AOI and must not be read as a calibrated area.',
        }],
        'sigma_f': {'method': 'first-order error propagation through the unmixing; Var(f) = Var(a)*(1/n_pre+1), '
                              'Var(a) = (sR^2 dR^2 + sN^2 dN^2)/|d|^4',
                    'sigma_band_source': 'per-pixel temporal std (ddof=1) of the pre-date reflectance, floored at '
                                         'the scene median of that statistic',
                    'sigma_floor_per_band_normalised': {d: [float(x) for x in head['per_fold'][d]['sigma_floor']]
                                                        for d in SR_POOL},
                    'bootstrap_alternative': 'not used; see docs/adr-f2-gate-v2.md'},
        'tau': {
            'value': head['tau'], 'alpha': ALPHA, **head['tau_info'],
            'provenance': 'ceil((n+1)(1-alpha))-th order statistic of the per-160 m-window max of '
                          'gate_v2\'s OWN score s = f/sigma_f, over F1\'s three leave-one-date-out PLACEBO '
                          'folds, restricted to the CALIBRATION (even-tile) half of the checkerboard split. '
                          'Null-event data only: no post-event date is loadable in this code path (B1 guard).',
            'value_no_normalisation_track': tracks['no_normalisation']['tau'],
            'f1_export_consumed': {
                'path': str(CALIB_SCORES.relative_to(ROOT)), 'sha256': calib_sha,
                'sha256_matches_f1_json': calib_sha == CALIB_SCORES_SHA256_EXPECTED,
                'n_scores': int(f1_scores.size),
                'tau_if_taken_verbatim_from_this_file': float(G.split_conformal_quantile(f1_scores, ALPHA)[0]),
                'why_not_used_verbatim': 'F1 exported max(d/sigma_v1) per window -- an SR-NDVI z-score, whose '
                                         'units and scale (range '
                                         f'{float(f1_scores.min()):.2f} to {float(f1_scores.max()):.2f}) are not '
                                         'those of gate_v2\'s unmixing score f/sigma_f. F1\'s own docstring says '
                                         'so ("gate_v2\'s own score (unmixing f/sigma_f) does not exist yet"). '
                                         'Split conformal is only valid when calibration and test scores come '
                                         'from the SAME score function, so tau is recomputed with gate_v2\'s '
                                         'score on EXACTLY F1\'s folds, splits and window grid. The verbatim '
                                         'number is reported above for transparency; applying it would make the '
                                         'gate flag nothing and produce a vacuous FAR of 0.'},
        },
        'lambda': {'value': lam_star, 'grid': LAM_GRID, 'curve': lam_rows,
                   'fit_split': 'F1 calibration (even tiles) ONLY; the test split was never used to choose lam',
                   'objective': 'split-half dihedral reproducibility: allocate with the SR change field built '
                                'from dihedral runs 0-3 and again from runs 4-7, and maximise the Jaccard index '
                                'of the two allocated sets over calibration pixels',
                   'cap_rationale': 'lam <= 0.5 is a pre-committed cap so the within-block SR rank always carries '
                                    'at least as much weight as the neighbour-agreement term',
                   'neighbour_agreement_definition': 'fraction of a 2.5 m pixel\'s 8 neighbours (8-connectivity, '
                                                     'zero-padded at the array edge) allocated in the provisional '
                                                     'pass-1 (pure-SR) allocation; the pixel itself is excluded',
                   'far_is_lambda_invariant': 'lam changes WHICH sub-pixels inside a detected block are allocated, '
                                              'never HOW MANY (round(16*f)) nor which blocks are detected, so it '
                                              'cannot move pixel or window FAR; it is therefore not tunable against '
                                              'the keep rule even in principle'},
        'sr_ranking': {
            'source': 'data/experiments-cache/wayanad_evidence/per_date_ndvi/<date>_dihedral_means.npy -- the real '
                      '2.5 m SEN2SR-lite SR NDVI, 8 dihedral runs per date, produced by the E1 overlap tiler '
                      '(experiments/wayanad_evidence/tiler.py + step3_sr.py, 128 px native tiles, stride 96, '
                      '16 px crop margin) through the pinned model on the corrected AOI crop',
            'shape_2p5m': list(dih[SR_POOL[0]].shape),
            'score': 'per-pixel SR NDVI difference d = mean(pre dates, 8-run dihedral mean) - held-out date',
            'blocks_with_within_block_variation_fraction': sr_varying,
            'b9_guard': 'trustsr.gate_v2.assert_sr_score_is_not_replicated raises if the ranking field is constant '
                        'inside every 4x4 block, i.e. if np.repeat(np.repeat(d_10m,4),4) is passed (x9\'s bug)',
            'not_rerun_note': 'SR inference was NOT re-run in F2: the cached product is already the real tiled '
                              'dihedral SR stack for exactly these dates and exactly this crop '
                              '(rows 256:896, cols 128:640 = 640x512 at 10 m -> 2560x2048 at 2.5 m, 8 runs/date). '
                              'Re-running it on CPU would reproduce the same bytes at a cost of hours.'},
        'no_data': {'source': 'experiments.wayanad_evidence.data.scl_valid (v1 logic, reused verbatim) on EVERY '
                              'date, AND-ed, propagated nearest-neighbour to 2.5 m, union the exclusion geometry '
                              'and the pixels where the SR product is non-finite',
                    'n_nodata_px_2p5m_per_fold': {f.held_out: int(f.nodata.sum()) for f in folds},
                    'n_valid_px_2p5m_per_fold': {f.held_out: int((~f.nodata).sum()) for f in folds}},
        'block_sum_test': {'max_abs_deviation': int(block_sum_max),
                           'max_abs_deviation_detected_blocks': int(max(
                               normalised_result['block_sum_max_deviation_detected'],
                               unnorm_result['block_sum_max_deviation_detected'])),
                           'pass': bool(block_sum_pass),
                           'per_fold_normalised': {g['held_out']: g['block_sum'] for g in normalised_result['per_fold']},
                           'definition': 'observed = count of CORE|ALLOCATED 2.5 m px in each block\'s 4x4 footprint '
                                         'of the OUTPUT class map; expected = round(16*f) recomputed from the '
                                         'unmixing fraction. Independent paths (B7).'},
        'tau_mutation_test': {'changes_output_map': bool(tau_mutation_changes_map), 'runs': mut,
                              'unit_test': 'tests/test_gate_v2.py::TestTauMutation'},
        'scoring_in_f1_harness': {
            'harness': 'trustsr/placebo_v2.py (F1), GATES[\'gate_v2\'], unchanged FAR code paths',
            'flagged_definition': 'CORE union ALLOCATED (configs/fix.yaml units.flagged_definition.v2); '
                                  'UNSUPPORTED is reported separately and never counted as flagged',
            'normalised_f3': normalised_result,
            'no_normalisation_comparator': unnorm_result,
        },
        'keep_rule': {
            'block_sum_max_deviation_is_0': bool(block_sum_pass),
            'window_far_test_le_alpha': keep_window_far,
            'window_far_test': win_test, 'pixel_far_test': pix_test,
            'alpha': ALPHA,
            'verdict': (
                f"Block-sum test PASSES (max deviation {int(block_sum_max)}). The scoring keep rule does NOT "
                f"hold: the test-split window FAR point estimate is {win_test['estimate']:.4f}, above "
                f"alpha = {ALPHA}. Its 95 % CI is [{win_test['lo']:.4f}, {win_test['hi']:.4f}], which does "
                f"contain alpha, but the pre-registered rule is stated on the point estimate and is therefore "
                f"NOT met. F2's overall status is FAIL. Per configs/fix.yaml f2_gate_v2.block_sum_test."
                f"keep_rule, the condition that stops Wave 3 (F7) is a block-sum failure, and that did not "
                f"happen; the FAR shortfall is a keep-rule failure for gate_v2's claim, not a Wave-3 stop."
            ) if not keep_window_far else (
                f"Block-sum test PASSES (max deviation {int(block_sum_max)}) and the test-split window FAR "
                f"point estimate {win_test['estimate']:.4f} <= alpha = {ALPHA}. Both keep rules met."),
            'exchangeability_caveat': fix_cfg['units']['window_caveat'],
            'wave_3_status': 'not blocked by F2' if block_sum_pass else
                             'STOPPED: the F2 block-sum test failed, so per configs/fix.yaml '
                             'f2_gate_v2.block_sum_test.keep_rule Wave 3 (F7) does not run',
        },
        'b_ids_fixed': {
            'B6': 'tau comes from PLACEBO (null-event) calibration-split window scores only; x4 calibrated on '
                  'pre_mean - post_mean of the real event pair with valid = ones',
            'B7': 'block_sum_deviation compares the OUTPUT class map\'s per-block CORE|ALLOCATED count with '
                  'round(16*f) recomputed from the fraction -- two independent paths, never a quantity with itself',
            'B8': f'config_sha256 = sha256(configs/fix.yaml) = {fix_hash}; no "pending-a3-output" anywhere',
            'B9': 'allocation is ranked by the real 2.5 m SR NDVI difference from the cached E1/E8 dihedral SR '
                  'stack, guarded by assert_sr_score_is_not_replicated; x9 ranked by np.repeat(np.repeat(d_10m,4),4)',
            'B10': 'tau is a required positional argument of apply_gate_v2 and raises if non-finite; the mutation '
                   'test above and tests/test_gate_v2.py::TestTauMutation show the map changes with tau',
            'B11': 'NO_DATA from the SCL validity mask on every date, nearest-neighbour to 2.5 m; x9 used NaN only',
            'B12': 'endmembers estimated by estimate_endmembers from stable/high-NDVI and NDVI<=0.2 pre-event '
                   'pixels outside disks+footprint; x9 hard-coded e_v/e_b constants',
        },
        'limitations': [
            'Three pool dates only (2024-01-16/21/26): the cached real 2.5 m dihedral SR product exists for no '
            'other pre-event date, so every fold\'s reference is the mean of just 2 dates and the three folds '
            'share one crop. Serial and spatial correlation across folds is not modelled, as in F1.',
            'tau is recomputed with gate_v2\'s own score rather than read verbatim from '
            'f1_calibration_scores.npz, because F1 exported a different score function (d/sigma_v1). The folds, '
            'the checkerboard split, the window grid and the alpha are exactly F1\'s. The verbatim number is '
            'reported in tau.f1_export_consumed.',
            'sigma_f propagates measurement noise only. Endmember uncertainty is reported separately as the '
            '+/- 1 robust SD sensitivity, not folded into the score.',
            'sigma_band carries a floor at the scene median of the per-pixel temporal std. Without it a handful '
            'of pixels with near-identical pre dates dominate the score (F1\'s exported d/sigma_v1 reaches 255 '
            'for this reason). The floor is fixed at the median and was not varied.',
            'SR inference was not re-run; the cached real tiled dihedral stack for these dates and this crop was '
            'used as-is (see sr_ranking.not_rerun_note).',
            f'The pre-registered e_b rule (NDVI <= 0.2) is not satisfiable on this evergreen AOI in January '
            f'(8 qualifying px over the whole AOI). The ceiling was escalated to {ndvi_low} before any result '
            f'was computed; see preregistration_deviations. e_b is therefore the darkest land cover present, '
            f'not true bare ground, and f is correspondingly upper-leaning.',
            'The gate is HIGHLY sensitive to the endmembers: shifting e_b by +1 robust SD multiplies the '
            'flagged pixel count by about '
            f"{1 + (sensitivity['perturbations']['e_b_plus_1sd']['relative_delta'] or 0):.1f} on the fold tested. "
            'The robust SD of e_b is large relative to |e_v - e_b| because only a few hundred dark-land pixels '
            'exist to estimate it from. The fraction f is not a calibrated area and must not be reported as one.',
            f'lam was fitted to {lam_star} on the calibration split, i.e. the neighbour-agreement term carries '
            'zero weight in the headline run. That is a fitted value, not a disabled parameter: split-half '
            'dihedral reproducibility was HIGHEST at lam = 0 and fell monotonically across the grid (see '
            'lambda.curve), and tests/test_gate_v2.py::test_lambda_changes_which_sub_pixels_are_chosen_but_never_'
            'how_many shows the term does change the map when lam > 0.',
            'UNSUPPORTED is very large on these folds (millions of 2.5 m px) because it is defined against the '
            'v1 dihedral sigma, which is tiny, so almost any SR NDVI difference exceeds k*sigma. UNSUPPORTED is '
            'reported separately and is NEVER counted as flagged (configs/fix.yaml units.flagged_definition.v2), '
            'so it does not enter any FAR above; it is a known v1-sigma artefact, not a gate-v2 result.',
            f"F3's per-date normalisation changes the test-split window FAR from "
            f"{unnorm_result['pooled_window_far_test']['estimate']:.4f} (without) to "
            f"{win_test['estimate']:.4f} (with). Both columns are reported; the headline follows "
            'configs/fix.yaml f3_noise_v2.interaction_test, which instructs F2 to apply the normalisation.',
            fix_cfg['units']['window_caveat'],
        ],
    }
    result.update(started_utc=started, finished_utc=datetime.now(timezone.utc).isoformat())
    write_json(ROOT / 'experiments/results/f2.json', result)
    write_report(result, ROOT / 'experiments/results/f2_REPORT.md')
    print(json.dumps({'status': status, 'block_sum_max_deviation': int(block_sum_max),
                      'tau': head['tau'], 'lam': lam_star,
                      'window_far_test': win_test['estimate'],
                      'window_far_ci': [win_test['lo'], win_test['hi']],
                      'pixel_far_test': pix_test['estimate']}, indent=2))
    return result


def write_report(r, path: Path):
    n = r['scoring_in_f1_harness']['normalised_f3']
    w, p = n['pooled_window_far_test'], n['pooled_pixel_far_test']
    u = r['scoring_in_f1_harness']['no_normalisation_comparator']
    em = r['endmembers']['headline_all_3_pre_dates']
    fmt = lambda x: f"{x['estimate']:.4f} [{x['lo']:.4f}, {x['hi']:.4f}] (num {x['numerator']:.0f} / den {x['denominator']:.0f}, {x['blocks']} blocks)"
    L = [
        '# F2 — gate v2 rebuilt (replaces the invalidated x4/x9 gate-v2 path)', '',
        f"**Status: {r['status']}** · evidence: real · config_sha256 `{r['config_sha256']}` (configs/fix.yaml)", '',
        'Pre-registered in `configs/fix.yaml` `f2_gate_v2`. Design: `docs/adr-f2-gate-v2.md`. '
        'Implementation: `trustsr/gate_v2.py`. Harness: F1\'s `trustsr/placebo_v2.py`, unchanged FAR code.', '',
        '## What each part fixes', '',
        '| B-ID | The old bug (x4 / x9) | What F2 does instead |', '|---|---|---|',
        '| B6 | conformal scores came from `pre_mean - post_mean` on the REAL event pair with `valid = ones` | τ is the order statistic of PLACEBO (null-event) calibration-split window scores; no post-event date is loadable in this path |',
        '| B7 | `check_block_sum_consistency` compared `h*w*100 m²` with itself | `block_sum_deviation` counts CORE\\|ALLOCATED px in the output class map and compares with `round(16f)` recomputed from the fraction — two independent paths |',
        '| B8 | `config_sha256 = "pending-a3-output"` | `config_sha256` is the sha256 of `configs/fix.yaml`, hashed at run time |',
        '| B9 | allocation ranked by `np.repeat(np.repeat(d_10m,4),4)` | ranked by the real 2.5 m SR NDVI difference from the cached E1-tiled / E8-dihedral SEN2SR-lite stack; `assert_sr_score_is_not_replicated` raises on a blocky field |',
        '| B10 | τ loaded from x4.json and never passed in (`kappa=None`) | τ is a **required positional argument** of `apply_gate_v2`; non-finite τ raises; mutation test proves the map moves |',
        '| B11 | NO_DATA from NaN pixels only (512 px vs v1\'s 205,956) | NO_DATA from the SCL validity mask on **every** date, nearest-neighbour to 2.5 m |',
        f"| B12 | `e_v`/`e_b` hard-coded constants | estimated from stable/high-NDVI and low-NDVI "
        f"(≤ {r['endmembers']['ndvi_low_used']}, see the deviation note below) pre-event pixels outside "
        f"the disks + footprint |",
        '', '## Endmembers (B12)', '',
        f"- `e_v` = {[round(v, 5) for v in em['e_v']]} from **{em['e_v_count']:,}** pixels "
        f"(robust SD {[round(v, 5) for v in em['e_v_robust_sd']]})",
        f"- `e_b` = {[round(v, 5) for v in em['e_b']]} from **{em['e_b_count']:,}** pixels "
        f"(robust SD {[round(v, 5) for v in em['e_b_robust_sd']]})",
        f"- rule e_v: {em['rules']['e_v']}", f"- rule e_b: {em['rules']['e_b']}",
        f"- excluded from the sample: {r['endmembers']['n_excluded_10m_px_full_aoi']:,} 10 m px of the full AOI "
        f"({r['endmembers']['exclusion_geometry']}); estimated on {r['endmembers']['estimated_on']}",
        f"- ±1 robust-SD sensitivity (flagged px count, τ and λ fixed): baseline "
        f"{r['endmembers']['sensitivity_plus_minus_1_robust_sd']['baseline']:.0f}, max |Δ| "
        f"{r['endmembers']['sensitivity_plus_minus_1_robust_sd']['max_abs_delta']:.0f} "
        f"({100 * (r['endmembers']['sensitivity_plus_minus_1_robust_sd']['max_relative_delta'] or 0):.1f} %)",
        '', '### Pre-registration deviation on `e_b` — stated up front', '',
    ] + sum([[f"- **item**: {d['item']}", f"  - **problem**: {d['problem']}",
              f"  - **resolution**: {d['resolution']}", f"  - **when**: {d['when']}",
              f"  - **consequence**: {d['consequence_stated_plainly']}"]
             for d in r['preregistration_deviations']], []) + [
        '',
        '| NDVI ceiling | qualifying px | meets min count (100) |', '|---|---|---|',
    ] + [f"| {t['ndvi_low']:.2f}{' **(pre-registered)**' if t['ndvi_low'] == 0.20 else ''}"
         f"{' **(used)**' if t['ndvi_low'] == r['endmembers']['ndvi_low_used'] else ''} "
         f"| {t['e_b_count']:,} | {t['meets_min_count']} |" for t in r['endmembers']['ndvi_low_ceiling_table']] + [
        '', '## τ (B6, B10)', '',
        f"- **τ = {r['tau']['value']:.6f}** = T_({r['tau']['k']}) of n = {r['tau']['n_calibration']} calibration "
        f"placebo window scores, k = ⌈(n+1)(1−α)⌉, α = {r['tau']['alpha']}",
        f"- provenance: {r['tau']['provenance']}",
        f"- F1's exported file `{r['tau']['f1_export_consumed']['path']}` sha256 verified: "
        f"`{r['tau']['f1_export_consumed']['sha256']}` (matches f1.json: "
        f"{r['tau']['f1_export_consumed']['sha256_matches_f1_json']}). "
        f"{r['tau']['f1_export_consumed']['why_not_used_verbatim']}",
        '', '## λ and allocation (B9)', '',
        f"- λ = {r['lambda']['value']} (grid {r['lambda']['grid']}), fit: {r['lambda']['fit_split']}",
        f"- objective: {r['lambda']['objective']}",
        f"- neighbour agreement: {r['lambda']['neighbour_agreement_definition']}",
        f"- {r['lambda']['far_is_lambda_invariant']}",
        f"- SR ranking field varies within 4×4 blocks for "
        f"{', '.join(f'{k}: {100*v:.1f} %' for k, v in r['sr_ranking']['blocks_with_within_block_variation_fraction'].items())} "
        'of blocks — a `np.repeat` field would be 0 % and would raise.',
        '', '## Block-sum test (B7)', '',
        f"- **max |deviation| = {r['block_sum_test']['max_abs_deviation']}** over "
        f"{sum(v['n_blocks'] for v in r['block_sum_test']['per_fold_normalised'].values()):,} block-evaluations "
        f"({r['block_sum_test']['max_abs_deviation_detected_blocks']} over detected blocks) → "
        f"{'PASS' if r['block_sum_test']['pass'] else 'FAIL'}",
        f"- {r['block_sum_test']['definition']}",
        '', '## gate_v2 in F1\'s harness, TEST split (odd tiles)', '',
        '| metric | estimate [95 % CI] (numerator / denominator) |', '|---|---|',
        f"| window FAR (160 m windows) | {fmt(w)} |",
        f"| pixel FAR (2.5 m px) | {fmt(p)} |",
        f"| window FAR, no F3 normalisation | {fmt(u['pooled_window_far_test'])} |",
        f"| pixel FAR, no F3 normalisation | {fmt(u['pooled_pixel_far_test'])} |",
        f"| window FAR, calibration split (in-sample, for reference) | {fmt(n['pooled_window_far_calibration'])} |",
        '',
        f"UNSUPPORTED per fold (reported separately, **never** counted as flagged): "
        + ', '.join(f"{g['held_out']}: {g['class_counts']['UNSUPPORTED']:,} px" for g in n['per_fold']) + '.', '',
        f"Keep rule (window FAR ≤ α = {r['keep_rule']['alpha']}): "
        f"**{'MET' if r['keep_rule']['window_far_test_le_alpha'] else 'NOT MET'}**.", '',
        f"**Verdict.** {r['keep_rule']['verdict']}", '',
        f"> Caveat (configs/fix.yaml `units.window_caveat`): {r['keep_rule']['exchangeability_caveat']}", '',
        f"Wave 3: {r['keep_rule']['wave_3_status']}", '',
        '## Limitations', '',
    ] + [f'- {x}' for x in r['limitations']]
    path.write_text('\n'.join(L) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
