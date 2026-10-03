"""F13 A0: is the gate's test-minus-calibration window-FAR gap real? (helpers; the run is `main` at the bottom)

Pre-registered in configs/f13_discovery.yaml `experiments.A0`. Thresholds, seeds and replicates are read from
that file only. F2's own FAIL is not revisited by any outcome here.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats

from experiments import f13_common as C


# ---------------------------------------------------------------- splits ----------------------------------------------------------------

def random_tile_split(shape_tiles, n_test: int, seed: int) -> np.ndarray:
    """(H_tiles, W_tiles) bool, True = test, exactly `n_test` True, whole 128 px tiles reassigned at random."""
    rng = np.random.default_rng(seed)
    flat = np.zeros(int(np.prod(shape_tiles)), dtype=bool)
    flat[rng.choice(flat.size, size=n_test, replace=False)] = True
    return flat.reshape(shape_tiles)


def random_window_split(shape_windows, seed: int) -> np.ndarray:
    """Window-level random split, equal halves (yaml A0.method.proposed_not_in_memo.random_split), True = test."""
    n = int(np.prod(shape_windows))
    return random_tile_split(shape_windows, n // 2, seed)


# ---------------------------------------------------------------- conformal tau under weights ----------------------------------------------------------------

def weighted_order_statistic(sorted_scores, weights, k: int) -> float:
    """k-th (1-based) smallest of the multiset in which sorted_scores[i] appears weights[i] times."""
    cum = np.cumsum(np.asarray(weights, dtype=np.int64))
    if k < 1 or k > int(cum[-1]):
        raise ValueError(f'k = {k} outside the multiset of size {int(cum[-1])}')
    return float(np.asarray(sorted_scores)[int(np.searchsorted(cum, k, side='left'))])


def gap_from_weights(rec: dict, is_test, weights, alpha: float) -> dict:
    """D = window FAR(test half) - window FAR(calibration half), each at the conformal tau of the calibration half.

    rec: per (fold, window) arrays: score (gate-v2 window max of s = f/sigma_f), in_pop (window enters tau's
    population), flaggable (>= 1 valid block allocates >= 1 sub-pixel, i.e. round(16 f) >= 1), valid_px
    (denominator of F2's FAR), tile_id. `weights` are bootstrap multiplicities (all ones = the point estimate).
    A window is flagged iff in_pop & score > tau & flaggable -- identical to apply_gate_v2's detection followed by
    exact-count allocation; the equivalence is asserted against the real gate at run time.
    """
    score = np.asarray(rec['score'], dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    is_test = np.asarray(is_test, dtype=bool)
    pop = ~is_test & np.asarray(rec['in_pop'], bool) & np.isfinite(score) & (w > 0)
    order = np.argsort(score[pop], kind='stable')
    s_sorted, w_sorted = score[pop][order], np.rint(w[pop][order]).astype(np.int64)
    n = int(w_sorted.sum())
    k = int(np.ceil((n + 1) * (1.0 - alpha)))                   # same expression as trustsr.gate_v2.split_conformal_quantile
    if k > n:
        raise ValueError(f'calibration population {n} too small for alpha = {alpha}')
    tau = weighted_order_statistic(s_sorted, w_sorted, k)
    with np.errstate(invalid='ignore'):
        flag = np.asarray(rec['in_pop'], bool) & (score > tau) & np.asarray(rec['flaggable'], bool)
    valid = np.asarray(rec['valid_px'], bool)
    out = {'tau': tau, 'k': k, 'n_cal_pop': n}
    for half, sel in (('cal', ~is_test), ('test', is_test)):
        num, den = float((w * flag)[sel].sum()), float((w * valid)[sel].sum())
        if den <= 0:
            raise ValueError(f'no valid windows in the {half} half')
        out[f'num_{half}'], out[f'den_{half}'], out[f'far_{half}'] = num, den, num / den
    out['D'] = out['far_test'] - out['far_cal']
    return out


def bootstrap_gap(rec: dict, is_test, alpha: float, replicates: int, seed: int, ci: float) -> dict:
    """Block bootstrap over 128 px tiles with tau recomputed inside every replicate (yaml A0.method)."""
    tile = np.asarray(rec['tile_id'])
    uniq, tidx = np.unique(tile, return_inverse=True)
    n_t = uniq.size
    point = gap_from_weights(rec, is_test, np.ones(tile.size), alpha)
    rng = np.random.default_rng(seed)
    ds, skipped = [], 0
    for _ in range(replicates):
        counts = np.bincount(rng.integers(0, n_t, size=n_t), minlength=n_t)
        try:
            ds.append(gap_from_weights(rec, is_test, counts[tidx].astype(np.float64), alpha)['D'])
        except ValueError:
            skipped += 1
    lo, hi = np.quantile(ds, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return {**point, 'lo': float(lo), 'hi': float(hi), 'replicates': replicates, 'seed': seed, 'ci': ci,
            'n_tiles': int(n_t), 'replicates_skipped': skipped}


def tile_permutation_reference(rec: dict, obs_is_test, n_tiles: int, n_test_tiles: int, alpha: float,
                               n_perm: int, seed: int) -> dict:
    """Where does the observed checkerboard D fall among D under random whole-tile splits (F2's tile counts)?

    tau is recomputed on each random calibration half. One-sided p = (1 + #{D_perm >= D_obs}) / (1 + n_perm).
    """
    tile = np.asarray(rec['tile_id'])
    obs = gap_from_weights(rec, obs_is_test, np.ones(tile.size), alpha)
    rng = np.random.default_rng(seed)
    ds = np.empty(n_perm)
    for i in range(n_perm):
        test_tiles = np.zeros(n_tiles, bool)
        test_tiles[rng.choice(n_tiles, size=n_test_tiles, replace=False)] = True
        ds[i] = gap_from_weights(rec, test_tiles[tile], np.ones(tile.size), alpha)['D']
    ge = int((ds >= obs['D']).sum())
    return {'D_obs': obs['D'], 'n_perm': n_perm, 'seed': seed, 'n_ge_obs': ge,
            'p_one_sided': (1 + ge) / (1 + n_perm),
            'perm_mean': float(ds.mean()), 'perm_sd': float(ds.std(ddof=1)),
            'perm_q025': float(np.quantile(ds, 0.025)), 'perm_q975': float(np.quantile(ds, 0.975)),
            'perm_values': ds}


# ---------------------------------------------------------------- per-window records ----------------------------------------------------------------

def window_arrays(s_block, valid_block, f_hat, nodata_2p5m, window_10m: int = 16) -> dict:
    """Per-160 m-window quantities of one fold, built from the arrays F2's own code path produces.

    score     window max of s = f / sigma_f over valid blocks (trustsr.gate_v2.window_max_score, reused)
    in_pop    the window has a finite score: it enters tau's calibration population and can be detected
    flaggable some valid block has round(16 * clip(f, 0, 1)) >= 1, i.e. detection would allocate >= 1 sub-pixel
              (apply_gate_v2 sets f_effective = f_hat with NaN -> 0 on every valid block of a detected window)
    valid_px  the window has >= 1 valid 2.5 m px (trustsr.placebo_v2.window_indicators' denominator)
    """
    from trustsr import gate_v2 as G
    from trustsr.bootstrap import block_sums
    score, win_v = G.window_max_score(s_block, window_10m, valid_block)
    in_pop = win_v & np.isfinite(score)
    n_blk = np.clip(np.round(16 * np.clip(np.nan_to_num(np.asarray(f_hat, float), nan=0.0), 0.0, 1.0)), 0, 16)
    alloc_block = np.asarray(valid_block, bool) & (n_blk > 0)
    h, w = alloc_block.shape
    flaggable = alloc_block.reshape(h // window_10m, window_10m, w // window_10m, window_10m).any(axis=(1, 3))
    valid_px = block_sums((~np.asarray(nodata_2p5m, bool)).astype(np.float64), window_10m * 4) > 0
    return {'score': score, 'in_pop': in_pop, 'flaggable': flaggable, 'valid_px': valid_px}


# ---------------------------------------------------------------- the pre-registered decision rule ----------------------------------------------------------------

def _excludes_zero(ci) -> bool:
    lo, hi = ci
    return lo > 0 or hi < 0


def a0_verdict(checkerboard_ci, random_ci) -> str:
    """configs/f13_discovery.yaml experiments.A0.keep_rule, verbatim. A CI endpoint at exactly 0 includes 0."""
    if _excludes_zero(random_ci):
        return 'MISCALIBRATED_SCORE'
    return 'TRUE' if _excludes_zero(checkerboard_ci) else 'FALSE'


# ---------------------------------------------------------------- analytic iid reference ----------------------------------------------------------------

def betabinom_parameters(n_cal: int, alpha: float) -> dict:
    """For a perfectly calibrated exchangeable score, the exceedance probability U of the split-conformal tau
    (the k-th of n calibration scores) is Beta(n + 1 - k, k)."""
    k = int(math.ceil((n_cal + 1) * (1.0 - alpha)))
    return {'k': k, 'a': n_cal + 1 - k, 'b': k}


def betabinom_tail(n_cal: int, n_test: int, alpha: float, x_min: int) -> float:
    """P(number of test exceedances >= x_min) for n_test iid windows under a perfectly calibrated gate."""
    p = betabinom_parameters(n_cal, alpha)
    return float(stats.betabinom.sf(x_min - 1, n_test, p['a'], p['b']))


# ---------------------------------------------------------------- power: injected NDVI drops ----------------------------------------------------------------

def inject_ndvi_drop(refl, mask, drop: float) -> np.ndarray:
    """Lower NDVI by exactly `drop` inside `mask`, preserving B04 + B08 (yaml power_requirement injection_rule).

    refl is (H, W, 4) in model band order [B04, B03, B02, B08]. B08 -= drop*S/2, B04 += drop*S/2.
    """
    out = np.array(refl, dtype=np.float64, copy=True)
    m = np.asarray(mask, bool)
    s = out[..., 0] + out[..., 3]
    red = out[..., 0] + drop * s / 2.0
    nir = out[..., 3] - drop * s / 2.0
    if (nir[m] < 0).any() or (red[m] < 0).any():
        raise ValueError('injection would create negative reflectance')
    out[..., 0] = np.where(m, red, out[..., 0])
    out[..., 3] = np.where(m, nir, out[..., 3])
    return out


def patch_hit(win_score, in_pop, n_blk, origin, tau: float, patch_px: int, window_px: int) -> dict:
    """Was an injected patch flagged by the gate at `tau`?  `n_blk` is round(16 f) on every valid 10 m block
    (0 on invalid blocks) computed from the INJECTED scene; `origin` is the patch's (row, col) on the 10 m grid.

    detected: the patch's 160 m window has score > tau (and is in the calibration population);
    flagged:  detected AND >= 1 patch block allocates >= 1 sub-pixel, so >= 1 flagged px lies inside the patch;
    flagged_px: the exact number of flagged 2.5 m px in the patch (allocation is exactly round(16 f) per block,
    independent of the ranking field, so the count does not need the SR product).
    """
    r, c = origin
    wi, wj = r // window_px, c // window_px
    detected = bool(in_pop[wi, wj] and win_score[wi, wj] > tau)
    n_in = int(np.asarray(n_blk)[r:r + patch_px, c:c + patch_px].sum()) if detected else 0
    return {'detected': detected, 'flagged': bool(detected and n_in > 0), 'flagged_px': n_in}


def place_patches(valid, eligible_windows, n: int, patch_px: int, window_px: int, seed: int) -> list:
    """n patch origins (row, col), each wholly inside a distinct eligible window and wholly on valid px."""
    from numpy.lib.stride_tricks import sliding_window_view
    valid = np.asarray(valid, bool)
    ok = sliding_window_view(valid, (patch_px, patch_px)).all(axis=(-1, -2))          # origin (r, c) is patch-valid
    span = window_px - patch_px + 1
    cands, offs = [], {}
    for wi, wj in zip(*np.nonzero(eligible_windows)):
        r0, c0 = wi * window_px, wj * window_px
        sub = ok[r0:r0 + span, c0:c0 + span]
        if sub.size and sub.any():
            cands.append((int(wi), int(wj)))
            offs[(int(wi), int(wj))] = [(r0 + int(r), c0 + int(c)) for r, c in zip(*np.nonzero(sub))]
    if len(cands) < n:
        raise ValueError(f'only {len(cands)} eligible windows can hold a patch; {n} requested')
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(cands), size=n, replace=False)
    return [offs[cands[i]][int(rng.integers(0, len(offs[cands[i]])))] for i in chosen]


# ================================================================ the run ================================================================

def _required_inputs(cfg_ev, ev_root):
    from experiments.f2_gate_v2 import SR_POOL
    cache = ev_root / cfg_ev['paths']['cache']
    paths = [ev_root / 'experiments/wayanad_evidence/outputs/footprint.tif', ev_root / 'experiments/results/f2.json']
    for d in SR_POOL:
        paths += [cache / f'{d}.npz', cache / 'per_date_ndvi' / f'{d}_dihedral_means.npy']
    return cache, paths


def regenerate_f2(cfg_ev, ev_root):
    """Rebuild F2's gate-v2 inputs through F2's own functions, in F2's own order (experiments/f2_gate_v2.py main)."""
    from experiments import f2_gate_v2 as F2
    cache, paths = _required_inputs(cfg_ev, ev_root)
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise C.Blocked('missing cached inputs: ' + '; '.join(missing) + ' | checked: ' + '; '.join(str(p) for p in paths))
    folds, dih, refl, valid, ndvi, excl_10, excl_hr, aoi = F2.build(cfg_ev, cache)
    del dih
    pool = F2.SR_POOL
    stack = lambda key, dates: np.stack([aoi[key][d] for d in dates])
    ndvi_low, _em_all, _table = F2.endmember_ceiling_table(stack('refl', pool), stack('valid', pool),
                                                           stack('ndvi', pool), aoi['exclude'])
    em = {f.held_out: F2.estimate_endmembers_at(stack('refl', f.pre_dates), stack('valid', f.pre_dates),
                                                stack('ndvi', f.pre_dates), aoi['exclude'], ndvi_low)
          for f in folds}
    nd = np.stack([ndvi[d] for d in pool])
    stable = (np.stack([valid[d] for d in pool]).all(axis=0) & ~excl_10
              & (nd.std(axis=0, ddof=1) <= 0.05) & np.isfinite(nd).all(axis=0))
    shape = folds[0].nodata.shape
    calib_w, test_w, tile_split = F2.window_split_masks(shape, cfg_ev.get('seed', 2024))
    return {'F2': F2, 'folds': folds, 'em': em, 'stable': stable, 'excl_10': excl_10, 'shape': shape,
            'calib_w': calib_w, 'test_w': test_w, 'tile_split': tile_split, 'ndvi_low': ndvi_low}


def score_track(R, normalise: bool):
    """Per-fold gate-v2 scores and window records for the normalised (F2 headline) or raw arm, plus tau."""
    from trustsr import gate_v2 as G
    F2 = R['F2']
    per_fold = {}
    for fold in R['folds']:
        em = R['em'][fold.held_out]
        if normalise:
            rp, rq, _off = F2.normalise_fold(fold, R['stable'])
        else:
            rp, rq = fold.refl_pre, fold.refl_post
        s, vb, sigma_bands, _floor, f_hat = F2.fold_score(fold, em['e_v'], em['e_b'], rp, rq)
        per_fold[fold.held_out] = {'rp': rp, 'rq': rq, 's': s, 'vb': vb, 'sigma_bands': sigma_bands, 'f_hat': f_hat,
                                   'wa': window_arrays(s, vb, f_hat, fold.nodata)}
    cal = np.concatenate([pf['wa']['score'][R['calib_w'] & pf['wa']['in_pop']] for pf in per_fold.values()])
    tau, info = G.split_conformal_quantile(cal, R['alpha'])
    return {'per_fold': per_fold, 'tau': tau, 'tau_info': info}


def real_gate_window_flags(R, track, fold, tau, k_unsupported):
    """The REAL gate (trustsr.placebo_v2.gate_v2 -> apply_gate_v2) run exactly as F2's run_track does, reduced to
    per-160 m-window flagged / valid indicators with F2's own window_indicators."""
    import dataclasses
    from trustsr import placebo_v2 as P
    pf = track['per_fold'][fold.held_out]
    em = R['em'][fold.held_out]
    f = dataclasses.replace(fold, refl_pre=pf['rp'], refl_post=pf['rq'], gate_v2_params={
        'e_v': em['e_v'], 'e_b': em['e_b'], 'sigma_bands': pf['sigma_bands'], 'tau': tau, 'lam': R['lam'],
        'window_size': 16, 'k_unsupported': k_unsupported, 'sigma_sr': fold.sigma_v1})
    flagged = P.gate_v2(f, k_unsupported)
    return P.window_indicators(flagged, ~fold.nodata, 64)


def assemble_records(track, n_windows=(40, 32)) -> dict:
    wa = [pf['wa'] for pf in track['per_fold'].values()]
    rec = {k: np.concatenate([w[k].ravel() for w in wa]) for k in ('score', 'in_pop', 'flaggable', 'valid_px')}
    wr, wc = np.divmod(np.arange(n_windows[0] * n_windows[1]), n_windows[1])
    rec['tile_id'] = np.tile((wr // 2) * (n_windows[1] // 2) + (wc // 2), len(wa))
    rec['pos_id'] = np.tile(np.arange(n_windows[0] * n_windows[1]), len(wa))
    return rec


def _ci_row(name, g, boot, unit='window FAR difference (test - calibration)'):
    return C.measurement(name, g['D'], numerator=None, denominator=None, lo=g['lo'], hi=g['hi'],
                         replicates=boot['replicates'], seed=boot['seed'], unit=unit,
                         far_test=g['far_test'], far_cal=g['far_cal'],
                         num_test=g['num_test'], den_test=g['den_test'], num_cal=g['num_cal'], den_cal=g['den_cal'],
                         tau=g['tau'], k=g['k'], n_cal_pop=g['n_cal_pop'], n_tiles=g['n_tiles'],
                         replicates_skipped=g['replicates_skipped'])


def run_power(R, track, split_name, split_index, test_w, tau, cfg):
    """Recall of injected NDVI drops at this split's tau (yaml power_requirement). Patches live only in windows of the
    split's TEST half that are valid and in tau's population; all three folds, one pooled estimate with a tile block
    bootstrap, plus the contamination baseline (patches whose window the placebo already flagged)."""
    from trustsr.bootstrap import ratio_bootstrap_ci
    import dataclasses
    F2 = R['F2']
    pw = cfg['power_requirement']
    prop = pw['proposed_not_in_memo']
    n_patches, patch_px, seed = prop['n_patches_per_fold_per_drop'], 10, cfg['seed']
    boot = cfg['statistics']['bootstrap']
    out = {}
    for drop_i, drop in enumerate(pw['injected_ndvi_drops']):
        nums, dens, per_fold, hit_rows = [], [], {}, []
        for fold_i, fold in enumerate(R['folds']):
            pf = track['per_fold'][fold.held_out]
            base = pf['wa']
            eligible = test_w & base['in_pop'] & base['valid_px']
            ss = int(np.random.SeedSequence([seed, fold_i, drop_i, split_index]).generate_state(1)[0])
            patches = place_patches(pf['vb'], eligible, n_patches, patch_px, 16, ss)
            mask10 = np.zeros(pf['vb'].shape, bool)
            for r, c in patches:
                mask10[r:r + patch_px, c:c + patch_px] = True
            inj = dataclasses.replace(fold, refl_post=inject_ndvi_drop(fold.refl_post, mask10, drop))
            rp, rq, _ = F2.normalise_fold(inj, R['stable'])
            em = R['em'][fold.held_out]
            s, vb, _sb, _fl, f_hat = F2.fold_score(inj, em['e_v'], em['e_b'], rp, rq)
            wa = window_arrays(s, vb, f_hat, fold.nodata)
            n_blk = np.where(vb, np.clip(np.round(16 * np.clip(np.nan_to_num(f_hat, nan=0.0), 0, 1)), 0, 16), 0).astype(int)
            base_flag = base['in_pop'] & (base['score'] > tau) & base['flaggable']
            hits = [patch_hit(wa['score'], wa['in_pop'], n_blk, p, tau, patch_px, 16) for p in patches]
            tiles = np.array([((r // 16) // 2) * 16 + ((c // 16) // 2) for r, c in patches])
            flagged = np.array([h['flagged'] for h in hits], float)
            num = np.bincount(tiles, weights=flagged, minlength=320)
            den = np.bincount(tiles, minlength=320).astype(float)
            nums.append(num); dens.append(den)
            contaminated = np.array([bool(base_flag[r // 16, c // 16]) for r, c in patches])
            per_fold[fold.held_out] = {'patches': len(patches), 'flagged': int(flagged.sum()),
                                       'already_flagged_by_placebo': int(contaminated.sum())}
            hit_rows.append((flagged, contaminated, np.array([h['flagged_px'] for h in hits], float)))
        flagged = np.concatenate([h[0] for h in hit_rows])
        contam = np.concatenate([h[1] for h in hit_rows])
        px = np.concatenate([h[2] for h in hit_rows])
        ci = ratio_bootstrap_ci(np.concatenate(nums), np.concatenate(dens), boot['replicates'], cfg['statistics']['ci'],
                                boot['seed'])
        clean = ~contam
        out[str(drop)] = {
            'drop': drop, 'split': split_name, 'tau': tau, 'recall': ci,
            'patches': int(flagged.size), 'flagged': int(flagged.sum()),
            'patches_in_windows_already_flagged_by_placebo': int(contam.sum()),
            'recall_excluding_those': (float(flagged[clean].sum() / clean.sum()) if clean.sum() else None),
            'denominator_excluding_those': int(clean.sum()),
            'pixel_recall_mean': float(px.mean() / (patch_px * patch_px * 16)),
            'per_fold': per_fold}
    return out


def main():
    import json
    import re
    import yaml
    from experiments.wayanad_evidence import config as CE
    started = C.utc_now()
    result = {'evidence': 'real', 'experiment_id': 'A0'}
    try:
        pre = C.require_committed_prereg()
        cfg = C.load_prereg()
        a0 = cfg['experiments']['A0']
        boot, seed = cfg['statistics']['bootstrap'], cfg['seed']
        alpha = float(re.search(r'alpha=([0-9.]+)', a0['method']['tau_per_split']).group(1))
        cfg_ev, ev_root, _ = CE.load()
        cfg_ev['seed'] = seed
        f2 = json.loads((C.ROOT / 'experiments/results/f2.json').read_text(encoding='utf-8'))
        R = regenerate_f2(cfg_ev, ev_root)
        R['alpha'], R['lam'] = alpha, float(f2['lambda']['value'])
        k_v1 = cfg_ev['change']['k']
        track = score_track(R, normalise=True)
        raw_track = score_track(R, normalise=False)

        # ---- Step 1: reproduction gate (F2's own numbers) -------------------------------------------------
        f2n = f2['scoring_in_f1_harness']['normalised_f3']
        f2r = f2['scoring_in_f1_harness']['no_normalisation_comparator']
        tau = track['tau']
        repro = {'tau': {'regenerated': tau, 'f2_json': f2['tau']['value'], 'abs_diff': abs(tau - f2['tau']['value']),
                         'equal_to_6dp': round(tau, 6) == round(f2['tau']['value'], 6)}}
        calib_w, test_w = R['calib_w'], R['test_w']
        real_cal = real_test = 0
        per_fold_rows, fast_equal = [], True
        for fold in R['folds']:
            fw, vw = real_gate_window_flags(R, track, fold, tau, k_v1)
            wa = track['per_fold'][fold.held_out]['wa']
            fast = wa['in_pop'] & (wa['score'] > tau) & wa['flaggable']
            fast_equal &= bool(np.array_equal(fast, fw)) and bool(np.array_equal(wa['valid_px'], vw))
            row = {'held_out': fold.held_out, 'test_num': float(fw[test_w].sum()), 'test_den': float(vw[test_w].sum()),
                   'cal_num': float(fw[calib_w].sum()), 'cal_den': float(vw[calib_w].sum())}
            ref = next(g for g in f2n['per_fold'] if g['held_out'] == fold.held_out)
            row['f2_json'] = {'test': [ref['window_far_test']['numerator'], ref['window_far_test']['denominator']],
                              'cal': [ref['window_far_calibration']['numerator'], ref['window_far_calibration']['denominator']]}
            row['matches_f2'] = (row['test_num'], row['test_den'], row['cal_num'], row['cal_den']) == (
                *row['f2_json']['test'], *row['f2_json']['cal'])
            per_fold_rows.append(row)
            real_cal += row['cal_num']; real_test += row['test_num']
        pooled_ref = (f2n['pooled_window_far_calibration']['numerator'], f2n['pooled_window_far_calibration']['denominator'],
                      f2n['pooled_window_far_test']['numerator'], f2n['pooled_window_far_test']['denominator'])
        repro['real_gate_per_fold'] = per_fold_rows
        repro['real_gate_pooled'] = {'cal': [real_cal, sum(r['cal_den'] for r in per_fold_rows)],
                                     'test': [real_test, sum(r['test_den'] for r in per_fold_rows)],
                                     'f2_json': {'cal': list(pooled_ref[:2]), 'test': list(pooled_ref[2:])}}
        repro['real_gate_matches_f2'] = bool(all(r['matches_f2'] for r in per_fold_rows)
                                             and (real_cal, sum(r['cal_den'] for r in per_fold_rows)) == pooled_ref[:2]
                                             and (real_test, sum(r['test_den'] for r in per_fold_rows)) == pooled_ref[2:])
        repro['fast_path_equals_real_gate_window_map_at_tau'] = fast_equal
        # fast path vs the real gate at two other taus (one fold), so the equivalence is not a one-tau coincidence
        f0 = R['folds'][0]; wa0 = track['per_fold'][f0.held_out]['wa']
        other = {}
        for mult in (0.5, 2.0):
            fw, _ = real_gate_window_flags(R, track, f0, tau * mult, k_v1)
            fast = wa0['in_pop'] & (wa0['score'] > tau * mult) & wa0['flaggable']
            other[str(mult)] = bool(np.array_equal(fast, fw))
        repro['fast_path_equals_real_gate_at_other_taus_fold0'] = other
        rec_n, rec_r = assemble_records(track), assemble_records(raw_track)
        is_cb = np.tile(test_w.ravel(), len(R['folds']))
        g_cb = gap_from_weights(rec_n, is_cb, np.ones(is_cb.size), alpha)
        repro['records_checkerboard_point'] = {k: g_cb[k] for k in ('num_cal', 'den_cal', 'num_test', 'den_test', 'tau')}
        repro['records_match_f2_json'] = (g_cb['num_cal'], g_cb['den_cal'], g_cb['num_test'], g_cb['den_test']) == pooled_ref
        g_raw = gap_from_weights(rec_r, is_cb, np.ones(is_cb.size), alpha)
        ref_raw = (f2r['pooled_window_far_calibration']['numerator'], f2r['pooled_window_far_calibration']['denominator'],
                   f2r['pooled_window_far_test']['numerator'], f2r['pooled_window_far_test']['denominator'])
        repro['raw_arm_records_match_f2_json'] = (g_raw['num_cal'], g_raw['den_cal'], g_raw['num_test'], g_raw['den_test']) == ref_raw
        gate_ok = (repro['tau']['equal_to_6dp'] and repro['real_gate_matches_f2'] and repro['records_match_f2_json']
                   and fast_equal and all(other.values()))
        result['reproduction_gate'] = repro
        if not gate_ok:
            result.update(status='FAIL', verdict='NOT_RUN',
                          reason='reproduction gate failed: A0 stopped before any statistic was computed; '
                                 'see reproduction_gate for the differences')
            return _finish(result, started)

        # ---- Steps 2-4: the pre-registered statistics ------------------------------------------------------
        rw = random_window_split((40, 32), seed)
        is_rnd = np.tile(rw.ravel(), len(R['folds']))
        cb = bootstrap_gap(rec_n, is_cb, alpha, boot['replicates'], boot['seed'], cfg['statistics']['ci'])
        rnd = bootstrap_gap(rec_n, is_rnd, alpha, boot['replicates'], boot['seed'], cfg['statistics']['ci'])
        verdict = a0_verdict((cb['lo'], cb['hi']), (rnd['lo'], rnd['hi']))
        sec = {}
        for name, rec_x in (('raw_arm', rec_r),):
            sec[name] = {'checkerboard': bootstrap_gap(rec_x, is_cb, alpha, boot['replicates'], boot['seed'], cfg['statistics']['ci']),
                         'random_window': bootstrap_gap(rec_x, is_rnd, alpha, boot['replicates'], boot['seed'], cfg['statistics']['ci'])}
        perm_tile = tile_permutation_reference(rec_n, is_cb, 320, int(R['tile_split'].sum()), alpha, boot['replicates'], seed)
        rec_pos = {**rec_n, 'tile_id': rec_n['pos_id']}
        perm_win = tile_permutation_reference(rec_pos, is_cb, 1280, 640, alpha, boot['replicates'], seed)
        for p in (perm_tile, perm_win):
            p.pop('perm_values')
        n_cal, n_test, x_obs = int(g_cb['n_cal_pop']), int(g_cb['den_test']), int(g_cb['num_test'])
        analytic = {'parameters': betabinom_parameters(n_cal, alpha), 'n_cal': n_cal, 'n_test': n_test, 'x_obs': x_obs,
                    'P_test_count_ge_x_obs': betabinom_tail(n_cal, n_test, alpha, x_obs),
                    'P_test_far_gt_alpha': betabinom_tail(n_cal, n_test, alpha, int(np.floor(alpha * n_test)) + 1),
                    'memo_value_for_x_obs': 0.072, 'memo_value_for_far_gt_alpha': 0.46,
                    'assumption': 'independent windows, perfectly exchangeable; the 3 folds share one set of '
                                  'windows so the true effective n is smaller (memo design effect 2-4)'}

        # ---- Step 5: power (yaml requires it for A0) ---------------------------------------------------------
        power = {'checkerboard': run_power(R, track, 'checkerboard', 0, test_w, cb['tau'], cfg),
                 'random_window': run_power(R, track, 'random_window', 1, rw, rnd['tau'], cfg)}
        vacuous = {k: bool(v['0.5']['flagged'] == 0) for k, v in power.items()}

        status = 'PASS' if verdict == 'TRUE' else 'FAIL'
        meas = [_ci_row('D_checkerboard', cb, boot), _ci_row('D_random_window_split', rnd, boot),
                _ci_row('D_checkerboard_raw_arm_secondary', sec['raw_arm']['checkerboard'], boot),
                _ci_row('D_random_window_raw_arm_secondary', sec['raw_arm']['random_window'], boot)]
        for split, rows in power.items():
            for drop, v in rows.items():
                meas.append(C.measurement(f'recall_{split}_drop_{drop}', v['recall']['estimate'],
                                          numerator=v['recall']['numerator'], denominator=v['recall']['denominator'],
                                          lo=v['recall']['lo'], hi=v['recall']['hi'], replicates=boot['replicates'],
                                          seed=boot['seed'], unit='injected-patch recall at the split tau',
                                          tau=v['tau'], recall_excluding_prior_false_alarm_windows=v['recall_excluding_those'],
                                          denominator_excluding=v['denominator_excluding_those'],
                                          patches_in_already_flagged_windows=v['patches_in_windows_already_flagged_by_placebo']))
        result.update(
            status=status, verdict=verdict,
            rule_applied=a0['keep_rule'], preregistration={**pre, 'experiment': 'A0'},
            checkerboard=cb, random_window_split=rnd, secondary_raw_arm=sec,
            tile_permutation_reference_primary_per_prompt=perm_tile,
            window_permutation_reference_secondary=perm_win,
            analytic_iid_reference=analytic, power=power, power_vacuous_flag=vacuous,
            measurements=meas,
            limitations=[
                'The three leave-one-date-out folds share ONE crop and the same valid windows (nodata is the union over all 3 dates), so '
                'folds are not independent; the bootstrap resamples whole 128 px tiles with all three folds together, which is wider than '
                "F2's per-fold-tile bootstrap (273 blocks).",
                'F2 FAIL (window FAR 0.0652 > 0.05) is not revisited by this result in either direction.',
                'tau is recomputed on gate_v2 own score on the same crop, so exchangeability here is spatial exchangeability within one '
                'crop only; it says nothing about a different crop, AOI or season (B1).',
                'The prompt asks for N random tile splits with N from the yaml; the yaml has no N for A0, so the only replicate count in it '
                '(statistics.bootstrap.replicates = 2000) is used. This reference is additional; the verdict uses only the yaml A0 rule '
                '(bootstrap CI under the checkerboard and under the single window-level random split, seed 2024).',
                'Power injects NDVI drops into raw held-out reflectance (10 x 10 px patches, B08+B04 preserved) and re-runs F3 normalisation, '
                'unmixing and scoring; allocation inside a flagged block is exactly round(16 f) px regardless of the SR ranking, so the '
                'SR product is not needed for recall. The prompt mentions mixing toward e_b; the yaml specifies the NDVI-drop rule, which is used.',
                'Patch recall counts a patch as hit if its 160 m window is detected, including windows the placebo already false-alarmed on; '
                'recall excluding those windows is reported beside it.'])
        return _finish(result, started)
    except C.Blocked as exc:
        result.update(status='BLOCKED', verdict='NOT_RUN', reason=str(exc))
        return _finish(result, started)


def _finish(result, started):
    import json
    out = C.write_result('a0', result, started)
    print(json.dumps({k: out.get(k) for k in ('status', 'verdict', 'reason')}, indent=2))
    return C.exit_code(out['status'])


if __name__ == '__main__':
    raise SystemExit(main())
