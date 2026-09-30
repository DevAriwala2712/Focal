"""F3 noise model v2 (new x5b): tests whether removing a per-date scene-wide offset BEFORE applying A5's sigma
estimators recovers the coverage that A5's own post-hoc diagnostic suggested was available.

Motivation (from configs/fix.yaml `f3_noise_v2`, itself pre-registered before this script's first real run):
A5 (`experiments/x5_noise_model.py`, `experiments/x5_REPORT.md`) MISSED its keep rule (headline leave-one-date-out
coverage 88.0 %, gap -7.5 pp) and diagnosed the cause as scene-wide per-date offsets (77 % / 70 % of mean d^2 in the
two outer folds). A5's own POST HOC diagnostic (computed after the miss, not part of its decision) showed that
subtracting the fold's stable-pixel median offset from `d` lifted coverage to 96.1 % (window) / 99.3 %
(eb_stratified) -- but that offset-removal was never itself pre-registered or tested as an estimator. This script
is that test, pre-registered in `configs/fix.yaml` BEFORE being run. It is a NEW experiment, not a reinterpretation
of A5: `experiments/x5_noise_model.py`, `experiments/results/x5.json` and `experiments/x5_REPORT.md` are untouched.

Pipeline (three real-data stages, one synthetic-injection stage):
  1. estimator: per-date robust MEDIAN offset relative to a pool-fitted reference, fit on stable pixels outside the
     1500 m disks and the footprint (see `normalise_dates` below for why a median offset and not a Theil-Sen gain).
  2. selection (real, E6 12-date cache, other AOI, NO Wayanad pixel): the same replicate/pool procedure A5 used
     (`experiments.x5_noise_model.e6_stack`, imported not reimplemented), with the normalisation step inserted
     before `trustsr.noise.estimate_s2_dates`. A "no_normalisation" comparator column runs the identical procedure
     without step 1, for reference only -- it does not affect which estimator wins.
  3. evaluation (real): leave-one-date-out over the 3 cached Wayanad pre dates. The per-run dihedral NDVI needed for
     this was already computed and cached under `data/experiments-cache/wayanad_evidence/per_date_ndvi/` (8 runs x
     4 dates, real SR output, built for the placebo harness); this script reads it directly rather than re-running
     the SR model. `trustsr.noise.predictive_sigma_from_dates` and `.coverage` are reused verbatim; the bootstrap
     and tiling helpers are imported from `experiments.x5_noise_model` (`summarize`, `az_of`, `tile_ids`,
     `stable_mask_2p5m`) since they are generic statistics/geometry code, not A5-specific claims.
  4. change preservation (synthetic injection on a real stable-pixel field): `experiments/results/f1.json` (F1's
     real placebo harness) did not exist when this script was written -- another agent is building it in parallel.
     A synthetic NDVI drop is injected into a small patch of real Wayanad stable pixels and carried through the
     SAME normalisation fit used in stage 3, to check the scalar offset does not absorb it.

Keep rule (configs/fix.yaml f3_noise_v2, unchanged from A5's threshold): pooled LODO coverage of the selected
estimator within +/- 2 pp of 95.45 %. A miss is reported as FAIL with the gap, not adjusted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from experiments.common import Blocked, environment_record, finalize_result
from experiments.x5_noise_model import (
    az_of, clean, e6_stack, odd_window_px, sha256_file, stable_mask_2p5m, summarize, tile_ids,
)
from risk.common import write_json
from trustsr import noise as N
from trustsr.bootstrap import paired_bootstrap_ci

# NaN-aware reductions below legitimately hit all-invalid pixels at crop edges / masked border pixels; those
# produce NaN (correctly propagated and filtered downstream by the various `mask`/`valid` arguments), not a bug.
warnings.filterwarnings('ignore', message='All-NaN slice encountered', category=RuntimeWarning)
warnings.filterwarnings('ignore', message='Mean of empty slice', category=RuntimeWarning)
warnings.filterwarnings('ignore', message='Degrees of freedom <= 0 for slice', category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[1]
FIX_CONFIG = 'configs/fix.yaml'
EXCEPTIONAL_CONFIG = 'configs/exceptional.yaml'          # a5.noise / a5.selection / a5.evaluation reused verbatim, not redefined
WAYANAD_CACHE = ROOT / 'data/experiments-cache/wayanad_evidence/per_date_ndvi'
SIMPLICITY = ['raw', 'window', 'eb_stratified', 'eb_window']


def load_configs():
    fix_path = ROOT / FIX_CONFIG
    fix = yaml.safe_load(fix_path.read_text(encoding='utf-8'))
    exceptional = yaml.safe_load((ROOT / EXCEPTIONAL_CONFIG).read_text(encoding='utf-8'))
    return fix, sha256_file(fix_path), exceptional


# ---- the estimator: per-date robust median-offset normalisation --------------------------------------------------------

def robust_offsets(stack: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-date scalar offset = median((date image - `ref`) on `mask`). `stack` is (T, H, W); `ref` (H, W)."""
    stack = np.asarray(stack, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        raise ValueError('no valid pixels to fit an offset')
    out = np.empty(stack.shape[0], dtype=np.float64)
    for t in range(stack.shape[0]):
        diff = stack[t] - ref
        out[t] = float(np.nanmedian(diff[mask]))
    return out


def normalise_dates(pool_stack: np.ndarray, pool_mask: np.ndarray, apply_stack=None, apply_mask=None):
    """Fit a robust per-date MEDIAN offset on `pool_stack` (T_pool, H, W) at `pool_mask` pixels (stable, outside
    disks + footprint), never using `apply_stack` (e.g. a held-out date) to build the reference or fit its own
    offset differently -- the held-out date's offset is estimated the same way any deployed date would be, from
    its own pixels against the pool-fitted reference.

    Why a median offset and NOT a Theil-Sen gain (slope): A5's diagnostics (x5_REPORT.md) attribute 70-97 % of
    mean d^2 in the leave-one-date-out folds to a SCENE-WIDE ADDITIVE shift (the fold's stable-pixel median NDVI
    moves by ~0.05 between January dates; there is no reported evidence of a per-pixel multiplicative distortion).
    A location-only (median) estimator is simpler, needs no assumption about a reference brightness axis, is
    robust to a minority of contaminated pixels (see the change-preservation test below), and -- with only 2-3
    pool dates per fold -- a second (gain) parameter would add estimation variance for a term A5 did not find
    evidence of. This choice is made and documented BEFORE the Wayanad evaluation in this script is run.

    Reference: per-pixel MEDIAN across pool dates only (median of medians is itself robust to a single
    contaminated pool date). Returns (norm_pool, norm_apply, offsets_pool, offsets_apply, ref).
    """
    pool_stack = np.asarray(pool_stack, dtype=np.float64)
    if pool_stack.shape[0] < 1:
        raise ValueError('need at least one pool date to fit a reference')
    with np.errstate(invalid='ignore'):
        ref = np.nanmedian(pool_stack, axis=0)
    off_pool = robust_offsets(pool_stack, ref, pool_mask)
    norm_pool = pool_stack - off_pool[:, None, None]
    if apply_stack is None:
        return norm_pool, None, off_pool, None, ref
    apply_stack = np.asarray(apply_stack, dtype=np.float64)
    single = apply_stack.ndim == 2
    if single:
        apply_stack = apply_stack[None]
    am = pool_mask if apply_mask is None else apply_mask
    off_apply = robust_offsets(apply_stack, ref, am)
    norm_apply = apply_stack - off_apply[:, None, None]
    if single:
        norm_apply, off_apply = norm_apply[0], off_apply[0]
    return norm_pool, norm_apply, off_pool, off_apply, ref


# ---- stage 2: estimator selection on E6 (real, other AOI, NO Wayanad pixel) ---------------------------------------------

def run_selection(cfg, wcfg, exceptional):
    """Same replicate/pool procedure as A5's `run_selection` (imported `e6_stack`), with `normalise_dates` inserted
    before `estimate_s2_dates`. The `no_normalisation` column runs the identical procedure without the offset step,
    reported for comparison only -- the winner is chosen on the NORMALISED track alone (pre-registered)."""
    sel, nz = exceptional['a5']['selection'], exceptional['a5']['noise']
    dates, stack, disk = e6_stack(wcfg)
    floor_var = nz['sigma_floor'] ** 2
    window = odd_window_px(nz['window_m'], 10.0)
    rng = np.random.default_rng(sel['seed'])
    n_pool = sel['n_pool_dates']
    methods = list(nz['estimators'])
    tracks = ('normalised', 'no_normalisation')
    rec = {tr: {m: {'nll': [], 'cov2': [], 'keff': [], 'n_px': []} for m in methods} for tr in tracks}
    offset_share, draws, offset_magnitudes = [], [], []
    for _ in range(sel['replicates']):
        pick = rng.permutation(len(dates))[:n_pool + 1]
        draws.append([dates[i] for i in pick])
        pool, held = stack[pick[:n_pool]], stack[pick[n_pool]]
        ok = np.isfinite(stack[pick]).all(axis=0) & ~disk
        # ---- normalised track ----
        norm_pool, norm_held, off_pool, off_held, _ref = normalise_dates(pool, ok, held, ok)
        offset_magnitudes.append(float(np.abs(np.concatenate([off_pool, [off_held]])).mean()))
        d_n = norm_held - norm_pool.mean(axis=0)
        level_n = np.nanmean(norm_pool, axis=0)
        labels_n, _ = N.quantile_strata(level_n, nz['n_strata'], ok)
        med_n = float(np.median(d_n[ok]))
        offset_share.append(med_n ** 2 / float(np.mean(d_n[ok] ** 2)))       # exploratory, reported not decided on
        # ---- unnormalised (raw A5-style) comparator track ----
        d_r = held - pool.mean(axis=0)
        level_r = np.nanmean(pool, axis=0)
        labels_r, _ = N.quantile_strata(level_r, nz['n_strata'], ok)
        for tr, d, labels, pool_use in (('normalised', d_n, labels_n, norm_pool), ('no_normalisation', d_r, labels_r, pool)):
            for m in methods:
                s2, _info = N.estimate_s2_dates(pool_use, m, strata_labels=labels, window=window, valid=ok,
                                                 floor_rel_median=nz['floor_rel_median'], nu0_max=nz['nu0_max'])
                sig2 = N.predictive_sigma2(s2, n_pool, 0.0, 0.0, floor=floor_var)
                use = ok & np.isfinite(sig2)
                nll = 0.5 * (np.log(2 * np.pi * sig2[use]) + d[use] ** 2 / sig2[use])
                sig = np.sqrt(sig2[use])
                rec[tr][m]['nll'].append(float(nll.mean()))
                rec[tr][m]['cov2'].append(float(np.mean(np.abs(d[use]) <= 2.0 * sig)))
                rec[tr][m]['keff'].append(N.effective_k(d[use], sig))
                rec[tr][m]['n_px'].append(int(use.sum()))
    summary = {tr: {} for tr in tracks}
    for tr in tracks:
        for m in methods:
            r = {k: np.asarray(v, dtype=float) for k, v in rec[tr][m].items()}
            summary[tr][m] = {'mean_nll': float(r['nll'].mean()), 'mean_coverage_k2': float(r['cov2'].mean()),
                              'gap_pp_vs_nominal': 100 * float(r['cov2'].mean() - exceptional['a5']['coverage_nominal']),
                              'mean_k_eff': float(r['keff'].mean()), 'mean_valid_px': float(r['n_px'].mean())}
    ranked = sorted(methods, key=lambda m: summary['normalised'][m]['mean_nll'])
    best, second = ranked[0], ranked[1]
    diffs = np.asarray(rec['normalised'][best]['nll']) - np.asarray(rec['normalised'][second]['nll'])
    bs = cfg['statistics']['bootstrap']
    paired = paired_bootstrap_ci(diffs, bs['replicates'], bs['ci'], bs['seed'])
    contains_zero = paired['lo'] <= 0.0 <= paired['hi']
    winner = min((best, second), key=SIMPLICITY.index) if contains_zero else best
    return {'evidence': 'real', 'dataset': 'E6 12-date cache: other AOI (34 km from the Wayanad slide), 10 m B04/B08',
            'no_wayanad_pixel': True, 'replicates': sel['replicates'], 'seed': sel['seed'], 'n_pool_dates': n_pool,
            'window_px_10m': window, 'n_strata': nz['n_strata'], 'dates': dates, 'first_three_draws': draws[:3],
            'summary': summary, 'ranking_by_mean_nll_normalised': ranked, 'best': best, 'runner_up': second,
            'mean_abs_fitted_offset': float(np.mean(offset_magnitudes)),
            'exploratory_scene_median_offset_share_of_mean_d2_after_normalisation': {
                'mean': float(np.mean(offset_share)), 'min': float(np.min(offset_share)), 'max': float(np.max(offset_share)),
                'note': 'share of the RESIDUAL d^2 that is still a scene-wide median offset, after normalisation; '
                        'should be far below A5\'s unnormalised 70-97 % if the estimator removed the effect it targets'},
            'paired_nll_diff_best_minus_runner_up': paired, 'tie_rule_triggered': bool(contains_zero and winner != best),
            'tie_within_ci': bool(contains_zero), 'winner': winner,
            'note': 'replicates share the same 12 dates and pixels: not independent; procedure and hyperparameters '
                    '(n_strata, window, floors, nu0_max) are exactly configs/exceptional.yaml a5.noise/a5.selection, reused verbatim'}


# ---- stage 3: Wayanad leave-one-date-out (real; reads the cached per-run dihedral NDVI, no SR re-run) -------------------

def load_wayanad_dihedral(dates):
    """Per-date (mean over 8 runs, variance over 8 runs, ddof=1) NDVI, from the cache already built for the
    placebo harness (data/experiments-cache/wayanad_evidence/per_date_ndvi/<date>_dihedral_means.npy, real SR
    output, 8 dihedral runs). No SR model is re-run here."""
    means, dvars, raw = {}, {}, {}
    for d in dates:
        f = WAYANAD_CACHE / f'{d}_dihedral_means.npy'
        if not f.is_file():
            raise Blocked(f'real per-date dihedral NDVI not cached: expected {f}', evidence='real')
        a = np.load(f).astype(np.float64)
        raw[d] = a
        with np.errstate(invalid='ignore'):
            means[d] = np.nanmean(a, axis=0)
            dvars[d] = np.nanvar(a, axis=0, ddof=1)
    return means, dvars, raw


def run_wayanad(cfg, wcfg, exceptional, winner):
    nz, st = exceptional['a5']['noise'], cfg['statistics']
    ev = exceptional['a5']['evaluation']
    pre = [str(d) for d in cfg['f7_wayanad_v2']['dates']['pre']]   # ['2024-01-16', '2024-01-21', '2024-01-26'], reused not redefined
                                                                    # (YAML parses unquoted dates as datetime.date; stringify)
    means, dvars, _raw = load_wayanad_dihedral(pre)
    stable, stable_counts = stable_mask_2p5m(wcfg, None, ev)       # `manifest` arg is unused by this helper; disks+footprint from wcfg
    full = np.all([np.isfinite(means[d]) for d in pre], axis=0)
    mask = stable & full
    tile = ev['tile_px_2p5m']
    window = odd_window_px(nz['window_m'], 2.5)
    floor_var = nz['sigma_floor'] ** 2
    nominal = exceptional['a5']['coverage_nominal']
    tid, par = tile_ids(mask.shape, tile)
    res = {'stable_pixels': {**stable_counts, 'valid_2p5m_all_pre_dates': int(full.sum()), 'stable_and_valid_2p5m': int(mask.sum()),
                             'window_px_2p5m': window, 'tile_px_2p5m': tile}}
    az, az_raw, fold_info, offsets_by_fold = {}, {}, {}, {}
    for h in pre:
        rem = [p for p in pre if p != h]
        pool_stack = np.stack([means[p] for p in rem])
        norm_pool, norm_held, off_pool, off_held, _ref = normalise_dates(pool_stack, mask, means[h], mask)
        d = norm_held - norm_pool.mean(axis=0)
        d_raw = means[h] - pool_stack.mean(axis=0)                 # comparator: same fold, WITHOUT normalisation
        dvar_rem = np.stack([dvars[p] for p in rem])
        sigma, info = N.predictive_sigma_from_dates(norm_pool, dvar_rem, dvars[h], method=winner, n_strata=nz['n_strata'],
                                                     window=window, valid=mask, floor=floor_var,
                                                     floor_rel_median=nz['floor_rel_median'], nu0_max=nz['nu0_max'])
        sigma_raw, _ = N.predictive_sigma_from_dates(pool_stack, dvar_rem, dvars[h], method=winner, n_strata=nz['n_strata'],
                                                      window=window, valid=mask, floor=floor_var,
                                                      floor_rel_median=nz['floor_rel_median'], nu0_max=nz['nu0_max'])
        az[h] = az_of(d, sigma)
        az_raw[h] = az_of(d_raw, sigma_raw)
        med_h = float(np.median(d[mask]))
        offsets_by_fold[h] = {'offset_pool': {p: float(o) for p, o in zip(rem, off_pool)}, 'offset_held': float(off_held)}
        fold_info[h] = {'held_out': h, 'remaining': rem, 'n_pre_for_formula': len(rem), 'dof': len(rem) - 1,
                        'median_d_after_normalisation': med_h,
                        'exploratory_offset_share_of_mean_d2_after_normalisation': med_h ** 2 / float(np.mean(d[mask] ** 2)),
                        'median_d_without_normalisation': float(np.median(d_raw[mask])),
                        'median_sigma': float(np.median(sigma[mask])), 'nu0': info.get('nu0')}
    lodo = summarize(az, mask, tile, st, nominal, f'LODO {winner} (normalised)')
    lodo_raw = summarize(az_raw, mask, tile, st, nominal, f'LODO {winner} (no normalisation, same estimator)')
    date_medians = {d: float(np.median(means[d][mask])) for d in pre}
    head = lodo['pooled']
    gap = head['gap_pp_vs_nominal']
    kept = abs(gap) <= 2.0
    res['lodo'] = {'evidence': 'real', 'n_pre_for_formula': 2, 'dof_per_pixel': 1,
                   'definition': 'd = held-out pre date minus mean of the other two pre dates, both normalised by the '
                                 'per-date median-offset estimator fit on stable pixels outside disks+footprint; '
                                 'sigma from the same two dates + held-out dihedral variance (trustsr.noise, unchanged); '
                                 'matches configs/exceptional.yaml a5.coverage_metric verbatim',
                   'folds': fold_info, 'offsets_by_fold': offsets_by_fold,
                   'estimator_normalised': lodo, 'estimator_no_normalisation_same_sigma_method': lodo_raw,
                   'median_ndvi_on_stable_px_by_date': date_medians,
                   'independent_dates_note': 'three folds share three dates: at most 3 date effects, not independent'}
    res['keep_rule'] = {'rule': "pooled LODO coverage of |d| <= 2 sigma for the selected estimator within +/- 2 pp of "
                                "95.45 %, same threshold A5 used (configs/fix.yaml f3_noise_v2.keep_rule)",
                        'estimator': winner, 'coverage': head['coverage_k2'], 'ci95': head['ci95'], 'gap_pp': gap,
                        'kept': bool(kept),
                        'coverage_without_normalisation_same_sigma_method': lodo_raw['pooled']['coverage_k2'],
                        'ci95_without_normalisation': lodo_raw['pooled']['ci95'],
                        'cause_note': None if kept else 'MISS reported. See folds/offsets_by_fold for per-fold offsets and '
                                                        'residual d.'}
    return res


# ---- stage 4: change-preservation test (synthetic injection on a real stable-pixel field) -------------------------------

def change_preservation_test(means: dict, mask: np.ndarray, *, rng_seed: int, drop: float, patch_px: int) -> dict:
    """Inject a synthetic NDVI drop into a small square patch of REAL stable pixels of the held-out date, then run
    the SAME `normalise_dates` fit used in stage 3 on the contaminated data. If the scalar median offset absorbed
    real local change, the fitted offset would shift substantially and/or the drop would be erased after
    normalisation. `experiments/results/f1.json` (F1's real placebo harness) did not exist when this ran (another
    agent is building it in parallel); this synthetic-injection check is used instead, as instructed.
    """
    dates = list(means)
    held = dates[-1]
    rem = dates[:-1]
    pool_stack = np.stack([means[p] for p in rem])
    rng = np.random.default_rng(rng_seed)
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        raise ValueError('no stable pixels to inject a synthetic change into')
    i = rng.integers(0, ys.size)
    r0, c0 = int(ys[i]), int(xs[i])
    h, w = mask.shape
    r1, c1 = min(r0 + patch_px, h), min(c0 + patch_px, w)
    patch = np.zeros(mask.shape, bool)
    patch[r0:r1, c0:c1] = True
    patch &= mask                                   # only pixels that are genuinely stable/valid are perturbed
    if patch.sum() == 0:
        raise ValueError('injected patch has zero stable pixels; adjust patch_px or seed')
    contaminated = means[held].copy()
    contaminated[patch] -= drop

    _, _, off_pool_before, off_held_before, _ = normalise_dates(pool_stack, mask, means[held], mask)
    _, _, off_pool_after, off_held_after, _ = normalise_dates(pool_stack, mask, contaminated, mask)

    normalised_before = means[held] - off_held_before
    normalised_after = contaminated - off_held_after
    measured_drop = float(np.median((normalised_before - normalised_after)[patch]))
    recovery_ratio = measured_drop / drop if drop else float('nan')
    offset_shift = float(off_held_after - off_held_before)
    return {'evidence': 'synthetic (injected change on a real Wayanad stable-pixel field)',
            'held_out_date': held, 'pool_dates': rem, 'injected_patch_px': int(patch.sum()),
            'stable_px_used_to_fit_offset': int(mask.sum()), 'injected_drop': drop,
            'fitted_offset_held_before_injection': float(off_held_before), 'fitted_offset_held_after_injection': float(off_held_after),
            'offset_shift_from_injection': offset_shift,
            'offset_shift_as_fraction_of_injected_drop': float(offset_shift / drop) if drop else None,
            'measured_drop_after_normalisation': measured_drop, 'recovery_ratio': recovery_ratio,
            'interpretation': ('the injected patch is a small fraction of the stable-pixel set the offset is fit on, so a '
                               'robust (median) offset should barely move, and the drop should survive normalisation '
                               'almost intact (recovery_ratio near 1.0); a ratio far from 1 would mean the estimator is '
                               'absorbing real local change into its scene-wide correction')}


def write_report(out, path: Path):
    m = out['measurements']
    se, wy, ct = m['selection'], m['wayanad'], m['change_preservation']
    L, kr = wy['lodo'], wy['keep_rule']
    pct = lambda x: f'{100 * x:.1f} %'
    ci = lambda a: f'[{100 * a[0]:.1f}, {100 * a[1]:.1f}]'
    lines = ['# F3 noise model v2 report', '',
             f"Status **{out['status']}**. Motivated by A5 diagnostics; pre-registered before its own run (configs/fix.yaml "
             f"`f3_noise_v2`; does not edit `experiments/x5_noise_model.py`). Selected estimator (E6-only selection, no "
             f"Wayanad pixel): **{out['selected_estimator']}**. Config sha256 `{out['config_sha256'][:16]}...`.", '',
             '## 1. Wayanad leave-one-date-out (real; headline)', '',
             f"Stable pixels: {wy['stable_pixels']['stable_and_valid_2p5m']:,} 2.5 m px per fold (outside the 1500 m disks, "
             'footprint and parent), from the real per-run dihedral NDVI cache (no SR re-run). Coverage of |d| <= 2 sigma, '
             'pooled over 3 folds; 95 % block-bootstrap CI; nominal 95.45 %.', '',
             f"| track | coverage k=2 [CI] | gap vs nominal |",
             '|---|---|---|',
             f"| normalised (F3 estimator) | {pct(kr['coverage'])} {ci(kr['ci95'])} | {kr['gap_pp']:+.1f} pp |",
             f"| same sigma method, no normalisation | {pct(kr['coverage_without_normalisation_same_sigma_method'])} "
             f"{ci(kr['ci95_without_normalisation'])} | {100*(kr['coverage_without_normalisation_same_sigma_method']-0.9545):+.1f} pp |",
             '', f"Keep rule (within +/- 2 pp of 95.45 %): **{'KEPT' if kr['kept'] else 'MISSED'}**.", '',
             'Per fold (median d after normalisation, offset share of residual d^2, fitted offsets):', '']
    for h, f in L['folds'].items():
        off = L['offsets_by_fold'][h]
        lines.append(f"- {h}: median d {f['median_d_after_normalisation']:+.4f} (was {f['median_d_without_normalisation']:+.4f} "
                     f"before normalisation), offset share {pct(f['exploratory_offset_share_of_mean_d2_after_normalisation'])}, "
                     f"offset(held) {off['offset_held']:+.4f}")
    lines += ['', f"Median NDVI on stable pixels by date: " + ', '.join(f'{d} {v:.3f}' for d, v in L['median_ndvi_on_stable_px_by_date'].items()) + '.', '',
              '## 2. Estimator selection (real, E6 12-date cache, other AOI, ' f"{se['replicates']} seeded draws; no Wayanad pixel)", '',
              '| sigma | mean held-out NLL (normalised) | coverage k=2 (normalised) | mean NLL (no normalisation) | coverage (no normalisation) |',
              '|---|---|---|---|---|']
    for name in ['raw', 'window', 'eb_stratified', 'eb_window']:
        sn, sr = se['summary']['normalised'][name], se['summary']['no_normalisation'][name]
        lines.append(f"| {name} | {sn['mean_nll']:.3f} | {pct(sn['mean_coverage_k2'])} | {sr['mean_nll']:.3f} | {pct(sr['mean_coverage_k2'])} |")
    p = se['paired_nll_diff_best_minus_runner_up']
    off_share = se['exploratory_scene_median_offset_share_of_mean_d2_after_normalisation']
    lines += ['', f"Winner {se['winner']}; paired NLL diff best - runner-up ({se['best']} - {se['runner_up']}) {p['mean']:.3f} "
              f"[{p['lo']:.3f}, {p['hi']:.3f}] ({'tie rule applied' if se['tie_within_ci'] else 'CI excludes 0'}). "
              f"Mean |fitted offset| {se['mean_abs_fitted_offset']:.4f} NDVI. Residual scene-offset share of mean d^2 after "
              f"normalisation: {pct(off_share['mean'])} (range {pct(off_share['min'])}-{pct(off_share['max'])}).", '',
              '## 3. Change-preservation test (synthetic injection on a real stable-pixel field)', '',
              f"Injected a {ct['injected_drop']:+.2f} NDVI drop into {ct['injected_patch_px']} real stable px of "
              f"{ct['held_out_date']} (out of {ct['stable_px_used_to_fit_offset']:,} stable px the offset is fit on). "
              f"Fitted offset shifted by {ct['offset_shift_from_injection']:+.5f} "
              f"({100*ct['offset_shift_as_fraction_of_injected_drop']:+.2f} % of the injected drop). "
              f"Measured drop after normalisation: {ct['measured_drop_after_normalisation']:+.4f} "
              f"(recovery ratio {ct['recovery_ratio']:.3f}; 1.0 = fully preserved).", '',
              '## Limitations', ''] + [f'- {x}' for x in out['limitations']]
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--injection-seed', type=int, default=None, help='overrides cfg seed for the injection patch location')
    ap.add_argument('--injection-drop', type=float, default=0.30, help='NDVI magnitude of the synthetic injected change')
    ap.add_argument('--injection-patch-px', type=int, default=40, help='side length (2.5 m px) of the injected square')
    args = ap.parse_args()
    started = datetime.now(timezone.utc).isoformat()
    cfg, cfg_hash, exceptional = load_configs()
    from experiments.wayanad_evidence import config as C
    wcfg, _wroot, _wconfig_hash = C.load(str(ROOT / 'configs/wayanad_evidence.yaml'))
    np.random.seed(cfg['seed'])
    try:
        stage = {}
        selection = run_selection(cfg, wcfg, exceptional)
        stage['selection'] = selection
        stage['wayanad'] = run_wayanad(cfg, wcfg, exceptional, selection['winner'])
        pre = [str(d) for d in cfg['f7_wayanad_v2']['dates']['pre']]
        means, _dvars, _raw = load_wayanad_dihedral(pre)
        ev = exceptional['a5']['evaluation']
        stable, _counts = stable_mask_2p5m(wcfg, None, ev)
        full = np.all([np.isfinite(means[d]) for d in pre], axis=0)
        mask = stable & full
        stage['change_preservation'] = change_preservation_test(
            means, mask, rng_seed=args.injection_seed if args.injection_seed is not None else cfg['seed'],
            drop=args.injection_drop, patch_px=args.injection_patch_px)
        kept = stage['wayanad']['keep_rule']['kept']
        result = {'status': 'PASS' if kept else 'FAIL', 'evidence': 'real',
                  'evidence_labels': {'selection': 'real (E6, other AOI, no Wayanad pixel)', 'wayanad.lodo': 'real',
                                      'change_preservation': 'synthetic (injected change on a real stable-pixel field)'},
                  'motivated_by': cfg['f3_noise_v2']['motivated_by'],
                  'status_note': cfg['f3_noise_v2']['status'],
                  'selected_estimator': selection['winner'], 'measurements': stage,
                  'x5_reference': {'file': 'experiments/x5_noise_model.py (not edited)', 'result': 'experiments/results/x5.json (not edited)',
                                   'report': 'experiments/x5_REPORT.md (not edited)'},
                  'limitations': ['median(d) is exactly 0 for each LODO fold by construction: offset_held is DEFINED as the '
                                  'median residual against the pool reference, so removing it zeroes the median identically. '
                                  'This is a definitional consequence of the estimator, not itself evidence; the non-trivial, '
                                  'non-tautological result is the COVERAGE (|d| <= 2 sigma), since sigma is estimated from the '
                                  'pool dates’ across-date and dihedral variance, never from d',
                                  'three Wayanad pre dates only: leave-one-date-out has n_pre = 2 (1 dof) and 3 non-independent folds',
                                  'block-bootstrap CIs resample space, not dates: they understate date-level uncertainty',
                                  'the median offset is a SINGLE scalar per date: it cannot correct within-date spatial '
                                  'structure (e.g. a cloud-edge gradient), only a scene-wide shift',
                                  'change-preservation test is synthetic (injected on real pixels): it is not evidence '
                                  'about a real event, only about this estimator\'s sensitivity to a minority-pixel change',
                                  'stable pixels lie inside the SR crop around the slide, outside the footprint and 1500 m disks',
                                  'coverage on stable pixels says nothing about power on real change']}
    except Blocked as exc:
        result = {'status': 'BLOCKED', 'evidence': exc.evidence, 'reason': str(exc)}
    out = finalize_result('f3', result, cfg_hash, started)
    out['environment'] = environment_record({})
    out['config_sha256'] = cfg_hash
    out['config'] = {'file': FIX_CONFIG, 'f3_noise_v2': cfg['f3_noise_v2']}
    write_json(ROOT / cfg['results_dir'] / 'f3.json', clean(out))
    if 'measurements' in out:
        write_report(clean(out), ROOT / cfg['results_dir'] / 'f3_REPORT.md')
    print(json.dumps({k: clean(out[k]) for k in ('status', 'selected_estimator') if k in out}, indent=2))
    if 'measurements' in out:
        print(json.dumps(clean(out['measurements']['wayanad']['keep_rule']), indent=2))
    raise SystemExit(0 if out['status'] == 'PASS' else 2)


if __name__ == '__main__':
    main()
