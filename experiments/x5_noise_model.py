"""X5 (agent A5): calibrate the gate noise model on real stable pixels. Decision rule and failure modes: docs/adr-x5-noise-model.md.

Stages (each labelled with its evidence type in the result JSON):
  0. synthetic: bias table of v1 sigma vs the new sigma on a simulation with a known truth (same generator as the unit tests).
  1. build: re-run the pretrained SEN2SR-lite (8 dihedral runs, deterministic, CPU) on the cached Wayanad crop and keep per-date
     NDVI mean and dihedral variance under ./scratch_a5 (per-date, NOT pooled). Verified against the cached pooled Welford state.
  2. selection (real, E6 12-date cache, other AOI, 10 m, NO Wayanad pixel): choose the small-n estimator by the pre-registered rule.
  3. Wayanad (real): leave-one-date-out coverage on stable pixels (headline), pre-mean vs December (season-confounded), k_eff,
     stable-pixel empirical comparator. Block-bootstrap CIs (128 px = 320 m blocks) and denominators.

Exit code 0 = keep rule met, 2 = keep rule missed (FAIL, reported with its cause) or BLOCKED.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from experiments.common import Blocked, array_sha256, environment_record, finalize_result
from risk.common import write_json
from trustsr import noise as N
from trustsr.bootstrap import block_sums, paired_bootstrap_ci, ratio_bootstrap_ci

ROOT = Path(__file__).resolve().parents[1]
CONFIG = 'configs/exceptional.yaml'
SCRATCH = ROOT / 'scratch_a5'
SIMPLICITY = ['raw', 'window', 'eb_stratified', 'eb_window']          # simplest first (tie rule in configs a5.selection)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(x):
    """JSON-safe: numpy scalars/arrays to Python, NaN/inf to None."""
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def odd_window_px(window_m: float, res_m: float) -> int:
    n = int(round(window_m / res_m))
    return n if n % 2 else n + 1


def load_config():
    path = ROOT / CONFIG
    cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
    return cfg, sha256_file(path)


# ---- stage 0: synthetic bias table ----------------------------------------------------------------------------------------

def synthetic_bias(cfg):
    """v1 vs new sigma on a simulation with a known truth (SYNTHETIC evidence; tests/test_noise.py asserts the same numbers)."""
    nz = cfg['a5']['noise']
    s = N.simulate_dates(seed=cfg['seed'])
    means, dvar, dm = N.moments_from_runs(s['x'])
    n = s['n_pre']
    d = means[-1] - means[:-1].mean(axis=0)
    _, pv = N.pool_moments(means[:n], dvar[:n], dm.count_stack()[:n])
    old2 = N.sigma_v1(np.sqrt(pv), np.sqrt(dvar[-1])) ** 2
    labels, _ = N.quantile_strata(s['u'], 8)
    out = {'evidence': 'synthetic', 'n_px': int(d.size), 'n_pre': n, 'runs_per_date': s['runs'], 'true_prior_dof': 10.0,
           'old_v1': {'variance_ratio_to_truth': float(old2.mean() / s['true_var'].mean()),
                      'sd_ratio_to_truth': float(np.sqrt(old2.mean() / s['true_var'].mean())),
                      'coverage_at_k2': N.coverage(d, np.sqrt(old2), 2.0), 'k_eff': N.effective_k(d, np.sqrt(old2))},
           'oracle_true_sigma_coverage_at_k2': N.coverage(d, np.sqrt(s['true_var']), 2.0)}
    for m in N.METHODS:
        kw = {'strata_labels': labels} if m == 'eb_stratified' else {}
        if m in ('window', 'eb_window'):
            continue                                   # the simulation has no spatial layout
        s2, info = N.estimate_s2_dates(means[:n], m, **kw)
        sig2 = N.predictive_sigma2(s2, n, dvar[:n].mean(axis=0), dvar[-1], floor=nz['sigma_floor'] ** 2)
        target = s['tau2'] + s['delta2'] / s['runs']
        out[m] = {'variance_ratio_to_truth': float(sig2.mean() / s['true_var'].mean()),
                  'coverage_at_k2': N.coverage(d, np.sqrt(sig2), 2.0), 'k_eff': N.effective_k(d, np.sqrt(sig2)),
                  'mse_log_variance_of_s2_dates': float(np.mean((np.log(np.maximum(s2, 1e-12)) - np.log(target)) ** 2)),
                  'nu0_estimated': info.get('nu0'), 'shrink_weight': info.get('shrink_weight')}
    return out


# ---- stage 1: per-date dihedral statistics ------------------------------------------------------------------------------

def build_date_stats(wcfg, wroot, wconfig_hash, force: bool):
    """Per-date NDVI mean / dihedral variance / count on the 2.5 m crop grid, cached under scratch_a5 with SHA-256s."""
    from experiments.wayanad_evidence import config as C
    manifest_path = SCRATCH / 'manifest.json'
    if manifest_path.is_file() and not force:
        m = json.loads(manifest_path.read_text(encoding='utf-8'))
        if m.get('wayanad_config_sha256') == wconfig_hash and all((SCRATCH / f).is_file() for f in m['files']):
            return m
    from experiments.wayanad_evidence.data import load_date
    from experiments.wayanad_evidence.gate import upsample
    from experiments.wayanad_evidence.sr import Runner, sr_variants
    from experiments.wayanad_evidence.stats import ndvi
    from experiments.wayanad_evidence.step3_sr import fill_invalid
    from experiments.wayanad_evidence.tiler import tile_plan
    cache = C.cache(wcfg, wroot)
    with np.load(cache / 'step1_state.npz') as z:
        state = {k: z[k] for k in z.files}
    dates = [str(d) for d in state['dates']]
    tile, margin, stride, scale = (wcfg['tiling'][k] for k in ('tile', 'crop_margin_px', 'stride', 'scale'))
    r0c, r1c, c0c, c1c = (int(v) for v in state['crop'])
    hc, wc = r1c - r0c, c1c - c0c
    valid_hr = upsample(state['valid_all'][r0c:r1c, c0c:c1c], scale)
    floor = wcfg['radiometry']['min_denominator']
    runner = Runner(wcfg, wroot)
    plan = tile_plan((hc, wc), tile, margin, stride)                         # fixed row-major tile order
    dm = N.DateMoments((hc * scale, wc * scale), dates)
    t_all = time.perf_counter()
    for date in dates:
        a = load_date(wcfg, cache, date)
        refl = a['refl'][:, r0c:r1c, c0c:c1c]
        model_in = fill_invalid(refl, a['valid_scl'][r0c:r1c, c0c:c1c])
        for t in plan:
            (rl, rh), (cl, ch) = t['keep_r'], t['keep_c']
            v = sr_variants(runner, model_in[:, t['r0']:t['r0'] + tile, t['c0']:t['c0'] + tile], f'{date} r0={t["r0"]} c0={t["c0"]}',
                            runner.oom_type, runner.cleanup, runner.memory)
            v = v[:, :, (rl - t['r0']) * scale:(rh - t['r0']) * scale, (cl - t['c0']) * scale:(ch - t['c0']) * scale]
            region = (slice(rl * scale, rh * scale), slice(cl * scale, ch * scale))
            nd = ndvi(v[:, 0], v[:, 3], floor)                                # NDVI per run, then moments (not NDVI of the mean)
            for k in range(8):
                dm.update_at(date, region, nd[k], where=valid_hr[region])
        print(f'built {date} ({time.perf_counter() - t_all:.0f} s)', flush=True)
    means, dvar, counts = dm.mean_stack(), dm.var_stack(), dm.count_stack()
    # consistency with the cached POOLED Welford state (step3): pooled per-date moments must reproduce it
    with np.load(cache / 'step3_state.npz') as z:
        cached = {k: z[k] for k in z.files}
    npre = len([d for d in dates if d != str(state['post'])])
    pre_idx = [dates.index(d) for d in dates if d != str(state['post'])]
    pm, pv = N.pool_moments(means[pre_idx], dvar[pre_idx], counts[pre_idx], ddof=1)
    both = np.isfinite(pv) & np.isfinite(cached['pre_std'])
    check = {'pre_mean_max_abs_diff': float(np.abs(pm[both] - cached['pre_mean'][both]).max()),
             'pre_std_max_abs_diff': float(np.abs(np.sqrt(pv[both]) - cached['pre_std'][both]).max()),
             'post_std_max_abs_diff': float(np.nanmax(np.abs(np.sqrt(dvar[dates.index(str(state['post']))]) - cached['post_std']))),
             'pre_count_equal': bool(np.array_equal(counts[pre_idx].sum(axis=0), cached['pre_count'])),
             'compared_px': int(both.sum()), 'n_pre': npre}
    SCRATCH.mkdir(exist_ok=True)
    files, hashes = [], {}
    for name, arr, dt in (('mean', means, np.float32), ('var', dvar, np.float32), ('count', counts, np.uint8)):
        f = f'{name}.npy'
        a32 = arr.astype(dt)
        np.save(SCRATCH / f, a32)
        files.append(f)
        hashes[f] = array_sha256(a32)
    manifest = {'dates': dates, 'pre': [str(d) for d in state['pre']], 'post': str(state['post']),
                'crop_10m': [r0c, r1c, c0c, c1c], 'shape_2p5m': [hc * scale, wc * scale], 'tiles_per_date': len(plan),
                'runs_per_date': 8, 'files': files, 'array_sha256': hashes, 'wayanad_config_sha256': wconfig_hash,
                'consistency_vs_cached_pooled_state': check, 'build_seconds': time.perf_counter() - t_all,
                'device': wcfg['model']['device'], 'seed': wcfg['seed']}
    manifest_path.write_text(json.dumps(clean(manifest), indent=2), encoding='utf-8')
    return manifest


# ---- masks -----------------------------------------------------------------------------------------------------------------

def stable_mask_2p5m(wcfg, manifest, stable_cfg):
    """Stable 2.5 m pixels: valid on all dates (8 runs each), parent false, outside the footprint and the 1500 m disks."""
    from affine import Affine
    from pyproj import Transformer
    from experiments.wayanad_evidence.gate import upsample
    from experiments.wayanad_evidence.geo import reference_grid
    from experiments.wayanad_evidence import config as C
    cache = C.cache(wcfg, ROOT)
    with np.load(cache / 'step1_state.npz') as z:
        r0, r1, c0, c1 = (int(v) for v in z['crop'])
        valid_all, parent, footprint = z['valid_all'][r0:r1, c0:c1], z['parent'][r0:r1, c0:c1], z['footprint'][r0:r1, c0:c1]
    transform, _ = reference_grid(wcfg['aoi'])
    t10 = transform * Affine.translation(c0, r0)
    rows, cols = np.indices(valid_all.shape)
    x, y = t10 * (cols + 0.5, rows + 0.5)
    to_utm = Transformer.from_crs('EPSG:4326', wcfg['aoi']['crs'], always_xy=True)
    radius = wcfg['aoi']['plausibility_radius_m']
    disk = np.zeros(valid_all.shape, bool)
    for key in stable_cfg['stable_disk_points']:
        p = wcfg['aoi']['plausibility_points'][key]
        px, py = to_utm.transform(p['lon'], p['lat'])
        disk |= (x - px) ** 2 + (y - py) ** 2 <= radius ** 2
    stable10 = valid_all & ~parent & ~footprint & ~disk
    scale = wcfg['tiling']['scale']
    counts = {'crop_px_10m': int(valid_all.size), 'valid_all_dates_10m': int(valid_all.sum()),
              'parent_10m': int(parent.sum()), 'footprint_10m': int(footprint.sum()), 'disk_10m': int(disk.sum()),
              'stable_10m': int(stable10.sum()), 'disk_radius_m': radius, 'disk_points': stable_cfg['stable_disk_points']}
    return upsample(stable10, scale), counts


# ---- statistics helpers -----------------------------------------------------------------------------------------------------

def tile_ids(shape, block):
    rr, cc = np.indices(shape)
    bi, bj = rr // block, cc // block
    return (bi * (-(-shape[1] // block)) + bj).astype(np.int32), ((bi + bj) % 2).astype(np.int8)


def hist_quantile(hist, q, zmax, nbins):
    cum = np.cumsum(hist)
    if cum[-1] <= 0:
        return np.nan
    idx = int(np.searchsorted(cum, q * cum[-1]))
    return (min(idx, nbins - 1) + 1) * zmax / nbins


def summarize(az_folds, mask, tile, stats_cfg, nominal, label, extra=None):
    """Coverage of |d| <= 2 sigma and k_eff from folds of |z| = |d|/sigma on `mask`. Pooled over folds: blocks sum across folds."""
    bs = stats_cfg['bootstrap']
    reps, ci, seed = bs['replicates'], bs['ci'], bs['seed']
    block = tile
    tid, par = tile_ids(mask.shape, block)
    nb = int(tid.max()) + 1
    zmax, nbins = 30.0, 3000
    out = {'label': label, 'folds': {}}
    num_pool, den_pool, hist_pool = np.zeros(nb), np.zeros(nb), np.zeros(nb * nbins)
    all_az, all_par = [], []
    for name, az in az_folds.items():
        ok = mask & np.isfinite(az) | (mask & np.isinf(az))
        cov = ok & (az <= 2.0)
        num, den = np.bincount(tid[cov], minlength=nb).astype(float), np.bincount(tid[ok], minlength=nb).astype(float)
        num_pool += num
        den_pool += den
        zc = np.minimum(az[ok], zmax * (1 - 1e-9))
        hist_pool += np.bincount(tid[ok] * nbins + (zc / zmax * nbins).astype(np.int64), minlength=nb * nbins)
        res = ratio_bootstrap_ci(num, den, reps, ci, seed)
        out['folds'][name] = {'coverage_k2': res['estimate'], 'ci95': [res['lo'], res['hi']], 'n_px': int(res['denominator']),
                              'blocks': res['blocks'], 'median_abs_z': float(np.median(az[ok])),
                              'robust_scale_of_z_madn': float(np.median(az[ok]) / 0.6745),
                              'frac_abs_z_gt_3': float(np.mean(az[ok] > 3.0)), 'frac_abs_z_gt_4': float(np.mean(az[ok] > 4.0))}
        all_az.append(az[ok])
        all_par.append(par[ok])
    pooled = ratio_bootstrap_ci(num_pool, den_pool, reps, ci, seed)
    az_cat, par_cat = np.concatenate(all_az), np.concatenate(all_par)
    k_all = float(np.quantile(az_cat, nominal))
    # k_eff CI: block bootstrap of the nominal quantile via per-block histograms (0.01 resolution)
    H = hist_pool.reshape(nb, nbins)
    rng = np.random.default_rng(seed)
    keep = np.flatnonzero(den_pool > 0)
    ks = np.empty(reps)
    for r in range(reps):
        pick = keep[rng.integers(0, keep.size, keep.size)]
        ks[r] = hist_quantile(H[pick].sum(axis=0), nominal, zmax, nbins)
    # spatial cross-fit: k from one checkerboard parity applied to the other
    k_par = {p: float(np.quantile(az_cat[par_cat == p], nominal)) for p in (0, 1)}
    num_cf, den_cf = np.zeros(nb), np.zeros(nb)
    for name, az in az_folds.items():
        ok = mask & (np.isfinite(az) | np.isinf(az))
        for p in (0, 1):
            sel = ok & (par == p)
            cov = sel & (az <= k_par[1 - p])
            num_cf += np.bincount(tid[cov], minlength=nb)
            den_cf += np.bincount(tid[sel], minlength=nb)
    cf = ratio_bootstrap_ci(num_cf, den_cf, reps, ci, seed)
    out.update({'pooled': {'coverage_k2': pooled['estimate'], 'ci95': [pooled['lo'], pooled['hi']],
                           'covered_px': pooled['numerator'], 'n_px': pooled['denominator'], 'blocks': pooled['blocks'],
                           'gap_pp_vs_nominal': 100.0 * (pooled['estimate'] - nominal),
                           'ci_contains_nominal': bool(pooled['lo'] <= nominal <= pooled['hi'])},
                'k_eff': {'value_all_stable': k_all, 'ci95_block_bootstrap': [float(np.quantile(ks, 0.025)), float(np.quantile(ks, 0.975))],
                          'value_parity0': k_par[0], 'value_parity1': k_par[1],
                          'crossfit_coverage_at_k_eff': {'coverage': cf['estimate'], 'ci95': [cf['lo'], cf['hi']],
                                                        'n_px': cf['denominator'], 'note': 'k fitted on one checkerboard parity, '
                                                        'evaluated on the other, both directions'}},
                'robust_scale_of_z_madn': float(np.median(az_cat) / 0.6745),
                'frac_abs_z_gt_3': float(np.mean(az_cat > 3.0)), 'frac_abs_z_gt_4': float(np.mean(az_cat > 4.0))})
    if extra:
        out.update(extra)
    return out


def az_of(d, sigma):
    """|d|/sigma; sigma = 0 gives inf (uncovered) unless d is 0."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.abs(d) / sigma


# ---- stage 2: selection on E6 ---------------------------------------------------------------------------------------------

def e6_stack(cfg_wayanad):
    """12 clear pre dates x (1000,1000) NDVI (NaN where SCL-invalid or denominator < floor) and the E6 suspected-disk mask."""
    from pyproj import Transformer
    from experiments.common import load_experiment_config, ndvi_bands
    from experiments.e6_season_matched import baseline_offset_ok, cache_path, disk_mask, reflectance, valid_from_scl
    from risk.r2_imagery import aoi_grid
    exp, root, _ = load_experiment_config('configs/experiments.yaml')
    s, im = exp['e6'], exp['phase0']['imagery']
    r2 = json.loads((root / s['r2_result']).read_text(encoding='utf-8'))
    dates = r2['pre_dates']
    cache = root / exp['paths']['cache'] / s['cache_subdir']
    missing = [d for d in dates if not cache_path(cache, d).is_file()]
    if missing:
        raise Blocked(f'E6 cache missing {len(missing)} of {len(dates)} dates under {cache}: {missing[:3]}...', evidence='real')
    rad = s['radiometry']
    floor = cfg_wayanad['radiometry']['min_denominator']
    stack = []
    for d in dates:
        with np.load(cache_path(cache, d)) as z:
            red, nir, scl, meta = z['red'], z['nir'], z['scl'], json.loads(str(z['meta']))
        if not all(baseline_offset_ok(b, s['min_processing_baseline']) for b in meta['processing_baselines']):
            raise Blocked(f'{d}: processing baseline below {s["min_processing_baseline"]}', evidence='real')
        ok = valid_from_scl(scl, im['cloud_classes'], im['shadow_classes'], im['invalid_classes'])
        nd = ndvi_bands(reflectance(red, rad['quantification'], rad['boa_add_offset']),
                        reflectance(nir, rad['quantification'], rad['boa_add_offset']), floor)
        stack.append(np.where(ok, nd, np.nan))
    stack = np.stack(stack).astype(np.float64)
    _, transform, shape = aoi_grid({**im, 'audit_resolution_m': 10})
    to_metric = Transformer.from_crs('EPSG:4326', im['crs'], always_xy=True)
    disk = np.zeros(shape, bool)
    for zone in s['suspected_areas']:
        disk |= disk_mask(transform, shape, to_metric.transform(zone['lon'], zone['lat']), zone['radius_m'])
    return dates, stack, disk


def run_selection(cfg, wcfg):
    sel, nz = cfg['a5']['selection'], cfg['a5']['noise']
    dates, stack, disk = e6_stack(wcfg)
    floor_var = nz['sigma_floor'] ** 2
    window = odd_window_px(nz['window_m'], 10.0)
    rng = np.random.default_rng(sel['seed'])
    n_pool = sel['n_pool_dates']
    methods = list(nz['estimators']) + ['no_inflation_reference']
    rec = {m: {'nll': [], 'cov2': [], 'keff': [], 'n_px': [], 'cov2c': [], 'nu0': []} for m in methods}
    offset_share, draws = [], []
    for _ in range(sel['replicates']):
        pick = rng.permutation(len(dates))[:n_pool + 1]
        draws.append([dates[i] for i in pick])
        pool, held = stack[pick[:n_pool]], stack[pick[n_pool]]
        ok = np.isfinite(stack[pick]).all(axis=0) & ~disk
        d = held - pool.mean(axis=0)
        level = np.nanmean(pool, axis=0)
        labels, _ = N.quantile_strata(level, nz['n_strata'], ok)
        med_all = float(np.median(d[ok]))
        offset_share.append(med_all ** 2 / float(np.mean(d[ok] ** 2)))          # exploratory: share of E[d^2] that is a scene-wide offset
        for m in methods:
            info = {}
            if m == 'no_inflation_reference':
                s2, _ = N.estimate_s2_dates(pool, 'raw')                        # sigma^2 = s2_dates, no (1 + 1/n) and no shrinkage
                sig2 = np.maximum(s2, floor_var)
            else:
                s2, info = N.estimate_s2_dates(pool, m, strata_labels=labels, window=window, valid=ok,
                                               floor_rel_median=nz['floor_rel_median'], nu0_max=nz['nu0_max'])
                sig2 = N.predictive_sigma2(s2, n_pool, 0.0, 0.0, floor=floor_var)
            use = ok & np.isfinite(sig2)
            nll = 0.5 * (np.log(2 * np.pi * sig2[use]) + d[use] ** 2 / sig2[use])
            sig = np.sqrt(sig2[use])
            rec[m]['nll'].append(float(nll.mean()))
            rec[m]['cov2'].append(float(np.mean(np.abs(d[use]) <= 2.0 * sig)))
            rec[m]['cov2c'].append(float(np.mean(np.abs(d[use] - med_all) <= 2.0 * sig)))     # exploratory: scene offset removed
            rec[m]['keff'].append(N.effective_k(d[use], sig))
            rec[m]['n_px'].append(int(use.sum()))
            rec[m]['nu0'].append(float(info['nu0']) if info.get('nu0') is not None else np.nan)
    summary = {}
    for m in methods:
        r = {k: np.asarray(v, dtype=float) for k, v in rec[m].items()}
        summary[m] = {'mean_nll': float(r['nll'].mean()), 'mean_coverage_k2': float(r['cov2'].mean()),
                      'exploratory_mean_coverage_k2_after_removing_scene_median_offset': float(r['cov2c'].mean()),
                      'nu0_median': None if np.isnan(r['nu0']).all() else float(np.nanmedian(r['nu0'])),
                      'nu0_fraction_at_cap': None if np.isnan(r['nu0']).all() else float(np.mean(r['nu0'] >= nz['nu0_max'] * (1 - 1e-9))),
                      'coverage_k2_rep_range_2.5_97.5': [float(np.percentile(r['cov2'], 2.5)), float(np.percentile(r['cov2'], 97.5))],
                      'gap_pp_vs_nominal': 100 * float(r['cov2'].mean() - cfg['a5']['coverage_nominal']),
                      'mean_k_eff': float(r['keff'].mean()), 'k_eff_rep_range_2.5_97.5': [float(np.percentile(r['keff'], 2.5)), float(np.percentile(r['keff'], 97.5))],
                      'mean_valid_px': float(r['n_px'].mean())}
    eligible = [m for m in nz['estimators']]
    ranked = sorted(eligible, key=lambda m: summary[m]['mean_nll'])
    best, second = ranked[0], ranked[1]
    diffs = np.asarray(rec[best]['nll']) - np.asarray(rec[second]['nll'])
    bs = cfg['statistics']['bootstrap']
    paired = paired_bootstrap_ci(diffs, bs['replicates'], bs['ci'], bs['seed'])
    contains_zero = paired['lo'] <= 0.0 <= paired['hi']
    winner = min((best, second), key=SIMPLICITY.index) if contains_zero else best
    return {'evidence': 'real', 'dataset': 'E6 12-date cache: old AOI (34 km from the slide), 10 m B04/B08, NDVI floor from wayanad config',
            'dihedral_terms': 'zero (no SR runs exist for E6): validates the s2_dates shrinkage only, n_pre = 3',
            'replicates': sel['replicates'], 'seed': sel['seed'], 'n_pool_dates': n_pool, 'window_px_10m': window,
            'n_strata': nz['n_strata'], 'dates': dates, 'first_three_draws': draws[:3], 'summary': summary,
            'exploratory_scene_median_offset_share_of_mean_d2': {'mean': float(np.mean(offset_share)), 'min': float(np.min(offset_share)),
                                                                  'max': float(np.max(offset_share))},
            'ranking_by_mean_nll': ranked, 'best': best, 'runner_up': second,
            'paired_nll_diff_best_minus_runner_up': paired, 'tie_rule_triggered': bool(contains_zero and winner != best),
            'tie_within_ci': bool(contains_zero), 'winner': winner,
            'note': 'replicates share the same 12 dates and the same pixels: they are not independent; the rep range is design-level '
                    'variability of the date draw, not a sampling CI'}


# ---- stage 3: Wayanad --------------------------------------------------------------------------------------------------------

def run_wayanad(cfg, wcfg, manifest, winner):
    from experiments.wayanad_evidence import config as C
    nz, ev, st = cfg['a5']['noise'], cfg['a5']['evaluation'], cfg['statistics']
    dates, post, pre = manifest['dates'], manifest['post'], manifest['pre']
    M = np.load(SCRATCH / 'mean.npy').astype(np.float64)
    V = np.load(SCRATCH / 'var.npy').astype(np.float64)
    C_ = np.load(SCRATCH / 'count.npy')
    stable, stable_counts = stable_mask_2p5m(wcfg, manifest, ev)
    full = (C_ == 8).all(axis=0) & np.isfinite(M).all(axis=0)
    mask = stable & full
    tile = ev['tile_px_2p5m']
    window = odd_window_px(nz['window_m'], 2.5)
    floor_var = nz['sigma_floor'] ** 2
    nominal = cfg['a5']['coverage_nominal']
    idx = {d: i for i, d in enumerate(dates)}
    methods = list(nz['estimators'])
    res = {'stable_pixels': {**stable_counts, 'valid_2p5m_all_dates': int(full.sum()), 'stable_and_valid_2p5m': int(mask.sum()),
                             'window_px_2p5m': window, 'tile_px_2p5m': tile}}
    tid, par = tile_ids(mask.shape, tile)

    def sigma_new(pre_ids, post_id, method):
        s, info = N.predictive_sigma_from_dates(M[pre_ids], V[pre_ids], V[post_id], method=method, n_strata=nz['n_strata'],
                                                window=window, valid=full, floor=floor_var,
                                                floor_rel_median=nz['floor_rel_median'], nu0_max=nz['nu0_max'])
        return s, info

    def sigma_old(pre_ids, post_id):
        _, pv = N.pool_moments(M[pre_ids], V[pre_ids], C_[pre_ids].astype(float), ddof=1)
        return N.sigma_v1(np.sqrt(pv), np.sqrt(V[post_id]))

    def stable_empirical(d, level, ok, n_pre_fold):
        """sigma from the RMS of d in the same NDVI stratum on the OTHER checkerboard parity (spatial cross-fit)."""
        labels, _ = N.quantile_strata(level, nz['n_strata'], ok)
        sig = np.full(d.shape, np.nan)
        for p in (0, 1):
            cal = ok & (par == p) & (labels >= 0)
            nb_ = int(labels.max()) + 1
            rms = np.sqrt(np.bincount(labels[cal], weights=d[cal] ** 2, minlength=nb_) / np.maximum(np.bincount(labels[cal], minlength=nb_), 1))
            tgt = ok & (par == 1 - p) & (labels >= 0)
            sig[tgt] = rms[labels[tgt]]
        return sig

    # ---------------- leave-one-date-out (headline) ----------------
    folds = {}
    az = {m: {} for m in methods + ['old_v1', 'stable_empirical']}
    az_c = {m: {} for m in methods + ['old_v1']}                       # exploratory: fold's stable median offset removed from d
    ratios, floored, dih_frac = {m: {} for m in methods}, {m: {} for m in methods}, {m: {} for m in methods}
    fold_info = {}
    nbk = int(tid.max()) + 1
    far = {m: {} for m in methods + ['old_v1']}                        # one-sided (gate-style S: pre_mean - new > 2 sigma) false-flag counts per block
    for h in pre:
        rem = [idx[p] for p in pre if p != h]
        hi = idx[h]
        d = M[hi] - M[rem].mean(axis=0)
        med_h = float(np.median(d[mask]))
        so = sigma_old(rem, hi)
        az['old_v1'][h] = az_of(d, so)
        az_c['old_v1'][h] = az_of(d - med_h, so)
        far['old_v1'][h] = (np.bincount(tid[mask & (-d > 2.0 * so)], minlength=nbk), np.bincount(tid[mask], minlength=nbk))
        fold_info[h] = {'held_out': h, 'remaining': [dates[i] for i in rem], 'n_pre_for_formula': len(rem), 'dof': len(rem) - 1,
                        'median_d': med_h, 'median_abs_d': float(np.median(np.abs(d[mask]))),
                        'exploratory_offset_share_of_mean_d2': med_h ** 2 / float(np.mean(d[mask] ** 2)),
                        'median_sigma_old': float(np.median(so[mask]))}
        dih = V[rem].mean(axis=0) / len(rem) + V[hi]
        for m in methods:
            sg, info = sigma_new(rem, hi, m)
            az[m][h] = az_of(d, sg)
            az_c[m][h] = az_of(d - med_h, sg)
            far[m][h] = (np.bincount(tid[mask & (-d > 2.0 * sg)], minlength=nbk), np.bincount(tid[mask], minlength=nbk))
            ratios[m][h] = float(np.median(sg[mask] / so[mask]))
            floored[m][h] = float(np.mean(sg[mask] <= nz['sigma_floor'] + 1e-15))
            dih_frac[m][h] = float(np.median(dih[mask] / sg[mask] ** 2))
            fold_info[h][f'median_sigma_{m}'] = float(np.median(sg[mask]))
            fold_info[h][f'nu0_{m}'] = info.get('nu0')
        level = M[rem].mean(axis=0)
        se = stable_empirical(d, level, mask, len(rem))
        az['stable_empirical'][h] = az_of(d, se)
    lodo = {}
    for m in methods + ['old_v1', 'stable_empirical']:
        extra = {}
        if m in methods:
            extra = {'median_sigma_new_over_sigma_old_by_fold': ratios[m],
                     'median_sigma_new_over_sigma_old_mean_of_folds': float(np.mean(list(ratios[m].values()))),
                     'fraction_sigma_floored_by_fold': floored[m],
                     'median_fraction_of_sigma2_from_dihedral_terms_by_fold': dih_frac[m]}
        if m in far:
            bs = st['bootstrap']
            num = sum(far[m][h][0] for h in pre).astype(float)
            den = sum(far[m][h][1] for h in pre).astype(float)
            pooled_far = ratio_bootstrap_ci(num, den, bs['replicates'], bs['ci'], bs['seed'])
            extra['one_sided_flag_rate_at_k2'] = {
                'definition': 'fraction of stable pixels with (mean of the other two dates - held-out date) > 2 sigma, i.e. the gate-style S = d > k sigma '
                              'on data with no real change (a false-flag rate; every flag is false by construction)',
                'pooled': pooled_far['estimate'], 'ci95': [pooled_far['lo'], pooled_far['hi']], 'flagged_px': pooled_far['numerator'],
                'n_px': pooled_far['denominator'],
                'by_fold': {h: float(far[m][h][0].sum() / far[m][h][1].sum()) for h in pre}}
        if m in az_c:
            cen = summarize(az_c[m], mask, tile, st, nominal, f'LODO centred {m}')
            extra['exploratory_after_removing_fold_median_offset'] = {
                'note': 'POST HOC diagnostic added after the headline miss was seen; not part of the pre-registered rule; sigma unchanged',
                'coverage_k2': cen['pooled'], 'k_eff': cen['k_eff']['value_all_stable'],
                'per_fold_coverage': {h: cen['folds'][h]['coverage_k2'] for h in pre}}
        lodo[m] = summarize(az[m], mask, tile, st, nominal, f'LODO {m}', extra)
    date_medians = {d: float(np.median(M[idx[d]][mask])) for d in dates}
    res['lodo'] = {'evidence': 'real', 'n_pre_for_formula': 2, 'dof_per_pixel': 1,
                   'definition': 'd = held-out pre date minus mean of the other two pre dates, both from the same 8-run per-date means; '
                                 'sigma from the other two dates + held-out date dihedral variance; stable pixels only',
                   'folds': fold_info, 'estimators': lodo, 'median_ndvi_on_stable_px_by_date': date_medians,
                   'independent_dates_note': 'three folds share the same three dates: at most 3 date effects, not independent'}
    # ---------------- pre-mean vs December (season-confounded) ----------------
    pre_ids, post_id = [idx[p] for p in pre], idx[post]
    d = M[pre_ids].mean(axis=0) - M[post_id]
    med = float(np.median(d[mask]))
    so = sigma_old(pre_ids, post_id)
    with np.load(C.cache(wcfg, ROOT) / 'step3_state.npz') as z:
        so_cached = np.sqrt(z['pre_std'] ** 2 + z['post_std'] ** 2)
    both = mask & np.isfinite(so) & np.isfinite(so_cached)
    dec = {'evidence': 'real', 'confounded_by_season': True, 'n_pre_for_formula': 3, 'dof_per_pixel': 2,
           'median_d_stable': med, 'median_abs_d_stable': float(np.median(np.abs(d[mask]))),
           'median_sigma_old_recomputed': float(np.median(so[mask])), 'median_sigma_old_cached_step4_definition': float(np.median(so_cached[both])),
           'sigma_old_recomputed_vs_cached_max_abs_diff': float(np.abs(so[both] - so_cached[both]).max()),
           'note': 'December NDVI on stable pixels is higher than the January pool, so d carries a systematic offset: this measures '
                   'bias more than noise calibration. The centred variant subtracts the stable median (diagnostic only).',
           'estimators': {}}
    az_d = {'old_v1': {'dec': az_of(d, so)}}
    az_dc = {'old_v1': {'dec': az_of(d - med, so)}}
    for m in methods:
        sg, info = sigma_new(pre_ids, post_id, m)
        az_d[m], az_dc[m] = {'dec': az_of(d, sg)}, {'dec': az_of(d - med, sg)}
        dec['estimators'][m] = {'median_sigma_new': float(np.median(sg[mask])), 'median_sigma_new_over_sigma_old': float(np.median(sg[mask] / so[mask])),
                                'nu0': info.get('nu0'), 'fraction_sigma_floored': float(np.mean(sg[mask] <= nz['sigma_floor'] + 1e-15))}
    for m in ['old_v1'] + methods:
        entry = summarize(az_d[m], mask, tile, st, nominal, f'Dec {m}')
        cen = summarize(az_dc[m], mask, tile, st, nominal, f'Dec centred {m}')
        dec['estimators'].setdefault(m, {})
        dec['estimators'][m].update({'coverage_k2': entry['pooled'], 'k_eff': entry['k_eff'],
                                     'centred_diagnostic_coverage_k2': cen['pooled'], 'centred_diagnostic_k_eff': cen['k_eff']['value_all_stable']})
    res['pre_mean_vs_december'] = dec
    # ---------------- headline / keep rule ----------------
    head = lodo[winner]['pooled']
    gap = head['gap_pp_vs_nominal']
    kept = abs(gap) <= 2.0
    old_head = lodo['old_v1']['pooled']
    diag = {'per_fold_coverage_new': {h: lodo[winner]['folds'][h]['coverage_k2'] for h in pre},
            'per_fold_median_d': {h: fold_info[h]['median_d'] for h in pre},
            'robust_scale_of_z_madn_new': lodo[winner]['robust_scale_of_z_madn'],
            'frac_abs_z_gt_3_new': lodo[winner]['frac_abs_z_gt_3'], 'frac_abs_z_gt_4_new': lodo[winner]['frac_abs_z_gt_4'],
            'k_eff_new': lodo[winner]['k_eff']['value_all_stable']}
    res['keep_rule'] = {'rule': 'pooled LODO coverage of |d| <= 2 sigma for the selected estimator within +/- 2 pp of 95.45 %, OR the gap reported with its cause',
                        'estimator': winner, 'coverage': head['coverage_k2'], 'ci95': head['ci95'], 'gap_pp': gap,
                        'kept': bool(kept), 'old_v1_coverage': old_head['coverage_k2'], 'old_v1_ci95': old_head['ci95'],
                        'diagnostics_for_cause': diag,
                        'cause_note': None if kept else 'MISS reported. Diagnostics above separate a scale error (robust scale of z far from 1) '
                                                        'from tail/heterogeneity or date-coherent effects (per-fold coverage and per-fold median d).'}
    return res


def write_report(out, path: Path):
    """Short markdown table generated from the result JSON (numbers are read, not retyped)."""
    m = out['measurements']
    sy, se, wy = m['synthetic_bias'], m['selection'], m['wayanad']
    L, D, kr = wy['lodo'], wy['pre_mean_vs_december'], wy['keep_rule']
    pct = lambda x: f'{100 * x:.1f} %'
    ci = lambda a: f'[{100 * a[0]:.1f}, {100 * a[1]:.1f}]'
    lines = ['# X5 noise-model report (agent A5)', '',
             f"Status **{out['status']}** (exit {0 if out['status'] == 'PASS' else 2}). Selected estimator (pre-registered rule, E6 only): "
             f"**{out['selected_estimator']}**. Config sha256 `{out['config_sha256_of_exceptional_yaml'][:16]}...`. "
             'Decision rule and failure modes: `docs/adr-x5-noise-model.md`. Evidence labels are per table.', '',
             '## 1. Wayanad leave-one-date-out (real; headline, n_pre = 2 for the formula, 1 dof)', '',
             f"Stable pixels: {wy['stable_pixels']['stable_and_valid_2p5m']:,} 2.5 m px per fold ({wy['stable_pixels']['stable_10m']:,} 10 m px "
             'outside the 1500 m disks, footprint and parent). Coverage of |d| <= 2 sigma, pooled over 3 folds; 95 % block-bootstrap CI (320 m blocks); '
             'nominal 95.45 %. Three folds share three dates: they are not independent.', '',
             '| sigma | coverage k = 2 [CI] | n px (3 folds) | k_eff [CI] | one-sided flag rate S at k = 2 [CI] | median sigma / v1 (mean of folds) | coverage after removing fold offset (post hoc) |',
             '|---|---|---|---|---|---|---|']
    for name in ['old_v1', 'raw', 'window', 'eb_stratified', 'eb_window', 'stable_empirical']:
        e = L['estimators'][name]
        p, k = e['pooled'], e['k_eff']
        far = e.get('one_sided_flag_rate_at_k2')
        c = e.get('exploratory_after_removing_fold_median_offset')
        ratio = e.get('median_sigma_new_over_sigma_old_mean_of_folds')
        lines.append(f"| {name} | {pct(p['coverage_k2'])} {ci(p['ci95'])} | {int(p['n_px']):,} | {k['value_all_stable']:.2f} [{k['ci95_block_bootstrap'][0]:.2f}, {k['ci95_block_bootstrap'][1]:.2f}] | "
                     f"{'' if far is None else pct(far['pooled']) + ' ' + ci(far['ci95'])} | {'' if ratio is None else f'{ratio:.2f}'} | "
                     f"{'' if c is None else pct(c['coverage_k2']['coverage_k2'])} |")
    lines += ['', f"Keep rule (coverage within +/- 2 pp of 95.45 %, else report the cause): **{'KEPT' if kr['kept'] else 'MISSED'}** "
              f"for {kr['estimator']}: {pct(kr['coverage'])} {ci(kr['ci95'])}, gap {kr['gap_pp']:+.1f} pp; v1: {pct(kr['old_v1_coverage'])} {ci(kr['old_v1_ci95'])}.", '',
              'Per fold (held-out date: coverage of the selected estimator, median d, share of mean d^2 that is a scene-wide offset):', '']
    for h, f in L['folds'].items():
        lines.append(f"- {h}: coverage {pct(L['estimators'][kr['estimator']]['folds'][h]['coverage_k2'])}, median d {f['median_d']:+.3f}, "
                     f"offset share {pct(f['exploratory_offset_share_of_mean_d2'])}")
    lines += ['', f"Median NDVI on stable pixels by date: " + ', '.join(f'{d} {v:.3f}' for d, v in L['median_ndvi_on_stable_px_by_date'].items()) + '.', '',
              '## 2. Pre-mean vs December (real; n_pre = 3; CONFOUNDED by season, measures bias as much as noise)', '',
              '| sigma | coverage k = 2 [CI] | k_eff [CI] | median sigma / v1 | coverage centred (post hoc) |', '|---|---|---|---|---|']
    for name in ['old_v1', 'raw', 'window', 'eb_stratified', 'eb_window']:
        e = D['estimators'][name]
        r = e.get('median_sigma_new_over_sigma_old')
        lines.append(f"| {name} | {pct(e['coverage_k2']['coverage_k2'])} {ci(e['coverage_k2']['ci95'])} | {e['k_eff']['value_all_stable']:.2f} "
                     f"[{e['k_eff']['ci95_block_bootstrap'][0]:.2f}, {e['k_eff']['ci95_block_bootstrap'][1]:.2f}] | {'' if r is None else f'{r:.2f}'} | "
                     f"{pct(e['centred_diagnostic_coverage_k2']['coverage_k2'])} |")
    lines += ['', f"Stable median d = {D['median_d_stable']:+.3f} NDVI (December is greener). A coverage near nominal here is NOT evidence of calibration: "
              'the offset happens to be of the order of the between-date spread.', '',
              '## 3. Estimator selection (real, E6 12-date cache, other AOI, 10 m, 3 pool dates + 1 held-out, '
              f"{se['replicates']} seeded draws; no Wayanad pixel)", '',
              '| sigma | mean held-out NLL | coverage k = 2 (mean; draw range 2.5-97.5 %) | mean k_eff | nu0 median (share at cap) |', '|---|---|---|---|---|']
    for name in ['raw', 'window', 'eb_stratified', 'eb_window', 'no_inflation_reference']:
        s = se['summary'][name]
        nu = '' if s['nu0_median'] is None else f"{s['nu0_median']:.3g} ({pct(s['nu0_fraction_at_cap'])})"
        lines.append(f"| {name} | {s['mean_nll']:.3f} | {pct(s['mean_coverage_k2'])} ({pct(s['coverage_k2_rep_range_2.5_97.5'][0])} - {pct(s['coverage_k2_rep_range_2.5_97.5'][1])}) | {s['mean_k_eff']:.2f} | {nu} |")
    p = se['paired_nll_diff_best_minus_runner_up']
    lines += ['', f"Winner {se['winner']}; paired NLL difference best - runner-up ({se['best']} - {se['runner_up']}) {p['mean']:.3f} [{p['lo']:.3f}, {p['hi']:.3f}] "
              f"({'CI contains 0: tie rule applied' if se['tie_within_ci'] else 'CI excludes 0: no tie rule needed'}).", '',
              '## 4. Synthetic simulation (SYNTHETIC; 120,000 pixels, known truth, n_pre = 3, 8 runs)', '',
              '| sigma | variance / true predictive variance | coverage k = 2 | k_eff |', '|---|---|---|---|',
              f"| v1 | {sy['old_v1']['variance_ratio_to_truth']:.3f} (SD ratio {sy['old_v1']['sd_ratio_to_truth']:.3f}) | {pct(sy['old_v1']['coverage_at_k2'])} | {sy['old_v1']['k_eff']:.2f} |",
              f"| raw | {sy['raw']['variance_ratio_to_truth']:.3f} | {pct(sy['raw']['coverage_at_k2'])} | {sy['raw']['k_eff']:.2f} |",
              f"| eb_stratified | {sy['eb_stratified']['variance_ratio_to_truth']:.3f} | {pct(sy['eb_stratified']['coverage_at_k2'])} | {sy['eb_stratified']['k_eff']:.2f} |", '',
              f"Oracle (true sigma) coverage {pct(sy['oracle_true_sigma_coverage_at_k2'])}.", '',
              '## Limitations', ''] + [f'- {x}' for x in out['limitations']]
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--rebuild', action='store_true', help='re-run the SR dihedral pass even if scratch_a5 is populated')
    args = ap.parse_args()
    started = datetime.now(timezone.utc).isoformat()
    cfg, cfg_hash = load_config()
    from experiments.wayanad_evidence import config as C
    wcfg, wroot, wconfig_hash = C.load(str(ROOT / 'configs/wayanad_evidence.yaml'))
    np.random.seed(cfg['seed'])
    try:
        stage = {}
        stage['synthetic_bias'] = synthetic_bias(cfg)
        manifest = build_date_stats(wcfg, wroot, wconfig_hash, args.rebuild)
        stage['dihedral_stats_build'] = {k: manifest[k] for k in ('dates', 'pre', 'post', 'tiles_per_date', 'runs_per_date', 'array_sha256',
                                                                   'consistency_vs_cached_pooled_state', 'build_seconds', 'device', 'seed')}
        selection = run_selection(cfg, wcfg)                      # NO Wayanad d/sigma is computed before this line
        stage['selection'] = selection
        stage['wayanad'] = run_wayanad(cfg, wcfg, manifest, selection['winner'])
        kept = stage['wayanad']['keep_rule']['kept']
        result = {'status': 'PASS' if kept else 'FAIL', 'evidence': 'real',
                  'evidence_labels': {'synthetic_bias': 'synthetic', 'selection': 'real (E6, other AOI, 10 m)', 'wayanad.lodo': 'real',
                                      'wayanad.pre_mean_vs_december': 'real, confounded by season'},
                  'selected_estimator': selection['winner'], 'measurements': stage,
                  'adr': 'docs/adr-x5-noise-model.md', 'adr_sha256': sha256_file(ROOT / 'docs/adr-x5-noise-model.md'),
                  'limitations': ['three pre dates only: leave-one-date-out has n_pre = 2 (1 dof) and 3 non-independent folds',
                                  'block-bootstrap CIs resample space, not dates: they understate date-level uncertainty',
                                  'December comparison is confounded by season',
                                  'stable pixels lie inside the SR crop around the slide, outside the footprint and 1500 m disks',
                                  'coverage on stable pixels says nothing about power on real change', 'CPU macOS run; SR determinism verified by step3 only on this platform']}
    except Blocked as exc:
        result = {'status': 'BLOCKED', 'evidence': exc.evidence, 'reason': str(exc)}
    out = finalize_result('x5', result, cfg_hash, started)
    out['environment'] = environment_record({})
    out['config'] = {'file': CONFIG, 'seed': cfg['seed'], 'a5': cfg['a5'], 'statistics': cfg['statistics']}
    out['config_sha256_of_exceptional_yaml'] = cfg_hash
    write_json(ROOT / cfg['results_dir'] / 'x5.json', clean(out))
    if 'measurements' in out:
        write_report(clean(out), ROOT / cfg['results_dir'] / 'x5_REPORT.md')
    print(json.dumps({k: clean(out[k]) for k in ('status', 'selected_estimator') if k in out}, indent=2))
    if 'measurements' in out:
        print(json.dumps(clean(out['measurements']['wayanad']['keep_rule']), indent=2))
    raise SystemExit(0 if out['status'] == 'PASS' else 2)


if __name__ == '__main__':
    main()
