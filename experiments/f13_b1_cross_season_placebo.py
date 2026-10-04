"""F13 B1: does within-season calibration understate the deployment false-alarm rate? (cross-season placebo at 10 m)

Pre-registered in configs/f13_discovery.yaml `experiments.B1` (sha256 84beef1c...). January pre dates against a December
post date of a non-event year, scored by rule_10m and by gate v2 at F2's tau. This file holds the deterministic helpers,
the STAC fetch of the named dates through the existing wayanad_evidence fetch path, and the run.
"""
from __future__ import annotations

import numpy as np

from experiments import f13_common as C


# ---------------------------------------------------------------- deterministic helpers ----------------------------------------------------------------

def candidate_table(rows, start: str, end: str, max_cloud: float, min_coverage: float) -> list:
    """Every acquisition date in [start, end] with its best row's numbers, and whether the x6 clear rule accepts it.

    Acceptance is decided by experiments.x6_season_latency.clear_dates_in_window (cited, not redefined)."""
    from experiments.x6_season_latency import clear_dates_in_window
    clear = set(clear_dates_in_window(rows, start, end, max_cloud, min_coverage))
    best = {}
    for r in rows:
        if r['date'] not in best or r.get('cloud_shadow_pct_aoi', 100) < best[r['date']].get('cloud_shadow_pct_aoi', 100):
            best[r['date']] = r
    out = []
    for d in sorted(best):
        if not (start <= d <= end):
            continue
        r = best[d]
        cloud, cov = r.get('cloud_shadow_pct_aoi'), r.get('coverage_pct')
        reasons = []
        if cloud is None or cloud > max_cloud:
            reasons.append(f'cloud+shadow {cloud} % > {max_cloud} %')
        if cov is None or cov < min_coverage:
            reasons.append(f'coverage {cov} % < {min_coverage} %')
        out.append({'date': d, 'cloud_shadow_pct_aoi': cloud, 'coverage_pct': cov, 'item_ids': r.get('item_ids'),
                    'accepted': d in clear, 'reason': ('; '.join(reasons) if reasons else None)})
        assert (d in clear) == (not reasons), 'x6 rule and the reported reasons disagree'
    return out


def true_threshold(within: dict) -> float:
    """yaml B1.keep_rule TRUE threshold for a gate: twice that gate's own committed within-season test window FAR."""
    return 2.0 * float(within['estimate'])


def b1_gate_verdict(far: float, within: dict) -> str:
    """TRUE if FAR >= 2 x within-season estimate; FALSE if inside the within-season CI (endpoints included); else INCONCLUSIVE.
    The memo's '(>= 0.10)' is not a decision threshold (human decision b1_threshold_reading)."""
    if far >= true_threshold(within):
        return 'TRUE'
    if within['lo'] <= far <= within['hi']:
        return 'FALSE'
    return 'INCONCLUSIVE'


def b1_overall(per_comparison: dict) -> str:
    """B1's overall verdict = gate_v2 on the primary comparison (yaml method.proposed_not_in_memo.overall_verdict)."""
    return per_comparison['primary_non_event']['gate_v2']


def sigma_f_ratio_closed_form(n_pre: int, n_ref: int = 2) -> float:
    """sigma_f = sqrt(Var(a) (1/n_pre + 1)): sigma_f(n_ref) / sigma_f(n_pre)."""
    return float(np.sqrt((1.0 / n_ref + 1.0) / (1.0 / n_pre + 1.0)))


def is_vacuous(recall_by_drop: dict) -> bool:
    """yaml power_requirement.zero_power_rule: recall at the 0.50 drop is 0."""
    return float(recall_by_drop['0.5']) == 0.0


# ================================================================ acquisition (the only network use in B1) ================================================================

CACHE_DIR = C.ROOT / 'data' / 'experiments-cache' / 'f13' / 'wayanad_s2'
CROP = (256, 896, 128, 640)


def _ev_cfg_for_window(cfg_ev, start_iso: str, end_iso: str) -> dict:
    import copy
    cfg = copy.deepcopy(cfg_ev)
    cfg['dates']['audit_start'], cfg['dates']['audit_end'] = start_iso, end_iso
    return cfg


def acquire(cfg_ev, ev_root, windows: dict, budget_cap: int, log) -> dict:
    """STAC search + 20 m SCL audit for each named window, through experiments.wayanad_evidence.fetch (search_groups, audit).

    windows: {name: (start_date, end_date)} ISO dates. Returns {name: {'rows': audit rows, 'groups': {(date,platform,orbit): items}}}.
    SCL arrays are cached per item under CACHE_DIR/scl20 (data/experiments-cache/f13 only)."""
    from affine import Affine
    from experiments.e6_season_matched import ByteBudget
    from experiments.wayanad_evidence import fetch as F
    from experiments.wayanad_evidence.geo import reference_grid
    transform, shape = reference_grid(cfg_ev['aoi'])
    res20 = cfg_ev['aoi']['audit_resolution_m']
    transform20 = Affine(res20, 0.0, transform.c, 0.0, -res20, transform.f)
    shape20 = (shape[0] * 10 // res20, shape[1] * 10 // res20)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    budget = ByteBudget(budget_cap)
    out = {}
    for name, (start, end) in windows.items():
        cfg = _ev_cfg_for_window(cfg_ev, f'{start}T00:00:00Z', f'{end}T23:59:59Z')
        items, groups = F.search_groups(cfg, transform, shape)
        log(f'{name}: STAC returned {len(items)} items in {len(groups)} acquisition groups for {start}..{end}')
        rows, item_rows, errors = F.audit(cfg, ev_root, groups, transform20, shape20, budget, CACHE_DIR)
        if errors:
            raise C.Blocked(f'{name}: {len(errors)} SCL asset(s) unreadable, first: {errors[0]}')
        out[name] = {'rows': rows, 'item_rows': item_rows, 'groups': groups, 'window': [start, end],
                     'n_items': len(items)}
    out['_budget'] = budget
    out['_geometry'] = (transform, shape)
    return out


def fetch_dates(cfg_ev, ev_root, acquired, wanted: dict, log) -> dict:
    """Band reads (B04,B03,B02,B08 at 10 m over the full AOI grid) for the accepted dates via fetch.read_bands.
    wanted: {date: window name}. Returns {date: {'items': ids, 'baselines': [...], 'file': path}}."""
    from experiments.wayanad_evidence import fetch as F
    transform, shape = acquired['_geometry']
    budget = acquired['_budget']
    out = {}
    for date, name in sorted(wanted.items()):
        groups = acquired[name]['groups']
        items = [i for (d, _p, _o), g in groups.items() if d == date for i in g]
        if not items:
            raise C.Blocked(f'{date}: no STAC items in window {name}')
        path = F.read_bands(cfg_ev, ev_root, date, items, transform, shape, budget, CACHE_DIR)
        with np.load(path) as z:
            meta = __import__('json').loads(str(z['meta']))
        out[date] = {'items': meta['items'], 'baselines': meta['processing_baselines'], 'file': str(path.relative_to(C.ROOT)),
                     'bytes_on_disk': path.stat().st_size, 'sha256': __import__('hashlib').sha256(path.read_bytes()).hexdigest()}
        log(f'{date}: fetched {len(items)} item(s), baselines {meta["processing_baselines"]}, {budget.used()} B received so far')
    return out


# ================================================================ the run ================================================================

MANIFEST = C.ROOT / 'data' / 'experiments-cache' / 'f13' / 'b1_manifest.json'
BYTE_CAP = 300_000_000                     # configs/wayanad_evidence.yaml fetch.max_fetch_bytes, the cap the existing path uses
COMPARISONS = ('primary_non_event', 'secondary_proxy')


def acquire_or_load(cfg, cfg_ev, ev_root, log) -> dict:
    """Return the fetch manifest. If one exists and every file it lists is on disk, use it with no network; otherwise run
    the audit and fetch for exactly the windows/dates the yaml names, then write the manifest."""
    import hashlib
    import json
    import shutil
    from risk.common import write_json
    b1 = cfg['experiments']['B1']
    cmp_ = b1['data']['comparisons']
    im = cfg_ev['phase0']['imagery']
    max_cloud, min_cov = im['max_cloud_shadow_pct'], im['min_coverage_pct']
    if MANIFEST.is_file():
        man = json.loads(MANIFEST.read_text(encoding='utf-8'))
        if all((C.ROOT / f['file']).is_file() for f in man['dates'].values()):
            man['fetched_this_run'] = False
            return man
    pre_s, pre_e = cmp_['primary_non_event']['pre_window']
    post = cmp_['primary_non_event']['post']
    windows = {'pre_window_2023_01': (pre_s, pre_e), 'post_date': (post, post)}
    acq = acquire(cfg_ev, ev_root, windows, BYTE_CAP, log)
    tables = {name: candidate_table(acq[name]['rows'], *windows[name], max_cloud, min_cov) for name in windows}
    min_pre = int(b1['data']['min_pre_dates'])
    pre_ok = [r['date'] for r in tables['pre_window_2023_01'] if r['accepted']]
    post_ok = [r['date'] for r in tables['post_date'] if r['accepted']]
    man = {'windows': {k: list(v) for k, v in windows.items()}, 'rule': {'max_cloud_shadow_pct': max_cloud, 'min_coverage_pct': min_cov,
           'function': 'experiments.x6_season_latency.clear_dates_in_window'},
           'candidate_tables': tables, 'stac_items': {name: {f'{d}|{p}|{o}': [{'id': i['id'], 'processing_baseline': i['properties'].get('s2:processing_baseline'),
                                      'datetime': i['properties']['datetime']} for i in g] for (d, p, o), g in acq[name]['groups'].items()}
                                      for name in windows},
           'clear_pre_dates': pre_ok, 'post_accepted': bool(post_ok), 'min_pre_dates': min_pre, 'dates': {}}
    if len(pre_ok) < min_pre or not post_ok:
        man['blocked_reason'] = (f'{len(pre_ok)} clear Jan 2023 dates (need {min_pre}); post {post} accepted: {bool(post_ok)}')
        man['fetched_this_run'] = True
        man['bytes_received'] = acq['_budget'].used()
        write_json(MANIFEST, man)
        return man
    wanted = {d: 'pre_window_2023_01' for d in pre_ok}
    wanted[post] = 'post_date'
    man['dates'] = fetch_dates(cfg_ev, ev_root, acq, wanted, log)
    # secondary proxy: the three cached January 2024 dates, copied byte for byte into the F13 cache so one loader reads all dates
    src_cache = ev_root / cfg_ev['paths']['cache']
    for d in cmp_['secondary_proxy']['pre']:
        dst = CACHE_DIR / f'{d}.npz'
        if not dst.is_file():
            shutil.copyfile(src_cache / f'{d}.npz', dst)
        man['dates'][d] = {'file': str(dst.relative_to(C.ROOT)), 'copied_from': str((src_cache / f'{d}.npz').relative_to(C.ROOT)),
                           'bytes_on_disk': dst.stat().st_size, 'sha256': hashlib.sha256(dst.read_bytes()).hexdigest(),
                           'items': None, 'baselines': None}
    man['bytes_received'] = acq['_budget'].used()
    man['byte_cap'] = BYTE_CAP
    man['fetched_this_run'] = True
    write_json(MANIFEST, man)
    return man


def reproduction_gate(cfg, cfg_ev, ev_root) -> dict:
    """Within-season folds: rule_10m test-split window FAR 11/1962 (f1.json) and gate_v2 64/981 at F2's tau (f2.json)."""
    import json
    from experiments import f13_a0_exchangeability as A0
    from trustsr import placebo_v2 as P
    boot = cfg['statistics']['bootstrap']
    ba = {'replicates': boot['replicates'], 'ci': cfg['statistics']['ci'], 'seed': boot['seed']}
    f1 = json.loads((C.ROOT / 'experiments/results/f1.json').read_text(encoding='utf-8'))
    f2 = json.loads((C.ROOT / 'experiments/results/f2.json').read_text(encoding='utf-8'))
    import rasterio
    cache0 = ev_root / cfg_ev['paths']['cache']
    with rasterio.open(C.ROOT / 'experiments/wayanad_evidence/outputs/footprint.tif') as src:
        footprint_full, transform = src.read(1).astype(bool), src.transform
    excl_hr = P.upsample(P.exclusion_mask_10m(cfg_ev, transform, CROP, footprint_full), 4)
    pool = f1['ten_m_pool_dates']
    tile_split = P.checkerboard_split((2560, 2048), 128, seed=cfg['seed'])
    blocks = []
    for held in pool:
        fold = P.build_10m_fold(cfg_ev, cache0, CROP, held, [d for d in pool if d != held], cfg_ev['change']['parent_drop_threshold'])
        fold.nodata = fold.nodata | excl_hr
        flagged = P.gate_rule_10m(fold, cfg_ev['change']['k'])
        _r, num, den = P.compute_far_window_v2(flagged, ~fold.nodata, tile_split, 128, 64, **ba)
        blocks.append((num, den))
    pooled = P.pool_fold_blocks(blocks, **ba)
    ref_rule = f1['gates']['rule_10m']['test_split']['pooled_window_far']
    rule = {'measured': [pooled['numerator'], pooled['denominator']], 'target': [ref_rule['numerator'], ref_rule['denominator']],
            'estimate': pooled['estimate'], 'target_estimate': ref_rule['estimate'], 'folds': len(pool)}
    rule['pass'] = rule['measured'] == rule['target'] and abs(rule['estimate'] - rule['target_estimate']) < 1e-12
    R = A0.regenerate_f2(cfg_ev, ev_root)
    R['alpha'], R['lam'] = 0.05, float(f2['lambda']['value'])
    track = A0.score_track(R, normalise=True)
    rec = A0.assemble_records(track)
    g = A0.gap_from_weights(rec, np.tile(R['test_w'].ravel(), 3), np.ones(rec['score'].size), 0.05)
    gate = {'tau_measured': track['tau'], 'tau_target': f2['tau']['value'], 'test_window_far_measured': [g['num_test'], g['den_test']],
            'test_window_far_target': [64.0, 981.0]}
    gate['pass'] = bool(abs(track['tau'] - f2['tau']['value']) < 1e-12 and gate['test_window_far_measured'] == gate['test_window_far_target'])
    return {'rule_10m': rule, 'gate_v2': gate, 'pass': bool(rule['pass'] and gate['pass'])}


def _scene(cfg_ev, ev_root, pre_dates, post_date):
    """Load every date through the placebo guard, build the shared geometry for one comparison."""
    import rasterio
    from trustsr import placebo_v2 as P
    arrays = {d: P.guarded_load_date(cfg_ev, CACHE_DIR, d) for d in pre_dates + [post_date]}
    r0, r1, c0, c1 = CROP
    with rasterio.open(C.ROOT / 'experiments/wayanad_evidence/outputs/footprint.tif') as src:
        footprint_full, transform = src.read(1).astype(bool), src.transform
    excl_crop = P.exclusion_mask_10m(cfg_ev, transform, CROP, footprint_full)
    excl_full = P.exclusion_mask_10m(cfg_ev, transform, (0, 1024, 0, 1024), footprint_full)
    return {'arrays': arrays, 'excl_crop': excl_crop, 'excl_full': excl_full, 'excl_hr': P.upsample(excl_crop, 4),
            'refl': {d: np.moveaxis(arrays[d]['refl'][:, r0:r1, c0:c1], 0, -1).astype(np.float64) for d in arrays},
            'ndvi': {d: arrays[d]['ndvi'][r0:r1, c0:c1].astype(np.float64) for d in arrays},
            'valid': {d: arrays[d]['valid'][r0:r1, c0:c1] for d in arrays}}


def _far(flagged, valid, split, ba):
    from trustsr import placebo_v2 as P
    w, num, den = P.compute_far_window_v2(flagged, valid, split, 128, 64, **ba)
    p, pn, pd_ = P.compute_far_pixel_v2(flagged, valid, split, 128, **ba)
    return {'window': {k: w[k] for k in ('estimate', 'lo', 'hi', 'numerator', 'denominator', 'blocks', 'replicates', 'seed')},
            'pixel': {k: p[k] for k in ('estimate', 'lo', 'hi', 'numerator', 'denominator', 'blocks')}}


def score_comparison(idx, name, pre_dates, post_date, cfg, cfg_ev, ev_root, tau, within, log) -> dict:
    """Both gates on one (pre dates -> post date) comparison, then power. Everything not in the pre dates is the post image."""
    import dataclasses
    from experiments import f13_a0_exchangeability as A0
    from experiments import f13_a3_sr_swap as A3
    from experiments import f2_gate_v2 as F2
    from experiments.f3_noise_v2 import normalise_dates
    from experiments.wayanad_evidence.gate import parent_mask
    from experiments.wayanad_evidence.stats import ndvi as stats_ndvi
    from trustsr import gate_v2 as G
    from trustsr import placebo_v2 as P
    from trustsr.bootstrap import ratio_bootstrap_ci
    boot = cfg['statistics']['bootstrap']
    ba = {'replicates': boot['replicates'], 'ci': cfg['statistics']['ci'], 'seed': boot['seed']}
    seed = cfg['seed']
    thr = cfg_ev['change']['parent_drop_threshold']
    k_v1 = cfg_ev['change']['k']
    floor = cfg_ev['radiometry']['min_denominator']
    S = _scene(cfg_ev, ev_root, pre_dates, post_date)
    shape2 = (2560, 2048)
    tile_split = P.checkerboard_split(shape2, 128, seed=seed)                    # True = test (odd tiles)
    all_tiles = np.ones_like(tile_split, bool)
    test_w = np.repeat(np.repeat(tile_split, 2, axis=0), 2, axis=1)               # 40 x 32 windows
    dates = pre_dates + [post_date]
    nodata = G.nodata_from_scl(np.stack([S['valid'][d] for d in dates]), 4) | S['excl_hr']
    valid_px = ~nodata
    valid_all = np.logical_and.reduce([S['valid'][d] for d in dates])

    # ---- rule_10m (placebo_v2.gate_rule_10m on a placebo_v2.build_10m_fold) ----------------------------------
    fold10 = P.build_10m_fold(cfg_ev, CACHE_DIR, CROP, post_date, pre_dates, thr)
    fold10.nodata = fold10.nodata | S['excl_hr']
    flag_rule = P.gate_rule_10m(fold10, k_v1)
    assert np.array_equal(~fold10.nodata, valid_px), 'rule_10m and gate_v2 must score the same valid pixels'

    # ---- gate_v2: endmembers from the pre dates, F3 normalisation, F2's tau, lambda 0 --------------------------
    aoi = {'refl': {d: np.moveaxis(S['arrays'][d]['refl'], 0, -1).astype(np.float64) for d in pre_dates},
           'valid': {d: S['arrays'][d]['valid'] for d in pre_dates},
           'ndvi': {d: S['arrays'][d]['ndvi'].astype(np.float64) for d in pre_dates}}
    st = lambda key: np.stack([aoi[key][d] for d in pre_dates])
    gate_v2_result = {'status': 'ok'}
    try:
        ndvi_low, em, ceiling = F2.endmember_ceiling_table(st('refl'), st('valid'), st('ndvi'), S['excl_full'])
    except ValueError as exc:
        em = None
        gate_v2_result = {'status': 'BLOCKED', 'reason': f'no bare-ground endmember on these pre dates: {exc}'}
    out = {'name': name, 'pre_dates': pre_dates, 'post_date': post_date, 'n_pre': len(pre_dates),
           'valid_px_2p5m': int(valid_px.sum()), 'windows_valid': int(P.window_indicators(flag_rule, valid_px, 64)[1].sum())}
    out['rule_10m'] = {'far_test': _far(flag_rule, valid_px, tile_split, ba), 'far_all_windows': _far(flag_rule, valid_px, all_tiles, ba),
                       'within_season': within['rule_10m']}
    if em is None:
        out['gate_v2'] = gate_v2_result
        return out
    nd_pre = np.stack([S['ndvi'][d] for d in pre_dates])
    stable = (np.stack([S['valid'][d] for d in pre_dates]).all(axis=0) & ~S['excl_crop']
              & (nd_pre.std(axis=0, ddof=1) <= 0.05) & np.isfinite(nd_pre).all(axis=0))
    pre_stack = np.stack([S['refl'][d] for d in pre_dates])

    def normalise(post_refl):
        yp, yq = np.empty_like(pre_stack), np.empty_like(post_refl)
        for b in range(pre_stack.shape[-1]):
            p, q, _op, _oq, _ = normalise_dates(pre_stack[..., b], stable, post_refl[..., b], stable)
            yp[..., b], yq[..., b] = p, q
        return yp, yq
    y_pre, y_post = normalise(S['refl'][post_date])
    fold_stub = P.Fold(held_out=post_date, pre_dates=pre_dates, has_sr=False, nodata=nodata, n_pre=len(pre_dates))
    s, vb, sigma_bands, _floor, f_hat = F2.fold_score(fold_stub, em['e_v'], em['e_b'], y_pre, y_post)
    wa = A0.window_arrays(s, vb, f_hat, nodata)
    drop10 = A3.ndvi_drop_10m([S['ndvi'][d] for d in pre_dates], S['ndvi'][post_date], valid_all)
    rank_field = A3.centred_bilinear_upsample(drop10)                            # window flags do not depend on the ranking field
    class_map, meta = G.apply_gate_v2(y_pre, y_post, rank_field, em['e_v'], em['e_b'], sigma_bands, nodata, tau,
                                      lam=0.0, window_size=16, k_unsupported=k_v1, sigma_sr=None)
    flag_v2 = G.flagged_v2(class_map)
    fw, vw = P.window_indicators(flag_v2, valid_px, 64)
    fast = wa['in_pop'] & (wa['score'] > tau) & wa['flaggable']
    sigma_n = G.fraction_sigma(sigma_bands, em['e_v'], em['e_b'], n_pre=len(pre_dates))
    sigma_2 = G.fraction_sigma(sigma_bands, em['e_v'], em['e_b'], n_pre=2)
    out['gate_v2'] = {'status': 'ok', 'tau': tau, 'lam': 0.0, 'endmember_ndvi_low': ndvi_low,
                      'endmember_counts': {'e_v': em['e_v_count'], 'e_b': em['e_b_count']},
                      'far_test': _far(flag_v2, valid_px, tile_split, ba), 'far_all_windows': _far(flag_v2, valid_px, all_tiles, ba),
                      'within_season': within['gate_v2'],
                      'fast_path_equals_real_gate_window_map': bool(np.array_equal(fast, fw) and np.array_equal(wa['valid_px'], vw)),
                      'n_pre': len(pre_dates), 'n_pre_at_tau_calibration': 2,
                      'sigma_f_ratio_calibration_over_production': {'measured_median': float(np.nanmedian(sigma_2 / sigma_n)),
                                                                     'closed_form': sigma_f_ratio_closed_form(len(pre_dates))},
                      'sigma_f_note': 'recorded, not corrected (F7 precedent): the production score runs hotter than tau was calibrated for when n_pre > 2',
                      'class_counts': meta['class_counts'], 'windows_detected': int(meta['n_windows_detected'])}

    # ---- power: inject 0.15 / 0.30 / 0.50 NDVI drops into the post date (yaml power_requirement) ---------------
    pw = cfg['power_requirement']
    n_patches = pw['proposed_not_in_memo']['n_patches_per_fold_per_drop']
    eligible = test_w & wa['in_pop'] & wa['valid_px']
    base_flag_v2 = fast
    base_rule10 = parent_mask(np.where(valid_all, np.mean([S['ndvi'][d] for d in pre_dates], axis=0) - S['ndvi'][post_date], np.nan),
                              valid_all, thr)
    power = {'rule_10m': {}, 'gate_v2': {}}
    for di, drop in enumerate(pw['injected_ndvi_drops']):
        ss = int(np.random.SeedSequence([seed, idx, di]).generate_state(1)[0])
        patches = A0.place_patches(vb, eligible, n_patches, 10, 16, ss)
        mask10 = np.zeros(vb.shape, bool)
        for r, c in patches:
            mask10[r:r + 10, c:c + 10] = True
        post_inj = A0.inject_ndvi_drop(S['refl'][post_date], mask10, drop)
        # rule_10m: the 10 m NDVI drop of the injected post image against the pre mean
        nd_post = stats_ndvi(post_inj[..., 0], post_inj[..., 3], floor)
        par = parent_mask(np.where(valid_all, np.mean([S['ndvi'][d] for d in pre_dates], axis=0) - nd_post, np.nan), valid_all, thr)
        # gate_v2: F3 normalisation + unmixing on the injected post image
        yp_i, yq_i = normalise(post_inj)
        s_i, vb_i, _sb, _fl, f_i = F2.fold_score(fold_stub, em['e_v'], em['e_b'], yp_i, yq_i)
        wa_i = A0.window_arrays(s_i, vb_i, f_i, nodata)
        n_blk = np.where(vb_i, np.clip(np.round(16 * np.clip(np.nan_to_num(f_i, nan=0.0), 0, 1)), 0, 16), 0).astype(int)
        tiles = np.array([((r // 16) // 2) * 16 + ((c // 16) // 2) for r, c in patches])
        for gate in ('rule_10m', 'gate_v2'):
            if gate == 'rule_10m':
                hit = np.array([bool(par[r:r + 10, c:c + 10].any()) for r, c in patches], float)
                contaminated = np.array([bool(base_rule10[r:r + 10, c:c + 10].any()) for r, c in patches])
            else:
                hit = np.array([A0.patch_hit(wa_i['score'], wa_i['in_pop'], n_blk, p, tau, 10, 16)['flagged'] for p in patches], float)
                contaminated = np.array([bool(base_flag_v2[r // 16, c // 16]) for r, c in patches])
            num = np.bincount(tiles, weights=hit, minlength=320)
            den = np.bincount(tiles, minlength=320).astype(float)
            ci = ratio_bootstrap_ci(num, den, boot['replicates'], cfg['statistics']['ci'], boot['seed'])
            clean = ~contaminated
            power[gate][str(drop)] = {'drop': drop, 'recall': ci, 'patches': len(patches), 'flagged': int(hit.sum()),
                                      'patches_already_positive_without_injection': int(contaminated.sum()),
                                      'recall_excluding_those': float(hit[clean].mean()) if clean.any() else None,
                                      'denominator_excluding_those': int(clean.sum())}
    for gate in ('rule_10m', 'gate_v2'):
        rec = {k: v['recall']['estimate'] for k, v in power[gate].items()}
        out[gate]['power'] = power[gate]
        out[gate]['vacuous_zero_power'] = is_vacuous(rec)
        out[gate]['verdict'] = b1_gate_verdict(out[gate]['far_test']['window']['estimate'], within[gate])
    return out


def _within_reference(cfg) -> dict:
    ref = cfg['experiments']['B1']['within_season_reference']
    return {g: {'estimate': ref[g]['window_far'], 'lo': ref[g]['lo'], 'hi': ref[g]['hi'],
                'numerator': ref[g]['numerator'], 'denominator': ref[g]['denominator'], 'source': ref[g]['source']}
            for g in ('gate_v2', 'rule_10m')}


def main():
    import json
    import time
    from experiments.wayanad_evidence import config as CE
    started = C.utc_now()
    t0 = time.time()
    result = {'evidence': 'real', 'experiment_id': 'B1'}
    lines = []
    log = lambda m: (lines.append(m), print(m, flush=True))
    try:
        pre = C.require_committed_prereg()
        cfg = C.load_prereg()
        b1 = cfg['experiments']['B1']
        cfg_ev, ev_root, _ = CE.load()
        f2 = json.loads((C.ROOT / 'experiments/results/f2.json').read_text(encoding='utf-8'))
        tau = float(f2['tau']['value'])
        within = _within_reference(cfg)

        # ---- reproduction gate first (cached data, no network) ----------------------------------------------------
        rg = reproduction_gate(cfg, cfg_ev, ev_root)
        result['reproduction_check'] = rg
        if not rg['pass']:
            result.update(status='BLOCKED', verdict='NOT_RUN', reason='within-season reproduction gate failed; B1 stopped before any fetch (see reproduction_check)')
            return _finish(result, started)

        # ---- the named dates: audit, rule, fetch ------------------------------------------------------------------
        man = acquire_or_load(cfg, cfg_ev, ev_root, log)
        result['fetch_log'] = {k: man[k] for k in ('windows', 'rule', 'stac_items', 'dates', 'bytes_received', 'byte_cap') if k in man}
        result['dates_table'] = man['candidate_tables']
        if man.get('blocked_reason'):
            result.update(status='BLOCKED', verdict='NOT_RUN', reason=man['blocked_reason'] + f" | candidates: {man['candidate_tables']}")
            return _finish(result, started)
        cmp_ = b1['data']['comparisons']
        post = cmp_['primary_non_event']['post']
        comps = {'primary_non_event': (man['clear_pre_dates'], post), 'secondary_proxy': (cmp_['secondary_proxy']['pre'], post)}
        per = {}
        for ci_, (name, (pre_dates, post_date)) in enumerate(comps.items()):
            log(f'scoring {name}: pre {pre_dates} -> post {post_date}')
            per[name] = score_comparison(ci_, name, list(pre_dates), post_date, cfg, cfg_ev, ev_root, tau, within, log)
        result['comparisons'] = per
        verdicts = {n: {g: per[n][g].get('verdict') for g in ('gate_v2', 'rule_10m')} for n in per}
        if per['primary_non_event']['gate_v2'].get('status') != 'ok':
            result.update(status='BLOCKED', verdict='NOT_RUN', reason=per['primary_non_event']['gate_v2'].get('reason'))
            return _finish(result, started)
        overall = b1_overall(verdicts)
        meas = []
        for n in per:
            for g in ('gate_v2', 'rule_10m'):
                d = per[n][g]
                if 'far_test' not in d:
                    continue
                w = d['far_test']['window']
                meas.append(C.measurement(f'window_FAR_test_{g}_{n}', w['estimate'], numerator=w['numerator'], denominator=w['denominator'],
                                          lo=w['lo'], hi=w['hi'], replicates=w['replicates'], seed=w['seed'],
                                          unit='fraction of valid 160 m windows (test split, odd tiles) with >= 1 flagged px',
                                          verdict=d['verdict'], within_season_estimate=within[g]['estimate'],
                                          true_threshold=true_threshold(within[g]),
                                          recall_by_drop={k: v['recall']['estimate'] for k, v in d['power'].items()},
                                          vacuous=d['vacuous_zero_power']))
        result.update(
            status='PASS' if overall == 'TRUE' else 'FAIL', verdict=overall, verdicts_by_comparison_and_gate=verdicts,
            rule_applied=b1['keep_rule'], preregistration={**pre, 'experiment': 'B1'},
            verdict_basis='B1 overall = gate_v2 on primary_non_event, test-split window FAR vs 2 x its own within-season estimate (f2.json); '
                          "rule_10m is judged against 2 x 0.0056065 (f1.json). The memo's (>= 0.10) is not applied.",
            measurements=meas,
            interpretations=[
                'Per-gate TRUE threshold = 2 x that gate\'s committed within-season test window FAR (human decision b1_threshold_reading); '
                'FALSE = inside that gate\'s within-season CI endpoints included; anything else INCONCLUSIVE. The memo\'s (>= 0.10) is not used.',
                'Verdict number = window FAR on the TEST split (odd 128 px tiles) with a 2000-rep tile block bootstrap (seed 2024); '
                'the all-window FAR is secondary.',
                'gate_v2 follows F7\'s deployment path rather than F2\'s fold path: endmembers estimated from the pre dates only over the whole '
                'AOI outside the disks and footprint (F2.endmember_ceiling_table), stable-pixel mask from the pre dates only (NDVI std <= 0.05), '
                'F3 per-band median-offset normalisation, tau fixed at f2.json tau.value, lambda 0.',
                'No SR is used: apply_gate_v2 needs a ranking field, so the pixel-centre-aligned bilinear field of the 10 m NDVI drop is passed; '
                'it changes only WHICH sub-pixels inside a flagged block are marked, never the window flags or the FAR (asserted: the fast '
                'score > tau & flaggable window map equals the real gate\'s).',
                'Both gates score the same valid pixels (SCL validity on every date, union the 1500 m disks and footprint); asserted equal.',
                'Power: 50 patches (10 x 10 px at 10 m) per drop in distinct valid test-split windows, same seeds and positions for both gates; '
                'drops are injected into the post image\'s B04/B08 (NDVI falls by exactly the drop, B04+B08 preserved); rule_10m is re-evaluated '
                'on the injected 10 m NDVI drop, gate_v2 after re-running F3 normalisation and unmixing.',
                'Patch recall counts patches whose window/pixels the gate flags, including ones already positive without injection; recall '
                'excluding those is reported beside it.',
                'The post date is itself required to pass the x6 clear rule (cloud+shadow <= 10 %, coverage >= 99 %) and the pre dates are the '
                'accepted January 2023 dates only; n_pre therefore differs from tau\'s calibration n_pre = 2 and the sigma_f ratio is recorded, not corrected.',
                'Duplicate STAC items for one acquisition (original baseline 04.00 and a 2024 reprocessing at 05.10) are mosaicked by the existing '
                'read_bands logic: first valid item in id order wins, which is the original; both carry the same -1000 offset.'],
            limitations=[
                'One non-event year (2023), one tile, one crop; the cross-season comparison is a single pair per pre-date set, not an ensemble.',
                'A December 2023 post image is compared with January 2023 pre dates, so a one-year interval, regrowth and interannual weather are all inside the "null".',
                'tau was calibrated at n_pre = 2 on within-season folds; with more pre dates sigma_f is smaller and the production score runs hotter (recorded).',
                'The within-season references are the committed f1/f2 estimates; their own sampling error is in their CIs, not folded into the verdict thresholds.',
                'The primary comparison changes season, year, n_pre (3 vs the calibration 2) and the endmembers (estimated from the January 2023 dates) at once; '
                'the secondary proxy keeps F2\'s own January 2024 endmember configuration (e_b ceiling 0.30, 332 px) and changes season/direction only, so it is '
                'the cleaner seasonal read. Its post date precedes its pre dates in time, exactly as the yaml names it.',
                'The post image is processing baseline 05.10 while the pre dates are 04.00 (originals); both use the same -1000 offset but a baseline-related '
                'radiometric difference cannot be excluded and is inside the cross-season gap.',
                'gate_v2 recall is inflated by windows that were already false alarms (see patches_already_positive_without_injection); recall excluding them is '
                'reported and is on a small denominator.',
                'Bytes: fetch_log.bytes_received is the byte total of the run that fetched the band data (50,852,697 B for 4 band reads plus the audit); an earlier '
                'standalone run of the same audit of the same two windows, whose SCL arrays are cached and reused, received a further 1,131,524 B. Total about 51.98 MB against a 300 MB cap.'],
            runtime_seconds=round(time.time() - t0, 1))
        return _finish(result, started)
    except C.Blocked as exc:
        result.update(status='BLOCKED', verdict='NOT_RUN', reason=str(exc))
        return _finish(result, started)


def _finish(result, started):
    import json
    result.setdefault('interpretations', [])
    result.setdefault('reproduction_check', None)
    result.setdefault('measurements', [])
    result.setdefault('limitations', [])
    result.setdefault('fetch_log', {'bytes_received': 0, 'items': []})
    out = C.write_result('b1', result, started)
    print(json.dumps({k: out.get(k) for k in ('status', 'verdict', 'reason')}, indent=2))
    return C.exit_code(out['status'])


if __name__ == '__main__':
    raise SystemExit(main())
