"""X6 (agent A6): season matching and event-to-image latency for the Wayanad before/after pair. REAL data.

(a) Anniversary pre dates (clear Nov-Dec 2023) vs the current January 2024 pool, on E6's exact stable-pixel false-drop metric.
(b) A first-valid-observation post composite from 2024-10-27, 2024-11-01, 2024-11-06 (SCL mask only), kept iff footprint
    validity >= 0.90 and the event-to-latest-contributing-image gap <= 99 days.
Every definition and keep rule is fixed in docs/adr-x6-season-latency.md and configs/x6.yaml (and configs/exceptional.yaml, a6)
before any run. The pure helpers below are unit-tested (tests/test_x6.py); `run` does the network and file work.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from datetime import date as _date
from pathlib import Path

import numpy as np
import yaml

from experiments.common import Blocked
from trustsr.bootstrap import block_sums, ratio_bootstrap_ci

NO_SOURCE = 255


# ---- pure helpers (unit-tested) --------------------------------------------------------------------------

def clear_dates_in_window(rows, start: str, end: str, max_cloud: float, min_coverage: float) -> list[str]:
    """Dates in [start, end] (inclusive, ISO strings) whose best acquisition row passes the AOI clear test."""
    best = {}
    for r in rows:
        if r['date'] not in best or r.get('cloud_shadow_pct_aoi', 100) < best[r['date']].get('cloud_shadow_pct_aoi', 100):
            best[r['date']] = r
    return sorted(d for d, r in best.items() if start <= d <= end
                  and r.get('coverage_pct', 0) >= min_coverage and r.get('cloud_shadow_pct_aoi', 100) <= max_cloud)


def gap_days(event: str, day: str) -> int:
    return (_date.fromisoformat(day) - _date.fromisoformat(event)).days


def stable_min_valid(n_dates: int, allowed_missing: int) -> int:
    """E6 allows two of twelve dates missing (>= 10 valid); keep the same count of allowed-missing dates, at least one valid."""
    return max(1, n_dates - allowed_missing)


def first_valid_composite(valid_layers) -> np.ndarray:
    """Per-pixel index of the FIRST layer (date order) whose mask is True; 255 where no layer is valid."""
    valid = np.asarray(valid_layers, bool)
    src = np.full(valid.shape[1:], NO_SOURCE, dtype=np.uint8)
    for i in range(valid.shape[0] - 1, -1, -1):
        src[valid[i]] = i
    return src


def select_by_source(layers, source, fill):
    """layers (n, ..., H, W); source (H, W) of layer indices (255 = none) -> (..., H, W) taking each pixel from its source layer."""
    layers = np.asarray(layers)
    out = np.full(layers.shape[1:], fill, dtype=layers.dtype)
    for i in range(layers.shape[0]):
        m = source == i
        if m.any():
            out[..., m] = layers[i][..., m]
    return out


def footprint_validity(source, footprint):
    """(valid fraction, valid px, footprint px): footprint pixels that have a source date, over all footprint pixels."""
    inside = np.asarray(footprint, bool)
    ok = inside & (np.asarray(source) != NO_SOURCE)
    return float(ok.sum() / inside.sum()), int(ok.sum()), int(inside.sum())


def latest_contributing_date(source, footprint, dates) -> dict:
    """Latest date supplying at least one FOOTPRINT pixel, and the footprint pixel count per date."""
    inside = np.asarray(footprint, bool)
    counts = {d: int((inside & (np.asarray(source) == i)).sum()) for i, d in enumerate(dates)}
    used = [d for d in dates if counts[d] > 0]
    return {'date': used[-1] if used else None, 'footprint_px_by_date': counts}


def pixel_gap_distribution(source, footprint, dates, event: str) -> dict:
    """Event-to-source-image gap in days for every footprint pixel that has a source."""
    src = np.asarray(source)[np.asarray(footprint, bool)]
    src = src[src != NO_SOURCE]
    if src.size == 0:
        return {'n': 0, 'min': None, 'median': None, 'max': None, 'mean': None}
    per_date = np.array([gap_days(event, d) for d in dates])
    g = per_date[src.astype(int)]
    return {'n': int(g.size), 'min': int(g.min()), 'median': float(np.median(g)), 'max': int(g.max()), 'mean': float(g.mean())}


def keep_rule_a(fraction_anniversary: float, fraction_january: float, hi: float) -> bool:
    """KEEP iff the anniversary false-drop fraction is lower AND the CI of (anniversary - january) lies entirely below 0."""
    return bool(fraction_anniversary < fraction_january and hi < 0)


def keep_rule_b(footprint_validity_fraction: float, gap, min_valid: float, max_gap: int) -> bool:
    return bool(gap is not None and footprint_validity_fraction >= min_valid and gap <= max_gap)


def window_transform(transform, rows, cols):
    """Affine of a (rows, cols) window of the grid; the window origin is an integer number of pixels from the grid origin."""
    from affine import Affine
    return transform * Affine.translation(cols[0], rows[0])


def three_date_subsets(n: int) -> list[list[int]]:
    return [list(c) for c in itertools.combinations(range(n), 3)]


def stable_pixels(stack, transform, centres_xy, radius_m: float, min_valid: int, max_std: float):
    """E6 stable definition: (stable, suspected). `centres_xy` are projected (x, y) in the raster CRS."""
    from experiments.e6_season_matched import disk_mask, stable_mask
    suspected = np.zeros(stack.shape[-2:], bool)
    for xy in centres_xy:
        suspected |= disk_mask(transform, suspected.shape, xy, radius_m)
    return stable_mask(stack, suspected, min_valid, max_std), suspected


def paired_false_drop(pre_a, pre_b, post, stable, threshold: float, block: int, replicates: int, ci: float, seed: int) -> dict:
    """False-drop fractions of two pre pools against one post, on ONE shared comparable set, with block-bootstrap CIs
    (trustsr.bootstrap) for each fraction and for the paired difference a - b."""
    from experiments.e6_season_matched import false_drop
    res = false_drop(pre_a, pre_b, post, stable, threshold)
    if not res['comparable_pixels']:
        raise ValueError('no comparable pixels')
    den = block_sums(res['comparable'], block)
    na, nb = block_sums(res['flags_a'], block), block_sums(res['flags_b'], block)
    return {'comparable_pixels': res['comparable_pixels'], 'flagged_a': int(res['flags_a'].sum()),
            'flagged_b': int(res['flags_b'].sum()), 'fraction_a': res['fraction_a'], 'fraction_b': res['fraction_b'],
            'a': ratio_bootstrap_ci(na, den, replicates, ci, seed), 'b': ratio_bootstrap_ci(nb, den, replicates, ci, seed),
            'difference': ratio_bootstrap_ci(na, den, replicates, ci, seed, num_b=nb),
            'block_px': block, 'blocks_with_data': int((den > 0).sum())}


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def ndvi_from_dn(red_dn, nir_dn, valid, conv: dict, rad: dict) -> np.ndarray:
    """NDVI under a convention: E6 exact (no clip, floor 0.01) or the evidence run's (clip at 0, floor 0.05); NaN where invalid."""
    from experiments.common import ndvi_bands
    from experiments.e6_season_matched import reflectance
    r = reflectance(red_dn, rad['quantification'], rad['boa_add_offset'])
    n = reflectance(nir_dn, rad['quantification'], rad['boa_add_offset'])
    if conv['clip_negative_reflectance']:
        r, n = np.clip(r, 0.0, None), np.clip(n, 0.0, None)
    return np.where(valid, ndvi_bands(r, n, conv['ndvi_min_denominator']), np.nan).astype(np.float32)


class ByteLog:
    """Cumulative received bytes across (re)runs: per-process ByteBudget totals appended to a JSON file."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.runs = json.loads(self.path.read_text()) if self.path.is_file() else []

    def record(self, label: str, budget) -> None:
        self.runs.append({'label': label, **budget.log()})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.runs, indent=2) + '\n')

    def total(self) -> int:
        return int(sum(r['bytes_received'] for r in self.runs))


# ---- config -----------------------------------------------------------------------------------------------

def load_x6_config(path='configs/x6.yaml'):
    """(x6 config, root, evidence cfg, exceptional cfg, per-file sha256 dict, combined sha256)."""
    from experiments.wayanad_evidence import config as EC
    path = Path(path).resolve()
    root = path.parent.parent
    cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
    if cfg['schema_version'] != 1:
        raise ValueError('unsupported x6 schema')
    ev, _, _ = EC.load(root / cfg['evidence_config'])
    exc = yaml.safe_load((root / cfg['exceptional_config']).read_text(encoding='utf-8'))
    files = {'x6': path, 'exceptional': root / cfg['exceptional_config'], 'wayanad_evidence': root / cfg['evidence_config'],
             'experiments': root / cfg['experiments_config'], 'phase0': root / ev['phase0_config']}
    shas = {k: sha256_file(v) for k, v in files.items()}
    combined = hashlib.sha256(''.join(shas[k] for k in sorted(shas)).encode()).hexdigest()
    return cfg, root, ev, exc, shas, combined


# ---- network plumbing (real data) --------------------------------------------------------------------------

def _stac_items(ev, start, end, transform, shape):
    """Search with the wayanad-evidence search code over a custom window; returns (items, groups)."""
    from experiments.wayanad_evidence import fetch as F
    c = json.loads(json.dumps(ev))
    c['dates']['audit_start'], c['dates']['audit_end'] = start, end
    return F.search_groups(c, transform, shape)


def _baselines(items_by_id, ids):
    return sorted({items_by_id[i]['properties']['s2:processing_baseline'] for i in ids if i in items_by_id})


def _baseline_key(v: str):
    return tuple(int(x) for x in v.split('.'))


def read_window_band(item, band, window_t, window_shape, crs, policy, budget):
    """One 10 m band of one item on an exact sub-window of the AOI grid (WarpedVRT, nearest). Retried 3x, then fails naming the URL."""
    import planetary_computer as pc
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT
    from risk.common import retry
    href = item['assets'][band]['href']

    def go():
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', GDAL_HTTP_TIMEOUT=policy['timeout_seconds']):
            with rasterio.open(pc.sign(href)) as src:
                if src.crs.to_string() != crs:
                    raise RuntimeError(f'{href.split("?")[0]}: CRS {src.crs} differs from AOI CRS {crs}')
                with WarpedVRT(src, crs=crs, transform=window_t, width=window_shape[1], height=window_shape[0],
                               resampling=Resampling.nearest, nodata=0) as vrt:
                    return vrt.read(1)
    out = retry(go, policy, href.split('?')[0])
    budget.check()
    return out


def fetch_anniversary_date(ev, x6, date, items, scl10_mosaic, per_item, transform, shape, budget, cache: Path):
    """B04/B08 over the AOI and B03/B02 over the crop window for one date -> cache/{date}.npz (idempotent).
    The first-valid-tile rule follows experiments.wayanad_evidence.fetch.read_bands (tile-valid = not SCL invalid)."""
    target = cache / f'{date}.npz'
    if target.is_file():
        return target
    a, b = x6['exp_a'], x6['exp_b']
    policy, crs = ev['phase0']['network'], ev['aoi']['crs']
    crop_r, crop_c = b['crop_rows'], b['crop_cols']
    crop_shape = (crop_r[1] - crop_r[0], crop_c[1] - crop_c[0])
    crop_t = window_transform(transform, crop_r, crop_c)
    full = {k: np.zeros(shape, 'uint16') for k in a['fetch_bands_full_aoi']}
    crop = {k: np.zeros(crop_shape, 'uint16') for k in a['fetch_bands_crop_only']}
    taken = np.zeros(shape, bool)
    for item, (_, valid) in zip(items, per_item):
        take = valid & ~taken
        if not take.any():
            continue
        for k in full:
            full[k][take] = read_window_band(item, k, transform, shape, crs, policy, budget)[take]
        take_crop = take[crop_r[0]:crop_r[1], crop_c[0]:crop_c[1]]
        if take_crop.any():
            for k in crop:
                crop[k][take_crop] = read_window_band(item, k, crop_t, crop_shape, crs, policy, budget)[take_crop]
        taken |= take
    meta = {'date': date, 'items': [i['id'] for i in items], 'bands_full_aoi': list(full), 'bands_crop_only': list(crop),
            'crop_rows': crop_r, 'crop_cols': crop_c, 'scl_policy': 'nearest neighbour 20 m -> 10 m; first valid tile wins',
            'bytes_received_so_far_this_process': budget.used()}
    np.savez_compressed(target, red=full['B04'], nir=full['B08'], green_crop=crop['B03'], blue_crop=crop['B02'],
                        scl=scl10_mosaic, taken=taken, meta=json.dumps(meta))
    return target


# ---- the run ----------------------------------------------------------------------------------------------

def run(x6_path='configs/x6.yaml'):
    from datetime import datetime, timezone

    import rasterio
    from affine import Affine
    from pyproj import Transformer

    from experiments.common import environment_record, finalize_result
    from experiments.e6_season_matched import ByteBudget, false_drop, pooled_mean
    from experiments.wayanad_evidence import fetch as F
    from experiments.wayanad_evidence.cogio import write_cog
    from experiments.wayanad_evidence.data import scl_valid
    from experiments.wayanad_evidence.geo import reference_grid
    from risk.common import write_json

    started = datetime.now(timezone.utc).isoformat()
    x6, root, ev, exc, shas, config_hash = load_x6_config(x6_path)
    a, b = x6['exp_a'], x6['exp_b']
    im, rad = ev['phase0']['imagery'], ev['radiometry']
    policy = ev['phase0']['network']
    stats = exc['statistics']['bootstrap']
    transform, shape = reference_grid(ev['aoi'])
    t20 = Affine(20.0, 0.0, transform.c, 0.0, -20.0, transform.f)
    shape20 = (shape[0] // 2, shape[1] // 2)
    main_cache, a6_cache = root / x6['paths']['main_cache'], root / x6['paths']['a6_cache']
    a6_cache.mkdir(parents=True, exist_ok=True)
    out_dir = root / x6['paths']['outputs']
    out_dir.mkdir(parents=True, exist_ok=True)
    crs = ev['aoi']['crs']
    crop_r, crop_c = b['crop_rows'], b['crop_cols']
    crop_shape = (crop_r[1] - crop_r[0], crop_c[1] - crop_c[0])
    crop_t = window_transform(transform, crop_r, crop_c)
    sl = (slice(crop_r[0], crop_r[1]), slice(crop_c[0], crop_c[1]))

    # The cap covers every X6 process together: earlier processes' bytes are subtracted from this process's allowance.
    bytelog = ByteLog(a6_cache.parent / 'fetch_log.json')
    cap = ev['fetch']['max_fetch_bytes']
    if bytelog.total() >= cap:
        raise RuntimeError(f'cumulative byte cap {cap} already reached by earlier X6 processes ({bytelog.total()})')
    budget = ByteBudget(cap - bytelog.total())
    result = {'evidence': 'real'}
    outputs = []

    # ------------------------------------------------------------- (b) decision from the cached 20 m SCL (no network)
    audit = json.loads((root / 'experiments/wayanad_evidence/outputs/audit_acquisitions.json').read_text())
    step2 = json.loads((root / 'experiments/wayanad_evidence/outputs/step2.json').read_text())
    with rasterio.open(root / x6['paths']['footprint']) as src:
        footprint = src.read(1).astype(bool)
        if src.transform != transform or src.crs.to_string() != crs:
            raise RuntimeError('footprint raster is not on the reference grid')
    rows_b = {d: min((r for r in audit['rows'] if r['date'] == d), key=lambda r: r['cloud_shadow_pct_aoi']) for d in b['composite_dates']}
    step2_table = {t['date']: t for t in step2['measurements']['table']}
    per_date_b, scl_b, per_item_b = {}, {}, {}
    for d in b['composite_dates']:
        scl10, per_item = F.scl_mosaic_10m(ev, [{'id': i} for i in rows_b[d]['item_ids']], main_cache)
        scl_b[d], per_item_b[d] = scl10, per_item
        ok = scl_valid(scl10, im)
        frac = float((ok & footprint).sum() / footprint.sum())
        per_date_b[d] = {'footprint_valid_fraction': frac, 'footprint_valid_px': int((ok & footprint).sum()),
                         'aoi_cloud_shadow_pct': rows_b[d]['cloud_shadow_pct_aoi'], 'platform': rows_b[d]['platform'],
                         'orbit': rows_b[d]['orbit'], 'item_ids': rows_b[d]['item_ids'],
                         'days_after_event': gap_days(ev['event_date'], d),
                         'step2_footprint_valid_fraction': step2_table[d]['footprint_valid_fraction'],
                         'reproduces_step2': abs(frac - step2_table[d]['footprint_valid_fraction']) < 1e-9}
    source = first_valid_composite([scl_valid(scl_b[d], im) for d in b['composite_dates']])
    val_frac, val_px, fp_px = footprint_validity(source, footprint)
    latest = latest_contributing_date(source, footprint, b['composite_dates'])
    gap = gap_days(ev['event_date'], latest['date']) if latest['date'] else None
    aoi_latest = latest_contributing_date(source, np.ones(shape, bool), b['composite_dates'])
    keep_b = keep_rule_b(val_frac, gap, b['min_footprint_valid'], b['max_gap_days'])
    part_b = {'composite_dates': b['composite_dates'], 'per_date': per_date_b, 'footprint_px': fp_px,
              'composite_footprint_valid_px': val_px, 'composite_footprint_validity': val_frac,
              'footprint_px_by_source_date': latest['footprint_px_by_date'],
              'latest_contributing_date_footprint': latest['date'], 'gap_days_footprint': gap,
              'latest_contributing_date_aoi': aoi_latest['date'],
              'gap_days_aoi': gap_days(ev['event_date'], aoi_latest['date']) if aoi_latest['date'] else None,
              'aoi_px_by_source_date': aoi_latest['footprint_px_by_date'],
              'aoi_composite_valid_fraction': float((source != NO_SOURCE).mean()),
              'per_pixel_gap_days_footprint': pixel_gap_distribution(source, footprint, b['composite_dates'], ev['event_date']),
              'gap_days_current_post': gap_days(ev['event_date'], a['post_date']),
              'rule': f"footprint validity >= {b['min_footprint_valid']} AND gap <= {b['max_gap_days']} days", 'keep': keep_b}

    # ------------------------------------------------------------- (a) anniversary audit (network)
    items_a, groups_a = _stac_items(ev, a['audit_start'], a['audit_end'], transform, shape)
    items_by_id = {i['id']: i for i in items_a}
    rows_a, _, errors_a = F.audit(ev, root, groups_a, t20, shape20, budget, a6_cache)
    if errors_a:
        raise Blocked(f'{len(errors_a)} SCL asset(s) unreadable in the anniversary audit, first: {errors_a[0]}', evidence='real')
    lo_d, hi_d = a['audit_start'][:10], a['audit_end'][:10]
    clear_all = clear_dates_in_window(rows_a, lo_d, hi_d, im['max_cloud_shadow_pct'], im['min_coverage_pct'])
    pool_dates = clear_dates_in_window(rows_a, a['pool_window'][0], a['pool_window'][1], im['max_cloud_shadow_pct'], im['min_coverage_pct'])
    audit_table = [{'date': r['date'], 'platform': r['platform'], 'orbit': r['orbit'], 'item_ids': r['item_ids'],
                    'cloud_shadow_pct_aoi': round(r['cloud_shadow_pct_aoi'], 3), 'coverage_pct': r['coverage_pct'],
                    'clear': r['date'] in clear_all, 'pooled': r['date'] in pool_dates} for r in rows_a]
    part_a = {'audit_window': [lo_d, hi_d], 'pool_window': a['pool_window'], 'audited_acquisitions': len(rows_a),
              'audit_table': audit_table, 'clear_dates_in_audit_window': clear_all, 'anniversary_pool_dates': pool_dates,
              'january_pool_dates': a['january_pool'], 'post_date': a['post_date'], 'min_pool_dates': a['min_pool_dates']}
    def anniversary(pool):
        """Fetch (idempotent, byte-capped), then compare `pool` with the January pool. Returns the result dict."""
        for d in pool:
            ids = [i for r in rows_a if r['date'] == d for i in r['item_ids']]
            baselines = _baselines(items_by_id, ids)
            if any(_baseline_key(v) < _baseline_key(rad['min_processing_baseline']) for v in baselines):
                raise Blocked(f'{d}: processing baseline {baselines} < {rad["min_processing_baseline"]}', evidence='real')
            scl10, per_item = F.scl_mosaic_10m(ev, [{'id': i} for i in ids], a6_cache)
            fetch_anniversary_date(ev, x6, d, [items_by_id[i] for i in ids], scl10, per_item, transform, shape, budget, a6_cache)
            print(f'cached {d}: {budget.used()} bytes received this process', flush=True)

        def load(path):
            with np.load(path) as z:
                return {'red': z['red'], 'nir': z['nir'], 'scl': z['scl'], 'taken': z['taken'], 'meta': json.loads(str(z['meta']))}

        data = {d: load(a6_cache / f'{d}.npz') for d in pool}
        for d in a['january_pool'] + [a['post_date']]:
            with np.load(main_cache / f'{d}.npz') as z:
                data[d] = {'red': z['dn'][0], 'nir': z['dn'][3], 'scl': z['scl'], 'taken': z['taken'], 'meta': json.loads(str(z['meta']))}
        pre_dates = pool + a['january_pool']
        ia, ij = list(range(len(pool))), list(range(len(pool), len(pre_dates)))
        to_metric = Transformer.from_crs('EPSG:4326', crs, always_xy=True)
        pts = ev['aoi']['plausibility_points']
        centres = [to_metric.transform(pts[k]['lon'], pts[k]['lat']) for k in a['disk_points']]
        min_valid = stable_min_valid(len(pre_dates), a['stable_allowed_missing_dates'])

        def compare(conv, exclude_footprint=False):
            def ndvi_of(d):
                v = data[d]
                return ndvi_from_dn(v['red'], v['nir'], scl_valid(v['scl'], im) & v['taken'], conv, rad)
            stack, post = np.stack([ndvi_of(d) for d in pre_dates]), ndvi_of(a['post_date'])
            stable, suspected = stable_pixels(stack, transform, centres, a['disk_radius_m'], min_valid, a['stable_max_std'])
            if exclude_footprint:
                stable = stable & ~footprint
            pool_a, pool_j = pooled_mean(stack, ia, a['pool_min_valid']), pooled_mean(stack, ij, a['pool_min_valid'])
            res = paired_false_drop(pool_a, pool_j, post, stable, conv['threshold'], stats['block_px_10m'],
                                    stats['replicates'], stats['ci'], stats['seed'])
            comp = stable & np.isfinite(pool_a) & np.isfinite(pool_j) & np.isfinite(post)
            res.update(stable_pixels=int(stable.sum()), suspected_disk_px=int(suspected.sum()), aoi_px=int(stable.size),
                       mean_ndvi_on_comparable={'anniversary_pool': float(pool_a[comp].mean()), 'january_pool': float(pool_j[comp].mean()),
                                                'post': float(post[comp].mean())},
                       mean_signed_drop={'anniversary_minus_post': float((pool_a[comp] - post[comp]).mean()),
                                         'january_minus_post': float((pool_j[comp] - post[comp]).mean())},
                       per_pre_date_mean_ndvi_on_comparable={d: (float(np.nanmean(stack[i][comp])) if np.isfinite(stack[i][comp]).any() else None)
                                                                for i, d in enumerate(pre_dates)},   # mean over pixels valid on that date
                       subsets=None)
            if len(pool) > 3:
                vals = []
                for sub in three_date_subsets(len(pool)):
                    fd = false_drop(pooled_mean(stack, sub, a['pool_min_valid']), pool_j, post, stable, conv['threshold'])
                    vals.append({'dates': [pool[i] for i in sub], 'comparable_pixels': fd['comparable_pixels'],
                                 'fraction_anniversary_subset': fd['fraction_a'], 'fraction_january': fd['fraction_b']})
                fr = np.array([v['fraction_anniversary_subset'] for v in vals])
                res['subsets'] = {'n_subsets': len(vals), 'min': float(fr.min()), 'median': float(np.median(fr)), 'max': float(fr.max()),
                                  'all': vals, 'note': 'each subset has its own comparable set (pixels valid in that subset pool), '
                                                       'so denominators differ slightly from the headline comparison'}
            return res

        cog_dir = root / x6['paths']['a6_cogs']
        for d in pool:      # crop-window 4-band COG of each fetched Nov-Dec 2023 date (Wave-2 SR input); uncommitted, data_a6
            with np.load(a6_cache / f'{d}.npz') as z:
                dn4 = np.stack([z['red'][sl], z['green_crop'], z['blue_crop'], z['nir'][sl]]).astype('uint16')
                scl_c = z['scl'][sl].astype('uint16')
            outputs.append(write_cog(
                cog_dir / f'anniversary_{d}_crop.tif', np.concatenate([dn4, scl_c[None]]), crop_t, crs, nodata=0,
                descriptions=['B04', 'B03', 'B02', 'B08', 'SCL'],
                tags={'date': d, 'items': ','.join(data[d]['meta']['items']), 'band_order': 'B04,B03,B02,B08,SCL',
                      'reflectance': '(DN-1000)/10000 for baseline>=04.00', 'scl_policy': 'validity mask only',
                      'crop_rows': crop_r, 'crop_cols': crop_c}))
        return {
            'pool_dates': pool, 'pre_dates_union': pre_dates,
            'stable_definition': {'min_valid_dates': min_valid, 'of_pre_dates': len(pre_dates), 'max_ndvi_std': a['stable_max_std'],
                                  'disk_radius_m': a['disk_radius_m'], 'disk_points': a['disk_points'],
                                  'footprint_excluded_in_primary': False},
            'primary': {'convention': a['primary'], **compare(a['primary'])},
            'sensitivity_evidence_conventions': {'convention': a['sensitivity'], **compare(a['sensitivity'])},
            'primary_excluding_footprint': {'convention': a['primary'], **compare(a['primary'], exclude_footprint=True)},
            'bootstrap': stats,
            'items_and_baselines': {d: {'items': data[d]['meta']['items'],
                                        'processing_baselines': data[d]['meta'].get('processing_baselines')
                                        or _baselines(items_by_id, data[d]['meta']['items'])} for d in data}}

    keep_a = False
    if len(pool_dates) < a['min_pool_dates']:
        part_a.update(ran=False, keep=False,
                      reason=f'only {len(pool_dates)} clear Nov-Dec 2023 date(s) {pool_dates}; need {a["min_pool_dates"]}. '
                             'Every audited date and its cloud+shadow percentage is in audit_table.')
    else:
        res_a = anniversary(pool_dates)
        keep_a = keep_rule_a(res_a['primary']['fraction_a'], res_a['primary']['fraction_b'], res_a['primary']['difference']['hi'])
        part_a.update(ran=True, keep=keep_a, rule='anniversary false-drop fraction < January AND the 95 % block-bootstrap CI of '
                                                 '(anniversary - january) lies below 0', **res_a)
    # E-a2: EXPLORATORY, post hoc (ADR deviations log 2). No keep rule; cannot change the verdict above.
    ex = x6['exploratory']
    relaxed = clear_dates_in_window(rows_a, a['pool_window'][0], a['pool_window'][1], ex['relaxed_pool_max_cloud_shadow_pct'],
                                    im['min_coverage_pct'])
    part_a['exploratory_relaxed_pool'] = {
        'label': 'EXPLORATORY, post hoc, no keep rule; cannot change the pre-registered verdict',
        'max_cloud_shadow_pct': ex['relaxed_pool_max_cloud_shadow_pct'], 'pool_dates': relaxed,
        **(anniversary(relaxed) if len(relaxed) >= 3 else {'ran': False, 'reason': f'only {len(relaxed)} dates'})}

    # E-b2: EXPLORATORY cumulative composite curve from the cached whole-year SCL (no network).
    cum_dates = sorted({r['date'] for r in audit['rows'] if ex['cumulative_composite_from'] <= r['date'] <= ex['cumulative_composite_until']})
    layers_cum, curve = [], []
    for d in cum_dates:
        ids = [i for r in audit['rows'] if r['date'] == d for i in r['item_ids']]
        scl10, _ = F.scl_mosaic_10m(ev, [{'id': i} for i in ids], main_cache)
        layers_cum.append(scl_valid(scl10, im))
        src_c = first_valid_composite(layers_cum)
        v_c, _, _ = footprint_validity(src_c, footprint)
        lat = latest_contributing_date(src_c, footprint, cum_dates[:len(layers_cum)])
        curve.append({'up_to_date': d, 'days_after_event': gap_days(ev['event_date'], d), 'footprint_validity': v_c,
                      'latest_contributing_date_footprint': lat['date'],
                      'gap_days_footprint': gap_days(ev['event_date'], lat['date']) if lat['date'] else None,
                      'median_pixel_gap_days': pixel_gap_distribution(src_c, footprint, cum_dates[:len(layers_cum)], ev['event_date'])['median']})
    first_ok = next((c for c in curve if c['footprint_validity'] >= b['min_footprint_valid']), None)
    part_b['exploratory_cumulative_curve'] = {
        'label': 'EXPLORATORY, no keep rule: first-valid composite of ALL post-event acquisitions up to each date, from cached SCL',
        'first_date_reaching_min_validity': first_ok, 'curve': curve}

    # ------------------------------------------------------------- (b) composite, only if its keep rule passed
    part_b['composite_written'] = False
    if keep_b:
        items_c, _ = _stac_items(ev, '2024-10-25T00:00:00Z', '2024-11-08T23:59:59Z', transform, shape)
        by_id_c = {i['id']: i for i in items_c}
        src_crop = source[sl]
        dn = np.zeros((4, *crop_shape), 'uint16')
        meta_b = {}
        for i, d in enumerate(b['composite_dates']):
            m = src_crop == i
            ids = rows_b[d]['item_ids']
            missing = [k for k in ids if k not in by_id_c]
            if missing:
                raise Blocked(f'{d}: STAC items {missing} not returned by the search; cannot fetch bands', evidence='real')
            meta_b[d] = {'item_ids': ids, 'processing_baselines': _baselines(by_id_c, ids), 'platform': rows_b[d]['platform'],
                         'relative_orbit': rows_b[d]['orbit'], 'pixels_in_crop': int(m.sum())}
            if not m.any():
                continue
            cached = a6_cache / 'composite' / f'{d}.npz'
            if cached.is_file():
                with np.load(cached) as z:
                    band_dn = z['dn']
            else:
                band_dn = np.zeros((4, *crop_shape), 'uint16')
                taken = np.zeros(crop_shape, bool)
                for item_id, (_, tile_valid) in zip(ids, per_item_b[d]):
                    take = tile_valid[sl] & ~taken
                    if not take.any():
                        continue
                    for bi, name in enumerate(F.MODEL_BANDS):
                        band_dn[bi][take] = read_window_band(by_id_c[item_id], name, crop_t, crop_shape, crs, policy, budget)[take]
                    taken |= take
                cached.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(cached, dn=band_dn)
            print(f'composite date {d}: {budget.used()} bytes received this process', flush=True)
            dn[:, m] = band_dn[:, m]
        outputs.append(write_cog(
            out_dir / 'post_composite_crop.tif', dn, crop_t, crs, nodata=0, descriptions=list(F.MODEL_BANDS),
            tags={'composite_dates': ','.join(b['composite_dates']),
                  'rule': 'per-pixel first valid observation in date order (SCL valid mask only)', 'band_order': 'B04,B03,B02,B08',
                  'reflectance': '(DN-1000)/10000 for baseline>=04.00', 'radiometric_harmonisation': 'NONE applied',
                  'crop_rows': crop_r, 'crop_cols': crop_c, 'nodata': '0 where no date is SCL-valid'}))
        outputs.append(write_cog(
            out_dir / 'post_composite_source_index.tif', src_crop.astype('uint8'), crop_t, crs, nodata=NO_SOURCE, categorical=True,
            descriptions=['source_date_index'],
            tags={'index_to_date': json.dumps(dict(enumerate(b['composite_dates']))), 'nodata': '255 = no valid date'}))
        manifest = {'experiment': 'x6', 'composite_dates': b['composite_dates'], 'rule': part_b['rule'], 'crs': crs,
                    'window': {'rows': crop_r, 'cols': crop_c, 'shape': list(crop_shape), 'transform': list(crop_t)[:6],
                               'parent_grid_transform': list(transform)[:6]},
                    'band_order': list(F.MODEL_BANDS), 'dtype': 'uint16 DN, reflectance=(DN-1000)/10000 (baseline>=04.00)',
                    'sources': meta_b, 'radiometric_harmonisation': 'NONE applied between dates (limitation)',
                    'scl_policy': 'validity mask only; classes 2,3,8,9,10 (shadow/cloud) and 0,1,11 (invalid) are not valid',
                    'pixels_by_source_in_crop': {d: int((src_crop == i).sum()) for i, d in enumerate(b['composite_dates'])},
                    'no_source_px_in_crop': int((src_crop == NO_SOURCE).sum())}
        write_json(out_dir / 'post_composite_manifest.json', manifest)
        part_b.update(composite_written=True, composite_manifest='post_composite_manifest.json', sources=meta_b,
                      crop_window={'rows': crop_r, 'cols': crop_c})
    else:
        part_b['composite_not_written_reason'] = 'keep rule (b) failed; nothing was fetched for the composite'

    # ------------------------------------------------------------- result
    bytelog.record('process', budget)
    raster_hashes = {str(p.relative_to(root)): sha256_file(p) for p in outputs}
    if keep_b:
        raster_hashes[str((out_dir / 'post_composite_manifest.json').relative_to(root))] = sha256_file(out_dir / 'post_composite_manifest.json')
    result.update(
        status='PASS' if (keep_a and keep_b) else 'FAIL',
        decision={'a_keep_anniversary': keep_a, 'b_keep_composite': keep_b},
        exp_a=part_a, exp_b=part_b,
        stac={'collection': im['collection'], 'stac_url': im['stac_url'],
              'anniversary_window_items': {r['date']: r['item_ids'] for r in rows_a},
              'composite_items': {d: rows_b[d]['item_ids'] for d in b['composite_dates']}},
        bytes={'cap_bytes': cap, 'total_received_all_x6_processes': bytelog.total(), 'per_process': bytelog.runs,
               'method': 'nettop per-PID bytes_in; cumulative across X6 reruns'},
        output_sha256=raster_hashes, config_sha256_files=shas,
        limitations=['Single AOI, single post date (2024-12-06), one year pair: one comparison, not a general result.',
                     'Stable pixels are seasonally steady vegetation by construction (NDVI std <= 0.05 across the pre dates); the '
                     'result does not speak for paddy, bare soil or plantation clearings.',
                     'No detection labels were used: a lower false-drop fraction is not a detection gain.',
                     'SCL validity is a model estimate and does not mean haze-free; the composite applies NO radiometric harmonisation between dates.',
                     'The footprint comes from the January-vs-December NDVI drop, so validity is measured on an outcome-derived region '
                     'used for date selection only.'])
    result = finalize_result('x6', result, config_hash, started)
    result['environment'] = environment_record(yaml.safe_load((root / x6['experiments_config']).read_text()).get('environment'))
    return result, root, x6


def main():
    import sys
    from datetime import datetime, timezone

    from experiments.common import finalize_result
    from risk.common import write_json
    try:
        result, root, x6 = run()
    except Blocked as exc:
        x6, root, _, _, _, h = load_x6_config()
        result = finalize_result('x6', {'status': 'BLOCKED', 'evidence': exc.evidence, 'reason': str(exc)}, h,
                                 datetime.now(timezone.utc).isoformat())
    write_json(root / x6['paths']['results'] / 'x6.json', result)
    print(json.dumps({k: result[k] for k in ('status', 'decision', 'reason') if k in result}, indent=2))
    sys.exit(0 if result['status'] == 'PASS' else 2)


if __name__ == '__main__':
    main()
