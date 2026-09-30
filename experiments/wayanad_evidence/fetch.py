"""STAC search, SCL audit of the new AOI, date selection and windowed 10 m band reads.

Reuses risk.r2_imagery (search_items, read_scl, summarize_scl) and risk.common (retry). Only B02/B03/B04/B08 are read as
spectral data; SCL is read only as a validity mask and is resampled by nearest neighbour (here: exact 2x replication of the
20 m grid, which is aligned to the 10 m grid by construction).
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

MODEL_BANDS = ['B04', 'B03', 'B02', 'B08']       # model order; storage names are preserved in the cache metadata


# ---- pure helpers (unit-tested) ---------------------------------------------------------------------------

def reflectance(dn, rad: dict) -> np.ndarray:
    """ESA L2A: rho = (DN + BOA_ADD_OFFSET) / QUANTIFICATION (offset applies for processing baseline >= 04.00), clipped."""
    rho = (np.asarray(dn, dtype=np.float64) + rad['boa_add_offset']) / rad['quantification']
    return np.clip(rho, rad['clip_min'], None).astype(np.float32)


def scl_to_10m(scl20: np.ndarray) -> np.ndarray:
    """Nearest-neighbour SCL 20 m -> 10 m on aligned grids: every 20 m pixel becomes a 2x2 block."""
    return np.repeat(np.repeat(scl20, 2, axis=0), 2, axis=1)


def mosaic_first_valid(layers):
    """[(array, valid_mask), ...] in a fixed order -> (mosaic, taken). The first valid tile wins; never cloud-minimised."""
    out = np.zeros_like(layers[0][0])
    taken = np.zeros(layers[0][0].shape, dtype=bool)
    for array, valid in layers:
        take = np.asarray(valid, bool) & ~taken
        out[take] = array[take]
        taken |= take
    return out, taken


def choose_dates(rows, dates_cfg: dict, event_date: str, max_cloud: float, min_coverage: float):
    """Preferred dates that pass the AOI clear test are kept; failing ones are replaced by the earliest clear dates."""
    best = {}
    for r in rows:
        if r['date'] not in best or r.get('cloud_shadow_pct_aoi', 100) < best[r['date']].get('cloud_shadow_pct_aoi', 100):
            best[r['date']] = r
    clear = sorted(d for d, r in best.items()
                   if r.get('coverage_pct', 0) >= min_coverage and r.get('cloud_shadow_pct_aoi', 100) <= max_cloud)
    checked = lambda ds: '; '.join(f"{d}: {best[d].get('cloud_shadow_pct_aoi', 'unread')}% cloud/shadow" for d in ds if d in best)
    replaced = {}
    pre = [d for d in dates_cfg['preferred_pre'] if d in clear]
    pool = [d for d in clear if d < event_date and d <= dates_cfg['max_pre_date'] and d not in pre]
    for d in dates_cfg['preferred_pre']:
        if d not in clear:
            if not pool:
                raise RuntimeError(f'No clear pre-event replacement for {d}. Checked: {checked(sorted(best))}')
            replaced[d] = pool.pop(0)
            pre.append(replaced[d])
    post = dates_cfg['preferred_post']
    if post not in clear:
        later = [d for d in clear if d > event_date]
        if not later:
            listed = sorted(d for d in best if d > event_date)
            raise RuntimeError(f'No clear post-event acquisition. Preferred post {post} and others checked: {checked(listed)}')
        replaced[post] = later[0]
        post = later[0]
    if len(pre) < dates_cfg['min_pre_dates']:
        raise RuntimeError(f'Only {len(pre)} clear pre-event dates; need {dates_cfg["min_pre_dates"]}')
    return {'pre': sorted(pre), 'post': post, 'replaced': replaced, 'source': 'fallback' if replaced else 'preferred',
            'clear_dates': clear}


# ---- network (retrying, byte-capped) ----------------------------------------------------------------------

def aoi_polygon(transform, shape, crs: str) -> dict:
    from pyproj import Transformer
    to_lonlat = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)
    x0, y0 = transform.c, transform.f
    x1, y1 = x0 + transform.a * shape[1], y0 + transform.e * shape[0]
    ring = [to_lonlat.transform(x, y) for x, y in [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]]
    return {'type': 'Polygon', 'coordinates': [[list(p) for p in ring]]}


def search_groups(cfg, transform, shape):
    """Whole-year L2A search on the AOI; same-pass tiles are grouped by (date, platform, relative orbit) as in R2."""
    from risk.r2_imagery import search_items
    im, policy = cfg['phase0']['imagery'], cfg['phase0']['network']
    settings = {**im, 'start': cfg['dates']['audit_start'], 'end': cfg['dates']['audit_end']}
    items = search_items(settings, policy, aoi_polygon(transform, shape, cfg['aoi']['crs']))
    groups = defaultdict(list)
    for item in items:
        p = item['properties']
        groups[(p['datetime'][:10], p.get('platform'), p.get('sat:relative_orbit'))].append(item)
    return items, {k: sorted(v, key=lambda i: i['id']) for k, v in sorted(groups.items(), key=lambda kv: str(kv[0]))}


def audit(cfg, root, groups, transform20, shape20, budget, cache_dir: Path):
    """SCL (20 m) for every item, cached per item; per-group mosaic statistics. Returns (rows, item_rows, errors)."""
    from risk.r2_imagery import read_scl, summarize_scl
    im, policy = cfg['phase0']['imagery'], cfg['phase0']['network']
    scl_dir = cache_dir / 'scl20'
    scl_dir.mkdir(parents=True, exist_ok=True)
    rows, item_rows, errors = [], [], []
    for (date, platform, orbit), items in groups.items():
        layers = []
        for item in items:
            path = scl_dir / f"{item['id']}.npy"
            try:
                if path.is_file():
                    scl = np.load(path)
                else:
                    scl = read_scl(item, {'crs': cfg['aoi']['crs']}, transform20, shape20, policy)
                    np.save(path, scl)
                    budget.check()
                stats = summarize_scl(scl, im['cloud_classes'], im['shadow_classes'], im['invalid_classes'])
                item_rows.append({'id': item['id'], 'date': date, **stats})
                layers.append((scl, ~np.isin(scl, im['invalid_classes'])))
            except Exception as exc:
                errors.append({'id': item['id'], 'error': str(exc).split('?')[0]})
        if not layers:
            continue
        mosaic, _ = mosaic_first_valid(layers)
        rows.append({'date': date, 'platform': platform, 'orbit': orbit, 'item_ids': [i['id'] for i in items],
                     **summarize_scl(mosaic, im['cloud_classes'], im['shadow_classes'], im['invalid_classes'])})
        print(f"{date}: {rows[-1]['cloud_shadow_pct_aoi']:.2f}% cloud/shadow; {rows[-1]['coverage_pct']:.2f}% valid; "
              f"{budget.used()} B", flush=True)
    return rows, item_rows, errors


def scl_mosaic_10m(cfg, items, cache_dir: Path):
    """SCL mosaic of one acquisition group at 10 m (replicated from the cached 20 m item arrays), plus 10 m item layers."""
    invalid = cfg['phase0']['imagery']['invalid_classes']
    per_item = [np.load(cache_dir / 'scl20' / f"{i['id']}.npy") for i in items]
    layers = [(s, ~np.isin(s, invalid)) for s in per_item]
    mosaic, _ = mosaic_first_valid(layers)
    return scl_to_10m(mosaic), [(scl_to_10m(s), scl_to_10m(v.astype('uint8')).astype(bool)) for s, v in layers]


def read_bands(cfg, root, date, items, transform10, shape10, budget, cache_dir: Path):
    """B04,B03,B02,B08 digital numbers at 10 m for one acquisition group -> cache/{date}.npz. Idempotent."""
    import planetary_computer as pc
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT
    from risk.common import retry
    target = cache_dir / f'{date}.npz'
    if target.is_file():
        return target
    policy, rad = cfg['phase0']['network'], cfg['radiometry']
    key = lambda v: tuple(int(x) for x in v.split('.'))
    baselines = {i['properties']['s2:processing_baseline'] for i in items}
    if any(key(b) < key(rad['min_processing_baseline']) for b in baselines):
        raise RuntimeError(f'{date}: processing baseline {sorted(baselines)} < {rad["min_processing_baseline"]}; '
                           'the -1000 offset convention does not apply')
    scl10, per_item = scl_mosaic_10m(cfg, items, cache_dir)
    crs = cfg['aoi']['crs']

    def read(href):
        def go():
            with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', GDAL_HTTP_TIMEOUT=policy['timeout_seconds']):
                with rasterio.open(pc.sign(href)) as src:
                    if src.crs.to_string() != crs:
                        raise RuntimeError(f'{href}: CRS {src.crs} differs from AOI CRS {crs}')
                    with WarpedVRT(src, crs=crs, transform=transform10, width=shape10[1], height=shape10[0],
                                   resampling=Resampling.nearest, nodata=0) as vrt:
                        return vrt.read(1)
        return retry(go, policy, href)

    dn = np.zeros((4, *shape10), dtype='uint16')
    taken = np.zeros(shape10, dtype=bool)
    for item, (_, valid) in zip(items, per_item):
        take = valid & ~taken
        if not take.any():
            continue
        for b, name in enumerate(MODEL_BANDS):
            dn[b][take] = read(item['assets'][name]['href'])[take]
            budget.check()
        taken |= take
    meta = {'date': date, 'items': [i['id'] for i in items], 'band_order': MODEL_BANDS,
            'processing_baselines': sorted(baselines), 'scl_policy': 'nearest neighbour (20 m -> 10 m replication); '
            'first valid tile wins on overlaps', 'bytes_received_so_far': budget.used()}
    np.savez_compressed(target, dn=dn, scl=scl10, taken=taken, meta=json.dumps(meta))
    return target
