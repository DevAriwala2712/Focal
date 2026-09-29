"""STAC Sentinel-2 L2A selection and aligned four-band Cloud-Optimized stack."""
from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np
import planetary_computer as pc
import rasterio
import requests
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

from risk.common import load_config, retry, write_json
from risk.r2_imagery import aoi_grid, read_scl, search_items, summarize_scl

BANDS = ('B04', 'B03', 'B02', 'B08')


def validate_cached_aoi(expected, cached):
    if expected.get('type') != 'Polygon' or cached.get('type') != 'Polygon':
        raise ValueError('Cached AOI geometry is not a polygon')
    left, right = np.asarray(expected['coordinates'], dtype=float), np.asarray(cached['coordinates'], dtype=float)
    if left.shape != right.shape or not np.allclose(left, right, rtol=0, atol=1e-8):
        raise ValueError('Cached STAC/SCL audit AOI differs from requested AOI')


def choose_acquisitions(rows, *, event_date, max_cloud, min_coverage, min_pre, max_pre):
    clear = sorted((row for row in rows if row['coverage_pct'] >= min_coverage
                    and row['cloud_shadow_pct_aoi'] <= max_cloud), key=lambda r: r['date'])
    pre = [r for r in clear if r['date'] < event_date]
    post = [r for r in clear if r['date'] > event_date]
    if len({r['date'] for r in pre}) < min_pre:
        raise ValueError(f'Only {len({r["date"] for r in pre})} clear pre-event dates; need {min_pre}')
    if not post:
        checked = '; '.join(f"{r['date']} {r['cloud_shadow_pct_aoi']}% cloud/shadow" for r in rows if r['date'] > event_date)
        raise ValueError(f'No clear post-event acquisition. Checked: {checked or "none"}')
    # One acquisition per date; deterministic tie-break by cloud/coverage/item ID.
    by_date = {}
    for row in pre:
        key = row['date']
        if key not in by_date or (row['cloud_shadow_pct_aoi'], -row['coverage_pct'], row['item_ids']) < (
                by_date[key]['cloud_shadow_pct_aoi'], -by_date[key]['coverage_pct'], by_date[key]['item_ids']):
            by_date[key] = row
    selected_pre = [by_date[d] for d in sorted(by_date)[-max_pre:]]
    post.sort(key=lambda r: (r['date'], r['cloud_shadow_pct_aoi'], -r['coverage_pct'], r['item_ids']))
    return selected_pre + [post[0]]


def parse_radiometry(xml: bytes) -> tuple[float, float]:
    root = ET.fromstring(xml)
    tags = lambda name: [e for e in root.iter() if e.tag.split('}')[-1] == name]
    quant = tags('BOA_QUANTIFICATION_VALUE')
    offsets = tags('BOA_ADD_OFFSET')
    if len(quant) != 1 or not offsets:
        raise ValueError('Product metadata lacks BOA quantification or offset')
    q, values = float(quant[0].text), {float(e.text) for e in offsets}
    if not math.isfinite(q) or q <= 0 or len(values) != 1:
        raise ValueError('Unsupported band-varying or invalid BOA radiometry')
    return q, values.pop()


def write_stack(path, arrays: np.ndarray, crs, transform, dates):
    if arrays.ndim != 4 or arrays.shape[0] != len(dates) or arrays.shape[1] != 5:
        raise ValueError('Stack must be date, four RGBN bands plus SCL, height, width')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, 'w', driver='COG', height=arrays.shape[-2], width=arrays.shape[-1],
                       count=len(dates)*5, dtype='float32', crs=crs, transform=transform,
                       compress='DEFLATE', blocksize=512, overview_resampling='nearest') as ds:
        for i, date in enumerate(dates):
            for j, band in enumerate((*BANDS, 'SCL')):
                index = i*5+j+1
                ds.write(arrays[i, j].astype('float32'), index)
                ds.set_band_description(index, f'{date}:{band}')


def _reference_grid(item, aoi, policy):
    from pyproj import Transformer
    from rasterio.windows import Window, from_bounds
    lon, lat, size_m, crs = aoi
    x, y = Transformer.from_crs('EPSG:4326', crs, always_xy=True).transform(lon, lat)
    box = (x-size_m/2, y-size_m/2, x+size_m/2, y+size_m/2)
    href = item['assets']['B04']['href']
    def read():
        with rasterio.open(pc.sign(href)) as src:
            if src.crs.to_string() != crs:
                from rasterio.warp import transform_bounds
                bounds = transform_bounds(crs, src.crs, *box)
            else:
                bounds = box
            raw = from_bounds(*bounds, transform=src.transform)
            left, top = math.floor(raw.col_off), math.floor(raw.row_off)
            right, bottom = math.ceil(raw.col_off+raw.width), math.ceil(raw.row_off+raw.height)
            window = Window(left, top, right-left, bottom-top)
            return src.crs, src.window_transform(window), (int(window.height), int(window.width))
    return retry(read, policy, href)


def _read_asset(item, band, crs, transform, shape, policy):
    href = item['assets'][band]['href']
    def read():
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', GDAL_HTTP_TIMEOUT=policy['timeout_seconds']):
            with rasterio.open(pc.sign(href)) as src:
                method = Resampling.nearest if band == 'SCL' else Resampling.bilinear
                with WarpedVRT(src, crs=crs, transform=transform, width=shape[1], height=shape[0],
                               resampling=method, nodata=0) as vrt:
                    return vrt.read(1)
    return retry(read, policy, href)


def _radiometry(item, policy):
    href = item['assets']['product-metadata']['href']
    def read():
        response = requests.get(pc.sign(href), timeout=policy['timeout_seconds'])
        response.raise_for_status()
        return parse_radiometry(response.content)
    return retry(read, policy, href)


def _audit(items, settings, policy):
    _, transform, shape = aoi_grid(settings)
    groups = defaultdict(list)
    for item in items:
        props = item['properties']
        groups[(props['datetime'][:10], props.get('platform'), props.get('sat:relative_orbit'))].append(item)
    rows = []
    for key, group in sorted(groups.items()):
        scl = np.zeros(shape, dtype=np.uint8)
        covered = np.zeros(shape, dtype=bool)
        for item in sorted(group, key=lambda i: i['id']):
            part = read_scl(item, settings, transform, shape, policy)
            valid = ~np.isin(part, settings['invalid_classes'])
            take = valid & ~covered
            scl[take] = part[take]
            covered |= valid
        rows.append({'date': key[0], 'platform': key[1], 'orbit': key[2],
                     'item_ids': [i['id'] for i in group], **summarize_scl(scl, settings['cloud_classes'],
                                                                             settings['shadow_classes'], settings['invalid_classes'])})
    return rows


def fetch(aoi, start, end, *, config_path='configs/pipeline.yaml', output=None, audit=None):
    """Fetch aligned L2A RGBN+SCL COG and JSON manifest for (lon,lat,size_m)."""
    cfg, root, _ = load_config(config_path)
    settings = {**cfg['imagery'], 'longitude': aoi[0], 'latitude': aoi[1], 'aoi_size_m': aoi[2],
                'start': start, 'end': end}
    policy = cfg['network']
    if audit:
        catalog = json.loads(Path(audit[0]).read_text(encoding='utf-8'))
        polygon, _, _ = aoi_grid(settings)
        validate_cached_aoi(polygon, catalog['aoi'])
        rows = json.loads(Path(audit[1]).read_text(encoding='utf-8'))['rows']
        items = catalog['items']
    else:
        polygon, _, _ = aoi_grid(settings)
        items = search_items(settings, policy, polygon)
        rows = _audit(items, settings, policy)
    chosen = choose_acquisitions(rows, event_date=settings['event_date'],
                                  max_cloud=settings['max_cloud_shadow_pct'],
                                  min_coverage=settings['min_coverage_pct'],
                                  min_pre=settings['min_pre_dates'], max_pre=cfg['fetch']['max_pre_dates'])
    lookup = {i['id']: i for i in items}
    crs, transform, shape = _reference_grid(lookup[chosen[0]['item_ids'][0]], (*aoi, settings['crs']), policy)
    arrays = np.zeros((len(chosen), 5, *shape), dtype=np.float32)
    manifest_rows = []
    for date_i, row in enumerate(chosen):
        covered = np.zeros(shape, dtype=bool)
        sources = []
        for item_id in row['item_ids']:
            item = lookup[item_id]
            scl = _read_asset(item, 'SCL', crs, transform, shape, policy).astype(np.uint8)
            valid = ~np.isin(scl, settings['invalid_classes']) & ~covered
            if not valid.any():
                continue
            q, offset = _radiometry(item, policy)
            bands = [_read_asset(item, band, crs, transform, shape, policy) for band in BANDS]
            valid &= np.logical_and.reduce([band != 0 for band in bands])
            for band_i, values in enumerate(bands):
                arrays[date_i, band_i, valid] = (values[valid].astype('float32') + offset) / q
            arrays[date_i, 4, valid] = scl[valid]
            covered |= valid
            sources.append({'id': item_id, 'quantification': q, 'offset': offset,
                            'asset_hrefs': {band: item['assets'][band]['href'] for band in (*BANDS, 'SCL')}})
        manifest_rows.append({'date': row['date'], 'cloud_shadow_pct_aoi': row['cloud_shadow_pct_aoi'],
                              'coverage_pct_aoi': row['coverage_pct'], 'sources': sources})
    out = Path(output) if output else root / cfg['fetch']['stack']
    write_stack(out, arrays, crs, transform, [row['date'] for row in chosen])
    manifest = {'stack': str(out), 'crs': str(crs), 'transform': list(transform)[:6],
                'shape': list(shape), 'band_order': list(BANDS), 'quality_band': 'SCL',
                'aoi': list(aoi), 'event_date': settings['event_date'], 'dates': manifest_rows}
    write_json(out.with_suffix('.json'), manifest)
    return out
