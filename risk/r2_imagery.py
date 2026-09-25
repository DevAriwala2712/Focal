"""Audit all annual acquisitions with AOI-level SCL, never catalog cloud cover."""
from __future__ import annotations

import csv
from collections import defaultdict

import numpy as np

from risk.common import request_json, retry, run_cli, write_json


def summarize_scl(scl, cloud_classes, shadow_classes, invalid_classes):
    scl = np.asarray(scl)
    if scl.size == 0:
        raise ValueError('Empty AOI')
    valid = ~np.isin(scl, invalid_classes)
    clouds = np.isin(scl, cloud_classes) & valid
    shadows = np.isin(scl, shadow_classes) & valid
    return {
        'coverage_pct': 100 * float(valid.mean()),
        'cloud_pct_aoi': 100 * float(clouds.mean()),
        'shadow_pct_aoi': 100 * float(shadows.mean()),
        'cloud_shadow_pct_aoi': 100 * float((clouds | shadows).mean()),
        'cloud_pct_valid': 100 * float(clouds.sum() / valid.sum()) if valid.any() else None,
        'aoi_pixels': int(scl.size),
        'valid_pixels': int(valid.sum()),
    }


def select_dates(rows, event_date, max_cloud, min_coverage, min_pre):
    clear = sorted({r['date'] for r in rows if r.get('coverage_pct', 0) >= min_coverage
                    and r.get('cloud_shadow_pct_aoi', 100) <= max_cloud})
    pre = [d for d in clear if d < event_date]
    post = [d for d in clear if d > event_date]
    reasons = []
    if len(pre) < min_pre:
        reasons.append(f'Only {len(pre)} distinct clear pre-event dates; need {min_pre}')
    if not post:
        checked = '; '.join(f"{r['date']}: {r.get('cloud_shadow_pct_aoi', 'unreadable')}% cloud/shadow, "
                            f"{r.get('coverage_pct', 'unknown')}% valid" for r in rows if r['date'] > event_date)
        reasons.append('No clear post-event acquisition among checked dates: ' + (checked or 'none'))
    return {'status': 'FAIL' if reasons else 'PASS', 'pre_dates': pre,
            'first_post_date': post[0] if post else None, 'reason': '; '.join(reasons)}


def aoi_grid(settings):
    from pyproj import Transformer
    from rasterio.transform import from_origin
    to_metric = Transformer.from_crs('EPSG:4326', settings['crs'], always_xy=True)
    to_lonlat = Transformer.from_crs(settings['crs'], 'EPSG:4326', always_xy=True)
    x, y = to_metric.transform(settings['longitude'], settings['latitude'])
    half = settings['aoi_size_m'] / 2
    corners = [(x-half, y-half), (x+half, y-half), (x+half, y+half), (x-half, y+half)]
    polygon = {'type': 'Polygon', 'coordinates': [[list(to_lonlat.transform(*p)) for p in corners + corners[:1]]]}
    resolution = settings['audit_resolution_m']
    side = settings['aoi_size_m'] / resolution
    if not side.is_integer():
        raise ValueError('AOI size must be divisible by audit resolution')
    return polygon, from_origin(x-half, y+half, resolution, resolution), (int(side), int(side))


def search_items(settings, policy, polygon):
    query = dict(collections=[settings['collection']], intersects=polygon,
                 datetime=f"{settings['start']}/{settings['end']}", limit=settings['page_limit'])
    url, body = settings['stac_url'].rstrip('/') + '/search', query
    items, seen_pages = {}, set()
    while url:
        key = (url, str(body))
        if key in seen_pages:
            raise RuntimeError('STAC pagination repeated a page')
        seen_pages.add(key)
        page = request_json(url, policy, body=body)
        for item in page['features']:
            items[item['id']] = item
        nxt = next((link for link in page.get('links', []) if link['rel'] == 'next'), None)
        if not nxt:
            break
        url = nxt['href']
        if nxt.get('method', 'GET') == 'POST':
            body = {**query, **nxt.get('body', {})} if nxt.get('merge') else nxt.get('body', {})
        else:
            body = None
    return list(items.values())


def read_scl(item, settings, transform, shape, policy):
    import planetary_computer as pc
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT
    href = item['assets']['SCL']['href']
    def read():
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', GDAL_HTTP_TIMEOUT=policy['timeout_seconds']):
            with rasterio.open(pc.sign(href)) as src:
                with WarpedVRT(src, crs=settings['crs'], transform=transform, width=shape[1], height=shape[0],
                               resampling=Resampling.nearest, nodata=0) as vrt:
                    return vrt.read(1)
    return retry(read, policy, href)


def probe(cfg, root):
    """Query a whole year, record every item and score acquisition mosaics."""
    settings, policy = cfg['imagery'], cfg['network']
    polygon, transform, shape = aoi_grid(settings)
    items = search_items(settings, policy, polygon)
    out = root / cfg['paths']['results']
    write_json(out / 'r2_catalog.json', {'aoi': polygon, 'items': items})
    groups = defaultdict(list)
    for item in items:
        # Preserve separate orbits/platforms; mosaic only tiles from the same pass.
        props = item['properties']
        key = (props['datetime'][:10], props.get('platform'), props.get('sat:relative_orbit'))
        groups[key].append(item)
    rows, item_rows, errors = [], [], []
    for key, group in sorted(groups.items(), key=lambda kv: str(kv[0])):
        mosaic = np.zeros(shape, dtype='uint8')
        covered = np.zeros(shape, dtype=bool)
        for item in sorted(group, key=lambda i: i['id']):
            try:
                scl = read_scl(item, settings, transform, shape, policy)
                stats = summarize_scl(scl, settings['cloud_classes'], settings['shadow_classes'], settings['invalid_classes'])
                item_rows.append({'id': item['id'], 'date': key[0], **stats})
                valid = ~np.isin(scl, settings['invalid_classes'])
                # First valid tile wins deterministically; never cloud-minimize overlaps.
                take = valid & ~covered
                mosaic[take] = scl[take]
                covered |= valid
            except Exception as exc:
                errors.append({'id': item['id'], 'error': str(exc).split('?')[0]})
        row = {'date': key[0], 'platform': key[1], 'orbit': key[2], 'item_ids': [i['id'] for i in group],
               **summarize_scl(mosaic, settings['cloud_classes'], settings['shadow_classes'], settings['invalid_classes'])}
        rows.append(row)
        print(f"{key[0]}: {row['cloud_shadow_pct_aoi']:.2f}% cloud/shadow; {row['coverage_pct']:.2f}% valid", flush=True)
        write_json(out / 'r2_acquisitions.json', {'rows': rows, 'items': item_rows, 'errors': errors})
    selection = select_dates(rows, settings['event_date'], settings['max_cloud_shadow_pct'],
                             settings['min_coverage_pct'], settings['min_pre_dates'])
    if errors:
        selection.update(status='BLOCKED', reason='Some SCL assets failed; first clear post-event date is not established.')
    if item_rows:
        with (out / 'r2_scenes.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(item_rows[0]))
            writer.writeheader()
            writer.writerows(item_rows)
    return {**selection, 'scene_count': len(items), 'acquisition_count': len(rows), 'errors': errors,
            'aoi': polygon, 'aoi_crs': settings['crs'], 'scl_shape': list(shape),
            'thresholds': settings, 'audit': 'r2_acquisitions.json', 'scene_table': 'r2_scenes.csv'}


if __name__ == '__main__':
    run_cli('r2', probe)
