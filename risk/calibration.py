"""A small real pre/post/label download proof for R4, not a training dataset."""
from __future__ import annotations

import math
import sqlite3
from datetime import datetime, timedelta

import numpy as np

from risk.common import digest, retry, write_json


def write_cog(path, data, crs, transform, bands, nodata):
    import rasterio
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, 'w', driver='COG', width=data.shape[-1], height=data.shape[-2],
                       count=data.shape[0], dtype=data.dtype, crs=crs, transform=transform,
                       nodata=nodata, compress='DEFLATE', overview_resampling='NEAREST') as dst:
        dst.write(data)
        dst.descriptions = tuple(bands)


def inventory_polygons(path):
    from shapely import from_wkb
    with sqlite3.connect(f'{path.as_uri()}?mode=ro', uri=True) as db:
        table, column, srs = db.execute('SELECT table_name, column_name, srs_id FROM gpkg_geometry_columns').fetchone()
        quote = lambda name: '"' + name.replace('"', '""') + '"'
        rows = db.execute(f'SELECT {quote(column)}, event_date FROM {quote(table)}').fetchall()
    if srs != 4326:
        raise ValueError('This R4 proof expects a WGS84 inventory; reproject explicitly for other inventories')
    result = []
    envelope_sizes = [0, 32, 48, 48, 64]
    for blob, date in rows:
        if blob[:2] != b'GP':
            raise ValueError('Invalid GeoPackage geometry')
        envelope = (blob[3] >> 1) & 7
        if envelope >= len(envelope_sizes):
            raise ValueError('Unsupported GeoPackage envelope')
        result.append((from_wkb(blob[8 + envelope_sizes[envelope]:]), date))
    return result


def download_sample(cfg, root, inventory):
    import planetary_computer as pc
    import rasterio
    from pyproj import Transformer
    from rasterio.enums import Resampling
    from rasterio.features import rasterize
    from rasterio.transform import from_origin
    from rasterio.vrt import WarpedVRT
    from shapely.ops import transform as transform_geometry
    from risk.r2_imagery import aoi_grid, read_scl, search_items, summarize_scl
    settings, policy = cfg['labels'], cfg['network']
    polygons = inventory_polygons(inventory)
    largest, event_time = max(polygons, key=lambda g: g[0].area)
    centre = largest.representative_point()
    event = datetime.fromisoformat(event_time)
    query = {**cfg['imagery'], 'longitude': centre.x, 'latitude': centre.y,
             'crs': settings['sample_crs'], 'aoi_size_m': settings['sample_pixels'] * settings['sample_resolution_m'],
             'audit_resolution_m': settings['sample_resolution_m'],
             'start': (event - timedelta(days=settings['sample_pre_days'])).strftime('%Y-%m-%dT00:00:00Z'),
             'end': (event + timedelta(days=settings['sample_post_days'])).strftime('%Y-%m-%dT23:59:59Z')}
    aoi, transform, shape = aoi_grid(query)
    # Snap the proof window to the metric 10 m lattice; all outputs share it.
    res = settings['sample_resolution_m']
    transform = from_origin(math.floor(transform.c / res) * res, math.ceil(transform.f / res) * res, res, res)
    items = search_items(query, policy, aoi)
    date = event.date().isoformat()
    groups = {label: [i for i in items if (i['properties']['datetime'][:10] < date if label == 'pre'
                                           else i['properties']['datetime'][:10] > date)] for label in ['pre', 'post']}
    checked, chosen, images, masks = [], {}, {}, {}
    for label, candidates in groups.items():
        candidates.sort(key=lambda i: (i['properties'].get('eo:cloud_cover', 100), i['id']))
        for item in candidates[:settings['sample_max_candidates']]:
            scl = read_scl(item, query, transform, shape, policy)
            stats = summarize_scl(scl, query['cloud_classes'], query['shadow_classes'], query['invalid_classes'])
            checked.append({'period': label, 'id': item['id'], **stats})
            if stats['coverage_pct'] < settings['sample_min_coverage_pct'] or stats['cloud_shadow_pct_aoi'] > settings['sample_max_cloud_shadow_pct']:
                continue
            arrays = []
            for band in cfg['model']['bands']:
                href = item['assets'][band]['href']
                def read_band():
                    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', GDAL_HTTP_TIMEOUT=policy['timeout_seconds']):
                        with rasterio.open(pc.sign(href)) as src:
                            with WarpedVRT(src, crs=query['crs'], transform=transform, width=shape[1], height=shape[0],
                                           resampling=Resampling.nearest, nodata=0) as vrt:
                                return vrt.read(1)
                arrays.append(retry(read_band, policy, href))
            images[label] = np.stack(arrays)
            masks[label] = ~np.isin(scl, query['cloud_classes'] + query['shadow_classes'] + query['invalid_classes'])
            masks[label] &= np.all(images[label] != 0, axis=0)
            chosen[label] = item
            break
        if label not in chosen:
            raise RuntimeError(f'No usable {label} sample in checked candidates: {checked}')
    project = Transformer.from_crs('EPSG:4326', query['crs'], always_xy=True).transform
    labels = rasterize([(transform_geometry(project, p), 1) for p, d in polygons if d == event_time],
                       out_shape=shape, transform=transform, fill=0, dtype='uint8')
    valid = masks['pre'] & masks['post']
    positive = int(((labels == 1) & valid).sum())
    if positive == 0:
        raise RuntimeError('Downloaded sample has no valid labelled landslide pixels')
    directory = root / cfg['paths']['cache'] / 'labels' / 'sample'
    for name, array, bands, nodata in [
        ('pre', images['pre'], cfg['model']['bands'], 0), ('post', images['post'], cfg['model']['bands'], 0),
        ('labels', labels[None], ['landslide'], None), ('valid', valid.astype('uint8')[None], ['valid'], None)]:
        write_cog(directory / f'{name}.tif', array, query['crs'], transform, bands, nodata)
    record = {'sample_only': True, 'suitable_for_calibration_alone': False,
              'warning': 'One access-proof crop, not a held-out calibration dataset; polygon inventory negatives can be incomplete.',
              'crs': query['crs'], 'shape': list(shape), 'transform': list(transform), 'event_date': date,
              'items': chosen, 'checked_candidates': checked, 'valid_landslide_pixels': positive,
              'valid_pct': 100 * float(valid.mean()), 'bands': cfg['model']['bands'],
              'radiometry': 'Stored source DN; use asset raster:bands scale/offset before NDVI or learning.',
              'files': {name: {'path': str((directory / f'{name}.tif').relative_to(root)).replace('\\', '/'),
                               'sha256': digest(directory / f'{name}.tif')} for name in ['pre', 'post', 'labels', 'valid']}}
    write_json(root / cfg['paths']['results'] / 'r4_sample.json', record)
    return {'paired_rasters_downloaded': True, 'pre_date': chosen['pre']['properties']['datetime'],
            'post_date': chosen['post']['properties']['datetime'], 'valid_landslide_pixels': positive,
            'valid_pct': record['valid_pct'], 'manifest': 'r4_sample.json', 'sample_only': True}
