"""Offline tile processing from an aligned stack to georeferenced trust outputs."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from risk.common import write_json
from trustsr.change import classify
from trustsr.sr import sr_geometry
from trustsr.trust import ndvi, trust_period


def _write_cog(path, values, crs, transform, descriptions, *, nodata=None):
    values = np.asarray(values)
    if values.ndim == 2:
        values = values[None]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, 'w', driver='COG', width=values.shape[-1], height=values.shape[-2],
                       count=values.shape[0], dtype=values.dtype, crs=crs, transform=transform,
                       nodata=nodata, blocksize=512, compress='DEFLATE', overview_resampling='nearest') as ds:
        ds.write(values)
        for index, name in enumerate(descriptions, 1):
            ds.set_band_description(index, name)
    return str(path.resolve())


def process_stack(stack, output_dir, sr_operator, *, event_date: str, tile_pixels=128, k=2.0,
                  parent_drop_threshold=.1, invalid_scl=(0, 1, 2, 3, 8, 9, 10, 11)):
    """Process one central tile and save COGs plus a provenance manifest."""
    out = Path(output_dir)
    with rasterio.open(stack) as src:
        if src.count % 5 or src.count < 20:
            raise ValueError('Expected at least three pre dates and one post date with RGBN+SCL')
        dates = [src.descriptions[index].split(':')[0] for index in range(0, src.count, 5)]
        if dates != sorted(dates) or len(set(dates)) != len(dates):
            raise ValueError('Stack dates must be distinct and sorted')
        width, height = min(tile_pixels, src.width), min(tile_pixels, src.height)
        window = Window((src.width-width)//2, (src.height-height)//2, width, height)
        transform, crs = src.window_transform(window), src.crs
        data = src.read(window=window).reshape(len(dates), 5, height, width)
    images = [np.asarray(data[i, :4], dtype=np.float32) for i in range(len(dates))]
    masks = [~np.isin(data[i, 4], invalid_scl) for i in range(len(dates))]
    pre = trust_period(images[:-1], sr_operator, masks[:-1])
    post = trust_period(images[-1:], sr_operator, masks[-1:])
    parent_pre = np.mean([ndvi(image) for image in images[:-1]], axis=0)
    parent_post = ndvi(images[-1])
    valid = pre['valid'] & post['valid'] & np.repeat(np.repeat(np.logical_and.reduce(masks), 4, -2), 4, -1)
    classes, counts = classify(pre['ndvi_mean'], post['ndvi_mean'], pre['ndvi_std'], post['ndvi_std'],
                               parent_pre, parent_post, valid, k=k,
                               parent_drop_threshold=parent_drop_threshold)
    fine_transform, _ = sr_geometry(transform, (height, width))
    out.mkdir(parents=True, exist_ok=True)
    bands = ['B04', 'B03', 'B02', 'B08']
    files = {
        'pre_10m': _write_cog(out/'pre_10m.tif', np.mean(images[:-1], axis=0), crs, transform, bands),
        'post_10m': _write_cog(out/'post_10m.tif', images[-1], crs, transform, bands),
        'pre_2p5m': _write_cog(out/'pre_2p5m.tif', pre['image_mean'], crs, fine_transform, bands),
        'post_2p5m': _write_cog(out/'post_2p5m.tif', post['image_mean'], crs, fine_transform, bands),
        'pre_ndvi_mean': _write_cog(out/'pre_ndvi_mean.tif', np.nan_to_num(pre['ndvi_mean'], nan=-9999).astype('float32'), crs, fine_transform, ['NDVI_PRE_MEAN'], nodata=-9999),
        'post_ndvi_mean': _write_cog(out/'post_ndvi_mean.tif', np.nan_to_num(post['ndvi_mean'], nan=-9999).astype('float32'), crs, fine_transform, ['NDVI_POST_MEAN'], nodata=-9999),
        'pre_ndvi_std': _write_cog(out/'pre_ndvi_std.tif', np.nan_to_num(pre['ndvi_std'], nan=-9999).astype('float32'), crs, fine_transform, ['NDVI_PRE_STD'], nodata=-9999),
        'post_ndvi_std': _write_cog(out/'post_ndvi_std.tif', np.nan_to_num(post['ndvi_std'], nan=-9999).astype('float32'), crs, fine_transform, ['NDVI_POST_STD'], nodata=-9999),
        'change': _write_cog(out/'change.tif', classes, crs, fine_transform, ['TRUST_CLASS'], nodata=0),
    }
    stack_manifest = Path(stack).with_suffix('.json')
    source_aoi = json.loads(stack_manifest.read_text(encoding='utf-8')).get('aoi') if stack_manifest.exists() else None
    metadata = {'source_stack': str(Path(stack).resolve()), 'source_aoi': source_aoi,
                'pre_dates': dates[:-1], 'post_date': dates[-1],
                'event_date': event_date,
                'post_event_lag_days': (date.fromisoformat(dates[-1]) - date.fromisoformat(event_date)).days,
                'grid_crs': str(crs), '10m_transform': list(transform)[:6],
                '2p5m_transform': list(fine_transform)[:6], '10m_shape': [height, width],
                'sr_shape': [height*4, width*4], 'classes': {'NO_DATA': 0, 'NO_CHANGE': 1,
                'OBSERVED': 2, 'INFERRED': 3, 'UNSUPPORTED': 4}, 'k': k,
                'parent_drop_threshold': parent_drop_threshold, 'invalid_scl': list(invalid_scl),
                'metrics': counts, 'files': files,
                'caveat': 'SR boundaries are model-inferred; NDVI disturbance is not landslide ground truth.'}
    write_json(out/'manifest.json', metadata)
    return {'files': files, 'metrics': counts, 'manifest': str((out/'manifest.json').resolve())}
