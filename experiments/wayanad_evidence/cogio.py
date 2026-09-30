"""Cloud-Optimized GeoTIFF writer: input CRS and exact affine preserved, internal tiling and overviews."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def write_cog(path, array, transform, crs, *, nodata=None, descriptions=None, tags=None, categorical=False):
    """Write a (bands, rows, cols) or (rows, cols) array as a COG. `categorical` -> nearest-neighbour overviews."""
    import rasterio
    array = np.asarray(array)
    if array.ndim == 2:
        array = array[None]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    predictor = 3 if array.dtype.kind == 'f' else 2
    with rasterio.open(path, 'w', driver='COG', width=array.shape[2], height=array.shape[1], count=array.shape[0],
                       dtype=array.dtype, crs=crs, transform=transform, nodata=nodata, compress='deflate',
                       predictor=predictor, blocksize=256, overview_resampling='nearest' if categorical else 'average',
                       BIGTIFF='IF_SAFER') as dst:
        dst.write(array)
        if descriptions:
            for i, d in enumerate(descriptions, 1):
                dst.set_band_description(i, d)
        if tags:
            dst.update_tags(**{k: str(v) for k, v in tags.items()})
    return path
