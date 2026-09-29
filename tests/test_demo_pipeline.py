import json
import numpy as np
import rasterio
from affine import Affine

from trustsr.fetch import write_stack
from trustsr.pipeline import process_stack


def test_real_grid_processing_writes_aligned_cogs_and_observed_change(tmp_path):
    transform = Affine.translation(500000, 1200000) * Affine.scale(10, -10)
    dates = ['2024-03-01', '2024-04-01', '2024-05-01', '2024-12-06']
    arrays = np.zeros((4, 5, 2, 2), dtype=np.float32)
    arrays[:, 0] = .2
    arrays[:, 3] = .6
    arrays[:, 4] = 4
    arrays[3, 0] = .4
    arrays[3, 3] = .4
    stack = tmp_path / 'stack.tif'
    write_stack(stack, arrays, 'EPSG:32643', transform, dates)
    def upscale(image):
        return np.repeat(np.repeat(image, 4, -2), 4, -1)
    result = process_stack(stack, tmp_path / 'out', upscale, tile_pixels=2, k=2,
                           parent_drop_threshold=.1, invalid_scl=[0, 1, 2, 3, 8, 9, 10, 11])
    with rasterio.open(result['files']['change']) as ds:
        assert ds.crs.to_string() == 'EPSG:32643'
        assert ds.transform == transform * Affine.scale(.25, .25)
        assert ds.bounds == (500000, 1199980, 500020, 1200000)
        assert ds.tags(ns='IMAGE_STRUCTURE')['LAYOUT'] == 'COG'
        assert np.all(ds.read(1) == 2)
    assert result['metrics']['observed_pixels'] == 64
    assert json.loads((tmp_path / 'out' / 'manifest.json').read_text())['post_date'] == '2024-12-06'
