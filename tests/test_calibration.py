import numpy as np


def test_calibration_cog_preserves_grid_and_mask_values(tmp_path):
    import rasterio
    from rasterio.transform import from_origin
    from risk.calibration import write_cog
    data = np.array([[[0, 1], [1, 0]]], dtype='uint8')
    transform = from_origin(500000, 400000, 10, 10)
    path = tmp_path / 'label.tif'
    write_cog(path, data, 'EPSG:32618', transform, ['landslide'], nodata=None)
    with rasterio.open(path) as ds:
        assert ds.tags(ns='IMAGE_STRUCTURE')['LAYOUT'] == 'COG'
        assert ds.crs.to_epsg() == 32618
        assert ds.transform == transform
        assert ds.bounds == (500000, 399980, 500020, 400000)
        assert ds.read().tolist() == [[[0, 1], [1, 0]]]
        assert ds.nodata is None
