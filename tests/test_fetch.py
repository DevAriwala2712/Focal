import numpy as np
import rasterio
from affine import Affine

from trustsr.fetch import choose_acquisitions, parse_radiometry, write_stack, validate_cached_aoi


def test_selects_three_latest_clear_pre_and_first_clear_post():
    rows = [
        {'date': '2024-02-01', 'cloud_shadow_pct_aoi': 1, 'coverage_pct': 100, 'item_ids': ['a']},
        {'date': '2024-03-01', 'cloud_shadow_pct_aoi': 2, 'coverage_pct': 100, 'item_ids': ['b']},
        {'date': '2024-04-01', 'cloud_shadow_pct_aoi': 3, 'coverage_pct': 100, 'item_ids': ['c']},
        {'date': '2024-05-01', 'cloud_shadow_pct_aoi': 4, 'coverage_pct': 100, 'item_ids': ['d']},
        {'date': '2024-08-01', 'cloud_shadow_pct_aoi': 30, 'coverage_pct': 100, 'item_ids': ['e']},
        {'date': '2024-12-06', 'cloud_shadow_pct_aoi': 5, 'coverage_pct': 100, 'item_ids': ['f']},
    ]
    chosen = choose_acquisitions(rows, event_date='2024-07-30', max_cloud=10,
                                  min_coverage=99, min_pre=3, max_pre=3)
    assert [x['date'] for x in chosen] == ['2024-03-01', '2024-04-01', '2024-05-01', '2024-12-06']


def test_no_post_error_lists_checked_cloud_percentages():
    rows = [dict(date=f'2024-0{i}-01', cloud_shadow_pct_aoi=0, coverage_pct=100, item_ids=[str(i)]) for i in (1, 2, 3)]
    rows.append(dict(date='2024-08-01', cloud_shadow_pct_aoi=44, coverage_pct=100, item_ids=['post']))
    import pytest
    with pytest.raises(ValueError, match=r'2024-08-01.*44'):
        choose_acquisitions(rows, event_date='2024-07-30', max_cloud=10,
                            min_coverage=99, min_pre=3, max_pre=3)


def test_radiometry_reads_product_xml_and_rejects_band_varying_offset():
    xml = b'<root><BOA_QUANTIFICATION_VALUE>10000</BOA_QUANTIFICATION_VALUE><BOA_ADD_OFFSET_VALUES_LIST><BOA_ADD_OFFSET band_id="0">-1000</BOA_ADD_OFFSET><BOA_ADD_OFFSET band_id="1">-1000</BOA_ADD_OFFSET></BOA_ADD_OFFSET_VALUES_LIST></root>'
    assert parse_radiometry(xml) == (10000.0, -1000.0)


def test_stack_is_exact_grid_cog_with_named_dates_and_scl(tmp_path):
    grid = Affine.translation(500000, 1200000) * Affine.scale(10, -10)
    values = np.ones((2, 5, 3, 4), dtype=np.float32)
    values[:, 4] = 4
    path = tmp_path / 'stack.tif'
    write_stack(path, values, 'EPSG:32643', grid, ['2024-05-05', '2024-12-06'])
    with rasterio.open(path) as ds:
        assert ds.crs.to_string() == 'EPSG:32643'
        assert ds.transform == grid
        assert ds.bounds == (500000, 1199970, 500040, 1200000)
        assert ds.tags(ns='IMAGE_STRUCTURE')['LAYOUT'] == 'COG'
        assert ds.descriptions == tuple(f'{date}:{band}' for date in ['2024-05-05', '2024-12-06']
                                        for band in ['B04', 'B03', 'B02', 'B08', 'SCL'])


def test_cached_scene_audit_cannot_be_reused_for_another_aoi():
    import pytest
    original = {'type': 'Polygon', 'coordinates': [[[76.1, 11.4], [76.2, 11.4], [76.1, 11.4]]]}
    wrong = {'type': 'Polygon', 'coordinates': [[[76.1, 11.8], [76.2, 11.8], [76.1, 11.8]]]}
    with pytest.raises(ValueError, match='AOI'):
        validate_cached_aoi(original, wrong)
