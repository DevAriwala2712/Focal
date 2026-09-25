import numpy as np
import pytest


def test_scl_missing_coverage_cannot_be_called_clear():
    from risk.r2_imagery import summarize_scl
    stats = summarize_scl(np.array([[0, 4], [9, 3]], dtype='uint8'), [8, 9, 10], [3], [0, 1])
    assert stats['coverage_pct'] == 75.0
    assert stats['cloud_pct_aoi'] == 25.0
    assert stats['cloud_shadow_pct_aoi'] == 50.0
    assert stats['cloud_pct_valid'] == pytest.approx(100 / 3)


def test_dates_exclude_event_day_and_need_three_distinct_pre_dates():
    from risk.r2_imagery import select_dates
    rows = [dict(date=d, cloud_shadow_pct_aoi=0, coverage_pct=100)
            for d in ['2024-01-01', '2024-01-01', '2024-02-01', '2024-07-30', '2024-08-01']]
    selection = select_dates(rows, '2024-07-30', 10, 99, 3)
    assert selection['pre_dates'] == ['2024-01-01', '2024-02-01']
    assert selection['first_post_date'] == '2024-08-01'
    assert selection['status'] == 'FAIL'


def test_no_clear_post_retains_dates_checked():
    from risk.r2_imagery import select_dates
    result = select_dates([dict(date='2024-08-01', cloud_shadow_pct_aoi=100, coverage_pct=100)],
                          '2024-07-30', 10, 99, 3)
    assert result['first_post_date'] is None
    assert '2024-08-01' in result['reason']


def test_invalid_scl_code_cannot_pass_as_clear_land():
    from risk.r2_imagery import summarize_scl
    with pytest.raises(ValueError, match='SCL'):
        summarize_scl(np.array([[255]], dtype='uint8'), [8, 9, 10], [3], [0, 1])


def test_oom_halves_tiles_but_never_hides_other_errors():
    from risk.common import retry_tiles
    sizes = []
    class OOM(RuntimeError):
        pass
    def operation(size):
        sizes.append(size)
        if size > 64:
            raise OOM('out of memory')
        return 'actual output'
    assert retry_tiles(operation, 256, 64, OOM) == ('actual output', 64, [256, 128])
    assert sizes == [256, 128, 64]
    with pytest.raises(ValueError, match='shape'):
        retry_tiles(lambda s: (_ for _ in ()).throw(ValueError('shape')), 128, 64, OOM)


def test_oom_at_minimum_fails_explicitly():
    from risk.common import retry_tiles
    class OOM(RuntimeError):
        pass
    with pytest.raises(RuntimeError, match='64'):
        retry_tiles(lambda s: (_ for _ in ()).throw(OOM('oom')), 64, 64, OOM)


def test_country_counts_deduplicate_revisits_and_report_border():
    from shapely.geometry import box, mapping
    from risk.r3_worldstrat import count_countries
    countries = {'features': [
        {'properties': {'ADMIN': 'India'}, 'geometry': mapping(box(0, 0, 2, 2))},
        {'properties': {'ADMIN': 'Nepal'}, 'geometry': mapping(box(2, 0, 4, 2))}]}
    rows = [{'': 'A', 'lon': '1', 'lat': '1'}, {'': 'A', 'lon': '1', 'lat': '1'},
            {'': 'B', 'lon': '3', 'lat': '1'}, {'': 'C', 'lon': '2', 'lat': '1'}]
    result = count_countries(rows, countries, ['India', 'Nepal'])
    assert result['india_count'] == 1
    assert result['south_asia_count'] == 2
    assert result['ambiguous_aoi_ids'] == ['C']


def test_training_manifest_does_not_allow_missing_or_synthetic_pairs(tmp_path):
    from risk.r5_finetune import validate_manifest
    with pytest.raises(ValueError, match='WorldStrat'):
        validate_manifest({'dataset': 'synthetic', 'pairs': []}, tmp_path, ['B04','B03','B02','B08'])
    with pytest.raises(ValueError, match='pairs'):
        validate_manifest({'dataset': 'WorldStrat', 'pairs': []}, tmp_path, ['B04','B03','B02','B08'])
