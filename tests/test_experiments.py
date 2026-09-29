"""CPU tests for the six-laws experiment helpers (mechanics only, synthetic inputs)."""
import numpy as np
import pytest


# ---- shared helpers -------------------------------------------------------------

def test_array_hash_binds_dtype_shape_and_bytes():
    from experiments.common import array_sha256
    a = np.arange(6, dtype='float32').reshape(2, 3)
    assert array_sha256(a) == array_sha256(a.copy())
    assert array_sha256(a) != array_sha256(a.reshape(3, 2))
    assert array_sha256(a) != array_sha256(a.astype('float64'))
    b = a.copy()
    b[0, 0] = np.nextafter(b[0, 0], np.float32(1))
    assert array_sha256(a) != array_sha256(b)


def test_array_hash_ignores_memory_layout():
    from experiments.common import array_sha256
    a = np.arange(12, dtype='float32').reshape(3, 4)
    assert array_sha256(a) == array_sha256(np.asfortranarray(a))


def test_result_requires_an_evidence_label():
    from experiments.common import finalize_result
    with pytest.raises(ValueError, match='evidence'):
        finalize_result('e0', {'status': 'PASS'}, 'h', 't0')
    with pytest.raises(ValueError, match='evidence'):
        finalize_result('e0', {'status': 'PASS', 'evidence': 'mixed'}, 'h', 't0')


def test_result_rejects_unknown_status_and_keeps_schema_keys():
    from experiments.common import finalize_result
    with pytest.raises(ValueError, match='status'):
        finalize_result('e0', {'status': 'MAYBE', 'evidence': 'synthetic'}, 'h', 't0')
    out = finalize_result('e0', {'status': 'BLOCKED', 'evidence': 'real', 'reason': 'no data'}, 'h', 't0')
    for key in ('status', 'evidence', 'experiment', 'started_utc', 'finished_utc', 'config_sha256', 'limitations'):
        assert key in out
    assert out['config_sha256'] == 'h'


def test_blocked_exception_becomes_blocked_not_fail():
    from experiments.common import Blocked, run_probe
    def probe(cfg, root):
        raise Blocked('CUDA GPU required', evidence='real')
    out = run_probe('e0', probe, {}, '.', 'h')
    assert out['status'] == 'BLOCKED' and 'CUDA' in out['reason'] and out['evidence'] == 'real'


def test_unexpected_exception_is_a_failure_not_a_block():
    from experiments.common import run_probe
    def probe(cfg, root):
        raise KeyError('bug')
    out = run_probe('e0', probe, {}, '.', 'h')
    assert out['status'] == 'FAIL' and 'KeyError' in out['error']


# ---- E7 streaming moments ----------------------------------------------------------

def test_welford_matches_two_pass_numpy_in_float64():
    from experiments.e7_streaming_moments import Welford
    rng = np.random.default_rng(0)
    stack = rng.normal(0.3, 0.1, size=(8, 5, 7)).astype('float32')
    w = Welford((5, 7))
    for run in stack:
        w.update(run)
    ref = stack.astype('float64')
    assert np.abs(w.mean - ref.mean(0)).max() <= 1e-12
    assert np.abs(w.variance(ddof=0) - ref.var(0)).max() <= 1e-12
    assert np.abs(w.variance(ddof=1) - ref.var(0, ddof=1)).max() <= 1e-12
    assert w.count == 8


def test_welford_survives_large_offset_where_naive_sums_fail():
    from experiments.e7_streaming_moments import Welford
    rng = np.random.default_rng(1)
    stack = (1e7 + rng.normal(0, 1e-2, size=(32, 4, 4))).astype('float64')
    w = Welford((4, 4))
    for run in stack:
        w.update(run)
    naive = (stack ** 2).mean(0) - stack.mean(0) ** 2
    two_pass = stack.var(0)
    assert np.abs(w.variance(0) - two_pass).max() < 1e-9
    assert np.abs(naive - two_pass).max() > 1e-6   # guards against a vacuous test


def test_welford_rejects_wrong_shape_and_underdetermined_variance():
    from experiments.e7_streaming_moments import Welford
    w = Welford((2, 2))
    with pytest.raises(ValueError, match='shape'):
        w.update(np.zeros((3, 3)))
    w.update(np.ones((2, 2)))
    with pytest.raises(ValueError, match='ddof'):
        w.variance(ddof=1)
    assert w.variance(ddof=0).max() == 0


def test_welford_state_does_not_grow_with_run_count():
    from experiments.e7_streaming_moments import Welford
    w = Welford((3, 3))
    for i in range(50):
        w.update(np.full((3, 3), float(i)))
    assert w.mean.shape == (3, 3) and w.m2.shape == (3, 3)
    assert w.mean.dtype == np.float64 and w.m2.dtype == np.float64


def test_e7_measurement_reports_flat_streaming_ram_and_growing_control():
    from experiments.e7_streaming_moments import measure
    cfg = {'run_counts': [8, 32, 96], 'shape': [4, 96, 96], 'ddof': 0,
           'max_abs_error': 1e-6, 'ram_growth_tolerance': 1.5}
    m = measure(cfg, seed=3)
    assert [r['runs'] for r in m['per_run_count']] == [8, 32, 96]
    assert all(r['max_abs_error_mean'] <= 1e-6 and r['max_abs_error_var'] <= 1e-6 for r in m['per_run_count'])
    streaming = [r['streaming_peak_bytes'] for r in m['per_run_count']]
    control = [r['stacked_peak_bytes'] for r in m['per_run_count']]
    assert streaming[-1] <= streaming[0] * 1.5
    assert control[-1] > control[0] * 4      # a stacked reference must visibly grow
    assert m['ram_flat'] is True and m['error_ok'] is True


# ---- E1 tile scheduler -------------------------------------------------------------

def _nearest_x4(batch):
    """Deterministic local operator: each input pixel becomes a 4x4 block."""
    return np.repeat(np.repeat(batch, 4, axis=2), 4, axis=3)


def test_tile_origins_clamp_the_edge_tile_instead_of_padding():
    from experiments.e1_tile_scheduler import tile_origins
    assert tile_origins(128, 128, 96) == [0]
    assert tile_origins(300, 128, 96) == [0, 96, 172]
    assert tile_origins(517, 128, 96) == [0, 96, 192, 288, 384, 389]   # 384+128 < 517 forces a clamped tile
    assert tile_origins(1000, 128, 96)[-1] == 872
    with pytest.raises(ValueError, match='smaller than tile'):
        tile_origins(100, 128, 96)


def test_tile_origins_cover_every_pixel_and_stay_inside_the_image():
    from experiments.e1_tile_scheduler import tile_origins
    for size in (128, 129, 300, 517, 1000):
        for offset in (0, 32):
            origins = tile_origins(size, 128, 96, offset=offset)
            assert origins == sorted(set(origins))
            assert origins[0] == 0 and origins[-1] == size - 128
            covered = np.zeros(size, dtype=bool)
            for o in origins:
                assert 0 <= o <= size - 128
                covered[o:o + 128] = True
            assert covered.all()
            gaps = np.diff(origins)
            assert gaps.size == 0 or gaps.max() <= 96   # neighbouring tiles always overlap by >= 32


def test_tile_lattice_is_anchored_to_the_reference_grid_origin():
    from experiments.e1_tile_scheduler import tile_origins
    # AOI starting on a lattice point keeps the lattice; a shifted AOI moves origins with it.
    assert tile_origins(300, 128, 96, anchor=96) == [0, 96, 172]
    assert tile_origins(300, 128, 96, anchor=10) == [0, 86, 172]
    # offset shifts the lattice, always keeping the clamped edge tiles
    assert tile_origins(300, 128, 96, offset=32) == [0, 32, 128, 172]


def test_feather_taper_only_on_interior_sides_and_strictly_positive():
    from experiments.e1_tile_scheduler import feather_weights
    both = feather_weights(512, 128, True, True)
    assert both.shape == (512,) and (both > 0).all() and both.max() <= 1
    assert both[0] < both[64] < both[128] and both[128] == 1 and both[-1] < both[-65]
    left_open = feather_weights(512, 128, False, True)
    assert left_open[0] == 1 and left_open[-1] < 1
    none = feather_weights(512, 128, False, False)
    assert (none == 1).all()


@pytest.mark.parametrize('size', [128, 300, 517])
@pytest.mark.parametrize('offset', [0, 32])
def test_stitched_tiles_equal_single_pass_for_a_local_operator(size, offset):
    from experiments.e1_tile_scheduler import super_resolve_tiled
    rng = np.random.default_rng(size)
    image = rng.random((4, size, size)).astype('float32')
    out = super_resolve_tiled(image, None, _nearest_x4, tile=128, stride=96, scale=4, feather=32, offset=offset)
    assert out.array.shape == (4, size * 4, size * 4)
    np.testing.assert_allclose(out.array, _nearest_x4(image[None])[0], atol=1e-6)


def test_only_native_128_inputs_reach_the_model_and_edge_tiles_are_not_padded():
    from experiments.e1_tile_scheduler import super_resolve_tiled
    shapes = []
    def op(batch):
        shapes.append(batch.shape)
        return _nearest_x4(batch)
    super_resolve_tiled(np.zeros((4, 517, 517), 'float32'), None, op, tile=128, stride=96, scale=4, feather=32)
    assert set(shapes) == {(1, 4, 128, 128)} and len(shapes) == 36


def test_output_affine_is_input_affine_times_quarter_scale_exactly():
    from affine import Affine
    from experiments.e1_tile_scheduler import super_resolve_tiled
    transform = Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 1300000.0)
    out = super_resolve_tiled(np.zeros((4, 300, 300), 'float32'), transform, _nearest_x4,
                              tile=128, stride=96, scale=4, feather=32)
    assert out.transform == transform * Affine.scale(1 / 4)
    assert (out.transform.a, out.transform.e) == (2.5, -2.5)
    # identical outer bounds: 300 px * 10 m == 1200 px * 2.5 m
    assert 300 * transform.a == out.array.shape[2] * out.transform.a


def test_blend_of_constant_tiles_is_monotone_and_seams_are_smooth():
    from experiments.e1_tile_scheduler import super_resolve_tiled
    counter = iter(range(100))
    def op(batch):
        return np.full((1, 4, 512, 512), float(next(counter)), dtype='float32')
    out = super_resolve_tiled(np.zeros((4, 128, 300), 'float32'), None, op, tile=128, stride=96, scale=4, feather=32)
    row = out.array[0, 100]
    assert row.min() >= 0 and row.max() <= 2
    assert np.abs(np.diff(row)).max() < 0.1     # feathered, not a step


def test_operator_with_wrong_output_shape_is_rejected():
    from experiments.e1_tile_scheduler import super_resolve_tiled
    with pytest.raises(ValueError, match='operator returned'):
        super_resolve_tiled(np.zeros((4, 128, 128), 'float32'), None, lambda b: b, tile=128, stride=96, scale=4, feather=32)


def test_spectral_consistency_is_zero_for_a_block_replicated_image():
    from experiments.e1_tile_scheduler import spectral_consistency
    rng = np.random.default_rng(0)
    image = rng.random((4, 16, 16)).astype('float32')
    sr = np.repeat(np.repeat(image, 4, axis=1), 4, axis=2)
    stats = spectral_consistency(sr, image, 4, ['B04', 'B03', 'B02', 'B08'])
    assert all(stats[b]['rmse'] < 1e-7 and stats[b]['max_abs'] < 1e-6 for b in stats)
    shifted = spectral_consistency(sr + np.array([0.1, 0, 0, 0], 'float32')[:, None, None], image, 4,
                                   ['B04', 'B03', 'B02', 'B08'])
    assert shifted['B04']['rmse'] == pytest.approx(0.1, abs=1e-6) and shifted['B03']['rmse'] < 1e-7
    assert shifted['B04']['bias'] == pytest.approx(0.1, abs=1e-6)


def test_synthetic_scene_is_deterministic_reflectance_like_and_crop_consistent():
    from experiments.common import synthetic_scene
    a, b = synthetic_scene(300, 7), synthetic_scene(300, 7)
    assert a.dtype == np.float32 and a.shape == (4, 300, 300)
    assert np.array_equal(a, b) and not np.array_equal(a, synthetic_scene(300, 8))
    assert a.min() >= 0.0 and a.max() <= 1.0 and a.std() > 0.02
    big = synthetic_scene(517, 7, full=1000)
    assert np.array_equal(big, synthetic_scene(1000, 7)[:, :517, :517])


def test_missing_real_crop_is_blocked_and_names_the_path(tmp_path):
    from experiments.common import Blocked, load_real_crop
    cfg = {'path': 'data/none.tif', 'dn_scale': 10000.0, 'dn_offset': 0.0}
    with pytest.raises(Blocked, match='data/none.tif') as info:
        load_real_crop(cfg, tmp_path)
    assert info.value.evidence == 'real'


def test_real_crop_refuses_unset_radiometry_and_wrong_geometry(tmp_path):
    import rasterio
    from affine import Affine
    from experiments.common import Blocked, load_real_crop
    path = tmp_path / 'crop.tif'
    data = np.full((4, 20, 20), 2000, 'uint16')
    with rasterio.open(path, 'w', driver='GTiff', height=20, width=20, count=4, dtype='uint16',
                       crs='EPSG:32643', transform=Affine(10, 0, 5e5, 0, -10, 1.3e6)) as dst:
        dst.write(data)
    with pytest.raises(Blocked, match='dn_scale'):
        load_real_crop({'path': 'crop.tif', 'dn_scale': None, 'dn_offset': None}, tmp_path)
    array, transform, crs = load_real_crop({'path': 'crop.tif', 'dn_scale': 10000.0, 'dn_offset': -1000.0}, tmp_path)
    assert array.dtype == np.float32 and array.shape == (4, 20, 20)
    assert np.allclose(array, 0.1) and transform.a == 10 and str(crs) == 'EPSG:32643'
    bad = tmp_path / 'coarse.tif'
    with rasterio.open(bad, 'w', driver='GTiff', height=20, width=20, count=4, dtype='uint16',
                       crs='EPSG:32643', transform=Affine(20, 0, 5e5, 0, -20, 1.3e6)) as dst:
        dst.write(data)
    with pytest.raises(Blocked, match='10 m'):
        load_real_crop({'path': 'coarse.tif', 'dn_scale': 10000.0, 'dn_offset': 0.0}, tmp_path)
