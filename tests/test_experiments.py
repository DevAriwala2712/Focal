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


# ---- shared NDVI + E2 tile invariance ----------------------------------------------

def test_ndvi_uses_red_and_nir_and_masks_unstable_denominators():
    from experiments.common import ndvi
    a = np.zeros((4, 1, 3), 'float32')
    a[0, 0] = [0.1, 0.2, 0.001]          # B04 red
    a[3, 0] = [0.3, 0.2, 0.001]          # B08 nir
    out = ndvi(a, min_denominator=0.01)
    assert out.dtype == np.float32 and out.shape == (1, 3)
    assert out[0, 0] == pytest.approx(0.5) and out[0, 1] == pytest.approx(0.0)
    assert np.isnan(out[0, 2])


def test_coverage_count_and_seam_lines_follow_the_tile_layout():
    from experiments.e2_tile_invariance import coverage_count, seam_lines
    counts = coverage_count(300, [0, 96, 172], 128)
    assert counts[0] == 1 and counts[96] == 2 and counts[127] == 2 and counts[128] == 1
    assert counts[172] == 2 and counts[223] == 2 and counts[224] == 1 and counts.min() == 1
    # tile-boundary lines inside the image (a hard cut steps here), in 4x output pixels;
    # the borders 0 and 1200 are image edges, not seams
    assert seam_lines(300, [0, 96, 172], 128, 4) == [384, 512, 688, 896]
    assert seam_lines(128, [0], 128, 4) == []


def test_zone_masks_partition_the_output_and_separate_shared_from_different_tiles():
    from experiments.e2_tile_invariance import zone_masks
    z = zone_masks(300, [0, 96, 172], [0, 32, 128, 172], 128, 4)
    assert set(z) == {'interior_different_tile', 'interior_same_tile', 'seam'}
    stacked = np.stack(list(z.values()))
    assert stacked.shape == (3, 1200, 1200) and (stacked.sum(0) == 1).all()   # exact partition
    # positions 0..31 are covered only by tile 0 in both runs: identical computation, not evidence
    assert z['interior_same_tile'][10 * 4, 10 * 4] and not z['interior_different_tile'][10 * 4, 10 * 4]
    # position 160: only tile 96 in run A, only tile 128 in run B -> a genuinely different tile
    assert z['interior_different_tile'][160 * 4, 160 * 4]
    # position 60 lies in one tile of run A but two tiles (origins 0 and 32) of run B -> seam
    assert z['seam'][60 * 4, 60 * 4]


def test_seam_score_detects_a_step_and_ignores_a_smooth_ramp():
    from experiments.e2_tile_invariance import seam_score
    ramp = np.tile(np.linspace(0, 1, 800, dtype='float32'), (4, 800, 1))
    smooth = seam_score(ramp, [], [448])
    assert smooth['ratio'] == pytest.approx(1.0, rel=0.05)
    step = ramp.copy()
    step[:, :, 448:] += 0.2
    stepped = seam_score(step, [], [448])
    assert stepped['ratio'] > 20 and stepped['seam_mean'] > stepped['elsewhere_mean']


def test_delta_stats_report_max_and_p99_over_a_mask():
    from experiments.e2_tile_invariance import delta_stats
    a = np.zeros((4, 10, 10), 'float32')
    b = a.copy()
    b[0, 0, 0] = 0.5
    b[1, 5, 5] = -0.25
    mask = np.ones((10, 10), bool)
    full = delta_stats(a, b, mask)
    assert full['max_abs'] == 0.5 and full['count'] == 400 and 0 <= full['p99_abs'] < 0.5
    mask[0, 0] = False
    assert delta_stats(a, b, mask)['max_abs'] == 0.25
    assert delta_stats(a, b, np.zeros((10, 10), bool))['count'] == 0


def test_seam_score_compares_against_the_same_subpixel_phase():
    from experiments.e2_tile_invariance import seam_score
    x = np.arange(800)
    profile = 0.01 * (x % 4 == 0).astype('float32') * (x > 0)      # block-boundary gradients everywhere
    img = np.cumsum(profile)[None, None, :].repeat(4, 0).repeat(8, 1).astype('float32')
    naive_like = seam_score(img, [], [448], scale=1)                # phase-blind
    phased = seam_score(img, [], [448], scale=4)                    # 448 % 4 == 0: same phase as most steps
    assert phased['ratio'] == pytest.approx(1.0, rel=0.05)
    assert naive_like['ratio'] > 2


def test_tile_bias_operator_cycles_a_known_step_per_tile():
    from experiments.e2_tile_invariance import biased_operator
    op = biased_operator(lambda b: np.zeros((b.shape[0], 4, 8, 8), 'float32'), 0.25, period=3)
    got = [float(op(np.zeros((1, 4, 2, 2), 'float32')).max()) for _ in range(4)]
    assert got == [0.0, 0.25, 0.5, 0.0]


def test_keep_tiles_exposes_raw_per_tile_outputs():
    from experiments.e1_tile_scheduler import super_resolve_tiled
    out = super_resolve_tiled(np.zeros((4, 128, 224), 'float32'), None, _nearest_x4, tile=128, stride=96,
                              scale=4, feather=32, keep_tiles=True)
    assert set(out.tiles) == {(0, 0), (0, 96)} and out.tiles[(0, 0)].shape == (4, 512, 512)
    assert super_resolve_tiled(np.zeros((4, 128, 128), 'float32'), None, _nearest_x4, tile=128, stride=96,
                               scale=4, feather=32).tiles == {}


def test_pair_disagreement_bins_raw_overlap_error_by_distance_to_the_nearer_tile_edge():
    from experiments.e2_tile_invariance import pair_disagreement
    zeros = np.zeros((4, 512, 512), 'float32')
    b = zeros.copy()
    b[:, :, :4] = 1.0                 # first input column of tile B disagrees with tile A
    tiles = {(0, 0): zeros, (0, 96): b}
    rows = pair_disagreement(tiles, 128, 4, [0, 2, 4, 8, 16])
    assert [r['bin'] for r in rows] == ['[0,2)', '[2,4)', '[4,8)', '[8,16)', '[16,inf)']
    assert rows[0]['max_abs'] == 1.0 and all(r['max_abs'] == 0.0 for r in rows[1:4])
    assert rows[4]['count'] == 0
    # overlap is 32 input px = 128 output cols x 512 rows x 4 bands, counted once per direction
    assert sum(r['count'] for r in rows) == 4 * 512 * 128


def test_pair_disagreement_also_covers_vertical_neighbours():
    from experiments.e2_tile_invariance import pair_disagreement
    zeros = np.zeros((4, 512, 512), 'float32')
    b = zeros.copy()
    b[:, :4, :] = 1.0                 # first input row of the lower tile disagrees
    rows = pair_disagreement({(0, 0): zeros, (96, 0): b}, 128, 4, [0, 2, 4, 8, 16])
    assert rows[0]['max_abs'] == 1.0 and all(r['max_abs'] == 0.0 for r in rows[1:4])


# ---- E3 hash-checked reruns ----------------------------------------------------------

def test_seed_everything_makes_numpy_and_torch_repeatable_and_enables_deterministic_mode():
    import torch
    from experiments.e3_hash_reruns import seed_everything
    seed_everything(5)
    a, an = torch.rand(3), np.random.rand(3)
    seed_everything(5)
    assert torch.equal(a, torch.rand(3)) and np.array_equal(an, np.random.rand(3))
    assert torch.are_deterministic_algorithms_enabled()


def test_compare_runs_reports_hashes_identity_and_max_delta():
    from experiments.e3_hash_reruns import compare_runs
    a = np.linspace(0, 1, 64, dtype='float32').reshape(4, 4, 4)
    same = compare_runs([a, a.copy()])
    assert same['identical'] and same['max_abs_delta'] == 0.0 and len(set(same['sha256'])) == 1
    b = a.copy()
    b[0, 0, 0] += 1e-3
    diff = compare_runs([a, b])
    assert not diff['identical'] and diff['max_abs_delta'] == pytest.approx(1e-3, rel=1e-3)
    assert len(set(diff['sha256'])) == 2
    with pytest.raises(ValueError, match='shape'):
        compare_runs([a, a[:, :2]])
    with pytest.raises(ValueError, match='two'):
        compare_runs([a])


def test_seeded_tiled_run_with_a_torch_operator_repeats_bit_for_bit_on_cpu():
    import torch
    from experiments.common import array_sha256, synthetic_scene
    from experiments.e1_tile_scheduler import super_resolve_tiled
    from experiments.e3_hash_reruns import seed_everything

    def run():
        seed_everything(11)
        conv = torch.nn.Conv2d(4, 4, 3, padding=1)
        def op(batch):
            with torch.inference_mode():
                return torch.nn.functional.interpolate(conv(torch.from_numpy(batch)), scale_factor=4,
                                                       mode='bicubic').numpy()
        return super_resolve_tiled(synthetic_scene(200, 1), None, op, tile=128, stride=96, scale=4,
                                   feather=32).array
    assert array_sha256(run()) == array_sha256(run())


# ---- E4 memory gate ---------------------------------------------------------------------

class _FakeMem:
    """Scripted stand-in for torch.cuda peak counters."""
    def __init__(self, script):
        self.script, self.resets = list(script), 0
        self._cur = (0, 0)
    def reset_peak(self):
        self.resets += 1
        self._cur = self.script.pop(0) if self.script else (0, 0)
    def max_allocated(self):
        return self._cur[0]
    def max_reserved(self):
        return self._cur[1]


def test_stage_memory_keeps_the_max_peak_per_stage_across_repeated_entries():
    from experiments.e4_memory_gate import StageMemory
    mem = _FakeMem([(100, 128), (300, 384), (0, 0)])
    rec = StageMemory(mem)
    with rec('forward'):
        pass
    with rec('forward'):
        pass
    with rec('blend'):
        pass
    assert rec.stages['forward'] == {'calls': 2, 'peak_allocated_bytes': 300, 'peak_reserved_bytes': 384}
    assert rec.stages['blend'] == {'calls': 1, 'peak_allocated_bytes': 0, 'peak_reserved_bytes': 0}
    assert mem.resets == 3
    assert rec.overall() == {'peak_allocated_bytes': 300, 'peak_reserved_bytes': 384}


def test_stage_memory_records_a_peak_even_when_the_stage_raises():
    from experiments.e4_memory_gate import StageMemory
    rec = StageMemory(_FakeMem([(7, 9)]))
    with pytest.raises(RuntimeError):
        with rec('forward'):
            raise RuntimeError('oom')
    assert rec.stages['forward']['peak_reserved_bytes'] == 9


def test_memory_gate_verdict_is_strictly_below_the_cap():
    from experiments.e4_memory_gate import gate_verdict
    gib = 2 ** 30
    assert gate_verdict(4 * gib - 1, 4.0) is True
    assert gate_verdict(4 * gib, 4.0) is False          # 'peak reserved < 4 GiB' is strict
    assert gate_verdict(128 * 2 ** 20, 4.0) is True


def test_e4_without_cuda_is_blocked_and_names_the_missing_hardware(tmp_path):
    import torch
    if torch.cuda.is_available():
        pytest.skip('CUDA present; the blocked path is not reachable')
    from experiments.common import run_probe
    from experiments.e4_memory_gate import probe
    out = run_probe('e4', probe, {'e4': {}, 'phase0': {}}, tmp_path, 'h')
    assert out['status'] == 'BLOCKED' and 'CUDA' in out['reason']
    assert out['evidence'] == 'synthetic'
    assert any('never executed on CUDA' in x for x in out['limitations'])


# ---- E8 dihedral 4 vs 8 + shared change helpers ------------------------------------------

def test_spearman_handles_monotone_reversed_and_tied_values():
    from experiments.common import spearman
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert spearman(a, a ** 3) == pytest.approx(1.0)
    assert spearman(a, -a) == pytest.approx(-1.0)
    x = np.array([1.0, 2.0, 2.0, 3.0])
    y = np.array([1.0, 3.0, 2.0, 4.0])
    ranks_x, ranks_y = np.array([1, 2.5, 2.5, 4]), np.array([1, 3, 2, 4])
    assert spearman(x, y) == pytest.approx(np.corrcoef(ranks_x, ranks_y)[0, 1])
    assert np.isnan(spearman(np.ones(5), a))


def test_classify_change_follows_the_design_table_and_no_data_overrides():
    from experiments.common import CLASS_CODES, classify_change
    d = np.array([1.0, 0.1, 1.0, 0.1, 1.0])
    sigma = np.full(5, 0.2)                                  # k=2 -> threshold 0.4
    parent = np.array([True, True, False, False, True])
    no_data = np.array([False, False, False, False, True])
    out = classify_change(d, sigma, parent, no_data, 2.0)
    names = {v: k for k, v in CLASS_CODES.items()}
    assert [names[c] for c in out] == ['OBSERVED', 'INFERRED', 'UNSUPPORTED', 'NO_CHANGE', 'NO_DATA']
    assert set(CLASS_CODES) == {'NO_CHANGE', 'OBSERVED', 'INFERRED', 'UNSUPPORTED', 'NO_DATA'}


def test_classify_change_uses_strict_greater_than_k_sigma():
    from experiments.common import CLASS_CODES, classify_change
    out = classify_change(np.array([0.4]), np.array([0.2]), np.array([False]), np.array([False]), 2.0)
    assert out[0] == CLASS_CODES['NO_CHANGE']                # d == k*sigma is not "d > k*sigma"


def test_dihedral_transform_inverse_is_identity_for_all_eight_and_the_first_four_are_rotations():
    from experiments.e8_dihedral import apply_dihedral, invert_dihedral
    rng = np.random.default_rng(0)
    x = rng.random((4, 6, 6)).astype('float32')
    outs = []
    for t in range(8):
        y = apply_dihedral(x, t)
        assert np.array_equal(invert_dihedral(y, t), x)
        outs.append(y)
    assert all(not np.array_equal(outs[i], outs[j]) for i in range(8) for j in range(i + 1, 8))
    for k in range(4):
        assert np.array_equal(outs[k], np.rot90(x, k, axes=(-2, -1)))
    with pytest.raises(ValueError, match='0..7'):
        apply_dihedral(x, 8)


def test_dihedral_stats_are_zero_sigma_for_an_equivariant_operator():
    from experiments.common import ndvi, synthetic_scene
    from experiments.e8_dihedral import dihedral_ndvi_stats
    tile = synthetic_scene(16, 3)
    for transforms in (range(4), range(8)):
        mean, std, valid = dihedral_ndvi_stats(_nearest_x4, tile, list(transforms), 0.01, 1)
        direct = ndvi(_nearest_x4(tile[None])[0], 0.01)
        assert valid.all() and np.abs(std).max() < 1e-6
        np.testing.assert_allclose(mean, direct, atol=1e-6)


def test_dihedral_stats_show_spread_for_a_non_equivariant_operator():
    from experiments.common import synthetic_scene
    from experiments.e8_dihedral import dihedral_ndvi_stats
    def shifting(batch):                                    # not equivariant: shifts content by one pixel
        return np.roll(_nearest_x4(batch), 1, axis=3)
    _, std, valid = dihedral_ndvi_stats(shifting, synthetic_scene(16, 3), list(range(8)), 0.01, 1)
    assert std[valid].mean() > 1e-3


def test_synthetic_change_pair_has_a_known_changed_square():
    from experiments.common import ndvi, synthetic_change_pair
    pre, post, truth = synthetic_change_pair(64, 5, (20, 24, 16, 16), noise=0.002)
    pre2, post2, truth2 = synthetic_change_pair(64, 5, (20, 24, 16, 16), noise=0.002)
    assert np.array_equal(pre, pre2) and np.array_equal(post, post2) and np.array_equal(truth, truth2)
    assert truth.sum() == 16 * 16 and truth[20:36, 24:40].all()
    drop = ndvi(pre, 0.01) - ndvi(post, 0.01)
    assert np.nanmedian(drop[truth]) > 0.3                  # a strong, known NDVI drop inside the square
    assert np.nanmedian(np.abs(drop[~truth])) < 0.02        # only noise outside it
