"""CPU tests for the Wayanad evidence experiment (tiny synthetic rasters, no network, no model)."""
import numpy as np
import pytest
from affine import Affine


# ---- geometry -----------------------------------------------------------------------

def test_sr_affine_is_input_affine_times_quarter_with_equal_bounds():
    from experiments.wayanad_evidence.geo import bounds, sr_grid
    transform = Affine(10.0, 0.0, 612340.0, 0.0, -10.0, 1272280.0)
    shape = (256, 384)
    sr_transform, sr_shape = sr_grid(transform, shape, 4)
    assert sr_transform == transform * Affine.scale(1 / 4)
    assert (sr_transform.a, sr_transform.e) == (2.5, -2.5)
    assert sr_shape == (1024, 1536)
    assert bounds(sr_transform, sr_shape) == bounds(transform, shape)


def test_reference_grid_is_snapped_and_tile_divisible():
    from experiments.wayanad_evidence.geo import reference_grid
    aoi = {'longitude': 76.160, 'latitude': 11.490, 'crs': 'EPSG:32643', 'size_m': 10240, 'snap_m': 20}
    transform, shape = reference_grid(aoi)
    assert shape == (1024, 1024)
    assert transform.a == 10.0 and transform.e == -10.0
    assert transform.c % 20 == 0 and transform.f % 20 == 0
    # the AOI centre lies inside, within one snap step of the middle
    from pyproj import Transformer
    x, y = Transformer.from_crs('EPSG:4326', 'EPSG:32643', always_xy=True).transform(76.160, 11.490)
    assert abs((transform.c + 5120) - x) <= 20 and abs((transform.f - 5120) - y) <= 20


def test_snap_crop_snaps_outward_to_tile_grid_and_clamps():
    from experiments.wayanad_evidence.geo import snap_crop
    assert snap_crop((130, 300, 5, 129), (1024, 1024), 128) == (128, 384, 0, 256)
    assert snap_crop((0, 1024, 0, 1024), (1024, 1024), 128) == (0, 1024, 0, 1024)
    assert snap_crop((1000, 1024, 1000, 1024), (1024, 1024), 128) == (896, 1024, 896, 1024)


def test_dilate_square_grows_by_exact_chebyshev_radius():
    from experiments.wayanad_evidence.geo import dilate_square
    m = np.zeros((21, 21), bool)
    m[10, 10] = True
    out = dilate_square(m, 3)
    assert out.sum() == 49 and out[7:14, 7:14].all()
    assert dilate_square(m, 0).sum() == 1
    edge = np.zeros((5, 5), bool)
    edge[0, 0] = True
    assert dilate_square(edge, 2).sum() == 9          # clipped at the border, no wrap-around


def test_label_components_matches_reference_flood_fill():
    from experiments.wayanad_evidence.geo import label_components
    rng = np.random.default_rng(3)
    for connectivity in (4, 8):
        mask = rng.random((40, 50)) < 0.4
        labels, n = label_components(mask, connectivity)
        assert (labels > 0).sum() == mask.sum() and (labels[~mask] == 0).all()
        # reference: brute-force flood fill
        seen = np.zeros(mask.shape, bool)
        steps = [(1, 0), (-1, 0), (0, 1), (0, -1)] + ([(1, 1), (1, -1), (-1, 1), (-1, -1)] if connectivity == 8 else [])
        ref = 0
        for r, c in zip(*np.nonzero(mask)):
            if seen[r, c]:
                continue
            ref += 1
            stack = [(r, c)]
            seen[r, c] = True
            ids = set()
            while stack:
                y, x = stack.pop()
                ids.add(labels[y, x])
                for dy, dx in steps:
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < mask.shape[0] and 0 <= xx < mask.shape[1] and mask[yy, xx] and not seen[yy, xx]:
                        seen[yy, xx] = True
                        stack.append((yy, xx))
            assert len(ids) == 1                         # one flood-fill component carries exactly one label
        assert n == ref


# ---- tiler --------------------------------------------------------------------------

@pytest.mark.parametrize('n', [128, 256, 384, 640, 1024])
def test_axis_plan_keeps_partition_every_pixel_once(n):
    from experiments.wayanad_evidence.tiler import axis_plan
    plan = axis_plan(n, tile=128, margin=16, stride=96)
    covered = np.zeros(n, int)
    for start, lo, hi in plan:
        assert 0 <= start <= n - 128                     # edge tiles are clamped, never padded
        assert start <= lo < hi <= start + 128
        if lo > 0:
            assert lo - start >= 16                      # at least 16 input px discarded per side...
        if hi < n:
            assert start + 128 - hi >= 16                # ...except at scene edges
        covered[lo:hi] += 1
    assert (covered == 1).all()


def test_tiler_covers_every_pixel_once_and_sends_only_128_inputs():
    from experiments.wayanad_evidence.tiler import stitch
    rng = np.random.default_rng(1)
    image = rng.random((4, 384, 640)).astype('float32')
    seen_shapes, writes = [], []

    def fake_model(batch):                              # deterministic nearest-neighbour x4, records what it is sent
        seen_shapes.append(tuple(batch.shape))
        return np.repeat(np.repeat(batch, 4, axis=-2), 4, axis=-1)

    out = stitch(image, fake_model, tile=128, margin=16, stride=96, scale=4, write_log=writes)
    assert out.shape == (4, 384 * 4, 640 * 4)
    assert set(s[-2:] for s in seen_shapes) == {(128, 128)}
    cover = np.zeros(out.shape[-2:], int)
    for r0, r1, c0, c1 in writes:
        cover[r0:r1, c0:c1] += 1
    assert (cover == 1).all()                            # each output pixel written exactly once
    np.testing.assert_array_equal(out, np.repeat(np.repeat(image, 4, axis=1), 4, axis=2))   # == single pass


def test_tiler_order_is_fixed_row_major():
    from experiments.wayanad_evidence.tiler import tile_plan
    a = tile_plan((384, 640), 128, 16, 96)
    b = tile_plan((384, 640), 128, 16, 96)
    assert a == b
    keys = [(t['r0'], t['c0']) for t in a]
    assert keys == sorted(keys)


def test_tiler_rejects_image_smaller_than_one_tile():
    from experiments.wayanad_evidence.tiler import tile_plan
    with pytest.raises(ValueError, match='128'):
        tile_plan((100, 256), 128, 16, 96)


# ---- dihedral and Welford -----------------------------------------------------------

def test_dihedral_transform_then_inverse_is_identity():
    from experiments.wayanad_evidence.stats import dihedral, inverse_dihedral
    rng = np.random.default_rng(0)
    x = rng.random((3, 5, 7))                            # non-square on purpose: rotations swap axes
    seen = set()
    for k in range(8):
        y = dihedral(x, k)
        np.testing.assert_array_equal(inverse_dihedral(y, k), x)
        seen.add(y.tobytes() + str(y.shape).encode())
    assert len(seen) == 8                                # eight genuinely different variants


def test_welford_matches_two_pass_numpy_within_1e_10():
    from experiments.wayanad_evidence.stats import Welford
    rng = np.random.default_rng(5)
    samples = rng.normal(0.5, 0.2, size=(24, 9, 11)) + 1e3      # large offset stresses naive sum-of-squares
    w = Welford((9, 11))
    for s in samples:
        w.update(s)
    np.testing.assert_allclose(w.mean, samples.mean(axis=0), rtol=0, atol=1e-10)
    np.testing.assert_allclose(w.std(ddof=1), samples.std(axis=0, ddof=1), rtol=0, atol=1e-10)
    assert (w.count == 24).all() and w.mean.dtype == np.float64


def test_welford_skips_nan_samples_per_pixel():
    from experiments.wayanad_evidence.stats import Welford
    rng = np.random.default_rng(6)
    samples = rng.normal(size=(10, 6, 6))
    samples[rng.random(samples.shape) < 0.3] = np.nan
    w = Welford((6, 6))
    for s in samples:
        w.update(s)
    with np.errstate(invalid='ignore'):
        np.testing.assert_allclose(w.mean, np.nanmean(samples, axis=0), atol=1e-12, equal_nan=True)
    ok = w.count >= 2
    np.testing.assert_allclose(w.std(ddof=1)[ok], np.nanstd(samples, axis=0, ddof=1)[ok], atol=1e-12)
    assert np.isnan(w.std(ddof=1)[~ok]).all()


# ---- fetch helpers ------------------------------------------------------------------

def _rows(spec):
    return [{'date': d, 'coverage_pct': 100.0, 'cloud_shadow_pct_aoi': c} for d, c in spec]


_DATES = {'preferred_pre': ['2024-01-11', '2024-01-21', '2024-01-31'], 'preferred_post': '2024-12-06',
          'min_pre_dates': 3, 'max_pre_date': '2024-05-31'}


def test_choose_dates_keeps_preferred_when_clear():
    from experiments.wayanad_evidence.fetch import choose_dates
    rows = _rows([('2024-01-11', 0), ('2024-01-21', 5), ('2024-01-31', 1), ('2024-02-05', 0), ('2024-12-06', 0.1)])
    out = choose_dates(rows, _DATES, '2024-07-30', 10, 99)
    assert out['pre'] == _DATES['preferred_pre'] and out['post'] == '2024-12-06' and out['source'] == 'preferred'


def test_choose_dates_replaces_only_failing_dates_by_earliest_clear():
    from experiments.wayanad_evidence.fetch import choose_dates
    rows = _rows([('2024-01-06', 0), ('2024-01-11', 0), ('2024-01-21', 60), ('2024-01-31', 1), ('2024-02-05', 0),
                  ('2024-08-13', 30), ('2024-12-06', 50), ('2024-12-16', 2), ('2024-12-21', 0)])
    out = choose_dates(rows, _DATES, '2024-07-30', 10, 99)
    assert out['pre'] == ['2024-01-11', '2024-01-31', '2024-01-06'] or out['pre'] == ['2024-01-06', '2024-01-11', '2024-01-31']
    assert out['post'] == '2024-12-16' and out['source'] == 'fallback'
    assert out['replaced']['2024-01-21'] == '2024-01-06'                    # earliest clear pre-event date not already chosen
    assert out['replaced']['2024-12-06'] == '2024-12-16'


def test_choose_dates_raises_naming_dates_and_cloud_when_no_clear_post():
    from experiments.wayanad_evidence.fetch import choose_dates
    rows = _rows([('2024-01-11', 0), ('2024-01-21', 0), ('2024-01-31', 0), ('2024-12-06', 55.5), ('2024-12-11', 80)])
    with pytest.raises(RuntimeError, match=r'2024-12-06.*55\.5'):
        choose_dates(rows, _DATES, '2024-07-30', 10, 99)


def test_mosaic_first_valid_tile_wins_and_is_deterministic():
    from experiments.wayanad_evidence.fetch import mosaic_first_valid
    a, b = np.full((2, 3), 1, 'uint16'), np.full((2, 3), 2, 'uint16')
    va = np.array([[1, 1, 0], [0, 0, 0]], bool)
    vb = np.array([[1, 0, 1], [1, 0, 0]], bool)
    out, taken = mosaic_first_valid([(a, va), (b, vb)])
    np.testing.assert_array_equal(out, [[1, 1, 2], [2, 0, 0]])
    np.testing.assert_array_equal(taken, [[1, 1, 1], [1, 0, 0]])


def test_reflectance_applies_baseline_offset_and_clips_at_zero():
    from experiments.wayanad_evidence.fetch import reflectance
    rad = {'quantification': 10000.0, 'boa_add_offset': -1000.0, 'clip_min': 0.0}
    out = reflectance(np.array([0, 999, 1000, 3500, 11000], 'uint16'), rad)
    np.testing.assert_allclose(out, [0.0, 0.0, 0.0, 0.25, 1.0])
    assert out.dtype == np.float32


def test_scl_at_10m_is_nearest_neighbour_replication_of_the_20m_grid():
    from experiments.wayanad_evidence.fetch import scl_to_10m
    scl20 = np.array([[4, 9], [3, 5]], 'uint8')
    np.testing.assert_array_equal(scl_to_10m(scl20), np.kron(scl20, np.ones((2, 2), 'uint8')))


def test_write_cog_preserves_crs_transform_nodata_and_is_a_real_cog(tmp_path):
    import rasterio
    from experiments.wayanad_evidence.cogio import write_cog
    transform = Affine(2.5, 0.0, 612340.0, 0.0, -2.5, 1272280.0)
    data = (np.arange(3 * 600 * 700) % 251).astype('uint8').reshape(3, 600, 700)
    path = tmp_path / 'x.tif'
    write_cog(path, data, transform, 'EPSG:32643', nodata=255, descriptions=['a', 'b', 'c'],
              tags={'label': 'test'}, categorical=True)
    with rasterio.open(path) as src:
        assert src.crs.to_epsg() == 32643 and src.transform == transform
        assert src.nodata == 255 and src.count == 3 and src.descriptions == ('a', 'b', 'c')
        assert src.tags()['label'] == 'test'
        assert src.profile['tiled'] and src.overviews(1)
        np.testing.assert_array_equal(src.read(), data)
        assert rasterio.transform.array_bounds(src.height, src.width, src.transform) == \
            rasterio.transform.array_bounds(600, 700, transform)


def test_numpy_mask_families_equal_upstream_tricks():
    torch = pytest.importorskip('torch')
    from sen2sr.models import tricks
    from experiments.wayanad_evidence.mask_check import families
    shape, radius = (32, 32), 8
    ours = families(shape, radius)
    np.testing.assert_allclose(ours['ideal'](), tricks.ideal_filter(shape, radius).numpy(), atol=1e-6)
    # upstream gaussian_filter cannot run in the pinned sen2sr (torch.exp receives a Python float): compare with a
    # scalar loop of the formula in its source, exp(-((u-c)^2 + (v-c)^2) / (2 cutoff^2))
    with pytest.raises(TypeError):
        tricks.gaussian_filter(shape, radius)
    import math
    ref = np.array([[math.exp(-((u - 16) ** 2 + (v - 16) ** 2) / (2 * radius ** 2)) for v in range(32)]
                    for u in range(32)], dtype=np.float32)
    np.testing.assert_allclose(ours['gaussian'](), ref, atol=1e-6)
    np.testing.assert_allclose(ours['butterworth'](3), tricks.butterworth_filter(shape, radius, 3).numpy(), atol=1e-6)
    with pytest.raises(TypeError):                       # same upstream bug in sigmoid_filter
        tricks.sigmoid_filter(shape, radius, 2.5)
    ref = np.array([[1 / (1 + math.exp((math.hypot(u - 16, v - 16) - radius) / 2.5)) for v in range(32)]
                    for u in range(32)], dtype=np.float32)
    np.testing.assert_allclose(ours['sigmoid'](2.5), ref, atol=1e-6)


def test_fit_picks_the_generating_family():
    from experiments.wayanad_evidence.mask_check import families, fit_families
    fam = families((64, 64), 16)
    for name, param in (('ideal', None), ('gaussian', None), ('butterworth', 4), ('sigmoid', 3.0)):
        mask = fam[name]() if param is None else fam[name](param)
        fit = fit_families(mask, 16)
        assert fit['best_family'] == name and fit['fits'][name]['max_abs_diff'] < 1e-6
    binary = fam['ideal']()
    assert fit_families(binary, 16)['is_binary'] is True


def test_radial_profile_of_isotropic_mask_is_monotone_for_gaussian():
    from experiments.wayanad_evidence.mask_check import families, radial_profile
    prof = radial_profile(families((64, 64), 16)['gaussian']())
    assert prof[0] == pytest.approx(1.0) and (np.diff(prof[:30]) <= 1e-12).all()


class _FakeOOM(Exception):
    pass


def test_oom_policy_retries_once_at_batch_one_after_emptying_cache():
    from experiments.wayanad_evidence.sr import infer_with_oom_policy
    calls, cleaned = [], []

    def fn(batch):
        calls.append(len(batch))
        if len(batch) > 1:
            raise _FakeOOM('out of memory')
        return np.repeat(np.repeat(batch, 4, axis=-2), 4, axis=-1)

    out = infer_with_oom_policy(fn, np.ones((8, 4, 128, 128), 'float32'), 'tile r=0 c=96', _FakeOOM,
                                cleanup=lambda: cleaned.append(1), memory=lambda: {})
    assert out.shape == (8, 4, 512, 512) and calls == [8] + [1] * 8 and cleaned == [1]


def test_oom_policy_fails_naming_the_tile_when_batch_one_also_ooms():
    from experiments.wayanad_evidence.sr import infer_with_oom_policy

    def fn(batch):
        raise _FakeOOM('out of memory')

    with pytest.raises(RuntimeError, match=r'tile r=32 c=64.*batch 1'):
        infer_with_oom_policy(fn, np.ones((8, 4, 128, 128), 'float32'), 'tile r=32 c=64', _FakeOOM,
                              cleanup=lambda: None, memory=lambda: {'allocated_mib': 1.0})


def test_welford_update_at_region_equals_full_array_update():
    from experiments.wayanad_evidence.stats import Welford
    rng = np.random.default_rng(4)
    samples = rng.normal(size=(6, 10, 12))
    full, part = Welford((10, 12)), Welford((10, 12))
    for s in samples:
        full.update(s)
        part.update_at((slice(0, 5), slice(0, 12)), s[:5])
        part.update_at((slice(5, 10), slice(0, 12)), s[5:])
    np.testing.assert_array_equal(full.mean, part.mean)
    np.testing.assert_array_equal(full.m2, part.m2)


def test_sr_variants_of_an_equivariant_model_are_identical_after_inversion():
    from experiments.wayanad_evidence.sr import sr_variants
    rng = np.random.default_rng(9)
    tile = rng.random((4, 128, 128)).astype('float32')
    nearest = lambda b: np.repeat(np.repeat(b, 4, axis=-2), 4, axis=-1)
    out = sr_variants(nearest, tile, 'tile r=0 c=0', _FakeOOM, cleanup=lambda: None, memory=lambda: {})
    assert out.shape == (8, 4, 512, 512)
    for k in range(8):
        np.testing.assert_array_equal(out[k], out[0])


def test_sr_variants_apply_the_transform_to_the_input_only():
    from experiments.wayanad_evidence.sr import sr_variants
    from experiments.wayanad_evidence.stats import dihedral
    seen = []
    model = lambda b: (seen.append(b.copy()), np.repeat(np.repeat(b, 4, axis=-2), 4, axis=-1))[1]
    tile = np.arange(4 * 128 * 128, dtype='float32').reshape(4, 128, 128)
    sr_variants(model, tile, 't', _FakeOOM, cleanup=lambda: None, memory=lambda: {})
    for k in range(8):
        np.testing.assert_array_equal(seen[0][k], dihedral(tile, k))


def test_footprint_valid_fraction_counts_only_footprint_pixels():
    from experiments.wayanad_evidence.step2_visibility import footprint_valid_fraction
    im = {'cloud_classes': [8, 9, 10], 'shadow_classes': [2, 3], 'invalid_classes': [0, 1, 11]}
    scl = np.full((10, 10), 4, 'uint8')
    scl[0:2, :] = 9                                   # cloud outside the footprint: must not matter
    footprint = np.zeros((10, 10), bool)
    footprint[4:8, 4:8] = True
    assert footprint_valid_fraction(scl, footprint, im) == (1.0, 16, 16)
    scl[4:6, 4:8] = 3                                 # shadow covering half of the footprint
    assert footprint_valid_fraction(scl, footprint, im) == (0.5, 8, 16)


def test_step3_glue_with_equivariant_fake_model_gives_zero_sigma_and_exact_ndvi(tmp_path):
    """run_sr end to end on a 256x256 crop with a nearest-neighbour 'model': moments, counts, spectral error, COG grid."""
    from types import SimpleNamespace
    import rasterio
    from experiments.wayanad_evidence.gate import upsample
    from experiments.wayanad_evidence.geo import reference_grid
    from experiments.wayanad_evidence.stats import ndvi
    from experiments.wayanad_evidence.step3_sr import run_sr

    cfg = {'tiling': {'tile': 128, 'crop_margin_px': 16, 'stride': 96, 'scale': 4}, 'radiometry': {'min_denominator': 0.05},
           'change': {'ddof': 1}, 'seed': 1, 'model': {'device': 'cpu', 'deterministic': True},
           'aoi': {'crs': 'EPSG:32643', 'longitude': 76.16, 'latitude': 11.49, 'size_m': 10240, 'snap_m': 20},
           'labels': {'model': 'pretrained SEN2SR-lite (NOT fine-tuned)'}}

    class FakeRunner:
        oom_type = _FakeOOM
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        cleanup = staticmethod(lambda: None)
        memory = staticmethod(lambda: {'peak_rss_mib': 1.0})

        def __call__(self, batch):
            assert batch.shape[-2:] == (128, 128)
            return np.repeat(np.repeat(batch, 4, axis=-2), 4, axis=-1)

    rng = np.random.default_rng(12)
    dates = ['d1', 'd2', 'd3', 'd4']
    arrays = {}
    for d in dates:
        refl = rng.uniform(0.05, 0.4, (4, 256, 256)).astype('float32')
        arrays[d] = {'refl': refl, 'valid_scl': np.ones((256, 256), bool)}
    state = {'dates': np.array(dates), 'pre': np.array(dates[:3]), 'post': np.array('d4'),
             'valid_all': np.ones((1024, 1024), bool), 'crop': np.array([0, 256, 0, 256])}
    out, cache = tmp_path / 'out', tmp_path / 'cache'
    out.mkdir(), cache.mkdir()
    res = run_sr(cfg, tmp_path, 'hash', FakeRunner(), arrays, state, out, cache)
    assert res['tiles_per_date'] == 9 and res['forward_passes_total'] == 9 * 8 * 4
    assert res['determinism']['tile_rerun_bytes_identical'] is True
    with np.load(cache / 'step3_state.npz') as z:
        assert (z['pre_count'] == 24).all() and (z['post_count'] == 8).all()
        nd = [ndvi(arrays[d]['refl'][0], arrays[d]['refl'][3], 0.05) for d in dates]
        np.testing.assert_allclose(z['post_mean'], upsample(nd[3], 4), atol=1e-12)
        np.testing.assert_allclose(z['pre_mean'], upsample(np.mean(nd[:3], axis=0), 4), atol=1e-12)
        assert z['post_std'].max() < 1e-12                                   # 8 identical runs -> sigma 0
        # pre pool = 8 copies of each of 3 dates: var(ddof=1) = 16/23 * (cross-date var, ddof=1)
        expected = np.sqrt(16 / 23) * np.std(nd[:3], axis=0, ddof=1)
        np.testing.assert_allclose(z['pre_std'], upsample(expected, 4), atol=1e-12)
    for d in dates:
        assert max(res['spectral_consistency'][d]['per_band_mae_identity_run'].values()) < 1e-6
    with rasterio.open(out / 'sr_post_2p5m.tif') as sr:
        base, _ = reference_grid(cfg['aoi'])
        assert sr.transform == base * Affine.scale(1 / 4) and (sr.height, sr.width) == (1024, 1024)


# ---- trust gate ---------------------------------------------------------------------

def _scene():
    """64x64 10 m scene. Known 16x16 square NDVI drop, an INFERRED-only block, a masked corner, an SR-only artefact."""
    from experiments.wayanad_evidence.gate import upsample
    rng = np.random.default_rng(11)
    pre10 = np.full((64, 64), 0.80)
    post10 = pre10.copy()
    post10[20:36, 20:36] = 0.20                        # known square: drop 0.60
    post10[40, 40] = 0.40                              # parent-visible, but the SR does not see it: INFERRED
    nodata10 = np.zeros((64, 64), bool)
    nodata10[20:22, 20:22] = True                      # cloud over one corner of the square
    d = upsample(pre10 - post10, 4) + rng.normal(0, 0.01, (256, 256))
    d[40 * 4:41 * 4, 40 * 4:41 * 4] = 0.05             # SR mean drop small here (<= k*sigma)
    d[200:203, 200:203] = 0.50                         # injected SR-only artefact; 10 m data show no change there
    sigma = np.full((256, 256), 0.05)
    return pre10, post10, nodata10, d, sigma


def test_synthetic_scene_square_observed_artefact_unsupported_masked_nodata():
    from experiments.wayanad_evidence.gate import (NO_CHANGE, NO_DATA, OBSERVED, INFERRED, UNSUPPORTED, classify,
                                                   parent_mask, upsample)
    pre10, post10, nodata10, d, sigma = _scene()
    parent10 = parent_mask(pre10 - post10, ~nodata10, 0.30)
    cls = classify(d, sigma, upsample(parent10, 4), upsample(nodata10, 4), k=2.0)
    square = np.zeros((256, 256), bool)
    square[80:144, 80:144] = True
    masked = upsample(nodata10, 4)
    assert (cls[masked] == NO_DATA).all()                                   # masked pixels are NO_DATA, overriding all
    assert (cls[square & ~masked] == OBSERVED).all()                        # known square (drop 0.6 >> 2 sigma) is OBSERVED
    assert (cls[200:203, 200:203] == UNSUPPORTED).all()                     # SR-only artefact
    assert (cls[160:164, 160:164] == INFERRED).all()                        # parent change, SR mean below k*sigma
    outside = ~square & ~masked
    outside[200:203, 200:203] = False
    outside[160:164, 160:164] = False
    assert (cls[outside] == NO_CHANGE).all()


def test_nodata_overrides_every_class_and_nan_is_nodata():
    from experiments.wayanad_evidence.gate import NO_DATA, classify
    d = np.array([[0.9, 0.9], [np.nan, 0.0]])
    sigma = np.full((2, 2), 0.05)
    parent = np.array([[True, False], [True, False]])
    nodata = np.array([[True, True], [False, False]])
    cls = classify(d, sigma, parent, nodata, k=2.0)
    assert cls[0, 0] == NO_DATA and cls[0, 1] == NO_DATA and cls[1, 0] == NO_DATA


def test_classify_uses_strict_inequality_d_greater_than_k_sigma():
    from experiments.wayanad_evidence.gate import INFERRED, OBSERVED, classify
    d, sigma = np.array([[0.10, 0.1000001]]), np.array([[0.05, 0.05]])
    cls = classify(d, sigma, np.array([[True, True]]), np.zeros((1, 2), bool), k=2.0)
    assert cls[0, 0] == INFERRED and cls[0, 1] == OBSERVED


def test_rho_is_one_for_block_constant_delta_and_zero_for_zero_mean_blocks():
    from experiments.wayanad_evidence.gate import object_rho
    rng = np.random.default_rng(2)
    block_values = rng.uniform(0.2, 0.9, size=(2, 2))
    const = np.kron(block_values, np.ones((4, 4)))                  # 8x8, constant inside every 4x4 block
    labels = np.ones((8, 8), int)
    assert object_rho(const, labels, 4)[0] == pytest.approx(1.0, abs=1e-12)
    pattern = np.tile(np.array([[1.0, -1.0], [-1.0, 1.0]]), (2, 2))  # 4x4 with zero block mean
    zero_mean = np.tile(pattern, (2, 2))                              # 8x8, every 4x4 block sums to zero
    assert object_rho(zero_mean, labels, 4)[0] == pytest.approx(0.0, abs=1e-12)
    mixed = const + 0.3 * zero_mean
    r = object_rho(mixed, labels, 4)[0]
    assert 0.0 < r < 1.0


def test_rho_is_computed_per_object_and_stays_in_unit_interval():
    from experiments.wayanad_evidence.gate import object_rho
    rng = np.random.default_rng(8)
    delta = rng.uniform(0.1, 1.0, (16, 16))
    labels = np.zeros((16, 16), int)
    labels[0:4, 0:8] = 1
    labels[8:12, 4:12] = 2
    rho = object_rho(delta, labels, 4)
    assert rho.shape == (2,) and ((rho >= 0) & (rho <= 1)).all()
    single = np.zeros_like(delta)
    single[0:4, 0:8] = delta[0:4, 0:8]                               # object 1 on its own, by the definition
    blocks = single.reshape(4, 4, 4, 4).mean(axis=(1, 3))
    expected = 16 * (blocks ** 2).sum() / (single ** 2).sum()
    assert rho[0] == pytest.approx(expected, rel=1e-12)


def test_ndvi_undefined_below_denominator_floor():
    from experiments.wayanad_evidence.stats import ndvi
    red = np.array([0.05, 0.01, 0.10])
    nir = np.array([0.45, 0.02, 0.10])
    out = ndvi(red, nir, min_denominator=0.05)
    assert out[0] == pytest.approx(0.8) and np.isnan(out[1]) and out[2] == pytest.approx(0.0)
