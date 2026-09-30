"""trustsr.hrbench: tiny synthetic arrays, CPU only. Everything here is `evidence: synthetic` (mechanics, never quality claims)."""
import numpy as np
import pytest

from trustsr import hrbench as hb


def field(shape, seed=0, smooth=3):
    """Smooth random field in [0.05, 0.95], (C,H,W) float64."""
    rng = np.random.default_rng(seed)
    x = rng.random(shape)
    k = np.ones(smooth) / smooth
    for ax in (-1, -2):
        x = np.apply_along_axis(lambda v: np.convolve(np.pad(v, smooth, mode='reflect'), k, 'same')[smooth:-smooth], ax, x)
    x = (x - x.min()) / (x.max() - x.min())
    return 0.05 + 0.9 * x


# ---------------------------------------------------------------- resampling on the exact grid

def test_nearest_is_exact_block_replication():
    x = field((4, 6, 5), 1)
    up = hb.resample(x, 4, 'nearest')
    assert up.shape == (4, 24, 20)
    assert np.array_equal(up, np.repeat(np.repeat(x, 4, axis=1), 4, axis=2))


@pytest.mark.parametrize('method', hb.BASELINES)
def test_constant_image_is_preserved(method):
    x = np.full((4, 9, 9), 0.3)
    assert np.allclose(hb.resample(x, 4, method), 0.3, atol=1e-12)


def test_block_mean_inverts_nearest_exactly_and_rejects_bad_factor():
    x = field((4, 8, 8), 2)
    assert np.abs(hb.block_mean(hb.resample(x, 4, 'nearest'), 4) - x).max() < 1e-15   # float mean, at most 1 ulp
    with pytest.raises(ValueError):
        hb.block_mean(np.zeros((1, 10, 10)), 4)


def test_bicubic_matches_torch_interpolate_half_pixel_convention():
    torch = pytest.importorskip('torch')
    x = field((4, 12, 12), 3)
    ref = torch.nn.functional.interpolate(torch.from_numpy(x)[None], scale_factor=4, mode='bicubic', align_corners=False)[0].numpy()
    assert np.abs(hb.resample(x, 4, 'bicubic') - ref).max() < 1e-9


def test_lanczos_matches_pil_in_the_interior():
    PIL = pytest.importorskip('PIL.Image')
    x = field((1, 16, 16), 4)[0].astype(np.float32)
    ref = np.asarray(PIL.fromarray(x, mode='F').resize((64, 64), PIL.LANCZOS))
    ours = hb.resample(x[None].astype(np.float64), 4, 'lanczos')[0]
    # PIL renormalises truncated kernels at the border; ours clamps indices. Interior (>= 3 input px = 12 output px) agrees.
    assert np.abs(ours[12:-12, 12:-12] - ref[12:-12, 12:-12]).max() < 2e-5


def test_bicubic_and_lanczos_reproduce_a_linear_ramp_at_block_mean_level():
    yy, xx = np.mgrid[0:16, 0:16]
    ramp = (0.2 + 0.01 * yy + 0.02 * xx)[None].astype(np.float64)
    for m in ('bicubic', 'lanczos'):
        back = hb.block_mean(hb.resample(ramp, 4, m), 4)
        assert np.abs(back - ramp)[:, 4:-4, 4:-4].max() < 2e-3        # interior; kernels are not exactly linear-preserving (lanczos)


# ---------------------------------------------------------------- PSNR / SSIM

def test_psnr_known_value_and_mask_is_respected():
    ref = np.full((4, 20, 20), 0.5)
    est = ref + 0.1
    mask = np.zeros((20, 20), bool)
    mask[5:15, 5:15] = True
    est_bad = est.copy()
    est_bad[:, ~mask] = 0.0                                          # garbage outside the mask must not matter
    assert hb.psnr(ref, est, mask) == pytest.approx(20.0, abs=1e-9)
    assert hb.psnr(ref, est_bad, mask) == pytest.approx(20.0, abs=1e-9)
    assert hb.psnr(ref, est, mask, data_range=2.0) == pytest.approx(20.0 + 20 * np.log10(2.0), abs=1e-9)


def test_psnr_is_capped_when_identical_and_needs_pixels():
    ref = field((4, 8, 8), 5)
    mask = np.ones((8, 8), bool)
    assert hb.psnr(ref, ref, mask) == pytest.approx(hb.PSNR_CAP_DB)
    with pytest.raises(ValueError):
        hb.psnr(ref, ref, np.zeros((8, 8), bool))


def test_per_band_psnr_shape():
    ref = field((4, 8, 8), 6)
    est = ref.copy()
    est[2] += 0.05
    out = hb.psnr_per_band(ref, est, np.ones((8, 8), bool))
    assert out.shape == (4,) and out[2] == pytest.approx(20 * np.log10(1 / 0.05)) and out[0] == hb.PSNR_CAP_DB


def test_ssim_identity_and_monotone_in_noise():
    ref = field((4, 40, 40), 7)
    mask = np.ones((40, 40), bool)
    rng = np.random.default_rng(0)
    assert hb.ssim(ref, ref, mask) == pytest.approx(1.0)
    s = [hb.ssim(ref, np.clip(ref + n * rng.standard_normal(ref.shape), 0, 1), mask) for n in (0.01, 0.05, 0.2)]
    assert s[0] > s[1] > s[2]


def test_ssim_only_counts_windows_fully_inside_the_mask():
    ref = field((4, 40, 40), 8)
    est = ref.copy()
    est[:, :12, :] += 0.4                                            # damage the top 12 rows only
    mask = np.ones((40, 40), bool)
    mask[:16, :] = False                                             # ... and mask them out (window radius 5 < 16-12+... fully excluded)
    assert hb.ssim(ref, est, mask) == pytest.approx(1.0)
    assert hb.ssim(ref, est, np.ones((40, 40), bool)) < 0.99
    with pytest.raises(ValueError):
        hb.ssim(ref, est, np.zeros((40, 40), bool))


def test_ssim_matches_skimage_when_available():
    skimage = pytest.importorskip('skimage.metrics')
    ref = field((1, 48, 48), 9)
    rng = np.random.default_rng(1)
    est = np.clip(ref + 0.05 * rng.standard_normal(ref.shape), 0, 1)
    full = skimage.structural_similarity(ref[0], est[0], data_range=1.0, gaussian_weights=True, sigma=1.5,
                                         use_sample_covariance=False, full=True)[1]
    assert hb.ssim(ref, est, np.ones((48, 48), bool)) == pytest.approx(full[5:-5, 5:-5].mean(), abs=1e-9)


# ---------------------------------------------------------------- 10 m downsample error

def test_downsample_error_zero_for_nearest_and_offset_for_biased_sr():
    lr = field((4, 8, 8), 10)
    sr = hb.resample(lr, 4, 'nearest')
    e0 = hb.downsample_error(sr, lr, 4)
    assert np.allclose(e0['mae'], 0) and np.allclose(e0['rmse'], 0)
    e1 = hb.downsample_error(sr + 0.05, lr, 4)
    assert np.allclose(e1['mae'], 0.05) and np.allclose(e1['bias'], 0.05) and np.allclose(e1['rmse'], 0.05)
    with pytest.raises(ValueError):
        hb.downsample_error(sr[:, :-4], lr, 4)                       # shape must be exactly scale x LR


def test_downsample_error_border_and_mask():
    lr = np.zeros((4, 10, 10))
    sr = np.zeros((4, 40, 40))
    sr[:, :4, :4] = 1.0                                              # error only in LR pixel (0,0)
    m = np.ones((10, 10), bool)
    assert hb.downsample_error(sr, lr, 4, lr_mask=m)['mae'][0] == pytest.approx(1 / 100)
    m[0, 0] = False
    assert hb.downsample_error(sr, lr, 4, lr_mask=m)['mae'][0] == 0.0


# ---------------------------------------------------------------- grid alignment check

def test_grid_shift_finds_zero_for_exact_subdivision_and_detects_a_shift():
    hr = field((4, 64, 64), 11, smooth=5)
    lr = hb.block_mean(hr, 4)
    g0 = hb.grid_shift(lr, hr, 4)
    assert (g0['dy'], g0['dx']) == (0, 0)
    shifted = np.roll(hr, (2, -1), axis=(1, 2))                      # shifted[i] = hr[i-2] rows, hr[j+1] cols
    g1 = hb.grid_shift(lr, shifted, 4)
    # (dy, dx) is the HR-pixel offset at which the HR grid must be read to line up with the LR grid
    assert (g1['dy'], g1['dx']) == (2, -1)
    assert g1['mse_best'] < g1['mse_zero']


# ---------------------------------------------------------------- padding / native-size adapter

def test_native_padding_roundtrip_with_nearest_operator():
    x = field((4, 121, 121), 12, smooth=5).astype(np.float32)
    calls = []

    def native_fn(b):                                               # (B,C,128,128) -> nearest x4
        calls.append(b.shape)
        return np.repeat(np.repeat(b, 4, axis=2), 4, axis=3)

    out = hb.sr_native_padded(native_fn, x, native=128, scale=4)
    assert calls == [(1, 4, 128, 128)] and out.shape == (4, 484, 484)
    assert np.array_equal(out, hb.resample(x, 4, 'nearest'))         # pad -> SR -> crop is transparent for a local operator
    with pytest.raises(ValueError):
        hb.sr_native_padded(native_fn, np.zeros((4, 129, 128), np.float32), native=128, scale=4)


# ---------------------------------------------------------------- evaluate_sr_fn (the shared harness)

def make_items(n=6, scale=4, size=48, seed=20, dataset='synthetic'):
    items = []
    for i in range(n):
        hr = field((4, size * scale, size * scale), seed + i, smooth=3)
        lr = hb.block_mean(hr, scale)
        items.append((dataset, hb.Sample(name=f'img{i}', lr=lr.astype(np.float32), hr=hr.astype(np.float32),
                                         mask=np.ones(hr.shape[-2:], bool), group=f's{i // 2}')))
    return items


BOOT = {'replicates': 300, 'ci': 0.95, 'seed': 2024}


def test_harness_oracle_beats_bicubic_and_bicubic_ties_itself():
    items = make_items()
    hr_by_name = {s.name: s.hr for _, s in items}
    # oracle keyed by content: returns the true HR for the LR it was given
    lookup = {s.lr.tobytes(): s.hr for _, s in items}
    res = hb.evaluate_sr_fn(lambda lr: lookup[lr.tobytes()], iter(items), border=8, bootstrap=BOOT)
    d = res['datasets']['synthetic']
    assert d['n_images'] == 6
    assert d['delta_vs_bicubic']['psnr']['lo'] > 0 and d['beats_bicubic'] is True
    assert d['delta_vs_bicubic']['ssim']['mean'] > 0
    tie = hb.evaluate_sr_fn(lambda lr: hb.resample(lr.astype(np.float64), 4, 'bicubic'), iter(items), border=8, bootstrap=BOOT)
    t = tie['datasets']['synthetic']['delta_vs_bicubic']['psnr']
    assert t['mean'] == pytest.approx(0, abs=1e-9) and tie['datasets']['synthetic']['beats_bicubic'] is False
    assert len(res['per_image']) == 6 and set(res['per_image'][0]['psnr']) == {'nearest', 'bicubic', 'lanczos', 'sr'}
    assert hr_by_name


def test_harness_nearest_sr_loses_and_strongest_baseline_is_named():
    items = make_items(n=8, seed=40)
    res = hb.evaluate_sr_fn(lambda lr: hb.resample(lr.astype(np.float64), 4, 'nearest'), iter(items), border=8, bootstrap=BOOT)
    d = res['datasets']['synthetic']
    assert d['delta_vs_bicubic']['psnr']['hi'] < 0 and d['beats_bicubic'] is False
    assert d['strongest_baseline'] in hb.BASELINES
    assert d['strongest_baseline'] != 'nearest'


def test_harness_x2_reference_reduces_sr_by_block_mean():
    items = make_items(n=4, scale=2, size=40, seed=60, dataset='x2')
    lookup = {s.lr.tobytes(): s.hr for _, s in items}
    # an x4 SR whose 2x2 block mean equals the x2 HR exactly (nearest replication of the truth)
    res = hb.evaluate_sr_fn(lambda lr: np.repeat(np.repeat(lookup[lr.tobytes()], 2, axis=1), 2, axis=2), iter(items),
                            border=8, bootstrap=BOOT)
    r = res['per_image'][0]
    assert r['ref_scale'] == 2 and r['psnr']['sr'] == pytest.approx(hb.PSNR_CAP_DB)


def test_harness_deterministic_and_cluster_sensitivity_present():
    items = make_items(n=6, seed=80)
    fn = lambda lr: hb.resample(lr.astype(np.float64), 4, 'lanczos')
    a = hb.evaluate_sr_fn(fn, iter(items), border=8, bootstrap=BOOT)
    b = hb.evaluate_sr_fn(fn, iter(items), border=8, bootstrap=BOOT)
    assert a == b
    d = a['datasets']['synthetic']
    assert d['n_groups'] == 3 and 'delta_vs_bicubic_by_group' in d


def test_harness_rejects_bad_sr_output_and_non_integer_reference_scale():
    items = make_items(n=2, seed=90)
    with pytest.raises(ValueError):
        hb.evaluate_sr_fn(lambda lr: np.zeros((4, 10, 10)), iter(items), border=8, bootstrap=BOOT)
    with pytest.raises(ValueError):
        hb.evaluate_sr_fn(lambda lr: np.full((4, 192, 192), np.nan), iter(items), border=8, bootstrap=BOOT)
    bad = hb.Sample(name='x', lr=np.zeros((4, 10, 10), np.float32), hr=np.zeros((4, 25, 25), np.float32),
                    mask=np.ones((25, 25), bool))
    with pytest.raises(ValueError):
        hb.evaluate_sr_fn(lambda lr: lr, iter([('bad', bad)]), border=0, bootstrap=BOOT)


def test_pooled_summary_has_all_images_and_dataset_means():
    items = make_items(n=3, seed=100, dataset='a') + make_items(n=4, seed=200, dataset='b')
    res = hb.evaluate_sr_fn(lambda lr: hb.resample(lr.astype(np.float64), 4, 'lanczos'), iter(items), border=8, bootstrap=BOOT)
    p = res['pooled']
    assert p['n_images'] == 7 and p['n_datasets'] == 2
    means = [res['datasets'][k]['delta_vs_bicubic']['psnr']['mean'] for k in ('a', 'b')]
    assert p['mean_of_dataset_mean_delta_psnr'] == pytest.approx(np.mean(means))


def test_make_sr_fn_tiles_large_and_pads_small_images_with_a_local_operator():
    from experiments.x2_hr_benchmark import make_sr_fn

    def native(b):
        assert b.shape[-2:] == (128, 128)
        return np.repeat(np.repeat(b, 4, axis=2), 4, axis=3)

    fn = make_sr_fn(native, {'native': 128, 'scale': 4, 'margin': 16, 'stride': 96}, RuntimeError, lambda: None)
    for shape in ((4, 200, 150), (4, 121, 121), (4, 128, 128), (4, 130, 90)):
        x = field(shape, 13).astype(np.float32)
        out = fn(x)
        assert out.shape == (4, shape[1] * 4, shape[2] * 4)
        assert np.array_equal(out, hb.resample(x, 4, 'nearest'))   # padding/stitching is transparent for a pointwise operator
