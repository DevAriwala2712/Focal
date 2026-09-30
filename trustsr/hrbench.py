"""HR-referenced validation harness for x4 super-resolution (A2). numpy only, CPU, deterministic.

Conventions (all stated so results are auditable):
* Reflectance in [0, 1] (DN / 10000); PSNR/SSIM data range 1.0. Image PSNR = 10 log10(range^2 / MSE), MSE pooled over all
  valid pixels and bands; capped at PSNR_CAP_DB (MSE floor 1e-12) so identical images stay finite.
* Baselines are computed on the exact x`scale` subdivision of the LR grid with half-pixel-centre geometry
  (output pixel i sits at input coordinate (i + 0.5)/scale - 0.5); edge taps clamp to the border pixel.
* SSIM: Gaussian window 11 x 11, sigma 1.5, population covariance (Wang et al.), evaluated only where the WHOLE window lies
  inside the mask; the image value is the mean over bands and such windows.
* Every candidate (SR and baselines) is clipped to [0, 1] before PSNR/SSIM (same for all methods). The 10 m downsample error
  uses the UNclipped SR.
* If the reference grid is coarser than the SR grid (e.g. 5 m Venus vs x4 = 2.5 m) every candidate is block-averaged by the
  exact integer ratio before comparison; nothing is resized by interpolation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from trustsr.bootstrap import paired_bootstrap_ci

BASELINES = ('nearest', 'bicubic', 'lanczos')
PSNR_CAP_DB = 120.0
_MSE_FLOOR = 1e-12


# ---------------------------------------------------------------- resampling on the exact grid

def dn_to_reflectance(dn, scale: float = 10000.0) -> np.ndarray:
    return (np.asarray(dn, dtype=np.float64) / scale).astype(np.float32)


def _cubic(t, a=-0.75):                       # torch/OpenCV bicubic convolution (a = -0.75)
    t = np.abs(t)
    return np.where(t <= 1, (a + 2) * t ** 3 - (a + 3) * t ** 2 + 1,
                    np.where(t < 2, a * t ** 3 - 5 * a * t ** 2 + 8 * a * t - 4 * a, 0.0))


def _lanczos(t, a=3):
    t = np.asarray(t, dtype=np.float64)
    return np.where(np.abs(t) < a, np.sinc(t) * np.sinc(t / a), 0.0)


def _weights(n_in: int, scale: int, method: str):
    """(index[n_out, taps], weight[n_out, taps]) for one axis."""
    n_out = n_in * scale
    x = (np.arange(n_out) + 0.5) / scale - 0.5
    if method == 'nearest':
        return (np.arange(n_out) // scale)[:, None], np.ones((n_out, 1))
    radius = 2 if method == 'bicubic' else 3
    base = np.floor(x).astype(int)
    taps = base[:, None] + np.arange(-radius + 1, radius + 1)[None, :]
    w = (_cubic if method == 'bicubic' else _lanczos)(taps - x[:, None])
    if method == 'lanczos':
        w = w / w.sum(axis=1, keepdims=True)
    return np.clip(taps, 0, n_in - 1), w


def resample(x, scale: int, method: str) -> np.ndarray:
    """(C,h,w) -> (C,h*scale,w*scale) on the exact subdivision grid. method in BASELINES."""
    if method not in BASELINES:
        raise ValueError(f'unknown method {method!r}')
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 3:
        raise ValueError('expected (C,H,W)')
    iy, wy = _weights(x.shape[1], scale, method)
    ix, wx = _weights(x.shape[2], scale, method)
    tmp = np.einsum('cokw,ok->cow', x[:, iy, :], wy)              # rows
    return np.einsum('choq,oq->cho', tmp[:, :, ix], wx)           # cols


def block_mean(x, factor: int) -> np.ndarray:
    """Exact area mean over factor x factor blocks of the last two axes."""
    x = np.asarray(x, dtype=np.float64)
    h, w = x.shape[-2:]
    if h % factor or w % factor:
        raise ValueError(f'{h}x{w} is not divisible by {factor}')
    return x.reshape(*x.shape[:-2], h // factor, factor, w // factor, factor).mean(axis=(-3, -1))


# ---------------------------------------------------------------- metrics

def _check(ref, est, mask):
    ref, est, mask = np.asarray(ref, np.float64), np.asarray(est, np.float64), np.asarray(mask, bool)
    if ref.shape != est.shape or ref.shape[-2:] != mask.shape:
        raise ValueError(f'shape mismatch ref {ref.shape} est {est.shape} mask {mask.shape}')
    if not mask.any():
        raise ValueError('mask selects no pixels')
    return ref, est, mask


def _psnr_from_mse(mse, data_range):
    return float(min(10 * np.log10(data_range ** 2 / max(mse, _MSE_FLOOR)), 10 * np.log10(data_range ** 2 / _MSE_FLOOR)))


def psnr(ref, est, mask, data_range: float = 1.0) -> float:
    ref, est, mask = _check(ref, est, mask)
    return _psnr_from_mse(float(((ref - est)[:, mask] ** 2).mean()), data_range)


def psnr_per_band(ref, est, mask, data_range: float = 1.0) -> np.ndarray:
    ref, est, mask = _check(ref, est, mask)
    return np.array([_psnr_from_mse(float(((r - e)[mask] ** 2).mean()), data_range) for r, e in zip(ref, est)])


def _gauss(win, sigma):
    x = np.arange(win) - (win - 1) / 2
    k = np.exp(-x ** 2 / (2 * sigma ** 2))
    return k / k.sum()


def _filt(a, k):
    a = sliding_window_view(a, len(k), axis=0) @ k
    return sliding_window_view(a, len(k), axis=1) @ k


def ssim(ref, est, mask, data_range: float = 1.0, win: int = 11, sigma: float = 1.5) -> float:
    ref, est, mask = _check(ref, est, mask)
    k = _gauss(win, sigma)
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    ones = np.ones(win) / win
    inside = np.isclose(_filt(mask.astype(np.float64), ones), 1.0, atol=1e-9)
    if not inside.any():
        raise ValueError('mask has no window that lies fully inside it')
    vals = []
    for r, e in zip(ref, est):
        mx, my = _filt(r, k), _filt(e, k)
        sxx, syy, sxy = _filt(r * r, k) - mx ** 2, _filt(e * e, k) - my ** 2, _filt(r * e, k) - mx * my
        m = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (sxx + syy + c2))
        vals.append(m[inside])
    return float(np.concatenate(vals).mean())


def downsample_error(sr, lr, scale: int, lr_mask=None) -> dict:
    """Per-band error between block_mean(sr, scale) and the LR input: mae, rmse, signed bias (SR - LR), reflectance units."""
    sr, lr = np.asarray(sr, np.float64), np.asarray(lr, np.float64)
    if sr.shape[-2:] != (lr.shape[-2] * scale, lr.shape[-1] * scale) or sr.shape[0] != lr.shape[0]:
        raise ValueError(f'SR {sr.shape} must be exactly {scale}x the LR {lr.shape}')
    m = np.ones(lr.shape[-2:], bool) if lr_mask is None else np.asarray(lr_mask, bool)
    if not m.any():
        raise ValueError('mask selects no pixels')
    diff = (block_mean(sr, scale) - lr)[:, m]
    return {'mae': np.abs(diff).mean(axis=1), 'rmse': np.sqrt((diff ** 2).mean(axis=1)), 'bias': diff.mean(axis=1)}


def grid_shift(lr, hr, scale: int, max_shift: int | None = None) -> dict:
    """Integer HR-pixel offset (dy, dx) at which block_mean(HR) best matches LR. (0, 0) = HR sits on the exact subdivision."""
    lr, hr = np.asarray(lr, np.float64), np.asarray(hr, np.float64)
    if hr.shape[-2:] != (lr.shape[-2] * scale, lr.shape[-1] * scale):
        raise ValueError('HR must be exactly scale x LR')
    ms = scale if max_shift is None else max_shift
    m = -(-ms // scale) * scale
    lr_c = lr[:, m // scale: lr.shape[1] - m // scale, m // scale: lr.shape[2] - m // scale]
    h, w = hr.shape[-2:]
    best, mse0 = None, None
    for dy in range(-ms, ms + 1):
        for dx in range(-ms, ms + 1):
            e = float(((block_mean(hr[:, m + dy: h - m + dy, m + dx: w - m + dx], scale) - lr_c) ** 2).mean())
            if dy == 0 and dx == 0:
                mse0 = e
            if best is None or e < best[0]:
                best = (e, dy, dx)
    return {'dy': best[1], 'dx': best[2], 'mse_best': best[0], 'mse_zero': mse0}


def sr_native_padded(native_fn: Callable, lr, native: int = 128, scale: int = 4) -> np.ndarray:
    """Run a fixed-input-size model on an image no larger than `native`: reflect-pad to native, run, crop. (C,h,w)->(C,h*s,w*s).
    Larger images must be tiled by the caller. Deviation from the production no-padding rule, documented where used."""
    lr = np.asarray(lr, dtype=np.float32)
    c, h, w = lr.shape
    if h > native or w > native:
        raise ValueError(f'{h}x{w} exceeds the native {native} px input; tile it')
    pt, pl = (native - h) // 2, (native - w) // 2
    padded = np.pad(lr, ((0, 0), (pt, native - h - pt), (pl, native - w - pl)), mode='reflect')
    out = np.asarray(native_fn(padded[None]))[0]
    return out[:, pt * scale: (pt + h) * scale, pl * scale: (pl + w) * scale]


# ---------------------------------------------------------------- harness

@dataclass
class Sample:
    name: str
    lr: np.ndarray                 # (C,h,w) reflectance, the model input grid
    hr: np.ndarray                 # (C,H,W) reflectance reference, H = ref_scale * h exactly
    mask: np.ndarray               # (H,W) bool, True = valid reference pixel
    group: str | None = None       # e.g. Sentinel-2 acquisition id, for the cluster-bootstrap sensitivity
    meta: dict = field(default_factory=dict)


def _ci(diffs, boot):
    return paired_bootstrap_ci(diffs, boot['replicates'], boot['ci'], boot['seed'])


def _delta_block(rec, key, a, b, boot):
    d = np.array([r[key][a] - r[key][b] for r in rec])
    return _ci(d, boot)


def evaluate_sr_fn(sr_fn: Callable, dataset_iter: Iterable, *, sr_scale: int = 4, border: int = 16, data_range: float = 1.0,
                   bootstrap: dict | None = None, clip: bool = True, extra: Callable | None = None) -> dict:
    """Score `sr_fn` (LR (C,h,w) float32 -> SR (C,h*sr_scale,w*sr_scale)) with the SAME protocol for any model/checkpoint.

    dataset_iter yields (dataset_name, Sample). `border` HR px are excluded on every side (opensr-test default 16).
    `extra(sample, candidates, lr)` may return {method: {metric: float}} (e.g. opensr-test groups); candidates are the clipped,
    reference-grid arrays {'sr','nearest','bicubic','lanczos'}. Returns per-image records, per-dataset summaries with
    paired-bootstrap CIs (SR - bicubic and SR - strongest baseline) and a pooled summary. PASS/keep logic lives in the caller,
    except `beats_bicubic` = (PSNR delta CI lower bound > 0), the pre-registered keep rule.
    """
    boot = bootstrap or {'replicates': 2000, 'ci': 0.95, 'seed': 2024}
    methods = ('sr',) + BASELINES
    records = []
    for ds, s in dataset_iter:
        lr = np.asarray(s.lr)
        hr = np.asarray(s.hr, np.float64)
        c, h, w = lr.shape
        if hr.shape[0] != c or hr.shape[1] % h or hr.shape[2] % w or hr.shape[1] // h != hr.shape[2] // w:
            raise ValueError(f'{ds}/{s.name}: HR {hr.shape} is not an integer multiple of LR {lr.shape}')
        ref_scale = hr.shape[1] // h
        if sr_scale % ref_scale:
            raise ValueError(f'{ds}/{s.name}: SR scale {sr_scale} is not a multiple of reference scale {ref_scale}')
        reduce = sr_scale // ref_scale
        sr = np.asarray(sr_fn(lr), dtype=np.float64)
        if sr.shape != (c, h * sr_scale, w * sr_scale):
            raise ValueError(f'{ds}/{s.name}: sr_fn returned {sr.shape}, expected {(c, h * sr_scale, w * sr_scale)}')
        if not np.isfinite(sr).all():
            raise ValueError(f'{ds}/{s.name}: sr_fn returned non-finite values')
        raw = {'sr': sr, **{m: resample(lr, sr_scale, m) for m in BASELINES}}
        cand = {}
        for m, a in raw.items():
            a = block_mean(a, reduce) if reduce > 1 else a
            cand[m] = np.clip(a, 0.0, 1.0) if clip else a
        mask = np.asarray(s.mask, bool).copy()
        if border:
            mask[:border], mask[-border:], mask[:, :border], mask[:, -border:] = False, False, False, False
        lr_mask = np.ones((h, w), bool)
        lb = border // ref_scale
        if lb:
            lr_mask[:lb], lr_mask[-lb:], lr_mask[:, :lb], lr_mask[:, -lb:] = False, False, False, False
        lr_mask &= (lr > 0).all(axis=0)
        rec = {'dataset': ds, 'name': s.name, 'group': s.group, 'ref_scale': ref_scale, 'n_valid_px': int(mask.sum()),
               'psnr': {}, 'ssim': {}, 'psnr_band': {}, 'downsample_error': {},
               'frac_outside_unit': {m: float(((raw[m] < 0) | (raw[m] > 1)).mean()) for m in methods}}
        for m in methods:
            rec['psnr'][m] = psnr(hr, cand[m], mask, data_range)
            rec['ssim'][m] = ssim(hr, cand[m], mask, data_range)
            rec['psnr_band'][m] = psnr_per_band(hr, cand[m], mask, data_range).tolist()
            de = downsample_error(raw[m], lr, sr_scale, lr_mask)
            rec['downsample_error'][m] = {k: v.tolist() for k, v in de.items()}
        if extra is not None:
            rec['extra'] = extra(s, cand, lr)
        for key in ('psnr', 'ssim'):
            if not all(np.isfinite(v) for v in rec[key].values()):
                raise ValueError(f'{ds}/{s.name}: non-finite {key}')
        records.append(rec)

    datasets = {}
    for ds in dict.fromkeys(r['dataset'] for r in records):
        rec = [r for r in records if r['dataset'] == ds]
        out = {'n_images': len(rec), 'n_valid_px_hr': int(sum(r['n_valid_px'] for r in rec)),
               'mean': {k: {m: float(np.mean([r[k][m] for r in rec])) for m in methods} for k in ('psnr', 'ssim')}}
        strongest = max(BASELINES, key=lambda m: out['mean']['psnr'][m])
        out['strongest_baseline'] = strongest
        out['delta_vs_bicubic'] = {k: _delta_block(rec, k, 'sr', 'bicubic', boot) for k in ('psnr', 'ssim')}
        out['delta_vs_strongest'] = {k: _delta_block(rec, k, 'sr', strongest, boot) for k in ('psnr', 'ssim')}
        out['beats_bicubic'] = bool(out['delta_vs_bicubic']['psnr']['lo'] > 0)
        out['images_sr_better_than_bicubic_psnr'] = int(sum(r['psnr']['sr'] > r['psnr']['bicubic'] for r in rec))
        groups = {}
        for r in rec:
            groups.setdefault(r['group'] if r['group'] is not None else r['name'], []).append(r['psnr']['sr'] - r['psnr']['bicubic'])
        out['n_groups'] = len(groups)
        out['delta_vs_bicubic_by_group'] = {'psnr': _ci([np.mean(v) for v in groups.values()], boot)}
        out['mean_downsample_error'] = {m: {k: np.mean([r['downsample_error'][m][k] for r in rec], axis=0).tolist()
                                            for k in ('mae', 'rmse', 'bias')} for m in methods}
        datasets[ds] = out
    d_all = np.array([r['psnr']['sr'] - r['psnr']['bicubic'] for r in records]) if records else np.array([])
    pooled = {'n_images': len(records), 'n_datasets': len(datasets)}
    if records:
        pooled['mean_delta_psnr_all_images'] = float(d_all.mean())
        pooled['mean_of_dataset_mean_delta_psnr'] = float(np.mean([d['delta_vs_bicubic']['psnr']['mean'] for d in datasets.values()]))
        pooled['delta_psnr_all_images_ci'] = _ci(d_all, boot)
    return {'protocol': {'sr_scale': sr_scale, 'border_px': border, 'data_range': data_range, 'clip_unit_interval': clip,
                         'bootstrap': boot, 'ssim': {'win': 11, 'sigma': 1.5, 'covariance': 'population'}},
            'per_image': records, 'datasets': datasets, 'pooled': pooled}
