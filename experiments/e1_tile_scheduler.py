"""E1: overlap tile scheduler at native 128 px input (no change to the model's Fourier hard constraint)."""
from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass, field

import numpy as np

from experiments.common import Blocked, load_real_crop, run_cli, synthetic_scene


def tile_origins(size: int, tile: int, stride: int, anchor: int = 0, offset: int = 0) -> list[int]:
    """1-D tile starts inside an AOI of `size` px.

    The lattice sits at global multiples of `stride` (+offset) on the reference grid; `anchor` is the
    AOI's start on that grid. Both edges are clamped tiles, never padded ones.
    """
    if size < tile:
        raise ValueError(f'AOI of {size} px is smaller than tile {tile}; padding is not allowed')
    last = size - tile
    first = -((offset - anchor) // stride)            # smallest k with k*stride+offset-anchor >= 0
    lattice = range(first * stride + offset - anchor, last + 1, stride)
    return sorted({0, last, *lattice})


def feather_weights(n: int, taper: int, low: bool, high: bool) -> np.ndarray:
    """1-D blend weights: raised-cosine ramps on interior sides only, strictly positive."""
    w = np.ones(n, dtype=np.float32)
    ramp = (0.5 - 0.5 * np.cos(np.pi * (np.arange(taper) + 1) / (taper + 1))).astype(np.float32)
    if low:
        w[:taper] = np.minimum(w[:taper], ramp)
    if high:
        w[n - taper:] = np.minimum(w[n - taper:], ramp[::-1])
    return w


@dataclass
class TiledResult:
    array: np.ndarray
    transform: object
    origins: list = field(default_factory=list)
    input_shapes: set = field(default_factory=set)


def super_resolve_tiled(image, transform, operator, *, tile, stride, scale, feather,
                        anchor=(0, 0), offset=(0, 0), batch_size=1, origins=None,
                        stage=lambda name: contextlib.nullcontext()) -> TiledResult:
    """Overlap-tile `operator` over image (C,H,W). Tile order is row-major and fixed."""
    channels, height, width = image.shape
    offset = (offset, offset) if isinstance(offset, int) else offset
    anchor = (anchor, anchor) if isinstance(anchor, int) else anchor
    if origins is None:
        rows = tile_origins(height, tile, stride, anchor[0], offset[0])
        cols = tile_origins(width, tile, stride, anchor[1], offset[1])
        origins = [(r, c) for r in rows for c in cols]
    out_tile, taper = tile * scale, feather * scale
    num = np.zeros((channels, height * scale, width * scale), dtype=np.float32)
    den = np.zeros((height * scale, width * scale), dtype=np.float32)
    weights, shapes = {}, set()
    for start in range(0, len(origins), batch_size):
        chunk = origins[start:start + batch_size]
        with stage('extract'):
            batch = np.stack([image[:, r:r + tile, c:c + tile] for r, c in chunk]).astype(np.float32)
        shapes.add(batch.shape)
        with stage('forward'):
            out = operator(batch)
        if out.shape != (len(chunk), channels, out_tile, out_tile):
            raise ValueError(f'operator returned {out.shape}, expected {(len(chunk), channels, out_tile, out_tile)}')
        with stage('blend'):
            for (r, c), sr in zip(chunk, out):
                key = (r > 0, r + tile < height, c > 0, c + tile < width)
                if key not in weights:
                    weights[key] = np.outer(feather_weights(out_tile, taper, key[0], key[1]),
                                            feather_weights(out_tile, taper, key[2], key[3]))
                w = weights[key]
                R, C = r * scale, c * scale
                num[:, R:R + out_tile, C:C + out_tile] += sr * w
                den[R:R + out_tile, C:C + out_tile] += w
    with np.errstate(invalid='ignore', divide='ignore'):
        result = np.where(den > 0, num / den, np.float32('nan')).astype(np.float32)
    if transform is not None:
        from affine import Affine
        transform = transform * Affine.scale(1 / scale)
    return TiledResult(result, transform, list(origins), shapes)


def spectral_consistency(sr, image, scale, bands) -> dict:
    """SR area-downsampled by `scale` (block mean) vs the input, per band."""
    c, h, w = image.shape
    down = sr.astype(np.float64).reshape(c, h, scale, w, scale).mean(axis=(2, 4))
    diff = down - image.astype(np.float64)
    return {b: {'rmse': float(np.sqrt(np.mean(d ** 2))), 'max_abs': float(np.abs(d).max()),
                'p99_abs': float(np.percentile(np.abs(d), 99)), 'bias': float(d.mean())}
            for b, d in zip(bands, diff)}


def model_operator(model, device='cpu'):
    import torch
    def operator(batch):
        with torch.inference_mode():
            return model(torch.from_numpy(batch).to(device)).cpu().numpy().astype(np.float32)
    return operator


def bicubic_control(image, scale, bands) -> dict:
    """No-model control: the constraint's own antialiased bicubic upsample, downsampled back."""
    import torch
    up = torch.nn.functional.interpolate(torch.from_numpy(image[None]), scale_factor=scale,
                                         mode='bicubic', antialias=True)[0].numpy()
    return spectral_consistency(up, image, scale, bands)


def run_size(model_op, scene, settings, transform, bands, scale) -> dict:
    """Run one AOI size and evaluate every E1 keep-criterion."""
    from affine import Affine
    size = scene.shape[-1]
    t0 = time.perf_counter()
    out = super_resolve_tiled(scene, transform, model_op, tile=settings['tile'], stride=settings['stride'],
                              scale=scale, feather=settings['feather'])
    seconds = time.perf_counter() - t0
    consistency = spectral_consistency(out.array, scene, scale, bands)
    tol = settings['spectral_rmse_tolerance']
    checks = {
        'only_128_inputs': out.input_shapes == {(1, 4, settings['tile'], settings['tile'])},
        'output_exactly_4x': out.array.shape == (4, size * scale, size * scale),
        'affine_is_input_times_scale_quarter': out.transform == transform * Affine.scale(1 / scale),
        'finite_and_fully_covered': bool(np.isfinite(out.array).all()),
        'spectral_rmse_within_tolerance': all(v['rmse'] <= tol for v in consistency.values()),
    }
    return {'aoi_px': size, 'tiles': len(out.origins), 'origins_rows': sorted({r for r, _ in out.origins}),
            'seconds': seconds, 'checks': checks, 'spectral_consistency': consistency,
            'bicubic_control_consistency': bicubic_control(scene, scale, bands),
            'output_affine': list(out.transform)[:6], 'passed': all(checks.values())}


def probe(cfg, root):
    """E1: overlap tiler at native 128 px input, real pinned SEN2SR-lite on CPU."""
    import torch
    from risk.model import load_model
    from affine import Affine
    settings, p0 = cfg['e1'], cfg['phase0']['model']
    torch.manual_seed(cfg['seed'])
    model = load_model(cfg['phase0'], root, device='cpu').eval()
    op = model_operator(model)
    transform = Affine(*settings['affine'])
    largest = max(settings['aoi_sizes'])
    rows = [run_size(op, synthetic_scene(s, cfg['seed'], full=largest), settings, transform, p0['bands'], p0['scale'])
            for s in settings['aoi_sizes']]
    direct_scene = synthetic_scene(p0['native_tile'], cfg['seed'], full=largest)
    with torch.inference_mode():
        direct = model(torch.from_numpy(direct_scene[None])).numpy()[0]
    single_tile_identity = float(np.abs(super_resolve_tiled(direct_scene, None, op, tile=settings['tile'],
        stride=settings['stride'], scale=p0['scale'], feather=settings['feather']).array - direct).max())
    try:
        real_crop = {'status': 'available'}
        load_real_crop(settings['real_crop'], root)
    except Blocked as exc:
        real_crop = {'status': 'BLOCKED', 'reason': str(exc)}
    ok = all(r['passed'] for r in rows)
    return {'status': 'PASS' if ok else 'FAIL', 'evidence': 'synthetic',
            'evidence_parts': {'synthetic_rgbn': 'PASS' if ok else 'FAIL', 'real_10m_crop': real_crop['status']},
            'criteria': {'spectral_rmse_tolerance_reflectance': settings['spectral_rmse_tolerance'],
                         'tolerance_rationale': settings['tolerance_rationale'], 'tile': settings['tile'],
                         'stride': settings['stride'], 'feather_input_px': settings['feather']},
            'measurements': {'device': 'cpu', 'torch': torch.__version__, 'per_size': rows,
                             'single_tile_vs_direct_model_max_abs': single_tile_identity},
            'real_crop': real_crop,
            'limitations': ['Synthetic scenes: mechanics, geometry and spectral consistency only; no accuracy or '
                            'image-quality claim.',
                            'Spectral consistency is largely enforced by the model\'s hard constraint, so it is a '
                            'sanity check on the tiler, not evidence of SR fidelity.',
                            'CPU run; GPU numerics and memory are not covered (see E3, E4).']
                           + (['Real 10 m crop not evaluated: ' + real_crop['reason']] if real_crop['status'] == 'BLOCKED' else [])}


if __name__ == '__main__':
    run_cli('e1', probe)
