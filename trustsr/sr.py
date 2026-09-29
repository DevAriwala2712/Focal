"""Four-band SEN2SR-lite inference on a fixed x4 grid."""
from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
from affine import Affine


def sr_geometry(transform: Affine, shape: tuple[int, int], scale: int = 4):
    if scale != 4:
        raise ValueError('SEN2SR-lite RGBN checkpoint is x4')
    return transform * Affine.scale(1 / scale, 1 / scale), (shape[0] * scale, shape[1] * scale)


def spectral_project(lr: np.ndarray, guessed: np.ndarray, scale: int = 4) -> np.ndarray:
    """Enforce exact 4x4 block mean consistency with each observed parent pixel."""
    if lr.ndim != 3 or guessed.shape != (lr.shape[0], lr.shape[1] * scale, lr.shape[2] * scale):
        raise ValueError('Expected C,H,W source and exactly x4 C,H,W reconstruction')
    block_means = guessed.reshape(lr.shape[0], lr.shape[1], scale, lr.shape[2], scale).mean((2, 4))
    correction = lr.astype(np.float32) - block_means
    return guessed.astype(np.float32) + np.repeat(np.repeat(correction, scale, -2), scale, -1)


def spectral_project_torch(lr, guessed, scale: int = 4):
    """Differentiable consistency projection for the 64px training fallback."""
    import torch.nn.functional as F
    if lr.ndim != 4 or guessed.shape != (lr.shape[0], lr.shape[1], lr.shape[2]*scale, lr.shape[3]*scale):
        raise ValueError('Expected N,C,H,W and exactly x4 N,C,H,W')
    correction = lr - F.avg_pool2d(guessed, kernel_size=scale, stride=scale)
    return guessed + correction.repeat_interleave(scale, -2).repeat_interleave(scale, -1)


def _starts(length: int, tile_size: int, overlap: int) -> list[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, tile_size - overlap))
    if starts[-1] != length - tile_size:
        starts.append(length - tile_size)
    return starts


def tiled_super_resolve(lr: np.ndarray, operator, *, tile_size: int = 128, overlap: int = 16) -> np.ndarray:
    """Blend overlapping tiles with positive edge weights; operator returns x4 tiles."""
    if lr.ndim != 3 or tile_size <= 0 or overlap < 0 or overlap >= tile_size:
        raise ValueError('Invalid C,H,W image or tiling parameters')
    bands, height, width = lr.shape
    out = np.zeros((bands, height * 4, width * 4), dtype=np.float32)
    weight = np.zeros((height * 4, width * 4), dtype=np.float32)
    ys, xs = _starts(height, tile_size, overlap), _starts(width, tile_size, overlap)
    for y in ys:
        for x in xs:
            tile = lr[:, y:min(y + tile_size, height), x:min(x + tile_size, width)]
            pred = np.asarray(operator(tile), dtype=np.float32)
            if pred.shape != (bands, tile.shape[1] * 4, tile.shape[2] * 4):
                raise ValueError('Tile operator returned wrong shape')
            hh, ww = pred.shape[-2:]
            wy, wx = np.ones(hh, dtype=np.float32), np.ones(ww, dtype=np.float32)
            feather = overlap * 4
            if y > 0:
                wy[:min(feather, hh)] = np.linspace(1 / (feather + 1), 1, min(feather, hh))
            if y + tile.shape[1] < height:
                wy[-min(feather, hh):] = np.minimum(wy[-min(feather, hh):], np.linspace(1, 1 / (feather + 1), min(feather, hh)))
            if x > 0:
                wx[:min(feather, ww)] = np.linspace(1 / (feather + 1), 1, min(feather, ww))
            if x + tile.shape[2] < width:
                wx[-min(feather, ww):] = np.minimum(wx[-min(feather, ww):], np.linspace(1, 1 / (feather + 1), min(feather, ww)))
            blend = wy[:, None] * wx[None, :]
            out[:, y*4:y*4+hh, x*4:x*4+ww] += pred * blend
            weight[y*4:y*4+hh, x*4:x*4+ww] += blend
    if np.any(weight == 0):
        raise RuntimeError('Tiling left uncovered pixels')
    return out / weight


def super_resolve(tile: np.ndarray, model, *, device: str = 'cuda', tile_size: int = 128,
                  overlap: int = 16, min_tile: int = 64) -> np.ndarray:
    """Run the pretrained SEN2SR CNN and exact parent consistency projection.

    The upstream Fourier hard constraint is fixed to a 512x512 output. This
    explicit projection permits 64-pixel fallback without altering CNN weights.
    """
    import torch
    if tile.shape[0] != 4 or not np.isfinite(tile).all():
        raise ValueError('Expected four finite bands in B04,B03,B02,B08 order')
    def infer(crop):
        x = torch.from_numpy(np.ascontiguousarray(crop[None], dtype=np.float32)).to(device)
        with torch.inference_mode():
            sr = model.sr_model(x).clamp_min(0).cpu().numpy()[0]
        return spectral_project(crop, sr)
    size = tile_size
    while True:
        try:
            blended = tiled_super_resolve(tile, infer, tile_size=size, overlap=min(overlap, size // 4))
            return spectral_project(tile, blended)
        except torch.cuda.OutOfMemoryError:
            gc.collect()
            torch.cuda.empty_cache()
            if size <= min_tile:
                raise RuntimeError(f'CUDA OOM at minimum tile size {min_tile}; cannot proceed') from None
            size = max(min_tile, size // 2)


def load_finetuned_model(cfg: dict, root: Path, *, device: str = 'cuda'):
    from safetensors.torch import load_file
    from risk.model import load_model
    checkpoint = root / cfg['training']['checkpoint']
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Fine-tuned checkpoint unavailable: {checkpoint}')
    model = load_model(cfg, root, trainable=True, device=device).eval()
    model.load_state_dict(load_file(str(checkpoint)))
    return model
