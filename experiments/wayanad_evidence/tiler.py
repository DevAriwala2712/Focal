"""Native-128 tiler: fixed row-major order, clamped edge tiles, centre-crop stitching (ADR-001). No padding, no 64 px path."""
from __future__ import annotations

import math

import numpy as np


def axis_plan(n: int, tile: int, margin: int, stride: int):
    """[(start, keep_lo, keep_hi)] along one axis. keep windows partition [0, n); each discards >= margin per interior side."""
    if n < tile:
        raise ValueError(f'axis length {n} is smaller than the {tile} px native input; the model accepts {tile} px only')
    if stride > tile - 2 * margin:
        raise ValueError('stride larger than tile - 2*margin would leave gaps between centre crops')
    count = 1 if n == tile else math.ceil((n - tile) / stride) + 1
    starts = [min(i * stride, n - tile) for i in range(count)]
    plan, lo = [], 0
    for i, start in enumerate(starts):
        hi = n if i == count - 1 else start + tile - margin
        plan.append((start, lo, hi))
        lo = hi
    return plan


def tile_plan(shape, tile: int, margin: int, stride: int):
    """Row-major list of tiles: r0, c0 (input origin) and keep windows in input px."""
    rows, cols = axis_plan(shape[0], tile, margin, stride), axis_plan(shape[1], tile, margin, stride)
    return [{'r0': r0, 'c0': c0, 'keep_r': (rl, rh), 'keep_c': (cl, ch)}
            for r0, rl, rh in rows for c0, cl, ch in cols]


def stitch(image, model_fn, *, tile, margin, stride, scale, write_log=None):
    """Run `model_fn` (B,C,tile,tile) -> (B,C,tile*scale,tile*scale) tile by tile and centre-crop stitch."""
    channels, h, w = image.shape
    out = np.zeros((channels, h * scale, w * scale), dtype=np.float32)
    for t in tile_plan((h, w), tile, margin, stride):
        r0, c0 = t['r0'], t['c0']
        sr = model_fn(image[None, :, r0:r0 + tile, c0:c0 + tile])[0]
        (rl, rh), (cl, ch) = t['keep_r'], t['keep_c']
        out[:, rl * scale:rh * scale, cl * scale:ch * scale] = sr[:, (rl - r0) * scale:(rh - r0) * scale,
                                                                   (cl - c0) * scale:(ch - c0) * scale]
        if write_log is not None:
            write_log.append((rl * scale, rh * scale, cl * scale, ch * scale))
    return out
