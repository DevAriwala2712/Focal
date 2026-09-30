"""Step 3a: identify the pinned Fourier low-pass mask (family fit) and probe real tile-grid disagreement (ADR-001)."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np


def _distance(shape):
    rows, cols = shape
    u, v = np.mgrid[0:rows, 0:cols]
    return np.hypot(u - rows // 2, v - cols // 2)


def families(shape, radius):
    """Vectorised numpy versions of the four upstream families in sen2sr/models/tricks.py (float32, centre = shape // 2)."""
    d = _distance(shape)
    return {
        'ideal': lambda: (d <= radius).astype(np.float32),
        'gaussian': lambda: np.exp(-(d ** 2) / (2 * radius ** 2)).astype(np.float32),
        'butterworth': lambda order: (1 / (1 + (d / radius) ** (2 * order))).astype(np.float32),
        # the exponent is clipped only to avoid an overflow warning; 1 / (1 + e^700) is already 0 in float32
        'sigmoid': lambda sharp: (1 / (1 + np.exp(np.clip((d - radius) / sharp, -700, 700)))).astype(np.float32),
    }


def radial_profile(mask):
    """Mean of the mask by integer distance from the centre pixel."""
    bins = np.rint(_distance(mask.shape)).astype(int)
    return np.bincount(bins.ravel(), weights=np.asarray(mask, dtype=np.float64).ravel()) / np.bincount(bins.ravel())


def _refine(err, lo, hi, rounds=8, points=41):
    """Coarse-to-fine 1-D minimisation of err(param) on [lo, hi] (no scipy)."""
    grid = np.geomspace(lo, hi, 300)
    best = grid[int(np.argmin([err(p) for p in grid]))]
    step = best * 0.05
    for _ in range(rounds):
        grid = np.linspace(max(best - step * 5, 1e-9), best + step * 5, points)
        best = grid[int(np.argmin([err(p) for p in grid]))]
        step = (grid[1] - grid[0])
    return float(best), float(err(best))


def fit_families(mask, radius):
    """Max |difference| of the stored mask against each upstream family built for its shape at `radius`."""
    mask = np.asarray(mask, dtype=np.float32)
    fam = families(mask.shape, radius)
    diff = lambda m: float(np.abs(mask - m).max())
    fits = {'ideal': {'max_abs_diff': diff(fam['ideal']()), 'param': None},
            'gaussian': {'max_abs_diff': diff(fam['gaussian']()), 'param': None}}
    orders = list(range(1, 41))
    errs = [diff(fam['butterworth'](o)) for o in orders]
    fits['butterworth'] = {'max_abs_diff': min(errs), 'param': {'order': orders[int(np.argmin(errs))]}}
    sharp, err = _refine(lambda s: diff(fam['sigmoid'](s)), 0.05, 50.0)
    fits['sigmoid'] = {'max_abs_diff': err, 'param': {'sharpness': sharp}}
    best = min(fits, key=lambda k: fits[k]['max_abs_diff'])
    unique = np.unique(mask)
    return {'fits': fits, 'best_family': best, 'reproduces_within_1e-6': fits[best]['max_abs_diff'] < 1e-6,
            'is_binary': bool(set(unique.tolist()) <= {0.0, 1.0}), 'radius': radius}


def fit_free_gaussian(mask):
    """Diagnostic only (not an upstream family at its radius): best Gaussian exp(-d^2 / 2 sigma^2) with free sigma."""
    mask = np.asarray(mask, dtype=np.float32)
    d = _distance(mask.shape)
    err = lambda s: float(np.abs(mask - np.exp(-(d ** 2) / (2 * s ** 2)).astype(np.float32)).max())
    sigma, diff = _refine(err, 1.0, 200.0)
    return {'sigma': sigma, 'max_abs_diff': diff}


def load_edge_probe(root: Path):
    """The synthetic FFT edge probe. The JSON is absent on this branch; the same content is read from the patch that adds it."""
    direct = root / 'experiments/probes/results/fft_edge_probe.json'
    if direct.is_file():
        return json.loads(direct.read_text(encoding='utf-8')), str(direct.relative_to(root))
    for patch in sorted(root.glob('0001-*.patch')):
        text = patch.read_text(encoding='utf-8')
        m = re.search(r'diff --git a/experiments/probes/results/fft_edge_probe\.json.*?\n@@[^\n]*\n(.*?)(?=\ndiff --git|\n-- \n|\Z)',
                      text, re.S)
        if m:
            body = '\n'.join(l[1:] for l in m.group(1).splitlines() if l.startswith('+'))
            return json.loads(body), f'{patch.name} (patch content; the JSON file itself is not on this branch)'
    return None, 'not found'


def seam_probe(model_fn, image, tile=128, offset=64, scale=4, bins=((0, 8), (8, 16), (16, 32), (32, 64), (64, 129))):
    """Two tile grids offset by `offset` input px, real model. p99/max |SR_A - SR_B| by distance (2.5 m px) from grid-A tile edge."""
    c, h, w = image.shape
    starts_a = list(range(0, h - tile + 1, tile))
    starts_b = [s + offset for s in starts_a if s + offset + tile <= h]
    out_a = np.zeros((c, h * scale, w * scale), np.float32)
    dist = np.zeros((h * scale, w * scale), np.float32)
    out_b = np.full_like(out_a, np.nan)
    for r in starts_a:
        for q in range(0, w - tile + 1, tile):
            sr = model_fn(image[None, :, r:r + tile, q:q + tile])[0]
            out_a[:, r * scale:(r + tile) * scale, q * scale:(q + tile) * scale] = sr
            i = np.arange(tile * scale)
            e = np.minimum(i, tile * scale - 1 - i)
            dist[r * scale:(r + tile) * scale, q * scale:(q + tile) * scale] = np.minimum(e[:, None], e[None, :])
    for r in starts_b:
        for q in [s + offset for s in range(0, w - tile + 1, tile) if s + offset + tile <= w]:
            sr = model_fn(image[None, :, r:r + tile, q:q + tile])[0]
            out_b[:, r * scale:(r + tile) * scale, q * scale:(q + tile) * scale] = sr
    delta = np.abs(out_a - out_b).max(axis=0)
    ok = np.isfinite(delta)
    res = {}
    for lo, hi in bins:
        sel = ok & (dist >= lo) & (dist < hi)
        if sel.any():
            res[f'{lo}-{hi}px_HR'] = {'p99_abs_delta_between_grids': float(np.percentile(delta[sel], 99)),
                                      'max_abs_delta': float(delta[sel].max()), 'pixels': int(sel.sum())}
    return res
