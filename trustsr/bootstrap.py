"""Bootstrap confidence intervals used by every pre-registered comparison (configs/exceptional.yaml `statistics`).

Two flavours:
* `paired_bootstrap_ci`: units are images/AOIs/dates; the input is one paired difference per unit.
* `ratio_bootstrap_ci`: pixel data. Pixels are spatially correlated, so whole blocks are resampled. The caller reduces a
  raster to per-block sufficient statistics (`block_sums`) and the ratio of sums (a rate such as FAR, false-drop fraction or
  coverage) is bootstrapped over blocks. Passing `num_b` bootstraps the difference of two rates that share denominators
  (a paired comparison on the same pixels).
"""
from __future__ import annotations

import numpy as np


def block_sums(a, block: int) -> np.ndarray:
    """Sum `a` over non-overlapping block x block tiles of the last two axes. Edge blocks are kept (never cropped).
    NaN counts as 0; bool input counts True. Returns shape (ceil(H/block), ceil(W/block))."""
    a = np.nan_to_num(np.asarray(a, dtype=np.float64), nan=0.0)
    h, w = a.shape[-2:]
    hb, wb = -(-h // block), -(-w // block)
    pad = np.zeros((hb * block, wb * block), dtype=np.float64)
    pad[:h, :w] = a
    return pad.reshape(hb, block, wb, block).sum(axis=(1, 3))


def _quantiles(samples, ci):
    lo, hi = np.quantile(samples, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return float(lo), float(hi)


def paired_bootstrap_ci(diffs, replicates: int, ci: float, seed: int) -> dict:
    """Percentile CI of the mean of per-unit paired differences, resampling units with replacement."""
    d = np.asarray(diffs, dtype=np.float64).ravel()
    d = d[np.isfinite(d)]
    if d.size == 0:
        raise ValueError('no finite paired differences')
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(replicates, d.size))
    lo, hi = _quantiles(d[idx].mean(axis=1), ci)
    return {'mean': float(d.mean()), 'lo': lo, 'hi': hi, 'n': int(d.size), 'replicates': replicates, 'ci': ci, 'seed': seed}


def ratio_bootstrap_ci(num, den, replicates: int, ci: float, seed: int, num_b=None) -> dict:
    """Block bootstrap of sum(num)/sum(den) (or, with `num_b`, sum(num)/sum(den) - sum(num_b)/sum(den)).
    `num`, `den` (and `num_b`) are per-block sums, one entry per block (any shape)."""
    n = np.asarray(num, dtype=np.float64).ravel()
    m = np.asarray(den, dtype=np.float64).ravel()
    b = None if num_b is None else np.asarray(num_b, dtype=np.float64).ravel()
    if n.shape != m.shape or (b is not None and b.shape != m.shape):
        raise ValueError('num, den and num_b must have the same number of blocks')
    if m.sum() <= 0:
        raise ValueError('zero denominator')
    keep = m > 0
    n, m = n[keep], m[keep]
    b = None if b is None else b[keep]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n.size, size=(replicates, n.size))
    dm = m[idx].sum(axis=1)
    est = n[idx].sum(axis=1) / dm
    point = n.sum() / m.sum()
    if b is not None:
        est = est - b[idx].sum(axis=1) / dm
        point = point - b.sum() / m.sum()
    lo, hi = _quantiles(est, ci)
    return {'estimate': float(point), 'lo': lo, 'hi': hi, 'numerator': float(n.sum()), 'denominator': float(m.sum()),
            'blocks': int(n.size), 'replicates': replicates, 'ci': ci, 'seed': seed}
