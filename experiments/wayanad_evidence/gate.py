"""Trust gate: parent (10 m) mask, five-class rule and the measurement-visible fraction rho (ADR-003)."""
from __future__ import annotations

import numpy as np

NO_CHANGE, OBSERVED, INFERRED, UNSUPPORTED, NO_DATA = 0, 1, 2, 3, 255
CLASS_NAMES = {NO_CHANGE: 'NO_CHANGE', OBSERVED: 'OBSERVED', INFERRED: 'INFERRED',
               UNSUPPORTED: 'UNSUPPORTED', NO_DATA: 'NO_DATA'}


def upsample(a, scale: int):
    """Replicate every 10 m pixel into scale x scale sub-pixels (the parent replicated to the 2.5 m grid)."""
    return np.repeat(np.repeat(np.asarray(a), scale, axis=-2), scale, axis=-1)


def parent_mask(drop, valid, threshold: float):
    """10 m parent change: NDVI drop above the (provisional) threshold on valid pixels only."""
    drop = np.asarray(drop)
    return np.asarray(valid, bool) & np.isfinite(drop) & (drop > threshold)


def classify(d, sigma, parent, nodata, k: float):
    """OBSERVED = S and P; INFERRED = P only; UNSUPPORTED = S only; NO_DATA (incl. non-finite d or sigma) overrides all.

    S = d > k*sigma (strict). `parent` and `nodata` are already on the same grid as d.
    """
    d, sigma = np.asarray(d, dtype=np.float64), np.asarray(sigma, dtype=np.float64)
    bad = np.asarray(nodata, bool) | ~np.isfinite(d) | ~np.isfinite(sigma)
    with np.errstate(invalid='ignore'):
        s = d > k * sigma
    p = np.asarray(parent, bool)
    cls = np.full(d.shape, NO_CHANGE, dtype=np.uint8)
    cls[s & p] = OBSERVED
    cls[~s & p] = INFERRED
    cls[s & ~p] = UNSUPPORTED
    cls[bad] = NO_DATA
    return cls


def object_rho(delta, labels, block: int = 4):
    """rho = ||P D||^2 / ||D||^2 per object (label 1..n), D = delta restricted to the object (zero elsewhere),
    P = replace every block x block tile by its mean. A fraction of signal energy in [0, 1], not a probability.
    Returns an array of length n; NaN for an object with zero energy."""
    delta = np.asarray(delta, dtype=np.float64)
    labels = np.asarray(labels)
    h, w = delta.shape
    if h % block or w % block:
        raise ValueError('grid must be a whole number of blocks')
    n = int(labels.max()) if labels.size else 0
    if n == 0:
        return np.zeros(0)
    rr, cc = np.nonzero(labels > 0)
    lab = labels[rr, cc].astype(np.int64)
    vals = delta[rr, cc]
    nb = (h // block) * (w // block)
    key = lab * nb + (rr // block) * (w // block) + (cc // block)
    uniq, inv = np.unique(key, return_inverse=True)
    block_sum = np.bincount(inv, weights=vals)
    proj = np.bincount(uniq // nb, weights=block_sum ** 2 / (block * block), minlength=n + 1)[1:]
    energy = np.bincount(lab, weights=vals ** 2, minlength=n + 1)[1:]
    out = np.full(n, np.nan)
    np.divide(proj, energy, out=out, where=energy > 0)
    return out
