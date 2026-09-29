"""Dihedral transforms, streaming (Welford) moments and NDVI."""
from __future__ import annotations

import numpy as np


def dihedral(x, k: int):
    """One of the 8 dihedral variants (k = 0..7) acting on the last two axes: rot90 k%4 times, flipped when k >= 4."""
    if not 0 <= k < 8:
        raise ValueError('k must be 0..7')
    y = np.flip(x, axis=-1) if k >= 4 else x
    return np.ascontiguousarray(np.rot90(y, k % 4, axes=(-2, -1)))


def inverse_dihedral(y, k: int):
    """Exact inverse of `dihedral(., k)`."""
    x = np.rot90(y, -(k % 4), axes=(-2, -1))
    return np.ascontiguousarray(np.flip(x, axis=-1) if k >= 4 else x)


class Welford:
    """Per-pixel streaming mean/variance in float64. NaN samples are skipped per pixel (count tracks valid samples)."""

    def __init__(self, shape):
        self.count = np.zeros(shape, dtype=np.float64)
        self.mean = np.zeros(shape, dtype=np.float64)
        self.m2 = np.zeros(shape, dtype=np.float64)

    def update(self, x, where=None):
        x = np.asarray(x, dtype=np.float64)
        ok = np.isfinite(x)
        if where is not None:
            ok &= where
        self.count += ok
        delta = np.where(ok, x - self.mean, 0.0)
        self.mean += np.divide(delta, self.count, out=np.zeros_like(delta), where=self.count > 0)
        delta2 = np.where(ok, x - self.mean, 0.0)
        self.m2 += delta * delta2

    def update_at(self, region, x, where=None):
        """Update only `region` (a tuple of slices) in place; the moments outside it are untouched."""
        view = Welford.__new__(Welford)
        view.count, view.mean, view.m2 = self.count[region], self.mean[region], self.m2[region]
        view.update(x, where)

    def variance(self, ddof: int = 1):
        out = np.full(self.count.shape, np.nan)
        np.divide(self.m2, self.count - ddof, out=out, where=self.count > ddof)
        return out

    def std(self, ddof: int = 1):
        return np.sqrt(self.variance(ddof))

    def valid_mean(self, min_count: int = 1):
        return np.where(self.count >= min_count, self.mean, np.nan)


def ndvi(red, nir, min_denominator: float):
    """(NIR - R) / (NIR + R); NaN where NIR + R < min_denominator (ill-conditioned: dark water, shadow)."""
    red = np.asarray(red, dtype=np.float64)
    nir = np.asarray(nir, dtype=np.float64)
    denom = nir + red
    out = np.full(denom.shape, np.nan)
    np.divide(nir - red, denom, out=out, where=denom >= min_denominator)
    return out
