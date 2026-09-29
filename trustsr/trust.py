"""Dihedral sensitivity estimates for four-band model reconstructions."""
from __future__ import annotations

import numpy as np


def dihedral(image: np.ndarray, index: int) -> np.ndarray:
    if index not in range(8):
        raise ValueError('Dihedral index must be 0..7')
    transformed = np.flip(image, -1) if index >= 4 else image
    return np.rot90(transformed, index % 4, axes=(-2, -1)).copy()


def undo_dihedral(image: np.ndarray, index: int) -> np.ndarray:
    if index not in range(8):
        raise ValueError('Dihedral index must be 0..7')
    unrotated = np.rot90(image, -(index % 4), axes=(-2, -1))
    return (np.flip(unrotated, -1) if index >= 4 else unrotated).copy()


def ndvi(image: np.ndarray, *, red_index: int = 0, nir_index: int = 3) -> np.ndarray:
    red, nir = image[red_index].astype(np.float32), image[nir_index].astype(np.float32)
    denominator = nir + red
    return np.divide(nir - red, denominator, out=np.full_like(red, np.nan), where=denominator > 1e-6)


def ndvi_moments(images, *, red_index: int = 0, nir_index: int = 3):
    count, mean, m2 = 0, None, None
    for image in images:
        value = ndvi(image, red_index=red_index, nir_index=nir_index)
        if not np.isfinite(value).all():
            raise ValueError('Invalid NDVI denominator or non-finite reflectance')
        count += 1
        if mean is None:
            mean, m2 = np.zeros_like(value), np.zeros_like(value)
        delta = value - mean
        mean += delta / count
        m2 += delta * (value - mean)
    if count == 0:
        raise ValueError('No clear model runs')
    return mean, np.sqrt(m2 / count)


def trust_date(lr: np.ndarray, sr_operator):
    """Run all eight transforms sequentially, computing NDVI before pooling."""
    return ndvi_moments(undo_dihedral(sr_operator(dihedral(lr, i)), i) for i in range(8))
