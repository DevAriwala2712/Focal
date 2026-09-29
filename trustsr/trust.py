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


def trust_period(images, sr_operator, valid_masks=None):
    """Pool each date's eight inversely transformed runs with streaming moments."""
    images = list(images)
    if not images:
        raise ValueError('No dates supplied')
    if valid_masks is None:
        valid_masks = [np.ones(image.shape[-2:], dtype=bool) for image in images]
    if len(valid_masks) != len(images):
        raise ValueError('One validity mask required per date')
    count, image_mean, ndvi_mean, ndvi_m2, valid = 0, None, None, None, None
    for image, mask in zip(images, valid_masks):
        if image.shape[0] != 4 or mask.shape != image.shape[-2:]:
            raise ValueError('Expected four bands and aligned 10 m validity')
        fine_valid = np.repeat(np.repeat(mask, 4, -2), 4, -1)
        for index in range(8):
            fine = undo_dihedral(sr_operator(dihedral(image, index)), index).astype(np.float32)
            value = ndvi(fine)
            run_valid = fine_valid & np.isfinite(value)
            valid = run_valid.copy() if valid is None else valid & run_valid
            value = np.nan_to_num(value, nan=0, posinf=0, neginf=0)
            count += 1
            if image_mean is None:
                image_mean = np.zeros_like(fine)
                ndvi_mean = np.zeros_like(value)
                ndvi_m2 = np.zeros_like(value)
            image_mean += (fine - image_mean) / count
            delta = value - ndvi_mean
            ndvi_mean += delta / count
            ndvi_m2 += delta * (value - ndvi_mean)
    ndvi_mean[~valid] = np.nan
    std = np.sqrt(ndvi_m2 / count)
    std[~valid] = np.nan
    return {'runs': count, 'image_mean': image_mean, 'ndvi_mean': ndvi_mean,
            'ndvi_std': std, 'valid': valid}
