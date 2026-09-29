"""SR fidelity and common-grid binary detection metrics."""
from __future__ import annotations

import math

import numpy as np


def _validate_pair(a, b):
    if np.shape(a) != np.shape(b) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Metrics require same-shape, finite arrays')


def spectral_rmse(lr, sr, *, scale=4):
    lr, sr = np.asarray(lr), np.asarray(sr)
    if lr.ndim != 3 or sr.shape != (lr.shape[0], lr.shape[1]*scale, lr.shape[2]*scale):
        raise ValueError('Expected C,H,W LR and exact x4 SR')
    parent = sr.reshape(lr.shape[0], lr.shape[1], scale, lr.shape[2], scale).mean((2, 4))
    return float(np.sqrt(np.mean(np.square(parent - lr))))


def psnr(truth, prediction, *, data_range=1.0):
    _validate_pair(truth, prediction)
    if data_range <= 0:
        raise ValueError('Data range must be positive')
    mse = float(np.mean(np.square(np.asarray(truth) - np.asarray(prediction))))
    return None if mse == 0 else 10 * math.log10(data_range**2 / mse)


def f1_score(truth, prediction, valid=None):
    _validate_pair(truth, prediction)
    truth, prediction = np.asarray(truth, bool), np.asarray(prediction, bool)
    valid = np.ones_like(truth, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    if valid.shape != truth.shape:
        raise ValueError('Validity mask shape mismatch')
    tp = int(np.count_nonzero(truth & prediction & valid))
    fp = int(np.count_nonzero(~truth & prediction & valid))
    fn = int(np.count_nonzero(truth & ~prediction & valid))
    return 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None


def assess_sr(lr, hr, prediction, *, split: str, data_range=1.0):
    _validate_pair(hr, prediction)
    if np.shape(hr) != (np.shape(lr)[0], np.shape(lr)[1]*4, np.shape(lr)[2]*4):
        raise ValueError('High-resolution target must align x4 with LR')
    from skimage.metrics import structural_similarity
    ssim = float(np.mean([structural_similarity(hr[i], prediction[i], data_range=data_range)
                          for i in range(hr.shape[0])]))
    return {'split': split, 'held_out': split in {'val', 'test'}, 'psnr_db': psnr(hr, prediction, data_range=data_range),
            'ssim': ssim, 'spectral_rmse': spectral_rmse(lr, prediction), 'data_range': data_range}
