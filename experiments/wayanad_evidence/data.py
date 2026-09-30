"""Per-date arrays from the cache: reflectance in model order, SCL validity, NDVI at 10 m."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.wayanad_evidence.fetch import MODEL_BANDS, reflectance
from experiments.wayanad_evidence.stats import ndvi


def scl_valid(scl, im: dict) -> np.ndarray:
    """Pixel usable: not cloud, cloud-shadow/dark or invalid in SCL (phase-0 classes). SCL is a mask only."""
    return ~np.isin(scl, list(im['cloud_classes']) + list(im['shadow_classes']) + list(im['invalid_classes']))


def load_date(cfg, cache_dir: Path, date: str) -> dict:
    """{'refl': (4,H,W) float32 [B04,B03,B02,B08], 'scl', 'valid_scl', 'ndvi', 'valid', 'meta'} for one acquisition."""
    path = Path(cache_dir) / f'{date}.npz'
    if not path.is_file():
        raise FileNotFoundError(f'{path} is not cached; run step1_baseline to fetch it')
    with np.load(path) as z:
        dn, scl, taken, meta = z['dn'], z['scl'], z['taken'], json.loads(str(z['meta']))
    assert meta['band_order'] == MODEL_BANDS
    refl = reflectance(dn, cfg['radiometry'])
    valid_scl = scl_valid(scl, cfg['phase0']['imagery']) & taken
    nd = ndvi(refl[0], refl[3], cfg['radiometry']['min_denominator'])
    return {'refl': refl, 'scl': scl, 'valid_scl': valid_scl, 'ndvi': nd,
            'valid': valid_scl & np.isfinite(nd), 'meta': meta}
