"""Shared plumbing for the six-laws experiments: config, hashing, result schema.

Reuses risk.common (digest, write_json, load_config) and risk.model (pinned loader);
nothing here downloads or loads a model itself.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from risk.common import digest, load_config, write_json

STATUSES = ('PASS', 'FAIL', 'BLOCKED')
EVIDENCE = ('synthetic', 'real')


class Blocked(Exception):
    """A prerequisite (real data, GPU, network) is missing. Never converted to PASS."""

    def __init__(self, reason: str, evidence: str = 'real', limitations=None):
        super().__init__(reason)
        self.evidence = evidence
        self.limitations = list(limitations or [])


def array_sha256(array: np.ndarray) -> str:
    """SHA-256 over dtype, shape and C-ordered bytes, so layout cannot change the hash."""
    a = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(f'{a.dtype.str}|{a.shape}|'.encode())
    h.update(a.tobytes())
    return h.hexdigest()


def load_experiment_config(path='configs/experiments.yaml'):
    """Return (config, repo_root, config_hash). The pinned phase-0 config is nested under 'phase0'."""
    path = Path(path).resolve()
    root = path.parent.parent
    cfg = yaml.safe_load(path.read_text(encoding='utf-8-sig'))
    if cfg['schema_version'] != 1:
        raise ValueError('Unsupported experiments schema')
    phase0, _, phase0_hash = load_config(root / cfg['phase0_config'])
    cfg['phase0'] = phase0
    combined = hashlib.sha256((digest(path) + phase0_hash).encode()).hexdigest()
    return cfg, root, combined


def finalize_result(name: str, result: dict, config_hash: str, started: str) -> dict:
    if result.get('status') not in STATUSES:
        raise ValueError(f'status must be one of {STATUSES}, got {result.get("status")!r}')
    if result.get('evidence') not in EVIDENCE:
        raise ValueError(f'evidence must be one of {EVIDENCE}, got {result.get("evidence")!r}')
    out = dict(result)
    out.setdefault('limitations', [])
    out.update(experiment=name, risk=name, started_utc=started,
               finished_utc=datetime.now(timezone.utc).isoformat(), config_sha256=config_hash)
    return out


def run_probe(name: str, probe, cfg, root, config_hash: str) -> dict:
    started = datetime.now(timezone.utc).isoformat()
    try:
        result = probe(cfg, root)
    except Blocked as exc:
        result = {'status': 'BLOCKED', 'evidence': exc.evidence, 'reason': str(exc), 'limitations': exc.limitations}
    except Exception as exc:  # a bug is a FAIL with its error, never a silent BLOCKED
        result = {'status': 'FAIL', 'evidence': 'synthetic',
                  'error': f'{type(exc).__name__}: {exc}'}
    return finalize_result(name, result, config_hash, started)


def run_cli(name: str, probe):
    parser = argparse.ArgumentParser(description=probe.__doc__)
    parser.add_argument('--config', default='configs/experiments.yaml')
    args = parser.parse_args()
    cfg, root, config_hash = load_experiment_config(args.config)
    result = run_probe(name, probe, cfg, root, config_hash)
    results = root / cfg['paths']['results']
    write_json(results / 'config_snapshots' / f'{config_hash}.json', cfg)
    write_json(results / f'{name}.json', result)
    print(json.dumps(result, indent=2, allow_nan=False))
    print(f'Result saved to {results / (name + ".json")}')
    raise SystemExit(0 if result['status'] == 'PASS' else 2)


# ---- inputs ---------------------------------------------------------------------------

_CLASS_SPECTRA = np.array([          # B04, B03, B02, B08 reflectance: forest, crop, bare soil, water
    [0.03, 0.05, 0.03, 0.35],
    [0.08, 0.10, 0.06, 0.30],
    [0.20, 0.22, 0.17, 0.28],
    [0.02, 0.03, 0.04, 0.02]], dtype=np.float64)


def _smooth_noise(rng, n: int, sigma: float) -> np.ndarray:
    freq = np.fft.fftfreq(n)
    kernel = np.exp(-2 * (np.pi * sigma) ** 2 * (freq[:, None] ** 2 + freq[None, :] ** 2))
    field = np.fft.ifft2(np.fft.fft2(rng.standard_normal((n, n))) * kernel).real
    return (field - field.mean()) / field.std()


def synthetic_scene(size: int, seed: int, full: int | None = None) -> np.ndarray:
    """Deterministic RGBN-like reflectance scene (B04,B03,B02,B08), float32 in [0,1].

    Generated at `full` px then cropped to `size`, so crops of one scene agree.
    Synthetic: exercises mechanics only, never image-quality claims.
    """
    full = full or size
    rng = np.random.default_rng(seed)
    classes = np.digitize(_smooth_noise(rng, full, 12.0), [-0.8, 0.0, 0.9])
    texture = 1 + 0.12 * _smooth_noise(rng, full, 1.5)
    scene = _CLASS_SPECTRA[classes].transpose(2, 0, 1) * texture[None]
    scene += 0.003 * rng.standard_normal(scene.shape)
    return np.clip(scene, 0.0, 1.0).astype('float32')[:, :size, :size]


def load_real_crop(settings: dict, root):
    """Read a cached 4-band 10 m GeoTIFF as reflectance. Anything missing or unverified -> Blocked."""
    import rasterio
    path = Path(root) / settings['path']
    if not path.is_file():
        raise Blocked(f'real 10 m RGBN crop not cached: expected {settings["path"]} '
                      '(4-band B04,B03,B02,B08 GeoTIFF, 10 m)', evidence='real')
    if settings.get('dn_scale') is None or settings.get('dn_offset') is None:
        raise Blocked(f'{settings["path"]} present but dn_scale/dn_offset unset; radiometry must be '
                      'verified from the provider asset, not assumed', evidence='real')
    with rasterio.open(path) as src:
        if src.count != 4 or src.crs is None:
            raise Blocked(f'{settings["path"]} must have 4 bands and a CRS', evidence='real')
        if not (abs(src.transform.a) == 10 == abs(src.transform.e)):
            raise Blocked(f'{settings["path"]} must be a 10 m grid, got {src.transform.a} x {src.transform.e}',
                          evidence='real')
        dn = src.read().astype('float64')
        return ((dn + settings['dn_offset']) / settings['dn_scale']).astype('float32'), src.transform, src.crs


def ndvi(array: np.ndarray, min_denominator: float, red: int = 0, nir: int = 3) -> np.ndarray:
    """NDVI from a (4,H,W) B04,B03,B02,B08 array; NaN where red+nir < min_denominator (unstable)."""
    r, n = array[red].astype(np.float64), array[nir].astype(np.float64)
    denom = r + n
    with np.errstate(invalid='ignore', divide='ignore'):
        out = np.where(denom >= min_denominator, (n - r) / denom, np.nan)
    return out.astype(np.float32)


# ---- change semantics (docs/design.md) -----------------------------------------------------

CLASS_CODES = {'NO_CHANGE': 0, 'OBSERVED': 1, 'INFERRED': 2, 'UNSUPPORTED': 3, 'NO_DATA': 4}


def classify_change(d, sigma, parent, no_data, k: float) -> np.ndarray:
    """S = d > k*sigma, P = parent drop. OBSERVED S&P, INFERRED P only, UNSUPPORTED S only; NO_DATA overrides."""
    s = np.asarray(d) > k * np.asarray(sigma)
    p = np.asarray(parent, dtype=bool)
    out = np.full(s.shape, CLASS_CODES['NO_CHANGE'], dtype=np.uint8)
    out[s & p] = CLASS_CODES['OBSERVED']
    out[~s & p] = CLASS_CODES['INFERRED']
    out[s & ~p] = CLASS_CODES['UNSUPPORTED']
    out[np.asarray(no_data, dtype=bool)] = CLASS_CODES['NO_DATA']
    return out


def _average_ranks(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind='mergesort')
    sorted_a = a[order]
    new_group = np.r_[True, sorted_a[1:] != sorted_a[:-1]]
    group = np.cumsum(new_group) - 1
    starts = np.flatnonzero(new_group)
    ends = np.r_[starts[1:], a.size] - 1
    ranks = np.empty(a.size, dtype=np.float64)
    ranks[order] = ((starts + ends) / 2 + 1)[group]
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rho with average ranks for ties; NaN if either input is constant."""
    ra, rb = _average_ranks(np.ravel(a)), _average_ranks(np.ravel(b))
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float('nan') if denom == 0 else float((ra * rb).sum() / denom)


def synthetic_change_pair(size: int, seed: int, square, noise: float):
    """(pre, post, truth). Both dates share one scene; the square (r0, c0, h, w, 10 m px) is forest before and
    bare/burned after, so the NDVI drop there is well defined. post also gets `noise`. Synthetic."""
    scene = synthetic_scene(size, seed).astype(np.float64)
    rng = np.random.default_rng([seed, 1])
    r0, c0, h, w = square
    inside = (slice(None), slice(r0, r0 + h), slice(c0, c0 + w))
    pre = scene.copy()
    pre[inside] = _CLASS_SPECTRA[0][:, None, None] * (1 + 0.05 * rng.standard_normal((1, h, w)))
    post = scene.copy()
    post[inside] = np.array([0.25, 0.22, 0.18, 0.20])[:, None, None]
    post += noise * rng.standard_normal(post.shape)
    truth = np.zeros((size, size), dtype=bool)
    truth[r0:r0 + h, c0:c0 + w] = True
    return np.clip(pre, 0, 1).astype('float32'), np.clip(post, 0, 1).astype('float32'), truth


def parent_drop_mask(pre: np.ndarray, post: np.ndarray, threshold: float, min_denominator: float):
    """10 m parent mask from the ORIGINAL data: (NDVI_pre - NDVI_post > threshold, NDVI defined on both dates)."""
    drop = ndvi(pre, min_denominator) - ndvi(post, min_denominator)
    valid = np.isfinite(drop)
    return valid & (np.nan_to_num(drop, nan=0.0) > threshold), valid
