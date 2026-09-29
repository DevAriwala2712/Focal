"""Load configs/wayanad_evidence.yaml together with the pinned phase-0 config; one combined hash identifies both."""
from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from risk.common import digest, load_config

DEFAULT = 'configs/wayanad_evidence.yaml'


def load(path: str | Path = DEFAULT):
    """Return (config, repo_root, config_hash). The phase-0 config is nested under cfg['phase0']."""
    path = Path(path).resolve()
    root = path.parent.parent
    cfg = yaml.safe_load(path.read_text(encoding='utf-8-sig'))
    if cfg['schema_version'] != 1:
        raise ValueError('Unsupported wayanad_evidence schema')
    phase0, _, phase0_hash = load_config(root / cfg['phase0_config'])
    if cfg['tiling']['tile'] != phase0['model']['native_tile'] or cfg['tiling']['scale'] != phase0['model']['scale']:
        raise ValueError('tiling.tile/scale must equal the pinned model native_tile/scale')
    cfg['phase0'] = phase0
    return cfg, root, hashlib.sha256((digest(path) + phase0_hash).encode()).hexdigest()


def outputs(cfg, root) -> Path:
    out = root / cfg['paths']['outputs']
    out.mkdir(parents=True, exist_ok=True)
    return out


def cache(cfg, root) -> Path:
    out = root / cfg['paths']['cache']
    out.mkdir(parents=True, exist_ok=True)
    return out
