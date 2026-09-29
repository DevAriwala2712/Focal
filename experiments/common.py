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

    def __init__(self, reason: str, evidence: str = 'real'):
        super().__init__(reason)
        self.evidence = evidence


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
        result = {'status': 'BLOCKED', 'evidence': exc.evidence, 'reason': str(exc)}
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
