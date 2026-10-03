"""Shared plumbing for the F13 discovery wave: the pre-registration, JSON hygiene, the result schema.

Every threshold, seed and replicate count comes from configs/f13_discovery.yaml (pre-registered in 0978e8a).
Nothing here edits that file or any existing result.
"""
from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from experiments.common import environment_record
from risk.common import write_json

ROOT = Path(__file__).resolve().parent.parent
PREREG = ROOT / 'configs' / 'f13_discovery.yaml'
RESULTS_DIR = ROOT / 'experiments' / 'results' / 'f13'
STATUSES = ('PASS', 'FAIL', 'BLOCKED')


class Blocked(Exception):
    """A named input is missing. Never converted to PASS, never substituted."""


def prereg_sha256() -> str:
    return hashlib.sha256(PREREG.read_bytes()).hexdigest()


def load_prereg() -> dict:
    return yaml.safe_load(PREREG.read_text(encoding='utf-8'))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def require_committed_prereg() -> dict:
    """Precondition: the yaml is tracked and byte-identical to HEAD. Returns {'sha256', 'commit'}."""
    def git(*args):
        return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True)
    tracked = git('ls-files', '--error-unmatch', str(PREREG.relative_to(ROOT)))
    if tracked.returncode != 0:
        raise Blocked('configs/f13_discovery.yaml is not tracked by git; finish Pass 0 (commit it) first')
    if git('diff', '--quiet', 'HEAD', '--', str(PREREG.relative_to(ROOT))).returncode != 0:
        raise Blocked('configs/f13_discovery.yaml differs from HEAD; the pre-registration must be committed unchanged')
    commit = git('log', '-1', '--format=%H', '--', str(PREREG.relative_to(ROOT))).stdout.strip()
    return {'sha256': prereg_sha256(), 'commit': commit}


def to_jsonable(x):
    """Recursively convert numpy types; NaN/inf -> None (write_json forbids NaN)."""
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return to_jsonable(x.tolist())
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return v if np.isfinite(v) else None
    return x


def measurement(name, estimate, numerator=None, denominator=None, lo=None, hi=None, replicates=None, seed=None,
                unit=None, **extra) -> dict:
    m = {'name': name, 'estimate': estimate, 'numerator': numerator, 'denominator': denominator,
         'lo': lo, 'hi': hi, 'replicates': replicates, 'seed': seed, 'unit': unit}
    m.update(extra)
    return m


def write_result(exp_id: str, result: dict, started: str) -> dict:
    """Complete and write experiments/results/f13/<id>.json. Refuses to overwrite nothing it should not: only f13/."""
    if result.get('status') not in STATUSES:
        raise ValueError(f'status must be one of {STATUSES}')
    if result.get('evidence') != 'real':
        raise ValueError('F13 pass 1 only reports evidence: real; BLOCKED results also say real (nothing substituted)')
    out = dict(result)
    out.update(experiment=f'f13_{exp_id}', started_utc=started, finished_utc=utc_now(),
               config_sha256=prereg_sha256(), config_file='configs/f13_discovery.yaml')
    out.setdefault('limitations', [])
    out['environment'] = environment_record()
    out = to_jsonable(out)
    write_json(RESULTS_DIR / f'{exp_id}.json', out)
    return out


def exit_code(status: str) -> int:
    return 0 if status == 'PASS' else 2
