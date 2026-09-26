"""Configuration, bounded downloads and honest result reporting."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml


def digest(path: Path, algorithm: str = 'sha256') -> str:
    h = hashlib.new(algorithm)
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    # Windows sync/indexing can briefly hold the destination open. Preserve
    # atomic replacement and the original error if the lock does not clear.
    for attempt in range(4):
        try:
            temp.replace(path)
            break
        except PermissionError:
            if attempt == 3:
                raise
            time.sleep(.1 * 2 ** attempt)


def retry(operation, policy: dict, label: str):
    for attempt in range(policy['retries'] + 1):
        try:
            return operation()
        except Exception as exc:
            if attempt == policy['retries']:
                raise RuntimeError(f'{label}: failed after {attempt + 1} attempts: {exc}') from exc
            time.sleep(policy['backoff_seconds'] * 2 ** attempt)


def download(url: str, target: Path, policy: dict, *, sha256=None, md5=None) -> Path:
    """Atomic, size-bounded retrieval. Cache identity includes URL and digest."""
    target = Path(target)
    sidecar = target.with_suffix(target.suffix + '.source.json')
    def valid():
        if not target.is_file():
            return False
        if sha256:
            return digest(target) == sha256
        if md5:
            return digest(target, 'md5') == md5
        if not sidecar.is_file():
            return False
        record = json.loads(sidecar.read_text(encoding='utf-8'))
        return record['url'] == url and record['sha256'] == digest(target)
    if valid():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + '.part')
    def retrieve():
        try:
            with requests.get(url, stream=True, timeout=policy['timeout_seconds']) as response:
                response.raise_for_status()
                limit = policy['max_download_bytes']
                if int(response.headers.get('Content-Length', 0)) > limit:
                    raise ValueError(f'download size exceeds configured limit {limit} bytes')
                size = 0
                with partial.open('wb') as stream:
                    for block in response.iter_content(1024 * 1024):
                        size += len(block)
                        if size > limit:
                            raise ValueError(f'download size exceeds configured limit {limit} bytes')
                        stream.write(block)
            if sha256 and digest(partial) != sha256:
                raise ValueError('SHA256 mismatch')
            if md5 and digest(partial, 'md5') != md5:
                raise ValueError('MD5 mismatch')
            partial.replace(target)
            write_json(sidecar, {'url': url, 'sha256': digest(target), 'bytes': target.stat().st_size})
            return target
        finally:
            partial.unlink(missing_ok=True)
    return retry(retrieve, policy, url)


def request_json(url: str, policy: dict, *, body=None) -> dict:
    def get():
        method = requests.post if body is not None else requests.get
        kwargs = {'json': body} if body is not None else {}
        with method(url, timeout=policy['timeout_seconds'], **kwargs) as response:
            response.raise_for_status()
            return response.json()
    return retry(get, policy, url)


def retry_tiles(operation, initial: int, minimum: int, oom_type, cleanup=lambda: None):
    if initial < minimum or minimum <= 0:
        raise ValueError('invalid tile limits')
    failed = []
    size = initial
    while True:
        try:
            return operation(size), size, failed
        except oom_type as exc:
            exc.__traceback__ = None
            failed.append(size)
            cleanup()
            if size <= minimum:
                raise RuntimeError(f'CUDA OOM at minimum tile size {minimum}; cannot proceed') from None
            size = max(minimum, size // 2)


def load_config(path: str | Path):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text(encoding='utf-8-sig'))
    if cfg['schema_version'] != 1 or cfg['model']['bands'] != ['B04', 'B03', 'B02', 'B08']:
        raise ValueError('Unsupported schema or model band order')
    return cfg, path.parent.parent, digest(path)


def run_cli(name: str, probe):
    parser = argparse.ArgumentParser(description=probe.__doc__)
    parser.add_argument('--config', default='configs/phase0.yaml')
    args = parser.parse_args()
    cfg, root, config_hash = load_config(args.config)
    started = datetime.now(timezone.utc).isoformat()
    try:
        result = probe(cfg, root)
    except Exception as exc:
        result = {'status': 'BLOCKED', 'reason': f'{type(exc).__name__}: {exc}'}
    result.update(risk=name, started_utc=started, finished_utc=datetime.now(timezone.utc).isoformat(),
                  config_sha256=config_hash)
    write_json(root / cfg['paths']['results'] / 'config_snapshots' / f'{config_hash}.json', cfg)
    out = root / cfg['paths']['results'] / f'{name}.json'
    write_json(out, result)
    print(json.dumps(result, indent=2))
    print(f'Result saved to {out}')
    raise SystemExit(0 if result['status'] == 'PASS' else 2)
