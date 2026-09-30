"""OpenSR-test data access for the A2 benchmark: pinned download with byte log, and a pickle audit.

opensr_test.load() does a bare pickle.load() of a file fetched without any size or hash check. Unpickling can execute
arbitrary code, so here every file is (1) fetched from a pinned dataset revision, (2) verified against the LFS sha256
published by the Hub, (3) scanned: a stub Unpickler records every global the pickle references WITHOUT importing or calling
anything, and loading proceeds only if all of them are on an allow-list of array/dataframe types.
"""
from __future__ import annotations

import hashlib
import io
import pickle
import time
from pathlib import Path

import requests

HF = 'https://huggingface.co'
REPO = 'isp-uv-es/opensr-test'
VERSION_KEY = '021'            # what opensr_test.load(version="v3") maps to (opensr_test/dataset.py, v1.3.3)

# Globals that a pickle of numpy arrays + a pandas DataFrame legitimately references. Anything else blocks loading.
ALLOWED_PREFIXES = ('numpy', 'pandas', 'collections.OrderedDict', 'builtins.')
ALLOWED_BUILTINS = {'dict', 'list', 'tuple', 'set', 'frozenset', 'slice', 'range', 'object', 'bytearray', 'complex',
                    'int', 'float', 'str', 'bytes', 'bool', 'getattr'}


class ByteLog:
    """Per-process log of every network request and the bytes it moved; enforces a hard cap."""

    def __init__(self, path: Path, cap_bytes: int):
        self.path, self.cap, self.total = Path(path), int(cap_bytes), 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(f'# A2 byte log, cap {self.cap} bytes, started {time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}\n')

    def add(self, what: str, nbytes: int):
        self.total += nbytes
        with self.path.open('a') as f:
            f.write(f'{time.strftime("%H:%M:%S", time.gmtime())}\t{nbytes}\t{self.total}\t{what}\n')
        if self.total > self.cap:
            raise RuntimeError(f'byte cap exceeded: {self.total} > {self.cap}')


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def hub_listing(retries: int = 3, backoff: float = 2.0, log: ByteLog | None = None) -> dict:
    """Pinned commit sha and {dataset: (size, lfs_sha256)} for the v3 pickles, from the Hub API."""
    def get(url):
        for attempt in range(retries + 1):
            try:
                r = requests.get(url, timeout=45)
                r.raise_for_status()
                if log is not None:
                    log.add(url, len(r.content))
                return r.json()
            except Exception as exc:
                if attempt == retries:
                    raise RuntimeError(f'{url}: failed after {attempt + 1} attempts: {exc}') from exc
                time.sleep(backoff * 2 ** attempt)
    meta = get(f'{HF}/api/datasets/{REPO}')
    sha = meta['sha']
    tree = get(f'{HF}/api/datasets/{REPO}/tree/{sha}/{VERSION_KEY}?recursive=true')
    files = {}
    for item in tree:
        parts = item['path'].split('/')
        if len(parts) == 3 and parts[2] == parts[1] + '.pkl' and 'lfs' in item:
            files[parts[1]] = {'size': item['size'], 'sha256': item['lfs']['oid'], 'path': item['path']}
    return {'revision': sha, 'files': files}


def fetch_pickle(dataset: str, listing: dict, cache: Path, log: ByteLog, retries: int = 3, backoff: float = 2.0) -> Path:
    """Download `021/<ds>/<ds>.pkl` at the pinned revision into cache, verify size and sha256. Retries 3x then names the URL."""
    info = listing['files'][dataset]
    target = Path(cache) / f'{dataset}.pkl'
    if target.is_file() and target.stat().st_size == info['size'] and sha256_file(target) == info['sha256']:
        log.add(f'cache hit {target.name} (0 network bytes)', 0)
        return target
    url = f'{HF}/datasets/{REPO}/resolve/{listing["revision"]}/{info["path"]}'
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix('.pkl.part')
    for attempt in range(retries + 1):
        got = 0
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with part.open('wb') as f:
                    for block in r.iter_content(1 << 20):
                        got += len(block)
                        if log.total + got > log.cap:
                            raise RuntimeError(f'byte cap would be exceeded fetching {url}')
                        f.write(block)
            log.add(f'GET {url}', got)
            if part.stat().st_size != info['size'] or sha256_file(part) != info['sha256']:
                raise ValueError('size/sha256 mismatch against the Hub LFS record')
            part.replace(target)
            return target
        except Exception as exc:
            if got:
                log.add(f'FAILED attempt {attempt + 1} {url}: {exc}', got)
            part.unlink(missing_ok=True)
            if attempt == retries:
                raise RuntimeError(f'{url}: failed after {attempt + 1} attempts: {exc}') from exc
            time.sleep(backoff * 2 ** attempt)


class _Stub:
    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        pass


class _AuditUnpickler(pickle.Unpickler):
    """Records every (module, name) the pickle asks for; returns an inert stub so that nothing is imported or executed."""

    def __init__(self, f):
        super().__init__(f)
        self.seen = set()

    def find_class(self, module, name):
        self.seen.add(f'{module}.{name}')
        return _Stub


def audit_globals(path: Path):
    """(sorted globals referenced by the pickle, completed). Collected without executing any of them.
    `completed` is False if the stub pass aborted early, in which case the list may be incomplete and loading is refused."""
    with Path(path).open('rb') as f:
        up = _AuditUnpickler(f)
        try:
            up.load()
            done = True
        except Exception:
            done = False
    return sorted(up.seen), done


def disallowed(globals_seen) -> list[str]:
    bad = []
    for g in globals_seen:
        if g.startswith('builtins.') and g.split('.', 1)[1] in ALLOWED_BUILTINS:
            continue
        if g.startswith(('numpy', 'pandas')) or g == 'collections.OrderedDict':
            continue
        bad.append(g)
    return bad


class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        g = f'{module}.{name}'
        if disallowed([g]):
            raise pickle.UnpicklingError(f'global not on allow-list: {g}')
        return super().find_class(module, name)


def safe_load(path: Path):
    """Audit, then load with an allow-list unpickler. Raises if the file references anything outside numpy/pandas."""
    seen, done = audit_globals(path)
    if not done:
        raise pickle.UnpicklingError(f'{path.name}: audit pass did not complete, global list may be partial; refusing to load')
    bad = disallowed(seen)
    if bad:
        raise pickle.UnpicklingError(f'{path.name} references non-allow-listed globals: {bad}')
    with Path(path).open('rb') as f:
        return _SafeUnpickler(io.BufferedReader(f)).load(), seen
