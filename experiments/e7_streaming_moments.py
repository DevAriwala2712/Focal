"""E7: per-pixel streaming mean/variance (Welford) vs two-pass numpy; RAM stays flat in run count."""
from __future__ import annotations

import tracemalloc

import numpy as np

from experiments.common import run_cli


class Welford:
    """Per-pixel streaming moments with float64 accumulation. State size is independent of run count."""

    def __init__(self, shape):
        self.shape = tuple(shape)
        self.count = 0
        self.mean = np.zeros(self.shape, dtype=np.float64)
        self.m2 = np.zeros(self.shape, dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        if x.shape != self.shape:
            raise ValueError(f'shape {x.shape} does not match accumulator shape {self.shape}')
        self.count += 1
        delta = x - self.mean                       # float64 after promotion
        self.mean += delta / self.count
        self.m2 += delta * (x - self.mean)

    def variance(self, ddof: int = 0) -> np.ndarray:
        if self.count - ddof <= 0:
            raise ValueError(f'ddof={ddof} needs more than {ddof} runs, have {self.count}')
        return self.m2 / (self.count - ddof)


def _run(seed: int, index: int, shape) -> np.ndarray:
    """One synthetic 'run' (float32, as a model would emit); regenerated on demand, never stored."""
    rng = np.random.default_rng([seed, index])
    return rng.normal(0.3, 0.1, size=shape).astype('float32')


def _peak(fn) -> tuple[object, int]:
    tracemalloc.start()
    tracemalloc.reset_peak()
    try:
        out = fn()
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    return out, peak


def measure(cfg: dict, seed: int) -> dict:
    shape, ddof = tuple(cfg['shape']), cfg['ddof']
    rows = []
    for n in cfg['run_counts']:
        def stream():
            w = Welford(shape)
            for i in range(n):
                w.update(_run(seed, i, shape))
            return w.mean, w.variance(ddof)
        (mean, var), stream_peak = _peak(stream)
        def stacked():
            stack = np.stack([_run(seed, i, shape) for i in range(n)]).astype('float64')
            return stack.mean(0), stack.var(0, ddof=ddof)
        (ref_mean, ref_var), stacked_peak = _peak(stacked)
        rows.append({'runs': n,
                     'max_abs_error_mean': float(np.abs(mean - ref_mean).max()),
                     'max_abs_error_var': float(np.abs(var - ref_var).max()),
                     'streaming_peak_bytes': stream_peak, 'stacked_peak_bytes': stacked_peak})
    error_ok = all(max(r['max_abs_error_mean'], r['max_abs_error_var']) <= cfg['max_abs_error'] for r in rows)
    ram_flat = rows[-1]['streaming_peak_bytes'] <= rows[0]['streaming_peak_bytes'] * cfg['ram_growth_tolerance']
    return {'per_run_count': rows, 'error_ok': error_ok, 'ram_flat': ram_flat}


def probe(cfg, root):
    """E7: Welford streaming moments vs two-pass numpy (synthetic stacks)."""
    settings = cfg['e7']
    m = measure(settings, cfg['seed'])
    return {'status': 'PASS' if m['error_ok'] and m['ram_flat'] else 'FAIL', 'evidence': 'synthetic',
            'criteria': {'max_abs_error': settings['max_abs_error'],
                         'ram_growth_tolerance': settings['ram_growth_tolerance'], 'ddof': settings['ddof'],
                         'accumulation_dtype': 'float64'},
            'measurements': m,
            'limitations': ['Synthetic Gaussian stacks prove the accumulator mechanics only.',
                            'tracemalloc counts Python/numpy allocations, not process RSS or GPU memory.',
                            'Inputs are float32 (as model output would be); both methods see the same float32 values.']}


if __name__ == '__main__':
    run_cli('e7', probe)
