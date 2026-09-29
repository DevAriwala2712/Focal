"""CPU tests for the six-laws experiment helpers (mechanics only, synthetic inputs)."""
import numpy as np
import pytest


# ---- shared helpers -------------------------------------------------------------

def test_array_hash_binds_dtype_shape_and_bytes():
    from experiments.common import array_sha256
    a = np.arange(6, dtype='float32').reshape(2, 3)
    assert array_sha256(a) == array_sha256(a.copy())
    assert array_sha256(a) != array_sha256(a.reshape(3, 2))
    assert array_sha256(a) != array_sha256(a.astype('float64'))
    b = a.copy()
    b[0, 0] = np.nextafter(b[0, 0], np.float32(1))
    assert array_sha256(a) != array_sha256(b)


def test_array_hash_ignores_memory_layout():
    from experiments.common import array_sha256
    a = np.arange(12, dtype='float32').reshape(3, 4)
    assert array_sha256(a) == array_sha256(np.asfortranarray(a))


def test_result_requires_an_evidence_label():
    from experiments.common import finalize_result
    with pytest.raises(ValueError, match='evidence'):
        finalize_result('e0', {'status': 'PASS'}, 'h', 't0')
    with pytest.raises(ValueError, match='evidence'):
        finalize_result('e0', {'status': 'PASS', 'evidence': 'mixed'}, 'h', 't0')


def test_result_rejects_unknown_status_and_keeps_schema_keys():
    from experiments.common import finalize_result
    with pytest.raises(ValueError, match='status'):
        finalize_result('e0', {'status': 'MAYBE', 'evidence': 'synthetic'}, 'h', 't0')
    out = finalize_result('e0', {'status': 'BLOCKED', 'evidence': 'real', 'reason': 'no data'}, 'h', 't0')
    for key in ('status', 'evidence', 'experiment', 'started_utc', 'finished_utc', 'config_sha256', 'limitations'):
        assert key in out
    assert out['config_sha256'] == 'h'


def test_blocked_exception_becomes_blocked_not_fail():
    from experiments.common import Blocked, run_probe
    def probe(cfg, root):
        raise Blocked('CUDA GPU required', evidence='real')
    out = run_probe('e0', probe, {}, '.', 'h')
    assert out['status'] == 'BLOCKED' and 'CUDA' in out['reason'] and out['evidence'] == 'real'


def test_unexpected_exception_is_a_failure_not_a_block():
    from experiments.common import run_probe
    def probe(cfg, root):
        raise KeyError('bug')
    out = run_probe('e0', probe, {}, '.', 'h')
    assert out['status'] == 'FAIL' and 'KeyError' in out['error']


# ---- E7 streaming moments ----------------------------------------------------------

def test_welford_matches_two_pass_numpy_in_float64():
    from experiments.e7_streaming_moments import Welford
    rng = np.random.default_rng(0)
    stack = rng.normal(0.3, 0.1, size=(8, 5, 7)).astype('float32')
    w = Welford((5, 7))
    for run in stack:
        w.update(run)
    ref = stack.astype('float64')
    assert np.abs(w.mean - ref.mean(0)).max() <= 1e-12
    assert np.abs(w.variance(ddof=0) - ref.var(0)).max() <= 1e-12
    assert np.abs(w.variance(ddof=1) - ref.var(0, ddof=1)).max() <= 1e-12
    assert w.count == 8


def test_welford_survives_large_offset_where_naive_sums_fail():
    from experiments.e7_streaming_moments import Welford
    rng = np.random.default_rng(1)
    stack = (1e7 + rng.normal(0, 1e-2, size=(32, 4, 4))).astype('float64')
    w = Welford((4, 4))
    for run in stack:
        w.update(run)
    naive = (stack ** 2).mean(0) - stack.mean(0) ** 2
    two_pass = stack.var(0)
    assert np.abs(w.variance(0) - two_pass).max() < 1e-9
    assert np.abs(naive - two_pass).max() > 1e-6   # guards against a vacuous test


def test_welford_rejects_wrong_shape_and_underdetermined_variance():
    from experiments.e7_streaming_moments import Welford
    w = Welford((2, 2))
    with pytest.raises(ValueError, match='shape'):
        w.update(np.zeros((3, 3)))
    w.update(np.ones((2, 2)))
    with pytest.raises(ValueError, match='ddof'):
        w.variance(ddof=1)
    assert w.variance(ddof=0).max() == 0


def test_welford_state_does_not_grow_with_run_count():
    from experiments.e7_streaming_moments import Welford
    w = Welford((3, 3))
    for i in range(50):
        w.update(np.full((3, 3), float(i)))
    assert w.mean.shape == (3, 3) and w.m2.shape == (3, 3)
    assert w.mean.dtype == np.float64 and w.m2.dtype == np.float64


def test_e7_measurement_reports_flat_streaming_ram_and_growing_control():
    from experiments.e7_streaming_moments import measure
    cfg = {'run_counts': [8, 32, 96], 'shape': [4, 96, 96], 'ddof': 0,
           'max_abs_error': 1e-6, 'ram_growth_tolerance': 1.5}
    m = measure(cfg, seed=3)
    assert [r['runs'] for r in m['per_run_count']] == [8, 32, 96]
    assert all(r['max_abs_error_mean'] <= 1e-6 and r['max_abs_error_var'] <= 1e-6 for r in m['per_run_count'])
    streaming = [r['streaming_peak_bytes'] for r in m['per_run_count']]
    control = [r['stacked_peak_bytes'] for r in m['per_run_count']]
    assert streaming[-1] <= streaming[0] * 1.5
    assert control[-1] > control[0] * 4      # a stacked reference must visibly grow
    assert m['ram_flat'] is True and m['error_ok'] is True
