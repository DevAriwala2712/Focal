# tests/test_f11_gate_v2_recalibration.py
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
F11_JSON = ROOT / 'experiments' / 'results' / 'f11.json'


def test_build_matched_folds_produces_n_pre_3_folds():
    from experiments.f11_gate_v2_recalibration import build_matched_folds
    pool = ['2024-01-16', '2024-01-21', '2024-01-26', '2024-02-05']
    specs = build_matched_folds(pool, n_pre=3)
    assert len(specs) == 4   # C(4,3)=4 reference sets x 1 remaining each
    for ref, held in specs:
        assert len(ref) == 3
        assert held not in ref


def test_perturb_e_b_reuses_existing_robust_sd_not_a_new_estimate():
    from experiments.f11_gate_v2_recalibration import perturb_e_b

    em = {'e_v': np.array([0.08, 0.06, 0.04, 0.40]), 'e_b': np.array([0.15, 0.12, 0.10, 0.08]),
          'e_b_robust_sd': np.array([0.01, 0.01, 0.01, 0.01])}
    plus = perturb_e_b(em, sign=+1)
    minus = perturb_e_b(em, sign=-1)
    assert np.allclose(plus, em['e_b'] + em['e_b_robust_sd'])
    assert np.allclose(minus, em['e_b'] - em['e_b_robust_sd'])


def test_sensitivity_robust_keep_rule_requires_all_three_conditions():
    from experiments.f11_gate_v2_recalibration import evaluate_sensitivity_robust_keep_rule

    alpha = 0.05
    all_pass = {'point_estimate': {'far_point': 0.03}, 'e_b_plus_1sd': {'far_point': 0.04},
               'e_b_minus_1sd': {'far_point': 0.02}}
    one_fail = {'point_estimate': {'far_point': 0.03}, 'e_b_plus_1sd': {'far_point': 0.09},
               'e_b_minus_1sd': {'far_point': 0.02}}

    assert evaluate_sensitivity_robust_keep_rule(all_pass, alpha) is True
    assert evaluate_sensitivity_robust_keep_rule(one_fail, alpha) is False


@pytest.mark.skipif(not F11_JSON.is_file(), reason='f11.json not yet produced by a real run')
def test_f11_json_status_and_hash_are_honest():
    r = json.loads(F11_JSON.read_text())
    assert r['evidence'] == 'real'
    assert r['status'] in ('PASS', 'FAIL', 'BLOCKED')
    import hashlib
    cfg_path = ROOT / 'configs' / 'f11_f12.yaml'
    assert r['config_sha256'] == hashlib.sha256(cfg_path.read_bytes()).hexdigest()


@pytest.mark.skipif(not F11_JSON.is_file(), reason='f11.json not yet produced by a real run')
def test_f11_reports_all_three_sensitivity_conditions():
    r = json.loads(F11_JSON.read_text())
    conditions = r['sensitivity_robust_keep_rule']['conditions']
    assert set(conditions.keys()) == {'point_estimate', 'e_b_plus_1sd', 'e_b_minus_1sd'}
    for cond in conditions.values():
        assert 'far_window' in cond
        assert 'ci' in cond['far_window']
        assert 'denominator' in cond['far_window']


@pytest.mark.skipif(not F11_JSON.is_file(), reason='f11.json not yet produced by a real run')
def test_f11_tau_differs_from_f2_tau():
    """Proves this is a real recalibration, not an accidental reuse of F2's old tau."""
    r = json.loads(F11_JSON.read_text())
    f2 = json.loads((ROOT / 'experiments' / 'results' / 'f2.json').read_text())
    assert r['tau'] != f2['tau']
