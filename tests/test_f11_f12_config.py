from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / 'configs' / 'f11_f12.yaml'


def test_config_file_exists_and_parses():
    assert CONFIG.is_file()
    cfg = yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    assert cfg['schema_version'] == 1


def test_config_has_required_top_level_keys():
    cfg = yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    for key in ('aoi_center', 'pool_extension', 'fold_construction',
                'endmember_sensitivity_keep_rule', 'tau_recomputation', 'f12_wayanad_v3'):
        assert key in cfg, f'missing top-level key: {key}'


def test_pool_extension_has_minimum_and_target():
    cfg = yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    pe = cfg['pool_extension']
    assert pe['minimum_n'] == 4
    assert pe['target_n'] == 6
    assert 'stop_rule' in pe


def test_fold_construction_n_pre_matches_f12_production():
    cfg = yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    assert cfg['fold_construction']['n_pre'] == 3
    assert cfg['fold_construction']['n_pre'] == len(cfg['f12_wayanad_v3']['dates']['pre'])


def test_endmember_sensitivity_keep_rule_requires_all_three_conditions():
    cfg = yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    conditions = cfg['endmember_sensitivity_keep_rule']['conditions']
    assert set(conditions) == {'point_estimate', 'e_b_plus_1sd', 'e_b_minus_1sd'}
    assert cfg['endmember_sensitivity_keep_rule']['alpha'] == 0.05


def test_tau_recomputation_is_not_verbatim_from_f1_or_f2():
    """The method must explain that tau is RECOMPUTED, not taken verbatim -- explaining the
    rejection necessarily uses the word 'verbatim' in a negated sentence ('never ... verbatim'),
    so this checks for the correct instruction phrase rather than absence of the word itself."""
    cfg = yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    method = cfg['tau_recomputation']['method']
    assert 'never' in method.lower() and 'verbatim' in method.lower()
    assert 'recomputed' in method.lower() or 'own score' in method.lower()


def test_config_hash_is_computable_and_stable():
    h1 = hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    h2 = hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    assert h1 == h2
    assert len(h1) == 64
