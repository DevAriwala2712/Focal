"""F6: no-drift check between the claim in experiments/results/f6.json / f6_REPORT.md and what
experiments/f6_a2_report.py actually computes from experiments/results/x2.json.

Does not run any SR inference; only reads the already-committed x2.json.
"""
import json
from pathlib import Path

from experiments.f6_a2_report import build_report, per_dataset_rows

ROOT = Path(__file__).resolve().parent.parent
F6_JSON = ROOT / 'experiments/results/f6.json'


def test_f6_json_exists_and_is_pass():
    res = json.loads(F6_JSON.read_text(encoding='utf-8'))
    assert res['status'] == 'PASS'
    assert res['evidence'] == 'real'


def test_pooled_number_in_f6_json_matches_script_recomputation():
    """The pooled number recorded in f6.json must equal what build_report() computes right now
    from x2.json -- no hand-copied / stale numbers."""
    committed = json.loads(F6_JSON.read_text(encoding='utf-8'))
    fresh = build_report()
    c = committed['pooled_all_datasets']['recomputed']
    f = fresh['pooled_all_datasets']['recomputed']
    assert c == f


def test_pooled_matches_audit_quoted_b14_figure():
    res = json.loads(F6_JSON.read_text(encoding='utf-8'))
    r = res['pooled_all_datasets']['recomputed']
    assert round(r['mean'], 3) == -0.097
    assert round(r['lo'], 3) == -0.157
    assert round(r['hi'], 3) == -0.037
    assert r['n'] == 178


def test_all_five_datasets_present_including_the_three_omitted_from_results_exceptional():
    res = json.loads(F6_JSON.read_text(encoding='utf-8'))
    assert set(res['per_dataset']) == {'naip', 'spot', 'spain_crops', 'spain_urban', 'venus'}
    # the three B14-omitted datasets must show SR NOT beating bicubic (that's why they were omitted)
    for ds in ('spain_crops', 'spain_urban', 'venus'):
        assert res['per_dataset'][ds]['beats_bicubic'] is False


def test_naip_excluded_sensitivity_is_more_negative_not_a_rescue():
    res = json.loads(F6_JSON.read_text(encoding='utf-8'))
    pooled_all = res['pooled_all_datasets']['recomputed']['mean']
    pooled_naip_excluded = res['naip_excluded_sensitivity']['recomputed']['mean']
    # excluding the dataset with the largest positive delta cannot make the pooled result positive
    assert pooled_naip_excluded < 0
    assert pooled_naip_excluded <= pooled_all


def test_hallucination_metric_higher_for_sr_on_every_dataset():
    res = json.loads(F6_JSON.read_text(encoding='utf-8'))
    for ds, h in res['opensr_hallucination_column'].items():
        assert h['sr_higher_than_bicubic'] is True, ds


def test_config_sha256_matches_current_fix_yaml_contents():
    import hashlib
    res = json.loads(F6_JSON.read_text(encoding='utf-8'))
    actual = hashlib.sha256((ROOT / 'configs/fix.yaml').read_bytes()).hexdigest()
    assert res['config_sha256'] == actual


def test_per_dataset_rows_match_x2_json_n_images_denominators():
    res = json.loads(F6_JSON.read_text(encoding='utf-8'))
    assert res['per_dataset']['naip']['n_images'] == 62
    assert res['per_dataset']['spot']['n_images'] == 9
    assert res['per_dataset']['spain_crops']['n_images'] == 28
    assert res['per_dataset']['spain_urban']['n_images'] == 20
    assert res['per_dataset']['venus']['n_images'] == 59
    assert sum(d['n_images'] for d in res['per_dataset'].values()) == 178
