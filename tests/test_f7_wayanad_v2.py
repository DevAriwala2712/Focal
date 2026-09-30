"""Tests for F7 (the Wayanad v2 production rerun).

Two things are checked here that nothing else checks:

  1. the real per-block sum test on F7's ACTUAL production class map (not a fixture, not F2's placebo
     folds), plus proof that the test is not tautological -- corrupting one allocated pixel in the output
     must move the deviation by exactly 1;
  2. the semantic guard on the v1-vs-v2 comparison: the code must refuse to put a v1 class next to a v2
     class whose definition differs (configs/fix.yaml f7_wayanad_v2.no_cross_class_comparison and
     f9_checks 'no_row_compares_classes_with_different_definitions').
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments import f7_wayanad_v2 as F7
from trustsr import gate_v2 as G

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / 'experiments/results/f7.json'
STATE = ROOT / 'data/experiments-cache/f7_production_state.npz'


def load_state():
    if not STATE.is_file():
        pytest.skip(f'{STATE} absent: run `python -m experiments.f7_wayanad_v2` first')
    with np.load(STATE) as z:
        return z['class_map'], z['f_effective'], z['nodata']


def load_result():
    if not RESULT.is_file():
        pytest.skip(f'{RESULT} absent: run `python -m experiments.f7_wayanad_v2` first')
    return json.loads(RESULT.read_text(encoding='utf-8'))


# ---------------------------------------------------------------- block-sum on the real output ----------------------------------------------------------------

def test_block_sum_deviation_is_zero_on_the_real_production_map():
    class_map, f_eff, nodata = load_state()
    dev = G.block_sum_deviation(class_map, f_eff, nodata, 4)
    assert dev['n_blocks'] == f_eff.size
    assert dev['max_abs_deviation_all_blocks'] == 0, dev
    assert dev['n_blocks_with_nonzero_deviation'] == 0, dev
    assert dev['pass'] is True


def test_block_sum_on_the_real_map_is_not_tautological():
    """Corrupt exactly one allocated pixel of the OUTPUT map; the deviation must become exactly 1.

    A self-comparison (x4's `h*w*100` vs `h*w*100`, B7) cannot react to this at all.
    """
    class_map, f_eff, nodata = load_state()
    flagged = G.flagged_v2(class_map)
    rr, cc = np.nonzero(flagged)
    assert rr.size, 'the production map flags nothing; there is nothing to corrupt'
    corrupted = class_map.copy()
    corrupted[rr[0], cc[0]] = G.NO_CHANGE
    dev = G.block_sum_deviation(corrupted, f_eff, nodata, 4)
    assert dev['max_abs_deviation_all_blocks'] == 1, dev
    assert dev['n_blocks_with_nonzero_deviation'] == 1, dev


def test_reported_block_sum_matches_a_fresh_recomputation():
    r = load_result()
    class_map, f_eff, nodata = load_state()
    dev = G.block_sum_deviation(class_map, f_eff, nodata, 4)
    assert r['block_sum_test']['max_abs_deviation_all_blocks'] == dev['max_abs_deviation_all_blocks']
    assert r['block_sum_test']['n_blocks'] == dev['n_blocks']
    assert r['block_sum_test']['n_detected_blocks'] == dev['n_detected_blocks']


# ---------------------------------------------------------------- the v1-vs-v2 semantic guard ----------------------------------------------------------------

def test_the_guard_accepts_only_registered_identical_definitions():
    assert F7.assert_well_defined_comparison('v1:OBSERVED|INFERRED', 'v2:CORE|ALLOCATED')
    assert F7.assert_well_defined_comparison('v1:NO_DATA', 'v2:NO_DATA')


@pytest.mark.parametrize('a,b', [
    ('v1:UNSUPPORTED', 'v2:UNSUPPORTED'),      # different gating predicate entirely
    ('v1:OBSERVED', 'v2:CORE'),                # not a refinement of one another
    ('v1:INFERRED', 'v2:ALLOCATED'),
])
def test_the_guard_refuses_classes_with_different_definitions(a, b):
    with pytest.raises(ValueError, match='refusing to compare'):
        F7.assert_well_defined_comparison(a, b)
    with pytest.raises(ValueError, match='refusing to compare'):
        F7.compare_sets(a, np.zeros((4, 4), bool), b, np.zeros((4, 4), bool))


def test_the_guard_refuses_unregistered_pairs_by_default():
    with pytest.raises(ValueError, match='not registered in COMPARABLE_PAIRS'):
        F7.assert_well_defined_comparison('v1:NO_CHANGE', 'v2:NO_CHANGE')


def test_no_registered_comparison_compares_a_quantity_with_itself():
    """f9_checks: 'no_test_compares_quantity_with_itself'. Every accepted pair must be v1 vs v2."""
    for a, b in F7.COMPARABLE_PAIRS:
        assert a != b
        assert a.startswith('v1:') and b.startswith('v2:'), (a, b)


def test_compare_sets_is_the_only_path_to_a_v1_v2_number_and_it_calls_the_guard(monkeypatch):
    called = []
    monkeypatch.setattr(F7, 'assert_well_defined_comparison',
                        lambda a, b: called.append((a, b)) or 'ok')
    a = np.zeros((4, 4), bool); a[0, 0] = True
    b = np.zeros((4, 4), bool); b[0, 0] = b[1, 1] = True
    out = F7.compare_sets('v1:OBSERVED|INFERRED', a, 'v2:CORE|ALLOCATED', b)
    assert called == [('v1:OBSERVED|INFERRED', 'v2:CORE|ALLOCATED')]
    assert out['a_px'] == 1 and out['b_px'] == 2 and out['intersection_px'] == 1
    assert out['iou'] == pytest.approx(0.5)
    assert out['a_m2'] == 6.25 and out['b_km2'] == pytest.approx(2 * 6.25 / 1e6)


def test_compare_sets_refuses_grids_that_do_not_match():
    with pytest.raises(ValueError, match='not on the same grid'):
        F7.compare_sets('v1:NO_DATA', np.zeros((4, 4), bool), 'v2:NO_DATA', np.zeros((8, 8), bool))


# ---------------------------------------------------------------- what the result file must say ----------------------------------------------------------------

def test_the_result_never_compares_unsupported_across_versions():
    r = load_result()
    u = r['v1_vs_v2']['unsupported_not_compared']
    assert u['compared'] is False
    assert 'DIFFERENT DEFINITIONS' in u['reason']
    # no difference / ratio / IoU key may exist on that block
    assert not any(k in u for k in ('difference', 'ratio', 'iou', 'b_minus_a_px', 'b_over_a'))
    for key, comp in r['v1_vs_v2']['comparisons'].items():
        if 'a_name' in comp:
            assert (comp['a_name'], comp['b_name']) in F7.COMPARABLE_PAIRS


def test_status_and_caveat_carry_gate_v2s_failed_far_keep_rule():
    r = load_result()
    assert r['status'] == 'FAIL', 'gate v2 failed its FAR keep rule; F7 may not report PASS'
    assert r['evidence'] == 'real'
    assert 'caveat' in r and '0.0652' in r['caveat']
    assert r['gate_v2_keep_rule_failure']['verdict'] == 'NOT MET'
    assert r['gate_v2_keep_rule_failure']['estimate'] > r['gate_v2_keep_rule_failure']['alpha']
    # the failure must also be on the figure caption, not only in the json
    assert 'FAILED' in r['wow_figure']['caption']
    assert 'not fine-tuned' in r['wow_figure']['caption'].lower()
    assert 'model reconstruction' in r['wow_figure']['caption'].lower()
    # and near the top of the human report
    report = (ROOT / 'experiments/results/f7_REPORT.md').read_text(encoding='utf-8')
    assert 'FAILED ITS OWN KEEP RULE' in report[:1500]
    assert '0.0652' in report[:2000]


def test_config_hash_is_the_real_hash_of_fix_yaml():
    from risk.common import digest
    r = load_result()
    assert r['config_sha256'] == digest(ROOT / 'configs/fix.yaml')
    assert r['config_sha256'] != 'pending-a3-output'


def test_dates_are_the_preregistered_ones_and_the_absent_fourth_is_not_fabricated():
    r = load_result()
    assert r['dates']['pre'] == ['2024-01-16', '2024-01-21', '2024-01-26']
    assert r['dates']['post'] == '2024-12-06'
    fourth = r['dates']['fourth_pre_date']
    assert fourth['candidate'] == '2023-12-27'
    if not fourth['available']:
        assert fourth['candidate'] not in r['dates']['pre']
        assert len(r['dates']['pre']) == 3


def test_the_sr_run_was_real_inference_and_the_pass_count_is_consistent():
    r = load_result()
    sr = r['sr_run']
    assert sr['is_real_inference'] is True
    assert sr['forward_passes_total'] == sr['tiles_per_date'] * 8 * len(sr['dates_run'])
    assert sr['sr_score_blocks_with_within_block_variation_fraction'] > 0, \
        'a blocky np.repeat field would be 0 and must raise (B9)'


def test_tau_is_used_not_merely_loaded():
    r = load_result()
    runs = r['detection']['tau_mutation_on_the_real_map']
    counts = {v['flagged_px'] for v in runs.values()}
    assert len(counts) > 1, 'changing tau must change the production map (B10)'
    assert runs['tau_times_0p5']['flagged_px'] >= runs['tau_fitted']['flagged_px'] \
        >= runs['tau_times_2']['flagged_px'], 'flagged count must be monotone decreasing in tau'


def test_areas_use_the_preregistered_pixel_area():
    r = load_result()
    for name, v in r['class_report'].items():
        assert v['m2'] == pytest.approx(v['pixels_2p5m'] * 6.25)
        assert v['km2'] == pytest.approx(v['m2'] / 1e6)
    total = sum(v['pixels_2p5m'] for v in r['class_report'].values())
    assert total == r['class_report_denominator']['total_2p5m_px']


def test_exclusion_geometry_is_not_baked_into_production_no_data():
    """v1 and v2 NO_DATA must agree exactly; if the 1500 m disks had leaked into v2's NO_DATA they
    could not, and the scar itself would have been erased."""
    r = load_result()
    nd = r['v1_vs_v2']['comparisons']['no_data']
    assert nd['a_px'] == nd['b_px'] and nd['iou'] == 1.0, nd


def test_e5_cascade_forward_passes_are_counted_not_estimated():
    r = load_result()
    c = r['e5_cascade']
    t, p = c['tiles'], c['forward_passes']
    n_dates = c['parameters']['dates']
    assert p['full_rerun_baseline'] == t['total'] * 8 * n_dates
    assert p['cascade'] == t['processed'] * 8 * n_dates
    assert p['saved'] == p['full_rerun_baseline'] - p['cascade']
    assert t['processed'] + t['skipped'] == t['total']
    assert p['full_rerun_baseline'] == r['sr_run']['forward_passes_total'], \
        'the baseline must be exactly the run this script performed'
    full = c['full_aoi_selection_for_context']
    assert full['sr_was_not_run_at_this_scale_in_f7'] is True
    assert full['forward_passes']['saved'] == (full['tiles']['total'] - full['tiles']['processed']) * 8 * n_dates
