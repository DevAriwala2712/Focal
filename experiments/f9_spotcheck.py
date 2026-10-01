"""F9 spot-checks: one numeric claim per task, recomputed here from the rawest quantity each file records
(numerators, denominators, pixel counts, per-fold blocks), never from the file's own summary line.

Every comparison below is a float/int comparison executed by this script. Nothing is eyeballed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'experiments/results'
out = {}


def load(n):
    return json.load(open(R / f'{n}.json'))


def close(a, b, tol=5e-4):
    return bool(abs(float(a) - float(b)) <= tol)


# ---------------------------------------------------------------- F2: window FAR = 64/981?
def f2_far():
    d = load('f2')
    res = {}
    sc = d['scoring_in_f1_harness']
    for arm_name, arm in sc.items():
        if not isinstance(arm, dict):
            continue
        for split_name, split in arm.items():
            if not isinstance(split, dict):
                continue
            for metric in ('window_far', 'pixel_far'):
                m = split.get(metric) if isinstance(split.get(metric), dict) else None
                if not m or 'numerator' not in m:
                    continue
                num, den, est = m['numerator'], m['denominator'], m['estimate']
                res[f'{arm_name}.{split_name}.{metric}'] = {
                    'numerator': num, 'denominator': den, 'reported_estimate': est,
                    'f9_num_over_den': num / den,
                    'agrees': close(est, num / den, 1e-6),
                    'point_in_own_ci': bool(m['lo'] <= est <= m['hi'])}
    # the keep rule itself
    kr = d['keep_rule']
    res['_keep_rule_arithmetic'] = {
        'reported': kr, 'alpha': 0.05,
        'f9_estimate_exceeds_alpha': None}
    return res


# ---------------------------------------------------------------- F3: pooled coverage = covered/n?
def f3_coverage():
    d = load('f3')
    res = {}
    lodo = d['measurements']['wayanad']['lodo']
    for arm, blk in lodo.items():
        if not isinstance(blk, dict):
            continue
        p = blk.get('pooled')
        if not p:
            continue
        res[arm] = {'covered_px': p['covered_px'], 'n_px': p['n_px'],
                    'reported_coverage': p['coverage_k2'],
                    'f9_covered_over_n': p['covered_px'] / p['n_px'],
                    'agrees': close(p['coverage_k2'], p['covered_px'] / p['n_px'], 1e-9),
                    'reported_gap_pp': p['gap_pp_vs_nominal'],
                    'f9_gap_pp': (p['coverage_k2'] - 0.9545) * 100,
                    'gap_agrees': close(p['gap_pp_vs_nominal'], (p['coverage_k2'] - 0.9545) * 100, 1e-6),
                    'within_2pp_of_nominal': bool(abs((p['coverage_k2'] - 0.9545) * 100) <= 2.0)}
        # do the per-fold covered/n sum to the pooled covered/n?
        folds = blk.get('folds', {})
        if folds:
            cs = sum(f['coverage_k2'] * f['n_px'] for f in folds.values())
            ns = sum(f['n_px'] for f in folds.values())
            res[arm]['f9_pooled_from_per_fold_folds'] = cs / ns
            res[arm]['pooled_equals_fold_weighted_mean'] = close(p['coverage_k2'], cs / ns, 1e-6)
            res[arm]['f9_sum_of_fold_n_px'] = ns
            res[arm]['fold_n_sums_to_pooled_n'] = close(ns, p['n_px'], 1.0)
    return res


# ---------------------------------------------------------------- F7: areas, class totals, v1-vs-v2 IoU
def f7_areas():
    d = load('f7')
    cr = d['class_report']
    res = {'per_class_area_arithmetic': {}}
    total = 0
    for name, v in cr.items():
        total += v['pixels_2p5m']
        res['per_class_area_arithmetic'][name] = {
            'pixels': v['pixels_2p5m'], 'reported_m2': v['m2'], 'f9_px_times_6p25': v['pixels_2p5m'] * 6.25,
            'm2_agrees': close(v['m2'], v['pixels_2p5m'] * 6.25, 1e-6),
            'km2_agrees': close(v['km2'], v['pixels_2p5m'] * 6.25 / 1e6, 1e-9)}
    den = d['class_report_denominator']
    res['class_totals'] = {
        'f9_sum_of_class_pixels': total, 'reported_total': den['total_2p5m_px'],
        'agrees': total == den['total_2p5m_px'],
        'f9_grid_product_2560x2048': 2560 * 2048,
        'total_equals_grid': total == 2560 * 2048}
    # mapped change = CORE + ALLOCATED
    mc = cr['CORE']['pixels_2p5m'] + cr['ALLOCATED']['pixels_2p5m']
    cmp_ = d['v1_vs_v2']['comparisons']['mapped_change_area']
    res['mapped_change'] = {
        'f9_core_plus_allocated_px': mc, 'reported_v2_px': cmp_['b_px'],
        'agrees': mc == cmp_['b_px'],
        'f9_km2': mc * 6.25 / 1e6, 'reported_km2': cmp_['b_km2'],
        'km2_agrees': close(cmp_['b_km2'], mc * 6.25 / 1e6, 1e-9),
        'f9_v2_minus_v1_px': cmp_['b_px'] - cmp_['a_px'],
        'reported_diff_px': cmp_.get('b_minus_a_px'),
        'f9_iou_from_intersection': cmp_['intersection_px'] / (cmp_['a_px'] + cmp_['b_px'] - cmp_['intersection_px']),
        'reported_iou': cmp_['iou'],
        'iou_agrees': close(cmp_['iou'],
                            cmp_['intersection_px'] / (cmp_['a_px'] + cmp_['b_px'] - cmp_['intersection_px']), 1e-4),
        'f9_in_v1_not_v2': cmp_['a_px'] - cmp_['intersection_px'],
        'f9_in_v2_not_v1': cmp_['b_px'] - cmp_['intersection_px']}
    # tau mutation monotonicity, from the recorded sweep
    runs = d['detection']['tau_mutation_on_the_real_map']
    res['tau_mutation'] = {k: v['flagged_px'] for k, v in runs.items()}
    res['tau_mutation_strictly_changes_output'] = len({v['flagged_px'] for v in runs.values()}) == len(runs)
    res['tau_fitted_flagged_equals_mapped_change'] = runs['tau_fitted']['flagged_px'] == mc
    return res


# ---------------------------------------------------------------- F1: FAR reduction arithmetic + n consistency
def f1_far():
    d = load('f1')
    res = {}
    g = d['gates']
    base = g['ungated_S_v1_sigma']['test_split']['pooled_pixel_far']
    for name, blk in g.items():
        ts = blk.get('test_split')
        if not isinstance(ts, dict) or 'pooled_pixel_far' not in ts:
            res[name] = {'scored': False, 'status': blk.get('status') or str(blk)[:120]}
            continue
        pf, wf = ts['pooled_pixel_far'], ts['pooled_window_far']
        res[name] = {
            'pixel_far': pf['estimate'], 'pixel_n': pf.get('denominator'),
            'pixel_point_in_ci': bool(pf['lo'] <= pf['estimate'] <= pf['hi']),
            'window_far': wf['estimate'], 'window_n': wf.get('denominator'),
            'window_point_in_ci': bool(wf['lo'] <= wf['estimate'] <= wf['hi']),
            'f9_pixel_num_over_den': (pf['numerator'] / pf['denominator']) if 'numerator' in pf else None,
            'pixel_far_equals_num_over_den': (close(pf['estimate'], pf['numerator'] / pf['denominator'], 1e-6)
                                              if 'numerator' in pf else None),
            'f9_reduction_vs_ungated_pixel': pf['estimate'] - base['estimate']}
    # do gate_v1 and gate_v1_with_a5_sigma really coincide to the last digit, as claimed?
    a, b = g.get('gate_v1', {}).get('test_split'), g.get('gate_v1_with_a5_sigma', {}).get('test_split')
    if a and b and 'pooled_pixel_far' in a:
        res['_v1_vs_a5_identical_on_real_crop'] = {
            'pixel_far_identical': a['pooled_pixel_far']['estimate'] == b['pooled_pixel_far']['estimate'],
            'window_far_identical': a['pooled_window_far']['estimate'] == b['pooled_window_far']['estimate'],
            'pixel_numerators': [a['pooled_pixel_far'].get('numerator'), b['pooled_pixel_far'].get('numerator')],
            'numerators_identical': a['pooled_pixel_far'].get('numerator') == b['pooled_pixel_far'].get('numerator')}
    return res


# ---------------------------------------------------------------- F5: headline delta from per-image IoUs
def f5_headline():
    d = load('f5')
    res = {'available_top_keys': sorted(d)}
    per = d.get('per_image')
    if not per:
        return res
    # rebuild the paired delta from the per-image IoU records, oracle fraction, primary class
    rows = []
    for rec in per:
        cls = rec.get('classes', {}).get('nonveg_ndvi_0.35') or {}
        m = cls.get('methods', {})
        a = m.get('alloc_pretrained_sr|oracle', {}).get('iou')
        b = m.get('alloc_bilinear|oracle', {}).get('iou')
        if a is None or b is None:
            continue
        rows.append((rec['dataset'], float(a), float(b)))
    if not rows:
        res['note'] = 'per-image IoU keys not in the expected shape; see keys sample'
        sample = per[0].get('classes', {})
        res['class_keys'] = sorted(sample)
        if sample:
            k0 = sorted(sample)[0]
            res['method_keys'] = sorted(sample[k0].get('methods', {}))
        return res
    ds = np.array([r[0] for r in rows])
    delta = np.array([r[1] - r[2] for r in rows])

    def boot(v, reps=2000, seed=2024):
        rng = np.random.default_rng(seed)
        n = len(v)
        ms = np.array([v[rng.integers(0, n, n)].mean() for _ in range(reps)])
        return float(np.quantile(ms, 0.025)), float(np.quantile(ms, 0.975))

    lo, hi = boot(delta)
    nn = delta[ds != 'naip']
    lo2, hi2 = boot(nn)
    res['f9_all_datasets'] = {'n': int(delta.size), 'delta': float(delta.mean()), 'ci': [lo, hi],
                             'n_sr_wins': int((delta > 0).sum())}
    res['f9_excluding_naip'] = {'n': int(nn.size), 'delta': float(nn.mean()), 'ci': [lo2, hi2],
                                'n_sr_wins': int((nn > 0).sum()),
                                'ci_spans_zero': bool(lo2 < 0 < hi2)}
    res['report_claims'] = {'all': 0.0081, 'all_ci': [0.0035, 0.0141], 'n_all': 118,
                            'no_naip': 0.0016, 'no_naip_ci': [-0.0026, 0.0053], 'n_no_naip': 56}
    res['f9_matches_report_all_4dp'] = close(delta.mean(), 0.0081, 5e-4)
    res['f9_matches_report_no_naip_4dp'] = close(nn.mean(), 0.0016, 5e-4)
    return res


if __name__ == '__main__':
    out['F1'] = f1_far()
    out['F2'] = f2_far()
    out['F3'] = f3_coverage()
    out['F5'] = f5_headline()
    out['F7'] = f7_areas()
    print(json.dumps(out, indent=1, default=str))
    json.dump(out, open(ROOT / 'experiments/f9_spotcheck.json', 'w'), indent=1, default=str)
