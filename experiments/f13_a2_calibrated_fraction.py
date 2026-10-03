"""F13 A2: does a calibrated fraction make sub-pixel allocation beat blocky? (isotonic, leave-one-dataset-out; O2)

Pre-registered in configs/f13_discovery.yaml `experiments.A2` (sha256 84beef1c...). The calibration is a decreasing
isotonic regression of the oracle block fraction on the block's 10 m NDVI, fitted on three datasets and applied to the
fourth. scikit-learn is not installed in the project environment, so pool-adjacent-violators is implemented here and
tested against a brute-force min-max reference (tests/test_f13.py::TestPAV).
"""
from __future__ import annotations

import numpy as np

from experiments import f13_common as C

BLOCK = 4


# ---------------------------------------------------------------- isotonic regression ----------------------------------------------------------------

def pav_increasing(y, w) -> np.ndarray:
    """Weighted L2 isotonic (non-decreasing) fit of y in the given order, by pool-adjacent-violators."""
    y = np.asarray(y, np.float64)
    w = np.asarray(w, np.float64)
    means, weights, counts = [], [], []
    for yi, wi in zip(y, w):
        means.append(yi); weights.append(wi); counts.append(1)
        while len(means) > 1 and means[-2] > means[-1]:
            m2, w2, c2 = means.pop(), weights.pop(), counts.pop()
            m1, w1, c1 = means.pop(), weights.pop(), counts.pop()
            means.append((m1 * w1 + m2 * w2) / (w1 + w2)); weights.append(w1 + w2); counts.append(c1 + c2)
    return np.repeat(means, counts)


def fit_isotonic_decreasing(x, y, w=None) -> dict:
    """Decreasing isotonic fit of y on x. Tied x values are pooled (weighted mean) first, as scikit-learn does.

    Returns {'x': knots ascending, 'y': fitted values (non-increasing)}."""
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    w = np.ones_like(x) if w is None else np.asarray(w, np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, w = x[ok], y[ok], w[ok]
    ux, inv = np.unique(x, return_inverse=True)
    sw = np.bincount(inv, weights=w)
    sy = np.bincount(inv, weights=w * y) / sw
    fitted = -pav_increasing(-sy, sw)
    return {'x': ux, 'y': fitted, 'n_points': int(x.size), 'n_knots': int(ux.size)}


def predict_isotonic(model: dict, x) -> np.ndarray:
    """Linear interpolation between knots, clipped to the end values outside them (scikit-learn's
    out_of_bounds='clip' behaviour); non-finite x -> 0 (no allocation)."""
    x = np.asarray(x, np.float64)
    out = np.interp(np.nan_to_num(x, nan=0.0), model['x'], model['y'])
    return np.where(np.isfinite(x), out, 0.0)


# ---------------------------------------------------------------- leave-one-dataset-out ----------------------------------------------------------------

def lodo_folds(datasets) -> list:
    datasets = list(datasets)
    return [{'held_out': d, 'train': [t for t in datasets if t != d]} for d in datasets]


def training_arrays(rows, train):
    """Concatenate (x, y) of the rows whose dataset is in `train`. The held-out dataset never enters."""
    sel = [r for r in rows if r['dataset'] in set(train)]
    return np.concatenate([r['x'] for r in sel]), np.concatenate([r['y'] for r in sel])


def valid_blocks(valid_hr, ndvi10) -> np.ndarray:
    """A 10 m block counts for fitting and MAE only if all 16 of its 2.5 m px are valid and its 10 m NDVI is finite."""
    from trustsr import alloc
    return alloc._blocks(np.asarray(valid_hr, bool), BLOCK).all(axis=-1) & np.isfinite(ndvi10)


def a2_verdict(mae: float, delta: dict) -> str:
    """yaml experiments.A2.keep_rule, verbatim, on point estimates."""
    excl = delta['lo'] > 0 or delta['hi'] < 0
    if mae <= 0.15 and delta['estimate'] > 0 and excl:
        return 'TRUE'
    if mae > 0.20 or delta['estimate'] <= 0:
        return 'FALSE'
    return 'INCONCLUSIVE'


# ================================================================ the run ================================================================

ARMS = ('blocky_10m', 'bilinear_ranked_calibrated', 'sr_ranked_calibrated', 'hr_ranked_realistic', 'hr_ranked_calibrated')
CONTEXT_ARMS = ('bilinear_ranked_realistic',)          # = F5's alloc_bilinear|realistic, for the calibration effect
NDVI_GRID = (-0.2, 0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def image_inputs(im) -> dict:
    from experiments.f13_a1_sr_vs_interpolation import THRESHOLD, interp_index
    from trustsr import alloc
    lr, hr, sr, valid = im['lr'], im['hr'], im['sr'], im['valid']
    s_lr, s_hr, s_sr = alloc.ndvi_rgbn(lr), alloc.ndvi_rgbn(hr), alloc.ndvi_rgbn(sr)
    truth = np.isfinite(s_hr) & (s_hr < THRESHOLD)
    lr_valid = (lr > 0).all(axis=0) & np.isfinite(lr).all(axis=0)
    frac_real, _diag = alloc.block_fraction_realistic(lr, s_lr, lr_valid, (0, 3))
    return {'dataset': im['dataset'], 'name': im['name'], 's_lr': s_lr, 's_hr': s_hr, 's_sr': s_sr,
            's_bil': interp_index(s_lr, 'bilinear'), 'truth': truth, 'valid': valid,
            'frac_oracle': alloc.block_fraction_oracle(truth & valid, BLOCK), 'frac_real': frac_real,
            'bv': valid_blocks(valid, s_lr), 'f5_iou': im['f5_iou']}


def score(x, f_cal) -> dict:
    from experiments.f13_a1_sr_vs_interpolation import THRESHOLD
    from trustsr import alloc
    masks = {'blocky_10m': alloc.blocky_mask(x['s_lr'], THRESHOLD, BLOCK),
             'bilinear_ranked_calibrated': alloc.allocate_by_rank(x['s_bil'], f_cal, BLOCK),
             'sr_ranked_calibrated': alloc.allocate_by_rank(x['s_sr'], f_cal, BLOCK),
             'hr_ranked_realistic': alloc.allocate_by_rank(x['s_hr'], x['frac_real'], BLOCK),
             'hr_ranked_calibrated': alloc.allocate_by_rank(x['s_hr'], f_cal, BLOCK),
             'bilinear_ranked_realistic': alloc.allocate_by_rank(x['s_bil'], x['frac_real'], BLOCK)}
    out = {}
    for arm, m in masks.items():
        r = alloc.iou(m, x['truth'], x['valid'])
        out[arm] = {'i': r['intersection'], 'u': r['union'], 'iou': r['iou'],
                    'bf1': alloc.boundary_f1(m, x['truth'], x['valid'], 1)['f1']}
    return out


def main():
    import json
    import time
    from experiments import f13_a1_sr_vs_interpolation as A1
    from trustsr.bootstrap import paired_bootstrap_ci, ratio_bootstrap_ci
    started = C.utc_now()
    t0 = time.time()
    result = {'evidence': 'real', 'experiment_id': 'A2'}
    try:
        A1.block_network()
        pre = C.require_committed_prereg()
        cfg = C.load_prereg()
        a2 = cfg['experiments']['A2']
        boot = cfg['statistics']['bootstrap']
        P = A1.prepare(cfg)
        result['reproduction_check'] = P['gate']
        result['data'], result['sr_cache'] = P['data'], P['sr_cache']
        if not P['gate']['pass']:
            result.update(status='BLOCKED', verdict='NOT_RUN',
                          reason='F5 reproduction gate failed; no A2 arm was computed (see reproduction_check)')
            return _finish(result, started)
        xs = [image_inputs(im) for im in P['images']]
        del P['images']
        datasets = list(cfg['experiments']['A1']['data']['datasets'])
        rows = [{'dataset': x['dataset'], 'x': x['s_lr'][x['bv']], 'y': x['frac_oracle'][x['bv']]} for x in xs]

        # ---- leave-one-dataset-out isotonic calibration ---------------------------------------------------------------
        folds, f_cal = [], [None] * len(xs)
        for fold in lodo_folds(datasets):
            xtr, ytr = training_arrays(rows, fold['train'])
            model = fit_isotonic_decreasing(xtr, ytr)
            held = [k for k, x in enumerate(xs) if x['dataset'] == fold['held_out']]
            for k in held:
                f_cal[k] = predict_isotonic(model, xs[k]['s_lr'])
            folds.append({**fold, 'n_training_blocks': int(xtr.size), 'n_knots': model['n_knots'],
                          'curve_f_at_ndvi': {str(v): float(predict_isotonic(model, np.array([v]))[0]) for v in NDVI_GRID},
                          'held_out_images': len(held)})

        # ---- fraction MAE over valid blocks ---------------------------------------------------------------------------
        abs_cal = np.array([np.abs(f_cal[k] - x['frac_oracle'])[x['bv']].sum() for k, x in enumerate(xs)])
        abs_real = np.array([np.abs(x['frac_real'] - x['frac_oracle'])[x['bv']].sum() for x in xs])
        nblk = np.array([x['bv'].sum() for x in xs], float)
        ds_of = np.array([x['dataset'] for x in xs])

        def mae_block(sel):
            c = ratio_bootstrap_ci(abs_cal[sel], nblk[sel], boot['replicates'], 0.95, boot['seed'])
            r = ratio_bootstrap_ci(abs_real[sel], nblk[sel], boot['replicates'], 0.95, boot['seed'])
            return {'calibrated': c, 'realistic_unmixing_same_blocks': r, 'n_images': int(sel.sum())}
        mae = {'pooled_all_folds': mae_block(np.ones(len(xs), bool)),
               'without_naip': mae_block(ds_of != 'naip'),
               'per_fold_held_out': {d: mae_block(ds_of == d) for d in datasets}}
        f5 = json.loads((C.ROOT / 'experiments/results/f5.json').read_text(encoding='utf-8'))
        f5_style = {'calibrated_per_image_mean_all_blocks': float(np.mean([np.abs(f_cal[k] - x['frac_oracle']).mean() for k, x in enumerate(xs)])),
                    'realistic_per_image_mean_all_blocks': float(np.mean([np.abs(x['frac_real'] - x['frac_oracle']).mean() for x in xs])),
                    'f5_json_realistic_block_fraction_mae_mean': f5['per_class']['nonveg_ndvi_0.35']['fraction_estimate']['ALL']['block_fraction_mae_mean']}
        f5_style['realistic_reproduces_f5'] = abs(f5_style['realistic_per_image_mean_all_blocks'] - f5_style['f5_json_realistic_block_fraction_mae_mean']) <= 1e-9

        # ---- allocation arms vs blocky ----------------------------------------------------------------------------------
        recs = [{'dataset': x['dataset'], 'name': x['name'], 'arms': score(x, f_cal[k])} for k, x in enumerate(xs)]
        eq = {'bilinear_ranked_realistic_equals_f5_alloc_bilinear_realistic':
              all(r['arms']['bilinear_ranked_realistic']['iou'] == x['f5_iou']['alloc_bilinear|realistic'] for r, x in zip(recs, xs)),
              'blocky_equals_f5_blocky': all(r['arms']['blocky_10m']['iou'] == x['f5_iou']['blocky_10m'] for r, x in zip(recs, xs))}
        result['reproduction_check']['arm_equivalence_with_f5_per_image'] = eq
        result['reproduction_check']['realistic_fraction_mae_reproduces_f5'] = f5_style['realistic_reproduces_f5']
        if not all(eq.values()):
            result.update(status='BLOCKED', verdict='NOT_RUN', reason='blocky / realistic arms do not equal F5 image by image')
            return _finish(result, started)

        def arr(rs, arm):
            return (np.array([r['arms'][arm]['i'] for r in rs], float), np.array([r['arms'][arm]['u'] for r in rs], float))

        def subset(which):
            if which == 'with_naip':
                return recs
            if which == 'without_naip':
                return [r for r in recs if r['dataset'] != 'naip']
            return [r for r in recs if r['dataset'] == which]

        table, contrasts = {}, {}
        for which in ('with_naip', 'without_naip') + tuple(datasets):
            rs = subset(which)
            table[which] = {}
            for arm in ARMS + CONTEXT_ARMS:
                i, u = arr(rs, arm)
                ci = ratio_bootstrap_ci(i, u, boot['replicates'], 0.95, boot['seed'])
                per = [r['arms'][arm]['iou'] for r in rs if r['arms'][arm]['iou'] is not None]
                bf = [r['arms'][arm]['bf1'] for r in rs if r['arms'][arm]['bf1'] is not None]
                table[which][arm] = {'iou_sum_ratio': ci['estimate'], 'lo': ci['lo'], 'hi': ci['hi'],
                                     'numerator': ci['numerator'], 'denominator': ci['denominator'], 'n_images': len(rs),
                                     'iou_per_image_mean_f5_style': float(np.mean(per)) if per else None,
                                     'boundary_f1_1px_per_image_mean': float(np.mean(bf)) if bf else None}
            if which in ('with_naip', 'without_naip'):
                contrasts[which] = {}
                ib, ub = arr(rs, 'blocky_10m')
                for arm in ARMS[1:] + CONTEXT_ARMS:
                    ia, ua = arr(rs, arm)
                    c = A1.paired_ratio_diff_ci(ia, ua, ib, ub, boot['replicates'], boot['seed'], 0.95)
                    d = [r['arms'][arm]['iou'] - r['arms']['blocky_10m']['iou'] for r in rs
                         if r['arms'][arm]['iou'] is not None and r['arms']['blocky_10m']['iou'] is not None]
                    pc = paired_bootstrap_ci(np.asarray(d), boot['replicates'], 0.95, boot['seed'])
                    c['f5_style_per_image_mean_iou_diff'] = {'estimate': pc['mean'], 'lo': pc['lo'], 'hi': pc['hi'],
                                                             'n_images': pc['n'], 'n_arm_better': int(sum(v > 0 for v in d))}
                    contrasts[which][f'{arm} - blocky_10m'] = c

        mae_pooled = mae['pooled_all_folds']['calibrated']['estimate']
        delta = contrasts['with_naip']['bilinear_ranked_calibrated - blocky_10m']
        verdict = a2_verdict(mae_pooled, delta)
        pivot = {'trigger': cfg['pivot_rules']['rules']['probabilistic_output']['trigger'],
                 'mae_gt_0p15': bool(mae_pooled > 0.15), 'calibrated_allocation_le_blocky': bool(delta['estimate'] <= 0),
                 'met': bool(mae_pooled > 0.15 or delta['estimate'] <= 0)}
        mc = mae['pooled_all_folds']['calibrated']
        meas = [C.measurement('fraction_MAE_calibrated_pooled_LODO', mc['estimate'], numerator=mc['numerator'],
                              denominator=mc['denominator'], lo=mc['lo'], hi=mc['hi'], replicates=boot['replicates'],
                              seed=boot['seed'], unit='mean |f_hat - f_oracle| over valid 10 m blocks (numerator = sum of abs errors, denominator = blocks)'),
                C.measurement('delta_IoU_bilinear_ranked_calibrated_minus_blocky_all_datasets', delta['estimate'],
                              numerator=None, denominator=None, lo=delta['lo'], hi=delta['hi'], replicates=delta['replicates'],
                              seed=delta['seed'], unit='difference of sum-ratio IoU', iou_a=delta['iou_a'],
                              numerator_a=delta['numerator_a'], denominator_a=delta['denominator_a'], iou_b=delta['iou_b'],
                              numerator_b=delta['numerator_b'], denominator_b=delta['denominator_b'], n_images=delta['n_images'])]
        result.update(
            status='PASS' if verdict == 'TRUE' else 'FAIL', verdict=verdict, rule_applied=a2['keep_rule'],
            preregistration={**pre, 'experiment': 'A2'},
            verdict_basis='pooled LODO fraction MAE (all four datasets) and the all-dataset paired sum-ratio IoU delta of '
                          'bilinear_ranked_calibrated minus blocky_10m with its image-paired bootstrap CI',
            pivot_probabilistic_output=pivot,
            folds=folds, fraction_mae=mae, fraction_mae_f5_style=f5_style,
            arms=table, contrasts_vs_blocky=contrasts,
            per_image=[{'dataset': r['dataset'], 'name': r['name'], 'iou': {a: v['iou'] for a, v in r['arms'].items()}} for r in recs],
            measurements=meas,
            interpretations=[
                'Isotonic regression is pool-adjacent-violators implemented here (scikit-learn is not installed), decreasing in '
                '10 m NDVI, equal weight per block, tied NDVI values pooled first; prediction interpolates linearly between '
                'knots and clips to the end values outside them (scikit-learn out_of_bounds="clip" behaviour).',
                'A 10 m block enters the fit and the MAE only if all 16 HR px are valid (F5 mask incl. the 16 px border) and its '
                '10 m NDVI is finite; the calibrated fraction is applied to every block, as F5 applies the realistic one.',
                'Fraction MAE pooled = sum of |f_hat - f_oracle| over valid blocks of all held-out sets / number of those blocks; '
                'F5 reported a per-image mean over all blocks, given beside it (fraction_mae_f5_style).',
                'IoU is the sum-ratio over images, paired image bootstrap; blocky_10m is computed the same way in this run, not '
                'taken from the yaml\'s per-image-mean 0.7596.',
                'HR ranking = HR NDVI at its native 2.5 m: HRharm is already on the truth grid, so block-averaging it to 2.5 m is the identity.',
                'In the FALSE rule "delta <= 0" is applied to the point estimate; TRUE also requires the CI to exclude 0.',
                'bilinear_ranked_realistic (= F5 alloc_bilinear|realistic) is a context arm that isolates the calibration effect; '
                'it has no rule attached.',
                'Runs under the .venv_a2 interpreter (pandas needed for the pickles), same versions as f5.json.'],
            limitations=[
                'Static land-cover boundaries (HR NDVI < 0.35) are a proxy for change boundaries, not a landslide label.',
                'Four datasets give four LODO folds; spot has only 9 images, so its fold is small.',
                'The calibration maps 10 m NDVI to a non-vegetation fraction for THIS truth definition; it is not a change fraction '
                'and is not validated for a pre/post difference.',
                'HRharm is harmonised to S2, which favours LR-consistent estimators; the same reference is used for every arm.'],
            runtime_seconds=round(time.time() - t0, 1), sr_seconds=P['sr_seconds'], environment_extra=A1.environment_extra())
        return _finish(result, started)
    except C.Blocked as exc:
        result.update(status='BLOCKED', verdict='NOT_RUN', reason=str(exc))
        return _finish(result, started)


def _finish(result, started):
    import json
    result.setdefault('interpretations', [])
    result.setdefault('reproduction_check', None)
    result.setdefault('measurements', [])
    out = C.write_result('a2', result, started)
    print(json.dumps({k: out.get(k) for k in ('status', 'verdict', 'reason')}, indent=2))
    return C.exit_code(out['status'])


if __name__ == '__main__':
    raise SystemExit(main())
