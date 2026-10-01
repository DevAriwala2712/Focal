"""F9 spot-check: recompute F6's headline numbers from x2.json WITHOUT using experiments/f6_a2_report.py.

Deliberately a different method from F6's:
  * a flat record table built straight off x2.json['per_image'] (pandas is not installed in this venv, so
    the table is a dict-of-arrays), not F6's dict walk;
  * the pooled point estimate by three separate routes (vectorised numpy, a longhand Python loop over the
    raw json, and a per-dataset weighted recombination), not F6's accumulator;
  * the CI by an independently written paired bootstrap (resample image ROW INDICES with a fresh
    numpy Generator), so agreement is not an artefact of sharing F6's bootstrap helper;
  * additionally a closed-form normal-approximation CI as a sanity band on the bootstrap CI.

If F6's -0.097 dB [-0.157, -0.037] / n=178 is right, all three routes must agree to ~0.001 dB on the
point estimate, and the bootstrap CI must land within a few thousandths of F6's.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    x2 = json.load(open(ROOT / 'experiments/results/x2.json'))
    rec = {k: [] for k in ('dataset', 'psnr_sr', 'psnr_bicubic', 'ha_sr', 'ha_bicubic')}
    for r in x2['per_image']:
        rec['dataset'].append(r['dataset'])
        rec['psnr_sr'].append(r['psnr']['sr'])
        rec['psnr_bicubic'].append(r['psnr']['bicubic'])
        rec['ha_sr'].append(r['extra']['sr']['ha_metric'])
        rec['ha_bicubic'].append(r['extra']['bicubic']['ha_metric'])
    ds = np.array(rec['dataset'])
    psnr_sr = np.array(rec['psnr_sr'], float)
    psnr_bic = np.array(rec['psnr_bicubic'], float)
    delta = psnr_sr - psnr_bic
    datasets = sorted(set(rec['dataset']))
    out = {'n_images_in_x2_per_image': int(delta.size),
           'n_datasets': len(datasets),
           'images_per_dataset': {d: int((ds == d).sum()) for d in datasets}}

    # ---- pooled point estimate, three independent routes
    mean_vec = float(delta.mean())
    mean_longhand = float(sum(r['psnr']['sr'] - r['psnr']['bicubic'] for r in x2['per_image']) / len(x2['per_image']))
    # third route: per-dataset means recombined with image-count weights (must equal the flat mean)
    mean_weighted = float(sum(delta[ds == d].sum() for d in datasets) / delta.size)
    out['pooled_point_estimate'] = {'numpy_vectorised': mean_vec, 'longhand_loop': mean_longhand,
                                   'per_dataset_weighted_recombination': mean_weighted,
                                   'all_three_agree_to_1e_12': bool(
                                       abs(mean_vec - mean_longhand) < 1e-12
                                       and abs(mean_vec - mean_weighted) < 1e-12)}
    mean_pandas = mean_vec

    # ---- independently written paired bootstrap over image row indices
    def boot_ci(vals, reps=2000, seed=2024, alpha=0.05):
        rng = np.random.default_rng(seed)
        n = len(vals)
        means = np.array([vals[rng.integers(0, n, n)].mean() for _ in range(reps)])
        return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))

    lo, hi = boot_ci(delta)
    se = delta.std(ddof=1) / np.sqrt(delta.size)
    out['pooled_ci'] = {'f9_independent_bootstrap_lo': lo, 'f9_independent_bootstrap_hi': hi,
                        'f9_normal_approx_lo': float(mean_pandas - 1.96 * se),
                        'f9_normal_approx_hi': float(mean_pandas + 1.96 * se)}

    # ---- what F6 and x2 claim
    f6 = json.load(open(ROOT / 'experiments/results/f6.json'))
    out['f6_json_pooled'] = f6['pooled_all_datasets']
    out['x2_own_pooled'] = x2['pooled']['delta_psnr_all_images_ci']
    claim = -0.097
    out['agreement'] = {
        'f9_point_vs_prereg_claim_-0.097_within_0.001': bool(abs(round(mean_pandas, 3) - claim) < 1e-9),
        'f9_point_vs_x2_own': bool(abs(mean_pandas - x2['pooled']['mean_delta_psnr_all_images']) < 1e-9),
        'f9_point_rounded_3dp': round(mean_pandas, 3),
        'f9_ci_rounded_3dp': [round(lo, 3), round(hi, 3)],
        'prereg_ci_claim': [-0.157, -0.037],
        'f9_ci_matches_prereg_at_3dp': bool(round(lo, 3) == -0.157 and round(hi, 3) == -0.037),
    }

    # ---- NAIP-excluded sensitivity, recomputed
    nn = delta[ds != 'naip']
    lo2, hi2 = boot_ci(nn)
    out['naip_excluded'] = {'n': int(nn.size), 'point': float(nn.mean()),
                            'ci': [lo2, hi2], 'point_rounded': round(float(nn.mean()), 3),
                            'f6_claim': f6.get('naip_excluded_sensitivity')}

    # ---- per-dataset table, recomputed
    out['per_dataset'] = {}
    for d in datasets:
        g = delta[ds == d]
        l, h = boot_ci(g)
        out['per_dataset'][d] = {'n': int(g.size), 'delta': round(float(g.mean()), 3),
                                'ci': [round(l, 3), round(h, 3)],
                                'beats_bicubic': bool(g.mean() > 0)}

    # ---- hallucination column: SR worse on every dataset?
    ha_sr = np.array(rec['ha_sr'], float)
    ha_bic = np.array(rec['ha_bicubic'], float)
    out['ha_metric'] = {d: {'sr': round(float(ha_sr[ds == d].mean()), 4),
                            'bicubic': round(float(ha_bic[ds == d].mean()), 4),
                            'sr_worse': bool(ha_sr[ds == d].mean() > ha_bic[ds == d].mean())}
                        for d in datasets}
    out['ha_sr_worse_on_every_dataset'] = bool(all(v['sr_worse'] for v in out['ha_metric'].values()))

    print(json.dumps(out, indent=1))
    json.dump(out, open(ROOT / 'experiments/f9_f6_recompute.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
