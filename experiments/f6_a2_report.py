"""F6: honest A2 reporting (no new SR runs).

RESULTS_EXCEPTIONAL.md (now quarantined, see experiments/results/INVALIDATED.md) cherry-picked
only the two opensr-test datasets (NAIP, SPOT) where SEN2SR-lite beat bicubic on PSNR, and omitted:
  - the pooled result across all 5 datasets / 178 images (SR underperforms bicubic pooled)
  - the Venus result (SR clearly worse than bicubic)
This is finding B14 in the audit (selective reporting).

This script does not run any SR inference. It reads the existing experiments/results/x2.json
(produced by experiments/x2_hr_benchmark.py, which this script does not touch or re-run) and:

  1. Lists every dataset's PSNR delta vs bicubic (mean, 95% CI, n images) -- including the three
     datasets omitted from RESULTS_EXCEPTIONAL.md (spain_crops, spain_urban, venus).
  2. Recomputes the pooled PSNR delta across all 178 images directly from x2.json's per-image
     data (does not just copy x2.json's own `pooled` block or the audit's quoted number), using
     the same paired-bootstrap estimator (trustsr.bootstrap.paired_bootstrap_ci) and the same
     bootstrap parameters (2000 replicates, 95% CI, seed 2024) that x2's own protocol used.
  3. Recomputes the pooled delta with NAIP excluded, since x2's own limitations note flags
     possible train/test leakage for NAIP (SEN2SR-lite was trained on SEN2NAIPv2).
  4. Reports the opensr-test hallucination ("ha_metric") column already computed by x2 via
     opensr_test.Metrics, for SR vs bicubic on every dataset.

Run: python3 -m experiments.f6_a2_report   (from repo root; pure stdlib + numpy, no GPU/opensr-test needed)
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from trustsr.bootstrap import paired_bootstrap_ci

ROOT = Path(__file__).resolve().parent.parent
X2_PATH = ROOT / 'experiments/results/x2.json'
FIX_YAML_PATH = ROOT / 'configs/fix.yaml'
BOOT = {'replicates': 2000, 'ci': 0.95, 'seed': 2024}   # matches x2's own protocol.bootstrap and configs/fix.yaml statistics.bootstrap


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_x2() -> dict:
    return json.loads(X2_PATH.read_text(encoding='utf-8'))


def per_image_psnr_diffs(x2: dict, exclude_datasets=()) -> list[float]:
    """Per-image PSNR(sr) - PSNR(bicubic), the same paired unit x2's own pooled CI uses."""
    return [im['psnr']['sr'] - im['psnr']['bicubic']
            for im in x2['per_image'] if im['dataset'] not in exclude_datasets]


def recompute_pooled(x2: dict, exclude_datasets=()) -> dict:
    diffs = per_image_psnr_diffs(x2, exclude_datasets)
    return paired_bootstrap_ci(diffs, BOOT['replicates'], BOOT['ci'], BOOT['seed'])


def per_dataset_rows(x2: dict) -> dict:
    rows = {}
    for ds, d in x2['datasets'].items():
        dv = d['delta_vs_bicubic']['psnr']
        rows[ds] = {
            'n_images': d['n_images'],
            'mean_psnr_sr': d['mean']['psnr']['sr'],
            'mean_psnr_bicubic': d['mean']['psnr']['bicubic'],
            'delta_vs_bicubic_psnr_db': dv['mean'],
            'ci95_lo': dv['lo'],
            'ci95_hi': dv['hi'],
            'beats_bicubic': d['beats_bicubic'],
        }
    return rows


def hallucination_column(x2: dict) -> dict:
    groups = x2.get('opensr_test_groups') or {}
    out = {}
    for ds, ms in groups.items():
        sr = ms.get('sr', {})
        bc = ms.get('bicubic', {})
        out[ds] = {
            'sr_ha_metric': sr.get('ha_metric'),
            'bicubic_ha_metric': bc.get('ha_metric'),
            'sr_higher_than_bicubic': (sr.get('ha_metric') is not None and bc.get('ha_metric') is not None
                                        and sr['ha_metric'] > bc['ha_metric']),
        }
    return out


def build_report() -> dict:
    x2 = load_x2()

    pooled_all = recompute_pooled(x2)
    pooled_naip_excluded = recompute_pooled(x2, exclude_datasets=('naip',))

    x2_pooled_claim = x2['pooled']['delta_psnr_all_images_ci']
    audit_claim = {'mean': -0.097, 'lo': -0.157, 'hi': -0.037, 'n': 178}   # RESULTS_EXCEPTIONAL.md / B14 quoted figure, rounded to 3dp

    matches_x2_json = (round(pooled_all['mean'], 6) == round(x2_pooled_claim['mean'], 6)
                        and round(pooled_all['lo'], 6) == round(x2_pooled_claim['lo'], 6)
                        and round(pooled_all['hi'], 6) == round(x2_pooled_claim['hi'], 6))
    matches_audit_claim_3dp = (round(pooled_all['mean'], 3) == audit_claim['mean']
                                and round(pooled_all['lo'], 3) == audit_claim['lo']
                                and round(pooled_all['hi'], 3) == audit_claim['hi']
                                and pooled_all['n'] == audit_claim['n'])

    return {
        'status': 'PASS' if (matches_x2_json and matches_audit_claim_3dp) else 'FAIL',
        'evidence': 'real',
        'config_sha256': sha256_of(FIX_YAML_PATH),
        'source': str(X2_PATH.relative_to(ROOT)),
        'source_x2_config_sha256': x2.get('config_sha256'),
        'finding': 'B14 (selective reporting): RESULTS_EXCEPTIONAL.md reported only NAIP and SPOT '
                   '(the 2/5 datasets where SR beat bicubic on PSNR) and omitted the pooled result '
                   'and the Venus result. This file reports all 5 datasets, the pooled result, a '
                   'NAIP-excluded sensitivity, and the hallucination-metric column.',
        'per_dataset': per_dataset_rows(x2),
        'pooled_all_datasets': {
            'recomputed': pooled_all,
            'matches_x2_json_pooled_block': matches_x2_json,
            'x2_json_pooled_block': x2_pooled_claim,
            'matches_audit_quoted_claim_3dp': matches_audit_claim_3dp,
            'audit_quoted_claim': audit_claim,
        },
        'naip_excluded_sensitivity': {
            'recomputed': pooled_naip_excluded,
            'delta_vs_pooled_all_mean_db': pooled_naip_excluded['mean'] - pooled_all['mean'],
        },
        'opensr_hallucination_column': hallucination_column(x2),
        'bootstrap_params': BOOT,
    }


def write_report_md(res: dict, path: Path) -> None:
    L = []
    L.append('# F6: honest A2 reporting (corrects B14 selective reporting)\n')
    L.append(f'Status: **{res["status"]}**  |  evidence: **{res["evidence"]}**  |  config_sha256: `{res["config_sha256"]}`\n')
    L.append('\n## What RESULTS_EXCEPTIONAL.md omitted (B14)\n')
    L.append(
        'RESULTS_EXCEPTIONAL.md reported PSNR gains for SEN2SR-lite only on the NAIP and SPOT '
        'opensr-test datasets -- the two of five where SR beats bicubic. It did not report the '
        'pooled result across all 5 datasets / 178 images, and did not report the Venus dataset, '
        'where SR is clearly worse than bicubic. That is a selective-reporting problem: a reader of '
        'RESULTS_EXCEPTIONAL.md alone would conclude SR reliably beats bicubic on PSNR, which the '
        'full x2.json data does not support.\n')
    L.append('\n## Per-dataset PSNR delta vs bicubic (all 5 datasets, including the 3 omitted)\n')
    L.append('| dataset | n images | delta vs bicubic (dB) | 95% CI | beats bicubic | reported in RESULTS_EXCEPTIONAL.md |')
    L.append('|---|---|---|---|---|---|')
    omitted = {'spain_crops', 'spain_urban', 'venus'}
    for ds, d in res['per_dataset'].items():
        L.append(f'| {ds} | {d["n_images"]} | {d["delta_vs_bicubic_psnr_db"]:+.3f} | '
                 f'[{d["ci95_lo"]:+.3f}, {d["ci95_hi"]:+.3f}] | {"YES" if d["beats_bicubic"] else "NO"} | '
                 f'{"no (B14)" if ds in omitted else "yes"} |')
    p = res['pooled_all_datasets']
    r = p['recomputed']
    L.append('\n## Pooled result across all 5 datasets / 178 images (recomputed from x2.json per-image data)\n')
    L.append(f'Recomputed: **{r["mean"]:+.3f} dB [{r["lo"]:+.3f}, {r["hi"]:+.3f}]**, n={r["n"]} images '
             f'(paired bootstrap, {r["replicates"]} replicates, {int(r["ci"]*100)}% CI, seed {r["seed"]}).\n')
    L.append(f'\nMatches x2.json\'s own `pooled.delta_psnr_all_images_ci` block: **{p["matches_x2_json_pooled_block"]}** '
             f'(x2.json: {p["x2_json_pooled_block"]["mean"]:+.3f} [{p["x2_json_pooled_block"]["lo"]:+.3f}, {p["x2_json_pooled_block"]["hi"]:+.3f}]).\n')
    L.append(f'\nMatches the audit\'s quoted B14 figure (-0.097 dB [-0.157, -0.037], 178 images) at 3 decimal places: '
             f'**{p["matches_audit_quoted_claim_3dp"]}**.\n')
    n = res['naip_excluded_sensitivity']
    nr = n['recomputed']
    L.append('\n## NAIP-excluded sensitivity\n')
    L.append(
        'x2.json\'s own limitations note: "SEN2SR-lite was trained on SEN2NAIPv2 (NAIP); train/test '
        'overlap with the OpenSR-test NAIP set cannot be excluded from public information." NAIP is '
        'also the dataset with the largest positive SR-vs-bicubic delta, so it is worth checking '
        'whether the pooled result depends on it.\n')
    L.append(f'\nPooled, NAIP excluded: **{nr["mean"]:+.3f} dB [{nr["lo"]:+.3f}, {nr["hi"]:+.3f}]**, n={nr["n"]} images.\n')
    L.append(f'\nChange vs the all-datasets pooled mean: {n["delta_vs_pooled_all_mean_db"]:+.3f} dB. '
             f'{"Excluding NAIP makes the pooled result more negative" if n["delta_vs_pooled_all_mean_db"] < 0 else "Excluding NAIP makes the pooled result less negative"} '
             '-- SR does not beat bicubic pooled either way, and removing the one dataset with a possible '
             'leakage advantage does not rescue the pooled PSNR claim.\n')
    L.append('\n## opensr-test hallucination metric (ha_metric): SR vs bicubic\n')
    L.append(
        'This is the motivation for a trust gate even on datasets where SR wins on PSNR: opensr-test\'s '
        'own consistency/correctness decomposition scores SR as *more* hallucination-prone than bicubic '
        'on every single dataset, including NAIP and SPOT where SR wins on PSNR.\n')
    L.append('\n| dataset | SR ha_metric | bicubic ha_metric | SR higher (worse) |')
    L.append('|---|---|---|---|')
    for ds, h in res['opensr_hallucination_column'].items():
        L.append(f'| {ds} | {h["sr_ha_metric"]:.4f} | {h["bicubic_ha_metric"]:.4f} | {"YES" if h["sr_higher_than_bicubic"] else "no"} |')
    L.append('\n## Reproducibility\n')
    L.append(f'Computed by `experiments/f6_a2_report.py` from `{res["source"]}` '
             f'(x2 config_sha256: `{res["source_x2_config_sha256"]}`). No SR inference was run; this is a '
             'pure recomputation over existing per-image PSNR values. See `tests/test_f6_a2_report.py` for '
             'the check that this report\'s pooled number matches what the script computes from x2.json.\n')
    Path(path).write_text('\n'.join(L) + '\n', encoding='utf-8')


def main():
    res = build_report()
    out_dir = ROOT / 'experiments/results'
    (out_dir / 'f6.json').write_text(json.dumps(res, indent=2, sort_keys=False) + '\n', encoding='utf-8')
    write_report_md(res, out_dir / 'f6_REPORT.md')
    print(json.dumps({k: res[k] for k in ('status', 'evidence', 'config_sha256')}, indent=2))
    print('pooled recomputed:', res['pooled_all_datasets']['recomputed'])


if __name__ == '__main__':
    main()
