# F6: honest A2 reporting (corrects B14 selective reporting)

Status: **PASS**  |  evidence: **real**  |  config_sha256: `7ee83389827340ed4e40b03917a7278d6bad11e2a6dac68460e8596c0111e7e6`


## What RESULTS_EXCEPTIONAL.md omitted (B14)

RESULTS_EXCEPTIONAL.md reported PSNR gains for SEN2SR-lite only on the NAIP and SPOT opensr-test datasets -- the two of five where SR beats bicubic. It did not report the pooled result across all 5 datasets / 178 images, and did not report the Venus dataset, where SR is clearly worse than bicubic. That is a selective-reporting problem: a reader of RESULTS_EXCEPTIONAL.md alone would conclude SR reliably beats bicubic on PSNR, which the full x2.json data does not support.


## Per-dataset PSNR delta vs bicubic (all 5 datasets, including the 3 omitted)

| dataset | n images | delta vs bicubic (dB) | 95% CI | beats bicubic | reported in RESULTS_EXCEPTIONAL.md |
|---|---|---|---|---|---|
| naip | 62 | +0.236 | [+0.193, +0.280] | YES | yes |
| spot | 9 | +0.130 | [+0.058, +0.198] | YES | yes |
| spain_crops | 28 | -0.007 | [-0.093, +0.080] | NO | no (B14) |
| spain_urban | 20 | +0.005 | [-0.093, +0.102] | NO | no (B14) |
| venus | 59 | -0.559 | [-0.638, -0.482] | NO | no (B14) |

## Pooled result across all 5 datasets / 178 images (recomputed from x2.json per-image data)

Recomputed: **-0.097 dB [-0.157, -0.037]**, n=178 images (paired bootstrap, 2000 replicates, 95% CI, seed 2024).


Matches x2.json's own `pooled.delta_psnr_all_images_ci` block: **True** (x2.json: -0.097 [-0.157, -0.037]).


Matches the audit's quoted B14 figure (-0.097 dB [-0.157, -0.037], 178 images) at 3 decimal places: **True**.


## NAIP-excluded sensitivity

x2.json's own limitations note: "SEN2SR-lite was trained on SEN2NAIPv2 (NAIP); train/test overlap with the OpenSR-test NAIP set cannot be excluded from public information." NAIP is also the dataset with the largest positive SR-vs-bicubic delta, so it is worth checking whether the pooled result depends on it.


Pooled, NAIP excluded: **-0.275 dB [-0.345, -0.203]**, n=116 images.


Change vs the all-datasets pooled mean: -0.178 dB. Excluding NAIP makes the pooled result more negative -- SR does not beat bicubic pooled either way, and removing the one dataset with a possible leakage advantage does not rescue the pooled PSNR claim.


## opensr-test hallucination metric (ha_metric): SR vs bicubic

This is the motivation for a trust gate even on datasets where SR wins on PSNR: opensr-test's own consistency/correctness decomposition scores SR as *more* hallucination-prone than bicubic on every single dataset, including NAIP and SPOT where SR wins on PSNR.


| dataset | SR ha_metric | bicubic ha_metric | SR higher (worse) |
|---|---|---|---|
| naip | 0.0598 | 0.0375 | YES |
| spot | 0.0600 | 0.0341 | YES |
| spain_crops | 0.0848 | 0.0642 | YES |
| spain_urban | 0.0874 | 0.0645 | YES |
| venus | 0.2049 | 0.1378 | YES |

## Reproducibility

Computed by `experiments/f6_a2_report.py` from `experiments/results/x2.json` (x2 config_sha256: `1f2b9d7d174dc0e606a8795427ec0d68e60761350c4e83866c0b3d92e8c67453`). No SR inference was run; this is a pure recomputation over existing per-image PSNR values. See `tests/test_f6_a2_report.py` for the check that this report's pooled number matches what the script computes from x2.json.

