# X5 noise-model report (agent A5)

Status **FAIL** (exit 2). Selected estimator (pre-registered rule, E6 only): **eb_stratified**. Config sha256 `41a31344763472bd...`. Decision rule and failure modes: `docs/adr-x5-noise-model.md`. Evidence labels are per table.

## 1. Wayanad leave-one-date-out (real; headline, n_pre = 2 for the formula, 1 dof)

Stable pixels: 2,357,687 2.5 m px per fold (147,393 10 m px outside the 1500 m disks, footprint and parent). Coverage of |d| <= 2 sigma, pooled over 3 folds; 95 % block-bootstrap CI (320 m blocks); nominal 95.45 %. Three folds share three dates: they are not independent.

| sigma | coverage k = 2 [CI] | n px (3 folds) | k_eff [CI] | one-sided flag rate S at k = 2 [CI] | median sigma / v1 (mean of folds) | coverage after removing fold offset (post hoc) |
|---|---|---|---|---|---|---|
| old_v1 | 54.1 % [53.3, 54.9] | 7,073,061 | 12.12 [11.39, 12.87] | 27.4 % [26.6, 28.2] |  | 81.3 % |
| raw | 71.7 % [71.4, 72.1] | 7,073,061 | 7.90 [7.42, 8.38] | 19.7 % [18.5, 20.8] | 1.67 | 89.2 % |
| window | 79.8 % [78.7, 80.9] | 7,073,061 | 3.61 [3.38, 3.85] | 16.5 % [14.9, 18.1] | 1.84 | 96.1 % |
| eb_stratified | 88.0 % [86.7, 89.1] | 7,073,061 | 2.59 [2.52, 2.67] | 11.2 % [10.1, 12.3] | 2.12 | 99.3 % |
| eb_window | 79.8 % [78.7, 80.9] | 7,073,061 | 3.61 [3.38, 3.85] | 16.5 % [14.9, 18.1] | 1.84 | 96.1 % |
| stable_empirical | 98.4 % [98.0, 98.8] | 7,073,061 | 1.63 [1.59, 1.69] |  |  |  |

Keep rule (coverage within +/- 2 pp of 95.45 %, else report the cause): **MISSED** for eb_stratified: 88.0 % [86.7, 89.1], gap -7.5 pp; v1: 54.1 % [53.3, 54.9].

Per fold (held-out date: coverage of the selected estimator, median d, share of mean d^2 that is a scene-wide offset):

- 2024-01-16: coverage 66.8 %, median d -0.057, offset share 77.4 %
- 2024-01-21: coverage 99.6 %, median d +0.008, offset share 8.2 %
- 2024-01-26: coverage 97.4 %, median d +0.048, offset share 70.3 %

Median NDVI on stable pixels by date: 2024-01-16 0.751, 2024-01-21 0.802, 2024-01-26 0.831, 2024-12-06 0.841.

## 2. Pre-mean vs December (real; n_pre = 3; CONFOUNDED by season, measures bias as much as noise)

| sigma | coverage k = 2 [CI] | k_eff [CI] | median sigma / v1 | coverage centred (post hoc) |
|---|---|---|---|---|
| old_v1 | 67.8 % [64.2, 71.5] | 6.89 [5.91, 8.00] |  | 81.6 % |
| raw | 80.7 % [78.0, 83.4] | 5.07 [4.34, 5.90] | 1.38 | 88.5 % |
| window | 84.3 % [81.6, 87.0] | 3.39 [3.01, 3.83] | 1.44 | 92.4 % |
| eb_stratified | 95.4 % [94.4, 96.4] | 2.00 [1.93, 2.09] | 1.57 | 97.0 % |
| eb_window | 84.3 % [81.6, 87.0] | 3.39 [3.01, 3.83] | 1.44 | 92.4 % |

Stable median d = -0.047 NDVI (December is greener). A coverage near nominal here is NOT evidence of calibration: the offset happens to be of the order of the between-date spread.

## 3. Estimator selection (real, E6 12-date cache, other AOI, 10 m, 3 pool dates + 1 held-out, 100 seeded draws; no Wayanad pixel)

| sigma | mean held-out NLL | coverage k = 2 (mean; draw range 2.5-97.5 %) | mean k_eff | nu0 median (share at cap) |
|---|---|---|---|---|
| raw | 1.654 | 82.2 % (38.9 % - 99.0 %) | 3.89 |  |
| window | -0.362 | 87.0 % (41.6 % - 99.9 %) | 2.50 |  |
| eb_stratified | -0.599 | 89.8 % (44.9 % - 99.9 %) | 2.08 | 1e+04 (79.0 %) |
| eb_window | -0.362 | 87.0 % (41.6 % - 99.9 %) | 2.50 | 1e+04 (100.0 %) |
| no_inflation_reference | 2.415 | 78.1 % (32.0 % - 98.6 %) | 4.48 |  |

Winner eb_stratified; paired NLL difference best - runner-up (eb_stratified - eb_window) -0.237 [-0.402, -0.117] (CI excludes 0: no tie rule needed).

## 4. Synthetic simulation (SYNTHETIC; 120,000 pixels, known truth, n_pre = 3, 8 runs)

| sigma | variance / true predictive variance | coverage k = 2 | k_eff |
|---|---|---|---|
| v1 | 0.530 (SD ratio 0.728) | 73.1 % | 5.39 |
| raw | 1.007 | 82.4 % | 4.18 |
| eb_stratified | 1.007 | 95.2 % | 2.03 |

Oracle (true sigma) coverage 95.4 %.

## Limitations

- three pre dates only: leave-one-date-out has n_pre = 2 (1 dof) and 3 non-independent folds
- block-bootstrap CIs resample space, not dates: they understate date-level uncertainty
- December comparison is confounded by season
- stable pixels lie inside the SR crop around the slide, outside the footprint and 1500 m disks
- coverage on stable pixels says nothing about power on real change
- CPU macOS run; SR determinism verified by step3 only on this platform
