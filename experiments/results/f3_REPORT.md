# F3 noise model v2 report

Status **PASS**. Motivated by A5 diagnostics; pre-registered before its own run (configs/fix.yaml `f3_noise_v2`; does not edit `experiments/x5_noise_model.py`). Selected estimator (E6-only selection, no Wayanad pixel): **eb_stratified**. Config sha256 `192345cd156583c6...`.

## 1. Wayanad leave-one-date-out (real; headline)

Stable pixels: 2,358,286 2.5 m px per fold (outside the 1500 m disks, footprint and parent), from the real per-run dihedral NDVI cache (no SR re-run). Coverage of |d| <= 2 sigma, pooled over 3 folds; 95 % block-bootstrap CI; nominal 95.45 %.

| track | coverage k=2 [CI] | gap vs nominal |
|---|---|---|
| normalised (F3 estimator) | 94.2 % [93.3, 95.0] | -1.3 pp |
| same sigma method, no normalisation | 85.8 % [84.5, 87.0] | -9.7 pp |

Keep rule (within +/- 2 pp of 95.45 %): **KEPT**.

Per fold (median d after normalisation, offset share of residual d^2, fitted offsets):

- 2024-01-16: median d +0.0000 (was -0.0566 before normalisation), offset share 0.0 %, offset(held) -0.0566
- 2024-01-21: median d +0.0000 (was +0.0084 before normalisation), offset share 0.0 %, offset(held) +0.0084
- 2024-01-26: median d +0.0000 (was +0.0484 before normalisation), offset share 0.0 %, offset(held) +0.0484

Median NDVI on stable pixels by date: 2024-01-16 0.751, 2024-01-21 0.802, 2024-01-26 0.831.

## 2. Estimator selection (real, E6 12-date cache, other AOI, 100 seeded draws; no Wayanad pixel)

| sigma | mean held-out NLL (normalised) | coverage k=2 (normalised) | mean NLL (no normalisation) | coverage (no normalisation) |
|---|---|---|---|---|
| raw | 1.845 | 83.1 % | 1.654 | 82.2 % |
| window | -0.701 | 89.6 % | -0.362 | 87.0 % |
| eb_stratified | -0.861 | 91.6 % | -0.599 | 89.8 % |
| eb_window | -0.700 | 89.6 % | -0.362 | 87.0 % |

Winner eb_stratified; paired NLL diff best - runner-up (eb_stratified - window) -0.160 [-0.251, -0.083] (CI excludes 0). Mean |fitted offset| 0.0435 NDVI. Residual scene-offset share of mean d^2 after normalisation: 1.4 % (range 0.0 %-19.1 %).

## 3. Change-preservation test (synthetic injection on a real stable-pixel field)

Injected a +0.30 NDVI drop into 1600 real stable px of 2024-01-26 (out of 2,358,286 stable px the offset is fit on). Fitted offset shifted by -0.00004 (-0.01 % of the injected drop). Measured drop after normalisation: +0.3000 (recovery ratio 1.000; 1.0 = fully preserved).

## Limitations

- median(d) is exactly 0 for each LODO fold by construction: offset_held is DEFINED as the median residual against the pool reference, so removing it zeroes the median identically. This is a definitional consequence of the estimator, not itself evidence; the non-trivial, non-tautological result is the COVERAGE (|d| <= 2 sigma), since sigma is estimated from the pool dates’ across-date and dihedral variance, never from d
- three Wayanad pre dates only: leave-one-date-out has n_pre = 2 (1 dof) and 3 non-independent folds
- block-bootstrap CIs resample space, not dates: they understate date-level uncertainty
- the median offset is a SINGLE scalar per date: it cannot correct within-date spatial structure (e.g. a cloud-edge gradient), only a scene-wide shift
- change-preservation test is synthetic (injected on real pixels): it is not evidence about a real event, only about this estimator's sensitivity to a minority-pixel change
- stable pixels lie inside the SR crop around the slide, outside the footprint and 1500 m disks
- coverage on stable pixels says nothing about power on real change
