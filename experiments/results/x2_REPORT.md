# A2: HR validation benchmark (x2)

Status: **FAIL**  |  evidence: **real**  |  verdict: **MIXED**


SR beats bicubic (PSNR CI lower bound > 0) on 2/5 datasets. Model: pinned SEN2SR-lite RGBN x4 (B04,B03,B02,B08), device `cpu`. Reference: OpenSR-test HRharm, reflectance = DN/10000, data range 1.0, 16 px border excluded, candidates clipped to [0,1]. CI = paired bootstrap over images (2000 replicates, 95 %, seed 2024). Keep rule: PSNR CI lower bound > 0.


| dataset | n images (valid HR px) | metric | nearest | bicubic | lanczos | SR | delta vs bicubic | 95 % CI | evidence |
|---|---|---|---|---|---|---|---|---|---|
| naip | 62 (12666848) | PSNR dB | 36.146 | 36.617 | 36.668 | 36.853 | +0.236 | [+0.193, +0.280] | real |
| naip | 62 (12666848) | SSIM | 0.8820 | 0.8956 | 0.8965 | 0.9014 | +0.0058 | [+0.0048, +0.0068] | real |
| spot | 9 (2067870) | PSNR dB | 32.868 | 33.276 | 33.319 | 33.406 | +0.130 | [+0.058, +0.198] | real |
| spot | 9 (2067870) | SSIM | 0.8020 | 0.8280 | 0.8297 | 0.8371 | +0.0091 | [+0.0040, +0.0147] | real |
| spain_crops | 28 (6451200) | PSNR dB | 32.630 | 32.895 | 32.915 | 32.889 | -0.007 | [-0.093, +0.080] | real |
| spain_crops | 28 (6451200) | SSIM | 0.8175 | 0.8322 | 0.8328 | 0.8339 | +0.0018 | [-0.0012, +0.0050] | real |
| spain_urban | 20 (4608000) | PSNR dB | 28.949 | 29.247 | 29.274 | 29.252 | +0.005 | [-0.093, +0.102] | real |
| spain_urban | 20 (4608000) | SSIM | 0.6873 | 0.7114 | 0.7128 | 0.7177 | +0.0062 | [-0.0001, +0.0124] | real |
| venus | 59 (2959902) | PSNR dB | 35.627 | 36.597 | 36.599 | 36.038 | -0.559 | [-0.638, -0.482] | real |
| venus | 59 (2959902) | SSIM | 0.9039 | 0.9235 | 0.9230 | 0.9119 | -0.0116 | [-0.0137, -0.0097] | real |

## Per-dataset verdict and strongest baseline

| dataset | SR beats bicubic | strongest baseline | SR minus strongest (PSNR dB, CI) | images SR>bicubic | by-S2-acquisition CI (n groups) |
|---|---|---|---|---|---|
| naip | YES | lanczos | +0.185 [+0.145, +0.225] | 56/62 | +0.212 [+0.161, +0.262] (47) |
| spot | YES | lanczos | +0.086 [+0.020, +0.147] | 7/9 | +0.130 [+0.058, +0.198] (9) |
| spain_crops | NO | lanczos | -0.026 [-0.104, +0.051] | 15/28 | +0.026 [-0.099, +0.148] (15) |
| spain_urban | NO | lanczos | -0.022 [-0.109, +0.064] | 11/20 | -0.011 [-0.147, +0.129] (10) |
| venus | NO | lanczos | -0.561 [-0.637, -0.487] | 3/59 | -0.542 [-0.635, -0.440] (40) |

Pooled (178 images, 5 datasets): mean PSNR delta -0.097 dB [-0.157, -0.037]; mean of dataset means -0.039 dB.


## 10 m downsample error (block mean 4x4 of unclipped SR vs the 10 m input; mean over images, reflectance units, bands B04,B03,B02,B08)

| dataset | method | MAE per band | bias per band |
|---|---|---|---|
| naip | sr | 0.00080, 0.00073, 0.00061, 0.00096 | -0.00000, +0.00000, -0.00000, +0.00000 |
| naip | bicubic | 0.00111, 0.00086, 0.00076, 0.00123 | +0.00000, +0.00000, +0.00000, -0.00000 |
| naip | lanczos | 0.00102, 0.00079, 0.00069, 0.00113 | +0.00000, +0.00000, +0.00000, +0.00000 |
| spot | sr | 0.00165, 0.00148, 0.00128, 0.00162 | +0.00000, -0.00000, -0.00000, -0.00000 |
| spot | bicubic | 0.00208, 0.00178, 0.00162, 0.00221 | -0.00000, -0.00000, -0.00000, -0.00000 |
| spot | lanczos | 0.00190, 0.00162, 0.00148, 0.00202 | -0.00000, -0.00000, -0.00000, -0.00000 |
| spain_crops | sr | 0.00091, 0.00084, 0.00071, 0.00113 | -0.00000, -0.00000, -0.00000, -0.00000 |
| spain_crops | bicubic | 0.00119, 0.00103, 0.00090, 0.00142 | -0.00000, -0.00000, -0.00000, -0.00000 |
| spain_crops | lanczos | 0.00109, 0.00094, 0.00082, 0.00131 | -0.00000, -0.00000, -0.00000, -0.00000 |
| spain_urban | sr | 0.00170, 0.00160, 0.00135, 0.00182 | +0.00000, -0.00000, -0.00000, -0.00000 |
| spain_urban | bicubic | 0.00223, 0.00191, 0.00173, 0.00242 | -0.00000, +0.00000, -0.00000, -0.00000 |
| spain_urban | lanczos | 0.00204, 0.00175, 0.00158, 0.00222 | +0.00000, +0.00000, +0.00000, -0.00000 |
| venus | sr | 0.00113, 0.00101, 0.00092, 0.00178 | -0.00000, -0.00000, -0.00000, -0.00000 |
| venus | bicubic | 0.00148, 0.00117, 0.00108, 0.00226 | -0.00000, -0.00000, -0.00000, -0.00000 |
| venus | lanczos | 0.00136, 0.00107, 0.00098, 0.00206 | -0.00000, -0.00000, -0.00000, -0.00000 |

## opensr-test 1.3.3 groups (mean over images; consistency: reflectance, spectral, spatial; synthesis; correctness: ha/om/im percent)

| dataset | method | n ok | reflectance | spectral (deg) | spatial | synthesis | hallucination | omission | improvement |
|---|---|---|---|---|---|---|---|---|---|
| naip | sr | 62/62 | 0.00122 | 0.3127 | 0.0000 | 0.00252 | 0.0598 | 0.8579 | 0.0823 |
| naip | bicubic | 62/62 | 0.00180 | 0.4008 | 0.0006 | 0.00149 | 0.0375 | 0.9223 | 0.0402 |
| spot | sr | 9/9 | 0.00227 | 0.4678 | 0.0000 | 0.00507 | 0.0600 | 0.8566 | 0.0833 |
| spot | bicubic | 9/9 | 0.00350 | 0.6481 | 0.0000 | 0.00289 | 0.0341 | 0.9280 | 0.0379 |
| spain_crops | sr | 28/28 | 0.00140 | 0.3162 | 0.0000 | 0.00452 | 0.0848 | 0.7948 | 0.1204 |
| spain_crops | bicubic | 28/28 | 0.00206 | 0.4103 | 0.0000 | 0.00322 | 0.0642 | 0.8600 | 0.0758 |
| spain_urban | sr | 20/20 | 0.00246 | 0.6058 | 0.0000 | 0.00798 | 0.0874 | 0.7872 | 0.1253 |
| spain_urban | bicubic | 20/20 | 0.00378 | 0.8076 | 0.0000 | 0.00602 | 0.0645 | 0.8566 | 0.0789 |
| venus | sr | 59/59 | 0.00224 | 0.6101 | 0.0000 | 0.00674 | 0.2049 | 0.4482 | 0.3469 |
| venus | bicubic | 59/59 | 0.00310 | 0.7754 | 0.0000 | 0.00542 | 0.1378 | 0.5866 | 0.2756 |

## Sensitivity: unharmonized HR (only sets whose raw radiometry is on the S2 scale)

| dataset | n | SR PSNR | bicubic PSNR | delta | 95 % CI |
|---|---|---|---|---|---|
| naip | 62 | 37.040 | 36.869 | +0.171 | [+0.123, +0.221] |
| spot | 9 | 32.919 | 32.782 | +0.137 | [+0.073, +0.200] |
| venus | 59 | 33.759 | 34.169 | -0.410 | [-0.488, -0.338] |

## Geometry check (does HRharm sit on the exact LR subdivision?)

| dataset | scale | HR origin on 10 m grid | shift search within 1 HR px | dataset-reported misalignment (LR px, mean/max) | S2 acquisitions |
|---|---|---|---|---|---|
| naip | x4 | 62/62 | 62/62 | 0.047 / 0.126 | 47 |
| spot | x4 | 9/9 | 9/9 | 0.036 / 0.060 | 9 |
| spain_crops | x4 | 28/28 | 14/28 | 0.081 / 0.261 | 15 |
| spain_urban | x4 | 20/20 | 11/20 | 0.083 / 0.189 | 10 |
| venus | x2 | 12/59 | 35/59 | 0.045 / 0.141 | 40 |

## Reproducibility

Seed 2024, deterministic algorithms on, torch threads 5, device cpu. Full SR recompute gives identical sha256 for every dataset: True. Bytes downloaded by this process: 77704 (cap 6000000000); dataset revision e4600b9c74a621adeec047e5f6cc7a2d70a58134.


## Limitations

- HRharm is harmonized to the Sentinel-2 L2A bands by the dataset authors (radiometry and sub-pixel alignment), which favours LR-consistent estimators such as bicubic; unharmonized HR is not on a common radiometric scale across sets (see raw_scale_check).
- No SCL/cloud mask ships with OpenSR-test; only zero-valued (no-data) pixels are masked.
- SEN2SR-lite was trained on SEN2NAIPv2 (NAIP); train/test overlap with the OpenSR-test NAIP set cannot be excluded from public information.
- NAIP LR is 121 px (< 128 native): reflect-padded to 128 and cropped; the padded border lies inside the 16 HR px excluded border. Production ADR-001 forbids padding; this is an evaluation-only adapter.
- Venus is x2 (5 m): every method is x4-upsampled then block-averaged 2x2 to the 5 m grid; the model output is not compared at 2.5 m there.
- Images sharing a Sentinel-2 acquisition are not independent; the by-group bootstrap is reported as a sensitivity.
- SEN2SR paper table not read (publisher pages return 403 / Cloudflare challenge); no published-number comparison is made.
