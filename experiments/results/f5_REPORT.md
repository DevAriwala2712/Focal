# F5 / A8 — does SR-informed sub-pixel allocation beat a naive ranking, against real HR truth?

`experiments/f5_a8_mapping.py` · `experiments/results/f5.json` · pre-registered in `configs/fix.yaml` block `a8`
`config_sha256` = `18cce7b1ea88855bc054b42e6cbe3219eef73b2909eb091d07d5ada3df170970` (sha256 of `configs/fix.yaml`)
Status **PASS** (the test ran end to end; `fix.yaml` a8 sets no numeric keep threshold, so the scientific answer is the
`verdict`). Evidence **real**. Verdict **SR_RANKING_BETTER — but only with NAIP included, and by ~1% of IoU.**

---

## Required caveat — read before any number below

**Static land-cover boundaries (vegetation / water) are a PROXY for change boundaries, not a landslide label.**
A8 measures whether SR-informed ranking places sub-pixel detail at real HR boundaries *in general*. It says nothing
about landslide-specific accuracy, and no result here licenses a claim about the Wayanad mapping or any change product.

---

## What was actually run

| | |
|---|---|
| Images | OpenSR-test HRharm, the sets A2 already cached and validated: `naip` (62), `spot` (9), `spain_crops` (28), `spain_urban` (20) = **119 images**, 72 unique Sentinel-2 acquisitions |
| Excluded | `venus` — its HR reference is 5 m (2x), so it cannot carry a 2.5 m truth mask |
| 10 m input | the **real paired Sentinel-2 L2A** image shipped with each HR reference. **No synthetic downsample is used anywhere in this experiment** |
| SR | **real pretrained SEN2SR-lite RGBN x4 forward pass**, CPU, 119 passes, deterministic (`torch.use_deterministic_algorithms(True)`, seed 2024), per-dataset SR output sha256 in `f5.json` |
| Truth at 2.5 m | HR NDVI < **0.35** → non-vegetation (primary); sensitivity at 0.25 and 0.45; HR NDWI > 0 → water (second class) |
| Evaluation area | HR pixels valid in all 4 bands, 16 HR px border excluded per side (the A2 / opensr-test convention) |
| Runtime | 100 s total on CPU (Apple arm64, torch 2.8.0, no CUDA) |

### Methods (every one produces a 2.5 m binary mask)

| method | what it is |
|---|---|
| `blocky_10m` | threshold at 10 m, replicate the block value to all 16 sub-pixels. No allocation. |
| `hard_threshold_v1_style` | v1's gate: the 2.5 m hard threshold on the SR index AND the replicated 10 m parent — `experiments/wayanad_evidence/gate.py`'s `OBSERVED = S & P`. No fraction, no block-sum constraint. |
| `alloc_bilinear` | block-sum-constrained allocation, sub-pixels ranked by the **bilinear-upsampled 10 m index**. The naive ranking baseline. |
| `alloc_pretrained_sr` | the **same** allocation, sub-pixels ranked by the index of the real pretrained SR output. |
| `alloc_finetuned_sr` | **BLOCKED** — F4's fine-tune has not landed. Never substituted by pretrained output. |
| heavy SR (Colab) | **NOT RUN** — Colab is human-operated and out of scope for this sandbox. |

Allocation rule: within each 10 m block take the `k = round(16 x fraction)` lowest-score sub-pixels. The two `alloc_*`
methods receive the *identical* fraction, so **they differ only in the ranking** — which is the thing under test.

### Fraction variants

* **oracle** — the block fraction read straight off the HR truth. Isolates ranking quality; this is the headline arm.
* **realistic** — the block fraction from two-endmember unmixing of the 10 m image alone (endmembers = mean reflectance
  of its own lowest / highest 10% index pixels). No HR information enters it. This is what a product could compute.

---

## Headline — paired IoU(`alloc_pretrained_sr`) − IoU(`alloc_bilinear`), oracle fraction, NDVI < 0.35

Image-paired percentile bootstrap, 2000 replicates, seed 2024. Sign reported as found.

| subset | delta IoU | 95% CI | n images | images where SR ranking wins |
|---|---|---|---|---|
| **all 4 datasets** | **+0.0081** | **[+0.0035, +0.0141]** | **118** | 87 / 118 |
| **excluding NAIP** | **+0.0016** | **[−0.0026, +0.0053]** | **56** | 30 / 56 |
| naip | +0.0139 | [+0.0065, +0.0238] | 62 | 57 / 62 |
| spot | −0.0002 | [−0.0010, +0.0007] | 9 | 1 / 9 |
| spain_crops | −0.0015 | [−0.0090, +0.0044] | 27 | 14 / 27 |
| spain_urban | +0.0066 | [+0.0025, +0.0129] | 20 | 15 / 20 |

(The denominator is 118, not 119: one `spain_crops` image has no non-vegetation truth pixel inside the valid area, so
its IoU is undefined for both methods and it is dropped from the paired comparison rather than scored as 0 or 1.)

**Reading this honestly.**

1. Pooled over all four datasets the CI excludes zero, so SR ranking *is* better than bilinear ranking — by
   **0.008 IoU on a base of 0.837, about 1% relative**. It is a real effect and a small one.
2. **Remove NAIP and the effect is no longer distinguishable from zero** (+0.0016, CI spans 0, n=56). NAIP is exactly
   the set where SEN2SR-lite train/test overlap cannot be excluded (SEN2NAIPv2 training data; see A2 limitations and
   F6), and NAIP alone carries the largest per-dataset effect (+0.0139). Of the three non-NAIP sets, one
   (`spain_urban`) is positive with a CI excluding zero, and two (`spot`, `spain_crops`) are flat or very slightly
   negative.
3. The defensible claim is therefore narrow: *with the possibly-leaking NAIP set included, SR ranking beats bilinear
   ranking by about 1% of IoU; without it, no pooled difference is detectable at n=56.*

### Where the effect lives: feature-width stratification

Truth components are binned by local width — `(2 x max distance-to-background − 1) x 2.5 m`, i.e. twice the largest
inscribed radius (exact for odd pixel widths, 1 px low for even; the bin edges sit far from that ambiguity). Stratum
IoU is computed inside the stratum dilated by 4 px (one 10 m parent block), so distant predictions neither help nor hurt.

| width bin | delta IoU, all datasets | 95% CI | n | delta IoU, excluding NAIP | 95% CI | n |
|---|---|---|---|---|---|---|
| 0–20 m | **+0.0164** | [+0.0106, +0.0233] | 102 | **+0.0077** | [+0.0018, +0.0132] | 47 |
| 20–50 m | +0.0051 | [+0.0004, +0.0092] | 43 | +0.0037 | [−0.0063, +0.0124] | 18 |
| 50–150 m | +0.0055 | [+0.0022, +0.0091] | 27 | +0.0027 | [−0.0017, +0.0064] | 11 |
| 150+ m | +0.0011 | [+0.0005, +0.0018] | 81 | +0.0006 | [−0.0000, +0.0014] | 41 |

This is the most specific positive result in F5, and the only one that survives removing NAIP: **the advantage of SR
ranking is concentrated in the narrowest features (0–20 m) and is ~15x smaller for features wider than 150 m.** That is
the direction the sub-pixel hypothesis predicts — wide blobs are nearly all interior, where no ranking matters, while
narrow features are all boundary. Absolute IoU in the 0–20 m bin is low for every method (0.272 for SR, 0.255 for
bilinear, 0.128 for blocky, oracle fraction, all datasets), so this is an improvement on a hard problem, not a solved one.

---

## The realistic fraction: allocation is *worse than not allocating at all*

| method (all datasets, NDVI<0.35) | mean IoU | median IoU | BF1 @1 px | BF1 @2 px | pooled area error |
|---|---|---|---|---|---|
| `blocky_10m` | 0.7596 | 0.8879 | 0.337 | 0.472 | −0.35% |
| `hard_threshold_v1_style` | 0.7625 | 0.8939 | 0.357 | 0.494 | −1.7% |
| `alloc_bilinear` \| oracle | 0.8286 | 0.9352 | 0.706 | 0.804 | 0.0% (exact by construction) |
| `alloc_pretrained_sr` \| oracle | **0.8367** | 0.9404 | 0.710 | 0.804 | 0.0% |
| `alloc_bilinear` \| realistic | 0.4456 | 0.5057 | 0.158 | 0.223 | −15.5% |
| `alloc_pretrained_sr` \| realistic | 0.4465 | 0.5058 | 0.156 | 0.223 | −15.5% |

* Under the **realistic** fraction the SR-vs-bilinear delta is +0.0009 [+0.0006, +0.0012] (n=119): non-zero, and
  practically nothing, because both methods are swamped by fraction error.
* `alloc_pretrained_sr|realistic` minus `blocky_10m` = **−0.313 IoU [−0.344, −0.282]**, n=119, better on 3/119 images.
  **With a fraction actually estimated from Sentinel-2, sub-pixel allocation loses to doing no allocation at all**,
  whichever ranking it uses.
* Why: the estimated block fraction is only weakly related to the true one — mean per-block MAE **0.400**, mean
  within-image Pearson r **0.294** (per dataset 0.15–0.36). Part of this is genuine estimation difficulty and part is
  definitional: a continuous two-endmember abundance is not the same quantity as "the count of 2.5 m sub-pixels whose
  NDVI falls below 0.35", and the 0.35 threshold is not the 50% mixing point between the endmembers. A different
  endmember choice would move these numbers. **The oracle-fraction headline is unaffected by any of this** — that is
  precisely why the pre-registration asked for both arms.
* Oracle minus realistic fraction, same ranking: +0.379 (bilinear) and +0.386 (SR) IoU, better on 118/118 images. The
  oracle-fraction arm is an upper bound, never a product number.

**Fairness note on the `alloc_* vs blocky_10m` comparison.** Under the oracle fraction the allocation methods receive
information (the exact block fraction) that `blocky_10m` and `hard_threshold_v1_style` do not. The +0.07 IoU that
`alloc_pretrained_sr|oracle` shows over `blocky_10m` is therefore **not** evidence that allocation beats blocky in
production. The production-relevant comparison is the realistic one, and it is −0.31.

---

## Sensitivity and the second class

| test | delta IoU (all) | 95% CI | n | delta IoU (no NAIP) | 95% CI | n |
|---|---|---|---|---|---|---|
| NDVI < 0.25 | +0.0063 | [+0.0029, +0.0104] | 112 | +0.0010 | [−0.0021, +0.0040] | 52 |
| NDVI < 0.35 (primary) | +0.0081 | [+0.0035, +0.0141] | 118 | +0.0016 | [−0.0026, +0.0053] | 56 |
| NDVI < 0.45 | +0.0054 | [+0.0022, +0.0105] | 119 | +0.0009 | [−0.0011, +0.0028] | 57 |
| water, NDWI > 0 | +0.0146 | [−0.0479, +0.0664] | 44 | −0.0203 | [−0.0970, +0.0295] | 32 |

Moving the NDVI threshold does not change the story: small positive with NAIP, null without. **The water class is null
in both directions** — only 44 of 119 images contain any NDWI>0 pixel inside the valid area, absolute IoU is low
(0.43 oracle / 0.004 realistic — the unmixing fraction essentially fails for water), and the CI is wide. No claim is
made about water.

---

## What was BLOCKED or skipped, and why

* **`alloc_finetuned_sr` — BLOCKED.** F4's fine-tune has not landed. Pretrained output was never used under this label
  and no fine-tuned number appears anywhere in `f5.json`.
* **Heavy SR (Colab upper reference) — NOT RUN.** Colab is human-operated; out of scope for this sandbox. Not
  substituted, not estimated.
* **`venus` — excluded.** 5 m HR reference; a 2.5 m truth mask cannot be defined on it. Stated in `f5.json`
  under `data.datasets_excluded`.
* Nothing was subsampled: the real pretrained SR forward pass ran on **all 119 images** of the four usable datasets.

## Limitations

1. Static land-cover boundaries are a proxy for change boundaries; this is not a landslide label.
2. HRharm is radiometrically harmonized to Sentinel-2 by the dataset authors, which favours LR-consistent estimators.
   Every method is scored against the same reference so the comparison is fair, but absolute IoU is not a
   field-validated number.
3. SEN2SR-lite was trained on SEN2NAIPv2 (NAIP); train/test overlap with the OpenSR-test NAIP set cannot be excluded.
   That is why every headline is reported with and without NAIP — and why the with-NAIP number should not be quoted
   alone.
4. `spain_crops` and `spain_urban` pass A2's geometry check on only 0.50 / 0.55 of images within 1 HR px. Sub-pixel
   misalignment penalises every method equally, but it inflates boundary error for all of them.
5. NAIP LR is 121 px and is reflect-padded to the model's 128 px native input, then cropped (A2's evaluation-only
   adapter; production ADR-001 forbids padding). The padded border lies inside the 16 HR px excluded border.
6. The realistic fraction uses one unmixing estimator with percentile endmembers; a different endmember choice would
   move every realistic-fraction number.
7. Images sharing a Sentinel-2 acquisition are not independent (119 images, 72 acquisitions). The bootstrap resamples
   images, so the CIs are mildly optimistic.

## Bottom line

* SR-informed sub-pixel ranking **does** carry information beyond a bilinear upsample, but the margin is ~1% of IoU,
  and pooled over the three datasets without a leakage concern it is not distinguishable from zero at n=56.
* The one effect that survives removing NAIP is that the advantage is **concentrated in features narrower than 20 m**
  (+0.0077 IoU, CI [+0.0018, +0.0132], n=47).
* None of this transfers to a deployed product as it stands: once the block fraction has to be estimated from
  Sentinel-2 rather than read off the truth, **allocation of any kind is 0.31 IoU worse than not allocating**. The
  bottleneck is the fraction, not the ranking.
