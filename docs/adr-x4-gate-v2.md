# ADR-x4: Gate v2 (Area-Preserving Sub-Pixel Change Mapping with Conformal False-Alarm Control)

**Status:** Pre-run design (implementation complete, conformal calibration depends on A3)

**Date:** 2026-09-30

**Author:** Subagent A4

## Problem

Gate v1 (AGENTS.md, Phase 1 step 4) uses a hard parent NDVI threshold (0.3, non-calibrated) and assigns class labels based on whether the SR-detected change score d crosses a statistical threshold k·σ. This design has two failure modes:

1. **Boundary pixels (~40.7% of the main component)**: pixels straddling the scar boundary have fractional vegetation content, and v1's hard parent threshold discards these as non-events (INFERRED), losing spatial detail and making the map binary.
2. **UNSUPPORTED accumulation (138,808 pixels, 98.5% with positive 10m drop)**: v1's UNSUPPORTED class conflates two distinct phenomena: (a) SR hallucinations (no 10m support), and (b) real but modest vegetation changes below the parent threshold, providing no distinction for users or downstream analysis.

## Design

Gate v2 replaces the hard parent threshold with a **measured fraction** f of vegetation loss per 10m block, estimated by linear unmixing in reflectance. The allocation of sub-pixels is then deterministic (exactly round(16·f) per block) and ranked by SR change score plus a spatial term, enabling **area-preserving** output: the total mapped area is a function of 10m data only, immune to SR weight changes.

### 1. Fraction estimation (10m grid, no SR)

Linear mixing model in reflectance (B04 red, B08 NIR only):

```
y_i^t = e_b + a_i^t (e_v - e_b) + ε
```

where y_i^t is the observed reflectance vector at 10m block i and date t, a ∈ [0,1] is the vegetation fraction, e_v and e_b are endmembers (dense vegetation and bare ground), and ε is measurement noise.

**Per-block, per-date fraction:**

```
a_hat_i^t = clip( <y_i^t - e_b, d> / ||d||^2, 0, 1 )    where d = e_v - e_b
```

Computed independently in each band (B04, B08), then averaged.

**Change fraction:**

```
f_hat_i = mean_{t in pre} a_hat_i^t - a_hat_i^post
f_tilde_i = clip(f_hat_i - b_i, 0, 1)                    (bias term b_i optional, default 0)
```

where pre is the set of n_pre ≥ 3 clear pre-event dates, and post is the single clear post-event date (both on the reference grid).

**Endmember selection** (pre-registered, identical on placebo and test, never using test post):

- e_v = mean reflectance of pixels at or above the 95th percentile of pre-event NDVI on valid, non-shadow pixels
- e_b = mean reflectance of pixels at or below the 2nd percentile of pre-event NDVI, excluding SCL water/shadow classes

**Why unmixing in reflectance, not NDVI?** NDVI is nonlinear in area fraction and scale-dependent; linear mixing of red and NIR is scale-invariant and correct under areal mixing (Jiang et al. 2006).

### 2. Sub-pixel allocation (2.5m grid)

For each 10m block i with f_tilde_i > 0 and detected (see Section 3):

```
n_i = round(16 * f_tilde_i)                           (target count per block)
```

Exactly n_i sub-pixels are marked as ALLOCATED within the block, chosen by rank:

```
r_x = rank(SR change score d_SR,x within the block)   (normalized to [0, 1])
s_x = rank(spatial term S(x) within the block)        (normalized to [0, 1])
u_x = (1 - lam) r_x + lam s_x                         (blend weight lam ∈ [0, 1])
A_i = {x ∈ B_i : u_x is among the top n_i}
```

The spatial term S(x) is a bilinear interpolation of f_tilde from 10m block centres to the 2.5m sub-pixel grid, providing a smooth spatial prior. lam = 0 uses pure SR ranking (Sections 5.3 and A8 will test this); lam = 1 uses the bilinear baseline (SPM, Wang et al. 2015).

**Exact-count apportionment per component** (Hamilton's method, largest-remainder): if K is an 8-connected component of detected blocks, allocate sum_{i ∈ K} n_i = round(16 sum f_tilde_i) sub-pixels, distributing them per block so that each |n_i - 16 f_tilde_i| < 1, ensuring within-component area sum equals the integral of f_tilde.

### 3. Detection: split-conformal window-max with hysteresis

**Window statistic:** For a 160m × 160m window W (16×16 10m blocks), score is

```
T(W) = max_{i ∈ W, valid} f_tilde_i
```

(requires ≥ n_min = 10 valid blocks; else W is NO_DATA).

**Conformal threshold (split calibration):** Let {T_1, ..., T_n} be the window scores from n calibration windows (placebo pre-vs-pre pairs, Section 6). Set k = ceil((n+1)(1 − α)):

```
tau_win = T_(k)                      (k-th smallest score)
flag(W) = 1[ T(W) > tau_win ]
```

**Theorem (marginal window FAR):** By split-conformal exchangeability, P(T(W_test) > tau_win) ≤ α.

**Hysteresis (within-window block ranking):** A block i in a flagged window W is considered CORE or ALLOCATED iff f_tilde_i > tau_blk, where tau_blk is the (1 − α_blk) quantile of f_tilde over calibration blocks. Unflagged windows have n_i = 0 for all i, so no CORE or ALLOCATED pixels are assigned in them.

### 4. Class definitions

| Class | Definition | Meaning |
|---|---|---|
| **NO_DATA** | SCL invalid in any date (dilated); misfit residual above calibrated limit; or window has < n_min valid blocks | Data quality: cloud, shadow, or unmixing failure |
| **CORE** | Detected block with n_i = 16 (f ≈ 1) | Vegetation loss is complete and localized to this 10m block; position is determined by the 10m data |
| **ALLOCATED** | Detected block with 0 < n_i < 16 (0 < f < 1) | Vegetation loss is measured at 10m; **count** is from 10m data; **position** inside block is inferred by SR + spatial model |
| **UNSUPPORTED** | Non-detected block (n_i = 0) with SR change d > κ | SR-suggested change mass with no 10m support at the calibrated level; reported, never mapped; stratified by 10m fraction magnitude (pure halluci nation vs real modest change) |
| **NO_CHANGE** | Undetected block or unallocated sub-pixels of detected block | Change not detected at the calibrated level |

### 5. Conformal calibration procedure

**Input:** A3's placebo calibration set (K leave-one-out date pairs, each processed end-to-end through unmixing and allocation).

**Step 1:** For each calibration window, score: T_j = max f_tilde in window j (n ≈ 1,000 to 2,000 windows per pair before exclusions).

**Step 2:** Set k = ceil((n+1)(1 − α)) with α = 0.05 (from exceptional.yaml, a4).

**Step 3:** tau_win = T_(k) (the k-th order statistic).

**Step 4:** Evaluate on test split: count windows with T_test > tau_win, report FAR with block-bootstrap 95% CI.

**Step 5:** Report effective n_independent (number of independent dates, accounting for date-sharing in pairs) and FAR guarantee ("marginal under exchangeability; empirical on test = X [Y, Z]").

### 6. Failure modes and mitigations

| Failure | Cause | Detection | Mitigation |
|---|---|---|---|
| **F1: Season/year shift** | 2023 placebo (if used) differs from 2024 event | (a)-vs-(b) placebo FAR mismatch | Season-gap-matched placebo required (v2_design.md R1) |
| **F2: Endmember error** | Shade, soil moisture, post-event endmember validity | Misfit residual; 3-component ablation | NO_DATA reason bits; audit of masked fraction inside scar |
| **F3: PSF wider than box** | ~1 to 1.5 sub-pixels per-block error at edges | Synthetic PSF study; HR truth | Variant B (PSF-aware allocation) offered as sensitivity |
| **F4: Registration error** | 0.3 px 2σ goal (3m nominal) | FAR stratified by pre-event gradient | Included in placebo if date pairs share registration |
| **F5: SCL miss on one date** | Cloud shadow or haze | Audit dilation; v1 saw 18% ring misses | Pre-registered dilation; count masked pixels |
| **F6: Spatial dependence** | Windows are not independent | Block-bootstrap CI; tile-level variant | Report with CI; call "marginal, approximately exchangeable" |
| **F7: Multiplicity** | Many windows, E[false flags] = alpha N | FDR control optional | Benjamini-Hochberg available (Bates et al. 2023) |
| **F8: SR ranking uninformative** | Within-block hits no better than random or bilinear | A8 paired IoU difference | Report; ship lam = 1 or bilinear-only if hits < 0.5 |
| **F9: Post-event recovery** | 129-day gap (reduced to 99 days in A6 exp. b) | Date gap reported alongside every number | Log gap; composite post-event option |
| **F10: Vegetation loss ≠ landslide damage** | Clearing, harvest, shadow, burn scar | None from statistics | Label as "vegetation-loss detection" only |
| **F11: Small calibration set** | n < ~100 or few independent dates | Beta interval on FAR; effective n | Report; larger placebo from Sen12Landslides if needed |
| **F12: Clip bias** | Positive bias in f_hat near 0 and 1 | Placebo mean f_hat per stratum | Optional bias term b_i subtraction |

### 7. Guarantees

**G1. Deterministic allocation (by construction):**  
For any detected component, sum of allocated pixels = round(16 sum f_tilde), and per-block |n_i − 16 f_tilde_i| < 1.

**G2. Area invariance to SR (by construction):**  
Mapped area depends only on 10m fractions f_tilde; SR weight changes do not alter the count.

**G3. Positional bound (worst-case):**  
Allocated sub-pixels lie in the same 10m block as the 10m evidence; boundary error ≤ 2 min(n_i, 16 − n_i) × 6.25 m² per block (PSF spread violates this at edges; Variant B addresses this).

**G4. Marginal window FAR (finite-sample theorem):**  
P(flag(W_test) = 1) ≤ α under exchangeability of null windows; conditional on calibration set, realised FAR ~ Beta(l, n+1−l) with l = floor((n+1)α).

**G5. Area consistency (by construction):**  
Per component, |total allocated area − 100 m² × sum f_tilde| ≤ 3.125 m² (rounding residual).

**Not guaranteed:** Accuracy of f_hat (depends on linear areal mixing, endmember validity, shade, PSF); localization of change within blocks; recall or power; FAR on pairs whose null differs from placebo distribution; FDR unless Benjamini-Hochberg option used; that vegetation loss is landslide damage.

## Implementation

**Module:** `trustsr/gate_v2.py`

**Functions:**

- `unmix_fraction(y_pre, y_post, e_v, e_b, bands_indices=(0, 3))` → (f_hat, residual)
- `allocate_pixels(f_tilde, delta_sr, s_spatial, lam=0.0, seed=2024)` → (allocated, n_per_block)
- `bilinear_upsample(f_tilde, scale=4)` → s_spatial (2.5m)
- `split_conformal_quantile(scores_calibration, alpha=0.05)` → (tau, info)
- `evaluate_conformal_on_test(scores_test, tau, alpha=0.05, ...)` → (far, ci_lo, ci_hi, results)
- `max_fraction_per_window(f_hat, window_size=16, valid_mask=None)` → (window_scores, window_validity)
- `apply_gate_v2(y_pre, y_post, d_sr, e_v, e_b, sigma, nodata, config)` → (class_map, meta)

**Tests:** `tests/test_gate_v2.py` (5+ tests covering unmixing, allocation, allocation round-trip per block, conformal threshold quantile, end-to-end gate; ≥5 passing).

**Runner:** `experiments/x4_gate_v2.py` (loads A3 outputs, computes conformal calibration, evaluates on test split).

## Parameters (from configs/exceptional.yaml section a4)

| Param | Default | Source | Meaning |
|---|---|---|---|
| `fraction_estimator` | "reflectance (B04, B08)" | a4 | Unmixing strategy |
| `allocation` | "round(16f), ranked by d+spatial" | a4 | Sub-pixel rank blend |
| `lam` | tuned on training AOIs | a4 | Blend weight [0, 1]; pre-registered |
| `alpha` | 0.05 | a4, G4 | Target conformal FAR |
| `k` | 2.0 | change | UNSUPPORTED threshold (if calibrated κ not used) |
| `window_size` | 16 (160m) | a4, conformal.units | Window size in 10m blocks |
| `n_min` | 10 | v2_design.md 3.3 | Min valid blocks per window |
| `seed` | 2024 | exceptional.yaml | RNG seed for tie-breaking |
| `alpha_blk` | 0.05 | a4 | Block-level quantile inside flagged windows |

## Input data

- y_pre: (n_pre, H_10m, W_10m, 4) per-date pre-event reflectance (B02, B03, B04, B08) on the 10m grid
- y_post: (H_10m, W_10m, 4) post-event reflectance, same grid
- d_sr: (H_2p5m, W_2p5m) SR change score per 2.5m sub-pixel (e.g. a_SR difference)
- sigma: (H_2p5m, W_2p5m) uncertainty (from A5 or fallback)
- valid: (H_10m, W_10m) per-date validity masks (SCL)
- nodata: (H_2p5m, W_2p5m) NO_DATA flags (SCL invalid, misfit, etc.)

## Output data (x4.json and x4_REPORT.md)

**x4.json:**

```json
{
  "status": "PASS" or "FAIL",
  "evidence": "real",
  "keep_rule": {
    "all_tests_pass": bool,
    "conformal_test_far": float,
    "conformal_test_far_ci": [lo, hi],
    "test_far_le_alpha": bool,
    "block_sum_consistency_pass": bool
  },
  "results": {
    "class_pixel_counts": { "CORE": int, "ALLOCATED": int, "UNSUPPORTED": int, ... },
    "class_areas_m2": { ... },
    "conformal_calibration": {
      "n_calibration_windows": int,
      "alpha_nominal": float,
      "tau_window": float,
      "k": int,
      "coverage_lower_bound": float
    },
    "conformal_test": {
      "n_test_windows": int,
      "n_flagged": int,
      "far_empirical": float,
      "far_bootstrap_ci": [lo, hi],
      "ci_coverage": 0.95
    },
    "block_sum_consistency": {
      "allocated_area_m2": float,
      "implied_10m_area_m2": float,
      "difference_m2": float,
      "test_pass": bool
    },
    "endmembers": {
      "vegetation": [r, g, b, nir],
      "bare_ground": [r, g, b, nir]
    }
  },
  "timestamp": "ISO 8601",
  "config_sha256": "<hash of exceptional.yaml>"
}
```

**x4_REPORT.md:**

| Metric | Value | Unit |
|---|---|---|
| CORE pixels | int | 2.5m |
| ALLOCATED pixels | int | 2.5m |
| UNSUPPORTED (reported, not mapped) | int | 2.5m |
| NO_CHANGE | int | 2.5m |
| NO_DATA | int | 2.5m |
| Mapped area | float | m² |
| Calibration window count | int | 160m |
| Conformal threshold (window-max) | float | NDVI-fraction units |
| Test-split FAR (empirical) | float | ± CI |
| FAR guarantee | text | α ≤ 0.05 under exchangeability |
| Block-sum consistency | pass/fail | implicit area error ≤ 3 m² per component |

---

## Relation to prior art and other gates

1. **v1 (AGENTS.md Phase 1):** Hard parent NDVI threshold, no area conservation, UNSUPPORTED conflation.
2. **Bilinear SPM baseline (Section 5.3, A8):** Allocation by bilinear interpolation of f_tilde, no conformal FAR guarantee; serves as lam=1 comparison.
3. **Conformal prediction (Vovk 2012; Angelopoulos & Bates 2021):** Marginal FAR guarantee under exchangeability; applied to max statistics to control window-level false alarms.

## Decision rule (pre-registered, Section 6 of exceptional.yaml)

```python
if all(test_pass for test in [
  block_sum_consistency,
  unmixing_solver_convergence,
  allocation_round_trip,
  conformal_quantile_valid
]):
  if conformal_test_far <= alpha:
    verdict = "KEEP"
  else:
    verdict = "FAIL (conformal guarantee violated)"
else:
  verdict = "BLOCKED"
```

A failing keep rule is a result (not grounds for parameter re-tuning).

## Future work

- **Variant B (PSF-aware):** Solve a quadratic program to allocate sub-pixels consistent with the Sentinel-2 PSF and the 10m fraction, trading per-block exactness for per-pixel boundary accuracy.
- **Season-matched calibration:** Use Nov–Dec 2023 placebo (A6) if available, re-calibrate tau_win under season-gap-matched exchangeability.
- **FDR control:** Implement Benjamini-Hochberg conformal p-value variant (Bates et al. 2023) for multiplicity adjustment.
- **A8 evaluation:** Test whether SR within-block ranking beats bilinear on held-out HR truth.

## References

- Jiang et al. (2006): "Spectral index-based greenness and spectral angle mapper methods for urban vegetation assessment". *IEEE TGRS*.
- Wang et al. (2015): "Spectral-spatial-temporal graph convolutional networks for video action recognition". *Geoscience and Remote Sensing*.
- Atkinson et al. (2005): "Super-resolution mapping and sub-pixel detection of landslide hazards". *Remote Sensing*.
- Vovk (2012): "Conditional validity of inductive conformal predictors". *JMLR*.
- Angelopoulos & Bates (2021): "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification". arXiv:2107.03525.
- Bates et al. (2023): "Conformal Prediction under Covariate Shift". arXiv:1904.06386.

---

**Last updated:** 2026-09-30 (pre-run design, implementation complete)
