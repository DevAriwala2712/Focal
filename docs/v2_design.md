# TrustSR v2 design: consistency-constrained, area-preserving sub-pixel change mapping with conformal false-alarm control

Agent A1, 2026-09-30. Design document, not a result. **No number in this file comes from a run of the v2 method; there is none yet.** Numbers come from (a) cited sources, (b) the repository's existing v1 evidence, or (c) the small synthetic calculations in Section 9, each labelled. Evidence tags as in [prior_art_teardown.md](prior_art_teardown.md): `[P]` primary read, `[S]` secondary, `[D]` derived here, `[U]` unverified.

## 0. Verdict on the starting hypothesis: REVISE (keep the architecture, change six parts, reject two claims)

Hypothesis under test: *consistency-constrained, area-preserving sub-pixel change mapping with conformal false-alarm control* (two-endmember unmixing in reflectance to a per-10 m-block change fraction f; allocate exactly round(16 f) of the 16 sub-pixels per block, ranked by an SR change score plus a spatial-dependence term; calibrate the block-detection threshold by split-conformal on placebo pre-vs-pre pairs so false-alarm rate <= alpha; classes CORE / ALLOCATED / UNSUPPORTED / NO_DATA).

| Part of the hypothesis | Verdict | Evidence |
|---|---|---|
| Unmix in reflectance, not NDVI | **Confirmed** | NDVI is nonlinear in area fraction and scale-dependent over partial cover (Jiang et al. 2006 `[P]`); linear mixing of red and NIR is scale-invariant. Illustration `[D]`: with assumed endmembers a 10 m NDVI drop of 0.30 is a bare fraction of about 0.47, so v1's parent rule discards mixed pixels below that. Matches the v1 diagnosis (40.7 % boundary pixels) in direction; the endmembers are assumed |
| Area-preserving allocation | **Confirmed as a design property, revised as a physical claim** | Preservation of counts by construction has precedent (pixel swapping, Atkinson 2005 `[P]` abstract). But S2's MTF requirement (0.15 to 0.3 at Nyquist, Gascon 2014 `[P]`) vs a box's 0.637 `[D]` means a pixel's value is not the mean of its own 10 m cell; a straight-edge calculation `[D]` gives a 0.06 to 0.10 fraction error (about 1 to 1.5 sub-pixels) in boundary blocks if a box is assumed. So "block sum equals true block area" is an **invariant of the map**, not a fact about the ground. Add a PSF-aware allocator (Section 4.3) |
| SR score ranks sub-pixels inside a block | **Kept only as a testable hypothesis** | Independent evidence gives no reason to expect SR to add information: TU Wien found the best interpolator (nearest neighbour, F1 0.485) ahead of every SR product (best SEN2SR-RGBN 0.478; SEN2SR-lite 0.468; bicubic 0.477) on building delineation `[P]` (different task, no CIs seen). Derived break-even (Section 5.3): allocation beats a hard-threshold blocky map in IoU only if within-block hit rate is at least 1/2 (for blocks with n <= 8 true sub-pixels), i.e. at least twice random for n=4. The bilinear-fraction ranking is the published SPM baseline (Wang, Atkinson, Shi 2015 `[P]`); A8's keep rule tests SR against it |
| Split-conformal calibration gives FAR <= alpha | **Confirmed with conditions; revised** | Finite-sample marginal validity under exchangeability, and Beta-distributed realised coverage given the calibration set (Angelopoulos and Bates `[P]`). **Revised:** (i) the placebo must be season-gap-matched (Section 6.3); an ordinary same-season pre-vs-pre placebo does **not** carry the guarantee to a January-to-December pair; (ii) calibrate a window-max statistic with hysteresis rather than a bare block threshold, because a block-level 5 % rate gives about 100 % window-level false alarms `[D]`; (iii) FAR is not FDR; offer conformal-p-value BH (Bates et al. 2023 `[P]`) as an option |
| Spatial-dependence term | **Kept, simplified** | Rank blend with a bilinear fraction surface; optional count-preserving pixel swap |
| Classes CORE / ALLOCATED / UNSUPPORTED / NO_DATA | **Kept, semantics tightened** | Section 3.5; UNSUPPORTED is redefined to avoid the v1 misreading (98.5 % of v1's UNSUPPORTED had a positive 10 m drop, EVIDENCE.md) |

**Rejected:** (R1) "pre-vs-pre placebo, alpha guaranteed" without season matching and exchangeability stated. (R2) Any statement that v2 localises change better than v1 or bilinear before A8 has run. **Also rejected as too strong:** "every allocated sub-pixel is measured": counts are measured at 10 m, positions are model-inferred.

**Where v2 can honestly claim to beat what exists.** Not on radiometric fidelity or resolution accuracy (unverified, Section 10). On two properties no candidate in the teardown offers, both *by construction or by finite-sample theorem*: (1) the mapped area is a deterministic function of 10 m data, independent of any SR weights, so SR hallucination cannot change how much change is reported; (2) a stated false-alarm rate on placebo-like no-event windows. Those two claims survive even if SR adds zero information.

## 1. Problem restated with the failure it fixes

v1 (`experiments/wayanad_evidence`): S = d > k sigma at 2.5 m, P = 10 m NDVI drop > 0.3, OBSERVED = S and P, INFERRED = P and not S, UNSUPPORTED = S and not P. Verified consequences (EVIDENCE.md, exceptional.yaml): OBSERVED union INFERRED equals P on valid pixels, so 2.5 m output adds nothing to the binary map; INFERRED is 515 of 117,584 parent px (0.44 %); sigma is 35 to 40 % too small (median sigma_pre 0.0326, sigma_post 0.0022); UNSUPPORTED is mostly real modest change; no HR validation; no fine-tuning; season-mismatched dates (Jan vs 2024-12-06, 129 days after the event).

v2 replaces the two thresholded NDVI maps by (a) a **measured fraction** at 10 m, (b) an **exact-count allocation** to 2.5 m, (c) an **empirical, placebo-calibrated** detection threshold in place of k sigma.

## 2. Notation

- Blocks i in I on the 10 m grid; sub-pixels x in B_i on the exactly aligned 2.5 m grid, |B_i| = s^2 = 16 (s = 4).
- Dates: pre set P = {p_1..p_m} (m >= 3), post date q. All on the reference grid; y_i^t in R^4 the reflectance vector (B04, B03, B02, B08); v_i^t in {0,1} validity from SCL (SCL is a mask only, never a model input).
- Endmembers e_v (dense vegetation) and e_b (bare ground) in R^4 (pre-registered primary uses only (B04, B08), two components); d = e_v - e_b.
- SR: for each date t the 8-run dihedral mean reflectance ybar_x^t (2.5 m) and per-pixel run standard deviation, from the existing trust module.

## 3. Method

### 3.1 Fractions (10 m, no SR)

Linear mixing model: y = e_b + a d + eps, a in [0,1].

```
a_hat_i^t = clip( <y_i^t - e_b, d> / ||d||^2 , 0, 1 )
r_i^t     = || (I - d d^T/||d||^2) (y_i^t - e_b) ||          # misfit residual
f_hat_i   = mean_{p in P} a_hat_i^p  -  a_hat_i^q            # vegetation-loss fraction
f_tilde_i = clip( f_hat_i - b_i , 0, 1 )                     # b_i: placebo null bias per stratum (Sec. 6.4), 0 if not used
```

Properties `[D]`: unbiased and linear in the true area fraction if the mixture is areal and linear and endmembers are correct; OLS variance sigma_y^2/||d||^2 per date; averaging over m pre dates reduces the pre-side variance to sigma_a^2/m only if errors are independent across dates (they are not: shared registration and sensor effects). We do **not** rely on a parametric noise model for decisions; the placebo does that job. With two bands and one parameter, the squared residual has one degree of freedom; with four bands, three.

Endmember rule (pre-registered, applied identically to placebo and test pairs, never using the test post image): e_v = mean reflectance vector of pixels at or above the 95th percentile of pre-event NDVI on valid, non-shadow pixels; e_b = mean vector of pixels at or below the 2nd percentile of pre-event NDVI excluding SCL water/shadow classes. **Caveat:** e_b from pre-event imagery may not represent fresh landslide debris (exposed regolith, wet soil); this is a documented ablation (e_b from post scar core pixels of the same scene vs pre-only), reported, and any variant that reads post-event pixels is flagged as non-calibratable because the placebo cannot replicate it.

### 3.2 SR change score (2.5 m)

Apply the same projection to the SR reflectance, per date, then difference:

```
a_SR,x^t = clip( <ybar_x^t - e_b, d>/||d||^2, 0, 1 )
Delta_x  = mean_{p in P} a_SR,x^p - a_SR,x^q
Delta'_x = Delta_x - mean_{y in B_i} Delta_y                  # within-block deviation: only relative ranking is used
```

The SR score never sets *how much* change is in a block, only *which* sub-pixels of the block carry it. (In v1 NDVI was computed per run then averaged, per AGENTS.md; here the projection is linear in reflectance, so mean-of-projections equals projection-of-mean except for the clip. Keep the per-run computation to stay consistent and to retain run-to-run dispersion.)

### 3.3 Detection: conformal window-max with hysteresis

Unit: window W = 16 x 16 blocks (160 m), the pre-registered `far_window`. Statistic:

```
T(W) = max_{i in W, valid} f_hat_i          (requires >= n_min valid blocks in W; else W is NO_DATA)
```

Calibration set: placebo windows (Section 6). Let T_1..T_n be their statistics. With k = ceil((n+1)(1 - alpha)):

```
tau_win = T_(k)          (k-th smallest);  if k > n then tau_win = +inf
flag(W) = 1[ T(W) > tau_win ]
```

**Theorem (marginal window FAR)** `[P]` (split conformal; Vovk; Angelopoulos and Bates): if T_1..T_n and T(W_test) are exchangeable, then P(T(W_test) > tau_win) <= alpha. Proof sketch `[D]`: by exchangeability the rank of T_test among the n+1 values is uniform on ties-broken ranks, so P(T_test > T_(k)) <= (n+1-k)/(n+1) <= alpha. Given the calibration set the realised coverage (1 - FAR) is Beta(n+1-l, l)-distributed (FAR ~ Beta(l, n+1-l)) with l = floor((n+1) alpha) (marginal over that set equals the guarantee).

Hysteresis for mapping: a block i in a flagged window is **detected** iff f_hat_i > tau_blk, where tau_blk is the (1 - alpha_blk) quantile of block-level f_hat over the calibration placebo blocks (pre-registered alpha_blk). Blocks in unflagged windows are never detected, whatever their f_hat. Hence the window-level guarantee is unaffected by the choice of tau_blk, which only trades precision and recall inside flagged windows.

Why a window statistic: if block nulls were independent with exceedance probability p, window FAR = 1 - (1-p)^256 `[D]`; p = 0.05 gives about 1 (1 - 0.95^256 = 1 - 2e-6), and 0.05 needs p about 2e-4. A block-level threshold at the usual 5 % is therefore useless as a **window** false-alarm control. Conformal on the max avoids assuming independence.

Optional FDR mode `[P]` (Bates et al. 2023): p_j = (1 + #{i: T_i >= T(W_j)}) / (n+1), then Benjamini-Hochberg over test windows; controls FDR under the paper's conditions (exchangeable null, and positive dependence proved for conformal p-values conditional on the calibration set). Off by default; reported as a sensitivity.

### 3.4 Exact-count allocation

For detected blocks: n_i must equal round(16 f_tilde_i) *and* component totals must be consistent. Use largest-remainder apportionment (Hamilton) per 8-connected component K of detected blocks:

```
N_K   = round( 16 * sum_{i in K} f_tilde_i )
n_i   = floor(16 f_tilde_i) + 1[ i is among the (N_K - sum floor) blocks with the largest fractional parts ]   (ties: raster order)
```

`[D]` Guarantees: sum_{i in K} n_i = N_K exactly; each n_i in {floor(16 f_tilde_i), ceil(16 f_tilde_i)} so 0 <= n_i <= 16 and |n_i - 16 f_tilde_i| < 1; and |6.25 m^2 * N_K - 100 m^2 * sum f_tilde_i| <= 3.125 m^2 per component. (floor is superadditive, so the remainder R = N_K - sum floor is >= 0, and R <= |K| because each fractional part < 1, so no block gets more than +1.)

Ranking inside each detected block, using a rank blend:

```
r_x   = within-block rank of Delta'_x, scaled to [0,1]            (SR evidence)
S(x)  = bilinear interpolation of f_tilde from block centres to the 2.5 m grid   (spatial dependence)
s_x   = within-block rank of S(x), scaled to [0,1]
u_x   = (1 - lam) r_x + lam s_x ,    lam in [0,1] chosen on training AOIs only
A_i   = the n_i sub-pixels of B_i with largest u_x    (ties: fixed raster order)
```

lam = 0 is pure SR ranking; lam = 1 is the interpolation baseline (bilinear SPM, Wang, Atkinson and Shi 2015 `[P]` list bilinear among SPM algorithms; the count-preserving top-n version here is our composition of that idea, not their exact algorithm). Optional refinement: count-preserving pixel swapping (Atkinson 2005 `[P]`) with a fixed, deterministic iteration budget.

Deterministic: fixed tie order, no random numbers, so outputs are byte-reproducible (fits E3 hash reruns).

### 3.4b PSF-aware variant (B)

Variant A above uses the box model: the map's block sum equals 16 f_tilde. Variant B replaces exact block sums by consistency with the blurred observation: let H be the 2.5 m-grid operator "PSF-weighted average sampled at block centres" (Gaussian with sigma from an MTF assumption, e.g. MTF(Nyquist) in {0.15, 0.20, 0.30}); choose the binary map c minimising

```
|| H c - f_tilde ||^2  -  mu * sum_x u_x c_x      subject to   sum_x c_x = 16 * sum_i f_tilde_i   (global count)
```

by greedy add and swap with a fixed budget. Variant B keeps global area conservation (a partition of unity: each sub-pixel's PSF weights over all blocks sum to 1 `[D]`) but drops the per-block exact count. It is evaluated against variant A on the synthetic PSF study and on HR truth; the pre-registered choice rule is in the build prompt. **A is the shipped default** because its guarantees are testable exactly.

### 3.5 Classes

| Class | Definition | Meaning |
|---|---|---|
| NO_DATA | any selected date invalid by SCL in the block (with a pre-registered dilation), or misfit r_i above the placebo-calibrated limit, or window with fewer than n_min valid blocks | reason bits are stored (SCL / misfit / coverage) so misfit removal inside the slide can be audited |
| CORE | sub-pixels of detected blocks with n_i = 16 | change area fully determined by 10 m data; no positional choice was made |
| ALLOCATED | sub-pixels chosen in detected blocks with 0 < n_i < 16 | **count** is measured at 10 m; **position** inside the block is inferred by SR/spatial model |
| UNSUPPORTED | sub-pixels in non-detected blocks (n_i = 0) with a high SR score: Delta_x above kappa, where kappa is the (1 - alpha_u) quantile of Delta over placebo sub-pixels | SR-suggested change with no 10 m support at the calibrated level: reported and counted, never mapped. Stratify by the 10 m fraction f_hat (0 to noise floor vs sub-threshold positive) so that "no support" is not confused with "modest real change" as in v1 |
| NO_CHANGE | everything else, including unselected sub-pixels of detected mixed blocks | not a class the guarantee speaks about |

Naming continuity: v1 OBSERVED roughly corresponds to CORE plus the measured share of ALLOCATED; v1 INFERRED roughly to ALLOCATED. The identity "OBSERVED union INFERRED equals P" is replaced by "mapped area equals 6.25 m^2 times the sum of n_i".

## 4. Guarantees, with assumptions

| # | Statement | Type | Assumptions |
|---|---|---|---|
| G1 | For every detected block, the number of allocated sub-pixels equals n_i; per component, sum n_i = round(16 sum f_tilde_i); mapped area equals 6.25 m^2 times sum n_i | Deterministic, by construction (variant A) | The code is correct; f_tilde is the input. Tested exactly on CPU |
| G2 | Mapped area is a function of 10 m data only, invariant to any change in SR weights or SR score | Deterministic | SR affects only within-block order |
| G3 | Any allocated sub-pixel lies in the same 10 m block as its 10 m evidence, so positional error is bounded by the block; per block the symmetric-difference area is at most 2 min(n_i, 16 - n_i) * 6.25 m^2 | Deterministic bound, worst case | Truth is inside the same block as the estimate's count; **PSF spread violates this at edges** (Section 9.2) |
| G4 | P(flag(W)) <= alpha for a no-event window W exchangeable with the calibration placebo windows; conditional on the calibration set the realised coverage is Beta(n+1-l, l), i.e. FAR ~ Beta(l, n+1-l) | Finite-sample theorem | Exchangeability of null windows between calibration and test; identical processing; no leakage of test information into endmembers or thresholds |
| G5 | Mapped area is within 3.125 m^2 per component of 100 m^2 * sum of f_tilde over detected blocks | Deterministic | Area is of **detected** blocks: undetected small fractions are excluded by design |

**Not guaranteed:** the accuracy of f_hat (depends on linear areal mixing, endmember validity, shade, PSF); where inside a block the change is; recall or power (report the power curve vs true fraction from placebo-injection or labelled events); FAR on any pair whose null distribution differs from the placebo's (season, year, sensor state); FDR unless the BH option is used and its conditions hold; that vegetation loss is landslide damage (it is not: harvest, clearing, shadow and burn scars also lower NDVI or vegetation fraction; no building-damage claim).

## 5. Analysis

### 5.1 Why SR cannot change the reported area (G2) and what SR can still hurt

SR moves counts nowhere but can misplace them inside a block. Worst case per block (G3): 2 min(n, 16 - n) sub-pixels wrong, up to 16 sub-pixels for n = 8, versus 8 for a hard-threshold blocky map at n = 8. So SR can make position worse than blocky in the worst case; whether it does on average is an empirical question (A8).

### 5.2 Random and oracle reference points `[D]`

For a block with n true sub-pixels, a random allocation of n sub-pixels has expected overlap n^2/16, so expected symmetric difference is 2(n - n^2/16). Blocky with threshold 0.5 (using the true fraction) has symmetric difference n for n < 8 and 16 - n for n > 8. Random allocation is therefore never better than blocky in symmetric difference (equal at n = 8). Numbers (n: blocky, random): 2: 2, 3.5; 4: 4, 6.0; 8: 8, 8.0; 12: 4, 6.0; 14: 2, 3.5. The oracle allocation has 0.

### 5.3 Break-even hit rate `[D]`

With overlap o (correct sub-pixels among the n allocated), symmetric difference is 2(n - o). Allocation beats blocky iff o >= n/2 for n <= 8, and o >= (3n - 16)/2 for n > 8. In hit-rate terms h = o/n: h >= 0.5 for n <= 8 (random has h = n/16: 0.125 at n = 2, 0.25 at n = 4, 0.5 at n = 8); h >= (3n-16)/(2n) for n > 8 (n = 12 needs 0.833; random has 0.75; n = 14 needs 0.929; random 0.875).

Consequence: **area-preserving allocation is chiefly an area and calibration device; it improves IoU only when within-block ranking is good.** The product therefore ships the 10 m fraction raster f_tilde as a first-class output beside the 2.5 m allocation, and reports IoU and area error separately, never a single "resolution gain".

### 5.4 Placebo-window arithmetic `[D]`

Wayanad AOI (10,240 m square): 64 x 64 = 4,096 windows per pair; the pre-registered split is a checkerboard of 128 px (1,280 m) tiles, giving about 2,048 calibration and 2,048 test windows per pair before exclusions (slide disk and footprint). Angelopoulos and Bates suggest about 1000 calibration points for stable coverage `[P]`, so one pair is enough in count but **not in independence**: windows within a tile are spatially correlated and pairs sharing a date are dependent. Report the effective number of independent dates (A3 honesty rule), and treat any FAR estimate as an estimate with a block-bootstrap interval, not as a proof.

## 6. Placebo design and the exchangeability assumption

### 6.1 Definition

A placebo pair is a pair of clear dates (a, b), no known event between them, processed by exactly the same pipeline as the test pair, including endmember estimation, masks and unmixing. Every flag it produces is a false alarm by construction (real non-event changes such as clearing are counted as false alarms, which makes the calibration conservative). Exclude the published-slide 1500 m disk and the footprint (exceptional.yaml a3).

### 6.2 Two placebo families

- **(a) Same-season:** leave-one-out among clear Jan to May 2024 dates (a3 spec). Captures sensor, atmosphere, registration noise. Does not contain the phenological offset between the January pre pool and the December post image.
- **(b) Season-gap-matched:** the same calendar pair one year earlier, e.g. January 2023 dates vs a November/December 2023 date (A6 exp. a fetches Nov-Dec 2023). Its null contains the January-to-December offset.

### 6.3 Requirement (revision R1)

Use (b) for the Jan-vs-Dec 2024 event pair. Use (a) only to characterise the noise floor. Report FAR of (a)-calibrated thresholds evaluated on (b) test windows: if it exceeds alpha, that **measures** the season effect. Whether 2023 is exchangeable with 2024 (monsoon strength differed; the event year was extreme) is `[U]`; state it as an assumption, and if 2023 data are not available the event pair carries **no** stated FAR guarantee, only the same-season guarantee.

### 6.4 Additional calibration details

- Null bias: E[max(f_hat, 0)] > 0 under null noise; estimate b_i as the placebo mean of f_hat per stratum and subtract (optional, pre-registered; effect on area is reported).
- Strata (Mondrian conformal, Vovk `[S]` via Angelopoulos and Bates): pre-event NDVI terciles (four bands only; DEM excluded by the project rule). Use stratified thresholds only where each stratum has at least 100 placebo windows (about 19 is the algebraic minimum for alpha = 0.05, `[D]`).
- Tile-level sensitivity: recalibrate on the max over a 128 px tile (64 windows) so that FAR per tile <= alpha, which implies per-window <= alpha `[D]` but is much more conservative; report both, keep the pre-registered window-level rule as primary.
- Any tunable step (endmember choice, lam, tau_blk, n_min, dilation) is fixed on the calibration/training split before the test split is scored (never_tune_on_test_split in exceptional.yaml).

## 7. Failure modes

| # | Failure | Effect | Detected by | Mitigation |
|---|---|---|---|---|
| F1 | Season/year shift between placebo and event pair | FAR > alpha on the event pair | (a)-vs-(b) placebo comparison | Season-gap-matched placebo (R1); otherwise no FAR claim |
| F2 | Endmember error, topographic shade (Wayanad slopes; L2A has no topographic correction) | Spurious f in shaded slopes; wrong f in scar | Misfit residual; 3-component ablation with a shade endmember; edge and slope-agnostic stratification by pre-event brightness | Misfit to NO_DATA with reason bits and audit of masked share inside slide |
| F3 | PSF wider than box | Halo of allocated pixels outside the true edge; block-level area error about 1 to 1.5 sub-pixels at edges `[D]`; totals conserved | Synthetic PSF study; HR truth | Variant B; report both |
| F4 | Registration error (goal 0.3 px 2 sigma = 3 m `[P]`) | False change along strong edges (water, roads, ridgelines) | FAR stratified by pre-event gradient magnitude | Included in placebo if placebo pairs share the same registration statistics; otherwise stratify |
| F5 | SCL misses cloud shadow or haze on one date | False f, invisible to placebo if not present there | v1 saw 18 % of the 30 m ring SCL-dark on the post date (EVIDENCE.md) | Pre-registered dilation; audit |
| F6 | Spatial dependence between windows | Realised FAR differs from alpha | Block bootstrap of FAR; tile-level variant | Report with CI; claim "marginal, approximately exchangeable" (Mao et al. 2024 `[P]` supports approximate local exchangeability under infill asymptotics only) |
| F7 | Multiplicity | Many windows, E[false flags] = alpha * N_null; FDR not controlled | FDR reported | BH option |
| F8 | SR ranking uninformative | ALLOCATED positions no better than bilinear | A8 paired IoU difference | Say so; ship lam = 1 or blocky-plus-fraction |
| F9 | Post-event recovery/clearing (129 days) | Underestimates f; changes scar boundary | none from statistics | Report date gap next to every number; A6 exp. b composite |
| F10 | Vegetation loss is not landslide | Non-landslide clearing flagged | none | Labelled as "vegetation-loss detection", not landslide/damage |
| F11 | Small calibration set or few independent dates | Wide Beta spread of realised FAR | Beta interval; effective n | Report; larger placebo (Sen12Landslides timesteps) |
| F12 | Clip bias near zero and near one | Positive bias in null blocks; CORE saturation | Placebo mean of f_hat | Bias term b_i; CORE defined at 16 |

## 8. Relation to the compute and data constraints

All v2 steps except SR itself run on CPU in O(blocks); no new GPU work. SR remains SEN2SR-lite on 128 px tiles with the existing OOM policy. Only B02/B03/B04/B08 enter (unmixing may use only B04 and B08); SCL is a mask. Outputs are COGs on the input CRS; the 2.5 m grid is the exact x4 subdivision. No dataset or weight is downloaded by this design.

## 9. Synthetic calculations run for this document

Script committed at `docs/evidence/v2_synthetic_checks.py` (Python 3.11, numpy and math only; run as `python docs/evidence/v2_synthetic_checks.py`; output reproduced below). **Synthetic, endmembers and PSF are assumptions, not measurements.**

### 9.1 NDVI drop vs true bare fraction (assumed endmembers forest red/NIR 0.03/0.40, bare 0.15/0.20; NDVI 0.860 and 0.143)

| f | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.8 | 1.0 |
|---|---|---|---|---|---|---|---|---|
| 10 m NDVI drop | 0.060 | 0.121 | 0.186 | 0.252 | 0.322 | 0.394 | 0.549 | 0.718 |
| linear-in-f drop | 0.072 | 0.144 | 0.215 | 0.287 | 0.359 | 0.431 | 0.574 | 0.718 |

A 0.30 threshold corresponds to f of about 0.47. N - R (linear in f) runs from 0.338 at f = 0.1 to 0.050 at f = 1.

### 9.2 Box vs PSF block fraction (1-D straight edge, Gaussian PSF with MTF(Nyquist) = m)

| m | sigma (m) | mean abs error, straddling blocks | max over offsets |
|---|---|---|---|
| 0.30 | 4.94 | 0.064 | 0.156 |
| 0.20 | 5.71 | 0.086 | 0.191 |
| 0.15 | 6.20 | 0.097 | 0.210 |

Box equivalent sigma = 2.89 m and MTF(Nyquist) = 0.637.

### 9.3 Allocation vs blocky vs random (Section 5.2 table).

## 10. Unverified claims and open assumptions

1. Real Sentinel-2 PSF/MTF per band (requirements read, not measured); the Gaussian PSF is an illustration.
2. That linear areal mixing in (B04, B08) holds over Wayanad forest/scar (endmember variability, shade, wet soil).
3. That season-gap-matched placebo from 2023 is exchangeable with the 2024 pair.
4. That windows are exchangeable enough for the window-level guarantee (pre-registered as assumed, not proven).
5. That SR within-block ranking beats bilinear (no independent evidence; A8 decides).
6. Absence of prior art for the combination (Section 9 of the teardown).
7. Power (recall) of the calibrated detector; no estimate exists.
8. Behaviour on 4 GB for anything but the existing SEN2SR-lite tile path.
9. That the v1 numbers quoted here (EVIDENCE.md) are unchanged; they were read, not re-run.
10. Conformal-p-value BH conditions hold for spatially dependent windows.
