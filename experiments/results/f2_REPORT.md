# F2 — gate v2 rebuilt (replaces the invalidated x4/x9 gate-v2 path)

**Status: FAIL** · evidence: real · config_sha256 `18cce7b1ea88855bc054b42e6cbe3219eef73b2909eb091d07d5ada3df170970` (configs/fix.yaml)

Pre-registered in `configs/fix.yaml` `f2_gate_v2`. Design: `docs/adr-f2-gate-v2.md`. Implementation: `trustsr/gate_v2.py`. Harness: F1's `trustsr/placebo_v2.py`, unchanged FAR code.

## What each part fixes

| B-ID | The old bug (x4 / x9) | What F2 does instead |
|---|---|---|
| B6 | conformal scores came from `pre_mean - post_mean` on the REAL event pair with `valid = ones` | τ is the order statistic of PLACEBO (null-event) calibration-split window scores; no post-event date is loadable in this path |
| B7 | `check_block_sum_consistency` compared `h*w*100 m²` with itself | `block_sum_deviation` counts CORE\|ALLOCATED px in the output class map and compares with `round(16f)` recomputed from the fraction — two independent paths |
| B8 | `config_sha256 = "pending-a3-output"` | `config_sha256` is the sha256 of `configs/fix.yaml`, hashed at run time |
| B9 | allocation ranked by `np.repeat(np.repeat(d_10m,4),4)` | ranked by the real 2.5 m SR NDVI difference from the cached E1-tiled / E8-dihedral SEN2SR-lite stack; `assert_sr_score_is_not_replicated` raises on a blocky field |
| B10 | τ loaded from x4.json and never passed in (`kappa=None`) | τ is a **required positional argument** of `apply_gate_v2`; non-finite τ raises; mutation test proves the map moves |
| B11 | NO_DATA from NaN pixels only (512 px vs v1's 205,956) | NO_DATA from the SCL validity mask on **every** date, nearest-neighbour to 2.5 m |
| B12 | `e_v`/`e_b` hard-coded constants | estimated from stable/high-NDVI and low-NDVI (≤ 0.3, see the deviation note below) pre-event pixels outside the disks + footprint |

## Endmembers (B12)

- `e_v` = [0.02403, 0.04097, 0.02807, 0.2566] from **178,141** pixels (robust SD [0.00519, 0.00845, 0.00667, 0.04873])
- `e_b` = [0.11133, 0.09587, 0.07623, 0.1819] from **332** pixels (robust SD [0.04487, 0.03645, 0.02429, 0.07087])
- rule e_v: valid on all 3 pre dates, outside exclusion, mean pre NDVI >= 0.8, per-pixel NDVI std across pre dates <= 0.05
- rule e_b: valid on all 3 pre dates, outside exclusion, NDVI <= 0.3 on EVERY pre date
- excluded from the sample: 186,987 10 m px of the full AOI (1500 m disks around configs/wayanad_evidence.yaml plausibility_points union experiments/wayanad_evidence/outputs/footprint.tif, via trustsr.placebo_v2.exclusion_mask_10m (reused, not redefined)); estimated on the whole 1024x1024 10 m AOI (not only F1's 640x512 crop), pre-event dates only
- ±1 robust-SD sensitivity (flagged px count, τ and λ fixed): baseline 3471, max |Δ| 24298 (700.0 %)

### Pre-registration deviation on `e_b` — stated up front

- **item**: configs/fix.yaml f2_gate_v2.endmembers.e_b: 'pre-event pixels with NDVI <= 0.2, valid on all pre dates, outside disks+footprint'
  - **problem**: On this AOI in January the rule selects almost nothing. Over the whole 1024x1024 AOI, outside the disks + footprint and valid on all three pre dates, only 8 pixels have NDVI <= 0.2 on every date (0 pixels inside F1's 640x512 crop). The Western Ghats AOI is evergreen; the only large bare surface is the landslide scar itself, which the exclusion geometry removes by construction -- and which must be removed, because using it would make e_b a label (B12).
  - **resolution**: The NDVI ceiling was escalated deterministically in 0.05 steps from the pre-registered 0.20 until at least 100 qualifying pixels existed. The first ceiling that qualified is 0.3. The full count/endmember table at every ceiling is in endmembers.ndvi_low_ceiling_table.
  - **when**: decided and executed BEFORE any tau, block-sum or FAR number was computed in this run; no threshold was moved after seeing a result
  - **consequence**: e_b at NDVI <= 0.3 is the darkest, least-vegetated land in the AOI, NOT true bare ground. It sits closer to e_v than a real soil/rock endmember would, which SHORTENS the mixing line and therefore INFLATES f for a given spectral change. The fraction f reported by this gate is an upper-leaning estimate of vegetation-loss fraction on this AOI and must not be read as a calibrated area.

| NDVI ceiling | qualifying px | meets min count (100) |
|---|---|---|
| 0.20 **(pre-registered)** | 8 | False |
| 0.25 | 80 | False |
| 0.30 **(used)** | 332 | True |
| 0.35 | 1,182 | True |
| 0.40 | 2,554 | True |
| 0.45 | 5,094 | True |
| 0.50 | 9,455 | True |

## τ (B6, B10)

- **τ = 3.889118** = T_(933) of n = 981 calibration placebo window scores, k = ⌈(n+1)(1−α)⌉, α = 0.05
- provenance: ceil((n+1)(1-alpha))-th order statistic of the per-160 m-window max of gate_v2's OWN score s = f/sigma_f, over F1's three leave-one-date-out PLACEBO folds, restricted to the CALIBRATION (even-tile) half of the checkerboard split. Null-event data only: no post-event date is loadable in this code path (B1 guard).
- F1's exported file `data/experiments-cache/f1_calibration_scores.npz` sha256 verified: `53a2fe8a3a7852d7648d6d8d17abaf982b1435d23ffb0afcd9fda8effd2679e4` (matches f1.json: True). F1 exported max(d/sigma_v1) per window -- an SR-NDVI z-score, whose units and scale (range -25.86 to 255.60) are not those of gate_v2's unmixing score f/sigma_f. F1's own docstring says so ("gate_v2's own score (unmixing f/sigma_f) does not exist yet"). Split conformal is only valid when calibration and test scores come from the SAME score function, so tau is recomputed with gate_v2's score on EXACTLY F1's folds, splits and window grid. The verbatim number is reported above for transparency; applying it would make the gate flag nothing and produce a vacuous FAR of 0.

## λ and allocation (B9)

- λ = 0.0 (grid [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]), fit: F1 calibration (even tiles) ONLY; the test split was never used to choose lam
- objective: split-half dihedral reproducibility: allocate with the SR change field built from dihedral runs 0-3 and again from runs 4-7, and maximise the Jaccard index of the two allocated sets over calibration pixels
- neighbour agreement: fraction of a 2.5 m pixel's 8 neighbours (8-connectivity, zero-padded at the array edge) allocated in the provisional pass-1 (pure-SR) allocation; the pixel itself is excluded
- lam changes WHICH sub-pixels inside a detected block are allocated, never HOW MANY (round(16*f)) nor which blocks are detected, so it cannot move pixel or window FAR; it is therefore not tunable against the keep rule even in principle
- SR ranking field varies within 4×4 blocks for 2024-01-16: 100.0 %, 2024-01-21: 100.0 %, 2024-01-26: 100.0 % of blocks — a `np.repeat` field would be 0 % and would raise.

## Block-sum test (B7)

- **max |deviation| = 0** over 983,040 block-evaluations (0 over detected blocks) → PASS
- observed = count of CORE|ALLOCATED 2.5 m px in each block's 4x4 footprint of the OUTPUT class map; expected = round(16*f) recomputed from the unmixing fraction. Independent paths (B7).

## gate_v2 in F1's harness, TEST split (odd tiles)

| metric | estimate [95 % CI] (numerator / denominator) |
|---|---|
| window FAR (160 m windows) | 0.0652 [0.0478, 0.0850] (num 64 / den 981, 273 blocks) |
| pixel FAR (2.5 m px) | 0.0036 [0.0022, 0.0054] (num 13167 / den 3617880, 273 blocks) |
| window FAR, no F3 normalisation | 0.0540 [0.0369, 0.0735] (num 53 / den 981, 273 blocks) |
| pixel FAR, no F3 normalisation | 0.0039 [0.0024, 0.0059] (num 14265 / den 3617880, 273 blocks) |
| window FAR, calibration split (in-sample, for reference) | 0.0489 [0.0335, 0.0671] (num 48 / den 981, 273 blocks) |

UNSUPPORTED per fold (reported separately, **never** counted as flagged): 2024-01-16: 2,257,067 px, 2024-01-21: 536,070 px, 2024-01-26: 44,092 px.

Keep rule (window FAR ≤ α = 0.05): **NOT MET**.

**Verdict.** Block-sum test PASSES (max deviation 0). The scoring keep rule does NOT hold: the test-split window FAR point estimate is 0.0652, above alpha = 0.05. Its 95 % CI is [0.0478, 0.0850], which does contain alpha, but the pre-registered rule is stated on the point estimate and is therefore NOT met. F2's overall status is FAIL. Per configs/fix.yaml f2_gate_v2.block_sum_test.keep_rule, the condition that stops Wave 3 (F7) is a block-sum failure, and that did not happen; the FAR shortfall is a keep-rule failure for gate_v2's claim, not a Wave-3 stop.

> Caveat (configs/fix.yaml `units.window_caveat`): pixels and neighbouring windows are spatially correlated; exchangeability assumed across windows, not proven

Wave 3: not blocked by F2

## Limitations

- Three pool dates only (2024-01-16/21/26): the cached real 2.5 m dihedral SR product exists for no other pre-event date, so every fold's reference is the mean of just 2 dates and the three folds share one crop. Serial and spatial correlation across folds is not modelled, as in F1.
- tau is recomputed with gate_v2's own score rather than read verbatim from f1_calibration_scores.npz, because F1 exported a different score function (d/sigma_v1). The folds, the checkerboard split, the window grid and the alpha are exactly F1's. The verbatim number is reported in tau.f1_export_consumed.
- sigma_f propagates measurement noise only. Endmember uncertainty is reported separately as the +/- 1 robust SD sensitivity, not folded into the score.
- sigma_band carries a floor at the scene median of the per-pixel temporal std. Without it a handful of pixels with near-identical pre dates dominate the score (F1's exported d/sigma_v1 reaches 255 for this reason). The floor is fixed at the median and was not varied.
- SR inference was not re-run; the cached real tiled dihedral stack for these dates and this crop was used as-is (see sr_ranking.not_rerun_note).
- The pre-registered e_b rule (NDVI <= 0.2) is not satisfiable on this evergreen AOI in January (8 qualifying px over the whole AOI). The ceiling was escalated to 0.3 before any result was computed; see preregistration_deviations. e_b is therefore the darkest land cover present, not true bare ground, and f is correspondingly upper-leaning.
- The gate is HIGHLY sensitive to the endmembers: shifting e_b by +1 robust SD multiplies the flagged pixel count by about 8.0 on the fold tested. The robust SD of e_b is large relative to |e_v - e_b| because only a few hundred dark-land pixels exist to estimate it from. The fraction f is not a calibrated area and must not be reported as one.
- lam was fitted to 0.0 on the calibration split, i.e. the neighbour-agreement term carries zero weight in the headline run. That is a fitted value, not a disabled parameter: split-half dihedral reproducibility was HIGHEST at lam = 0 and fell monotonically across the grid (see lambda.curve), and tests/test_gate_v2.py::test_lambda_changes_which_sub_pixels_are_chosen_but_never_how_many shows the term does change the map when lam > 0.
- UNSUPPORTED is very large on these folds (millions of 2.5 m px) because it is defined against the v1 dihedral sigma, which is tiny, so almost any SR NDVI difference exceeds k*sigma. UNSUPPORTED is reported separately and is NEVER counted as flagged (configs/fix.yaml units.flagged_definition.v2), so it does not enter any FAR above; it is a known v1-sigma artefact, not a gate-v2 result.
- F3's per-date normalisation changes the test-split window FAR from 0.0540 (without) to 0.0652 (with). Both columns are reported; the headline follows configs/fix.yaml f3_noise_v2.interaction_test, which instructs F2 to apply the normalisation.
- pixels and neighbouring windows are spatially correlated; exchangeability assumed across windows, not proven
