# F7 — Wayanad v2 production rerun (replaces the invalidated x9)

**Status: FAIL** · evidence: real · `config_sha256` `18cce7b1ea88855bc054b42e6cbe3219eef73b2909eb091d07d5ada3df170970` (sha256 of `configs/fix.yaml`)

> ## ⚠ THE DETECTOR THIS MAP COMES FROM FAILED ITS OWN KEEP RULE
>
> gate v2's window false-alarm rate on F1's placebo **test** split is **0.0652 [0.0478, 0.0850]** against a pre-registered requirement of **≤ 0.05**. The keep rule is **NOT MET**.
>
> The detector whose output this map is was measured, on null-event data, to raise false alarms in about 6.5 % of 160 m windows where the pre-registered ceiling was 5 %. F7 is therefore reported as FAIL. These class counts describe what THIS gate configuration produces on the real landslide; they are not a validated detection.
>
> **F5, independently:** F5 (experiments/results/f5.json) found that under REALISTIC (unmixing-based, non-oracle) fraction estimation, SR-ranked sub-pixel allocation was substantially WORSE than naive blocky assignment (paired IoU -0.313). F7 allocates with realistic, unmixing-derived fractions, so F5 is a direct prior signal that the sub-pixel allocation quality in this map is not to be relied on.

## What was actually run

- **Real SR inference, not a cache read.** pinned SEN2SR-lite, 35 native 128 px tiles per date (crop margin 16 px, stride 96), 8 dihedral variants per tile, NDVI per run after inverting the transform. **1,120 real forward passes** through the pinned SEN2SR-lite on cpu (2024-01-16: 11.39s, 2024-01-21: 11.37s, 2024-01-26: 11.39s, 2024-12-06: 11.45s).
- Ran on: the real cached crop rows 256:896, cols 128:640 of the 1024x1024 corrected-AOI 10 m grid = 640x512 at 10 m -> 2560x2048 at 2.5 m. This is exactly the region v1, F1 and F2 use, which is what makes the v1-vs-v2 comparison below well defined. The full 1024x1024 AOI was NOT super-resolved in this run.
- Dates: pre 2024-01-16, 2024-01-21, 2024-01-26 → post 2024-12-06 (event 2024-07-30).
- 4th pre date `2023-12-27`: 2023-12-27 is NOT cached for the corrected AOI (no 10 m bands, no SCL, no SR product) and this run does not fetch imagery. F7 therefore uses the 3 pre-registered pre dates only. No substitute date was used and none was invented. This matches F1's report, which recorded the same absence.
- τ = **3.889118** = T_(933) of n = 981 placebo calibration window scores (α = 0.05); matches `f2.json`: True. Verbatim-from-F1 τ would be 119.91 and would flag nothing.
- F3 normalisation applied: True (119,158 stable 10 m px used to fit the offsets).
- λ = 0.0 (experiments/results/f2.json lambda.value, fitted on F1's calibration split by split-half dihedral reproducibility; not refitted here (refitting on production data would use the event)).

### Reproduction check on the SR

| date | cached product present | identical (NaN-aware) | max abs diff |
|---|---|---|---|
| 2024-01-16 | True | True | 0.0 |
| 2024-01-21 | True | True | 0.0 |
| 2024-01-26 | True | True | 0.0 |
| 2024-12-06 | True | True | 0.0 |

v1 step3 moments, max |this run − cached|: {"pre_mean": 0.0, "pre_std": 0.0, "post_mean": 0.0, "post_std": 0.0}

## Per-class report (2.5 m, 6.25 m² per pixel)

| class | pixels | m² | km² |
|---|---:|---:|---:|
| NO_CHANGE | 4,733,514 | 29,584,462.50 | 29.584463 |
| CORE | 17,648 | 110,300.00 | 0.110300 |
| ALLOCATED | 239,122 | 1,494,512.50 | 1.494512 |
| UNSUPPORTED | 46,640 | 291,500.00 | 0.291500 |
| NO_DATA | 205,956 | 1,287,225.00 | 1.287225 |

Denominator: 5,242,880 px (32.7680 km²) in the crop, of which 5,036,924 are valid. percentages of a class should be taken against valid_2p5m_px, not the total

**Mapped change (v2)** = CORE ∪ ALLOCATED = 256,770 px = **1.604812 km²**. UNSUPPORTED is never counted as flagged.

## Block-sum test on this production map (B7)

- **max |deviation| = 0** over 327,680 10 m blocks (81,549 with round(16f) > 0 — note this is a different count from the 200,814 blocks inside a conformally detected window, many of which round to zero sub-pixels), PASS.
- observed = count of CORE|ALLOCATED px in the block 4x4 footprint of the OUTPUT class map; expected = round(16*f) recomputed from the unmixing fraction. Two independent paths; the allocator's n_per_block is not used (B7).
- Recomputed independently of the gate's own call: True. Scope: THIS production map on the real AOI crop -- not F2's placebo folds.

## Detection

- τ = 3.889118; 810 of 1273 valid 160 m windows detected; 200,814 10 m blocks detected.
- τ is used, not merely loaded (x9's B10 bug): tau_times_0p5 → 287,477 flagged px, tau_fitted → 256,770 flagged px, tau_times_2 → 123,803 flagged px.

- **n_pre mismatch.** sigma_f = sqrt(Var(a)(1/n_pre + 1)), so with 3 pre dates sigma_f is SMALLER than in the 2-pre-date placebo folds tau was calibrated on. The same physical change therefore scores HIGHER here than it would have in calibration: the production gate is slightly MORE sensitive than the threshold was calibrated for, which pushes the true false-alarm rate ABOVE the already-failing 0.0652. At a threshold rescaled to match calibration sensitivity the flagged count would be 246,911 instead of 256,770; the headline uses τ as calibrated.

## v1 vs v2 — only on quantities with identical definitions

Only quantities with IDENTICAL definitions on both sides are compared. The guard `assert_well_defined_comparison` raises on any other pair; see INCOMPARABLE_PAIRS and tests/test_f7_wayanad_v2.py.

| quantity | v1 | v2 | v2 − v1 | IoU |
|---|---:|---:|---:|---:|
| mapped change area (km²) | 0.734900 | 1.604812 | +0.869912 | 0.2448 |
| mapped change area (2.5 m px) | 117,584 | 256,770 | +139,186 | — |
| NO_DATA (2.5 m px) | 205,956 | 205,956 | +0 | 1.0000 |

- v1 flagged = `v1:OBSERVED|INFERRED`, v2 flagged = `v2:CORE|ALLOCATED`. mapped change area: the set of 2.5 m pixels the product maps as change. v1 flagged = OBSERVED union INFERRED; v2 flagged = CORE union ALLOCATED (configs/fix.yaml units.flagged_definition). Same grid, same crop, same dates, same denominator.
- In v1 but not v2: 43,960 px. In v2 but not v1: 183,146 px.

### Sub-pixel divergence from a blocky 10 m parent

A 10 m pixel is a BOUNDARY pixel if its 4-neighbourhood is not constant in v1's 10 m parent mask. The "blocky parent" is that 10 m parent decision replicated 4x4 to 2.5 m -- exactly what a no-SR product would emit. A boundary pixel COUNTS if v2's 16 sub-pixel flagged labels are not all equal to the blocky parent value there. Blocks touching NO_DATA are excluded from both sides.

- 10 m blocks scored: **314,671**; boundary blocks: **4,412**.
- **Boundary 10 m pixels whose 2.5 m allocation differs from the blocky parent: 3,946** (89.4 % of boundary blocks).
- Anywhere in the crop: 80,704 blocks differ; 80,446 blocks carry genuine sub-pixel structure (0 < allocated < 16).
- ⚠ F5 (experiments/results/f5.json) found that under REALISTIC (unmixing-based, non-oracle) fraction estimation, SR-ranked sub-pixel allocation was substantially WORSE than naive blocky assignment (paired IoU -0.313). F7 allocates with realistic, unmixing-derived fractions, so F5 is a direct prior signal that the sub-pixel allocation quality in this map is not to be relied on.

### UNSUPPORTED — NOT compared

DIFFERENT DEFINITIONS -- not compared. v1 UNSUPPORTED = (d_SR > k*sigma_SR) AND NOT parent, where 'parent' is a fixed 0.30 NDVI-drop threshold on the 10 m image. v2 UNSUPPORTED = (d_SR > k*sigma_SR) AND the 10 m block allocated zero sub-pixels, where 'allocated zero' comes from a conformal window detection on the unmixing score plus round(16f) = 0. The gating predicate is a different function of different inputs, so the two counts are not two measurements of one quantity. Both are reported separately, neither is differenced against the other.

- v1 UNSUPPORTED: 138,808 px (0.867550 km²) — d_SR > k*sigma_SR AND NOT parent-positive (parent = 10 m NDVI drop > 0.30)
- v2 UNSUPPORTED: 46,640 px (0.291500 km²) — d_SR > k*sigma_SR AND the 10 m block allocated zero sub-pixels (no conformal window detection, or round(16f) = 0)
- No difference, ratio or IoU is computed between them. Enforced by experiments.f7_wayanad_v2.assert_well_defined_comparison raises on this pair.

## E5 parent-first cascade — forward passes saved

- Tiles: 35 total, 22 fired, 13 ring, 0 audit, **0 skipped**.
- Forward passes: full re-run baseline **1,120** (exactly what this script ran) vs cascade **1,120** → **0 saved (0.0 %)**. tiles x 8 dihedral runs x 4 dates; the full baseline is exactly what this script ran.
- Cost: 0 2.5 m px would have no SR at all, containing 0 of v2's 256,770 flagged px (0.00 %). v1 stitching gives every 2.5 m pixel exactly one owning tile, so a skipped tile leaves its keep-window with no SR at all. The cascade's recall is bounded by the 10 m parent mask by construction (E5's own stated limitation).
- ⚠ This crop is a 6.4 x 5.1 km box CENTRED ON THE SCAR, so nearly every tile is parent-positive or in the buffer ring: the cascade processes all of them and saves nothing here. That is a fact about the crop, not about the cascade. The full-AOI selection below is the honest scale at which the cascade has anything to skip.
- **Full AOI for context** (the full 1024x1024 10 m AOI parent mask (experiments/wayanad_evidence step1_state.npz), same E5 parameters, same 8 runs x 4 dates cost model; SR was NOT run at this scale in F7): 121 tiles, 30 skipped → 2,912 vs 3,872 passes, **960 saved (24.8 %)**.

## The figure

- Rendered: True → `experiments/results/f7_wow_figure.png`
- Underlying arrays: `experiments/results/f7_wow_figure.npz` (window at 2.5 m row 512, col 1536, 64×64 px = 160 m).

> Wayanad, corrected AOI 11.490 N 76.160 E, AT THE SCAR MARGIN. Pre 2024-01-16, 2024-01-21, 2024-01-26  ->  post 2024-12-06. MIDDLE: the 10 m parent decision replicated 4x4 (blocky) -- the slider's 'before'. RIGHT: gate v2's 2.5 m map -- the slider's 'after'. pretrained SEN2SR-lite, NOT fine-tuned (F4 has not shipped). Model reconstruction, not observation. NOT A VALIDATED DETECTOR: this gate's window false-alarm rate on F1's placebo test split is 0.0652 [0.0478, 0.0850], above the pre-registered ceiling of 0.05 -- the keep rule FAILED.

## Limitations

- THE GATE FAILED ITS OWN FAR KEEP RULE (0.0652 [0.0478, 0.0850] vs <= 0.05). This map is what that gate produces, not a validated detection.
- F5 (experiments/results/f5.json) found that under REALISTIC (unmixing-based, non-oracle) fraction estimation, SR-ranked sub-pixel allocation was substantially WORSE than naive blocky assignment (paired IoU -0.313). F7 allocates with realistic, unmixing-derived fractions, so F5 is a direct prior signal that the sub-pixel allocation quality in this map is not to be relied on.
- Only 3 pre dates. 2023-12-27 is pre-registered as an allowed 4th pre date but is not cached for the corrected AOI and this run fetches no imagery; see dates.fourth_pre_date for every path checked. No substitute was used.
- One post-event date (2024-12-06), 4 months after the 2024-07-30 event: regrowth, seasonality and illumination differences between a January pre-pool and a December post date are confounded with the landslide. There is no second post date to separate them.
- tau was calibrated on 2-pre-date placebo folds and applied to a 3-pre-date production score. sigma_f shrinks with n_pre, so the production gate is slightly MORE sensitive than the threshold was calibrated for -- pushing the true FAR further above the already-failing value. Measured and reported in n_pre_mismatch_between_calibration_and_production, not corrected.
- SR ran on the 640x512 (10 m) crop, not the full 1024x1024 AOI. Stated exactly in sr_run.ran_on.
- e_b is the darkest land cover in the AOI, not true bare ground (the pre-registered NDVI <= 0.20 rule selects 8 px). f is upper-leaning and is NOT a calibrated area.
- The gate remains highly sensitive to the endmembers; see endmembers.sensitivity_plus_minus_1_robust_sd for the effect on THIS map.
- v2 UNSUPPORTED and v1 UNSUPPORTED have different definitions and are NOT differenced anywhere; both are reported with their definitions.
- The E5 cascade numbers are a tile-selection and forward-pass count on the real parent mask, plus a measurement of what the skipped tiles would have cost. The cascade's recall is bounded by the 10 m parent mask by construction.
- No labels were used and no 2.5 m ground truth exists, so no accuracy, F1 or IoU-against-truth is reported anywhere. The IoU in v1_vs_v2 is agreement between two products, not accuracy.
- pretrained SEN2SR-lite (NOT fine-tuned)
- k = 2.0 (NOT calibrated)
- retrospective comparison
- pixels and neighbouring windows are spatially correlated; exchangeability assumed across windows, not proven

## Verdict

THE EXPERIMENT RAN CORRECTLY; THE DETECTOR IT MEASURES DID NOT PASS ITS OWN KEEP RULE. F7 executed end to end on real data -- real tiled SR (1120 real forward passes), a tau recomputed from placebo folds only, a block-sum test that passes on this production map -- but the gate it runs is F2's gate v2, whose pre-registered detection keep rule FAILED: window FAR 0.0652 [0.0478, 0.0850] against a required <= 0.05. Every class count, area and comparison in this report describes WHAT THIS GATE CONFIGURATION PRODUCES on the real Wayanad scene. None of it is a validated detection, and none of it may be quoted as one. Status is FAIL for exactly this reason. F5 (experiments/results/f5.json) found that under REALISTIC (unmixing-based, non-oracle) fraction estimation, SR-ranked sub-pixel allocation was substantially WORSE than naive blocky assignment (paired IoU -0.313). F7 allocates with realistic, unmixing-derived fractions, so F5 is a direct prior signal that the sub-pixel allocation quality in this map is not to be relied on.

