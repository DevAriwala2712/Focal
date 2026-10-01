# ADR — F2: gate v2, rebuilt

- **Status**: accepted (implementation), keep rule **not met** (see `experiments/results/f2.json`)
- **Date**: 2026-09-30
- **Supersedes in practice**: `docs/adr-x4-gate-v2.md` (the design is kept; its *implementation path* through
  `experiments/x4_calibration.py`, `experiments/x4_gate_v2.py` and `experiments/x9_v2_production.py` is
  quarantined — `experiments/results/INVALIDATED.md`)
- **Pre-registration**: `configs/fix.yaml` `f2_gate_v2`, sha256
  `18cce7b1ea88855bc054b42e6cbe3219eef73b2909eb091d07d5ada3df170970`
- **Code**: `trustsr/gate_v2.py`, `experiments/f2_gate_v2.py`, `tests/test_gate_v2.py`,
  `trustsr/placebo_v2.py` (`GATES['gate_v2']`)

## Context

The A10 audit found seven defects (B6–B12) in the gate-v2 path. All seven were in **how gate v2 was
called and calibrated**, not in the idea of gate v2. ADR-x4's design — unmix to a per-10 m fraction,
allocate exactly `round(16f)` sub-pixels, threshold by split conformal, report CORE / ALLOCATED /
UNSUPPORTED / NO_DATA — is unchanged and re-affirmed here. What changes is that every number the design
depends on is now produced by real, auditable code.

A deliberate consequence: this rebuild produces a **worse-looking** result than the quarantined one. x9
detected everything (B10) with no real SR in the ranking (B9) and almost no NO_DATA (B11), so it mapped a
large, clean-looking scar. F2 maps less, admits 2.8 M NO_DATA pixels per fold, and fails its own FAR keep
rule. That is the point.

## Decisions

### 1. Endmembers are estimated from data (B12)

`estimate_endmembers(refl_pre, valid_pre, ndvi_pre, exclude, ...)`. x9 hard-coded
`e_v = [0.08, 0.06, 0.04, 0.4]` and `e_b = [0.15, 0.12, 0.10, 0.08]` as literals, so the fraction `f` was
a function of two numbers nobody measured.

- `e_v` = per-band median of the temporal-mean reflectance over pixels that are valid on every pre date,
  have mean NDVI ≥ 0.80 and per-pixel NDVI std across the pre dates ≤ 0.05. Both thresholds are the
  published E6 stable-pixel statistics already cited in `configs/wayanad_evidence.yaml` `change`; neither
  was chosen after seeing an F2 number.
- `e_b` = same statistic over pixels with NDVI ≤ 0.20 on **every** pre date.
- Both samples exclude the 1500 m plausibility disks and the landslide footprint, using
  `trustsr.placebo_v2.exclusion_mask_10m` — F1's geometry, reused, not redefined.

**Why an exclusion is not a label.** The function has no `labels`, `class_map`, `y_post` or
`footprint_is_change` argument, and `gate_v2_signature_audit()` asserts that in code. `exclude` only
*removes* pixels from the estimation sample; `test_endmembers_ignore_everything_inside_the_exclusion`
sets the excluded region to absurd values and asserts the endmembers are bit-identical. Removing the
event area is required precisely so the scar cannot become the bare-ground endmember.

**Deviation, recorded up front.** The pre-registered `e_b` rule (NDVI ≤ 0.20) selects **8 pixels** over
the whole 1024² AOI and **0** inside F1's crop: the Western Ghats AOI is evergreen in January and the only
large bare surface is the scar, which is excluded by construction. The ceiling was escalated
deterministically in 0.05 steps until ≥ 100 pixels qualified (0.30, n = 332), **before** any τ, block-sum
or FAR number existed, and the full count/endmember table at every ceiling from 0.20 to 0.50 is reported.
The honest consequence: `e_b` is the darkest land cover present, not soil or rock; the mixing line is
shorter than a true vegetation–soil line, so `f` is upper-leaning and is not a calibrated area.

**Sensitivity is large and is reported, not buried.** Shifting `e_b` by +1 robust SD multiplies the
flagged-pixel count by ≈ 8 on the fold tested. This is the single largest source of uncertainty in the
gate and it exists because only a few hundred dark-land pixels are available to estimate `e_b` from.

### 2. `σ_f` by error propagation, not bootstrap

`a = (y − e_b)·d / |d|²`, so `Var(a) = (σ_R² d_R² + σ_N² d_N²)/|d|⁴` and, with `n_pre` independent pre
dates, `Var(f) = Var(a)(1/n_pre + 1)`. Chosen over a bootstrap because with 2–3 pre dates a bootstrap over
dates has essentially no resolution, whereas the propagation formula is exact for the linear estimator
actually used. `σ_band` is the per-pixel temporal std of the pre-date reflectance, **floored at the scene
median** of that statistic. The floor is a variance floor, stated and fixed at the median, never varied:
without it, pixels whose two or three pre dates happen to agree to ~1e-6 produce scores of order 1e5 and a
single such pixel sets τ. F1's exported `d/σ_v1` window scores reach 255 for exactly this reason.

Endmember uncertainty is deliberately **not** folded into `σ_f`: a scene-wide endmember shift is a
systematic error, not independent per-pixel noise, and averaging it into a per-pixel σ would understate it.
It is reported separately as the ±1 robust-SD sensitivity.

### 3. τ from placebo data only, and τ is a required argument (B6, B10)

x4 built its conformal calibration scores from `pre_mean − post_mean` on the **real event pair** with
`valid = ones`. A conformal threshold calibrated on data containing the event it is meant to detect
guarantees nothing. F2 takes τ from F1's leave-one-date-out **placebo** folds, in which every positive is a
false alarm by construction, restricted to the **calibration (even-tile)** half of the checkerboard split:
τ = T₍k₎, k = ⌈(n+1)(1−α)⌉, α = 0.05, n = 981 → **τ = 3.889118**.

x9 then loaded `tau_win` from x4.json and never passed it into `apply_gate_v2` (`kappa=None`, and the
signature had no τ argument at all), so detection never ran and every block with `f > 0` was allocated.
In F2, `tau` is the **eighth positional parameter of `apply_gate_v2` with no default**; `None`, `NaN` and
`inf` all raise. `TestTauMutation` asserts (a) omitting τ is a `TypeError`, (b) non-finite τ raises, (c)
the output map changes with τ, and (d) the flagged count is monotone in τ. The run itself records
τ×0.5 → 22 399 flagged px, τ → 3 471, τ×2 → 1 055.

**One substantive departure from the letter of the pre-registration.** `configs/fix.yaml` says to take τ
from `data/experiments-cache/f1_calibration_scores.npz`. That file's sha256 was verified
(`53a2fe8a…2679e4`, matches f1.json) and it is consumed and reported — but F1 exported
`max(d/σ_v1)` per window, an SR-NDVI z-score ranging −25.9 to 255.6, and F1's own docstring says so
("gate_v2's own score (unmixing f/σ_f) does not exist yet"). Split conformal is only valid when
calibration and test scores come from the **same** score function. Taken verbatim, τ would be 119.9, the
gate would flag nothing, and the reported FAR would be a vacuous 0 — a pass by construction. So τ is
recomputed with gate_v2's own score on **exactly** F1's folds, splits, window grid and α. The verbatim
number is reported alongside. This makes the keep rule harder, not easier.

### 4. Allocation ranked by real SR (B9)

x9's "SR change score" was `np.repeat(np.repeat(d_10m, 4), 4)` — the 10 m difference replicated. No SR
signal entered allocation at all; the 2.8 s whole-AOI runtime is the tell.

F2 ranks by the real 2.5 m SR NDVI difference from the cached **E1-tiled, E8-dihedral SEN2SR-lite** stack
(`per_date_ndvi/<date>_dihedral_means.npy`, 8 runs × 2560 × 2048 per date), produced by the audited v1
pipeline on exactly this crop. SR inference was not re-run: the cached product *is* the real tiled
dihedral stack for these dates and this crop, and re-running it on CPU would reproduce the same bytes over
hours. The field varies within **100 %** of 4×4 blocks; `assert_sr_score_is_not_replicated` raises if it
varies within none, i.e. if x9's construction is ever passed again
(`test_a_blocky_replicated_field_is_rejected` feeds it the literal `np.repeat(np.repeat(...))`).

**Neighbour agreement**, defined precisely: the fraction of a pixel's 8 neighbours (8-connectivity,
zero-padded at the edge) allocated in a provisional pass-1 pure-SR allocation. Two passes keep it
non-circular. The blend is `u = (1−λ)·rank(d_SR) + λ·rank(na)` on within-block ranks, so λ ≤ 0.5 — a
pre-committed cap — always leaves the SR term at least equal weight.

**λ fitted on the calibration split only**, by split-half dihedral reproducibility: allocate using the SR
field built from dihedral runs 0–3, again from runs 4–7, and maximise the Jaccard index of the two
allocated sets over calibration pixels. This objective is label-free (placebo data has no truth) and uses
only variation the model itself produces. Result: λ = 0.0; Jaccard is highest at 0 and falls monotonically
across the grid. λ = 0 is a **fitted** value, not a disabled parameter — the term demonstrably changes the
map when λ > 0 (`test_lambda_changes_which_sub_pixels_are_chosen_but_never_how_many`). Note also that λ
cannot be tuned against the keep rule even in principle: it changes *which* sub-pixels in a detected block
are allocated, never *how many* (`round(16f)`) nor *which* blocks are detected, so neither pixel nor window
FAR depends on it.

### 5. NO_DATA from SCL on every date (B11)

x9 used `np.isnan(d_sr) | np.isnan(sigma_sr)` and reported 512 NO_DATA px where v1 reported 205 956 —
cloud and shadow pixels were silently mapped as change. `nodata_from_scl` AND-s
`experiments.wayanad_evidence.data.scl_valid` (v1's logic, reused verbatim) over every date and propagates
nearest-neighbour to 2.5 m; a 10 m pixel masked on any date yields exactly 16 NO_DATA sub-pixels
**regardless of whether its value is finite**. A 10 m block touching any NO_DATA sub-pixel is excluded from
scoring and allocates nothing, so NO_DATA can never sit half-inside an allocated block. F2 reports
2 836 569 NO_DATA px per fold (including the exclusion geometry) against 2 406 311 valid.

### 6. A real block-sum test (B7)

x4's `check_block_sum_consistency` compared `h*w*100 m²` against `h*w*100 m²`. Under
`configs/fix.yaml` `f9_checks.stop_rule`, a tautological test means the claim is **dropped**, not
downgraded.

`block_sum_deviation(class_map, f_effective, nodata)` derives the two sides independently: *observed* by
counting CORE|ALLOCATED pixels in each block's 4×4 footprint **of the output class map**, *expected* by
recomputing `round(16f)` from the unmixing fraction. The allocator's own `n_per_block` is not a parameter,
and a test asserts it isn't (`test_block_sum_does_not_receive_the_allocator_count`). A second test corrupts
one allocated pixel in the output and asserts the deviation becomes exactly 1 — a self-comparison cannot do
that.

Result: **max |deviation| = 0** over 983 040 block-evaluations. The Wave-3 stop condition does not fire.

## Consequences

| | x4 / x9 (quarantined) | F2 |
|---|---|---|
| endmembers | 8 hard-coded literals | estimated, 178 141 / 332 px, sensitivity reported |
| calibration data | real event pair, `valid = ones` | placebo folds, calibration split only |
| τ in the gate | loaded, never passed | required positional arg; mutation-tested |
| allocation ranking | `np.repeat(np.repeat(d_10m,4),4)` | real 2.5 m dihedral SR, guarded |
| NO_DATA | NaN only (512 px) | SCL every date (2 836 569 px/fold) |
| block-sum test | `h*w*100` vs `h*w*100` | output map vs `round(16f)`, max dev 0 |
| `config_sha256` | `"pending-a3-output"` | sha256 of `configs/fix.yaml`, hashed at run time |
| whole-AOI runtime | 2.8 s | 21 s for 3 folds × 2 tracks + λ grid + sensitivity |

**The keep rule is not met.** Test-split window FAR = 0.0652 [0.0478, 0.0850] against α = 0.05
(n = 981 windows; pixel FAR 0.0036 [0.0022, 0.0054] over 3 617 880 valid px). The CI contains α, but the
pre-registered rule is stated on the point estimate, so gate_v2's FAR claim **fails**. Two honest readings,
both recorded: split conformal guarantees a *marginal* exceedance rate under exchangeability, and
`configs/fix.yaml` `units.window_caveat` already warns that "pixels and neighbouring windows are spatially
correlated; exchangeability assumed across windows, not proven" — three folds sharing one crop, with an
interleaved checkerboard, are about as far from exchangeable as a split can be. But that is an explanation,
not a pass.

Per `configs/fix.yaml` `f2_gate_v2.block_sum_test.keep_rule`, the condition that stops Wave 3 (F7) is a
**block-sum failure**. That did not happen. F7 is therefore not blocked by F2, but it must carry gate_v2's
failed FAR keep rule forward rather than treating gate v2 as validated.

## Rejected alternatives

- **Bootstrap `σ_f` over pre dates.** With 2–3 dates a date-bootstrap has no resolution; the propagation
  formula is exact for the linear estimator used.
- **Taking τ verbatim from F1's npz.** Statistically invalid (different score function) and would have
  produced a vacuous FAR of 0. Rejected in favour of the harder, valid recomputation.
- **Widening the `e_b` NDVI ceiling silently, or using inside-footprint pixels for `e_b`.** The first hides
  a pre-registration failure; the second turns the endmember into a label (B12). Escalated openly instead,
  with the full table published.
- **Dropping the `σ_band` floor.** Produces the F1 pathology (scores to 255 driven by a handful of pixels)
  and lets one pixel set τ.
- **Fitting λ to minimise FAR.** FAR is λ-invariant by construction, so this would have been theatre.
