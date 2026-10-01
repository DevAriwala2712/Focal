# Gate v2 Recalibration (F11/F12) — Design Spec

## Context

On `experiments/fix-v2`, F2 rebuilt gate v2 (fixing B6–B12) and F7 reran the Wayanad production
map with it. Both are honest, real, and both FAILED the pre-registered window-FAR keep rule
(`configs/fix.yaml` `f2_gate_v2.scoring`): test-split window FAR = 0.0652 [0.0478, 0.0850] vs
required ≤ α = 0.05 (n=981 windows). F9 (the adversarial verifier) audited this failure and found
it is real, not an arithmetic slip, with three contributing, independently-verified causes:

1. **n_pre mismatch.** `apply_gate_v2` computes `sigma_f` using `n_pre = y_pre.shape[0]` at call
   time (`trustsr/gate_v2.py:569`). F7's production call passes 3 pre dates (n_pre=3). But τ was
   calibrated on F1's placebo folds, which are leave-one-out over a pool of only 3 SR-dependent
   dates — every fold's reference set has exactly 2 remaining dates (n_pre=2). Since
   `sigma_f ∝ sqrt(1/n_pre + 1)`, the n_pre=2 calibration population has systematically *larger*
   sigma_f (hence smaller scores) than the n_pre=3 production population for the same underlying
   signal. τ, calibrated against the smaller-scored population, is too low for production: F9
   measured `sigma_f(n_pre=2)/sigma_f(n_pre=3) = 1.06066` exactly, meaning 0.0652 is a **lower
   bound** on the true production FAR, not the number itself.
2. **Calibration/test exchangeability gap.** F9 found gate v2's calibration-split (even-tile)
   window FAR is ≤ α in *both* normalisation arms (0.04893), while test-split (odd-tile) FAR is
   0.0652 (normalised) / 0.0540 (un-normalised). The gate is properly calibrated on its own
   calibration data and simply doesn't generalise to the held-out split — consistent with
   `configs/fix.yaml`'s own pre-registered caveat ("exchangeability assumed across windows, not
   proven") and with only 3 pool dates sharing one crop (low degrees of freedom, correlated folds).
3. **Endmember instability.** `e_b` (bare-ground endmember) is estimated from only 332 pixels on
   this evergreen AOI; ±1 robust SD on `e_b` multiplies the flagged pixel count by ~8×. This is
   already honestly reported by F2 as a sensitivity diagnostic, but is not currently part of the
   keep rule — a gate could "pass" by luck of where `e_b` happened to land.

**Goal of F11/F12:** address all three causes with genuine methodology, re-score gate v2 honestly,
and rerun Wayanad production with whatever the recalibrated gate turns out to produce — including
a second FAIL, if that's what happens. This is not a campaign to force a PASS.

## Key finding that reshaped this design

An earlier draft of this spec proposed folding `e_b`'s estimation uncertainty directly into
`sigma_f` (the per-pixel detection score's denominator), so a noisily-estimated endmember would
make detection more conservative. **Rejected after reading `trustsr/gate_v2.py:245-257`'s own
docstring**, which already considered and explicitly rejected this:

> "Endmember uncertainty is deliberately NOT folded in here... because a scene-wide endmember
> shift is a systematic, not an independent per-pixel, error."

This is correct: `e_b` is one fixed (if imperfectly known) number applied identically to every
pixel in a given run. Its uncertainty is about which stable pixels the *gate-building* step
happened to sample — an epistemic, correlated, scene-wide quantity — not an independent per-pixel
radiometric noise draw that averages out across a window. Folding it into `sigma_f` would let a
shared bias masquerade as per-pixel noise, which is the same class of statistical error the gate
rebuild was supposed to eliminate. **F11 does not touch `fraction_sigma`.** Instead, endmember
instability is addressed by making the **keep rule itself** robustness-checked (see below).

## Non-negotiables inherited from the fix program (unchanged)

AGENTS.md / RISK_REPORT.md / the Fourier hard constraint stay untouched. Only B02/B03/B04/B08.
Native 128 px tiles only. Results are PASS/FAIL/BLOCKED/INVALIDATED; BLOCKED and INVALIDATED are
never PASS. `evidence: real` or `evidence: synthetic` on every result. Never fabricate a date,
metric, or GPU number. Never tune on the test split. A keep rule is committed before the run it
governs. Earlier result files (`f2.json`, `f7.json`, their reports) are never edited — F11/F12
supersede them with new files, same quarantine discipline as F0 used for x3/x4/x9.

## Architecture

Two new sequential experiments on branch `experiments/fix-v3` (branched from `experiments/fix-v2`
tip, which stays untouched — F0–F10's hashes and results are not disturbed):

- **F11** (gate v2 recalibration): extends the SR-dependent placebo pool, redesigns calibration-
  fold construction to match production's n_pre exactly, and adds an endmember-sensitivity-robust
  keep rule. Re-scores gate v2 end-to-end on F1's (extended) harness.
- **F12** (Wayanad v3 rerun): reruns the production map using F11's recalibrated τ, on the same
  dates/crop F7 used, comparing v1/v2(F7)/v3 only on well-defined shared quantities.

Pre-registered in a **new** file `configs/f11_f12.yaml`, committed before F11 runs. `configs/fix.yaml`
and its hash (cited by every F0–F10 result) are never edited.

## Component 1: extend the SR-dependent placebo pool

**File:** `experiments/f11_extend_pool.py` (new)

F1's STAC audit (`experiments/results/f1.json` `stac_audit.table`) already found 8 clear Jan–Jun
2024 dates beyond the 3 already SR-processed (2024-01-16/21/26): 2024-02-05, 02-10, 02-15 (10 m
bands already fetched by F1), 02-20, 03-01, 03-06, 03-11, 03-26 (not fetched). This task:

1. Fetches 10 m bands for any of the remaining candidate dates not yet cached (reuse the same
   Planetary Computer STAC code path F1 used — `experiments/wayanad_evidence/` fetch logic — same
   AOI, same SCL clear-fraction rule, cited not redefined).
2. Runs real SR inference (E1 tiler + E8 8-dihedral) on the same crop region used throughout
   (`rows 256:896, cols 128:640` at 10 m, matching F1/F2/F7), for enough additional dates to reach
   a **minimum pool of N=6** SR-dependent dates (3 existing + 3 more). This is a hard minimum, not
   a target to approach: N≥4 is the absolute floor required for fold construction (see Component 2)
   to produce even a single reference set of size 3 with something left over to hold out; N=6 is
   chosen to give C(6,3)=20 reference-set combinations, a meaningful increase in combinatorial
   coverage over F1's original 3 trivial leave-one-out folds.
3. Writes the per-date dihedral NDVI product to
   `data/experiments-cache/wayanad_evidence/per_date_ndvi/<date>_dihedral_means.npy`, matching the
   existing cache format exactly, so it's usable by `trustsr/placebo_v2.py` and
   `trustsr/gate_v2.py` unchanged.
4. **Stop rule:** if real SR processing cannot reach N≥4 within a ~45-minute CPU budget (budget
   derived from F7's measured ~46s for 4 dates × 35 tiles × 8 dihedral on this crop; 3 more dates
   at that rate is on the order of minutes, so 45 minutes is a generous ceiling, not a tight one),
   F11 is **BLOCKED**, not forced with a mismatched n_pre. State exactly which dates were
   processed and why any remaining ones were not.

## Component 2: fold construction matching production's n_pre

**File:** `trustsr/placebo_v2.py` (add a new function; the existing leave-one-out `build_folds`
used by F1 is untouched — F1's own result stands as-is)

```python
def build_folds_matching_n_pre(pool_dates: list[str], n_pre: int) -> list[Fold]:
    """Enumerate every size-n_pre reference subset of pool_dates; for each, every remaining pool
    date is a held-out comparison. This matches production's n_pre exactly (unlike leave-one-out
    over the whole pool, which only matches n_pre = len(pool_dates) - 1), and multiplies the number
    of raw (reference, held-out) pairs from len(pool_dates) to C(len(pool_dates), n_pre) * (len(pool_dates) - n_pre).

    Folds from overlapping reference sets are NOT independent (they share dates) -- this is
    reported honestly via an `effective_independent_dates` count, same spirit as F1's own pool
    reporting, not hidden behind the larger raw pair count.
    """
```

For N=6, n_pre=3: C(6,3)=20 reference sets × 3 held-out each = 60 raw pairs (vs F1's original 3
leave-one-out pairs). Each fold's reference set has exactly 3 dates — matching
`apply_gate_v2`'s production call exactly, closing the n_pre mismatch (cause 1) as a direct
consequence of this redesign, not a separate patch.

Checkerboard calibration/test tile split (even/odd, seed from `configs/f11_f12.yaml`) is applied
per fold exactly as F1 did — unchanged mechanism, just more folds and the right n_pre feeding it.

**Test:** `test_build_folds_matching_n_pre_produces_reference_sets_of_exact_size`,
`test_build_folds_matching_n_pre_count_equals_binomial_coefficient`,
`test_leave_one_out_fold_builder_is_untouched` (F1's regression guard).

## Component 3: endmember-sensitivity-robust keep rule

**File:** `experiments/f11_gate_v2_recalibration.py` (new)

No change to `fraction_sigma` or `apply_gate_v2`'s internals. Instead, the keep rule itself is
strengthened: compute τ once (from the point-estimate `e_v, e_b` on the extended CALIBRATION
folds), then score gate v2 on the TEST split **three times**, holding τ and every other parameter
fixed, varying only the endmembers:

1. point estimate (`e_v`, `e_b` as `estimate_endmembers` returns them)
2. `e_b` shifted by **+1 robust SD** (same robust SD F2 already computes in `endmember_sensitivity`
   — no new estimation code, reuse the existing function's output)
3. `e_b` shifted by **−1 robust SD**

**New keep rule (replaces F2's point-estimate-only version for F11):** window FAR ≤ α on the TEST
split in **all three** conditions, each reported with its own CI and denominator. This directly
operationalises cause 3: a gate that only "passes" when `e_b` happens to land at its point estimate
is not a robust detector, and F11's keep rule says so explicitly rather than reporting the
sensitivity as a footnote.

**Test:** `test_sensitivity_robust_keep_rule_requires_all_three_conditions_pass`,
`test_perturbed_e_b_reuses_endmember_sensitivity_output_not_a_new_estimate`.

## Component 4: τ recomputation on the extended, n_pre-matched pool

`experiments/f11_gate_v2_recalibration.py` recomputes τ exactly as F2 did (same formula: the
⌈(n+1)(1−α)⌉-th order statistic of the gate's own `f/σ_f` score, α=0.05 — never the raw F1-npz
score verbatim, for the same reason F2 rejected that: a different score function invalidates split
conformal), but now over the CALIBRATION half of the extended, n_pre=3-matched folds from
Component 2. τ is still a required, explicit argument of `apply_gate_v2` — no default, same B10
guard F2 already has, unchanged.

**Test:** reuse F2's `TestTauMutation` pattern against the new τ
(`test_tau_recomputed_from_extended_pool_differs_from_f2_tau`, confirming this is a genuine new
calibration, not an accidental reuse of F2's old value).

## Component 5: F12 — Wayanad v3 rerun

**File:** `experiments/f12_wayanad_v3.py` (new)

Identical structure to F7: same corrected AOI, same pre/post dates (2024-01-16/21/26 pre,
2024-12-06 post), same crop, F3's normalisation applied (per `configs/fix.yaml`
`f3_noise_v2.interaction_test`, unchanged), real E1/E8 SR on the production crop. The only change
from F7 is which τ is passed into `apply_gate_v2` (F11's recalibrated τ, point-estimate endmembers
— production itself is not re-run three times for the sensitivity conditions; that robustness
check lives in F11's placebo scoring, not in the one production map F12 produces).

Block-sum test rerun independently (allocation code is untouched, so this should still pass — if
it doesn't, that's a stop condition, not something to paper over). v1/v2(F7)/v3 comparison only on
shared, identically-defined quantities (reuse F7's `assert_well_defined_comparison` guard
verbatim — do not redefine it).

**Stop rule:** if F11 is BLOCKED (pool extension failed) or F11's new keep rule fails, F12 still
runs (block-sum is independent of the keep-rule outcome) but must carry F11's result forward
exactly as F7 carried F2's forward — no softening.

**Test:** `test_f12_block_sum_still_passes`, `test_f12_v1_v3_comparison_excludes_mismatched_definitions`
(reusing F7's guard test pattern).

## Pre-registration file: `configs/f11_f12.yaml`

New file, committed before F11 runs, containing: pool extension minimum (N≥4 hard floor, N=6
target), the stop rule for pool extension, the fold-construction formula (Component 2), the
three-condition sensitivity-robust keep rule (Component 3) with α=0.05 stated explicitly, the τ
recomputation method (Component 4, explicitly NOT the F1-npz-verbatim score), F12's date/crop
block (identical to `configs/fix.yaml`'s `f7_wayanad_v2` block, cited not redefined), and the
`config_sha256` field every `f11.json`/`f12.json` must carry (computed at commit time, never
hardcoded, never "pending").

## Error handling / stop rules summary

- Pool extension can't reach N≥4 in budget → F11 **BLOCKED**, F12 does not run with a mismatched
  n_pre (would reintroduce cause 1 rather than fix it).
- F11's three-condition keep rule fails (any of the three) → F11 **FAIL**, reported honestly with
  all three numbers; F12 still runs (block-sum is independent) but carries the FAIL forward.
- F12's block-sum regresses → stop, report; this would indicate a real bug introduced by this
  change, not an expected outcome.
- Either experiment is a valid, reportable outcome if it fails — this spec does not guarantee a
  PASS and treats a second honest FAIL as acceptable, publishable output.

## Testing summary (all pytest, real + synthetic fixtures as noted)

- Pool extension: a test that the cache file format matches the existing per-date dihedral NDVI
  schema exactly (so downstream code needs no format-specific branches).
- Fold construction: exact-size reference sets, correct combinatorial count, leave-one-out builder
  untouched (regression guard).
- Sensitivity-robust keep rule: requires all three conditions, reuses `endmember_sensitivity`'s
  existing robust-SD output rather than re-deriving it.
- τ: mutation test (changing τ changes the map — reuse F2's `TestTauMutation` pattern), and a test
  that F11's τ differs from F2's τ (proves this is a real recalibration).
- F12: block-sum test (reuse F2/F7's independent-paths pattern — never compare a quantity with
  itself), v1/v2/v3 comparison guard (reuse F7's `assert_well_defined_comparison`).
- Full existing suite (F1–F10) rerun to confirm zero regressions, since `fraction_sigma` and
  `apply_gate_v2` are explicitly NOT modified by this work.
