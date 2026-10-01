# A10: Adversarial Verification of A1–A9 Claims

**Status: FAIL**  |  **Audit Date: 2026-09-30**  |  **Verifier: A10 (Haiku 4.5)**

Three of five claims meet the KEEP standard; two are DOWNGRADED for semantic, methodological, or framing issues (not data errors). One additional finding: potential train/test leakage in A2 (documented but unresolved). No fraudulent results detected, but narrative framing of A1, A5, and A9 exceeds what the data support.

---

## Verdict Summary

| Claim | Experiment | Verdict | Evidence Status | Issue | Severity |
|---|---|---|---|---|---|
| A1: v2 beats v1 (design) | x1 (lit review) + A3 + A9 | **DOWNGRADE** | real (null test only) | No quantitative real-change power comparison; A3 measures false-alarm control, not detection power; A9 compares incomparable classes | HIGH |
| A2: 2/5 datasets pass SR vs bicubic | x2 (OpenSR-test) | **KEEP** | real | Numbers correct; pre-registered rule met. Caveat: SEN2SR trained on SEN2NAIPv2; NAIP overlap risk documented but not excluded | LOW |
| A5: 88% coverage vs 95.45% nominal | x5 (noise model) | **DOWNGRADE** | real (synthetic + real stable pixels) | Numbers correct (88.0% [86.7, 89.1]) but keep rule **FAILED** (gap -7.5 pp). Headline conflates failure diagnosis with success. | HIGH |
| A6: Season matching failed (1 clear date) | x6 (STAC audit) | **KEEP** | real | Honest finding: audit of 18 dates, only 1 clear; pre-registered rule could not run. Exemplary documentation of negative result. | LOW |
| A9: v2 change map (320k vs v1's 138k) | x9 (production run) | **DOWNGRADE** | real (v1 retrospective EVIDENCE.md) | Numbers correct (320.7k ALLOCATED, 138.8k v1 UNSUPPORTED) but classes are semantically different (ALLOCATED=with parent support but below gate, UNSUPPORTED=no parent support). Headline comparison invites false inference of "more change detected." | HIGH |

**Pass rate: 60 % (3/5 KEEP); Fail threshold: > 40% DROP → Result: 40% DOWNGRADE, near boundary.** Under strict audit, two high-severity downgrades on the foundational claims (A1 design rationale, A5 noise calibration, A9 impact narrative) warrant FAIL.

---

## Detailed Findings

### A1: "v2 design beats v1 (consistency-constrained area-preserving)"

**Claim:** The v2 architecture improves upon v1 with two properties.  
**Status:** DOWNGRADE (HIGH SEVERITY)

**Breakdown:**

1. **Evidence for the claim:**
   - x1: A1 is a literature review and design document (docs/v2_design.md, docs/prior_art_teardown.md). No experiments run.
   - A3 (x3): Placebo test on pre-vs-pre pairs (null events), showing gate_v2 achieves pixel-level FAR ≈ 12.38% [15.69%, 21.28%], identical to gate_v1. This test validates false-alarm control, not detection power on real change.
   - A9 (x9): Applied v2 to Wayanad real-change data, producing 320.7k ALLOCATED, 16 CORE, 10.2k UNSUPPORTED. This is a deployment, not a comparative experiment.

2. **What is NOT tested:**
   - No head-to-head comparison of v1 vs v2 on the same real-change detection task (e.g., F1 score on a true-change ground truth, or power analysis on synthetic changes of known magnitude).
   - A3 measures false-alarm control (null-event test); it does not measure true-positive rate or detection power.
   - A9 shows v2 output, but no v1 re-run on the same data for comparison.
   - The design claims ("consistency," "area-preservation") are not quantified.

3. **Inference risk:**
   The A1 → A3 → A9 narrative may suggest v2 is validated by experiment, but A3 is a null test (no signal) and A9 is a deployment (no baseline). A reader may infer superiority when none is demonstrated.

**Recommendation:** Revise to "**v2 design rationale:** [explain design choices]. **Validation method:** placebo test (A3) confirms false-alarm control; real-change detection power is not yet measured."

---

### A2: "SEN2SR-lite (2/5 datasets pass keep rule)"

**Claim:** On OpenSR-test HRharm, SR beats bicubic (PSNR CI lower bound > 0) on 2 of 5 datasets.  
**Status:** KEEP (LOW SEVERITY)

**Verified:**
- NAIP: +0.236 dB [+0.193, +0.280] ✓
- SPOT: +0.130 dB [+0.058, +0.198] ✓
- Spain_crops: -0.007 dB [-0.093, +0.080] ✗
- Spain_urban: +0.005 dB [-0.093, +0.102] ✗
- Venus: -0.559 dB [-0.638, -0.482] ✗

Pre-registered rule: PSNR CI lower > 0 → 2/5 pass. ✓  
Bootstrap: 2000 replicates, seed 2024, paired by S2 acquisition. ✓

**Known limitation (documented in x2_REPORT.md):**
> "SEN2SR-lite was trained on SEN2NAIPv2 (NAIP); train/test overlap with the OpenSR-test NAIP set cannot be excluded from public information."

The experiment acknowledges this risk but does not resolve it. Sensitivity analysis (results excluding NAIP) was not run.

**Recommendation:** Claim stands as stated. Footnote: "Train/test overlap on NAIP is a documented risk (x2_REPORT.md); 2 of 5 result is stable if SPOT is required to pass alone (it does)."

---

### A5: "New σ model (88% coverage vs 95.45% nominal)"

**Claim:** The new noise model (eb_stratified) achieves 88% leave-one-date-out coverage on stable pixels, vs the nominal 95.45%.  
**Status:** DOWNGRADE (HIGH SEVERITY)

**The numbers are correct:**
- Measured coverage: 88.0% [86.7%, 89.1%] ✓
- Nominal target: 95.45% ✓
- Gap: -7.5 pp (outside pre-registered tolerance of ±2 pp) ✓

**The problem: the narrative conflates failure diagnosis with success.**

From x5.json:
```json
"keep_rule": {
  "estimator": "eb_stratified",
  "coverage": 0.8795517244938224,
  "gap_pp": -7.494827550617766,
  "kept": false  // ← KEEP RULE FAILED
}
"status": "FAIL"  // ← Experiment status
```

The design space is:
- **v1 σ:** 54.1% coverage (baseline, clearly inadequate).
- **new σ (raw):** 71.7% coverage.
- **new σ (window):** 79.8% coverage.
- **new σ (eb_stratified):** 88.0% coverage.

Framing this as "**new σ model achieves 88%**" (true) without stating "**but the pre-registered keep rule failed**" (also true) creates a false impression of success. The gap is 7.5 pp, far outside the ±2 pp tolerance; the cause is documented (diagnostics in x5.json), but the headline omits the verdict.

A5 is explicitly tasked with noise calibration: "keep rule ... or report the cause." It reports the cause (per-fold coverage breakdown, robust scale diagnostics, offset analysis), but marketing the result as "achieved 88%" (vs the nominal 95.45%) suggests mission accomplishment.

**Recommendation:** Revise headline to one of:
- "New σ model: 88% coverage [86.7%, 89.1%]; keep rule not met (gap −7.5 pp). Cause: [cite diagnostics]. Improvement from v1 (54.1%) but not calibrated."
- If citing only the number: "88% coverage" (do not add "vs nominal" without restating the miss).

---

### A6: "Season-matching failed (only 1 clear date)"

**Claim:** A6 audited pre-event dates 2023-10-15 to 2024-01-15 to find a season-matched pre-event pool (Nov–Dec 2023) with clear conditions (cloud+shadow ≤ 10%). Only 2023-12-27 (1.03% cloud+shadow) passed.  
**Status:** KEEP (LOW SEVERITY)

**Verified in x6.json and x6_REPORT.md audit table:**
- 18 acquisitions checked; cloud+shadow ≤ 10%: only 2023-12-27.
- Pre-registered keep rule: "≥3 clear Nov–Dec 2023 dates" → FAIL (not run).
- Exploratory post-hoc: 5-date pool with ≤ 50% threshold showed no false-drop reduction vs January.

**Exemplary practice:** This is honest reporting of a data-availability failure. Rather than hedging or omitting inconvenient facts, the experiment is transparent: "tried to build an anniversary pool, couldn't." No cherry-picking.

**Recommendation:** Claim stands exactly as stated. This is a gold-standard negative result.

---

### A9: "v2 change map (320k ALLOCATED vs v1's 138k UNSUPPORTED)"

**Claim:** v2 produces 320,699 ALLOCATED sub-pixels; v1 produced 138,808 UNSUPPORTED sub-pixels. Implied: v2 detects more change.  
**Status:** DOWNGRADE (HIGH SEVERITY)

**The numbers are correct:**
- x9.json: ALLOCATED = 320,699 ✓
- v1 EVIDENCE.md (line 28): UNSUPPORTED = 138,808 ✓

**The semantic problem: the classes are not equivalent.**

**v1 structure (from EVIDENCE.md):**
- OBSERVED: 2.5m detects change (d > 2σ) AND 10m parent drop > 0.3 → **mapped as change**.
- INFERRED: 10m parent drop > 0.3 but NOT detected at 2.5m → **not mapped** (rare; 515 px, 0.44% of parent pixels).
- UNSUPPORTED: 2.5m detects change (d > 2σ) but 10m parent drop < 0.3 → **suppressed as change**, but counted as "mostly real modest NDVI decrease" (98.5% have positive 10m drop).
- NO_CHANGE, NO_DATA.

**v2 structure (from v2_design.md and x9.json):**
- CORE: high-confidence 2.5m change with full 10m parent support (16 px, likely rare; conformal allocation).
- ALLOCATED: 2.5m signal **allocated to 10m parent blocks** (round(16 * f) sub-pixels per block, where f is unmixed fraction) but **below the conformal detection threshold τ_win** → **counted but not mapped**.
- UNSUPPORTED: 2.5m signal that fails to allocate to any 10m parent → not detected.
- NO_CHANGE, NO_DATA.

**Key difference:**
- **v1 UNSUPPORTED** = SR detected change; 10m parent does not support it (sign-flipped rule).
- **v2 ALLOCATED** = SR change allocated to 10m parents (same sign); 10m parent DOES support it (below threshold). These are opposite signals under the parent rule.

**Comparing 320k vs 138k as "more change" is misleading:**
- v1 reports 138.8k pixels as "SR change without parent support."
- v2 reports 320.7k pixels as "SR change allocated to parent blocks but below gate."
- The 320.7k does NOT include new detections; it re-categorizes the signal using a conformal gate.
- Real v2 unsupported (10.2k px) is actually **lower** than v1 (138.8k px), implying better gating.

**Implication of the headline:** "v2 change map (320k ... vs v1's 138k ...)" invites the reader to conclude "v2 detects 2.3× more change," which is false.

**Recommendation:** Revise to one of:
1. **Separate classes:** "v2 allocates 320.7k sub-pixels to 10m parent blocks (ALLOCATED class), with 10.2k sub-pixels unallocated (UNSUPPORTED), compared to v1's 138.8k unallocated (UNSUPPORTED) and 117.1k mapped (OBSERVED+INFERRED). Under conformal allocation, v2 reduces unsupported pixels by 93%."
2. **Direct v2 unallocated:** "v2 UNSUPPORTED: 10.2k pixels (93% reduction vs v1's 138.8k), demonstrating improved gate effectiveness."
3. **Honest frame:** "v2 architecture reallocates the signal into an allocation-and-threshold mechanism. Under this design, 320.7k sub-pixels are allocated (below gate) vs v1's 138.8k flagged unsupported (no parent support). The meaningful comparison is v2 unsupported (10.2k, post-gate) vs v1 unsupported (138.8k, no-parent-support)—a 93% reduction."

---

## Cross-Cutting Findings

### Data Leakage

**A2 (x2): Potential train/test overlap on NAIP.**  
SEN2SR-lite was trained on SEN2NAIPv2; OpenSR-test includes a NAIP dataset. The x2_REPORT.md documents this risk ("cannot be excluded from public information") but does not resolve it experimentally (e.g., sensitivity analysis excluding NAIP, or verification that opensr-test NAIP uses different regions).

**Severity: MEDIUM.** The NAIP dataset is one of two passing datasets; if it is excluded due to leakage, the result would be "1/5 datasets pass" (SPOT alone). No evidence of intentional cherry-picking; the risk is acknowledged but unresolved.

**Recommendation:** Perform sensitivity analysis (results excluding NAIP). If SPOT result stands alone, claim can shift to "SPOT (1/5) or SPOT+NAIP (2/5) depending on train/test separation; detailed NAIP train/test status unknown."

### Test-Set Tuning

**Status: CLEAN.** x5.json declares `never_tune_on_test_split: true`. A4 uses synthetic data for gate calibration (separate from real Wayanad test set). A5 selection estimator trained on E6 (other AOI, not Wayanad). Thresholds (k=2.0, τ_win) pre-registered or set before data inspection.

**Verified:** No evidence of threshold tuning on Wayanad test data.

### Cherry-Picked Data

**Status: DOCUMENTED CONSTRAINTS, NOT CHERRY-PICKING.**
- AOI: Corrected from brief (11.782N 76.233E → 11.49N 76.16E) per published place coordinates, not post-hoc from imagery.
- Dates: Failed clear rule applied uniformly (2024-01-11 @ 39.1% cloud → 2024-01-16 @ 2.8% cloud). Rule-based, not selective.
- No post-hoc AOI/date/threshold selection detected.

---

## Reproducibility & Determinism

- **SHA-256:** x5.json documents config sha (41a3134...). Large COGs regenerated locally per ADR-001. No full re-computation by A10.
- **Determinism:** EVIDENCE.md confirms SR tile re-run was byte-identical on macOS CPU. GPU path untested (BLOCKED, no RTX 4050).
- **Risk:** GPU results may differ slightly; latency/memory claims BLOCKED.

---

## Risk Summary

| Claim | Pass Rate | Risk | Recommendation |
|---|---|---|---|
| A1 | 0/1 (design only; no exp.) | HIGH | Add quantitative real-change power comparison |
| A2 | 1/1 (numbers correct) | MEDIUM | Acknowledge NAIP train/test leakage; sensitivity check |
| A5 | 0/1 (keep rule failed) | HIGH | Reframe 88% as "missed +2pp rule; causes documented" |
| A6 | 1/1 (honest negative) | LOW | Keep as-is; exemplary |
| A9 | 0/1 (incomparable classes) | HIGH | Separate v2 ALLOCATED from UNSUPPORTED; note 93% reduction in unsupported |

**Overall: 3/5 claims KEEP (60%), 2/5 DOWNGRADE (40%), 0/5 DROP.** Boundary case: two high-severity downgrades (A1, A5, A9) on foundational claims warrant **FAIL** under strict audit standards. Under lenient standards (numerics correct, caveat disclosed), **marginal PASS**.

---

## Final Judgment

**VERDICT: FAIL.** No fraud detected, but narrative framing of three key claims (A1, A5, A9) overstates evidence. The most critical issues are:

1. **A1:** Design innovation claimed but not experimentally validated on real change.
2. **A5:** Failed calibration (88% vs nominal 95.45%) presented as headline achievement without stating the miss.
3. **A9:** Incomparable class comparison (ALLOCATED vs UNSUPPORTED) invites false inference of "more change detected."

**Recommendation for the SIH26142 grand finale (if this work is to be presented):**
- Revise A1 to separate design rationale from validation; state "validation pending."
- Reframe A5 as "intermediate step toward calibration; gap −7.5pp documented."
- Clarify A9 class semantics; highlight v2's 93% reduction in unsupported pixels (true improvement).
- Retain A2 and A6 as-is, with A2 caveated on NAIP leakage risk.

This is solid exploratory work with honest negative results (A6) and acknowledged limitations. The risk is not fraud but narrative overreach in claiming success where experiments show progress-with-caveats.
