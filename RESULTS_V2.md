# TrustSR — RESULTS_V2: the verified claim table

**Integrator: F10** · **Verifier: F9** · **Branch: `task/f10`** (based on `experiments/fix-v2`) · **Date: 2026-10-01**

Replaces `RESULTS_EXCEPTIONAL.md` and `experiments/results/x11.json`, both INVALIDATED (B13, B14). See
`experiments/results/INVALIDATED.md` for the full quarantine and what supersedes what.

**`config_sha256` = `18cce7b1ea88855bc054b42e6cbe3219eef73b2909eb091d07d5ada3df170970`** — the sha256 of
`configs/fix.yaml`, the pre-registration every claim below traces back to. Computed by F10 with
`hashlib.sha256(open('configs/fix.yaml','rb').read()).hexdigest()`, not copied from any result file or
brief. It matches all six `f*.json`.

Every arithmetic figure in this document was recomputed by F10 in code (numerator/denominator, sums,
ratios, unit conversions, and the row tallies) before being written down. Nothing here was eyeballed.

---

## Bottom line, stated as the data support it

**The headline scientific result of this wave is negative.**

1. **Gate v2 does not meet its pre-registered false-alarm rule.** Test-split window FAR is **0.0652**
   against a required **≤ 0.05**. F2 is **FAIL**. F7, which deploys that gate on the real Wayanad scene,
   is **FAIL** for the same reason and carries the failure forward. Neither status is softened anywhere
   below.
2. **Sub-pixel allocation hurts once the block fraction has to be estimated from Sentinel-2.** Under the
   realistic (unmixing) fraction, allocation is **−0.313 IoU [−0.344, −0.282]** *worse* than doing no
   allocation at all (n=119, better on 3/119 images). The bottleneck is the fraction, not the ranking.
3. **SR does not beat bicubic on PSNR pooled.** **−0.097 dB [−0.157, −0.037]** over 178 images; removing
   the one dataset with unexcludable train/test overlap (NAIP) makes it *more* negative, **−0.275 dB**.
   SR is also more hallucination-prone than bicubic on **every** one of the five datasets.
4. **The one positive result is small and leakage-dependent.** F5's headline **+0.0081 IoU** for SR-informed
   ranking holds only *with NAIP included*; **excluding NAIP it is +0.0016 [−0.0026, +0.0053] at n=56 —
   not distinguishable from zero.** These two numbers must never be separated.

What *did* hold: the placebo test is a genuine null test (F1, PASS), the noise model v2 meets its coverage
keep rule (F3, PASS, 94.2% vs nominal 95.45%, gap −1.27 pp), the honest A2 re-report is complete (F6, PASS),
and F5 ran end to end (PASS, verdict qualified as above).

F9 audited 49 claims and went looking specifically for the defects that invalidated the x-wave — a
fabricated number, a test comparing a quantity with itself, a threshold loaded but never applied, a
threshold moved after seeing a result — and **found none of them**.

---

## The cross-cutting finding: calibrated, but it does not generalise

This is the substantive scientific finding of the wave, and it is not an arithmetic slip.

| split | F3-normalised arm | un-normalised arm | vs α = 0.05 |
|---|---|---|---|
| **calibration** (even tiles, in-sample) | **0.04893** (48/981) | **0.04893** (48/981) | **≤ α, in both arms** |
| **test** (odd tiles, out-of-sample) | **0.0652** (64/981) | **0.0540** (53/981) | **> α, in both arms** |

Split conformal pins the calibration-split exceedance rate near α *by construction*, and it does exactly
that. The gate is **properly calibrated**. It then **fails on the test split in both normalisation arms** —
so the FAIL is robust to F2's choice of which arm to headline (F2 headlines the *worse* arm, per
`fix.yaml`'s `interaction_test`).

The three placebo folds share **one crop** with an interleaved checkerboard split, which is precisely the
situation `fix.yaml`'s pre-registered `units.window_caveat` warned about. **This is a real exchangeability
gap across the spatial split, not a calibration bug and not a rounding error.** Fixing it requires
calibration windows that are genuinely exchangeable with the test windows — a different crop or a
different AOI — not a threshold adjustment.

---

## Negative and failed results, in full

These are not footnotes. They are the load-bearing results of this wave.

| # | result | number | denominator | file |
|---|---|---|---|---|
| 1 | **F2's window FAR keep rule FAILED** | **0.0652 [0.0478, 0.0850]** vs required **≤ 0.05** | 981 test-split windows (num 64) | `f2.json` |
| 2 | **F7 status FAIL** — the gate it deploys failed its own FAR rule; the 1.60 km² map is *what this gate configuration produces*, **not a validated detection** | — | 5,242,880 px crop, 5,036,924 valid | `f7.json` |
| 3 | **F7's 0.0652 is a LOWER BOUND on the true FAR** — τ calibrated at n_pre=2, applied at n_pre=3; σ_f(2)/σ_f(3) = 1.06066, so the production score runs ~6.07% hot and the gate is *more* trigger-happy than τ was set for | ratio **1.0606601717798212** | — | `f7.json` |
| 4 | **F5's headline collapses without NAIP** | **+0.0016 [−0.0026, +0.0053]**, CI spans zero, 30/56 wins | **n = 56 images** | `f5.json` |
| 5 | **F5: allocation is worse than not allocating**, with a realistically estimated fraction | **−0.313 IoU [−0.344, −0.282]**, better on 3/119 | n = 119 images | `f5.json` |
| 6 | **F6: SR loses to bicubic on PSNR pooled** (the B14 omission) | **−0.097 dB [−0.157, −0.037]** | n = 178 images | `f6.json` / `x2.json` |
| 7 | **F6: Venus is badly negative** (omitted entirely by `RESULTS_EXCEPTIONAL.md`) | **−0.559 dB [−0.638, −0.482]** | n = 59 images | `f6.json` / `x2.json` |
| 8 | **F6: excluding NAIP makes it worse, not better** | **−0.275 dB [−0.345, −0.203]** | n = 116 images | `f6.json` |
| 9 | **F6: SR is more hallucination-prone than bicubic on all 5 datasets** (opensr-test `ha_metric`) | 5/5 worse | 178 images / 5 datasets | `f6.json` |
| 10 | **A5's coverage keep rule was MISSED** (original x-wave, honestly reported at the time) | 87.96% [86.72, 89.09], **gap −7.49 pp** vs ±2 pp tolerance; `kept: false` | 7,073,061 stable 2.5 m px | `x5.json` |
| 11 | **A6: both pre-registered keep rules FAILED** — only 1 clear Nov–Dec 2023 date, so the anniversary test never ran; post-composite validity 0.806 < 0.90 | 1 clear date of 18 audited; 0.806 = 31,456/39,019 | 18 acquisitions; 39,019 footprint px | `x6.json` |
| 12 | **F2: λ = 0.0**, so the neighbour-agreement term contributes **nothing** to any reported result — no claim that it helps is supported | fitted optimum at grid boundary | 6-point grid | `f2.json` |
| 13 | **F7: the E5 cascade saves 0 forward passes on this crop** (an honest null) | 0 of 1,120 (0.0%) | 35 tiles × 8 runs × 4 dates | `f7.json` |
| 14 | **F2: the gate is highly sensitive to `e_b`** — +1 robust SD multiplies flagged pixels by ~8 | baseline 3,471 px, max \|Δ\| 24,298 | 1 fold | `f2.json` |
| 15 | **F1: `rule_10m`'s pool (N=6) is double the SR gates' (N=3)**, so its FAR row is not on equal footing | — | — | `f1.json` |

**Blocked, not estimated:** F5's `alloc_finetuned_sr` (F4's fine-tune never landed — pretrained output was
never substituted under that label), F5's heavy/Colab SR arm (NOT RUN), F1's 2023-12-27 proxy pair (not
cached for the corrected AOI, not fetched), F7's 4th pre date 2023-12-27 (same reason, no substitute
invented), and gate v2 in F1's own table (BLOCKED at F1's run time, per the pre-registration).

---

## Claim table

All 48 claims that F9 verdicted **KEEP** or **DOWNGRADE**, one row per F9 claim, in F9's order.

- **Verdict** is F9's, copied from `experiments/results/f9.json` `claims_audited[].verdict` — not F10's opinion.
- **DOWNGRADE** never means "the arithmetic is wrong". In all five cases the arithmetic reproduced exactly;
  what is downgraded is a *mechanism claim* or the *interpretive weight* of a number. Each row says which.
- **Baseline** / **delta** are left as `—` where the source experiment reports no comparator. F10 did not
  invent one.
- **Evidence** labels: `real` = real Sentinel-2 / real HR imagery; `real (recomputation)` = no new inference,
  recomputed from a stored real-data artefact; `real + synthetic fixture` = real result plus a constructed
  fixture used to prove a mechanism; `identity` = definitional/arithmetic identity, not an empirical result;
  `declared` = a blocked or superseded status, declared rather than estimated; `docs` = a documentation defect.

### F1 — the real placebo (null-event) FAR test · status **PASS**

| task | claim | baseline | delta | 95% CI | denominator | evidence | verdict | file |
|---|---|---|---|---|---|---|---|---|
| F1 | `rule_10m` pixel FAR = **0.000265** | ungated SR 0.4033 | **−0.4030** | [0.0000654, 0.000563] | 7,232,544 valid 2.5 m px (pool N=6) | real | KEEP | `f1.json` |
| F1 | `rule_10m` window FAR = **0.00561** (uses this gate's own flag map — B3 fixed) | ungated SR 0.8532 (window) | — | [0.00201, 0.01074] | 1,962 windows (pool N=6) | real | KEEP | `f1.json` |
| F1 | `ungated_S_v1_sigma` pixel FAR = **0.4033** — the ungated baseline every reduction is measured against | — (is the baseline) | — | [0.3548, 0.4593] | 3,617,880 valid 2.5 m px (pool N=3) | real | KEEP | `f1.json` |
| F1 | `ungated_S_v1_sigma` window FAR = **0.8532** | — (is the baseline) | — | [0.8183, 0.8864] | 981 windows | real | KEEP | `f1.json` |
| F1 | `gate_v1` pixel FAR = **0.000495**, window FAR = **0.00714** | ungated SR 0.4033 / 0.8532 | −0.4028 (pixel) | [0.0000854, 0.001097] / [0.00104, 0.01437] | 3,617,880 px / 981 windows (numerator 1,792) | real | KEEP | `f1.json` |
| F1 | `gate_v1` and `gate_v1_with_a5_sigma` are numerically **identical on the real crop by design, not by code aliasing** (`OBSERVED ∪ INFERRED = parent` wherever d and σ are finite) | — | 0 px difference on the real crop | — | 1,792 flagged px of 3,617,880; 5 GATES entries = 5 distinct `__code__` objects; F9's mirrored fixtures diverge by 2,048 px | real + synthetic fixture | KEEP | `f1.json` |
| F1 | FAR reduction vs ungated SR (pixel, test split): `rule_10m` **−0.4030**, `gate_v1` **−0.4028**, `gate_v1_with_a5_sigma` **−0.4028** | ungated SR 0.4033 | as stated | — | 7,232,544 px (N=6) / 3,617,880 px (N=3) — **not equal footing, disclosed** | real | KEEP | `f1.json` |
| F1 | The placebo/calibration code path **cannot load the post-event date** (B1 hard guard); malformed date spellings fail closed | — | — | — | **16 of 16** adversarial load attempts behaved correctly, incl. the post date monkeypatched into `SR_POOL` on τ's own path | real (attack test) | KEEP | `f1.json` |
| F1 | `gate_v2` = **BLOCKED** in F1's 5-gate table, per the pre-registration; never a copy of `gate_v1` | — | — | — | — | declared | KEEP | `f1.json` |

> **Do not reprint F1's `gate_v2 = BLOCKED` row as current.** It was correct at F1's run time and is now
> **superseded**: after F2 landed, `placebo_v2.gate_v2` is the real gate. `f1_REPORT.md` still describes it
> as a stub that raises `NotImplementedError`.

### F2 — gate v2 rebuilt · status **FAIL**

| task | claim | baseline | delta | 95% CI | denominator | evidence | verdict | file |
|---|---|---|---|---|---|---|---|---|
| F2 | Block-sum test: **max \|deviation\| = 0** → PASS. Non-tautological, proven by sabotage (B7 recreated ⇒ both anti-tautology tests fail) | — | 0 | — | 983,040 block-evaluations | real | KEEP | `f2.json` |
| F2 | **Window FAR keep rule NOT MET → F2 status FAIL**: **0.0652** on the test split vs required ≤ 0.05. Robust to the arm choice (un-normalised 0.0540 also fails) | required ≤ 0.05 (α) | **+0.0152 over the ceiling** | [0.0478, 0.0850] | **981 test-split windows** (num 64, 273 blocks) | real | KEEP | `f2.json` |
| F2 | Pixel FAR = **0.0036** | — | — | [0.0022, 0.0054] | 3,617,880 valid 2.5 m px (num 13,167) | real | KEEP | `f2.json` |
| F2 | **τ = 3.889118** = T₍₉₃₃₎ of the calibration-split placebo window scores, k = ⌈(n+1)(1−α)⌉, α = 0.05. Null-event data only | — | — | — | n = 981 calibration windows; k = 933 confirmed | real | KEEP | `f2.json` |
| F2 | **τ is used, not merely loaded** (B10 fixed) — required positional arg, non-finite raises; sabotaging the applying line makes both mutation tests fail | — | — | — | 2 sabotage-verified mutation tests | real | KEEP | `f2.json` |
| F2 | Endmembers **estimated from data, not hard-coded** (B12 fixed): `e_v` from **178,141 px**, `e_b` from **332 px**; `estimate_endmembers` takes no label-like argument | — | — | — | 178,141 px / 332 px; 186,987 px excluded by disks+footprint | real | KEEP | `f2.json` |
| F2 | **The gate is HIGHLY sensitive to `e_b`**: +1 robust SD multiplies the flagged pixel count by **~8×**. The single largest uncertainty in the gate; `f` is **not a calibrated area** | 3,471 flagged px | max \|Δ\| **24,298 px** (700%) | — | 1 fold, τ and λ held fixed | real | KEEP | `f2.json` |
| F2 | **λ = 0.0**, fitted on F1's calibration split only by split-half dihedral Jaccard. Genuinely wired (sabotage-verified) — the optimum simply lies at the grid boundary. **Consequence: the neighbour-agreement term contributes nothing to any reported result** | — | — | — | 6-point grid [0.0 … 0.5]; Jaccard highest at 0, falling monotonically | real | KEEP | `f2.json` |
| F2 | The `e_b` NDVI-ceiling escalation (0.20 → 0.30) — **the stated "made the keep rule HARDER" mechanism is NOT ESTABLISHED** (see the deviations section below) | — | mean `f` 0.395 → 0.821 (2.08× inflation) **but** mean score s 20.36 → 10.59 | — | 4 mixing-line lengths measured by F9; full 0.20–0.50 ceiling table published pre-hoc (8 / 80 / 332 / 1,182 / 2,554 / 5,094 / 9,455 qualifying px) | real | **DOWNGRADE** | `f2.json` |
| F2 | Recomputing τ with gate v2's **own** score (rather than F1's verbatim export) **made the keep rule HARDER, not easier** — and it is the choice that cost F2 its keep rule | verbatim τ = **119.90589141845703** ⇒ flags ~nothing ⇒ window FAR = 0 ⇒ **vacuous PASS** | recomputed τ = 3.889118 ⇒ **0.0652 ⇒ FAIL** | — | F1's npz (sha256 `53a2fe8a…fd2679e4`, independently recomputed); F1's scores span −25.86 to 255.60 vs gate v2's order 1–10 | real | KEEP | `f2.json` |

### F3 — noise model v2 · status **PASS**

| task | claim | baseline | delta | 95% CI | denominator | evidence | verdict | file |
|---|---|---|---|---|---|---|---|---|
| F3 | Normalised pooled LODO coverage of \|d\| ≤ 2σ = **94.2%**, gap **−1.27 pp** vs nominal 95.45% → **keep rule KEPT** (within ±2 pp) | nominal 95.45% | **−1.27 pp** | [93.3, 95.0] | **7,074,856 px** (num 6,663,111; 2,358,286 stable px/fold × 3 folds) | real | KEEP | `f3.json` |
| F3 | Same σ method **without** normalisation = **85.8%**, gap **−9.65 pp** — fails the rule, and shows the normalisation does real work rather than being cosmetic | nominal 95.45% | **−9.65 pp** | [84.5, 87.0] | 7,074,856 px (num 6,069,894) | real | KEEP | `f3.json` |
| F3 | Estimator (`eb_stratified`) selected on **E6 only** — no Wayanad pixel — so the Wayanad evaluation is out-of-sample for the choice; paired NLL diff vs runner-up **−0.160**, CI excludes 0 | runner-up (`window`) | **−0.160** | [−0.251, −0.083] | E6 12-date cache (other AOI), 100 seeded draws | real | KEEP | `f3.json` |
| F3 | Change-preservation: a +0.30 NDVI drop injected into real stable px is recovered at ratio **1.000** (fitted offset moved by −0.01% of the drop) | injected +0.30 | recovered +0.3000 | — | **1,600 injected px** of 2,358,286 stable px | **synthetic injection on real pixels** — bounds sensitivity to a minority-pixel change, **not** behaviour on a real event | KEEP | `f3.json` |
| F3 | median(d) = 0 for every LODO fold after normalisation — **kept as a stated identity, not as evidence**. `offset_held` *is* the median residual, so this is definitional; F3 names it as such and rests the keep rule on **coverage** instead (σ comes from across-date and dihedral variance, never from d) | — | 0 by construction | — | 3 folds | **identity** | KEEP | `f3.json` |
| F3 | `f3_REPORT.md` cites `config_sha256` **`192345cd156583c6…`** — **STALE and truncated**. That is the hash at F3's own YAML-fix commit `298e214`; the merged branch's file hashes to `18cce7b1…`. `f3.json` **was** amended (commit `0059c2f`, two lines only); the report was not | correct hash `18cce7b1…f170970` | — | — | — | **docs** | **DOWNGRADE** | `f3_REPORT.md` |

> F3's headline coverage table has no *n* column of its own (the 2,358,286 figure sits in prose immediately
> above it, and `f3.json` carries `n_px` on every node). A documentation gap, not a defect — but cite the
> denominator from `f3.json`, not from the table.

### F5 / A8 — HR-referenced sub-pixel mapping · status **PASS** (verdict qualified)

> **Required caveat, before any F5 number:** static land-cover boundaries (vegetation / water) are a
> **proxy** for change boundaries, **not a landslide label**. Nothing in F5 licenses a claim about
> landslide-specific accuracy or about the Wayanad map.

| task | claim | baseline | delta | 95% CI | denominator | evidence | verdict | file |
|---|---|---|---|---|---|---|---|---|
| F5 | Headline paired IoU(`alloc_pretrained_sr`) − IoU(`alloc_bilinear`), oracle fraction, all 4 datasets = **+0.0081** — **with NAIP included, and only ~1% of IoU on a base of 0.837.** Must never be quoted without the next row | `alloc_bilinear` 0.8286 IoU | **+0.0081** | [+0.0035, +0.0141] | **n = 118 images**, SR wins 87/118 | real | KEEP | `f5.json` |
| F5 | **Headline EXCLUDING NAIP = +0.0016 — not distinguishable from zero.** NAIP is exactly the set where SEN2SR-lite train/test overlap cannot be excluded | `alloc_bilinear` | **+0.0016** | **[−0.0026, +0.0053]** (spans zero) | **n = 56 images**, SR wins 30/56 | real | KEEP | `f5.json` |
| F5 | The **only** positive surviving NAIP removal is the **0–20 m feature-width stratum**: **+0.0077**. Effect is ~15× smaller for features wider than 150 m — the direction the sub-pixel hypothesis predicts. Absolute IoU in that bin is low (0.272 SR / 0.255 bilinear) | `alloc_bilinear` 0.255 IoU in-bin | **+0.0077** | [+0.0018, +0.0132] | **n = 47 images** (NAIP excluded) | real | KEEP | `f5.json` |
| F5 | **Under the REALISTIC (unmixing) fraction, `alloc_pretrained_sr` is −0.313 IoU WORSE than `blocky_10m`** — with a fraction actually estimated from Sentinel-2, **sub-pixel allocation loses to doing no allocation at all**, whichever ranking it uses. Mechanism: per-block fraction MAE 0.400, within-image Pearson r 0.294 | `blocky_10m` 0.7596 IoU | **−0.313** | [−0.344, −0.282] | **n = 119 images**, better on only **3/119** | real | KEEP | `f5.json` |
| F5 | The water (NDWI > 0) class is **null in both directions; no claim is made** | — | +0.0146 (all) / −0.0203 (no NAIP) | [−0.0479, +0.0664] / [−0.0970, +0.0295] | **n = 44 of 119** images contain any NDWI>0 px | real | KEEP | `f5.json` |
| F5 | `alloc_bilinear` and `alloc_pretrained_sr` differ **only in the ranking field**, so the headline is a genuine ranking comparison and **not a quantity compared with itself** (identical fraction, identical allocator; a self-comparison would return exactly 0.0) | — | non-zero on 87 of 118 images | — | 118 paired images | real | KEEP | `f5.json` |
| F5 | `alloc_finetuned_sr` **BLOCKED** (F4 has not landed) and heavy/Colab SR **NOT RUN**; **pretrained output was never substituted** under the fine-tuned label. `venus` correctly excluded (5 m HR cannot carry a 2.5 m truth mask) | — | — | — | no fine-tuned number appears anywhere in `f5.json` | declared | KEEP | `f5.json` |

### F6 — honest A2 reporting (corrects B14 selective reporting) · status **PASS**

| task | claim | baseline | delta | 95% CI | denominator | evidence | verdict | file |
|---|---|---|---|---|---|---|---|---|
| F6 | **Pooled PSNR delta vs bicubic across all 5 datasets = −0.097 dB** — the number `RESULTS_EXCEPTIONAL.md` omitted (B14). **SR does not beat bicubic pooled.** | bicubic 35.028 dB | **−0.097 dB** (SR 34.931 dB) | [−0.157, −0.037] | **n = 178 images** | real (recomputation from `x2.json`) | KEEP | `f6.json` |
| F6 | Per-dataset table, **all 5 including the 3 omitted**: naip **+0.236** (n=62), spot **+0.130** (n=9), spain_crops **−0.007** (n=28), spain_urban **+0.005** (n=20), **venus −0.559** (n=59) | bicubic 36.617 / 33.276 / 32.895 / 29.247 / 36.597 dB | as stated | [+0.193,+0.280] / [+0.058,+0.198] / [−0.093,+0.080] / [−0.093,+0.102] / **[−0.638,−0.482]** | 62 / 9 / 28 / 20 / 59 images = 178 | real (recomputation from `x2.json`) | KEEP | `f6.json` |
| F6 | **NAIP-excluded sensitivity = −0.275 dB.** Removing the possibly-leaking dataset makes the pooled result **MORE negative** — it does not rescue the PSNR claim | bicubic 34.179 dB | **−0.275 dB** | [−0.345, −0.203] | **n = 116 images** | real (recomputation from `x2.json`) | KEEP | `f6.json` |
| F6 | opensr-test `ha_metric`: **SR is more hallucination-prone than bicubic on EVERY one of the 5 datasets**, including the two where SR wins on PSNR. This is the honest motivation for a trust gate | bicubic 0.0375 / 0.0341 / 0.0642 / 0.0645 / 0.1378 | SR 0.0598 / 0.0600 / 0.0848 / 0.0874 / 0.2049 — **worse 5/5** | — | 5 datasets, 178 images | real (recomputation from `x2.json`) | KEEP | `f6.json` |
| F6 | `f6_REPORT.md` cites `config_sha256` **`7ee83389…0111e7e6`** — **STALE**. That is the hash of `configs/fix.yaml` as originally pre-registered (`020880f`), the file F6 legitimately computed against, but the merged branch's file hashes to `18cce7b1…`. `f6.json` **was** amended (commit `6d2f722`, two lines only); the report was not | correct hash `18cce7b1…f170970` | — | — | — | **docs** | **DOWNGRADE** | `f6_REPORT.md` |

### F7 — Wayanad v2 production rerun · status **FAIL**

> **⚠ THE DETECTOR THIS MAP COMES FROM FAILED ITS OWN KEEP RULE.** Window FAR **0.0652 [0.0478, 0.0850]**
> against a pre-registered **≤ 0.05**, and the true FAR is **worse** than that (τ/n_pre mismatch). Every
> count below describes **what this gate configuration produces** on the real Wayanad scene. **None of it
> is a validated detection and none of it may be quoted as one.**

| task | claim | baseline | delta | 95% CI | denominator | evidence | verdict | file |
|---|---|---|---|---|---|---|---|---|
| F7 | Per-class report (2.5 m, 6.25 m²/px): NO_CHANGE **4,733,514** px · CORE **17,648** · ALLOCATED **239,122** · UNSUPPORTED **46,640** · NO_DATA **205,956**, with m² and km² | — | — | — | classes sum to **5,242,880 px = 2560 × 2048** exactly; **5,036,924 valid** — take class percentages against valid px, not the total | real | KEEP | `f7.json` |
| F7 | Mapped change (v2) = CORE ∪ ALLOCATED = **256,770 px = 1.604812 km²** | — | — | — | 5,036,924 valid px | real | **DOWNGRADE** | `f7.json` |
| F7 | v1 vs v2 mapped change area: **0.734900 → 1.604812 km²** (+0.869912), **IoU 0.2448**; in v1 not v2 **43,960 px**, in v2 not v1 **183,146 px**. Well defined: same quantity, grid, crop, dates and denominator on both sides | v1 0.734900 km² (117,584 px) | **+0.869912 km²** (+139,186 px) | — | 5,036,924 valid px | real | KEEP | `f7.json` |
| F7 | **UNSUPPORTED is NOT compared across v1 and v2** because the gating predicates differ; the allow-list guard `assert_well_defined_comparison` (default deny) enforces it. Both reported separately with their definitions: v1 **138,808 px** (0.867550 km²), v2 **46,640 px** (0.291500 km²) — **never differenced** | — | **no difference, ratio or IoU exists** between them | — | — | real | KEEP | `f7.json` |
| F7 | Block-sum test on the **real production map**: **max \|deviation\| = 0** → PASS. Re-executed by F9 and non-tautological under sabotage | — | 0 | — | **327,680 10 m blocks** (81,549 with round(16f) > 0) | real | KEEP | `f7.json` |
| F7 | **τ is used on the production map**, not merely loaded: τ×0.5 → **287,477** px, τ → **256,770** px, τ×2 → **123,803** px — strictly monotone, and the fitted-τ count equals the headline mapped-change area exactly | — | — | — | 5,036,924 valid px; 810 of 1,273 valid windows detected | real | KEEP | `f7.json` |
| F7 | **1,120 real SR forward passes** (35 tiles × 8 dihedral × 4 dates) and the run reproduces the cached product exactly (**max abs diff 0.0** on all 4 dates). The SR ranking field varies within 100% of 4×4 blocks, so x9's `np.repeat` construction (B9) is excluded and would raise | — | — | — | 35 × 8 × 4 = 1,120 | real | KEEP | `f7.json` |
| F7 | **The τ/n_pre mismatch (calibrated at n_pre=2, applied at n_pre=3) pushes the TRUE false-alarm rate ABOVE the already-failing 0.0652.** σ_f = √(Var(a)(1/n_pre + 1)) ⇒ σ_f(2)/σ_f(3) = **1.0606601717798212**, so the production score is ~**6.07% higher** than at calibration. Measured, disclosed, **not corrected** | 0.0652 (as calibrated) | true FAR **> 0.0652** | — | at a threshold rescaled to match calibration sensitivity the flagged count would be 246,911 instead of 256,770 | real | KEEP | `f7.json` |
| F7 | E5 cascade saves **0 forward passes (0.0%)** on this scar-centred crop — an honest null about this crop, not about the cascade; **960 of 3,872 (24.8%)** on the full AOI, **where SR was NOT run** | full re-run 1,120 passes | **0 saved** (crop) / 960 saved (full-AOI selection count) | — | 35 tiles (crop) / 121 tiles, 30 skipped (full AOI) | real (crop) / **selection count only, SR not run** (full AOI) | KEEP | `f7.json` |
| F7 | **3,946 of 4,412** boundary 10 m blocks (**89.4%**) have a 2.5 m allocation differing from the blocky parent | blocky 10 m parent | 3,946 blocks differ | — | **4,412 boundary blocks** of 314,671 scored (NO_DATA-touching blocks excluded from both sides) | real | **DOWNGRADE** | `f7.json` |
| F7 | **F7 overall status = FAIL**, because the gate it runs failed its own FAR keep rule. Stated in the first line, in a blockquote above every number, and in the figure caption; F7's own test asserts it appears within the first 1,500 characters | required ≤ 0.05 | FAIL | — | — | real | KEEP | `f7.json` |

**Why the two F7 downgrades, precisely** — in both cases the arithmetic is exact and reproduced to the last digit:

- **Mapped change 1.604812 km²** is downgraded **as a substantive claim**, not for any arithmetic defect. The
  detector that produced it failed its pre-registered FAR rule, the true FAR is worse still, and F5's
  realistic-fraction result (−0.313 IoU) is a direct prior against the sub-pixel allocation. Quotable only
  as *"what this gate configuration produces"*.
- **89.4% of boundary blocks differ from the blocky parent** is downgraded because **differing from a blocky
  parent is not evidence of being better than one** — and F5 measured that under exactly F7's
  realistic-fraction regime the allocation is 0.313 IoU *worse* than blocky. **It must never be presented as
  "SR adds information".**

Equally: the **block-sum PASS is a plumbing invariant**, not a scientific result. It verifies that the
allocator and class map agree on how many sub-pixels a block got. **F5's −0.313 IoU is the mapping-quality
evidence, and it is negative.**

---

## Sound x-wave results, not part of F9's 49

These experiments were never quarantined and may be cited freely (`experiments/results/INVALIDATED.md`).
They sit outside F9's 49-claim audit because F9 audited the fix wave (F1–F7); they are included here so the
human has one table, and each is labelled with its own original status.

| task | claim | baseline | delta | 95% CI | denominator | evidence | status | file |
|---|---|---|---|---|---|---|---|---|
| A5 | Noise model v1 coverage of \|d\| ≤ 2σ improved to **87.96%** — but the **pre-registered keep rule was MISSED**: gap **−7.49 pp** against a ±2 pp tolerance, `keep_rule.kept = false`. Honestly reported as a FAIL at the time; **superseded by F3**, which meets the same rule at 94.2% (gap −1.27 pp) | v1 σ 54.12% [53.30, 54.91]; nominal 95.45% | **−7.49 pp vs nominal** | [86.72, 89.09] | **7,073,061 stable 2.5 m px** (pooled LODO, 3 folds) | real | **FAIL (keep rule missed)** | `x5.json` |
| A6 | **Both** pre-registered keep rules **FAILED**. (a) Anniversary pre-dates: only **1** clear Nov–Dec 2023 date (2023-12-27, 1.0% cloud+shadow) of **18** acquisitions audited, so the ≥3-date rule was unmet and the false-drop test **never ran**. (b) Post composite: footprint validity **0.806** (31,456/39,019 px) against a required **≥ 0.90**, gap 99 d | (a) ≥3 clear dates required; (b) ≥0.90 validity, ≤99 d gap | (a) −2 dates; (b) −0.094 validity | — | (a) 18 acquisitions, 2023-10-15 to 2024-01-15; (b) 39,019 footprint px | real (STAC + SCL audit) | **FAIL (honest)** | `x6.json` |
| A2 | The A2 HR benchmark itself is sound; **B14 was a defect in how it was reported, not in the experiment.** Its complete numbers are the five F6 rows above, which recompute from `x2.json` per-image data | — | — | — | 178 images, 5 datasets | real | sound; reported honestly by F6 | `x2.json` |

A6's exemplary-reporting behaviour is worth naming: it declined to pool anything, ran no test it could not
justify, and labelled its one post-hoc relaxation as exploratory with no keep rule. `RESULTS_EXCEPTIONAL.md`
listed that as a claim; here it is simply the correct conduct, not a result.

---

## Row tally

| | count |
|---|---|
| Claims F9 audited | 49 |
| **KEEP** rows in the table above | **43** |
| **DOWNGRADE** rows in the table above | **5** |
| **Total rows in the F9 claim table** | **48** |
| **DROP — excluded entirely** | **1** |
| Additional sound x-wave rows (A5, A6, A2 — outside F9's 49) | 3 |

Per task (F9 claim table only): F1 **9** · F2 **10** · F3 **6** · F5 **7** · F6 **5** · F7 **11** = **48**.
Counts computed by F10 from `f9.json`'s `claims_audited` list, not typed by hand.

### The one DROP, and why it is not here

F9 dropped a single claim: `f1_REPORT.md`'s assertion that the hash
`7ee83389…0111e7e6` *"is 66 hex chars and cannot be a valid SHA-256 digest"*. **Both halves are false** —
`len()` returns **64**, the string is well-formed hex, and it is the exact sha256 of `configs/fix.yaml` at
the F0 pre-registration commit `020880f`, i.e. precisely the hash F6 legitimately computed before the YAML
fix landed. A downgrade is for a claim that is true but weaker than stated; there is no weaker true version
of this one, so it is dropped.

**It has no scientific content and no scientific consequence.** F1's own `config_sha256` is correct and
independently verified. It is excluded from the claim table entirely. Its one lasting value is as a process
signal: it is the single place in the wave where a quantitative assertion was made by *inspecting* a string
rather than *measuring* it — which is why `f9_checks.arithmetic_checked_by: code, not prose` exists.

---

## Pre-registration deviations: the two F2/F7 spec deviations, stated as F9 verified them

F9 **measured** both "it made things harder" claims rather than accepting the argument. The two came out
differently, and the difference matters.

### 1. τ recomputation (F2) — **VERIFIED: it genuinely made the bar harder**

`configs/fix.yaml` says take τ from F1's npz. F1 exported `max(d/σ_v1)` — a **different score function**,
spanning −25.86 to 255.60, whereas gate v2's own window scores are order 1–10. Split conformal is only valid
when calibration and test scores come from the same score function, so F2 recomputed τ with gate v2's score
on exactly F1's folds, splits and window grid.

F9 recomputed the verbatim τ from the npz as **119.90589141845703**, matching `f2.json` to the last digit.
**Applying it would flag ~nothing ⇒ window FAR = 0 ⇒ a vacuous PASS of the ≤ 0.05 rule.** Recomputing gave
**0.0652 ⇒ FAIL**.

**The deviation is unambiguously the harder choice, and it is the choice that cost F2 its keep rule** — the
opposite of post-hoc rationalisation. This claim stands as stated.

### 2. `e_b` NDVI ceiling 0.20 → 0.30 (F2) — **direction NOT ESTABLISHED**

The pre-registered rule (NDVI ≤ 0.20) is not satisfiable on this evergreen Western Ghats AOI in January:
only **8 pixels** qualify over the whole 1024×1024 AOI outside the disks + footprint (**0** inside F1's
crop). The only large bare surface is the landslide scar itself, which the exclusion geometry removes by
construction — and *must* remove, because using it would make `e_b` a label (B12). The ceiling was escalated
deterministically in 0.05 steps until ≥100 pixels qualified; 0.30 is the first that did (332 px).

F2 and the ADR then argued: a shorter mixing line inflates `f`, therefore inflates the score, therefore
raises FAR, therefore harder. **F9 measured this and the chain breaks at the third link:**

| mixing-line length | mean `f` | mean `round(16f)` | mean score `s` | max `s` |
|---|---|---|---|---|
| 1.00 | 0.395 | 6.31 | **20.36** | 51.14 |
| 0.75 | 0.524 | 8.40 | 20.26 | 47.60 |
| 0.50 | 0.674 | 10.82 | 17.39 | 31.80 |
| 0.25 | 0.821 | 13.20 | **10.59** | 15.90 |

`f` inflates **2.08×** — that part holds. But the **detection score s = f/σ_f FALLS, it does not rise**,
because σ_f carries its own 1/\|e_v − e_b\| factor that more than offsets the inflation of `f`. A shorter
mixing line makes the conformal detector fire **less** readily, not more. And since τ is itself recomputed
conformally from the same score, the calibration exceedance rate is pinned near α whatever `e_b` is,
leaving the **net test-split effect indeterminate**.

**The correct framing, and the only one to use:** the escalation was **disclosed pre-hoc**, with the full
0.20–0.50 ceiling table published, **before any τ, FAR or block-sum number existed in this run — and the
keep rule failed regardless.** No pass was manufactured; this is not evidence of gaming. But **its effect on
the difficulty of the keep rule is NOT ESTABLISHED, and must not be described as having made anything
harder.** Only the stated mechanism is wrong; no reported number depends on it.

**The consequence that does carry:** `e_b` at NDVI ≤ 0.30 is the darkest, least-vegetated land in the AOI,
**not true bare ground**. It sits closer to `e_v` than a real soil/rock endmember would. **The fraction `f`
is upper-leaning and must not be read as a calibrated area** — reinforced by the ±1 robust-SD sensitivity
(~8× on flagged pixel count).

---

## Human actions still needed

### F4 (A7 fine-tune) — **OPEN. Never dispatched in this fix wave.**

**Verified by checking the filesystem, not assumed:**

| artifact | present? |
|---|---|
| `experiments/f4_finetune.py` | **ABSENT** |
| `notebooks/a7_finetune.ipynb` | **ABSENT** |
| `experiments/results/f4.json` | **ABSENT** |
| `notebooks/colab_finetune.ipynb` | present (pre-existing, not an F4 artifact) |

Only **F1, F2, F3, F5, F6, F7, F9** ran in this wave. F4 was never dispatched, so no F4 result exists and
none is claimed. Per `configs/fix.yaml`'s `a7` block, its status remains **`status_until_colab_upload:
BLOCKED`**.

**Exact next step:** a human runs the fine-tune on GPU (Colab) per `docs/r5_data_contract.md`, with an
AOI-level split made *before* cropping and shuffled batches drawn from many AOIs (the pre-registered guard
against B15's 3-batch cycling), keeping `hard_constraint` frozen and unchanged from
`AGENTS.md`/`hard_constraint.safetensor`. Required artifacts: safetensors + sha256 + a reload-agreement
test, per-epoch held-out-AOI PSNR, and the `guard_test` that **fails if the loss sequence is periodic in the
number of distinct batches**. Then evaluate the checkpoint against the `a7` keep rule *and* the A2 benchmark
baselines in `x2.json`. Per the pre-registration: **if the rule fails, ship pretrained and say so.**

Until that lands, **F5's `alloc_finetuned_sr` arm stays BLOCKED** and every SR number in this document is
**pretrained SEN2SR-lite, NOT fine-tuned**.

### F8 (GPU / RTX 3050 physical checks) — **OPEN. Never dispatched in this fix wave.**

| artifact | present? |
|---|---|
| `experiments/f8_gpu.py` | **ABSENT** |
| `experiments/results/f8.json` | **ABSENT** |

Every timing and memory figure in this repo comes from CPU or from the single 6 GB RTX 4050 run described in
R1. **No physical 4 GB deployment claim is supported by anything here.**

**Exact next step:** a human with the RTX 3050 runs the five pre-registered checks in `configs/fix.yaml`'s
`f8_gpu` block — `nvidia_smi_vram`, `e4_memory_gate_full_aoi`, `e8_4_vs_8_dihedral_real_tiles`,
`gpu_hash_drift`, `seconds_per_km2` — and writes `f8.json`. The `physical_4gb_test_condition` fires **only
if total memory ≤ 4 GiB**. If there is no CUDA device, the pre-registration requires the **exact
instructions be recorded in `f8.json`** rather than the checks being silently skipped.

### Two stale hashes to fix in the reports (F9 downgrades 3 and 4)

`f3_REPORT.md` prints `192345cd156583c6…` (truncated) and `f6_REPORT.md` prints `7ee83389…0111e7e6`. Both
`f3.json` and `f6.json` **were** correctly amended (commits `0059c2f` and `6d2f722`, two lines each — F9 read
the full diffs and confirmed no other field was touched). The reports were not. **Cite `18cce7b1…f170970`,
or the JSONs, never those two report headers.** F10 did not edit the f-series reports (out of scope), so this
remains open.

Context that makes these harmless-but-worth-fixing: `configs/fix.yaml` **as pre-registered at `020880f` did
not parse at all** (`yaml.safe_load` raises `ParserError` — an `f9_checks` block sequence followed by sibling
mapping keys at the same indent). Two independent fixes landed (F3's `298e214`, F1's `0da15a3`); F9 verified
`yaml.safe_load(298e214) == yaml.safe_load(0da15a3)` is **True** — they differ only by a comment. **No check
name, threshold or semantic ever changed.**

### Three latent items F9 recorded, none a live defect

1. **Two unguarded dihedral loaders** (`placebo_v2._load_dihedral`, `f2_gate_v2.load_dihedral` — raw
   `np.load` on a date-templated path). F9 confirmed **by attack, not by reading**, that they are provably
   unreachable with a post date, because the guarded 10 m load over the same date list executes first in
   every caller and raises. **Worth hardening.**
2. **Several F7 tests assert properties of `f7.json`** rather than re-executing code — notably
   `test_tau_is_used_not_merely_loaded`, which reads its three numbers back out of the JSON and so cannot
   catch a code regression. Those claims stand on the `trustsr`-level mutation tests instead, which F9
   sabotage-verified. **Test hygiene, not a false claim.**
3. **One pre-existing test failure on the branch**, `tests/test_placebo.py::TestFARPixel::test_far_pixel_basic`,
   in the quarantined legacy `trustsr/placebo.py`, which the fix wave never modified. Full repo suite:
   **329 passed, 1 failed, 1 skipped**. It touches no F1–F7 claim.

### AOI correction (B16) — **done and verified**

The project brief's Wayanad AOI centre (11.782°N, 76.233°E) lies **~34 km north of the 30 July 2024
landslide**. The corrected centre, used by `configs/fix.yaml` `aoi_center` and every experiment in this
wave, is **11.490°N, 76.160°E**.

- `docs/design.md` — **fixed** (plus a mojibake encoding cleanup on the touched line).
- `docs/playbook_strategy.md` — **fixed**, with the correction and its reason stated inline.
- `README.md` — **verified: no AOI coordinates appear in it at all**, so no fix was needed. The earlier B16
  note listing README.md was inaccurate; `experiments/results/INVALIDATED.md` has been corrected.
- `AGENTS.md` and `RISK_REPORT.md` **intentionally retain the brief's original AOI** as the pre-registered
  record of what was originally specified. That is deliberate provenance, not an unfixed bug.

---

## What would change these verdicts

F9 reran test suites, attacked guards, sabotaged thresholds and recomputed from stored artefacts. It did
**not** regenerate the SR stack or the placebo folds from imagery. Two things would strengthen or overturn
this audit:

1. **A real end-to-end re-execution of F1/F2/F7 from Sentinel-2 imagery.**
2. **An independent check of the cached E1/E8 dihedral product itself**, which every SR-dependent number in
   F1, F2 and F7 inherits without re-deriving. F7's reproduction check confirms this run *matches the cache*
   (max abs diff 0.0) — it does **not** independently validate the cache.

Beyond the audit, the substantive open question is the one the cross-cutting finding names: gate v2 is
correctly calibrated and **does not generalise across the spatial split**. Three folds sharing one crop
cannot settle that. The next experiment that matters is a placebo calibration on genuinely exchangeable
windows — a different crop or AOI — not a threshold adjustment.

---

## Standing labels on every number in this document

- **pretrained SEN2SR-lite, NOT fine-tuned** (F4 has not shipped).
- **k = 2.0, NOT calibrated.**
- **Retrospective comparison**, not an operational or real-time result.
- **No labels and no 2.5 m ground truth exist for Wayanad**, so **no accuracy, F1 or IoU-against-truth is
  reported anywhere for the Wayanad map.** F7's v1-vs-v2 IoU is agreement between two products, not accuracy.
- **One post-event date** (2024-12-06), 129 days after the event: regrowth, seasonality and illumination
  differences between a January pre-pool and a December post date are **confounded with the landslide**, and
  there is no second post date to separate them.
- **Pixels and neighbouring windows are spatially correlated**; exchangeability across windows is **assumed,
  not proven** (`configs/fix.yaml` `units.window_caveat`).
- **The 2.5 m output is a model reconstruction, not a new sensor observation.**

## Provenance

| | |
|---|---|
| Pre-registration | `configs/fix.yaml`, sha256 `18cce7b1ea88855bc054b42e6cbe3219eef73b2909eb091d07d5ada3df170970` |
| Verifier | F9 — `experiments/results/f9.json`, `f9_REPORT.md` (49 claims: 43 keep, 5 downgrade, 1 drop) |
| Experiments | `f1.json` `f2.json` `f3.json` `f5.json` `f6.json` `f7.json` (+ reports); sound x-wave: `x2.json` `x5.json` `x6.json` |
| Quarantine | `experiments/results/INVALIDATED.md`, `invalidated.json` — x3, x4, x9, x10, x11, `RESULTS_EXCEPTIONAL.md` |
| Not run | F4 (A7 fine-tune), F8 (GPU/RTX 3050) — both **BLOCKED**, artifacts verified absent |
