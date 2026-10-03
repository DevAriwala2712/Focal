# TrustSR Status

TrustSR is an experimental Sentinel-2 super-resolution pipeline with a trust-gating mechanism for change detection, currently in a Phase 0 research and validation stage.

## Executive status

| Area | Status | Evidence | Meaning |
|---|---|---|---|
| SR inference | PASS | `risk/results/r1.json` | Native SEN2SR-lite inference works at the published 128×128 input size. |
| 4 GiB compatibility | BLOCKED | `risk/results/r1.json` | Arbitrary tile sizes (e.g., 512×512) and the OOM fallback (64×64) fail due to a fixed Fourier mask. No physical 4 GB GPU was tested (only an allocator cap). |
| Imagery | PARTIAL | `risk/results/r2.json` | Clear pre-event imagery is found, but the first post-event clear image is 2024-12-06, 129 days after the Wayanad event. |
| Training | BLOCKED | `risk/results/r5.json` | Fine-tuning code exists but 0 real training steps have been run due to missing WorldStrat paired data manifests. |
| Trust gate | FAIL | `experiments/results/f2.json` | The trust gate fails its pre-registered False Alarm Rate (FAR) test on held-out windows (0.0652 vs target ≤ 0.05). |
| SR analytical benefit | FAIL | `experiments/results/f5.json`, `f6.json` | SR performs worse than bicubic interpolation in realistic sub-pixel unmixing and baseline PSNR (excluding the NAIP leak). |
| Wayanad demonstration | BLOCKED | `experiments/results/f7.json` | F7 mapped 1.60 km² of change, but it is not a validated detection due to the F2 FAR failure and lack of ground truth. |
| Reproducibility/audit | PASS | `experiments/results/f1.json`, `RESULTS_V2.md` | The pipeline's placebo tests, noise models, and cryptographic provenance auditing are successfully implemented. |

## What TrustSR is actually trying to solve

Sentinel-2 provides observed 10 m multispectral information. Super-resolution (SR) produces model-reconstructed 2.5 m detail. A central challenge in environmental monitoring is that this reconstructed spatial detail must not automatically become evidence of environmental damage (like landslides).

The proposed TrustSR system therefore tries to separate original-resolution-supported change from inferred fine spatial detail and unsupported SR-only artifacts. In essence, it aims to prevent SR hallucinations from being falsely reported as physical damage on the ground.

## What has actually been demonstrated

- Native SEN2SR-lite inference works at the published 128×128 input size (`risk/results/r1.json`).
- The audit and provenance harness is fully implemented and cryptographically hashes configurations (`experiments/results/f1.json`).
- Placebo and noise-model tests that are not dependent on real-world change detections passed successfully, such as the A5 noise model coverage (`RESULTS_V2.md`).
- Claim auditing and reproducibility machinery that passed, strictly preventing silent metric modifications and enforcing preregistered test policies.

## What has failed

- **F2 window FAR = 0.0652** versus the required ≤ 0.05 threshold (`experiments/results/f2.json`).
- **F7 therefore remains FAIL** and its mapped ~1.60 km² output is not a validated detection.
- **F5 realistic sub-pixel allocation = approximately −0.313 IoU** (`experiments/results/f5.json`).
- **F5's non-NAIP result = approximately +0.0016 IoU** with confidence intervals crossing zero.
- **F6 SR versus bicubic pooled PSNR = approximately −0.097 dB** (`experiments/results/f6.json`).
- **Excluding NAIP makes the PSNR result more negative**, clearly illustrating the NAIP data leakage/dependence.
- SR was more hallucination-prone across the reported five datasets.
- **R5 fine-tuning has 0 real training steps** (`risk/results/r5.json`).
- **R1 fixed-shape/4 GiB compatibility remains unresolved**; the test only used a 6 GB memory allocator cap rather than proving physical 4 GB compatibility, and arbitrary shapes fail.
- **First accepted clear Wayanad post-event image is 2024-12-06**, which is 129 days after the 2024-07-30 event (`risk/results/r2.json`).

## What this means scientifically

The current TrustSR formulation has NOT demonstrated an effective end-to-end landslide detection pipeline. This is not equivalent to saying the overall research direction is impossible, but the current evidence suggests the problem is deeper than a missing implementation detail.

In particular, the current SR model has not demonstrated downstream analytical benefit over simpler interpolation. The current trust gate calibrates on its calibration split but fails on the held-out split. The repository has correctly identified this as an exchangeability/generalization problem, meaning that simple threshold tuning alone is not an acceptable solution.

## Recommended next research direction

The safer and more defensible direction, which remains compatible with the core design, is as follows:

1. Use original 10 m Sentinel-2 evidence for the actual change decision.
2. Use SR as a localization/visualization hypothesis rather than independent evidence.
3. Require original 10 m support before a fine-resolution change becomes an `OBSERVED` result.
4. Treat SR-only features as `UNSUPPORTED`.
5. Evaluate whether SR improves spatial localization over bicubic before making it a core component.
6. Re-run trust calibration using genuinely independent AOIs/events rather than interleaved windows from one crop.
7. Complete the real fine-tuning smoke test on verified training data.
8. Only build the Wayanad demo after the preceding scientific gates are satisfied.

These are proposed next steps, not completed functionality.

## Current blockers

| Priority | Blocker | Why it matters | Required evidence to unblock |
|---|---|---|---|
| P0 | Trust gate generalization (F2) | A failing False Alarm Rate invalidates all downstream detection claims. | Passing FAR (≤ 0.05) on genuinely independent AOIs. |
| P0 | SR analytical deficit (F5/F6) | SR currently degrades analysis compared to bicubic interpolation. | Validated positive spatial metrics (IoU/PSNR) excluding NAIP leakage. |
| P1 | Arbitrary shape inference (R1) | Blocks the required 4 GiB OOM tiling fallback (64px). | Successful inference of dynamic tensor shapes without fixed Fourier masks. |
| P1 | Validated fine-tuning (R5) | Needed to prove the SR model can actually adapt to the domain. | Decreasing loss and a saved checkpoint using the WorldStrat pair manifest. |

## Definition of "Phase 1 ready"

Phase 1 should not begin merely because the code executes without syntax errors. Under the repository's acceptance logic, Phase 1 requires:
- R1-R5 results genuinely recorded as PASS.
- Unresolved blockers either explicitly resolved or accepted by a revised project brief.
- Independent calibration and test data splits.
- Validated training evidence (R5).
- Documented geometry and provenance.
- No unsupported claims in the README or demonstration UI.

## Timeline

The following represents rough engineering effort bands, but the scientific outcome remains uncertain:
- **Status/documentation cleanup:** Days
- **Phase 0 closure:** Weeks
- **Independent gate validation:** Weeks to Months
- **SR analytical comparison:** Weeks to Months
- **End-to-end demo:** Blocked pending successful scientific validation

## Bottom line

**Is TrustSR working?**
Not yet as a validated end-to-end solution. The current formulation has produced important negative results. However, the project is not a dead end because these failure modes are now clearly measurable and specific. The next stage is a strict scientific redesign and validation stage, not just more engineering implementation.
