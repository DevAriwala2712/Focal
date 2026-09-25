# Phase 0 implementation ledger

Plan: docs/phase0_plan.md. Spec: docs/design.md. Baseline: 57bcc55.

User authorized implementation after receiving both documents. Execute locally on branch phase0-risk-tests and stop before Phase 1.

Ruling: use this existing checkout on a dedicated branch, with a project virtual environment, because the checkout is clean and no concurrent implementation is occurring. No product dependencies alter the user's global Python.

Ruling: use the available Python 3.13 runtime if the pinned wheels resolve; Python 3.11 was a proposed trial, not a requirement. Record the actual tested environment.

Ruling: R5 is only a PASS after a real Colab run on WorldStrat pairs. Local/synthetic tests validate mechanics only and cannot replace that result.

Pre-flight: all scripts consume a YAML config and shared download/result handling. R1/R5 share the exact pinned model loader; R3 supplies dataset access evidence, not automatically approved training pairs. R2 is an independent SCL audit. R4 must distinguish inventories, paired images, and single-date masks.

## Execution checklist

- [ ] Shared config, bounded downloads/retries, JSON results; CPU failure-path tests.
- [ ] R1 actual pretrained model benchmark and memory/shape evidence.
- [ ] R2 annual STAC/SCL audit, date selection and no-data tests.
- [ ] R3 unique AOI country counts, archive size and licence evidence.
- [ ] R4 actual label/download checks and truthful pair-availability status.
- [ ] R5 validated pair manifest, 50-step runner and Colab notebook; run or record concrete blocker.
- [ ] Full CPU suite, source review, evidence-backed RISK_REPORT and reproduction instructions.

Each script writes status PASS/FAIL/BLOCKED, UTC timestamp, config hash, observed measurements and explicit limitations. Downloads retry three times after the initial attempt, with exponential backoff. Test results are separate from scientific feasibility results.
