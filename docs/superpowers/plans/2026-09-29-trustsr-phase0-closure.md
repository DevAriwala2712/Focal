# TrustSR Phase 0 Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the measured SEN2SR-lite tile-shape and real Colab fine-tuning risks, then publish an evidence-backed Phase 0 gate decision before any product component is built.

**Architecture:** Keep risk experiments in `risk/`, their small CPU checks in `tests/`, and measured JSON under `risk/results/`. Reuse the pinned upstream model loader, source-verified WorldStrat pair preparation and existing Colab notebook. A failed scientific prerequisite remains a failed or blocked result in `RISK_REPORT.md`.

**Tech Stack:** Python 3.10+, pinned PyTorch/CUDA, rasterio, NumPy, pytest, safetensors, Colab, SEN2SR-lite `NonReference_RGBN_x4`.

**Spec:** [AGENTS.md](../../../AGENTS.md), [project blueprint](../../project_blueprint.md), [technical design](../../design.md), [current evidence](../../../RISK_REPORT.md).

## Global Constraints

- Use only B02/B03/B04/B08 as spectral inputs; the upstream model order is B04/B03/B02/B08.
- Fine-tune the pinned pretrained SEN2SR-lite weights; no training from scratch or silent replacement of its hard constraint.
- All GPU work is sequential/tiled and must meet the 4 GiB target; on CUDA OOM halve input tile size down to 64 px, then fail explicitly if still impossible.
- All raster outputs are COGs in the input CRS with exactly aligned ×4 affine grids; do not silently crop.
- `configs/phase0.yaml` owns paths, thresholds, seeds and model hashes. Do not claim a physically tested 4 GB GPU from a 6 GB card or an allocator cap.
- Downloads retry three times with backoff and name the failing URL.
- Phase 1 stays closed until R1-R5 have recorded outcomes and the remaining blockers are resolved or the project brief is explicitly revised.
- Treat an optimizer smoke pass as optimizer evidence, not held-out SR or landslide accuracy.

## Review Focus

- **Non-native tile at scene edge:** Task 1 tests 64-pixel inference and output dimensions without padding disguised as memory reduction.
- **Fourier constraint fidelity:** Task 1 checks that a native 128-pixel run matches the original pinned forward path and that non-native outputs retain the configured spectral-consistency test.
- **OOM after partial optimization:** Task 2 tests that a retry starts again from pinned pretrained weights and step zero.
- **Bad or unavailable training pair:** Task 2 checks manifest hashes, split membership, band order, affine grid and mask before spending GPU time.
- **Misleading status after a newer run:** Task 3 reconciles JSON, the preparation ledger, Colab logs and the prose report; no stale “missing pairs” statement remains after preparation.

---

## File map

| File | Responsibility |
| --- | --- |
| `risk/model.py` | Pinned loader and any source-faithful shape adapter, kept separate from metric collection. |
| `risk/r1_model_fits.py` | Timed/memory probe for native, 64-input and tiled 512-input cases; shape/OOM causes. |
| `tests/test_model_probe.py` | Native equivalence and small-shape/constraint properties on CPU. |
| `tests/test_risk_core.py` | Retry policy, including exhausted 64-pixel fallback. |
| `risk/worldstrat_prepare.py`, `docs/r5_data_contract.md` | Real pair provenance and smoke-only radiometric interpretation. Change only if source inspection finds an actual defect. |
| `risk/r5_finetune.py`, `tests/test_training.py` | Manifest validation, 50-step optimization, OOM restart, checkpoint reload evidence. |
| `notebooks/colab_finetune.ipynb` | Reproducible isolated Colab invocation; actual runtime evidence comes from an executed run. |
| `configs/phase0.yaml`, `risk/results/*.json`, `RISK_REPORT.md`, `README.md` | Frozen settings, measured evidence, gate decision and reproduction directions. |

## Task 1: Determine and verify the SEN2SR-lite tile policy

**Files:**
- Modify: `risk/model.py`, `risk/r1_model_fits.py` only if a source-faithful solution is established
- Test: `tests/test_model_probe.py`, `tests/test_risk_core.py`
- Modify: `configs/phase0.yaml`, `RISK_REPORT.md` with measured outcomes

**Interfaces:**
- Consumes: `load_model(cfg, root, *, trainable=False, device='cuda')` from `risk/model.py`; pinned `HardConstraint` source/weights.
- Produces: one documented `run_model_tile(model, tile: torch.Tensor) -> torch.Tensor` path if shape adaptation is validated; `risk.r1_model_fits.probe(cfg, root) -> dict` remains the result interface.

- [ ] **Step 1: Characterize the constraint.** Read the pinned `load.py`, `sen2sr.models.tricks.HardConstraint`, the 512×512 weight tensor and `srmodel.forward`. Record how the mask was built or state that the generation parameters cannot be recovered. The native 128-input result is the reference.
- [ ] **Step 2: Write failing CPU tests.** Assert `run_model_tile` returns exactly 4× spatial dimensions for 64 and 128 inputs; assert its 128 result matches the unmodified pinned forward path within the configured numerical tolerance; assert the spectral downsample check is no worse than the native-path reference on the same synthetic input class. Test a 512-input *scene* through multiple bounded model tiles, not as a claimed native 512 forward.
- [ ] **Step 3: Verify the tests fail for the present fixed mask.** Run `python -m pytest tests/test_model_probe.py tests/test_risk_core.py -q`; record the expected 64-input shape failure before implementation.
- [ ] **Step 4: Implement only a justified adapter.** Keep the pretrained SR weights and original 128 path unchanged. If the mask's construction can be reproduced from pinned source/weights, generate a size-matched constraint for 64 and verify the tests. If it cannot, record R1 FAIL with the reason; do not resize a frequency mask or pad 64 to 128 merely to obtain a passing shape.
- [ ] **Step 5: Run the full CPU suite and the actual R1 probe.** Run `python -m pytest -q` and `python -m risk.r1_model_fits`. Record peak allocated/reserved memory, process/device reading, per-tile timing, output dimensions, failures and the exact 4 GiB method. A 512 scene passes only if tiled composition works under the budget; do not call a 512 native model forward supported.
- [ ] **Step 6: Commit the tested adapter/probe and its report update.** Use a single focused commit. If the source-faithful 64 path is impossible, commit the failure evidence and keep the gate closed instead of committing an unvalidated approximation.

## Task 2: Execute the real 50-step Colab smoke test

**Files:**
- Modify if needed: `risk/worldstrat_prepare.py`, `risk/r5_finetune.py`, `notebooks/colab_finetune.ipynb`
- Test: `tests/test_worldstrat_prepare.py`, `tests/test_training.py`
- Output: `risk/results/r5.json`, checkpoint hash and actual runtime log; checkpoint binary stays ignored

**Interfaces:**
- Consumes: `data/worldstrat/pairs.json` under the execution checkout, with at least three real train AOIs and matching prepared COG hashes; `validate_manifest(manifest, base, bands) -> list[dict]`; `load_crop(pair, size, scale) -> list[np.ndarray]`.
- Produces: `risk.r5_finetune.probe(cfg, root) -> dict` with 50 losses, first/last five-step means, weight delta, gradient check, checkpoint SHA-256, reload match, AOI IDs and Colab GPU/runtime.

- [ ] **Step 1: Reconcile source pairs.** Inspect the current [preparation evidence](../../../risk/results/worldstrat_preparation.json) and original members. Confirm Landcover-118968, Landcover-151915 and Landcover-1534788 are in the publisher train split, have same-day source acquisitions, named RGBN bands, documented reconstructed HR bounds, exact ×4 prepared grids, valid land crops and recorded empirical gain/offset. The ignored prepared files live in the original execution checkout; a new Git worktree does not copy them.
- [ ] **Step 2: Write or retain failure-path tests.** Assert a changed source hash, shifted HR grid, invalid crop, wrong split, wrong band order and missing Colab runtime all stop before training. Assert an injected OOM after an optimizer step causes a fresh model/optimizer at step zero on the next supported tile; exhausted 64-pixel OOM reports failure.
- [ ] **Step 3: Verify tests.** Run `python -m pytest tests/test_training.py tests/test_worldstrat_prepare.py -q`; expected PASS for guard tests. Fix only a reproduced failure. Do not substitute the artificial CPU optimizer test for the Colab experiment.
- [ ] **Step 4: Execute the pinned notebook in an authenticated Colab GPU runtime.** Install its isolated pinned environment, stage the verified manifest/files and run exactly `training.steps: 50` using the 4 GiB budget policy. Record the actual GPU, CUDA/PyTorch versions, manifest hash and all 50 losses. If sign-in, asset transfer, package installation or 64-pixel fallback blocks execution, keep R5 BLOCKED/FAIL and name that condition.
- [ ] **Step 5: Verify the checkpoint and claim.** Require finite nonzero gradients, changed trainable weights, lower final than initial five-step mean, a saved checkpoint hash, and a matching reloaded output. Record `training_steps_run: 50` only after completion. Note that empirically harmonized SPOT DN targets make this a mechanics smoke test.
- [ ] **Step 6: Commit the new result metadata and any tested code fix.** Keep binary checkpoints and source imagery out of Git; retain URLs/hashes needed to reproduce them.

## Task 3: Reconcile Phase 0 evidence and decide the gate

**Files:**
- Modify: `RISK_REPORT.md`, `README.md`, `docs/r5_data_contract.md` if its prerequisite/status text is stale
- Read: `risk/results/r1.json` through `r5.json`, `risk/results/worldstrat_preparation.json`, `docs/implementation-ledger.md`

**Interfaces:**
- Consumes: measured R1/R5 results and the existing R2-R4 evidence.
- Produces: a dated Phase 0 pass/fail/block table with numbers, provenance and an explicit Phase 1 decision.

- [ ] **Step 1: Compare every headline to its evidence file.** Distinguish “pairs prepared” from “Colab training ran.” Preserve R2's 2024-12-06 first accepted post-date, R3's region/count/licence basis and R4's sample-only label limits.
- [ ] **Step 2: Update the report.** For each R1-R5 row, state PASS, FAIL or BLOCKED, measured numbers, test conditions and what the result means for the proposed product. Say plainly when physical 4 GB hardware was unavailable. Keep the 129-day Wayanad gap and original Landslide4Sense-host failure visible.
- [ ] **Step 3: Update README reproduction and gate text.** Commands and demo status must match what actually ran. If any required risk remains unresolved, say Phase 1 remains closed and list the exact blocker. Do not create `trustsr/`, `demo/` or metrics that have not been produced.
- [ ] **Step 4: Verify links, JSON and tests.** Run `python -m json.tool risk/results/r1.json` and `python -m json.tool risk/results/r5.json`; run `python -m pytest -q`; inspect `git diff --check` and compare the written numbers with the JSON.
- [ ] **Step 5: Commit the report and stop.** End the execution turn with the five-result summary and explicit gate decision. Phase 1 requires a separate decision after the evidence is reviewable.

## Self-review and handoff

This plan covers the two remaining Phase 0 blockers and report reconciliation; the conditional Phase 1 module sequence lives in [project_blueprint.md](../../project_blueprint.md). It does not treat WorldStrat smoke targets as physical reflectance, Landslide4Sense as pre/post data, or a 6 GB card as a physical 4 GB test. Before executing, reread the live branch and evidence: the parallel development chat may have advanced R5 since this plan was written. Implement the tasks that remain, preserving its work.
