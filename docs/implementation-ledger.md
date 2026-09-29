# Phase 0 implementation ledger

Plan: docs/phase0_plan.md. Spec: docs/design.md. Baseline: 57bcc55.

User authorized implementation after receiving both documents. Execute locally on branch phase0-risk-tests and stop before Phase 1.

Ruling: use this existing checkout on a dedicated branch, with a project virtual environment, because the checkout is clean and no concurrent implementation is occurring. No product dependencies alter the user's global Python.

Ruling: use the available Python 3.13 runtime if the pinned wheels resolve; Python 3.11 was a proposed trial, not a requirement. Record the actual tested environment.

Ruling: R5 is only a PASS after a real Colab run on WorldStrat pairs. Local/synthetic tests validate mechanics only and cannot replace that result.

Pre-flight: all scripts consume a YAML config and shared download/result handling. R1/R5 share the exact pinned model loader; R3 supplies dataset access evidence, not automatically approved training pairs. R2 is an independent SCL audit. R4 must distinguish inventories, paired images, and single-date masks.

## Execution checklist

- [x] Shared config, bounded downloads/retries, JSON results; CPU failure-path tests.
- [x] R1 actual pretrained model benchmark and memory/shape evidence.
- [x] R2 annual STAC/SCL audit, date selection and no-data tests.
- [x] R3 unique AOI country counts, archive size and licence evidence.
- [x] R4 actual label/download checks and truthful pair-availability status.
- [x] R5 validated pair manifest, 50-step runner and Colab notebook; run or record concrete blocker.
- [x] Full CPU suite, source review, evidence-backed RISK_REPORT and reproduction instructions.

Each script writes status PASS/FAIL/BLOCKED, UTC timestamp, config hash, observed measurements and explicit limitations. Downloads retry three times after the initial attempt, with exponential backoff. Test results are separate from scientific feasibility results.

## Execution results

- Shared/download and scientific guard tests were written before their implementations; missing functions failed, then passed. Additional guards reject unknown SCL codes, zero radiometric scales, shifted paired grids and model hash mismatches.
- R1 invoked the actual pretrained model on CUDA: 128px native input succeeds; 512/64 input fails the fixed Fourier-mask shape requirement. Keep that failure visible; no adapter was invented during the risk phase.
- R2: all 147 SCL assets audited, 73 acquisitions, 12 clear pre-dates, first accepted post-date 2024-12-06.
- R3: metadata/country/licence audit passes. 71 India, 125 South Asia. No full WorldStrat imagery archive fetched.
- R4: 838-polygon inventory and real pre/post/label COG proof retrieved; original Landslide4Sense host DNS fails after retries, but the pinned IBM-NASA mirror supplied a genuine image/mask sample. Sample-access PASS.
- R5: three genuine WorldStrat train AOIs were prepared and a Colab T4 run completed 50 optimizer steps; final loss window decreased, weights changed, and checkpoint reload matched. Smoke-test PASS, without an SR accuracy claim.
- Ruling: the upstream trainable loader does not enable Conv3XC.train_mode. Explicitly enable the existing differentiable branch after loading weights; CPU characterization verifies forward equivalence and actual gradient flow. This preserves pretrained initialization.
- Ruling: pin affine 2.4.0 with rasterio 1.4.3 to avoid the new affine 3.x API deprecation encountered during setup. Full suite passes after pinning.
- Independent reviewer dispatch failed due its usage limit. No successful external review is claimed; local review was completed and found the train-mode issue.
- Final verification before the Colab run: 23 CPU tests pass; pip check and real R4 COG/grid inspection pass. The Colab isolated environment later passed pip check and R5. No Phase 1 work performed.

## Continuation: recover R4 access

Ruling: use the IBM-NASA hosted redistribution as a documented alternate download, retaining original-host failures and no claim of byte-equivalence. Pinned revision and file hashes recorded. Added a test first (failed on missing inspector), then implemented bounded image/mask retrieval with RGBN-only reading and binary-mask validation. Full suite: 19 passed. R4 rerun returned PASS with actual mask containing 405 positive pixels. Colab is at Google sign-in; user must establish an authenticated session. WorldStrat ZIP selective-access feasibility is under investigation; no real prepared pairs claimed.

Source exploration retrieved 12 original members from the newer WorldStrat ZIP release using validated HTTP byte ranges, local-header CRC32 and file SHA256. Source files remain ignored originals; no raster output or training manifest fabricated. All three HR files omit their geotransform; LR bounds equal published metadata. The zero-cloud metadata was contradicted by SCL for two candidates; the third is all water. Preserve these findings for crop selection and documented georeferencing/radiometry reconstruction. Source downloader pinned at 39fe2232b2e9f1efe845795e6c3415026373fde3. All exploration processes terminated successfully.

## Real-pair preparation and Colab continuation

Ruling: Phase 0 smoke training may use empirically harmonized paired targets, explicitly labelled as such, rather than claim an unverified physical DN-to-reflectance conversion. Fit one positive gain/offset per RGBN band on each training crop after HR area resampling; reject correlation below configured 0.5. No holdout data enters this fit. This validates optimizer mechanics only and does not establish SR or NDVI accuracy. Preserve original geographic CRS and exact x4 affine subdivision; label spatial resolution nominal, not exact metric 2.5m.

Ruling: restore missing HR geotransforms from publisher WGS84 bounds only after verifying those bounds match the LR raster. Retain original files unchanged. Prefer central clear land crops to avoid documented black HR borders; reject remaining nodata.

Tests were red before implementing member CRC/size/name validation, clear-land crop selection and empirical gain recovery. A transient OneDrive replacement lock was reproduced with a failing test, then fixed with bounded atomic-replace retries. Full suite now 23 passed.

Preparation succeeded for Landcover-118968, Landcover-151915 and Landcover-1534788, each from the published train split and same-day HR/LR acquisitions. See worldstrat_preparation.json for hashes, source members, rejected candidates, coefficients and grids. The generated pairs are ignored data artifacts. Colab T4 base runtime used torch 2.11.0+cu128. Standard venv failed at ensurepip; `venv --without-pip` plus pip's `--python` method installed the complete pinned dependencies without altering base packages. The first isolated training attempt lacked Matplotlib; the next inherited Colab's inline backend, which was unavailable in the venv. With `MPLBACKEND=Agg`, 50 real-data steps completed, checkpoint saved and reloaded, and R5 returned PASS. The browser could not transfer the ZIP archive to this checkout, so the local R5 JSON was transcribed from visible Colab output and the checkpoint remains only in Colab's temporary runtime. See RISK_REPORT for the measured values and remaining Phase 1 gate failures.
