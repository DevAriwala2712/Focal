# TrustSR design and decision record

Status: design baseline plus Phase 0 implementation, 2026-09-26. The risk harness is implemented, but the Phase 0 gate has not passed; Phase 1 is closed. The execution evidence in [RISK_REPORT.md](../RISK_REPORT.md) supersedes initial availability assumptions below.
Authority: [AGENTS.md](../AGENTS.md), the user's GPU correction, and the [full supplied NTRO statement](problem_statement.md).
The statement is copied byte-for-byte from the root file; an official SIH statement ID/URL has not been supplied or independently authenticated.

## Purpose and success criteria

Build a reproducible Sentinel-2 L2A 10 m â†’ 2.5 m (Ã—4) pipeline for landslide damage assessment. Demonstrate the 30 July 2024 Wayanad event using a roughly 10 Ã— 10 km AOI centred on 11.782Â°N, 76.233Â°E. This is an offline research demonstration for the SIH finale, not an operational emergency service.

Trustworthiness takes priority over sharpness. The 2.5 m output is a model reconstruction, not a new sensor observation. Report where change has independent support in the original 10 m signal, where spatial detail is inferred, and where SR-only change is unsupported. Every displayed date, metric, and checkpoint must have provenance.

## Decisions and rejected options

| Option | Decision | Reason |
| --- | --- | --- |
| A: pretrained SEN2SR-lite RGBN Ã—4, fine-tuning, original-resolution change support and transform sensitivity | Selected by project brief; feasibility still gated by Phase 0 | Small pretrained four-band model, explicit spectral constraint, inspectable inference/training path, and a manageable demo scope |
| B: diffusion SR | Rejected for this project | Iterative sampling and uncertainty ensembles would add memory/runtime and validation burden. Plausible invented textures are especially problematic for damage mapping. This is a project trade-off, not a claim that every diffusion method is inaccurate |
| C: train our own SR model from scratch | Rejected | Violates the brief; insufficient compute, paired-data coverage and time to establish reliability |
| Pure interpolation | Retain as a baseline, not the proposed SR solution | Helps establish whether learned SR adds analytical value |

Out of scope: 20 m/60 m **spectral inputs**, other disaster types, training from scratch, alternative SR architecture development, an operational deployment, and automatic building/road destruction claims. SCL is an explicit exception: it is a categorical quality layer typically supplied at 20 m, used only for masking with nearest-neighbour resampling, never as an SR input.

## Verified model interface

Use `SEN2SRLite/NonReference_RGBN_x4`, not the ten-band composite model. The upstream README and model metadata specify input order **B04, B03, B02, B08** (RGBN), shape NÃ—4Ã—128Ã—128 and output NÃ—4Ã—512Ã—512. The fetched metadata reports 472,496 parameters; this is not a VRAM measurement.

The documented inference API is `mlstac.download(file=..., output_dir=...)`, then `mlstac.load(model_dir).compiled_model(device=device)`, then `model(batch)`.
The inspected model `load.py` defines `trainable_model(path, device=...)` separately. Its `compiled_model` freezes SR parameters. Training must use a verified trainable entry point, assert nonzero gradients and changed weights, and retain the frozen hard-constraint module. The wrapper exposes `sr_model` and `hard_constraint`.
Do not infer trainability from a successful forward pass.

Preserve storage-band names and explicitly reorder for the model. Validate reflectance scale/offset from actual provider assets; do not blindly divide every source by 10,000. Retain invalid masks even if finite fill values are required for inference.
Pin upstream code and checkpoint revisions/hashes at download time; the currently inspected GitHub revision is 8e21bb669bcc6e8eb953a1fb24dfc5bb59dc18a4. The Hugging Face revision has not yet been pinned.

## Compute and geometry

Actual hardware: RTX 4050 Laptop GPU, 6,141 MiB, native Windows. Preserve the brief's 4 GiB compatibility target as a separate acceptance test. Running on a 6 GB card does not prove 4 GB compatibility. Record both allocated/reserved PyTorch peaks and device memory; a process allocator cap is only an approximation to a physical 4 GB GPU.

All GPU operations, including fine-tuning and eight transformed runs, must process tiles sequentially. On CUDA OOM halve tile size down to 64 input pixels, then fail explicitly. Test support for each shape rather than assuming that the published 128-pixel model/constraint accepts 64 or 512. Benchmark 512 input pixels separately from its normal 512-pixel **output**.

Reference grid: first accepted pre-event 10 m image. Reproject every date onto that full grid, marking missing coverage as no-data instead of silently cropping. Output affine transform = input transform Ã— scale(1/4, 1/4); dimensions Ã—4; identical CRS and bounds. Never upsample to a cosmetically similar but shifted grid. All raster outputs are COGs, including masks and diagnostics; categorical overviews/resampling use nearest neighbour.

## Components after Phase 0

| Module | Contract and responsibility | Depends on |
| --- | --- | --- |
| fetch | AOI/time window â†’ aligned four-band multi-date stack, SCL masks, JSON date/provenance manifest | R2 clear-date audit |
| sr | Normalized RGBN tile â†’ Ã—4 tile; overlap and feathered blending; OOM retries | R1 and R5 |
| trust | Eight dihedral transforms per date, inverse transforms, NDVI per run, per-pixel mean/std | sr, valid masks |
| change | NDVI drop plus original 10 m support â†’ five classes and unsupported counts | trust, fetch, calibration |
| evaluate | Held-out SR fidelity, spectral consistency and common-grid downstream metrics | R3/R4, leakage-safe splits |
| demo | Precomputed Wayanad before/after view, resolution toggle, overlay, metrics; optional one-tile live run | verified outputs |

Use configs/*.yaml for paths, thresholds, cloud classes, tile/overlap sizes, model identifiers, training settings, random seeds and evaluation choices. Pin an actually tested dependency set, not an invented lockfile.

## Trust and change semantics

Compute NDVI separately for each transformed run after inversion. Pool clear pre-event dates and runs for the pre-event moments; post-event moments use the first accepted post-event date. Use streaming moments to bound memory. Cross-date variance includes seasonal change and registration error; dihedral dispersion is a sensitivity proxy, not calibrated probability or complete epistemic uncertainty. Eight agreeing runs can share the same error.

For valid pixels let d = mean(NDVI_pre) âˆ’ mean(NDVI_post) and Ïƒ = sqrt(Ïƒ_preÂ² + Ïƒ_postÂ²). Let S = d > kÏƒ and P = independently thresholded 10 m parent NDVI drop.

| S | P | Class |
| --- | --- | --- |
| true | true | OBSERVED: original-resolution-supported change; the 2.5 m boundary is still inferred |
| false | true | INFERRED: parent change supported, fine position uncertain |
| true | false | UNSUPPORTED: exclude from change, count and log |
| false | false | NO_CHANGE |
| any | any | NO_DATA overrides all when cloud/shadow/invalid data occurs in any selected date |

k defaults to 2.0 until calibrated. The 10 m parent-drop threshold remains a calibration decision; it must be explicit and independent of SR. Specify valid NDVI denominators, temporal aggregation, cloud/shadow classes and minimum coverage before evaluating real change. A conservative proposed mask excludes SCL 0,1,3,8,9,10,11 and treats class 2 separately after inspection.

Report precision/recall/F1 separately for OBSERVED and OBSERVEDâˆªINFERRED: their union equals P for valid pixels, so that union cannot demonstrate added binary detection benefit over the parent mask on the same grid. Measure unsupported rejection and localization separately. NDVI drop alone indicates vegetation disturbance, not uniquely landslide damage.

## Data and evaluation limits

WorldStrat pairs need band, radiometric, temporal and geometric inspection. Its 1.5 m high-resolution product must be properly resampled to the exact 2.5 m target grid, not relabelled as native 2.5 m. Inspect the native multispectral/pansharpened provenance, especially NIR. Split by AOI/event before cropping to avoid leakage. A 50-step loss decrease is a smoke test, not generalization evidence.

The arXiv dataset supplies inventories and a recipe for acquiring Sentinel-2 dates, rather than a verified ready-to-download pre/post raster archive. Landslide4Sense is single-image segmentation and cannot alone validate a temporal detector. Select only the allowed four bands if used; 10 m masks do not validate 2.5 m boundary accuracy. Calibrate on one split and evaluate on a separate event/geographic split.

Compute PSNR/SSIM with a documented data range and mask, plus per-band error after downsampling SR back to 10 m. Compare downstream predictions on a common labelled grid; upsampling a 10 m label is not independent high-resolution truth. OpenSR-test is optional if installation passes.

The published Wayanad 507 buildings / 8.38 km roads are a Sentinel-1-plus-OSM contextual baseline. A future count comparison needs matching footprint, event dates, historical OSM version and intersection rule. They are not an F1 ground truth and must not be presented as TrustSR results.

## NTRO deliverable traceability

| Supplied requirement | Planned evidence |
| --- | --- |
| Sentinel-2 10 m to <4 m | Ã—4 2.5 m COG; resolution/extent tests |
| Pre-processing | SCL audit, radiometry, registration, aligned stack and manifest |
| Paired-data training | WorldStrat pair provenance, split manifest, fine-tuning log and reloadable checkpoint |
| Geographic and spectral consistency | Affine/CRS tests and downsample consistency metrics |
| Accuracy assessment / HR validation | Held-out PSNR/SSIM; spectral consistency; common-grid landslide precision/recall/F1 |
| Uncertainty and error | Class legend, NDVI sensitivity maps, unsupported counts, limitations and calibration record |
| Analytical usefulness | Reproducible Wayanad demonstration and honest comparison to 10 m baseline |

## Acceptance and stop rules

Phase 0 comes first: five standalone scripts, evidence and a completed RISK_REPORT.md. Each result is PASS, FAIL or BLOCKED/NOT RUN; missing numbers stay missing. No Phase 1 request until that report exists with actual results and unresolved blockers stated.

Later CPU-only synthetic tests must cover exact georeferencing, spectral consistency, transform inverses, known change/unsupported artefact/no-data, and tiling using a deterministic local operator. Real CNN tiled equality is not guaranteed without sufficient context; add separate seam measurements rather than promising exact equality for the learned model.
Downloads retry three times with backoff and fail naming the URL. No clear post-event image produces a dated cloud/coverage audit and an explicit stop.
