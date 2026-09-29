# TrustSR project blueprint

Status: proposed architecture and acceptance contract, 2026-09-29. [RISK_REPORT.md](../RISK_REPORT.md) is the source of measured results. [AGENTS.md](../AGENTS.md) and the [supplied NTRO statement](problem_statement.md) define the task; [the playbook interpretation](playbook_strategy.md) informs presentation and sequencing.

## Mission and decision boundary

Deliver a reproducible Sentinel-2 L2A, four-band (B02/B03/B04/B08) 10 m → nominal 2.5 m research pipeline for retrospective Wayanad landslide-area assessment. The output is a model reconstruction. The trust gate must make the origin of every claimed change legible, preserve missing-data status, and prevent an SR-only artefact from becoming a damage claim. Prefer a visibly less sharp, well-supported result to an attractive unsupported boundary.

Success is not “the image looks sharper.” It is a complete chain from source item IDs and acquisition dates to aligned COGs, model/checkpoint hashes, calibrated decision rules, held-out metrics and a demo that makes limitations apparent. Operational rapid response, building-level destruction, and independent 2.5 m ground truth are outside the evidence currently available.

## Current gate and next decision

| Risk | Current evidence | Decision |
| --- | --- | --- |
| R1 model/compute | Genuine pretrained 128×128 forward passed; direct 64/512 inputs fail the loaded 512×512 Fourier mask. PyTorch allocation was measured on 6 GB hardware, with a 4 GiB allocator cap. | Prove a source-faithful small-tile route and physical 4 GB compatibility before product inference. |
| R2 Wayanad imagery | 12 accepted pre-event dates; first accepted post-event date 2024-12-06, 129 days after the event. | Keep dates visible and restrict claims to retrospective comparison. |
| R3 pairs | WorldStrat metadata, size, licence and regional AOIs audited. Three train-AOI crops were subsequently prepared with documented affine/radiometric reconstruction. | Treat prepared targets as smoke-test data, not physically calibrated HR truth. |
| R4 labels | A real Colombia pre/post crop with inventory labels and a Landslide4Sense mirror image/mask were obtained. | Landslide4Sense is single-date segmentation; build separate held-out temporal calibration/evaluation sets. |
| R5 fine-tuning | Last reported probe: zero steps. The later preparation ledger records three candidate train pairs and a connected Colab T4 with isolated environment setup in progress. | Obtain a completed 50-step Colab log and reloadable checkpoint; refresh the risk report with the actual result. |

The status above is deliberately time scoped. The other chat may continue Phase 0; new results must be incorporated from committed evidence before changing the gate. The [Phase 0 closure plan](superpowers/plans/2026-09-29-trustsr-phase0-closure.md) is the immediate execution document. **No Phase 1 module is claimed complete.**

## System contract

```text
STAC L2A item IDs + SCL policy + AOI
  → full-grid, dated 10 m RGBN stack and validity COGs + manifest
  → pinned SEN2SR-lite, fine-tuned weights, overlap-aware tiled ×4 reconstruction
  → eight transform/inverse runs per date; NDVI per run; pre/post moments
  → 2.5 m NDVI drop + 10 m parent support + masks
  → class COGs, uncertainty and unsupported diagnostics, metrics
  → offline, date-labelled Wayanad demo; optional bounded one-tile live run
```

Each arrow has a contract: CRS, affine transform, bounds, band order, reflectance conversion, mask semantics, date, source IDs, checksums and software/model revision are recorded. A failed contract stops the run with a named input or URL. Every raster result is a COG. The SR affine is the 10 m affine multiplied by a 1/4 scale; width and height are ×4 and bounds remain equal. Resample the categorical SCL mask by nearest neighbour. Do not silently crop dates to their intersection.

| Unit | Input → output | Test and failure boundary |
| --- | --- | --- |
| `trustsr.fetch` | AOI and date window → aligned four-band stack, quality masks, JSON manifest | AOI-level SCL cloud/coverage policy; at least three accepted pre-dates; explicit list of checked post-dates if none qualifies; full reference-grid reprojection. |
| `trustsr.sr` | One 10 m RGBN tile/COG → exact-grid ×4 tile/COG | Pinned upstream loading API and weights; bounded overlap and feathering; CPU deterministic stitch test; memory/OOM retry to 64 only if 64 is genuinely supported. |
| `trustsr.trust` | Each clear date's eight dihedral SR runs → per-pixel image moments and NDVI moments | Transform/inverse identity, NDVI before aggregation, pre-event pooling across dates, bounded memory. Dispersion is sensitivity, not calibrated probability. |
| `trustsr.change` | Pre/post NDVI moments, original 10 m signal, validity masks → class/diagnostic COGs | `d = NDVI_pre_mean − NDVI_post_mean`; `σ = sqrt(σ_pre² + σ_post²)`; `k` configured; parent support independently calculated at 10 m; NO_DATA overrides all. |
| `trustsr.evaluate` | Held-out pairs and landslide labels → metrics JSON and report | Masked PSNR/SSIM/data range, downsample spectral error, same-grid baseline comparison, leakage-safe geographic/event splits, unsupported count. |
| `demo/` | Precomputed COGs/manifest/metrics → local map | Date and model-reconstruction labels, 10 m/2.5 m toggle, before/after swipe, class legend, metrics provenance, one-tile live cap. |

The five valid-pixel classes are mutually exclusive: **OBSERVED** when a significant 2.5 m drop and a changed 10 m parent agree; **INFERRED** when the parent changed but the 2.5 m test is uncertain; **UNSUPPORTED** when only the 2.5 m test fires (excluded from positive change but counted); **NO_CHANGE** otherwise; and **NO_DATA** for cloud, shadow or invalid observations on any selected date. “Observed” supports a 10 m change, not a surveyed subpixel outline. Report OBSERVED and OBSERVED∪INFERRED separately; the latter equals the parent-positive footprint on the valid 2.5 m grid and must not be sold as improved binary detection.

## Engineering decisions to freeze before Phase 1

1. **Reference grid and bands.** Select the first accepted pre-event 10 m grid; preserve its CRS and full bounds. Storage and model band orders may differ; name bands and reorder explicitly. SCL is quality metadata, never an SR spectral input.
2. **Radiometry.** Preserve provider encoding and document actual scale/offset. The WorldStrat smoke preparation uses empirical per-band harmonization of SPOT DN; it does not establish physical reflectance or downstream NDVI fidelity. A demo checkpoint needs separate held-out spectral validation.
3. **Model shape and memory.** The inspected pinned loader's `HardConstraint` multiplies by a fixed 512×512 mask. A 64-input retry cannot be assumed to work; padding 64 to 128 would not prove lower peak memory. Resolve and remeasure this before treating OOM halving as implemented end to end.
4. **Change calibration.** Set `k = 2.0` as a provisional configuration value only. Define the 10 m parent threshold independently, choose `k` on labelled calibration events, and report performance on separate held-out events. Do not tune to the one selected positive Colombia crop or the Wayanad display.
5. **Label scope.** The Colombia inventory supplies event polygons rasterized to 10 m and reconstructable pre/post imagery. Landslide4Sense supplies single-date 10 m segmentation masks. [GLaD4CD](landscape.md) is an untested packaged bi-temporal change-label candidate. None is currently verified as independent 2.5 m landslide boundaries or building damage counts. Match every reported metric to the label type and grid it actually tests.
6. **Reproducibility.** Put paths, thresholds, SCL classes, AOI, dates, tiling, seeds and model IDs in pinned YAML; keep large imagery and checkpoints out of Git. Record provenance and hashes for generated artifacts, not only source URLs.

## Delivery sequence

**Gate 0 — close Phase 0.** Resolve R1's non-native tile shape without weakening the spectral constraint, demonstrate the memory policy, complete R5 on verified pairs in Colab, and update RISK_REPORT.md with pass/fail numbers. Reassess the delayed Wayanad date and labels. Stop if any required risk remains blocked.

**Increment 1 — trustworthy inputs and baseline.** Implement `fetch` and the original 10 m NDVI-change baseline. Ship an aligned stack/manifest, explicit cloud audit and CPU georeference tests. This is useful even if SR later fails.

**Increment 2 — SR and trust signals.** Implement tiled `sr`, then `trust`; demonstrate exact-grid output, seam behaviour, spectral consistency and transform inversion on synthetic inputs before running real imagery.

**Increment 3 — change rule and calibration.** Implement the five-class map and count unsupported pixels. Keep the rule and parent threshold in config. Calibrate only on event-held-out labels with documented label resolution.

**Increment 4 — evaluation and demo.** Compare 10 m, 2.5 m, and gated results on a common labelled grid; publish metrics with denominators and confidence limits if defensible. Build the offline Wayanad experience from immutable outputs. Treat live one-tile inference as optional after the offline path is reliable.

Each increment is a separate, reviewable commit with its own CPU tests. The README's reproduction section must reflect the actual commands and outputs after each increment; it must never imply that a planned demo already runs. The exact technical baseline and known alternatives remain in [design.md](design.md).

## Release and presentation gate

A jury-ready claim needs a source ID or measured artifact, a valid comparison, and a clear limit. In particular:

- Show the acquisition dates and 129-day gap in every Wayanad view; no immediate-response statement.
- Show the unsupported-pixel count even if it weakens the visual story.
- Label 2.5 m detail as reconstructed; label original 10 m support separately.
- Do not report 2.5 m F1 against an upsampled 10 m mask as independent high-resolution accuracy.
- Keep a local precomputed demo, its manifest, and its metric report together; if live GPU or network access fails, the evidence remains reviewable.
- Publish a dependency and model/checkpoint lock only from a tested runtime. The current [Windows lock](../requirements-lock-windows.txt) is Phase 0 evidence, not a verified Colab or production lock.

The final deck structure and ten-second workflow are defined in [playbook_strategy.md](playbook_strategy.md); existing alternatives and data-fit questions are in [landscape.md](landscape.md). The plan for closing the current gate is deliberately separate from Phase 1 execution.
