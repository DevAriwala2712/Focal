# TrustSR Phase 0 risk register

**Status: preliminary resource audit; Phase 0 NOT COMPLETE.** Updated 2026-09-26.
This report exists early to make missing evidence visible. It is not a Phase 0 completion report and does not unlock Phase 1. No model-fit, cloud-clearance, fine-tuning or accuracy numbers have been invented.

| Risk | Current status | Verified evidence | Missing before a pass |
| --- | --- | --- | --- |
| R1 model fits | BLOCKED / benchmark not run | RTX 4050 Laptop GPU: 6,141 MiB; installed torch 2.13.0+cpu, CUDA unavailable. Four-band lite README, metadata, loader and wrapper inspected | CUDA environment, pinned weights, real 128/512-input timings and peaks, 4 GiB-budget test, shape/OOM fallback checks |
| R2 imagery exists | NOT RUN | Planetary Computer Sentinel-2 L2A collection page accessible; AOI/time window defined | Full STAC/SCL AOI audit, â‰¥3 clear pre-dates and first clear post-date. Clear post-event availability is UNKNOWN, not absent |
| R3 paired data | PARTIAL; not a pass | WorldStrat Zenodo record lists 107.0 GB of files; processed HR 41.5 GB and L2A 26.8 GB. Metadata CSV downloaded; licences identified | Country-boundary AOI counts for India/South Asia; actual usable four-band paired samples and alignment/radiometry inspection |
| R4 calibration labels | PARTIAL; not a pass | arXiv paper and author repository accessible; inventories plus acquisition notebooks advertised. Landslide4Sense official repository and training link found | Actual pre/post/label sample download; Landslide4Sense archive retrieval and label inspection. Firecrawl request to its training share returned INVALID_ARGUMENT, which is a retrieval-tool failure, not proof the dataset is unavailable |
| R5 fine-tuning | BLOCKED / not run | Separate trainable loader exists; inference loader freezes parameters | WorldStrat sample pairs, working Colab GPU, 50 actual training steps, loss trend, weight changes, checkpoint save/reload |

## Implications

- The available 6 GB GPU is stronger than the originally recorded 4 GB device, but the required compatibility test remains separate.
- Current Python can import CPU PyTorch; rasterio, pystac-client and PyYAML are absent. Build a tested isolated environment before R1/R2.
- WorldStrat regional counts remain unknown: its metadata contains coordinates, not a country column. Do not fabricate a count from the global map.
- The arXiv release requires image acquisition through notebooks. Landslide4Sense is pixel-wise single-image segmentation at about 10 m; it does not itself supply the temporal calibration pairs or independent 2.5 m truth.
- The Wayanad 507 buildings / 8.38 km roads reference is a published Sentinel-1/OSM assessment, not measured TrustSR output or pixel labels.
- Finale date is unconfirmed; the relative schedule in docs/timeline.md is provisional.

## Evidence and next action

Environment output: [environment](docs/environment.md). Verified source URLs and interpretation: [sources](docs/sources.md). Reproducible risk-script specifications: [Phase 0 plan](docs/phase0_plan.md). The five risk scripts have not yet been implemented in this planning pass.

Replace each partial/blocked entry with measured results and linked logs as tests are executed. Preserve failures. Stop and report after all five risks have been assessed; request Phase 1 only then, with scope based on what actually passed.
