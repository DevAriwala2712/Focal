# TrustSR

Trust-gated Sentinel-2 super-resolution for landslide assessment, with Wayanad (30 July 2024) as the showcase. Planning stage only; no pipeline or validated results yet.

## Project documents

- [Full supplied NTRO problem statement](docs/problem_statement.md)
- [Design and rejected alternatives](docs/design.md)
- [Sources and dataset limitations](docs/sources.md)
- [Measured local environment](docs/environment.md)
- [Timeline](docs/timeline.md)
- [Phase 0 execution plan](docs/phase0_plan.md)
- [Preliminary risk register](RISK_REPORT.md)

## How to reproduce the Wayanad demo

The demo is not implemented. First execute and report the five Phase 0 risks described in AGENTS.md. Verify clear imagery and label/data access before building the pipeline. The intended route is pinned inputs and weights â†’ aligned 10 m COGs â†’ tiled Ã—4 SR â†’ trust/change COGs â†’ held-out evaluation â†’ offline Streamlit demo.

Hardware measured on 2026-09-26: RTX 4050 Laptop GPU with 6,141 MiB VRAM, native Windows 11. Keep the separate 4 GiB compatibility target. The present Python environment has CPU-only PyTorch.

Do not interpret planning documents, model parameter counts, or dataset landing pages as successful GPU, download or accuracy tests.
