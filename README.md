# TrustSR

A trust-gated Sentinel-2 super-resolution project for landslide assessment, using Wayanad (30 July 2024) as the showcase.

**Current status: Phase 0 risk harness implemented; gate not passed. No Phase 1 pipeline or demo yet.** See [RISK_REPORT.md](RISK_REPORT.md) for real results and blockers.

## Run Phase 0 on Windows

Use Python 3.10+; the tested runtime is native Windows 11 / Python 3.13.14 on an RTX 4050 Laptop GPU. The existing global Python environment is unchanged.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-phase0.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m risk.r1_model_fits
.\.venv\Scripts\python.exe -m risk.r2_imagery
.\.venv\Scripts\python.exe -m risk.r3_worldstrat
.\.venv\Scripts\python.exe -m risk.r4_labels
.\.venv\Scripts\python.exe -m risk.r5_finetune
```

All probes accept `--config configs/phase0.yaml`. Paths resolve relative to the project root inferred from that config. They write evidence under risk/results/. Exit 0 means PASS; exit 2 means FAIL/BLOCKED, with a JSON explanation. An unavailable network/data prerequisite must not be converted to a pass.

The direct dependency pins are in requirements-phase0.txt. To reproduce the exact tested Windows package set, install requirements-lock-windows.txt with `--extra-index-url https://download.pytorch.org/whl/cu126`. The CUDA wheel is a substantial download. Colab is a separate environment whose actual packages must be recorded; use [the notebook](notebooks/colab_finetune.ipynb).

R2 reads the full year's SCL windows. R3 downloads metadata and country polygons, not the full WorldStrat archives. R4 downloads one actual temporal/label access-proof crop and checks the original Landslide4Sense share plus a pinned IBM-NASA image/mask mirror. Model/data caches and checkpoints are ignored by Git. Prepare real pairs with `python -m risk.worldstrat_prepare`. R5 intentionally stops locally: it requires genuine prepared WorldStrat pairs and Colab. Follow the [data contract](docs/r5_data_contract.md).

## How to reproduce the Wayanad demo

The demo is **not yet implemented**. The executed audit found 12 clear pre-event dates and the first post-event date meeting the configured cloud/shadow threshold on 6 December 2024. That is 129 days after the landslide.

Before building the demo, resolve the risk report's fixed-mask tiling limitation, held-out calibration preparation and real-data Colab training. The intended later route remains: pinned inputs/weights -> aligned 10m COGs -> tiled x4 SR -> trust/change COGs -> held-out evaluation -> offline Streamlit demo. A 6 GB GPU run is not proof of physical 4 GB compatibility.

## Documents

- [Full supplied NTRO problem statement](docs/problem_statement.md)
- [Design and rejected alternatives](docs/design.md)
- [Source register](docs/sources.md)
- [Environment history](docs/environment.md)
- [Provisional timeline](docs/timeline.md)
- [Phase 0 plan](docs/phase0_plan.md)
- [Implementation ledger](docs/implementation-ledger.md)

The finale date remains unconfirmed. No accuracy, fine-tuning success or immediate disaster-response claim is established by this risk harness.
