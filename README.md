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

The front end is a static site in `demo/web/` (the "Cartographic Precision" design: eight screens sharing one shell) plus a tiny Python server. Every number and image on it is read from the Wayanad evidence run; nothing is hand-typed. Anything the run did not measure is shown as NOT RUN / BLOCKED.

```bash
# 1. (once, or after re-running the evidence steps) build the site data from the run outputs; it re-checks that
#    classify() reproduces the stored class map and that rho matches rho_objects.csv, and refuses to build otherwise
.venv/bin/python scripts/build_web_data.py

# 2. serve it
.venv/bin/python demo/server.py            # http://127.0.0.1:8765

# 3. guards (pages present, numbers match step4.json, downloads whitelisted, no invented claims)
.venv/bin/python -m pytest tests/test_web_demo.py
```

Needs the evidence outputs in `experiments/wayanad_evidence/outputs/` (the large `stack_*.tif` and `sr_post_2p5m.tif` are gitignored; regenerate them with the steps in `experiments/wayanad_evidence/EVIDENCE.md`). `demo/web/data/` is generated and gitignored. Pages load Tailwind and fonts from CDNs, so open them once online before an offline demo.

Screens: How it works · Data & dates (73 real acquisitions, post-event visibility audit) · Map workspace (swipe, 10 m / 2.5 m, live k re-gate, pixel inspector) · Objects (2,217 change objects with ρ) · Evidence · Experiments · Demo (jury view) · Exports (real files, sizes, SHA-256).

Not wired yet: the spec'd "Run live" GPU button (the control is present but disabled; it needs the RTX 4050 machine). The k slider is exploratory only, since k = 2.0 remains uncalibrated.

Current evidence status: 12 clear pre-event dates existed on the old AOI, and the corrected AOI (11.490°N, 76.160°E) run used pre 2024-01-16/21/26 and post 2024-12-06 (129 days after the landslide). Phase 0 is not passed; no accuracy claim is established.

## Documents

- [Project blueprint and acceptance gates](docs/project_blueprint.md)
- [SIH playbook interpretation and jury narrative](docs/playbook_strategy.md)
- [Research landscape and dataset fit](docs/landscape.md)
- [Phase 0 closure implementation plan](docs/superpowers/plans/2026-09-29-trustsr-phase0-closure.md)
- [Full supplied NTRO problem statement](docs/problem_statement.md)
- [Design and rejected alternatives](docs/design.md)
- [Source register](docs/sources.md)
- [Environment history](docs/environment.md)
- [Provisional timeline](docs/timeline.md)
- [Phase 0 plan](docs/phase0_plan.md)
- [Implementation ledger](docs/implementation-ledger.md)

The finale date remains unconfirmed. No accuracy, fine-tuning success or immediate disaster-response claim is established by this risk harness.
