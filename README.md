# TrustSR

TrustSR is a four-band Sentinel-2 L2A 10 m → model-inferred 2.5 m (×4) research pipeline for vegetation-change assessment. A 10 m parent-pixel gate distinguishes **OBSERVED support**, **INFERRED fine position**, **UNSUPPORTED SR-only change**, and **NO_DATA**. “Observed” means support in the original 10 m NDVI signal; it does not mean a 2.5 m sensor observed the boundary or that a landslide was independently confirmed.

**Critical location correction:** the brief's point (11.782°N, 76.233°E) is 36.43 km from the [NRSC/ISRO mapped Chooralmala landslide point](https://bhuvan-app1.nrsc.gov.in/disaster/usrtasks/landslide/doc/Charter_1029_VAP_3_31july2024.pdf) (11.466763°N, 76.136272°E). Phase 0 audited the supplied point; the event-site demo uses a separate audit and never conflates the two. See [the correction record](docs/location_correction.md).

## Run the offline demo

The repository bundles the real precomputed event-site tile as Cloud-Optimized GeoTIFFs in [demo/assets](demo/assets). From the project root on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run demo/app.py
```

Open `http://localhost:8501`. The browser shows a before/after swipe, a 10 m / 2.5 m switch, the trust overlay and counts. The optional **Run live on GPU** button needs the full source stack created below and an NVIDIA CUDA GPU. The checked-in 2.3 MB fine-tuned checkpoint is a 50-step smoke result, not an accuracy-validated production model. No data is fabricated when source assets are missing.

## How to reproduce the Wayanad demo

The tested runtime is native Windows 11, Python 3.13, an RTX 4050 Laptop GPU with 6,141 MiB; GPU jobs enforce a 4 GiB PyTorch allocator cap. The cap is not proof of operation on a physical 4 GB GPU. Python 3.10+ is required; the pinned direct dependencies are in [requirements.txt](requirements.txt). WSL needs its own CUDA driver/runtime and rasterio wheels.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m scripts.audit_event_aoi
.\.venv\Scripts\python.exe -m scripts.fetch_wayanad
.\.venv\Scripts\python.exe -m scripts.run_demo_tile
.\.venv\Scripts\python.exe -m scripts.evaluate_results
.\.venv\Scripts\python.exe -m scripts.package_demo
```

The audit reads AOI-level SCL for all 2024 acquisitions. Fetch uses its selected three most recent clear pre-event dates and first clear post-event date, checks each product's BOA offset/quantification metadata, reprojects to the first pre-event 10 m grid, and writes `data/wayanad/event_stack.tif` plus a manifest. The full stack and large source caches are ignored by Git. `run_demo_tile` processes a central 128×128-pixel input tile with eight dihedral runs per date, then writes aligned COGs and `data/wayanad/demo/manifest.json`. `package_demo` copies only validated real tile outputs into the offline bundle and rejects a mismatched AOI.

For the event-site 10 km AOI, the first clear post-event acquisition under the configured ≤10% cloud/shadow and ≥99% coverage rule is **16 December 2024**, **139 days after the landslide**. The result is a delayed, seasonally confounded comparison, not immediate disaster response. The earlier Phase 0 December 6 date belongs only to the incorrect brief point.

The checkpoint can be regenerated from the three provenance-verified WorldStrat training pairs with `python -m risk.worldstrat_prepare` followed by `python -m scripts.finetune_local`. Their HR values were empirically harmonized for a smoke test, not independently physically calibrated. The Phase 0 Colab T4 run is separately recorded in [RISK_REPORT.md](RISK_REPORT.md) and [the notebook](notebooks/colab_finetune.ipynb).

## Evidence and limits

See [results/REPORT.md](results/REPORT.md) and [results/metrics.json](results/metrics.json). The three WorldStrat PSNR/SSIM values are **training-pair diagnostics only**. Held-out SR fidelity, calibrated `k`, and landslide-detection F1 are **unavailable**: an independent multi-event paired label split has not been prepared. `scripts/calibrate_k.py` deliberately refuses the single Phase 0 access-proof crop, retaining default `k=2.0`. Landslide4Sense supplies single-image masks, not before/after truth. The published 507-building / 8.38-km-road figure is contextual research from a different sensor/method, not a TrustSR measurement.

The required modules are [fetch.py](trustsr/fetch.py), [sr.py](trustsr/sr.py), [trust.py](trustsr/trust.py), [change.py](trustsr/change.py), and [evaluate.py](trustsr/evaluate.py). CPU synthetic tests cover grid identity, spectral consistency, transform inverses, tile stitching/OOM fallback, known change classes and COG outputs. The original Phase 0 evidence remains in [RISK_REPORT.md](RISK_REPORT.md), and [design.md](docs/design.md) records the rejected diffusion and from-scratch alternatives.
