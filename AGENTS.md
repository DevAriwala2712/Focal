# Task
Build "TrustSR": a Sentinel-2 super-resolution pipeline (10 m → 2.5 m, ×4) with a trust-gated
before/after change detector for landslide damage, demoed on the Wayanad landslide (Kerala, India,
30 July 2024, ~11.782°N 76.233°E). This is for the Smart India Hackathon grand finale; the problem
statement is from NTRO (National Technical Research Organisation).

The core idea: super-resolution invents detail, and in change detection an invented edge looks like
damage. Every change we report must be labelled as observed or model-inferred. Trustworthiness beats
visual sharpness in every trade-off you make.

# Hard constraints (read these twice)
- Compute: Google Colab and one RTX 4050 Laptop GPU (6 GB VRAM, user correction; measured 6141 MiB). No training from scratch. Every GPU
  step must run on 4 GB with tiling; add automatic tile-size reduction on CUDA OOM.
- Start from the pretrained SEN2SR lightweight model (github.com/ESAOpenSR/SEN2SR). Read its README
  and source to get the real loading/inference API; do not guess function names. Fine-tune only.
- Bands: only the four 10 m bands (B02, B03, B04, B08). No 20 m or 60 m bands.
- Geospatial integrity: every raster output is a Cloud-Optimized GeoTIFF in the same CRS as its
  input, and the 2.5 m grid is exactly aligned to the 10 m grid (each 10 m pixel = 4×4 SR pixels).
- Never fabricate data. If imagery, a dataset, or a download is unavailable, stop that step and report
  what you checked.

# Phase 0 — Risk tests FIRST. Stop after this phase and report.
Write each as a script in `risk/` and record results in `RISK_REPORT.md` (pass/fail, numbers, and
what it means for the plan):
- R1 Model fits: load SEN2SR-lite, run one 128×128 (10 m) tile; report peak VRAM, time per tile,
  and whether a 512×512 tile fits in 4 GB.
- R2 Imagery exists: via STAC (Microsoft Planetary Computer `sentinel-2-l2a`, or Copernicus Data
  Space), list L2A scenes over a ~10×10 km AOI centred on Wayanad from 2024-01-01 to 2024-12-31 with
  AOI-level cloud %, using the SCL band. Identify ≥3 clear pre-event dates and the first clear
  post-event date. If none is clear post-event, say so plainly.
- R3 Paired training data: find WorldStrat (Sentinel-2 + SPOT pairs). Report total size, how many
  AOIs fall in India or South Asia, and licence.
- R4 Calibration labels: check whether a labelled pre/post Sentinel-2 landslide dataset is
  downloadable (start with arXiv 2405.20161 "Landslide mapping from Sentinel-2 imagery through
  change detection"). Also confirm Landslide4Sense download. Report what labels each actually gives.
- R5 Fine-tune smoke test: run 50 fine-tuning steps of SEN2SR-lite on a few WorldStrat pairs in
  Colab; confirm loss decreases and a checkpoint saves.
Do not start Phase 1 until the report is written; end your turn with a summary of it.

# Phase 1+ — Components (one module each, testable alone)
Repo layout: `trustsr/{fetch,sr,trust,change,evaluate}.py`, `demo/`, `configs/*.yaml`,
`tests/`, `scripts/` (CLI entry points), `notebooks/colab_finetune.ipynb`.

1. fetch — `fetch(aoi, start, end) -> stack.tif`: pulls L2A 10 m bands + SCL cloud mask, selects
   clear dates (config: max cloud %, min pre-event dates = 3), reprojects all dates onto the first
   pre-event grid, writes an aligned multi-date stack plus a JSON manifest of dates used.
2. sr — `super_resolve(tile) -> tile_2p5m`: SEN2SR-lite with fine-tuned weights; tiled inference
   with overlap and feathered blending (no visible seams); preserves georeferencing.
3. trust — for each date, run sr under the 8 dihedral transforms (flips/rotations, inverted
   after). Pre-event: pool runs across all clear pre-event dates. Output per-pixel mean image and
   per-pixel std for NDVI (compute NDVI per run, then mean/std — not NDVI of the mean).
4. change — per 2.5 m pixel: d = NDVI_pre_mean − NDVI_post_mean, σ = sqrt(σ_pre² + σ_post²).
   Also compute the same NDVI drop on the original 10 m data ("parent" pixel).
   Classes: OBSERVED = d > k·σ and 10 m parent shows change; INFERRED = 10 m parent shows change
   but d ≤ k·σ (real change nearby, exact shape is the model's guess); UNSUPPORTED = d > k·σ but
   parent shows no change (likely hallucination → treated as no change, but counted and logged);
   NO_CHANGE; NO_DATA (cloud/shadow in any date). k from config, default 2.0 until calibrated;
   `scripts/calibrate_k.py` picks k maximising F1 on the Phase-0 calibration set.
5. evaluate — (a) SR fidelity on held-out WorldStrat tiles (and OpenSR-test via the `opensr-test`
   package if it installs): PSNR, SSIM, plus spectral consistency = error between SR output
   downsampled back to 10 m and the 10 m input. (b) Downstream: landslide detection F1 at 10 m vs
   2.5 m vs 2.5 m + trust gate (Landslide4Sense / calibration set), and the UNSUPPORTED pixel
   count. Write `results/metrics.json` and a short `results/REPORT.md` table.
6. demo — Streamlit app: map with a before/after swipe slider, a toggle between 10 m and 2.5 m,
   the change overlay (OBSERVED red, INFERRED amber, NO_DATA grey), and a metrics panel. Default
   mode loads precomputed Wayanad outputs from disk; a "Run live" button processes one small tile
   end-to-end on the local GPU.

# Error handling
- No clear post-event image in the window → raise a clear error listing dates and cloud % checked.
- CRS/extent mismatch between dates → reproject to the reference grid; never silently crop.
- CUDA OOM → halve the tile size and retry, down to 64 px; then fail with a message.
- Any download failure → retry 3× with backoff, then fail naming the URL.

# Tests (pytest, CPU-only, tiny synthetic rasters, run in < 2 min)
- Georef: SR output bounds equal input bounds; pixel size exactly 1/4.
- Spectral consistency: SR output downsampled to 10 m stays within a tolerance of the input.
- Trust: dihedral transform + inverse is an identity on a test array.
- Change: synthetic scene with a known square NDVI drop + noise → square is OBSERVED, an injected
  SR-only artefact is UNSUPPORTED, masked pixels are NO_DATA.
- Tiling: stitched overlapping-tile output equals a single-pass result on a small image.

# Working rules
- Python 3.10+, PyTorch, rasterio, numpy, pystac-client, streamlit, pytest. Pin versions.
- All parameters in `configs/*.yaml`; no hard-coded paths or thresholds.
- Commit after each component with its tests passing. Keep the README's "How to reproduce the
  Wayanad demo" section current.
- When uncertain about an external API or dataset detail, check the source and state what you
  verified.

# Reminder of the non-negotiables
Phase 0 first, then stop and report. 4 GB VRAM. Only B02/B03/B04/B08. Georeferencing preserved
exactly. Never invent data or numbers — report what's missing instead.
