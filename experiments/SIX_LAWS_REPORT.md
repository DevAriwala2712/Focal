# Six-laws experiments: report

Branch `experiments/six-laws`, one commit per experiment. Results: `experiments/results/eN.json`. Settings: `configs/experiments.yaml` (every threshold below was fixed there before the run it governs). Nothing in this report edits the model's Fourier hard constraint, `AGENTS.md` or `RISK_REPORT.md`.

**Where this ran:** macOS arm64, CPU only, no CUDA. **Deviation from the pins:** torch 2.8.0 (macOS build) instead of the pinned `2.8.0+cu126`, which does not exist for this platform; every other pin from `requirements-phase0.txt` was kept (only the torch/torchvision pair was swapped). Every `eN.json` records this under `environment`. The real pinned SEN2SR-lite (revision `469e9b3a…`, all four artifact hashes verified by the existing loader) ran on CPU. **CPU hashes and timings are macOS-arm64-specific.** MPS was not used as a CUDA substitute.

**Real data fetched (E6 only, approved):** Sentinel-2 L2A B04, B08 and SCL, the 12 R2 clear pre dates plus 2024-12-06, R2 AOI grid at 10 m, from Planetary Computer, hard cap 250 MB (`e6.max_fetch_bytes`). Measured with a per-process counter (`nettop`): **83.1 MB received for 12 dates** (about 6.9 MB per date); the first date came from an earlier aborted run and is not in that figure, so the total is about 90 MB (estimate for that one date). Log: `experiments/results/e6_fetch_log.json`. B02/B03 were not fetched.

## 1. Results

| ID | Idea | Status | Evidence | Key numbers | Verdict |
|---|---|---|---|---|---|
| E1 | Overlap tiler at native 128 px | **PASS** (synthetic part); real-crop part **BLOCKED** | synthetic | AOI 128/300/517/1000 px use 1/9/36/121 tiles, all inputs exactly 128×128. Output exactly 4×, affine exactly `input × scale(1/4)`. Spectral consistency per-band RMSE 0.0041–0.0068 (tolerance 0.01); per-band max 0.040–0.072. A no-model bicubic control gives RMSE 0.0039–0.0064, so the residual is the constraint's own filter mismatch, not the tiler | **ADOPT** as the fix for R1 (mechanics). Not shown on real imagery |
| E2 | Tile invariance (grid offset 0 vs 32) | **PASS** | synthetic | Interior (different tile in each run): max \|Δ\| 3.3e-7, p99 1.3e-7 (limit 0.02). Seam/overlap zones: max 4.4e-4, p99 4.4e-5. NDVI \|Δ\|: interior 2.7e-5, seam max 0.021 / p99 6.4e-4. Seam score (gradient across seams ÷ elsewhere, same sub-pixel phase) 1.03 (offset 0), 1.07 (offset 32); a known injected step reads 1.44 hard-cut vs 1.03 feathered | **ADOPT** (stride 96, feather 32) |
| E3 | Hash-checked reruns | **PASS** on CPU; GPU half **BLOCKED** | synthetic | CPU SHA-256 identical across fresh-process reruns at 1 and 4 threads, and identical across the two thread counts (**macOS-arm64-specific hashes**). Control: reversing tile order changes the hash (max Δ 1.5e-7), so a fixed order is a real requirement | **ADOPT** for CPU. GPU drift **NEEDS DATA**: needs the RTX 4050 machine |
| E4 | Memory gate at 128 px | **BLOCKED** | (synthetic input; needs GPU) | No memory number exists. Stage-tracking code is implemented and unit-tested against a fake memory API; it has never run on CUDA | **NEEDS DECISION** only after a run on the RTX 4050 machine; currently no evidence either way |
| E5 | Parent-first cascade | **PASS** (synthetic mechanics); Wayanad part **BLOCKED** | synthetic | 36 tiles: 4 fired, 12 ring, 5 audit, 15 skipped (**41.7 % skipped**). Forward passes 168 vs 288 (**58.3 %**). On 2,329,600 fully-processed pixels: stats bitwise identical (Δ = 0.0), class agreement **100 %**. Boundary pixels partly covered by skipped tiles: 997,888 px, 99.57 % agreement. Audit UNSUPPORTED: 61,269 found in 221,729 audit-only valid px (27.6 %); estimate for skipped area 221,780 px vs full-run reference 221,188 | Mechanics **ADOPT**; value on real data **NEEDS DATA** |
| E6 | Season-matched (January) pre dates | **FAIL** (hypothesis falsified) | **real** | On 186,843 comparable stable pixels (18.7 % of the AOI): false-drop fraction **0.81 % pooling all 12 pre dates vs 0.95 % pooling January only**. Difference (all12 − Jan) −0.13 pp, block-bootstrap 95 % CI [−0.20, −0.08] pp, i.e. January is *worse*, not better. Mean NDVI on stable pixels: all-12 pool 0.723, January 0.743, post (2024-12-06) 0.810. Post-monsoon December is greener than every pre date, so January is not the matching season here | **REJECT** January-only pooling (this AOI, this post date) |
| E7 | Streaming moments (Welford) | **PASS** | synthetic | Max abs error ≤ 3.9e-16 at 8/32/96 runs (limit 1e-6). Streaming peak 2.4–3.0 MB flat vs stacked control 10 → 35 → 102 MB | **ADOPT** |
| E8 | 4 vs 8 dihedral transforms | **BLOCKED** | real (needed) | The verdict needs real 10 m RGBN tiles run through the real model on the RTX 4050 machine. Synthetic mechanics only: ρ(σ4, σ8) = 0.964, five-class agreement 97.1 % (bar: 0.95 / 99 %) | **NEEDS DATA**; keep 8 until then |

Test suite: 91 tests pass in about 3 s (the existing 23 included). Exit codes: E1, E2, E3, E5, E7 exit 0; E4, E6, E8 exit 2 (E6 is a real FAIL, E4 and E8 are BLOCKED).

### Things found along the way (each changed the experiment, none tuned a pass mark)
- **E2 interior was vacuous at first.** Both grids contain the same corner tiles, so a naive "interior" was identical computation (Δ ≈ 3e-7 by construction). Interior is now split into *different-tile* (informative, the one the keep rule applies to) and *same-tile* (sanity, exactly 0). The 0.02 tolerance was not touched.
- **E2 seam score was blind at first** (ratio 1.22 for both feathered and hard cut) for two reasons: gradients are larger at the 4×4 sub-pixel boundaries where seams sit, and the model's tiles agree so closely that even a hard cut is invisible. It now compares same-phase gradients and has a positive control with an injected step.
- **Where the model stops being tile-invariant:** raw tile-to-tile disagreement in overlaps is float noise (≤ 3.6e-7) at ≥ 8 input px from a tile edge, 2.4e-3 at 4–8 px, 1.9e-2 at 2–4 px, 0.18 at 0–2 px. The border effect reaches about 8 input px (32 output px), which the 32 px overlap plus feathering are positioned to hide.
- **E5 parent mask fires spuriously over water/dark pixels** (median red+NIR 0.035): NDVI is ill-conditioned there and noise alone crosses the 0.15 threshold, so every tile fired. E5 uses a local NDVI floor of 0.05, standing in for SCL water/shadow masking on real data.
- **E5's 27.6 % UNSUPPORTED is a synthetic artefact**, not SR hallucination: date-to-date noise exceeds the dihedral σ (σ measures model sensitivity, not radiometric noise). It does show that a dihedral σ alone can badly under-cover real inter-date variability; the design already pools pre dates for this.
- **E8 validity differs between the 4- and 8-sets** (1,668 px valid only with 4 transforms in the synthetic run): any NaN NDVI in one of the extra four runs invalidates that pixel for the 8-set. Comparisons use pixels valid under both.
- **E6 stable-pixel definition and data checks.** Stable was fixed before results: ≥ 10 of 12 valid pre dates, NDVI std across them ≤ 0.05, outside a 1.5 km disk around the project's stated event point (no labelled polygons exist). Sanity checks on the fetched data: 92–100 % valid coverage per date, processing baselines 05.10/05.11 (the −1000 offset convention applies; the COGs carry no scale/offset tags, so the ESA convention was applied, not read from the asset), plausible dry-season NDVI dip (0.74 in January to 0.52 in early March) and recovery. The first fetch attempt was stopped after one date because a system-wide byte counter read about 2 MB/s of unrelated idle traffic; it was replaced by a per-process counter before continuing.
- **Possible AOI error (unverified, needs a human check).** The configured centre 11.782°N, 76.233°E matches Wikipedia's infobox for the 2024 Wayanad landslides (11°46′54″N). A web search for the slide's own coordinates returned the crown/headscarp at about 11.465°N, 76.135°E with an ~8 km runout (Mundakkai–Chooralmala), roughly 35 km south of the R2 AOI. That looks like a degrees-minutes typo (46′ vs 27′) in the source. If right, the R2 10 × 10 km AOI does not contain the slide, which would affect every Wayanad claim (R2 clear dates, and the E6 result above, which would then describe stable land 35 km from the slide and not the slide itself). I did not edit `AGENTS.md`, `docs/`, or configs for this; please verify against a primary source (e.g. GSI or the ICL report) before Phase 1.

## 2. Evidence for the human decision on the 64 px fallback (E4)

**Measured evidence:** none new. E4 is BLOCKED (no CUDA GPU here; it needs the RTX 4050 machine). The only existing measurement is R1 (RTX 4050 Laptop, 6 GiB card, allocator capped at 4 GiB): native 128 px input peaks at 100.48 MiB allocated / 128 MiB reserved. E1–E3 add, on CPU, that a 128 px tiler covers AOIs up to 1000 px with only 128×128 inputs, so the model itself never needs to run at another size. R1 also measured that direct 64 px and 512 px inputs fail the fixed 512×512 Fourier mask; **the 64 px fallback as written cannot currently execute**, independent of memory.

An allocator cap on a 6 GB card is not a physical 4 GB GPU: it excludes the CUDA context, library workspaces and other processes.

**Lines a human would need to change if they decide to drop the fallback (not changed here):**

- `AGENTS.md:72`: `- CUDA OOM → halve the tile size and retry, down to 64 px; then fail with a message.`
- `docs/design.md:39` repeats the rule ("On CUDA OOM halve tile size down to 64 input pixels, then fail explicitly").
- `AGENTS.md:12–13` (compute constraint: "Every GPU step must run on 4 GB with tiling; add automatic tile-size reduction on CUDA OOM.") would also need rewording, since at a fixed 128 px there is no tile size left to reduce. Batch size 1 and a lower-precision path would be the remaining levers.

Recommended precondition: run `python -m experiments.e4_memory_gate` on the RTX 4050 machine first.

## 3. Blockers

| Item | Exactly what is missing |
|---|---|
| E1 real crop | A cached 4-band (B04, B03, B02, B08) 10 m GeoTIFF at `data/experiments-cache/real_rgbn_10m.tif`, at least 1000 px on a side, with `e1.real_crop.dn_scale` / `dn_offset` set. The E6 fetch has only B04, B08 and SCL, so **B02 and B03 still need a (separately approved) download** |
| E3 GPU reruns | **RTX 4050 machine** (NVIDIA GPU, CUDA torch). MPS was not substituted |
| E4 | **RTX 4050 machine**. A physical 4 GB card is needed for any *physical* 4 GB claim |
| E5 Wayanad | Cached 4-band pre and post 10 m stacks on one grid at `data/experiments-cache/wayanad_pre_10m.tif` / `wayanad_post_10m.tif` (B02/B03 are missing from the E6 cache), and the E5 code path that consumes them, which is not written yet (it only detects absence) |
| E8 verdict | Real 10 m RGBN tile pairs at `data/experiments-cache/real_pre_10m.tif` / `real_post_10m.tif` (B02/B03 not fetched), run through the real model on the **RTX 4050 machine**. Deliberately left BLOCKED here |
| Real-data claims generally | A decision on the possible AOI error above: the R2 AOI may not contain the slide |

## 4. What these results do not show

- **No accuracy claim.** All PASS results use synthetic scenes; they show mechanics, geometry and internal consistency, not super-resolution fidelity, NDVI accuracy or landslide detection skill.
- **No physical 4 GB claim** and no GPU number of any kind. E4 has no measurement.
- **No claim about the 4-vs-8 question (E8)**: its synthetic mechanics run is labelled as such and is not a verdict. The one real-data result (E6) is one AOI, one post date and one season pair, on stable pixels only; it says nothing about detection skill, and if the AOI misses the slide it describes no landslide at all.
- **CPU bit-reproducibility is macOS-arm64-specific**: it holds here (this CPU, OS and the macOS torch 2.8.0 build, at 1 and 4 threads); it does not transfer to other hardware, the pinned `+cu126` build, or GPUs.
- **Spectral consistency is largely enforced by the model's hard constraint**, so the small E1 errors are a check on the tiler, not evidence of SR quality.
- **Uncalibrated thresholds:** the parent drop threshold (0.15), k = 2.0 and the NDVI floors are placeholders for mechanics.
- **No Phase 1 readiness.** Phase 0's gate (R5 unexecuted, delayed post-event imagery, the AOI question) is unchanged by these experiments.
