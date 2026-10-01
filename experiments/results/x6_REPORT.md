# X6: season matching and latency (Wayanad, corrected AOI 11.49 N 76.16 E)

Evidence: **real** Sentinel-2 L2A (Planetary Computer), SCL as mask only, B02/B03/B04/B08 only. Machine-readable: `experiments/results/x6.json`
(exit code 2 = FAIL). Rules: `docs/adr-x6-season-latency.md`, `configs/x6.yaml`, `configs/exceptional.yaml` (a6). Both pre-registered keep rules **FAILED**.

| Sub-experiment | Pre-registered rule | Result | Verdict |
|---|---|---|---|
| (a) Anniversary pre dates | >= 3 clear Nov-Dec 2023 dates, then f(anniv) < f(Jan) with CI of the difference below 0 | **Only 1 clear date** (2023-12-27, 1.03 % cloud+shadow); the test did not run | **FAIL (not run)** |
| (b) Post composite | footprint validity >= 0.90 AND gap <= 99 d | validity **0.806** (31,456 / 39,019 px), gap **99 d** (2024-11-06 supplies 2,562 footprint px) | **FAIL** (validity) |

## (a) Audit, 2023-10-15 to 2024-01-15 (18 acquisitions, all 100 % coverage; clear = cloud+shadow <= 10 %, from SCL)
Cloud+shadow %: 10-18 40.3, 10-23 52.9, 10-28 86.8, 11-02 53.3, 11-07 25.9, 11-12 88.7, 11-17 62.7, 11-22 60.3, 11-27 39.6, 12-02 46.2,
12-07 76.1, 12-12 24.0, 12-17 100.0, 12-22 90.9, **12-27 1.0**, 2024-01-01 80.2, 01-06 86.8, 01-11 39.1. The retreating monsoon leaves one clear
date; nothing was pooled and no false-drop keep test could be run. Bytes for the audit: 2.9 MB.

### Exploratory, post hoc (ADR deviations log 2; no keep rule; cannot change the verdict)
Relaxed pool = Nov-Dec 2023 dates with cloud+shadow <= 50 % (threshold chosen after seeing the table): 2023-11-07, 11-27, 12-02, 12-12, 12-27 (5 dates)
vs the January pool 2024-01-16/21/26, post 2024-12-06. E6's metric and stable definition (union of 8 pre dates, >= 6 valid, NDVI std <= 0.05, 1500 m disks
around the published crown/Mundakkai/bridge points); block bootstrap 32 px, 2000 replicates, seed 2024.

| Convention | Comparable px (10 m) | f(anniv) | f(Jan) | f(anniv) - f(Jan), 95 % CI |
|---|---|---|---|---|
| E6 exact (threshold 0.15, floor 0.01, no clip) | 83,521 | 2.347 % (1,960 px) | 2.278 % | +0.068 pp [-0.027, +0.166] |
| Evidence run (0.30, floor 0.05, clip) | 83,521 | 1.278 % (1,067 px) | 1.221 % | +0.056 pp [+0.006, +0.123] |
| E6 exact, footprint also excluded | 83,052 | 2.265 % | 2.200 % | +0.065 pp [-0.031, +0.161] |

Reading: on these pixels the anniversary-style pool has **no lower false-drop fraction** than January (point estimate slightly higher; at threshold 0.30 the
CI excludes 0 on the wrong side). Mean stable NDVI: anniversary pool 0.764, January pool 0.748, post 0.788; mean signed drop (pre - post) -0.024 anniversary vs
-0.040 January, so the anniversary pool is closer to the December level in the mean, but that does not reduce the count of pixels dropping > 0.15. All 3-date
subsets of the 5: false-drop 1.58 % (min), 2.28 % (median), 2.51 % (max), so pool-size does not explain it. The comparable set is only 8 % of the AOI (cloudy
dates leave few pixels valid on >= 6 of 8 dates), the pool contains SCL-missed thin cloud, and 2023-11-27 and 2023-12-12 each have two STAC items of the same
tile with different processing baselines (05.09 and 05.10) that the first-valid-tile rule mixes. Processing baselines: 2023 dates 05.09/05.10, January 05.10, post 05.11.

## (b) Composite, from cached SCL (each date reproduces the step-2 footprint-valid figure exactly)
| Date | Platform | AOI cloud+shadow % | Footprint valid | Footprint px supplied to composite |
|---|---|---|---|---|
| 2024-10-27 (89 d) | S2A | 42.3 | 72.5 % | 28,287 |
| 2024-11-01 (94 d) | S2B | 59.9 | 43.5 % | 607 |
| 2024-11-06 (99 d) | S2A | 34.5 | 46.5 % | 2,562 |

Composite footprint validity **80.6 %** (needs 90 %). Latest contributing date 2024-11-06, gap **99 d** (current post: 129 d); per-pixel footprint gap median 89 d,
max 99 d. AOI-wide the composite is 86.8 % valid. Nothing was fetched for the composite and no composite COG exists, as the rule requires.

### Exploratory (no network, no keep rule): cumulative first-valid composite of ALL post-event acquisitions up to date T
Footprint validity: 24.6 % by 2024-08-23 and ~32 % through 2024-10-22; 81.9 % at 10-27; 84.6 % at 11-06; 85.3 % at 11-16 (gap 109 d); it first reaches 0.90 only
at **2024-12-06 (97.1 %, gap 129 d)**, i.e. the current post date. So under this SCL rule no earlier composite meets 0.90, and the 129-day latency is the cost of the rule.

## Bytes
44,571,739 received across all X6 processes (nettop per-PID, cap 300,000,000): audit 2.9 MB, five relaxed-pool dates ~41 MB, remainder STAC searches. Some
processes' counters read low or 0 on cache-only reruns; the total is a lower bound of network use but the fetches are complete and cached.

## Implication for the Wayanad v2 run
Keep the current pair: January 2024 pre dates and post 2024-12-06 (129 d). Neither idea passed its rule: an anniversary pre set cannot be built under the clear-date
rule (1 date), the exploratory relaxed pool shows no false-drop gain, and a partial-latency composite reaches only 80.6 % footprint validity. If latency matters more
than the validity rule, the 10-27/11-01/11-06 composite (80.6 %, 99 d, ~72 % of footprint pixels from the 89-day image) is available as a documented, lower-validity
option, but this experiment did not test it against any pre pool and it is not a KEEP. The single clear 2023-12-27 date could be added as a fourth pre date; nothing here tests that.

## What this does NOT show
No detection accuracy or labels; no test of the composite (or of a Nov pre pool) as post/pre; nothing about seasons other than Nov-Dec vs Jan on one AOI and one post
date; stable pixels are seasonally steady vegetation only; SCL validity is not haze-free; no radiometric harmonisation; the exploratory numbers were chosen after
seeing the audit and are descriptive; the footprint is derived from the January-vs-December NDVI drop.

## Outputs
Committed: `x6.json`, this report. Not committed (regenerable, `data_a6/`): cached DN windows `data_a6/cache/*.npz`, byte log `data_a6/fetch_log.json`, and
crop-window 4-band COGs `data_a6/cogs/anniversary_{2023-11-07,11-27,12-02,12-12,12-27}_crop.tif` (B04,B03,B02,B08,SCL, same CRS EPSG:32643 and 10 m lattice, crop rows 256:896, cols 128:640);
SHA-256 of each is in `x6.json` (`output_sha256`). No composite COGs (keep rule failed).
