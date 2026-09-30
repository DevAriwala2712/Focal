# ADR X6: season matching and event-to-image latency for the Wayanad pair

Status: pre-registered (written before any X6 code, tests or data fetch). Branch: worktree of `experiments/exceptional`.
Governs: `experiments/x6_season_latency.py`, `configs/x6.yaml`. Keep rules are the ones in `configs/exceptional.yaml`
(`a6`), repeated here word for word; nothing below may be edited after a result is seen.

## Context

Verified diagnosis (lead engineer, from the existing Wayanad run): pre dates 2024-01-16/21/26 are dry season; the post date
2024-12-06 is 129 days after the 2024-07-30 event and post-monsoon green. Mean stable-pixel NDVI is 0.743 in January against
0.810 in December, so seasonal greening shows up as a negative NDVI drop and the two sides are season-mismatched.
Experiment E6 (old, wrong AOI) already falsified "January-only pooling is better": false-drop 0.95 % Jan-only vs 0.81 % for
all 12 dates pooled, CI of the difference [-0.20, -0.08] pp. Separately every post-event image before 2024-12-06 fails the
footprint-validity rule (>= 90 % of the footprint SCL-valid) on the corrected AOI; 2024-10-27 (72.5 %), 2024-11-01 (43.5 %) and
2024-11-06 (46.5 %) are partially valid. The bottleneck attacked is the measurement quality of the before/after pair, not compute.

## Decisions fixed now

### Experiment (a): anniversary pre dates

* **AOI and grid:** the corrected AOI (11.49 N, 76.16 E, EPSG:32643, 10,240 m square) and the exact 10 m grid from
  `experiments.wayanad_evidence.geo.reference_grid`, so every array is pixel-for-pixel alignable with the existing stack.
* **Audit:** STAC search (Planetary Computer `sentinel-2-l2a`) for acquisitions 2023-10-15 to 2024-01-15; SCL at 20 m over the AOI;
  a date is clear iff AOI cloud+shadow <= 10 % and coverage >= 99 % (classes and thresholds from `configs/phase0.yaml`, the same
  rule and the same code path as the existing whole-year audit). Every audited date is listed in the report, clear or not.
* **Anniversary pool:** the clear dates that fall in **2023-11-01 .. 2023-12-31** (the post date 2024-12-06 is at the centre of this
  window one year earlier, so it is the season-matched window; 2023-10-15..31 and 2024-01-01..15 dates are audited and listed but not
  pooled). If fewer than 3 clear dates exist the experiment does **not** run and the report says so plainly (status FAIL, rule unmet).
* **January pool:** the current pool 2024-01-16, 2024-01-21, 2024-01-26 (arrays read from the existing cache, not re-downloaded).
  **Post:** 2024-12-06 (existing cache).
* **Metric: E6's exact stable-pixel false-drop metric**, by calling the E6 functions (`valid_from_scl`, `reflectance`, `stable_mask`,
  `pooled_mean`, `false_drop`): reflectance `(DN - 1000)/10000` without clipping, NDVI floor `red+nir >= 0.01`, a pixel is a false drop if
  pooled pre NDVI minus post NDVI > the E6 parent-drop threshold **0.15** (`configs/experiments.yaml`, `change.parent_drop_threshold`),
  pool needs >= 2 valid dates (E6 `pool_min_valid`).
* **Stable pixels (defined on pre-event NDVI only, before any outcome is looked at):** computed on the union stack of all pre dates
  of both pools (N anniversary + 3 January). Stable = valid on >= (N + 3) - 2 of them (E6: 10 of 12, i.e. at most two missing),
  NDVI std across the valid dates <= 0.05, and outside a 1500 m disk around each published slide coordinate of
  `configs/wayanad_evidence.yaml` (`aoi.plausibility_points`: crown, Mundakkai, Chooralmala bridge). E6's disk was centred on the old
  (wrong) AOI point, which lies ~34 km outside this AOI, so it is replaced by these published points. The footprint (derived from a
  January-vs-December NDVI drop, i.e. from the outcome image) is **not** used to define stable pixels in the primary analysis.
* **Comparison:** both pools evaluated on ONE shared comparable set (stable, both pools finite, post finite), so the denominators are
  identical. Fractions `f_anniv`, `f_jan`; difference `f_anniv - f_jan`.
* **Uncertainty:** `configs/exceptional.yaml` `statistics`: spatial block bootstrap, 32 x 32 px (320 m) blocks, 2000 replicates, 95 %,
  seed 2024, via `trustsr.bootstrap.block_sums` and `ratio_bootstrap_ci(..., num_b=...)` (paired: same blocks, same denominators).
* **KEEP (a) iff `f_anniv < f_jan` AND the 95 % CI of `f_anniv - f_jan` lies entirely below 0.** Otherwise the anniversary set is not
  adopted on this evidence.
* **Secondary, descriptive (no keep rule):** (i) the same comparison at the evidence-run threshold 0.30 with that run's reflectance
  conventions (clip at 0, NDVI floor 0.05); (ii) stable pixels additionally excluding the footprint; (iii) mean pooled NDVI and mean
  signed drop per pool (the seasonal-greening signature); (iv) **pool-size control**: E6 pooled 12 dates against 3, and a mean over
  more dates is less noisy for reasons unrelated to season, so if N > 3 the false-drop fraction is also reported for every 3-date
  subset of the anniversary dates (min, median, max).

### Experiment (b): post composite

* **Dates and rule:** 2024-10-27, 2024-11-01, 2024-11-06, in that date order; per pixel the **first** date whose SCL is valid
  (not SCL classes 2, 3, 8, 9, 10, 0, 1, 11: the shadow, cloud and invalid classes of `configs/phase0.yaml`). Validity mosaics per date are built exactly as in the existing step 2
  (`scl_mosaic_10m`, cached 20 m SCL replicated to 10 m), so each date's footprint-valid fraction must reproduce the step-2 table
  (72.5 / 43.5 / 46.5 %); if it does not, that is reported as a discrepancy.
* **Footprint:** `experiments/wayanad_evidence/outputs/footprint.tif` (largest parent-positive component, 200 m square buffer). It comes
  from the January-vs-December comparison, so it is an established-by-outcome region used only for date selection, as in step 2.
* **Quantities:** footprint validity of the composite = valid footprint px / footprint px. Event-to-latest-contributing-image gap =
  days from 2024-07-30 to the latest date that supplies at least one **footprint** pixel (the AOI-wide latest contributing date and the
  per-pixel gap distribution, median and maximum over footprint pixels, are reported as well).
* **KEEP (b) iff footprint validity >= 0.90 AND gap <= 99 days** (was 129).
* **Only if KEEP:** fetch B02/B03/B04/B08 for the crop window (rows 256:896, cols 128:640 of the AOI grid) of those dates, write the
  composite as an aligned 4-band COG (DN uint16, band order B04, B03, B02, B08, nodata 0 where no date is valid), a per-pixel
  source-date index COG and a JSON manifest. **No radiometric harmonisation between dates is applied**; processing baselines of each
  date are listed. If not KEEP, nothing is fetched for (b) and the failure is reported.

### Bytes and network

Everything downloaded counts against `fetch.max_fetch_bytes` = 300,000,000 of `configs/wayanad_evidence.yaml`, measured with the
per-process `nettop` counter (`ByteBudget`), checked after every read; each read is retried 3 times with backoff by
`risk.common.retry` and then fails naming the URL. Only B02/B03/B04/B08 (+ SCL as a mask, never as a model input). Stop and report rather
than exceed the cap. No network means BLOCKED, naming the failed calls.

## Failure modes recorded in advance

1. **Composite date seams and mixed illumination.** The composite takes each pixel from a different acquisition (Sentinel-2A / 2B
   five days apart, same relative orbit). Sun angle, atmospheric correction and haze differ, and no harmonisation is applied, so
   lines between source regions can appear as NDVI steps. The per-pixel source index is shipped so downstream code can test for this;
   whether it matters was not measured.
2. **SCL validity is not haze-free.** The first-valid rule trusts SCL. Thin cloud, haze and missed cloud shadow classed as valid vegetation
   or bare soil are exactly what a change detector mistakes for vegetation loss, and the footprint is where partial validity was
   the problem.
3. **Latency barely changes the confounder.** 99 days instead of 129 is 30 fewer days of debris clearing and regrowth; the monsoon
   recovery phase of early November is the same phase as 6 December, only slightly earlier. The latency rule passes by construction as
   soon as 2024-11-06 supplies one footprint pixel; the informative part of (b) is the validity condition.
4. **The composite does not improve season matching to the pre pool by itself.** A late-October to early-November post image is
   matched by a late-October to early-November pre pool, which is neither Nov-Dec 2023 (partly) nor January. This ADR does not test
   composite-as-post against any pre pool (the crop-only fetch cannot support a full-AOI stable-pixel test).
5. **Interannual change in (a).** December 2023 and December 2024 are different years: plantation harvest or rotation, a wetter or drier
   autumn, or a change of processing baseline between 2023 and 2024 acquisitions all show up as false drops against the anniversary pool
   and would bias (a) against the anniversary set. The processing baselines of the 2023 and 2024 dates are listed to make a baseline
   discontinuity visible.
6. **Pool size confounds the comparison** (above, control iv). A difference could come from averaging more dates, not from the season.
7. **Nov-Dec 2023 is the retreating monsoon**: few clear dates are plausible, pools may have N = 3 or fewer, and cloudy-date
   leakage through SCL would add noise. If N < 3 the experiment reports it and stops.
8. **Stable mask is pre-event data, so it excludes pixels with strong seasonal NDVI swing** (std <= 0.05 across a union that spans
   Nov-Dec and January). The comparison therefore speaks for seasonally steady vegetation, not for every land cover (paddy, bare
   soil, plantation clearings are under-represented).
9. **One AOI, one post date, one year pair.** A single comparison, not a general result; the difference of two fractions with a block
   bootstrap CI says nothing about detection skill (no labels are used).

## What this ADR does not claim

No detection accuracy, no calibrated threshold (0.15 and 0.30 are placeholders fixed earlier), no claim that NDVI drop equals landslide
damage, no claim about the SR model. A failed keep rule is a result and is reported with its numbers.

## Deviations log

1. **Pre-registered run completed first, both keep rules failed** (config sha256 of the run: x6.yaml `fe37199f...`, combined
   `fbd67e03...`; log kept in `data_a6/run1.log`, first result in `data_a6/x6_prereg_run1.json`): (a) only ONE clear Nov-Dec 2023
   date exists (2023-12-27, 1.03 % cloud+shadow), so experiment (a) did not run, as pre-registered; (b) the composite covers 80.6 % of
   the footprint (< 0.90), gap 99 days, so no composite was fetched. Those verdicts are final and are not revisited below.
2. **Exploratory additions, made AFTER seeing those results and the audit cloud percentages (so post hoc, and labelled so in every
   output). They have no keep rule, cannot change any verdict above, and are reported separately:**
   * **E-a2 relaxed pool.** Because the pre-registered pool is empty, the same false-drop machinery is run on the Nov-Dec 2023 dates whose
     AOI cloud+shadow is <= 50 % (per-pixel SCL validity still applies; the threshold 50 is a post-hoc choice): 2023-11-07, 2023-11-27,
     2023-12-02, 2023-12-12, 2023-12-27. Same stable definition (union of these 5 and the 3 January dates, >= 6 valid of 8, std <= 0.05,
     published-point disks), same metric, same bootstrap, plus the 3-date-subset control. Reported as descriptive numbers with CIs, with
     the explicit caveat that the pool is cloud-contaminated (SCL misses thin cloud) and the choice of threshold was made after looking.
   * **E-b2 cumulative composite curve (no network).** From the cached whole-year SCL: for every post-event acquisition date T from 2024-08-13
     to 2024-12-06, the footprint validity and latest-contributing gap of the first-valid composite of ALL post-event acquisitions up to and
     including T. This shows how much latency the 0.90 validity rule costs, without any fetch or claim of image quality.
   Config keys for these were added to `configs/x6.yaml` after the first run (`exploratory` section); no pre-registered key was changed.
