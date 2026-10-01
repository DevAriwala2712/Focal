# Addendum to the six-laws experiment prompt (2026-09-29)

Append this to the existing prompt. All its ground rules apply: branch `experiments/six-laws`, results in `experiments/results/eN.json` with the repo schema, `evidence: synthetic|real`, BLOCKED is never PASS, no edits to AGENTS.md, RISK_REPORT.md or the hard constraint. "6A" in the original prompt means E6.

Read `docs/architecture/TRUSTSR_ARCHITECTURE.md` (ADR-001 to ADR-005) before starting.

## E1 · Step 0 — identify the pinned Fourier mask (run before the tiler)
Load `hard_constraint.safetensor` from the pinned cache through the existing loader. Report: shape, dtype, number of unique values, min/max, the radial profile (mean by integer distance from the centre), and the max |difference| against each upstream family in `sen2sr/models/tricks.py` built for (512, 512) with radius 64: `ideal_filter`, `gaussian_filter`, and `butterworth_filter`/`sigmoid_filter` with the best-fitting order or sharpness (report the fit, don't guess). Write `experiments/results/e1_mask.json`.
- If one family reproduces the stored mask within 1e-6: record `mask_family` and note that a 256×256 version is reproducible. Do **not** enable 64 px inputs (ADR-001).
- Set the E1 stride from this result: smooth mask → test 96 and 112; ideal → 96 only. Crop at least 16 input px per tile side except at scene edges.
- Compare your real-model seam numbers with the synthetic probe in `experiments/probes/results/fft_edge_probe.json` and say whether the network's receptive field adds edge error beyond the constraint.

## E9 · Footprint-level post-event date audit (real data only)
Idea (ADR-002): the post-event date should be chosen by validity over the scar, not over the whole 10 km AOI.
1. Before looking at any post-event SCL, write into `configs/experiments.yaml`: footprint buffer (m), minimum valid fraction inside the footprint, SCL classes (reuse phase0.yaml), and the candidate dates (every 2024 acquisition after 2024-07-30 with AOI cloud+shadow < 100% in `risk/results/r2_acquisitions.json`).
2. Footprint: 10 m NDVI drop between the pooled January 2024 pre-dates and 2024-12-06, thresholded with the configured parent threshold, then dilated by the buffer. Save it as a COG with its manifest. It is used only for date selection, never as a label.
3. For each candidate date, read SCL on the reference grid (nearest neighbour) and report valid fraction inside the footprint, pixel count and earliest date passing the threshold.
4. Also write a per-pixel "first clear post-event date" COG over the footprint.
5. Repeat the pre-event search for 2023 acquisitions in the same calendar months as the chosen post-date; report how many clear 2023 dates exist.

Keep if a date earlier than 2024-12-06 passes the threshold. Otherwise report FAIL with the per-date table — that is a valid finding for the pitch. No network or no STAC access → BLOCKED naming the URL.

## Report additions
In `experiments/SIX_LAWS_REPORT.md`, add rows for E1-step-0 and E9, and in section 2 list the AGENTS.md lines quoted in ADR-001 and the R2 rule affected by ADR-002.
