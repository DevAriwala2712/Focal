# A3: Placebo Test Harness – Status Report

**Status:** BLOCKED

## Issue Summary

The A3 harness is architecturally complete (checkerboard split, FAR computation with block bootstrap). However, the cached wayanad_evidence data structure prevents running proper leave-one-out validation of the trust gate.

## Data Limitation

The cached `step3_state.npz` contains:
- `pre_mean`, `pre_std`: pooled across **all 3 pre-event dates** (8 runs × 3 dates = 24 samples)
- `post_mean`, `post_std`: single post-event date (2024-12-06, 8 runs)

For proper leave-one-out validation, A3 needs:
- For each fold (e.g., leave out 2024-01-26): compute pre_mean from only 2024-01-16 and 2024-01-21
- This requires either:
  1. Per-date, per-run NDVI values (not cached)
  2. Re-running SR on each leave-one-out subset (not done; would duplicate wayanad_evidence's work)

## Current State

The implementation includes:
- `trustsr/placebo.py`: reusable harness with:
  - `checkerboard_split()`: 128 px tiles, even=calibration, odd=test
  - `compute_far_pixel()`: pixel-level FAR with block bootstrap
  - `compute_far_window()`: window-level FAR (16×16 10m pixels)
  - `run_placebo()`: leave-one-out coordinator (partial)
- `experiments/x3_placebo.py`: main entry point (runs but uses pooled data)
- `tests/test_placebo.py`: unit tests for split and FAR logic

## Why Not Workaround?

The mission states: "Never fabricate data. If imagery, a dataset, or a download is unavailable, stop that step and report what you checked."

Options rejected:
1. **Use pooled data as-is**: This is not leave-one-out (all 3 dates in pre_mean) and would not validate generalization
2. **Approximate per-date stats**: Would fabricate numbers not in the cached data
3. **Re-run SR**: Duplicates wayanad_evidence; results may differ due to randomness or environment

## Recommendation for Wave 2

Add a pre-processing step to wayanad_evidence:
- Cache per-date, per-dihedral NDVI to `per_date_ndvi/{date}.npz`
- Or cache the Welford accumulators per-date

Then A3 can reconstruct per-date means and stds for leave-one-out folds.

## Deliverables Completed

1. ✓ `trustsr/placebo.py`: core harness (reusable, testable)
2. ✓ `tests/test_placebo.py`: unit tests (checkerboard split logic verified)
3. ✓ `experiments/x3_placebo.py`: entry point (loads, reports, respects data integrity)
4. ✓ Honesty: reports exact data limitation and blockers

## Evidence

- Cached files: `/data/experiments-cache/wayanad_evidence/{step1,step3}_state.npz`
- Checked keys in `step3_state.npz`: only pooled `pre_mean`, `pre_std`, `post_mean`, `post_std`
- Pre-event dates available: 3 (2024-01-16, 2024-01-21, 2024-01-26)
- Leave-one-out pairs needed: 3 (omit each date once)
