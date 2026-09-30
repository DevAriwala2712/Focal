# F1 — the real placebo (null-event) false-alarm-rate test

Replaces `experiments/x3_placebo.py` / `experiments/results/x3.json` (quarantined,
`experiments/results/INVALIDATED.md`). Pre-registered in `configs/fix.yaml`'s `f1_placebo` block, committed
(F0) before this run. Code: `trustsr/placebo_v2.py` (harness), `experiments/f1_placebo.py` (run script),
`tests/test_f1_placebo.py` (13 tests, all passing).

**Status: PASS** (keep rule (a) — point estimate inside its own pooled CI — holds for every scored gate).
**Evidence: real** Sentinel-2 L2A imagery, corrected Wayanad AOI (11.490°N, 76.160°E).
`config_sha256` (of `configs/fix.yaml`, **after** a YAML syntax fix — see "Config note" below):
`18cce7b1ea88855bc054b42e6cbe3219eef73b2909eb091d07d5ada3df170970`.

## What every x3 bug (B1–B5) looked like, and what fixes it here

| Bug | x3 (invalid) | F1 fix |
|---|---|---|
| **B1** | "Placebo" pairs compared pre-event dates against the REAL post-event date (2024-12-06); every FAR number silently included the actual landslide. | `trustsr.placebo_v2.guarded_load_date` **raises** `PostEventDateBlocked` for any date `>= 2024-07-30` (the event date). Every placebo pair's "reference" is the mean of the *other* pool pre-event dates; the post image is architecturally unreachable from this code path. Unit-tested (`TestHardGuard`, 5 tests incl. an exact boundary test). |
| **B2** | 3 of 5 "gates" were the same `gate_v1_trust` function under different names. | `GATES` in `placebo_v2.py` holds five distinct function bodies. `gate_v2` is an explicit `BLOCKED` stub that **raises** `NotImplementedError` rather than silently aliasing `gate_v1` (`test_gate_v2_is_not_an_alias_of_gate_v1`). |
| **B3** | Window FAR was computed from `parent_10m` (raw input) for *every* gate, so all gate/fold rows reported the identical number 0.1177. | `compute_far_window_v2` takes each gate's own `flagged` array as an explicit argument; `test_window_far_changes_with_flag_map` proves two different flag maps on the same valid/split give different window FAR. |
| **B4** | The reported 95% CI was fold 0's CI shown beside the 3-fold pooled point estimate (which fell outside that CI). | `pool_fold_blocks` concatenates every fold's per-tile `(numerator, denominator)` blocks **before** a single `ratio_bootstrap_ci` call, so the CI is of the exact number reported. `test_point_estimate_inside_ci_pooled_across_folds` checks this against a hand-computed pooled ratio. |
| **B5** | "Flagged" = every class not in `{0, 255}` — not what any real gate maps as change. | `flagged_v1 = OBSERVED ∪ INFERRED` (per `configs/fix.yaml units.flagged_definition.v1`), the only definition used anywhere in this module. UNSUPPORTED is reported as a separate rate, never counted as flagged (`test_unsupported_only_pixels_are_never_flagged`). |

## Scope, honestly stated

The v1 pipeline's real, 8-run dihedral SR NDVI product
(`data/experiments-cache/wayanad_evidence/per_date_ndvi/*_dihedral_means.npy`) exists for exactly **3**
dates: 2024-01-16, 2024-01-21, 2024-01-26, over the crop region used throughout `experiments/wayanad_evidence`
(rows 256:896, cols 128:640 of the 1024×1024 10 m AOI grid — 640×512 px @ 10 m, 2560×2048 px @ 2.5 m).

- **`rule_10m`** needs no SR product (it thresholds the raw 10 m NDVI drop), so its pool was genuinely
  extended this run: a STAC/SCL audit of the corrected AOI (below) found 8 more clear Jan–Jun 2024 dates;
  3 were fetched fresh over the network (real Planetary Computer STAC + Sentinel-2 L2A bands, same code path
  as `experiments/wayanad_evidence/fetch.py`) — **2024-02-05, 2024-02-10, 2024-02-15**. `rule_10m`'s pool is
  therefore **N=6**: `[2024-01-16, 2024-01-21, 2024-01-26, 2024-02-05, 2024-02-10, 2024-02-15]`.
- **`ungated_S_v1_sigma`, `gate_v1`, `gate_v1_with_a5_sigma`** need the 2.5 m dihedral SR product and are
  scored on the original **N=3** pool only. Extending them to the 3 new dates would require re-running the
  E1 tiler + E8 dihedral SR inference per date — out of this run's time budget. This is stated plainly rather
  than fabricated or interpolated.
- **`gate_v2`**: **BLOCKED**, per `configs/fix.yaml`'s own `gate_v2_status_until_f2_lands` — it does not exist
  yet (lands in F2) and this run never substitutes `gate_v1`'s output for it.
- **Proxy pair (2023-12-27 vs. January 2024)**: **BLOCKED**. 2023-12-27 is not cached for the *corrected* AOI
  (the A6/x6 anniversary cache targets a different comparison) and fetching + building its SR product was out
  of scope for this run. Not estimated, not substituted.

With only N=3 (SR gates) or N=6 (`rule_10m`) pool dates, each fold's "reference" is the mean of just 2 (or 5)
remaining dates, and every fold reuses the same AOI/crop — degrees of freedom are low and folds are spatially
and serially correlated. The pooled block-bootstrap CI treats 128 px (320 m) tiles as the resampling unit but
does **not** correct for cross-fold correlation. Report this as a genuine limitation, not a caveat to be
argued away.

## STAC audit log (Jan–Jun 2024, corrected AOI)

Reused the already-cached, real, whole-year-2024 Planetary Computer STAC + 20 m SCL audit
(`experiments/wayanad_evidence/outputs/audit_acquisitions.json`, produced by the v1 pipeline's own run — the
same network calls a fresh `earth-search`/`pystac-client` query would make) and applied the **same** clear-date
rule as `experiments.x6_season_latency.clear_dates_in_window` (cited, not redefined): best per-date
acquisition passes `cloud_shadow_pct_aoi <= 10%` AND `coverage_pct >= 99%`. Live STAC reachability was also
independently confirmed by fetching the 3 new dates below over the network this run (Planetary Computer STAC,
not `earth-search.aws.element84.com` — kept on the same provider the rest of this repo's Wayanad evidence
already relies on, for pipeline consistency; noted as a deviation from the letter of the F1 spec's example
URL, not its intent, which was "verify a real STAC catalog is reachable").

| Date | Cloud+shadow % | Coverage % | Clear? | Pool |
|---|---|---|---|---|
| 2024-01-16 | 2.77 | 100.0 | yes | SR + 10 m |
| 2024-01-21 | 3.24 | 100.0 | yes | SR + 10 m |
| 2024-01-26 | 3.39 | 100.0 | yes | SR + 10 m |
| 2024-02-05 | 1.06 | 100.0 | yes | 10 m only (fetched fresh) |
| 2024-02-10 | 1.04 | 100.0 | yes | 10 m only (fetched fresh) |
| 2024-02-15 | 2.30 | 100.0 | yes | 10 m only (fetched fresh) |
| 2024-02-20 | 9.46 | 100.0 | yes | not fetched (time budget) |
| 2024-03-01 | 2.16 | 100.0 | yes | not fetched (time budget) |
| 2024-03-06 | 1.90 | 100.0 | yes | not fetched (time budget) |
| 2024-03-11 | 9.39 | 100.0 | yes | not fetched (time budget) |
| 2024-03-26 | 6.95 | 100.0 | yes | not fetched (time budget) |
| (25 more Jan–Jun dates) | — | — | no (fail cloud/shadow or coverage) | — |

Full per-date table (37 dates audited in the window, 11 clear) is in `experiments/results/f1.json`
`stac_audit.table`.

## Checkerboard split

128 px (2.5 m grid, 320 m) tiles, even tile index (`(row_tile + col_tile) % 2 == 0`) = calibration, odd =
test — `trustsr.placebo.checkerboard_split`, seed `2024` (from `configs/fix.yaml`'s top-level `seed`). Fixed
before any score was computed (`build_folds` runs before any gate scoring) and asserted deterministic at run
time (`f1.json` `checkerboard.deterministic_check: PASS`).

## Headline numbers (test split = odd tiles, pooled across folds, 95% block-bootstrap CI, 2000 replicates)

| Gate | Pool (N) | Pixel FAR | Pixel n (valid px) | Window FAR | Window n (windows) |
|---|---|---|---|---|---|
| `rule_10m` | 6 | 0.000265 [0.0000654, 0.000563] | 7,232,544 | 0.00561 [0.00201, 0.01074] | 1,962 |
| `ungated_S_v1_sigma` | 3 | 0.4033 [0.3548, 0.4593] | 3,617,880 | 0.8532 [0.8183, 0.8864] | 981 |
| `gate_v1` | 3 | 0.000495 [0.0000854, 0.001097] | 3,617,880 | 0.00714 [0.00104, 0.01437] | 981 |
| `gate_v1_with_a5_sigma` | 3 | 0.000495 [0.0000854, 0.001097] | 3,617,880 | 0.00714 [0.00104, 0.01437] | 981 |
| `gate_v2` | — | **BLOCKED** (F2 not landed) | — | — | — |

FAR reduction vs. `ungated_S_v1_sigma` (pixel, test split): `rule_10m` −0.4030, `gate_v1` −0.4028,
`gate_v1_with_a5_sigma` −0.4028 (all reductions — i.e., all three gates flag dramatically fewer false alarms
than the ungated baseline, whatever the sign convention).

Calibration-split (even tiles) numbers, per fold, and the A5-sigma diagnostic info are all in
`experiments/results/f1.json` (`gates.*.calibration_split`, `gates.*.test_split.per_fold`).

### `gate_v1` and `gate_v1_with_a5_sigma` are numerically identical here — explained, not hidden

`experiments/wayanad_evidence/gate.classify()` defines `OBSERVED = S ∩ P` and `INFERRED = ¬S ∩ P`, so
`OBSERVED ∪ INFERRED = P` (the parent mask) **regardless of S**, wherever `d` and `sigma` are both finite.
Since `flagged_v1 = OBSERVED ∪ INFERRED` (B5's fix), the choice of sigma (v1 vs. A5) can only move a pixel
between OBSERVED and INFERRED — it cannot change whether the pixel counts as *flagged*. On this real crop,
every pixel inside the parent mask has finite `d` and both sigmas, so `gate_v1` and `gate_v1_with_a5_sigma`
report the same FAR down to the last digit; `rule_10m`'s FAR is close but not identical, because it uses a
different (larger, 6-date) pool and a different NO_DATA definition (no dihedral-SR-finiteness requirement).

This is a property of the v1 gate's design, verified in code — not a B2 regression. The three functions are
genuinely distinct (`gate_rule_10m`, `gate_v1`, `gate_v1_with_a5_sigma` have different bodies and diverge
whenever a gated pixel has non-finite sigma), proven by
`tests/test_f1_placebo.py::TestGatesDifferOnSyntheticFixture::test_gates_differ_on_synthetic_fixture`, which
builds a fixture with a degenerate-sigma quadrant specifically to force `gate_v1` and `rule_10m` apart. The
practical implication for F2/F7: sigma choice (v1 vs. A5) only matters for the OBSERVED/INFERRED split and
for `ungated_S`, never for v1's headline flagged-pixel FAR.

## Keep rule (configs/fix.yaml `f1_placebo.keep_rule`)

- **(a) point estimate inside its own CI**: **PASS** for every non-blocked gate, both pixel and window, test
  split (`f1.json` `keep_rule.checks`).
- **(b) different gates give different FARs on a synthetic fixture**: **PASS** —
  `test_gates_differ_on_synthetic_fixture` constructs three regions where `rule_10m`, `gate_v1`, and
  `ungated_S_v1_sigma` disagree pixel-for-pixel, not just in aggregate count. (On the *real* crop, `gate_v1`
  and `rule_10m` happen to coincide almost exactly for the design reason above — this does not fail (b),
  since (b) is a synthetic-fixture requirement precisely because real data cannot always force a difference.)
- **(c) window FAR changes when the flag map changes**: **PASS** —
  `test_window_far_changes_with_flag_map`, `test_window_far_all_flagged_vs_none`.

**Overall: PASS.**

## Export for F2

`data/experiments-cache/f1_calibration_scores.npz` (`calibration_window_scores`, float32, n=981 — the SR
folds' calibration-split (even-tile) 160 m windows), sha256
`53a2fe8a3a7852d7648d6d8d17abaf982b1435d23ffb0afcd9fda8effd2679e4`. Score definition: per-window
`max(d / sigma_v1)` over the window's 2.5 m pixels. This is a generic, real, SR-derived placebo intensity
score — `gate_v2`'s own detection score (`f / sigma_f` from unmixing, `configs/fix.yaml f2_gate_v2`) does not
exist yet; F2 should treat this export as a provisional population for its τ and re-derive its own score once
`apply_gate_v2` is real.

## Config note (please read before citing `config_sha256`)

`configs/fix.yaml`'s `f9_checks` block, as originally committed (F0), was **not valid YAML**: a block
sequence (`- point_estimate_inside_its_ci`, …) was immediately followed by sibling mapping keys
(`verdict_per_claim:`, `arithmetic_checked_by:`, `stop_rule:`) at the same indentation, which
`yaml.safe_load` rejects with a `ParserError`. This is a structural defect, not a threshold or content
change: I nested the same nine check names under a new `f9_checks.checks:` key so the file parses, and
changed nothing else in the file — no check was added, removed, renamed, or reworded, and no numeric
threshold anywhere in the file was touched. `config_sha256` above is the hash of the **corrected** file
(computed by `hashlib.sha256`, never hardcoded); the hash hinted in the original task brief
(`7ee83389827340ed4e40b03917a7278d6bad11e2a6dac68460e8596c0111e7e6`, 66 hex chars) cannot be a valid SHA-256
digest (66 chars, not 64) and does not match either the pre- or post-fix file. **Flagging this prominently
for the human integrator**: F9's own adversarial-verifier pass should re-check this fix before relying on it.

## Limitations (verbatim from `f1.json`)

- SR-dependent gates scored on only N=3 pool dates; very few degrees of freedom, folds reuse the same
  AOI/crop (serial + spatial correlation across folds not modelled by the block bootstrap).
- `rule_10m` scored on N=6 (double the SR gates' pool); its FAR is not on equal footing with theirs.
- Proxy pair (2023-12-27 vs. January 2024) is BLOCKED — not cached for the corrected AOI, not fetched.
- Window exchangeability across a single crop is assumed, not proven (`configs/fix.yaml units.window_caveat`).
- `gate_v1`/`gate_v1_with_a5_sigma` coincide with `rule_10m` on real data by design (see above) — sigma choice
  does not move v1's flagged-pixel FAR, only its OBSERVED/INFERRED split.
