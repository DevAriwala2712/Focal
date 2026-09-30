# Build prompt: TrustSR v2 (area-preserving sub-pixel change mapping with conformal false-alarm control)

Paste everything below the line into a fresh Claude Code session started in a git worktree cut from `experiments/exceptional` (or its successor). It is self-contained. The design and its evidence are in `docs/v2_design.md` and `docs/prior_art_teardown.md`; read them first.

---

You are building **TrustSR v2**, the change-mapping core for a Smart India Hackathon grand-finale entry (NTRO problem: deep-learning super-resolution of 10 m Sentinel-2 to <4 m, with validation against high-resolution references and explicit uncertainty). The demo case is the Wayanad landslide of 2024-07-30.

## 0. Non-negotiables (from AGENTS.md; do not relax)

- Only the four 10 m bands B02, B03, B04, B08 enter any model or index. SCL is a categorical validity mask only. No DEM, no 20 m or 60 m bands.
- Pinned pretrained SEN2SR-lite RGBN x4 is the SR model; fine-tuning only, no training from scratch. Every GPU step must run on 4 GB with tiling and automatic tile-size halving on CUDA OOM (down to the model's supported minimum, then fail with a clear message). v2 itself adds **no GPU work**.
- Georeferencing exact: every raster output is a Cloud-Optimized GeoTIFF in the input CRS; the 2.5 m grid is the exact x4 subdivision of the 10 m grid (each 10 m block = 4x4 sub-pixels, same bounds).
- **Never fabricate data or numbers.** If imagery, a dataset or a download is unavailable, stop that step and report what you checked (URL, error, retries). Downloads retry 3 times with backoff, then fail naming the URL. Respect `fetch.max_fetch_bytes` in `configs/wayanad_evidence.yaml` and keep a per-process byte log.
- All parameters in `configs/*.yaml`; no hard-coded paths or thresholds. Python 3.10+, pinned versions, CPU tests under 2 minutes total.
- Every result JSON carries `config_sha256`, `evidence: real|synthetic`, the git commit, and the labels: model = pretrained SEN2SR-lite (NOT fine-tuned unless A7 shipped), k / thresholds calibrated or not, retrospective comparison, the pre/post dates and the day gap.
- Never edit `AGENTS.md`, `RISK_REPORT.md`, the pre-registered keep rules in `configs/exceptional.yaml`, or `docs/sources.md`. Never loosen a keep rule after seeing a result. A failed keep rule is a result: report it.

## 1. What already exists (inspect, do not assume)

- `experiments/wayanad_evidence/` (v1 gate, trust statistics, EVIDENCE.md with verified numbers) and `experiments/e1..e8`, `experiments/common.py`.
- `configs/exceptional.yaml`: pre-registered programme. Sections a2 (HR benchmark), a3 (placebo harness), a4 (gate v2), a5 (noise model), a6 (season and latency), a7 (fine-tune), a8 (HR-referenced evaluation), statistics (block bootstrap), units (far_pixel, far_window). Your work implements **a4** and consumes **a3** and **a8**. If `trustsr/bootstrap.py` exists (block and paired bootstrap CIs, tested) use it for every interval; do not write another bootstrap.
- Cached data (gitignored, `data/experiments-cache/wayanad_evidence/`, main checkout) include the aligned stacks and 8-run SR statistics for the corrected AOI (11.49 N, 76.16 E, 10,240 m square, EPSG:32643). If any cache you need is missing, say BLOCKED and name it; do not regenerate silently.
- Verified v1 diagnosis you are fixing: OBSERVED union INFERRED equals the 10 m parent mask; INFERRED 515 of 117,584 parent 2.5 m px (0.44 %); sigma_pre median 0.0326 vs sigma_post 0.0022; UNSUPPORTED 138,808 px, 98.5 % with a positive 10 m drop; 40.7 % of the main scar's 10 m pixels are boundary pixels; Jan 2024 pre dates vs post 2024-12-06 (129 days after the event).

## 2. The method to implement (full math in docs/v2_design.md)

Per 10 m block i, with 4x4 sub-pixels B_i:

1. **Fractions (10 m, no SR).** Two-endmember linear unmixing in reflectance: a_hat = clip(<y - e_b, d>/||d||^2, 0, 1), d = e_v - e_b, using (B04, B08) as the primary spec. f_hat_i = mean over pre dates of a_hat minus a_hat(post). Misfit residual r_i (NO_DATA reason bit). Endmembers fixed by a pre-registered percentile rule on the **pre-event** stack only (e_v = mean of pixels at or above the 95th NDVI percentile; e_b = mean of pixels at or below the 2nd NDVI percentile, excluding SCL water/shadow), applied identically to placebo and test pairs.
2. **SR change score (2.5 m).** Same projection on the 8-run mean SR reflectance per date; Delta_x = mean_pre a_SR - a_SR(post); Delta'_x = Delta_x minus its block mean. Used only for ranking inside a block.
3. **Detection.** Window W = 16x16 blocks. T(W) = max over valid blocks of f_hat. tau_win = the k-th smallest placebo window statistic, k = ceil((n+1)(1-alpha)), alpha = 0.05 (else +inf). Flag W iff T(W) > tau_win. Block detected iff its window is flagged and f_hat_i > tau_blk, tau_blk the (1 - alpha_blk) quantile of placebo block f_hat (alpha_blk in config).
4. **Exact-count allocation.** f_tilde = clip(f_hat - b_stratum, 0, 1). Per 8-connected component K of detected blocks: N_K = round(16 sum f_tilde), largest-remainder apportionment (ties by raster order) gives n_i. Rank sub-pixels inside a block by u_x = (1 - lam) r_x + lam s_x, r_x the within-block rank of Delta'_x, s_x the within-block rank of the bilinear interpolation of f_tilde; select the top n_i (ties by raster order). Deterministic, no random numbers.
5. **Variant B (PSF-aware)** as in design section 3.4b; implement after variant A is tested; not the default.
6. **Classes** (uint8 COG, nodata 255): 0 NO_CHANGE, 1 CORE (detected block with n_i = 16), 2 ALLOCATED (chosen sub-pixels of detected blocks with 0 < n_i < 16), 3 UNSUPPORTED (sub-pixels in non-detected blocks with Delta_x above a placebo-calibrated kappa; counted and reported, never mapped), 255 NO_DATA (reason bits in a companion COG). Also write the 10 m f_tilde raster as a first-class product (float32 COG) and the per-window flag raster.

Baselines you must implement for comparison (same masks, same grid): `blocky_10m` (hard-threshold 10 m map replicated to 2.5 m), `hard_threshold_v1_style` (v1 rule), `alloc_bilinear` (lam = 1), `alloc_pretrained_sr` (lam = 0 and lam tuned), `alloc_finetuned_sr` only if A7 shipped, `ungated_S_v1_sigma`, `gate_v1`, `gate_v1_with_A5_sigma`, and interpolation baselines nearest/bicubic for SR fidelity.

## 3. Repository additions

```
trustsr/v2/__init__.py
trustsr/v2/unmix.py        fractions, misfit, endmember rule
trustsr/v2/conformal.py    window statistic, split-conformal threshold, Beta interval, BH option, Mondrian strata
trustsr/v2/allocate.py     largest remainder, ranking, variant A, variant B
trustsr/v2/classes.py      class raster, NO_DATA reason bits, UNSUPPORTED
trustsr/v2/psf.py          Gaussian PSF operator H from an MTF assumption
scripts/v2_calibrate.py    placebo calibration (consumes the a3 harness)
scripts/v2_run.py          apply to a pair -> COGs + JSON
scripts/v2_evaluate_hr.py  a8 evaluation
configs/v2.yaml            ALL parameters and the keep rules below, committed BEFORE any run
tests/test_v2_*.py         CPU-only, tiny synthetic rasters
docs/v2_results.md         results with denominators, labels, keep-rule verdicts
```

Commit after each component with its tests passing. Commit `configs/v2.yaml` (including its `keep_rules` block and parameter ranges) **before** running anything it governs, and put its sha256 in every result.

## 4. Tests (write first; all CPU; total under 2 minutes)

- **T1 count consistency (blocking).** For random f_tilde on a synthetic 64x64-block grid, every detected block has exactly n_i allocated sub-pixels with n_i in {floor(16 f), ceil(16 f)}; per connected component sum n_i = round(16 sum f_tilde); component area error <= 3.125 m^2 (0.5 sub-pixel) as a float assertion. A failure **stops** later waves.
- **T2 SR independence (blocking).** Replace the SR score by three unrelated random fields and by its negation: per-block counts and the total mapped area are identical; only positions change.
- **T3 georef.** Output bounds equal input bounds; pixel size exactly 1/4; each 10 m block maps to exactly its 4x4 sub-pixels; CRS unchanged; COG validity check.
- **T4 unmixing.** Synthetic scene with known areal fractions (checkerboard sub-pixel truth, box PSF) plus Gaussian noise recovers f within 3 standard errors on 99 % of blocks; a nonlinear-NDVI fraction estimator is shown to be biased on the same scene (the test asserts the bias exceeds a configured bound).
- **T5 conformal validity.** Monte Carlo, exchangeable synthetic window statistics (n_cal = 1000, alpha = 0.05, 500 repetitions): mean realised FAR within 0.05 +- 0.005, and the empirical distribution of realised coverage (1 - FAR) given the calibration set matches Beta(n+1-l, l), l = floor((n+1) alpha), i.e. realised FAR ~ Beta(l, n+1-l) (Kolmogorov-Smirnov p > 0.01 or a fixed tolerance in config). A second test with a deliberate mean shift in the test statistic asserts FAR exceeds alpha (the guarantee fails when exchangeability fails).
- **T6 hysteresis.** Detection outside flagged windows never occurs; changing tau_blk cannot change the window-level flag set.
- **T7 NO_DATA.** Masked pixels in any date are NO_DATA in every output; NO_DATA overrides all classes; reason bits set.
- **T8 determinism.** Two runs produce byte-identical arrays and JSON (except timestamps, which are excluded from the hash).
- **T9 PSF study.** Reproduce `docs/evidence/v2_synthetic_checks.py` numbers within 1e-6 (the fixed synthetic table in the design doc).

## 5. Experiments, in order (stop and report at any BLOCKED)

**X1 synthetic PSF study.** 2-D synthetic scenes with random polygonal change, PSF Gaussian with MTF(Nyquist) in {0.15, 0.20, 0.30}, noise at three levels, 200 scenes each. Compare variant A vs B: per-block area RMSE and IoU vs truth. Evidence: synthetic.

**X2 placebo calibration (a3 harness).** Placebo pairs: (a) leave-one-out among clear Jan-May 2024 dates; (b) season-gap-matched pairs from 2023 (Jan 2023 dates vs a Nov/Dec 2023 date) fetched in accordance with A6 exp. a; if 2023 imagery cannot be fetched under the byte cap, mark (b) BLOCKED and state that the event pair then carries no FAR claim. Exclude the published-slide 1500 m disk and the footprint. Checkerboard split by 128 px tile (even = calibration, odd = test). Calibrate tau_win and tau_blk on the calibration tiles only; score test tiles. Report at pixel, block and **window** level (units in exceptional.yaml), plus tile-level sensitivity. Report the effective number of independent dates (pairs share dates).

**X3 HR-referenced evaluation (a8).** On HR sets available through opensr-test / SEN2NAIPv2 (whatever is accessible; name exactly what failed), construct synthetic pre/post pairs by applying known vegetation-to-bare masks at HR and degrading, so truth is exact. Truth rule from a8: HR NDVI < 0.35 non-vegetation, sensitivity at 0.25 and 0.45. Methods and metrics per a8; fractions both `oracle_from_hr` and `realistic_from_s2`; strata by feature width (0-20, 20-50, 50-150, 150+ m).

**X4 Wayanad application.** Apply the calibrated detector to the Jan-2024 vs 2024-12-06 pair (and the A6 composite if fetched). No accuracy claim (no labels); report class areas, counts, UNSUPPORTED stratified by 10 m fraction, NO_DATA by reason, block-sum consistency, and the 129-day gap on every figure. A9 external reference only if a clearly licensed polygon exists.

**X5 optional transfer.** Sen12Landslides or GLaD4CD only if downloadable within the byte cap and licences checked; four bands only. Record exactly what was listed vs downloaded.

## 6. Pre-registered keep rules (copy verbatim into `configs/v2.yaml: keep_rules`; do not edit after any run)

- **K1 count consistency.** KEEP iff T1 and T2 pass and, on every real run, 100 % of detected blocks satisfy n_i in {floor(16 f), ceil(16 f)} and every component satisfies the component identity. Any violation STOPS wave 3.
- **K2 conformal machinery.** KEEP iff T5 passes.
- **K3 conformal on real placebo (a4 rule).** KEEP "window-level false-alarm rate is controlled at alpha = 0.05" iff the **test-split** window-level FAR point estimate for gate v2 is <= 0.05, with the 95 % block-bootstrap CI reported. If the point estimate is <= 0.05 but the CI upper bound exceeds 0.05, label the claim "controlled in point estimate, not in interval". Report the same for gate_v1, gate_v1_with_A5_sigma, ungated_S_v1_sigma and rule_10m on the same windows; the headline is the FAR reduction of gate v2 vs ungated SR with CI and denominators.
- **K4 season effect.** Report FAR of a family-(a)-calibrated threshold evaluated on family-(b) test windows. If FAR > 0.05, the verdict is "season effect measured; (a)-calibrated threshold is not valid for a Jan-to-Dec pair" and the event pair uses (b) only. If (b) is BLOCKED, the event-pair output is labelled "no false-alarm guarantee (placebo not season-matched)".
- **K5 SR adds information (a8).** Paired IoU(alloc_pretrained_sr) minus IoU(alloc_bilinear), image-paired block bootstrap 95 % CI, reported whatever its sign. KEEP "SR ranking adds within-block information" iff the CI lower bound is > 0 on the pooled evaluation and > 0 on at least half of the feature-width strata; otherwise the claim is not made and the shipped default is lam = 1.
- **K6 allocation vs blocky.** KEEP "allocation improves IoU over blocky_10m" iff the paired CI lower bound is > 0 under both oracle and realistic fractions. Otherwise claim only what K7 supports.
- **K7 area.** KEEP "allocation reduces absolute area error vs hard_threshold_v1_style" iff the paired CI lower bound of (blocky error minus allocation error) is > 0.
- **K8 PSF-aware variant.** Variant B replaces A as default iff (i) on X1 its per-block area RMSE is lower than A's with paired CI excluding 0 at all three MTF values and (ii) on X3 its IoU difference (B minus A) has CI lower bound >= -0.01. Otherwise A ships and B is reported.
- **K9 fractions vs NDVI.** KEEP "reflectance unmixing beats NDVI-rescaled fractions" iff per-block fraction RMSE is lower with paired CI excluding 0 on X3 realistic fractions.
- **K10 determinism.** KEEP iff T8 passes and two full X4 runs give identical hashes.
- **K11 Wayanad.** No F1, precision or recall is reported without a labelled reference. Class areas, counts and dates only, each with its denominator.
- **K12 no unverified claim in outputs.** Every "X beats Y" in `docs/v2_results.md` cites the result JSON and its CI; otherwise it is written as untested.

## 7. Deliverables and reporting

1. Code, tests and `configs/v2.yaml`, one commit per component (tests passing), message ends with the co-author line supplied by the session.
2. `experiments/results/v2_*.json` (with config hash, evidence label, commit) and `docs/v2_results.md` (a table of every keep rule with PASS / FAIL / BLOCKED, numbers, denominators, CIs).
3. Update the README's "How to reproduce the Wayanad demo" only for steps you actually ran.
4. Final message: status per experiment (PASS / FAIL / BLOCKED), each keep-rule verdict, what could not be run and why, and the list of assumptions from `docs/v2_design.md` section 10 that were tested vs still untested. Do not claim generalisation beyond the datasets actually used.

## 8. Things to watch (decisions already made; do not silently change)

- The placebo must run the **identical** pipeline (endmember rule, masks, dilation, unmixing). A variant that reads post-event pixels for endmembers is reported as non-calibratable.
- Do not tune lam, tau_blk, n_min, dilation or strata on the test split.
- v1's NDVI-per-run then mean convention stays for SR NDVI diagnostics; v2's decision statistic is the reflectance-space fraction.
- The output is a **vegetation-loss** product. Do not label it landslide damage, buildings or roads.
- If the 4 GB check matters for a step you run, cap the PyTorch allocator at 4 GiB and say that this approximates but does not verify a 4 GB card.
