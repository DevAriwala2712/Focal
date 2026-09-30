# ADR-X5: predictive noise model for the trust gate (agent A5)

Status: decision rule and failure modes fixed BEFORE any Wayanad coverage number was computed (git commit that adds this
file precedes the commit that adds `experiments/results/x5.json`). The section "Outcome" at the bottom is appended after the run.

## Problem

Gate v1 sets `sigma = sqrt(sigma_pre^2 + sigma_post^2)`. `sigma_pre` is the std of all pre-date x dihedral-run samples
(3 x 8 = 24, Welford, ddof 1); `sigma_post` is the std of the 8 dihedral runs of the single post date. Measured on the cached
Wayanad state: median sigma_pre 0.0326, median sigma_post 0.0022.

What the gate needs is the spread of ONE NEW observation about the pre mean: `d = new - mean_pre`. Two things are missing in v1.

1. The pooled sample std weights the between-date spread as if it were 24 independent draws: with `n_pre` date means `m_t` and
   `R` runs per date, `sigma_pre^2 ~ R(n-1)/(nR-1) * s2_dates` (`s2_dates` = variance of the date means, ddof 1). For `n = 3,
   R = 8` this is `16/23 = 0.70 s2_dates`.
2. The predictive variance of `d` is `s2_dates * (1 + 1/n)` (one new date effect plus the error of the pre mean): 1.33 s2_dates.
   v1 has no "+1" and no "+1/n"; `sigma_post` is dihedral only (0.002) and carries no temporal term.

So, analytically and independent of the data, `sigma_v1 / sigma_predictive = sqrt(0.696/1.333) = 0.72`: v1 is ~28 % below
the predictive SD, equivalently the predictive SD is ~38 % above v1 (`n_pre = 3`; for the leave-one-date-out check with 2
remaining dates the factors are 0.533 s2 vs 1.5 s2: v1 is 40 % low, predictive 68 % above). Note the mechanism: pooling does
contain the between-date spread (that is why sigma_pre is 15x sigma_post). The brief's statement "24 near-identical samples"
is therefore not the mechanism; the deficit is the wrong weight/dof and the missing predictive inflation. The claim is
tested numerically (synthetic) and against real held-out dates (script) rather than assumed.

## Model

    sigma^2 = s2_dates * (1 + 1/n_pre) + s2_dihedral_pre / n_pre + s2_dihedral_post          (configs/exceptional.yaml a5.formula)

* `s2_dates`: variance ACROSS dates of the per-date mean NDVI (each the mean over that date's 8 dihedral runs). NDVI is
  computed per run first, then averaged. Never pooled runs.
* The post temporal variance is borrowed from the pre dates (post is a single date, so it has no variance of its own): the
  "1" in `(1 + 1/n_pre)`. Exchangeability of dates is assumed (not proven; see failure modes).
* `s2_dihedral_pre` = mean over pre dates of the within-date dihedral variance; `s2_dihedral_post` likewise for the post date.
  The date-mean variance already contains `delta^2/R` of dihedral noise, so the two dihedral terms are conservative by
  construction; the simulation quantifies it (about 0.4 % of variance at delta/tau = 0.07) and the real run reports the
  fraction of sigma^2 they contribute.
* Floor: `sigma^2 >= a5.noise.sigma_floor^2` (0.005 NDVI). The fraction of valid pixels floored is reported.

### Small-n estimator (n_pre = 3 gives 2 degrees of freedom per pixel)

A per-pixel `s2_dates` with 2 dof is chi-squared(2)/2 distributed: the resulting z = d/sigma has t(2) tails, covers 82 % at
k = 2 (simulated), and is useless pixel by pixel. Candidates (all in `trustsr/noise.py`, all computed on the same data):

| id | estimator | idea |
|---|---|---|
| raw | s2 from the dates | unbiased, 2 dof; the literal formula |
| window | mean of raw s2 in a 90 m window | pools dof spatially; biased at variance edges |
| eb_stratified | empirical Bayes toward the mean s2 of its NDVI stratum (20 equal-count strata of the pre-mean NDVI) | pools dof across land-cover/NDVI-alike pixels |
| eb_window | empirical Bayes toward the local 90 m window mean | local prior, keeps pixel-specific evidence |

Empirical Bayes: `s2 ~ sigma^2 chi2_dof/dof`, `sigma^2 ~ scaled-inv-chi2(nu0, s0^2)`. The prior degrees of freedom `nu0` come
from the log-variance moment estimator `var(log s2/prior) = trigamma(dof/2) + trigamma(nu0/2)` (Smyth-2004 style, verified by
simulation: true 10, recovered 10.2). The returned value is the POSTERIOR MEAN of sigma^2,
`((nu0-2) prior + dof s2) / (nu0-2+dof)`, because the predictive distribution of `d` is then a scaled t whose variance is exactly
that; using the posterior scale instead would be biased low by `2/(nu0+dof)` (16 % at nu0 = 10). It is unbiased for the
variance (not for the SD: Jensen).

## Decision rule (fixed before any Wayanad coverage was computed)

The estimator is chosen on data that contain NO Wayanad pixel: the E6 12-date cache (old AOI, 10 m B04/B08, 12 clear pre
dates), by repeated seeded subsampling of 3 pool dates + 1 held-out date (config `a5.selection`, 100 replicates).
Criterion: mean held-out Gaussian NLL of `d` under N(0, sigma^2) with sigma floored. Winner = lowest mean NLL; if the paired
per-replicate difference vs the runner-up has a 95 % CI containing 0, take the simpler (order raw < window < eb_stratified <
eb_window). The dihedral terms are zero in that stage (no SR runs exist for E6): it validates the `s2_dates` shrinkage only.
The winner becomes the headline estimator for Wayanad; all four are still reported there, none is dropped.

Wayanad headline quantity (config `a5.evaluation`): pooled leave-one-date-out coverage `P(|d| <= 2 sigma)` on real stable pixels,
`d = held-out date - mean(other pre dates)`, sigma from the other two dates (`n_pre = 2`, 1 dof) and the held-out date's
dihedral variance. Keep rule (pre-registered, not changed): headline coverage within +/- 2 pp of 95.45 %, OR the gap is
reported with its cause. `k_eff` and old-sigma coverage are reported next to it; `k_eff` is cross-fitted by 128 px checkerboard
tiles (fit on even tiles, evaluate on odd, and the reverse). No threshold is tuned on the Wayanad coverage.

Secondary (labelled): pre-mean vs December (`n_pre = 3`, CONFOUNDED by season: December NDVI is higher on stable pixels, so
this measures bias more than noise calibration); a stable-pixel empirical sigma (per-stratum SD of d, spatially cross-fitted)
as a comparator; validation of the n_pre = 3 estimators on E6 held-out dates.

## Failure modes to check for (declared in advance)

1. Heavy tails: registration error at edges, thin cloud, haze, changed crops give |d| far beyond Gaussian; coverage at k = 2
   stays below 95.45 % however good the variance is. Then `k_eff > 2` and the miss is reported with this cause.
2. Coherent date effects: an atmospheric offset moves a whole date, so pixel counts overstate independent evidence. There are
   3 pre dates (3 folds that share dates, not independent); block-bootstrap CIs cover space, NOT dates, and understate the
   uncertainty across dates.
3. Season: the December comparison mixes a systematic NDVI offset into `d`.
4. n_pre = 2 in the leave-one-date-out check (1 dof) is a harsher test than the n_pre = 3 deployment; the E6 subsampling checks
   n_pre = 3 but has no dihedral term and a different resolution and season set.
5. NDVI strata do not capture spatial heterogeneity at edges (forest/clearing): shrinkage toward the wrong stratum inflates or
   deflates sigma there.
6. The floor may dominate in flat dark/dense pixels; reported as a fraction.
7. Selection on E6 may pick an estimator that is not best on Wayanad. That is accepted: the rule was fixed first.
8. `d` on stable pixels excludes the slide footprint and pixels with a 10 m parent drop > 0.30 (a mild truncation of large
   positive `pre - Dec` drops, absent in the leave-one-date-out `d`).

## What this does not show

Nothing here says that any flagged pixel is a real landslide; it calibrates the noise scale on stable pixels of one AOI and
three January dates. It does not test power (sensitivity) and does not calibrate against labelled change.

## Outcome (appended after the run)

(see `experiments/results/x5_REPORT.md` and `x5.json`)
