# TrustSR — Adversarial Research Memo

Oct 3, 2026 · @Boss

## Summary

Before writing significant new code, run four cheap analyses on artefacts that already exist, and use them to decide whether super-resolution stays in the evidence path at all. The repository's results already point away from SR as the source of analytical value; three findings below change what should be tested next.

**1. The gate keep rule cannot tell a working gate from a broken one.** I computed the operating characteristic of F2's rule (test-split window FAR ≤ α, with τ set by split conformal at α = 0.05, n = 981 calibration windows). A *perfectly* calibrated, perfectly exchangeable gate with independent windows exceeds 0.05 on the test split with probability **0.46**, and reaches 0.0652 or higher with probability **0.072**. With spatially correlated windows (design effect 2 to 4) that probability rises to **0.14–0.22**. F2's CI \[0.0478, 0.0850\] contains 0.05. F2's FAIL stands as pre-registered, but the interpretation "calibrated but does not generalise" is **not established**. F11's three-condition version of the same rule is likely to fail whatever the gate's quality.

**2. The sub-pixel bottleneck is the fraction estimate, and SR is nearly irrelevant to it.** From `f5.json`: with the oracle fraction, allocation ranked by plain **bilinear** interpolation beats the blocky 10 m map by about +0.069 IoU (0.829 vs 0.760). Swapping bilinear for SR ranking adds +0.0016 excluding NAIP (CI spans 0). With the realistic fraction, allocation falls to 0.446. Allocation can work; SR is not what makes it work; the fraction estimator is what breaks it.

**3. For Wayanad, spatial resolution is probably not the binding constraint.** The binding constraints are observation timing (first clear post date 129 days after the event, one post date) and season (December post greener than every January pre date, mean NDVI 0.81 vs 0.72). 97.8 % of SR change energy is already resolvable at 10 m. The placebo null used to calibrate the gate is within-season (January–March pairs), while deployment is cross-season (January pre vs December post). F11 does not touch that gap.

**Tools note.** I cloned the repository and read the result files. I could not reach the imagery or OpenSR-test caches from this sandbox (they live on the team machine; Hugging Face is not reachable here), so the only thing I executed is the arithmetic in finding 1. Every other number is `[observed]` from repository JSON, and every experiment outcome below is `[predicted]`.

## 1. Real problem

**Objective, without the architecture:** decide where, when and how much land surface changed, at a spatial grain finer than 10 m where the evidence supports it, with a stated false-alarm rate, and never let a model's reconstruction count as an observation.

The NTRO statement asks for something adjacent but different: a < 4 m product from 10 m Sentinel-2 that is validated against high-resolution references and carries explicit uncertainty. Change detection is one of four named applications, not the requirement. The memo keeps both in view, because a direction that solves the science but drops the requirement loses the finale.

| Category | Content | Source |
| --- | --- | --- |
| Directly observed | Four 10 m bands per clear date; SCL validity; acquisition dates; 6 clear pre dates (Jan–Feb 2024); one accepted post date, 2024-12-06 | R2, F11 pool |
| Directly observed | Block-mean reflectance of each 10 m pixel (the only spatial fact SR is forced to keep, via the Fourier hard constraint) | design, E1 |
| Inferred | Any 2.5 m pattern inside a 10 m pixel; any sub-pixel change fraction; any boundary position finer than 10 m | SR, unmixing |
| Inferred | That an NDVI drop is a landslide rather than season, regrowth or illumination | NDVI rule |
| Unavailable | 2.5 m change truth for any landslide; licensed Wayanad polygon; any clear optical post image within 90 days; GPU and physical 4 GB numbers | RESULTS\_V2 |
| Assumed | Placebo pairs are exchangeable with the deployment pair; windows are exchangeable across a checkerboard split; static land-cover boundaries proxy change boundaries; linear two-endmember mixing in red/NIR | fix.yaml caveats, F5 |
| Operational constraint | Pretrained SEN2SR-lite, fine-tune only; B02/B03/B04/B08 only; 4 GB target; exact grid; finale date unknown | AGENTS.md |
| Scientific constraint | Pre-registered keep rules; no tuning on test; BLOCKED is never PASS; synthetic proves mechanics only | AGENTS.md, fix.yaml |

**Success criterion (operational).** On held-out data with high-resolution truth, a method counts as solving the problem if it (a) beats the strongest simple baseline (blocky 10 m, bilinear-threshold) on IoU and boundary F1 for features under 50 m wide, with a 95 % CI excluding zero and NAIP excluded; (b) holds a false-alarm rate at or below a stated α under a null that matches the deployment pair (same seasonal offset, same latency); and (c) reports detection power at that α on injected or labelled changes. A method that meets (b) by flagging nothing fails (c).

## 2. Current failure diagnosis

Each failure has a mechanism that does not need "a better SR model" to explain it, and most mechanisms point upstream of SR.

| Failure `[observed]` | Likely mechanism `[inferred]` | Confidence | What would confirm it |
| --- | --- | --- | --- |
| SR loses to bicubic on PSNR: −0.097 dB pooled, −0.275 dB without NAIP; Venus −0.559 dB | The hard constraint pins every frequency the sensor measured. Everything above the 10 m Nyquist limit comes from a prior learned on NAIP (US aerial). Outside that domain the prior adds error, not signal. | High | Gain tracks domain similarity to NAIP: spain\_urban (regular geometry) is the only non-NAIP positive ranking stratum, +0.0066 \[+0.0025, +0.0129\] |
| SR ranking adds +0.0016 IoU over bilinear without NAIP | Inside a 10 m block, the only measured fact is the block mean. Bilinear already orders sub-pixels by the neighbouring block means; SR re-orders them by prior. The prior rarely knows where *this* boundary is. | High | SR-threshold ≈ bilinear-threshold with an identical detector (experiment A1) |
| SR more hallucination-prone than bicubic on 5/5 datasets; ungated SR flags 40.3 % of pixels on no-event placebo pairs | Prior-generated texture is not stable across dates: two clear images of unchanged ground get different invented detail, so their difference looks like change. Eight dihedral runs of one input share the prior's bias, so dihedral σ understates real variance (v1 coverage 54 %). | High | Per-pixel SR difference variance on placebo pairs exceeds bilinear's by a large factor at matched 10 m consistency |
| Allocation with the realistic fraction: 0.446 IoU vs blocky 0.760 (3/119 better) | Fraction MAE 0.40, r 0.29. The estimator uses per-image percentile endmembers; the target class is an NDVI threshold, not a material; NDVI mixes non-linearly; terrain shading in the Ghats moves both endmembers. Hard-thresholding the 10 m pixel is itself a well-tuned fraction estimator (it rounds at about 0.5), which is why blocky is a strong baseline. | High | A calibrated fraction (isotonic, leave-one-dataset-out) cuts MAE below \~0.15 and allocation beats blocky (experiment A2) |
| Change fraction unstable: ±1 robust SD on `e_b` changes flagged pixels about 8× | A change fraction is the difference of two noisy fractions, so errors add. `e_b` came from 332 pixels at NDVI ≤ 0.30: dark vegetation, not bare ground, so the mixing line is short. | High | Already measured (F2). No further test needed to accept it. |
| Neighbour term λ = 0 | The spatial-dependence prior adds nothing measurable at this scale with this score. | Medium | Already measured. Accept and drop. |
| Gate window FAR 0.0652 vs ≤ 0.05 | Mostly sampling noise plus a rule with about a 46 % false-fail rate (Summary, finding 1). Real, but separate, gaps: n\_pre mismatch (×1.06 on σ\_f), three correlated dates on one crop, and a within-season null applied to a cross-season deployment. | Medium-high | A block-bootstrap test of test-vs-calibration FAR difference (A0); a cross-season placebo (B1) |
| E5 cascade saves 0 of 1,120 passes on the crop | The crop is centred on the scar, so every tile has a firing parent. An honest null about the crop, not the method. | High | Not worth testing further now |

**Is spatial resolution the bottleneck?** For Wayanad, the evidence says no. The scar is found at 10 m within 6–8 m of published coordinates; 97.8 % of SR change energy is resolvable at 10 m; and the binding limits are a 129-day gap, one post date and a seasonal greening confound. Resolution may bind for features under about 20 m wide (runout channels, roads, stream banks), which is exactly where F5's only surviving positive stratum sits (+0.0077 excluding NAIP, absolute IoU 0.27).

**Is the objective poorly formulated?** In three places, yes:

- **The 2.5 m layer is subordinate by definition.** v1 classes make OBSERVED ∪ INFERRED equal the 10 m parent, so trust was operationalised as "agrees with the 10 m decision". A gate built that way can only delete; it can never add evidence the 10 m rule lacked.
- **False-alarm rate is scored without power.** The verbatim-τ path would have passed with FAR = 0 by flagging nothing. No result reports recall at the chosen α.
- **The flagship scene cannot be validated.** No 2.5 m change truth exists for Wayanad, and the problem statement scores validation against high-resolution references. A Wayanad-centred plan cannot produce the validated claim the statement asks for.

## 3. Hidden assumptions

Six of the nine assumptions below (A1 to A6) are already contradicted by repository evidence; each gets an inverse formulation rather than a tweak.

| Assumption | Evidence | Confidence it holds | If false, replace with |
| --- | --- | --- | --- |
| A1. Finer pixels improve the change decision | ρ = 0.978 energy at 10 m; F6, F5 | Low | Resolution is not the bottleneck; observation timing, season and truth are |
| A2. SR texture carries site-specific information | F5 ranking +0.0016 (no NAIP); ha worse 5/5 | Low | SR carries a domain prior; use it only where the prior matches (urban grids), or not at all |
| A3. A hard 2.5 m boundary is the right output | Fraction MAE 0.40; `e_b` ±1 SD → 8× flags | Low | A calibrated probability per sub-pixel, or an expected area with CI per 10 m pixel |
| A4. Two-endmember linear unmixing in red/NIR gives usable fractions | MAE 0.40, r 0.29; no bare-ground endmember on this AOI | Low | Fraction from a calibrated monotone map of the 10 m index, learned on HR-referenced scenes |
| A5. Within-season placebo pairs represent the deployment null | Deployment is Jan pre vs Dec post; NDVI 0.72 vs 0.81 | Low | Null pairs must copy the deployment's seasonal offset and latency |
| A6. A point FAR ≤ α on one test split is a meaningful keep rule | 46 % false-fail for a perfect gate | Low | Test FAR exchangeability (test − calibration with block bootstrap) plus power at α |
| A7. Static land-cover boundaries proxy change boundaries | Untested | Medium | Validate on bi-temporal HR change pairs, even a few |
| A8. The 4-band, optical-only input set is fixed | AGENTS.md | Constraint, not evidence | Another sensor supplies post-event timing (requires a human decision; section 12) |
| A9. One flagship event (Wayanad) can carry the argument | No truth for it | Low | Validate on labelled inventories; show Wayanad only as an illustration |

## 4. Solution hypotheses

Fourteen formulations; H1 is the current system, and twelve (H3 to H14) differ from it in information source, representation, inference or decision mechanism, not in backbone or threshold.

| # | Formulation | Category | Different from current? | Mechanism | Existing evidence |
| --- | --- | --- | --- | --- | --- |
| H1 | SR-ranked area-preserving allocation + conformal gate (TrustSR v2) | Current | — | 10 m fixes how much, SR picks where | F2 FAIL, F5 realistic −0.313 |
| H2 | v2 with a calibrated fraction estimator, SR ranking kept | Current, repaired | Partly | Fix the measured bottleneck, keep SR | None yet |
| H3 | **Interpolate-then-threshold**: bilinear/bicubic 10 m index, threshold at 2.5 m; no fraction, no SR | Simplification | Yes | Classic super-resolution mapping baseline; boundary placed by neighbour gradients | Not in F5's method list: a missing baseline |
| H4 | **Calibrated-fraction allocation, bilinear ranking, no SR** | Multi-scale | Yes | Monotone map from 10 m index to fraction, learned on HR scenes; allocate by bilinear | Oracle fraction + bilinear = 0.829 vs blocky 0.760 |
| H5 | **10 m probabilistic change, no sub-pixel output**: P(change) and expected changed area with CI per 10 m pixel | Removing SR / probabilistic | Yes | Report what is measured; area via fraction with uncertainty | `rule_10m` pixel FAR 0.000265 |
| H6 | **Per-pixel time-series break detection** with a multi-year harmonic seasonal model (BFAST/CCDC-style) | Temporal | Yes | Seasonality is modelled, not matched; the break date is evidence | Season confound (0.72 vs 0.81 NDVI) is unmodelled today |
| H7 | **Per-pixel first-valid-observation compositing**: use partially cloudy scenes pixel by pixel to shorten latency | Temporal | Yes | Latency is per pixel, not per scene; scene-level 90 % validity rule discards usable pixels | 2024-10-27 is 72.5 % valid, 99 days post |
| H8 | **SAR-first post-event detection** (Sentinel-1 backscatter change, coherence loss), optical for later delineation | Multi-modal | Yes | Radar sees through monsoon cloud; 12-day revisit | Published Wayanad study used S1; needs AGENTS.md decision |
| H9 | **Terrain-constrained localisation**: slope and drainage define where debris flow can be; change allocated along flow paths | Physics | Yes | Debris flows follow channels; a strong spatial prior that is physical, not learned | Untested; DEM is forbidden only as an SR input |
| H10 | **Reconstruction-confidence map as the product**: SR image + per-pixel score of how much is observed vs prior, validated as an error predictor on HR data | Problem redefinition | Yes | Answers the statement's uncertainty requirement directly | ρ machinery exists; A2 cache has HR truth |
| H11 | **Model disagreement as hallucination flag**: SEN2SR-lite vs bicubic vs a second SR model; disagreement predicts error | Ensemble | Yes | Different priors disagree where neither observed | ha\_metric difference exists per dataset |
| H12 | **Analyst triage**: rank 10 m tiles by calibrated change evidence; SR shown only as a visual aid; metric = recall at a review budget | Human-in-the-loop | Yes | Optimise the decision, not the map | NTRO is an analyst organisation |
| H13 | **Sample-based area estimation** (good-practice stratified estimator): change area with CI from a small reference sample, map as stratifier | Classical RS | Yes | Unbiased area even from a biased map | Standard practice; not used |
| H14 | Multi-temporal sub-pixel reconstruction from S2 geolocation jitter | Temporal SR | Yes | Many shifted looks | Killed: jitter (\~1.5 m) is far below the PSF (\~3–5 m), per project context |

**Categories skipped:** retrieval of analogous events (an inventory of past debris-flow shapes could shape a prior, but with no 2.5 m truth it cannot be validated in a week) and a learned end-to-end change network (needs labelled training pairs the project does not have).

## 5. Anti-solutions (no SR allowed)

Three complete designs that never call a super-resolution model; each must be beaten before SR earns a place in the evidence path.

**AS-1 "Ten metres plus time."** Per-pixel harmonic model of NDVI (or red/NIR reflectance) fitted on all valid observations from 2019–2024, including partially cloudy scenes per pixel. Change score = standardised residual of post observations against the seasonal prediction, using F3's normalisation and noise model. Threshold set by split conformal on placebo years: the same calendar offset (January → December) in years with no event, on a different AOI or crop from the test windows. Output: per 10 m pixel P(change), first date of detection, and a change-area estimate with CI. `[predicted]` This removes the season confound by construction and should match or beat the current gate on placebo FAR. Weakness: it never resolves anything narrower than 10 m.

**AS-2 "Bilinear super-resolution mapping."** AS-1's 10 m change probability, plus sub-pixel placement by one of two fixed rules: threshold a bilinear-upsampled change score at 2.5 m (H3), or allocate round(16 · f̂) sub-pixels by bilinear rank, where f̂ is a calibrated fraction (H4). No learned prior can invent texture. `[predicted]` Under the oracle fraction this reaches about 0.83 IoU on F5's proxy; the realistic number depends entirely on the fraction calibration. This is the strongest honest answer to the statement's title, because "super-resolution mapping" classically means sub-pixel mapping.

**AS-3 "Radar for when, optics for where."** Sentinel-1 VV/VH backscatter change and coherence loss detect the event within one 12-day cycle; Sentinel-2 at 10 m delineates once a clear scene exists; terrain flow paths (H9) constrain the delineation. `[predicted]` Cuts the event-to-evidence gap from 129 days to ≤ 12. Requires a human decision to widen the input set (section 12).

**What the anti-solutions establish.** If AS-2 matches SR-ranked allocation within CI on HR-referenced data, SR adds nothing to the decision and can stay only as a labelled visual layer. That is the single comparison the project has not yet made with identical downstream logic.

## 6. Missing information

The information that would make this problem easy is mostly obtainable from other sources or avoidable by changing the task; very little of it should be inferred by SR.

| Missing information | Measure? | Other sensor/source? | Infer? | Avoid by changing the task? | Recommended route |
| --- | --- | --- | --- | --- | --- |
| Sub-10 m change truth for a landslide | Only with HR imagery | Maxar/Planet open-data releases, UAV-LiDAR authors, NRSC, Sen12Landslides / GLaD4CD (10 m labels only) | No | Partly: validate the mapping step on static HR scenes (F5 proxy) and the detection step on 10 m labels | **Obtain** for a few scenes; otherwise avoid |
| Change fraction per 10 m pixel | Indirectly | — | Yes, but needs a calibrated estimator | Yes: report P(change) and expected area instead of positions | **Infer with calibration**, else **avoid** |
| Position of change inside a 10 m pixel | No | HR optical; SAR at 5×20 m does not help | Weakly (gradients of neighbours) | Yes: output a probability surface | **Approximate** (bilinear) or **avoid** |
| Post-event state within weeks | Not by optics in monsoon | Sentinel-1 (12-day revisit, cloud-independent) | No | No | **Other sensor** (needs decision) |
| Seasonal baseline for December | Yes: earlier years' Decembers | Same sensor, archive back to 2017 | Harmonic model | — | **Measure** (archive) |
| Bare-ground endmember on an evergreen AOI | No (no bare ground outside the scar) | Spectral libraries; other AOIs' bare pixels | Yes | Yes: drop unmixing, use calibrated index-to-fraction map | **Avoid** |
| Deployment-matched null distribution | Yes: placebo pairs with the same calendar offset in non-event years | Same sensor archive | — | — | **Measure** |
| Detection power at the chosen α | Yes: injected changes (F3 already does this) and labelled inventories | — | — | — | **Measure** |
| Where debris can physically travel | Yes | Copernicus GLO-30 DEM (constraint review needed) | — | — | **Other source**, as a prior on the change map, never as an SR input |

The one row where SR could in principle help is "position inside a 10 m pixel", and F5 already measured that its contribution there is +0.0016 IoU without NAIP.

## 7. Oracle and anti-experiments

Each oracle hands one component perfect information so a failure can be pinned to a single stage; each anti-experiment removes something the pipeline claims to need.

| ID | Give perfect / remove | Isolates | Data | Read-out | If it fails even here |
| --- | --- | --- | --- | --- | --- |
| O1 | Perfect fraction; ranking ∈ {random, bilinear, bicubic, SR, HR-at-5 m} | Value of ranking information, and of a real 5 m sensor | F5 cache | IoU and boundary F1 by width bin | If SR ≈ bilinear and HR-at-5 m ≫ both, the prior is the problem, not resolution |
| O2 | Perfect ranking field (HR downsampled to 2.5 m); realistic fraction | Ceiling on what any SR could achieve with today's fraction | F5 cache | IoU vs blocky 0.760 | If still below blocky, fraction dominates and no SR can rescue allocation |
| O3 | Perfect exchangeability: random (not checkerboard) window split of the same placebo scores | Whether the gate gap is spatial structure or noise | F1/F2 npz | Test − calibration FAR, block bootstrap | Gap persists under random split → score is miscalibrated, not a split artefact |
| O4 | Same detector, field swapped: SR vs bicubic vs bilinear, threshold at 2.5 m | SR's analytical value with nothing else changing | F5 cache | Paired IoU, no NAIP | SR ≤ bilinear → SR leaves the evidence path |
| O5 | Perfect labels, degraded geometry: rasterise inventory polygons at 2.5, 10 and 20 m | How much landslide area lives below 10 m at all | Colombia inventory (838 polygons, R4); Sen12Landslides if cached | Area share by width; IoU of 10 m blocky truth vs 2.5 m truth | If 10 m truth already gets IoU ≥ 0.85, resolution cannot be the main bottleneck for landslides |
| X1 | Remove SR from the production map (bilinear ranking in F7's exact pipeline) | Whether SR changes the Wayanad answer | F7 cache | IoU(SR map, bilinear map); area difference | IoU ≥ 0.95 → SR is decorative here |
| X2 | Remove history: 1 pre date vs 6 | Value of temporal pooling | F11 6-date pool | Placebo FAR and noise coverage | Small change → time is not helping the way it is used |
| X3 | Remove resolution: run the 10 m rule at 20 m | Whether Wayanad needs 10 m, let alone 2.5 m | R2 imagery | Scar recall, distance to published crown | Same scar at 20 m → resolution is far from binding here |
| X4 | Replace the null: cross-season placebo (Jan pre vs Dec of a non-event year; Jan 2024 pool vs 2023-12-27) | Whether within-season calibration understates deployment FAR | 10 m archive (small STAC fetch) | Window FAR at F2's τ and at `rule_10m` | FAR ≫ within-season → season, not resolution, is the main error source |

## 8. Cheapest falsification experiments

The first five run on artefacts already cached, cost under a working day together, and between them decide whether SR stays in the evidence path. Ranked by information gained per hour; all outcomes are `[predicted]` until run.

| Rank | ID · hypothesis | Mechanism | If true / if false | Cheapest decisive experiment · data · effort | Metric | Abandon when |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | **A0** · The gate fails because test windows are not exchangeable with calibration windows | Spatial structure across a checkerboard on one crop | True: test − cal FAR CI excludes 0 under checkerboard and closes under a random split. False: CI includes 0 in both | O3 on F1/F2 window scores · cached npz · 1–2 h | FAR difference, block bootstrap (tile blocks), 2,000 reps | CI includes 0 → drop the "does not generalise" narrative; FAIL still stands as pre-registered |
| 2 | **A3** · SR changes the Wayanad map | SR ranking moves sub-pixels | True: IoU(SR map, bilinear map) < 0.9. False: ≥ 0.95 | X1: F7 pipeline, bilinear ranking field, all else fixed · F7 cache · 1 h | Map IoU, area difference (km²) | IoU ≥ 0.95 → SR is decorative in production |
| 3 | **A1** · SR adds analytical information over interpolation | Prior places boundaries better than neighbour gradients | True: SR-threshold − bilinear-threshold > 0, CI excludes 0, no NAIP, in 0–50 m bins. False: CI spans 0 or negative | O4 + O1 (random, bicubic arms) · F5 cache · 2–3 h | Paired IoU and boundary F1 (1 px) by width bin | No positive bin without NAIP → SR leaves the evidence path |
| 4 | **A4** · Resolution does not bind at Wayanad | Scar width ≫ 10 m | True: 20 m rule recovers the same scar (recall ≥ 0.9 of 10 m component area). False: < 0.7 | X3 · R2 imagery · 1 h | Area recall vs 10 m component; crown distance | — (informs framing only) |
| 5 | **B2** · Most landslide area lies in features narrower than 20 m | Runout channels are narrow | True: ≥ 30 % of inventory area in < 20 m wide parts; IoU(10 m truth, 2.5 m truth) < 0.8. False: < 10 %; IoU ≥ 0.85 | O5 · Colombia inventory (cached, R4) · 2–3 h, labels only, no imagery | Area share by width; rasterisation IoU | False → resolution cannot be the main lever for landslides; pick narrow-feature applications or drop sub-pixel claims |
| 6 | **A2** · A calibrated fraction makes allocation beat blocky | Monotone index-to-fraction map fixes non-linearity and endmember bias | True: fraction MAE ≤ 0.15 and allocation − blocky > 0, CI excludes 0. False: MAE > 0.2 or allocation ≤ blocky | Isotonic regression of block HR fraction on 10 m NDVI, leave-one-dataset-out · F5 cache · 4 h | Fraction MAE; IoU vs blocky 0.760 | False → stop hard sub-pixel maps; switch to probability (H5) |
| 7 | **B1** · Within-season calibration understates deployment FAR | December greener than January | True: cross-season placebo window FAR at F2's τ ≥ 2× within-season (≥ 0.10). False: within CI of 0.065 | X4 · small STAC fetch (2023-12-27 plus a non-event Jan→Dec pair) · 0.5–1 day | Window FAR, `rule_10m` and gate v2 | True → every current threshold is invalid for Wayanad; seasonal modelling (H6) becomes priority |
| 8 | **C1** · A reconstruction-confidence map predicts where SR is wrong | Low-frequency share is observed; prior-only detail is error-prone | True: AUROC for \|SR − HR\| > t beats the trivial predictor (local 10 m gradient magnitude) by ≥ 0.05. False: ≤ 0.02 | H10 on A2 cache · 1 day | AUROC, reliability curve | False → the uncertainty deliverable reverts to dihedral spread with a stated caveat |
| 9 | **B3** · A harmonic seasonal model beats the pre-pool mean on null FAR | Season is modelled | True: placebo FAR at matched power drops ≥ 30 %. False: < 10 % | H6 on 10 m archive · 1 day | Window FAR at fixed injected-change recall | False → keep pre-pool with season-matched dates |
| 10 | **C2** · SAR shortens latency with usable agreement | Cloud-independent post-event data | True: S1 acquisition ≤ 12 days post; IoU with 10 m Dec parent ≥ 0.4. False: no scene or IoU < 0.2 | H8 · Planetary Computer S1 RTC · 1 day; **only after the constraint decision** | Days to first evidence; agreement IoU | False → optical-only, accept latency, say so |

**Power must ride with every FAR.** Any gate experiment above also reports recall on F3-style injected drops (0.15, 0.30, 0.50 NDVI) at its α. A gate that wins FAR by flagging nothing is a failure, not a pass.

## 9. Research tree

Each of the five formulations gets at least one decisive test; five branches should be killed now on existing evidence.

```text
Original problem: where, when and how much changed, finer than 10 m where supported, with bounded false alarms
├── A. SR as evidence (current TrustSR)
│   ├── A1  SR-threshold vs bilinear-threshold, same detector ........ TEST (rank 3)
│   ├── A3  swap SR → bilinear in F7 production ...................... TEST (rank 2)
│   ├── A-k1 fine-tune SEN2SR-lite (F4) before A1/A3 ................. KILL for now
│   ├── A-k2 heavier / diffusion SR ................................. KILL
│   └── A-k3 neighbour term λ ....................................... KILL (λ = 0 measured)
├── B. Ten metres + time (no SR in the decision)
│   ├── A0  is the gate gap real? (random vs checkerboard split) ..... TEST (rank 1)
│   ├── B1  cross-season placebo null ................................ TEST (rank 7)
│   ├── B3  harmonic seasonal model .................................. TEST (rank 9)
│   └── H7  per-pixel first-valid compositing ........................ PARK (after B1)
├── C. Sub-pixel mapping without a learned prior
│   ├── A2  calibrated fraction + bilinear allocation ................ TEST (rank 6)
│   ├── H5  probabilistic output if A2 fails ......................... CONDITIONAL
│   └── H14 multi-temporal jitter reconstruction ..................... KILL
├── D. Redefine the deliverable
│   ├── C1  reconstruction-confidence map, validated on HR ........... TEST (rank 8)
│   ├── H12 analyst triage at a review budget ........................ PARK (pitch framing)
│   └── H13 sample-based area estimate with CI ....................... PARK (needs reference sample)
└── E. Another sensor / physics
    ├── C2  Sentinel-1 latency and agreement ......................... GATED on constraint decision
    └── H9  terrain flow-path prior .................................. PARK (after B2)

Resolution sanity checks feeding every branch: A4 (20 m degrade), B2 (label geometry)
```

| Killed branch | Evidence | Assumption disproven | Do not try again unless |
| --- | --- | --- | --- |
| Fine-tuning as the next step (F4) | F5: even a perfect ranking field barely matters next to the fraction; F6: SR below bicubic outside NAIP | A2 (texture carries site information) | A1 or O1 shows SR ranking beats bilinear without NAIP |
| Heavier or diffusion SR | ha\_metric worse 5/5 for the light model; diffusion adds sampled texture and exceeds 4 GB | A2 | Same as above, and a 4 GB path exists |
| Neighbour-agreement term λ | Fitted λ = 0 at the grid boundary | Spatial smoothness prior helps | A different score family is adopted |
| Multi-temporal jitter SR | Jitter \~1.5 m vs PSF \~3–5 m | Jitter carries new information | Measured jitter on this AOI is ≥ PSF scale |
| More within-season τ tuning (F11 as the main path) | Keep rule has \~46 % false-fail; null is within-season | A5, A6 | B1 shows cross-season FAR ≈ within-season FAR |

F11 can still finish, because it is cheap and pre-registered; it should not be treated as the route to a Wayanad claim.

## 10. Top 3 directions to test first

These three are chosen because each addresses a measured failure, each can be killed in under two days with data already in hand, and together they cover detection, mapping and the statement's uncertainty requirement. None is declared the winner.

| Criterion | D1 · Ten metres + time, deployment-matched null (branch B) | D2 · Sub-pixel mapping by calibrated fraction, SR only as an ablation arm (branch C, with A1 inside) | D3 · SR product with a validated reconstruction-confidence map (branch D) |
| --- | --- | --- | --- |
| Evidence fit | Targets the season confound, the within-season null and the 129-day gap; `rule_10m` already has pixel FAR 0.000265 | Targets the measured bottleneck: oracle-fraction allocation +0.069 IoU over blocky, realistic fraction −0.313 | Targets the statement's "validated against HR" and "account for uncertainty" lines, which no current result satisfies |
| Falsifiability | A0 and B1 decide it: FAR difference and cross-season FAR, each a single number with a CI | A1 and A2 decide it: one paired IoU delta each | C1 decides it: AUROC vs a trivial predictor |
| Data feasibility | F1/F2 npz cached; one or two extra 10 m dates via STAC | F5 OpenSR-test cache (119 images, 4 datasets) | A2 cache (178 images, 5 datasets) |
| Compute | CPU, minutes | CPU, minutes to an hour; no new SR passes if the SR outputs are cached | CPU; SR outputs already exist from A2 |
| Novelty | Low as method; high as an honest finding ("the event needs time, not pixels") | Medium: classic sub-pixel mapping with modern calibration and a conformal bound | Medium-high: an observed-vs-prior map validated as an error predictor is uncommon in SR papers |
| Downstream relevance | Directly governs whether any Wayanad change claim is defensible | Governs whether any < 10 m change map is defensible | Governs whether the SR deliverable can be shown honestly to NTRO |
| Implementation cost | 1–2 days | 1 day | 1 day |

**How they interact.** D1 decides whether the change decision is sound at 10 m. D2 decides whether anything finer than 10 m can be placed without a learned prior. D3 decides whether SR survives as a product rather than as evidence. If all three come back as predicted, the project becomes: 10 m-and-time detection, bilinear-calibrated sub-pixel delineation, and an SR image whose observed and invented parts are measured and labelled.

## 11. Seven-day experiment program

Days 1–2 use only cached artefacts and settle whether SR stays in the evidence path; days 3–5 settle the null and the seasonal model; days 6–7 test the deliverable and write the new pre-registration. No training anywhere in the week.

Day 0 (2 h): write `configs/f13_discovery.yaml` with every keep rule in section 8, commit it, then run. Each experiment writes the existing result schema, labelled `evidence: real`.

```text
Experiment 1  (Day 1, 2 h)  A0  gate exchangeability: random vs checkerboard split of cached placebo scores
              → tells us whether "does not generalise" is real or sampling noise
Experiment 2  (Day 1, 1 h)  A3  F7 pipeline with bilinear ranking in place of SR
              → tells us whether SR changes the Wayanad answer at all
Experiment 3  (Day 1, 1 h)  A4  10 m rule degraded to 20 m at Wayanad
              → tells us how far resolution is from binding on the flagship event
Experiment 4  (Day 2, 3 h)  A1  SR- vs bicubic- vs bilinear-threshold, plus random and HR-at-5 m ranking (O1, O4)
              → tells us whether a learned prior adds any analytical information, by feature width
Experiment 5  (Day 2, 4 h)  A2  isotonic fraction calibration, leave-one-dataset-out, then allocation (O2)
              → tells us whether sub-pixel allocation can beat blocky with a realistic fraction
Experiment 6  (Day 3, 3 h)  B2  inventory polygons rasterised at 2.5 / 10 / 20 m
              → tells us how much landslide area lives below 10 m at all
Experiment 7  (Day 3–4)     B1  cross-season placebo (Jan → Dec, non-event) at 10 m
              → tells us whether current thresholds are valid for a December post date
Experiment 8  (Day 5)       B3  harmonic seasonal model vs pre-pool mean, FAR at fixed injected recall
              → tells us whether modelling season beats matching dates
Experiment 9  (Day 6)       C1  reconstruction-confidence AUROC vs trivial predictor on A2 cache
              → tells us whether SR can be shipped with a validated "observed vs invented" map
Experiment 10 (Day 6, opt.) C2  Sentinel-1 availability audit only (dates within 12 days of 2024-07-30)
              → tells us whether the sensor decision is even worth raising; no processing until decided
Day 7       Synthesis: experiment log per section 8 template, update research tree, apply section 12,
            pre-register the next formulation before any build
```

**Order logic.** Experiments 1–3 can each end a branch in an hour. Experiment 4 is the single most decision-relevant result in the week. Experiments 7–8 need a small fetch, so they follow the cached-data runs. If experiment 4 is clearly negative and experiment 2 gives IoU ≥ 0.95, day 6 shifts entirely to C1, because the SR deliverable then depends on it.

## 12. Pivot and abandon criteria

These thresholds belong in the Day 0 pre-registration, written before any of the experiments run.

| Decision | Trigger (all from section 8 experiments) |
| --- | --- |
| **Keep TrustSR with SR in the evidence path** | A1: SR-threshold or SR-ranked beats the bilinear equivalent by ≥ 0.01 IoU in the 0–50 m bins, NAIP excluded, CI excluding 0; **and** A3: IoU(SR map, bilinear map) < 0.9 on Wayanad; **and** A2 or the oracle shows allocation beats blocky |
| **Remove SR from the decision; keep it as a labelled visual layer** | A1 CI spans 0 or is negative without NAIP, **or** A3 IoU ≥ 0.95. SR then appears only as "model reconstruction", with the C1 map if C1 passes |
| **Remove SR entirely** | A1 negative **and** C1 fails (AUROC gain ≤ 0.02 over the trivial predictor). Ship bilinear sub-pixel mapping and say why, with F6's PSNR table as the evidence |
| **Add another sensor (Sentinel-1)** | B1 cross-season FAR ≥ 2× within-season **or** no clear optical post scene within 60 days for the target event; **and** the C2 audit finds an S1 acquisition within 12 days |
| **Switch the output from a hard map to a probabilistic map** | A2 fraction MAE > 0.15 after calibration, **or** calibrated allocation ≤ blocky. Report P(change) per 10 m pixel and expected area with CI; score with Brier and reliability, not IoU |
| **Reframe the whole project** | B2 shows < 10 % of landslide area in parts narrower than 20 m **and** A4 finds the Wayanad scar at 20 m. Then sub-10 m landslide mapping is not where value lies: reframe to (a) an honest SR product with a validated observed-vs-invented map for NTRO's stated applications, and (b) narrow-feature applications (roads, stream banks, field boundaries) where F5's only surviving gain sits |
| **Abandon the conformal gate** | A0 shows a real gap **and** B1 shows a large cross-season gap **and** B3 does not close it. The gate's exchangeability assumption cannot be met with this archive |

**Constraint decisions this would need (human, not this memo).** Adding Sentinel-1 conflicts with the AGENTS.md line "Bands: only the four 10 m bands (B02, B03, B04, B08). No 20 m or 60 m bands." as currently interpreted for all inputs. A terrain prior conflicts with the project-context rule "no DEM as input" if that is read beyond the SR model. Neither line is changed here. The cheap audits (C2, B2) produce the evidence a human needs to decide.

## 13. Stop now, and what to do next

**Stop immediately:**

- **Fine-tuning (F4) and any larger SR model.** Nothing measured so far says ranking quality is the limit; the fraction and the null are.
- **Treating F11 as the path to a Wayanad claim.** Its keep rule fails a perfect gate about half the time at the point estimate, more often across three `e_b` conditions, and its null stays within-season. Let it finish if it is nearly done; do not build on it.
- **Quoting the Wayanad map, the 89.4 % "differs from blocky", ρ medians or any NAIP-inclusive number as evidence** that SR adds information.
- **Engineering for scale before science**: E5 cascade work, GPU/4 GB optimisation, demo UI. They answer "can we implement it", not "does it work".
- **Scoring any gate on FAR without power.** Every FAR row needs recall at α beside it.
- **Adding parameters to rescue the gate** (λ, new endmember ceilings, score tweaks). λ = 0 was measured; a new knob is a new hypothesis and needs its own pre-registration.

**What should we do next, before writing significant new code?**

1. **Write and commit one pre-registration** (`configs/f13_discovery.yaml`) holding the keep rules and pivot thresholds in sections 8 and 12, including the power requirement. About two hours.
2. **Run the three one-hour checks on cached artefacts**: A0 (is the gate gap real?), A3 (does SR change the Wayanad map?), A4 (does Wayanad even need 10 m?). Each needs tens of lines, not a module.
3. **Run A1 and A2 on the F5 cache**: SR vs interpolation with an identical detector, and a calibrated fraction. These two numbers decide whether SR belongs in the decision path and whether sub-pixel maps can beat blocky.
4. **Read the results against section 12 before choosing what to build.** Only then fetch new dates (B1), model season (B3) or raise the sensor question (C2).
5. **Confirm the finale date.** If it is under three weeks away, run only steps 1–3 and B1, and pitch the honest version: detection at 10 m with a measured null, sub-pixel placement only where it beats blocky, SR shown with its invented parts labelled.

The single most decision-relevant unknown is the A1 result: whether a learned prior places any boundary better than bilinear interpolation once NAIP is removed. The second is B1: whether the false-alarm rate survives a December post date.
