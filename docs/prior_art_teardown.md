# Prior-art teardown: Sentinel-2 to <4 m for trustworthy change mapping

Agent A1, read 2026-09-30. Literature and code reading only: no dataset or weight was downloaded, nothing was run except one small synthetic calculation (Section 8, labelled). Every source and what was verified from it is in [sources.md](sources.md) under "A1 prior-art pass, 2026-09-30".

**Evidence tags.** `[P]` = read on the primary page (paper page, README, model card, metadata). `[S]` = secondary description of a primary that I could not read (named). `[D]` = derived here from stated assumptions, with the derivation shown. `[U]` = unverified. A "better than" statement is only made where a primary source gives a comparable metric; otherwise it is marked `[U]`.

**Primaries I could not read** (so everything below about them is `[S]` or `[U]`): the SEN2SR journal paper (Remote Sensing of Environment 334:115222, 2026; ScienceDirect returned 403, the SSRN preprint returned 403), the LDSR-S2 paper (IEEE JSTARS 18:6940-6952, 2025; the UV repository copy was blocked by an anti-bot page and IEEE Xplore returned no text), the Sen4x code (announced as "will be" public), Atkinson (2005) full text (Southampton eprints returned 403; abstract seen through a search snippet only), Ling and Foody (2019) DeepSRM full text (403).

## 1. Verdict in one paragraph

No existing product delivers "Sentinel-2 to <4 m that is trustworthy for change mapping" as such. The closest **SR** components are SEN2SR (best consistency story, smallest, permissive licence), LDSR-S2 (only one with a per-pixel uncertainty output, but 10 to 20 times the compute) and the multi-image SR family (the only family that genuinely adds measurement information). The closest **task** solutions are 10 m bi-temporal landslide detectors that use more than four bands and a DEM. The one independent, downstream, head-to-head test of SEN2SR-lite, LDSR-S2 and SR4RS that I found (a 2025 TU Wien thesis, building delineation) found that **no SR product beat plain interpolation** on that task. That is the most important fact for TrustSR's design: 2.5 m information from a single-date SR model must be treated as unproven until a paired test against interpolation says otherwise. The v2 design in [v2_design.md](v2_design.md) is built so that its main guarantees do not depend on SR adding information.

## 2. Candidate table

| Candidate | What it is `[P]` unless tagged | Bands / scale | Licence | Independent downstream evidence found | Change / uncertainty story |
|---|---|---|---|---|---|
| SEN2SR-Lite (`NonReference_RGBN_x4`) | SPAN CNN, 472,496 params, model card "memory requirement 1 GB", trained on SEN2NAIPv2 (mlm.json) | B04,B03,B02,B08, x4, fixed 128 to 512 tile | Weights CC0-1.0 (HF); code: repo badge CC0-1.0, PyPI page says MIT (conflict, unresolved) | TU Wien building delineation: F1 0.468, IoU 0.341 (bicubic 0.477 / 0.348; NN 0.485 / 0.354) | Low-frequency hard constraint; no uncertainty output; LAM (local attention map) tool for spatial sensitivity |
| SEN2SR full (`SEN2SR/NonReference_RGBN_x4`, "SENSR", MambaSRv1) | 13,759,444 params, 36.7 MB, requires CUDA and `mamba_ssm`, same 1 GB "memory requirement" in metadata | RGBN x4 (variant list also has 10+20 m to 2.5 m and RSWIR x2) | as above | TU Wien: SEN2SR-RGBN (Mamba) F1 0.478, IoU 0.349, the best SR in that test, 0.001 above bicubic | same as lite |
| LDSR-S2 (`opensr-model`) | Latent diffusion, RGBN x4 128 to 512; `[S]` conditioning on the encoded LR image for spectral consistency; per-pixel uncertainty from repeated sampling | RGBN x4 | LICENSE file exists, terms not retrieved `[U]` | TU Wien: F1 0.450, IoU 0.323 (worst of the CNN/GAN/diffusion set except DeepSent) | Sample std as per-pixel uncertainty; ESA page: "10x" sampling and a "factor of 10 to 20" compute cost |
| Sen4x / multi-image SR | Swin2SR-style single-image branch plus HighRes-net-style recursive fusion over 8 revisits, about 30 M params, Pleiades Neo targets | RGBN x4, Hanoi only | Code promised, not verified | Own paper: land-cover OA 74.6 %, mIoU 51.6 % vs Swin2SR 71.4 %, HighResNet 58.3 %, HR reference 85.6 % | "Multi-temporal fusion mitigates hallucination" (their claim, `[P]` abstract-level summary) |
| calebrob6/s2-superres | Optimisation of a Gaussian-splat scene with per-observation sub-pixel shifts, about 30 revisits, no neural net | B02,B03,B04,B08; "2 to 5 m" | MIT | none; edge-contrast and Laplacian-variance experiments only | README states the limit: about 30 shifted observations "mostly contain redundant spatial information"; jitter about 1.5 m std vs PSF 3 to 5 m |
| SR4RS | SRGAN trained on Spot-6/7 vs Sentinel-2 over France, OTBTF/Docker | RGBN, 2.5 m | MIT | TU Wien: F1 0.451, IoU 0.325; "hallucinations are noticeable" in their visual check | none |
| S2DR3 | 12 bands to 1 m, "features down to 3 m"; released Oct 2023 by Yosef Akhtman | all 12 bands | none found `[U]` | none found | Only a third-party repository (README plus notebook, no weights) and search-snippet descriptions were readable; the author's Medium page returned 403. TU Wien could not run it either (weights only on request) `[S]` |
| OpenSR-SRGAN | Configurable SRGAN framework | RGBN or 6-band | CC BY 4.0 (paper) / Apache-2.0 (repo listing) | own paper: PSNR 31.45 dB, SSIM 0.81 on SEN2NAIP x4 | Explicitly no uncertainty; "research and benchmarking blueprint"; two A100 GPUs for training |
| DiffFuSR | RGB diffusion SR plus fusion network to all 12 bands at 2.5 m | 12 bands | not retrieved | Best average rank 2.85 on OpenSR-test | hallucination scored by OpenSR-test only |
| BBUnet (arXiv 2405.20161) | Bitemporal UNet, 12 bands plus DEM, 256x256 patches at 10 m | 12 S2 bands + elevation/slope/aspect | not retrieved | Haiti test: F1 0.348, precision 0.582, recall 0.249 | none |
| Sen12Landslides (Sci. Data 2025) | 75,000 annotations, 12,000+ patches, 15 regions, 128x128 at 10 m, 15 timesteps, 10 S2 bands | B02 to B12 (no 60 m) | CC BY 4.0 | U-ConvLSTM S2 F1 0.83 +- 0.01 on the paper's own split | Precision higher than recall |

The F1 values in the last three rows are **not comparable** with each other or with the SR rows: different regions, labels, splits and inputs.

## 3. Top-3 teardown

Selection rule: the three components a TrustSR-class pipeline would actually consider building on, ranked by how much a primary source lets me say about them. Change-detection networks are treated in Section 6 because they are not SR.

### 3.1 SEN2SR (lite and full)

- **Product.** A pip package (`sen2sr`, version 0.8.5 in `pyproject.toml`) and Hugging Face weights (`tacofoundation/SEN2SR`, about 583 to 609 MB in total across variants) with Colab notebooks. Published as "A radiometrically and spatially consistent super-resolution framework for Sentinel-2" (Aybar, Contreras, Donike, Portales-Julia, Mateo-Garcia, Gomez-Chova; RSE 334:115222, 2026) `[P]` citation, `[U]` content.
- **Workflow.** Normalised 128x128 patches in, 512x512 out; for larger scenes tile with overlapping margins (the README says 32 px margins; this repository uses a 16 px crop margin) `[P]`. The wrapper is `SR net -> clamp(>=0) -> HardConstraint(lr, sr)` `[P]` (docs/evidence/sen2sr-wrapper.txt).
- **Technology.** Lite: SPAN-type CNN. Full: Mamba. The hard constraint blends the LR input into low frequencies of the output through a fixed Fourier low-pass mask; in this repository the stored mask was measured to be a Gaussian of sigma about 35 px on a 512 FFT (EVIDENCE.md, step 3a). Secondary description (TU Wien thesis, section 2.2.2.6): trained on synthetic SEN2NAIPv2 and CloudSEN12; evaluated on OpenSR-test and cross-sensor SEN2NAIPv2; downstream flood (WorldFloods) and methane-plume (MARS-S2L) tests; models above about 15 M parameters "did not achieve significantly better results" than smaller ones; Mamba beat CNN `[S]`.
- **Validation reference.** Training and validation pairs come from NAIP (continental United States only) `[S]`. The SEN2NAIP construction: a strictly co-temporal cross-sensor subset (within one day) is used to train degradation models that create the synthetic LR from NAIP, and the synthetic subset is what is used for training `[S]` (TU Wien section 2.1). No independent held-out set outside the United States is reported in the pages I could read.
- **Compute.** Model card: "memory requirement 1 GB" for both lite and full, fp16 I/O `[P]` (declared, not measured). Measured here: lite 128 tile, 100.48 MiB peak allocated, 0.1208 s per tile on an RTX 4050 (RISK_REPORT R1). The full model needs CUDA and `mamba_ssm`; its behaviour under a 4 GB cap is **unmeasured** `[U]`.
- **Licence.** Weights CC0-1.0 `[P]`. Code: CC0 badge on GitHub, MIT on PyPI `[P]`, unresolved.
- **Independent downstream evidence.** TU Wien (Hollendonner, 2025): SEN2SR-RGBN IoU 0.349 / F1 0.478; SEN2SR-Lite 0.341 / 0.468; bicubic 0.348 / 0.477; nearest neighbour 0.354 / 0.485; orthophoto 0.552 / 0.681. Conclusion of the thesis: "for building delineation, Super-Resolution currently offers no advantage over interpolation". In the no-building stratum (1,133 images) the better SR products did better than interpolation (SEN2SR-RGBN 0.941, SEN2SR-Lite 0.953, SR4RS 0.941, Evoland 0.971, Swin2Mose 0.953 vs 0.894 to 0.918 for the interpolators), which the thesis reads as fewer false positives on empty scenes; LDSR-S2 (0.912) and DeepSent (0.900) did not. **Comparability gap:** building delineation with a UNet trained per product, Austria, one split, no confidence intervals reported in the results tables I read, not change detection, not landslides.
- **Assumptions that break for TrustSR (4 GB, 4-band, change mapping, trust labels).**
  1. *Consistency is not correctness.* The constraint pins block-scale means (measured mean absolute downsample error here: 0.0004 to 0.0005 in visible bands, 0.0022 to 0.0029 in NIR, EVIDENCE.md) and says nothing about where inside a 10 m block change sits. A change map built from the SR output inherits SR-chosen edge positions with no error bar.
  2. *Training-domain assumption.* Synthetic degradation of NAIP over the United States (`[S]`) is assumed to match Sentinel-2's real PSF, noise and registration; a tropical montane forest with cloud shadow and terrain shading is far from that domain `[U]`.
  3. *Fixed 128/512 geometry.* The pinned mask has fixed 512x512 shape, so the required 64 px OOM fallback fails (RISK_REPORT R1).
  4. *Nothing labels change.* No uncertainty output; the LAM tool measures spatial sensitivity, not error.
  5. *The full model is not shown to fit 4 GB* `[U]`.

### 3.2 LDSR-S2 / OpenSR latent diffusion

- **Product.** `opensr-model` (v1.0.0), Colab notebooks, companion `opensr-utils` for whole-tile tiling and GeoTIFF I/O; companion benchmark `opensr-test` (MIT) `[P]`.
- **Workflow.** RGBN 128 to 512; uncertainty by "running the SR model multiple times (e.g., 10x sampling) for a single LR input"; per-pixel confidence intervals from the sample standard deviation `[P]` (ESA project page).
- **Technology.** Latent diffusion adapted so that the encoded LR image conditions the diffusion for spectral consistency `[S]` (abstract-level summaries of the JSTARS paper).
- **Validation reference.** SEN2NAIP-trained; compared against SR4RS, Satlas SRGAN and others on OpenSR-test; reported best on all metrics except synthesis `[S]` (TU Wien section 2.2.2.5 summarising the paper). OpenSR-test metrics (README `[P]`): reflectance (L1), spectral (spectral angle), spatial (phase correlation), synthesis, and correctness split into hallucination ("a detail (high-gradient) in the SR image that is not present in the HR image"), omission and improvement. Datasets: NAIP 62 x4, SPOT 9 x4, Venus 59 x2, Spain crops 28, Spain urban 20. Very small sets; the SPOT set has 9 images.
- **Compute.** ESA project page: multiple samples "significantly increases the computational cost (by a factor of 10 to 20)". A DiffFuSR paper reports 1.74 s for its own diffusion inference (a different model) `[P]`; no GPU-memory number for LDSR-S2 was found `[U]`.
- **Licence.** Repository has a LICENSE file whose terms I could not read `[U]`.
- **Independent downstream evidence.** TU Wien: F1 0.450, IoU 0.323, below every interpolation method and below SEN2SR. Their explanation, which is a hypothesis and not a test: generative backbones "likely introduced hallucinations".
- **Assumptions that break.**
  1. *Uncertainty is about appearance, not about the change decision.* The ESA page itself reports wider intervals in "high-texture areas, such as houses or fine field lines", i.e. the sample spread tracks texture. Nothing shows that spread predicts a wrong NDVI-drop decision.
  2. *No guarantee attached.* A standard deviation of sampler output is not calibrated to any error rate; there is no threshold with a stated false-alarm rate.
  3. *Cost.* Two dates times N samples times tiles; the project's 4 GB and time budget make this a poor fit `[D]`.
  4. *Same synthetic-domain assumption as SEN2SR.*

### 3.3 Multi-image (multi-temporal) Sentinel-2 SR: Sen4x, HighRes-net family, calebrob6/s2-superres

- **Product.** Research code. Sen4x (arXiv 2505.24799) announces code at github.com/ADB-Data-Division/sen4x "will be made publicly available" `[P]`; I did not verify that it is released. BreizhSR (EarthVision 2024) extends SRDiff and HighRes-net to irregular time series `[P]` abstract-level. s2-superres is a runnable MIT repository `[P]`.
- **Workflow.** Sen4x: eight revisits chosen by temporal proximity, cloud cover and spectral quality, fused recursively, plus a single-image branch. s2-superres: about 30 observations, per-observation sub-pixel shifts, PyTorch optimisation, "a few minutes" per 256x256 crop `[P]`.
- **Validation reference.** Pleiades Neo (Sen4x), SPOT-6 (BreizhSR), none (s2-superres). Sen4x downstream: land cover in Hanoi, OA 74.6 % (mIoU 51.6 %) vs 71.4 % single-image and 58.3 % pure multi-image; PSNR and SSIM poorly predicted downstream utility, only LPIPS did `[P]` summary. **Comparability gap:** one city, one task, own split, no change task.
- **Compute.** Sen4x: 189.6 ms per 64x64 patch, hardware not stated `[P]`. s2-superres: minutes per crop.
- **Why it matters.** It is the only family whose extra information has a physical mechanism (sub-pixel jitter between revisits) instead of a learned prior. The repository's own README bounds it: jitter of about 1.5 m std against a 3 to 5 m PSF, and about 30 observations "mostly contain redundant spatial information".
- **Assumptions that break.**
  1. *Static scene across the stack.* The pre-event side of a change map can pool many clear dates (a static scene); the post-event side usually has one or a few clear dates (129 days late in the current Wayanad run), so information is asymmetric between the two sides of the difference `[D]`.
  2. *Phenology and cloud.* Wayanad's clear pre-event dates are January to May only (RISK_REPORT R2), so a stack is season-limited.
  3. *Registration.* Sub-pixel jitter is the signal here and registration error is the noise; the Sentinel-2 multi-temporal registration goal is 0.3 pixel at 2 sigma (Gascon 2014) `[P]` (a mission planning goal, not a measured value), i.e. the same order as the jitter being exploited.
  4. *Not shown on landslides.*

## 4. Independent evidence on "does SR help downstream"

| Source | Task | Result | Gap |
|---|---|---|---|
| TU Wien thesis (Hollendonner 2025), read in full text | building delineation, Austria, UNet per product | best interpolator (nearest neighbour, F1 0.485) ahead of every SR product; best SR (SEN2SR Mamba, 0.478) +0.001 vs bicubic (0.477) | not change; single split, no CIs seen |
| Evoland (via TU Wien 2.2.2.2) `[S]` | small-water-body detection, spectral index, France | improved detection of small water bodies, especially using enhanced 20 m bands | 20 m bands, not our 4-band constraint |
| Swin2Mose, SEN2SR (via TU Wien) `[S]` | land cover (SeasoNet); flood and methane plume | reported improvements by the model authors | authors' own tests, secondary description |
| Sen4x `[P]` | land cover, Hanoi | multi-image plus single-image > single-image > multi-image alone | own test |

So the evidence is mixed and task-dependent, and it contains no landslide or change-detection test of any of these SR products. I found no primary source that evaluates single-image SR for bi-temporal landslide change detection (search performed, not exhaustive; treat "none exists" as `[U]`).

## 5. Sub-pixel mapping (SPM) for change

- **Concept `[P]`.** Soft classification or unmixing gives class proportions per coarse pixel but not their location; SPM allocates them to sub-pixels using the spatial dependence assumption (nearer sub-pixels are more likely the same class) (Atkinson 2005 abstract; PolyU/SPM summaries).
- **Pixel swapping `[P]` abstract.** Start from a random allocation of the soft proportions to hard sub-pixel classes, then iteratively swap the sub-pixel of the target class with the minimum attraction value and the background sub-pixel with the maximum, if the swap raises spatial correlation. Because it only swaps, the number of target sub-pixels per coarse pixel stays equal to the initial allocation: **proportion (area) preservation is by construction**. That property is what the v2 hypothesis calls area-preserving.
- **SPM for change detection `[P]`.** Wang, Atkinson and Shi (IEEE TGRS 53(4), 2015): five fast SPM algorithms (bilinear, bicubic, sub-pixel/pixel spatial attraction, kriging, radial basis function interpolation) applied to sub-pixel resolution change detection; abstract states feasibility, no quantitative comparison in the abstract. **Consequence:** bilinear/bicubic interpolation of fractions is itself a published SPM baseline, so "allocate by an interpolated fraction surface" is the honest null model for "does SR add anything".
- **Supervised SPM `[S]`.** DeepSRM (Ling and Foody 2019): a CNN maps coarse fractional images to fine indicator images trained on existing fine maps; superior to conventional SPM on **simulated** images (search-result summary; full text not read). Comparability gap: simulated data.
- **Not found.** No source combining SPM with SR-model change scores, or with a statistical false-alarm guarantee, in the searches I ran (`[U]`, absence of evidence).

## 6. Landslide Sentinel-2 change detectors and datasets

| Item | Facts read `[P]` | Fit to TrustSR |
|---|---|---|
| arXiv 2405.20161 (Monopoli, Montello, Rossi) | 34,920 polygons across Haiti (14,482), Central Sulawesi (6,749), Luding (8,194), Mesetas Colombia (838), Iburi (4,657); BBUnet uses 12 bands plus DEM elevation, slope, aspect; Haiti F1 0.348 (P 0.582, R 0.249); authors cite "the relatively low resolution of S2 ... in comparison to landslides' size" | Uses more than four bands and DEM (excluded); shows that at 10 m recall is the weakness (0.249) |
| Sen12Landslides (Sci. Data, Nov 2025) | 128x128 at 10 m, 15 timesteps per patch, B02-B12, CC BY 4.0, Hugging Face `paulhoehn/Sen12Landslides`; F1 S2 0.83 +- 0.01 (U-ConvLSTM); date-confidence scores derived from NDVI vegetation loss; models more precise than recall | **Usable as a multi-event labelled set and, because of the timesteps, as a source of pre-vs-pre placebo windows** (four bands only). Not downloaded. Date labels are themselves NDVI-derived, which correlates with an NDVI-drop detector `[D]` |
| GLaD4CD v1 (Zenodo 10800338, 9 Mar 2024) | 174 events; `LANDSLIDE_DATASET.zip` 2.0 GB; bands B02,B03,B04,B08,CLM; CC BY 4.0; change maps in LABEL for the validation set | Matches the band constraint; label count small |
| GLaD4CD v2 (Zenodo 14226448, 27 Nov 2024) | `GLaD4CD_2.0.zip` listed as 294.7 MB; **13 bands** (v2), TRAIN and TEST folders; change maps only for TEST (17 pairs) | The 2.0 GB to 294.7 MB drop is unexplained `[U]`; only 17 labelled pairs; use only B02/B03/B04/B08 |
| Landslide4Sense-2022 | 3,799 train / 245 val / 800 test patches, 128x128, about 10 m, 12 S2 bands plus slope and DEM, MIT; U-Net baseline validation F1 57.82 % (P 51.75, R 65.50) | Single-date segmentation; no temporal labels |

Every one of these labels is a 10 m-scale inventory. None gives 2.5 m truth, so none can measure "SR adds information", only detection at 10 m or coarser.

## 7. Theory that bears on trust claims

- **Hallucination is structural, not only model-specific.** Iagaru, Gottschling, Hansen and Garnier (arXiv 2605.13146): hallucinations arise from the ill-posedness of the inverse problem; the paper states necessary and sufficient conditions and "computable bounds on their magnitude that depend only on the forward model" `[P]` abstract-level. I have not read the derivations, so I use it only as motivation.
- **Conformal calibration `[P]`.** Angelopoulos and Bates (arXiv 2107.07511, read in full text): marginal validity over the calibration set; conditional on the calibration set the realised coverage is Beta(n+1-l, l) with l = floor((n+1) alpha) (Vovk); "about 1000 calibration points" give coverage typically in .88 to .92 for the 90 % case; the validity proof needs exchangeability of calibration and test points, and adjacent geographic or temporal points violate it (their Section 5.3 example explicitly says so). Angelopoulos et al. (ICML 2022) apply distribution-free intervals to image-to-image regression including a super-resolution microscopy example `[S]`.
- **Conformal p-values and FDR `[P]`.** Bates, Candes, Lei, Romano and Sesia (Annals of Statistics 51(1), 2023): conformal p-values are marginally valid, mutually dependent, positively dependent, and enable exact FDR control.
- **Spatial exchangeability `[P]` abstract.** Mao, Martin and Reich (JASA 119(546), 2024): spatial data can be treated as exactly or approximately exchangeable in a wide range of settings; local approximate exchangeability is proved under an infill asymptotic regime. This supports, but does not prove, using windows as the exchangeable unit.

## 8. Physical facts that constrain any sub-pixel design

1. **The 10 m pixel is not a box average.** ESA's Sentinel-2 requirement (Gascon, 28 Jan 2014 slide) is MTF 0.15 to 0.3 at Nyquist for the 10 m bands, multi-temporal registration 0.3 pixel at 2 sigma "goal with GCPs", absolute radiometric uncertainty 3 % goal / 5 % threshold, inter-band relative radiometric uncertainty 3 % `[P]` (requirements and goals, not measured performance). For reference a perfect 10 m box has MTF sinc(0.5) = 0.637 at Nyquist `[D]`. So a pixel integrates a **wider** footprint than its own 10 m cell.
2. **Calculation (synthetic, run here, assumptions stated).** One-dimensional straight edge; block cell 10 m wide; pixel response Gaussian with MTF(Nyquist) = m, sigma = sqrt(-ln m / (2 pi^2 f_N^2)) with f_N = 1/20 m^-1. For m = 0.30, 0.20, 0.15 (sigma 4.94, 5.71, 6.20 m) the mean absolute difference between the true in-cell area fraction and the fraction the pixel reports, over blocks that straddle the edge, is 0.064, 0.086, 0.097 (max over all offsets 0.156, 0.191, 0.210), i.e. roughly one to one and a half sub-pixels per boundary block. A Gaussian PSF is an illustration; the real PSF was not measured here.
3. **NDVI is not linear in area fraction.** Jiang et al. (RSE 101(3), 2006): NDVI shows scale dependence over heterogeneous surfaces and nonlinearity over partial cover, especially with dark soil or shadow; a scale-invariant index based on linear mixing of red and NIR (SDVI) is proposed `[P]`. Illustrative calculation (assumed, not measured, endmembers red/NIR 0.03/0.40 for forest and 0.15/0.20 for bare ground): a 10 m NDVI drop of 0.30 corresponds to a bare fraction of about 0.47, and a 10 m NDVI drop of 0.12 to a fraction of 0.20. This matches, in direction, the v1 diagnosis (a 0.3 parent threshold discards mixed boundary pixels), but the endmember values are assumptions.
4. **Registration and radiometry set a floor.** 0.3 px at 2 sigma is 3 m, larger than one 2.5 m sub-pixel; any 2.5 m change score along strong edges is sensitive to it `[D]`.

## 9. What nobody does (and where TrustSR may differ)

Searched, not exhaustive: I did not find a source that (a) constrains a 2.5 m change map to the 10 m evidence's area exactly, (b) calibrates a detection threshold on no-event placebo pairs with a stated false-alarm guarantee, or (c) reports whether SR-derived within-block ranking beats a fraction-interpolation baseline. Each is `[U]` as a novelty claim; a jury may know a counter-example. The defensible wording is "we did not find prior art for this combination".

## 10. Unverified claims (do not present as fact)

1. SEN2SR paper contents (hard-constraint definition, benchmark numbers, downstream tests): known only through the TU Wien thesis.
2. SEN2SR training domain and synthetic degradation details (via the thesis).
3. LDSR-S2 paper contents beyond the ESA project page and repository README.
4. LDSR-S2 licence; Sen4x code release and licence; DiffFuSR licence.
5. SEN2SR full-model VRAM behaviour on 4 GB, and the model card's "1 GB" figure (declared, not measured).
6. S2DR3 methodology and licence.
7. GLaD4CD v2 size and band count discrepancy (2.0 GB / 5 bands vs 294.7 MB / 13 bands).
8. Any claim that single-image SR helps or hurts landslide change detection (no test found).
9. Absence of prior art for the combination in Section 9.
10. Real Sentinel-2 PSF and measured MTF for B02/B03/B04/B08 (requirements read, not measurements).
11. Whether TU Wien's small IoU/F1 differences are statistically significant (no confidence intervals in the tables I read).
