# TrustSR Phase 0 execution report

Updated 2026-09-29. **Phase 0 experiments complete; the Phase 1 gate is NOT PASSED. Phase 1 has not started.**
All five risk scripts have real results. R5 completed 50 steps on three genuine WorldStrat pairs in Colab and saved/reloaded a checkpoint. R1 still fails the requested 512px and 64px tile shapes, and R2's first clear post-event Wayanad scene is 129 days late. Those findings require a design decision before Phase 1.

## Results

| Risk | Outcome | Observed result | Implication |
| --- | --- | --- | --- |
| R1 model fits | FAIL overall; native 128 input passes | RTX 4050 Laptop, 6140.5 MiB; FP32 128x128 -> 512x512: median **0.1208 s/tile**, peak allocated **100.48 MiB**, peak reserved **128 MiB**. Direct 512 and 64 input tiles fail fixed-mask shape checks | Native 128 inference is viable; arbitrary tile sizes and required 64px OOM fallback need a justified adapter before the production SR component |
| R2 imagery | PASS for configured availability test | **147 STAC items, 73 acquisition groups**, all SCL reads succeeded. **12 distinct clear pre-event dates**. First clear post-event date: **2024-12-06**, **0.0664%** AOI cloud/shadow | Clear imagery exists, but that post-date is **129 days after the event**; this selection cannot support an immediate-response claim |
| R3 paired-data resource | PASS for metadata/size/licence audit | **3,928 unique AOI IDs**, **71 India**, **125 South Asia**, **107,036,250,648 bytes** across record files | Regional coverage exists; these counts do not prove usable cloud-free/co-registered training pairs |
| R4 labels | PASS for sample access | Author inventory: **838 polygons**. Real Colombia pre/post crop: **748 valid labelled landslide pixels**. IBM-NASA Landslide4Sense training image/mask: **405 landslide pixels** | Both label types are downloadable; the full held-out calibration split still needs preparation |
| R5 fine-tune | PASS smoke test | **50** steps on three verified WorldStrat train AOIs in Colab T4; first five-loss mean **0.01166859**, last five-loss mean **0.01159419**; checkpoint saved and reloaded with matching output | The optimizer path works on real pairs; this is not an SR accuracy or disaster-detection result |

Machine-readable evidence: [R1](risk/results/r1.json), [R2](risk/results/r2.json), [R3](risk/results/r3.json), [R4](risk/results/r4.json), [R5](risk/results/r5.json). Exact policies are in [configuration](configs/phase0.yaml); per-run policy snapshots are in risk/results/config_snapshots/.

## R1: what was actually measured

The pinned upstream SEN2SR-lite NonReference_RGBN_x4 model and weights were loaded, not a substitute network. Inputs were seeded synthetic tensors **only for memory and latency measurement**; no image-quality inference is made. Batch size 1, FP32, one warm-up and three timed forwards; CUDA synchronization surrounds timing. Timing includes input tensor generation and output checks, excludes loading.

The 128px run's three times were 0.1181, 0.1250 and 0.1208 seconds. PyTorch's peak counters include warm-up. The device-wide memory reading **after** the run was 1,218.5 MiB, which includes context/other device use and is not a measured device-wide peak.

A 4 GiB PyTorch allocation cap was imposed on the 6 GB card. This excludes CUDA context/other processes and **does not verify a physical 4 GB GPU**. A direct 512px input produced a 2048-vs-512 dimension error; 64px produced a 256-vs-512 error. Both arise from the pretrained Fourier constraint's 512x512 output mask. There was no CUDA OOM in these three probes. Unit tests, rather than this hardware run, exercise OOM halving and minimum-size failure.

Source review also found that upstream trainable_model constructs Conv3XC with train_mode=False. That branch recomputes detached inference weights. The R5 loader now enables the existing train_mode=True path after loading pretrained weights and leaves the hard constraint frozen. A CPU test confirms equivalent forward values on a tiny upstream model and nonzero gradients reaching its convolution weights. This is a mechanics test, not WorldStrat training.

## R2: clear-date audit

AOI: a metric 10,000m square centred at 11.782N, 76.233E in EPSG:32643. Audit grid: 500x500 at SCL's 20m spacing. The only non-10m layer read is the explicitly permitted categorical SCL quality layer.

Policy fixed before selection: cloud classes 8/9/10; dark-area/shadow classes 2/3; invalid classes 0/1/11. “Clear” requires cloud-plus-shadow <=10% of **all AOI pixels** and valid coverage >=99%. The scene table separately reports cloud-only and valid-pixel cloud fractions. This is an SCL-based estimate, not a manual assertion of perfect visibility.

Clear pre-event dates:
2024-01-11, 2024-01-21, 2024-01-31, 2024-02-05, 2024-02-10, 2024-02-15,
2024-03-01, 2024-03-06, 2024-03-31, 2024-04-05, 2024-04-30, 2024-05-05.

Same-pass tiles are mosaicked deterministically with the first valid tile at overlaps; cloud minima are not cherry-picked across dates. Event-day scenes are excluded from pre/post selection. No catalogue cloud filter was applied to the annual query.

Evidence: [every scene's AOI statistics](risk/results/r2_scenes.csv), [acquisition audit](risk/results/r2_acquisitions.json), [raw STAC catalogue](risk/results/r2_catalog.json). The late post-image and early clear pre-images introduce seasonal/recovery confounding. Keep acquisition dates prominent in the eventual demo.

## R3: actual region counts and storage

The metadata checksum matches the publisher. 62,848 revisit rows deduplicate to 3,928 AOI IDs. Country assignment uses AOI centres and Natural Earth v5.1.2 1:10m country polygons; 35 centres are unmatched and zero ambiguous. Boundaries determine disputed/coastal assignments; counts are not jurisdictional claims. The 125 South Asia count uses Afghanistan, Bangladesh, Bhutan, India, Maldives, Nepal, Pakistan and Sri Lanka. Iran is reported separately: 22.

Full per-country counts, AOI assignments and exclusions: [country audit](risk/results/r3_aoi_countries.json). Metadata and boundary SHA256 hashes are in R3.

Processed HR + L2A archives require **68,306,275,318 bytes compressed**, before extraction. The full collection was not downloaded: only metadata, boundaries and licence were needed for this risk. HR terms: CC BY-NC 4.0; other dataset content: CC BY 4.0; code: BSD-3-Clause. Sample-pair band/radiometry/grid suitability remains an R5 prerequisite.

## R4: downloaded labels versus usable calibration data

The pinned author repository's Colombia GeoPackage was downloaded and inspected: 838 features in EPSG:4326, event 2019-12-24. Its notebook describes Sentinel-2 acquisition and rasterization, not an already packaged temporal raster archive.

The access-proof sample reconstructs a 128x128, 10m window near the largest inventory polygon. Actual dates: pre **2019-05-24**, post **2020-01-09**. Four spectral bands only (RGBN) plus SCL for validity. The pre/post, binary inventory label and valid mask are COGs on exactly the same CRS/grid/bounds. Valid overlap is **99.4385%**; **748** valid pixels intersect labelled landslides. [Sample manifest and source items](risk/results/r4_sample.json) contain hashes and candidate checks; COG files live in ignored data/risk-cache/labels/sample/.

This selected positive crop is not an unbiased validation set. Inventory negatives can be incomplete, the pre-date is seasonally distant, and labels at 10m do not provide independent 2.5m truth. Prepare multiple held-out event/AOI samples before calibrating k or reporting F1.

Both the official Landslide4Sense training share and download URL failed name resolution at cloud.iarai.ac.at after an initial attempt plus three retries. The retrieved official loader/README describe HDF5 image key img (128x128x14) and mask key mask. The subsequent IBM-NASA mirror check downloaded a genuine image_1.h5/mask_1.h5 training pair at pinned revision 4b291891badf301b5c75c2153f0f8fe00eeb1435. The image has shape 128x128x14; only zero-based channels [3,2,1,7] (RGBN) are read into the spectral sample. The binary mask contains 405 landslide pixels. SHA256 hashes and exact source URLs are recorded in R4. This is a redistribution and has not been byte-compared against the unavailable IARAI original. It confirms a sample download, not the full archive, georeferencing, temporal labels or independent 2.5m truth. Do not substitute these single-date segmentation labels for temporal pairs.

## R5: real-data Colab fine-tune smoke test

[Runner](risk/r5_finetune.py), [Colab notebook](notebooks/colab_finetune.ipynb), and [real-data contract](docs/r5_data_contract.md) are provided. Before training, the runner requires a WorldStrat manifest, >=3 distinct train AOIs, actual matching hashes, four named RGBN bands, exact x4 COG geometry, explicit finite radiometric scale/offset and valid crops. Source provenance and the published split still require inspection during data preparation.

The runner restarts from pretrained weights if CUDA OOM causes a tile-size retry, records all 50 losses, checks real parameter changes/gradients, saves safetensors and verifies a reload. The fixed-mask 64px limitation remains, so an OOM at 128 must currently be reported, not bypassed.

Three same-day publisher WorldStrat train AOIs passed source-member CRC/SHA checks, valid land-crop checks and the four-band/x4 COG manifest validation: Landcover-118968, Landcover-151915 and Landcover-1534788. The [preparation audit](risk/results/worldstrat_preparation.json) records the selected and rejected candidates, source URLs, crop positions, transformations and radiometric fits. WorldStrat SPOT high-resolution files lack a geotransform, so their bounds were reconstructed from published metadata and checked against the Sentinel-2 raster. HR DN values were empirically fitted per band to paired LR reflectance for this smoke test. These are not physically calibrated HR reflectances. The source grid is geographic with nominal 10m sampling and exact x4 subdivision, not a guaranteed metric 2.5m grid.

The Colab run used a Tesla T4, pinned torch 2.8.0+cu126 and the repository's full pinned requirements in an isolated virtual environment. Colab's notebook-only Matplotlib backend initially blocked import; setting `MPLBACKEND=Agg` for the subprocess resolved it. The successful run used a 128px LR tile, no OOM retry, and reported 50 finite losses. The mean of the first five losses was **0.0116685908**, compared with **0.0115941893** for the last five (a **0.64%** decrease). Trainable weights changed by L1 sum **34.2322**, finite nonzero gradients were seen, and reloaded checkpoint output matched. The checkpoint SHA256 reported by Colab was `dd284235ddfcc82ee61da09055f012782442a5360eee483c59251e5b6aa2b1aa`. [R5 result](risk/results/r5.json) contains every loss and the exact runtime/config/manifest hashes. It was transcribed from the visible Colab output after its browser download did not deliver a local file; the binary checkpoint remains in Colab's temporary runtime and is not part of this repository. This limits independent artifact verification but not the observed save/reload smoke-test result. [Live notebook](https://colab.research.google.com/drive/1F97r5UJKhszCpoKFGaEXxvCxw-HB_rYd) shows the run and the initial backend failure.

## Verification and phase gate

- 23 CPU tests passed in the isolated environment after this report update (latest `pytest -q` exit code 0).
- pip check: no broken requirements.
- Real R4 COG/grid inspection passed.
- The pinned Colab environment installed and passed `pip check`; the real-data R5 run completed on T4 and reloaded its saved checkpoint.
- Independent review was attempted but the reviewer could not run because of its usage limit. Local source/code review found and corrected the Conv3XC gradient-path issue; there is no claim of completed independent review.
- Package/source pins and runtime: [requirements](requirements-phase0.txt), [Windows lock](requirements-lock-windows.txt), [environment evidence](risk/results/environment.json).

**Stop here.** Before requesting Phase 1: resolve the fixed-mask tile policy, use the accessible Landslide4Sense mirror and prepare held-out calibration/evaluation data, and accept or revise the delayed Wayanad imagery claim. The R5 smoke test is complete, but none of these open points may be treated as passed.

## Source-selection audit

An earlier WorldStrat sample check retrieved three other train AOIs and found the HR geotransform/radiometry issue plus SCL cloud/water discrepancies despite metadata cloud values of zero. [Original sample evidence](risk/results/r5_source_samples.json) and [upstream source inspection](risk/results/worldstrat_preparation_sources.json) remain for traceability. Those candidates were not used for R5. The subsequent preparation audit selected three fully valid land crops and the Colab result above is the actual smoke-test evidence. Full source archive checksums were not verified; selected ZIP members had CRC and SHA256 checks.

## Post-Phase-0 implementation follow-up (2026-09-29)

The Phase 0 findings above are retained as executed; subsequent user instructions authorized implementation. A critical map check changed the event-site plan: the brief's point (11.782°N, 76.233°E) is **36.43 km** from the [NRSC/ISRO Chooralmala map point](https://bhuvan-app1.nrsc.gov.in/disaster/usrtasks/landslide/doc/Charter_1029_VAP_3_31july2024.pdf) (11.466763°N, 76.136272°E). R2's original December 6 date is only for the incorrect brief AOI. [Correction details](docs/location_correction.md) and the [new annual SCL audit](risk/results/event_aoi/r2.json) make this explicit. The corrected 10 km AOI had **73 acquisitions, nine clear pre-event dates and first clear post-event 2024-12-16**, **139 days** after the event. The selected demo inputs are **2024-03-01, 2024-03-06, 2024-04-10 and 2024-12-16**; every selected product's XML reported quantification **10,000** and BOA offset **−1,000**.

R1's original direct-size failure was resolved at the product inference layer. The pinned pretrained CNN accepts 64px input. TrustSR tiles larger inputs and applies an explicit parent-mean projection after the CNN and again after feathering, replacing the fixed-size upstream Fourier hard constraint for inference. Under a **4 GiB PyTorch allocator cap** on the **6,140.5 MiB RTX 4050**, synthetic 64/128/512px inputs all completed with 256/512/2048px outputs. [GPU measurements](results/gpu_tiles.json) show peak allocated **8.13/21.98/21.98 MiB** and wall times **0.514/0.103/1.882 s** respectively. This is an allocator-cap check, **not** a physical 4 GB GPU validation. CPU tests cover exact x4 geometry, near-exact spectral consistency, tiled deterministic equivalence and CUDA OOM halving/failure at 64px.

The Colab R5 checkpoint remains untransferred. A **separate local 50-step run** on the same three verified WorldStrat training AOIs saved the [bundled checkpoint](checkpoints/r5-smoke.safetensors), SHA256 `bbbdad6bda5dfadfc4099dc08dbf54ecacd9fe31e1640b85ff6f6cac75e12709`. Its first/last five-step loss means were **0.0116682859/0.0115935471**, with changed weights and finite gradients. [Local record](results/local_finetune.json). This is only a smoke fine-tune; no held-out generalization claim follows.

The corrected event-site 128×128 input tile was processed with eight dihedral runs per date and saved as aligned 2.5m/10m COGs in [offline demo assets](demo/assets). Its real class counts are **15,229 OBSERVED-support**, **99 INFERRED-position**, **15,748 UNSUPPORTED (rejected)** and **15,584 NO_DATA** 2.5m pixels. The pre/post parent-grid spectral RMSEs are **1.00×10⁻⁸ / 6.11×10⁻⁹ reflectance**. [Evaluation table](results/REPORT.md) and [machine-readable metrics](results/metrics.json) mark held-out WorldStrat PSNR/SSIM, landslide F1 and k calibration unavailable. The three PSNR/SSIM values in metrics.json are training-pair diagnostics only. The single Phase 0 Colombia crop is insufficient for independent calibration; `scripts/calibrate_k.py` refuses it and leaves k=2.0. No damage or building/road impact count is claimed from the demo.

The [Sen12Landslides repository](https://github.com/PaulH97/Sen12Landslides) was inspected as a possible independent temporal-label source. Its published structure includes multi-date Sentinel-2 patches, masks and splits; no patch or held-out label was downloaded or scored. The first harmonized Sentinel-2 archive in its dataset repository is **1,474,236,626 bytes**, so this is a future validation path, not a passed result.

## Held-out validation follow-up (2026-09-29)

The separate publisher WorldStrat **validation** split supplied three AOIs that do not overlap the three fine-tuning AOIs. Source ZIP members passed CRC/SHA checks, the prepared four-band pairs passed COG/grid/hash checks, and the local fine-tuned model yielded mean **33.5289 dB PSNR**, **0.8577 SSIM** and **8.18×10⁻⁹** parent-grid spectral RMSE. [Per-AOI metrics](results/metrics.json) and [preparation audit](risk/results/worldstrat_val_candidates.json) give the underlying data. WorldStrat HR DN was fitted per band to its paired LR reflectance, so these numbers measure spatial fidelity under empirical harmonization, not independent physical reflectance accuracy.

The [Sen12Landslides harmonized S12LS-LD test split](https://github.com/PaulH97/Sen12Landslides) supplied real four-band 10 m before/after imagery, SCL, event dates and 10 m landslide masks. The preparation script streamed selected members from three publisher archives and hashed the extracted `.nc` files. Six usable patches from Chimanimani and Italy formed the calibration set. Seven Kyrgyzstan patches were inspected; five with multiple event dates sharing one mask were excluded, leaving **two** independent evaluation patches. Clear bracketing dates were selected from each patch's 15 acquisitions with at most 10% invalid SCL pixels. [Manifest](results/sen12_manifest.json) and [evaluation record](results/sen12_evaluation.json) contain files, hashes, dates and exclusions.

Every tested k from **0.5 through 4.0 tied** at calibration F1 **0.09289**, so the threshold is not identified. The configured default **k=2.0** was retained by an explicit tie policy. On the held-out Kyrgyzstan event, all predictions were brought to the only available **10 m mask grid**: 10 m F1 **0.02352** (TP 27, FP 2,144, FN 98), ungated SR F1 **0.01464** (TP 119, FP 16,013, FN 6), and trust-gated F1 **0.02352**. The gate rejected **177,864** 2.5 m UNSUPPORTED pixels across the two patches. The trust result matches the 10 m gate at 10 m by construction; this is a safety filter, not a demonstrated accuracy gain. The sample was deliberately positive-patch-enriched and small, with no independent 2.5 m boundary truth. These low scores are evidence **against** presenting TrustSR as an accurate operational landslide detector or claiming Wayanad damage counts.

The 4 GiB allocator-cap check remains a proxy on a 6 GB card; no physical 4 GB GPU was available for this run. The Wayanad post-image remains 139 days late and has no independent matching landslide mask. The demonstration is functional, but those limitations prevent a claim that the disaster-detection objective has been scientifically validated.
