# TrustSR Phase 0 execution report

Updated 2026-09-26. **Phase 0 gate NOT PASSED. Phase 1 has not started.**
The five scripts are implemented and have been invoked. R1/R2/R3 and the accessible part of R4 produced real measurements; R5 stopped at its prerequisites with zero training steps. A blocked experiment is not a pass.

## Results

| Risk | Outcome | Observed result | Implication |
| --- | --- | --- | --- |
| R1 model fits | FAIL overall; native 128 input passes | RTX 4050 Laptop, 6140.5 MiB; FP32 128x128 -> 512x512: median **0.1208 s/tile**, peak allocated **100.48 MiB**, peak reserved **128 MiB**. Direct 512 and 64 input tiles fail fixed-mask shape checks | Native 128 inference is viable; arbitrary tile sizes and required 64px OOM fallback need a justified adapter before the production SR component |
| R2 imagery | PASS for configured availability test | **147 STAC items, 73 acquisition groups**, all SCL reads succeeded. **12 distinct clear pre-event dates**. First clear post-event date: **2024-12-06**, **0.0664%** AOI cloud/shadow | Clear imagery exists, but that post-date is **129 days after the event**; this selection cannot support an immediate-response claim |
| R3 paired-data resource | PASS for metadata/size/licence audit | **3,928 unique AOI IDs**, **71 India**, **125 South Asia**, **107,036,250,648 bytes** across record files | Regional coverage exists; these counts do not prove usable cloud-free/co-registered training pairs |
| R4 labels | PARTIAL / BLOCKED overall | Author inventory: **838 polygons**. Real Colombia pre/post crop downloaded with **748 valid labelled landslide pixels**. Original Landslide4Sense host fails DNS after retries | Temporal-labelled data can be reconstructed from the paper's release, but the full calibration split and Landslide4Sense content are not verified |
| R5 fine-tune | BLOCKED; **0 steps run** | Prepared WorldStrat pair manifest absent; current runtime is native Windows, not Colab | Colab notebook/runner ready for real prepared pairs; no loss-decrease or saved trained-checkpoint claim |

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

Both the official Landslide4Sense training share and download URL failed name resolution at cloud.iarai.ac.at after an initial attempt plus three retries. The retrieved official loader/README describe HDF5 image key img (128x128x14) and mask key mask. No real Landslide4Sense archive/mask was obtained. The script can inspect a successfully retrieved ZIP, but its synthetic ZIP unit test is not data-download evidence. Do not claim Landslide4Sense access or substitute its single-date segmentation labels for temporal pairs.

## R5: runnable, but not executed scientifically

[Runner](risk/r5_finetune.py), [Colab notebook](notebooks/colab_finetune.ipynb), and [real-data contract](docs/r5_data_contract.md) are provided. Before training, the runner requires a WorldStrat manifest, >=3 distinct train AOIs, actual matching hashes, four named RGBN bands, exact x4 COG geometry, explicit finite radiometric scale/offset and valid crops. Source provenance and the published split still require inspection during data preparation.

The runner restarts from pretrained weights if CUDA OOM causes a tile-size retry, records all 50 losses, checks real parameter changes/gradients, saves safetensors and verifies a reload. The fixed-mask 64px limitation remains, so an OOM at 128 must currently be reported, not bypassed.

Current evidence is **BLOCKED with zero optimizer steps** because no prepared real pairs or Colab runtime are present. The CPU optimizer unit test runs 50 steps on a tiny artificial network only to validate software mechanics.

## Verification and phase gate

- 18 CPU tests passed in approximately 10 seconds in the isolated environment.
- pip check: no broken requirements.
- Real R4 COG/grid inspection passed.
- Colab notebook code cells compile; notebook was not executed in Colab.
- Independent review was attempted but the reviewer could not run because of its usage limit. Local source/code review found and corrected the Conv3XC gradient-path issue; there is no claim of completed independent review.
- Package/source pins and runtime: [requirements](requirements-phase0.txt), [Windows lock](requirements-lock-windows.txt), [environment evidence](risk/results/environment.json).

**Stop here.** Before requesting Phase 1: resolve the fixed-mask tile policy, obtain/inspect Landslide4Sense or explicitly revise that requirement, prepare real WorldStrat pairs and complete the Colab smoke test, and accept or revise the delayed Wayanad imagery claim. No phase can treat an unresolved entry as passed.
