# TrustSR source register

## Execution update (2026-09-26)

The sections below preserve the initial source audit; later verified results are in [RISK_REPORT.md](../RISK_REPORT.md). WorldStrat has measured country-centre counts: 71 India, 125 across the defined eight South Asian countries. R2 queried the live Planetary Computer catalogue and read SCL for all 147 items; first accepted post-event date is 2024-12-06. The arXiv author's Colombia inventory and one real pre/post crop were downloaded. Both original Landslide4Sense training URLs fail DNS resolution locally; a pinned IBM-NASA mirror supplied one genuine sample. R5 completed 50 real-data steps in Colab on 2026-09-29.

Additional inspected implementation sources:
- [SEN2SR Conv3XC/CNNSR](https://github.com/ESAOpenSR/SEN2SR/blob/8e21bb669bcc6e8eb953a1fb24dfc5bb59dc18a4/sen2sr/models/opensr_baseline/cnn.py): train_mode is separate from PyTorch's train/eval flag; the risk loader explicitly enables its differentiable branch for fine-tuning.
- [Hard constraint](https://github.com/ESAOpenSR/SEN2SR/blob/8e21bb669bcc6e8eb953a1fb24dfc5bb59dc18a4/sen2sr/models/tricks.py): Fourier mask dimensions explain the observed non-128 input failures.
- [Landslide4Sense HDF5 loader](https://github.com/iarai/Landslide4Sense-2022/blob/main/dataset/landslide_dataset.py): confirms img/mask keys and image channel-last layout; archive content is still unavailable from the original host.
- [Natural Earth v5.1.2 country polygons](https://raw.githubusercontent.com/nvkelso/natural-earth-vector/v5.1.2/geojson/ne_10m_admin_0_countries.geojson): downloaded and hashed for AOI-centre classification; disputed/coastal assignments follow this dataset.
- [PyTorch official version instructions](https://pytorch.org/get-started/previous-versions/): CUDA 12.6 builds of torch 2.8.0 and torchvision 0.23.0 installed and tested in the project environment.

Hugging Face model revision is now pinned to 469e9b3a049f9320ba66b3a39b9655dbd31480b3; actual loader/weight hashes are in configs/phase0.yaml and risk/results/r1.json. The earlier unpinned/undownloaded statements below describe the initial planning pass only.

Checked 2026-09-26. “Accessible” means the page/source was retrieved, not that its data, weights or executable examples have passed a test. Links here are the reference list for [design](design.md) and [risk status](../RISK_REPORT.md).

## NTRO and project brief

- [Full local problem statement](problem_statement.md): exact copy of ../problem_statement.md supplied in the workspace. Official SIH problem ID and canonical statement URL are not yet supplied/verified; do not claim independent authentication.
- [Brief's video](https://www.youtube.com/watch?v=cQoHSStTEdM) and [brief's data browser](https://browser.dataspace.copernicus.eu): preserved from the supplied statement; video contents were not reviewed.
- [Project instructions](../AGENTS.md): scope, AOI centre, phase gate and tests. Hardware description updated to the measured RTX 4050; 4 GB compatibility requirement retained.

## SEN2SR

- [Repository and README](https://github.com/ESAOpenSR/SEN2SR), inspected GitHub revision [8e21bb669bcc6e8eb953a1fb24dfc5bb59dc18a4](https://github.com/ESAOpenSR/SEN2SR/tree/8e21bb669bcc6e8eb953a1fb24dfc5bb59dc18a4).
- [Four-band model metadata](https://huggingface.co/tacofoundation/sen2sr/resolve/main/SEN2SRLite/NonReference_RGBN_x4/mlm.json): RGBN order B04/B03/B02/B08; ×4; declared 472,496 parameters. Parameter count is not VRAM consumption.
- [Model loader source](https://huggingface.co/tacofoundation/SEN2SR/resolve/main/SEN2SRLite/NonReference_RGBN_x4/load.py): inspected separate trainable_model and compiled_model functions. The inference loader disables SR weight gradients.
- [Wrapper source](https://github.com/ESAOpenSR/SEN2SR/blob/8e21bb669bcc6e8eb953a1fb24dfc5bb59dc18a4/sen2sr/nonreference.py): SR, nonnegative clamp, then frozen hard constraint.
- [Package requirements](https://github.com/ESAOpenSR/SEN2SR/blob/8e21bb669bcc6e8eb953a1fb24dfc5bb59dc18a4/pyproject.toml): source version 0.8.5, Python ≥3.10,<4.0, NumPy ≥2.0.2. These are upstream declarations, not a tested project lock.
- [Lite Colab notebook](https://colab.research.google.com/drive/1x65GoI5hOfgX61LhtbATSBm7HySUHSw9?usp=sharing): official README link; notebook execution not tested.
- [Weights](https://huggingface.co/tacofoundation/SEN2SR/resolve/main/SEN2SRLite/NonReference_RGBN_x4/model.safetensor) and [constraint weights](https://huggingface.co/tacofoundation/SEN2SR/resolve/main/SEN2SRLite/NonReference_RGBN_x4/hard_constraint.safetensor): URLs verified in metadata; binary download/hash not yet verified.

Snapshots of the inspected model metadata, loader and wrapper are under evidence/ as text for review. Pin the Hugging Face revision and hash downloaded weights before running remote loader code. README badge and repository license detection differ; inspect actual license files/model terms before redistribution rather than assuming the badge settles it.

## WorldStrat paired SR data

- [Author repository](https://github.com/worldstrat/worldstrat)
- [Zenodo v1, DOI 10.5281/zenodo.6810792](https://zenodo.org/records/6810792)
- [Paper and datasheet](https://arxiv.org/abs/2207.06418)
- [Metadata download](https://zenodo.org/records/6810792/files/metadata.csv?download=1) — downloaded to evidence/worldstrat-metadata.csv.
- [Published split](https://zenodo.org/records/6810792/files/stratified_train_val_test_split.csv?download=1)
- [Processed HR archive](https://zenodo.org/records/6810792/files/hr_dataset.tar.gz?download=1)
- [L2A archive](https://zenodo.org/records/6810792/files/lr_dataset_l2a.tar.gz?download=1)
- [Licence file](https://zenodo.org/records/6810792/files/LICENSE.txt?download=1)
- [Kaggle mirror linked by authors](https://www.kaggle.com/datasets/jucor1/worldstrat) — not download-tested.

Zenodo lists **107.0 GB** total files: processed HR 41.5 GB, raw HR 11.3 GB, L1C 27.4 GB, L2A 26.8 GB, plus metadata/docs. For this project processed HR + L2A is approximately **68.3 GB compressed**, before extraction; this is a calculated planning subtotal, not disk usage measured here. Do not download L1C unnecessarily.

The dataset pairs high-resolution Airbus/SPOT imagery with multiple Sentinel-2 revisits. HR is distributed under CC BY-NC 4.0; other dataset content is CC BY 4.0; code is BSD-3-Clause. Preserve attribution and inspect terms before reuse/distribution. This is a report of publisher terms, not legal clearance.

India/South Asia AOI counts are **not yet computed**. The downloaded CSV has 62,848 rows and 3,928 distinct first-column AOI identifiers; its MD5 matches the published dfeb3348e79b719bf03c230d5d258839. See [metadata audit](evidence/worldstrat-metadata-audit.json). Metadata has coordinates and repeated revisits, so country counting needs AOI deduplication and versioned country boundaries. Do not count metadata rows as distinct places or confuse global coverage with usable cloud-free regional pairs.

## Landslide calibration and segmentation labels

- [Monopoli, Montello and Rossi, arXiv:2405.20161](https://arxiv.org/abs/2405.20161)
- [Read paper v1](https://arxiv.org/html/2405.20161v1)
- [Author code and inventories](https://github.com/links-ads/igarss-landslide-delineation)
- [Image acquisition notebooks](https://github.com/links-ads/igarss-landslide-delineation/tree/main/processing)

The paper frames detection as newly occurring landslides from before/after Sentinel-2 images. The repository instructs users to obtain imagery via processing notebooks; an immediately downloadable packaged pre/post raster dataset has not been verified. Inventory access and end-to-end image acquisition are separate checks. Labels are landslide inventory/segmentation labels, not verified building damage or 2.5 m boundary truth. The paper's use of DEM does not authorize extra inputs for TrustSR.

- [Official Landslide4Sense-2022 repository](https://github.com/iarai/Landslide4Sense-2022)
- [Training data share](https://cloud.iarai.ac.at/index.php/s/KrwKngeXN7KjkFm)
- [Validation data share](https://cloud.iarai.ac.at/index.php/s/N6TacGsfr5nRNWr)

The official repository describes 3,799 training, 245 validation and 800 test patches, 128×128 pixels, approximately 10 m. Each image has 12 Sentinel-2 bands plus slope and DEM; only B02/B03/B04/B08 are eligible here. Training masks are pixel-wise landslide/background labels. The original competition validation images were supplied without labels. No ready-made before/after pairs are established. The training-share retrieval returned a tool INVALID_ARGUMENT error; archive accessibility remains unverified. The initially guessed /iarai/Landslide4Sense URL returned 404; use the -2022 repository above.

## Accuracy assessment

- [OpenSR-test repository](https://github.com/ESAOpenSR/opensr-test)
- [Documentation](https://esaopensr.github.io/opensr-test)
- [Python distribution](https://pypi.org/project/opensr-test/)

README inspected: evaluates consistency, synthesis and correctness; documents opensr_test.Metrics().compute(lr=..., sr=..., hr=...). Installation was not attempted. Keep held-out PSNR/SSIM and downsample consistency as required metrics even if this optional package fails.

## Imagery access and quality

- [Microsoft Planetary Computer Sentinel-2 L2A](https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a) — retrieved; provides L2A imagery as COGs.
- [STAC collection](https://planetarycomputer.microsoft.com/api/stac/v1/collections/sentinel-2-l2a)
- [STAC API root](https://planetarycomputer.microsoft.com/api/stac/v1)
- [Copernicus Data Space browser](https://browser.dataspace.copernicus.eu)
- [Copernicus STAC documentation](https://documentation.dataspace.copernicus.eu/APIs/STAC.html) — retrieval tool returned INVALID_ARGUMENT in this pass; endpoint details not verified.

R2 must query the actual 2024 scenes and read SCL over the AOI. None of these landing pages proves a clear post-event acquisition. Asset signing, access limits, nodata and scale/offset need validation on real assets.

## Wayanad showcase and external impact reference

- AOI centre **11.782°N, 76.233°E**, approximately 10×10 km, comes from the project brief. It is a study centre, not a surveyed landslide boundary.
- Event date **30 July 2024** is supported by the [published Wayanad damage assessment](https://dataforpublicgood.org.in/blog/landslide-damage-assessment-wayanad-kerala/), published 8 April 2025.
- That assessment reports **507 buildings** and **8.38 km of roads** affected. It delineates an affected area using pre/post **Sentinel-1** and intersects **OpenStreetMap** amenities.

Use those figures as attributed contextual reference totals. They are not Sentinel-2/SR performance metrics, independently verified destruction labels, or a target to tune TrustSR toward. Any quantitative comparison needs the same footprint, OSM snapshot, geometry inclusion rules and units; the page does not supply enough verified inputs to reproduce those totals in this pass.

## Evidence levels

Verified: supplied statement copy, local environment, primary repository/record contents and small WorldStrat metadata download.
Not yet verified: GPU inference/fine-tuning, large data/weight downloads, regional AOI counts, clear Wayanad scene dates, Colab execution, evaluation scores and finale date.

### Phase 0 continuation: verified alternate access

- Landslide4Sense redistribution: https://huggingface.co/datasets/ibm-nasa-geospatial/Landslide4sense at revision 4b291891badf301b5c75c2153f0f8fe00eeb1435. Actual training image/mask downloaded and inspected; see R4 hashes. Original IARAI host still fails DNS. Single-date binary segmentation, not temporal damage labels.
- Publisher WorldStrat ZIP release: https://zenodo.org/records/15382551. HTTP byte ranges supported in the HR ZIP probe. This release is separate from the earlier R3 tarball inventory; do not mix archive sizes or metadata hashes.
