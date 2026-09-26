# R5 real-data input contract

R5 has not run on WorldStrat in Colab. Its executable runner deliberately stops if the prepared pair manifest or Colab GPU is missing. A synthetic CPU unit test exercises the optimizer, but is not a substitute for the scientific smoke test.

## Required preparation

1. Acquire at least three distinct **training** AOIs from WorldStrat's published split. Record the source archive URLs, original member names, source dates, original CRS, product type and checksums. Use L2A low-resolution imagery. Do not use test/validation AOIs for this smoke test.
2. Inspect the native HR product's band order, radiometry and NIR provenance. WorldStrat's 1.5 m grid is not the requested 2.5 m target grid. Resample a genuine four-band HR reference onto the exact x4 subdivision of the LR affine grid, preserving bounds and CRS. Document the resampling operation; do not replace the HR target with an upsampled LR image.
3. Select a cloud-free, no-data-free crop of at least 128x128 LR pixels and its matching 512x512 HR extent. The runner reads the upper-left crop, so prepare the verified crop at that location. Check both acquisitions for temporal differences. A future preprocessing component should automate these checks; no such component is claimed in Phase 0.
4. Write both prepared rasters as COGs with exactly four band descriptions **B04, B03, B02, B08** in that order. Retain the true reflectance encoding and record scale/offset for each product. If reflectance is already float32, use scale 1 and offset 0 only after verifying it.
5. Put the prepared files and `pairs.json` in the path configured by `training.pair_manifest`. Record the prepared files' SHA256 hashes. Preserve any cloud/invalid mask as COG nodata; R5 rejects masked crops.

## Manifest schema

The following describes fields; it is not a usable or fabricated dataset manifest. `pairs` must contain at least `training.min_pairs` distinct AOI IDs.

| Field | Required value/meaning |
| --- | --- |
| dataset | WorldStrat |
| pairs[].aoi_id | Actual identifier in the published metadata/split |
| pairs[].split | train, verified against the published split |
| pairs[].lr / hr | Relative paths from the manifest directory to prepared COGs |
| pairs[].lr_sha256 / hr_sha256 | Actual prepared-file hashes |
| pairs[].source_urls | Nonempty array of genuine publisher source URLs |
| pairs[].lr_scale / hr_scale | Finite positive multipliers into comparable reflectance units |
| pairs[].lr_offset / hr_offset | Finite additive offsets after multiplication |

Also retain original member names, original hashes, acquisition dates and preparation notes in each entry. The runner checks hashes, four-band names, raster CRS/bounds, x4 affine grid, COG layout, nonzero positive scale, finite values and no-data. A user-written dataset name or URL is not independent proof of provenance; inspect originals before trusting a manifest.

## Colab execution and evidence

Use `notebooks/colab_finetune.ipynb`, upload the project and these prepared pairs, select a GPU runtime, install pinned requirements, and run the notebook. The runner starts from pinned pretrained weights, enables the upstream Conv3XC training branch, retains the frozen hard constraint, uses Adam/L1 for the configured 50 steps, and saves/reloads safetensors.

Required evidence: `risk/results/r5.json`, all 50 losses, first/last five-step means, nonzero finite gradients, changed weights, checkpoint hash and reload agreement, GPU/runtime details, pair manifest hash and AOI IDs. A lower training loss only validates the smoke test; it says nothing about held-out SR fidelity or landslide accuracy.

Known limitation: upstream hard-constraint weights have a fixed 512x512 output shape. The OOM handler halves tile size, but 64px inputs currently fail shape validation. Do not silently resize or remove the constraint to force a pass; this requires a separately justified adapter and new measurements before Phase 1.

## Verified source issues from selective retrieval

See risk/results/r5_source_samples.json and worldstrat_preparation_sources.json. Three genuine training AOIs from publisher release 15382551 are cached under data/risk-cache/worldstrat-zip/originals. HR four-band files omit geotransforms; source code exports SPOT DN. LR bounds match metadata, but LR grids are EPSG:4326 with nominal rather than exact metric 10m sampling. Original files must remain unchanged; any reconstructed bounds or empirical radiometric fit must be explicit in prepared-file provenance. The current samples include an all-water AOI and cloud/shadow pixels despite zero metadata cloud cover, so they are not automatically approved training pairs.

## Implemented preparation path

Run `python -m risk.worldstrat_prepare` using configs/phase0.yaml. It selectively downloads publisher ZIP members, verifies CRC32 and records SHA256, confirms train membership and dates, selects a fully valid 128px land crop using SCL/dataMask, reconstructs the documented HR bounds, and writes four-band COGs. Full-archive checksums are not verified by selective retrieval. Empirical per-band positive affine fits harmonize area-averaged HR DN to the paired LR values, with configured correlation rejection; these are smoke-only targets, not physically calibrated HR reflectance. The manifest stores all coefficients, source hashes and reconstruction notes. Input geographic CRS is preserved, with an exact x4 affine subdivision and nominal ~10m/~2.5m sampling.

The previous requirement to record true reflectance encoding remains applicable to physically calibrated products; this explicit empirical path records that no such HR calibration is claimed. No prepared target is an upsampled LR surrogate. Holdout evaluation must be separately designed before accepting smoke-test weights for the demo.
