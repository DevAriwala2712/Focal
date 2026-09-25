# Phase 0 execution plan

Planning only, 2026-09-26. These are script specifications, not claims that scripts already exist or tests have run. See ../RISK_REPORT.md for current evidence. Product components remain gated.

| Script to write | Work | Required evidence and decision |
| --- | --- | --- |
| risk/r1_model_fits.py | Pin/review loader and weights; load RGBN lite; warm up; synchronize CUDA; benchmark batch-one 128 and independently 512 input tiles; cap allocator for a separate 4 GiB-budget run; test 64 fallback shape | Input/output shapes, precision, device, peak allocated/reserved/device memory, elapsed seconds, OOM/shape error, checkpoint hash. Never call a 6 GB run a physical 4 GB pass |
| risk/r2_imagery.py | STAC query all of 2024 over the metric 10 km AOI; paginate; deduplicate/mosaic same acquisitions; read SCL on full AOI; record missing-coverage and cloud/shadow fractions separately | CSV/JSON for every acquisition; threshold/classes and denominator; â‰¥3 clear pre-event dates; first clear post-event date or explicit none. Catalog-wide cloud % is not an AOI measurement |
| risk/r3_worldstrat.py | Query archive sizes; verify metadata checksum; deduplicate AOIs; spatially join AOI centres to a versioned country boundary dataset; inspect actual usable RGBN pairs | Total bytes and storage plan, India count and South Asia count with AOI IDs/boundary policy; licences; pair shapes/bands/dates. No bounding-box count presented as country count |
| risk/r4_labels.py | Download inventory labels and inspect processing notebooks; retrieve one actual pre/post sample and mask; check Landslide4Sense training archive and one labelled HDF5 pair | URL/status/checksum, label meaning/resolution/CRS and whether pre/post data actually exist; separately report landing-page access versus data download |
| risk/r5_finetune.py | Run in Colab on a few verified WorldStrat pairs using the trainable model, batch-one tiling and OOM reduction; 50 optimizer steps | Device/runtime, valid pair IDs, loss every step, first/last five-step means, finite nonzero gradients and parameter delta, saved/reloaded checkpoint and output agreement |

Define South Asia before counting: Afghanistan, Bangladesh, Bhutan, India, Maldives, Nepal, Pakistan and Sri Lanka (project operational definition). Report Iran separately if comparing with UN M49 Southern Asia. Deduplicate metadata revisits using the AOI identifier; assign by AOI centre and report ambiguous borders rather than guessing.

R2: define strict pre-event timestamps before 2024-07-30, and exclude event-day scenes unless acquisition timing proves post-event status. Use AOI validity coverage as well as cloud thresholds. Threshold selection must be recorded before selecting dates. SCL is allowed as quality metadata despite the spectral-band restriction.

R5: verify provider scale/offset, HR multispectral provenance and exact Ã—4 pair alignment first. Hold the validation AOIs apart. If Colab is unavailable, write/run the preparation checks and mark the requested Colab run BLOCKED; local results cannot be silently substituted.

Shared configuration belongs in configs/phase0.yaml when scripts are implemented. Use a separate tested Python environment; pin dependencies after resolution, run pip check, record freeze and CUDA wheel version. Preserve source IDs, access dates, hashes and logs in risk/results/. Retry failed downloads three times with backoff, then report the URL. Large archives should be downloaded only once with checksum verification and sufficient disk space.

Stop after updating RISK_REPORT.md with all five outcomes. A failed scientific prerequisite may require changing scope; it never becomes a pass because the demo deadline is close.
