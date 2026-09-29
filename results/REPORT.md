# TrustSR evaluation status

The Wayanad tile is a delayed December 2024 comparison. Its NDVI drop is not a labelled landslide outcome.

| Measure | Result | Interpretation |
| --- | ---: | --- |
| Pre-event spectral RMSE (reflectance) | 0.00000001 | Parent-block consistency on real Wayanad inputs |
| Post-event spectral RMSE (reflectance) | 0.00000001 | Parent-block consistency on real Wayanad inputs |
| OBSERVED pixels | 15229 | 2.5 m positions are still model-inferred |
| INFERRED pixels | 99 | Changed 10 m parent, uncertain fine location |
| UNSUPPORTED pixels | 15748 | SR-only change rejected |
| Held-out WorldStrat PSNR/SSIM | unavailable | Prepared pairs are all train AOIs |
| Landslide F1 (10 m / 2.5 m / trust) | unavailable | No suitable held-out temporal label set |

Training-pair PSNR/SSIM are in metrics.json for optimizer diagnostics only; they are not generalization evidence.
The earlier 507-building and 8.38-km-road figures are a published contextual baseline, not TrustSR counts.
