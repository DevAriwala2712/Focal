# TrustSR evaluation status

The Wayanad tile is a delayed December 2024 comparison. Its NDVI drop is not a labelled landslide outcome.

| Measure | Result | Interpretation |
| --- | ---: | --- |
| Pre-event spectral RMSE (reflectance) | 0.00000001 | Parent-block consistency on real Wayanad inputs |
| Post-event spectral RMSE (reflectance) | 0.00000001 | Parent-block consistency on real Wayanad inputs |
| OBSERVED pixels | 15229 | 2.5 m positions are still model-inferred |
| INFERRED pixels | 99 | Changed 10 m parent, uncertain fine location |
| UNSUPPORTED pixels | 15748 | SR-only change rejected |
| Held-out WorldStrat PSNR/SSIM | 33.529 dB / 0.8577 | 3 publisher validation AOIs; per-pair radiometric fit limits interpretation |
| Event-held-out landslide F1 (10 m / raw SR / trust) | 0.0235 / 0.0146 / 0.0235 | 2 Sen12Landslides patches; 10 m labels; positive-enriched sample |
| Event-held-out UNSUPPORTED pixels | 177864 | Rejected fine pixels across tested patches |
| k calibration | 2.0 | All candidates tied; retained default, not identified |

Training-pair PSNR/SSIM are in metrics.json for optimizer diagnostics only.
WorldStrat HR DN required a per-pair fit to LR reflectance; held-out scores measure spatial fidelity under that fit, not physical radiometric accuracy.
The event-held-out landslide sample is small and enriched for positive patches. Its low F1 is evidence against a reliable operational detector; it is not a Wayanad accuracy estimate.
The earlier 507-building and 8.38-km-road figures are a published contextual baseline, not TrustSR counts.
