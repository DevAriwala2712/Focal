# TrustSR implementation plan

Goal: deliver a reproducible four-band Sentinel-2 to 2.5 m pipeline and a clearly qualified Wayanad demonstration.

Architecture: use the verified SEN2SR-lite loader and real WorldStrat pairs. Keep date selection and geospatial alignment separate from tiled inference, uncertainty, classification, evaluation, and presentation. All scientific claims are backed by saved provenance or marked unavailable.

Stack: Python 3.13, PyTorch 2.8, rasterio 1.4, NumPy 2.2, Streamlit, pytest. Spec: [AGENTS.md](../../../AGENTS.md), [design.md](../../design.md), [RISK_REPORT.md](../../../RISK_REPORT.md).

## Tasks

1. Resolve R1 fixed-size constraint and produce a local fine-tuned checkpoint from the verified three WorldStrat pairs. Add tests for exact parent-pixel spectral consistency, tile fallback, and seam-free deterministic stitching. Run CPU tests and real GPU probe. Commit SR component.
2. Implement AOI date selection, four-band and SCL retrieval, radiometry, full-grid reprojection, COG stack, and manifest. Test geometry, masks, failure messages, and retry behavior. Run a small real Wayanad crop. Commit fetch component.
3. Implement eight dihedral runs, inverse transforms, NDVI per run, and pre/post moment pooling. Test identity and hand-computed moments. Commit trust component.
4. Implement 10 m parent support, five classes, masks, and unsupported counts. Test a known square, artefact, and cloud. Commit change component.
5. Implement held-out SR and downstream evaluation with explicit unavailable metrics and calibration leakage controls. Test metric math and missing-label behavior. Commit evaluate component.
6. Build Streamlit offline demo and one-tile live path. Verify launch and real precomputed outputs, document delayed post-event date and reproduction, then push branch.

Review focus: exact CRS/bounds/affine, only B02/B03/B04/B08, 4 GiB target, truthful uncertainty and dataset provenance. The December 2024 clear image makes this a delayed comparison, not an immediate response result. A Colab smoke checkpoint was saved but could not be transferred, so local fine-tuning is a distinct recorded run.
