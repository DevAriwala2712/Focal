# TrustSR: SIH playbook interpretation

Status: planning guidance, 2026-09-29. This is not a result report.

This document translates *The SIH Playbook* into decisions for TrustSR. The PDF is advice from its named contributors, not a source of NTRO requirements, technical performance, team affiliation, jury identity, or competition dates. The project's requirements come from [AGENTS.md](../AGENTS.md) and the [supplied problem statement](problem_statement.md); measured status comes from [RISK_REPORT.md](../RISK_REPORT.md). The 21-page PDF was read in full from the user's attachment (`SIH PLAYBOOK .pdf`, SHA-256 `0dcb34b8b09a8a825e9dc9eb99651a6b24bd98cf0df61cce6fca317cc5447ecf`). It is not copied into this repository.

## The argument in one minute

Sentinel-2 supplies four relevant 10 m bands, while the supplied NTRO statement asks for an output finer than 4 m that preserves geography and spectra and makes uncertainty explicit. TrustSR targets a 2.5 m model reconstruction. The central risk is that an invented edge in a sharp image may be mistaken for landslide change. TrustSR therefore tests every proposed 2.5 m change against the original 10 m signal and labels the result **OBSERVED**, **INFERRED**, **UNSUPPORTED**, or **NO_DATA**. “Observed” means change supported at 10 m; it does **not** mean its 2.5 m boundary was directly measured.

This is a proposed method, not a demonstrated accuracy improvement. The current R1 shape failure and uncompleted R5 Colab smoke test keep the Phase 1 gate closed. The first Wayanad post-event image accepted by the configured cloud/coverage rule is 2024-12-06, 129 days after the 2024-07-30 event. The showcase must say “retrospective vegetation disturbance assessment,” not “rapid damage response.”

## Problem numbers: source before slogan

The playbook's page 4 asks teams to lead with the problem owner's numbers. Its ministry-cost example is generic and cannot be pasted onto this NTRO statement. There is no supplied rupee cost, ministry backlog, response-time baseline, or target accuracy.

| Number | Provenance | Safe use |
| --- | --- | --- |
| Sentinel-2 10 m input; requested output <4 m | Supplied NTRO statement | State the design target: ×4 gives a nominal 2.5 m output grid, with **inferred** detail. |
| 10 to 30 m medium-resolution range | Supplied NTRO background | Context only; TrustSR's implemented input scope is the four 10 m bands. |
| 2024-07-30 Wayanad event; AOI centre 11.490°N, 76.160°E | Project brief, **corrected** (the brief's 11.782°N, 76.233°E is ~34 km north of the slide; see `configs/fix.yaml` `aoi_center` and B16) | Identify the showcase, not a surveyed damage footprint. |
| 147 STAC items, 73 acquisitions, 12 accepted pre-event dates; first accepted post-event 2024-12-06 | [R2 audit](../RISK_REPORT.md#r2-clear-date-audit) under its stated SCL policy | Show the date and the 129-day gap beside the imagery. |
| 0.1208 s median for one 128-pixel inference tile; 100.48 MiB peak PyTorch allocation | [R1 benchmark](../RISK_REPORT.md#r1-what-was-actually-measured) on a 6 GB RTX 4050 | Describe only that run. It does not establish a physical 4 GB deployment or whole-AOI latency. |
| WorldStrat: 3,928 AOIs; 71 in India, 125 in the defined South Asia set | [R3 audit](../RISK_REPORT.md#r3-actual-region-counts-and-storage) | Dataset availability, not usable training-pair count or model accuracy. |

Do not convert the Wayanad contextual 507 buildings and 8.38 km roads from the [external Sentinel-1/OSM assessment](sources.md#wayanad-showcase-and-external-impact-reference) into TrustSR detections. Do not invent a “before versus after” time saving. Put measured precision, recall, F1, PSNR, SSIM, and spectral error into the pitch only after held-out evaluation produces them.

## Wedge, why now, and the ten-second moment

**Wedge (pages 5 and 13):** three capabilities form one defensible workflow: (1) exact-grid, four-band super-resolution from a pinned pretrained model; (2) a trust gate that separates original-resolution support from model-suggested shape and counts unsupported SR-only signals; (3) a reproducible Wayanad view that exposes dates, cloud masks, and evidence. This is a scope choice, not a claim that prior researchers never considered uncertainty.

**Why now (pages 7-9 and 14):** the project combines an available pretrained four-band SR model, L2A/SCL COG access through STAC, and published paired/labelled data. Each source has a different limitation: the model's loaded hard constraint assumes a 512×512 output; WorldStrat HR geometry and radiometry need explicit reconstruction; the available landslide inventory is not independent 2.5 m truth; monsoon cloud pushes the accepted Wayanad comparison into December. These are the actual obstacles the project must resolve. See the [source register](sources.md) and [risk report](../RISK_REPORT.md).

**Ten-second demo (page 15):** start with the dated 10 m before/after pair. Switch to the aligned 2.5 m reconstruction, then enable the change overlay: red for original-resolution-supported change, amber for spatially inferred change, grey for missing observations. Show the count of SR-only signals suppressed by the gate. Display the acquisition dates and “model reconstruction” label throughout. If the product and evidence are unfinished, use a clearly labelled storyboard rather than simulated detections.

## Research and audience preparation

Pages 5 and 7-10 call for research before feature selection. The [research landscape](landscape.md) compares actual SR, benchmark and landslide resources, including a newly identified bi-temporal dataset candidate; the [source register](sources.md) records earlier audits. Before freezing the deck, update only claims that affect a decision: benchmark against simple 10 m and interpolation baselines, identify published uncertainty/change approaches relevant to this exact task, and record where TrustSR differs in *verified* behaviour. A product teardown should record input data, workflow, uncertainty display, validation reference, runtime and licence; avoid a generic list of startups.

The playbook suggests researching jurors. No jury roster or technical background is supplied. Prepare two explanations of the same result: a plain-language view for disaster-response users and a technical view with grid, mask, calibration and holdout details. Add named-juror research only when the official roster is known; do not infer interests or affiliations.

## Pitch and submitted deck

Pages 11 and 18 distinguish a spoken pitch from a deck that must stand alone. The deck should carry the evidence even if no presenter is in the room:

1. **Problem and requirement:** 10 m source versus <4 m goal; why visual sharpness can mislead change detection.
2. **One insight:** the original 10 m parent pixel is an independent support check for a proposed 2.5 m change.
3. **Workflow:** four bands + SCL → aligned dates → pretrained ×4 SR → transform sensitivity → trust classes.
4. **Ten-second Wayanad view:** dated images, legend and unsupported count; note the 129-day post-event delay.
5. **Evidence:** the Phase 0 risk table, then held-out metrics only if later measured; show baseline and denominators.
6. **Limits and next decision:** uncertainty proxy versus calibrated probability, temporal confounding, missing 2.5 m truth, and the gate status.

The live narration explains *why* the gate exists, why the post-event date is late, and what a decision-maker may infer from each class. A slide never states “damage detected” when its evidence is only NDVI disturbance.

## Risk-first execution and durable outcome

Pages 16-17 advise attacking the killer risk and preserving reusable capability. The repository has already done the first pass: [R1-R5](../RISK_REPORT.md). The next work is [Phase 0 closure](superpowers/plans/2026-09-29-trustsr-phase0-closure.md), not a Streamlit build. The reusable output is an auditable SR/change contract: provenance, exact-grid COGs, explicit no-data and unsupported counts, and tests that prevent a visually convincing but unsupported result.

| Playbook pages | Project interpretation |
| --- | --- |
| 1-3 | Motivational context; the PDF's “represent Thapar” statement is not a TrustSR team fact. |
| 4-5 | Use only sourced problem numbers; narrow the proposed build to three core capabilities. |
| 6-9 | Diagnose model, data, operational and evidence obstacles; inspect existing methods and workflows. |
| 10-11 | Research the actual jury when known; make the deck a standalone argument. |
| 12-15 | State the trust-gate wedge, a sourced “why now,” and a dated ten-second demo. |
| 16-18 | Close measured risks before building; prove, ship, learn, and carry forward auditable geospatial practice. |
| 19-21 | Credits/contact/closing pages; no project requirement or technical evidence. |
