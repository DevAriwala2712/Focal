# Requests for Agent 1

- **What:** Score ungated SR with the F3 σ in the F1 harness.
  **Why:** So the headline placebo comparison is against the strongest honest baseline.
  **Which file:** `experiments/f1.py` or equivalent test harness configuration.

- **What:** Fix the stale hashes in `f3_REPORT.md` and `f6_REPORT.md`.
  **Why:** They must reflect the actual correct state so that audits pass without error.
  **Which file:** `experiments/results/f3_REPORT.md`, `experiments/results/f6_REPORT.md`.

- **What:** Export the 2.5 m SR RGB, σ and class rasters as small COGs for the demo crop if the cache is not shareable.
  **Why:** Required to serve the offline demo effectively.
  **Which file:** `demo/export_assets.py` will read them, but they need to be produced in `experiments/wayanad_evidence/outputs/` or similar.
