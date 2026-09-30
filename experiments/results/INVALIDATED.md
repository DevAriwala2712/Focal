# INVALIDATED — do not cite these results

This file quarantines results on `experiments/exceptional` that do not survive re-audit under
`experiments/fix-v2` (see the orchestration prompt for B1–B16). Original result files are left
byte-for-byte unchanged; do not edit or delete them. Machine-readable version:
`experiments/results/invalidated.json`.

| File | Status | Invalidated by | Reason (short) | Superseded by |
|---|---|---|---|---|
| `experiments/x3_placebo.py`, `experiments/results/x3.json`, `x3_REPORT.md` | INVALIDATED | B1, B2, B3, B4, B5 | Placebo pairs use the real post-event date (not a null test); 3 of 5 gates are the same function; window FAR is copy-pasted across gates; reported CI is a single fold's CI, not the pooled CI; FAR counts every non-zero class as flagged | `f1_placebo.py` → `f1.json` (pending) |
| `experiments/x4_calibration.py`, `experiments/results/x4.json`, `x4_REPORT.md` | INVALIDATED | B6, B7, B8 | Conformal scores are calibrated on the pre/post pair containing the real event; the block-sum "consistency" check compares a quantity with itself; `config_sha256` was `"pending-a3-output"` | `trustsr/gate_v2.py` (F2) → `f2` result (pending) |
| `experiments/x9_v2_production.py`, `experiments/results/x9.json`, `x9_REPORT.md` | INVALIDATED | B9, B10, B11, B12 | The "SR change score" is the 10 m difference nearest-neighbour-replicated (no SR signal enters allocation); τ_win is loaded but never passed to `apply_gate_v2` (no detection step — every f>0 block is ALLOCATED); NO_DATA only checks NaNs, ignoring the SCL validity mask; unmixing endmembers are hard-coded constants, not estimated from data | `f7_wayanad_v2.py` (F7) → `f7.json` (pending, depends on F2) |
| `experiments/results/x10.json`, `x10_REPORT.md` | INVALIDATED | B13 | Verifier (Haiku) skipped the required clean-clone rerun, missed B1–B12, and miscounted its own KEEP tally (states 3/5, lists 2) | `f9.json` (pending) |
| `experiments/results/x11.json`, `RESULTS_EXCEPTIONAL.md` | INVALIDATED | B13, B14 | Integrator marked overall CONDITIONAL PASS on top of an invalid A10 audit; omits A2's pooled result (−0.097 dB [−0.157, −0.037], 178 images) and the Venus result (−0.559 dB); its "11.8% placebo FAR" headline is B1–B5's invalid number | `RESULTS_V2.md` (F10, pending) |

## Sound and NOT touched by this quarantine (cite freely)
E1–E8 (`experiments/e1..e8*.py`, `experiments/results/e1..e8.json`); `experiments/wayanad_evidence/` (v1
diagnosis, cited verbatim in `configs/exceptional.yaml`); `experiments/x2_*.py` / `x2.json` /
`x2_REPORT.md` (A2 HR benchmark — see B14 for the one omission in how it was *reported*, not in the
experiment itself); `experiments/x5_noise_model.py` / `x5.json` / `x5_REPORT.md` (A5, keep rule missed
but honestly reported); `experiments/x6_season_latency.py` / `x6.json` / `x6_REPORT.md` (A6, honest
FAIL).

## Doc fixes tracked separately (B16)
`docs/design.md` L9, `docs/playbook_strategy.md` L21, and `README.md` still cite the wrong AOI
(11.782°N 76.233°E). Corrected centre: 11.490°N 76.160°E. Fixed in F10.
