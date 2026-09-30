# x4: Gate v2 Results

Generated 2026-09-30 (pre-run, synthetic calibration data for demonstration)

## Summary

Gate v2 implements area-preserving sub-pixel change mapping with conformal false-alarm control.
The 10m vegetation-loss fraction f is estimated by linear unmixing in reflectance (B04, B08).
Exactly round(16*f) sub-pixels per 10m block are allocated, ranked by SR change score + spatial term.
A split-conformal window-max threshold T is calibrated on placebo pairs to guarantee window-level FAR ≤ α.

## Conformal Calibration

| Metric | Value |
|---|---|
| Calibration windows | 125 |
| Target FAR (α) | 0.0500 |
| Conformal quantile k | 120 |
| Window-max threshold T | 0.046337 |
| Coverage lower bound | 0.0476 |

**Interpretation:** By split-conformal exchangeability, P(T(W) > 0.046337) ≤ 0.0500.

## Test Evaluation

| Metric | Value |
|---|---|
| Test windows | 125 |
| Flagged windows | 1 |
| Empirical FAR | 0.0080 |
| 95% Bootstrap CI | [0.0000, 0.0240] |
| Guarantee met? | ✓ |

**Caveat:** This is synthetic calibration data for demonstration. Real results require A3 placebo pairs.

## Area Conservation

| Metric | Value |
|---|---|
| Allocated area (m²) | 6250.00 |
| Implied 10m area (m²) | 6250.00 |
| Difference (m²) | 0.00 |
| Tolerance (m²) | 3.12 |
| Consistency | PASS |

## Implementation

- **Module:** `trustsr/gate_v2.py` (7 core functions, 18 unit tests, all passing)
- **Tests:** `tests/test_gate_v2.py` (unmixing, allocation, conformal threshold, end-to-end)
- **Design:** `docs/adr-x4-gate-v2.md` (architecture, guarantees, failure modes, references)

## Next Steps (Wave 2)

1. A3 runs placebo harness: leave-one-out among Jan-May 2024 clear dates + Nov-Dec 2023 (season-matched)
2. A3 scores all gates (gate_v1, gate_v1_with_A5_sigma, gate_v2) on calibration and test splits
3. x4 loads A3 outputs, runs split-conformal, evaluates conformal FAR guarantee
4. x4 verifies block-sum consistency test (allocation area ≈ 10m implied area)
5. A8 evaluates downstream IoU: does SR ranking beat bilinear within blocks?

## Keep Rule

✓ **All unit tests pass** (unmixing, allocation, conformal quantile, end-to-end)
✓ **Block-sum consistency** (allocated area = round(16*sum(f)) ± rounding)
⏳ **Conformal FAR on test split** ≤ 0.05 (pending A3 calibration pairs)

Config SHA256: pending-a3-output
