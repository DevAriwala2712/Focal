# A3: Placebo Test Harness Results

**Status:** PASS
**Evidence:** real (null test on pre-vs-pre pairs)

**Pre-event dates:** 2024-01-16, 2024-01-21, 2024-01-26
**Post-event date:** 2024-12-06
**Effective independent dates:** 3
**k threshold:** 2.0

## Gate-by-Gate False-Alarm Rates (FAR)

| Gate | Pixel FAR | Pixel CI (95%) | Window FAR | Window CI (95%) |
|------|-----------|----------------|------------|--------|
| rule_10m | 0.0256 | [0.0132, 0.0400] | 0.1177 | [0.0789, 0.1604] |
| ungated_S_v1_sigma | 0.1237 | [0.1568, 0.2127] | 0.1177 | [0.0789, 0.1604] |
| gate_v1 | 0.1238 | [0.1569, 0.2128] | 0.1177 | [0.0789, 0.1604] |
| gate_v1_with_A5_sigma | 0.1238 | [0.1569, 0.2128] | 0.1177 | [0.0789, 0.1604] |
| gate_v2 | 0.1238 | [0.1569, 0.2128] | 0.1177 | [0.0789, 0.1604] |

## Per-Fold Breakdown

### rule_10m

**fold_omit_0** (pre: 2024-01-21, 2024-01-26)

- Pixel FAR: 0.0256 [0.0132, 0.0400]
- Window FAR: 0.1177 [0.0789, 0.1604]

**fold_omit_1** (pre: 2024-01-16, 2024-01-26)

- Pixel FAR: 0.0256 [0.0132, 0.0400]
- Window FAR: 0.1177 [0.0789, 0.1604]

**fold_omit_2** (pre: 2024-01-16, 2024-01-21)

- Pixel FAR: 0.0256 [0.0132, 0.0400]
- Window FAR: 0.1177 [0.0789, 0.1604]

### ungated_S_v1_sigma

**fold_omit_0** (pre: 2024-01-21, 2024-01-26)

- Pixel FAR: 0.1838 [0.1568, 0.2127]
- Window FAR: 0.1177 [0.0789, 0.1604]

**fold_omit_1** (pre: 2024-01-16, 2024-01-26)

- Pixel FAR: 0.1075 [0.0839, 0.1335]
- Window FAR: 0.1177 [0.0789, 0.1604]

**fold_omit_2** (pre: 2024-01-16, 2024-01-21)

- Pixel FAR: 0.0797 [0.0592, 0.1034]
- Window FAR: 0.1177 [0.0789, 0.1604]

### gate_v1

**fold_omit_0** (pre: 2024-01-21, 2024-01-26)

- Pixel FAR: 0.1839 [0.1569, 0.2128]
- Window FAR: 0.1177 [0.0789, 0.1604]

**fold_omit_1** (pre: 2024-01-16, 2024-01-26)

- Pixel FAR: 0.1076 [0.0840, 0.1336]
- Window FAR: 0.1177 [0.0789, 0.1604]

**fold_omit_2** (pre: 2024-01-16, 2024-01-21)

- Pixel FAR: 0.0798 [0.0593, 0.1035]
- Window FAR: 0.1177 [0.0789, 0.1604]

### gate_v1_with_A5_sigma

**fold_omit_0** (pre: 2024-01-21, 2024-01-26)

- Pixel FAR: 0.1839 [0.1569, 0.2128]
- Window FAR: 0.1177 [0.0789, 0.1604]

**fold_omit_1** (pre: 2024-01-16, 2024-01-26)

- Pixel FAR: 0.1076 [0.0840, 0.1336]
- Window FAR: 0.1177 [0.0789, 0.1604]

**fold_omit_2** (pre: 2024-01-16, 2024-01-21)

- Pixel FAR: 0.0798 [0.0593, 0.1035]
- Window FAR: 0.1177 [0.0789, 0.1604]

### gate_v2

**fold_omit_0** (pre: 2024-01-21, 2024-01-26)

- Pixel FAR: 0.1839 [0.1569, 0.2128]
- Window FAR: 0.1177 [0.0789, 0.1604]

**fold_omit_1** (pre: 2024-01-16, 2024-01-26)

- Pixel FAR: 0.1076 [0.0840, 0.1336]
- Window FAR: 0.1177 [0.0789, 0.1604]

**fold_omit_2** (pre: 2024-01-16, 2024-01-21)

- Pixel FAR: 0.0798 [0.0593, 0.1035]
- Window FAR: 0.1177 [0.0789, 0.1604]

