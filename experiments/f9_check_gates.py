"""F9 verification scratch script (check 6: gates_in_a_table_differ_in_code).

The historical bug to hunt (B2, in x3) was `gate_v1_with_a5_sigma` and `gate_v2_placeholder` both being
bound to `gate_v1_trust` -- three table rows, one function. So:

  (a) compare the __code__ objects of all 5 entries of placebo_v2.GATES pairwise: any two rows sharing a
      code object is a B2 regression;
  (b) F1's report claims gate_v1 and gate_v1_with_a5_sigma are numerically IDENTICAL on the real crop BY
      DESIGN (flagged = OBSERVED|INFERRED = the parent mask wherever d and sigma are finite), not by code
      aliasing. Test that claim both ways:
        - they must be different functions (different code objects, different sigma field read);
        - a fixture where ONLY sigma_a5 is degenerate must make them DIVERGE. F1's own shipped fixture
          NaNs both sigmas in the same quadrant, so it does NOT separate these two gates; this script
          builds the fixture that does.
  (c) F5's method table: alloc_bilinear and alloc_pretrained_sr are the same allocator by design, so the
      thing to verify is that they are fed DIFFERENT ranking fields (otherwise the F5 headline compares a
      quantity with itself -- B7-shaped).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from trustsr.placebo_v2 import (
    GATES, Fold, gate_rule_10m, gate_ungated_s_v1_sigma, gate_v1, gate_v1_with_a5_sigma,
)

ROOT = Path(__file__).resolve().parents[1]
out = {}


# ---------------------------------------------------------------- (a) pairwise code identity
def code_identity():
    names = list(GATES)
    ids = {n: GATES[n].__code__ for n in names}
    shared = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if ids[a] is ids[b]:
                shared.append([a, b])
    return {'gate_names': names,
            'n_distinct_code_objects': len({id(c) for c in ids.values()}),
            'pairs_sharing_a_code_object': shared,
            'qualnames': {n: GATES[n].__qualname__ for n in names},
            'source_first_line': {n: GATES[n].__code__.co_firstlineno for n in names},
            'B2_regression': bool(shared)}


# ---------------------------------------------------------------- (b) the fixture that separates v1 / a5
def v1_vs_a5_divergence():
    """parent True everywhere, d large, sigma_v1 finite, sigma_a5 NaN on the right half.

    classify() marks a pixel NO_DATA where its sigma is non-finite, so gate_v1 (v1 sigma) must flag the
    right half while gate_v1_with_a5_sigma (A5 sigma) must not. If they were one function this is
    impossible.
    """
    h = w = 64
    parent = np.ones((h, w), bool)
    nodata = np.zeros((h, w), bool)
    d = np.full((h, w), 5.0)
    sigma_v1 = np.full((h, w), 0.1)
    sigma_a5 = np.full((h, w), 0.1)
    sigma_a5[:, 32:] = np.nan                     # ONLY the A5 sigma is degenerate here
    fold = Fold(held_out='f9_fixture', pre_dates=['a', 'b'], has_sr=True, parent_hr=parent,
                nodata=nodata, d=d, sigma_v1=sigma_v1, sigma_a5=sigma_a5, n_pre=2)
    a, b = gate_v1(fold, 2.0), gate_v1_with_a5_sigma(fold, 2.0)
    # and the reverse asymmetry, to prove neither name is silently reading the other's sigma
    sigma_v1b = np.full((h, w), 0.1); sigma_v1b[:, 32:] = np.nan
    fold2 = Fold(held_out='f9_fixture_rev', pre_dates=['a', 'b'], has_sr=True, parent_hr=parent,
                 nodata=nodata, d=d, sigma_v1=sigma_v1b, sigma_a5=np.full((h, w), 0.1), n_pre=2)
    a2, b2 = gate_v1(fold2, 2.0), gate_v1_with_a5_sigma(fold2, 2.0)
    return {
        'fixture': 'parent=True, d=5.0, k=2.0, 64x64; only sigma_a5 NaN on cols 32:',
        'gate_v1_flagged_px': int(a.sum()), 'gate_v1_with_a5_sigma_flagged_px': int(b.sum()),
        'maps_differ': bool(not np.array_equal(a, b)),
        'n_px_where_they_disagree': int((a != b).sum()),
        'reverse_fixture_only_sigma_v1_nan': {
            'gate_v1_flagged_px': int(a2.sum()), 'gate_v1_with_a5_sigma_flagged_px': int(b2.sum()),
            'maps_differ': bool(not np.array_equal(a2, b2)),
            'each_gate_reads_its_own_sigma': bool(int(a.sum()) != int(a2.sum())
                                                  and int(b.sum()) != int(b2.sum()))},
    }


# ---------------------------------------------------------------- (b2) reproduce F1's real-crop identity claim
def identity_on_a_finite_fixture():
    """F1's design argument: wherever d and BOTH sigmas are finite, flagged = parent for either sigma, so
    the two gates coincide. Confirm that is what happens -- i.e. the real-crop identity is explained, not
    a symptom of aliasing."""
    h = w = 64
    rng = np.random.RandomState(2024)
    parent = rng.rand(h, w) > 0.5
    fold = Fold(held_out='x', pre_dates=['a', 'b'], has_sr=True, parent_hr=parent,
                nodata=np.zeros((h, w), bool), d=rng.normal(0, 1, (h, w)),
                sigma_v1=np.full((h, w), 0.3), sigma_a5=np.full((h, w), 0.9), n_pre=2)
    a, b = gate_v1(fold, 2.0), gate_v1_with_a5_sigma(fold, 2.0)
    return {'sigmas_differ_by_3x_but_all_finite': True,
            'gate_v1_flagged_px': int(a.sum()), 'a5_flagged_px': int(b.sum()),
            'maps_identical': bool(np.array_equal(a, b)),
            'both_equal_parent_mask': bool(np.array_equal(a, parent) and np.array_equal(b, parent)),
            'interpretation': 'identity on finite data is a property of flagged = OBSERVED|INFERRED = '
                              'parent, reproduced here with two deliberately very different sigmas; it is '
                              'NOT evidence of code aliasing'}


# ---------------------------------------------------------------- (c) F5 ranking fields
def f5_rankings_differ():
    d = json.load(open(ROOT / 'experiments/results/f5.json'))
    sr = d.get('sr_output_sha256') or d.get('sr', {})
    per = d.get('methods_ranking_fields') or {}
    return {'f5_json_top_keys': sorted(d)[:25], 'ranking_field_block_present': bool(per),
            'sr_sha_block': str(sr)[:400]}


if __name__ == '__main__':
    out['a_code_identity'] = code_identity()
    out['b_v1_vs_a5_divergence_fixture'] = v1_vs_a5_divergence()
    out['b2_identity_explained_on_finite_data'] = identity_on_a_finite_fixture()
    out['c_f5'] = f5_rankings_differ()
    print(json.dumps({k: v for k, v in out.items() if k != 'c_f5'}, indent=1))
    json.dump(out, open(ROOT / 'experiments/f9_gate_distinctness.json', 'w'), indent=1)
