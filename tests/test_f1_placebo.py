"""Tests for F1 (trustsr/placebo_v2.py, experiments/f1_placebo.py): the real placebo harness that replaces
the invalidated experiments/x3_placebo.py. Each test is pinned to the B-ID it proves is fixed.
"""
from __future__ import annotations

import numpy as np
import pytest

from trustsr.placebo_v2 import (
    Fold, PostEventDateBlocked, compute_far_pixel_v2, compute_far_window_v2, gate_rule_10m, gate_ungated_s_v1_sigma,
    gate_v1, gate_v1_with_a5_sigma, gate_v2_blocked, guarded_load_date, pool_fold_blocks,
)
from trustsr.placebo import checkerboard_split


# ---------------------------------------------------------------- (d) B1: hard guard ----------------------------------------------------------------

class TestHardGuard:
    def test_raises_for_the_event_date(self):
        with pytest.raises(PostEventDateBlocked):
            guarded_load_date({}, None, '2024-07-30')

    def test_raises_for_the_post_image_date(self):
        with pytest.raises(PostEventDateBlocked):
            guarded_load_date({}, None, '2024-12-06')

    def test_raises_for_any_date_after_the_event(self):
        with pytest.raises(PostEventDateBlocked):
            guarded_load_date({}, None, '2025-01-01')

    def test_does_not_raise_for_a_pre_event_date(self, monkeypatch):
        # patch the real loader so this test needs no cache/network; only the guard's boundary is under test
        import experiments.wayanad_evidence.data as data_mod
        monkeypatch.setattr(data_mod, 'load_date', lambda cfg, cache, date: {'date': date})
        out = guarded_load_date({}, None, '2024-07-29')
        assert out == {'date': '2024-07-29'}

    def test_boundary_is_exact(self, monkeypatch):
        """One day before the event must NOT raise; the event date itself MUST raise."""
        import experiments.wayanad_evidence.data as data_mod
        monkeypatch.setattr(data_mod, 'load_date', lambda cfg, cache, date: {'date': date})
        guarded_load_date({}, None, '2024-07-29')          # no raise
        with pytest.raises(PostEventDateBlocked):
            guarded_load_date({}, None, '2024-07-30')


# ---------------------------------------------------------------- (c) B3: window FAR uses the gate's own flag map ----------------------------------------------------------------

class TestWindowFarUsesOwnFlagMap:
    def test_window_far_changes_with_flag_map(self):
        """Two different flag maps on the SAME valid/split must give different window FAR (the direct fix for
        B3, where x3/run_placebo computed window FAR from the raw parent for every gate regardless of what it flagged).
        """
        shape = (128, 128)                     # one 128px tile = 4x4 (64px) windows
        valid = np.ones(shape, bool)
        tile_split = checkerboard_split(shape, 128, seed=2024)   # single tile -> in one split only
        split = ~tile_split if not tile_split[0, 0] else tile_split

        flagged_a = np.zeros(shape, bool)
        flagged_a[0:64, 0:64] = True            # 1 of 4 windows flagged

        flagged_b = np.zeros(shape, bool)       # nothing flagged

        boot = {'replicates': 200, 'ci': 0.95, 'seed': 2024}
        res_a, _, _ = compute_far_window_v2(flagged_a, valid, split, 128, 64, **boot)
        res_b, _, _ = compute_far_window_v2(flagged_b, valid, split, 128, 64, **boot)
        assert res_a['estimate'] != res_b['estimate']
        assert res_b['estimate'] == 0.0

    def test_window_far_all_flagged_vs_none(self):
        shape = (128, 128)
        valid = np.ones(shape, bool)
        tile_split = np.ones((1, 1), bool)      # single tile, entirely included
        boot = {'replicates': 100, 'ci': 0.95, 'seed': 1}
        none_flagged = np.zeros(shape, bool)
        all_flagged = np.ones(shape, bool)
        res_none, _, _ = compute_far_window_v2(none_flagged, valid, tile_split, 128, 64, **boot)
        res_all, _, _ = compute_far_window_v2(all_flagged, valid, tile_split, 128, 64, **boot)
        assert res_none['estimate'] == 0.0
        assert res_all['estimate'] == 1.0


# ---------------------------------------------------------------- (b) B2: gates are distinct code, distinct results possible ----------------------------------------------------------------

class TestGatesDifferOnSyntheticFixture:
    def _fixture(self):
        """A small synthetic fold where the three real-vs-parent gates are DESIGNED to disagree:
        - top-left quadrant: parent True, d/sigma finite and large -> S true (OBSERVED for both v1 sigmas)
        - top-right quadrant: parent True, sigma_v1 is NaN (a degenerate dihedral estimate) -> gate_v1 must
          mark this NO_DATA (excluded from flagged) while rule_10m (no d/sigma dependency) still flags it
        - bottom-left quadrant: parent False, d/sigma_v1 huge -> S true, P false -> ungated_S flags it,
          rule_10m and gate_v1 do not (UNSUPPORTED is never flagged)
        - bottom-right quadrant: nothing
        """
        h = w = 64
        parent = np.zeros((h, w), bool)
        parent[:32, :] = True                      # top half parent-positive
        nodata = np.zeros((h, w), bool)
        d = np.full((h, w), 0.01)
        sigma_v1 = np.full((h, w), 0.1)
        sigma_a5 = np.full((h, w), 0.1)
        d[:32, :32] = 5.0                            # top-left: large drop, will exceed k*sigma
        sigma_v1[:32, 32:] = np.nan                   # top-right: degenerate sigma -> NO_DATA under classify()
        sigma_a5[:32, 32:] = np.nan
        d[32:, :32] = 5.0                             # bottom-left: large drop but parent is False here
        fold = Fold(held_out='synthetic', pre_dates=['synthetic_a', 'synthetic_b'], has_sr=True,
                    parent_hr=parent, nodata=nodata, d=d, sigma_v1=sigma_v1, sigma_a5=sigma_a5, n_pre=2)
        return fold

    def test_gates_differ_on_synthetic_fixture(self):
        fold = self._fixture()
        k = 2.0
        r10 = gate_rule_10m(fold, k)
        s = gate_ungated_s_v1_sigma(fold, k)
        v1 = gate_v1(fold, k)
        v1_a5 = gate_v1_with_a5_sigma(fold, k)

        # rule_10m flags the whole parent-positive region including the NaN-sigma quadrant
        assert r10[:32, :].all()
        # gate_v1 must NOT flag the NaN-sigma quadrant (NO_DATA there), so it disagrees with rule_10m
        assert not v1[:32, 32:].any()
        assert r10.sum() != v1.sum()
        # ungated_S flags the bottom-left (S true, P false) quadrant that NEITHER rule_10m NOR gate_v1 flags
        assert s[32:, :32].all()
        assert not r10[32:, :32].any()
        assert not v1[32:, :32].any()
        assert s.sum() != v1.sum()
        assert not np.array_equal(s, r10)             # different pixel SETS even where counts coincide
        assert not np.array_equal(s, v1)
        assert not np.array_equal(r10, v1)

    def test_gate_v2_is_not_an_alias_of_gate_v1(self):
        fold = self._fixture()
        with pytest.raises(NotImplementedError):
            gate_v2_blocked(fold, 2.0)


# ---------------------------------------------------------------- (a) B4: pooled point estimate lies inside its own pooled CI ----------------------------------------------------------------

class TestPooledEstimateInsideCI:
    def test_point_estimate_inside_ci_single_block_set(self):
        rng = np.random.RandomState(0)
        num = rng.binomial(20, 0.1, size=50).astype(float)
        den = np.full(50, 20.0)
        result = pool_fold_blocks([(num, den)], replicates=2000, ci=0.95, seed=2024)
        assert result['lo'] <= result['estimate'] <= result['hi']

    def test_point_estimate_inside_ci_pooled_across_folds(self):
        """The B4 regression case: pooling several folds' blocks and checking the SAME pooled estimate the
        CI was computed from (x3/run_placebo reported fold 0's CI beside the 3-fold pooled point estimate,
        which fell outside it).
        """
        rng = np.random.RandomState(1)
        folds = [(rng.binomial(20, p, size=30).astype(float), np.full(30, 20.0)) for p in (0.05, 0.15, 0.25)]
        pooled = pool_fold_blocks(folds, replicates=2000, ci=0.95, seed=2024)
        assert pooled['lo'] <= pooled['estimate'] <= pooled['hi']
        # sanity: the pooled point estimate must equal sum(num)/sum(den) over ALL folds' blocks, not one fold's
        all_num = np.concatenate([n for n, _ in folds])
        all_den = np.concatenate([d for _, d in folds])
        assert pooled['estimate'] == pytest.approx(all_num.sum() / all_den.sum())

    def test_far_pixel_v2_estimate_inside_its_ci(self):
        rng = np.random.RandomState(2024)
        shape = (256, 256)
        flagged = rng.rand(*shape) > 0.9
        valid = np.ones(shape, bool)
        tile_split = checkerboard_split(shape, 128, seed=2024)
        result, _, _ = compute_far_pixel_v2(flagged, valid, tile_split, 128, replicates=500, ci=0.95, seed=2024)
        assert result['lo'] <= result['estimate'] <= result['hi']


# ---------------------------------------------------------------- B5: flagged excludes UNSUPPORTED ----------------------------------------------------------------

class TestFlaggedExcludesUnsupported:
    def test_unsupported_only_pixels_are_never_flagged(self):
        """A pixel with S true and P false is UNSUPPORTED under classify(); gate_v1/_a5 must never flag it,
        matching configs/fix.yaml units.flagged_definition.v1 = OBSERVED union INFERRED (B5's fix)."""
        h = w = 16
        parent = np.zeros((h, w), bool)
        d = np.full((h, w), 5.0)                # S true everywhere (d >> k*sigma)
        sigma = np.full((h, w), 0.1)
        nodata = np.zeros((h, w), bool)
        fold = Fold(held_out='x', pre_dates=['a', 'b'], has_sr=True, parent_hr=parent, nodata=nodata,
                    d=d, sigma_v1=sigma, sigma_a5=sigma, n_pre=2)
        assert not gate_v1(fold, 2.0).any()
        assert not gate_v1_with_a5_sigma(fold, 2.0).any()
        assert gate_ungated_s_v1_sigma(fold, 2.0).all()   # the ungated baseline DOES flag it (by design: no gating)
