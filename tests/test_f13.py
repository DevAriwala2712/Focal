"""F13 pass 1 (A0, A3, A4) helper tests. CPU-only, tiny synthetic arrays, no cache and no network.

These prove mechanics of deterministic helpers only. Every experiment result comes from the real cached data.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import pytest

from experiments import f13_a0_exchangeability as A0
from experiments import f13_a3_sr_swap as A3
from experiments import f13_a4_degrade20m as A4
from experiments import f13_a1_sr_vs_interpolation as A1
from experiments import f13_a2_calibrated_fraction as A2
from experiments import f13_b2_inventory_width as B2
from experiments import f13_b1_cross_season_placebo as B1
from experiments import f13_common as C

ROOT = Path(__file__).resolve().parent.parent
PREREG_SHA256 = '84beef1cebb14dae451ca0220b301924f817e5dd21f5a6dc308a7a762a68dad7'   # committed in 0978e8a


# ---------------------------------------------------------------- shared ----------------------------------------------------------------

class TestPreregistration:
    def test_prereg_hash_is_the_committed_one(self):
        # fails if anyone edits configs/f13_discovery.yaml after it was committed (the pre-registration)
        assert C.prereg_sha256() == hashlib.sha256((ROOT / 'configs/f13_discovery.yaml').read_bytes()).hexdigest()
        assert C.prereg_sha256() == PREREG_SHA256

    def test_thresholds_come_from_the_yaml(self):
        cfg = C.load_prereg()
        assert cfg['statistics']['bootstrap']['replicates'] == 2000
        assert cfg['seed'] == 2024
        assert cfg['power_requirement']['injected_ndvi_drops'] == [0.15, 0.30, 0.50]

    def test_json_sanitiser_removes_nan_and_numpy_types(self):
        out = C.to_jsonable({'a': np.float64('nan'), 'b': np.int64(3), 'c': np.array([1.0, np.inf]), 'd': np.bool_(True)})
        assert out == {'a': None, 'b': 3, 'c': [1.0, None], 'd': True}


# ---------------------------------------------------------------- A0: splits ----------------------------------------------------------------

class TestSplits:
    def test_random_tile_split_has_exact_counts_and_is_deterministic(self):
        a = A0.random_tile_split((20, 16), n_test=160, seed=7)
        b = A0.random_tile_split((20, 16), n_test=160, seed=7)
        assert a.shape == (20, 16) and a.dtype == bool
        assert int(a.sum()) == 160
        assert np.array_equal(a, b)

    def test_random_tile_split_differs_by_seed_and_from_checkerboard(self):
        from trustsr.placebo import checkerboard_split
        a = A0.random_tile_split((20, 16), 160, seed=7)
        b = A0.random_tile_split((20, 16), 160, seed=8)
        cb = checkerboard_split((2560, 2048), 128)
        assert cb.shape == (20, 16) and int(cb.sum()) == 160          # F2's own tile counts
        assert not np.array_equal(a, b)
        assert not np.array_equal(a, cb)

    def test_random_window_split_equal_halves(self):
        w = A0.random_window_split((40, 32), seed=2024)
        assert int(w.sum()) == 640 and int((~w).sum()) == 640


# ---------------------------------------------------------------- A0: conformal order statistic with weights ----------------------------------------------------------------

class TestWeightedOrderStatistic:
    def test_matches_repeat_and_sort(self):
        rng = np.random.default_rng(0)
        scores = np.sort(rng.normal(size=50))
        w = rng.integers(0, 4, size=50)
        for k in (1, 7, int(w.sum())):
            expect = np.sort(np.repeat(scores, w))[k - 1]
            assert A0.weighted_order_statistic(scores, w, k) == expect

    def test_k_beyond_total_weight_raises(self):
        with pytest.raises(ValueError):
            A0.weighted_order_statistic(np.array([1.0, 2.0]), np.array([1, 1]), 3)


def _records(cal_scores, test_scores, test_flaggable=None):
    n_c, n_t = len(cal_scores), len(test_scores)
    score = np.r_[cal_scores, test_scores].astype(float)
    n = n_c + n_t
    flaggable = np.ones(n, bool)
    if test_flaggable is not None:
        flaggable[n_c:] = test_flaggable
    rec = {'score': score, 'in_pop': np.ones(n, bool), 'flaggable': flaggable, 'valid_px': np.ones(n, bool),
           'tile_id': np.arange(n), 'pos_id': np.arange(n)}
    is_test = np.r_[np.zeros(n_c, bool), np.ones(n_t, bool)]
    return rec, is_test


class TestGapStatistic:
    def test_known_fixture(self):
        # n=39 calibration scores 1..39, alpha=0.05 -> k=ceil(40*0.95)=38 -> tau=38
        rec, is_test = _records(np.arange(1, 40), [0.5, 38.5, 40.0, 41.0, 2.0])
        g = A0.gap_from_weights(rec, is_test, np.ones(44), alpha=0.05)
        assert g['tau'] == 38.0 and g['k'] == 38 and g['n_cal_pop'] == 39
        assert (g['num_cal'], g['den_cal']) == (1.0, 39.0)           # only score 39 exceeds 38
        assert (g['num_test'], g['den_test']) == (3.0, 5.0)
        assert g['D'] == pytest.approx(3 / 5 - 1 / 39)

    def test_unflaggable_window_is_not_a_false_alarm(self):
        # a window whose score exceeds tau but which allocates 0 sub-pixels is not flagged (round(16 f) = 0)
        rec, is_test = _records(np.arange(1, 40), [0.5, 38.5, 40.0, 41.0, 2.0],
                                test_flaggable=[True, True, False, True, True])
        g = A0.gap_from_weights(rec, is_test, np.ones(44), alpha=0.05)
        assert g['num_test'] == 2.0

    def test_equality_with_tau_is_not_flagged(self):
        rec, is_test = _records(np.arange(1, 40), [38.0])
        assert A0.gap_from_weights(rec, is_test, np.ones(40), alpha=0.05)['num_test'] == 0.0

    def test_weights_change_tau_like_duplicated_records(self):
        rec, is_test = _records(np.arange(1, 40), [0.5, 38.5])
        w = np.ones(41); w[38] = 2                                   # duplicate the score-39 record
        g = A0.gap_from_weights(rec, is_test, w, alpha=0.05)
        assert g['n_cal_pop'] == 40 and g['k'] == 39 and g['tau'] == 39.0
        assert g['num_cal'] == 0.0 and g['den_cal'] == 40.0

    def test_window_without_valid_px_does_not_count(self):
        rec, is_test = _records(np.arange(1, 40), [50.0, 60.0])
        rec['valid_px'][-1] = False; rec['in_pop'][-1] = False
        g = A0.gap_from_weights(rec, is_test, np.ones(41), alpha=0.05)
        assert (g['num_test'], g['den_test']) == (1.0, 1.0)


class TestBootstrapGap:
    def _iid(self, n=400, seed=1):
        rng = np.random.default_rng(seed)
        score = rng.normal(size=n)
        tile = np.arange(n) // 4
        rec = {'score': score, 'in_pop': np.ones(n, bool), 'flaggable': np.ones(n, bool),
               'valid_px': np.ones(n, bool), 'tile_id': tile, 'pos_id': np.arange(n)}
        is_test = (tile % 2).astype(bool)
        return rec, is_test

    def test_deterministic_given_seed_and_brackets_point_estimate(self):
        rec, is_test = self._iid()
        a = A0.bootstrap_gap(rec, is_test, alpha=0.05, replicates=300, seed=5, ci=0.95)
        b = A0.bootstrap_gap(rec, is_test, alpha=0.05, replicates=300, seed=5, ci=0.95)
        assert a == b
        assert a['lo'] <= a['D'] <= a['hi']

    def test_one_replicate_equals_brute_force_duplication(self):
        rec, is_test = self._iid(n=80, seed=2)
        rng = np.random.default_rng(9)
        n_tiles = int(rec['tile_id'].max()) + 1
        draw = rng.integers(0, n_tiles, size=n_tiles)
        counts = np.bincount(draw, minlength=n_tiles)
        g_fast = A0.gap_from_weights(rec, is_test, counts[rec['tile_id']].astype(float), alpha=0.05)
        # brute force: physically repeat every tile's records `count` times
        idx = np.concatenate([np.flatnonzero(rec['tile_id'] == t).repeat(1).tolist() * counts[t] for t in range(n_tiles)]).astype(int)
        rec2 = {k: v[idx] for k, v in rec.items()}
        g_brute = A0.gap_from_weights(rec2, is_test[idx], np.ones(len(idx)), alpha=0.05)
        for key in ('tau', 'D', 'num_cal', 'den_cal', 'num_test', 'den_test'):
            assert g_fast[key] == pytest.approx(g_brute[key])


class TestTilePermutationReference:
    def test_exchangeable_data_places_observed_gap_in_the_bulk(self):
        rng = np.random.default_rng(3)
        n = 800
        rec = {'score': rng.normal(size=n), 'in_pop': np.ones(n, bool), 'flaggable': np.ones(n, bool),
               'valid_px': np.ones(n, bool), 'tile_id': np.arange(n) // 4, 'pos_id': np.arange(n)}
        obs_split = ((rec['tile_id'] % 2) == 1)
        out = A0.tile_permutation_reference(rec, obs_split, n_tiles=200, n_test_tiles=100, alpha=0.05,
                                            n_perm=300, seed=11)
        assert out['p_one_sided'] > 0.05

    def test_shifted_test_half_is_in_the_upper_tail(self):
        rng = np.random.default_rng(3)
        n = 800
        tile = np.arange(n) // 4
        obs_split = (tile % 2) == 1
        score = rng.normal(size=n) + 2.0 * obs_split                 # test windows systematically higher
        rec = {'score': score, 'in_pop': np.ones(n, bool), 'flaggable': np.ones(n, bool),
               'valid_px': np.ones(n, bool), 'tile_id': tile, 'pos_id': np.arange(n)}
        out = A0.tile_permutation_reference(rec, obs_split, n_tiles=200, n_test_tiles=100, alpha=0.05,
                                            n_perm=300, seed=11)
        assert out['p_one_sided'] < 0.01


class TestBetaBinomialReference:
    def test_degenerate_threshold_is_certain(self):
        assert A0.betabinom_tail(n_cal=981, n_test=981, alpha=0.05, x_min=0) == pytest.approx(1.0)

    def test_matches_monte_carlo_of_the_conformal_mechanism(self):
        # independent of scipy.betabinom: U ~ Beta(n+1-k, k) is the exceedance probability of the conformal tau
        n, alpha = 981, 0.05
        k = math.ceil((n + 1) * (1 - alpha))
        rng = np.random.default_rng(0)
        u = rng.beta(n + 1 - k, k, size=400_000)
        x = rng.binomial(n, u)
        mc = float((x >= 64).mean())
        assert A0.betabinom_tail(n_cal=n, n_test=n, alpha=alpha, x_min=64) == pytest.approx(mc, abs=0.004)

    def test_returns_the_conformal_parameters_it_used(self):
        info = A0.betabinom_parameters(n_cal=981, alpha=0.05)
        assert (info['k'], info['a'], info['b']) == (933, 49, 933)


# ---------------------------------------------------------------- A0: window flag fast path + injection + patch placement ----------------------------------------------------------------

class TestInjection:
    def _refl(self):
        rng = np.random.default_rng(0)
        r = np.zeros((20, 20, 4))
        r[..., 0] = 0.04 + 0.01 * rng.random((20, 20))               # B04
        r[..., 3] = 0.35 + 0.02 * rng.random((20, 20))               # B08
        r[..., 1] = r[..., 2] = 0.05
        return r

    def test_ndvi_falls_by_exactly_the_drop_inside_and_nowhere_else(self):
        refl = self._refl()
        mask = np.zeros((20, 20), bool); mask[5:15, 5:15] = True
        out = A0.inject_ndvi_drop(refl, mask, 0.30)
        nd = lambda a: (a[..., 3] - a[..., 0]) / (a[..., 3] + a[..., 0])
        assert np.allclose(nd(out)[mask], nd(refl)[mask] - 0.30)
        assert np.array_equal(out[~mask], refl[~mask])
        assert np.allclose((out[..., 0] + out[..., 3])[mask], (refl[..., 0] + refl[..., 3])[mask])

    def test_does_not_mutate_its_input(self):
        refl = self._refl(); before = refl.copy()
        A0.inject_ndvi_drop(refl, np.ones((20, 20), bool), 0.5)
        assert np.array_equal(refl, before)

    def test_refuses_to_create_negative_reflectance(self):
        refl = np.zeros((4, 4, 4)); refl[..., 0] = 0.30; refl[..., 3] = 0.02    # NDVI already negative
        with pytest.raises(ValueError):
            A0.inject_ndvi_drop(refl, np.ones((4, 4), bool), 0.5)


class TestPatchPlacement:
    def test_distinct_windows_inside_window_all_valid_and_deterministic(self):
        valid = np.ones((64, 64), bool); valid[0:16, 0:16] = False       # one fully invalid window
        eligible_windows = np.ones((4, 4), bool)
        a = A0.place_patches(valid, eligible_windows, n=8, patch_px=10, window_px=16, seed=3)
        b = A0.place_patches(valid, eligible_windows, n=8, patch_px=10, window_px=16, seed=3)
        assert a == b and len(a) == 8
        wins = {(r // 16, c // 16) for r, c in a}
        assert len(wins) == 8                                            # no two patches share a window
        for r, c in a:
            assert (r // 16, c // 16) != (0, 0)
            assert (r + 9) // 16 == r // 16 and (c + 9) // 16 == c // 16  # wholly inside one window
            assert valid[r:r + 10, c:c + 10].all()

    def test_not_enough_eligible_windows_raises(self):
        valid = np.ones((32, 32), bool)
        with pytest.raises(ValueError):
            A0.place_patches(valid, np.ones((2, 2), bool), n=5, patch_px=10, window_px=16, seed=0)

    def test_only_eligible_windows_are_used(self):
        valid = np.ones((64, 64), bool)
        elig = np.zeros((4, 4), bool); elig[1, 2] = elig[3, 3] = True
        got = A0.place_patches(valid, elig, n=2, patch_px=10, window_px=16, seed=0)
        assert {(r // 16, c // 16) for r, c in got} == {(1, 2), (3, 3)}


# ---------------------------------------------------------------- A0: per-window records from block arrays ----------------------------------------------------------------

class TestWindowArrays:
    def _blocks(self):
        # 2 x 2 windows of 16 x 16 blocks => 32 x 32 blocks; nodata on the 2.5 m grid is 128 x 128
        s = np.full((32, 32), 1.0); f = np.full((32, 32), 0.5)
        valid = np.ones((32, 32), bool)
        nodata = np.zeros((128, 128), bool)
        return s, valid, f, nodata

    def test_score_is_window_max_over_valid_blocks_only(self):
        s, valid, f, nodata = self._blocks()
        s[3, 3] = 9.0; s[20, 20] = 7.0
        valid[20, 20] = False                                  # the 7.0 block is invalid and must not count
        out = A0.window_arrays(s, valid, f, nodata)
        assert out['score'].shape == (2, 2)
        assert out['score'][0, 0] == 9.0 and out['score'][1, 1] == 1.0

    def test_window_with_no_valid_block_is_outside_the_tau_population(self):
        s, valid, f, nodata = self._blocks()
        valid[0:16, 16:32] = False                             # window (0, 1) has no valid block
        out = A0.window_arrays(s, valid, f, nodata)
        assert out['in_pop'].tolist() == [[True, False], [True, True]]

    def test_flaggable_needs_a_block_that_allocates_at_least_one_subpixel(self):
        s, valid, f, nodata = self._blocks()
        f[0:16, 0:16] = 0.02                                   # round(16 * 0.02) = 0 -> nothing to allocate
        f[16:32, 0:16] = 0.02; f[20, 3] = 0.5                  # one block allocates 8
        f[0:16, 16:32] = -0.3                                  # negative fraction clips to 0
        out = A0.window_arrays(s, valid, f, nodata)
        assert out['flaggable'].tolist() == [[False, False], [True, True]]

    def test_invalid_block_does_not_make_a_window_flaggable(self):
        s, valid, f, nodata = self._blocks()
        f[0:16, 0:16] = 0.0; f[2, 2] = 0.9; valid[2, 2] = False
        assert not A0.window_arrays(s, valid, f, nodata)['flaggable'][0, 0]

    def test_valid_px_is_pixel_based_not_block_based(self):
        s, valid, f, nodata = self._blocks()
        nodata[0:64, 64:128] = True                            # every 2.5 m px of window (0, 1) is NO_DATA
        nodata[64:128, 0:64] = True; nodata[70, 70 - 64] = False   # window (1, 0): a single valid px survives
        out = A0.window_arrays(s, valid, f, nodata)
        assert out['valid_px'].tolist() == [[True, False], [True, True]]

    def test_nan_fraction_is_treated_as_zero_like_apply_gate_v2(self):
        s, valid, f, nodata = self._blocks()
        f[:] = np.nan
        assert not A0.window_arrays(s, valid, f, nodata)['flaggable'].any()


class TestPatchHit:
    def _grid(self):
        score = np.array([[5.0, 1.0], [1.0, 1.0]])             # 2 x 2 windows of 16 blocks
        in_pop = np.ones((2, 2), bool)
        n_blk = np.zeros((32, 32), int)
        return score, in_pop, n_blk

    def test_detected_window_with_allocating_blocks_flags_the_patch_and_counts_px(self):
        score, in_pop, n_blk = self._grid()
        n_blk[2:12, 2:12] = 4                                   # patch origin (2, 2), 10 x 10 blocks, 4 px each
        hit = A0.patch_hit(score, in_pop, n_blk, (2, 2), tau=3.0, patch_px=10, window_px=16)
        assert hit == {'detected': True, 'flagged': True, 'flagged_px': 400}

    def test_window_at_or_below_tau_is_not_detected(self):
        score, in_pop, n_blk = self._grid()
        n_blk[2:12, 2:12] = 4
        hit = A0.patch_hit(score, in_pop, n_blk, (2, 2), tau=5.0, patch_px=10, window_px=16)   # equal to the score
        assert hit == {'detected': False, 'flagged': False, 'flagged_px': 0}

    def test_detected_window_whose_patch_allocates_nothing_is_not_flagged(self):
        score, in_pop, n_blk = self._grid()
        n_blk[20:30, 20:30] = 9                                 # allocation exists, but outside the patch
        hit = A0.patch_hit(score, in_pop, n_blk, (2, 2), tau=3.0, patch_px=10, window_px=16)
        assert hit == {'detected': True, 'flagged': False, 'flagged_px': 0}

    def test_window_outside_the_tau_population_is_never_detected(self):
        score, in_pop, n_blk = self._grid()
        in_pop[0, 0] = False; n_blk[2:12, 2:12] = 4
        assert not A0.patch_hit(score, in_pop, n_blk, (2, 2), tau=0.0, patch_px=10, window_px=16)['detected']


class TestAssembleRecords:
    def _track(self):
        mk = lambda off: {'wa': {'score': np.full((40, 32), float(off)), 'in_pop': np.ones((40, 32), bool),
                                 'flaggable': np.ones((40, 32), bool), 'valid_px': np.ones((40, 32), bool)}}
        return {'per_fold': {'2024-01-16': mk(1), '2024-01-21': mk(2), '2024-01-26': mk(3)}}

    def test_stacks_the_three_folds_over_the_same_1280_window_positions(self):
        rec = A0.assemble_records(self._track())
        assert rec['score'].shape == (3 * 1280,)
        assert rec['score'][:1280].tolist() == [1.0] * 1280 and rec['score'][-1280:].tolist() == [3.0] * 1280
        assert np.array_equal(rec['pos_id'][:1280], np.arange(1280)) and np.array_equal(rec['pos_id'][:1280], rec['pos_id'][1280:2560])

    def test_tile_id_groups_each_2x2_windows_into_one_128px_tile_320_tiles(self):
        rec = A0.assemble_records(self._track())
        t = rec['tile_id'][:1280].reshape(40, 32)
        assert int(rec['tile_id'].max()) == 319 and np.unique(rec['tile_id']).size == 320
        assert t[0, 0] == t[0, 1] == t[1, 0] == t[1, 1] == 0
        assert t[0, 2] == 1 and t[2, 0] == 16 and t[39, 31] == 319
        assert np.bincount(rec['tile_id'][:1280]).tolist() == [4] * 320


class TestA0VerdictRule:
    """yaml experiments.A0.keep_rule, applied verbatim:
    TRUE = checkerboard CI excludes 0 AND random-split CI includes 0; FALSE = both CIs include 0;
    MISCALIBRATED_SCORE = random-split CI excludes 0 (O3). Nothing else exists, so nothing else is returned."""

    def test_true_when_only_the_checkerboard_gap_is_real(self):
        assert A0.a0_verdict((0.002, 0.03), (-0.01, 0.02)) == 'TRUE'
        assert A0.a0_verdict((-0.04, -0.001), (-0.01, 0.02)) == 'TRUE'          # a negative gap also excludes 0

    def test_false_when_both_include_zero(self):
        assert A0.a0_verdict((-0.001, 0.03), (-0.01, 0.02)) == 'FALSE'

    def test_miscalibrated_when_the_random_split_excludes_zero(self):
        assert A0.a0_verdict((0.002, 0.03), (0.001, 0.02)) == 'MISCALIBRATED_SCORE'
        assert A0.a0_verdict((-0.001, 0.03), (0.001, 0.02)) == 'MISCALIBRATED_SCORE'

    def test_a_ci_endpoint_exactly_at_zero_counts_as_including_zero(self):
        assert A0.a0_verdict((0.0, 0.03), (-0.01, 0.02)) == 'FALSE'


# ================================================================ A3: SR ranking vs bilinear ranking ================================================================

class TestPartialBlockMask:
    def test_marks_only_blocks_with_0_lt_n_lt_16_expanded_to_2p5m(self):
        n = np.array([[0, 16], [5, 16]])
        m = A3.partial_block_mask(n)
        assert m.shape == (8, 8)
        expect = np.zeros((8, 8), bool); expect[4:8, 0:4] = True
        assert np.array_equal(m, expect)

    def test_n_of_one_and_fifteen_are_partial(self):
        n = np.array([[1, 15]])
        assert A3.partial_block_mask(n).all()


class TestIoU:
    def test_known_value_and_counts(self):
        a = np.zeros((4, 4), bool); a[0, :] = True; a[1, :2] = True          # 6 px
        b = np.zeros((4, 4), bool); b[0, :] = True; b[2, :3] = True          # 7 px, 4 shared
        out = A3.iou(a, b, np.ones((4, 4), bool))
        assert (out['intersection'], out['union']) == (4, 9)
        assert out['iou'] == pytest.approx(4 / 9)

    def test_domain_restricts_both_sets(self):
        a = np.ones((2, 2), bool); b = np.ones((2, 2), bool)
        dom = np.array([[True, False], [False, False]])
        assert A3.iou(a, b, dom)['union'] == 1

    def test_empty_union_is_nan_not_one(self):
        z = np.zeros((2, 2), bool)
        assert np.isnan(A3.iou(z, z, np.ones((2, 2), bool))['iou'])


class TestBlockCountsAndMovement:
    def test_equal_counts_pass_and_unequal_raise(self):
        a = np.zeros((8, 8), bool); a[0, 0] = a[0, 1] = True
        b = np.zeros((8, 8), bool); b[3, 3] = b[2, 2] = True                  # same block (0,0), same count, other pixels
        A3.assert_equal_block_counts(a, b)
        b[0, 0] = True
        with pytest.raises(AssertionError):
            A3.assert_equal_block_counts(a, b)

    def test_moved_fraction(self):
        a = np.zeros((4, 4), bool); a[0, 0] = a[0, 1] = True
        assert A3.moved_fraction(a, a) == 0.0
        b = np.zeros((4, 4), bool); b[3, 3] = b[3, 2] = True
        assert A3.moved_fraction(a, b) == 1.0
        c = np.zeros((4, 4), bool); c[0, 0] = c[3, 3] = True
        assert A3.moved_fraction(a, c) == 0.5

    def test_moved_fraction_of_empty_set_is_nan(self):
        z = np.zeros((4, 4), bool)
        assert np.isnan(A3.moved_fraction(z, z))


class TestCentredBilinear:
    def test_impulse_response_is_symmetric_about_the_block_centre(self):
        a = np.zeros((6, 6)); a[2, 2] = 1.0
        up = A3.centred_bilinear_upsample(a)
        assert up.shape == (24, 24)
        # block (2,2) covers sub-pixels 8..11; its centre lies between 9 and 10
        assert up[9, 9] == pytest.approx(up[10, 10]) == pytest.approx(up[9, 10])
        assert up[9, 9] == up.max()

    def test_constant_field_stays_constant(self):
        assert np.allclose(A3.centred_bilinear_upsample(np.full((5, 7), 0.3)), 0.3)

    def test_gate_v2_bilinear_is_not_centred(self):
        # documents WHY a centred arm is reported beside the prompt-specified gate_v2.bilinear_upsample:
        # that function stretches the grid so a block's peak lands on a single sub-pixel (shifted half a sub-pixel)
        from trustsr.gate_v2 import bilinear_upsample
        a = np.zeros((6, 6)); a[2, 2] = 1.0
        up = bilinear_upsample(a)
        assert up[9, 9] != pytest.approx(up[10, 10])


class TestBilinearRankingField:
    def test_drop_is_pre_mean_minus_post_with_invalid_filled_by_zero(self):
        pre = [np.full((4, 4), 0.8), np.full((4, 4), 0.6)]
        post = np.full((4, 4), 0.5)
        valid = np.ones((4, 4), bool); valid[0, 0] = False
        d = A3.ndvi_drop_10m(pre, post, valid)
        assert d[1, 1] == pytest.approx(0.2) and d[0, 0] == 0.0

    def test_nan_ndvi_is_zero_filled(self):
        pre = [np.full((2, 2), np.nan)]; post = np.full((2, 2), 0.5)
        assert np.all(A3.ndvi_drop_10m(pre, post, np.ones((2, 2), bool)) == 0.0)


class TestA3VerdictRule:
    """yaml experiments.A3.keep_rule: TRUE IoU < 0.90; FALSE IoU >= 0.95; INCONCLUSIVE in between."""

    def test_regions(self):
        assert A3.a3_verdict(0.5) == 'TRUE'
        assert A3.a3_verdict(0.8999) == 'TRUE'
        assert A3.a3_verdict(0.90) == 'INCONCLUSIVE'
        assert A3.a3_verdict(0.9499) == 'INCONCLUSIVE'
        assert A3.a3_verdict(0.95) == 'FALSE'
        assert A3.a3_verdict(1.0) == 'FALSE'


# ================================================================ A4: does Wayanad need 10 m? ================================================================

class TestAggregate2x2:
    def test_reflectance_is_averaged_not_ndvi(self):
        refl = np.zeros((4, 2, 2)); refl[0] = [[0.10, 0.02], [0.10, 0.02]]; refl[3] = [[0.40, 0.40], [0.40, 0.40]]   # B04 .. B08
        valid = np.ones((2, 2), bool)
        r20, v20 = A4.aggregate_reflectance_2x2(refl, valid)
        assert r20.shape == (4, 1, 1) and v20.shape == (1, 1)
        assert r20[0, 0, 0] == pytest.approx(0.06) and r20[3, 0, 0] == pytest.approx(0.40)
        ndvi = lambda r, n: (n - r) / (n + r)
        mean_of_ndvi = np.mean([ndvi(0.10, 0.40), ndvi(0.02, 0.40)])
        ndvi_of_mean = ndvi(0.06, 0.40)
        assert not np.isclose(mean_of_ndvi, ndvi_of_mean)                      # the two differ, so the choice matters
        assert A4.ndvi_from_reflectance(r20, 0.05)[0, 0] == pytest.approx(ndvi_of_mean)

    def test_cell_is_invalid_if_any_of_its_four_children_is_invalid(self):
        valid = np.ones((4, 4), bool); valid[0, 1] = False; valid[3, 3] = False
        _, v20 = A4.aggregate_reflectance_2x2(np.ones((4, 4, 4)), valid)
        assert v20.tolist() == [[False, True], [True, False]]

    def test_odd_dimensions_raise_instead_of_cropping(self):
        with pytest.raises(ValueError):
            A4.aggregate_reflectance_2x2(np.ones((4, 5, 4)), np.ones((5, 4), bool))


class TestScarComponent:
    def _mask(self):
        m = np.zeros((40, 40), bool)
        m[2:12, 2:12] = True            # 100 px, far from the crown
        m[20:24, 20:23] = True          # 12 px, touches the disk
        m[26:28, 21:24] = True          # 6 px, touches the disk (row 26 is 4 px from the crown row 22)
        return m

    def test_picks_the_largest_component_that_touches_the_disk_not_the_largest_overall(self):
        comp, info = A4.scar_component(self._mask(), crown_rc=(22, 21), radius_px=5)
        assert int(comp.sum()) == 12 and info['n_components_touching_disk'] == 2
        assert comp[21, 21]

    def test_8_connectivity_joins_diagonal_pixels(self):
        m = np.zeros((10, 10), bool); m[2, 2] = m[3, 3] = m[4, 4] = True
        comp, _ = A4.scar_component(m, crown_rc=(3, 3), radius_px=3)
        assert int(comp.sum()) == 3

    def test_no_component_in_the_disk_raises(self):
        with pytest.raises(ValueError):
            A4.scar_component(np.zeros((10, 10), bool), crown_rc=(5, 5), radius_px=3)


class TestDistanceAndAgreement:
    def test_min_distance_from_a_point_to_pixel_centres_in_metres(self):
        from affine import Affine
        tr = Affine(10, 0, 500000, 0, -10, 1000000)             # 10 m px, origin (500000, 1000000)
        comp = np.zeros((10, 10), bool); comp[3, 4] = True      # centre x = 500000 + 4.5*10, y = 1000000 - 3.5*10
        d = A4.min_distance_m(comp, tr, (500045.0 + 30.0, 999965.0))
        assert d == pytest.approx(30.0)

    def test_area_recall_and_upsampled_iou(self):
        scar10 = np.zeros((8, 8), bool); scar10[2:6, 2:6] = True                 # 16 px at 10 m
        mask20 = np.zeros((4, 4), bool); mask20[1:2, 1:3] = True                 # 2 cells = 8 px at 10 m, inside the scar
        up = A4.upsample_nn(mask20, 2)
        assert up.shape == (8, 8) and int(up.sum()) == 8
        assert A4.area_recall(up, scar10) == pytest.approx(0.5)
        assert A3.iou(up, scar10, np.ones((8, 8), bool))['iou'] == pytest.approx(0.5)

    def test_recall_of_an_empty_scar_is_nan(self):
        assert np.isnan(A4.area_recall(np.ones((4, 4), bool), np.zeros((4, 4), bool)))


class TestA4VerdictRule:
    """yaml experiments.A4.keep_rule: TRUE recall >= 0.90; FALSE recall < 0.70; INCONCLUSIVE in between."""

    def test_regions(self):
        assert A4.a4_verdict(0.90) == 'TRUE' and A4.a4_verdict(1.0) == 'TRUE'
        assert A4.a4_verdict(0.8999) == 'INCONCLUSIVE' and A4.a4_verdict(0.70) == 'INCONCLUSIVE'
        assert A4.a4_verdict(0.6999) == 'FALSE'


# ================================================================ PASS 2 — A1: SR vs interpolation ================================================================

class TestResampleAlignment:
    """Every interpolation in Pass 2 goes through trustsr.hrbench.resample (pixel-centre aligned)."""

    @pytest.mark.parametrize('method', ['bilinear', 'bicubic'])
    def test_constant_field_stays_constant(self, method):
        from trustsr.hrbench import resample
        assert np.allclose(resample(np.full((1, 6, 7), 0.37), 4, method), 0.37)

    def test_linear_ramp_is_the_ramp_at_subpixel_centres(self):
        # bilinear only: hrbench's bicubic uses the Keys kernel with a = -0.75 (torch/OpenCV), which does not reproduce a
        # ramp exactly (max error ~0.095 px); its centre alignment is tested by symmetry below instead
        from trustsr.hrbench import resample
        a = np.tile(np.arange(8, dtype=float), (8, 1))[None]          # value = column index of the 10 m pixel centre
        up = resample(a, 4, 'bilinear')[0]
        x = (np.arange(32) + 0.5) / 4 - 0.5                           # sub-pixel centres in source coordinates
        inner = slice(4, 28)                                          # away from the clamped border
        assert np.allclose(up[:, inner], np.broadcast_to(x[inner], (32, 24)))

    @pytest.mark.parametrize('method', ['bilinear', 'bicubic'])
    def test_interp_index_impulse_is_symmetric_about_the_block_centre(self, method):
        a = np.zeros((6, 6)); a[2, 2] = 1.0
        up = A1.interp_index(a, method)
        assert up.shape == (24, 24)
        # block (2,2) covers sub-pixels 8..11; centre alignment makes the response symmetric about 9.5
        assert np.allclose(up[8:12, 8:12], up[8:12, 8:12][::-1, ::-1])
        assert up[9, 9] == pytest.approx(up[10, 10]) and up[9, 9] == pytest.approx(up.max())

    def test_interp_index_fills_nan_with_zero_like_f5(self):
        a = np.full((4, 4), 0.8); a[1, 1] = np.nan
        up = A1.interp_index(a, 'bilinear')
        from trustsr.hrbench import resample
        filled = a.copy(); filled[1, 1] = 0.0
        assert np.isfinite(up).all() and np.allclose(up, resample(filled[None], 4, 'bilinear')[0])


class TestHrAt5m:
    def test_constant_reflectance_gives_constant_ndvi(self):
        hr = np.zeros((4, 16, 16)); hr[0] = 0.05; hr[3] = 0.45
        f = A1.hr_at_5m_field(hr)
        assert f.shape == (16, 16) and np.allclose(f, 0.8)

    def test_averages_reflectance_over_2x2_before_ndvi(self):
        hr = np.zeros((4, 4, 4)); hr[3] = 0.4
        hr[0] = np.array([[0.1, 0.02, 0.1, 0.02], [0.1, 0.02, 0.1, 0.02]] * 2)
        f = A1.hr_at_5m_field(hr)
        expect = (0.4 - 0.06) / (0.4 + 0.06)                         # NDVI of the 5 m mean reflectance
        assert np.allclose(f, expect)


class TestRandomRanking:
    def test_reproducible_per_image_and_different_between_images(self):
        a = A1.random_scores((8, 8), seed=2024, image_index=3)
        b = A1.random_scores((8, 8), seed=2024, image_index=3)
        c = A1.random_scores((8, 8), seed=2024, image_index=4)
        assert np.array_equal(a, b) and not np.array_equal(a, c)


class TestSumRatio:
    def test_sum_ratio_is_not_the_mean_of_ratios(self):
        inter, union = np.array([1, 90]), np.array([10, 100])
        assert A1.sum_ratio(inter, union) == pytest.approx(91 / 110)
        assert A1.sum_ratio(inter, union) != pytest.approx(np.mean(inter / union))

    def test_zero_total_union_is_nan(self):
        assert np.isnan(A1.sum_ratio(np.array([0, 0]), np.array([0, 0])))


class TestPairedRatioDiffBootstrap:
    def _data(self, n=60, shift=0.0, seed=0):
        rng = np.random.default_rng(seed)
        ub = rng.integers(50, 150, n).astype(float)
        ib = np.floor(ub * rng.uniform(0.3, 0.7, n))
        ia = np.minimum(ub, np.floor(ib + shift * ub))
        return ia, ub.copy(), ib, ub

    def test_point_estimate_is_the_difference_of_sum_ratios(self):
        ia, ua, ib, ub = self._data(shift=0.05)
        out = A1.paired_ratio_diff_ci(ia, ua, ib, ub, replicates=200, seed=1, ci=0.95)
        assert out['estimate'] == pytest.approx(ia.sum() / ua.sum() - ib.sum() / ub.sum())
        assert out['lo'] <= out['estimate'] <= out['hi']

    def test_same_image_indices_for_both_arms(self):
        # identical arms -> every replicate difference is exactly 0, which only holds if images are paired
        ia, ua, _ib, _ub = self._data()
        out = A1.paired_ratio_diff_ci(ia, ua, ia, ua, replicates=200, seed=1, ci=0.95)
        assert out['lo'] == 0.0 and out['hi'] == 0.0

    def test_deterministic_and_detects_a_shift(self):
        ia, ua, ib, ub = self._data(shift=0.1)
        a = A1.paired_ratio_diff_ci(ia, ua, ib, ub, replicates=300, seed=7, ci=0.95)
        b = A1.paired_ratio_diff_ci(ia, ua, ib, ub, replicates=300, seed=7, ci=0.95)
        assert a == b and a['lo'] > 0

    def test_matches_a_brute_force_replicate(self):
        ia, ua, ib, ub = self._data(n=20, shift=0.03)
        out = A1.paired_ratio_diff_ci(ia, ua, ib, ub, replicates=50, seed=3, ci=0.95, return_draws=True)
        idx = np.random.default_rng(3).integers(0, 20, size=(50, 20))[0]
        brute = ia[idx].sum() / ua[idx].sum() - ib[idx].sum() / ub[idx].sum()
        assert out['draws'][0] == pytest.approx(brute)


class TestPooledStratum:
    def test_pooled_0_50_is_the_union_of_the_two_narrow_bins(self):
        from trustsr import alloc
        m = np.zeros((40, 40), bool)
        m[2:5, 2:30] = True                  # 3 px wide -> 7.5 m  -> 0-20
        m[10:22, 2:30] = True                # 12 px wide -> 27.5 m -> 20-50
        m[25:40, 0:40] = True                # 15 px wide -> 35 m? no: (2*8-1)*2.5 = 37.5 m -> 20-50
        w, _, _ = alloc.component_width_m(m, 2.5)
        bins = alloc.width_bin_masks(w, m)
        pooled = A1.pooled_stratum(w, m)
        assert np.array_equal(pooled, bins['0-20'] | bins['20-50'])
        assert not (pooled & (w > 50)).any()


class TestA1VerdictRule:
    """yaml A1: TRUE if either primary contrast (without NAIP, pooled 0-50 m) has estimate > 0 with CI excluding 0;
    FALSE if both CIs span 0 or are negative."""

    def test_true_when_one_contrast_is_significantly_positive(self):
        assert A1.a1_verdict([{'estimate': 0.02, 'lo': 0.001, 'hi': 0.04}, {'estimate': -0.01, 'lo': -0.03, 'hi': 0.01}]) == 'TRUE'

    def test_false_when_both_span_zero_or_are_negative(self):
        assert A1.a1_verdict([{'estimate': 0.01, 'lo': -0.001, 'hi': 0.02}, {'estimate': -0.02, 'lo': -0.04, 'hi': -0.01}]) == 'FALSE'

    def test_lower_bound_exactly_zero_does_not_exclude_zero(self):
        assert A1.a1_verdict([{'estimate': 0.01, 'lo': 0.0, 'hi': 0.02}, {'estimate': 0.0, 'lo': -0.01, 'hi': 0.01}]) == 'FALSE'

    def test_pivot_threshold_needs_0p01_and_ci_excluding_zero(self):
        assert A1.pivot_met({'estimate': 0.01, 'lo': 0.002, 'hi': 0.02}) is True
        assert A1.pivot_met({'estimate': 0.0099, 'lo': 0.002, 'hi': 0.02}) is False
        assert A1.pivot_met({'estimate': 0.02, 'lo': -0.001, 'hi': 0.04}) is False


# ================================================================ PASS 2 — A2: calibrated fraction ================================================================

def _brute_isotonic_increasing(y, w):
    """Reference: f_i = max_{j<=i} min_{k>=i} weighted mean(y[j..k]) (min-max formula for the L2 isotonic fit)."""
    n = len(y)
    out = np.empty(n)
    for i in range(n):
        best = -np.inf
        for j in range(i + 1):
            worst = np.inf
            for k in range(i, n):
                worst = min(worst, np.average(y[j:k + 1], weights=w[j:k + 1]))
            best = max(best, worst)
        out[i] = best
    return out


class TestPAV:
    @pytest.mark.parametrize('seed', [0, 1, 2, 3])
    def test_increasing_fit_matches_brute_force(self, seed):
        rng = np.random.default_rng(seed)
        y = rng.normal(size=12); w = rng.integers(1, 4, 12).astype(float)
        assert np.allclose(A2.pav_increasing(y, w), _brute_isotonic_increasing(y, w))

    def test_decreasing_fit_is_monotone_and_matches_brute_force(self):
        rng = np.random.default_rng(5)
        x = np.sort(rng.uniform(size=15)); y = -x + 0.3 * rng.normal(size=15)
        model = A2.fit_isotonic_decreasing(x, y)
        fitted = A2.predict_isotonic(model, x)
        assert np.all(np.diff(fitted) <= 1e-12)
        assert np.allclose(fitted, -_brute_isotonic_increasing(-y, np.ones(15)))

    def test_tied_x_values_are_pooled_before_fitting(self):
        x = np.array([0.1, 0.1, 0.5, 0.9]); y = np.array([1.0, 0.0, 0.2, 0.1])
        model = A2.fit_isotonic_decreasing(x, y)
        assert A2.predict_isotonic(model, np.array([0.1]))[0] == pytest.approx(0.5)

    def test_prediction_interpolates_between_knots_and_clips_outside(self):
        model = A2.fit_isotonic_decreasing(np.array([0.0, 1.0]), np.array([1.0, 0.0]))
        assert A2.predict_isotonic(model, np.array([0.5]))[0] == pytest.approx(0.5)
        assert A2.predict_isotonic(model, np.array([-3.0, 4.0])).tolist() == [1.0, 0.0]

    def test_non_finite_input_predicts_zero(self):
        model = A2.fit_isotonic_decreasing(np.array([0.0, 1.0]), np.array([1.0, 0.0]))
        assert A2.predict_isotonic(model, np.array([np.nan]))[0] == 0.0


class TestLODO:
    def test_four_folds_each_holding_out_one_dataset_and_never_fitting_on_it(self):
        folds = A2.lodo_folds(['naip', 'spot', 'spain_crops', 'spain_urban'])
        assert [f['held_out'] for f in folds] == ['naip', 'spot', 'spain_crops', 'spain_urban']
        for f in folds:
            assert f['held_out'] not in f['train'] and len(f['train']) == 3

    def test_fit_rows_exclude_the_held_out_dataset(self):
        rows = [{'dataset': 'a', 'x': np.array([0.1]), 'y': np.array([0.9])},
                {'dataset': 'b', 'x': np.array([0.8]), 'y': np.array([0.1])},
                {'dataset': 'c', 'x': np.array([0.5]), 'y': np.array([0.5])}]
        x, y = A2.training_arrays(rows, train=['a', 'b'])
        assert sorted(x.tolist()) == [0.1, 0.8] and 0.5 not in x.tolist()


class TestValidBlocks:
    def test_block_is_valid_only_if_all_16_px_valid_and_ndvi_finite(self):
        valid = np.ones((8, 8), bool); valid[0, 0] = False
        ndvi10 = np.array([[0.5, 0.5], [np.nan, 0.5]])
        assert A2.valid_blocks(valid, ndvi10).tolist() == [[False, True], [False, True]]


class TestA2VerdictRule:
    """yaml A2: TRUE = MAE <= 0.15 AND delta > 0 with CI excluding 0; FALSE = MAE > 0.20 OR delta <= 0; else INCONCLUSIVE."""

    def test_true(self):
        assert A2.a2_verdict(0.15, {'estimate': 0.01, 'lo': 0.001, 'hi': 0.02}) == 'TRUE'

    def test_false_on_mae(self):
        assert A2.a2_verdict(0.2001, {'estimate': 0.05, 'lo': 0.01, 'hi': 0.09}) == 'FALSE'

    def test_false_on_non_positive_delta(self):
        assert A2.a2_verdict(0.10, {'estimate': 0.0, 'lo': -0.01, 'hi': 0.01}) == 'FALSE'

    def test_inconclusive_between(self):
        assert A2.a2_verdict(0.18, {'estimate': 0.02, 'lo': 0.01, 'hi': 0.03}) == 'INCONCLUSIVE'
        assert A2.a2_verdict(0.10, {'estimate': 0.01, 'lo': -0.001, 'hi': 0.02}) == 'INCONCLUSIVE'


# ================================================================ PASS 3 — B1: cross-season placebo ================================================================

class TestCandidateTable:
    ROWS = [
        {'date': '2023-01-05', 'cloud_shadow_pct_aoi': 2.0, 'coverage_pct': 100.0, 'item_ids': ['a']},
        {'date': '2023-01-10', 'cloud_shadow_pct_aoi': 10.0, 'coverage_pct': 99.0, 'item_ids': ['b']},     # both thresholds exactly met
        {'date': '2023-01-15', 'cloud_shadow_pct_aoi': 10.01, 'coverage_pct': 100.0, 'item_ids': ['c']},   # just too cloudy
        {'date': '2023-01-20', 'cloud_shadow_pct_aoi': 1.0, 'coverage_pct': 98.99, 'item_ids': ['d']},     # just too little coverage
        {'date': '2023-01-20', 'cloud_shadow_pct_aoi': 60.0, 'coverage_pct': 100.0, 'item_ids': ['e']},    # worse duplicate of the same date
        {'date': '2023-02-03', 'cloud_shadow_pct_aoi': 0.0, 'coverage_pct': 100.0, 'item_ids': ['f']},     # outside the window
    ]

    def test_uses_the_x6_rule_inclusive_thresholds_and_window(self):
        t = B1.candidate_table(self.ROWS, '2023-01-01', '2023-01-31', 10, 99)
        assert [r['date'] for r in t] == ['2023-01-05', '2023-01-10', '2023-01-15', '2023-01-20']
        assert [r['accepted'] for r in t] == [True, True, False, False]

    def test_every_rejection_names_its_reason_and_every_row_carries_its_numbers(self):
        t = {r['date']: r for r in B1.candidate_table(self.ROWS, '2023-01-01', '2023-01-31', 10, 99)}
        assert 'cloud' in t['2023-01-15']['reason'] and '10.01' in t['2023-01-15']['reason']
        assert 'coverage' in t['2023-01-20']['reason'] and '98.99' in t['2023-01-20']['reason']
        assert t['2023-01-05']['reason'] is None and t['2023-01-05']['cloud_shadow_pct_aoi'] == 2.0

    def test_uses_the_best_acquisition_of_a_date(self):
        t = {r['date']: r for r in B1.candidate_table(self.ROWS, '2023-01-01', '2023-01-31', 10, 99)}
        assert t['2023-01-20']['cloud_shadow_pct_aoi'] == 1.0      # the 60 % duplicate is not what is reported

    def test_the_post_date_goes_through_the_same_rule(self):
        t = B1.candidate_table([{'date': '2023-12-27', 'cloud_shadow_pct_aoi': 30.0, 'coverage_pct': 100.0, 'item_ids': ['p']}],
                               '2023-12-27', '2023-12-27', 10, 99)
        assert t[0]['accepted'] is False


class TestB1VerdictRule:
    """yaml B1.keep_rule per gate: TRUE if FAR >= 2 x that gate's own within-season FAR; FALSE if inside its within-season CI."""
    GATE = {'estimate': 0.0652395514780836, 'lo': 0.04780674254466574, 'hi': 0.08498175581546036}
    RULE = {'estimate': 0.0056065239551478085, 'lo': 0.002006999422753962, 'hi': 0.010736333610531435}

    def test_thresholds_are_twice_the_committed_within_season_estimates_not_the_memos_0p10(self):
        assert B1.true_threshold(self.GATE) == pytest.approx(0.1304791029561672)
        assert B1.true_threshold(self.RULE) == pytest.approx(0.011213047910295617)

    def test_regions_for_gate_v2(self):
        assert B1.b1_gate_verdict(0.1304791029561672, self.GATE) == 'TRUE'          # at the threshold counts
        assert B1.b1_gate_verdict(0.13, self.GATE) == 'INCONCLUSIVE'                # the memo's 0.10 / just below 2x is not TRUE
        assert B1.b1_gate_verdict(0.10, self.GATE) == 'INCONCLUSIVE'
        assert B1.b1_gate_verdict(0.0652, self.GATE) == 'FALSE'
        assert B1.b1_gate_verdict(0.0478067425, self.GATE) == 'INCONCLUSIVE'        # just below the CI
        assert B1.b1_gate_verdict(0.04780674254466574, self.GATE) == 'FALSE'        # CI endpoints are inside
        assert B1.b1_gate_verdict(0.08498175581546036, self.GATE) == 'FALSE'
        assert B1.b1_gate_verdict(0.0851, self.GATE) == 'INCONCLUSIVE'

    def test_regions_for_rule_10m(self):
        assert B1.b1_gate_verdict(0.0113, self.RULE) == 'TRUE'
        assert B1.b1_gate_verdict(0.0108, self.RULE) == 'INCONCLUSIVE'
        assert B1.b1_gate_verdict(0.0056, self.RULE) == 'FALSE'
        assert B1.b1_gate_verdict(0.0, self.RULE) == 'INCONCLUSIVE'

    def test_overall_verdict_is_gate_v2_on_the_primary_comparison(self):
        per = {'primary_non_event': {'gate_v2': 'FALSE', 'rule_10m': 'TRUE'}, 'secondary_proxy': {'gate_v2': 'TRUE', 'rule_10m': 'TRUE'}}
        assert B1.b1_overall(per) == 'FALSE'


class TestSigmaRatioAndZeroPower:
    def test_sigma_f_ratio_matches_the_closed_form(self):
        # sigma_f = sqrt(Var(a) (1/n_pre + 1)) -> sigma_f(2) / sigma_f(n) = sqrt(1.5 / (1/n + 1))
        assert B1.sigma_f_ratio_closed_form(2) == pytest.approx(1.0)
        assert B1.sigma_f_ratio_closed_form(3) == pytest.approx(1.0606601717798212)   # F7's measured ratio

    def test_zero_power_rule(self):
        assert B1.is_vacuous({'0.15': 0.2, '0.3': 0.1, '0.5': 0.0}) is True
        assert B1.is_vacuous({'0.15': 0.0, '0.3': 0.0, '0.5': 0.01}) is False


# ================================================================ PASS 3 — B2: inventory width share ================================================================

def _gpkg_blob(geom, srs_id=4326, envelope=False):
    import struct
    from shapely import wkb
    flags = 0b00000001 | (0b0010 if envelope else 0)             # little-endian; envelope indicator 1 = [minx,maxx,miny,maxy]
    head = b'GP' + bytes([0, flags]) + struct.pack('<i', srs_id)
    if envelope:
        minx, miny, maxx, maxy = geom.bounds
        head += struct.pack('<4d', minx, maxx, miny, maxy)
    return head + wkb.dumps(geom, hex=False, byte_order=1)


def _make_gpkg(path, geoms, table='colombia_landslides', srs_id=4326):
    import sqlite3
    con = sqlite3.connect(path)
    con.execute('CREATE TABLE gpkg_contents (table_name TEXT, data_type TEXT, srs_id INTEGER, min_x REAL, min_y REAL, max_x REAL, max_y REAL)')
    con.execute(f'CREATE TABLE {table} (fid INTEGER PRIMARY KEY, geom BLOB)')
    con.execute('CREATE TABLE gpkg_geometry_columns (table_name TEXT, column_name TEXT, geometry_type_name TEXT, srs_id INTEGER)')
    con.execute('INSERT INTO gpkg_contents VALUES (?,?,?,?,?,?,?)', (table, 'features', srs_id, 0, 0, 1, 1))
    con.execute('INSERT INTO gpkg_geometry_columns VALUES (?,?,?,?)', (table, 'geom', 'MULTIPOLYGON', srs_id))
    for i, g in enumerate(geoms):
        con.execute(f'INSERT INTO {table} (fid, geom) VALUES (?,?)', (i + 1, _gpkg_blob(g, srs_id, envelope=(i % 2 == 0))))
    con.commit(); con.close()


class TestGpkgReader:
    def test_parses_blobs_with_and_without_envelope(self):
        from shapely.geometry import Polygon
        poly = Polygon([(0, 0), (4, 0), (4, 3), (0, 3)])
        for env in (False, True):
            g = B2.parse_gpkg_geometry(_gpkg_blob(poly, envelope=env))
            assert g.equals(poly) and g.area == 12.0

    def test_rejects_an_empty_geometry_blob(self):
        from shapely.geometry import Point
        blob = bytearray(_gpkg_blob(Point(1, 2)))
        blob[3] |= 0b00010000                                          # the 'empty' flag
        with pytest.raises(ValueError):
            B2.parse_gpkg_geometry(bytes(blob))

    def test_rejects_non_gpkg_bytes(self):
        with pytest.raises(ValueError):
            B2.parse_gpkg_geometry(b'XX\x00\x01\x00\x00\x00\x00')

    def test_reads_every_feature_with_its_srs(self, tmp_path):
        from shapely.geometry import Polygon, MultiPolygon
        geoms = [Polygon([(0, 0), (1, 0), (1, 1)]), MultiPolygon([Polygon([(5, 5), (6, 5), (6, 6)])]), Polygon([(9, 9), (10, 9), (10, 10)])]
        p = tmp_path / 'inv.gpkg'; _make_gpkg(p, geoms)
        out = B2.read_inventory(p, 'colombia_landslides')
        assert out['srs_id'] == 4326 and len(out['geometries']) == 3
        assert out['geometries'][0].equals(geoms[0])

    def test_verify_inventory_reports_every_failed_check(self, tmp_path):
        import hashlib
        from shapely.geometry import Polygon
        p = tmp_path / 'inv.gpkg'; _make_gpkg(p, [Polygon([(0, 0), (1, 0), (1, 1)])])
        good = hashlib.sha256(p.read_bytes()).hexdigest()
        assert B2.verify_inventory(p, good, 'colombia_landslides', 1)['ok'] is True
        bad = B2.verify_inventory(p, '0' * 64, 'colombia_landslides', 838)
        assert bad['ok'] is False and bad['checks']['sha256'] is False and bad['checks']['feature_count'] is False


class TestNestedGrid:
    def test_snapped_to_20m_and_nested_8_and_4(self):
        g = B2.nested_grid((3.0, 5.0, 47.0, 61.0))
        assert (g['x0'], g['y0']) == (0.0, 80.0)
        assert g['shape'] == {2.5: (32, 24), 10.0: (8, 6), 20.0: (4, 3)}
        assert g['transform'][2.5].a == 2.5 and g['transform'][2.5].e == -2.5 and g['transform'][2.5].c == g['x0']
        assert g['transform'][20.0].c == g['x0'] and g['transform'][20.0].f == g['y0']

    def test_origin_is_on_the_20m_lattice_even_when_a_10m_snap_would_not_be(self):
        for b in [(13.0, 5.0, 47.0, 61.0), (-27.0, -9.0, 41.2, 19.9), (1234567.0 + 3, 2.0, 1234600.0, 31.0)]:
            g = B2.nested_grid(b)
            assert g['x0'] % 20 == 0 and g['y0'] % 20 == 0
            assert g['x0'] <= b[0] and b[0] - g['x0'] < 20

    def test_covers_the_bounds(self):
        g = B2.nested_grid((-33.3, -7.7, 41.2, 19.9))
        assert g['x0'] <= -33.3 and g['y0'] >= 19.9
        assert g['x0'] + g['shape'][20.0][1] * 20 >= 41.2 and g['y0'] - g['shape'][20.0][0] * 20 <= -7.7


class TestRasterise:
    def test_pixel_centre_rule_on_nested_grids(self):
        from shapely.geometry import box
        g = B2.nested_grid((0.0, 0.0, 40.0, 40.0))
        r = B2.rasterise([box(1, 21, 14, 34)], g)                    # a 13 m square, away from every pixel-centre boundary
        assert int(r[2.5].sum()) == 36 and int(r[10.0].sum()) == 1 and int(r[20.0].sum()) == 1

    def test_a_polygon_that_misses_every_pixel_centre_burns_nothing_at_that_resolution(self):
        from shapely.geometry import box
        g = B2.nested_grid((0.0, 0.0, 40.0, 40.0))
        r = B2.rasterise([box(15, 21, 25, 31)], g)                   # 10 m square straddling two 20 m cells, centres at x = 10 and 30
        assert int(r[20.0].sum()) == 0                               # all_touched=True would burn 2-4 px
        assert int(r[2.5].sum()) == 16

    def test_rasterisation_is_direct_not_a_downsample(self):
        from shapely.geometry import box
        g = B2.nested_grid((0.0, 0.0, 40.0, 40.0))
        r = B2.rasterise([box(1, 21, 14, 34)], g)
        coarse_from_fine = r[2.5].reshape(4, 4, 4, 4).any(axis=(1, 3))        # 16 x 16 px at 2.5 m -> 4 x 4 at 10 m
        assert int(coarse_from_fine.sum()) > int(r[10.0].sum())


class TestRasterIoU:
    def test_nearest_upsampled_coarse_vs_fine(self):
        fine = np.zeros((8, 8), bool); fine[0:4, 0:4] = True; fine[0:2, 4:6] = True       # 16 + 4 px
        coarse = np.zeros((2, 2), bool); coarse[0, 0] = True                              # upsamples to the 4x4 block only
        out = B2.raster_iou(coarse, fine, 4)
        assert (out['intersection'], out['union']) == (16, 20) and out['iou'] == pytest.approx(0.8)

    def test_identical_rasters_give_one(self):
        a = np.zeros((4, 4), bool); a[1:3, 1:3] = True
        assert B2.raster_iou(a, a, 1)['iou'] == 1.0


class TestLocalThickness:
    """Per-pixel local thickness = the diameter of the largest inscribed disk that covers the pixel, in pixels
    (2 r - 1 with r the Euclidean distance to the background, as trustsr.alloc.component_width_m)."""

    def test_strip_of_known_width_has_that_thickness_everywhere(self):
        for w in (3, 7, 12):
            m = np.zeros((40, 80), bool); m[10:10 + w, 5:75] = True
            t = B2.local_thickness_px(m)
            inner = t[10:10 + w, 15:65]
            assert np.all(inner >= w - 1) and np.all(inner <= w + 1)            # +-1 px is the digital-disk ambiguity
            assert t[~m].max() == 0

    def test_odd_strip_widths_are_exact(self):
        m = np.zeros((30, 60), bool); m[10:17, 5:55] = True                     # 7 px wide
        assert np.all(B2.local_thickness_px(m)[10:17, 15:45] == 7.0)

    def test_disk_thickness_is_its_diameter(self):
        yy, xx = np.mgrid[-30:31, -30:31]
        m = (yy ** 2 + xx ** 2) <= 20 ** 2
        t = B2.local_thickness_px(m)
        assert 39 <= t[30, 30] <= 41 and np.all(t[m] >= 39 - 1)                 # every pixel lies under the central disk

    def test_l_shape_arms_have_the_arm_width_and_the_corner_is_at_most_a_bit_wider(self):
        w = 9
        m = np.zeros((60, 60), bool); m[40:40 + w, 5:55] = True; m[5:50, 5:5 + w] = True
        t = B2.local_thickness_px(m)
        assert t[44, 40] == w and t[20, 9] == w                                  # mid-arm pixels
        assert w <= t[m].max() <= 1.25 * w + 1                                   # the corner junction admits a slightly larger disk

    def test_matches_brute_force_on_random_blobs(self):
        from scipy import ndimage
        rng = np.random.default_rng(0)
        for _ in range(3):
            m = np.pad(ndimage.gaussian_filter(rng.random((20, 20)), 2) > 0.5, 3)   # blobs sit inside a background margin, as in the real raster
            r = ndimage.distance_transform_edt(m)
            brute = np.zeros(m.shape)
            for qy, qx in zip(*np.nonzero(m)):
                best = 0.0
                for py, px in zip(*np.nonzero(m)):
                    if np.hypot(qy - py, qx - px) <= r[py, px] + 1e-9:
                        best = max(best, 2 * r[py, px] - 1)
                brute[qy, qx] = best
            assert np.allclose(B2.local_thickness_px(m), brute)

    def test_empty_mask(self):
        assert B2.local_thickness_px(np.zeros((5, 5), bool)).max() == 0


class TestThinShare:
    def test_area_share_below_threshold_in_metres(self):
        thick = np.array([[0, 7, 7, 30], [0, 0, 30, 30]], float)            # px thickness
        mask = thick > 0
        s = B2.thin_share(thick, mask, px_m=2.5, threshold_m=20.0)          # 7 px = 17.5 m (<20), 30 px = 75 m
        assert (s['thin_px'], s['area_px']) == (2, 5) and s['share'] == pytest.approx(0.4)

    def test_threshold_is_strict(self):
        thick = np.array([[8.0, 8.0]]); mask = thick > 0                    # exactly 20 m is not "< 20 m"
        assert B2.thin_share(thick, mask, 2.5, 20.0)['thin_px'] == 0


class TestB2VerdictRule:
    """yaml B2.keep_rule: TRUE = thin share >= 0.30 AND IoU(10 m, 2.5 m) < 0.80; FALSE = share < 0.10 AND IoU >= 0.85."""

    def test_regions(self):
        assert B2.b2_verdict(0.30, 0.7999) == 'TRUE'
        assert B2.b2_verdict(0.30, 0.80) == 'INCONCLUSIVE'
        assert B2.b2_verdict(0.2999, 0.5) == 'INCONCLUSIVE'
        assert B2.b2_verdict(0.0999, 0.85) == 'FALSE'
        assert B2.b2_verdict(0.10, 0.9) == 'INCONCLUSIVE'
        assert B2.b2_verdict(0.05, 0.8499) == 'INCONCLUSIVE'


class TestB2AnalyseEndToEnd:
    """A synthetic inventory with a hand-derived answer. Polygon edges are 1 m inside pixel-centre boundaries so the
    pixel-centre rule is unambiguous:  strip x 1001..1011, y 5001..5401 (10 x 400 m = 4 x 160 px at 2.5 m = 640 px, thickness
    4 px = 10 m < 20 m);  square x 2001..2199, y 5001..5199 (198 m = 80 x 80 px at 2.5 m = 6400 px, thick)."""

    def _polys(self):
        from shapely.geometry import box
        return [box(1001, 5001, 1011, 5401), box(2001, 5001, 2199, 5199)]

    def test_area_share_and_rasterisation_iou_match_the_geometry(self):
        out = B2.analyse(self._polys(), {'replicates': 200, 'seed': 2024})
        s = out['shares']['local_thickness']
        assert s['area_px'] == 640 + 6400
        # the strip is entirely thin; the square's four sharp corner tips are also covered only by small inscribed disks, so a
        # few corner pixels are thin too (local thickness is a property of the covering disk, not of the polygon's side length)
        assert 640 <= s['thin_px'] <= 640 + 4 * 6
        assert s['share'] == pytest.approx(640 / 7040, abs=0.004)
        # IoU is between RASTERS (nearest-upsampled coarse vs the 2.5 m raster), not against the polygon. The 2.5 m raster covers the
        # strip at x 1000..1010 (4 px) and the square at 2000..2200: 4000 + 40000 m2. The 10 m raster covers exactly the same cells
        # (IoU 1); the 20 m raster widens the strip to 1000..1020: 8000 + 40000 m2, so IoU = 44000 / 48000.
        assert out['rasterisation_iou']['10.0']['iou'] == pytest.approx(1.0)
        assert out['rasterisation_iou']['20.0']['iou'] == pytest.approx(44000 / 48000)
        assert out['rasterisation_iou']['20.0']['iou'] < out['rasterisation_iou']['10.0']['iou']
        assert out['connected_components_2p5m'] == 2

    def test_a_polygon_thinner_than_a_coarse_pixel_is_lost_at_that_resolution(self):
        from shapely.geometry import box
        out = B2.analyse([box(1001, 5001, 1008, 5201)], {'replicates': 100, 'seed': 2024})     # 7 m x 200 m
        assert out['rasterisation_iou']['20.0']['iou'] == 0.0           # no 20 m pixel centre falls inside: the feature is lost
        assert out['rasterisation_iou']['10.0']['iou'] > 0.0
        assert out['shares']['local_thickness']['share'] == 1.0
