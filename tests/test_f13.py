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
