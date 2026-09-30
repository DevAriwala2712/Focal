"""Unit tests for trustsr.alloc and the F5/A8 pipeline wiring (synthetic fixtures only, no data or model needed)."""
from __future__ import annotations

import numpy as np
import pytest

from trustsr import alloc, hrbench as hb
from trustsr.gate_v2 import unmix_fraction


# ---------------------------------------------------------------- block plumbing

def test_blocks_roundtrip():
    a = np.arange(8 * 12).reshape(8, 12).astype(float)
    assert np.array_equal(alloc._unblocks(alloc._blocks(a, 4), 4), a)


def test_blocks_groups_the_right_subpixels():
    a = np.zeros((8, 8))
    a[0:4, 4:8] = 7.0
    b = alloc._blocks(a, 4)
    assert b.shape == (2, 2, 16)
    assert np.array_equal(b[0, 1], np.full(16, 7.0))
    assert b[0, 0].sum() == 0


def test_block_fraction_oracle_counts_subpixels():
    truth = np.zeros((4, 8), bool)
    truth[0, 0:3] = True            # 3 of the 16 sub-pixels of block (0,0)
    f = alloc.block_fraction_oracle(truth, 4)
    assert f.shape == (1, 2)
    assert f[0, 0] == pytest.approx(3 / 16)
    assert f[0, 1] == 0.0


# ---------------------------------------------------------------- allocation

def test_allocate_by_rank_respects_the_block_sum_and_picks_the_lowest():
    score = np.arange(16).reshape(4, 4).astype(float)
    out = alloc.allocate_by_rank(score, np.array([[5 / 16]]), 4)
    assert out.sum() == 5
    assert np.array_equal(np.sort(score[out]), np.arange(5))


def test_allocate_by_rank_ranks_non_finite_last():
    score = np.full((4, 4), np.nan)
    score.ravel()[:2] = [0.5, 0.1]
    out = alloc.allocate_by_rank(score, np.array([[2 / 16]]), 4)
    assert out.sum() == 2
    assert out.ravel()[0] and out.ravel()[1]


def test_allocate_by_rank_rejects_mismatched_fraction_grid():
    with pytest.raises(ValueError):
        alloc.allocate_by_rank(np.zeros((8, 8)), np.zeros((1, 1)), 4)


def test_sr_and_bilinear_ranking_give_provably_different_iou():
    """Sanity: the pipeline CAN detect a ranking difference when one exists.

    One 10 m block, half non-target / half target. The oracle fraction is 8/16 for both methods, so the ONLY thing that
    can differ is the ranking. `good` ranks the true target sub-pixels lowest (perfect IoU); `bad` ranks the opposite
    ones lowest (zero IoU). If the harness ever reported these as equal it would be unable to see a real effect.
    """
    truth = np.zeros((4, 4), bool)
    truth[:, :2] = True
    good = np.where(truth, 0.0, 1.0)
    bad = np.where(truth, 1.0, 0.0)
    frac = alloc.block_fraction_oracle(truth, 4)
    valid = np.ones((4, 4), bool)
    m_good = alloc.allocate_by_rank(good, frac, 4)
    m_bad = alloc.allocate_by_rank(bad, frac, 4)
    assert alloc.iou(m_good, truth, valid)['iou'] == 1.0
    assert alloc.iou(m_bad, truth, valid)['iou'] == 0.0
    assert alloc.iou(m_good, truth, valid)['iou'] - alloc.iou(m_bad, truth, valid)['iou'] == 1.0


def test_oracle_fraction_is_at_least_as_good_as_a_wrong_realistic_fraction():
    """Oracle has strictly more information: with the same (perfect) ranking it cannot do worse."""
    rng = np.random.default_rng(0)
    truth = rng.random((16, 16)) < 0.4
    valid = np.ones((16, 16), bool)
    score = np.where(truth, 0.0, 1.0) + 1e-6 * rng.random((16, 16))   # a ranking that knows the target
    oracle = alloc.block_fraction_oracle(truth, 4)
    realistic = np.clip(oracle + 0.25, 0, 1)                          # a biased estimate of the same fraction
    iou_oracle = alloc.iou(alloc.allocate_by_rank(score, oracle, 4), truth, valid)['iou']
    iou_real = alloc.iou(alloc.allocate_by_rank(score, realistic, 4), truth, valid)['iou']
    assert iou_oracle == 1.0
    assert iou_oracle >= iou_real
    assert iou_real < 1.0


# ---------------------------------------------------------------- baseline methods

def test_blocky_mask_has_no_subpixel_structure():
    s10 = np.array([[0.1, 0.9]])
    out = alloc.blocky_mask(s10, 0.5, 4)
    assert out.shape == (4, 8)
    assert out[:, :4].all() and not out[:, 4:].any()


def test_hard_threshold_v1_style_is_gated_by_the_parent():
    s10 = np.array([[0.9]])                      # parent says "not target"
    s_sr = np.zeros((4, 4))                      # sub-pixels say "target"
    assert not alloc.hard_threshold_v1_style(s_sr, s10, 0.5, 4).any()
    s10 = np.array([[0.1]])                      # parent agrees -> the sub-pixel test decides
    s_sr = np.array([[0.0, 0.0, 1.0, 1.0]] * 4)
    out = alloc.hard_threshold_v1_style(s_sr, s10, 0.5, 4)
    assert out[:, :2].all() and not out[:, 2:].any()


# ---------------------------------------------------------------- metrics

def test_iou_and_area_error_denominators():
    truth = np.zeros((4, 4), bool)
    truth[0, :2] = True
    pred = np.zeros((4, 4), bool)
    pred[0, 1:3] = True
    valid = np.ones((4, 4), bool)
    r = alloc.iou(pred, truth, valid)
    assert (r['intersection'], r['union'], r['iou']) == (1, 3, pytest.approx(1 / 3))
    a = alloc.area_error(pred, truth, valid)
    assert (a['pred_px'], a['truth_px'], a['relative']) == (2, 2, 0.0)


def test_iou_respects_the_valid_mask():
    truth = np.zeros((4, 4), bool)
    pred = np.zeros((4, 4), bool)
    pred[3, 3] = True
    valid = np.ones((4, 4), bool)
    valid[3, 3] = False
    assert alloc.iou(pred, truth, valid)['union'] == 0


def test_boundary_f1_perfect_and_disjoint():
    truth = np.zeros((20, 20), bool)
    truth[5:15, 5:15] = True
    valid = np.ones((20, 20), bool)
    assert alloc.boundary_f1(truth, truth, valid, 1)['f1'] == pytest.approx(1.0)
    shifted = np.zeros((20, 20), bool)
    shifted[5:15, 6:16] = True
    r1 = alloc.boundary_f1(shifted, truth, valid, 1)
    r2 = alloc.boundary_f1(shifted, truth, valid, 2)
    assert 0.0 < r1['f1'] <= r2['f1'] <= 1.0
    empty = np.zeros((20, 20), bool)
    assert alloc.boundary_f1(empty, truth, valid, 1)['f1'] == 0.0
    assert alloc.boundary_f1(empty, empty, valid, 1)['f1'] is None


def test_area_error_is_none_without_truth():
    z = np.zeros((4, 4), bool)
    assert alloc.area_error(z, z, np.ones((4, 4), bool))['relative'] is None


# ---------------------------------------------------------------- width stratification

def test_component_width_bins_a_synthetic_shape():
    """A 1-px-wide bar (2.5 m), an 11-px bar (27.5 m) and a 41-px block (102.5 m) land in three different bins."""
    m = np.zeros((200, 200), bool)
    m[10, 10:60] = True                 # width 1 px  -> 2.5 m   -> '0-20'
    m[40:51, 10:120] = True             # width 11 px -> 27.5 m  -> '20-50'
    m[100:141, 10:141] = True           # width 41 px -> 102.5 m -> '50-150'
    width, labels, per = alloc.component_width_m(m, 2.5)
    assert labels.max() == 3
    assert width[10, 30] == pytest.approx(2.5)
    assert width[45, 60] == pytest.approx(27.5)
    assert width[120, 70] == pytest.approx(102.5)
    bins = alloc.width_bin_masks(width, m)
    assert bins['0-20'][10, 30] and not bins['20-50'][10, 30]
    assert bins['20-50'][45, 60] and not bins['0-20'][45, 60]
    assert bins['50-150'][120, 70]
    assert not bins['150+'].any()
    assert sum(int(b.sum()) for b in bins.values()) == int(m.sum())


def test_width_bin_edges_are_the_preregistered_ones():
    assert alloc.WIDTH_BIN_NAMES == ('0-20', '20-50', '50-150', '150+')
    assert alloc.WIDTH_BIN_EDGES_M == (0.0, 20.0, 50.0, 150.0, float('inf'))


def test_stratum_iou_ignores_predictions_far_from_the_stratum():
    truth = np.zeros((40, 40), bool)
    truth[10:14, 10:14] = True
    pred = truth.copy()
    pred[30:34, 30:34] = True           # a false positive far away, outside the dilated region
    valid = np.ones((40, 40), bool)
    assert alloc.stratum_iou(pred, truth, valid, 4)['iou'] == pytest.approx(1.0)
    near = truth.copy()
    near[10:14, 15:19] = True           # a false positive 1 px away; cols 15-17 fall inside the dilated region
    assert alloc.stratum_iou(near, truth, valid, 4)['iou'] == pytest.approx(16 / 28)
    assert alloc.stratum_iou(pred, np.zeros((40, 40), bool), valid, 4)['iou'] is None


# ---------------------------------------------------------------- unmixing

def test_unmix_fraction_static_recovers_a_known_mixture():
    e_t = np.array([0.30, 0.25, 0.20, 0.25])
    e_o = np.array([0.03, 0.05, 0.03, 0.40])
    a_true = np.array([[0.0, 0.25], [0.5, 1.0]])
    y = np.moveaxis(e_o + a_true[..., None] * (e_t - e_o), -1, 0)
    frac, resid = alloc.unmix_fraction_static(y, e_t, e_o, (0, 3))
    assert frac == pytest.approx(a_true, abs=1e-9)
    assert resid == pytest.approx(np.zeros_like(a_true), abs=1e-12)


def test_unmix_fraction_static_agrees_with_gate_v2_on_a_matched_case():
    """gate_v2.unmix_fraction is the pre/post CHANGE estimator; this is its single-observation projection.

    With a pre stack of pure vegetation (abundance 1) the change estimator's loss fraction is 1 - a_post, which must
    equal 1 - (the static vegetation abundance) pixel for pixel.
    """
    e_v = np.array([0.03, 0.05, 0.03, 0.40])
    e_b = np.array([0.30, 0.25, 0.20, 0.25])
    rng = np.random.default_rng(3)
    a_post = rng.random((5, 6))
    y_post = e_b + a_post[..., None] * (e_v - e_b)              # (H,W,4), gate_v2 layout
    y_pre = np.broadcast_to(e_v, (2,) + y_post.shape).copy()
    f_change, _ = unmix_fraction(y_pre, y_post, e_v, e_b, bands_indices=(0, 3))
    a_static, _ = alloc.unmix_fraction_static(np.moveaxis(y_post, -1, 0), e_v, e_b, (0, 3))
    assert f_change == pytest.approx(1.0 - a_static, abs=1e-9)


def test_endmembers_from_score_use_only_the_observation():
    refl = np.stack([np.linspace(0, 1, 100).reshape(10, 10)] * 4)
    score = refl[0]
    e_t, e_o = alloc.endmembers_from_score(refl, score, quantile=0.1)
    assert (e_t < e_o).all()


def test_block_fraction_realistic_never_leaves_the_unit_interval():
    rng = np.random.default_rng(1)
    refl = rng.random((4, 16, 16)) * 0.5
    score = alloc.ndvi_rgbn(refl)
    frac, diag = alloc.block_fraction_realistic(refl, score, None, (0, 3))
    assert frac.shape == (16, 16)
    assert (frac >= 0).all() and (frac <= 1).all()
    assert diag['band_indices'] == [0, 3]


# ---------------------------------------------------------------- indices and bilinear resampling

def test_ndvi_and_ndwi_band_order():
    refl = np.zeros((4, 1, 1))
    refl[:, 0, 0] = [0.1, 0.2, 0.05, 0.5]            # B04, B03, B02, B08
    assert alloc.ndvi_rgbn(refl)[0, 0] == pytest.approx((0.5 - 0.1) / 0.6)
    assert alloc.ndwi_rgbn(refl)[0, 0] == pytest.approx((0.2 - 0.5) / 0.7)


def test_bilinear_resample_is_available_but_not_an_a2_baseline():
    assert 'bilinear' in hb.RESAMPLE_METHODS
    assert 'bilinear' not in hb.BASELINES                     # A2's scored baseline set must not change
    x = np.arange(5, dtype=float).reshape(1, 1, 5)
    out = hb.resample(x, 4, 'bilinear')
    assert out.shape == (1, 4, 20)
    # half-pixel-centre geometry: output column j samples the input at (j + 0.5)/4 - 0.5, exactly reproduced for a ramp
    j = np.arange(4, 16)
    assert out[0, 0, 4:16] == pytest.approx((j + 0.5) / 4 - 0.5)
    assert hb.block_mean(out, 4)[0, :, 1:4] == pytest.approx(x[0, :, 1:4])   # area-preserving away from the clamped edge
    flat = np.full((1, 4, 4), 0.7)
    assert hb.resample(flat, 4, 'bilinear') == pytest.approx(np.full((1, 16, 16), 0.7))
