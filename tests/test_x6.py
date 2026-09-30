"""CPU tests for the X6 helpers (season matching and latency). Synthetic rasters only; written before the helpers."""
import itertools

import numpy as np
import pytest
from affine import Affine

from experiments import x6_season_latency as X


# ---- audit / date helpers ------------------------------------------------------------------

def _row(date, cloud, cov=100.0):
    return {'date': date, 'cloud_shadow_pct_aoi': cloud, 'coverage_pct': cov}


def test_clear_dates_in_window_applies_window_and_clear_rule_inclusively():
    rows = [_row('2023-10-31', 1), _row('2023-11-01', 10.0), _row('2023-11-11', 10.01), _row('2023-11-21', 2, cov=98.9),
            _row('2023-12-31', 0.0), _row('2024-01-05', 0.0)]
    got = X.clear_dates_in_window(rows, '2023-11-01', '2023-12-31', 10.0, 99.0)
    assert got == ['2023-11-01', '2023-12-31']          # boundary dates in, >10 % out, <99 % coverage out, outside window out


def test_clear_dates_in_window_uses_best_row_per_date():
    rows = [_row('2023-12-01', 50), _row('2023-12-01', 3)]
    assert X.clear_dates_in_window(rows, '2023-11-01', '2023-12-31', 10, 99) == ['2023-12-01']


def test_gap_days_matches_the_diagnosis_numbers():
    assert X.gap_days('2024-07-30', '2024-12-06') == 129
    assert X.gap_days('2024-07-30', '2024-11-06') == 99


def test_stable_min_valid_allows_two_missing_like_e6():
    assert X.stable_min_valid(12, 2) == 10              # E6: 10 of 12
    assert X.stable_min_valid(6, 2) == 4
    assert X.stable_min_valid(2, 2) == 1                # never below one valid date


# ---- composite -----------------------------------------------------------------------------

def test_first_valid_composite_earliest_date_wins_and_none_is_255():
    a = np.array([[1, 1, 0, 0]], bool)
    b = np.array([[1, 0, 1, 0]], bool)
    c = np.array([[0, 0, 1, 0]], bool)
    src = X.first_valid_composite([a, b, c])
    assert src.dtype == np.uint8
    assert src.tolist() == [[0, 0, 1, 255]]             # px0: a and b valid -> a; px2: b and c valid -> b; px3: none


def test_select_by_source_picks_the_values_of_the_source_date():
    layers = np.stack([np.full((2, 2), 10), np.full((2, 2), 20), np.full((2, 2), 30)])
    src = np.array([[0, 1], [2, 255]], np.uint8)
    out = X.select_by_source(layers, src, fill=-1)
    assert out.tolist() == [[10, 20], [30, -1]]


def test_select_by_source_handles_band_axis():
    layers = np.stack([np.zeros((4, 2, 2)), np.ones((4, 2, 2))])
    src = np.array([[0, 1], [1, 255]], np.uint8)
    out = X.select_by_source(layers, src, fill=7)
    assert out.shape == (4, 2, 2)
    assert out[0].tolist() == [[0, 1], [1, 7]]


def test_footprint_validity_counts_only_inside_the_footprint():
    src = np.array([[0, 255, 255], [1, 2, 255]], np.uint8)
    footprint = np.array([[1, 1, 0], [1, 1, 0]], bool)
    frac, valid, total = X.footprint_validity(src, footprint)
    assert (valid, total) == (3, 4) and frac == pytest.approx(0.75)


def test_latest_contributing_date_ignores_dates_that_only_contribute_outside_the_footprint():
    dates = ['2024-10-27', '2024-11-01', '2024-11-06']
    src = np.array([[0, 1, 2], [0, 0, 255]], np.uint8)
    footprint_in = np.array([[1, 1, 1], [1, 1, 0]], bool)
    assert X.latest_contributing_date(src, footprint_in, dates)['date'] == '2024-11-06'
    footprint_out = np.array([[1, 1, 0], [1, 1, 0]], bool)   # the 11-06 pixel is outside
    got = X.latest_contributing_date(src, footprint_out, dates)
    assert got['date'] == '2024-11-01'
    assert got['footprint_px_by_date'] == {'2024-10-27': 3, '2024-11-01': 1, '2024-11-06': 0}


def test_gap_distribution_is_per_footprint_pixel():
    dates = ['2024-10-27', '2024-11-01', '2024-11-06']
    src = np.array([[0, 1, 2, 255]], np.uint8)
    fp = np.ones((1, 4), bool)
    d = X.pixel_gap_distribution(src, fp, dates, '2024-07-30')
    assert d['median'] == 94 and d['max'] == 99 and d['min'] == 89 and d['n'] == 3


def test_keep_rules_at_the_boundaries():
    assert X.keep_rule_a(0.008, 0.0095, hi=-0.0001) is True
    assert X.keep_rule_a(0.008, 0.0095, hi=0.0) is False      # CI touching 0 is not excluded
    assert X.keep_rule_a(0.0100, 0.0095, hi=-0.001) is False  # point estimate not lower
    assert X.keep_rule_b(0.90, 99, 0.90, 99) is True
    assert X.keep_rule_b(0.8999, 99, 0.90, 99) is False
    assert X.keep_rule_b(0.95, 100, 0.90, 99) is False


# ---- geometry ---------------------------------------------------------------------------

def test_window_transform_stays_on_the_10m_lattice():
    t = Affine(10.0, 0, 500020.0, 0, -10.0, 1271000.0)
    w = X.window_transform(t, rows=(256, 896), cols=(128, 640))
    assert w.a == 10.0 and w.e == -10.0
    assert w.c == 500020.0 + 128 * 10 and w.f == 1271000.0 - 256 * 10
    assert ((w.c - t.c) / 10) == 128 and ((t.f - w.f) / 10) == 256


# ---- paired false-drop metric --------------------------------------------------------------

def _scene(seed=0, n=128, offset_a=0.0, offset_b=0.0, noise=0.01):
    rng = np.random.default_rng(seed)
    base = 0.75 + 0.02 * rng.standard_normal((n, n))
    post = base + noise * rng.standard_normal((n, n))
    pre_a = base + offset_a + noise * rng.standard_normal((n, n))
    pre_b = base + offset_b + noise * rng.standard_normal((n, n))
    return pre_a.astype('float32'), pre_b.astype('float32'), post.astype('float32')


def test_paired_false_drop_shared_denominator_and_ci_excludes_zero_for_a_real_offset():
    pre_a, pre_b, post = _scene(offset_a=0.0, offset_b=0.12)        # B's pre is systematically too high vs post
    stable = np.ones(post.shape, bool)
    r = X.paired_false_drop(pre_a, pre_b, post, stable, threshold=0.10, block=32, replicates=400, ci=0.95, seed=3)
    assert r['comparable_pixels'] == post.size
    assert r['fraction_a'] < r['fraction_b']
    assert r['difference']['hi'] < 0 and r['difference']['estimate'] < 0
    assert r['difference']['denominator'] == post.size


def test_paired_false_drop_ci_straddles_zero_when_pools_are_equivalent():
    pre_a, pre_b, post = _scene(seed=5, offset_a=0.0, offset_b=0.0)
    stable = np.ones(post.shape, bool)
    r = X.paired_false_drop(pre_a, pre_b, post, stable, threshold=0.05, block=32, replicates=400, ci=0.95, seed=3)
    assert r['difference']['lo'] <= 0 <= r['difference']['hi']


def test_paired_false_drop_only_counts_stable_and_finite_pixels():
    pre_a, pre_b, post = _scene()
    post[:16] = np.nan
    stable = np.ones(post.shape, bool)
    stable[:, :16] = False
    r = X.paired_false_drop(pre_a, pre_b, post, stable, threshold=0.0, block=32, replicates=50, ci=0.95, seed=1)
    assert r['comparable_pixels'] == (128 - 16) * (128 - 16)


def test_three_date_subsets_are_all_combinations_in_order():
    assert X.three_date_subsets(4) == [list(c) for c in itertools.combinations(range(4), 3)]
    assert X.three_date_subsets(3) == [[0, 1, 2]]
    assert X.three_date_subsets(2) == []


def test_stable_area_excludes_published_disks_and_unstable_pixels():
    n = 64
    t = Affine(10.0, 0, 0.0, 0, -10.0, 640.0)
    stack = np.full((4, n, n), 0.7, np.float32)
    stack[:, 0:8, 0:8] += np.array([0, 0.3, -0.3, 0.3], np.float32)[:, None, None]      # high variance block
    stack[0, 60:, 60:] = np.nan                                                        # only 3 valid dates: still stable (>= 2 min)
    stable, suspected = X.stable_pixels(stack, t, [(320.0, 320.0)], radius_m=50.0, min_valid=3, max_std=0.05)
    assert not stable[0:8, 0:8].any()                   # high std excluded
    assert not stable[n // 2, n // 2]                   # inside the disk
    assert suspected[n // 2, n // 2]
    assert stable[20, 20]
    assert stable[61, 61]


def test_sha256_file_is_content_addressed(tmp_path):
    p = tmp_path / 'a.bin'
    p.write_bytes(b'abc')
    assert X.sha256_file(p) == 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'


def test_byte_cap_stops_before_the_next_read():
    from experiments.e6_season_matched import ByteBudget
    used = {'v': 0}
    b = ByteBudget(1000, counter=lambda: used['v'])
    used['v'] = 999
    b.check()
    used['v'] = 1001
    with pytest.raises(RuntimeError, match='byte cap'):
        b.check()
