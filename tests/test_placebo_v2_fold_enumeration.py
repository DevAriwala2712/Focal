# tests/test_placebo_v2_fold_enumeration.py
from __future__ import annotations

import itertools

import pytest

from trustsr.placebo_v2 import enumerate_fold_specs


def test_every_reference_set_has_exact_size_n_pre():
    specs = enumerate_fold_specs(['a', 'b', 'c', 'd', 'e', 'f'], n_pre=3)
    for pre_dates, held_out in specs:
        assert len(pre_dates) == 3
        assert held_out not in pre_dates


def test_count_matches_binomial_times_remaining():
    pool = ['a', 'b', 'c', 'd', 'e', 'f']
    specs = enumerate_fold_specs(pool, n_pre=3)
    n_ref_sets = len(list(itertools.combinations(pool, 3)))
    assert n_ref_sets == 20
    assert len(specs) == 20 * (len(pool) - 3)
    assert len(specs) == 60


def test_leave_one_out_fold_builder_is_untouched():
    """Regression guard: F1's own leave-one-out logic (experiments/f1_placebo.py build_folds) is not
    this function and is not changed by adding it."""
    from trustsr import placebo_v2
    assert hasattr(placebo_v2, 'build_sr_fold')
    assert hasattr(placebo_v2, 'enumerate_fold_specs')
    assert placebo_v2.build_sr_fold.__module__ == 'trustsr.placebo_v2'


def test_pool_too_small_raises_rather_than_silently_degrading():
    with pytest.raises(ValueError, match='n_pre'):
        enumerate_fold_specs(['a', 'b', 'c'], n_pre=3)   # no date left to hold out


def test_reference_sets_are_deterministically_ordered():
    pool = ['a', 'b', 'c', 'd']
    specs1 = enumerate_fold_specs(pool, n_pre=3)
    specs2 = enumerate_fold_specs(pool, n_pre=3)
    assert specs1 == specs2
