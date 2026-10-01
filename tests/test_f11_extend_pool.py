# tests/test_f11_extend_pool.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from experiments.f11_extend_pool import run_sr_for_dates

ROOT = Path(__file__).resolve().parent.parent


def test_run_sr_for_dates_returns_one_array_per_date(monkeypatch):
    """Does not hit the network or run the real model -- patches the per-date SR step to a small
    deterministic stand-in and checks the orchestration (shape, keys, dtype) is correct."""
    from experiments import f11_extend_pool as mod

    called_with = []

    def fake_sr_one_date(cfg, root, cache, date, crop, plan, runner):
        called_with.append(date)
        hc, wc = crop[1] - crop[0], crop[3] - crop[2]
        scale = cfg['tiling']['scale']
        return np.zeros((8, hc * scale, wc * scale), dtype=np.float32)

    monkeypatch.setattr(mod, '_sr_one_date', fake_sr_one_date)
    monkeypatch.setattr(mod, '_build_runner_and_plan', lambda cfg, root, crop: (None, [{'r0': 0, 'c0': 0}]))

    cfg = {'tiling': {'scale': 4, 'tile': 128, 'crop_margin_px': 16, 'stride': 96}}
    out = run_sr_for_dates(cfg, ROOT, ROOT, ['2024-02-05', '2024-02-10'], (0, 10, 0, 10))

    assert set(out.keys()) == {'2024-02-05', '2024-02-10'}
    assert called_with == ['2024-02-05', '2024-02-10']
    for arr in out.values():
        assert arr.shape == (8, 40, 40)
        assert arr.dtype == np.float32


def test_cache_write_matches_existing_dihedral_format(tmp_path):
    """The written .npy file must load back as (8, H, W) float32 -- the exact format
    trustsr.placebo_v2._load_dihedral already expects, so no downstream code needs a format branch."""
    from experiments.f11_extend_pool import write_dihedral_cache

    arr = np.random.rand(8, 20, 20).astype(np.float32)
    out_path = write_dihedral_cache(tmp_path, '2024-02-05', arr)

    assert out_path.name == '2024-02-05_dihedral_means.npy'
    loaded = np.load(out_path)
    assert loaded.shape == (8, 20, 20)
    assert loaded.dtype == np.float32
    assert np.array_equal(loaded, arr)
