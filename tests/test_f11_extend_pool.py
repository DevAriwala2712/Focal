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


def test_main_cache_detection_skips_already_cached_candidate(tmp_path, monkeypatch):
    """Regression test for the exact logic that broke: main()'s existing/candidates/need
    computation. Pre-creates .npy stand-ins for the 3 original EXISTING_SR_POOL dates plus one
    candidate date ('2024-02-05'), then proves that candidate is recognized as already-done --
    excluded from `candidates` and never passed to run_sr_for_dates -- while the other, uncached
    candidates are still processed."""
    import yaml

    from experiments import f11_extend_pool as mod

    cache_dir = tmp_path / 'per_date_ndvi'
    cache_dir.mkdir(parents=True)

    candidate_dates = ['2024-02-05', '2024-02-10', '2024-02-15']
    already_cached_candidate = '2024-02-05'

    # Pre-create cache files for the 3 original pool dates + 1 candidate already "cached".
    for d in mod.EXISTING_SR_POOL + [already_cached_candidate]:
        np.save(cache_dir / f'{d}_dihedral_means.npy', np.zeros((8, 2, 2), dtype=np.float32))

    f11_config = {
        'pool_extension': {
            'minimum_n': 4,
            'target_n': 6,
            'candidate_dates': candidate_dates,
            'crop': [0, 10, 0, 10],
        }
    }
    config_path = tmp_path / 'f11_f12.yaml'
    config_path.write_text(yaml.safe_dump(f11_config), encoding='utf-8')

    monkeypatch.setattr(mod, 'CACHE_DIR', cache_dir)
    monkeypatch.setattr(mod, 'FIX11_CONFIG', config_path)
    monkeypatch.setattr(mod, 'digest', lambda path: 'deadbeef')
    monkeypatch.setattr(mod.C, 'load', lambda: ({}, tmp_path, 'hash'))
    monkeypatch.setattr(mod.C, 'cache', lambda cfg, root: tmp_path)
    monkeypatch.setattr(mod, 'write_json', lambda path, value: None)

    sr_calls = []

    def fake_run_sr_for_dates(cfg, root, cache, dates, crop):
        sr_calls.append(list(dates))
        out = {}
        for d in dates:
            arr = np.zeros((8, 2, 2), dtype=np.float32)
            mod.write_dihedral_cache(cache_dir, d, arr)
            out[d] = arr
        return out

    monkeypatch.setattr(mod, 'run_sr_for_dates', fake_run_sr_for_dates)

    result = mod.main(verbose=False)

    # The already-cached candidate must be recognized as existing, not re-processed.
    assert already_cached_candidate in result['existing_pool']
    processed_dates = [p['date'] for p in result['newly_processed']]
    assert already_cached_candidate not in processed_dates
    assert all(already_cached_candidate not in call for call in sr_calls)

    # The 3 original dates + the 1 cached candidate = 4 existing pool dates.
    assert sorted(result['existing_pool']) == sorted(mod.EXISTING_SR_POOL + [already_cached_candidate])

    # Only the genuinely uncached candidates should ever be passed to run_sr_for_dates.
    for call in sr_calls:
        for d in call:
            assert d in {'2024-02-10', '2024-02-15'}


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
