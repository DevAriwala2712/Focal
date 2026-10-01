"""F11 step 1: extend the SR-dependent placebo pool from 3 to >= 4 (target 6) real dates.

Pre-registered in configs/f11_f12.yaml `pool_extension`. Reuses the exact low-level SR primitives
experiments/f7_wayanad_v2.py::run_real_sr uses (Runner, tile_plan, sr_variants, fill_invalid,
sr_ndvi), but without that function's pre/post-date Welford split -- every date here is just another
pool date, no event involved. Writes each date's (8, H, W) float32 dihedral NDVI to
data/experiments-cache/wayanad_evidence/per_date_ndvi/<date>_dihedral_means.npy, the exact format
trustsr.placebo_v2._load_dihedral already reads.

Run: `python -m experiments.f11_extend_pool` from the repo root.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from experiments.wayanad_evidence import config as C
from experiments.wayanad_evidence.data import load_date
from experiments.wayanad_evidence.step3_sr import fill_invalid
from experiments.wayanad_evidence.stats import ndvi as sr_ndvi
from experiments.wayanad_evidence.tiler import tile_plan
from risk.common import digest, write_json

ROOT = Path(__file__).resolve().parent.parent
FIX11_CONFIG = ROOT / 'configs' / 'f11_f12.yaml'
EXISTING_SR_POOL = ['2024-01-16', '2024-01-21', '2024-01-26']
CACHE_DIR = ROOT / 'data' / 'experiments-cache' / 'wayanad_evidence' / 'per_date_ndvi'


def write_dihedral_cache(cache_dir: Path, date: str, arr: np.ndarray) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f'{date}_dihedral_means.npy'
    np.save(out_path, arr.astype(np.float32))
    return out_path


def _build_runner_and_plan(cfg, root, crop):
    from experiments.wayanad_evidence.sr import Runner
    tile, margin, stride = cfg['tiling']['tile'], cfg['tiling']['crop_margin_px'], cfg['tiling']['stride']
    hc, wc = crop[1] - crop[0], crop[3] - crop[2]
    plan = tile_plan((hc, wc), tile, margin, stride)
    runner = Runner(cfg, root)
    return runner, plan


def _sr_one_date(cfg, root, cache, date, crop, plan, runner):
    """Real tiled SR + 8-dihedral for one date on the given crop. No pre/post distinction."""
    from experiments.wayanad_evidence.sr import sr_variants
    scale = cfg['tiling']['scale']
    floor = cfg['radiometry']['min_denominator']
    r0c, r1c, c0c, c1c = crop
    hc, wc = r1c - r0c, c1c - c0c
    a = load_date(cfg, cache, date)
    refl = a['refl'][:, r0c:r1c, c0c:c1c]
    model_in = fill_invalid(refl, a['valid_scl'][r0c:r1c, c0c:c1c])
    nd_runs = np.zeros((8, hc * scale, wc * scale), dtype=np.float32)
    for n, t in enumerate(plan):
        r0, c0 = t['r0'], t['c0']
        (rl, rh), (cl, ch) = t['keep_r'], t['keep_c']
        v = sr_variants(runner, model_in[:, r0:r0 + cfg['tiling']['tile'], c0:c0 + cfg['tiling']['tile']],
                        f'{date} tile#{n}', runner.oom_type, runner.cleanup, runner.memory)
        hr = (slice((rl - r0) * scale, (rh - r0) * scale), slice((cl - c0) * scale, (ch - c0) * scale))
        vv = v[:, :, hr[0], hr[1]]
        out_region = (slice(rl * scale, rh * scale), slice(cl * scale, ch * scale))
        nd = sr_ndvi(vv[:, 0], vv[:, 3], floor)
        for k in range(8):
            nd_runs[k, out_region[0], out_region[1]] = nd[k]
    return nd_runs


def run_sr_for_dates(cfg, root, cache, dates: list[str], crop: tuple[int, int, int, int]) -> dict[str, np.ndarray]:
    """Real tiled SR + 8-dihedral for every date in `dates`, no pre/post split. Returns
    {date: (8, H, W) float32 SR NDVI}."""
    runner, plan = _build_runner_and_plan(cfg, root, crop)
    out = {}
    for date in dates:
        out[date] = _sr_one_date(cfg, root, cache, date, crop, plan, runner)
    return out


def main(verbose=True):
    started = datetime.now(timezone.utc).isoformat()
    f11_cfg = yaml.safe_load(FIX11_CONFIG.read_text(encoding='utf-8'))
    pe = f11_cfg['pool_extension']
    crop = tuple(pe['crop'])

    cfg_ev, ev_root, ev_hash = C.load()
    cache = C.cache(cfg_ev, ev_root)

    existing = [d for d in EXISTING_SR_POOL + pe['candidate_dates']
                if (CACHE_DIR / f'{d}_dihedral_means.npy').is_file()]
    need = pe['target_n'] - len(existing)
    candidates = [d for d in pe['candidate_dates'] if d not in existing]

    processed, skipped = [], []
    budget_seconds = 45 * 60
    t0 = time.perf_counter()
    for date in candidates:
        if len(processed) >= need:
            break
        elapsed = time.perf_counter() - t0
        if elapsed > budget_seconds:
            skipped.append({'date': date, 'reason': f'45-minute CPU budget exceeded ({elapsed:.0f}s elapsed)'})
            continue
        t_date0 = time.perf_counter()
        arrays = run_sr_for_dates(cfg_ev, ev_root, cache, [date], crop)
        write_dihedral_cache(CACHE_DIR, date, arrays[date])
        seconds = time.perf_counter() - t_date0
        processed.append({'date': date, 'seconds': seconds})
        if verbose:
            print(f'  F11 pool extension: {date} done in {seconds:.1f}s', flush=True)

    final_pool = existing + [p['date'] for p in processed]
    status = 'PASS' if len(final_pool) >= pe['minimum_n'] else 'BLOCKED'
    result = {
        'status': status,
        'evidence': 'real',
        'config_sha256': digest(FIX11_CONFIG),
        'started_utc': started,
        'finished_utc': datetime.now(timezone.utc).isoformat(),
        'existing_pool': existing,
        'newly_processed': processed,
        'skipped': skipped,
        'final_pool': sorted(final_pool),
        'final_pool_size': len(final_pool),
        'minimum_required': pe['minimum_n'],
        'target': pe['target_n'],
        'blocked_reason': None if status == 'PASS' else
            f'only reached {len(final_pool)} pool dates, below the minimum {pe["minimum_n"]} required '
            f'for fold_construction to hold any date out of an n_pre=3 reference set',
    }
    write_json(ROOT / 'experiments' / 'results' / 'f11_pool_extension.json', result)
    if verbose:
        print(f'F11 pool extension: {status}, final pool = {result["final_pool"]}', flush=True)
    return result


if __name__ == '__main__':
    main()
