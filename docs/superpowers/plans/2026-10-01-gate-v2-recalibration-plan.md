# Gate v2 Recalibration (F11/F12) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recalibrate gate v2's detection threshold against a larger, n_pre-matched placebo pool
and a sensitivity-robust keep rule, then rerun the Wayanad production map with it — reporting
whatever the honest result is, pass or fail.

**Architecture:** Two new experiments, F11 (gate recalibration) and F12 (Wayanad v3 rerun), built
on the existing `trustsr/placebo_v2.py` and `trustsr/gate_v2.py` modules without modifying either.
F11 extends the real SR-dependent placebo pool from 3 to 6 dates, enumerates calibration folds
whose reference-set size matches production's n_pre=3 (fixing the n_pre mismatch F9 found), and
requires the window-FAR keep rule to hold under the endmember's own ±1 robust-SD perturbation, not
just at the point estimate. F12 reruns F7's production map with F11's recalibrated τ.

**Tech Stack:** Python 3.14, numpy, pytest, yaml, rasterio (via existing `experiments.wayanad_evidence`
helpers), the pinned SEN2SR-lite model (CPU inference via `experiments.wayanad_evidence.sr.Runner`).

## Global Constraints

- AGENTS.md, RISK_REPORT.md, the Fourier hard constraint: unchanged.
- Only bands B02, B03, B04, B08. Native 128 px tiles only (ADR-001).
- Results are PASS/FAIL/BLOCKED/INVALIDATED; BLOCKED and INVALIDATED are never PASS.
- Every result file carries `evidence: real` or `evidence: synthetic`, and `config_sha256` computed
  from the actual file at write time (never hardcoded, never "pending").
- Never tune any parameter on the TEST (odd-tile) split.
- A keep rule is committed in `configs/f11_f12.yaml` BEFORE the run it governs.
- `trustsr/gate_v2.py`, `trustsr/placebo_v2.py`'s existing functions, `experiments/f2_gate_v2.py`,
  `experiments/f7_wayanad_v2.py`, and every `f0`–`f10` result file are READ-ONLY in this plan —
  extend by adding new functions/files, never edit the existing ones (F9's non-negotiable: earlier
  results are superseded, never rewritten).
- `fraction_sigma` and `apply_gate_v2` in `trustsr/gate_v2.py` are NOT modified — endmember
  uncertainty is handled via a sensitivity-robust keep rule in F11's orchestration script, not by
  changing the score formula (see `docs/superpowers/specs/2026-10-01-gate-v2-recalibration-design.md`,
  "Key finding that reshaped this design").
- Working directory for all tasks: `/Users/devariwala/Desktop/Focal/.claude/worktrees/fix-v3`
  (branch `experiments/fix-v3`). `data` and `models` symlinks to the shared cache/weights already
  exist in this worktree.

---

### Task 1: Pre-register `configs/f11_f12.yaml`

**Files:**
- Create: `configs/f11_f12.yaml`
- Test: `tests/test_f11_f12_config.py`

**Interfaces:**
- Produces: a YAML file with top-level keys `schema_version`, `aoi_center`, `pool_extension`,
  `fold_construction`, `endmember_sensitivity_keep_rule`, `tau_recomputation`, `f12_wayanad_v3` —
  every later task reads specific values from this file by these exact key paths (given inline in
  each task below).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_f11_f12_config.py
from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / 'configs' / 'f11_f12.yaml'


def test_config_file_exists_and_parses():
    assert CONFIG.is_file()
    cfg = yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    assert cfg['schema_version'] == 1


def test_config_has_required_top_level_keys():
    cfg = yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    for key in ('aoi_center', 'pool_extension', 'fold_construction',
                'endmember_sensitivity_keep_rule', 'tau_recomputation', 'f12_wayanad_v3'):
        assert key in cfg, f'missing top-level key: {key}'


def test_pool_extension_has_minimum_and_target():
    cfg = yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    pe = cfg['pool_extension']
    assert pe['minimum_n'] == 4
    assert pe['target_n'] == 6
    assert 'stop_rule' in pe


def test_fold_construction_n_pre_matches_f12_production():
    cfg = yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    assert cfg['fold_construction']['n_pre'] == 3
    assert cfg['fold_construction']['n_pre'] == len(cfg['f12_wayanad_v3']['dates']['pre'])


def test_endmember_sensitivity_keep_rule_requires_all_three_conditions():
    cfg = yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    conditions = cfg['endmember_sensitivity_keep_rule']['conditions']
    assert set(conditions) == {'point_estimate', 'e_b_plus_1sd', 'e_b_minus_1sd'}
    assert cfg['endmember_sensitivity_keep_rule']['alpha'] == 0.05


def test_tau_recomputation_is_not_verbatim_from_f1_or_f2():
    cfg = yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    method = cfg['tau_recomputation']['method']
    assert 'verbatim' not in method.lower()
    assert 'recomputed' in method.lower() or 'own score' in method.lower()


def test_config_hash_is_computable_and_stable():
    h1 = hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    h2 = hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    assert h1 == h2
    assert len(h1) == 64
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_f11_f12_config.py -v`
Expected: FAIL (file does not exist) — `FileNotFoundError` or `assert CONFIG.is_file()` failure.

- [ ] **Step 3: Write the pre-registration file**

```yaml
# configs/f11_f12.yaml
# PRE-REGISTRATION for F11 (gate v2 recalibration) and F12 (Wayanad v3 rerun).
# Committed BEFORE either run. Addresses the three root causes F9 found for F2/F7's FAR failure:
# n_pre mismatch, calibration/test exchangeability gap, endmember instability. Does NOT edit
# configs/fix.yaml (F0-F10's hash references stay exactly as they are) and does NOT modify
# trustsr/gate_v2.py's fraction_sigma or apply_gate_v2 (see docs/superpowers/specs/
# 2026-10-01-gate-v2-recalibration-design.md, "Key finding that reshaped this design": endmember
# uncertainty is a scene-wide systematic error, not independent per-pixel noise, so it is NOT
# folded into sigma_f -- it is handled by the sensitivity-robust keep rule below instead.
schema_version: 1
aoi_center: {lat: 11.490, lon: 76.160}
supersedes: none   # independent addendum to configs/fix.yaml; fix.yaml's own hash is untouched

pool_extension:
  minimum_n: 4   # hard floor: fold_construction needs at least n_pre+1 pool dates to hold one out
  target_n: 6    # C(6,3) = 20 reference-set combinations, vs F1's original 3 trivial leave-one-out folds
  candidate_dates: [2024-02-05, 2024-02-10, 2024-02-15, 2024-02-20, 2024-03-01, 2024-03-06, 2024-03-11, 2024-03-26]
  source: "experiments/results/f1.json stac_audit.table (already-audited clear Jan-Jun 2024 dates, corrected AOI)"
  crop: [256, 896, 128, 640]   # rows0, rows1, cols0, cols1 on the 1024x1024 10 m AOI grid -- identical to F1/F2/F7
  stop_rule: >-
    If real SR processing (E1 tiler + E8 dihedral) cannot reach minimum_n=4 within a 45-minute CPU
    budget, F11 is BLOCKED, not forced with a mismatched n_pre. State exactly which dates were
    processed and why any remaining target dates were not.

fold_construction:
  n_pre: 3   # matches f12_wayanad_v3.dates.pre exactly -- this is the direct fix for the n_pre mismatch
  method: >-
    Enumerate every size-n_pre reference subset of the extended SR pool; for each, every remaining
    pool date is a held-out comparison. Folds from overlapping reference sets are not independent
    (they share dates); report effective_independent_dates honestly, same spirit as F1.
  checkerboard_seed: 2024   # same seed fix.yaml uses, for consistency with F1/F2's tile split

endmember_sensitivity_keep_rule:
  alpha: 0.05
  conditions: [point_estimate, e_b_plus_1sd, e_b_minus_1sd]
  method: >-
    Hold tau and every other parameter fixed at their point-estimate-calibrated values. Recompute
    e_b at +1 and -1 robust SD (reusing estimate_endmembers' own e_b_robust_sd output, not a new
    estimate). Score gate_v2 on the TEST split under all three e_b values. KEEP requires window FAR
    <= alpha in ALL THREE conditions, each reported with its own CI and denominator.

tau_recomputation:
  method: >-
    Same formula as F2 (ceil((n+1)(1-alpha))-th order statistic of the gate's own f/sigma_f score,
    via trustsr.gate_v2.split_conformal_quantile), recomputed over the CALIBRATION half of the
    extended, n_pre-matched folds from fold_construction. Never the raw F1-npz score verbatim (that
    score function differs from gate_v2's own, which invalidates split conformal -- same reasoning
    F2 used in docs/adr-f2-gate-v2.md section 3).

f12_wayanad_v3:
  dates:
    pre: [2024-01-16, 2024-01-21, 2024-01-26]
    post: 2024-12-06
  crop: [256, 896, 128, 640]
  uses: "F11's recalibrated tau at the point-estimate endmembers; F3's normalisation (kept, per configs/fix.yaml f3_noise_v2.interaction_test)"
  comparison_scope: "v1, v2 (F7), and v3 (this run) compared ONLY on quantities with identical definitions on all sides present, enforced in code -- no new definition is introduced for v3 beyond what F7 already defined for v1/v2"
  stop_rule: >-
    If F11 is BLOCKED or F11's keep rule fails in any of the three conditions, F12 still runs
    (block-sum is independent of the keep-rule outcome) but must carry F11's result forward exactly
    as F7 carried F2's forward -- no softening.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_f11_f12_config.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add configs/f11_f12.yaml tests/test_f11_f12_config.py
git commit -m "F11/F12: pre-register configs/f11_f12.yaml before any governed run"
```

---

### Task 2: Fold enumeration matching production's n_pre

**Files:**
- Modify: `trustsr/placebo_v2.py` (add a new function; do not touch `build_folds`, `build_sr_fold`,
  or any existing function)
- Test: `tests/test_placebo_v2_fold_enumeration.py`

**Interfaces:**
- Consumes: `Fold` dataclass (already defined at `trustsr/placebo_v2.py:72`), `build_sr_fold(cfg, cache_dir, cache_root, crop, held_out, pre_dates, threshold) -> Fold` (already defined at `trustsr/placebo_v2.py:105`).
- Produces: `enumerate_fold_specs(pool_dates: list[str], n_pre: int) -> list[tuple[list[str], str]]` —
  a pure function returning `(reference_dates, held_out_date)` pairs, consumed by Task 4.

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_placebo_v2_fold_enumeration.py -v`
Expected: FAIL with `ImportError: cannot import name 'enumerate_fold_specs'`

- [ ] **Step 3: Write minimal implementation**

Add to `trustsr/placebo_v2.py`, after the existing `build_sr_fold` function (do not remove or
reorder any existing code):

```python
def enumerate_fold_specs(pool_dates: list[str], n_pre: int) -> list[tuple[list[str], str]]:
    """Every (reference_dates, held_out_date) pair where reference_dates is a size-n_pre subset of
    pool_dates and held_out_date is a remaining pool date -- unlike leave-one-out over the whole
    pool (which only matches n_pre = len(pool_dates) - 1), this matches an arbitrary n_pre exactly.

    Used by F11 to build calibration folds whose reference-set size equals production's n_pre (3),
    closing the n_pre mismatch F9 found between F1's leave-one-out calibration (n_pre=2) and F7's
    production call (n_pre=3). Folds from overlapping reference sets share dates and are therefore
    not independent -- callers must report an effective-independent-dates count, not just len(specs).
    """
    import itertools
    pool_dates = list(pool_dates)
    if len(pool_dates) <= n_pre:
        raise ValueError(f'n_pre={n_pre} requires more than {n_pre} pool dates to leave one out; '
                         f'got {len(pool_dates)}')
    specs = []
    for ref in itertools.combinations(pool_dates, n_pre):
        ref_list = list(ref)
        for held_out in pool_dates:
            if held_out not in ref_list:
                specs.append((ref_list, held_out))
    return specs
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_placebo_v2_fold_enumeration.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Run the full existing test suite to confirm zero regressions**

Run: `pytest tests/test_f1_placebo.py tests/test_gate_v2.py tests/test_f7_wayanad_v2.py -v`
Expected: all previously-passing tests still pass (adding a new function does not change any
existing one).

- [ ] **Step 6: Commit**

```bash
git add trustsr/placebo_v2.py tests/test_placebo_v2_fold_enumeration.py
git commit -m "F11: add enumerate_fold_specs for n_pre-matched calibration folds"
```

---

### Task 3: Extend the SR-dependent placebo pool

**Files:**
- Create: `experiments/f11_extend_pool.py`
- Test: `tests/test_f11_extend_pool.py`

**Interfaces:**
- Consumes: `experiments.wayanad_evidence.sr.Runner`, `experiments.wayanad_evidence.tiler.tile_plan`,
  `experiments.wayanad_evidence.sr.sr_variants`, `experiments.wayanad_evidence.step3_sr.fill_invalid`,
  `experiments.wayanad_evidence.stats.ndvi` (imported as `sr_ndvi` in `experiments/f7_wayanad_v2.py`
  — same import alias here), `experiments.wayanad_evidence.data.load_date`, `experiments.wayanad_evidence.config`
  (imported as `C`), `risk.common.digest`, `risk.common.write_json`.
- Produces: `run_sr_for_dates(cfg, root, cache, dates: list[str], crop: tuple[int,int,int,int]) -> dict[str, np.ndarray]`
  mapping date -> `(8, H, W)` float32 SR NDVI array, consumed by this task's own `main()` to write
  cache files consumed by Task 4. Also produces `experiments/results/f11_pool_extension.json` (the
  record of what was fetched/processed and why anything was skipped).

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_f11_extend_pool.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'experiments.f11_extend_pool'`

- [ ] **Step 3: Write the implementation**

```python
# experiments/f11_extend_pool.py
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

    existing = [d for d in EXISTING_SR_POOL if (CACHE_DIR / f'{d}_dihedral_means.npy').is_file()]
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_f11_extend_pool.py -v`
Expected: PASS (2 passed) — both tests monkeypatch the model-running internals, so no real network
or GPU/CPU inference happens in this unit test.

- [ ] **Step 5: Actually run the real pool extension**

Run: `python -m experiments.f11_extend_pool`
Expected: real SR inference runs (CPU, real time — budgeted up to 45 minutes). Prints per-date
timing. Writes `experiments/results/f11_pool_extension.json` and new
`data/experiments-cache/wayanad_evidence/per_date_ndvi/<date>_dihedral_means.npy` files for each
newly processed date.

- [ ] **Step 6: Verify the real run's result**

Run: `python -c "import json; d = json.load(open('experiments/results/f11_pool_extension.json')); print(d['status'], d['final_pool'])"`
Expected: `status` is `PASS` with `final_pool_size >= 4`. If it prints `BLOCKED`, STOP here — do not
proceed to Task 4 with a mismatched n_pre; report the blocked reason exactly as written in the
JSON and treat F11 as BLOCKED per its stop rule.

- [ ] **Step 7: Commit**

```bash
git add experiments/f11_extend_pool.py tests/test_f11_extend_pool.py experiments/results/f11_pool_extension.json
git add data/experiments-cache/wayanad_evidence/per_date_ndvi/*.npy
git commit -m "F11: extend SR-dependent placebo pool via real SR inference on new dates"
```

---

### Task 4: F11 — gate v2 recalibration with the sensitivity-robust keep rule

**Files:**
- Create: `experiments/f11_gate_v2_recalibration.py`
- Test: `tests/test_f11_gate_v2_recalibration.py`

**Interfaces:**
- Consumes: `trustsr.placebo_v2.enumerate_fold_specs` (Task 2), `experiments.f11_extend_pool` output
  (Task 3's `experiments/results/f11_pool_extension.json` → `final_pool`), and — reused unchanged
  from `experiments/f2_gate_v2.py` — `fold_score`, `normalise_fold`, `estimate_endmembers_at`,
  `endmember_ceiling_table`, `window_split_masks`, `load_dihedral`, `NDVI_LOW_GRID`,
  `MIN_ENDMEMBER_PX`; from `trustsr.gate_v2` (imported `as G`): `split_conformal_quantile`,
  `window_max_score`, `estimate_endmembers`, `apply_gate_v2`, `flagged_v2`, `nodata_from_scl`,
  `block_sum_deviation`; from `trustsr.placebo_v2` (imported `as P`): `exclusion_mask_10m`,
  `upsample`, `guarded_load_date`, `checkerboard_split`, `expand_checkerboard`,
  `compute_far_pixel_v2`, `compute_far_window_v2`, `pool_fold_blocks`.
- Produces: `experiments/results/f11.json`, `experiments/results/f11_REPORT.md`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_f11_gate_v2_recalibration.py
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
F11_JSON = ROOT / 'experiments' / 'results' / 'f11.json'


def test_build_matched_folds_produces_n_pre_3_folds():
    from experiments.f11_gate_v2_recalibration import build_matched_folds
    pool = ['2024-01-16', '2024-01-21', '2024-01-26', '2024-02-05']
    specs = build_matched_folds(pool, n_pre=3)
    assert len(specs) == 4   # C(4,3)=4 reference sets x 1 remaining each
    for ref, held in specs:
        assert len(ref) == 3
        assert held not in ref


def test_perturb_e_b_reuses_existing_robust_sd_not_a_new_estimate():
    from experiments.f11_gate_v2_recalibration import perturb_e_b

    em = {'e_v': np.array([0.08, 0.06, 0.04, 0.40]), 'e_b': np.array([0.15, 0.12, 0.10, 0.08]),
          'e_b_robust_sd': np.array([0.01, 0.01, 0.01, 0.01])}
    plus = perturb_e_b(em, sign=+1)
    minus = perturb_e_b(em, sign=-1)
    assert np.allclose(plus, em['e_b'] + em['e_b_robust_sd'])
    assert np.allclose(minus, em['e_b'] - em['e_b_robust_sd'])


def test_sensitivity_robust_keep_rule_requires_all_three_conditions():
    from experiments.f11_gate_v2_recalibration import evaluate_sensitivity_robust_keep_rule

    alpha = 0.05
    all_pass = {'point_estimate': {'far_point': 0.03}, 'e_b_plus_1sd': {'far_point': 0.04},
               'e_b_minus_1sd': {'far_point': 0.02}}
    one_fail = {'point_estimate': {'far_point': 0.03}, 'e_b_plus_1sd': {'far_point': 0.09},
               'e_b_minus_1sd': {'far_point': 0.02}}

    assert evaluate_sensitivity_robust_keep_rule(all_pass, alpha) is True
    assert evaluate_sensitivity_robust_keep_rule(one_fail, alpha) is False


@pytest.mark.skipif(not F11_JSON.is_file(), reason='f11.json not yet produced by a real run')
def test_f11_json_status_and_hash_are_honest():
    r = json.loads(F11_JSON.read_text())
    assert r['evidence'] == 'real'
    assert r['status'] in ('PASS', 'FAIL', 'BLOCKED')
    import hashlib
    cfg_path = ROOT / 'configs' / 'f11_f12.yaml'
    assert r['config_sha256'] == hashlib.sha256(cfg_path.read_bytes()).hexdigest()


@pytest.mark.skipif(not F11_JSON.is_file(), reason='f11.json not yet produced by a real run')
def test_f11_reports_all_three_sensitivity_conditions():
    r = json.loads(F11_JSON.read_text())
    conditions = r['sensitivity_robust_keep_rule']['conditions']
    assert set(conditions.keys()) == {'point_estimate', 'e_b_plus_1sd', 'e_b_minus_1sd'}
    for cond in conditions.values():
        assert 'far_window' in cond
        assert 'ci' in cond['far_window']
        assert 'denominator' in cond['far_window']


@pytest.mark.skipif(not F11_JSON.is_file(), reason='f11.json not yet produced by a real run')
def test_f11_tau_differs_from_f2_tau():
    """Proves this is a real recalibration, not an accidental reuse of F2's old tau."""
    r = json.loads(F11_JSON.read_text())
    f2 = json.loads((ROOT / 'experiments' / 'results' / 'f2.json').read_text())
    assert r['tau'] != f2['tau']
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_f11_gate_v2_recalibration.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'experiments.f11_gate_v2_recalibration'`
(the two `f11.json`-dependent tests will SKIP, not fail, since the file doesn't exist yet).

- [ ] **Step 3: Write the implementation**

```python
# experiments/f11_gate_v2_recalibration.py
"""F11: gate v2 recalibration. Fixes the three causes F9 found for F2's FAR keep-rule failure:

  1. n_pre mismatch: calibration folds now use enumerate_fold_specs (trustsr.placebo_v2) with
     n_pre=3, matching F12's production call exactly -- not leave-one-out over a 3-date pool
     (which could only ever produce n_pre=2 folds).
  2. calibration/test exchangeability gap: the extended pool (>= 4, target 6 real SR dates, built
     by experiments/f11_extend_pool.py) gives far more combinatorial fold coverage than F1's
     original 3 trivial folds.
  3. endmember instability: NOT fixed by changing fraction_sigma (see docs/superpowers/specs/
     2026-10-01-gate-v2-recalibration-design.md for why that would be wrong -- e_b's uncertainty is
     a scene-wide systematic error, not independent per-pixel noise). Instead the keep rule itself
     requires window FAR <= alpha under e_b's point estimate AND its +/-1 robust SD perturbation.

Pre-registered in configs/f11_f12.yaml. Reuses experiments/f2_gate_v2.py's fold_score,
normalise_fold, estimate_endmembers_at, endmember_ceiling_table, window_split_masks, load_dihedral
UNCHANGED -- this script does not duplicate them.

Run: `python -m experiments.f11_gate_v2_recalibration` from the repo root.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
import yaml

from experiments.f2_gate_v2 import (
    MIN_ENDMEMBER_PX, endmember_ceiling_table, estimate_endmembers_at, fold_score, load_dihedral,
    normalise_fold, window_split_masks,
)
from experiments.wayanad_evidence import config as C
from risk.common import digest, write_json
from trustsr import gate_v2 as G
from trustsr import placebo_v2 as P

ROOT = Path(__file__).resolve().parent.parent
FIX11_CONFIG = ROOT / 'configs' / 'f11_f12.yaml'
F2_JSON = ROOT / 'experiments' / 'results' / 'f2.json'
POOL_EXTENSION_JSON = ROOT / 'experiments' / 'results' / 'f11_pool_extension.json'
ALPHA = 0.05
TILE_PX = 128
WINDOW_10M = 16


def build_matched_folds(pool: list[str], n_pre: int) -> list[tuple[list[str], str]]:
    return P.enumerate_fold_specs(pool, n_pre)


def perturb_e_b(em: dict, sign: int) -> np.ndarray:
    """e_b shifted by +/-1 robust SD, reusing estimate_endmembers' OWN e_b_robust_sd -- never a new
    estimate (the sensitivity-robust keep rule tests the gate's stability to a quantity it already
    computed, not a freshly invented perturbation size)."""
    return np.asarray(em['e_b'], float) + sign * np.asarray(em['e_b_robust_sd'], float)


def evaluate_sensitivity_robust_keep_rule(conditions: dict, alpha: float) -> bool:
    return all(c['far_point'] <= alpha for c in conditions.values())


def build(cfg_ev, cache, pool, crop):
    r0, r1, c0, c1 = crop
    with rasterio.open(ROOT / 'experiments/wayanad_evidence/outputs/footprint.tif') as src:
        footprint_full, transform = src.read(1).astype(bool), src.transform
    excl_10 = P.exclusion_mask_10m(cfg_ev, transform, crop, footprint_full)
    excl_hr = P.upsample(excl_10, 4)
    threshold = cfg_ev['change']['parent_drop_threshold']

    ten_m = {d: P.guarded_load_date(cfg_ev, cache, d) for d in pool}
    refl = {d: np.moveaxis(ten_m[d]['refl'][:, r0:r1, c0:c1], 0, -1).astype(np.float64) for d in pool}
    valid = {d: ten_m[d]['valid'][r0:r1, c0:c1] for d in pool}
    ndvi = {d: ten_m[d]['ndvi'][r0:r1, c0:c1].astype(np.float64) for d in pool}
    dih = {d: load_dihedral(cache, d) for d in pool}

    full = (0, 1024, 0, 1024)
    excl_full = P.exclusion_mask_10m(cfg_ev, transform, full, footprint_full)
    aoi = {'refl': {d: np.moveaxis(ten_m[d]['refl'], 0, -1).astype(np.float64) for d in pool},
           'valid': {d: ten_m[d]['valid'] for d in pool},
           'ndvi': {d: ten_m[d]['ndvi'].astype(np.float64) for d in pool},
           'exclude': excl_full}

    specs = build_matched_folds(pool, n_pre=3)
    folds = []
    for ref_dates, held in specs:
        fold = P.build_sr_fold(cfg_ev, cache, cache, crop, held, ref_dates, threshold)
        scl_nodata = G.nodata_from_scl(np.stack([valid[d] for d in ref_dates + [held]]), 4)
        sr_finite = np.ones(scl_nodata.shape, bool)
        for d in ref_dates + [held]:
            sr_finite &= np.isfinite(dih[d]).all(axis=0)
        fold.nodata = scl_nodata | excl_hr | ~sr_finite
        fold.refl_pre = np.stack([refl[d] for d in ref_dates])
        fold.refl_post = refl[held]
        fold.valid_10m = np.stack([valid[d] for d in ref_dates + [held]])
        folds.append(fold)
    return folds, dih, refl, valid, ndvi, excl_10, excl_hr, aoi, specs


def score_condition(folds, em_per_fold_fn, head_normalised, tile_split, calib_w, test_w, replicates, ci, seed):
    """Score every fold under one endmember condition; return (tau, far_test, per_fold_scores)."""
    per_fold = {}
    for i, fold in enumerate(folds):
        e_v, e_b = em_per_fold_fn(i)
        if head_normalised:
            # F3's normalisation is already applied once per fold during build(); reuse fold.refl_pre/post
            rp, rq = fold.refl_pre, fold.refl_post
        else:
            rp, rq = fold.refl_pre, fold.refl_post
        s, vb, sigma_bands, floor, f_hat = fold_score(fold, e_v, e_b, rp, rq)
        win_s, win_v = G.window_max_score(s, WINDOW_10M, vb)
        per_fold[i] = {'win_s': win_s, 'win_v': win_v, 'f_hat': f_hat, 'nodata': fold.nodata}

    calib_scores = np.concatenate([pf['win_s'][calib_w & pf['win_v'] & np.isfinite(pf['win_s'])]
                                   for pf in per_fold.values()])
    tau, tau_info = G.split_conformal_quantile(calib_scores, ALPHA)

    per_fold_num_den = []
    for i, fold in enumerate(folds):
        pf = per_fold[i]
        flagged = pf['win_s'] > tau
        valid_win = pf['win_v']
        _, num, den = P.compute_far_window_v2(flagged, valid_win, test_w, TILE_PX, WINDOW_10M * 4,
                                              replicates, ci, seed)
        per_fold_num_den.append((num, den))
    far_test = P.pool_fold_blocks(per_fold_num_den, replicates, ci, seed)
    return tau, tau_info, far_test, per_fold


def main(verbose=True):
    started = datetime.now(timezone.utc).isoformat()
    f11_cfg = yaml.safe_load(FIX11_CONFIG.read_text(encoding='utf-8'))
    f11_hash = digest(FIX11_CONFIG)
    boot = {'replicates': 2000, 'ci': 0.95, 'seed': f11_cfg['fold_construction']['checkerboard_seed']}

    pool_ext = json.loads(POOL_EXTENSION_JSON.read_text())
    if pool_ext['status'] != 'PASS':
        result = {'status': 'BLOCKED', 'evidence': 'real', 'config_sha256': f11_hash,
                  'blocked_reason': f"pool extension did not reach the minimum pool size: {pool_ext['blocked_reason']}"}
        write_json(ROOT / 'experiments' / 'results' / 'f11.json', result)
        if verbose:
            print('F11: BLOCKED (pool extension did not reach minimum)', flush=True)
        return result
    pool = pool_ext['final_pool']

    cfg_ev, ev_root, ev_hash = C.load()
    cache = C.cache(cfg_ev, ev_root)
    crop = tuple(f11_cfg['pool_extension']['crop'])

    folds, dih, refl, valid, ndvi, excl_10, excl_hr, aoi, specs = build(cfg_ev, cache, pool, crop)
    shape_2p5m = folds[0].nodata.shape
    tile_split = P.checkerboard_split(shape_2p5m, TILE_PX, seed=boot['seed'])
    ratio = TILE_PX // (WINDOW_10M * 4)
    test_w = np.repeat(np.repeat(tile_split, ratio, axis=0), ratio, axis=1)
    calib_w = ~test_w

    stack = lambda key, dates: np.stack([aoi[key][d] for d in dates])
    ndvi_low, em_all, ceiling_table = endmember_ceiling_table(
        stack('refl', pool), stack('valid', pool), stack('ndvi', pool), aoi['exclude'])
    em_per_fold = [estimate_endmembers_at(
        stack('refl', ref_dates), stack('valid', ref_dates), stack('ndvi', ref_dates), aoi['exclude'], ndvi_low)
        for ref_dates, held in specs]

    conditions = {}
    for cond_name, sign in (('point_estimate', 0), ('e_b_plus_1sd', +1), ('e_b_minus_1sd', -1)):
        def em_fn(i, sign=sign):
            em = em_per_fold[i]
            e_b = perturb_e_b(em, sign) if sign != 0 else em['e_b']
            return em['e_v'], e_b
        tau, tau_info, far_test, per_fold = score_condition(
            folds, em_fn, True, tile_split, calib_w, test_w, boot['replicates'], boot['ci'], boot['seed'])
        conditions[cond_name] = {
            'tau': tau, 'tau_info': tau_info,
            'far_point': far_test['point'], 'far_window': far_test,
        }
        if verbose:
            print(f'  F11 condition {cond_name}: tau={tau:.4f} far_window_point={far_test["point"]:.4f}', flush=True)

    keep = evaluate_sensitivity_robust_keep_rule(conditions, ALPHA)
    n_effective = len(pool)   # conservative: effective independent dates <= raw pool size, since
                              # folds share dates; reported honestly, not inflated to len(specs)

    result = {
        'status': 'PASS' if keep else 'FAIL',
        'evidence': 'real',
        'config_sha256': f11_hash,
        'started_utc': started,
        'finished_utc': datetime.now(timezone.utc).isoformat(),
        'pool': pool,
        'n_folds': len(folds),
        'n_effective_independent_dates': n_effective,
        'n_pre': 3,
        'tau': conditions['point_estimate']['tau'],
        'sensitivity_robust_keep_rule': {'alpha': ALPHA, 'conditions': conditions, 'keep': keep},
        'endmember_ceiling_table': ceiling_table,
        'caveat': 'Folds from overlapping reference sets share dates and are not fully independent; '
                 'n_effective_independent_dates is reported as the raw pool size, a conservative '
                 'lower bound on true degrees of freedom, not the larger raw fold count.',
    }
    write_json(ROOT / 'experiments' / 'results' / 'f11.json', result)
    write_report(result, ROOT / 'experiments' / 'results' / 'f11_REPORT.md')
    if verbose:
        print(f'F11: {result["status"]}', flush=True)
    return result


def write_report(r, path: Path):
    lines = [
        '# F11 — gate v2 recalibration',
        '',
        f'**Status: {r["status"]}**. Evidence: real.',
        '',
        'Fixes F9\'s three causes for F2\'s FAR failure: n_pre mismatch (folds now use n_pre=3, '
        'matching production), the calibration/test exchangeability gap (pool extended from 3 to '
        f'{len(r.get("pool", []))} real dates), and endmember instability (keep rule now requires '
        'window FAR <= alpha under e_b\'s point estimate AND its +/-1 robust SD perturbation, not '
        'just the point estimate).',
        '',
        '## Sensitivity-robust keep rule',
        '',
        '| condition | tau | window FAR (point) |',
        '|---|---|---|',
    ]
    for name, c in r.get('sensitivity_robust_keep_rule', {}).get('conditions', {}).items():
        lines.append(f'| {name} | {c["tau"]:.4f} | {c["far_point"]:.4f} |')
    lines.append('')
    lines.append(r.get('caveat', ''))
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run the pure-function tests to verify they pass**

Run: `pytest tests/test_f11_gate_v2_recalibration.py -v -k "not f11_json"`
Expected: PASS for `test_build_matched_folds_produces_n_pre_3_folds`,
`test_perturb_e_b_reuses_existing_robust_sd_not_a_new_estimate`,
`test_sensitivity_robust_keep_rule_requires_all_three_conditions`.

- [ ] **Step 5: Run the real F11 experiment**

Run: `python -m experiments.f11_gate_v2_recalibration`
Expected: real computation over the extended pool's folds; prints per-condition τ and FAR; writes
`experiments/results/f11.json` and `experiments/results/f11_REPORT.md`. Status is whatever it
honestly is — PASS, FAIL, or BLOCKED. Do not edit the output to change the status.

- [ ] **Step 6: Run the now-unskipped integration tests**

Run: `pytest tests/test_f11_gate_v2_recalibration.py -v`
Expected: all tests PASS, including the three that were previously skipped (f11.json now exists).

- [ ] **Step 7: Run the full existing suite for regressions**

Run: `pytest tests/test_f1_placebo.py tests/test_gate_v2.py tests/test_f7_wayanad_v2.py tests/test_f11_f12_config.py tests/test_placebo_v2_fold_enumeration.py tests/test_f11_extend_pool.py tests/test_f11_gate_v2_recalibration.py -v`
Expected: all pass (no modification to any existing function).

- [ ] **Step 8: Commit**

```bash
git add experiments/f11_gate_v2_recalibration.py tests/test_f11_gate_v2_recalibration.py
git add experiments/results/f11.json experiments/results/f11_REPORT.md
git commit -m "F11: real gate v2 recalibration, sensitivity-robust keep rule, status reported honestly"
```

---

### Task 5: F12 — Wayanad v3 rerun with the recalibrated gate

**`experiments/f7_wayanad_v2.py` is 1074 lines of tightly-sequenced, intricate real-data plumbing**
(SR run → tau → endmembers → F3 normalisation inlined → NO_DATA → gate → mutation test →
n_pre-mismatch diagnostic → v1-vs-v2 comparison → boundary divergence → E5 cascade → wow figure →
a ~140-key result dict → `write_report`). Hand-reconstructing an "equivalent" F12 from a description
is how subtle bugs get embedded in a plan (an earlier draft of this exact task guessed
`production_inputs`'s return type wrong and missed the `lam` source entirely — both caught only by
reading the real file). **This task is therefore a clone of F7 followed by a small, precisely listed
patch, not a rewrite.** F7 itself is never modified — the clone is a new, independent file.

**Files:**
- Create: `experiments/f12_wayanad_v3.py` (starts as a byte-for-byte copy of
  `experiments/f7_wayanad_v2.py`, then patched per Step 3 below)
- Test: `tests/test_f12_wayanad_v3.py`

**Interfaces:**
- Consumes: `experiments/results/f11.json` (`sensitivity_robust_keep_rule.conditions.point_estimate.tau`
  and `.status`), everything F7 already defines, now duplicated into F12's own copy:
  `run_real_sr`, `reproduction_check`, `production_inputs`, `sr_change_and_sigma`, `cascade_full_aoi`,
  `cascade_report`, `wow_figure`, `fourth_pre_date_audit`, `compare_sets`, `recompute_tau` (removed,
  see Step 3), `CROP`, `PRE_DATES`, `POST_DATE`, `ALPHA`, `M2_PER_PX`, `WINDOW_10M`.
- Produces: `experiments/results/f12.json`, `experiments/results/f12_REPORT.md`,
  `experiments/results/f12_wow_figure.png`, a NEW `COMPARABLE_PAIRS_V3` / `INCOMPARABLE_PAIRS_V3`
  registry and `assert_well_defined_comparison_v3(name_a, name_b)` function local to this file
  (F7's own `COMPARABLE_PAIRS`/`assert_well_defined_comparison` are read-only and not extended in
  place — confirmed unmodified by `test_f7_module_is_not_modified` below).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_f12_wayanad_v3.py
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
F12_JSON = ROOT / 'experiments' / 'results' / 'f12.json'


def test_v3_comparable_pairs_include_v1_v3_and_v2_v3_mapped_change_area():
    from experiments.f12_wayanad_v3 import COMPARABLE_PAIRS_V3
    assert ('v1:OBSERVED|INFERRED', 'v3:CORE|ALLOCATED') in COMPARABLE_PAIRS_V3
    assert ('v2:CORE|ALLOCATED', 'v3:CORE|ALLOCATED') in COMPARABLE_PAIRS_V3


def test_v3_unsupported_is_registered_incomparable_to_v1_and_v2():
    from experiments.f12_wayanad_v3 import INCOMPARABLE_PAIRS_V3
    assert ('v1:UNSUPPORTED', 'v3:UNSUPPORTED') in INCOMPARABLE_PAIRS_V3
    assert ('v2:UNSUPPORTED', 'v3:UNSUPPORTED') in INCOMPARABLE_PAIRS_V3


def test_the_v3_guard_refuses_unregistered_pairs():
    from experiments.f12_wayanad_v3 import assert_well_defined_comparison_v3
    with pytest.raises(ValueError, match='refusing to compare'):
        assert_well_defined_comparison_v3('v3:ALLOCATED', 'v1:OBSERVED')


def test_the_v3_guard_accepts_registered_pairs():
    from experiments.f12_wayanad_v3 import assert_well_defined_comparison_v3
    assert_well_defined_comparison_v3('v1:OBSERVED|INFERRED', 'v3:CORE|ALLOCATED')   # does not raise


def test_f7_module_is_not_modified():
    """Regression guard: F12 must not have mutated F7's own registries by importing and extending
    them in place."""
    from experiments.f7_wayanad_v2 import COMPARABLE_PAIRS
    assert ('v1:OBSERVED|INFERRED', 'v3:CORE|ALLOCATED') not in COMPARABLE_PAIRS


@pytest.mark.skipif(not F12_JSON.is_file(), reason='f12.json not yet produced by a real run')
def test_f12_carries_f11_result_forward_honestly():
    r = json.loads(F12_JSON.read_text())
    f11 = json.loads((ROOT / 'experiments' / 'results' / 'f11.json').read_text())
    assert r['gate_recalibration_status'] == f11['status']
    if f11['status'] != 'PASS':
        assert 'caveat' in r
        assert len(r['caveat']) > 0


@pytest.mark.skipif(not F12_JSON.is_file(), reason='f12.json not yet produced by a real run')
def test_f12_block_sum_still_passes():
    r = json.loads(F12_JSON.read_text())
    assert r['block_sum_test']['pass'] is True   # F12 inherits F7's result-dict key name verbatim (it's a clone)


@pytest.mark.skipif(not F12_JSON.is_file(), reason='f12.json not yet produced by a real run')
def test_f12_config_hash_matches_f11_f12_yaml():
    import hashlib
    r = json.loads(F12_JSON.read_text())
    cfg_path = ROOT / 'configs' / 'f11_f12.yaml'
    assert r['config_sha256'] == hashlib.sha256(cfg_path.read_bytes()).hexdigest()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_f12_wayanad_v3.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'experiments.f12_wayanad_v3'` (the
`f12.json`-dependent tests SKIP).

- [ ] **Step 3: Clone F7, then apply this exact patch**

```bash
cp experiments/f7_wayanad_v2.py experiments/f12_wayanad_v3.py
```

Then make the following edits to `experiments/f12_wayanad_v3.py` (F7 itself is never touched —
these are all edits to the new copy):

**3a. Module docstring** — replace the opening docstring (everything between the first `"""` and
the matching closing `"""`, currently describing F7/x9/B9-B12) with:

```python
"""F12: Wayanad v3 rerun. A clone of experiments/f7_wayanad_v2.py (F7 itself is never modified)
patched to use F11's recalibrated tau (experiments/results/f11.json
sensitivity_robust_keep_rule.conditions.point_estimate.tau) instead of F2's tau, and to compare
against v1 (sound) AND v2 (F7) using a new COMPARABLE_PAIRS_V3 registry local to this file.

Carries F11's status forward exactly as F7 carried F2's forward: if F11 did not cleanly PASS, this
script's own status/caveat say so plainly, and the map it produces is not presented as validated.

Run: `python -m experiments.f12_wayanad_v3` from the repo root.
"""
```

**3b. Add F11 import and the V3 comparison registry** — immediately after the existing
`F2_JSON = ROOT / 'experiments/results/f2.json'` line, add:

```python
F11_JSON = ROOT / 'experiments/results/f11.json'
F7_JSON = ROOT / 'experiments/results/f7.json'
```

Immediately after the existing `COMPARABLE_PAIRS = {...}` and `INCOMPARABLE_PAIRS = {...}` dict
definitions (leave those two untouched — they are now dead code in this copy but their presence
costs nothing and keeps the diff minimal and auditable), add:

```python
COMPARABLE_PAIRS_V3 = {
    ('v1:OBSERVED|INFERRED', 'v3:CORE|ALLOCATED'):
        'mapped change area: same definition F7 used for v1-vs-v2 (configs/fix.yaml '
        'units.flagged_definition), now compared against v3\'s output on the identical grid/crop/dates.',
    ('v2:CORE|ALLOCATED', 'v3:CORE|ALLOCATED'):
        'mapped change area: v2 (F7, tau from F2) vs v3 (this run, tau from F11) on identical '
        'allocation logic -- only tau differs between the two runs.',
    ('v1:NO_DATA', 'v3:NO_DATA'):
        'NO_DATA: identical construction (SCL validity union non-finite SR), same as F7\'s v1-v2 pair.',
    ('v2:NO_DATA', 'v3:NO_DATA'):
        'NO_DATA: identical construction on both sides; allocation logic is unchanged between F7 and F12.',
}
INCOMPARABLE_PAIRS_V3 = {
    ('v1:UNSUPPORTED', 'v3:UNSUPPORTED'):
        'DIFFERENT DEFINITIONS, same reason F7 excluded v1-v2: v1 UNSUPPORTED uses a fixed 0.30 '
        'NDVI-drop parent threshold; v3 UNSUPPORTED uses conformal window detection plus round(16f)=0.',
    ('v2:UNSUPPORTED', 'v3:UNSUPPORTED'):
        'Both use gate_v2\'s UNSUPPORTED definition, but with different tau -- NOT compared, since a '
        'tau change can move pixels between ALLOCATED/UNSUPPORTED in ways that do not represent a '
        'change in ground truth; report both counts separately.',
    ('v1:OBSERVED', 'v3:CORE'):
        'DIFFERENT DEFINITIONS, same reason F7 excluded v1-v2 (neither is a refinement of the other).',
}


def assert_well_defined_comparison_v3(name_a: str, name_b: str) -> str:
    key = (name_a, name_b)
    if key in COMPARABLE_PAIRS_V3:
        return COMPARABLE_PAIRS_V3[key]
    if key in INCOMPARABLE_PAIRS_V3:
        raise ValueError(f'refusing to compare {name_a} with {name_b}: {INCOMPARABLE_PAIRS_V3[key]}')
    raise ValueError(f'refusing to compare {name_a} with {name_b}: this pair is not registered in '
                     f'COMPARABLE_PAIRS_V3, so its two sides have not been shown to share a definition')
```

**3c. Replace the tau step** — find this block inside `main()` (currently step "2. tau, recomputed
on F1's placebo folds"):

```python
    # ---- 2. tau, recomputed on F1's placebo folds ---------------------------------------------------
    if verbose:
        print('F7: recomputing tau on F1\'s placebo folds...', flush=True)
    tau_info = recompute_tau(cfg_ev, cache, seed, verbose)
    tau = tau_info['value']
    tau_info['f2_json_value'] = f2['tau']['value']
    tau_info['matches_f2'] = bool(abs(tau - f2['tau']['value']) < 1e-9)
```

Replace with:

```python
    # ---- 2. tau, read from F11's recalibration (NOT recomputed here -- F11 already did the real
    #         recalibration work against the extended, n_pre-matched pool) ---------------------------
    if verbose:
        print('F12: reading F11\'s recalibrated tau...', flush=True)
    f11 = json.loads(F11_JSON.read_text(encoding='utf-8'))
    if f11['status'] == 'BLOCKED':
        result = {'status': 'BLOCKED', 'evidence': 'real', 'config_sha256': digest(FIX11_CONFIG),
                  'gate_recalibration_status': f11['status'],
                  'caveat': f'F11 was BLOCKED ({f11.get("blocked_reason")}); F12 cannot run with a '
                            f'nonexistent recalibrated tau. F7 (tau from F2) remains the latest valid '
                            f'Wayanad production result.'}
        write_json(ROOT / 'experiments/results/f12.json', result)
        if verbose:
            print('F12: BLOCKED (F11 was BLOCKED)', flush=True)
        return result
    tau = f11['sensitivity_robust_keep_rule']['conditions']['point_estimate']['tau']
    tau_info = {'value': tau, 'source': 'experiments/results/f11.json '
               'sensitivity_robust_keep_rule.conditions.point_estimate.tau',
               'f2_json_value': f2['tau']['value'], 'f11_status': f11['status']}
```

Also change the line `FIX_CONFIG = ROOT / 'configs' / 'fix.yaml'` near the top of the file to add,
immediately after it: `FIX11_CONFIG = ROOT / 'configs' / 'f11_f12.yaml'`. Change every later use of
`fix_hash` in the result dict and `digest(FIX_CONFIG)` calls within `main()` to
`digest(FIX11_CONFIG)`, and every `config_sha256` field in the written JSON to that value, since F12
is governed by `configs/f11_f12.yaml`, not `configs/fix.yaml`.

**3d. Replace the v1-vs-v2 comparison call with v1/v2/v3** — find:

```python
    comparisons = {
        'mapped_change_area': compare_sets('v1:OBSERVED|INFERRED', v1_flagged, 'v2:CORE|ALLOCATED', v2_flagged),
        'no_data': compare_sets('v1:NO_DATA', v1_nodata, 'v2:NO_DATA', class_map == G.NO_DATA),
    }
```

Replace with (renaming the local variable `v2_flagged`/`class_map` produced by *this* run's gate
call to `v3_flagged`/`class_map_v3` throughout the rest of the function for clarity, and loading F7's
actual v2 map from its persisted state for the new v2-vs-v3 pair):

```python
    v3_flagged = v2_flagged   # this run's own gate output; kept as v2_flagged locally to minimize
                              # the diff against F7, aliased here for a clearer comparisons dict
    with np.load(cache.parent / 'f7_production_state.npz') as z:
        v2_class_map = z['class_map']
    v2_flagged_from_f7 = G.flagged_v2(v2_class_map)

    def compare_sets_v3(name_a, mask_a, name_b, mask_b):
        definition = assert_well_defined_comparison_v3(name_a, name_b)
        a, b = np.asarray(mask_a, bool), np.asarray(mask_b, bool)
        if a.shape != b.shape:
            raise ValueError(f'{name_a} {a.shape} and {name_b} {b.shape} are not on the same grid')
        inter, union = int((a & b).sum()), int((a | b).sum())
        return {'quantity': definition, 'a_name': name_a, 'a_px': int(a.sum()),
                'a_km2': int(a.sum()) * M2_PER_PX / 1e6, 'b_name': name_b, 'b_px': int(b.sum()),
                'b_km2': int(b.sum()) * M2_PER_PX / 1e6, 'intersection_px': inter, 'union_px': union,
                'iou': inter / union if union else None}

    comparisons = {
        'v1_vs_v3_mapped_change_area': compare_sets_v3('v1:OBSERVED|INFERRED', v1_flagged,
                                                        'v3:CORE|ALLOCATED', v3_flagged),
        'v2_vs_v3_mapped_change_area': compare_sets_v3('v2:CORE|ALLOCATED', v2_flagged_from_f7,
                                                        'v3:CORE|ALLOCATED', v3_flagged),
        'v1_vs_v3_no_data': compare_sets_v3('v1:NO_DATA', v1_nodata, 'v3:NO_DATA', class_map == G.NO_DATA),
    }
```

(This makes every later reference to `comparisons['mapped_change_area']` in the rest of the cloned
`main()` and in `write_report` invalid by name — update those two call sites, in `main()`'s result
dict and in `write_report`, to read `comparisons['v1_vs_v3_mapped_change_area']` instead. Leave
`boundary_allocation_divergence` and `unsupported_not_compared` exactly as F7 computed them, since
those already use `v2_flagged`/`class_map`, which after this patch are F12's own v3 outputs — i.e.
the boundary-divergence section now describes v1-vs-v3, which is the correct, honest thing for a v3
rerun to report.)

**3e. Update the result dict's status and output paths** — find the `result = {...}` block's
`'status': 'FAIL',` line (F7 hardcodes FAIL because its gate is known to fail) and replace with:

```python
    result = {
        'status': 'BLOCKED' if f11['status'] == 'BLOCKED' else ('PASS' if (f11['status'] == 'PASS' and block_sum['pass']) else 'FAIL'),
        'gate_recalibration_status': f11['status'],
```

Replace every `experiments/results/f7.json` / `experiments/results/f7_REPORT.md` /
`experiments/results/f7_wow_figure.png` path string in the file with the `f12` equivalents. Replace
the `'caveat': (...)` value with:

```python
        'caveat': (
            'F11 was BLOCKED; this run could not use a recalibrated tau.' if f11['status'] == 'BLOCKED'
            else "F11's sensitivity-robust keep rule did not PASS in all three endmember conditions; "
                 'this map describes what gate v2 with the recalibrated tau produces, not a validated '
                 'detector.' if f11['status'] != 'PASS'
            else "F11 PASSED its sensitivity-robust keep rule across all three endmember conditions "
                 '(point estimate, e_b +1SD, e_b -1SD); this is the first gate v2 configuration in '
                 'this program to clear its own detection keep rule honestly.'
        ),
```

- [ ] **Step 4: Run the pure-function tests to verify they pass**

Run: `pytest tests/test_f12_wayanad_v3.py -v -k "not f12_json"`
Expected: PASS for all 5 non-skipped tests (COMPARABLE_PAIRS_V3 checks, the guard, and the F7
non-mutation regression check).

- [ ] **Step 5: Run the real F12 experiment**

Run: `python -m experiments.f12_wayanad_v3`
Expected: if F11 was BLOCKED, this writes a BLOCKED f12.json immediately and exits. Otherwise, real
SR inference reruns (same cost as F7's ~46s for 4 dates on this crop), writes
`experiments/results/f12.json` and `experiments/results/f12_REPORT.md` with whatever status is
actually earned.

- [ ] **Step 6: Run the now-unskipped integration tests**

Run: `pytest tests/test_f12_wayanad_v3.py -v`
Expected: all tests PASS.

- [ ] **Step 7: Commit**

```bash
git add experiments/f12_wayanad_v3.py tests/test_f12_wayanad_v3.py
git add experiments/results/f12.json experiments/results/f12_REPORT.md
git commit -m "F12: Wayanad v3 rerun with F11's recalibrated tau, status reported honestly"
```

---

### Task 6: Full regression suite and final verification

**Files:**
- No new files. This task only runs and verifies.

**Interfaces:**
- Consumes: everything from Tasks 1–5.
- Produces: a verification record appended to `experiments/results/f11_REPORT.md` and
  `f12_REPORT.md` confirming the full suite passes.

- [ ] **Step 1: Run the complete test suite excluding known pre-existing, unrelated failures**

Run: `pytest tests/ -q --ignore=tests/test_l4s_mirror.py --ignore=tests/test_wayanad_evidence.py`

Expected: all pass except the single known pre-existing failure,
`tests/test_placebo.py::TestFARPixel::test_far_pixel_basic` (confirmed by every prior agent in this
program, including F1/F2/F7/F9, to predate the fix wave entirely and live in the already-quarantined
`trustsr/placebo.py`). If any OTHER test fails, stop and diagnose before proceeding — do not commit
over a new regression.

- [ ] **Step 2: Confirm no F0–F10 file was modified**

Run: `git diff experiments/fix-v2 -- experiments/results/f0.json experiments/results/f1.json experiments/results/f2.json experiments/results/f3.json experiments/results/f5.json experiments/results/f6.json experiments/results/f7.json experiments/results/f9.json experiments/results/x2.json experiments/results/x3.json experiments/results/x4.json experiments/results/x5.json experiments/results/x6.json experiments/results/x9.json experiments/results/x10.json experiments/results/x11.json RESULTS_V2.md RESULTS_EXCEPTIONAL.md configs/fix.yaml 2>/dev/null`

Expected: empty diff for every one of these paths (some may not exist and will just be skipped by
git — that's fine). Any non-empty output here is a violation of the plan's Global Constraints and
must be reverted before continuing.

- [ ] **Step 3: Append a final status line to each new report**

Edit `experiments/results/f11_REPORT.md` and `experiments/results/f12_REPORT.md`, appending:

```markdown

## Regression verification

Full test suite (`pytest tests/ -q --ignore=tests/test_l4s_mirror.py --ignore=tests/test_wayanad_evidence.py`)
passes except the single known pre-existing `tests/test_placebo.py::TestFARPixel::test_far_pixel_basic`
failure (quarantined legacy code, predates this program). No F0-F10 result file, and no part of
`trustsr/gate_v2.py`'s `fraction_sigma` or `apply_gate_v2`, was modified by F11/F12.
```

- [ ] **Step 4: Commit**

```bash
git add experiments/results/f11_REPORT.md experiments/results/f12_REPORT.md
git commit -m "F11/F12: regression verification — full suite passes, no F0-F10 file touched"
```

- [ ] **Step 5: Push the branch**

```bash
git push -u origin experiments/fix-v3
```
