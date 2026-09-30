"""F9 verification scratch script (check 2: no_placebo_code_path_reads_post_event_date).

This does NOT read the guard and reason about it. It ATTEMPTS to load the post-event date through every
entry point F1/F2/F7 use for placebo/calibration scoring, and records whether the attempt was refused.

Paths attacked:
  1. trustsr.placebo_v2.guarded_load_date          -- the sanctioned door
  2. trustsr.placebo_v2.build_sr_fold              -- held_out = post date (also proves the UNGUARDED
                                                     _load_dihedral is unreachable with a post date,
                                                     because guarded_load_date runs first)
  3. trustsr.placebo_v2.build_sr_fold              -- post date smuggled into pre_dates
  4. trustsr.placebo_v2.build_10m_fold             -- post date in pre_dates and as held_out
  5. experiments.f2_gate_v2.build                  -- SR_POOL monkeypatched to contain the post date
                                                     (this is the function F7's recompute_tau calls, so
                                                     it is the tau/calibration path)
  6. malformed / non-ISO spellings of the post date -- does the guard fail OPEN or CLOSED?
"""
from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

import numpy as np

import trustsr.placebo_v2 as P

ROOT = Path(__file__).resolve().parents[1]
POST = '2024-12-06'
results = []


def attempt(name, fn, expect_refused=True):
    try:
        fn()
        outcome, detail = 'LOADED', 'no exception raised'
    except P.PostEventDateBlocked as e:
        outcome, detail = 'REFUSED_BY_GUARD', f'PostEventDateBlocked: {e}'
    except Exception as e:                                  # noqa: BLE001
        outcome, detail = f'OTHER_ERROR({type(e).__name__})', str(e)[:200]
    ok = (outcome == 'REFUSED_BY_GUARD') if expect_refused else (outcome == 'LOADED')
    results.append({'attack': name, 'outcome': outcome, 'detail': detail,
                    'expected_refusal': expect_refused, 'verdict': 'PASS' if ok else 'FAIL'})
    print(f'  [{"PASS" if ok else "FAIL"}] {name}: {outcome}')
    return ok


def main():
    # use the SAME real config loader F1/F2/F7 use, so the fold builders get as far as they would in a
    # real run and the guard is genuinely the thing that stops them (an empty cfg would crash earlier,
    # which would prove nothing)
    from experiments.wayanad_evidence import config as C
    cfg, ev_root, _h = C.load()
    cache = C.cache(cfg, ev_root)
    crop = (256, 896, 128, 640)

    print('1. the sanctioned door')
    attempt('guarded_load_date(post_date)', lambda: P.guarded_load_date(cfg, cache, POST))
    attempt('guarded_load_date(event_date)', lambda: P.guarded_load_date(cfg, cache, '2024-07-30'))
    attempt('guarded_load_date(datetime.date(2024,12,6))',
            lambda: P.guarded_load_date(cfg, cache, datetime.date(2024, 12, 6)))

    print('2-3. build_sr_fold (the SR placebo fold builder)')
    attempt('build_sr_fold(held_out=post)',
            lambda: P.build_sr_fold(cfg, cache, cache, crop, POST, ['2024-01-16', '2024-01-21'], 0.30))
    attempt('build_sr_fold(post smuggled into pre_dates)',
            lambda: P.build_sr_fold(cfg, cache, cache, crop, '2024-01-26', ['2024-01-16', POST], 0.30))

    print('4. build_10m_fold')
    attempt('build_10m_fold(held_out=post)',
            lambda: P.build_10m_fold(cfg, cache, crop, POST, ['2024-01-16', '2024-01-21'], 0.30))
    attempt('build_10m_fold(post in pre_dates)',
            lambda: P.build_10m_fold(cfg, cache, crop, '2024-01-26', ['2024-01-16', POST], 0.30))

    print("5. F2's build() -- the function F7's recompute_tau calls to make tau")
    import experiments.f2_gate_v2 as F2

    def attack_f2_pool():
        saved = F2.SR_POOL
        try:
            F2.SR_POOL = ['2024-01-16', '2024-01-21', POST]
            F2.build(cfg, cache)
        finally:
            F2.SR_POOL = saved
    attempt('f2_gate_v2.build with post date in SR_POOL', attack_f2_pool)

    print('6. malformed spellings -- must fail CLOSED, never open')
    for s in ['2024-12-06', '2024-7-30', '20241206', '2024/12/06', '2025-01-01', '9999-01-01']:
        attempt(f'guarded_load_date({s!r})', lambda s=s: P.guarded_load_date(cfg, cache, s))

    print('7. control: a genuine pre-event date must NOT be refused (the guard is not just always-raise)')
    import experiments.wayanad_evidence.data as dm
    real = dm.load_date
    dm.load_date = lambda cfg, cache, date: {'date': date, 'sentinel': 'stub'}
    try:
        attempt('guarded_load_date(2024-07-29) [stubbed loader]',
                lambda: P.guarded_load_date(cfg, cache, '2024-07-29'), expect_refused=False)
        attempt('guarded_load_date(2024-01-16) [stubbed loader]',
                lambda: P.guarded_load_date(cfg, cache, '2024-01-16'), expect_refused=False)
    finally:
        dm.load_date = real

    n_fail = sum(1 for r in results if r['verdict'] == 'FAIL')
    print(f'\n{len(results)} attacks, {n_fail} FAILED')
    json.dump({'n_attacks': len(results), 'n_failed': n_fail, 'attacks': results},
              open(ROOT / 'experiments/f9_guard_attacks.json', 'w'), indent=1)
    return 1 if n_fail else 0


if __name__ == '__main__':
    sys.exit(main())
