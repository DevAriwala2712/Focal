"""Step 2 (ADR-002): was the scar visible earlier? Valid SCL fraction inside the footprint per post-event date. Report only."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np

from experiments.common import Blocked, finalize_result
from experiments.wayanad_evidence import config as C
from experiments.wayanad_evidence.data import scl_valid
from experiments.wayanad_evidence.fetch import scl_mosaic_10m
from risk.common import write_json


def footprint_valid_fraction(scl10, footprint, im):
    """(valid fraction, valid pixel count, footprint pixel count) of SCL-usable pixels inside the footprint only."""
    inside = np.asarray(footprint, bool)
    ok = scl_valid(scl10, im) & inside
    return (float(ok.sum() / inside.sum()), int(ok.sum()), int(inside.sum()))


def run(cfg, root, config_hash):
    started = datetime.now(timezone.utc).isoformat()
    out, cache = C.outputs(cfg, root), C.cache(cfg, root)
    step1 = json.loads((out / 'step1.json').read_text(encoding='utf-8'))
    if step1['status'] == 'BLOCKED':
        raise Blocked('step 1 is BLOCKED, so no footprint exists: ' + step1.get('reason', ''), evidence='real')
    audit = json.loads((out / 'audit_acquisitions.json').read_text(encoding='utf-8'))
    with np.load(cache / 'step1_state.npz') as z:
        footprint, post = z['footprint'], str(z['post'])
    im = cfg['phase0']['imagery']
    cand = [r for r in audit['rows'] if cfg['event_date'] < r['date'] < post and r['cloud_shadow_pct_aoi'] < 100.0]
    by_date = {}
    for r in cand:
        by_date.setdefault(r['date'], []).append(r)
    table = []
    for date in sorted(by_date):
        row = min(by_date[date], key=lambda r: r['cloud_shadow_pct_aoi'])
        items = [{'id': i} for r_ in by_date[date] for i in r_['item_ids']]
        scl10, _ = scl_mosaic_10m(cfg, items, cache)
        frac, valid_px, fp_px = footprint_valid_fraction(scl10, footprint, im)
        table.append({'date': date, 'days_after_event': (datetime.fromisoformat(date) - datetime.fromisoformat(cfg['event_date'])).days,
                      'aoi_cloud_shadow_pct': row['cloud_shadow_pct_aoi'], 'footprint_valid_fraction': frac,
                      'footprint_valid_px': valid_px, 'footprint_px': fp_px,
                      'passes': frac >= cfg['footprint']['min_valid_fraction']})
    passing = [t for t in table if t['passes']]
    earliest = passing[0] if passing else None
    result = {'status': 'PASS', 'evidence': 'real',
              'measurements': {'threshold_min_valid_fraction': cfg['footprint']['min_valid_fraction'],
                               'candidate_rule': 'every acquisition date after the event and before the chosen post date with '
                                                 'AOI cloud+shadow < 100 % in this AOI\'s audit (audit_acquisitions.json)',
                               'chosen_post_date': post, 'table': table,
                               'earliest_passing_date': earliest,
                               'note': 'report only: the post date of this run was NOT switched. The footprint is used for date '
                                       'selection, never as a label.'},
              'limitations': ['SCL cloud/shadow classes are a model-based estimate; footprint-valid does not mean the scar was '
                              'visible without haze or shadow',
                              'footprint comes from the January-vs-post-date NDVI drop, so it is a circular basis for anything but date selection']}
    return finalize_result('step2', result, config_hash, started)


def main():
    cfg, root, config_hash = C.load()
    try:
        result = run(cfg, root, config_hash)
    except Blocked as exc:
        result = finalize_result('step2', {'status': 'BLOCKED', 'evidence': 'real', 'reason': str(exc)}, config_hash,
                                 datetime.now(timezone.utc).isoformat())
    write_json(C.outputs(cfg, root) / 'step2.json', result)
    print(json.dumps(result, indent=2, default=str))


if __name__ == '__main__':
    main()
