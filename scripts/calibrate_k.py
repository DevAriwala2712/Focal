"""Calibrate k only with an independent multi-event 10 m label split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from risk.common import load_config, write_json
from trustsr.evaluate import calibrate_k


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default='risk/results/r4_sample.json')
    parser.add_argument('--output', default='results/calibration.json')
    args = parser.parse_args()
    cfg, root, _ = load_config('configs/pipeline.yaml')
    path = Path(args.manifest)
    document = json.loads(path.read_text(encoding='utf-8'))
    if document.get('sample_only') or document.get('suitable_for_calibration_alone') is False:
        raise ValueError('Phase-0 R4 is a single access-proof sample; no independent calibration set exists. k remains 2.0.')
    entries = document.get('samples', [])
    if len({e['event_id'] for e in entries if e.get('split') == 'calibration'}) < 2:
        raise ValueError('Calibration needs at least two distinct labelled events in the calibration split')
    samples = []
    for entry in entries:
        if entry['split'] != 'calibration':
            continue
        source = (path.parent/entry['npz']).resolve()
        with np.load(source, allow_pickle=False) as archive:
            samples.append({key: archive[key] for key in ('pre_mean', 'post_mean', 'pre_std', 'post_std',
                                                           'parent_pre', 'parent_post', 'valid', 'labels')})
    result = calibrate_k(samples, cfg['calibration']['candidate_k'],
                         parent_drop_threshold=cfg['change']['parent_drop_threshold'])
    result['source_manifest'] = str(path.resolve())
    write_json(root/args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    import sys
    try:
        main()
    except ValueError as exc:
        print(f'Calibration unavailable: {exc}', file=sys.stderr)
        raise SystemExit(2) from None
