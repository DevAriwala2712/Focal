"""Prepare held-out WorldStrat validation AOIs from publisher split metadata."""
from __future__ import annotations

import json

from risk.common import load_config
from risk.worldstrat_prepare import probe


if __name__ == '__main__':
    config, root, _ = load_config('configs/phase0.yaml')
    result = probe(config, root, split='val', target_pairs=3,
                   manifest_path='data/worldstrat/validation/pairs.json')
    print(json.dumps(result, indent=2))
    if result['status'] != 'PASS':
        raise SystemExit(2)
