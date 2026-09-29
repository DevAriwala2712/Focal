"""Independently audit the NRSC-mapped Wayanad landslide-site AOI."""
from __future__ import annotations

import json

from risk.common import load_config, write_json
from risk.r2_imagery import probe


def main():
    cfg, root, _ = load_config('configs/pipeline.yaml')
    cfg['paths']['results'] = 'risk/results/event_aoi'
    result = probe(cfg, root)
    write_json(root/cfg['paths']['results']/'r2.json', result)
    print(json.dumps(result, indent=2))
    if result['status'] != 'PASS':
        raise RuntimeError(result['reason'])


if __name__ == '__main__':
    main()
