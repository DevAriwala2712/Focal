"""Reproduce the audited 2024 Chooralmala landslide-site imagery stack."""
from risk.common import load_config
from trustsr.fetch import fetch


if __name__ == '__main__':
    cfg, root, _ = load_config('configs/pipeline.yaml')
    settings = cfg['imagery']
    evidence = root/'risk/results/event_aoi'
    output = fetch((settings['longitude'], settings['latitude'], settings['aoi_size_m']),
                   settings['start'], settings['end'],
                   audit=(evidence/'r2_catalog.json', evidence/'r2_acquisitions.json'))
    print(output)
