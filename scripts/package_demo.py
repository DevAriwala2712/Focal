"""Copy validated real COG outputs into the offline Streamlit bundle."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from risk.common import digest, write_json
from risk.common import load_config


def main():
    root = Path(__file__).resolve().parent.parent
    cfg, _, _ = load_config(root/'configs/pipeline.yaml')
    source = root/'data/wayanad/demo'
    manifest = json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    stac = json.loads((root/cfg['fetch']['stack']).with_suffix('.json').read_text(encoding='utf-8'))
    expected = [cfg['imagery']['longitude'], cfg['imagery']['latitude'], cfg['imagery']['aoi_size_m']]
    if manifest.get('source_aoi') != expected or stac.get('aoi') != expected:
        raise ValueError('Refusing to package a tile from the wrong AOI')
    target = root/'demo/assets'
    target.mkdir(parents=True, exist_ok=True)
    checksums = {}
    for key, path in manifest['files'].items():
        original = Path(path)
        if original.parent.resolve() != source.resolve():
            raise ValueError('Demo source manifest points outside the validated output directory')
        destination = target/original.name
        shutil.copy2(original, destination)
        checksums[key] = digest(destination)
        manifest['files'][key] = original.name
    manifest['source_stack'] = 'Reproduce with scripts/fetch_wayanad.py; 10 km source COG is not bundled.'
    manifest['source_item_ids'] = {row['date']: [item['id'] for item in row['sources']] for row in stac['dates']}
    manifest['checkpoint_sha256'] = digest(root/'checkpoints/r5-smoke.safetensors')
    manifest['asset_sha256'] = checksums
    write_json(target/'manifest.json', manifest)
    print(f'Packaged {len(checksums)} real COGs in {target}')


if __name__ == '__main__':
    main()
