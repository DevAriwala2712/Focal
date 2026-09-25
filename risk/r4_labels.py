"""Distinguish downloaded inventories from ready-to-use temporal labels."""
from __future__ import annotations

import json
import sqlite3

from risk.common import digest, download, run_cli


def inspect_inventory(path):
    with sqlite3.connect(f'{path.as_uri()}?mode=ro', uri=True) as db:
        table, srs_id, min_x, min_y, max_x, max_y = db.execute(
            "SELECT table_name, srs_id, min_x, min_y, max_x, max_y FROM gpkg_contents WHERE data_type='features'").fetchone()
        safe = '"' + table.replace('"', '""') + '"'
        count = db.execute(f'SELECT COUNT(*) FROM {safe}').fetchone()[0]
        columns = [row[1] for row in db.execute(f'PRAGMA table_info({safe})')]
        dates = db.execute(f'SELECT DISTINCT event_date FROM {safe}').fetchall() if 'event_date' in columns else []
    return {'table': table, 'feature_count': count, 'srs_id': srs_id,
            'bounds': [min_x, min_y, max_x, max_y], 'columns': columns,
            'event_dates': [d[0] for d in dates], 'sha256': digest(path)}


def probe(cfg, root):
    """Download actual inventory and check the original Landslide4Sense share."""
    settings, policy = cfg['labels'], cfg['network']
    cache = root / cfg['paths']['cache'] / 'labels'
    findings = {'inventory': {}, 'l4s': {}, 'errors': []}
    try:
        inventory = download(settings['inventory_url'], cache / 'colombia.gpkg', policy)
        findings['inventory'] = inspect_inventory(inventory)
        notebook = download(settings['notebook_url'], cache / 'processing.ipynb', policy)
        cells = json.loads(notebook.read_text(encoding='utf-8'))['cells']
        code = '\n'.join(''.join(c['source']) for c in cells if c['cell_type'] == 'code')
        findings['inventory'].update(
            download_verified=True, recipe_uses_planetary_computer='planetarycomputer.microsoft.com' in code,
            notebook_sha256=digest(notebook), label_values_in_recipe=[0, 255],
            label_meaning='Landslide polygons rasterized to the Sentinel-2 grid; not building-damage labels',
            paired_rasters_downloaded=False)
        from risk.calibration import download_sample
        findings['inventory'].update(download_sample(cfg, root, inventory))
    except Exception as exc:
        findings['errors'].append({'source': settings['inventory_url'], 'error': str(exc)})
    for key in ['l4s_training_share', 'l4s_download_url']:
        try:
            path = download(settings[key], cache / (key + '.response'), policy)
            with path.open('rb') as f:
                magic = f.read(8)
            findings['l4s'][key] = {'retrieved_bytes': path.stat().st_size, 'sha256': digest(path),
                                    'zip_header': magic.startswith(b'PK'), 'html_only': magic.lstrip().startswith(b'<')}
        except Exception as exc:
            findings['errors'].append({'source': settings[key], 'error': str(exc)})
    return {'status': 'BLOCKED', **findings,
            'reason': 'See inventory.paired_rasters_downloaded for the real pair access check. '
                      'Landslide4Sense image/mask content must be inspected before calibration use.',
            'l4s_label_contract_from_readme': 'Single-date 128x128, ~10 m, 14 channels; training binary masks. '
                      'Only channels B02/B03/B04/B08 permitted. Not paired temporal or 2.5 m truth.',
            'checked_urls': settings}


if __name__ == '__main__':
    run_cli('r4', probe)
