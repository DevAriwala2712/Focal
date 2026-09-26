"""Distinguish downloaded inventories from ready-to-use temporal labels."""
from __future__ import annotations

import json
import io
import sqlite3
import zipfile

from risk.common import digest, download, run_cli


def inspect_l4s_pair(image_path, mask_path, band_indices):
    import h5py
    import numpy as np
    if list(band_indices) != [3, 2, 1, 7]:
        raise ValueError('Only RGBN band indices [3, 2, 1, 7] are permitted')
    with h5py.File(image_path, 'r') as f:
        shape = f['img'].shape
        if shape != (128, 128, 14):
            raise ValueError('Unexpected Landslide4Sense image shape')
        image = np.stack([f['img'][:, :, i] for i in band_indices], axis=-1)
    with h5py.File(mask_path, 'r') as f:
        labels = f['mask'][:]
    if labels.shape != (128, 128) or not np.isin(labels, [0, 1]).all():
        raise ValueError('Expected a 128x128 binary mask')
    if not np.isfinite(image).all():
        raise ValueError('Nonfinite values in selected RGBN bands')
    return {'labelled_pair_verified': True, 'image_shape': list(shape),
            'selected_shape': list(image.shape), 'selected_band_indices': list(band_indices),
            'mask_values': np.unique(labels).tolist(), 'landslide_pixels': int(labels.sum()),
            'image_sha256': digest(image_path), 'mask_sha256': digest(mask_path),
            'temporal_pair': False, 'georeferencing_verified': False}


def inspect_l4s_archive(path, max_member_bytes):
    import h5py
    import numpy as np
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        for name in sorted(names):
            if '/img/image_' not in name or not name.endswith('.h5'):
                continue
            mask = name.replace('/img/image_', '/mask/mask_')
            if mask not in names:
                continue
            if any(archive.getinfo(n).file_size > max_member_bytes for n in [name, mask]):
                raise ValueError('HDF5 archive member exceeds download limit')
            with h5py.File(io.BytesIO(archive.read(name)), 'r') as f:
                image = f['img'][:]
            with h5py.File(io.BytesIO(archive.read(mask)), 'r') as f:
                labels = f['mask'][:]
            if image.shape != (128, 128, 14) or labels.shape != (128, 128):
                raise ValueError('Landslide4Sense image/mask shape differs from published contract')
            if not np.isfinite(image).all() or not np.isin(labels, [0, 1]).all():
                raise ValueError('Invalid Landslide4Sense values')
            return {'labelled_pair_verified': True, 'image_member': name, 'mask_member': mask,
                    'image_shape': list(image.shape), 'mask_values': np.unique(labels).tolist()}
    raise ValueError('No matching Landslide4Sense image/mask HDF5 files in archive')


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
            if magic.startswith(b'PK'):
                findings['l4s'][key].update(inspect_l4s_archive(path, policy['max_download_bytes']))
        except Exception as exc:
            findings['errors'].append({'source': settings[key], 'error': str(exc)})
    if 'l4s_mirror' in settings:
        mirror = settings['l4s_mirror']
        try:
            image = download(mirror['image_url'], cache / 'ibm-nasa' / 'image_1.h5', policy)
            mask = download(mirror['mask_url'], cache / 'ibm-nasa' / 'mask_1.h5', policy)
            findings['l4s']['ibm_nasa_mirror'] = {
                **inspect_l4s_pair(image, mask, mirror['band_indices']),
                'source': mirror, 'provenance': 'IBM-NASA hosted redistribution; not byte-compared with unavailable IARAI original'}
        except Exception as exc:
            findings['errors'].append({'source': mirror['image_url'], 'error': str(exc)})
    complete = (findings['inventory'].get('paired_rasters_downloaded', False) and
                any(value.get('labelled_pair_verified', False) for value in findings['l4s'].values()))
    return {'status': 'PASS' if complete else 'BLOCKED', **findings,
            'reason': ('Both sources yielded inspectable label samples; a held-out calibration split still needs preparation.'
                       if complete else 'See inventory.paired_rasters_downloaded for the real pair access check. '
                       'Landslide4Sense image/mask content must be inspected before calibration use.'),
            'l4s_label_contract_from_readme': 'Single-date 128x128, ~10 m, 14 channels; training binary masks. '
                      'Only channels B02/B03/B04/B08 permitted. Not paired temporal or 2.5 m truth.',
            'checked_urls': settings}


if __name__ == '__main__':
    run_cli('r4', probe)
