"""Bounded publisher ZIP access and explicit smoke-only pair preparation."""
from __future__ import annotations

import ast
import csv
import io
import json
import struct
import zipfile
import zlib
from pathlib import Path

import numpy as np
import requests

from risk.common import digest, download, retry, run_cli, write_json


def decode_member(data, name, limit):
    h = struct.unpack('<4s5H3I2H', data[:30])
    if h[0] != b'PK\x03\x04' or h[2] != 0 or h[3] != 8:
        raise ValueError('Unsupported ZIP local header')
    if h[7] > limit or h[8] > limit:
        raise ValueError('ZIP member exceeds limit')
    if data[30:30+h[9]].decode('utf-8') != name:
        raise ValueError('ZIP member name mismatch')
    start = 30 + h[9] + h[10]
    decoder = zlib.decompressobj(-15)
    output = decoder.decompress(data[start:start+h[7]], limit + 1)
    if len(output) != h[8] or not decoder.eof or zlib.crc32(output) != h[6]:
        raise ValueError('ZIP size or CRC mismatch')
    return output


def byte_range(url, start, count, policy):
    if count <= 0 or count > policy['max_download_bytes']:
        raise ValueError('Byte range exceeds configured limit')
    end = start + count - 1
    def get():
        with requests.get(url, headers={'Range': f'bytes={start}-{end}'},
                          timeout=policy['timeout_seconds'], stream=True) as response:
            if response.status_code != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {start}-{end}/'):
                raise ValueError('Server did not return the exact requested byte range')
            data = bytearray()
            for block in response.iter_content(1024 * 1024):
                data.extend(block)
                if len(data) > count:
                    raise ValueError('Oversized range response')
            if len(data) != count:
                raise ValueError('Truncated range response')
            return bytes(data)
    return retry(get, policy, url)


class RemoteZip(io.RawIOBase):
    def __init__(self, url, size, policy):
        self.url, self.size, self.policy, self.position = url, size, policy, 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = (self.position if whence == 1 else self.size if whence == 2 else 0) + offset
        if self.position < 0:
            raise ValueError('Negative ZIP seek')
        return self.position

    def read(self, count=-1):
        count = self.size - self.position if count < 0 else min(count, self.size-self.position)
        if count == 0:
            return b''
        # Central directory can exceed the per-request limit; fetch bounded chunks.
        parts = []
        while count:
            amount = min(count, self.policy['max_download_bytes'])
            parts.append(byte_range(self.url, self.position, amount, self.policy))
            self.position += amount
            count -= amount
        return b''.join(parts)


class _LazyArchiveIndex(dict):
    """Lazy dict that loads from JSON only on first access. Avoids holding entire index in memory."""
    def __init__(self, path):
        self.path = path
        self._loaded = False
        super().__init__()

    def _ensure_loaded(self):
        if not self._loaded:
            records = json.loads(self.path.read_text(encoding='utf-8'))
            super().update({i['name']: i for i in records})
            self._loaded = True

    def __getitem__(self, key):
        self._ensure_loaded()
        return super().__getitem__(key)

    def __contains__(self, key):
        self._ensure_loaded()
        return super().__contains__(key)

    def get(self, key, default=None):
        self._ensure_loaded()
        return super().get(key, default)


def archive_index(archive, cache, policy):
    path = cache / (archive['name'] + '.index.json')
    if not path.exists():
        with zipfile.ZipFile(RemoteZip(archive['url'], archive['bytes'], policy)) as z:
            records = [{'name': i.filename, 'size': i.file_size,
                        'compressed_size': i.compress_size, 'offset': i.header_offset} for i in z.infolist()]
        path.write_text(json.dumps(records), encoding='utf-8')
    return _LazyArchiveIndex(path)


def get_member(archive, index, name, target, policy):
    sidecar = target.with_suffix(target.suffix + '.source.json')
    if target.exists() and sidecar.exists():
        record = json.loads(sidecar.read_text())
        if record['archive_url'] == archive['url'] and record['member'] == name and record['sha256'] == digest(target):
            return record
    info = index[name]
    header = byte_range(archive['url'], info['offset'], 30, policy)
    h = struct.unpack('<4s5H3I2H', header)
    if h[7] != info['compressed_size'] or h[8] != info['size']:
        raise ValueError('Index and local header sizes differ')
    body = byte_range(archive['url'], info['offset']+30, h[9]+h[10]+h[7], policy)
    data = decode_member(header+body, name, policy['max_download_bytes'])
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix('.part')
    temp.write_bytes(data)
    temp.replace(target)
    record = {'archive_url': archive['url'], 'member': name, 'sha256': digest(target),
              'bytes': len(data), 'crc32': h[6], 'full_archive_checksum_verified': False}
    write_json(sidecar, record)
    return record


def choose_crop(scl, valid, size, land_classes, min_land_fraction):
    good = np.isin(scl, [4, 5, 6, 7]) & (valid == 1)
    land = np.isin(scl, land_classes)
    rows = sorted(range(scl.shape[0]-size+1), key=lambda i: abs(i-(scl.shape[0]-size)/2))
    cols = sorted(range(scl.shape[1]-size+1), key=lambda i: abs(i-(scl.shape[1]-size)/2))
    for y in rows:
        for x in cols:
            sl = np.s_[y:y+size, x:x+size]
            if good[sl].all() and land[sl].mean() >= min_land_fraction:
                return y, x
    return None


def harmonize(low, high_dn, scale, min_correlation):
    """Fit per-band DN -> LR units; empirical harmonization, NOT physical calibration."""
    c, h, w = low.shape
    coarse = high_dn.reshape(c, h, scale, w, scale).mean((2, 4))
    out, coefficients = [], []
    for a, b, high in zip(coarse, low, high_dn):
        if a.std() < 1e-8 or b.std() < 1e-8:
            raise ValueError('Insufficient variance for radiometric fit')
        correlation = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
        if not np.isfinite(correlation) or correlation < min_correlation:
            raise ValueError(f'Poor HR/LR correlation: {correlation:.4f}')
        gain, offset = np.linalg.lstsq(np.column_stack([a.ravel(), np.ones(a.size)]), b.ravel(), rcond=None)[0]
        if gain <= 0:
            raise ValueError('Nonpositive radiometric gain')
        fitted = high * gain + offset
        if not np.isfinite(fitted).all():
            raise ValueError('Nonfinite harmonized target')
        out.append(fitted)
        coefficients.append({'gain': float(gain), 'offset': float(offset), 'correlation': correlation,
                             'coarse_rmse': float(np.sqrt(np.mean((a*gain+offset-b)**2)))})
    return np.asarray(out, dtype='float32'), coefficients


def probe(cfg, root):
    """Prepare three real clear land crops, preserving geographic CRS and x4 grid."""
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.warp import reproject, Resampling
    from rasterio.windows import Window
    from risk.calibration import write_cog
    from risk.r5_finetune import validate_manifest
    settings, policy = cfg['preparation'], cfg['network']
    cache = root / settings['cache']
    cache.mkdir(parents=True, exist_ok=True)
    files = {}
    for key in ['metadata', 'split']:
        spec = settings[key]
        files[key] = download(spec['url'], cache/spec['name'], policy, sha256=spec['sha256'])
    with files['split'].open(encoding='utf-8') as f:
        train = {r['tile'] for r in csv.DictReader(f) if r['split'] == 'train'}
    with files['metadata'].open(encoding='utf-8') as f:
        candidates = [r for r in csv.DictReader(f) if r['tile'] in train and
                      abs(int(r['delta'])) <= settings['max_date_gap_days'] and
                      float(r['cloud_cover']) <= settings['max_metadata_cloud_pct'] and
                      r['IPCC Class'] in settings['land_cover_classes']]
    candidates.sort(key=lambda r: (abs(int(r['delta'])), float(r['cloud_cover']), r['tile'], int(r['n'])))
    archives = settings['archives']
    indices = {key: archive_index(spec, cache, policy) for key, spec in archives.items()}
    manifest_path = root / cfg['training']['pair_manifest']
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pairs, checked, seen = [], [], set()
    size, scale = cfg['model']['native_tile'], cfg['model']['scale']
    for row in candidates:
        aoi, n = row['tile'], row['n']
        if aoi in seen:
            continue
        seen.add(aoi)
        if len(checked) >= settings['max_candidates']:
            break
        print(f'Inspecting {aoi}, revisit {n}', flush=True)
        try:
            folder = cache/'prepared-sources'/aoi
            provenance = []
            paths = {}
            for kind, suffix in [('scl', 'SCL'), ('valid', 'dataMask'), ('lr', 'L2A_data')]:
                name = f'lr_dataset/{aoi}/L2A/{aoi}-{n}-{suffix}.tiff'
                paths[kind] = folder/Path(name).name
                provenance.append(get_member(archives['lr'], indices['lr'], name, paths[kind], policy))
            with rasterio.open(paths['scl']) as s, rasterio.open(paths['valid']) as v, rasterio.open(paths['lr']) as lr:
                if s.transform != lr.transform or v.transform != lr.transform or s.shape != lr.shape or v.shape != lr.shape or s.crs != lr.crs or v.crs != lr.crs:
                    raise ValueError('Source quality grids differ')
                crop = choose_crop(s.read(1), v.read(1), size, settings['land_scl_classes'], settings['min_land_fraction'])
                if crop is None:
                    raise ValueError('No fully valid clear land crop')
                y, x = crop
                bounds = ast.literal_eval(row['bounds'])
                if not np.allclose(list(lr.bounds), bounds, rtol=0, atol=1e-10):
                    raise ValueError('Published bounds and LR bounds differ')
                window = Window(x, y, size, size)
                low = lr.read([4,3,2,8], window=window)
                transform, crs = lr.window_transform(window), lr.crs
            name = f'hr_dataset/{aoi}/{aoi}_ps.tiff'
            paths['hr'] = folder/Path(name).name
            provenance.append(get_member(archives['hr'], indices['hr'], name, paths['hr'], policy))
            with rasterio.open(paths['hr']) as hr:
                if hr.count != 4 or hr.crs != crs or not hr.transform.is_identity:
                    raise ValueError('HR source differs from inspected publisher contract')
                source = hr.read().astype('float32')
                if not np.isfinite(source).all():
                    raise ValueError('Invalid HR source')
                restored = from_bounds(*bounds, hr.width, hr.height)
                high_transform = transform * transform.scale(1/scale, 1/scale)
                high = np.full((4,size*scale,size*scale), np.nan, dtype='float32')
                for b in range(4):
                    reproject(source[b], high[b], src_transform=restored, src_crs=crs,
                              dst_transform=high_transform, dst_crs=crs,
                              src_nodata=0, dst_nodata=np.nan, resampling=Resampling.average)
            if not np.isfinite(low).all() or not np.isfinite(high).all():
                raise ValueError('Crop contains invalid or nodata pixels')
            target, coefficients = harmonize(low, high, scale, settings['min_band_correlation'])
            lr_path, hr_path = manifest_path.parent/f'{aoi}_lr.tif', manifest_path.parent/f'{aoi}_hr.tif'
            write_cog(lr_path, low, crs, transform, cfg['model']['bands'], None)
            write_cog(hr_path, target, crs, high_transform, cfg['model']['bands'], None)
            pairs.append({'aoi_id': aoi, 'split': 'train', 'lr': lr_path.name, 'hr': hr_path.name,
                          'lr_sha256': digest(lr_path), 'hr_sha256': digest(hr_path),
                          'lr_scale': 1, 'lr_offset': 0, 'hr_scale': 1, 'hr_offset': 0,
                          'source_urls': [spec['url'] for spec in archives.values()],
                          'originals': provenance, 'metadata': row, 'crop_row_col': [y,x],
                          'hr_transform_reconstruction': 'Published WGS84 bounds; verified against LR bounds; north-up raster',
                          'radiometry': {'method': 'Per-band affine fit from averaged HR DN to paired LR reflectance',
                                         'physical_hr_reflectance': False, 'coefficients': coefficients},
                          'resolution': 'Native geographic ~10m LR and exact x4 subdivision; not exact metric 2.5m',
                          'use': 'Training smoke test only; no validation or accuracy claim'})
            checked.append({'aoi': aoi, 'status': 'PREPARED'})
        except Exception as exc:
            checked.append({'aoi': aoi, 'status': 'REJECTED', 'reason': str(exc)})
            print(f'Rejected: {exc}', flush=True)
        write_json(root/cfg['paths']['results']/'worldstrat_candidates.json', {'checked': checked})
        if len(pairs) == cfg['training']['min_pairs']:
            break
    if len(pairs) < cfg['training']['min_pairs']:
        return {'status': 'BLOCKED', 'prepared': len(pairs), 'checked': checked, 'reason': 'Not enough verified pairs; no ready manifest published'}
    manifest = {'dataset': 'WorldStrat', 'pairs': pairs, 'smoke_only': True,
                'metadata_sha256': digest(files['metadata']), 'split_sha256': digest(files['split'])}
    validate_manifest(manifest, manifest_path.parent, cfg['model']['bands'])
    write_json(manifest_path, manifest)
    return {'status': 'PASS', 'prepared': len(pairs), 'checked': checked,
            'manifest_sha256': digest(manifest_path), 'manifest': manifest,
            'scope': 'Empirically harmonized real WorldStrat pairs for optimizer smoke test only'}


if __name__ == '__main__':
    run_cli('worldstrat_preparation', probe)
