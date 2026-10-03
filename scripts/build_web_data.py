"""Build the data behind demo/web from the real Wayanad evidence outputs and experiment results.

Nothing here invents a number: every value is read from experiments/wayanad_evidence/outputs/*, experiments/results/*.json,
or recomputed from those rasters (and cross-checked against the stored class map). Missing inputs stop the build.

    .venv/bin/python scripts/build_web_data.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.wayanad_evidence.gate import classify, object_rho, upsample  # noqa: E402
from experiments.wayanad_evidence.geo import label_components  # noqa: E402

OUT_DIR = ROOT / 'experiments' / 'wayanad_evidence' / 'outputs'
RES = ROOT / 'experiments' / 'results'
WEB = ROOT / 'demo' / 'web' / 'data'
SCALE = 4
K = 2.0
PIXEL_M2 = 6.25


def need(p: Path) -> Path:
    if not p.exists():
        raise SystemExit(f'missing required input: {p}')
    return p


def jload(p: Path):
    return json.loads(need(p).read_text(encoding='utf-8'))


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def stretch_rgb(r, g, b, lo=0.0, hi=0.22, gamma=1 / 1.4):
    """One fixed stretch for every date/resolution so before/after and 10 m/2.5 m are comparable."""
    rgb = np.stack([r, g, b], -1).astype('float32')
    rgb = np.clip((rgb - lo) / (hi - lo), 0, 1) ** gamma
    return (rgb * 255 + 0.5).astype('uint8')


def read_stack(date):
    with rasterio.open(need(OUT_DIR / f'stack_{date}.tif')) as ds:
        a = ds.read().astype('float32')
    refl = (a[:4] - 1000.0) / 10000.0  # ESA baseline>=04.00 convention, as in the run's own metadata
    return refl, a[4]


def main():
    (WEB / 'layers').mkdir(parents=True, exist_ok=True)
    (WEB / 'raster').mkdir(parents=True, exist_ok=True)
    manifest = jload(OUT_DIR / 'manifest.json')
    step1 = jload(OUT_DIR / 'step1.json')['measurements']
    step2 = jload(OUT_DIR / 'step2.json')
    step3 = jload(OUT_DIR / 'step3.json')['measurements']
    step4 = jload(OUT_DIR / 'step4.json')['measurements']
    audit = jload(OUT_DIR / 'audit_acquisitions.json')
    pre, post = manifest['pre_dates'], manifest['post_date']
    r0, r1 = step1['crop']['rows']
    c0, c1 = step1['crop']['cols']
    h10, w10 = r1 - r0, c1 - c0

    # ---- 10 m crops (B04,B03,B02,B08 order in the stacks) ----
    pre_refl = np.mean([read_stack(d)[0] for d in pre], axis=0)[:, r0:r1, c0:c1]
    last_pre = pre[-1]
    lp_refl = read_stack(last_pre)[0][:, r0:r1, c0:c1]
    post_refl = read_stack(post)[0][:, r0:r1, c0:c1]

    def save_png(arr, name):
        Image.fromarray(arr).save(WEB / 'layers' / name, optimize=True)

    save_png(stretch_rgb(*lp_refl[:3]), 'pre_10m.png')
    save_png(stretch_rgb(*post_refl[:3]), 'post_10m.png')

    with rasterio.open(need(OUT_DIR / 'sr_post_2p5m.tif')) as ds:
        sr = ds.read().astype('float32')
        sr_bounds, sr_transform, sr_shape = ds.bounds, ds.transform, ds.shape
        crs = ds.crs
    save_png(stretch_rgb(*sr[:3]), 'post_sr_2p5m.png')
    # post 10 m upsampled by pixel replication on the 2.5 m grid, so 10 m / 2.5 m compare pixel-for-pixel
    save_png(upsample(stretch_rgb(*post_refl[:3]).transpose(2, 0, 1), SCALE).transpose(1, 2, 0), 'post_10m_on_2p5m_grid.png')

    # ---- gate inputs, all on the 2.5 m grid of the crop ----
    with rasterio.open(need(OUT_DIR / 'ndvi_drop_2p5m.tif')) as ds:
        d = ds.read(1)
    with rasterio.open(need(OUT_DIR / 'ndvi_sigma_2p5m.tif')) as ds:
        sigma = ds.read(1)
    with rasterio.open(need(OUT_DIR / 'class_map_2p5m.tif')) as ds:
        cls_stored = ds.read(1)
    with rasterio.open(need(OUT_DIR / 'parent_mask.tif')) as ds:
        parent_full = ds.read(1)
    parent10 = (parent_full[r0:r1, c0:c1] == 1)
    nodata = cls_stored == 255  # NO_DATA is k-independent
    parent_hr = upsample(parent10, SCALE)

    # cross-check: re-running the project's own classify() on the shipped rasters must reproduce the stored class map
    again = classify(np.where(np.isnan(d), 0, d), np.where(np.isnan(sigma), 0, sigma), parent_hr, nodata, K)
    if not np.array_equal(again, cls_stored):
        raise SystemExit('classify() on shipped d/sigma/parent/nodata does not reproduce class_map_2p5m.tif; refusing to build')

    def q16(a, scale=10000.0):
        a = np.nan_to_num(a, nan=0.0)  # NO_DATA pixels are masked by class 255 in the viewer
        return np.clip(np.round(a * scale), -32767, 32767).astype('<i2')

    q16(d).tofile(WEB / 'raster' / 'd.i16')
    q16(sigma).tofile(WEB / 'raster' / 'sigma.i16')
    parent10.astype('u1').tofile(WEB / 'raster' / 'parent10.u8')
    cls_stored.tofile(WEB / 'raster' / 'class_k2.u8')
    # 10 m band reflectance x10000 (uint16): [pre-mean(3 dates), post] x [B04,B03,B02,B08]
    np.clip(np.round(np.concatenate([pre_refl, post_refl]) * 10000), 0, 65535).astype('<u2').tofile(WEB / 'raster' / 'bands10.u16')
    with rasterio.open(need(OUT_DIR / 'parent_drop.tif')) as ds:
        pdrop = ds.read(1)[r0:r1, c0:c1]
    q16(pdrop).tofile(WEB / 'raster' / 'parent_drop10.i16')

    palette = {1: (184, 60, 30, 210), 2: (232, 160, 32, 210), 3: (91, 110, 125, 150), 255: (120, 120, 120, 140)}
    ov = np.zeros(cls_stored.shape + (4,), 'u1')
    for c, rgba in palette.items():
        ov[cls_stored == c] = rgba
    save_png(ov, 'class_k2.png')

    # ---- geo: corners in lon/lat for the crop, for the pixel inspector ----
    tr = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)
    l, b, r, t = sr_bounds
    corners = {n: tr.transform(x, y) for n, (x, y) in {'nw': (l, t), 'ne': (r, t), 'sw': (l, b), 'se': (r, b)}.items()}
    inv = Transformer.from_crs('EPSG:4326', crs, always_xy=True)
    pts_px = {}
    for name, (plat, plon) in {'crown': (11.465104, 76.134982), 'mundakkai': (11.4859333, 76.1559113), 'chooralmala_bridge': (11.4992, 76.1601)}.items():
        col, row = ~sr_transform * inv.transform(plon, plat)
        pts_px[name] = [round(col, 1), round(row, 1)]

    # ---- objects (recompute S-objects, same labelling + rho as step4, match the CSV) ----
    with np.errstate(invalid='ignore'):
        S = (d > K * sigma) & ~nodata
    labels, n_obj = label_components(S, 8)
    delta = np.where(S, np.nan_to_num(d), 0.0)
    rho = object_rho(delta, labels, SCALE)
    csv_rows = list(csv.DictReader(open(need(OUT_DIR / 'rho_objects.csv'), encoding='utf-8')))
    if len(csv_rows) != n_obj:
        raise SystemExit(f'object count {n_obj} != csv {len(csv_rows)}')
    rr, cc = np.nonzero(labels > 0)
    lab = labels[rr, cc]
    sizes = np.bincount(lab, minlength=n_obj + 1)[1:]
    cy = np.bincount(lab, weights=rr, minlength=n_obj + 1)[1:] / np.maximum(sizes, 1)
    cx = np.bincount(lab, weights=cc, minlength=n_obj + 1)[1:] / np.maximum(sizes, 1)
    in_parent = np.bincount(lab, weights=parent_hr[rr, cc].astype(float), minlength=n_obj + 1)[1:] / np.maximum(sizes, 1)
    ymin = np.full(n_obj + 1, 1 << 30); ymax = np.full(n_obj + 1, -1); xmin = ymin.copy(); xmax = ymax.copy()
    np.minimum.at(ymin, lab, rr); np.maximum.at(ymax, lab, rr); np.minimum.at(xmin, lab, cc); np.maximum.at(xmax, lab, cc)
    objects = []
    for i, row in enumerate(csv_rows):
        if int(row['px_2p5m']) != int(sizes[i]):
            raise SystemExit(f'object {i + 1} size mismatch vs csv')
        x_m, y_m = sr_transform * (cx[i] + 0.5, cy[i] + 0.5)
        lon, lat = tr.transform(x_m, y_m)
        objects.append({
            'id': int(row['object']), 'px': int(row['px_2p5m']), 'area_m2': float(row['area_m2']),
            'rho': None if row['rho'] == '' else float(row['rho']),
            'unsupported_fraction': float(row['unsupported_pixel_fraction']),
            'parent_fraction': round(float(in_parent[i]), 4),
            'cx': round(float(cx[i]), 1), 'cy': round(float(cy[i]), 1),
            'bbox': [int(xmin[i + 1]), int(ymin[i + 1]), int(xmax[i + 1]), int(ymax[i + 1])],
            'lon': round(lon, 6), 'lat': round(lat, 6)})
    if rho is not None:
        csv_rho = np.array([np.nan if r['rho'] == '' else float(r['rho']) for r in csv_rows])
        if not np.allclose(np.nan_to_num(csv_rho), np.nan_to_num(rho), atol=1e-5):
            raise SystemExit('recomputed rho does not match rho_objects.csv')
    (WEB / 'objects.json').write_text(json.dumps(objects, separators=(',', ':')), encoding='utf-8')

    # ---- acquisitions (real STAC audit) ----
    acq = [{'date': r['date'], 'platform': r['platform'], 'cloud_shadow_pct': round(r['cloud_shadow_pct_aoi'], 2),
            'coverage_pct': r['coverage_pct'], 'item_ids': r['item_ids']} for r in audit['rows']]

    # ---- experiments ----
    exps = []
    for eid in ['e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'e7', 'e8', 'x2', 'x3', 'x4', 'x5', 'x6', 'x9']:
        j = jload(RES / f'{eid}.json')
        exps.append({'id': eid.upper(), 'status': j.get('status'), 'evidence': j.get('evidence'),
                     'evidence_parts': j.get('evidence_parts'), 'reason': j.get('reason'),
                     'limitations': j.get('limitations'), 'started_utc': j.get('started_utc') or j.get('timestamp'),
                     'verdict': j.get('verdict'), 'criteria': j.get('criteria'), 'keep_rule': j.get('keep_rule'),
                     'summary': j.get('summary'), 'file': f'experiments/results/{eid}.json'})
    x2 = jload(RES / 'x2.json')
    x2_rows = []
    for name, v in x2['datasets'].items():
        dp = v['delta_vs_bicubic']['psnr']
        x2_rows.append({'dataset': name, 'n': v['n_images'], 'psnr_sr': v['mean']['psnr']['sr'], 'psnr_bicubic': v['mean']['psnr']['bicubic'],
                        'delta': dp['mean'], 'lo': dp['lo'], 'hi': dp['hi'], 'beats': x2['keep_rule']['per_dataset_beats_bicubic'][name]})
    x3 = jload(RES / 'x3.json')['gates_scored']
    x3_rows = {g: v['summary'] for g, v in x3.items() if 'summary' in v}
    x4 = jload(RES / 'x4.json')['results']
    x5 = jload(RES / 'x5.json')
    x9 = jload(RES / 'x9.json')
    x10 = jload(RES / 'x10.json')
    x11 = jload(RES / 'x11.json')
    x6 = jload(RES / 'x6.json')

    # ---- exports: real files, real sizes, real hashes ----
    export_files = [
        ('Maps & spatial rasters', 'class_map_2p5m.tif', OUT_DIR / 'class_map_2p5m.tif', '2.5 m class map (EPSG:32643): 0 NO_CHANGE, 1 OBSERVED, 2 INFERRED, 3 UNSUPPORTED, 255 NO_DATA'),
        ('Maps & spatial rasters', 'ndvi_drop_2p5m.tif', OUT_DIR / 'ndvi_drop_2p5m.tif', 'd = mean NDVI(pre) - mean NDVI(post), from 8 dihedral SR runs per date'),
        ('Maps & spatial rasters', 'ndvi_sigma_2p5m.tif', OUT_DIR / 'ndvi_sigma_2p5m.tif', 'sigma = sqrt(sigma_pre^2 + sigma_post^2)'),
        ('Maps & spatial rasters', 'parent_mask.tif', OUT_DIR / 'parent_mask.tif', '10 m parent change mask (threshold 0.3, provisional)'),
        ('Maps & spatial rasters', 'sr_post_2p5m.tif', OUT_DIR / 'sr_post_2p5m.tif', 'Model reconstruction, 2024-12-06 (B04,B03,B02,B08), identity run. Not ground truth'),
        ('Provenance', 'manifest.json', OUT_DIR / 'manifest.json', 'Dates, STAC item IDs, processing baselines, SCL policy, grid'),
        ('Provenance', 'step1.json', OUT_DIR / 'step1.json', 'Date audit, fetch, baseline and parent statistics'),
        ('Provenance', 'step3.json', OUT_DIR / 'step3.json', 'SR run, mask check, spectral consistency, runtime'),
        ('Provenance', 'step4.json', OUT_DIR / 'step4.json', 'Gate counts, areas, rho distributions'),
        ('Figures & ledgers', 'figure.png', OUT_DIR / 'figure.png', 'Evidence figure produced by the run'),
        ('Figures & ledgers', 'rho_objects.csv', OUT_DIR / 'rho_objects.csv', 'Per-object rho, area and UNSUPPORTED fraction (2,217 objects)'),
        ('Figures & ledgers', 'EVIDENCE.md', ROOT / 'experiments' / 'wayanad_evidence' / 'EVIDENCE.md', 'Numbers table with labels and limitations'),
        ('Figures & ledgers', 'RESULTS_EXCEPTIONAL.md', ROOT / 'RESULTS_EXCEPTIONAL.md', 'A10-audited claims table'),
    ]
    exports = [{'group': g, 'name': n, 'path': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size, 'sha256': sha256(p), 'desc': desc}
               for g, n, p, desc in export_files if need(p)]

    site = {
        'built_from': 'experiments/wayanad_evidence/outputs, experiments/results',
        'labels': manifest['labels'], 'config_sha256': manifest['config_sha256'],
        'event_date': manifest['event_date'], 'pre_dates': pre, 'post_date': post,
        'gap_days': step1['dates']['gap_days_event_to_post'], 'item_ids': manifest['item_ids'],
        'processing_baselines': manifest['processing_baselines'],
        'crs': str(crs), 'aoi_bounds_utm': step1['aoi']['bounds_epsg32643'], 'aoi_shape_px': step1['aoi']['shape_px'],
        'aoi_centre': {'lat': 11.490, 'lon': 76.160},
        'crop': {'rows': [r0, r1], 'cols': [c0, c1], 'shape_10m': [h10, w10], 'shape_2p5m': list(sr_shape),
                 'bounds_utm': [sr_bounds.left, sr_bounds.bottom, sr_bounds.right, sr_bounds.top], 'corners_lonlat': corners},
        'k': K, 'parent_threshold': 0.3, 'parent_threshold_status': 'provisional, set before viewing data',
        'pixel_area_m2': PIXEL_M2,
        'plausibility_points': {
            'crown': {'lat': 11.465104, 'lon': 76.134982, 'distance_m': step1['plausibility']['distance_m_to_published_points']['crown'], 'px': pts_px['crown']},
            'mundakkai': {'lat': 11.4859333, 'lon': 76.1559113, 'distance_m': step1['plausibility']['distance_m_to_published_points']['mundakkai'], 'px': pts_px['mundakkai']},
            'chooralmala_bridge': {'lat': 11.4992, 'lon': 76.1601, 'distance_m': step1['plausibility']['distance_m_to_published_points']['chooralmala_bridge'], 'px': pts_px['chooralmala_bridge']}},
        'step1': step1, 'step2': step2.get('measurements', step2), 'step3': {k: step3[k] for k in step3 if k != 'mask_check'},
        'step4': step4,
        'acquisitions': acq,
        'experiments': exps, 'x2': {'rows': x2_rows, 'pooled': x2['pooled'], 'verdict': x2['verdict'], 'summary': x2['summary']},
        'x3': x3_rows, 'x4': x4,
        'x5': {'status': x5['status'], 'selected': x5['selected_estimator'], 'limitations': x5['limitations']},
        'x6': {'status': x6['status'], 'reason': x6['exp_a']['reason'], 'audit_table': x6['exp_a']['audit_table'],
               'composite_gap_days': x6['exp_b']['gap_days_footprint']},
        'x9': x9['class_distribution'], 'x9_conformal_threshold': x9['a4_conformal_threshold'],
        'x10': {'summary': x10['summary'], 'verdicts': x10['verdicts'], 'cross': x10['cross_cutting_findings']},
        'x11': {'overall': x11['overall_verdict'], 'summary': x11['summary']},
        'exports': exports,
    }
    (WEB / 'site.json').write_text(json.dumps(site, separators=(',', ':'), default=float), encoding='utf-8')
    (WEB / 'SHA256SUMS').write_text(''.join(f"{e['sha256']}  {e['path']}\n" for e in exports), encoding='utf-8')
    print('objects', n_obj, 'exports', len(exports), 'site.json', (WEB / 'site.json').stat().st_size, 'bytes')


if __name__ == '__main__':
    main()
