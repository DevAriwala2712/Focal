"""F13 B2: how much landslide area lives in features narrower than 20 m? (labels only; no imagery, no network)

Pre-registered in configs/f13_discovery.yaml `experiments.B2` (sha256 84beef1c...). The Colombia inventory polygons
are rasterised DIRECTLY at 2.5, 10 and 20 m on nested grids (pixel-centre rule), the per-pixel local thickness of the
2.5 m raster decides the area share of parts narrower than 20 m, and the rasterisation IoU of the 10 m and 20 m
rasters against the 2.5 m one says how much a coarse pixel loses.
"""
from __future__ import annotations

import hashlib
import sqlite3
import struct

import numpy as np
from scipy import ndimage

from experiments import f13_common as C

RESOLUTIONS = (2.5, 10.0, 20.0)


# ---------------------------------------------------------------- GeoPackage (no fiona / geopandas in the project env) ----------------------------------------------------------------

def parse_gpkg_geometry(blob):
    """Standard GeoPackage binary geometry -> shapely geometry. Header: 'GP', version, flags, srs_id, envelope, then WKB."""
    from shapely import wkb
    blob = bytes(blob)
    if blob[:2] != b'GP':
        raise ValueError('not a GeoPackage geometry blob (missing GP magic)')
    flags = blob[3]
    if (flags >> 4) & 1:
        raise ValueError('empty geometry')
    env_ind = (flags >> 1) & 0b111
    env_len = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}.get(env_ind)
    if env_len is None:
        raise ValueError(f'invalid envelope indicator {env_ind}')
    return wkb.loads(blob[8 + env_len:])


def read_inventory(path, table: str) -> dict:
    con = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    try:
        srs = con.execute('SELECT srs_id FROM gpkg_contents WHERE table_name = ?', (table,)).fetchone()
        col = con.execute('SELECT column_name FROM gpkg_geometry_columns WHERE table_name = ?', (table,)).fetchone()
        if srs is None or col is None:
            raise ValueError(f'table {table!r} is not a feature table of {path}')
        rows = con.execute(f'SELECT "{col[0]}" FROM "{table}" ORDER BY rowid').fetchall()
    finally:
        con.close()
    geoms = [parse_gpkg_geometry(r[0]) for r in rows if r[0] is not None]
    return {'srs_id': int(srs[0]), 'geometries': geoms, 'n_rows': len(rows)}


def verify_inventory(path, expected_sha256: str, table: str, expected_features: int) -> dict:
    from pathlib import Path
    p = Path(path)
    checks = {'exists': p.is_file()}
    if not checks['exists']:
        return {'ok': False, 'checks': checks}
    checks['sha256'] = hashlib.sha256(p.read_bytes()).hexdigest() == expected_sha256
    try:
        inv = read_inventory(p, table)
        checks['table'] = True
        checks['feature_count'] = inv['n_rows'] == expected_features
    except Exception:
        checks['table'], checks['feature_count'] = False, False
    return {'ok': all(checks.values()), 'checks': checks}


# ---------------------------------------------------------------- nested grids and rasterisation ----------------------------------------------------------------

def nested_grid(bounds_m):
    """Grids at 2.5, 10 and 20 m sharing an origin snapped to the 20 m lattice, so every 10 m px is exactly 4 x 4 px at
    2.5 m and every 20 m px is 8 x 8 px at 2.5 m (2 x 2 at 10 m). bounds = (minx, miny, maxx, maxy) in metres."""
    from affine import Affine
    minx, miny, maxx, maxy = bounds_m
    x0, y0 = float(np.floor(minx / 20.0) * 20.0), float(np.ceil(maxy / 20.0) * 20.0)
    n20_w, n20_h = int(np.ceil((maxx - x0) / 20.0)), int(np.ceil((y0 - miny) / 20.0))
    shape, transform = {}, {}
    for res in RESOLUTIONS:
        f = int(round(20.0 / res))
        shape[res] = (n20_h * f, n20_w * f)
        transform[res] = Affine(res, 0.0, x0, 0.0, -res, y0)
    return {'x0': x0, 'y0': y0, 'shape': shape, 'transform': transform}


def rasterise(geometries, grid) -> dict:
    """Burn the polygons into each resolution separately (rasterio.features.rasterize, all_touched=False = the
    pixel-centre rule). This is a direct rasterisation at every resolution, not a down-sampling of the finest one."""
    from rasterio import features
    out = {}
    for res in RESOLUTIONS:
        out[res] = features.rasterize([(g, 1) for g in geometries], out_shape=grid['shape'][res],
                                      transform=grid['transform'][res], fill=0, all_touched=False, dtype='uint8').astype(bool)
    return out


def raster_iou(coarse, fine, factor: int) -> dict:
    up = np.repeat(np.repeat(np.asarray(coarse, bool), factor, axis=0), factor, axis=1)
    fine = np.asarray(fine, bool)
    inter, union = int((up & fine).sum()), int((up | fine).sum())
    return {'iou': (inter / union) if union else float('nan'), 'intersection': inter, 'union': union}


# ---------------------------------------------------------------- local thickness ----------------------------------------------------------------

def local_thickness_px(mask) -> np.ndarray:
    """Per-pixel local thickness in pixels: 2 r - 1 of the largest inscribed disk (centre p in the mask, radius
    r(p) = Euclidean distance to the nearest background pixel) that covers the pixel (|q - p| <= r(p)). Background = 0.

    The '- 1' is trustsr.alloc.component_width_m's digital correction: a strip w px wide (odd w) has r = (w + 1) / 2 at
    its centre line, so 2 r - 1 = w exactly. Exact algorithm: process the distinct radii in descending order; every
    pixel is assigned by the first (largest) disk that covers it.
    """
    m = np.asarray(mask, bool)
    out = np.zeros(m.shape, dtype=np.float64)
    if not m.any():
        return out
    pad = np.pad(m, 1, constant_values=False)
    r = ndimage.distance_transform_edt(pad)[1:-1, 1:-1]
    remaining = m.copy()
    for rho in np.unique(r[m])[::-1]:
        centres = m & (r == rho)
        cover = ndimage.distance_transform_edt(~centres) <= rho + 1e-9
        hit = cover & remaining
        out[hit] = 2.0 * rho - 1.0
        remaining &= ~hit
        if not remaining.any():
            break
    return out


def thin_share(thickness_px, mask, px_m: float, threshold_m: float) -> dict:
    t = np.asarray(thickness_px, float)[np.asarray(mask, bool)]
    thin = int((t * px_m < threshold_m).sum())
    return {'thin_px': thin, 'area_px': int(t.size), 'share': (thin / t.size) if t.size else float('nan')}


def b2_verdict(share_thin: float, iou_10m_vs_2p5m: float) -> str:
    """yaml experiments.B2.keep_rule on point estimates."""
    if share_thin >= 0.30 and iou_10m_vs_2p5m < 0.80:
        return 'TRUE'
    if share_thin < 0.10 and iou_10m_vs_2p5m >= 0.85:
        return 'FALSE'
    return 'INCONCLUSIVE'


# ================================================================ the run ================================================================

EXPECTED_SHA256 = 'c55da5d5d5c72dd6eed2ad1173ce261a3b1eb4080b17b04ba7a49e1a2e04d29f'
TABLE = 'colombia_landslides'
N_FEATURES = 838
GPKG_RELATIVE = 'data/risk-cache/labels/colombia.gpkg'       # risk/r4_labels.py: cache / 'labels' / 'colombia.gpkg', cache = data/risk-cache
THIN_M = 20.0
TILE = 128
WIDTH_EDGES_M = (20.0, 50.0, 150.0)


def search_for_gpkg() -> dict:
    """Every place a copy could plausibly be: the repo tree (incl. worktrees), its parent and the home directory
    (excluding Library). Recorded so a BLOCKED result names exactly what was searched."""
    from pathlib import Path
    roots = [C.ROOT, C.ROOT.parent, Path.home()]
    found, searched = [], []
    for root in roots:
        searched.append(str(root))
        for p in root.rglob('*.gpkg') if root == C.ROOT else root.glob('*/*/*.gpkg'):
            if 'Library' in p.parts:
                continue
            found.append(str(p))
    return {'searched_roots': searched, 'expected_path': str(C.ROOT / GPKG_RELATIVE), 'gpkg_files_found': sorted(set(found))}


def thickness_maps(mask25):
    """Local thickness (px) of the 2.5 m raster, computed per connected-component window (the raster is mostly empty).
    Also the F5-style per-component width (trustsr.alloc.component_width_m) for the secondary share."""
    from trustsr import alloc
    labels, n = ndimage.label(mask25, structure=ndimage.generate_binary_structure(2, 2))
    thick = np.zeros(mask25.shape, dtype=np.float32)
    width_cc = np.zeros(mask25.shape, dtype=np.float32)
    for sl in ndimage.find_objects(labels):
        win = tuple(slice(max(s.start - 2, 0), s.stop + 2) for s in sl)
        sub = mask25[win]
        thick[win] = np.maximum(thick[win], local_thickness_px(sub).astype(np.float32))
        w, _l, _p = alloc.component_width_m(sub, 2.5)
        width_cc[win] = np.maximum(width_cc[win], (w / 2.5).astype(np.float32))
    return thick, width_cc, int(n)


def analyse(geoms, boot) -> dict:
    """Everything B2 measures from inventory polygons already in EPSG:32618 (metres)."""
    from trustsr.bootstrap import block_sums, ratio_bootstrap_ci
    minx, miny, maxx, maxy = [f(g.bounds[i] for g in geoms) for i, f in enumerate((min, min, max, max))]
    grid = nested_grid((minx - 40, miny - 40, maxx + 40, maxy + 40))
    r = rasterise(geoms, grid)
    thick, width_cc, n_cc = thickness_maps(r[2.5])
    area = int(r[2.5].sum())
    shares = {'local_thickness': thin_share(thick, r[2.5], 2.5, THIN_M),
              'component_width_secondary': thin_share(width_cc, r[2.5], 2.5, THIN_M)}
    t_m = thick[r[2.5]] * 2.5
    bins = {'<20': float((t_m < 20).mean()), '20-50': float(((t_m >= 20) & (t_m < 50)).mean()),
            '50-150': float(((t_m >= 50) & (t_m < 150)).mean()), '>=150': float((t_m >= 150).mean())}
    thin_map = r[2.5] & (thick * 2.5 < THIN_M)
    area_t, thin_t = block_sums(r[2.5], TILE), block_sums(thin_map, TILE)
    share_ci = ratio_bootstrap_ci(thin_t, area_t, boot['replicates'], 0.95, boot['seed'])
    ious = {}
    for res, f in ((10.0, 4), (20.0, 8)):
        up = np.repeat(np.repeat(r[res], f, axis=0), f, axis=1)
        inter_t, union_t = block_sums(up & r[2.5], TILE), block_sums(up | r[2.5], TILE)
        ci = ratio_bootstrap_ci(inter_t, union_t, boot['replicates'], 0.95, boot['seed']) if union_t.sum() > 0 else None
        ious[str(res)] = {**raster_iou(r[res], r[2.5], f), 'ci': ci}
    return {'grid_shapes': {str(k): list(v) for k, v in grid['shape'].items()}, 'polygon_area_px_2p5m': area,
            'connected_components_2p5m': n_cc, 'shares': shares, 'thickness_bins_share': bins,
            'share_ci': share_ci, 'rasterisation_iou': ious}


def main():
    import json
    import time
    from shapely.ops import transform as shp_transform
    from pyproj import Transformer
    started = C.utc_now()
    t0 = time.time()
    result = {'evidence': 'real', 'experiment_id': 'B2'}
    try:
        pre = C.require_committed_prereg()
        cfg = C.load_prereg()
        b2 = cfg['experiments']['B2']
        boot = cfg['statistics']['bootstrap']
        path = C.ROOT / GPKG_RELATIVE
        v = verify_inventory(path, EXPECTED_SHA256, TABLE, N_FEATURES)
        result['inventory_check'] = {'path': GPKG_RELATIVE, **v, 'expected_sha256': EXPECTED_SHA256,
                                     'expected_table': TABLE, 'expected_features': N_FEATURES}
        if not v['ok']:
            s = search_for_gpkg()
            result['search'] = s
            raise C.Blocked(f"Colombia inventory not usable at {GPKG_RELATIVE} (checks: {v['checks']}); never downloaded. "
                            f"Searched {s['searched_roots']} (repo tree recursively, parent and home to depth 3, Library excluded): "
                            f"{len(s['gpkg_files_found'])} .gpkg files found {s['gpkg_files_found']}. To run B2 place the file "
                            f"whose sha256 is {EXPECTED_SHA256[:8]}...{EXPECTED_SHA256[-7:]} at {GPKG_RELATIVE}.")
        inv = read_inventory(path, TABLE)
        to_utm = Transformer.from_crs(f"EPSG:{inv['srs_id']}", 'EPSG:32618', always_xy=True).transform
        geoms = [shp_transform(to_utm, g) for g in inv['geometries']]
        A = analyse(geoms, boot)
        share_ci, ious, shares = A['share_ci'], A['rasterisation_iou'], A['shares']
        verdict = b2_verdict(share_ci['estimate'], ious['10.0']['iou'])
        meas = [C.measurement('area_share_thinner_than_20m', share_ci['estimate'], numerator=share_ci['numerator'],
                              denominator=share_ci['denominator'], lo=share_ci['lo'], hi=share_ci['hi'],
                              replicates=boot['replicates'], seed=boot['seed'], unit='2.5 m px with local thickness < 20 m / polygon px'),
                *[C.measurement(f'IoU_{res}m_vs_2p5m', d['iou'], numerator=d['intersection'], denominator=d['union'],
                                lo=d['ci']['lo'], hi=d['ci']['hi'], replicates=boot['replicates'], seed=boot['seed'],
                                unit='IoU of the nearest-upsampled raster vs the 2.5 m raster') for res, d in ious.items()]]
        result.update(
            status='PASS' if verdict == 'TRUE' else 'FAIL', verdict=verdict, rule_applied=b2['keep_rule'],
            preregistration={**pre, 'experiment': 'B2'},
            o5_note=b2['keep_rule']['o5_note'], o5_note_outcome={'iou_10m_vs_2p5m': ious['10.0']['iou'],
                                                              'ge_0p85': bool(ious['10.0']['iou'] >= 0.85)},
            data={'features': len(geoms), 'srs_id_in': inv['srs_id'], 'crs_out': 'EPSG:32618', 'connected_components_2p5m': A['connected_components_2p5m'],
                  'grid_shapes': A['grid_shapes'], 'polygon_area_px_2p5m': A['polygon_area_px_2p5m']},
            shares=shares, thickness_bins_share=A['thickness_bins_share'], rasterisation_iou=ious, measurements=meas,
            interpretations=[
                'Local thickness = diameter of the largest inscribed disk covering the pixel, (2 r - 1) px with r the Euclidean distance to '
                'the background (alloc.component_width_m digital convention); "< 20 m" is strict.',
                'IoU(10 m, 2.5 m) and IoU(20 m, 2.5 m) use nearest-neighbour upsampling of the directly rasterised coarse raster.',
                'The raster is windowed per 8-connected component for the thickness; the grid has a 40 m margin so no polygon touches the edge.',
                'Bootstrap tiles are 128 px of the 2.5 m grid (320 m) over the inventory extent; tiles with no polygon area drop out.'],
            limitations=['Labels only: inventory polygons are not Sentinel-2 detections, and they are one event in one region (Colombia, 2019-12-24).',
                         'Inventory polygon geometry may itself be generalised; thin parts could be under- or over-drawn.'],
            runtime_seconds=round(time.time() - t0, 1))
        return _finish(result, started)
    except C.Blocked as exc:
        result.update(status='BLOCKED', verdict='NOT_RUN', reason=str(exc))
        return _finish(result, started)


def _finish(result, started):
    import json
    result.setdefault('interpretations', [])
    result.setdefault('reproduction_check', None)
    result.setdefault('measurements', [])
    result.setdefault('limitations', [])
    result.setdefault('fetch_log', {'network': 'none (B2 is labels only; the GeoPackage is never downloaded)', 'bytes_received': 0, 'items': []})
    out = C.write_result('b2', result, started)
    print(json.dumps({k: out.get(k) for k in ('status', 'verdict', 'reason')}, indent=2))
    return C.exit_code(out['status'])


if __name__ == '__main__':
    raise SystemExit(main())
