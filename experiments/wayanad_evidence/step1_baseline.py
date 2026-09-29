"""Step 1: audit the (re-centred) AOI, fetch RGBN+SCL for the chosen dates, 10 m NDVI change baseline, scar crop."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np

from experiments.common import Blocked, finalize_result
from experiments.e6_season_matched import ByteBudget
from experiments.wayanad_evidence import config as C
from experiments.wayanad_evidence import fetch as F
from experiments.wayanad_evidence.cogio import write_cog
from experiments.wayanad_evidence.data import load_date
from experiments.wayanad_evidence.gate import parent_mask
from experiments.wayanad_evidence.geo import (bounds, dilate_square, label_components, reference_grid, snap_crop)
from risk.common import write_json


def published_point_distances(cfg, transform, labels, label_id):
    """Distance (m) from each published place coordinate to the nearest pixel centre of the component. Plausibility only."""
    from pyproj import Transformer
    to_utm = Transformer.from_crs('EPSG:4326', cfg['aoi']['crs'], always_xy=True)
    rr, cc = np.nonzero(labels == label_id)
    xs, ys = transform.c + (cc + 0.5) * transform.a, transform.f + (rr + 0.5) * transform.e
    out = {}
    for name, p in cfg['aoi']['plausibility_points'].items():
        x, y = to_utm.transform(p['lon'], p['lat'])
        out[name] = float(np.hypot(xs - x, ys - y).min())
    return out


def run(cfg, root, config_hash):
    started = datetime.now(timezone.utc).isoformat()
    out, cache = C.outputs(cfg, root), C.cache(cfg, root)
    transform, shape = reference_grid(cfg['aoi'])
    res20 = cfg['aoi']['audit_resolution_m']
    from affine import Affine
    transform20 = Affine(res20, 0.0, transform.c, 0.0, -res20, transform.f)
    shape20 = (shape[0] * 10 // res20, shape[1] * 10 // res20)
    budget = ByteBudget(cfg['fetch']['max_fetch_bytes'])

    items, groups = F.search_groups(cfg, transform, shape)
    rows, item_rows, errors = F.audit(cfg, root, groups, transform20, shape20, budget, cache)
    write_json(out / 'audit_acquisitions.json', {'aoi_bounds': bounds(transform, shape), 'crs': cfg['aoi']['crs'],
                                                 'scene_count': len(items), 'rows': rows, 'items': item_rows, 'errors': errors})
    if errors:
        raise Blocked(f'{len(errors)} SCL asset(s) unreadable, first: {errors[0]}; dates are not established', evidence='real')
    im = cfg['phase0']['imagery']
    chosen = F.choose_dates(rows, cfg['dates'], cfg['event_date'], im['max_cloud_shadow_pct'], im['min_coverage_pct'])
    dates = chosen['pre'] + [chosen['post']]
    for date in dates:
        items_d = [i for (d, _, _), g in groups.items() if d == date for i in g]
        F.read_bands(cfg, root, date, items_d, transform, shape, budget, cache)
        print(f'cached {date}: {budget.used()} bytes received', flush=True)

    arrays = {d: load_date(cfg, cache, d) for d in dates}
    pre, post = chosen['pre'], chosen['post']
    valid_all = np.logical_and.reduce([arrays[d]['valid'] for d in dates])
    ndvi_pre_mean = np.mean([arrays[d]['ndvi'] for d in pre], axis=0)
    drop = np.where(valid_all, ndvi_pre_mean - arrays[post]['ndvi'], np.nan).astype('float32')
    parent = parent_mask(drop, valid_all, cfg['change']['parent_drop_threshold'])
    labels, n_comp = label_components(parent, cfg['change']['parent_connectivity'])
    if n_comp == 0:
        raise Blocked('no parent-positive pixel at the configured threshold; nothing to crop', evidence='real')
    sizes = np.bincount(labels.ravel())[1:]
    big = int(np.argmax(sizes)) + 1
    component = labels == big
    buffer_px = int(round(cfg['footprint']['buffer_m'] / 10))
    footprint = dilate_square(component, buffer_px)
    rr, cc = np.nonzero(footprint)
    crop = snap_crop((rr.min(), rr.max() + 1, cc.min(), cc.max() + 1), shape, cfg['tiling']['tile'])
    distances = published_point_distances(cfg, transform, labels, big)
    near = {k: v <= cfg['aoi']['plausibility_radius_m'] for k, v in distances.items()}

    # ---- rasters (COGs, input CRS, reference grid) ----
    crs = cfg['aoi']['crs']
    for d in dates:
        with np.load(cache / f'{d}.npz') as z:
            stack = np.concatenate([z['dn'], z['scl'][None].astype('uint16')], axis=0)
        write_cog(out / f'stack_{d}.tif', stack, transform, crs, nodata=0,
                  descriptions=[*F.MODEL_BANDS, 'SCL'], tags={
                      'date': d, 'items': ','.join(arrays[d]['meta']['items']), 'band_order': 'B04,B03,B02,B08,SCL',
                      'reflectance': '(DN-1000)/10000 for baseline>=04.00', 'scl_policy': 'validity mask only, nearest neighbour'})
    mask_cog = np.where(valid_all, parent.astype('uint8'), 255).astype('uint8')
    write_cog(out / 'parent_mask.tif', mask_cog, transform, crs, nodata=255, categorical=True, descriptions=['parent_mask'],
              tags={'values': '0 no change, 1 parent change, 255 NO_DATA',
                    'threshold': f"{cfg['change']['parent_drop_threshold']} ({cfg['change']['parent_threshold_status']})",
                    'pre_dates': ','.join(pre), 'post_date': post})
    write_cog(out / 'parent_drop.tif', drop, transform, crs, nodata=float('nan'), descriptions=['mean NDVI_pre - NDVI_post'],
              tags={'pre_dates': ','.join(pre), 'post_date': post})
    write_cog(out / 'footprint.tif', footprint.astype('uint8'), transform, crs, nodata=None, categorical=True,
              descriptions=['footprint'], tags={'buffer_m': cfg['footprint']['buffer_m'],
                                                'use': 'date selection and cropping only; never a label'})

    np.savez_compressed(cache / 'step1_state.npz', dates=np.array(dates), pre=np.array(pre), post=np.array(post),
                        valid_all=valid_all, drop=drop, parent=parent, component=component, footprint=footprint,
                        crop=np.array(crop))
    px_area = 10.0 * 10.0
    per_date = {d: {'scl_invalid_px': int((~arrays[d]['valid_scl']).sum()),
                    'ndvi_undefined_px': int((arrays[d]['valid_scl'] & ~np.isfinite(arrays[d]['ndvi'])).sum()),
                    'valid_px': int(arrays[d]['valid'].sum())} for d in dates}
    scar_plausible = bool(all(near.values()))
    measurements = {
        'dates': {'pre': pre, 'post': post, 'gap_days_event_to_post': (datetime.fromisoformat(post) - datetime.fromisoformat(cfg['event_date'])).days,
                  'source': chosen['source'], 'replaced_vs_preferred': chosen['replaced']},
        'aoi': {'bounds_epsg32643': bounds(transform, shape), 'shape_px': list(shape)},
        'no_data': {'per_date': per_date, 'union_invalid_px': int((~valid_all).sum()), 'union_invalid_fraction': float((~valid_all).mean())},
        'parent': {'threshold': cfg['change']['parent_drop_threshold'], 'threshold_status': cfg['change']['parent_threshold_status'],
                   'positive_px_aoi': int(parent.sum()), 'area_m2_aoi': float(parent.sum() * px_area),
                   'component_count': int(n_comp), 'largest_component_px': int(sizes[big - 1]),
                   'largest_component_area_m2': float(sizes[big - 1] * px_area),
                   'second_largest_px': int(np.sort(sizes)[-2]) if n_comp > 1 else 0},
        'footprint': {'buffer_m': cfg['footprint']['buffer_m'], 'px': int(footprint.sum()), 'area_m2': float(footprint.sum() * px_area)},
        'crop': {'rows': [crop[0], crop[1]], 'cols': [crop[2], crop[3]], 'shape_px': [crop[1] - crop[0], crop[3] - crop[2]],
                 'tiles_128': [(crop[1] - crop[0]) // 128, (crop[3] - crop[2]) // 128]},
        'plausibility': {'distance_m_to_published_points': distances, 'within_radius': near,
                         'radius_m': cfg['aoi']['plausibility_radius_m'],
                         'note': 'published coordinates used only to check the found component sits at the slide; never a label'},
        'bytes_received': budget.log(),
    }
    write_json(out / 'manifest.json', {
        'experiment': 'wayanad_evidence', 'labels': cfg['labels'], 'config_sha256': config_hash,
        'event_date': cfg['event_date'], 'pre_dates': pre, 'post_date': post,
        'item_ids': {d: arrays[d]['meta']['items'] for d in dates},
        'processing_baselines': {d: arrays[d]['meta']['processing_baselines'] for d in dates},
        'scl_policy': {'classes_invalid': {k: im[k] for k in ('cloud_classes', 'shadow_classes', 'invalid_classes')},
                       'role': 'validity mask only, never a model input', 'resampling': 'nearest neighbour (20 m -> 10 m replication)',
                       'ndvi_floor': cfg['radiometry']['min_denominator'], 'mosaic': 'first valid tile wins (sorted item id)'},
        'radiometry': cfg['radiometry'], 'crs': crs, 'grid_transform': list(transform)[:6], 'grid_shape': list(shape),
        'rasters': sorted(p.name for p in out.glob('*.tif'))})
    result = {'status': 'PASS' if scar_plausible else 'FAIL', 'evidence': 'real', 'measurements': measurements,
              'limitations': ['parent threshold is provisional (set before viewing data), not calibrated',
                              'NDVI drop indicates vegetation disturbance, not uniquely landslide damage',
                              f"pre dates are January-May dry season and the post date is {post}: seasonal/recovery confounding",
                              'retrospective comparison, not a rapid-response result']}
    if not scar_plausible:
        result['reason'] = ('largest parent-positive component is not within the plausibility radius of every published slide '
                            f'coordinate: {distances}')
    return finalize_result('step1', result, config_hash, started)


def main():
    cfg, root, config_hash = C.load()
    try:
        result = run(cfg, root, config_hash)
    except Blocked as exc:
        result = finalize_result('step1', {'status': 'BLOCKED', 'evidence': 'real', 'reason': str(exc)}, config_hash,
                                 datetime.now(timezone.utc).isoformat())
    write_json(C.outputs(cfg, root) / 'step1.json', result)
    print(json.dumps(result, indent=2, default=str))


if __name__ == '__main__':
    main()
