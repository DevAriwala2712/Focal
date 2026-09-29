"""Step 4: trust gate on the 2.5 m crop: d, sigma, S, P, five classes (COG), rho per S-object, areas and baselines."""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone

import numpy as np
from affine import Affine

from experiments.common import Blocked, finalize_result
from experiments.wayanad_evidence import config as C
from experiments.wayanad_evidence.cogio import write_cog
from experiments.wayanad_evidence.gate import (CLASS_NAMES, INFERRED, NO_CHANGE, NO_DATA, OBSERVED, UNSUPPORTED, classify,
                                               object_rho, upsample)
from experiments.wayanad_evidence.geo import label_components, reference_grid, sr_grid
from risk.common import write_json


def distribution(values):
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {'n': 0}
    q = np.percentile(v, [0, 10, 25, 50, 75, 90, 100])
    return {'n': int(v.size), **{k: float(x) for k, x in zip(['min', 'p10', 'p25', 'median', 'p75', 'p90', 'max'], q)},
            'mean': float(v.mean())}


def run(cfg, root, config_hash):
    started = datetime.now(timezone.utc).isoformat()
    out, cache = C.outputs(cfg, root), C.cache(cfg, root)
    for s in ('step1', 'step3'):
        st = json.loads((out / f'{s}.json').read_text(encoding='utf-8'))
        if st['status'] != 'PASS':
            raise Blocked(f"{s} status is {st['status']}; the gate needs its outputs. {st.get('reason', '')}", evidence='real')
    with np.load(cache / 'step1_state.npz') as z:
        s1 = {k: z[k] for k in z.files}
    with np.load(cache / 'step3_state.npz') as z:
        s3 = {k: z[k] for k in z.files}
    step3 = json.loads((out / 'step3.json').read_text(encoding='utf-8'))['measurements']
    scale, k = cfg['tiling']['scale'], cfg['change']['k']
    r0, r1, c0, c1 = (int(v) for v in s1['crop'])
    pre, post = [str(d) for d in s1['pre']], str(s1['post'])
    valid_hr = upsample(s1['valid_all'][r0:r1, c0:c1], scale)
    full = (s3['pre_count'] == 8 * len(pre)) & (s3['post_count'] == 8)
    nodata = ~valid_hr | ~full
    d = s3['pre_mean'] - s3['post_mean']
    sigma = np.sqrt(s3['pre_std'] ** 2 + s3['post_std'] ** 2)
    parent10 = s1['parent'][r0:r1, c0:c1]
    parent_hr = upsample(parent10, scale)
    cls = classify(d, sigma, parent_hr, nodata, k)
    with np.errstate(invalid='ignore'):
        S = (d > k * sigma) & ~nodata
    px = (10.0 / scale) ** 2
    counts = {CLASS_NAMES[c]: int((cls == c).sum()) for c in (NO_CHANGE, OBSERVED, INFERRED, UNSUPPORTED, NO_DATA)}
    area = lambda n: float(n * px)

    # ---- rho per connected S-object (ADR-003) ----
    labels, n_obj = label_components(S, cfg['change']['rho_connectivity'])
    delta = np.where(S, d, 0.0)
    rho = object_rho(delta, labels, scale)
    sizes = np.bincount(labels.ravel(), minlength=n_obj + 1)[1:]
    unsup_frac = np.bincount(labels[S].ravel(), weights=(cls[S] == UNSUPPORTED).astype(float), minlength=n_obj + 1)[1:] / np.maximum(sizes, 1)
    big = sizes >= cfg['change']['rho_min_object_px']
    energy = np.bincount(labels[S].ravel(), weights=d[S] ** 2, minlength=n_obj + 1)[1:]
    with open(out / 'rho_objects.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['object', 'px_2p5m', 'area_m2', 'rho', 'unsupported_pixel_fraction'])
        for i in range(n_obj):
            w.writerow([i + 1, int(sizes[i]), area(sizes[i]), '' if not np.isfinite(rho[i]) else f'{rho[i]:.6f}', f'{unsup_frac[i]:.4f}'])
    energy_weighted = float(np.nansum(rho * energy) / energy.sum()) if energy.sum() > 0 else None
    rho_report = {
        'definition': 'rho = ||P D||^2 / ||D||^2, D = d restricted to the S-object (zero elsewhere), P = 4x4 block mean; '
                      'a fraction of signal energy in [0,1], NOT a probability; area averaging only approximates the S2 PSF',
        'objects_total': int(n_obj), 'objects_ge_min_px': int(big.sum()), 'min_px_for_main_distribution': cfg['change']['rho_min_object_px'],
        'all_objects': distribution(rho), 'objects_ge_min_px_distribution': distribution(rho[big]),
        'energy_weighted_mean_rho': energy_weighted,
        'pixel_weighted_mean_rho': float(np.nansum(rho * sizes) / sizes[np.isfinite(rho)].sum()) if n_obj else None,
        'majority_unsupported_objects': distribution(rho[unsup_frac > 0.5]),
        'majority_observed_or_inferred_objects': distribution(rho[unsup_frac <= 0.5]),
        'csv': 'rho_objects.csv'}

    # ---- rasters ----
    transform, _ = reference_grid(cfg['aoi'])
    t_sr, shape_sr = sr_grid(transform * Affine.translation(c0, r0), (r1 - r0, c1 - c0), scale)
    crs = cfg['aoi']['crs']
    tags = {'classes': '0 NO_CHANGE, 1 OBSERVED, 2 INFERRED, 3 UNSUPPORTED (excluded from change), 255 NO_DATA',
            'pre_dates': ','.join(pre), 'post_date': post, 'model': cfg['labels']['model'], 'k': cfg['labels']['k'],
            'comparison': cfg['labels']['comparison'],
            'parent_threshold': f"{cfg['change']['parent_drop_threshold']} ({cfg['change']['parent_threshold_status']})"}
    write_cog(out / 'class_map_2p5m.tif', cls, t_sr, crs, nodata=255, categorical=True, descriptions=['class'], tags=tags)
    write_cog(out / 'ndvi_drop_2p5m.tif', np.where(nodata, np.nan, d).astype('float32'), t_sr, crs, nodata=float('nan'),
              descriptions=['d = mean NDVI_pre - mean NDVI_post (SR)'], tags=tags)
    write_cog(out / 'ndvi_sigma_2p5m.tif', np.where(nodata, np.nan, sigma).astype('float32'), t_sr, crs, nodata=float('nan'),
              descriptions=['sigma = sqrt(sigma_pre^2 + sigma_post^2)'], tags=tags)

    valid_n = int((~nodata).sum())
    measurements = {
        'labels': cfg['labels'], 'k': k, 'pre_dates': pre, 'post_date': post,
        'crop_2p5m_px': list(shape_sr), 'pixel_area_m2': px,
        'class_counts_px': counts, 'class_areas_m2': {n: area(c) for n, c in counts.items()},
        'valid_px': valid_n,
        'no_data': {'total_px': int(nodata.sum()), 'scl_or_ndvi_invalid_at_10m_px': int((~valid_hr).sum()),
                    'sr_ndvi_undefined_in_some_run_px': int((valid_hr & ~full).sum())},
        'S_ungated_positive_px': int(S.sum()), 'S_ungated_area_m2': area(S.sum()),
        'unsupported_suppressed_px': counts['UNSUPPORTED'],
        'unsupported_fraction_of_S': float(counts['UNSUPPORTED'] / S.sum()) if S.sum() else None,
        'baselines': {'parent_area_10m_crop_m2': float(parent10.sum() * 100.0), 'parent_px_10m_crop': int(parent10.sum()),
                      'parent_area_10m_aoi_m2': float(s1['parent'].sum() * 100.0),
                      'parent_area_largest_component_m2': float(s1['component'].sum() * 100.0)},
        'observed_area_m2': area(counts['OBSERVED']),
        'observed_union_inferred_area_m2': area(counts['OBSERVED'] + counts['INFERRED']),
        'nrsc_scale_check': {'nrsc_main_scarp_m2': cfg['scale_check']['nrsc_main_scarp_m2'],
                             'note': 'scale check only. NRSC maps the main scarp with different data and a different '
                                     'definition; TrustSR areas are NDVI-drop pixels (scarp plus runout vegetation loss) '
                                     'and NRSC is NOT a label'},
        'rho': rho_report,
        'spectral_consistency': step3['sr']['spectral_consistency'],
        'spectral_tolerance_mae_reported_against': cfg['spectral_consistency']['tolerance_mae'],
    }
    limitations = [cfg['labels']['model'], cfg['labels']['k'], cfg['labels']['comparison'],
                   'parent threshold provisional (set before viewing data), not calibrated',
                   'no accuracy/F1: no labels were used', 'no 2.5 m ground truth exists',
                   'sigma is dihedral + cross-date dispersion, a sensitivity proxy, not a calibrated probability']
    return finalize_result('step4', {'status': 'PASS', 'evidence': 'real', 'measurements': measurements, 'limitations': limitations},
                           config_hash, started)


def main():
    cfg, root, config_hash = C.load()
    try:
        result = run(cfg, root, config_hash)
    except Blocked as exc:
        result = finalize_result('step4', {'status': 'BLOCKED', 'evidence': 'real', 'reason': str(exc)}, config_hash,
                                 datetime.now(timezone.utc).isoformat())
    write_json(C.outputs(cfg, root) / 'step4.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
