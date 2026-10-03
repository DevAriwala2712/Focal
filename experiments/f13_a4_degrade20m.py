"""F13 A4: does Wayanad need 10 m at all? The 10 m parent rule degraded to 20 m.

Pre-registered in configs/f13_discovery.yaml `experiments.A4`. Reflectance (never NDVI) is area-averaged 2 x 2 to 20 m,
the NDVI drop is recomputed, and the same 0.30 threshold is applied. A 20 m cell is valid only if all four 10 m children
are valid on every date. Informs framing only; it does not feed the gate.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from experiments import f13_common as C


# ---------------------------------------------------------------- helpers ----------------------------------------------------------------

def aggregate_reflectance_2x2(refl, valid):
    """(B, H, W) reflectance and (H, W) validity -> 2 x 2 mean reflectance (B, H/2, W/2) and 20 m validity.

    The mean is over reflectance, so NDVI computed afterwards is the NDVI of the area-averaged signal (what a 20 m
    sensor would see), not the average of four NDVIs. A cell is valid only if every one of its four children is."""
    refl = np.asarray(refl, dtype=np.float64)
    valid = np.asarray(valid, bool)
    b, h, w = refl.shape
    if h % 2 or w % 2 or valid.shape != (h, w):
        raise ValueError(f'2 x 2 aggregation needs even dimensions and a matching validity mask; got {refl.shape}, {valid.shape}')
    r20 = refl.reshape(b, h // 2, 2, w // 2, 2).mean(axis=(2, 4))
    v20 = valid.reshape(h // 2, 2, w // 2, 2).all(axis=(1, 3))
    return r20, v20


def ndvi_from_reflectance(refl, min_denominator: float) -> np.ndarray:
    """NDVI of a (4, H, W) [B04, B03, B02, B08] array with the evidence run's own function and floor."""
    from experiments.wayanad_evidence.stats import ndvi
    return ndvi(refl[0], refl[3], min_denominator)


def upsample_nn(a, factor: int = 2) -> np.ndarray:
    return np.repeat(np.repeat(np.asarray(a), factor, axis=0), factor, axis=1)


def scar_component(mask, crown_rc, radius_px: float, connectivity: int = 8):
    """The largest connected component of `mask` that has at least one pixel within `radius_px` of the crown pixel
    (yaml A4.method: largest parent-positive component within plausibility_radius_m of the crown)."""
    mask = np.asarray(mask, bool)
    structure = ndimage.generate_binary_structure(2, 2 if connectivity == 8 else 1)
    labels, n = ndimage.label(mask, structure=structure)
    rr, cc = np.indices(mask.shape)
    disk = np.hypot(rr - crown_rc[0], cc - crown_rc[1]) <= radius_px
    touching = np.unique(labels[disk & mask])
    touching = touching[touching > 0]
    if touching.size == 0:
        raise ValueError('no parent-positive component within the plausibility radius of the crown')
    areas = np.bincount(labels.ravel(), minlength=n + 1)
    best = int(touching[np.argmax(areas[touching])])
    return labels == best, {'n_components_total': int(n), 'n_components_touching_disk': int(touching.size),
                            'area_px': int(areas[best])}


def min_distance_m(comp, transform, point_xy) -> float:
    """Minimum distance (metres, CRS units) from a point to the centres of the component's pixels."""
    rows, cols = np.nonzero(np.asarray(comp, bool))
    if rows.size == 0:
        return float('nan')
    x = transform.c + (cols + 0.5) * transform.a
    y = transform.f + (rows + 0.5) * transform.e
    return float(np.hypot(x - point_xy[0], y - point_xy[1]).min())


def area_recall(mask_up, scar10) -> float:
    """Fraction of the 10 m scar component's area that the upsampled 20 m rule also marks."""
    scar = np.asarray(scar10, bool)
    n = int(scar.sum())
    return float((np.asarray(mask_up, bool) & scar).sum() / n) if n else float('nan')


def a4_verdict(recall: float) -> str:
    """yaml experiments.A4.keep_rule on the point estimate: TRUE >= 0.90, FALSE < 0.70, INCONCLUSIVE between."""
    if recall >= 0.90:
        return 'TRUE'
    if recall < 0.70:
        return 'FALSE'
    return 'INCONCLUSIVE'


# ================================================================ the run ================================================================

def _distances_m(comp, transform, points_utm: dict) -> dict:
    return {name: min_distance_m(comp, transform, xy) for name, xy in points_utm.items()}


def main():
    import json
    from affine import Affine
    from pyproj import Transformer
    from experiments import f13_a3_sr_swap as A3
    from experiments.wayanad_evidence import config as CE
    from experiments.wayanad_evidence.data import load_date
    from experiments.wayanad_evidence.geo import label_components, reference_grid
    from experiments.wayanad_evidence.gate import parent_mask
    from trustsr.bootstrap import block_sums, ratio_bootstrap_ci
    started = C.utc_now()
    result = {'evidence': 'real', 'experiment_id': 'A4'}
    try:
        pre = C.require_committed_prereg()
        cfg = C.load_prereg()
        a4 = cfg['experiments']['A4']
        boot, ci = cfg['statistics']['bootstrap'], cfg['statistics']['ci']
        cfg_ev, ev_root, _ = CE.load()
        cache = ev_root / cfg_ev['paths']['cache']
        pre_dates, post_date = a4['data']['dates']['pre'], a4['data']['dates']['post']
        dates = pre_dates + [post_date]
        required = [cache / f'{d}.npz' for d in dates] + [ev_root / 'experiments/wayanad_evidence/outputs/step1.json']
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise C.Blocked('missing cached inputs: ' + '; '.join(missing) + ' | checked: ' + '; '.join(map(str, required)))
        thr = float(cfg_ev['change']['parent_drop_threshold'])
        if thr != 0.30:
            raise C.Blocked(f'wayanad_evidence parent_drop_threshold is {thr}, not the 0.30 pre-registered in the yaml')
        floor = float(cfg_ev['radiometry']['min_denominator'])
        transform, shape = reference_grid(cfg_ev['aoi'])
        transform20 = Affine(20.0, 0.0, transform.c, 0.0, -20.0, transform.f)
        to_utm = Transformer.from_crs('EPSG:4326', cfg_ev['aoi']['crs'], always_xy=True)
        pts = {name: to_utm.transform(p['lon'], p['lat']) for name, p in cfg_ev['aoi']['plausibility_points'].items()}
        crown_xy = pts['crown']
        crown_rc = (int((transform.f - crown_xy[1]) // 10), int((crown_xy[0] - transform.c) // 10))
        radius_m = float(cfg_ev['aoi']['plausibility_radius_m'])

        arrays = {d: load_date(cfg_ev, cache, d) for d in dates}

        # ---- 10 m rule, exactly as the evidence run (step1_baseline) -----------------------------------------
        valid_all = np.logical_and.reduce([arrays[d]['valid'] for d in dates])
        ndvi_pre = np.mean([arrays[d]['ndvi'] for d in pre_dates], axis=0)
        drop10 = np.where(valid_all, ndvi_pre - arrays[post_date]['ndvi'], np.nan).astype('float32')
        parent10 = parent_mask(drop10, valid_all, thr)
        comp10, info10 = scar_component(parent10, crown_rc, radius_m / 10.0, 8)
        labels10, _ = label_components(parent10, 8)
        sizes = np.bincount(labels10.ravel())[1:]
        largest_overall = labels10 == (int(np.argmax(sizes)) + 1)
        step1 = json.loads((ev_root / 'experiments/wayanad_evidence/outputs/step1.json').read_text(encoding='utf-8'))['measurements']
        d10 = _distances_m(comp10, transform, pts)
        ref = step1['plausibility']['distance_m_to_published_points']
        repro = {'scar_px_10m': int(comp10.sum()), 'step1_largest_component_px': step1['parent']['largest_component_px'],
                 'scar_is_the_largest_component_overall': bool(np.array_equal(comp10, largest_overall)),
                 'positive_px_aoi': int(parent10.sum()), 'step1_positive_px_aoi': step1['parent']['positive_px_aoi'],
                 'distances_m': d10, 'step1_distances_m': ref}
        repro['matches_step1'] = bool(repro['scar_px_10m'] == repro['step1_largest_component_px']
                                      and repro['positive_px_aoi'] == repro['step1_positive_px_aoi']
                                      and repro['scar_is_the_largest_component_overall']
                                      and all(abs(d10[k] - ref[k]) < 1e-6 for k in ref))
        result['reproduction_gate'] = repro
        if not repro['matches_step1']:
            result.update(status='FAIL', verdict='NOT_RUN', reason='reproduction gate failed: the 10 m scar component is not '
                          'the evidence run\'s (step1.json); A4 stopped before the 20 m rule was computed')
            return _finish(result, started)

        # ---- 20 m rule: area-average REFLECTANCE 2x2, recompute NDVI, same threshold, strict validity ------------
        nd20, v20 = {}, []
        for d in dates:
            r20, v = aggregate_reflectance_2x2(arrays[d]['refl'], arrays[d]['valid'])
            nd20[d] = ndvi_from_reflectance(r20, floor)
            v20.append(v & np.isfinite(nd20[d]))
        valid20 = np.logical_and.reduce(v20)
        drop20 = np.where(valid20, np.mean([nd20[d] for d in pre_dates], axis=0) - nd20[post_date], np.nan).astype('float32')
        parent20 = parent_mask(drop20, valid20, thr)
        crown_rc20 = (crown_rc[0] // 2, crown_rc[1] // 2)
        try:
            comp20, info20 = scar_component(parent20, crown_rc20, radius_m / 20.0, 8)
        except ValueError as exc:
            comp20, info20 = np.zeros_like(parent20), {'error': str(exc)}
        up_all, up_comp = upsample_nn(parent20, 2), upsample_nn(comp20, 2)
        valid20_up = upsample_nn(valid20, 2)
        ones = np.ones(comp10.shape, bool)
        recall = area_recall(up_all, comp10)
        verdict = a4_verdict(recall)
        inter_t = block_sums(up_all & comp10, 32)
        den_t = block_sums(comp10, 32)
        rci = ratio_bootstrap_ci(inter_t, den_t, boot['replicates'], ci, boot['seed'])
        comp10_valid20 = comp10 & valid20_up
        d20 = _distances_m(comp20, transform20, pts) if comp20.any() else {k: None for k in pts}
        meas = [
            C.measurement('area_recall_20m_rule_vs_10m_scar', recall, numerator=int((up_all & comp10).sum()),
                          denominator=int(comp10.sum()), lo=rci['lo'], hi=rci['hi'], replicates=boot['replicates'],
                          seed=boot['seed'], unit='fraction of the 10 m scar component area also positive under the 20 m rule',
                          bootstrap_unit='32 x 32 px (10 m) = 320 m tiles; the scar covers few tiles, so the CI is coarse',
                          blocks=rci['blocks']),
            C.measurement('IoU_20m_scar_component_vs_10m_scar_component', (lambda r: r['iou'])(A3.iou(up_comp, comp10, ones)),
                          numerator=A3.iou(up_comp, comp10, ones)['intersection'], denominator=A3.iou(up_comp, comp10, ones)['union'],
                          unit='IoU after nearest-neighbour upsampling 20 m -> 10 m'),
            C.measurement('IoU_20m_parent_mask_vs_10m_scar_component', A3.iou(up_all, comp10, ones)['iou'],
                          numerator=A3.iou(up_all, comp10, ones)['intersection'], denominator=A3.iou(up_all, comp10, ones)['union'],
                          unit='IoU of ALL 20 m positive cells (upsampled) vs the 10 m scar component; extra positives count against it')]
        result.update(
            status='PASS' if verdict == 'TRUE' else 'FAIL', verdict=verdict, rule_applied=a4['keep_rule'],
            preregistration={**pre, 'experiment': 'A4'},
            verdict_basis='area recall = 10 m scar-component px that are also positive under the 20 m rule / 10 m scar-component px, point estimate (yaml method.proposed_not_in_memo.area_recall)',
            scar={'definition': a4['method']['proposed_not_in_memo']['scar_component'],
                  'area_10m': {'px': int(comp10.sum()), 'm2': float(comp10.sum() * 100.0), 'connected_components_touching_disk': info10['n_components_touching_disk']},
                  'area_20m': {'cells': int(comp20.sum()), 'm2': float(comp20.sum() * 400.0), 'info': info20},
                  'area_ratio_20m_over_10m': (float(comp20.sum() * 400.0 / (comp10.sum() * 100.0)) if comp10.sum() else None)},
            area_recall={'estimate': recall, 'numerator': int((up_all & comp10).sum()), 'denominator': int(comp10.sum()), 'ci': rci},
            area_recall_component_only=area_recall(up_comp, comp10),
            iou={'scar_component_20m_vs_10m': A3.iou(up_comp, comp10, ones), 'parent_mask_20m_vs_scar_10m': A3.iou(up_all, comp10, ones),
                 'whole_aoi_parent_masks': A3.iou(up_all, parent10, ones)},
            validity_effect={'scar_px_10m_in_invalid_20m_cells': int((comp10 & ~valid20_up).sum()),
                             'share_of_scar_px': float((comp10 & ~valid20_up).sum() / comp10.sum()),
                             'recall_on_scar_px_with_valid_20m_cell': area_recall(up_all & valid20_up, comp10_valid20),
                             'denominator_valid_only': int(comp10_valid20.sum()),
                             'note': 'sensitivity only: scar px whose 2 x 2 cell has an invalid child cannot be positive at 20 m'},
            distance_to_published_points_m={'ten_m_component': d10, 'twenty_m_component': d20,
                                            'points': {k: {'lon': v['lon'], 'lat': v['lat']} for k, v in cfg_ev['aoi']['plausibility_points'].items()},
                                            'definition': 'minimum distance from the point to the centre of any pixel of the component; published coordinates are a plausibility check, never a label'},
            positive_counts={'parent_10m_px': int(parent10.sum()), 'parent_20m_cells': int(parent20.sum()),
                             'parent_20m_cells_as_10m_px': int(parent20.sum() * 4)},
            measurements=meas,
            limitations=[
                'One scene, one event, one scar: this informs framing only (yaml: abandon_when "—"); it is not a generalisation about landslides.',
                'The 10 m scar component is a threshold-0.30 NDVI-drop component (provisional, set before viewing data), not a surveyed polygon; '
                'recall is agreement with the 10 m rule, not accuracy against truth. No 2.5 m or 10 m change truth exists for Wayanad.',
                'A 20 m cell is invalid if any child is invalid on any date, so cloud/shadow edges inside the scar cannot be positive at 20 m; '
                'validity_effect reports how many scar px that touches and the recall restricted to the rest.',
                'Both rules use the single post date 2024-12-06 (129 days after the event); season and regrowth are confounded with the landslide at both resolutions.',
                'The recall CI resamples 320 m tiles of one component and is coarse; the verdict uses the point estimate as pre-registered.',
                'The prompt says "the component intersecting the published crown location"; the committed yaml says the largest component within '
                '1500 m of the crown, which is what is used. The 10 m component lies 5.7 m from the crown point, so it does not strictly intersect it.'])
        return _finish(result, started)
    except C.Blocked as exc:
        result.update(status='BLOCKED', verdict='NOT_RUN', reason=str(exc))
        return _finish(result, started)


def _finish(result, started):
    import json
    out = C.write_result('a4', result, started)
    print(json.dumps({k: out.get(k) for k in ('status', 'verdict', 'reason')}, indent=2))
    return C.exit_code(out['status'])


if __name__ == '__main__':
    raise SystemExit(main())
