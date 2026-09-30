"""F5 / A8: does sub-pixel allocation ranked by REAL SR output beat a naive bilinear ranking, against real HR truth?

Pre-registered in `configs/fix.yaml` block `a8`. Nothing here is landslide-specific: the target is a static land-cover
class (non-vegetation by HR NDVI, water by HR NDWI) on the OpenSR-test HRharm images A2 already cached and validated.
That makes the truth a PROXY for a change boundary, never a landslide label - see the caveat in the report.

Protocol
--------
* Images: OpenSR-test HRharm sets whose HR grid is the exact 4x subdivision of the paired real Sentinel-2 L2A 10 m input
  (naip, spot, spain_crops, spain_urban). venus is 5 m (2x) and cannot carry a 2.5 m truth, so it is excluded and said so.
* The 10 m input is the REAL paired Sentinel-2 L2A image shipped with each HR reference. No synthetic downsample is used
  anywhere in this experiment.
* Truth at 2.5 m: HR NDVI < 0.35 -> non-vegetation (primary; sensitivity at 0.25 and 0.45). HR NDWI > 0 -> water.
* Methods (all produce a 2.5 m binary mask):
    blocky_10m               threshold at 10 m, replicate to all 16 sub-pixels (no allocation)
    hard_threshold_v1_style  v1's gate: 2.5 m hard threshold on the SR index AND the replicated 10 m parent
    alloc_bilinear           block-sum-constrained allocation, sub-pixels ranked by bilinear-upsampled 10 m index
    alloc_pretrained_sr      same allocation, ranked by the index of REAL pretrained SEN2SR-lite output
    alloc_finetuned_sr       BLOCKED until F4 lands; never substituted by pretrained output
    heavy_sr                 not run (Colab-only, human-operated)
* Fractions: `oracle` (block fraction read off the HR truth: isolates ranking quality) and `realistic` (block fraction
  from two-endmember unmixing of the 10 m image alone: what the product would compute, error and all).
* Metrics per image: IoU, boundary F1 at 1 px and 2 px on the 2.5 m grid, signed/absolute area error. Strata: truth
  components binned by local width (0-20, 20-50, 50-150, 150+ m).
* Headline: paired IoU(alloc_pretrained_sr) - IoU(alloc_bilinear) over images, percentile bootstrap 95% CI, reported
  with whatever sign it has, with and without NAIP (possible SEN2NAIPv2 train/test overlap; see A2/F6).

Run (the venv that has pandas + torch + the pinned model, i.e. the A2 environment):
    python -m experiments.f5_a8_mapping
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import warnings
from pathlib import Path

import numpy as np
import yaml

from experiments.common import Blocked, array_sha256, environment_record, finalize_result
from risk.common import load_config, write_json
from trustsr import alloc, hrbench as hb
from trustsr.bootstrap import paired_bootstrap_ci

NAME = 'f5'
warnings.filterwarnings('ignore')

BLOCK = 4
PIXEL_M = 2.5
BORDER_HR_PX = 16                      # same exclusion as A2 (opensr-test Config.border_mask default)
ALLOC_METHODS = ('alloc_bilinear', 'alloc_pretrained_sr')
FRACTIONS = ('oracle', 'realistic')

# Cache and model directories are looked up here; every candidate is an existing A2 artefact, nothing is re-downloaded.
CACHE_CANDIDATES = ('data_a2/opensr_test',
                    '../agent-aee3717766a1e05f9/data_a2/opensr_test',
                    '../../../data_a2/opensr_test')


# ---------------------------------------------------------------- class definitions

def class_specs(a8cfg):
    """Each spec: (name, index, threshold, unmixing band pair). Index 0 is the pre-registered PRIMARY class.

    Thresholds are pre-registered in `configs/fix.yaml` a8.truth as prose; every value used below must appear there, so
    editing the config without editing this list (or the reverse) fails loudly instead of silently diverging.
    """
    out = [('nonveg_ndvi_0.35', 'ndvi', 0.35, (0, 3)),
           ('nonveg_ndvi_0.25', 'ndvi', 0.25, (0, 3)),
           ('nonveg_ndvi_0.45', 'ndvi', 0.45, (0, 3)),
           ('water_ndwi_0.0', 'ndwi', 0.0, (1, 3))]
    prose = f"{a8cfg['truth']['primary']} | {a8cfg['truth']['secondary']}"
    for name, kind, thr, _bands in out:
        token = f'{thr:g}' if kind == 'ndvi' else '0'
        if token not in prose:
            raise ValueError(f'{name}: threshold {thr} is not the pre-registered fix.yaml a8.truth: {prose!r}')
    if 'NDVI' not in prose or 'NDWI' not in prose:
        raise ValueError(f'fix.yaml a8.truth no longer names NDVI and NDWI: {prose!r}')
    return out


def _pearson(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or a[ok].std() == 0 or b[ok].std() == 0:
        return None
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def score_from(refl, kind: str) -> np.ndarray:
    """Score where LOW = more likely the target class. Non-vegetation: NDVI. Water: -NDWI."""
    if kind == 'ndvi':
        return alloc.ndvi_rgbn(refl)
    if kind == 'ndwi':
        return -alloc.ndwi_rgbn(refl)
    raise ValueError(kind)


def score_threshold(kind: str, threshold: float) -> float:
    """Truth/decision rule is always `score < thr` after the sign convention of `score_from`."""
    return threshold if kind == 'ndvi' else -threshold


# ---------------------------------------------------------------- one image

def evaluate_image(lr, hr, valid, sr, specs, boundary_tol) -> dict:
    """All methods x fraction variants x classes for one image. `sr`, `lr`, `hr` are (C,.,.) reflectance in [0,1]."""
    lr_valid = (lr > 0).all(axis=0) & np.isfinite(lr).all(axis=0)
    out = {'classes': {}}
    for name, kind, thr, band_pair in specs:
        s_thr = score_threshold(kind, thr)
        s_hr, s_sr, s_lr = score_from(hr, kind), score_from(sr, kind), score_from(lr, kind)
        # bilinear ranking = bilinear upsample of the 10 m INDEX (the naive ranking baseline in the spec)
        s_bil = hb.resample(np.nan_to_num(s_lr, nan=0.0)[None], BLOCK, 'bilinear')[0]
        truth = np.isfinite(s_hr) & (s_hr < s_thr)

        frac_oracle = alloc.block_fraction_oracle(truth & valid, BLOCK)
        frac_real, unmix_diag = alloc.block_fraction_realistic(lr, s_lr, lr_valid, band_pair)
        rank = {'alloc_bilinear': s_bil, 'alloc_pretrained_sr': s_sr}

        masks = {}
        masks[('blocky_10m', None)] = alloc.blocky_mask(s_lr, s_thr, BLOCK)
        masks[('hard_threshold_v1_style', None)] = alloc.hard_threshold_v1_style(s_sr, s_lr, s_thr, BLOCK)
        for m in ALLOC_METHODS:
            masks[(m, 'oracle')] = alloc.allocate_by_rank(rank[m], frac_oracle, BLOCK)
            masks[(m, 'realistic')] = alloc.allocate_by_rank(rank[m], frac_real, BLOCK)

        width_m, _labels, _per = alloc.component_width_m(truth & valid, PIXEL_M)
        strata = alloc.width_bin_masks(width_m, truth & valid)

        per_method = {}
        for (m, f), mask in masks.items():
            key = m if f is None else f'{m}|{f}'
            rec = {'method': m, 'fraction': f, **alloc.iou(mask, truth, valid),
                   'area_error': alloc.area_error(mask, truth, valid),
                   'boundary_f1': {str(t): alloc.boundary_f1(mask, truth, valid, t) for t in boundary_tol},
                   'strata': {b: alloc.stratum_iou(mask, sm, valid)
                              for b, sm in strata.items()}}
            per_method[key] = rec
        out['classes'][name] = {
            'threshold': thr, 'index': kind,
            'truth_px': int((truth & valid).sum()), 'valid_px': int(valid.sum()),
            'truth_fraction': float((truth & valid).sum() / max(valid.sum(), 1)),
            'stratum_truth_px': {b: int(sm.sum()) for b, sm in strata.items()},
            'fraction_estimate': {
                'unmixing': unmix_diag,
                'pearson_r_vs_oracle': _pearson(frac_real.ravel(), frac_oracle.ravel()),
                'mean_abs_error_vs_oracle': float(np.abs(frac_real - frac_oracle).mean()),
                'mean_signed_error_vs_oracle': float((frac_real - frac_oracle).mean()),
                'oracle_mean': float(frac_oracle.mean()), 'realistic_mean': float(frac_real.mean())},
            'methods': per_method}
    return out


# ---------------------------------------------------------------- aggregation

def _mean(vals):
    v = [x for x in vals if x is not None and np.isfinite(x)]
    return (float(np.mean(v)) if v else None), len(v)


def _median(vals):
    v = [x for x in vals if x is not None and np.isfinite(x)]
    return float(np.median(v)) if v else None


def summarize(records, boundary_tol, class_name) -> dict:
    """Per-dataset and pooled means of every metric, with the denominator (number of images) beside each."""
    keys = sorted({k for r in records for k in r['metrics']['classes'][class_name]['methods']})
    datasets = sorted({r['dataset'] for r in records})
    out = {}
    for ds in datasets + ['ALL', 'ALL_no_naip']:
        rec = ([r for r in records if r['dataset'] == ds] if ds in datasets else
               (records if ds == 'ALL' else [r for r in records if r['dataset'] != 'naip']))
        blk = {'n_images': len(rec), 'n_images_with_truth': sum(
            r['metrics']['classes'][class_name]['truth_px'] > 0 for r in rec), 'methods': {}}
        for key in keys:
            m = [r['metrics']['classes'][class_name]['methods'][key] for r in rec]
            iou_mean, n_iou = _mean([x['iou'] for x in m])
            pred_px = sum(x['area_error']['pred_px'] for x in m)
            truth_px = sum(x['area_error']['truth_px'] for x in m)
            entry = {'iou_mean': iou_mean, 'n_images_iou_defined': n_iou,
                     'iou_median': _median([x['iou'] for x in m]),
                     # per-image relative area error explodes on images whose truth area is a handful of pixels, so the
                     # median and the pooled ratio-of-sums are reported beside the mean. All three share n_images.
                     'abs_area_error_mean': _mean([x['area_error']['abs_relative'] for x in m])[0],
                     'abs_area_error_median': _median([x['area_error']['abs_relative'] for x in m]),
                     'signed_area_error_mean': _mean([x['area_error']['relative'] for x in m])[0],
                     'signed_area_error_median': _median([x['area_error']['relative'] for x in m]),
                     'pooled_area_error': {'pred_px': pred_px, 'truth_px': truth_px,
                                           'relative': (float((pred_px - truth_px) / truth_px) if truth_px else None)},
                     'boundary_f1_mean': {}}
            for t in boundary_tol:
                v, n = _mean([x['boundary_f1'][str(t)]['f1'] for x in m])
                entry['boundary_f1_mean'][str(t)] = {'mean': v, 'n_images': n}
            entry['strata_iou_mean'] = {}
            for b in alloc.WIDTH_BIN_NAMES:
                v, n = _mean([x['strata'][b]['iou'] for x in m])
                entry['strata_iou_mean'][b] = {'mean': v, 'n_images': n,
                                               'truth_px': int(sum(x['strata'][b]['n_truth'] for x in m))}
            blk['methods'][key] = entry
        out[ds] = blk
    return out


def fraction_summary(records, class_name, datasets) -> dict:
    """How good the `realistic` block fraction is against the `oracle` one - the other half of an allocation's error."""
    out = {}
    for ds in list(datasets) + ['ALL']:
        rec = records if ds == 'ALL' else [r for r in records if r['dataset'] == ds]
        fe = [r['metrics']['classes'][class_name]['fraction_estimate'] for r in rec]
        out[ds] = {'n_images': len(fe),
                   'block_fraction_mae_mean': _mean([f['mean_abs_error_vs_oracle'] for f in fe])[0],
                   'block_fraction_pearson_r_mean': _mean([f['pearson_r_vs_oracle'] for f in fe])[0],
                   'block_fraction_signed_error_mean': _mean([f['mean_signed_error_vs_oracle'] for f in fe])[0],
                   'oracle_fraction_mean': _mean([f['oracle_mean'] for f in fe])[0],
                   'realistic_fraction_mean': _mean([f['realistic_mean'] for f in fe])[0]}
    return out


def compact_records(records) -> list:
    """Per-image rows trimmed to the numbers a reader (or a verifier) needs; the full dicts stay in memory only."""
    out = []
    for r in records:
        row = {'dataset': r['dataset'], 'name': r['name'], 'group': r['group'], 'valid_px': r['valid_px'], 'classes': {}}
        for cname, c in r['metrics']['classes'].items():
            fe = c['fraction_estimate']
            row['classes'][cname] = {
                'truth_px': c['truth_px'], 'truth_fraction': round(c['truth_fraction'], 6),
                'stratum_truth_px': c['stratum_truth_px'],
                'fraction_mae_vs_oracle': round(fe['mean_abs_error_vs_oracle'], 6),
                'fraction_pearson_r_vs_oracle': (None if fe['pearson_r_vs_oracle'] is None
                                                 else round(fe['pearson_r_vs_oracle'], 6)),
                'methods': {k: {'iou': v['iou'],
                                'boundary_f1': {t: b['f1'] for t, b in v['boundary_f1'].items()},
                                'area_error_relative': v['area_error']['relative'],
                                'strata_iou': {b: s['iou'] for b, s in v['strata'].items()}}
                            for k, v in c['methods'].items()}}
        out.append(row)
    return out


def paired_delta(records, class_name, key_a, key_b, boot, subset=None) -> dict:
    """Image-paired difference of IoU between two method|fraction keys, with a percentile bootstrap CI."""
    rec = records if subset is None else [r for r in records if subset(r)]
    diffs, names = [], []
    for r in rec:
        ms = r['metrics']['classes'][class_name]['methods']
        a, b = ms[key_a]['iou'], ms[key_b]['iou']
        if a is None or b is None:
            continue
        diffs.append(a - b)
        names.append(f"{r['dataset']}/{r['name']}")
    if not diffs:
        return {'n': 0, 'note': 'no image has both IoUs defined'}
    ci = paired_bootstrap_ci(np.asarray(diffs), boot['replicates'], boot['ci'], boot['seed'])
    return {**ci, 'a': key_a, 'b': key_b, 'n_images': len(diffs),
            'n_images_a_better': int(sum(d > 0 for d in diffs)), 'n_images_tied': int(sum(d == 0 for d in diffs)),
            'mean_iou_a': float(np.mean([r['metrics']['classes'][class_name]['methods'][key_a]['iou']
                                         for r in rec if r['metrics']['classes'][class_name]['methods'][key_a]['iou'] is not None])),
            'mean_iou_b': float(np.mean([r['metrics']['classes'][class_name]['methods'][key_b]['iou']
                                         for r in rec if r['metrics']['classes'][class_name]['methods'][key_b]['iou'] is not None])),
            'datasets': sorted({n.split('/')[0] for n in names})}


def stratum_delta(records, class_name, key_a, key_b, boot, subset=None) -> dict:
    rec = records if subset is None else [r for r in records if subset(r)]
    out = {}
    for b in alloc.WIDTH_BIN_NAMES:
        diffs = []
        for r in rec:
            ms = r['metrics']['classes'][class_name]['methods']
            x, y = ms[key_a]['strata'][b]['iou'], ms[key_b]['strata'][b]['iou']
            if x is None or y is None:
                continue
            diffs.append(x - y)
        out[b] = ({'n': 0} if not diffs else
                  {**paired_bootstrap_ci(np.asarray(diffs), boot['replicates'], boot['ci'], boot['seed']),
                   'n_images': len(diffs), 'n_images_a_better': int(sum(d > 0 for d in diffs))})
    return out


# ---------------------------------------------------------------- data + model

def find_dir(root: Path, candidates, needed) -> Path | None:
    for c in candidates:
        p = (root / c).resolve()
        if p.is_dir() and all((p / n).exists() for n in needed):
            return p
    return None


def load_datasets(cache: Path, names):
    from experiments.x2_data import safe_load
    return {n: safe_load(cache / f'{n}.pkl')[0] for n in names}


def build_sr_fn(cfg, root, device):
    """Real pretrained SEN2SR-lite forward pass, reflect-padded to the 128 px native input for images below it."""
    import torch
    from risk.model import load_model
    model = load_model(cfg['phase0'], root, trainable=False, device=device)

    def model_fn(batch):
        with torch.no_grad():
            return model(torch.from_numpy(np.ascontiguousarray(batch, np.float32)).to(device)).float().cpu().numpy()

    def sr_fn(lr):
        lr = np.asarray(lr, np.float32)
        if lr.shape[1] > 128 or lr.shape[2] > 128:
            raise ValueError(f'image {lr.shape} exceeds the 128 px native input; A8 uses only <=128 px OpenSR-test tiles')
        return hb.sr_native_padded(model_fn, lr, 128, BLOCK)
    return sr_fn


# ---------------------------------------------------------------- probe

def probe(cfg, root, args) -> dict:
    import torch
    a2 = cfg['a2run']
    seed = int(a2['seed'])
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cache = Path(args.cache_dir).resolve() if args.cache_dir else find_dir(
        root, CACHE_CANDIDATES, [f'{d}.pkl' for d in a2['datasets']])
    if cache is None:
        raise Blocked('no cached OpenSR-test pickles found (A2 cache missing); '
                      "no usable HR images -> no 'SR adds information' claim anywhere", evidence='real')
    model_root = Path(args.model_root).resolve() if args.model_root else root
    if not (model_root / cfg['phase0']['paths']['model'] / 'load.py').is_file():
        raise Blocked(f'pretrained SEN2SR-lite artefacts not found under {model_root}', evidence='real')

    raw = load_datasets(cache, a2['datasets'])
    idx, dn = a2['lr_band_index'], a2['dn_scale']

    # keep only sets whose HR reference is the exact 4x subdivision (2.5 m); anything coarser cannot carry a 2.5 m truth
    usable, excluded = [], {}
    for ds, d in raw.items():
        lr_shape, hr_shape = np.asarray(d['L2A']).shape, np.asarray(d['HRharm']).shape
        s = hr_shape[-1] // lr_shape[-1]
        if s == BLOCK:
            usable.append(ds)
        else:
            excluded[ds] = f'HR reference is {s}x the 10 m grid ({10.0 / s} m), not the 2.5 m grid A8 requires'
    if not usable:
        raise Blocked("no OpenSR-test set has a 2.5 m HR reference; no usable HR images -> no 'SR adds information' "
                      'claim anywhere', evidence='real')

    sr_fn = build_sr_fn(cfg, model_root, device)
    specs = class_specs(cfg['fix']['a8'])
    boundary_tol = (1, 2)

    records, sr_hashes = [], {}
    t0 = time.time()
    for ds in usable:
        d = raw[ds]
        l2a, hrh, md = np.asarray(d['L2A']), np.asarray(d['HRharm']), d['metadata']
        n = len(md) if args.limit is None else min(args.limit, len(md))
        srs = []
        for i in range(n):
            lr = hb.dn_to_reflectance(l2a[i][idx], dn)
            hr = hb.dn_to_reflectance(hrh[i], dn)
            valid = (hrh[i] > 0).all(axis=0) & np.isfinite(hr).all(axis=0)
            valid[:BORDER_HR_PX], valid[-BORDER_HR_PX:] = False, False
            valid[:, :BORDER_HR_PX], valid[:, -BORDER_HR_PX:] = False, False
            sr = np.clip(np.asarray(sr_fn(lr), np.float64), 0.0, 1.0)
            srs.append(sr.astype(np.float32))
            if not valid.any():
                continue
            records.append({'dataset': ds, 'name': str(md['roi'].iloc[i]), 'group': str(md['lr_gee_id'].iloc[i]),
                            'valid_px': int(valid.sum()),
                            'metrics': evaluate_image(lr.astype(np.float64), hr.astype(np.float64), valid, sr,
                                                      specs, boundary_tol)})
        sr_hashes[ds] = array_sha256(np.stack(srs))
    elapsed = time.time() - t0

    if not records:
        raise Blocked("every candidate HR image was fully masked; no usable HR images -> no 'SR adds information' "
                      'claim anywhere', evidence='real')

    boot = cfg['statistics']['bootstrap']
    bootstrap = {'replicates': int(boot['replicates']), 'ci': float(boot['ci']), 'seed': int(boot['seed'])}
    primary = specs[0][0]

    per_class = {}
    for name, *_ in specs:
        per_class[name] = {'summary': summarize(records, boundary_tol, name),
                           'fraction_estimate': fraction_summary(records, name, usable)}

    headline = {}
    for frac in FRACTIONS:
        a, b = f'alloc_pretrained_sr|{frac}', f'alloc_bilinear|{frac}'
        headline[frac] = {
            'all_datasets': paired_delta(records, primary, a, b, bootstrap),
            'excluding_naip': paired_delta(records, primary, a, b, bootstrap, subset=lambda r: r['dataset'] != 'naip'),
            'per_dataset': {ds: paired_delta(records, primary, a, b, bootstrap,
                                             subset=lambda r, ds=ds: r['dataset'] == ds) for ds in usable},
            'by_feature_width_all_datasets': stratum_delta(records, primary, a, b, bootstrap),
            'by_feature_width_excluding_naip': stratum_delta(records, primary, a, b, bootstrap,
                                                            subset=lambda r: r['dataset'] != 'naip')}

    other_classes_headline = {name: {'oracle': paired_delta(records, name, 'alloc_pretrained_sr|oracle',
                                                            'alloc_bilinear|oracle', bootstrap),
                                     'oracle_excluding_naip': paired_delta(
                                         records, name, 'alloc_pretrained_sr|oracle', 'alloc_bilinear|oracle',
                                         bootstrap, subset=lambda r: r['dataset'] != 'naip')}
                              for name, *_ in specs[1:]}

    # secondary comparisons that give the headline a scale: SR allocation vs the two non-allocating methods
    context = {frac: {against: paired_delta(records, primary, f'alloc_pretrained_sr|{frac}', against, bootstrap)
                      for against in ('blocky_10m', 'hard_threshold_v1_style')} for frac in FRACTIONS}

    # sanity: the oracle fraction has strictly more information than the estimated one
    oracle_vs_realistic = {m: paired_delta(records, primary, f'{m}|oracle', f'{m}|realistic', bootstrap)
                           for m in ALLOC_METHODS}

    h_all = headline['oracle']['all_datasets']
    verdict = ('SR_RANKING_BETTER' if h_all['lo'] > 0 else
               'BILINEAR_RANKING_BETTER' if h_all['hi'] < 0 else 'NO_DETECTABLE_DIFFERENCE')
    return {
        'status': 'PASS',
        'status_meaning': ('PASS = the pre-registered A8 test ran end to end on real HR imagery with real pretrained SR '
                           'inference and the headline delta is reported with its CI. fix.yaml a8 sets no numeric keep '
                           'threshold; the scientific answer is in `verdict`, which may be negative.'),
        'evidence': 'real',
        'verdict': verdict,
        'summary': (f"paired IoU(alloc_pretrained_sr) - IoU(alloc_bilinear) on {primary}, oracle fraction, "
                    f"all datasets: {h_all['mean']:+.4f} [{h_all['lo']:+.4f}, {h_all['hi']:+.4f}] "
                    f"over n={h_all['n_images']} images -> {verdict}"),
        'proxy_caveat': ('Static land-cover boundaries (vegetation / water) are a PROXY for change boundaries, not a '
                         'landslide label. A8 measures whether SR-informed ranking places sub-pixel detail at real HR '
                         'boundaries in general; it says nothing about landslide-specific accuracy.'),
        'headline': headline,
        'headline_primary_class': primary,
        'headline_other_classes': other_classes_headline,
        'context_vs_non_allocating_methods': context,
        'oracle_vs_realistic_fraction': oracle_vs_realistic,
        'per_class': per_class,
        'per_image': compact_records(records),
        'methods': {
            'blocky_10m': 'implemented (trustsr.alloc.blocky_mask)',
            'hard_threshold_v1_style': ('implemented (trustsr.alloc.hard_threshold_v1_style); it is '
                                        'experiments/wayanad_evidence/gate.py OBSERVED = S & P, the 2.5 m hard '
                                        'threshold AND the replicated 10 m parent, with no fraction constraint'),
            'alloc_bilinear': 'implemented (bilinear-upsampled 10 m index as the ranking score)',
            'alloc_pretrained_sr': f'implemented, REAL pretrained SEN2SR-lite forward pass on {device}',
            'alloc_finetuned_sr': 'BLOCKED: F4 fine-tune has not landed. Never substituted by pretrained output.',
            'heavy_sr_optional_colab_only': 'NOT RUN: Colab is human-operated and out of scope for this sandbox.'},
        'protocol': {
            'pixel_size_m': PIXEL_M, 'block': BLOCK, 'border_hr_px_excluded': BORDER_HR_PX,
            'low_resolution_input': ('REAL paired Sentinel-2 L2A 10 m image shipped with each OpenSR-test HR '
                                     'reference. No synthetic downsample is used anywhere in this experiment.'),
            'reference': 'HRharm (harmonized to the S2 L2A bands by the dataset authors), the A2 primary reference',
            'lr_band_index': idx, 'model_band_order': a2['model_band_order'], 'dn_scale': dn,
            'truth': {'non_vegetation': 'HR NDVI < threshold', 'water': 'HR NDWI > 0',
                      'primary_threshold': 0.35, 'sensitivity_thresholds': [0.25, 0.45]},
            'allocation': 'k = round(16 * fraction) lowest-score sub-pixels per 10 m block (trustsr.alloc.allocate_by_rank)',
            'fraction_oracle': 'block mean of the HR truth mask',
            'fraction_realistic': ('two-endmember unmixing of the 10 m image alone; endmembers = mean reflectance of '
                                   'its lowest/highest 10% index pixels (trustsr.alloc.block_fraction_realistic). '
                                   'No HR information enters it.'),
            'boundary_f1': 'inner 4-connected boundary pixels, matched within 1 and 2 px (2.5 / 5.0 m)',
            'feature_width': ('per truth component, (2 * max distance-to-background - 1) * 2.5 m; bins '
                              f'{list(alloc.WIDTH_BIN_NAMES)} m'),
            'stratum_iou': 'IoU inside the stratum dilated by 4 px (one 10 m parent block)',
            'bootstrap': bootstrap},
        'data': {'cache_dir': str(cache), 'datasets_used': usable, 'datasets_excluded': excluded,
                 'n_images': len(records),
                 'n_images_per_dataset': {ds: sum(r['dataset'] == ds for r in records) for ds in usable},
                 'n_unique_s2_acquisitions': len({r['group'] for r in records}),
                 'hub_repo': a2['hub']['repo'], 'hub_revision': a2['hub']['revision']},
        'determinism': {'seed': seed, 'device': device, 'sr_output_sha256': sr_hashes,
                        'use_deterministic_algorithms': True},
        'runtime_seconds': round(elapsed, 1),
        'limitations': [
            'Static land-cover boundaries are a proxy for change boundaries; this is not a landslide label.',
            'HRharm is radiometrically harmonized to Sentinel-2 by the dataset authors, which favours LR-consistent '
            'estimators; the same reference is used for every method, so the comparison is fair, but absolute IoU is '
            'not a field-validated number.',
            'SEN2SR-lite was trained on SEN2NAIPv2 (NAIP). Train/test overlap with the OpenSR-test NAIP set cannot be '
            'excluded, so the headline is reported with and without NAIP.',
            'spain_crops and spain_urban pass A2 geometry at only 0.50 / 0.55 of images within 1 HR px; sub-pixel '
            'misalignment penalises every method equally but inflates boundary error for all of them.',
            'NAIP LR is 121 px and is reflect-padded to the 128 px native input, then cropped (A2 evaluation-only '
            'adapter); the padded border lies inside the 16 HR px excluded border.',
            'venus is excluded: its HR reference is 5 m, so it cannot carry a 2.5 m truth mask.',
            'alloc_finetuned_sr is BLOCKED (F4 not landed) and heavy SR was not run (Colab is human-only).',
            'The realistic fraction uses one unmixing estimator with percentile endmembers; a different endmember '
            'choice would move the realistic-fraction numbers (the oracle-fraction headline is unaffected).']}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default='configs/fix.yaml')
    ap.add_argument('--a2-config', default='configs/a2.yaml')
    ap.add_argument('--prereg', default='configs/exceptional.yaml')
    ap.add_argument('--cache-dir', default=None, help='directory holding the A2 OpenSR-test pickles')
    ap.add_argument('--model-root', default=None, help='root that contains models/SEN2SRLite_RGBN')
    ap.add_argument('--limit', type=int, default=None, help='debug: cap images per dataset')
    args = ap.parse_args()

    fixpath = Path(args.config).resolve()
    root = fixpath.parent.parent
    fix = yaml.safe_load(fixpath.read_text(encoding='utf-8'))
    a2 = yaml.safe_load((root / args.a2_config).read_text(encoding='utf-8'))
    prereg = yaml.safe_load((root / args.prereg).read_text(encoding='utf-8'))
    phase0, _, _ = load_config(root / a2['phase0_config'])
    cfg = {'fix': fix, 'a2run': a2, 'phase0': phase0, 'statistics': prereg['statistics']}
    config_hash = hashlib.sha256(fixpath.read_bytes()).hexdigest()
    started = time.strftime('%Y-%m-%dT%H:%M:%S+00:00', time.gmtime())
    try:
        result = probe(cfg, root, args)
    except Blocked as exc:
        result = {'status': 'BLOCKED', 'evidence': exc.evidence, 'reason': str(exc), 'limitations': exc.limitations}
    except Exception as exc:
        import traceback
        result = {'status': 'FAIL', 'evidence': 'real', 'error': f'{type(exc).__name__}: {exc}',
                  'traceback': traceback.format_exc()}
    out = finalize_result(NAME, result, config_hash, started)
    out['config_sha256_file'] = 'configs/fix.yaml'
    out['environment'] = environment_record()
    res_dir = root / 'experiments' / 'results'
    write_json(res_dir / 'f5.json', out)
    print(json.dumps({k: out.get(k) for k in ('status', 'verdict', 'summary', 'reason', 'error')}, indent=2))
    raise SystemExit(0 if out['status'] == 'PASS' else 2)


if __name__ == '__main__':
    main()
