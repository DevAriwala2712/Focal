"""F13 A1: does super-resolution add analytical information over interpolation? (O4 + O1 on the F5 cache)

Pre-registered in configs/f13_discovery.yaml `experiments.A1` (sha256 84beef1c...). Thresholds, arms, strata, seeds
and the keep rule come from that file only. Every interpolation uses trustsr.hrbench.resample (pixel-centre aligned).
IoU is sum(intersection) / sum(union) over images (yaml statistics.ratio_metrics), bootstrapped by resampling images
paired across arms; F5's per-image mean is reported beside every number.

This module also holds the OpenSR-test plumbing A2 reuses: offline data and model loading, the F5 reproduction gate
and the SR cache under data/experiments-cache/f13/sr_opensr/.
"""
from __future__ import annotations

import numpy as np

from experiments import f13_common as C

BLOCK = 4
THRESHOLD = 0.35                       # yaml A1 data.truth / proposed detector: HR NDVI < 0.35 and 2.5 m NDVI < 0.35
NARROW_BINS = ('0-20', '20-50')


# ---------------------------------------------------------------- deterministic helpers ----------------------------------------------------------------

def interp_index(index10, method: str) -> np.ndarray:
    """x4 interpolation of a 10 m index with the pixel-centre-aligned resampler; NaN -> 0 first (F5 precedent)."""
    from trustsr.hrbench import resample
    return resample(np.nan_to_num(np.asarray(index10, np.float64), nan=0.0)[None], BLOCK, method)[0]


def hr_at_5m_field(hr_refl) -> np.ndarray:
    """yaml A1: HR reflectance block-averaged to 5 m, NDVI, bilinear-upsampled (centre aligned) back to 2.5 m."""
    from trustsr import alloc
    from trustsr.hrbench import block_mean, resample
    nd5 = alloc.ndvi_rgbn(block_mean(hr_refl, 2))
    return resample(np.nan_to_num(nd5, nan=0.0)[None], 2, 'bilinear')[0]


def random_scores(shape, seed: int, image_index: int) -> np.ndarray:
    """The random ranking arm: uniform scores, reproducible per image from (yaml seed, image index)."""
    return np.random.default_rng([seed, image_index]).random(shape)


def sum_ratio(inter, union) -> float:
    i, u = float(np.sum(inter)), float(np.sum(union))
    return i / u if u > 0 else float('nan')


def paired_ratio_diff_ci(int_a, uni_a, int_b, uni_b, replicates: int, seed: int, ci: float, return_draws=False) -> dict:
    """Difference of two sum-ratio IoUs, bootstrapped by resampling IMAGES with the same indices for both arms.

    A replicate whose resampled union is 0 in either arm is skipped and counted.
    """
    ia, ua, ib, ub = (np.asarray(v, np.float64) for v in (int_a, uni_a, int_b, uni_b))
    n = ia.size
    est = sum_ratio(ia, ua) - sum_ratio(ib, ub)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(replicates, n))
    da, db = ua[idx].sum(axis=1), ub[idx].sum(axis=1)
    ok = (da > 0) & (db > 0)
    draws = ia[idx].sum(axis=1)[ok] / da[ok] - ib[idx].sum(axis=1)[ok] / db[ok]
    lo, hi = np.quantile(draws, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    out = {'estimate': est, 'lo': float(lo), 'hi': float(hi), 'n_images': int(n), 'replicates': replicates,
           'replicates_skipped': int((~ok).sum()), 'seed': seed, 'ci': ci,
           'iou_a': sum_ratio(ia, ua), 'iou_b': sum_ratio(ib, ub),
           'numerator_a': float(ia.sum()), 'denominator_a': float(ua.sum()),
           'numerator_b': float(ib.sum()), 'denominator_b': float(ub.sum())}
    if return_draws:
        out['draws'] = draws
    return out


def pooled_stratum(width_m, truth_valid) -> np.ndarray:
    """The yaml's verdict stratum: union of the 0-20 and 20-50 m width bins (trustsr.alloc.width_bin_masks)."""
    from trustsr import alloc
    bins = alloc.width_bin_masks(width_m, truth_valid)
    return bins['0-20'] | bins['20-50']


def _significant_positive(c) -> bool:
    return c['estimate'] > 0 and c['lo'] > 0


def a1_verdict(primary_contrasts) -> str:
    """yaml experiments.A1.keep_rule: TRUE if either primary contrast (without NAIP, pooled 0-50 m) has a positive paired
    IoU delta whose CI excludes 0; FALSE when both CIs span 0 or are negative. The two exhaust the cases."""
    return 'TRUE' if any(_significant_positive(c) for c in primary_contrasts) else 'FALSE'


def pivot_met(c) -> bool:
    """pivot_rules.keep_sr_in_evidence_path, A1 part: estimate >= 0.01 AND CI excluding 0."""
    return bool(c['estimate'] >= 0.01 and (c['lo'] > 0 or c['hi'] < 0))


# ================================================================ shared OpenSR-test plumbing (A1 and A2) ================================================================

F5_TARGETS = {                         # f5.json per_class.nonveg_ndvi_0.35.summary.ALL.methods.<key>.iou_mean (per-image mean)
    'blocky_10m': 0.759622759769101,
    'alloc_bilinear|oracle': 0.8286258719415689,
    'alloc_pretrained_sr|oracle': 0.8367043654137548,
    'alloc_bilinear|realistic': 0.4456180831263151,
    'alloc_pretrained_sr|realistic': 0.4464903424954483}
SR_CACHE = C.ROOT / 'data' / 'experiments-cache' / 'f13' / 'sr_opensr'


def block_network():
    """No network in Pass 2: HF offline and every socket connect raises."""
    import os
    import socket
    os.environ['HF_HUB_OFFLINE'] = '1'

    def _deny(*_a, **_k):
        raise RuntimeError('network access is forbidden in F13 Pass 2')
    socket.socket.connect = _deny
    socket.create_connection = _deny


def _cfg():
    import yaml
    from risk.common import load_config
    fix = yaml.safe_load((C.ROOT / 'configs/fix.yaml').read_text(encoding='utf-8'))
    a2 = yaml.safe_load((C.ROOT / 'configs/a2.yaml').read_text(encoding='utf-8'))
    phase0, _, _ = load_config(C.ROOT / a2['phase0_config'])
    return fix, a2, phase0


def verify_model_offline(phase0) -> dict:
    """Every pinned weight file present with its sha256, checked HERE, so load_model's download() returns early."""
    from risk.common import digest
    d = C.ROOT / phase0['paths']['model']
    rows = {}
    for name, sha in phase0['model']['sha256'].items():
        f = d / name
        got = digest(f) if f.is_file() else None
        rows[name] = {'path': str(f.relative_to(C.ROOT)), 'present': f.is_file(), 'sha256_ok': got == sha,
                      'bytes': f.stat().st_size if f.is_file() else None}
    if not all(r['sha256_ok'] for r in rows.values()):
        raise C.Blocked('pinned SEN2SR-lite weights missing or hash mismatch (load_model would download): '
                        + '; '.join(f'{k}: {v}' for k, v in rows.items()))
    return rows


def prepare(prereg_cfg) -> dict:
    """Load the cache, run real pretrained SR once per image, reproduce F5, write/verify the SR cache.

    Returns {'images': [...], 'gate': {...}, 'data': {...}, 'model': {...}, 'sr_cache': {...}}. Each image carries lr,
    hr (float64 reflectance), valid (HR mask with F5's 16 px border), sr (float64, clipped as F5) and identifiers.
    Raises C.Blocked on any missing input; the caller stops if gate['pass'] is False.
    """
    import json
    import time
    import torch
    from experiments import f5_a8_mapping as F5
    from experiments.common import array_sha256
    from risk.common import digest, write_json
    from trustsr import hrbench as hb
    fix, a2, phase0 = _cfg()
    a1 = prereg_cfg['experiments']['A1']
    cache = C.ROOT / a1['data']['cache']
    datasets = list(a1['data']['datasets'])
    need = [cache / f'{d}.pkl' for d in datasets]
    missing = [str(p) for p in need if not p.is_file()]
    if missing:
        raise C.Blocked('F5 OpenSR-test cache missing: ' + '; '.join(missing) + f' | checked {cache}')
    model_rows = verify_model_offline(phase0)
    seed = int(a2['seed'])
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    sr_fn = F5.build_sr_fn({'phase0': phase0}, C.ROOT, 'cpu')
    raw = F5.load_datasets(cache, datasets)
    idx_b, dn = a2['lr_band_index'], a2['dn_scale']
    specs = F5.class_specs(fix['a8'])[:1]                     # primary class only: nonveg_ndvi_0.35
    if specs[0][0] != 'nonveg_ndvi_0.35' or specs[0][2] != THRESHOLD:
        raise C.Blocked(f'F5 primary class is {specs[0]}, not nonveg_ndvi_0.35')

    images, f5_records, stacks, counts = [], [], {}, {}
    gidx = 0
    t0 = time.time()
    for ds in datasets:
        d = raw[ds]
        l2a, hrh, md = np.asarray(d['L2A']), np.asarray(d['HRharm']), d['metadata']
        counts[ds] = int(len(md))
        srs = []
        for i in range(len(md)):
            lr = hb.dn_to_reflectance(l2a[i][idx_b], dn)
            hr = hb.dn_to_reflectance(hrh[i], dn)
            valid = (hrh[i] > 0).all(axis=0) & np.isfinite(hr).all(axis=0)
            b = F5.BORDER_HR_PX
            valid[:b], valid[-b:], valid[:, :b], valid[:, -b:] = False, False, False, False
            sr = np.clip(np.asarray(sr_fn(lr), np.float64), 0.0, 1.0)
            srs.append(sr.astype(np.float32))
            if not valid.any():
                continue
            name = str(md['roi'].iloc[i])
            lr64, hr64 = lr.astype(np.float64), hr.astype(np.float64)
            f5_records.append({'dataset': ds, 'name': name, 'group': str(md['lr_gee_id'].iloc[i]),
                               'valid_px': int(valid.sum()),
                               'metrics': F5.evaluate_image(lr64, hr64, valid, sr, specs, (1, 2))})
            images.append({'dataset': ds, 'name': name, 'index_in_dataset': i, 'global_index': gidx,
                           'lr': lr64, 'hr': hr64, 'valid': valid, 'sr': sr})
            gidx += 1
        stacks[ds] = np.stack(srs)
    sr_seconds = time.time() - t0

    # ---- the F5 reproduction gate ---------------------------------------------------------------------------------
    f5 = json.loads((C.ROOT / 'experiments/results/f5.json').read_text(encoding='utf-8'))
    summ = F5.summarize(f5_records, (1, 2), specs[0][0])['ALL']['methods']
    rows = {k: {'target': v, 'measured': summ[k]['iou_mean'], 'abs_diff': abs(summ[k]['iou_mean'] - v),
                'n_images': summ[k]['n_images_iou_defined']} for k, v in F5_TARGETS.items()}
    sr_hash = {ds: {'measured': array_sha256(stacks[ds]), 'f5_json': f5['determinism']['sr_output_sha256'][ds]}
               for ds in datasets}
    for v in sr_hash.values():
        v['match'] = v['measured'] == v['f5_json']
    gate = {'iou_mean_ALL_targets': rows, 'tolerance': 1e-9,
            'iou_pass': all(r['abs_diff'] <= 1e-9 for r in rows.values()),
            'sr_output_sha256_per_dataset': sr_hash, 'sr_hash_pass': all(v['match'] for v in sr_hash.values()),
            'n_images_evaluated': len(f5_records), 'n_images_per_dataset': counts}
    gate['pass'] = bool(gate['iou_pass'] and gate['sr_hash_pass'] and counts == {
        'naip': 62, 'spot': 9, 'spain_crops': 28, 'spain_urban': 20})
    for im, rec in zip(images, f5_records):                  # per-image F5 IoUs, for the arm-equivalence checks
        im['f5_iou'] = {k: rec['metrics']['classes'][specs[0][0]]['methods'][k]['iou'] for k in F5_TARGETS}

    # ---- SR cache: written once, verified on every later run ----------------------------------------------------------
    manifest_path = SR_CACHE / 'manifest.json'
    files = {}
    for ds in datasets:
        for i in range(len(stacks[ds])):
            nm = str(raw[ds]['metadata']['roi'].iloc[i]).replace('/', '_')
            rel = f'{ds}/{i:03d}_{nm}.npy'
            p = SR_CACHE / rel
            arr = stacks[ds][i]
            if not p.is_file():
                p.parent.mkdir(parents=True, exist_ok=True)
                np.save(p, arr)
            if not np.array_equal(np.load(p), arr):
                raise C.Blocked(f'cached SR {p} differs from this run; refusing to overwrite it')
            files[rel] = {'sha256_file': digest(p), 'array_sha256': array_sha256(arr), 'shape': list(arr.shape)}
    manifest = {'what': 'pretrained SEN2SR-lite x4 output (B04,B03,B02,B08 reflectance, clipped to [0,1], float32) for every '
                        'image of the F5 OpenSR-test cache, in cache order; written once by F13 Pass 2',
                'source_cache': str(cache.relative_to(C.ROOT)), 'hub_revision': a1['data']['hub']['revision'],
                'model_files': model_rows, 'per_dataset_stack_sha256': {k: v['measured'] for k, v in sr_hash.items()},
                'matches_f5_sr_output_sha256': gate['sr_hash_pass'], 'files': files}
    if manifest_path.is_file():
        old = json.loads(manifest_path.read_text(encoding='utf-8'))
        if {k: v for k, v in old.items() if k != 'written_utc'} != manifest:
            raise C.Blocked(f'{manifest_path} exists with different content; refusing to overwrite')
        manifest_state = 'verified existing manifest'
    else:
        manifest['written_utc'] = C.utc_now()
        write_json(manifest_path, manifest)
        manifest_state = 'written'
    del raw, stacks
    return {'images': images, 'gate': gate,
            'data': {'cache': str(cache.relative_to(C.ROOT)), 'hub_revision': a1['data']['hub']['revision'],
                     'n_images_per_dataset': counts, 'n_images_evaluated': len(images),
                     'revision_evidence': '.claude/worktrees/agent-aee3717766a1e05f9/data_a2/x2_fetch_log.txt records the '
                                          'download URLs at this revision with byte counts equal to the cached pickles'},
            'model': model_rows, 'sr_seconds': round(sr_seconds, 1),
            'sr_cache': {'dir': str(SR_CACHE.relative_to(C.ROOT)), 'manifest': str(manifest_path.relative_to(C.ROOT)),
                         'state': manifest_state, 'n_files': len(files)}}


def environment_extra() -> dict:
    import sys
    return {'python_executable': sys.executable,
            'why_this_interpreter': 'the project .venv lacks pandas, which the OpenSR-test pickles need; this is the '
                                    '.venv_a2 interpreter F5 and A2 used (same python/numpy/torch versions as f5.json)'}


# ================================================================ A1 ================================================================

THRESHOLD_ARMS = ('sr_threshold', 'bilinear_threshold', 'bicubic_threshold')
SENSITIVITY_ARMS = ('bilinear_refl_threshold', 'bicubic_refl_threshold')
RANKING_ARMS = ('random_ranked', 'bilinear_ranked', 'bicubic_ranked', 'sr_ranked', 'hr_at_5m_ranked')
SCOPES = ('whole', '0-20', '20-50', '50-150', '150+', 'pooled_0_50')
PRIMARY = (('sr_threshold', 'bilinear_threshold'), ('sr_ranked', 'bilinear_ranked'))
SECONDARY = (('sr_threshold', 'bicubic_threshold'), ('sr_ranked', 'bicubic_ranked'),
             ('hr_at_5m_ranked', 'bilinear_ranked'), ('bilinear_ranked', 'random_ranked'))
SENSITIVITY = (('sr_threshold', 'bilinear_refl_threshold'), ('sr_threshold', 'bicubic_refl_threshold'),
               ('bilinear_refl_threshold', 'bilinear_threshold'))


def image_masks(im, seed: int) -> dict:
    from trustsr import alloc
    from trustsr.hrbench import resample
    lr, hr, sr, valid = im['lr'], im['hr'], im['sr'], im['valid']
    s_lr, s_hr, s_sr = alloc.ndvi_rgbn(lr), alloc.ndvi_rgbn(hr), alloc.ndvi_rgbn(sr)
    truth = np.isfinite(s_hr) & (s_hr < THRESHOLD)
    frac = alloc.block_fraction_oracle(truth & valid, BLOCK)
    below = lambda f: np.isfinite(f) & (f < THRESHOLD)
    fields = {'bilinear': interp_index(s_lr, 'bilinear'), 'bicubic': interp_index(s_lr, 'bicubic')}
    m = {'sr_threshold': below(s_sr),
         'bilinear_threshold': below(fields['bilinear']), 'bicubic_threshold': below(fields['bicubic']),
         'bilinear_refl_threshold': below(alloc.ndvi_rgbn(resample(lr, BLOCK, 'bilinear'))),
         'bicubic_refl_threshold': below(alloc.ndvi_rgbn(resample(lr, BLOCK, 'bicubic'))),
         'random_ranked': alloc.allocate_by_rank(random_scores(s_sr.shape, seed, im['global_index']), frac, BLOCK),
         'bilinear_ranked': alloc.allocate_by_rank(fields['bilinear'], frac, BLOCK),
         'bicubic_ranked': alloc.allocate_by_rank(fields['bicubic'], frac, BLOCK),
         'sr_ranked': alloc.allocate_by_rank(s_sr, frac, BLOCK),
         'hr_at_5m_ranked': alloc.allocate_by_rank(hr_at_5m_field(hr), frac, BLOCK)}
    return m, truth, int((~np.isfinite(s_lr)).sum())


def score_image(im, seed: int) -> dict:
    from experiments.f5_a8_mapping import PIXEL_M
    from trustsr import alloc
    masks, truth, n_nan_lr = image_masks(im, seed)
    valid = im['valid']
    tv = truth & valid
    width, _l, _p = alloc.component_width_m(tv, PIXEL_M)
    strata = alloc.width_bin_masks(width, tv)
    strata['pooled_0_50'] = pooled_stratum(width, tv)
    regions = {k: alloc.dilate(v, 4) & valid for k, v in strata.items()}
    out = {'dataset': im['dataset'], 'name': im['name'], 'n_nonfinite_lr_ndvi_px': n_nan_lr,
           'stratum_truth_px': {k: int(v.sum()) for k, v in strata.items()}, 'arms': {}}
    for arm, mask in masks.items():
        rec = {}
        r = alloc.iou(mask, truth, valid)
        rec['whole'] = {'i': r['intersection'], 'u': r['union'], 'iou': r['iou'],
                        'bf1': {t: alloc.boundary_f1(mask, truth, valid, t)['f1'] for t in (1, 2)}}
        for k, sm in strata.items():
            s = alloc.stratum_iou(mask, sm, valid)
            bf = ({t: alloc.boundary_f1(mask, sm, regions[k], t)['f1'] for t in (1, 2)} if sm.any()
                  else {1: None, 2: None})
            rec[k] = {'i': s['intersection'], 'u': s['union'], 'iou': s['iou'], 'bf1': bf}
        out['arms'][arm] = rec
    out['f5_equivalence'] = {
        'bilinear_ranked_equals_f5_alloc_bilinear_oracle': out['arms']['bilinear_ranked']['whole']['iou'] == im['f5_iou']['alloc_bilinear|oracle'],
        'sr_ranked_equals_f5_alloc_pretrained_sr_oracle': out['arms']['sr_ranked']['whole']['iou'] == im['f5_iou']['alloc_pretrained_sr|oracle']}
    return out


def _subset(recs, which):
    return recs if which == 'with_naip' else [r for r in recs if r['dataset'] != 'naip']


def _arrays(recs, arm, scope):
    i = np.array([r['arms'][arm][scope]['i'] for r in recs], float)
    u = np.array([r['arms'][arm][scope]['u'] for r in recs], float)
    return i, u


def _per_image_mean_diff(recs, a, b, scope, key, boot):
    from trustsr.bootstrap import paired_bootstrap_ci
    diffs = []
    for r in recs:
        x = r['arms'][a][scope]['iou'] if key == 'iou' else r['arms'][a][scope]['bf1'][key]
        y = r['arms'][b][scope]['iou'] if key == 'iou' else r['arms'][b][scope]['bf1'][key]
        if x is not None and y is not None:
            diffs.append(x - y)
    if not diffs:
        return {'n_images': 0}
    c = paired_bootstrap_ci(np.asarray(diffs), boot['replicates'], 0.95, boot['seed'])
    return {'estimate': c['mean'], 'lo': c['lo'], 'hi': c['hi'], 'n_images': c['n'],
            'n_a_better': int(sum(d > 0 for d in diffs)), 'replicates': boot['replicates'], 'seed': boot['seed']}


def contrast(recs, a, b, boot) -> dict:
    out = {}
    for which in ('with_naip', 'without_naip'):
        rs = _subset(recs, which)
        out[which] = {}
        for scope in SCOPES:
            ia, ua = _arrays(rs, a, scope)
            ib, ub = _arrays(rs, b, scope)
            if ua.sum() <= 0 or ub.sum() <= 0:
                out[which][scope] = {'n_images': len(rs), 'note': 'no truth in this stratum'}
                continue
            row = paired_ratio_diff_ci(ia, ua, ib, ub, boot['replicates'], boot['seed'], 0.95)
            row['n_images_with_stratum_truth'] = int(sum(r['stratum_truth_px'].get(scope, 1) > 0 for r in rs)) if scope != 'whole' else len(rs)
            row['f5_style_per_image_mean_iou_diff'] = _per_image_mean_diff(rs, a, b, scope, 'iou', boot)
            row['boundary_f1_1px_per_image_mean_diff'] = _per_image_mean_diff(rs, a, b, scope, 1, boot)
            row['boundary_f1_2px_per_image_mean_diff_secondary'] = _per_image_mean_diff(rs, a, b, scope, 2, boot)
            out[which][scope] = row
    return out


def arm_table(recs, arms, boot) -> dict:
    from trustsr.bootstrap import ratio_bootstrap_ci
    out = {}
    datasets = sorted({r['dataset'] for r in recs})
    for which in ('with_naip', 'without_naip') + tuple(datasets):
        rs = _subset(recs, which) if which in ('with_naip', 'without_naip') else [r for r in recs if r['dataset'] == which]
        out[which] = {}
        for arm in arms:
            out[which][arm] = {}
            for scope in SCOPES:
                i, u = _arrays(rs, arm, scope)
                if u.sum() <= 0:
                    out[which][arm][scope] = None
                    continue
                ci = ratio_bootstrap_ci(i, u, boot['replicates'], 0.95, boot['seed'])
                per = [r['arms'][arm][scope]['iou'] for r in rs if r['arms'][arm][scope]['iou'] is not None]
                bf = [r['arms'][arm][scope]['bf1'][1] for r in rs if r['arms'][arm][scope]['bf1'][1] is not None]
                out[which][arm][scope] = {'iou_sum_ratio': ci['estimate'], 'lo': ci['lo'], 'hi': ci['hi'],
                                          'numerator': ci['numerator'], 'denominator': ci['denominator'],
                                          'n_images': len(rs), 'iou_per_image_mean_f5_style': float(np.mean(per)) if per else None,
                                          'boundary_f1_1px_per_image_mean': float(np.mean(bf)) if bf else None}
    return out


def main():
    import json
    import time
    started = C.utc_now()
    t0 = time.time()
    result = {'evidence': 'real', 'experiment_id': 'A1'}
    try:
        block_network()
        pre = C.require_committed_prereg()
        cfg = C.load_prereg()
        a1 = cfg['experiments']['A1']
        boot = cfg['statistics']['bootstrap']
        P = prepare(cfg)
        result['reproduction_check'] = P['gate']
        result['data'], result['sr_cache'] = P['data'], P['sr_cache']
        if not P['gate']['pass']:
            result.update(status='BLOCKED', verdict='NOT_RUN',
                          reason='F5 reproduction gate failed; no A1 arm was computed (see reproduction_check)')
            return _finish(result, started)

        recs = [score_image(im, cfg['seed']) for im in P['images']]
        eq = {k: all(r['f5_equivalence'][k] for r in recs) for k in recs[0]['f5_equivalence']}
        result['reproduction_check']['arm_equivalence_with_f5_per_image'] = eq
        if not all(eq.values()):
            result.update(status='BLOCKED', verdict='NOT_RUN', reason='bilinear_ranked / sr_ranked do not equal F5\'s '
                          'allocation arms image by image; something other than the declared arm changed')
            return _finish(result, started)

        contrasts = {'primary': {f'{a} - {b}': contrast(recs, a, b, boot) for a, b in PRIMARY},
                     'secondary': {f'{a} - {b}': contrast(recs, a, b, boot) for a, b in SECONDARY},
                     'sensitivity_reflectance_first': {f'{a} - {b}': contrast(recs, a, b, boot) for a, b in SENSITIVITY}}
        verdict_rows = [contrasts['primary'][f'{a} - {b}']['without_naip']['pooled_0_50'] for a, b in PRIMARY]
        verdict = a1_verdict(verdict_rows)
        pivot = {f'{a} - {b}': {'met': pivot_met(r), 'estimate': r['estimate'], 'lo': r['lo'], 'hi': r['hi']}
                 for (a, b), r in zip(PRIMARY, verdict_rows)}
        meas = []
        for (a, b), r in zip(PRIMARY, verdict_rows):
            meas.append(C.measurement(f'delta_IoU_{a}_minus_{b}_pooled_0_50_without_naip', r['estimate'],
                                      numerator=None, denominator=None, lo=r['lo'], hi=r['hi'],
                                      replicates=r['replicates'], seed=r['seed'],
                                      unit='difference of sum-ratio IoU (sum intersection / sum union over images)',
                                      iou_a=r['iou_a'], numerator_a=r['numerator_a'], denominator_a=r['denominator_a'],
                                      iou_b=r['iou_b'], numerator_b=r['numerator_b'], denominator_b=r['denominator_b'],
                                      n_images=r['n_images']))
            w = contrasts['primary'][f'{a} - {b}']['with_naip']['pooled_0_50']
            meas.append(C.measurement(f'delta_IoU_{a}_minus_{b}_pooled_0_50_with_naip', w['estimate'],
                                      lo=w['lo'], hi=w['hi'], replicates=w['replicates'], seed=w['seed'],
                                      unit='difference of sum-ratio IoU', numerator=None, denominator=None,
                                      iou_a=w['iou_a'], numerator_a=w['numerator_a'], denominator_a=w['denominator_a'],
                                      iou_b=w['iou_b'], numerator_b=w['numerator_b'], denominator_b=w['denominator_b'],
                                      n_images=w['n_images']))
        result.update(
            status='PASS' if verdict == 'TRUE' else 'FAIL', verdict=verdict,
            rule_applied=a1['keep_rule'], preregistration={**pre, 'experiment': 'A1'},
            verdict_basis='primary contrasts, without_naip, pooled 0-50 m stratum, sum-ratio IoU delta with image-paired bootstrap CI',
            pivot_threshold_keep_sr_in_evidence_path=pivot,
            contrasts=contrasts,
            arms=arm_table(recs, THRESHOLD_ARMS + SENSITIVITY_ARMS + RANKING_ARMS, boot),
            per_image=[{'dataset': r['dataset'], 'name': r['name'], 'n_nonfinite_lr_ndvi_px': r['n_nonfinite_lr_ndvi_px'],
                        'stratum_truth_px': r['stratum_truth_px'],
                        'iou': {arm: {s: r['arms'][arm][s]['iou'] for s in ('whole', 'pooled_0_50')} for arm in r['arms']}}
                       for r in recs],
            measurements=meas,
            interpretations=[
                'IoU is sum(intersection)/sum(union) over images (yaml statistics.ratio_metrics); F5 reported per-image mean IoU, '
                'which is given beside every contrast (f5_style_per_image_mean_iou_diff) and every arm.',
                'Image-paired bootstrap: images are resampled with the same indices for both arms (image = tile, yaml '
                'resampling_unit.hr_image_benchmarks); replicates whose resampled union is 0 are skipped and counted.',
                "Verdict stratum 'pooled 0-50 m' = trustsr.alloc.stratum_iou on the union of the 0-20 and 20-50 m truth width "
                'bins (widths from alloc.component_width_m on truth & valid, as F5), evaluated in that union dilated by 4 px.',
                'A1 TRUE if EITHER primary contrast is positive with CI excluding 0 (yaml: "for either primary contrast"); FALSE '
                'otherwise (yaml: "for both primary contrasts ... CI spans 0 or the estimate is negative"); these exhaust the cases.',
                'Threshold arms interpolate the 10 m NDVI with trustsr.hrbench.resample (pixel-centre aligned, F5 precedent of '
                'interpolating the index, NaN -> 0); interpolating reflectance first is reported as a labelled sensitivity.',
                'random_ranked uses uniform scores from numpy default_rng([2024, image index]) per image (yaml seed 2024).',
                'hr_at_5m: HR reflectance block-averaged 2x2 to 5 m, NDVI, then hrbench bilinear x2 back to 2.5 m.',
                'Boundary F1 is reported as a per-image mean with an image-paired bootstrap (the yaml ratio_metrics rule names '
                'FAR, IoU and recall, not F1); per stratum it is evaluated inside the dilated stratum region.',
                'The pivot check (estimate >= 0.01 and CI excluding 0) is reported per primary contrast and does not change the verdict.',
                'The scripts run under .claude/worktrees/agent-aee3717766a1e05f9/.venv_a2/bin/python (F5\'s interpreter) because '
                'the project .venv has no pandas, which the cached pickles require; versions match f5.json environment.'],
            limitations=[
                'Static land-cover boundaries (HR NDVI < 0.35) are a proxy for change boundaries, not a landslide label (F5 caveat).',
                'SEN2SR-lite is pretrained, not fine-tuned; it was trained on NAIP, so without_naip is the verdict set.',
                'HRharm is harmonised to S2 by the dataset authors; spain_crops and spain_urban have weaker geometric agreement (A2 report).',
                'Width strata use whole-component widths (F5 definition); a wide component with narrow arms counts as wide.',
                'NAIP LR is 121 px, reflect-padded to the 128 px model input (F5 adapter); the padded border lies in the excluded 16 px.'],
            runtime_seconds=round(time.time() - t0, 1), sr_seconds=P['sr_seconds'], environment_extra=environment_extra())
        return _finish(result, started)
    except C.Blocked as exc:
        result.update(status='BLOCKED', verdict='NOT_RUN', reason=str(exc))
        return _finish(result, started)


def _finish(result, started):
    import json
    result.setdefault('interpretations', [])
    result.setdefault('reproduction_check', None)
    result.setdefault('measurements', [])
    out = C.write_result('a1', result, started)
    print(json.dumps({k: out.get(k) for k in ('status', 'verdict', 'reason')}, indent=2))
    return C.exit_code(out['status'])


if __name__ == '__main__':
    raise SystemExit(main())
