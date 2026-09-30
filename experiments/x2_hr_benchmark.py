"""A2: score pretrained SEN2SR-lite RGBN x4 against nearest/bicubic/Lanczos on OpenSR-test (real HR references).

Run from the repo root inside the venv that has opensr-test (requirements-a2.txt):
    .venv_a2/bin/python -m experiments.x2_hr_benchmark
Exit 0 = PASS (SR beats bicubic on every evaluated dataset by the pre-registered rule), 2 = FAIL or BLOCKED.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import yaml

from experiments.common import Blocked, array_sha256, environment_record, finalize_result
from risk.common import digest, load_config, write_json
from trustsr import hrbench as hb

NAME = 'x2'
warnings.filterwarnings('ignore')


# ---------------------------------------------------------------- SR callable for any image size

def make_sr_fn(model_fn, tiling, oom_type, cleanup):
    """LR (C,h,w) float32 -> (C,4h,4w). <=128 px: reflect-pad to 128, run, crop. Larger: pad short axes, centre-crop tiling
    with the ADR-001 tiler. Every model call goes through the ADR-001 OOM policy (retry once at batch 1)."""
    from experiments.wayanad_evidence.sr import infer_with_oom_policy
    from experiments.wayanad_evidence.tiler import stitch
    n, s = tiling['native'], tiling['scale']

    def call(batch):
        return infer_with_oom_policy(model_fn, batch, 'x2 tile', oom_type, cleanup, lambda: {})

    def sr_fn(lr):
        lr = np.asarray(lr, np.float32)
        c, h, w = lr.shape
        if h <= n and w <= n:
            return hb.sr_native_padded(call, lr, n, s)
        ph, pw = max(0, n - h), max(0, n - w)
        pad = np.pad(lr, ((0, 0), (0, ph), (0, pw)), mode='reflect') if (ph or pw) else lr
        out = stitch(pad, call, tile=n, margin=tiling['margin'], stride=tiling['stride'], scale=s)
        return out[:, :h * s, :w * s]
    return sr_fn


# ---------------------------------------------------------------- data

def build_samples(name, d, a2, use_raw=False):
    idx, sc = a2['lr_band_index'], a2['dn_scale']
    l2a, hr_h, hr_raw, md = np.asarray(d['L2A']), np.asarray(d['HRharm']), np.asarray(d['HR']), d['metadata']
    out = []
    for i in range(len(md)):
        lr = hb.dn_to_reflectance(l2a[i][idx], sc)
        ref = hb.dn_to_reflectance((hr_raw if use_raw else hr_h)[i], sc)
        mask = (hr_raw[i] > 0).all(axis=0) & (hr_h[i] > 0).all(axis=0) & np.isfinite(ref).all(axis=0)
        out.append((name, hb.Sample(name=str(md['roi'].iloc[i]), lr=lr, hr=ref, mask=mask, group=str(md['lr_gee_id'].iloc[i]),
                                    meta={'crs': str(md['crs'].iloc[i]), 'affine': str(md['affine'].iloc[i])})))
    return out


def geometry_report(name, d, a2):
    md = d['metadata']
    idx = a2['lr_band_index']
    lr = np.asarray(d['L2A'])[:, idx].astype(np.float64)
    hr = np.asarray(d['HRharm']).astype(np.float64)
    s = hr.shape[-1] // lr.shape[-1]
    aff = np.array([[float(x) for x in str(a).split(',')] for a in md['affine']])
    shifts = [hb.grid_shift(lr[i], hr[i], s, a2['geometry_check']['max_shift_hr_px']) for i in range(len(md))]
    zero = sum((g['dy'], g['dx']) == (0, 0) for g in shifts)
    within1 = sum(max(abs(g['dy']), abs(g['dx'])) <= 1 for g in shifts)
    ok10 = int(((aff[:, 2] % 10 == 0) & (aff[:, 5] % 10 == 0)).sum())
    return {'n': len(md), 'ref_scale': int(s), 'hr_shape': list(hr.shape[1:]), 'lr_shape': list(lr.shape[1:]),
            'pixel_size_m': sorted(set(aff[:, 0].tolist())), 'hr_origin_on_10m_grid': ok10,
            'shift_search_zero_offset': int(zero), 'shift_search_within_1_hr_px': int(within1),
            'shift_pass_fraction': float(within1 / len(md)),
            'meta_spatial_misalignment_lr_px': {'mean': float(md['spatial'].astype(float).mean()),
                                                'max': float(md['spatial'].astype(float).max())},
            'unique_s2_acquisitions': int(md['lr_gee_id'].nunique()),
            'mean_abs_dn_blockmean_harm_vs_lr': np.round(np.abs(hr.reshape(len(md), 4, lr.shape[2], s, lr.shape[3], s)
                                                                 .mean(axis=(3, 5)) - lr).mean(axis=(0, 2, 3)), 1).tolist()}


# ---------------------------------------------------------------- opensr-test groups

def make_extra(a2):
    import torch
    import opensr_test
    cfg = opensr_test.Config(correctness_distance=a2['opensr_test']['correctness_distance'])

    def extra(sample, cand, lr):
        out = {}
        t_lr, t_hr = torch.from_numpy(np.asarray(lr, np.float32)), torch.from_numpy(np.asarray(sample.hr, np.float32))
        for m in ('sr', 'bicubic'):
            try:
                r = opensr_test.Metrics(cfg).compute(lr=t_lr, sr=torch.from_numpy(cand[m].astype(np.float32)), hr=t_hr)
                out[m] = {k: float(v) for k, v in r.items()}
            except Exception as exc:                        # recorded, never silently dropped
                out[m] = {'error': f'{type(exc).__name__}: {exc}'}
        return out
    return extra


def summarize_extra(records):
    keys = ['reflectance', 'spectral', 'spatial', 'synthesis', 'ha_metric', 'om_metric', 'im_metric']
    res = {}
    for ds in dict.fromkeys(r['dataset'] for r in records):
        rec = [r for r in records if r['dataset'] == ds]
        res[ds] = {}
        for m in ('sr', 'bicubic'):
            ok = [r['extra'][m] for r in rec if 'error' not in r['extra'][m]]
            res[ds][m] = {'n_ok': len(ok), 'n_failed': len(rec) - len(ok),
                          **{k: (float(np.nanmean([o[k] for o in ok])) if ok else None) for k in keys}}
    return res


# ---------------------------------------------------------------- probe

def probe(cfg, root):
    a2 = cfg['a2run']
    import torch
    from experiments.x2_data import ByteLog, fetch_pickle, hub_listing, safe_load
    seed = a2['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    device = 'cuda' if a2['device'] in ('auto', 'cuda') and torch.cuda.is_available() else 'cpu'
    if a2['device'] == 'cuda' and device != 'cuda':
        raise Blocked('device cuda requested but unavailable', evidence='real')

    results = root / a2['results_dir']
    log = ByteLog(results / 'x2_fetch_log.txt', a2['byte_cap'])
    try:
        import opensr_test
        osr_version = opensr_test.__version__
    except Exception as exc:
        raise Blocked(f'opensr-test not importable in this environment ({type(exc).__name__}: {exc}); '
                      'use the venv from requirements-a2.txt', evidence='real')
    try:
        listing = hub_listing(log=log)
        if listing['revision'] != a2['hub']['revision']:
            raise Blocked(f'Hub revision moved: pinned {a2["hub"]["revision"]}, found {listing["revision"]}', evidence='real')
        raw = {ds: safe_load(fetch_pickle(ds, listing, root / a2['cache_dir'], log)) for ds in a2['datasets']}
    except Blocked:
        raise
    except Exception as exc:
        raise Blocked(f'OpenSR-test data could not be fetched/verified: {type(exc).__name__}: {exc}', evidence='real')

    from risk.model import load_model
    model = load_model(cfg['phase0'], root, trainable=False, device=device)
    cache = {}
    hashes_first = {}

    def model_fn(batch):
        with torch.no_grad():
            return model(torch.from_numpy(np.ascontiguousarray(batch, np.float32)).to(device)).float().cpu().numpy()

    core = make_sr_fn(model_fn, a2['tiling'], torch.OutOfMemoryError, lambda: torch.cuda.empty_cache() if device == 'cuda' else None)

    def sr_fn(lr):
        key = hashlib.sha1(np.ascontiguousarray(lr).tobytes()).hexdigest()
        if key not in cache:
            cache[key] = core(lr).astype(np.float32)
        return cache[key]

    boot = cfg['statistics']['bootstrap']
    bootstrap = {'replicates': boot['replicates'], 'ci': boot['ci'], 'seed': boot['seed']}
    mt = a2['metrics']
    geometry = {ds: geometry_report(ds, raw[ds][0], a2) for ds in a2['datasets']}

    items = [it for ds in a2['datasets'] for it in build_samples(ds, raw[ds][0], a2)]
    extra = make_extra(a2) if a2['opensr_test']['enabled'] else None
    t0 = time.time()
    res = hb.evaluate_sr_fn(sr_fn, iter(items), sr_scale=a2['tiling']['scale'], border=mt['border_px'],
                            data_range=mt['data_range'], bootstrap=bootstrap, clip=mt['clip_unit_interval'], extra=extra)
    elapsed = time.time() - t0

    # sensitivity: unharmonized HR, only where its radiometry is on the S2 reflectance scale
    raw_items = [it for ds in a2['sensitivity_raw_hr'] for it in build_samples(ds, raw[ds][0], a2, use_raw=True)]
    sens = hb.evaluate_sr_fn(sr_fn, iter(raw_items), sr_scale=a2['tiling']['scale'], border=mt['border_px'],
                             data_range=mt['data_range'], bootstrap=bootstrap, clip=mt['clip_unit_interval'])
    raw_scale_check = {}
    for ds in a2['datasets']:
        d = raw[ds][0]
        hm, hr_ = np.asarray(d['HRharm']).astype(np.float64).mean(axis=(0, 2, 3)), np.asarray(d['HR']).astype(np.float64).mean(axis=(0, 2, 3))
        raw_scale_check[ds] = {'raw_over_harm_band_mean_ratio': np.round(hr_ / hm, 3).tolist()}

    # determinism: SR hashes, then a full recompute with an empty cache
    def sr_hashes():
        return {ds: array_sha256(np.stack([sr_fn(s.lr) for d_, s in items if d_ == ds])) for ds in a2['datasets']}
    h1 = sr_hashes()
    rerun_equal = None
    if a2['rerun_check']:
        cache.clear()
        h2 = sr_hashes()
        rerun_equal = h1 == h2

    per_ds = res['datasets']
    verdict_by_ds = {ds: bool(v['beats_bicubic']) for ds, v in per_ds.items()}
    n_beat = sum(verdict_by_ds.values())
    status = 'PASS' if n_beat == len(verdict_by_ds) else 'FAIL'
    verdict = 'ADOPT' if status == 'PASS' else ('REJECT' if n_beat == 0 else 'MIXED')
    freeze = root / 'data_a2' / 'freeze.txt'
    limitations = [
        'HRharm is harmonized to the Sentinel-2 L2A bands by the dataset authors (radiometry and sub-pixel alignment), which favours LR-consistent estimators such as bicubic; unharmonized HR is not on a common radiometric scale across sets (see raw_scale_check).',
        'No SCL/cloud mask ships with OpenSR-test; only zero-valued (no-data) pixels are masked.',
        'SEN2SR-lite was trained on SEN2NAIPv2 (NAIP); train/test overlap with the OpenSR-test NAIP set cannot be excluded from public information.',
        'NAIP LR is 121 px (< 128 native): reflect-padded to 128 and cropped; the padded border lies inside the 16 HR px excluded border. Production ADR-001 forbids padding; this is an evaluation-only adapter.',
        'Venus is x2 (5 m): every method is x4-upsampled then block-averaged 2x2 to the 5 m grid; the model output is not compared at 2.5 m there.',
        'Images sharing a Sentinel-2 acquisition are not independent; the by-group bootstrap is reported as a sensitivity.',
        'SEN2SR paper table not read (publisher pages return 403 / Cloudflare challenge); no published-number comparison is made.']
    return {
        'status': status, 'evidence': 'real', 'verdict': verdict,
        'summary': f'SR beats bicubic (PSNR CI lower bound > 0) on {n_beat}/{len(verdict_by_ds)} datasets',
        'keep_rule': {'per_dataset_beats_bicubic': verdict_by_ds},
        'protocol': res['protocol'], 'datasets': per_ds, 'pooled': res['pooled'], 'per_image': res['per_image'],
        'opensr_test_groups': summarize_extra(res['per_image']) if extra else None,
        'sensitivity_raw_hr': {'datasets': sens['datasets'], 'pooled': sens['pooled']},
        'raw_scale_check': raw_scale_check, 'geometry': geometry,
        'data': {'hub_repo': a2['hub']['repo'], 'revision': listing['revision'], 'files': listing['files'],
                 'bytes_downloaded_this_process': log.total, 'byte_cap': a2['byte_cap'], 'opensr_test_version': osr_version,
                 'license': 'MIT (dataset card and package)'},
        'determinism': {'seed': seed, 'use_deterministic_algorithms': True, 'sr_output_sha256': h1,
                        'full_recompute_hashes_equal': rerun_equal, 'torch_threads': torch.get_num_threads()},
        'device': device, 'inference_seconds_incl_metrics': round(elapsed, 1),
        'pip_freeze_sha256': digest(freeze) if freeze.is_file() else None,
        'limitations': limitations}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default='configs/a2.yaml')
    args = ap.parse_args()
    a2path = Path(args.config).resolve()
    a2 = yaml.safe_load(a2path.read_text(encoding='utf-8'))
    root = a2path.parent.parent
    pre = root / a2['preregistration']
    pre_cfg = yaml.safe_load(pre.read_text(encoding='utf-8'))
    phase0, _, _ = load_config(root / a2['phase0_config'])
    cfg = {**pre_cfg, 'phase0': phase0, 'a2run': a2}
    config_hash = hashlib.sha256((digest(pre) + digest(a2path)).encode()).hexdigest()
    started = time.strftime('%Y-%m-%dT%H:%M:%S+00:00', time.gmtime())
    try:
        result = probe(cfg, root)
    except Blocked as exc:
        result = {'status': 'BLOCKED', 'evidence': exc.evidence, 'reason': str(exc), 'limitations': exc.limitations}
    except Exception as exc:
        import traceback
        result = {'status': 'FAIL', 'evidence': 'real', 'error': f'{type(exc).__name__}: {exc}', 'traceback': traceback.format_exc()}
    out = finalize_result(NAME, result, config_hash, started)
    out['config_sha256_parts'] = {'configs/exceptional.yaml': digest(pre), args.config: digest(a2path)}
    out['environment'] = environment_record()
    res_dir = root / a2['results_dir']
    write_json(res_dir / 'x2.json', out)
    from experiments.x2_report import write_report
    write_report(out, res_dir / 'x2_REPORT.md')
    print(json.dumps({k: out.get(k) for k in ('status', 'verdict', 'summary', 'reason', 'error')}, indent=2))
    raise SystemExit(0 if out['status'] == 'PASS' else 2)


if __name__ == '__main__':
    main()
