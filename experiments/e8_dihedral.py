"""E8: 4 vs 8 dihedral transforms. Does NDVI sigma from {id, rot90, rot180, rot270} match the full 8?"""
from __future__ import annotations

import numpy as np

from experiments.common import (CLASS_CODES, Blocked, classify_change, load_real_crop, ndvi, run_cli,
                                spearman, synthetic_change_pair)
from experiments.e7_streaming_moments import Welford
from experiments.e1_tile_scheduler import model_operator


def apply_dihedral(x: np.ndarray, t: int) -> np.ndarray:
    """t = 4*flip + k: flip along W, then rotate 90*k degrees. t in 0..3 are the pure rotations."""
    if not 0 <= t < 8:
        raise ValueError(f'dihedral index must be in 0..7, got {t}')
    y = x[..., ::-1] if t >= 4 else x
    return np.ascontiguousarray(np.rot90(y, t % 4, axes=(-2, -1)))


def invert_dihedral(y: np.ndarray, t: int) -> np.ndarray:
    if not 0 <= t < 8:
        raise ValueError(f'dihedral index must be in 0..7, got {t}')
    z = np.rot90(y, -(t % 4), axes=(-2, -1))
    return np.ascontiguousarray(z[..., ::-1] if t >= 4 else z)


def dihedral_ndvi_stats(operator, tile: np.ndarray, transforms, min_denominator: float, ddof: int):
    """(mean, std, valid) of per-run NDVI. NDVI is computed per run AFTER inverting the transform."""
    acc = None
    for t in transforms:
        sr = invert_dihedral(operator(apply_dihedral(tile, t)[None])[0], t)
        n = ndvi(sr, min_denominator)
        acc = acc or Welford(n.shape)
        acc.update(n)
    mean, var = acc.mean, acc.variance(ddof)
    valid = np.isfinite(mean) & np.isfinite(var)
    return mean, np.sqrt(var), valid


def evaluate(operator, pairs, settings: dict, md: float, change: dict, scale: int) -> dict:
    """Compare sigma and five-class output from 4 vs 8 transforms over (pre10, post10) tile pairs.

    A pixel is comparable when it is valid under BOTH variants. Validity is not identical: NDVI undefined in any
    of the four extra runs invalidates the pixel only for the 8-set, so those pixels are counted separately.
    """
    four, eight = list(range(4)), list(range(8))
    acc = {n: {'sigma': [], 'valid': [], 'cls': []} for n in ('4', '8')}
    for pre, post in pairs:
        stats = {name: {n: dihedral_ndvi_stats(operator, date, ts, md, settings['ddof'])
                        for n, ts in (('4', four), ('8', eight))} for name, date in (('pre', pre), ('post', post))}
        parent10 = ndvi(pre, md) - ndvi(post, md)
        up = lambda m: np.repeat(np.repeat(m, scale, 0), scale, 1)
        parent_valid = up(np.isfinite(parent10))
        parent = up(np.nan_to_num(parent10, nan=0.0) > change['parent_drop_threshold'])
        for n in ('4', '8'):
            (m0, s0, v0), (m1, s1, v1) = stats['pre'][n], stats['post'][n]
            valid = v0 & v1 & parent_valid
            sigma = np.sqrt(np.nan_to_num(s0) ** 2 + np.nan_to_num(s1) ** 2)
            acc[n]['sigma'].append(sigma.ravel())
            acc[n]['valid'].append(valid.ravel())
            acc[n]['cls'].append(classify_change(np.nan_to_num(m0 - m1), sigma, parent, ~valid,
                                                 change['k']).ravel())
    cat = lambda n, key: np.concatenate(acc[n][key])
    v4, v8 = cat('4', 'valid'), cat('8', 'valid')
    both = v4 & v8
    s4, s8, c4, c8 = cat('4', 'sigma')[both], cat('8', 'sigma')[both], cat('4', 'cls')[both], cat('8', 'cls')[both]
    rho = spearman(s4, s8)
    agreement = float((c4 == c8).mean())
    names = {v: k for k, v in CLASS_CODES.items()}
    confusion = {f'{names[a]}->{names[b]}': int(((c4 == a) & (c8 == b)).sum())
                 for a in names for b in names if a != b and ((c4 == a) & (c8 == b)).any()}
    keep4 = bool(rho >= settings['min_spearman'] and agreement >= settings['min_class_agreement'])
    return {'tile_pairs': len(pairs), 'comparable_pixels': int(both.sum()),
            'valid_only_with_4_transforms': int((v4 & ~v8).sum()), 'valid_only_with_8_transforms': int((v8 & ~v4).sum()),
            'spearman_sigma4_sigma8': rho, 'five_class_agreement': agreement,
            'disagreeing_pixels': int((c4 != c8).sum()),
            'class_counts_8': {names[c]: int((c8 == c).sum()) for c in names if c != CLASS_CODES['NO_DATA']},
            'confusion_4_to_8': confusion,
            'sigma_median': {'4': float(np.median(s4)), '8': float(np.median(s8))},
            'k': change['k'], 'would_keep_4': keep4}


def real_pairs(settings, root, tile: int):
    pre, _, _ = load_real_crop(settings['real_pre'], root)
    post, _, _ = load_real_crop(settings['real_post'], root)
    if pre.shape != post.shape:
        raise Blocked('real pre/post crops differ in shape; they must share one 10 m grid', evidence='real')
    n = min(settings['tiles'], (pre.shape[1] // tile) * (pre.shape[2] // tile))
    per_row = pre.shape[2] // tile
    return [(pre[:, (i // per_row) * tile:(i // per_row + 1) * tile, (i % per_row) * tile:(i % per_row + 1) * tile],
             post[:, (i // per_row) * tile:(i // per_row + 1) * tile, (i % per_row) * tile:(i % per_row + 1) * tile])
            for i in range(n)]


def probe(cfg, root):
    """E8: 4 vs 8 dihedral transforms through the real pinned model (CPU); verdict needs real tiles."""
    import torch
    from risk.model import load_model
    s, p0, tile = cfg['e8'], cfg['phase0']['model'], cfg['phase0']['model']['native_tile']
    torch.manual_seed(cfg['seed'])
    op = model_operator(load_model(cfg['phase0'], root, device='cpu').eval())
    md, change = cfg['ndvi']['min_denominator'], cfg['change']
    synthetic = [synthetic_change_pair(tile, cfg['seed'] + i, s['square'], s['noise'])[:2] for i in range(s['tiles'])]
    mechanics = evaluate(op, synthetic, s, md, change, p0['scale'])
    common = {'criteria': {'keep_4_if': f"spearman >= {s['min_spearman']} and five-class agreement >= "
                                        f"{s['min_class_agreement']} at k = {change['k']}; otherwise keep 8",
                           'ddof': s['ddof'], 'parent_drop_threshold_uncalibrated_placeholder':
                           change['parent_drop_threshold']},
              'limitations': ['The parent-drop threshold is an uncalibrated placeholder chosen for mechanics, not a '
                              'calibrated value; k = 2.0 is the design default, also uncalibrated.',
                              'Synthetic tiles test mechanics only. NDVI sigma from a dihedral ensemble is a sensitivity '
                              'proxy, not calibrated probability or accuracy.',
                              'CPU run of the real pinned model.']}
    try:
        pairs = real_pairs(s, root, tile)
    except Blocked as exc:
        return {'status': 'BLOCKED', 'evidence': 'real',
                'reason': f'{exc}. The keep/drop verdict stays BLOCKED here: it needs real 10 m tiles run through the real model '
                          'on the RTX 4050 machine.',
                'mechanics_synthetic': {**mechanics, 'note': 'mechanics only; NOT a verdict on 4 vs 8'},
                'measurements': {}, **common}
    real = evaluate(op, pairs, s, md, change, p0['scale'])
    return {'status': 'PASS' if real['would_keep_4'] else 'FAIL', 'evidence': 'real',
            'measurements': {'real': real}, 'mechanics_synthetic': mechanics, **common}


if __name__ == '__main__':
    run_cli('e8', probe)
