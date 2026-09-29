"""E5: parent-first cascade. Super-resolve only tiles whose 10 m parent fires (+ buffer + seeded audit sample)."""
from __future__ import annotations

import math

import numpy as np

from experiments.common import (CLASS_CODES, Blocked, classify_change, load_real_crop, parent_drop_mask, run_cli,
                                synthetic_change_pair)
from experiments.e1_tile_scheduler import model_operator, super_resolve_tiled, tile_origins
from experiments.e8_dihedral import dihedral_ndvi_stats


def select_tiles(origins, parent, tile: int, min_pixels: int, rings: int, audit_fraction: float, seed: int) -> dict:
    """Tiles to process: fired (parent pixels >= min_pixels) + Chebyshev buffer rings + seeded audit of the rest."""
    row_idx = {r: i for i, r in enumerate(sorted({r for r, _ in origins}))}
    col_idx = {c: i for i, c in enumerate(sorted({c for _, c in origins}))}
    by_index = {(row_idx[r], col_idx[c]): (r, c) for r, c in origins}
    fired = {(r, c) for r, c in origins if int(parent[r:r + tile, c:c + tile].sum()) >= min_pixels}
    ring = set()
    for r, c in fired:
        i, j = row_idx[r], col_idx[c]
        for di in range(-rings, rings + 1):
            for dj in range(-rings, rings + 1):
                near = by_index.get((i + di, j + dj))
                if near is not None and near not in fired:
                    ring.add(near)
    pool = sorted(set(origins) - fired - ring)
    n_audit = int(math.ceil(audit_fraction * len(pool) - 1e-12)) if audit_fraction > 0 else 0
    picked = np.random.default_rng(seed).choice(len(pool), size=n_audit, replace=False) if n_audit else []
    audit = {pool[i] for i in picked}
    processed = fired | ring | audit
    return {'fired': fired, 'ring': ring, 'audit': audit, 'processed': processed, 'skipped': set(origins) - processed}


def covered(size: int, origins, tile: int) -> np.ndarray:
    """Input-resolution mask of pixels covered by at least one of `origins`."""
    m = np.zeros((size, size), dtype=bool)
    for r, c in origins:
        m[r:r + tile, c:c + tile] = True
    return m


def fully_processed(size: int, all_origins, processed, tile: int) -> np.ndarray:
    """Pixels whose every covering tile was processed: their blend is arithmetically identical to a full run."""
    total = np.zeros((size, size), dtype=np.int32)
    done = np.zeros((size, size), dtype=np.int32)
    for r, c in all_origins:
        total[r:r + tile, c:c + tile] += 1
        if (r, c) in processed:
            done[r:r + tile, c:c + tile] += 1
    return (total > 0) & (done == total)


class EnsembleStats:
    """Per-tile operator: dihedral-ensemble NDVI [mean, variance, valid] at SR resolution; counts model passes."""

    def __init__(self, operator, transforms, min_denominator: float, ddof: int):
        self.operator, self.transforms, self.md, self.ddof = operator, list(transforms), min_denominator, ddof
        self.passes = 0

    def _counted(self, batch):
        self.passes += batch.shape[0]
        return self.operator(batch)

    def __call__(self, batch: np.ndarray) -> np.ndarray:
        out = []
        for tile in batch:
            mean, std, valid = dihedral_ndvi_stats(self._counted, tile, self.transforms, self.md, self.ddof)
            out.append(np.stack([np.nan_to_num(mean), np.nan_to_num(std) ** 2, valid.astype(np.float32)]))
        return np.stack(out).astype(np.float32)


def _classes(stats_pre, stats_post, parent_up, parent_valid_up, k: float) -> np.ndarray:
    (m0, v0, ok0), (m1, v1, ok1) = stats_pre, stats_post
    valid = np.isfinite(m0) & np.isfinite(m1) & (ok0 > 1 - 1e-6) & (ok1 > 1 - 1e-6) & parent_valid_up
    return classify_change(np.nan_to_num(m0 - m1), np.sqrt(np.nan_to_num(v0) + np.nan_to_num(v1)),
                           parent_up, ~valid, k)


def run_cascade(operator, pre, post, cfg: dict) -> dict:
    """Full run vs cascade on the same input. `operator` is the raw SR operator (batch -> SR batch)."""
    size, tile, scale = pre.shape[-1], cfg['tile'], cfg['scale']
    axis = tile_origins(size, tile, cfg['stride'])
    origins = [(r, c) for r in axis for c in axis]
    parent, parent_valid = parent_drop_mask(pre, post, cfg['parent_drop_threshold'], cfg['min_denominator'])
    up = lambda m: np.repeat(np.repeat(m, scale, 0), scale, 1)
    sel = select_tiles(origins, parent, tile, cfg['min_parent_pixels'], cfg['rings'], cfg['audit_fraction'], cfg['seed'])

    def run(subset):
        op = EnsembleStats(operator, cfg['transforms'], cfg['min_denominator'], cfg['ddof'])
        stats = [super_resolve_tiled(img, None, op, tile=tile, stride=cfg['stride'], scale=scale,
                                     feather=cfg['feather'], out_channels=3, origins=subset).array
                 for img in (pre, post)]
        return stats, op.passes
    full_stats, full_passes = run(origins)
    casc_stats, casc_passes = run([o for o in origins if o in sel['processed']])
    args = (up(parent), up(parent_valid), cfg['k'])
    cls_full, cls_casc = _classes(*full_stats, *args), _classes(*casc_stats, *args)

    fp = up(fully_processed(size, origins, sel['processed'], tile))
    touched = up(covered(size, sel['processed'], tile))
    partial = touched & ~fp
    delta = max(float(np.abs(c[:, fp] - f[:, fp]).max()) for c, f in zip(casc_stats, full_stats))
    unsupported = CLASS_CODES['UNSUPPORTED']
    no_data = CLASS_CODES['NO_DATA']
    # audit estimator: UNSUPPORTED rate in pixels seen ONLY through audit tiles (not fired/ring), fully processed
    audit_only = fp & ~up(covered(size, sel['fired'] | sel['ring'], tile))
    audit_valid = audit_only & (cls_casc != no_data)
    skipped_only = ~touched
    audit_rate = float((cls_casc[audit_valid] == unsupported).mean()) if audit_valid.any() else None
    truth_valid = skipped_only & (cls_full != no_data)
    return {
        'tiles': {'total': len(origins), 'fired': len(sel['fired']), 'ring': len(sel['ring']),
                  'audit': len(sel['audit']), 'skipped': len(sel['skipped'])},
        'tiles_skipped_fraction': len(sel['skipped']) / len(origins),
        'forward_passes': {'full': full_passes, 'cascade': casc_passes, 'ratio': casc_passes / full_passes},
        'identical_on_fully_processed': {
            'pixels': int(fp.sum()), 'max_abs_stat_delta': delta,
            'class_agreement': float((cls_casc[fp] == cls_full[fp]).mean()) if fp.any() else None},
        'partially_processed_boundary': {
            'pixels': int(partial.sum()),
            'class_agreement': float((cls_casc[partial] == cls_full[partial]).mean()) if partial.any() else None},
        'audit_unsupported': {
            'audit_only_valid_pixels': int(audit_valid.sum()),
            'unsupported_pixels_found': int((cls_casc[audit_valid] == unsupported).sum()),
            'unsupported_rate_estimate_for_skipped': audit_rate,
            'skipped_only_pixels': int(skipped_only.sum()),
            'skipped_only_parent_valid_pixels': int((skipped_only & up(parent_valid)).sum()),
            'estimated_unsupported_pixels_in_skipped':
                None if audit_rate is None else audit_rate * int((skipped_only & up(parent_valid)).sum()),
            # available only because this is a synthetic mechanics run with a full-run reference
            'reference_full_run_unsupported_rate_in_skipped':
                float((cls_full[truth_valid] == unsupported).mean()) if truth_valid.any() else None,
            'reference_full_run_unsupported_pixels_in_skipped': int((cls_full[skipped_only] == unsupported).sum()),
            'reference_full_run_valid_pixels_in_skipped': int(truth_valid.sum())},
        'class_counts_full_on_fully_processed': {n: int((cls_full[fp] == c).sum()) for n, c in CLASS_CODES.items()},
        'selection': sel}


def _jsonable(res: dict) -> dict:
    out = dict(res)
    out['selection'] = {k: sorted([list(t) for t in v]) for k, v in res['selection'].items()}
    return out


def probe(cfg, root):
    """E5: parent-first cascade vs full run; real pinned model on CPU, synthetic changed square."""
    import torch
    from risk.model import load_model
    s, s1, p0, ch = cfg['e5'], cfg['e1'], cfg['phase0']['model'], cfg['change']
    torch.manual_seed(cfg['seed'])
    op = model_operator(load_model(cfg['phase0'], root, device='cpu').eval())
    flat = {'tile': s1['tile'], 'stride': s1['stride'], 'scale': p0['scale'], 'feather': s1['feather'],
            'transforms': list(range(s['transforms'])), 'ddof': s['ddof'], 'rings': s['rings'],
            'min_parent_pixels': s['min_parent_pixels'], 'audit_fraction': s['audit_fraction'], 'seed': cfg['seed'],
            'k': ch['k'], 'parent_drop_threshold': ch['parent_drop_threshold'],
            'min_denominator': s['min_denominator']}
    pre, post, truth = synthetic_change_pair(s['aoi_px'], cfg['seed'], s['square'], s['noise'])
    res = run_cascade(op, pre, post, flat)
    fp = res['identical_on_fully_processed']
    ok = fp['class_agreement'] == 1.0 and fp['max_abs_stat_delta'] == 0.0 and \
        res['forward_passes']['cascade'] < res['forward_passes']['full']
    parts = {'synthetic_mechanics': 'PASS' if ok else 'FAIL'}
    try:
        load_real_crop(s['real_pre'], root)
        load_real_crop(s['real_post'], root)
        parts['wayanad_real'] = 'NOT_IMPLEMENTED_FOR_PRESENT_DATA'
    except Blocked as exc:
        parts['wayanad_real'] = 'BLOCKED'
        wayanad = str(exc)
    return {'status': 'PASS' if ok else 'FAIL', 'evidence': 'synthetic', 'evidence_parts': parts,
            'criteria': {'rule': 'cascade == full on fully processed pixels (bitwise stats, identical classes) and '
                                 'fewer forward passes', 'transforms': s['transforms'], 'rings': s['rings'],
                         'audit_fraction': s['audit_fraction'], 'min_parent_pixels': s['min_parent_pixels'],
                         'parent_drop_threshold_uncalibrated_placeholder': ch['parent_drop_threshold'], 'k': ch['k']},
            'measurements': _jsonable(res),
            'real_wayanad': {'status': parts['wayanad_real'], 'reason': wayanad if 'wayanad' in dir() else None},
            'limitations': ['UNSUPPORTED here is dominated by synthetic date-to-date noise exceeding the dihedral sigma '
                            '(sigma measures model sensitivity, not radiometric noise), NOT by SR hallucination. The '
                            'high rate characterises this synthetic set-up only.',
                            'Synthetic scene with one known changed square: mechanics only. The skipped fraction and '
                            'UNSUPPORTED rate here say nothing about real Wayanad imagery.',
                            'Pixels covered by both a processed and a skipped tile are blended from fewer tiles than in a '
                            'full run; they are reported separately (partially_processed_boundary), not claimed identical.',
                            'The UNSUPPORTED estimate uses few audited tiles and spatially correlated pixels; it has no '
                            'confidence interval here. The full-run reference exists only because this is synthetic.',
                            'Parent threshold and k are uncalibrated placeholders; the cascade never skips a tile whose '
                            'change the parent mask cannot see, so its recall is bounded by the 10 m parent mask.']
                           + (['Real Wayanad part not run: ' + wayanad] if 'wayanad' in dir() else [])}


if __name__ == '__main__':
    run_cli('e5', probe)
