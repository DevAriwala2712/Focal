"""E2: tile invariance. Same image, tile grid offset by 0 and 32 px; measure what changes and where."""
from __future__ import annotations

import numpy as np

from experiments.common import ndvi, run_cli, synthetic_scene
from experiments.e1_tile_scheduler import model_operator, super_resolve_tiled, tile_origins


def coverage_count(size: int, origins, tile: int) -> np.ndarray:
    counts = np.zeros(size, dtype=np.int32)
    for o in origins:
        counts[o:o + tile] += 1
    return counts


def seam_lines(size: int, origins, tile: int, scale: int) -> list[int]:
    """Output-pixel lines where a tile starts or ends inside the image (a hard cut steps here)."""
    lines = {o * scale for o in origins if o > 0} | {(o + tile) * scale for o in origins if o + tile < size}
    return sorted(lines)


def _adjacent_pairs(tiles: dict):
    """Yield (tile_a, tile_b, axis) for neighbouring tiles that share a row (axis 2) or a column (axis 1)."""
    rows = sorted({r for r, _ in tiles})
    cols = sorted({c for _, c in tiles})
    for r in rows:
        for c0, c1 in zip(cols, cols[1:]):
            yield (r, c0), (r, c1), 2
    for c in cols:
        for r0, r1 in zip(rows, rows[1:]):
            yield (r0, c), (r1, c), 1


def pair_disagreement(tiles: dict, tile: int, scale: int, edges) -> list[dict]:
    """Raw |A-B| between adjacent overlapping tiles (before blending), binned by input-px distance to the
    nearer of the two tile edges. Horizontal and vertical neighbours are pooled."""
    errs, dists = [], []
    for ta, tb, axis in _adjacent_pairs(tiles):
        pa, pb = ta[axis - 1], tb[axis - 1]              # start of each tile along the shared axis
        overlap = pa + tile - pb
        if overlap <= 0:
            continue
        a, b = tiles[ta], tiles[tb]
        n = overlap * scale
        cut = (slice(None),) * axis
        aa = a[cut + (slice((pb - pa) * scale, None),)]
        bb = b[cut + (slice(0, n),)]
        j = np.arange(n)
        d = np.minimum(j // scale, (n - 1 - j) // scale)
        diff = np.abs(aa.astype(np.float64) - bb.astype(np.float64))
        errs.append(diff.ravel())
        shape = [1, 1, 1]
        shape[axis] = n
        dists.append(np.broadcast_to(d.reshape(shape), diff.shape).ravel())
    err, dist = np.concatenate(errs), np.concatenate(dists)
    rows = []
    for lo, hi in zip(edges, list(edges[1:]) + [None]):
        m = (dist >= lo) if hi is None else ((dist >= lo) & (dist < hi))
        vals = err[m]
        rows.append({'bin': f'[{lo},{"inf" if hi is None else hi})', 'count': int(vals.size),
                     'max_abs': float(vals.max()) if vals.size else None,
                     'p99_abs': float(np.percentile(vals, 99)) if vals.size else None})
    return rows


def _owner(size: int, origins, tile: int) -> tuple[np.ndarray, np.ndarray]:
    """(coverage count, owning tile origin or -1 where the pixel is not covered exactly once)."""
    counts = coverage_count(size, origins, tile)
    owner = np.full(size, -1, dtype=np.int64)
    for o in origins:
        owner[o:o + tile] = np.where(counts[o:o + tile] == 1, o, owner[o:o + tile])
    return counts, owner


def zone_masks(size: int, origins_a, origins_b, tile: int, scale: int) -> dict:
    """Exact three-way partition of the (square) output into:

    interior_different_tile: one tile in BOTH runs, but a different tile each run (informative interior)
    interior_same_tile: one tile in both runs and the same tile (identical computation; sanity only)
    seam: any pixel where at least one run blends two or more tiles
    """
    ca, oa = _owner(size, origins_a, tile)
    cb, ob = _owner(size, origins_b, tile)
    single = ((ca[:, None] * ca[None, :]) == 1) & ((cb[:, None] * cb[None, :]) == 1)
    differ = (oa != ob)[:, None] | (oa != ob)[None, :]
    up = lambda m: np.repeat(np.repeat(m, scale, axis=0), scale, axis=1)
    return {'interior_different_tile': up(single & differ), 'interior_same_tile': up(single & ~differ),
            'seam': up(~single)}


def delta_stats(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> dict:
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))[:, mask]
    if d.size == 0:
        return {'count': 0, 'max_abs': None, 'p99_abs': None, 'mean_abs': None}
    return {'count': int(d.size), 'max_abs': float(d.max()), 'p99_abs': float(np.percentile(d, 99)),
            'mean_abs': float(d.mean())}


def seam_score(array: np.ndarray, row_lines, col_lines, scale: int = 1) -> dict:
    """Mean |adjacent difference| across seam lines vs across non-seam pairs of the SAME sub-pixel phase.

    The x4 model output has 4x4 block structure, so gradients depend on boundary index mod `scale`;
    comparing against all pairs would flag every block boundary as a seam. ratio ~ 1: no visible seam.
    """
    seam_g, expected_g = [], []
    for axis, lines in ((1, row_lines), (2, col_lines)):
        if not lines:                                # like with like: only axes that have seams
            continue
        g = np.abs(np.diff(array.astype(np.float64), axis=axis)).mean(axis=(0, 2 if axis == 1 else 1))
        boundary = np.arange(1, g.size + 1)          # g[i] is the pair across boundary i+1
        is_seam = np.isin(boundary, lines)
        for x in lines:
            same_phase = (boundary % scale == x % scale) & ~is_seam
            seam_g.append(g[x - 1])
            expected_g.append(g[same_phase].mean())
    if not seam_g:
        return {'seam_mean': None, 'elsewhere_mean': None, 'ratio': None, 'seam_lines': 0}
    return {'seam_mean': float(np.mean(seam_g)), 'elsewhere_mean': float(np.mean(expected_g)),
            'ratio': float(np.sum(seam_g) / np.sum(expected_g)), 'seam_lines': len(seam_g)}


def biased_operator(operator, delta: float, period: int = 2):
    """Known-seam control: add delta*(tile_number % period) so real discontinuities exist to detect."""
    state = {'n': 0}
    def wrapped(batch):
        out = operator(batch).copy()
        for i in range(out.shape[0]):
            out[i] += delta * ((state['n'] + i) % period)
        state['n'] += out.shape[0]
        return out
    return wrapped


def probe(cfg, root):
    """E2: tile-grid offset 0 vs 32 with the real pinned model on CPU."""
    import torch
    from risk.model import load_model
    s, s1, p0 = cfg['e2'], cfg['e1'], cfg['phase0']['model']
    torch.manual_seed(cfg['seed'])
    op = model_operator(load_model(cfg['phase0'], root, device='cpu').eval())
    scene = synthetic_scene(s['aoi_px'], cfg['seed'])
    common = dict(tile=s1['tile'], stride=s1['stride'], scale=p0['scale'])
    scale = p0['scale']
    runs = {f'offset_{o}': super_resolve_tiled(scene, None, op, feather=s1['feather'], offset=o,
                                               keep_tiles=True, **common)
            for o in s['offsets']}
    names = list(runs)
    a, b = runs[names[0]], runs[names[1]]
    zones = zone_masks(s['aoi_px'], sorted({r for r, _ in a.origins}), sorted({r for r, _ in b.origins}),
                       s1['tile'], scale)
    md = cfg['ndvi']['min_denominator']
    na, nb = ndvi(a.array, md)[None], ndvi(b.array, md)[None]
    valid = np.isfinite(na[0]) & np.isfinite(nb[0])

    size = s['aoi_px']

    def axes(run):
        return sorted({r for r, _ in run.origins}), sorted({c for _, c in run.origins})

    def seams(run):
        rl, cl = axes(run)
        return seam_score(run.array, seam_lines(size, rl, s1['tile'], scale),
                          seam_lines(size, cl, s1['tile'], scale), scale)


    scores = {n: seams(r) for n, r in runs.items()}
    controls = {}
    for label, feather in (('known_seam_feathered', s1['feather']), ('known_seam_hard_cut', s['hard_cut_feather'])):
        controls[label] = seams(super_resolve_tiled(scene, None, biased_operator(op, s['control_bias'], s['control_period']),
                                                    feather=feather, offset=s['offsets'][0], **common))
    tol = s['interior_max_tolerance']
    informative = delta_stats(a.array, b.array, zones['interior_different_tile'])
    ok = informative['count'] > 0 and informative['max_abs'] <= tol
    return {'status': 'PASS' if ok else 'FAIL', 'evidence': 'synthetic',
            'criteria': {'interior_max_abs_tolerance_reflectance': tol, 'applies_to': 'interior_different_tile',
                         'tolerance_rationale': s['tolerance_rationale'], 'offsets_px': s['offsets'],
                         'aoi_px': s['aoi_px']},
            'measurements': {
                'interior_different_tile_reflectance': informative,
                'interior_same_tile_reflectance_sanity': delta_stats(a.array, b.array, zones['interior_same_tile']),
                'seam_reflectance': delta_stats(a.array, b.array, zones['seam']),
                'interior_different_tile_ndvi': delta_stats(na, nb, zones['interior_different_tile'] & valid),
                'seam_ndvi': delta_stats(na, nb, zones['seam'] & valid),
                'zone_fractions': {k: float(v.mean()) for k, v in zones.items()},
                'seam_score_ratio': scores,
                'seam_score_known_seam_controls': {'bias_reflectance': s['control_bias'],
                                                   'bias_period': s['control_period'], **controls},
                'raw_tile_pair_disagreement_by_edge_distance_input_px': {
                    n: pair_disagreement(r.tiles, s1['tile'], scale, s['edge_distance_bins']) for n, r in runs.items()},
                'tiles': {n: len(r.origins) for n, r in runs.items()}},
            'limitations': ['Synthetic scene; the numbers characterise this model on this scene, not real imagery.',
                            'A learned CNN plus a global Fourier constraint is not tile-invariant; nonzero deltas are expected.',
                            'interior_different_tile is a small fraction of the image (see zone_fractions); the '
                            'interior_same_tile check exists because pixels computed by an identical tile in both runs '
                            'would otherwise make the interior look perfect.',
                            'The seam score is a same-phase gradient ratio, not a perceptual or accuracy metric; the '
                            'known-seam controls show it detects an injected step (and that feathering softens it).',
                            'CPU only.']}


if __name__ == '__main__':
    run_cli('e2', probe)
