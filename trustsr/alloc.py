"""Sub-pixel allocation and boundary-accuracy metrics for A8 (F5).

A8 asks one question: when a 10 m observation says "this 10 m block is 40% non-vegetation", does ranking the 16
sub-pixels by a super-resolved index put the non-vegetation in the right 2.5 m places, better than ranking them by a
naive bilinear upsample?

The pieces here are deliberately generic ("target class" = the class being mapped, whose score is LOW):

* `unmix_fraction_static` - the single-observation analogue of `trustsr.gate_v2.unmix_fraction`. gate_v2's version is a
  pre/post CHANGE estimator (it takes a stack of pre dates and a post date and returns a vegetation-LOSS fraction);
  A8 has no event and one date, so the same projection onto the two-endmember line is exposed for a single observation
  rather than duplicated inside the change estimator. `test_alloc.py` checks the two agree on a matched case.
* `allocate_by_rank` - the allocation rule shared by every `alloc_*` method: within a 10 m block, take the k sub-pixels
  with the lowest score, where k = round(16 * fraction). The block sum is respected exactly (up to the rounding of k),
  so the methods differ ONLY in the ranking, which is the thing under test.
* metrics - IoU, boundary F1 at a pixel tolerance, signed/absolute area error.
* `component_width_m` - feature-width stratification, so "does SR ranking help more on narrow features?" is answerable.

Conventions
-----------
* Masks are 2-D bool on the 2.5 m grid; `block` = 4 (10 m / 2.5 m).
* `score` is "lower = more likely to be the target class". Non-vegetation -> score = NDVI. Water -> score = -NDWI.
* Non-finite scores are ranked last (never allocated).
* Every metric takes an explicit `valid` mask and every denominator is returned alongside the value.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

BLOCK = 4
WIDTH_BIN_EDGES_M = (0.0, 20.0, 50.0, 150.0, float('inf'))
WIDTH_BIN_NAMES = ('0-20', '20-50', '50-150', '150+')


# ---------------------------------------------------------------- indices

def normalized_difference(refl, a: int, b: int, min_denominator: float = 1e-6) -> np.ndarray:
    """(band a - band b) / (band a + band b) for a (C,H,W) reflectance cube; NaN where the denominator is tiny."""
    refl = np.asarray(refl, dtype=np.float64)
    num, den = refl[a] - refl[b], refl[a] + refl[b]
    out = np.full(num.shape, np.nan)
    np.divide(num, den, out=out, where=np.abs(den) > min_denominator)
    return out


def ndvi_rgbn(refl, min_denominator: float = 1e-6) -> np.ndarray:
    """NDVI for the SEN2SR-lite band order [B04, B03, B02, B08]: (B08 - B04) / (B08 + B04)."""
    return normalized_difference(refl, 3, 0, min_denominator)


def ndwi_rgbn(refl, min_denominator: float = 1e-6) -> np.ndarray:
    """McFeeters NDWI for band order [B04, B03, B02, B08]: (B03 - B08) / (B03 + B08)."""
    return normalized_difference(refl, 1, 3, min_denominator)


# ---------------------------------------------------------------- block reductions

def _blocks(a, block: int) -> np.ndarray:
    """(H,W) -> (H/block, W/block, block*block), sub-pixels of one 10 m block on the last axis."""
    a = np.asarray(a)
    h, w = a.shape[-2:]
    if h % block or w % block:
        raise ValueError(f'{h}x{w} is not a whole number of {block}x{block} blocks')
    return a.reshape(h // block, block, w // block, block).transpose(0, 2, 1, 3).reshape(h // block, w // block, block * block)


def _unblocks(a, block: int) -> np.ndarray:
    """Inverse of `_blocks`."""
    hb, wb, n = a.shape
    if n != block * block:
        raise ValueError(f'last axis {n} != {block * block}')
    return a.reshape(hb, wb, block, block).transpose(0, 2, 1, 3).reshape(hb * block, wb * block)


def block_fraction_oracle(truth, block: int = BLOCK) -> np.ndarray:
    """Exact target fraction per 10 m block, read off the HR truth mask. Isolates ranking quality."""
    return _blocks(np.asarray(truth, bool).astype(np.float64), block).mean(axis=-1)


def upsample(a, block: int = BLOCK) -> np.ndarray:
    """Replicate each 10 m value into block x block sub-pixels (same rule as wayanad_evidence.gate.upsample)."""
    return np.repeat(np.repeat(np.asarray(a), block, axis=-2), block, axis=-1)


# ---------------------------------------------------------------- fractions from the 10 m observation

def unmix_fraction_static(y, e_target, e_other, band_indices=(0, 3)) -> tuple[np.ndarray, np.ndarray]:
    """Two-endmember abundance of `e_target` for a single observation, by projection onto the endmember line.

    Single-date analogue of `trustsr.gate_v2.unmix_fraction`, which solves the same projection for a pre/post change.

    Args:
        y: (C,H,W) reflectance.
        e_target, e_other: (C,) endmember reflectance. The returned fraction is the abundance of `e_target`.
        band_indices: the two bands the projection uses.

    Returns:
        (fraction in [0,1], residual sum of squares off the endmember line).
    """
    y = np.asarray(y, dtype=np.float64)
    e_target, e_other = np.asarray(e_target, np.float64), np.asarray(e_other, np.float64)
    bi = np.asarray(band_indices, dtype=int)
    if bi.size != 2:
        raise ValueError('band_indices must name exactly two bands')
    if e_target.shape != (y.shape[0],) or e_other.shape != (y.shape[0],):
        raise ValueError('endmembers must be 1-D with one entry per band')
    d = e_target[bi] - e_other[bi]
    dd = float(np.dot(d, d))
    if dd <= 0:
        raise ValueError('endmembers are identical in the selected bands; no mixing direction')
    dy = np.moveaxis(y[bi], 0, -1) - e_other[bi]                      # (H,W,2)
    proj = dy @ d / dd
    resid = ((dy - proj[..., None] * d) ** 2).sum(axis=-1)
    return np.clip(proj, 0.0, 1.0), resid


def endmembers_from_score(refl, score, valid=None, quantile: float = 0.10):
    """(e_target, e_other) = mean reflectance of the lowest- and highest-`quantile` score pixels.

    Uses only the 10 m observation, never the HR truth, so it is what the deployed product could compute.
    """
    refl = np.asarray(refl, dtype=np.float64)
    score = np.asarray(score, dtype=np.float64)
    ok = np.isfinite(score) & np.isfinite(refl).all(axis=0)
    if valid is not None:
        ok &= np.asarray(valid, bool)
    if ok.sum() < 8:
        raise ValueError('too few valid pixels to estimate endmembers')
    s = score[ok]
    lo, hi = np.quantile(s, quantile), np.quantile(s, 1.0 - quantile)
    a, b = ok & (score <= lo), ok & (score >= hi)
    if not a.any() or not b.any():
        raise ValueError('degenerate score distribution; no endmember pixels')
    return refl[:, a].mean(axis=1), refl[:, b].mean(axis=1)


def block_fraction_realistic(refl10, score10, valid10=None, band_indices=(0, 3), quantile: float = 0.10):
    """Target fraction per 10 m block estimated by unmixing the 10 m observation itself (no HR truth used).

    Returns (fraction (h,w), diagnostics dict).
    """
    e_t, e_o = endmembers_from_score(refl10, score10, valid10, quantile)
    frac, resid = unmix_fraction_static(refl10, e_t, e_o, band_indices)
    return frac, {'endmember_target': e_t.tolist(), 'endmember_other': e_o.tolist(),
                  'mean_residual': float(np.nanmean(resid)), 'quantile': quantile,
                  'band_indices': list(map(int, band_indices))}


# ---------------------------------------------------------------- the methods

def allocate_by_rank(score, fraction, block: int = BLOCK) -> np.ndarray:
    """Within each 10 m block, mark the k sub-pixels with the LOWEST score, k = round(block^2 * fraction).

    `score` is on the 2.5 m grid, `fraction` on the 10 m grid. Non-finite scores rank last. Ties break by raster order
    (stable sort), which is deterministic and identical for every method.
    """
    score = np.asarray(score, dtype=np.float64)
    fraction = np.asarray(fraction, dtype=np.float64)
    h, w = score.shape
    if fraction.shape != (h // block, w // block):
        raise ValueError(f'fraction {fraction.shape} does not match {score.shape} at block {block}')
    s = _blocks(np.where(np.isfinite(score), score, np.inf), block)
    order = np.argsort(s, axis=-1, kind='stable')
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.broadcast_to(np.arange(block * block), order.shape), axis=-1)
    k = np.clip(np.rint(np.nan_to_num(fraction, nan=0.0) * block * block), 0, block * block).astype(np.int64)
    return _unblocks(rank < k[..., None], block)


def blocky_mask(score10, threshold: float, block: int = BLOCK) -> np.ndarray:
    """The naive baseline: threshold at 10 m and replicate the block value to all 16 sub-pixels. No allocation."""
    s = np.asarray(score10, dtype=np.float64)
    return upsample(np.isfinite(s) & (s < threshold), block)


def hard_threshold_v1_style(score_sr, score10, threshold: float, block: int = BLOCK) -> np.ndarray:
    """v1's mapped class: the 2.5 m hard threshold on the SR index, AND the replicated 10 m parent.

    This is `experiments/wayanad_evidence/gate.py`'s OBSERVED = S & P with S the sub-pixel test and P the replicated
    10 m parent (`gate.upsample`). It uses no fraction and no block-sum constraint, which is exactly what A8 tests
    against the allocation methods.
    """
    s = np.asarray(score_sr, dtype=np.float64)
    return (np.isfinite(s) & (s < threshold)) & blocky_mask(score10, threshold, block)


# ---------------------------------------------------------------- metrics

def iou(pred, truth, valid) -> dict:
    p = np.asarray(pred, bool) & np.asarray(valid, bool)
    t = np.asarray(truth, bool) & np.asarray(valid, bool)
    inter, union = int((p & t).sum()), int((p | t).sum())
    return {'iou': (float(inter / union) if union else None), 'intersection': inter, 'union': union,
            'n_pred': int(p.sum()), 'n_truth': int(t.sum())}


def _inner_boundary(mask) -> np.ndarray:
    """Pixels of `mask` with at least one 4-neighbour outside it."""
    m = np.asarray(mask, bool)
    return m & ~ndimage.binary_erosion(m, structure=ndimage.generate_binary_structure(2, 1), border_value=1)


def boundary_f1(pred, truth, valid, tolerance_px: int) -> dict:
    """Boundary F1: a boundary pixel matches if a boundary pixel of the other mask lies within `tolerance_px`.

    Boundaries are extracted on the full masks, then restricted to `valid`, so the evaluation border does not create
    boundaries of its own. Returns None when both boundary sets are empty.
    """
    v = np.asarray(valid, bool)
    bp, bt = _inner_boundary(pred) & v, _inner_boundary(truth) & v
    if not bp.any() and not bt.any():
        return {'f1': None, 'precision': None, 'recall': None, 'n_pred_boundary': 0, 'n_truth_boundary': 0,
                'tolerance_px': int(tolerance_px)}
    if not bp.any() or not bt.any():
        return {'f1': 0.0, 'precision': 0.0, 'recall': 0.0, 'n_pred_boundary': int(bp.sum()),
                'n_truth_boundary': int(bt.sum()), 'tolerance_px': int(tolerance_px)}
    d_to_t = ndimage.distance_transform_edt(~bt)
    d_to_p = ndimage.distance_transform_edt(~bp)
    prec = float((d_to_t[bp] <= tolerance_px).mean())
    rec = float((d_to_p[bt] <= tolerance_px).mean())
    f1 = 0.0 if (prec + rec) == 0 else float(2 * prec * rec / (prec + rec))
    return {'f1': f1, 'precision': prec, 'recall': rec, 'n_pred_boundary': int(bp.sum()),
            'n_truth_boundary': int(bt.sum()), 'tolerance_px': int(tolerance_px)}


def area_error(pred, truth, valid) -> dict:
    """Signed and absolute relative area error of the mapped class. `None` relative error when the truth area is 0."""
    v = np.asarray(valid, bool)
    np_, nt = int((np.asarray(pred, bool) & v).sum()), int((np.asarray(truth, bool) & v).sum())
    rel = (float((np_ - nt) / nt) if nt else None)
    return {'pred_px': np_, 'truth_px': nt, 'signed_px': np_ - nt,
            'relative': rel, 'abs_relative': (None if rel is None else abs(rel))}


# ---------------------------------------------------------------- feature-width stratification

def component_width_m(mask, pixel_size_m: float, connectivity: int = 1):
    """Per-pixel local width of each connected component of `mask`, in metres.

    Width of a component = (2 * max Euclidean distance-to-background inside it - 1) * pixel_size_m, i.e. twice the
    largest inscribed radius. Exact for odd pixel widths and 1 px low for even ones; documented because the bin edges
    (20/50/150 m) sit far from that 2.5 m ambiguity.

    Returns (width_m (H,W), labels (H,W), per-component width array indexed by label-1).
    """
    m = np.asarray(mask, bool)
    labels, n = ndimage.label(m, structure=ndimage.generate_binary_structure(2, connectivity))
    width = np.zeros(m.shape, dtype=np.float64)
    if n == 0:
        return width, labels, np.zeros(0)
    edt = ndimage.distance_transform_edt(m)
    per = np.asarray(ndimage.maximum(edt, labels, index=np.arange(1, n + 1)), dtype=np.float64)
    per = (2.0 * per - 1.0) * float(pixel_size_m)
    width[m] = per[labels[m] - 1]
    return width, labels, per


def width_bin_masks(width_m, mask, edges=WIDTH_BIN_EDGES_M):
    """{bin_name: bool mask} splitting `mask` by the per-pixel component width. Bins are (lo, hi]."""
    w = np.asarray(width_m, dtype=np.float64)
    m = np.asarray(mask, bool)
    out = {}
    for i, name in enumerate(WIDTH_BIN_NAMES[:len(edges) - 1]):
        lo, hi = edges[i], edges[i + 1]
        out[name] = m & (w > lo) & (w <= hi)
    return out


def dilate(mask, radius_px: int) -> np.ndarray:
    return ndimage.binary_dilation(np.asarray(mask, bool), structure=ndimage.generate_binary_structure(2, 1),
                                   iterations=int(radius_px))


def stratum_iou(pred, truth_stratum, valid, radius_px: int = 4) -> dict:
    """IoU restricted to one width stratum.

    The evaluation region R is the stratum dilated by `radius_px` (4 px = 10 m, one parent block). Truth inside R is
    exactly the stratum, so IoU = |P n T_s| / |(P n R) u T_s|: predictions far away from these features neither help
    nor hurt, and a prediction that misses by up to one parent block still counts against the stratum.
    """
    v = np.asarray(valid, bool)
    t = np.asarray(truth_stratum, bool) & v
    if not t.any():
        return {'iou': None, 'intersection': 0, 'union': 0, 'n_truth': 0, 'radius_px': int(radius_px)}
    region = dilate(t, radius_px) & v
    p = np.asarray(pred, bool) & region
    inter, union = int((p & t).sum()), int((p | t).sum())
    return {'iou': (float(inter / union) if union else None), 'intersection': inter, 'union': union,
            'n_truth': int(t.sum()), 'radius_px': int(radius_px)}
