"""Grid geometry, tile-grid snapping, dilation and connected components (numpy only)."""
from __future__ import annotations

import numpy as np
from affine import Affine


def reference_grid(aoi: dict):
    """Snapped 10 m grid over a square AOI: (Affine, (rows, cols)). Origin lies on a `snap_m` lattice."""
    from pyproj import Transformer
    x, y = Transformer.from_crs('EPSG:4326', aoi['crs'], always_xy=True).transform(aoi['longitude'], aoi['latitude'])
    half, snap = aoi['size_m'] / 2, aoi['snap_m']
    side = aoi['size_m'] / 10.0
    if not side.is_integer():
        raise ValueError('AOI size must be a whole number of 10 m pixels')
    x0 = round((x - half) / snap) * snap
    y0 = round((y + half) / snap) * snap
    return Affine(10.0, 0.0, float(x0), 0.0, -10.0, float(y0)), (int(side), int(side))


def sr_grid(transform: Affine, shape, scale: int):
    """Output grid of a x`scale` SR: transform * scale(1/scale), dimensions x scale, identical bounds."""
    return transform * Affine.scale(1 / scale), (shape[0] * scale, shape[1] * scale)


def bounds(transform: Affine, shape):
    """(left, bottom, right, top) for a north-up grid."""
    return (transform.c, transform.f + transform.e * shape[0], transform.c + transform.a * shape[1], transform.f)


def snap_crop(bbox, grid_shape, tile: int):
    """Snap (r0, r1, c0, c1) (exclusive ends) outward to the `tile` lattice anchored at the grid origin, clamped."""
    r0, r1, c0, c1 = bbox
    return (max(0, r0 // tile * tile), min(grid_shape[0], -(-r1 // tile) * tile),
            max(0, c0 // tile * tile), min(grid_shape[1], -(-c1 // tile) * tile))


def dilate_square(mask: np.ndarray, radius: int) -> np.ndarray:
    """Chebyshev (square) dilation by `radius` px, clipped at the border, no wrap-around."""
    out = np.asarray(mask, bool)
    if radius <= 0:
        return out.copy()
    for axis in (0, 1):
        n = out.shape[axis]
        pad = [(0, 0), (0, 0)]
        pad[axis] = (radius + 1, radius)
        c = np.cumsum(np.pad(out.astype(np.int32), pad), axis=axis)
        hi = np.take(c, np.arange(n) + 2 * radius + 1, axis=axis)
        lo = np.take(c, np.arange(n), axis=axis)
        out = (hi - lo) > 0
    return out


def label_components(mask: np.ndarray, connectivity: int = 8):
    """Connected components by row runs + union-find: (labels 1..n, n). No scipy dependency."""
    if connectivity not in (4, 8):
        raise ValueError('connectivity must be 4 or 8')
    mask = np.asarray(mask, bool)
    h, w = mask.shape
    d = np.diff(np.pad(mask, ((0, 0), (1, 1))).astype(np.int8), axis=1)
    sr, sc = np.nonzero(d == 1)
    _, ec = np.nonzero(d == -1)
    n = len(sr)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    row_start = np.searchsorted(sr, np.arange(h + 1))
    slack = 1 if connectivity == 8 else 0
    for r in range(h - 1):
        a0, a1, b0, b1 = row_start[r], row_start[r + 1], row_start[r + 1], row_start[r + 2]
        i, j = a0, b0
        while i < a1 and j < b1:
            if sc[i] < ec[j] + slack and sc[j] < ec[i] + slack:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)
            if ec[i] <= ec[j]:
                i += 1
            else:
                j += 1
    roots = np.array([find(i) for i in range(n)], dtype=np.int64)
    _, ids = np.unique(roots, return_inverse=True)
    labels = np.zeros((h, w), dtype=np.int32)
    for k in range(n):
        labels[sr[k], sc[k]:ec[k]] = ids[k] + 1
    return labels, (int(ids.max()) + 1 if n else 0)
