"""E6: season-matched pre-event dates (REAL data only): all clear pre dates vs three January dates.

Stable pixels are defined BEFORE any result is looked at (see docstring of stable_mask and configs/experiments.yaml):
valid on >= min_valid of the pre dates, NDVI std across those dates <= max_std, and outside every suspected-slide
disk. False-drop fraction = share of stable comparable pixels with (pooled pre NDVI - post NDVI) > the parent
threshold. Work is at 10 m; no super-resolution is involved.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import warnings
from pathlib import Path

import numpy as np

from experiments.common import Blocked, load_experiment_config, ndvi_bands, run_cli


def reflectance(dn, scale: float, offset: float) -> np.ndarray:
    """ESA L2A: rho = (DN + BOA_ADD_OFFSET) / QUANTIFICATION. Offset is -1000 only for baseline >= 04.00."""
    return ((np.asarray(dn, dtype=np.float64) + offset) / scale).astype(np.float32)


def baseline_offset_ok(baseline: str, minimum: str) -> bool:
    key = lambda v: tuple(int(x) for x in v.split('.'))
    return key(baseline) >= key(minimum)


def valid_from_scl(scl, cloud, shadow, invalid) -> np.ndarray:
    return ~np.isin(scl, list(cloud) + list(shadow) + list(invalid))


def stable_mask(stack: np.ndarray, suspected: np.ndarray, min_valid: int, max_std: float) -> np.ndarray:
    """Stable = >= min_valid valid dates, NDVI std over valid pre dates <= max_std, outside suspected areas."""
    valid_count = np.isfinite(stack).sum(axis=0)
    with warnings.catch_warnings(), np.errstate(invalid='ignore'):
        warnings.simplefilter('ignore', RuntimeWarning)          # all-NaN pixels are expected
        std = np.nanstd(stack, axis=0)
    return (valid_count >= min_valid) & np.isfinite(std) & (std <= max_std) & ~suspected


def pooled_mean(stack: np.ndarray, indices, min_valid: int) -> np.ndarray:
    sub = stack[list(indices)]
    count = np.isfinite(sub).sum(axis=0)
    with warnings.catch_warnings(), np.errstate(invalid='ignore'):
        warnings.simplefilter('ignore', RuntimeWarning)
        mean = np.nanmean(sub, axis=0)
    return np.where(count >= min_valid, mean, np.nan).astype(np.float32)


def false_drop(pre_a, pre_b, post, stable, threshold: float) -> dict:
    """Fraction of stable pixels flagged as a drop under pool A and pool B, on ONE shared comparable pixel set."""
    comparable = stable & np.isfinite(pre_a) & np.isfinite(pre_b) & np.isfinite(post)
    flags_a = comparable & (np.nan_to_num(pre_a) - np.nan_to_num(post) > threshold)
    flags_b = comparable & (np.nan_to_num(pre_b) - np.nan_to_num(post) > threshold)
    n = int(comparable.sum())
    return {'comparable_pixels': n, 'flags_a': flags_a, 'flags_b': flags_b, 'comparable': comparable,
            'fraction_a': float(flags_a.sum() / n) if n else None, 'fraction_b': float(flags_b.sum() / n) if n else None}


def block_bootstrap_diff(flags_a, flags_b, comparable, block: int, n: int, seed: int, level: float = 95.0):
    """CI of fraction_a - fraction_b, resampling spatial blocks (pixels are strongly spatially correlated)."""
    h, w = comparable.shape
    rows = range(0, h, block)
    cols = range(0, w, block)
    stats = np.array([[comparable[r:r + block, c:c + block].sum(), flags_a[r:r + block, c:c + block].sum(),
                       flags_b[r:r + block, c:c + block].sum()] for r in rows for c in cols], dtype=np.float64)
    stats = stats[stats[:, 0] > 0]
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        pick = stats[rng.integers(0, len(stats), len(stats))]
        total = pick[:, 0].sum()
        diffs.append((pick[:, 1].sum() - pick[:, 2].sum()) / total)
    tail = (100 - level) / 2
    return float(np.percentile(diffs, tail)), float(np.percentile(diffs, 100 - tail))


def disk_mask(transform, shape, centre_xy, radius_m: float) -> np.ndarray:
    rows, cols = np.indices(shape)
    x, y = transform * (cols + 0.5, rows + 0.5)
    return (x - centre_xy[0]) ** 2 + (y - centre_xy[1]) ** 2 <= radius_m ** 2


# ---- byte budget ------------------------------------------------------------------------------------

def parse_interface_ibytes(netstat_text: str, interface: str) -> int:
    """Received bytes of `interface` from `netstat -ib` output (its <Link#> row; Ibytes is 5th from the end)."""
    for line in netstat_text.splitlines():
        tokens = line.split()
        if len(tokens) >= 8 and tokens[0] == interface and tokens[2].startswith('<Link#'):
            return int(tokens[-5])
    raise ValueError(f'no <Link#> row for interface {interface}')


def default_interface_ibytes() -> int:
    route = subprocess.run(['route', '-n', 'get', 'default'], capture_output=True, text=True, check=True).stdout
    interface = next(l.split(':')[1].strip() for l in route.splitlines() if l.strip().startswith('interface:'))
    text = subprocess.run(['netstat', '-ib'], capture_output=True, text=True, check=True).stdout
    return parse_interface_ibytes(text, interface)


class ByteBudget:
    """Bytes received on the default interface since construction. System-wide, so an UPPER BOUND on this
    process's traffic (any other traffic on the machine counts too). GDAL exposes no per-read byte counter."""

    def __init__(self, cap_bytes: int, counter=default_interface_ibytes):
        try:
            self.counter, self.start = counter, counter()
        except Exception as exc:
            raise Blocked(f'cannot count network bytes ({exc}); refusing to download without a byte counter',
                          evidence='real') from exc
        self.cap_bytes, self.peak = cap_bytes, 0

    def used(self) -> int:
        self.peak = max(self.peak, self.counter() - self.start)
        return self.peak

    def check(self) -> None:
        if self.used() > self.cap_bytes:
            raise RuntimeError(f'network byte cap exceeded: {self.peak} > {self.cap_bytes} bytes; stopped fetching')

    def log(self) -> dict:
        return {'cap_bytes': self.cap_bytes, 'bytes_received_upper_bound': self.peak,
                'method': 'default-interface received-bytes delta (netstat -ib); includes any other traffic'}


# ---- cache + (explicit, opt-in) fetch --------------------------------------------------------

def cache_path(cache: Path, date: str) -> Path:
    return Path(cache) / f'{date}.npz'


def load_cache(cache: Path, dates) -> dict:
    missing = [d for d in dates if not cache_path(cache, d).is_file()]
    if missing:
        raise Blocked(f'real 10 m imagery not cached: {len(missing)} of {len(dates)} per-date files missing under '
                      f'{cache} (first: {cache_path(cache, missing[0]).name}; last: {cache_path(cache, missing[-1]).name}). '
                      'Each needs B04/B08 digital numbers and SCL on the R2 AOI grid at 10 m; '
                      'run `python -m experiments.e6_season_matched --fetch` once network use is approved.',
                      evidence='real')
    out = {}
    for d in dates:
        with np.load(cache_path(cache, d)) as z:
            out[d] = {k: z[k] for k in ('red', 'nir', 'scl')} | {'meta': json.loads(str(z['meta']))}
    return out


def fetch(cfg, root, dates) -> dict:
    """Windowed reads of B04/B08/SCL from Planetary Computer onto the R2 AOI grid at 10 m. Network; opt-in only."""
    import planetary_computer as pc
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT
    from risk.common import retry
    from risk.r2_imagery import aoi_grid, read_scl
    im, policy, s6 = cfg['phase0']['imagery'], cfg['phase0']['network'], cfg['e6']
    polygon, transform, shape = aoi_grid({**im, 'audit_resolution_m': 10})
    catalog = json.loads((root / s6['catalog']).read_text(encoding='utf-8'))['items']
    cache = root / cfg['paths']['cache'] / s6['cache_subdir']
    cache.mkdir(parents=True, exist_ok=True)
    budget = ByteBudget(s6['max_fetch_bytes'])

    def read(href, resampling):
        def go():
            with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', GDAL_HTTP_TIMEOUT=policy['timeout_seconds']):
                with rasterio.open(pc.sign(href)) as src:
                    with WarpedVRT(src, crs=im['crs'], transform=transform, width=shape[1], height=shape[0],
                                   resampling=resampling, nodata=0) as vrt:
                        return vrt.read(1), src.scales[0], src.offsets[0]
        return retry(go, policy, href)
    for date in dates:
        items = sorted((i for i in catalog if i['properties']['datetime'][:10] == date), key=lambda i: i['id'])
        if not items:
            raise RuntimeError(f'no catalogued item for {date}')
        red = np.zeros(shape, 'uint16'); nir = red.copy(); scl = np.zeros(shape, 'uint8')
        taken = np.zeros(shape, bool)
        tags, baselines = set(), set()
        for item in items:
            baseline = item['properties']['s2:processing_baseline']
            if not baseline_offset_ok(baseline, s6['min_processing_baseline']):
                raise Blocked(f'{item["id"]}: processing baseline {baseline} < {s6["min_processing_baseline"]}; '
                              'the -1000 offset convention does not apply', evidence='real')
            baselines.add(baseline)
            s_scl = read_scl(item, im, transform, shape, policy)        # R2's own SCL reader, at 10 m
            r, sc_r, of_r = read(item['assets']['B04']['href'], Resampling.nearest)
            n, sc_n, of_n = read(item['assets']['B08']['href'], Resampling.nearest)
            tags.add((float(sc_r), float(of_r), float(sc_n), float(of_n)))
            budget.check()
            take = ~np.isin(s_scl, im['invalid_classes']) & ~taken          # first valid tile wins, as in R2
            red[take], nir[take], scl[take] = r[take], n[take], s_scl[take]
            taken |= take
        np.savez_compressed(cache_path(cache, date), red=red, nir=nir, scl=scl,
                            meta=json.dumps({'date': date, 'items': [i['id'] for i in items],
                                             'processing_baselines': sorted(baselines),
                                             'raster_scale_offset_tags': sorted(tags)}))
        print(f'cached {date}: {budget.used()} bytes received so far (cap {budget.cap_bytes})', flush=True)
    return budget.log()


def probe(cfg, root):
    """E6: pool all clear pre dates vs January-only, false-drop fraction on stable pixels (real 10 m data)."""
    from risk.r2_imagery import aoi_grid
    from pyproj import Transformer
    s, im = cfg['e6'], cfg['phase0']['imagery']
    r2 = json.loads((root / s['r2_result']).read_text(encoding='utf-8'))
    pre_dates, post_date = r2['pre_dates'], r2['first_post_date']
    if not set(s['january_dates']) <= set(pre_dates):
        raise RuntimeError(f'January dates {s["january_dates"]} are not all clear pre dates in R2: {pre_dates}')
    cache = root / cfg['paths']['cache'] / s['cache_subdir']
    data = load_cache(cache, pre_dates + [post_date])
    rad = s['radiometry']
    for d, v in data.items():
        for b in v['meta']['processing_baselines']:
            if not baseline_offset_ok(b, s['min_processing_baseline']):
                raise Blocked(f'{d}: processing baseline {b} below {s["min_processing_baseline"]}', evidence='real')
        for sc_r, of_r, sc_n, of_n in v['meta']['raster_scale_offset_tags']:
            if (sc_r, sc_n) not in ((1.0, 1.0), (1 / rad['quantification'],) * 2) or (of_r, of_n) not in ((0.0, 0.0), (rad['boa_add_offset'] / rad['quantification'],) * 2):
                raise Blocked(f'{d}: raster scale/offset tags {sc_r, of_r} disagree with the assumed ESA convention', evidence='real')
    md = cfg['ndvi']['min_denominator']
    def ndvi_of(v):
        ok = valid_from_scl(v['scl'], im['cloud_classes'], im['shadow_classes'], im['invalid_classes'])
        n = ndvi_bands(reflectance(v['red'], rad['quantification'], rad['boa_add_offset']),
                       reflectance(v['nir'], rad['quantification'], rad['boa_add_offset']), md)
        return np.where(ok, n, np.nan)
    stack = np.stack([ndvi_of(data[d]) for d in pre_dates])
    post = ndvi_of(data[post_date])
    _, transform, shape = aoi_grid({**im, 'audit_resolution_m': 10})
    to_metric = Transformer.from_crs('EPSG:4326', im['crs'], always_xy=True)
    suspected = np.zeros(shape, bool)
    for zone in s['suspected_areas']:
        suspected |= disk_mask(transform, shape, to_metric.transform(zone['lon'], zone['lat']), zone['radius_m'])
    stable = stable_mask(stack, suspected, s['stable_min_valid_dates'], s['stable_max_std'])
    idx_all = list(range(len(pre_dates)))
    idx_jan = [pre_dates.index(d) for d in s['january_dates']]
    pool_all, pool_jan = pooled_mean(stack, idx_all, s['pool_min_valid']), pooled_mean(stack, idx_jan, s['pool_min_valid'])
    thr = cfg['change']['parent_drop_threshold']
    res = false_drop(pool_all, pool_jan, post, stable, thr)
    lo, hi = block_bootstrap_diff(res['flags_a'], res['flags_b'], res['comparable'], s['bootstrap_block_px'],
                                  s['bootstrap_n'], cfg['seed'])
    fa, fb = res['fraction_a'], res['fraction_b']
    january_lower = fa is not None and fb < fa
    ci_supports = lo > 0                      # CI of (all - january) strictly above zero
    ok = january_lower and (ci_supports or not s['require_ci_excluding_zero'])
    return {'status': 'PASS' if ok else 'FAIL', 'evidence': 'real',
            'criteria': {'rule': 'January pooling has a lower false-drop fraction' + (
                ' AND the block-bootstrap CI of (all12 - january) excludes zero' if s['require_ci_excluding_zero'] else ''),
                         'stable_definition': {'min_valid_dates': s['stable_min_valid_dates'],
                                               'max_ndvi_std_across_pre_dates': s['stable_max_std'],
                                               'suspected_areas_excluded': s['suspected_areas']},
                         'parent_drop_threshold_uncalibrated_placeholder': thr, 'pre_dates_all': pre_dates,
                         'pre_dates_january': s['january_dates'], 'post_date': post_date},
            'measurements': {'stable_pixels': int(stable.sum()), 'aoi_pixels': int(stable.size),
                             'comparable_pixels': res['comparable_pixels'],
                             'false_drop_fraction_all_pre_dates': fa, 'false_drop_fraction_january': fb,
                             'difference_all_minus_january': None if fa is None else fa - fb,
                             'bootstrap_ci95_all_minus_january': [lo, hi], 'january_lower': january_lower},
            'limitations': ['Single post date, single AOI, single season pair: one comparison, not a general result.',
                            'The AOI centre may not contain the landslide (see report); if so nearly every flagged '
                            'pixel is a false drop by construction, which suits this test but not a detection claim.',
                            'No labelled slide polygons were available; suspected areas are configured disks, not labels.',
                            'NDVI drop flags vegetation change, not landslides; the threshold is an uncalibrated placeholder.']}


if __name__ == '__main__':
    import sys
    if '--fetch' in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument('--fetch', action='store_true')
        parser.add_argument('--config', default='configs/experiments.yaml')
        args = parser.parse_args()
        cfg_, root_, _ = load_experiment_config(args.config)
        r2_ = json.loads((root_ / cfg_['e6']['r2_result']).read_text(encoding='utf-8'))
        from risk.common import write_json
        log = fetch(cfg_, root_, r2_['pre_dates'] + [r2_['first_post_date']])
        write_json(root_ / cfg_['paths']['results'] / 'e6_fetch_log.json', log)
        print(json.dumps(log, indent=2))
    else:
        run_cli('e6', probe)
