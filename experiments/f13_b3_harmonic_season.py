"""F13 B3: does a harmonic seasonal model beat the pre-pool mean on the null false-alarm rate at matched power?

Pre-registered in configs/f13_discovery.yaml `experiments.B3` (sha256 84beef1c...). Per-pixel OLS NDVI(t) = a0 + a1 t +
two annual harmonics on the 2019-2023 archive, scored against the Jan-2023 pool-mean baseline at matched patch recall.
The archive crosses Sentinel-2 processing baseline 04.00 (2022-01-25): from there L2A digital numbers carry a +1000
BOA offset which is removed per item from its s2:processing_baseline before reflectance and NDVI.
"""
from __future__ import annotations

import math

import numpy as np

from experiments import f13_common as C

N_HARMONICS = 2
PARAMS = 2 + 2 * N_HARMONICS


# ---------------------------------------------------------------- radiometry: the processing-baseline offset ----------------------------------------------------------------

def parse_baseline(baseline) -> tuple:
    if not isinstance(baseline, str) or baseline.count('.') != 1:
        raise ValueError(f'cannot parse processing baseline {baseline!r}; refusing to guess an offset')
    try:
        major, minor = (int(x) for x in baseline.split('.'))
    except ValueError as exc:
        raise ValueError(f'cannot parse processing baseline {baseline!r}') from exc
    return major, minor


def boa_offset(baseline) -> float:
    """BOA_ADD_OFFSET for an item: -1000 from baseline 04.00 (acquisitions from 2022-01-25), 0 before it."""
    return -1000.0 if parse_baseline(baseline) >= (4, 0) else 0.0


def item_reflectance(dn, baseline, quantification: float = 10000.0) -> np.ndarray:
    """rho = (DN + offset) / 10000, clipped at 0 (experiments.wayanad_evidence.fetch.reflectance's convention, with the
    offset taken per item from its own processing baseline instead of one global value)."""
    return np.clip((np.asarray(dn, dtype=np.float64) + boa_offset(baseline)) / quantification, 0.0, None)


def item_ndvi(dn_red, dn_nir, baseline, min_denominator: float) -> np.ndarray:
    r, n = item_reflectance(dn_red, baseline), item_reflectance(dn_nir, baseline)
    den = n + r
    out = np.full(den.shape, np.nan)
    np.divide(n - r, den, out=out, where=den >= min_denominator)
    return out


def step_report(dates, ndvi_per_date, change_date: str, days: int = 30) -> dict:
    """Median of the per-date stable-pixel NDVI in the `days` before and after `change_date`."""
    from datetime import date as D
    c = D.fromisoformat(change_date)
    before, after = [], []
    for d, v in zip(dates, ndvi_per_date):
        delta = (D.fromisoformat(d) - c).days
        if -days <= delta < 0:
            before.append(v)
        elif 0 <= delta <= days:
            after.append(v)
    mb = float(np.median(before)) if before else float('nan')
    ma = float(np.median(after)) if after else float('nan')
    return {'change_date': change_date, 'days': days, 'n_before': len(before), 'n_after': len(after),
            'median_before': mb, 'median_after': ma, 'step': ma - mb}


# ---------------------------------------------------------------- the harmonic model ----------------------------------------------------------------

def design_matrix(t_days) -> np.ndarray:
    """[1, t/365.25, cos(2 pi k t/365.25), sin(2 pi k t/365.25)] for k = 1, 2 (t in days). The trend column is scaled to
    years for conditioning; the fitted values are identical to the unscaled yaml form."""
    t = np.asarray(t_days, dtype=np.float64)
    cols = [np.ones_like(t), t / 365.25]
    for k in range(1, N_HARMONICS + 1):
        cols += [np.cos(2 * np.pi * k * t / 365.25), np.sin(2 * np.pi * k * t / 365.25)]
    return np.stack(cols, axis=1)


def fit_harmonic(ndvi, valid, t_days, min_obs: int = 12, chunk: int = 65536, max_cond: float = 1e12) -> dict:
    """Per-pixel OLS on the valid observations. ndvi, valid: (T, H, W). Returns coef (H, W, 6), resid_std (H, W; RSS over
    n - 6 degrees of freedom), n_valid (H, W). Pixels with fewer than `min_obs` valid observations, or whose normal matrix is
    ill-conditioned (e.g. all observations at one time), are NaN (NO_DATA). Pixels are processed in chunks."""
    nd = np.asarray(ndvi, dtype=np.float64)
    T, H, W = nd.shape
    v = np.asarray(valid, bool) & np.isfinite(nd)
    X = design_matrix(t_days)                                          # (T, 6)
    XX = (X[:, :, None] * X[:, None, :]).reshape(T, PARAMS * PARAMS)
    n_valid = v.sum(axis=0)
    coef = np.full((H * W, PARAMS), np.nan)
    resid = np.full(H * W, np.nan)
    flat_v, flat_y = v.reshape(T, -1), np.where(v, nd, 0.0).reshape(T, -1)
    nv = n_valid.ravel()
    idx = np.flatnonzero(nv >= max(min_obs, PARAMS + 1))
    for start in range(0, idx.size, chunk):
        sel = idx[start:start + chunk]
        w = flat_v[:, sel].astype(np.float64)
        xtx = (XX.T @ w).T.reshape(-1, PARAMS, PARAMS)
        xty = (X.T @ flat_y[:, sel]).T
        with np.errstate(all='ignore'):
            good = np.isfinite(cond := np.linalg.cond(xtx)) & (cond < max_cond)
        sol = np.full((sel.size, PARAMS), np.nan)
        if good.any():
            sol[good] = np.linalg.solve(xtx[good], xty[good][..., None])[..., 0]
        with np.errstate(invalid='ignore'):
            err = np.where(flat_v[:, sel], flat_y[:, sel] - X @ sol.T, 0.0)
            res = np.sqrt((err ** 2).sum(axis=0) / (nv[sel] - PARAMS))
        res[~good] = np.nan
        coef[sel], resid[sel] = sol, res
    return {'coef': coef.reshape(H, W, PARAMS), 'resid_std': resid.reshape(H, W), 'n_valid': n_valid}


def predict_harmonic(coef, t_days) -> np.ndarray:
    """coef (H, W, 6), t_days (m,) -> (m, H, W)."""
    return np.einsum('mp,hwp->mhw', design_matrix(t_days), np.asarray(coef, dtype=np.float64))


def harmonic_score(pred, observed, resid_std) -> np.ndarray:
    """(prediction - observed) / residual std: positive where the observed NDVI is below the seasonal expectation."""
    with np.errstate(invalid='ignore', divide='ignore'):
        return (np.asarray(pred) - np.asarray(observed)) / np.asarray(resid_std)


def pool_score(pool_ndvi, observed) -> np.ndarray:
    """(pool mean - observed) / pool std (ddof = 1), per pixel. pool_ndvi: (n, H, W)."""
    p = np.asarray(pool_ndvi, dtype=np.float64)
    sd = p.std(axis=0, ddof=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        s = (p.mean(axis=0) - np.asarray(observed)) / sd
    return np.where(sd > 0, s, np.nan)


# ---------------------------------------------------------------- matched power and the metric ----------------------------------------------------------------

def recall_at(patch_maxima, threshold: float) -> float:
    m = np.asarray(patch_maxima, dtype=np.float64)
    return float(np.mean(m >= threshold))


def matched_threshold(patch_maxima, target_recall):
    """The largest threshold whose patch recall (fraction of injected patches whose maximum score is >= threshold) is at
    least `target_recall`: the r-th largest patch maximum, r = ceil(target * n). Only injected positives are used."""
    m = np.asarray(patch_maxima, dtype=np.float64)
    m = m[np.isfinite(m)]
    n = m.size
    r = int(math.ceil(target_recall * n - 1e-12))
    if n == 0 or r < 1 or r > n:
        raise ValueError(f'cannot reach recall {target_recall} with {n} finite patch maxima')
    return float(np.sort(m)[::-1][r - 1])


def matched_threshold_strict(patch_maxima, target_recall):
    """Sensitivity reading of the yaml's "exceeds": a window is flagged if its score is STRICTLY greater than the threshold.
    The threshold giving exactly ceil(target * n) exceeding patches is the next-lower patch maximum (r + 1-th largest)."""
    m = np.sort(np.asarray(patch_maxima, dtype=np.float64)[np.isfinite(patch_maxima)])[::-1]
    r = int(math.ceil(target_recall * m.size - 1e-12))
    if m.size == 0 or r < 1 or r >= m.size:
        raise ValueError(f'cannot form a strict threshold for recall {target_recall} with {m.size} finite patch maxima')
    return float(m[r])


def relative_reduction(num_pool, den_pool, num_alt, den_alt) -> float:
    far_p = float(np.sum(num_pool)) / float(np.sum(den_pool))
    far_a = float(np.sum(num_alt)) / float(np.sum(den_alt))
    return float('nan') if far_p == 0 else (far_p - far_a) / far_p


def relative_reduction_ci(num_pool, den_pool, num_alt, den_alt, replicates: int, seed: int, ci: float, return_draws=False) -> dict:
    """Block bootstrap over tiles: the SAME resampled tiles feed both methods; each method's threshold is held fixed
    (its per-tile flagged counts are inputs), only the tiles are redrawn."""
    np_, dp, na, da = (np.asarray(v, dtype=np.float64) for v in (num_pool, den_pool, num_alt, den_alt))
    n = np_.size
    idx = np.random.default_rng(seed).integers(0, n, size=(replicates, n))
    far_p = np_[idx].sum(axis=1) / dp[idx].sum(axis=1)
    far_a = na[idx].sum(axis=1) / da[idx].sum(axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        draws = (far_p - far_a) / far_p
    ok = np.isfinite(draws)
    lo, hi = np.quantile(draws[ok], [(1 - ci) / 2, 1 - (1 - ci) / 2])
    out = {'estimate': relative_reduction(np_, dp, na, da), 'lo': float(lo), 'hi': float(hi), 'replicates': replicates,
           'replicates_undefined': int((~ok).sum()), 'seed': seed, 'ci': ci, 'n_tiles': int(n),
           'far_pool': float(np_.sum() / dp.sum()), 'far_alt': float(na.sum() / da.sum()),
           'numerator_pool': float(np_.sum()), 'denominator_pool': float(dp.sum()),
           'numerator_alt': float(na.sum()), 'denominator_alt': float(da.sum())}
    if return_draws:
        out['draws'] = draws
    return out


def b3_verdict(reduction: float) -> str:
    """yaml experiments.B3.keep_rule on the point estimate: TRUE >= 0.30, FALSE < 0.10, INCONCLUSIVE between."""
    if reduction >= 0.30:
        return 'TRUE'
    if reduction < 0.10:
        return 'FALSE'
    return 'INCONCLUSIVE'


# ---------------------------------------------------------------- archive helpers ----------------------------------------------------------------

def crop_transform(transform, row0: int, col0: int):
    """Affine of the crop whose top-left pixel is (row0, col0) of the grid `transform`."""
    from affine import Affine
    return Affine(transform.a, transform.b, transform.c + col0 * transform.a + row0 * transform.b,
                  transform.d, transform.e, transform.f + col0 * transform.d + row0 * transform.e)


def mosaic_ndvi(layers, min_denominator: float):
    """layers: [(dn_red, dn_nir, baseline, take_mask), ...] in priority order (the first item to cover a pixel wins).

    Returns (ndvi_corrected, ndvi_uncorrected, taken). Corrected applies each item's own processing-baseline offset;
    uncorrected treats every item as baseline < 04.00 (kept only to measure what the correction does). NaN where a pixel
    is not taken or the NDVI denominator is below `min_denominator`."""
    shape = np.asarray(layers[0][3]).shape
    red_c, nir_c = np.zeros(shape), np.zeros(shape)
    red_u, nir_u = np.zeros(shape), np.zeros(shape)
    taken = np.zeros(shape, bool)
    for dn_red, dn_nir, baseline, take in layers:
        t = np.asarray(take, bool) & ~taken
        if not t.any():
            continue
        red_c[t], nir_c[t] = item_reflectance(np.asarray(dn_red)[t], baseline), item_reflectance(np.asarray(dn_nir)[t], baseline)
        red_u[t], nir_u[t] = item_reflectance(np.asarray(dn_red)[t], '03.00'), item_reflectance(np.asarray(dn_nir)[t], '03.00')
        taken |= t

    def nd(r, n):
        den = n + r
        out = np.full(shape, np.nan)
        np.divide(n - r, den, out=out, where=taken & (den >= min_denominator))
        return out.astype(np.float32)
    return nd(red_c, nir_c), nd(red_u, nir_u), taken


def stable_step_check(dates, corrected, uncorrected, valid, excluded, change_date: str, days: int = 30,
                      min_date_valid_fraction: float = 0.5, max_std: float = 0.05) -> dict:
    """Median NDVI of stable pixels, per date, in the `days` either side of `change_date`, with and without the baseline
    correction. Dates whose valid fraction (outside `excluded`) is below `min_date_valid_fraction` are listed and left out;
    stable = outside `excluded`, valid on every date used, and NDVI std (ddof 1, corrected series) <= `max_std`."""
    from datetime import date as D
    c = D.fromisoformat(change_date)
    excl = np.asarray(excluded, bool)
    corr, unc, val = np.asarray(corrected, float), np.asarray(uncorrected, float), np.asarray(valid, bool)
    used, low = [], []
    for i, d in enumerate(dates):
        delta = (D.fromisoformat(d) - c).days
        if not (-days <= delta <= days):
            continue
        (used if val[i][~excl].mean() >= min_date_valid_fraction else low).append(i)
    sub = np.stack([corr[i] for i in used])
    ok = ~excl & np.stack([val[i] for i in used]).all(axis=0) & np.isfinite(sub).all(axis=0)
    with np.errstate(invalid='ignore'):
        ok &= sub.std(axis=0, ddof=1) <= max_std
    med = lambda stack: [float(np.median(stack[k][ok])) for k in range(len(used))] if ok.any() else [float('nan')] * len(used)
    used_dates = [dates[i] for i in used]
    m_c, m_u = med(sub), med(np.stack([unc[i] for i in used]))
    return {'change_date': change_date, 'days': days, 'n_stable_px': int(ok.sum()), 'n_dates_used': len(used),
            'dates_used': used_dates, 'dates_excluded_low_validity': [dates[i] for i in low],
            'median_ndvi_per_date': {'corrected': dict(zip(used_dates, m_c)), 'uncorrected': dict(zip(used_dates, m_u))},
            'corrected': step_report(used_dates, m_c, change_date, days), 'uncorrected': step_report(used_dates, m_u, change_date, days)}


def patch_maxima(score, patches, size: int) -> np.ndarray:
    """Max score inside each size x size patch (NaN if the patch holds no finite score)."""
    s = np.asarray(score, dtype=np.float64)
    out = np.full(len(patches), np.nan)
    for k, (r, c) in enumerate(patches):
        blk = s[r:r + size, c:c + size]
        if np.isfinite(blk).any():
            out[k] = np.nanmax(blk)
    return out


def window_counts_per_tile(score, thr: float, tile_mask, window_px: int):
    """(flagged windows, valid windows) per tile. A window is valid if it holds a finite score and flagged if any score >= thr.
    A tile is a 2 x 2 block of windows; `tile_mask` (tile rows x tile cols) selects which tiles count."""
    s = np.asarray(score, dtype=np.float64)
    H, W = s.shape
    blk = s.reshape(H // window_px, window_px, W // window_px, window_px)
    finite = np.isfinite(blk)
    with np.errstate(invalid='ignore'):
        flagged = (blk >= thr).any(axis=(1, 3))
    valid = finite.any(axis=(1, 3))
    tm = np.repeat(np.repeat(np.asarray(tile_mask, bool), 2, axis=0), 2, axis=1)
    f, v = (flagged & tm).astype(float), (valid & tm).astype(float)
    th, tw = H // (2 * window_px), W // (2 * window_px)
    return f.reshape(th, 2, tw, 2).sum(axis=(1, 3)), v.reshape(th, 2, tw, 2).sum(axis=(1, 3))


# ================================================================ archive acquisition (network; STAC items for the yaml's window only) ================================================================

ARCHIVE_DIR = C.ROOT / 'data' / 'experiments-cache' / 'f13' / 's2_archive'
MANIFEST_B3 = C.ROOT / 'data' / 'experiments-cache' / 'f13' / 'b3_manifest.json'
CROP = (256, 896, 128, 640)
SAFETY_MARGIN_BYTES = 25_000_000           # > the bytes of the groups in flight, so the cap is never exceeded rather than detected after


def _read_crop_band(href, transform_crop, shape_crop, crs, policy):
    """One band over the crop window at 10 m. Same WarpedVRT read as experiments.wayanad_evidence.fetch.read_bands, windowed."""
    import planetary_computer as pc
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT
    from risk.common import retry

    def go():
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', GDAL_HTTP_TIMEOUT=policy['timeout_seconds']):
            with rasterio.open(pc.sign(href)) as src:
                if src.crs.to_string() != crs:
                    raise RuntimeError(f'{href}: CRS {src.crs} differs from AOI CRS {crs}')
                with WarpedVRT(src, crs=crs, transform=transform_crop, width=shape_crop[1], height=shape_crop[0],
                               resampling=Resampling.nearest, nodata=0) as vrt:
                    return vrt.read(1)
    return retry(go, policy, href)


def _fetch_group(key, items, cfg_ev, transform_crop, shape_crop, min_den):
    """One acquisition group -> ARCHIVE_DIR/<date>_<platform>_<orbit>.npz, or a recorded skip when no crop pixel is usable."""
    import hashlib
    import json
    from experiments.wayanad_evidence import fetch as F
    from experiments.wayanad_evidence.data import scl_valid
    date, platform, orbit = key
    r0, r1, c0, c1 = CROP
    name = f'{date}_{platform}_{orbit}'
    target = ARCHIVE_DIR / f'{name}.npz'
    baselines = {i['id']: i['properties'].get('s2:processing_baseline') for i in items}
    if target.is_file():
        with np.load(target) as z:
            rec = json.loads(str(z['meta']))
        rec.update(file=str(target.relative_to(C.ROOT)), bytes_on_disk=target.stat().st_size,
                   sha256=hashlib.sha256(target.read_bytes()).hexdigest())
        return {**rec, 'cached': True}
    scl10, per_item = F.scl_mosaic_10m(cfg_ev, items, CACHE_DIR_SCL)
    scl_crop = scl10[r0:r1, c0:c1]
    taken_scl = np.zeros(scl_crop.shape, bool)
    for _s, v in per_item:
        taken_scl |= v[r0:r1, c0:c1]
    valid_scl = scl_valid(scl_crop, cfg_ev['phase0']['imagery']) & taken_scl
    rec = {'name': name, 'date': date, 'platform': platform, 'orbit': orbit, 'items': baselines,
           'scl_valid_px_crop': int(valid_scl.sum()), 'crop_px': int(valid_scl.size)}
    if not valid_scl.any():
        return {**rec, 'status': 'skipped_no_valid_pixel_in_crop', 'file': None, 'cached': False}
    layers, used_ids, taken = [], [], np.zeros(scl_crop.shape, bool)
    crs = cfg_ev['aoi']['crs']
    policy = cfg_ev['phase0']['network']
    for item, (_s, v) in zip(items, per_item):
        take = v[r0:r1, c0:c1] & ~taken
        if not take.any():
            continue
        red = _read_crop_band(item['assets']['B04']['href'], transform_crop, shape_crop, crs, policy)
        nir = _read_crop_band(item['assets']['B08']['href'], transform_crop, shape_crop, crs, policy)
        layers.append((red, nir, item['properties'].get('s2:processing_baseline'), take))
        used_ids.append(item['id'])
        taken |= take
    corr, unc, got = mosaic_ndvi(layers, min_den)
    valid = valid_scl & got & np.isfinite(corr)
    rec.update(status='fetched', used_items=used_ids, used_baselines=[b for (_r, _n, b, _t) in layers],
               valid_px_crop=int(valid.sum()))
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, ndvi=corr, ndvi_uncorrected=unc, valid=valid, meta=json.dumps(rec))
    rec['file'] = str(target.relative_to(C.ROOT))
    rec['bytes_on_disk'] = target.stat().st_size
    rec['sha256'] = hashlib.sha256(target.read_bytes()).hexdigest()
    return {**rec, 'cached': False}


CACHE_DIR_SCL = C.ROOT / 'data' / 'experiments-cache' / 'f13' / 'wayanad_s2'      # SCL 20 m arrays shared with B1's audit


def fetch_archive(cfg, cfg_ev, ev_root, log, workers: int = 4) -> dict:
    """Every Sentinel-2 L2A acquisition over the AOI from the yaml's archive_window start to the target date inclusive."""
    import json
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from affine import Affine
    from experiments.e6_season_matched import ByteBudget
    from experiments.wayanad_evidence import fetch as F
    from experiments.wayanad_evidence.geo import reference_grid
    from risk.common import write_json
    b3 = cfg['experiments']['B3']['data']
    start, end_arch, target = b3['archive_window'][0], b3['archive_window'][1], b3['target']
    cap = int(cfg['experiments']['B3']['method']['proposed_not_in_memo']['fetch_budget_bytes'])
    transform, shape = reference_grid(cfg_ev['aoi'])
    res20 = cfg_ev['aoi']['audit_resolution_m']
    transform20 = Affine(res20, 0.0, transform.c, 0.0, -res20, transform.f)
    shape20 = (shape[0] * 10 // res20, shape[1] * 10 // res20)
    r0, r1, c0, c1 = CROP
    tcrop, scrop = crop_transform(transform, r0, c0), (r1 - r0, c1 - c0)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR_SCL.mkdir(parents=True, exist_ok=True)
    budget = ByteBudget(cap)
    cfg_w = __import__('copy').deepcopy(cfg_ev)
    cfg_w['dates']['audit_start'], cfg_w['dates']['audit_end'] = f'{start}T00:00:00Z', f'{target}T23:59:59Z'
    items, groups = F.search_groups(cfg_w, transform, shape)
    log(f'STAC: {len(items)} items in {len(groups)} acquisition groups, {start}..{target}')
    rows, _item_rows, errors = F.audit(cfg_w, ev_root, groups, transform20, shape20, budget, CACHE_DIR_SCL)
    if errors:
        raise C.Blocked(f'{len(errors)} SCL asset(s) unreadable, first: {errors[0]}')
    min_den = cfg_ev['radiometry']['min_denominator']
    keys = sorted(groups, key=lambda k: (k[0], str(k[1]), str(k[2])))
    results, stopped_at = {}, None
    t0 = __import__('time').time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs, i = {}, 0
        pending = iter(keys)
        done_count = 0
        while True:
            while len(futs) < workers:
                k = next(pending, None)
                if k is None:
                    break
                if budget.used() > cap - SAFETY_MARGIN_BYTES:
                    stopped_at = k
                    pending = iter(())
                    break
                futs[ex.submit(_fetch_group, k, groups[k], cfg_w, tcrop, scrop, min_den)] = k
            if not futs:
                break
            for f in as_completed(list(futs)):
                k = futs.pop(f)
                results[k] = f.result()
                done_count += 1
                if done_count % 20 == 0:
                    log(f'  {done_count}/{len(keys)} groups done, {budget.used()} B received, {__import__("time").time() - t0:.0f} s')
                break
    fetched = [r for r in results.values() if r['status'] == 'fetched']
    man = {'archive_window': [start, end_arch], 'target': target, 'crop': list(CROP), 'byte_cap': cap, 'bytes_received': budget.used(),
           'n_stac_items': len(items), 'n_groups': len(keys), 'n_fetched': len(fetched),
           'n_skipped_no_valid_pixel': sum(r['status'] != 'fetched' for r in results.values()),
           'stopped_at_group': (list(stopped_at) if stopped_at else None), 'complete': stopped_at is None and len(results) == len(keys),
           'groups': [results[k] for k in keys if k in results]}
    write_json(MANIFEST_B3, man)
    return man


# ================================================================ the run ================================================================

def load_archive(manifest: dict, target: str, archive_end: str):
    """Stack the fetched per-group NDVI files. Returns (train, target_obs): train = observations dated <= archive_end,
    target_obs = the target date's observation, each with dates, t (days since 2019-01-01), ndvi, valid."""
    from datetime import date as D
    t0 = D.fromisoformat('2019-01-01')
    rows = [g for g in manifest['groups'] if g['status'] == 'fetched']
    rows.sort(key=lambda g: (g['date'], str(g['platform']), str(g['orbit'])))
    nd, nu, va, dates, names, bl = [], [], [], [], [], []
    for g in rows:
        with np.load(C.ROOT / g['file']) as z:
            nd.append(z['ndvi']); nu.append(z['ndvi_uncorrected']); va.append(z['valid'])
        dates.append(g['date']); names.append(g['name']); bl.append(g.get('used_baselines'))
    t = np.array([(D.fromisoformat(d) - t0).days for d in dates], dtype=np.float64)
    nd, nu, va = np.stack(nd), np.stack(nu), np.stack(va)
    is_target = np.array([d == target for d in dates])
    is_train = np.array([d <= archive_end for d in dates])
    if not is_target.any():
        raise C.Blocked(f'target date {target} was not fetched (no valid crop pixel or not in the archive)')
    sel = lambda m: {'dates': [d for d, k in zip(dates, m) if k], 'names': [n for n, k in zip(names, m) if k], 't': t[m],
                     'ndvi': nd[m], 'ndvi_uncorrected': nu[m], 'valid': va[m], 'baselines': [b for b, k in zip(bl, m) if k]}
    return sel(is_train), sel(is_target)


def run_scores(train, target, pool_dates, excl_crop):
    """Harmonic and pool-mean scores of the target observation, both on the common validity mask."""
    obs = np.where(target['valid'][0], target['ndvi'][0], np.nan).astype(np.float64)
    fit = fit_harmonic(train['ndvi'], train['valid'], train['t'], min_obs=12)
    pred = predict_harmonic(fit['coef'], target['t'][:1])[0]
    pool_idx = [train['dates'].index(d) for d in pool_dates]
    pool_valid = train['valid'][pool_idx].all(axis=0)
    pool_nd = train['ndvi'][pool_idx].astype(np.float64)
    return obs, fit, pred, pool_idx, pool_valid, pool_nd


def main():
    import json
    import time
    from experiments import f13_a0_exchangeability as A0
    from experiments import f13_b1_cross_season_placebo as B1
    from experiments.wayanad_evidence import config as CE
    from trustsr import placebo_v2 as P
    from trustsr.bootstrap import ratio_bootstrap_ci
    started = C.utc_now()
    t0 = time.time()
    result = {'evidence': 'real', 'experiment_id': 'B3'}
    log = lambda m: print(m, flush=True)
    try:
        pre = C.require_committed_prereg()
        cfg = C.load_prereg()
        b3 = cfg['experiments']['B3']
        boot, seed = cfg['statistics']['bootstrap'], cfg['seed']
        pw = cfg['power_requirement']
        cfg_ev, ev_root, _ = CE.load()

        # ---- B3 needs B1's clear January 2023 dates as the pool baseline --------------------------------------------
        b1_path = C.ROOT / 'experiments/results/f13/b1.json'
        if not b1_path.is_file() or not B1.MANIFEST.is_file():
            raise C.Blocked(f'B1 result/manifest missing ({b1_path}, {B1.MANIFEST}); B3 needs its clear January 2023 pool')
        b1 = json.loads(b1_path.read_text(encoding='utf-8'))
        b1m = json.loads(B1.MANIFEST.read_text(encoding='utf-8'))
        if b1['status'] == 'BLOCKED' or b1m.get('blocked_reason') or not b1m.get('clear_pre_dates'):
            raise C.Blocked('B1 primary is BLOCKED (no clear January 2023 pool); B3 is BLOCKED too')
        pool_dates = list(b1m['clear_pre_dates'])

        # ---- the archive -------------------------------------------------------------------------------------------
        if MANIFEST_B3.is_file() and json.loads(MANIFEST_B3.read_text(encoding='utf-8')).get('complete'):
            man = json.loads(MANIFEST_B3.read_text(encoding='utf-8'))
        else:
            man = fetch_archive(cfg, cfg_ev, ev_root, log)
        result['fetch_log'] = {k: man[k] for k in ('archive_window', 'target', 'crop', 'byte_cap', 'bytes_received', 'n_stac_items', 'n_groups',
                                                    'n_fetched', 'n_skipped_no_valid_pixel', 'stopped_at_group', 'complete')}
        result['fetch_log']['groups'] = [{k: g.get(k) for k in ('name', 'status', 'items', 'used_baselines', 'scl_valid_px_crop', 'valid_px_crop', 'bytes_on_disk')}
                                         for g in man['groups']]
        if not man['complete']:
            raise C.Blocked(f"archive fetch stopped at {man['stopped_at_group']} before the {man['byte_cap']} B cap would be exceeded: "
                            f"{man['n_fetched']} of {man['n_groups']} groups fetched, {man['bytes_received']} B received. Never subsampled.")
        train, target = load_archive(man, b3['data']['target'], b3['data']['archive_window'][1])
        log(f"archive: {len(train['dates'])} training observations {train['dates'][0]}..{train['dates'][-1]}, target {target['dates']}")
        missing_pool = [d for d in pool_dates if d not in train['dates']]
        if missing_pool:
            raise C.Blocked(f'pool dates {missing_pool} are not in the fetched archive')

        # ---- exclusions: the 1500 m disks and footprint, as everywhere else -------------------------------------------
        import rasterio
        with rasterio.open(C.ROOT / 'experiments/wayanad_evidence/outputs/footprint.tif') as src:
            footprint_full, transform = src.read(1).astype(bool), src.transform
        excl = P.exclusion_mask_10m(cfg_ev, transform, CROP, footprint_full)

        # ---- radiometry: the processing-baseline offset ---------------------------------------------------------------
        base_counts = {}
        for d, bs in zip(train['dates'], train['baselines']):
            for b in (bs or []):
                base_counts[b] = base_counts.get(b, 0) + 1
        step = stable_step_check(train['dates'], train['ndvi'], train['ndvi_uncorrected'], train['valid'], excl, '2022-01-25', days=30)
        first_04 = min((d for d, bs in zip(train['dates'], train['baselines']) if bs and any(parse_baseline(b) >= (4, 0) for b in bs)), default=None)
        result['radiometry_check'] = {
            'offset_rule': 'boa_offset(s2:processing_baseline): -1000 for baseline >= 04.00, else 0, applied per item before reflectance and NDVI',
            'implementation': 'experiments/f13_b3_harmonic_season.py item_reflectance / mosaic_ndvi (the repo\'s fetch.reflectance applies one global -1000 '
                              'and read_bands refuses baselines < 04.00, so it cannot serve a 2019-2023 archive)',
            'items_by_baseline_used': base_counts, 'first_observation_with_baseline_ge_04.00': first_04,
            'stable_pixel_step_at_2022_01_25': step,
            'unit_test': 'tests/test_f13.py::TestNoStepAtTheBaselineChange (stable synthetic series: flat when corrected, step > 0.2 when not)'}

        # ---- the two scores ---------------------------------------------------------------------------------------------
        obs, fit, pred, pool_idx, pool_valid, pool_nd = run_scores(train, target, pool_dates, excl)
        res = fit['resid_std']
        s_h = harmonic_score(pred, obs, res)
        s_p = pool_score(pool_nd, obs)
        V = target['valid'][0] & pool_valid & np.isfinite(s_h) & np.isfinite(s_p) & ~excl
        s_h, s_p = np.where(V, s_h, np.nan), np.where(V, s_p, np.nan)
        tile_split = P.checkerboard_split((2560, 2048), 128, seed=seed)            # True = test
        test_w = np.repeat(np.repeat(tile_split, 2, axis=0), 2, axis=1)
        H, W = V.shape
        win_ok = V.reshape(H // 16, 16, W // 16, 16).any(axis=(1, 3))
        eligible = test_w & win_ok
        patches = A0.place_patches(V, eligible, pw['proposed_not_in_memo']['n_patches_per_fold_per_drop'], 10, 16,
                                   int(np.random.SeedSequence([seed, 3]).generate_state(1)[0]))
        maxima = {'harmonic': {}, 'pool': {}}
        for drop in pw['injected_ndvi_drops']:
            inj = obs.copy()
            for r, c in patches:
                inj[r:r + 10, c:c + 10] -= drop
            maxima['harmonic'][str(drop)] = patch_maxima(np.where(V, harmonic_score(pred, inj, res), np.nan), patches, 10)
            maxima['pool'][str(drop)] = patch_maxima(np.where(V, pool_score(pool_nd, inj), np.nan), patches, 10)
        scores = {'harmonic': s_h, 'pool': s_p}
        thr, thr_strict = {}, {}
        for m in scores:
            thr[m] = matched_threshold(maxima[m]['0.3'], 0.8)
            thr_strict[m] = matched_threshold_strict(maxima[m]['0.3'], 0.8)
        # null FAR on the TEST split at each method's matched threshold (thresholds come from injected positives only)
        tm = tile_split
        counts = {m: window_counts_per_tile(scores[m], thr[m], tm, 16) for m in scores}
        counts_strict = {m: window_counts_per_tile(scores[m], thr_strict[m], tm, 16) for m in scores}
        counts_all = {m: window_counts_per_tile(scores[m], thr[m], np.ones_like(tm), 16) for m in scores}
        keep = counts['pool'][1].ravel() > 0
        if not np.array_equal(counts['pool'][1], counts['harmonic'][1]):
            raise AssertionError('methods must be scored on identical valid windows')
        flat = lambda a: a.ravel()[keep]
        red = relative_reduction_ci(flat(counts['pool'][0]), flat(counts['pool'][1]), flat(counts['harmonic'][0]), flat(counts['harmonic'][1]),
                                    boot['replicates'], boot['seed'], cfg['statistics']['ci'])
        keep_s = counts_strict['pool'][1].ravel() > 0
        red_strict = relative_reduction_ci(counts_strict['pool'][0].ravel()[keep_s], counts_strict['pool'][1].ravel()[keep_s],
                                           counts_strict['harmonic'][0].ravel()[keep_s], counts_strict['harmonic'][1].ravel()[keep_s],
                                           boot['replicates'], boot['seed'], cfg['statistics']['ci'])
        far_ci = {m: ratio_bootstrap_ci(flat(counts[m][0]), flat(counts[m][1]), boot['replicates'], cfg['statistics']['ci'], boot['seed']) for m in scores}
        recall = {m: {k: {'recall': recall_at(maxima[m][k], thr[m]), 'patches': int(np.isfinite(maxima[m][k]).sum()),
                          'flagged': int((maxima[m][k] >= thr[m]).sum())} for k in maxima[m]} for m in scores}
        for m in scores:
            nums = lambda k: np.bincount(np.array([((r // 16) // 2) * 16 + ((c // 16) // 2) for r, c in patches]),
                                         weights=(maxima[m][k] >= thr[m]).astype(float), minlength=320)
            den = np.bincount(np.array([((r // 16) // 2) * 16 + ((c // 16) // 2) for r, c in patches]), minlength=320).astype(float)
            for k in maxima[m]:
                ci = ratio_bootstrap_ci(nums(k), den, boot['replicates'], cfg['statistics']['ci'], boot['seed'])
                recall[m][k]['ci'] = {kk: ci[kk] for kk in ('estimate', 'lo', 'hi', 'numerator', 'denominator')}
        verdict = b3_verdict(red['estimate']) if np.isfinite(red['estimate']) else 'INCONCLUSIVE'
        fv = lambda a: float(np.nanmedian(a))
        diag = {'training_observations': len(train['dates']), 'by_year': {y: sum(d.startswith(y) for d in train['dates']) for y in sorted({d[:4] for d in train['dates']})},
                'pixels_with_harmonic_fit': int(np.isfinite(fit['coef'][..., 0]).sum()), 'crop_px': int(V.size),
                'n_valid_obs_per_pixel_quantiles': {q: float(np.quantile(fit['n_valid'], q)) for q in (0.05, 0.5, 0.95)},
                'median_resid_std': fv(res), 'common_valid_px': int(V.sum()), 'common_valid_windows': int(counts['pool'][1].sum()),
                'pool_dates': pool_dates, 'target': target['dates'][0], 'patches': len(patches)}
        meas = [C.measurement('relative_FAR_reduction_harmonic_vs_pool', red['estimate'], numerator=red['numerator_pool'] - red['numerator_alt'],
                              denominator=red['denominator_pool'], lo=red['lo'], hi=red['hi'], replicates=boot['replicates'], seed=boot['seed'],
                              unit='(FAR_pool - FAR_harmonic) / FAR_pool at matched patch recall 0.80 (0.30 drop), test-split windows',
                              far_pool=red['far_pool'], far_harmonic=red['far_alt'], flagged_pool=red['numerator_pool'], flagged_harmonic=red['numerator_alt'],
                              valid_windows=red['denominator_pool'], thresholds=thr)]
        for m in scores:
            f = far_ci[m]
            meas.append(C.measurement(f'window_FAR_test_{m}', f['estimate'], numerator=f['numerator'], denominator=f['denominator'], lo=f['lo'], hi=f['hi'],
                                      replicates=boot['replicates'], seed=boot['seed'], unit='fraction of valid 160 m windows with >= 1 px score >= threshold',
                                      threshold=thr[m], recall_by_drop={k: v['recall'] for k, v in recall[m].items()}))
        result.update(
            status='PASS' if verdict == 'TRUE' else 'FAIL', verdict=verdict, rule_applied=b3['keep_rule'],
            preregistration={**pre, 'experiment': 'B3'},
            verdict_basis='relative FAR reduction (FAR_pool - FAR_harmonic) / FAR_pool at matched recall, point estimate, test-split windows',
            thresholds={'matched_ge_rth_largest_patch_maximum': thr, 'sensitivity_strict_exceeds': thr_strict},
            far={m: {'test': far_ci[m], 'all_windows': {'numerator': float(counts_all[m][0].sum()), 'denominator': float(counts_all[m][1].sum()),
                                                       'estimate': float(counts_all[m][0].sum() / counts_all[m][1].sum())}} for m in scores},
            recall=recall, relative_reduction=red, sensitivity_strict_exceeds={'relative_reduction': red_strict,
                'far': {m: float(counts_strict[m][0].sum() / counts_strict[m][1].sum()) for m in scores},
                'note': 'window flagged iff score > threshold with the threshold at the (r+1)-th largest patch maximum; flags more null windows than the >= reading'},
            archive_diagnostics=diag, measurements=meas,
            interpretations=[
                'Harmonic model: per-pixel OLS of NDVI on [1, t/365.25, cos/sin(2 pi k t/365.25), k = 1, 2] over every valid archive observation dated 2019-01-01..2023-12-26 '
                '(the Jan 2023 pool dates are among them); residual std on n - 6 degrees of freedom; < 12 valid observations or an ill-conditioned normal matrix -> NO_DATA.',
                'Pool baseline = the clear January 2023 dates B1 accepted (2023-01-01, 01-06, 01-11), taken from the archive\'s own corrected NDVI so both methods share one radiometric product.',
                'Both methods are scored on a COMMON valid mask: target valid, all pool dates valid, a finite harmonic fit, outside the disks and footprint; windows are valid if they hold a finite score, so '
                'the FAR denominators are identical.',
                'Matched power: 50 injected patches (10 x 10 px) in valid test-split windows, ONE set of positions shared by all three drops; NDVI falls by exactly the drop; each method\'s threshold is the '
                'r-th largest patch maximum at the 0.30 drop, r = ceil(0.8 n), so patch recall at 0.30 is >= 0.80 by construction (in-sample). Null windows never enter the threshold.',
                'Reading of the yaml\'s "exceeds": the main result flags score >= threshold (the supremum reading); a strict > threshold at the next-lower patch maximum is reported as a labelled sensitivity.',
                'CI: block bootstrap over test-split tiles (32 x 32 px at 10 m = 128 px at 2.5 m), 2000 reps, seed 2024, resampling the SAME tiles for both methods with each method\'s threshold held fixed.',
                'FAR is the test-split window FAR (windows with >= 1 flagged px / valid windows); the all-window FAR is secondary.',
                'The target 2023-12-27 is excluded from the fit; the stable-pixel check uses pixels valid on every used date within 30 days of 2022-01-25 (dates with < 50 % valid crop pixels listed and left out) and NDVI std <= 0.05.'],
            limitations=[
                'One target date, one tile, one 640 x 512 crop; a single pool of three dates against a model fitted on 2019-2023, so the FAR rests on one realisation of the null per method.',
                'The null windows contain whatever real change happened between the baselines and 2023-12-27 (it is not a verified no-change scene); both methods share it.',
                'Matching uses 50 patches, so the threshold is the 40th largest of 50 maxima and recall steps are 0.02.',
                'The pool method uses three dates (pool std ddof 1 from three samples is very noisy), which is the yaml\'s baseline; a longer pool would change its null FAR.',
                'Reprocessed duplicate items (baseline 05.xx, 2024) exist for some acquisitions; the original item is mosaicked first, as in the existing fetch path.'],
            runtime_seconds=round(time.time() - t0, 1))
        return _finish(result, started)
    except C.Blocked as exc:
        result.update(status='BLOCKED', verdict='NOT_RUN', reason=str(exc))
        return _finish(result, started)


def _finish(result, started):
    import json
    result.setdefault('interpretations', [])
    result.setdefault('reproduction_check', None)
    result.setdefault('measurements', [])
    result.setdefault('limitations', [])
    result.setdefault('fetch_log', {'bytes_received': 0, 'items': []})
    out = C.write_result('b3', result, started)
    print(json.dumps({k: out.get(k) for k in ('status', 'verdict', 'reason')}, indent=2))
    return C.exit_code(out['status'])


if __name__ == '__main__':
    raise SystemExit(main())
