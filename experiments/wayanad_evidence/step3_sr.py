"""Step 3: 3a Fourier-mask check, 3b/3c tiled pretrained SR of the scar crop with 8 dihedral runs and Welford NDVI moments."""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone

import numpy as np

from experiments.common import Blocked, finalize_result
from experiments.wayanad_evidence import config as C
from experiments.wayanad_evidence import mask_check as M
from experiments.wayanad_evidence.cogio import write_cog
from experiments.wayanad_evidence.data import load_date
from experiments.wayanad_evidence.gate import upsample
from experiments.wayanad_evidence.geo import sr_grid, reference_grid
from experiments.wayanad_evidence.sr import Runner, sr_variants
from experiments.wayanad_evidence.stats import Welford, ndvi
from experiments.wayanad_evidence.tiler import tile_plan
from risk.common import write_json


def fill_invalid(refl, valid):
    """Finite band-median fill of invalid pixels so the model sees no NaN/zeros. They stay NO_DATA downstream."""
    out = refl.copy()
    for b in range(out.shape[0]):
        out[b][~valid] = np.median(refl[b][valid]) if valid.any() else 0.0
    return out


def mask_check(cfg, root, runner, real_window):
    mask = runner.hard_constraint_mask
    radius = min(mask.shape[0] // 2, mask.shape[1] // 2) // cfg['tiling']['scale']
    fit = M.fit_families(mask, radius)
    profile = M.radial_profile(mask)
    probe, probe_source = M.load_edge_probe(root)
    seam = M.seam_probe(runner, real_window)
    smooth = (not fit['is_binary']) and fit['best_family'] != 'ideal'
    return {'shape': list(mask.shape), 'dtype': str(mask.dtype), 'unique_values': int(np.unique(mask).size),
            'min': float(mask.min()), 'max': float(mask.max()), 'centre_value': float(mask[mask.shape[0] // 2, mask.shape[1] // 2]),
            'corner_value': float(mask[0, 0]), 'radius_used_for_fit': radius, 'fit': fit,
            'radial_profile_by_integer_distance': [float(v) for v in profile],
            'classification': 'smooth' if smooth else 'ideal/binary',
            'stride_allowed_by_mask_rule': cfg['tiling']['stride_if_smooth_mask'] if smooth else 96,
            'stride_used': cfg['tiling']['stride'],
            'stride_note': 'stride 112 would need an 8 px centre-crop margin, contradicting the >= 16 px rule; stride 96 with a '
                           '16 px margin is used whichever family fits',
            'upstream_tricks_bug': 'sen2sr.models.tricks.gaussian_filter and sigmoid_filter raise TypeError in the pinned package '
                                   '(torch.exp on a Python float); the families were re-implemented in numpy from their source formulas',
            'synthetic_edge_probe': {'source': probe_source, 'ideal': (probe or {}).get('ideal'), 'gaussian': (probe or {}).get('gaussian')},
            'real_model_seam_probe': {'input_window_px': list(real_window.shape[-2:]), 'grid_offset_input_px': 64,
                                      'by_distance_from_tile_edge': seam,
                                      'note': 'real weights on a real image window; p99/max |SR_gridA - SR_gridB| over bands, '
                                              'binned by distance (2.5 m px) from the grid-A tile edge'}}


def run_sr(cfg, root, config_hash, runner, arrays_by_date, state, out, cache):
    tile, margin, stride, scale = (cfg['tiling'][k] for k in ('tile', 'crop_margin_px', 'stride', 'scale'))
    dates, pre, post = [str(d) for d in state['dates']], [str(d) for d in state['pre']], str(state['post'])
    r0c, r1c, c0c, c1c = (int(v) for v in state['crop'])
    hc, wc = r1c - r0c, c1c - c0c
    valid_all = state['valid_all'][r0c:r1c, c0c:c1c]
    valid_hr = upsample(valid_all, scale)
    floor = cfg['radiometry']['min_denominator']
    acc = {'pre': Welford((hc * scale, wc * scale)), 'post': Welford((hc * scale, wc * scale))}
    plan = tile_plan((hc, wc), tile, margin, stride)
    tile_log, spectral, negative = [], {}, {}
    for date in pre + [post]:
        a = arrays_by_date[date]
        refl = a['refl'][:, r0c:r1c, c0c:c1c]
        valid_scl = a['valid_scl'][r0c:r1c, c0c:c1c]
        model_in = fill_invalid(refl, valid_scl)
        sr_id = np.zeros((4, hc * scale, wc * scale), np.float32)
        err_sum, err_cnt, neg_cnt, tot = np.zeros((8, 4)), np.zeros(4), 0, 0
        target = acc['post'] if date == post else acc['pre']
        for n, t in enumerate(plan):
            r0, c0 = t['r0'], t['c0']
            (rl, rh), (cl, ch) = t['keep_r'], t['keep_c']
            name = f'{date} tile#{n} r0={r0} c0={c0}'
            t_start = time.perf_counter()
            variants = sr_variants(runner, model_in[:, r0:r0 + tile, c0:c0 + tile], name, runner.oom_type,
                                   runner.cleanup, runner.memory)
            seconds = time.perf_counter() - t_start
            hr = (slice((rl - r0) * scale, (rh - r0) * scale), slice((cl - c0) * scale, (ch - c0) * scale))
            v = variants[:, :, hr[0], hr[1]]
            out_region = (slice(rl * scale, rh * scale), slice(cl * scale, ch * scale))
            nd = ndvi(v[:, 0], v[:, 3], floor)                                   # NDVI per run, then moments
            for k in range(8):
                target.update_at(out_region, nd[k], where=valid_hr[out_region])
            sr_id[:, out_region[0], out_region[1]] = v[0]
            kh, kw = rh - rl, ch - cl
            down = v.reshape(8, 4, kh, scale, kw, scale).mean(axis=(3, 5))
            ok = valid_scl[rl:rh, cl:ch]
            err_sum += (np.abs(down - refl[:, rl:rh, cl:ch]) * ok).sum(axis=(2, 3))
            err_cnt += ok.sum()
            neg_cnt += int((v < 0).sum())
            tot += v.size
            tile_log.append({'date': date, 'tile': n, 'r0': r0, 'c0': c0, 'seconds_8_runs': seconds, **runner.memory()})
            if n % 10 == 0:
                print(f'{date} tile {n + 1}/{len(plan)} {seconds:.2f}s', flush=True)
        spectral[date] = {'per_band_mae_identity_run': dict(zip(['B04', 'B03', 'B02', 'B08'], (err_sum[0] / err_cnt).tolist())),
                          'per_band_mae_mean_of_8_runs': dict(zip(['B04', 'B03', 'B02', 'B08'], (err_sum.mean(axis=0) / err_cnt).tolist())),
                          'valid_10m_px': int(err_cnt[0]), 'reference': 'area-mean 4x4 downsample of the SR run vs the 10 m input '
                                                                       'reflectance (valid SCL pixels only)'}
        negative[date] = neg_cnt / tot
        np.save(cache / f'sr_identity_{date}.npy', sr_id)
    # ---- determinism: one tile twice, byte-identical ----
    t = plan[0]
    a_ = arrays_by_date[post]['refl'][:, r0c:r1c, c0c:c1c]
    tin = fill_invalid(a_, arrays_by_date[post]['valid_scl'][r0c:r1c, c0c:c1c])[:, t['r0']:t['r0'] + tile, t['c0']:t['c0'] + tile]
    h1 = hashlib.sha256(sr_variants(runner, tin, 'det-1', runner.oom_type, runner.cleanup, runner.memory).tobytes()).hexdigest()
    h2 = hashlib.sha256(sr_variants(runner, tin, 'det-2', runner.oom_type, runner.cleanup, runner.memory).tobytes()).hexdigest()
    np.savez_compressed(cache / 'step3_state.npz', pre_mean=acc['pre'].mean, pre_std=acc['pre'].std(cfg['change']['ddof']),
                        pre_count=acc['pre'].count, post_mean=acc['post'].mean, post_std=acc['post'].std(cfg['change']['ddof']),
                        post_count=acc['post'].count, crop=np.array(state['crop']))
    # ---- SR COG of the post date (identity run), SR affine = 10 m affine x scale(1/4) ----
    transform, _ = reference_grid(cfg['aoi'])
    from affine import Affine
    t10 = transform * Affine.translation(c0c, r0c)
    t_sr, shape_sr = sr_grid(t10, (hc, wc), scale)
    sr_post = np.load(cache / f'sr_identity_{post}.npy')
    write_cog(out / 'sr_post_2p5m.tif', sr_post, t_sr, cfg['aoi']['crs'], nodata=None, descriptions=['B04', 'B03', 'B02', 'B08'],
              tags={'label': f"model reconstruction: {cfg['labels']['model']}", 'date': post,
                    'run': 'identity dihedral variant', 'grid': 'input affine x scale(1/4)'})
    n_tiles = len(plan)
    secs = np.array([e['seconds_8_runs'] for e in tile_log])
    return {'crop_input_px': [hc, wc], 'crop_sr_px': list(shape_sr), 'tiles_per_date': n_tiles,
            'forward_passes_total': n_tiles * 8 * len(dates), 'seconds_per_tile_8_runs': {
                'median': float(np.median(secs)), 'p95': float(np.percentile(secs, 95)), 'max': float(secs.max()),
                'total': float(secs.sum())},
            'device': cfg['model']['device'], 'cuda_available': bool(runner.torch.cuda.is_available()),
            'memory': runner.memory(),
            'gpu_memory_peak': None if not runner.torch.cuda.is_available() else 'see per-tile log',
            'spectral_consistency': spectral, 'sr_negative_reflectance_fraction': negative,
            'determinism': {'seed': cfg['seed'], 'deterministic_algorithms': cfg['model']['deterministic'],
                            'tile_rerun_bytes_identical': h1 == h2, 'sha256_first_tile_8_runs': h1},
            'tile_log_file': 'step3_tile_log.json', 'tile_log': tile_log,
            'stats_files': ['cache/step3_state.npz'], 'sr_cog': 'sr_post_2p5m.tif',
            'pre_pool': {'dates': pre, 'runs_per_date': 8, 'samples_per_pixel_max': int(acc['pre'].count.max())},
            'post': {'date': post, 'samples_per_pixel_max': int(acc['post'].count.max())}}


def run(cfg, root, config_hash):
    started = datetime.now(timezone.utc).isoformat()
    out, cache = C.outputs(cfg, root), C.cache(cfg, root)
    step1 = json.loads((out / 'step1.json').read_text(encoding='utf-8'))
    if step1['status'] != 'PASS':
        raise Blocked(f"step 1 status is {step1['status']}: no verified scar crop to super-resolve. {step1.get('reason', '')}",
                      evidence='real')
    with np.load(cache / 'step1_state.npz') as z:
        state = {k: z[k] for k in z.files}
    arrays = {str(d): load_date(cfg, cache, str(d)) for d in state['dates']}
    runner = Runner(cfg, root)
    r0c, r1c, c0c, c1c = (int(v) for v in state['crop'])
    post = str(state['post'])
    win = min(384, ((r1c - r0c) // 128) * 128, ((c1c - c0c) // 128) * 128)
    a = arrays[post]
    window = fill_invalid(a['refl'][:, r0c:r0c + win, c0c:c0c + win], a['valid_scl'][r0c:r0c + win, c0c:c0c + win])
    mc = mask_check(cfg, root, runner, window)
    write_json(out / 'step3a_mask.json', mc)
    sr = run_sr(cfg, root, config_hash, runner, arrays, state, out, cache)
    write_json(out / 'step3_tile_log.json', sr['tile_log'])
    sr['tile_log'] = f"{len(sr['tile_log'])} entries in step3_tile_log.json"
    limitations = [cfg['labels']['model'], 'ran on CPU (no CUDA here): GPU peak memory and GPU time per tile are NOT measured; '
                   'the R1 RTX 4050 numbers are the only GPU evidence and are synthetic-input',
                   'no physical 4 GB GPU test', 'reflectance scale/offset convention follows ESA processing baseline; the '
                   'SEN2SR training-data scaling was not verified against the model card']
    return finalize_result('step3', {'status': 'PASS', 'evidence': 'real', 'measurements': {'mask_check': mc, 'sr': sr},
                                     'limitations': limitations}, config_hash, started)


def main():
    cfg, root, config_hash = C.load()
    try:
        result = run(cfg, root, config_hash)
    except Blocked as exc:
        result = finalize_result('step3', {'status': 'BLOCKED', 'evidence': 'real', 'reason': str(exc)}, config_hash,
                                 datetime.now(timezone.utc).isoformat())
    write_json(C.outputs(cfg, root) / 'step3.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'measurements'}, indent=2))


if __name__ == '__main__':
    main()
