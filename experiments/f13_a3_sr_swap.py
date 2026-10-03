"""F13 A3: does the super-resolution ranking change the Wayanad map?

Pre-registered in configs/f13_discovery.yaml `experiments.A3`. F7's production path is run twice with
everything fixed except the `d_sr` argument of `apply_gate_v2` (SR field vs bilinear field). Thresholds come from
the yaml only. Both maps come from gate v2, which FAILED its FAR keep rule (f2.json), so A3 measures agreement
between two rankings, not accuracy.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from experiments import f13_common as C

SCALE = 4


# ---------------------------------------------------------------- helpers ----------------------------------------------------------------

def partial_block_mask(n_per_block, scale: int = SCALE) -> np.ndarray:
    """2.5 m mask of blocks with 0 < n < scale^2 allocated sub-pixels. Full and empty blocks are identical under any
    ranking (block-sum constraint), so only these blocks can differ between two rankings."""
    n = np.asarray(n_per_block)
    partial = (n > 0) & (n < scale * scale)
    return np.repeat(np.repeat(partial, scale, axis=0), scale, axis=1)


def iou(a, b, domain) -> dict:
    a, b, d = np.asarray(a, bool), np.asarray(b, bool), np.asarray(domain, bool)
    inter, union = int((a & b & d).sum()), int(((a | b) & d).sum())
    return {'iou': (inter / union) if union else float('nan'), 'intersection': inter, 'union': union,
            'n_a': int((a & d).sum()), 'n_b': int((b & d).sum())}


def _block_counts(a, scale: int = SCALE) -> np.ndarray:
    a = np.asarray(a, bool)
    h, w = a.shape
    return a.reshape(h // scale, scale, w // scale, scale).sum(axis=(1, 3))


def assert_equal_block_counts(a, b, scale: int = SCALE) -> None:
    ca, cb = _block_counts(a, scale), _block_counts(b, scale)
    if not np.array_equal(ca, cb):
        raise AssertionError(f'per-block allocated counts differ in {int((ca != cb).sum())} blocks: '
                             'something other than the ranking changed')


def moved_fraction(a, b) -> float:
    """Fraction of the pixels in `a` that are not in `b` (a and b have equal per-block counts)."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n = int(a.sum())
    return float((a & ~b).sum() / n) if n else float('nan')


def centred_bilinear_upsample(f, scale: int = SCALE) -> np.ndarray:
    """Pixel-centre-aligned bilinear x4 (sub-pixel s samples the source at (s + 0.5) / 4 - 0.5)."""
    return ndimage.zoom(np.asarray(f, dtype=np.float64), scale, order=1, mode='nearest', grid_mode=True)


def ndvi_drop_10m(pre_ndvi, post_ndvi, valid) -> np.ndarray:
    """10 m NDVI drop = mean of the pre-date NDVI minus the post-date NDVI; invalid / non-finite -> 0.0, the same
    neutral fill F5's bilinear ranking uses (np.nan_to_num(nan=0.0)); invalid blocks are NO_DATA downstream anyway."""
    pre = np.mean([np.asarray(p, dtype=np.float64) for p in pre_ndvi], axis=0)
    drop = pre - np.asarray(post_ndvi, dtype=np.float64)
    return np.where(np.asarray(valid, bool), np.nan_to_num(drop, nan=0.0), 0.0)


def a3_verdict(iou_value: float) -> str:
    """yaml experiments.A3.keep_rule on the point estimate: TRUE < 0.90, FALSE >= 0.95, INCONCLUSIVE between."""
    if iou_value < 0.90:
        return 'TRUE'
    if iou_value >= 0.95:
        return 'FALSE'
    return 'INCONCLUSIVE'


# ================================================================ the run ================================================================

def _compare(flag_a, flag_b, n_per_block, nodata, boot, ci) -> dict:
    """All pre-registered and prompt-requested agreement numbers between two flagged maps with equal block counts."""
    from trustsr.bootstrap import block_sums, ratio_bootstrap_ci
    valid = ~np.asarray(nodata, bool)
    part = partial_block_mask(n_per_block)
    out = {}
    for name, dom in (('whole_map', valid), ('partial_blocks_only', valid & part)):
        r = iou(flag_a, flag_b, dom)
        inter = block_sums(np.asarray(flag_a, bool) & np.asarray(flag_b, bool) & dom, 128)
        union = block_sums((np.asarray(flag_a, bool) | np.asarray(flag_b, bool)) & dom, 128)
        r['ci'] = ratio_bootstrap_ci(inter, union, boot['replicates'], ci, boot['seed'])
        r['bootstrap_unit'] = '128 px (2.5 m) tiles with a non-empty union'
        out[name] = r
    n = np.asarray(n_per_block)
    pb = (n > 0) & (n < SCALE * SCALE)
    n_pb = n[pb].astype(np.float64)
    out['whole_map_inflation'] = {'full_block_px': int((n == SCALE * SCALE).sum()) * SCALE * SCALE, 'share_of_flagged_px': float((n == SCALE * SCALE).sum() * SCALE * SCALE / max(int(n.sum()), 1)), 'whole_minus_partial_iou': out['whole_map']['iou'] - out['partial_blocks_only']['iou']}
    out['block_census'] = {
        'blocks_total': int(n.size), 'blocks_empty': int((n == 0).sum()), 'blocks_full': int((n == SCALE * SCALE).sum()),
        'blocks_partial': int(pb.sum()), 'allocated_px_total': int(n.sum()), 'allocated_px_in_partial_blocks': int(n_pb.sum())}
    out['random_ranking_reference_partial_blocks'] = {
        'expected_iou': float((n_pb ** 2 / 16).sum() / (2 * n_pb - n_pb ** 2 / 16).sum()) if n_pb.size else None,
        'definition': 'analytic expectation if each partial block took a uniformly random subset of its n sub-pixels '
                      'in BOTH maps: sum(n^2/16) / sum(2n - n^2/16). Context only; not a pre-registered comparator.'}
    a, b = np.asarray(flag_a, bool) & valid, np.asarray(flag_b, bool) & valid
    out['moved'] = {'fraction_of_all_allocated_px': moved_fraction(a, b),
                    'fraction_of_allocated_px_in_partial_blocks': moved_fraction(a & part, b & part),
                    'px_moved': int((a & ~b).sum()), 'allocated_px': int(a.sum())}
    out['mapped_area'] = {'px_a': int(a.sum()), 'px_b': int(b.sum()), 'difference_px': int(a.sum() - b.sum()),
                          'difference_km2': float((a.sum() - b.sum()) * 6.25 / 1e6),
                          'note': 'zero by construction (round(16 f) sub-pixels per block in both arms); a plumbing check, not a finding'}
    return out


def main():
    import json
    from experiments import f2_gate_v2 as F2
    from experiments import f7_wayanad_v2 as F7
    from experiments.f3_noise_v2 import normalise_dates
    from experiments.wayanad_evidence import config as CE
    from experiments.wayanad_evidence.data import load_date
    from trustsr import gate_v2 as G
    started = C.utc_now()
    result = {'evidence': 'real', 'experiment_id': 'A3'}
    try:
        pre = C.require_committed_prereg()
        cfg = C.load_prereg()
        a3 = cfg['experiments']['A3']
        boot, ci = cfg['statistics']['bootstrap'], cfg['statistics']['ci']
        cfg_ev, ev_root, _ = CE.load()
        cache = ev_root / cfg_ev['paths']['cache']
        dates = F7.PRE_DATES + [F7.POST_DATE]
        state_path = cache.parent / 'f7_production_state.npz'
        required = [state_path, cache / 'step3_state.npz', ev_root / 'experiments/results/f2.json', ev_root / 'experiments/results/f7.json',
                    ev_root / 'experiments/wayanad_evidence/outputs/footprint.tif']
        for d in dates:
            required += [cache / f'{d}.npz', cache / 'per_date_ndvi' / f'{d}_dihedral_means.npy']
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise C.Blocked('missing cached inputs: ' + '; '.join(missing) + ' | checked: ' + '; '.join(map(str, required)))
        f2 = json.loads((ev_root / 'experiments/results/f2.json').read_text(encoding='utf-8'))
        f7 = json.loads((ev_root / 'experiments/results/f7.json').read_text(encoding='utf-8'))
        tau, lam, k_v1 = float(f2['tau']['value']), float(f2['lambda']['value']), cfg_ev['change']['k']
        r0, r1, c0, c1 = F7.CROP

        # ---- F7's production path, from cache (F7.main steps 3-6; no SR inference, no network) --------------
        arrays = {d: load_date(cfg_ev, cache, d) for d in dates}
        sr = {'arrays': arrays, 'valid_10m': {d: arrays[d]['valid'][r0:r1, c0:c1] for d in dates}}
        refl, ndvi10, valid, excl_crop, _excl_full, aoi = F7.production_inputs(cfg_ev, cache, sr, F7.CROP)
        st = lambda key: np.stack([aoi[key][d] for d in F7.PRE_DATES])
        _nl, em, _table = F2.endmember_ceiling_table(st('refl'), st('valid'), st('ndvi'), aoi['exclude'])
        nd_pre = np.stack([ndvi10[d] for d in F7.PRE_DATES])
        stable = (np.stack([valid[d] for d in F7.PRE_DATES]).all(axis=0) & ~excl_crop
                  & (nd_pre.std(axis=0, ddof=1) <= 0.05) & np.isfinite(nd_pre).all(axis=0))
        pre_stack = np.stack([refl[d] for d in F7.PRE_DATES])
        post_stack = refl[F7.POST_DATE]
        y_pre, y_post = np.empty_like(pre_stack), np.empty_like(post_stack)
        for b in range(pre_stack.shape[-1]):
            p, q, _op, _oq, _ = normalise_dates(pre_stack[..., b], stable, post_stack[..., b], stable)
            y_pre[..., b], y_post[..., b] = p, q
        dih = {d: np.load(cache / 'per_date_ndvi' / f'{d}_dihedral_means.npy') for d in dates}
        valid_all_10 = np.logical_and.reduce([valid[d] for d in dates])
        # d_sr / sigma_sr: the cached v1 Welford moments of the real tiled dihedral SR (float64 NDVI accumulation), which
        # F7 verified equal to its own fresh moments (f7.json sr_run.reproduction_check.v1_step3_moments: max diff 0.0).
        # A first attempt that rebuilt them from the float32 per-date dihedral means moved ONE pixel between NO_CHANGE and
        # UNSUPPORTED (CORE/ALLOCATED identical) and was discarded; it touched no CORE/ALLOCATED pixel and no A3 number.
        with np.load(cache / 'step3_state.npz') as z:
            if tuple(int(v) for v in z['crop']) != tuple(F7.CROP):
                raise C.Blocked(f'step3_state.npz crop {z["crop"].tolist()} is not F7\'s crop {list(F7.CROP)}')
            pre_mean, post_mean, pre_std, post_std = (z[k] for k in ('pre_mean', 'post_mean', 'pre_std', 'post_std'))
        d_sr = pre_mean - post_mean
        sigma_sr = np.sqrt(pre_std ** 2 + post_std ** 2)
        nodata = G.nodata_from_scl(np.stack([valid[d] for d in dates]), SCALE)
        sr_finite = np.isfinite(d_sr) & np.isfinite(sigma_sr)
        for d in dates:
            sr_finite &= np.isfinite(dih[d]).all(axis=0)
        nodata = nodata | ~sr_finite
        sigma_bands, _floor = G.band_sigma_from_pre(y_pre, valid=None)
        gate = dict(lam=lam, window_size=F7.WINDOW_10M, k_unsupported=k_v1, sigma_sr=sigma_sr)
        em_args = (y_pre, y_post)

        class_sr, meta_sr = G.apply_gate_v2(*em_args, d_sr, em['e_v'], em['e_b'], sigma_bands, nodata, tau, **gate)

        # ---- reproduction gate: the SR arm must be F7's stored production map, exactly ------------------------
        with np.load(state_path) as z:
            st_class, st_f, st_nodata = z['class_map'], z['f_effective'], z['nodata']
        counts = {n: int((class_sr == code).sum()) for code, n in G.CLASS_NAMES.items()}
        ref_counts = {n: f7['class_report'][n]['pixels_2p5m'] for n in counts}
        repro = {'class_map_identical_to_f7_state': bool(np.array_equal(class_sr, st_class)),
                 'f_effective_identical': bool(np.array_equal(meta_sr['f_effective'], st_f)),
                 'nodata_identical': bool(np.array_equal(nodata, st_nodata)),
                 'class_counts': counts, 'class_counts_f7_json': ref_counts, 'class_counts_match': counts == ref_counts,
                 'tau': tau, 'tau_source': 'experiments/results/f2.json tau.value (A0 regenerated it bit-identically)',
                 'n_class_map_px_differing': int((class_sr != st_class).sum())}
        result['reproduction_gate'] = repro
        if not (repro['class_map_identical_to_f7_state'] and repro['f_effective_identical'] and repro['nodata_identical']
                and repro['class_counts_match']):
            result.update(status='FAIL', verdict='NOT_RUN', reason='reproduction gate failed: the SR arm is not F7\'s '
                          'production map, so A3 stopped before the bilinear arm was run')
            return _finish(result, started)

        # ---- the bilinear arm: only d_sr changes ------------------------------------------------------------
        drop10 = ndvi_drop_10m([ndvi10[d] for d in F7.PRE_DATES], ndvi10[F7.POST_DATE], valid_all_10)
        arms = {'bilinear_gate_v2_function': G.bilinear_upsample(drop10, SCALE),
                'bilinear_centred_sensitivity': centred_bilinear_upsample(drop10, SCALE)}
        flag_sr = G.flagged_v2(class_sr)
        comps, guard = {}, {}
        for name, field in arms.items():
            try:
                class_b, meta_b = G.apply_gate_v2(*em_args, field, em['e_v'], em['e_b'], sigma_bands, nodata, tau,
                                                  check_sr=True, **gate)
            except Exception as exc:                      # the B9 guard rejecting the field is itself a finding
                guard[name] = f'{type(exc).__name__}: {exc}'
                continue
            guard[name] = 'accepted by assert_sr_score_is_not_replicated'
            if not np.array_equal(meta_b['n_per_block'], meta_sr['n_per_block']):
                raise AssertionError(f'{name}: n_per_block differs from the SR arm; something other than ranking changed')
            flag_b = G.flagged_v2(class_b)
            assert_equal_block_counts(flag_sr, flag_b)
            comps[name] = _compare(flag_sr, flag_b, meta_sr['n_per_block'], nodata, boot, ci)
        result['b9_guard'] = guard
        if 'bilinear_gate_v2_function' not in comps:
            result.update(status='FAIL', verdict='NOT_RUN', reason='the B9 guard rejected the bilinear field (finding); '
                          'A3 stopped as the prompt requires', measurements=[])
            return _finish(result, started)

        prim = comps['bilinear_gate_v2_function']
        whole = prim['whole_map']
        verdict = a3_verdict(whole['iou'])
        meas = []
        for arm, comp in comps.items():
            for scope in ('whole_map', 'partial_blocks_only'):
                m = comp[scope]
                meas.append(C.measurement(f'IoU_SR_vs_{arm}_{scope}', m['iou'], numerator=m['intersection'],
                                          denominator=m['union'], lo=m['ci']['lo'], hi=m['ci']['hi'],
                                          replicates=boot['replicates'], seed=boot['seed'], unit='IoU of CORE|ALLOCATED px'))
        result.update(
            status='PASS' if verdict == 'TRUE' else 'FAIL', verdict=verdict, rule_applied=a3['keep_rule'],
            preregistration={**pre, 'experiment': 'A3'},
            verdict_basis='whole-map IoU of CORE|ALLOCATED over valid 2.5 m px, arm bilinear_gate_v2_function, point estimate (yaml metrics.primary)',
            comparisons=comps, measurements=meas,
            limitations=[
                'Both maps come from gate v2, which FAILED its FAR keep rule (window FAR 0.0652 > 0.05, f2.json); A3 measures agreement '
                'between two rankings inside that gate, not accuracy against any truth (none exists for Wayanad).',
                'The whole-map IoU the yaml uses is computed on flagged px only, so empty blocks contribute nothing; full blocks (n = 16) are '
                'identical under any ranking and do add to both numerator and denominator, so it is inflated by that share (full-block share of '
                'flagged px is in comparisons.*.block_census). The partial-block IoU (0 < n < 16) is reported beside it and is the informative '
                'number; it does not override the yaml verdict.',
                'gate_v2.bilinear_upsample (named by the prompt) maps sub-pixel j to source coordinate j*(h-1)/(4h-1), which stretches the grid and '
                'shifts the field by up to ~1.5 sub-pixels toward the lower right; a pixel-centre-aligned bilinear arm is reported as a sensitivity. '
                'The verdict uses the prompt-specified function.',
                'The bilinear field is the 10 m NDVI drop of the raw (un-normalised) bands, which is what the SR field is built from; invalid '
                'pixels are filled with 0 (F5 precedent), which only touches blocks adjacent to NO_DATA.',
                'The mapped-area difference is zero by construction (block-sum); it is not evidence.',
                'SR is the cached real tiled SEN2SR-lite dihedral stack (F7 verified it against a fresh run); no SR inference was run here.'])
        return _finish(result, started)
    except C.Blocked as exc:
        result.update(status='BLOCKED', verdict='NOT_RUN', reason=str(exc))
        return _finish(result, started)


def _finish(result, started):
    import json
    out = C.write_result('a3', result, started)
    print(json.dumps({k: out.get(k) for k in ('status', 'verdict', 'reason')}, indent=2))
    return C.exit_code(out['status'])


if __name__ == '__main__':
    raise SystemExit(main())
