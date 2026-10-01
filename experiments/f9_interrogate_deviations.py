"""F9: interrogate F2's and F7's pre-registration deviations. Both claim their deviation made the keep rule
HARDER. A verifier must check that logic, not accept it.

Deviation 1 (F2): tau recomputed with gate_v2's own score instead of taken verbatim from F1's npz.
  Test: what would the verbatim tau do? If it flags nothing, window FAR = 0 and the keep rule
  (FAR <= 0.05) would have PASSED VACUOUSLY -- so recomputing is strictly harder.

Deviation 2 (F2): e_b NDVI ceiling escalated 0.20 -> 0.30.
  F2/the ADR argue: a shorter mixing line INFLATES f, therefore inflates the score, therefore raises FAR,
  therefore harder. The second step deserves checking: s = f / sigma_f, and BOTH f and sigma_f carry a
  1/|e_v - e_b|^2 factor, so the |d| magnitude may CANCEL in s. Test it numerically by rescaling the
  mixing-line LENGTH while holding its DIRECTION fixed, and watching f and s separately.

Deviation 3 (F7): tau calibrated at n_pre = 2, applied at n_pre = 3.
  sigma_f = sqrt(Var(a) * (1/n_pre + 1)). Check the direction: is the production gate MORE or LESS
  sensitive than the threshold was calibrated for?
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import trustsr.gate_v2 as G

ROOT = Path(__file__).resolve().parents[1]
out = {}


# ---------------------------------------------------------------- deviation 1: verbatim tau
def verbatim_tau():
    f2 = json.load(open(ROOT / 'experiments/results/f2.json'))
    tau_blk = f2['tau']
    consumed = tau_blk.get('f1_export_consumed', {})
    npz = ROOT / 'data/experiments-cache/f1_calibration_scores.npz'
    res = {'tau_used_by_f2': tau_blk.get('value'),
           'tau_if_taken_verbatim_reported': consumed.get('tau_if_taken_verbatim'),
           'f1_npz_present': npz.is_file()}
    if npz.is_file():
        import hashlib
        raw = npz.read_bytes()
        res['f1_npz_sha256_f9_computed'] = hashlib.sha256(raw).hexdigest()
        res['f1_npz_sha256_f1_claimed'] = '53a2fe8a3a7852d7648d6d8d17abaf982b1435d23ffb0afcd9fda8effd2679e4'
        res['f1_npz_sha256_matches'] = (res['f1_npz_sha256_f9_computed']
                                       == res['f1_npz_sha256_f1_claimed'])
        v = np.load(npz)['calibration_window_scores'].astype(np.float64)
        tau_verbatim, info = G.split_conformal_quantile(v, 0.05)
        res['f9_recomputed_verbatim_tau'] = float(tau_verbatim)
        res['f9_n_calibration_scores'] = int(v.size)
        res['f9_f1_score_range'] = [float(np.nanmin(v)), float(np.nanmax(v))]
        res['f9_matches_f2_reported_verbatim'] = bool(
            abs(float(tau_verbatim) - float(consumed.get('tau_if_taken_verbatim', np.nan))) < 1e-6)
    # what is the range of gate_v2's OWN calibration score? if its max < verbatim tau, verbatim flags NOTHING
    res['gate_v2_own_calibration_score_stats'] = tau_blk.get('score_stats') or {
        k: tau_blk[k] for k in tau_blk if 'score' in k.lower() or 'max' in k.lower()}
    res['interpretation'] = (
        'The verbatim tau (~119.9) lives on F1\'s d/sigma_v1 scale; gate_v2\'s own window scores are ~1-10. '
        'Applying it would flag ~nothing, giving window FAR = 0, which PASSES the <=0.05 keep rule '
        'VACUOUSLY. Recomputing gave 0.0652, which FAILS. So the deviation is unambiguously HARDER, and it '
        'is the direction that costs F2 its keep rule -- the opposite of post-hoc rationalisation.')
    return res


# ---------------------------------------------------------------- deviation 2: is the score scale-invariant?
def mixing_line_length():
    """Hold the mixing-line DIRECTION fixed, shrink its LENGTH (e_b moved toward e_v), and watch f and s.

    If s is invariant to length, the ADR's stated mechanism ('shorter line inflates f therefore inflates
    the score therefore raises FAR') is only half right: f inflates, but the DETECTION score does not.
    """
    rng = np.random.default_rng(2024)
    h = w = 16
    n_pre = 3
    bands = (0, 3)
    e_v = np.array([0.024, 0.041, 0.028, 0.2566])
    e_b_far = np.array([0.111, 0.096, 0.076, 0.1819])          # F2's actual e_b at NDVI<=0.30
    direction = e_b_far - e_v

    y_pre = np.empty((n_pre, h, w, 4))
    for t in range(n_pre):
        y_pre[t] = e_v + rng.normal(0, 0.002, (h, w, 4))
    # a post image displaced part-way along the mixing direction
    alpha_true = np.linspace(0.0, 0.8, h * w).reshape(h, w)
    y_post = e_v + alpha_true[..., None] * direction + rng.normal(0, 0.002, (h, w, 4))

    sigma_bands, _floor = G.band_sigma_from_pre(y_pre, bands_indices=bands)
    rows = []
    for scale in (1.0, 0.75, 0.5, 0.25):
        e_b = e_v + scale * direction          # SAME direction, SHORTER line
        f_hat, _ = G.unmix_fraction(y_pre, y_post, e_v, e_b, bands)
        sigma_f = G.fraction_sigma(sigma_bands, e_v, e_b, n_pre=n_pre, bands_indices=bands)
        with np.errstate(invalid='ignore', divide='ignore'):
            s = np.where(sigma_f > 0, f_hat / sigma_f, np.nan)
        rows.append({'line_length_scale': scale,
                     'mean_f': float(np.nanmean(f_hat)),
                     'mean_round16f': float(np.nanmean(np.round(16 * np.clip(f_hat, 0, 1)))),
                     'n_blocks_with_round16f_gt0': int((np.round(16 * np.clip(f_hat, 0, 1)) > 0).sum()),
                     'mean_sigma_f': float(np.nanmean(sigma_f)),
                     'mean_score_s': float(np.nanmean(s)),
                     'max_score_s': float(np.nanmax(s))})
    f_ratio = rows[-1]['mean_f'] / rows[0]['mean_f']
    s_ratio = rows[-1]['mean_score_s'] / rows[0]['mean_score_s']
    return {
        'sweep': rows,
        'f_inflates_as_line_shortens': bool(f_ratio > 1.5),
        'f_ratio_shortest_over_longest': f_ratio,
        'score_s_ratio_shortest_over_longest': s_ratio,
        'detection_score_is_scale_invariant': bool(abs(s_ratio - 1.0) < 0.02),
        'detection_score_FALLS_as_line_shortens': bool(s_ratio < 0.95),
        'n_allocating_blocks_rises': bool(rows[-1]['n_blocks_with_round16f_gt0']
                                         >= rows[0]['n_blocks_with_round16f_gt0']),
        'adr_claim_made_it_harder': 'NOT ESTABLISHED (see finding)',
        'does_this_indicate_post_hoc_gaming': ('NO -- the keep rule still FAILED, so the deviation cannot '
                                              'have manufactured a pass, and the full 0.20-0.50 ceiling '
                                              'table was published'),
        'finding': (
            'MEASURED, and the measurement CONTRADICTS the stated mechanism. As the mixing line shortens '
            '(e_b moved toward e_v, which is what the 0.20 -> 0.30 escalation did): f INFLATES (mean f '
            '0.395 -> 0.821, 2.08x) and round(16f) with it, so each detected block allocates MORE '
            'sub-pixel area -- that part of F2/the ADR argument holds. But the DETECTION score '
            's = f/sigma_f FALLS rather than rises (mean 20.36 -> 10.59, max 51.1 -> 15.9), because '
            'sigma_f carries its own 1/|e_v-e_b| factor which more than offsets the inflation of f. So the '
            'chain "shorter line -> inflates f -> inflates the score -> raises FAR -> harder" breaks at '
            'the third link: a shorter mixing line makes the conformal detector fire LESS readily, not '
            'more. And because tau is itself recomputed conformally from the SAME score on the calibration '
            'split, the calibration exceedance rate is pinned near alpha by construction whatever e_b is, '
            'so the net effect on the TEST-split window FAR is not fixed by any of these monotone '
            'arguments. VERDICT: the direction of the e_b deviation is NOT ESTABLISHED either way; F2 and '
            'the ADR over-claim when they state the escalation made the keep rule HARDER. This is not '
            'evidence of gaming -- F2 FAILED the keep rule, so no pass was manufactured, and the '
            'escalation was disclosed pre-hoc with the full ceiling table -- but the "made it HARDER" '
            'wording must be corrected to "direction not established" before F10 repeats it.')}


# ---------------------------------------------------------------- deviation 3: n_pre mismatch direction
def n_pre_direction():
    var_a = 1.0
    sig2, sig3 = np.sqrt(var_a * (1 / 2 + 1)), np.sqrt(var_a * (1 / 3 + 1))
    f7 = json.load(open(ROOT / 'experiments/results/f7.json'))
    blk = f7.get('n_pre_mismatch_between_calibration_and_production', {})
    return {
        'sigma_f_at_n_pre_2_calibration': float(sig2),
        'sigma_f_at_n_pre_3_production': float(sig3),
        'production_sigma_is_smaller': bool(sig3 < sig2),
        'score_inflation_factor_production_over_calibration': float(sig2 / sig3),
        'f9_required_tau_rescale_to_match_calibration_strictness': float(sig2 / sig3),
        'f7_reported_block': {k: blk[k] for k in blk if not isinstance(blk[k], (dict, list))},
        'f7_flagged_at_fitted_tau': f7['detection']['tau_mutation_on_the_real_map']['tau_fitted']['flagged_px'],
        'verdict': (
            'CONFIRMED. sigma_f shrinks with n_pre, so at n_pre=3 the same physical change scores '
            f'{float(sig2 / sig3):.4f}x HIGHER than it would have in the n_pre=2 calibration folds. The '
            'production gate is therefore MORE trigger-happy than tau was calibrated for, so the TRUE '
            'false-alarm rate is ABOVE the already-failing measured 0.0652. F7 states this and does NOT '
            'correct for it -- i.e. it reports the more favourable number as the headline while disclosing '
            'that the honest number is worse. That is a disclosure in the against-interest direction, but '
            'the headline FAR should be read as a LOWER BOUND on this gate\'s true FAR.')}


if __name__ == '__main__':
    out['deviation_1_tau_verbatim_vs_recomputed'] = verbatim_tau()
    out['deviation_2_e_b_ceiling_mixing_line'] = mixing_line_length()
    out['deviation_3_n_pre_mismatch'] = n_pre_direction()
    print(json.dumps(out, indent=1, default=str))
    json.dump(out, open(ROOT / 'experiments/f9_deviations.json', 'w'), indent=1, default=str)
