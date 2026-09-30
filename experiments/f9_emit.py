"""Generates experiments/results/f9.json.

The verdict counts are COMPUTED from the claims list, never typed by hand: A10's own failure (B13) was a
summary that said 3/5 KEEP while listing 2, and this script exists so F9 cannot repeat it.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / 'configs/fix.yaml'
CONFIG_SHA = hashlib.sha256(FIX.read_bytes()).hexdigest()

K, D, X = 'keep', 'downgrade', 'drop'

# (task, claim, verdict, reason, checks_bearing_on_it)
CLAIMS = [
    # ---------------------------------------------------------------- F1
    ('F1', 'rule_10m pixel FAR = 0.000265 [0.0000654, 0.000563], n = 7,232,544 valid 2.5 m px (pool N=6)', K,
     'Recomputed 0.00026546675692536403 = numerator/denominator exactly; point inside its own CI; n stated.',
     ['point_estimate_inside_its_ci', 'denominators_stated', 'no_placebo_code_path_reads_post_event_date']),
    ('F1', 'rule_10m window FAR = 0.00561 [0.00201, 0.01074], n = 1,962 windows', K,
     'Recomputed from num/den; inside own CI; window n stated; window FAR uses this gate\'s own flag map (B3 fixed).',
     ['point_estimate_inside_its_ci', 'denominators_stated']),
    ('F1', 'ungated_S_v1_sigma pixel FAR = 0.4033 [0.3548, 0.4593], n = 3,617,880 px', K,
     'Recomputed exactly; inside own CI; serves as the ungated baseline.',
     ['point_estimate_inside_its_ci', 'denominators_stated']),
    ('F1', 'ungated_S_v1_sigma window FAR = 0.8532 [0.8183, 0.8864], n = 981 windows', K,
     'Recomputed exactly; inside own CI.',
     ['point_estimate_inside_its_ci', 'denominators_stated']),
    ('F1', 'gate_v1 pixel FAR = 0.000495 / window FAR = 0.00714, n = 3,617,880 px / 981 windows', K,
     'Recomputed exactly (numerator 1792); both point estimates inside their own CIs.',
     ['point_estimate_inside_its_ci', 'denominators_stated']),
    ('F1', 'gate_v1 and gate_v1_with_a5_sigma are numerically IDENTICAL on the real crop BY DESIGN, not by '
           'code aliasing', K,
     'Independently verified: the 5 GATES entries have 5 distinct __code__ objects (no shared pair, so no B2 '
     'regression); F9 built the fixture F1 lacks (only sigma_a5 degenerate) and the two gates diverge by '
     '2048 px, and the mirrored fixture flips the asymmetry, proving each reads its OWN sigma; on all-finite '
     'data with sigmas differing 3x both equal the parent mask exactly, reproducing F1\'s stated mechanism.',
     ['gates_in_a_table_differ_in_code', 'no_test_compares_quantity_with_itself']),
    ('F1', 'FAR reduction vs ungated SR: rule_10m -0.4030, gate_v1 -0.4028, gate_v1_with_a5_sigma -0.4028', K,
     'Arithmetic reproduced from the pooled estimates. Caveat already disclosed by F1 and retained: rule_10m '
     'is on a 6-date pool and the SR gates on a 3-date pool, so its row is not on equal footing.',
     ['point_estimate_inside_its_ci', 'denominators_stated']),
    ('F1', 'the placebo/calibration code path cannot load the post-event date (B1 hard guard)', K,
     '16/16 adversarial load attempts behaved correctly: guarded_load_date, build_sr_fold (held_out AND '
     'smuggled into pre_dates), build_10m_fold, and f2_gate_v2.build with the post date monkeypatched into '
     'SR_POOL all raised PostEventDateBlocked under the REAL config; malformed spellings (2024-7-30, '
     '20241206, 2024/12/06) fail CLOSED; pre-event controls still load. The unguarded dihedral loaders '
     '(placebo_v2._load_dihedral, f2_gate_v2.load_dihedral) are provably unreachable with a post date '
     'because guarded_load_date runs first in every caller.',
     ['no_placebo_code_path_reads_post_event_date']),
    ('F1', 'gate_v2 = BLOCKED in F1\'s 5-gate table', K,
     'Correct AS OF F1\'s run, but now SUPERSEDED: after F2 landed, placebo_v2.gate_v2 is the real gate and '
     'raises only when gate_v2_params are absent. F1_REPORT.md still describes it as a stub that raises '
     'NotImplementedError. F10 must not reprint F1\'s BLOCKED row as current.',
     ['gates_in_a_table_differ_in_code']),
    ('F1', 'the hash 7ee83389...0111e7e6 hinted in the task brief "is 66 hex chars and cannot be a valid '
           'SHA-256 digest"', X,
     'FALSE, and disproven by running len(): the string is 64 hex characters and is a perfectly well-formed '
     'SHA-256 -- it is the exact sha256 of configs/fix.yaml as pre-registered at commit 020880f, i.e. the '
     'hash F6 legitimately computed before F1\'s YAML fix. DROPPED rather than downgraded because the claim '
     'is not imprecise, it is wrong in both of its two assertions, and it was reached by eyeballing a string '
     'instead of measuring it -- precisely the failure mode f9_checks exists to catch. Nothing scientific '
     'rests on it: F1\'s own config_sha256 is correct.',
     ['every_result_carries_fix_yaml_hash']),
    # ---------------------------------------------------------------- F2
    ('F2', 'block-sum test: max |deviation| = 0 over 983,040 block-evaluations -> PASS', K,
     'NOT tautological, and proven so by sabotage: F9 rewrote block_sum_deviation so that `observed` became '
     'a second copy of the `round(16f)` formula (recreating x4\'s B7 self-comparison) and both anti-tautology '
     'tests then FAILED (test_the_check_is_not_tautological_it_can_fail, and the one on the real production '
     'map). Reverted; git diff clean. Signature audit confirms n_per_block is not a parameter. Evidential '
     'weight is nonetheless limited: it verifies allocator/class-assignment plumbing, not science.',
     ['no_test_compares_quantity_with_itself']),
    ('F2', 'window FAR keep rule: 0.0652 [0.0478, 0.0850] on the test split vs required <= 0.05 -> NOT MET, '
           'F2 status FAIL', K,
     'The FAILURE is the verified result. 0.0652395514780836 = 64/981 exactly; point inside its own CI; '
     'n = 981 windows stated. F9 additionally confirmed the FAIL is robust to F2\'s arm choice: the '
     'un-normalised comparator is 0.0540 = 53/981, also > 0.05. Note the CALIBRATION-split rate is 0.04893 '
     'in both arms (conformal pins it near alpha by construction), so the failure is a test-split '
     'generalisation gap consistent with the disclosed exchangeability caveat, not a calibration bug.',
     ['point_estimate_inside_its_ci', 'denominators_stated', 'negative_results_present_in_claim_table']),
    ('F2', 'pixel FAR = 0.0036 [0.0022, 0.0054], n = 3,617,880 valid 2.5 m px', K,
     'Recomputed 0.003639424193173903 = 13167/3617880 exactly; inside own CI.',
     ['point_estimate_inside_its_ci', 'denominators_stated']),
    ('F2', 'tau = 3.889118 = T_(933) of n = 981 placebo CALIBRATION-split window scores, alpha = 0.05', K,
     'Order-statistic index k = ceil((n+1)(1-alpha)) = 933 for n=981 confirmed; calibration data is '
     'null-event only (guard attack 5 proves the post date cannot enter recompute_tau\'s path). F1\'s exported '
     'npz sha256 independently recomputed as 53a2fe8a...2679e4 -- MATCHES F1\'s and F2\'s claim.',
     ['no_placebo_code_path_reads_post_event_date', 'every_result_carries_fix_yaml_hash']),
    ('F2', 'tau is USED, not merely loaded (B10 fixed)', K,
     'Proven by sabotage: F9 replaced the tau-application line with `win_valid & isfinite(win_scores)` '
     '(recreating x9\'s B10 bug) and both TestTauMutation tests FAILED. So the mutation test is not vacuous. '
     'tau is also a required positional parameter with no default; None/NaN/inf raise. Reverted; diff clean.',
     ['every_loaded_threshold_is_used_mutation_test']),
    ('F2', 'endmembers estimated from data, not hard-coded: e_v from 178,141 px, e_b from 332 px (B12 fixed)', K,
     'Counts and rules stated with denominators; estimate_endmembers takes no label-like argument '
     '(signature-audited in code); the exclusion only REMOVES pixels and a test sets the excluded region to '
     'absurd values and asserts bit-identical endmembers.',
     ['denominators_stated']),
    ('F2', 'the gate is HIGHLY sensitive to e_b: +1 robust SD multiplies flagged pixel count by ~8 '
           '(baseline 3,471, max |delta| 24,298)', K,
     'A negative result about robustness, reported with its baseline denominator and carried into F7. This '
     'is the single largest uncertainty in the gate and F2 does not bury it.',
     ['denominators_stated', 'negative_results_present_in_claim_table']),
    ('F2', 'lambda = 0.0, fitted on F1\'s calibration split only by split-half dihedral Jaccard', K,
     'NOT an unused loaded parameter, so the stop rule does not fire. lambda is read from f2.json by F7 '
     '(line 578) and passed into every apply_gate_v2 call (lines 630/651/660/679); F9 sabotaged the lambda '
     'application and test_lambda_changes_which_sub_pixels_are_chosen_but_never_how_many then FAILED, so the '
     'parameter is genuinely wired and its mutation test is not vacuous. The fitted optimum simply lies at '
     'the grid boundary (Jaccard highest at 0, falling monotonically). Consequence to carry forward: the '
     'neighbour-agreement term contributes NOTHING to any reported result, so no claim that it helps is '
     'supported -- which F2 states.',
     ['every_loaded_threshold_is_used_mutation_test']),
    ('F2', 'the e_b NDVI-ceiling escalation (0.20 -> 0.30) made the keep rule HARDER, not easier', D,
     'DIRECTION NOT ESTABLISHED. F9 measured it instead of accepting the argument: shortening the mixing line '
     'does inflate f (mean f 0.395 -> 0.821, 2.08x) and round(16f) with it, but the DETECTION score '
     's = f/sigma_f FALLS rather than rises (mean 20.36 -> 10.59, max 51.1 -> 15.9) because sigma_f carries '
     'its own 1/|e_v-e_b| factor that more than offsets f. The chain "shorter line -> inflates f -> inflates '
     'the score -> raises FAR -> harder" therefore breaks at the third link, and since tau is recomputed '
     'conformally from the same score the calibration rate is pinned near alpha whatever e_b is, leaving the '
     'net test-split effect indeterminate. NOT evidence of gaming: the escalation was disclosed pre-hoc with '
     'the full 0.20-0.50 ceiling table and the keep rule FAILED anyway, so no pass was manufactured. The '
     '"made it HARDER" wording must be corrected to "direction not established" before F10 repeats it.',
     ['negative_results_present_in_claim_table']),
    ('F2', 'recomputing tau with gate_v2\'s own score (instead of F1\'s verbatim export) made the keep rule '
           'HARDER, not easier', K,
     'VERIFIED numerically. F9 recomputed the verbatim tau from F1\'s npz as 119.90589141845703, matching '
     'f2.json\'s tau_if_taken_verbatim_from_this_file to the last digit; F1\'s scores span -25.86 to 255.60 '
     'whereas gate_v2\'s own window scores are order 1-10. Applying 119.9 would flag ~nothing, giving window '
     'FAR = 0 and a VACUOUS PASS of the <=0.05 rule. Recomputing produced 0.0652 and a FAIL. The deviation '
     'is unambiguously the harder choice and it is the choice that cost F2 its keep rule -- the opposite of '
     'post-hoc rationalisation.',
     ['every_loaded_threshold_is_used_mutation_test']),
    # ---------------------------------------------------------------- F3
    ('F3', 'normalised pooled LODO coverage of |d| <= 2 sigma = 94.2% [93.3, 95.0], n = 7,074,856 px, '
           'gap -1.27 pp -> keep rule KEPT', K,
     'Recomputed 0.9418016423231794 = 6,663,111/7,074,856 exactly; gap_pp exact; |gap| 1.27 <= 2.0 so KEPT; '
     'pooled equals the per-fold n-weighted mean exactly and the fold n sum to the pooled n exactly; point '
     'inside its own CI.',
     ['point_estimate_inside_its_ci', 'denominators_stated']),
    ('F3', 'same sigma method WITHOUT normalisation = 85.8% [84.5, 87.0], gap -9.65 pp (fails the rule)', K,
     'Recomputed 0.8579530099269865 = 6,069,894/7,074,856 exactly. This contrast is what shows the '
     'normalisation does real work rather than being cosmetic, and it is reported alongside the headline.',
     ['point_estimate_inside_its_ci', 'negative_results_present_in_claim_table']),
    ('F3', 'estimator selected on E6 only (eb_stratified); paired NLL diff vs runner-up '
           '-0.160 [-0.251, -0.083], CI excludes 0', K,
     'Selection used no Wayanad pixel, so the Wayanad evaluation is genuinely out-of-sample for the choice; '
     'CI excludes zero; point inside its CI.',
     ['point_estimate_inside_its_ci']),
    ('F3', 'change-preservation: a +0.30 NDVI drop injected into 1,600 real stable px is recovered at ratio '
           '1.000 (fitted offset moved by -0.01% of the drop)', K,
     'Arithmetic consistent and the denominator (1,600 of 2,358,286 stable px) is stated. Weight limited and '
     'F3 says so: the injection is synthetic, so this bounds the estimator\'s sensitivity to a minority-pixel '
     'change, not its behaviour on a real event.',
     ['denominators_stated']),
    ('F3', 'median(d) = 0 for every LODO fold after normalisation', K,
     'This IS a tautology -- offset_held is DEFINED as the median residual, so removing it zeroes the median '
     'identically -- but the stop rule does not fire, because F3 identifies it as definitional in its own '
     'limitations and explicitly declines to use it as evidence, resting the keep rule on COVERAGE instead '
     '(sigma is estimated from pool across-date and dihedral variance, never from d). Kept as a stated '
     'identity, not as a result. Had F3 offered it as evidence this would have been a DROP.',
     ['no_test_compares_quantity_with_itself']),
    ('F3', 'f3_REPORT.md cites config_sha256 `192345cd156583c6...`', D,
     'STALE. 192345cd... is the hash of configs/fix.yaml at commit 298e214 (F3\'s own YAML fix, without F1\'s '
     'comment); the merged branch kept F1\'s version, hash 18cce7b1.... f3.json WAS amended (commit 0059c2f) '
     'but the report was not, so the human-readable file disagrees with the machine-readable one. The number '
     'is also truncated to 16 chars. Downgrade, not drop: the underlying result is unaffected (F9 confirmed '
     'the two YAML variants are semantically IDENTICAL under yaml.safe_load).',
     ['every_result_carries_fix_yaml_hash']),
    # ---------------------------------------------------------------- F5
    ('F5', 'headline paired IoU(alloc_pretrained_sr) - IoU(alloc_bilinear), oracle fraction, all 4 datasets '
           '= +0.0081 [+0.0035, +0.0141], n = 118 images, SR wins on 87/118', K,
     'INDEPENDENTLY REPRODUCED by F9 from the per-image IoU records with its own bootstrap: +0.008078, '
     'CI [+0.00347, +0.01410], n = 118, 87 wins. The n=118 (not 119) exclusion of one spain_crops image with '
     'no truth pixel is documented and correct.',
     ['point_estimate_inside_its_ci', 'denominators_stated', 'no_test_compares_quantity_with_itself']),
    ('F5', 'headline EXCLUDING NAIP = +0.0016 [-0.0026, +0.0053], n = 56 -- not distinguishable from zero', K,
     'INDEPENDENTLY REPRODUCED: +0.001592, CI [-0.00261, +0.00532], n = 56, 30/56 wins, CI spans zero. This '
     'is the key negative and F5 leads with it rather than hiding it; NAIP is exactly the set where '
     'SEN2SR-lite train/test overlap cannot be excluded.',
     ['point_estimate_inside_its_ci', 'denominators_stated', 'negative_results_present_in_claim_table']),
    ('F5', 'the only positive surviving NAIP removal is the 0-20 m width stratum: +0.0077 [+0.0018, +0.0132], '
           'n = 47', K,
     'Point inside its CI, n stated, and the direction matches what the sub-pixel hypothesis predicts '
     '(effect ~15x smaller for features wider than 150 m). Absolute IoU in that bin is low (0.272 SR / 0.255 '
     'bilinear), which F5 states.',
     ['point_estimate_inside_its_ci', 'denominators_stated']),
    ('F5', 'under the REALISTIC (unmixing) fraction, alloc_pretrained_sr is -0.313 IoU [-0.344, -0.282] '
           'WORSE than blocky_10m, n = 119, better on only 3/119 images', K,
     'The most consequential negative in the wave and it is stated plainly, with the mechanism (per-block '
     'fraction MAE 0.400, within-image Pearson r 0.294) and the fairness note that the oracle arm gives the '
     'allocators information blocky does not have. Correctly carried into F7 as a prior against F7\'s map.',
     ['point_estimate_inside_its_ci', 'denominators_stated', 'negative_results_present_in_claim_table']),
    ('F5', 'the water (NDWI > 0) class is null in both directions; no claim made', K,
     'CI spans zero with and without NAIP; n = 44 of 119 images contain any NDWI>0 pixel, stated.',
     ['point_estimate_inside_its_ci', 'denominators_stated', 'negative_results_present_in_claim_table']),
    ('F5', 'alloc_bilinear and alloc_pretrained_sr differ ONLY in the ranking field, so the headline is a '
           'genuine ranking comparison and not a quantity compared with itself', K,
     'Verified in code: both receive the IDENTICAL fraction and the same allocator, but rank by s_bil '
     '(bilinear upsample of the 10 m index) versus s_sr (index of the real SR forward pass) -- two '
     'independently constructed fields. A self-comparison would have produced exactly 0.0; the observed '
     'delta is non-zero with 87/118 wins.',
     ['no_test_compares_quantity_with_itself', 'gates_in_a_table_differ_in_code']),
    ('F5', 'alloc_finetuned_sr BLOCKED and heavy/Colab SR NOT RUN; pretrained output never substituted', K,
     'No fine-tuned number appears anywhere in f5.json; venus correctly excluded (5 m HR cannot carry a 2.5 m '
     'truth mask). Blocked things are declared, not estimated.', []),
    # ---------------------------------------------------------------- F6
    ('F6', 'pooled PSNR delta vs bicubic across all 5 datasets = -0.097 dB [-0.157, -0.037], n = 178 images', K,
     'INDEPENDENTLY RECOMPUTED from x2.json by three separate routes (vectorised numpy, a longhand Python '
     'loop over the raw json, and a per-dataset weighted recombination), agreeing to < 1e-12: '
     '-0.09707162370791389. F9\'s own separately written paired bootstrap reproduced the CI as '
     '[-0.1574637956167713, -0.03696577159967953], and a normal-approximation CI brackets it. This is the '
     'B14 number RESULTS_EXCEPTIONAL.md omitted and it is now correct and carried.',
     ['point_estimate_inside_its_ci', 'denominators_stated', 'negative_results_present_in_claim_table']),
    ('F6', 'per-dataset table, all 5 including the 3 omitted: naip +0.236 (n=62), spot +0.130 (n=9), '
           'spain_crops -0.007 (n=28), spain_urban +0.005 (n=20), venus -0.559 (n=59)', K,
     'All five point estimates and CIs independently reproduced to 3 dp by F9 with its own bootstrap; every '
     'row carries its image count. The venus -0.559 row is the one whose omission was B14.',
     ['point_estimate_inside_its_ci', 'denominators_stated', 'negative_results_present_in_claim_table']),
    ('F6', 'NAIP-excluded sensitivity: -0.275 dB [-0.345, -0.203], n = 116 images', K,
     'Independently reproduced: -0.2750402373233242, CI [-0.34499, -0.20261], n = 116. Removing the possibly '
     'leaking dataset makes the pooled result MORE negative, which F6 states.',
     ['point_estimate_inside_its_ci', 'denominators_stated', 'negative_results_present_in_claim_table']),
    ('F6', 'opensr-test ha_metric: SR is more hallucination-prone than bicubic on EVERY one of the 5 datasets', K,
     'All ten means independently reproduced to 4 dp; SR > bicubic on all 5. This is the honest motivation '
     'for a trust gate even where SR wins on PSNR.',
     ['denominators_stated']),
    ('F6', 'f6_REPORT.md cites config_sha256 `7ee83389...0111e7e6`', D,
     'STALE. That is the hash of configs/fix.yaml as originally pre-registered (commit 020880f) -- the file '
     'F6 legitimately computed against -- but the merged branch\'s file hashes to 18cce7b1.... f6.json WAS '
     'amended (commit 6d2f722) and the report was not. Downgrade, not drop: F9 confirmed by yaml.safe_load '
     'that the pre-fix file does NOT parse at all and that the two post-fix variants are semantically '
     'identical, so no field F6 reads changed.',
     ['every_result_carries_fix_yaml_hash']),
    # ---------------------------------------------------------------- F7
    ('F7', 'per-class report: NO_CHANGE 4,733,514 / CORE 17,648 / ALLOCATED 239,122 / UNSUPPORTED 46,640 / '
           'NO_DATA 205,956 px, with m2 and km2', K,
     'Every m2 = px x 6.25 and every km2 = m2/1e6 verified exactly; the five classes sum to 5,242,880 px, '
     'which equals the 2560 x 2048 grid exactly; the valid-pixel denominator is stated explicitly and the '
     'file warns that class percentages must be taken against valid px, not the total.',
     ['denominators_stated']),
    ('F7', 'mapped change (v2) = CORE u ALLOCATED = 256,770 px = 1.604812 km2', D,
     'The ARITHMETIC is exact (CORE+ALLOCATED = 256,770; x 6.25 / 1e6 = 1.6048125 km2, matching to 1e-9). '
     'Downgraded as a SUBSTANTIVE claim, not for any arithmetic defect: the detector that produced it failed '
     'its own pre-registered FAR keep rule (0.0652 vs <= 0.05), F9 confirmed the true FAR is worse still '
     '(n_pre mismatch), and F5\'s realistic-fraction result (-0.313 IoU) is a direct prior against the '
     'sub-pixel allocation. Quotable only as "what this gate configuration produces", exactly as F7 says.',
     ['denominators_stated', 'negative_results_present_in_claim_table']),
    ('F7', 'v1 vs v2 mapped change area: 0.734900 -> 1.604812 km2 (+0.869912), IoU 0.2448; in v1 not v2 '
           '43,960 px, in v2 not v1 183,146 px', K,
     'IoU independently recomputed from the intersection as 0.24481761048116252, matching exactly; the '
     'difference in px (+139,186) and both set differences reproduce exactly. The comparison is well defined: '
     'both sides are the same quantity (the set of 2.5 m px the product maps as change) on the same grid, '
     'crop, dates and denominator.',
     ['no_row_compares_classes_with_different_definitions', 'denominators_stated']),
    ('F7', 'UNSUPPORTED is NOT compared across v1 and v2 because the definitions differ; the guard '
           'assert_well_defined_comparison enforces it', K,
     'F9 invoked the guard directly. It is ALLOW-LIST based (default deny) and raised ValueError on '
     '(v1:UNSUPPORTED, v2:UNSUPPORTED), on (v1:OBSERVED, v2:CORE), on (v1:INFERRED, v2:ALLOCATED), on the '
     'unregistered (v1:NO_CHANGE, v2:NO_CHANGE), and on the self-pair (v1:NO_DATA, v1:NO_DATA). compare_sets '
     '-- the only path to a v1-v2 number -- also raised on the UNSUPPORTED pair. Both counts are reported '
     'separately with their definitions and no difference, ratio or IoU exists between them in f7.json.',
     ['no_row_compares_classes_with_different_definitions', 'no_test_compares_quantity_with_itself']),
    ('F7', 'block-sum test on the real production map: max |deviation| = 0 over 327,680 blocks -> PASS', K,
     'Re-executed by F9 against the cached production state and non-tautological under sabotage (making '
     '`observed` a copy of the expected formula made this exact test FAIL). Recomputed independently of the '
     'gate\'s own call. Same limited evidential weight as F2\'s: a plumbing invariant, not a science result.',
     ['no_test_compares_quantity_with_itself']),
    ('F7', 'tau is used on the production map, not merely loaded: tau x0.5 -> 287,477 px, tau -> 256,770 px, '
           'tau x2 -> 123,803 px', K,
     'Strictly monotone and all three distinct; the fitted-tau count equals the reported mapped-change area '
     'exactly, so the sweep and the headline are the same computation. Supported by the required-positional '
     'signature and by F9\'s sabotage of the tau-application line. Caveat recorded as a finding: F7\'s own '
     'test reads these numbers back OUT of f7.json rather than re-running the gate, so that particular test '
     'cannot catch a code regression -- the claim stands on the trustsr-level mutation tests instead.',
     ['every_loaded_threshold_is_used_mutation_test']),
    ('F7', '1,120 real SR forward passes (35 tiles x 8 dihedral x 4 dates) and the run reproduces the cached '
           'product exactly (max abs diff 0.0 on all 4 dates)', K,
     'The pass count is an identity over tiles x 8 x dates that F7\'s test asserts, and the per-date '
     'reproduction check is NaN-aware with max abs diff 0.0. The SR ranking field varies within 100% of 4x4 '
     'blocks, so x9\'s np.repeat construction (B9) is excluded and would raise.',
     ['denominators_stated']),
    ('F7', 'the tau/n_pre mismatch (calibrated at n_pre=2, applied at n_pre=3) pushes the TRUE false-alarm '
           'rate ABOVE the already-failing 0.0652', K,
     'VERIFIED independently. sigma_f = sqrt(Var(a)(1/n_pre + 1)), so sigma_f(2) / sigma_f(3) = '
     'sqrt(1.5/1.3333) = 1.0606601717798212 -- F9 reproduced exactly the ratio f7.json reports. The '
     'production score is therefore ~6.07% higher than at calibration, i.e. the gate is MORE trigger-happy '
     'than the threshold was calibrated for, and the direction of the claim is correct. F7 does not correct '
     'for it and reports the more favourable number as the headline while disclosing this, so the headline '
     '0.0652 must be read as a LOWER BOUND on the gate\'s true FAR.',
     ['every_loaded_threshold_is_used_mutation_test', 'negative_results_present_in_claim_table']),
    ('F7', 'E5 cascade saves 0 forward passes (0.0%) on this scar-centred crop; 960 of 3,872 (24.8%) on the '
           'full AOI, where SR was NOT run', K,
     'Both counts are identities over tiles x 8 x dates that F7\'s tests assert, the crop baseline (1,120) is '
     'exactly the run performed, and the full-AOI row is explicitly flagged as a selection count with SR not '
     'run at that scale. The "0 saved" is an honest null about this crop.',
     ['denominators_stated', 'negative_results_present_in_claim_table']),
    ('F7', '3,946 of 4,412 boundary 10 m blocks (89.4%) have a 2.5 m allocation differing from the blocky '
           'parent', D,
     'Arithmetic and denominators are fine (314,671 blocks scored, 4,412 boundary, NO_DATA-touching blocks '
     'excluded from both sides). Downgraded because DIFFERING from a blocky parent is not evidence of being '
     'BETTER than one, and F5 measured that under exactly F7\'s realistic-fraction regime the allocation is '
     '0.313 IoU WORSE than blocky. F7 itself attaches that warning. Must never be presented as "SR adds '
     'information".',
     ['denominators_stated', 'negative_results_present_in_claim_table']),
    ('F7', 'F7 overall status = FAIL because the gate it runs failed its own FAR keep rule', K,
     'The correct call, and it is unmissable: the FAIL is in the first line, a blockquote warning sits above '
     'all numbers, the figure caption carries it, and F7\'s own test asserts the report states it within the '
     'first 1,500 characters. F7 also imports F5\'s negative as a prior against its own map.',
     ['negative_results_present_in_claim_table']),
]

CHECK_NAMES = [
    'point_estimate_inside_its_ci',
    'no_placebo_code_path_reads_post_event_date',
    'every_loaded_threshold_is_used_mutation_test',
    'no_test_compares_quantity_with_itself',
    'every_result_carries_fix_yaml_hash',
    'gates_in_a_table_differ_in_code',
    'denominators_stated',
    'no_row_compares_classes_with_different_definitions',
    'negative_results_present_in_claim_table',
]

CHECKS = {
    'point_estimate_inside_its_ci': {
        'method': 'experiments/f9_check_ci.py walks every f*.json, finds every (point estimate, CI) pair '
                  'under each convention actually used (lo/hi + point, ci95_lo/ci95_hi, and a 2-list ci key '
                  'with a point sibling) and evaluates lo <= point <= hi in code. 175 pairs found in total.',
        'F1': 'PASS - 76 CI pairs, 0 outside their own CI',
        'F2': 'PASS - 34 CI pairs, 0 outside',
        'F3': 'PASS - 15 CI pairs, 0 outside',
        'F5': 'PASS - 40 CI pairs, 0 outside',
        'F6': 'PASS - 9 CI pairs, 0 outside',
        'F7': 'PASS - 1 CI pair (the keep-rule failure block), inside',
        'note': 'A first pass flagged one apparent violation in f3.json keep_rule; it was F9\'s own pairing '
                'heuristic, because that node carries TWO point estimates (normalised and un-normalised '
                'coverage) and the bare `coverage` was being matched against `ci95_without_normalisation`. '
                'Paired correctly, both are inside their own CIs. The scanner was fixed rather than the '
                'finding being reported. This is the exact B4 defect that invalidated x3, and it does not '
                'recur anywhere in the fix wave.'},
    'no_placebo_code_path_reads_post_event_date': {
        'method': 'experiments/f9_try_break_guard.py does not read the guard and reason about it -- it '
                  'ATTEMPTS 16 loads of the post-event date through every placebo/calibration entry point, '
                  'using the same real config loader the experiments use.',
        'F1': 'PASS - guarded_load_date, build_sr_fold (post as held_out AND smuggled into pre_dates) and '
              'build_10m_fold all raised PostEventDateBlocked; malformed spellings (2024-7-30, 20241206, '
              '2024/12/06, 9999-01-01) fail CLOSED; the pre-event controls 2024-07-29 and 2024-01-16 still '
              'load, so the guard is not merely always-raise; the boundary is exact.',
        'F2': 'PASS - f2_gate_v2.build with the post date monkeypatched into SR_POOL raised '
              'PostEventDateBlocked. This is the function F7\'s recompute_tau calls, so the tau/calibration '
              'path is covered.',
        'F3': 'PASS - F3 reads only the cached pre-date dihedral products; its LODO date list is the 3 pre '
              'dates and no post-date path exists in it.',
        'F5': 'N/A - F5 uses opensr-test HR imagery, not the Wayanad time series.',
        'F6': 'N/A - F6 is a pure recomputation over x2.json per-image PSNR values; no imagery is loaded.',
        'F7': 'PASS with the correct exception - F7 loads the post date DELIBERATELY for the production map '
              'via experiments.wayanad_evidence.data.load_date (a different, intentional route), while '
              'recompute_tau goes through f2_gate_v2.build and therefore through the guard. The two paths '
              'are separate and F7 documents this.',
        'bypass_audit': 'Two UNGUARDED dihedral loaders exist (trustsr.placebo_v2._load_dihedral and '
                        'experiments.f2_gate_v2.load_dihedral, both raw np.load on a date-templated path). '
                        'They are provably unreachable with a post date: in every caller the guarded 10 m '
                        'load over the same date list executes FIRST and raises. Worth hardening, not a '
                        'live defect.'},
    'every_loaded_threshold_is_used_mutation_test': {
        'method': 'For each loaded threshold, confirm a mutation test exists, RUN it, then SABOTAGE the line '
                  'that applies the threshold and confirm the test now FAILS (the does-the-test-test-anything '
                  'check). Every sabotage was reverted and `git diff` confirmed empty afterwards.',
        'F1': 'PASS - k and the parent threshold move the gate outputs; the synthetic-fixture test '
              'discriminates rule_10m / ungated_S / gate_v1 pixel-for-pixel.',
        'F2': 'PASS - tau: replacing `win_detected = win_valid & (win_scores > tau)` with '
              '`win_valid & isfinite(win_scores)` (x9\'s B10 bug) made BOTH TestTauMutation tests fail '
              '(test_changing_tau_changes_the_output_map, test_detection_is_monotone_in_tau). lambda: '
              'forcing the lam==0 early return always made '
              'test_lambda_changes_which_sub_pixels_are_chosen_but_never_how_many fail. Both thresholds are '
              'genuinely wired and neither test is vacuous.',
        'F3': 'PASS - k = 2 drives the reported coverage, and the with/without-normalisation arms differ by '
              '8.4 pp (94.2% vs 85.8%), so the fitted offsets demonstrably change the output.',
        'F5': 'PASS - the NDVI truth threshold is exercised at 0.25 / 0.35 / 0.45 and every one produces a '
              'different delta and n, so the threshold is used, not decorative.',
        'F7': 'PASS - tau x0.5 / tau / tau x2 give 287,477 / 256,770 / 123,803 flagged px, strictly '
              'monotone, and the fitted-tau count equals the headline mapped-change area exactly. FINDING: '
              'F7\'s own test asserts these values by reading them back out of f7.json, so it cannot catch a '
              'code regression; the claim rests on the trustsr-level sabotage-verified tests instead.',
        'verdict': 'No unused loaded parameter found anywhere in F1-F7, so the stop rule\'s '
                   '"unused loaded parameter -> DROP" branch does not fire for any claim.'},
    'no_test_compares_quantity_with_itself': {
        'method': 'Audit both sides of every key equality/comparison for independent derivation, then prove '
                  'non-tautology by sabotage rather than by inspection.',
        'F1': 'PASS - window FAR takes each gate\'s OWN flagged array as an explicit argument (B3\'s fix); '
              'the pooled-CI test checks the pooled estimate against a hand-computed sum(num)/sum(den) over '
              'all folds, not against itself.',
        'F2': 'PASS, sabotage-proven - block_sum_deviation derives `observed` by counting CORE|ALLOCATED px '
              'in the OUTPUT class map and `expected` by recomputing round(16f) from the unmixing fraction; '
              'the allocator\'s n_per_block is not a parameter and a signature test asserts that. F9 '
              'rewrote `observed` into a second copy of the expected formula (exactly x4\'s B7 bug) and '
              'test_the_check_is_not_tautological_it_can_fail FAILED. This is the direct B7 regression check '
              'and it holds.',
        'F3': 'PASS with a declared exception - median(d) = 0 per fold IS definitional, and F3 names it as '
              'such in its limitations and rests the keep rule on coverage instead, where sigma comes from '
              'the pool\'s across-date and dihedral variance and never from d.',
        'F5': 'PASS - the two alloc_* methods receive the IDENTICAL fraction and allocator but rank by two '
              'independently built fields (bilinear upsample of the 10 m index vs the real SR forward-pass '
              'index). A self-comparison would return exactly 0.0; the measured delta is non-zero on 87 of '
              '118 images.',
        'F6': 'PASS - F6\'s pooled number is checked against x2.json\'s own independently computed pooled '
              'block, and F9 recomputed it a third time by three further routes.',
        'F7': 'PASS, sabotage-proven - the same block-sum sabotage made '
              'test_block_sum_on_the_real_map_is_not_tautological fail on the REAL production map; and '
              'test_no_registered_comparison_compares_a_quantity_with_itself asserts every accepted pair is '
              'v1-vs-v2, with the guard refusing a self-pair when F9 fed it one.'},
    'every_result_carries_fix_yaml_hash': {
        'method': 'F9 computed hashlib.sha256(open("configs/fix.yaml","rb").read()).hexdigest() itself and '
                  'compared byte-for-byte against each f*.json config_sha256, then traced the hash lineage '
                  'across every commit that touched the file.',
        'f9_computed_fix_yaml_sha256': CONFIG_SHA,
        'F1': 'PASS - matches exactly', 'F2': 'PASS - matches exactly',
        'F3': 'PASS in f3.json (matches exactly); FAIL in f3_REPORT.md, which still cites the stale, '
              'truncated `192345cd156583c6...`',
        'F5': 'PASS - matches exactly', 'F6': 'PASS in f6.json (matches exactly); FAIL in f6_REPORT.md, '
              'which still cites the stale `7ee83389...0111e7e6`',
        'F7': 'PASS - matches exactly',
        'lineage_verified': {
            '020880f (F0 pre-registration)': '7ee83389827340ed4e40b03917a7278d6bad11e2a6dac68460e8596c0111e7e6'
                                             ' -- does NOT parse (yaml.safe_load raises ParserError)',
            '298e214 (F3\'s YAML fix)': '192345cd156583c61789d171b673ecbd00347eaf8f485811f6abfb9ad5a905b3',
            '0da15a3 (F1\'s commented YAML fix, and the merged branch)':
                '18cce7b1ea88855bc054b42e6cbe3219eef73b2909eb091d07d5ada3df170970',
            'semantic_equality': 'yaml.safe_load(298e214) == yaml.safe_load(0da15a3) is True, so the two '
                                 'independent fixes differ only in a comment; the F0 original genuinely '
                                 'does not parse, which validates the reason for the fix.'},
        'amendment_honesty': 'BOTH "F0 amend" commits are honest. 6d2f722 (f6.json) and 0059c2f (f3.json) '
                             'each change exactly 2 lines: the config_sha256 value and a NEW '
                             'config_sha256_note explaining why. No other field was touched in either '
                             'commit, and F9 verified this by reading the full diffs, not the messages.'},
    'gates_in_a_table_differ_in_code': {
        'method': 'Compare the __code__ objects of all 5 placebo_v2.GATES entries pairwise (the B2 pattern), '
                  'then build the synthetic fixture that should force the suspicious pair apart.',
        'F1': 'PASS - 5 gate names, 5 DISTINCT code objects, 0 pairs sharing one, so no B2 regression. F1\'s '
              'claim that gate_v1 and gate_v1_with_a5_sigma coincide on the real crop BY DESIGN (rather than '
              'by aliasing) is VERIFIED: F9 built a fixture in which only sigma_a5 is degenerate and the two '
              'diverge by 2,048 px (4,096 vs 2,048 flagged), the mirrored fixture flips the asymmetry '
              '(proving each reads its own sigma field), and on all-finite data with sigmas differing 3x '
              'both equal the parent mask exactly -- reproducing the stated mechanism '
              '(flagged = OBSERVED u INFERRED = parent wherever d and sigma are finite). FINDING: F1\'s own '
              'shipped fixture NaNs BOTH sigmas in the same quadrant, so it separates rule_10m from gate_v1 '
              'but does NOT separate gate_v1 from gate_v1_with_a5_sigma. F9 supplied the missing fixture; '
              'F1\'s test suite has that gap.',
        'F2': 'PASS - gate_v2 is a distinct function that calls apply_gate_v2 and raises rather than falling '
              'back to any other gate when gate_v2_params are absent. The legacy alias '
              '`gate_v2_blocked = gate_v2` exists but is NOT a second row of the gate table, so it is not a '
              'B2 instance.',
        'F5': 'PASS - the 6 method rows are distinct code paths; the two alloc_* rows deliberately share the '
              'allocator and differ in the ranking field, which is the thing under test and is verified to '
              'differ.',
        'F3': 'N/A - no gate table.', 'F6': 'N/A - no gate table.',
        'F7': 'PASS - F7 runs the single F2 gate; no multi-gate table.'},
    'denominators_stated': {
        'method': 'Scripted: require an n-like sibling key for every CI-bearing node in the JSONs, then scan '
                  'every report for a percentage or ratio whose line and enclosing table header carry no n.',
        'F1': 'PASS - every FAR row states both pixel n (valid px) and window n (windows); 0 of 76 CI nodes '
              'lack an n sibling.',
        'F2': 'PASS - every rate is reported as "(num X / den Y, Z blocks)"; 0 of 34 CI nodes lack an n.',
        'F3': 'PASS with a documentation gap - f3.json carries n_px on every pooled and per-fold node, and '
              'the report states "2,358,286 stable px per fold" in prose immediately above the headline '
              'table, but the headline coverage table itself has no n column, and 4 summary nodes duplicate '
              'a number whose n lives in the detailed node. No bare, unattributable ratio.',
        'F5': 'PASS - every table carries an "n images" column; the n=118 vs 119 discrepancy is explained.',
        'F6': 'PASS - every row carries its image count; the pooled row states n = 178.',
        'F7': 'PASS - the class table gives pixels, m2 and km2 with an explicit denominator line '
              '(5,242,880 px in the crop, 5,036,924 valid) and an instruction to take percentages against '
              'valid px; the one summary CI node without an n sibling is the keep-rule block, whose n = 981 '
              'windows is stated in f2.json and in the report prose.'},
    'no_row_compares_classes_with_different_definitions': {
        'method': 'F9 invoked F7\'s guard directly with matched, mismatched, unregistered and self-paired '
                  'class names, and also drove the numeric path.',
        'F7': 'PASS - the guard is an ALLOW-LIST (default deny). It accepted only the two registered pairs '
              '(v1:OBSERVED|INFERRED vs v2:CORE|ALLOCATED, and v1:NO_DATA vs v2:NO_DATA) and raised '
              'ValueError("refusing to compare") on (v1:UNSUPPORTED, v2:UNSUPPORTED), (v1:OBSERVED, '
              'v2:CORE), (v1:INFERRED, v2:ALLOCATED), the unregistered (v1:NO_CHANGE, v2:NO_CHANGE) and the '
              'self-pair (v1:NO_DATA, v1:NO_DATA). compare_sets -- the only route to a v1-v2 number -- also '
              'raised, and a monkeypatch test proves compare_sets always calls the guard. UNSUPPORTED was '
              'therefore correctly EXCLUDED: f7.json holds no difference, ratio or IoU between the two '
              'UNSUPPORTED counts, and a test asserts those keys are absent.',
        'F1': 'N/A', 'F2': 'N/A', 'F3': 'N/A', 'F5': 'N/A (all methods scored against one truth definition)',
        'F6': 'N/A'},
    'negative_results_present_in_claim_table': {
        'method': 'Regex-locate each required negative in each REPORT.md and record HOW EARLY it appears, so '
                  '"present" cannot mean "buried in an appendix".',
        'F1': 'PASS - PASS status at 3% in, gate_v2 BLOCKED at 12%, and a Limitations section that states '
              'the N=3 degrees-of-freedom problem and the unequal rule_10m pool without arguing them away.',
        'F2': 'PASS - "Status: FAIL" is in the first line (char 70); the 0.0652 number and "NOT MET" appear '
              'in the Verdict section, and the Limitations list the ~8x endmember sensitivity, the '
              'uncalibrated f, the lam=0 fit and the UNSUPPORTED artefact.',
        'F3': 'PASS - KEPT verdict and the 85.8% un-normalised contrast both in the first quarter; the '
              'definitional median(d)=0 tautology is self-declared.',
        'F5': 'PASS - NAIP dependence at 30% under an explicit "Reading this honestly" heading, the null '
              'without NAIP right after, the -0.313 realistic-fraction result under its own heading '
              '("allocation is worse than not allocating at all"), and both repeated in the Bottom line.',
        'F6': 'PASS - B14 named in the first line, venus -0.559 and the pooled -0.097 both in the first '
              'half, and the whole document exists to publish the omitted negatives.',
        'F7': 'PASS, the strongest of the six - FAIL in the first line, a blockquote warning above every '
              'number, F5\'s -0.313 imported as a prior at 7% in, the n_pre mismatch at 34%, the failure in '
              'the figure caption, and a test that asserts the report says so within its first 1,500 chars.',
        'verdict': 'F10 will have all the material. Every negative F9 was asked to look for is present and '
                   'prominent; none is buried.'},
}

CROSS_CUTTING = [
    {'finding': 'config_sha256 amendments are honest, but the REPORTS were not amended with the JSONs',
     'detail': 'The two "F0 amend" commits (6d2f722 for f6.json, 0059c2f for f3.json) each change exactly '
               'the config_sha256 plus a new explanatory note field and NOTHING else -- F9 read both full '
               'diffs. But f3_REPORT.md still prints `192345cd156583c6...` and f6_REPORT.md still prints '
               '`7ee83389...0111e7e6`, so for those two tasks the human-readable and machine-readable '
               'records disagree. All six f*.json carry the correct 18cce7b1... hash.',
     'severity': 'low (documentation)', 'action_for_f10': 'Correct both report headers; cite only 18cce7b1...'},
    {'finding': 'configs/fix.yaml as pre-registered did not parse, and the two independent fixes are '
                'semantically identical',
     'detail': 'yaml.safe_load raises ParserError on the F0 original (020880f). F1 and F3 each fixed the '
               'f9_checks block independently; F9 confirmed yaml.safe_load of F3\'s fix (298e214) equals '
               'yaml.safe_load of F1\'s fix (0da15a3) exactly, so the merged file differs from F3\'s only by '
               'a comment. No check name, threshold or semantic changed. The amendment story holds up.',
     'severity': 'informational', 'action_for_f10': 'None; record the lineage.'},
    {'finding': 'F1_REPORT.md contains a false factual assertion about a hash, reached by eyeballing',
     'detail': 'F1 asserts the hash 7ee83389...0111e7e6 "(66 hex chars) cannot be a valid SHA-256 digest". '
               'len() returns 64 and the string is valid hex -- and it is precisely the sha256 of '
               'configs/fix.yaml at the F0 pre-registration commit, i.e. the hash F6 correctly computed '
               'before the YAML fix. Both halves of the assertion are wrong. Nothing scientific depends on '
               'it, but it is the same species of error (asserting arithmetic from inspection) that '
               'f9_checks.arithmetic_checked_by exists to forbid.',
     'severity': 'low (but a process signal)', 'action_for_f10': 'Delete or correct that paragraph.'},
    {'finding': 'F2\'s and the ADR\'s "the e_b escalation made the keep rule HARDER" is not established',
     'detail': 'Measured, not argued: shortening the mixing line inflates f 2.08x, but the detection score '
               's = f/sigma_f FALLS (mean 20.36 -> 10.59) because sigma_f carries its own 1/|e_v-e_b| '
               'factor. Since tau is recomputed conformally from the same score, the calibration exceedance '
               'rate is pinned near alpha whatever e_b is, so the net test-split effect is indeterminate. '
               'This is NOT gaming -- the deviation was disclosed pre-hoc with the full 0.20-0.50 ceiling '
               'table and the keep rule FAILED anyway, so no pass was manufactured -- but the wording '
               'over-claims.',
     'severity': 'medium (an over-claimed mechanism in an ADR F10 will quote)',
     'action_for_f10': 'Restate as "direction not established"; keep the honest consequence (e_b is the '
                       'darkest land cover, not bare ground, so f is not a calibrated area).'},
    {'finding': 'the tau/n_pre mismatch makes F7\'s headline FAR a LOWER BOUND',
     'detail': 'Confirmed by F9: sigma_f(n_pre=2)/sigma_f(n_pre=3) = 1.06066, exactly the ratio f7.json '
               'reports, so the production gate scores ~6.07% higher than at calibration and is more '
               'trigger-happy than tau was calibrated for. F7 measures this, declines to correct it, and '
               'headlines the more favourable 0.0652.',
     'severity': 'medium', 'action_for_f10': 'Always present 0.0652 as a lower bound on this gate\'s true '
                                             'FAR, never as the FAR.'},
    {'finding': 'gate v2 is calibrated but does not generalise across the checkerboard split',
     'detail': 'Window FAR is 0.04893 on the CALIBRATION split (<= alpha, as split conformal guarantees by '
               'construction) and 0.0652 on the TEST split in the F3-normalised arm -- and 0.0540 on test '
               'even in the un-normalised arm, so BOTH arms fail. The failure is therefore a real '
               'exchangeability/generalisation gap across three folds sharing one crop, exactly the caveat '
               'configs/fix.yaml units.window_caveat pre-registered, not an arithmetic slip.',
     'severity': 'high (this is the substantive scientific finding of the wave)',
     'action_for_f10': 'State that the gate\'s conformal guarantee does not survive the spatial split, and '
                       'that the FAIL is robust to the normalisation arm.'},
    {'finding': 'lambda = 0.0 means the neighbour-agreement term contributes nothing to any reported result',
     'detail': 'lambda is genuinely wired (F9 sabotage-verified its mutation test) and 0.0 is a fitted '
               'optimum, not a default, so the stop rule does not fire and nothing is dropped. But no '
               'reported number benefits from the term, so no claim that neighbour agreement improves '
               'allocation is supported. F2 says this.',
     'severity': 'low', 'action_for_f10': 'Do not list neighbour agreement as a contributing component.'},
    {'finding': 'several F7 tests assert properties of f7.json rather than re-executing the code',
     'detail': 'Roughly half of tests/test_f7_wayanad_v2.py reads f7.json and asserts about its contents '
               '(including test_tau_is_used_not_merely_loaded). Those would still pass if the code rotted, '
               'so long as the result file is unchanged. The three block-sum tests DO recompute from the '
               'cached production state and are genuine. Every F7 claim F9 kept is independently supported '
               'by trustsr-level tests that F9 sabotage-verified, so no verdict changes.',
     'severity': 'low (test hygiene)', 'action_for_f10': 'Note as future work; not blocking.'},
    {'finding': 'two unguarded dihedral loaders exist, currently unreachable with a post-event date',
     'detail': 'trustsr.placebo_v2._load_dihedral and experiments.f2_gate_v2.load_dihedral are raw np.load '
               'calls on a date-templated path with no date guard. In every present caller the guarded 10 m '
               'load over the same date list runs first and raises, which F9 confirmed by attack. A future '
               'caller that loaded only the SR product would bypass B1\'s guard entirely.',
     'severity': 'low now, latent', 'action_for_f10': 'Recommend routing both through the guard.'},
    {'finding': 'the block-sum PASS is a plumbing invariant, not a scientific result',
     'detail': 'It is genuinely non-tautological -- F9 proved that by sabotage in both F2 and F7 -- but '
               'allocate_pixels enforces exactly round(16f) by construction (`pos < n_target`), so the test '
               'can only fail if class assignment loses pixels. It verifies the allocator and the class map '
               'agree; it says nothing about whether the allocation is correct. F5\'s -0.313 is the claim '
               'that speaks to that, and it is negative.',
     'severity': 'informational',
     'action_for_f10': 'Do not present "block-sum PASS" as evidence the mapping is good.'},
    {'finding': 'one pre-existing test failure on the branch, outside the fix wave',
     'detail': 'tests/test_placebo.py::TestFARPixel::test_far_pixel_basic FAILS (asserts estimate > 0.4, '
               'gets 0.0). It tests the quarantined legacy trustsr/placebo.py, which the fix wave never '
               'modified (git log over 49a135f..11de65f for that module and test is empty), so it is '
               'pre-existing and touches no F1-F7 claim. Full suite: 329 passed, 1 failed, 1 skipped.',
     'severity': 'low', 'action_for_f10': 'Either fix or explicitly quarantine that legacy test.'},
]


def main():
    counts = Counter(v for _, _, v, _, _ in CLAIMS)
    by_task = {}
    for t, _, v, _, _ in CLAIMS:
        by_task.setdefault(t, Counter())[v] += 1

    doc = {
        'adversarial_verifier': 'F9',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'config_sha256': CONFIG_SHA,
        'config_sha256_source': 'hashlib.sha256 of configs/fix.yaml, computed by F9, not copied from any '
                                'f*.json or from the task brief',
        'supersedes': 'experiments/results/x10.json + x10_REPORT.md (INVALIDATED, B13)',
        'worktree': {
            'path': '.claude/worktrees/fix-v2-f9', 'branch': 'task/f9',
            'head_at_start': '11de65f',
            'external_artifacts_present': [
                'data -> shared Sentinel-2 / experiments cache (symlink, NOT in git history). Required by '
                'F1/F2/F3/F7 and by the F7 block-sum tests, which read '
                'data/experiments-cache/f7_production_state.npz.',
                'models -> pretrained SEN2SR-lite weights (symlink, NOT in git history).',
                'python env: the worktree has no .venv, so the parent repository\'s '
                '/Users/devariwala/Desktop/Focal/.venv (Python 3.11.15, numpy/torch/scipy; NO pandas) was '
                'used. This is a third external dependency and is recorded for completeness.'],
            'tracked_files_modified_by_f9': 'none (git diff empty at completion); F9 wrote only '
                                           'experiments/results/f9.json, f9_REPORT.md and experiments/f9_* '
                                           'scratch scripts'},
        'summary': {
            'n_claims_audited': len(CLAIMS),
            'keep': counts[K], 'downgrade': counts[D], 'drop': counts[X],
            'counts_sum_to_total': counts[K] + counts[D] + counts[X] == len(CLAIMS),
            'per_task': {t: dict(c) for t, c in sorted(by_task.items())},
            'mandatory_checks_run': len(CHECK_NAMES),
            'self_check_note': 'These counts are computed from the claims list by experiments/f9_emit.py, '
                               'not typed by hand. A10 was invalidated (B13) partly for stating 3/5 KEEP '
                               'while listing 2, so F9 makes that arithmetic impossible to get wrong.',
            'headline': 'All five experiments\' self-reported statuses are CORRECT: F1 PASS, F2 FAIL, F3 '
                        'PASS, F5 PASS-with-caveats, F6 PASS, F7 FAIL. F9 found no fabricated number, no '
                        'tautological test, no unused loaded parameter and no post-hoc threshold movement. '
                        'Every arithmetic claim F9 recomputed reproduced. The defects found are one false '
                        'factual assertion about a hash (DROPPED), one over-claimed deviation mechanism '
                        '(DOWNGRADED), two stale hashes in reports (DOWNGRADED), and two F7 numbers that '
                        'are arithmetically right but must not be read as detection quality (DOWNGRADED).'},
        'claims_audited': [
            {'task': t, 'claim': c, 'verdict': v, 'reason': r, 'checks_bearing_on_it': ch}
            for t, c, v, r, ch in CLAIMS],
        'checks': CHECKS,
        'cross_cutting_findings': CROSS_CUTTING,
        'reproducibility': {
            'test_suites_rerun_in_this_fresh_worktree': {
                'command': 'PYTHONPATH=. python -m pytest tests/test_f1_placebo.py tests/test_f3_noise_v2.py '
                           'tests/test_gate_v2.py tests/test_f5_a8_mapping.py tests/test_alloc.py '
                           'tests/test_f7_wayanad_v2.py tests/test_f6_a2_report.py -v',
                'result': '114 passed in 1.08 s, 0 failed, 0 skipped',
                'note': 'These are fast synthetic-fixture unit tests plus three that recompute from the '
                        'cached F7 production state; they are not a re-execution of the experiments.'},
            'full_repo_suite': '329 passed, 1 failed, 1 skipped. The single failure '
                               '(tests/test_placebo.py::TestFARPixel::test_far_pixel_basic) is in the '
                               'quarantined legacy placebo module, untouched by the fix wave, and affects no '
                               'F1-F7 claim.',
            'sabotage_experiments': [
                'tau application in apply_gate_v2 replaced by an always-detect expression -> both '
                'TestTauMutation tests FAILED, so they are not vacuous. Reverted.',
                'block_sum_deviation `observed` replaced by a copy of the `expected` formula (x4\'s B7 '
                'self-comparison) -> test_the_check_is_not_tautological_it_can_fail AND '
                'test_block_sum_on_the_real_map_is_not_tautological both FAILED. Reverted.',
                'allocate_pixels forced to ignore lam -> '
                'test_lambda_changes_which_sub_pixels_are_chosen_but_never_how_many FAILED. Reverted.',
                'After every sabotage, trustsr/gate_v2.py was restored from a byte copy and `git diff '
                '--stat` confirmed empty.'],
            'independent_recomputations': {
                'F6 pooled PSNR delta': 'Recomputed from x2.json per-image records by three routes that do '
                                        'not use experiments/f6_a2_report.py (vectorised numpy, a longhand '
                                        'Python loop, and a per-dataset weighted recombination), agreeing to '
                                        '< 1e-12 at -0.09707162370791389. An independently written paired '
                                        'bootstrap reproduced [-0.1574637956167713, -0.03696577159967953]; '
                                        'a normal-approximation CI brackets it. Per-dataset (all 5), '
                                        'NAIP-excluded (-0.275, n=116) and all ten ha_metric means also '
                                        'reproduced.',
                'F5 headline': 'Rebuilt the paired delta from f5.json per-image IoU records with F9\'s own '
                               'bootstrap: all-datasets +0.008078 CI [+0.00347, +0.01410] n=118 87 wins '
                               '(report +0.0081 [+0.0035, +0.0141]); excluding NAIP +0.001592 '
                               'CI [-0.00261, +0.00532] n=56 30 wins, CI spans zero (report +0.0016).',
                'F1/F2 FARs': 'Every reported FAR recomputed as numerator/denominator and matched to 1e-6: '
                              'rule_10m 0.00026546675692536403; ungated_S 0.4032900483155881; gate_v1 and '
                              'gate_v1_with_a5_sigma both 0.0004953176998684312 with identical numerator '
                              '1792; gate_v2 window 0.065239551 = 64/981 and pixel 0.003639424 = '
                              '13167/3617880.',
                'F3 coverage': 'Pooled coverage recomputed as covered/n (0.9418016423231794 = '
                               '6663111/7074856 and 0.8579530099269865 = 6069894/7074856), gap_pp '
                               'recomputed, pooled shown equal to the per-fold n-weighted mean, and fold n '
                               'shown to sum to the pooled n -- all exact.',
                'F7 areas and IoU': 'Every m2 = px x 6.25 and km2 = m2/1e6 verified; the five class counts '
                                    'sum to 5,242,880 = 2560 x 2048 exactly; mapped change = CORE + '
                                    'ALLOCATED = 256,770 px = 1.6048125 km2; v1-vs-v2 IoU recomputed from '
                                    'the intersection as 0.24481761048116252, matching exactly.',
                'F2 tau order statistic': 'k = ceil((n+1)(1-alpha)) = ceil(982 x 0.95) = 933 for n = 981, '
                                          'consistent with the reported T_(933).'},
            'sha256_comparisons': {
                'configs/fix.yaml (F9-computed)': CONFIG_SHA,
                'all six f*.json config_sha256': 'MATCH (f1, f2, f3, f5, f6, f7)',
                'f3_REPORT.md / f6_REPORT.md cited hashes': 'MISMATCH (stale; see cross-cutting findings)',
                'data/experiments-cache/f1_calibration_scores.npz':
                    '53a2fe8a3a7852d7648d6d8d17abaf982b1435d23ffb0afcd9fda8effd2679e4 -- F9 recomputed it '
                    'and it MATCHES both f1.json\'s export claim and f2.json\'s consumption claim, so the '
                    'F1 -> F2 artefact chain is verified end to end.',
                'verbatim tau from that npz': 'F9 recomputed 119.90589141845703, matching f2.json\'s '
                                              'tau_if_taken_verbatim_from_this_file exactly.'},
            'f9_scratch_scripts': ['experiments/f9_check_ci.py', 'experiments/f9_try_break_guard.py',
                                   'experiments/f9_check_gates.py', 'experiments/f9_recompute_f6.py',
                                   'experiments/f9_spotcheck.py',
                                   'experiments/f9_interrogate_deviations.py', 'experiments/f9_emit.py']},
        'risk_summary': {
            'overall': 'The fix wave is materially sound and materially honest. Unlike the x-wave it '
                       'replaces, F9 could not find a single fabricated number, tautological test, unused '
                       'threshold or post-hoc threshold movement, and every number recomputed reproduced. '
                       'The scientific bottom line is negative and correctly reported: gate v2 does not '
                       'meet its false-alarm rule, and sub-pixel allocation does not help once the fraction '
                       'must be estimated from Sentinel-2.',
            'top_risks_for_f10': [
                'Quoting F7\'s 1.60 km2 mapped-change area as a detection. It is what a gate that FAILED its '
                'FAR rule produces, and the true FAR is worse than measured.',
                'Quoting F5\'s +0.0081 headline without "NAIP included" attached. Without NAIP -- the one '
                'dataset with an unexcludable train/test overlap -- the effect is indistinguishable from '
                'zero at n=56.',
                'Repeating the ADR\'s claim that the e_b escalation made the keep rule harder. F9 measured '
                'the mechanism and it does not hold.',
                'Presenting "block-sum deviation 0" as evidence of mapping quality. It is a plumbing '
                'invariant; F5\'s -0.313 IoU is the quality evidence and it is negative.',
                'Citing the stale config hashes still printed in f3_REPORT.md and f6_REPORT.md.'],
            'what_would_change_these_verdicts': 'A real re-execution of F1/F2/F7 end to end (F9 reran test '
                                                'suites and recomputed from stored artefacts, but did not '
                                                'regenerate the SR stack or the placebo folds from '
                                                'imagery); and an independent check of the E1/E8 cached '
                                                'dihedral product itself, which every SR-dependent number '
                                                'in F1/F2/F7 inherits without re-deriving.'},
        'pass_fail_logic': {
            'stop_rule_applied': 'configs/fix.yaml f9_checks.stop_rule -- "a tautological test or an unused '
                                 'loaded parameter -> that claim is DROPPED, not downgraded".',
            'stop_rule_fired': 'NO. F9 searched specifically for both triggers and found neither. Every '
                               'candidate was tested by sabotage rather than by inspection: the two '
                               'block-sum tests (F2 and F7) both FAIL when made self-comparing, so they are '
                               'not tautological; tau and lambda both have mutation tests that FAIL when '
                               'the applying line is disabled, so neither is an unused loaded parameter. '
                               'F3\'s median(d) = 0 IS definitional, but F3 declares it as such and rests '
                               'its keep rule on coverage instead, so no CLAIM rests on a tautology -- had '
                               'F3 offered it as evidence, that claim would have been DROPPED.',
            'the_one_drop': {
                'claim': 'F1_REPORT.md: the hash 7ee83389...0111e7e6 "(66 hex chars) cannot be a valid '
                         'SHA-256 digest".',
                'why_dropped_not_downgraded': 'A downgrade is for a claim that is true but weaker than '
                                              'stated. This claim is simply false in both of its '
                                              'assertions: len() returns 64, the string is valid hex, and '
                                              'it is the exact sha256 of configs/fix.yaml at the F0 '
                                              'pre-registration commit -- the hash F6 legitimately computed '
                                              'before the YAML fix. There is no weaker true version to '
                                              'retreat to, so it is dropped. It is also the one place in '
                                              'the wave where a quantitative assertion was made by '
                                              'inspecting a string instead of measuring it, which '
                                              'f9_checks.arithmetic_checked_by forbids.',
                'consequence': 'None scientific. F1\'s own config_sha256 is correct and independently '
                               'verified; only the paragraph dismissing the other hash must go.'},
            'why_the_downgrades_are_not_drops': [
                'F2 e_b-deviation direction: the deviation was disclosed PRE-HOC with the full 0.20-0.50 '
                'ceiling table and the keep rule FAILED anyway, so it cannot have been used to make a '
                'failing result look better. Only the stated mechanism is wrong, and no number depends on '
                'it.',
                'F3 and F6 report hashes: stale, not wrong at origin -- each was the true hash of the file '
                'that task actually read, the JSONs were honestly amended, and F9 verified the YAML variants '
                'are semantically identical, so no result is affected.',
                'F7 mapped-change area and boundary-divergence count: the arithmetic is exact and '
                'reproduced. They are downgraded on INTERPRETATION -- the detector failed its FAR rule and '
                'F5 is a prior against the allocation quality -- not on correctness, so the numbers remain '
                'citable with their attached caveat.'],
            'statuses_confirmed': 'F1 PASS, F2 FAIL, F3 PASS/KEPT, F5 PASS-with-major-caveat, F6 PASS, '
                                 'F7 FAIL -- all six self-reported statuses are CORRECT as stated.'},
    }
    p = ROOT / 'experiments/results/f9.json'
    p.write_text(json.dumps(doc, indent=2) + '\n', encoding='utf-8')
    print(f'wrote {p}')
    print(f'claims={len(CLAIMS)} keep={counts[K]} downgrade={counts[D]} drop={counts[X]} '
          f'sum_ok={counts[K]+counts[D]+counts[X]==len(CLAIMS)}')
    for t, c in sorted(by_task.items()):
        print(f'   {t}: ' + ', '.join(f'{k}={v}' for k, v in sorted(c.items())))


if __name__ == '__main__':
    main()
