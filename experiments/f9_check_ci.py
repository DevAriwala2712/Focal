"""F9 verification scratch script (check 1 + check 7).

check_1 point_estimate_inside_its_ci : walk every f*.json, find every (point estimate, CI) pair under
  any of the conventions actually used in these files, and assert numerically lo <= point <= hi.
check_7 denominators_stated          : for every such CI-bearing node, require an n-like sibling key.

Nothing is eyeballed; every comparison is a float comparison done here.
"""
from __future__ import annotations

import glob
import json
import os

# point-estimate key -> the CI it belongs to, for the conventions present in f1..f7.json
POINT_KEYS = ('estimate', 'mean', 'coverage_k2', 'coverage', 'value_all_stable',
              'delta_vs_bicubic_psnr_db', 'delta_psnr_db', 'value')
N_KEYS = ('n', 'n_px', 'n_images', 'n_windows', 'numerator', 'denominator', 'den', 'n_blocks',
          'covered_px', 'n_pairs', 'n_valid_px', 'n_images_a_better', 'blocks', 'n_windows_total')


def ci_candidates(node: dict):
    """Yield (label, point, lo, hi) for every CI convention present in this dict."""
    # (a) lo/hi as sibling scalars + a point key
    if 'lo' in node and 'hi' in node and isinstance(node['lo'], (int, float)):
        for pk in POINT_KEYS:
            if pk in node and isinstance(node[pk], (int, float)):
                yield f'lo/hi+{pk}', float(node[pk]), float(node['lo']), float(node['hi'])
                break
    # (b) ci95_lo / ci95_hi
    if 'ci95_lo' in node and 'ci95_hi' in node:
        for pk in POINT_KEYS:
            if pk in node and isinstance(node[pk], (int, float)):
                yield f'ci95_lo/hi+{pk}', float(node[pk]), float(node['ci95_lo']), float(node['ci95_hi'])
                break
    # (c) a 2-list CI under a ci-ish key + a point key sibling.
    #     A node may carry MORE THAN ONE point estimate (f3's keep_rule holds both the normalised and the
    #     un-normalised coverage). A suffixed CI key must be paired with the point estimate carrying the
    #     same suffix, never with the bare one -- pairing them blindly manufactures a false violation.
    SUFFIX_PAIRS = {'ci95_without_normalisation': 'coverage_without_normalisation_same_sigma_method'}
    for k, v in node.items():
        kl = k.lower()
        if not (kl == 'ci' or kl.startswith('ci95') or kl.endswith('_ci') or 'ci95' in kl):
            continue
        if not (isinstance(v, list) and len(v) == 2
                and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v)):
            continue
        if kl in SUFFIX_PAIRS:
            pk = SUFFIX_PAIRS[kl]
            if pk in node:
                yield f'{k}+{pk}', float(node[pk]), float(v[0]), float(v[1])
            continue
        for pk in POINT_KEYS:
            if pk in node and isinstance(node[pk], (int, float)) and not isinstance(node[pk], bool):
                yield f'{k}+{pk}', float(node[pk]), float(v[0]), float(v[1])
                break


def walk(node, path, rows):
    if isinstance(node, dict):
        for label, pt, lo, hi in ci_candidates(node):
            has_n = [k for k in node if k in N_KEYS]
            rows.append({'path': path, 'convention': label, 'point': pt, 'lo': lo, 'hi': hi,
                         'inside': lo <= pt <= hi, 'n_keys': has_n})
        for k, v in node.items():
            walk(v, f'{path}/{k}', rows)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            walk(v, f'{path}[{i}]', rows)


def main():
    summary = {}
    for f in sorted(glob.glob('experiments/results/f*.json')):
        task = os.path.basename(f).split('.')[0].upper()
        rows = []
        walk(json.load(open(f)), '', rows)
        bad = [r for r in rows if not r['inside']]
        no_n = [r for r in rows if not r['n_keys']]
        summary[task] = {'n_ci_pairs_found': len(rows),
                         'n_outside_own_ci': len(bad),
                         'outside': [{k: r[k] for k in ('path', 'convention', 'point', 'lo', 'hi')} for r in bad],
                         'n_without_denominator_sibling': len(no_n),
                         'without_denominator_sample': [r['path'] for r in no_n[:15]]}
        print(f'{task}: {len(rows)} CI pairs, {len(bad)} OUTSIDE own CI, '
              f'{len(no_n)} lacking an n-sibling')
        for r in bad:
            print('   OUTSIDE:', r['path'], r['convention'], r['point'], r['lo'], r['hi'])
        for p in no_n[:15]:
            print('   no-n:', p)
    json.dump(summary, open('experiments/f9_ci_scan.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
