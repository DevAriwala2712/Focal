"""A2 diagnosis of WHY SR does not beat bicubic on some OpenSR-test sets (writes experiments/results/x2_diagnosis.json).

Candidate causes checked, each with the same paired-bootstrap CI as the main run:
  1. residual sub-pixel misalignment: PSNR after the best integer HR-pixel shift (+-3 px) chosen per image AND per method
  2. tile/border effects: PSNR delta with border 0, 16 (main) and 32 HR px excluded
  3. radiometric bias: mean signed error (candidate - reference) per band, SR vs bicubic
  4. sensor/scale gap: Venus is x2; also report bicubic x2 applied directly (not via x4 then 2x2 mean)
Run: .venv_a2/bin/python -m experiments.x2_diagnose  (needs the cached OpenSR-test pickles and models/ from the main run)
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import yaml

from experiments.x2_data import safe_load
from experiments.x2_hr_benchmark import build_samples, make_sr_fn
from risk.common import load_config, write_json
from trustsr import hrbench as hb
from trustsr.bootstrap import paired_bootstrap_ci

warnings.filterwarnings('ignore')


def shifted_psnr(ref, est, mask, ms):
    """Best PSNR over integer HR-px shifts of `est` in [-ms, ms]^2 (interior compared; identical protocol for every method)."""
    h, w = ref.shape[-2:]
    best = -1e9
    for dy in range(-ms, ms + 1):
        for dx in range(-ms, ms + 1):
            r = ref[:, ms:h - ms, ms:w - ms]
            e = est[:, ms + dy:h - ms + dy, ms + dx:w - ms + dx]
            m = mask[ms:h - ms, ms:w - ms]
            best = max(best, hb.psnr(r, e, m))
    return best


def main(config='configs/a2.yaml'):
    import torch
    root = Path(config).resolve().parent.parent
    a2 = yaml.safe_load((root / config).read_text())
    boot = {'replicates': 2000, 'ci': 0.95, 'seed': 2024}
    torch.manual_seed(a2['seed'])
    torch.use_deterministic_algorithms(True)
    from risk.model import load_model
    phase0, _, _ = load_config(root / a2['phase0_config'])
    model = load_model(phase0, root, trainable=False, device='cpu')
    core = make_sr_fn(lambda b: model(torch.from_numpy(np.ascontiguousarray(b, np.float32))).detach().numpy(), a2['tiling'],
                      torch.OutOfMemoryError, lambda: None)
    out = {'evidence': 'real', 'datasets': {}}
    for ds in a2['datasets']:
        d, _ = safe_load(root / a2['cache_dir'] / f'{ds}.pkl')
        items = build_samples(ds, d, a2)
        rec = {'shifted': [], 'border': {0: [], 16: [], 32: []}, 'bias': {'sr': [], 'bicubic': []}, 'x2_direct': [], 'shift_hist': {}}
        for _, s in items:
            sr = core(s.lr).astype(np.float64)
            ref_scale = s.hr.shape[-1] // s.lr.shape[-1]
            red = 4 // ref_scale
            cand = {'sr': sr, 'bicubic': hb.resample(s.lr, 4, 'bicubic')}
            cand = {k: np.clip(hb.block_mean(v, red) if red > 1 else v, 0, 1) for k, v in cand.items()}
            hr = s.hr.astype(np.float64)
            base = s.mask
            rec['shifted'].append(shifted_psnr(hr, cand['sr'], base, 3) - shifted_psnr(hr, cand['bicubic'], base, 3))
            for b in rec['border']:
                m = base.copy()
                if b:
                    m[:b], m[-b:], m[:, :b], m[:, -b:] = False, False, False, False
                rec['border'][b].append(hb.psnr(hr, cand['sr'], m) - hb.psnr(hr, cand['bicubic'], m))
            m16 = base.copy()
            m16[:16], m16[-16:], m16[:, :16], m16[:, -16:] = False, False, False, False
            for k in ('sr', 'bicubic'):
                rec['bias'][k].append((cand[k] - hr)[:, m16].mean(axis=1))
            g = hb.grid_shift(s.lr.astype(np.float64), hr, ref_scale, 4)
            rec['shift_hist'][f'{g["dy"]},{g["dx"]}'] = rec['shift_hist'].get(f'{g["dy"]},{g["dx"]}', 0) + 1
            if ref_scale == 2:
                direct = np.clip(hb.resample(s.lr, 2, 'bicubic'), 0, 1)
                rec['x2_direct'].append(hb.psnr(hr, cand['sr'], m16) - hb.psnr(hr, direct, m16))
        ci = lambda v: paired_bootstrap_ci(v, boot['replicates'], boot['ci'], boot['seed'])
        out['datasets'][ds] = {
            'n': len(items),
            'delta_psnr_shift_tolerant_pm3px': ci(rec['shifted']),
            'delta_psnr_by_border_px': {str(b): ci(v) for b, v in rec['border'].items()},
            'mean_signed_error_per_band': {k: np.mean(v, axis=0).tolist() for k, v in rec['bias'].items()},
            'best_shift_histogram_hr_px_dy_dx': dict(sorted(rec['shift_hist'].items(), key=lambda kv: -kv[1])[:8]),
            'delta_psnr_sr_vs_bicubic_x2_direct': ci(rec['x2_direct']) if rec['x2_direct'] else None}
        print(ds, json.dumps(out['datasets'][ds]['delta_psnr_shift_tolerant_pm3px']), flush=True)
    write_json(root / a2['results_dir'] / 'x2_diagnosis.json', out)


if __name__ == '__main__':
    main()
