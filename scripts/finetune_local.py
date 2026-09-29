"""Produce a distinct local checkpoint from provenance-verified WorldStrat pairs."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from risk.common import digest, load_config, write_json
from risk.model import load_model
from risk.r5_finetune import load_crop, train_steps, validate_manifest
from trustsr.sr import spectral_project_torch


class VariableTileTraining(torch.nn.Module):
    """Use the pretrained CNN with differentiable parent constraint below 128px."""
    def __init__(self, sr_model):
        super().__init__()
        self.sr_model = sr_model

    def forward(self, x):
        return spectral_project_torch(x, self.sr_model(x).clamp_min(0))


def main():
    cfg, root, _ = load_config('configs/phase0.yaml')
    manifest_path = root / cfg['training']['pair_manifest']
    records = validate_manifest(json.loads(manifest_path.read_text(encoding='utf-8')),
                                manifest_path.parent, cfg['model']['bands'])
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU unavailable for local fine-tune')
    torch.manual_seed(cfg['model']['seed'])
    budget = cfg['model']['budget_gib'] * 2**30 / torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1, budget))
    size = cfg['model']['native_tile']
    while True:
        try:
            model = load_model(cfg, root, trainable=True).eval()
            samples = [tuple(torch.from_numpy(a[None]) for a in load_crop(p, size, 4)) for p in records]
            training_model = model if size == cfg['model']['native_tile'] else VariableTileTraining(model.sr_model)
            losses, delta, gradients = train_steps(training_model, samples, cfg['training']['steps'],
                                                   cfg['training']['learning_rate'], 'cuda')
            break
        except torch.cuda.OutOfMemoryError:
            del model
            torch.cuda.empty_cache()
            size //= 2
            if size < cfg['model']['min_tile']:
                raise RuntimeError('CUDA OOM at 64 px during local fine-tune') from None
    checkpoint = root / cfg['training']['checkpoint']
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    save_file({key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}, str(checkpoint))
    window = cfg['training']['comparison_window']
    result = {'runtime': 'local Windows RTX 4050', 'colab': False, 'status': 'PASS' if
              sum(losses[-window:]) < sum(losses[:window]) and delta > 0 and gradients else 'FAIL',
              'steps': len(losses), 'first_window_mean': sum(losses[:window]) / window,
              'last_window_mean': sum(losses[-window:]) / window, 'weight_l1_change': delta,
              'tile_size': size, 'checkpoint_sha256': digest(checkpoint),
              'pair_manifest_sha256': digest(manifest_path), 'aoi_ids': [r['aoi_id'] for r in records]}
    write_json(root / 'results' / 'local_finetune.json', result)
    print(json.dumps(result, indent=2))
    if result['status'] != 'PASS':
        raise RuntimeError('Local fine-tune did not meet smoke criteria')


if __name__ == '__main__':
    main()
