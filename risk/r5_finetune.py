"""Fifty real fine-tuning steps; no synthetic substitute for WorldStrat/Colab."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

from risk.common import digest, retry_tiles, run_cli


def validate_manifest(manifest, base, bands, *, allowed_splits=('train',)):
    if manifest.get('dataset') != 'WorldStrat':
        raise ValueError('Only provenance-checked WorldStrat pairs are accepted')
    if not manifest.get('pairs'):
        raise ValueError('Manifest has no pairs')
    import rasterio
    records = []
    for entry in manifest['pairs']:
        for field in ['aoi_id', 'lr', 'hr', 'lr_sha256', 'hr_sha256', 'source_urls', 'split']:
            if not entry.get(field):
                raise ValueError(f'Missing pair provenance: {field}')
        if entry['split'] not in allowed_splits:
            raise ValueError(f"Pair split {entry['split']} is not allowed here; expected {allowed_splits}")
        lr_path, hr_path = base / entry['lr'], base / entry['hr']
        for key, path in [('lr', lr_path), ('hr', hr_path)]:
            if digest(path) != entry[key + '_sha256']:
                raise ValueError(f'{key} checksum mismatch')
        with rasterio.open(lr_path) as lr, rasterio.open(hr_path) as hr:
            if lr.count != 4 or hr.count != 4 or list(lr.descriptions) != bands or list(hr.descriptions) != bands:
                raise ValueError('Pair must contain exactly four named RGBN bands in model order')
            if lr.crs is None or lr.crs != hr.crs or lr.bounds != hr.bounds:
                raise ValueError('Pair CRS/bounds mismatch')
            if hr.width != 4 * lr.width or hr.height != 4 * lr.height or not hr.transform.almost_equals(lr.transform * lr.transform.scale(.25, .25)):
                raise ValueError('Pair is not on the exact x4 grid')
            if lr.tags(ns='IMAGE_STRUCTURE').get('LAYOUT') != 'COG' or hr.tags(ns='IMAGE_STRUCTURE').get('LAYOUT') != 'COG':
                raise ValueError('Prepared training rasters must be COGs')
        for key in ['lr_scale', 'lr_offset', 'hr_scale', 'hr_offset']:
            if key not in entry:
                raise ValueError(f'Missing radiometry: {key}')
            if not isinstance(entry[key], (int, float)) or not math.isfinite(entry[key]):
                raise ValueError(f'Invalid finite radiometry value: {key}')
            if key.endswith('_scale') and entry[key] <= 0:
                raise ValueError(f'Radiometric scale must be positive: {key}')
        records.append({**entry, 'lr_path': lr_path, 'hr_path': hr_path})
    return records


def load_crop(pair, size, scale):
    import numpy as np
    import rasterio
    from rasterio.windows import Window
    arrays = []
    for key, multiplier in [('lr', 1), ('hr', scale)]:
        with rasterio.open(pair[key + '_path']) as src:
            if src.width < size * multiplier or src.height < size * multiplier:
                raise ValueError(f'Pair {pair["aoi_id"]} is too small for tile {size}')
            a = src.read(window=Window(0, 0, size * multiplier, size * multiplier), masked=True)
        if np.ma.getmaskarray(a).any():
            raise ValueError('Training crop contains masked/no-data pixels')
        a = np.asarray(a, dtype='float32') * pair[key + '_scale'] + pair[key + '_offset']
        if not np.isfinite(a).all():
            raise ValueError('Training crop contains non-finite values')
        arrays.append(a)
    return arrays


def train_steps(model, samples, steps, learning_rate, device):
    """Return full loss history and actual weight change; callable on CPU for tests."""
    import torch
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError('Model has no trainable parameters')
    before = [p.detach().cpu().clone() for p in params]
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    losses, grad_seen = [], False
    model.train()
    if hasattr(model, 'hard_constraint'):
        model.hard_constraint.eval()
    for step in range(steps):
        x_cpu, target_cpu = samples[step % len(samples)]
        x, target = x_cpu.to(device), target_cpu.to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x)
        if prediction.shape != target.shape:
            raise ValueError('SR output and HR target shapes differ')
        loss = torch.nn.functional.l1_loss(prediction, target)
        if not torch.isfinite(loss):
            raise ValueError('Non-finite training loss')
        loss.backward()
        for p in params:
            if p.grad is not None:
                if not torch.isfinite(p.grad).all():
                    raise ValueError('Non-finite training gradient')
                grad_seen |= bool(torch.count_nonzero(p.grad))
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    delta = sum(float((p.detach().cpu() - old).abs().sum()) for p, old in zip(params, before))
    return losses, delta, grad_seen


def probe(cfg, root):
    """Require real prepared pairs and a Colab GPU, train, save and reload weights."""
    manifest_path = root / cfg['training']['pair_manifest']
    is_colab = bool(os.environ.get('COLAB_RELEASE_TAG') or os.environ.get('COLAB_GPU'))
    blockers = []
    if not manifest_path.is_file():
        blockers.append(f'WorldStrat prepared pair manifest missing: {cfg["training"]["pair_manifest"]}')
    if not is_colab:
        blockers.append('This runtime is not Google Colab; a local run cannot satisfy R5')
    if blockers:
        return {'status': 'BLOCKED', 'reason': '; '.join(blockers), 'training_steps_run': 0, 'colab': is_colab}
    import gc
    import numpy as np
    import torch
    from safetensors.torch import load_file, save_file
    from risk.model import load_model
    pairs = validate_manifest(json.loads(manifest_path.read_text(encoding='utf-8')), manifest_path.parent, cfg['model']['bands'])
    if len({p['aoi_id'] for p in pairs}) < cfg['training']['min_pairs']:
        raise ValueError('Not enough distinct training AOIs for the configured smoke test')
    if not torch.cuda.is_available():
        raise RuntimeError('Colab CUDA GPU is unavailable')
    torch.manual_seed(cfg['model']['seed'])
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1, cfg['model']['budget_gib'] * 2**30 / total))
    def run(size):
        # Restart from pretrained weights and step zero after OOM, rather than
        # retaining a partly updated optimizer or falsely counting failed steps.
        model = load_model(cfg, root, trainable=True)
        samples = [tuple(torch.from_numpy(a[None]) for a in load_crop(p, size, cfg['model']['scale'])) for p in pairs]
        losses, delta, grads = train_steps(model, samples, cfg['training']['steps'], cfg['training']['learning_rate'], 'cuda')
        model.eval()
        with torch.inference_mode():
            reference = model(samples[0][0].to('cuda')).cpu()
        weights = {k: v.detach().cpu().contiguous().clone() for k, v in model.state_dict().items()}
        del model
        gc.collect()
        torch.cuda.empty_cache()
        checkpoint = root / cfg['training']['checkpoint']
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        save_file(weights, str(checkpoint))
        restored = load_model(cfg, root, trainable=True).eval()
        restored.load_state_dict(load_file(str(checkpoint)))
        with torch.inference_mode():
            reloaded = restored(samples[0][0].to('cuda')).cpu()
        window = cfg['training']['comparison_window']
        initial, final = float(np.mean(losses[:window])), float(np.mean(losses[-window:]))
        reload_ok = torch.allclose(reference, reloaded, atol=cfg['training']['reload_atol'], rtol=0)
        return {'status': 'PASS' if final < initial and delta > 0 and grads and reload_ok else 'FAIL',
                'losses': losses, 'first_window_mean': initial, 'last_window_mean': final,
                'weight_l1_change': delta, 'nonzero_finite_gradients': grads, 'checkpoint_sha256': digest(checkpoint),
                'checkpoint': cfg['training']['checkpoint'], 'reload_matches': reload_ok,
                'training_steps_run': len(losses), 'colab': True, 'aois': [p['aoi_id'] for p in pairs],
                'device': torch.cuda.get_device_name(0), 'torch': torch.__version__}
    def cleanup():
        gc.collect()
        torch.cuda.empty_cache()
    result, tile, failed = retry_tiles(run, cfg['model']['native_tile'], cfg['model']['min_tile'], torch.cuda.OutOfMemoryError, cleanup)
    return {**result, 'tile_size': tile, 'oom_sizes': failed, 'pair_manifest_sha256': digest(manifest_path)}


if __name__ == '__main__':
    run_cli('r5', probe)
