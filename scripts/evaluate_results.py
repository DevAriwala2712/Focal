"""Record real training diagnostics and explicit unavailable validation metrics."""
from __future__ import annotations

import json

import rasterio
import torch

from risk.common import digest, load_config, write_json
from risk.r5_finetune import load_crop, validate_manifest
from trustsr.evaluate import assess_sr, spectral_rmse
from trustsr.sr import load_finetuned_model, super_resolve


def main():
    cfg, root, _ = load_config('configs/phase0.yaml')
    pipeline, _, _ = load_config('configs/pipeline.yaml')
    pair_manifest = root/cfg['training']['pair_manifest']
    records = validate_manifest(json.loads(pair_manifest.read_text(encoding='utf-8')),
                                pair_manifest.parent, cfg['model']['bands'])
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU unavailable for SEN2SR evaluation')
    fraction = pipeline['model']['budget_gib']*2**30/torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1, fraction))
    model = load_finetuned_model(pipeline, root).eval()
    diagnostics = []
    for record in records:
        lr, hr = load_crop(record, 128, 4)
        prediction = super_resolve(lr, model, device='cuda')
        diagnostics.append({'aoi_id': record['aoi_id'], **assess_sr(lr, hr, prediction, split=record['split'])})
    demo_manifest = json.loads((root/'data/wayanad/demo/manifest.json').read_text(encoding='utf-8'))
    with rasterio.open(demo_manifest['files']['pre_10m']) as src:
        pre_lr = src.read()
    with rasterio.open(demo_manifest['files']['pre_2p5m']) as src:
        pre_sr = src.read()
    with rasterio.open(demo_manifest['files']['post_10m']) as src:
        post_lr = src.read()
    with rasterio.open(demo_manifest['files']['post_2p5m']) as src:
        post_sr = src.read()
    metrics = {'training_pair_diagnostic_only': diagnostics,
               'held_out_worldstrat': None,
               'held_out_reason': 'Only three WorldStrat train AOIs were prepared; no independent validation/test pairs are available.',
               'downstream_f1_10m': None, 'downstream_f1_2p5m': None, 'downstream_f1_trust': None,
               'downstream_reason': 'No held-out georeferenced pre/post landslide labels aligned to this Wayanad tile.',
               'opensr_test': None, 'opensr_test_reason': 'Optional package dry-run would upgrade pinned timm and add substantial dependencies; benchmark dataset not installed.',
               'wayanad_tile': {'pre_spectral_rmse': spectral_rmse(pre_lr, pre_sr),
                                'post_spectral_rmse': spectral_rmse(post_lr, post_sr),
                                'class_counts': demo_manifest['metrics'],
                                'pre_dates': demo_manifest['pre_dates'], 'post_date': demo_manifest['post_date'],
                                'post_event_lag_days': demo_manifest['post_event_lag_days']},
               'checkpoint_sha256': digest(root/pipeline['training']['checkpoint']),
               'pair_manifest_sha256': digest(pair_manifest)}
    write_json(root/'results/metrics.json', metrics)
    lines = ['# TrustSR evaluation status', '',
             'The Wayanad tile is a delayed December 2024 comparison. Its NDVI drop is not a labelled landslide outcome.', '',
             '| Measure | Result | Interpretation |', '| --- | ---: | --- |',
             f"| Pre-event spectral RMSE (reflectance) | {metrics['wayanad_tile']['pre_spectral_rmse']:.8f} | Parent-block consistency on real Wayanad inputs |",
             f"| Post-event spectral RMSE (reflectance) | {metrics['wayanad_tile']['post_spectral_rmse']:.8f} | Parent-block consistency on real Wayanad inputs |",
             f"| OBSERVED pixels | {demo_manifest['metrics']['observed_pixels']} | 2.5 m positions are still model-inferred |",
             f"| INFERRED pixels | {demo_manifest['metrics']['inferred_pixels']} | Changed 10 m parent, uncertain fine location |",
             f"| UNSUPPORTED pixels | {demo_manifest['metrics']['unsupported_pixels']} | SR-only change rejected |",
             '| Held-out WorldStrat PSNR/SSIM | unavailable | Prepared pairs are all train AOIs |',
             '| Landslide F1 (10 m / 2.5 m / trust) | unavailable | No suitable held-out temporal label set |', '',
             'Training-pair PSNR/SSIM are in metrics.json for optimizer diagnostics only; they are not generalization evidence.',
             'The earlier 507-building and 8.38-km-road figures are a published contextual baseline, not TrustSR counts.', '']
    (root/'results/REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
