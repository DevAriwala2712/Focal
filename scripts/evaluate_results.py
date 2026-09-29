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
    validation_manifest = root/'data/worldstrat/validation/pairs.json'
    held_out = None
    held_out_reason = 'No verified WorldStrat validation pairs have been prepared.'
    if validation_manifest.is_file():
        val_records = validate_manifest(json.loads(validation_manifest.read_text(encoding='utf-8')),
                                        validation_manifest.parent, cfg['model']['bands'],
                                        allowed_splits=('val',))
        train_aois = {record['aoi_id'] for record in records}
        if train_aois & {record['aoi_id'] for record in val_records}:
            raise ValueError('WorldStrat validation AOI overlaps fine-tuning AOIs')
        val_scores = []
        for record in val_records:
            lr, hr = load_crop(record, 128, 4)
            prediction = super_resolve(lr, model, device='cuda')
            val_scores.append({'aoi_id': record['aoi_id'], **assess_sr(lr, hr, prediction, split='val')})
        held_out = {'pairs': val_scores,
                    'mean_psnr_db': sum(row['psnr_db'] for row in val_scores)/len(val_scores),
                    'mean_ssim': sum(row['ssim'] for row in val_scores)/len(val_scores),
                    'mean_spectral_rmse': sum(row['spectral_rmse'] for row in val_scores)/len(val_scores),
                    'pair_count': len(val_scores),
                    'manifest_sha256': digest(validation_manifest),
                    'caveat': 'HR DN is fit per band to each paired LR tile; this is a held-out spatial-fidelity comparison, not calibrated physical reflectance accuracy.'}
        held_out_reason = None
    demo_manifest = json.loads((root/'data/wayanad/demo/manifest.json').read_text(encoding='utf-8'))
    with rasterio.open(demo_manifest['files']['pre_10m']) as src:
        pre_lr = src.read()
    with rasterio.open(demo_manifest['files']['pre_2p5m']) as src:
        pre_sr = src.read()
    with rasterio.open(demo_manifest['files']['post_10m']) as src:
        post_lr = src.read()
    with rasterio.open(demo_manifest['files']['post_2p5m']) as src:
        post_sr = src.read()
    sen12_path = root/'results/sen12_evaluation.json'
    sen12 = json.loads(sen12_path.read_text(encoding='utf-8')) if sen12_path.is_file() else None
    downstream = sen12['evaluation'] if sen12 else None
    metrics = {'training_pair_diagnostic_only': diagnostics,
               'held_out_worldstrat': held_out,
               'held_out_reason': held_out_reason,
               'downstream_f1_10m': downstream['f1_10m'] if downstream else None,
               'downstream_f1_2p5m': downstream['f1_raw_2p5m_aggregated_to_10m'] if downstream else None,
               'downstream_f1_trust': downstream['f1_trust_2p5m_aggregated_to_10m'] if downstream else None,
               'downstream_unsupported_pixels': downstream['unsupported_2p5m_pixels'] if downstream else None,
               'downstream_scope': ('Sen12Landslides event-held-out 10 m masks; positive-enriched patches; '
                                    'SR predictions aggregated to 10 m; Wayanad has no independent labels.' if downstream else None),
               'downstream_reason': None if downstream else 'No independent multi-event temporal label run yet.',
               'calibration': ({'selected_k': sen12['selected_k'],
                                'identifiable': sen12['calibration']['identifiable'],
                                'events': sen12['calibration_events']} if sen12 else None),
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
             (f"| Held-out WorldStrat PSNR/SSIM | {held_out['mean_psnr_db']:.3f} dB / {held_out['mean_ssim']:.4f} | {held_out['pair_count']} publisher validation AOIs; per-pair radiometric fit limits interpretation |"
              if held_out else '| Held-out WorldStrat PSNR/SSIM | unavailable | No verified validation pairs |'),
             (f"| Event-held-out landslide F1 (10 m / raw SR / trust) | {downstream['f1_10m']:.4f} / {downstream['f1_raw_2p5m_aggregated_to_10m']:.4f} / {downstream['f1_trust_2p5m_aggregated_to_10m']:.4f} | {sum(p['evaluation_group'] == 'evaluation' for p in sen12['patches'])} Sen12Landslides patches; 10 m labels; positive-enriched sample |"
              if downstream else '| Landslide F1 (10 m / 2.5 m / trust) | unavailable | No suitable held-out temporal label set |'),
             (f"| Event-held-out UNSUPPORTED pixels | {downstream['unsupported_2p5m_pixels']} | Rejected fine pixels across tested patches |"
              if downstream else ''),
             (f"| k calibration | {sen12['selected_k']:.1f} | {'Unique F1 optimum' if sen12['calibration']['identifiable'] else 'All candidates tied; retained default, not identified'} |"
              if sen12 else ''), '',
             'Training-pair PSNR/SSIM are in metrics.json for optimizer diagnostics only.',
             'WorldStrat HR DN required a per-pair fit to LR reflectance; held-out scores measure spatial fidelity under that fit, not physical radiometric accuracy.',
             'The event-held-out landslide sample is small and enriched for positive patches. Its low F1 is evidence against a reliable operational detector; it is not a Wayanad accuracy estimate.',
             'The earlier 507-building and 8.38-km-road figures are a published contextual baseline, not TrustSR counts.', '']
    (root/'results/REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
