"""Calibrate on two Sen12 event inventories and score an unseen third event."""
from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

import h5py
import numpy as np
import torch

from risk.common import load_config, write_json
from trustsr.change import INFERRED, OBSERVED, classify
from trustsr.evaluate import calibrate_k, f1_score
from trustsr.sr import load_finetuned_model, super_resolve
from trustsr.trust import ndvi, trust_period


class UnusableTemporalPatchError(ValueError):
    """Published patch lacks one unambiguous, clear before/after event pair."""


def read_patch(path: Path, expected_hash: str, invalid_scl, max_invalid_pct):
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
        raise ValueError(f'Sen12 member checksum mismatch: {path.name}')
    with h5py.File(path) as source:
        raw_event = source.attrs['event_date']
        event_text = raw_event.decode() if isinstance(raw_event, bytes) else str(raw_event)
        if ',' in event_text:
            raise UnusableTemporalPatchError(f'Multiple events share one inventory mask: {path.name}')
        event_date = date.fromisoformat(event_text)
        raw_units = source['time'].attrs['units']
        units = raw_units.decode() if isinstance(raw_units, bytes) else str(raw_units)
        origin = date.fromisoformat(units.removeprefix('days since '))
        dates = [origin + timedelta(days=int(value)) for value in source['time'][:]]
        acceptable = [i for i in range(len(dates))
                      if 100*np.isin(source['SCL'][i], invalid_scl).mean() <= max_invalid_pct]
        pre_candidates = [i for i in acceptable if dates[i] < event_date]
        post_candidates = [i for i in acceptable if dates[i] >= event_date]
        if not pre_candidates or not post_candidates:
            raise UnusableTemporalPatchError(f'No clear dates bracketing event: {path.name}')
        pre_index = max(pre_candidates, key=lambda i: dates[i])
        post_index = min(post_candidates, key=lambda i: dates[i])
        pre_date, post_date = dates[pre_index], dates[post_index]
        if not pre_date < event_date <= post_date:
            raise ValueError(f'Published before/after indices do not bracket event: {path.name}')
        def image(index):
            return np.stack([source[band][index].T for band in ('B04', 'B03', 'B02', 'B08')]).astype('float32')/10000
        before, after = image(pre_index), image(post_index)
        pre_scl, post_scl = source['SCL'][pre_index].T, source['SCL'][post_index].T
        labels = source['MASK'][post_index].T.astype(bool)
        if not np.array_equal(labels, source['MASK'][pre_index].T.astype(bool)):
            raise UnusableTemporalPatchError(f'Landslide inventory mask changes between dates: {path.name}')
        valid = ~np.isin(pre_scl, invalid_scl) & ~np.isin(post_scl, invalid_scl)
        valid &= (before > 0).all(axis=0) & (after > 0).all(axis=0)
        if not valid.any():
            raise UnusableTemporalPatchError(f'No valid paired pixels: {path.name}')
    return before, after, valid, labels, {'pre_date': str(pre_date), 'post_date': str(post_date),
                                         'event_date': str(event_date),
                                         'valid_parent_pixels': int(valid.sum()),
                                         'labelled_parent_pixels': int((labels & valid).sum())}


def prepare_features(record, model, cfg, root):
    before, after, valid_parent, labels, metadata = read_patch(
        root/record['file'], record['sha256'], cfg['change']['invalid_scl'],
        cfg['validation']['sen12_max_invalid_pct'])
    operator = lambda tile: super_resolve(tile, model, device='cuda',
                                          tile_size=cfg['sr']['tile_size'],
                                          overlap=cfg['sr']['overlap'],
                                          min_tile=cfg['sr']['min_tile'])
    pre = trust_period([before], operator, [valid_parent])
    post = trust_period([after], operator, [valid_parent])
    arrays = {'pre_mean': pre['ndvi_mean'], 'post_mean': post['ndvi_mean'],
              'pre_std': pre['ndvi_std'], 'post_std': post['ndvi_std'],
              'parent_pre': ndvi(before), 'parent_post': ndvi(after),
              'valid': pre['valid'] & post['valid'], 'labels': labels}
    return arrays, metadata


def score(samples, k, threshold):
    actual, coarse, fine, trusted, observed_only, validity = [], [], [], [], [], []
    unsupported = 0
    for sample in samples:
        classes, counts = classify(sample['pre_mean'], sample['post_mean'], sample['pre_std'],
                                   sample['post_std'], sample['parent_pre'], sample['parent_post'],
                                   sample['valid'], k=k, parent_drop_threshold=threshold)
        unsupported += counts['unsupported_pixels']
        h, w = sample['parent_pre'].shape
        parent_valid = sample['valid'].reshape(h, 4, w, 4).all((1, 3))
        d = sample['pre_mean'] - sample['post_mean']
        sigma = np.sqrt(sample['pre_std']**2 + sample['post_std']**2)
        raw_fine = (d > k*sigma) & sample['valid']
        actual.append(sample['labels'].ravel())
        coarse.append((sample['parent_pre'] - sample['parent_post'] > threshold).ravel())
        fine.append(raw_fine.reshape(h, 4, w, 4).any((1, 3)).ravel())
        trusted.append(np.isin(classes, [OBSERVED, INFERRED]).reshape(h, 4, w, 4).any((1, 3)).ravel())
        observed_only.append((classes == OBSERVED).reshape(h, 4, w, 4).any((1, 3)).ravel())
        validity.append(parent_valid.ravel())
    truth, valid = np.concatenate(actual), np.concatenate(validity)
    predictions = {'10m': np.concatenate(coarse), 'raw_2p5m': np.concatenate(fine),
                   'trust': np.concatenate(trusted), 'observed_only': np.concatenate(observed_only)}
    confusion = {}
    for name, prediction in predictions.items():
        confusion[name] = {'tp': int(np.count_nonzero(truth & prediction & valid)),
                           'fp': int(np.count_nonzero(~truth & prediction & valid)),
                           'fn': int(np.count_nonzero(truth & ~prediction & valid))}
    return {'f1_10m': f1_score(truth, np.concatenate(coarse), valid),
            'f1_raw_2p5m_aggregated_to_10m': f1_score(truth, np.concatenate(fine), valid),
            'f1_trust_2p5m_aggregated_to_10m': f1_score(truth, np.concatenate(trusted), valid),
            'f1_observed_only_aggregated_to_10m': f1_score(truth, np.concatenate(observed_only), valid),
            'confusion': confusion,
            'unsupported_2p5m_pixels': unsupported,
            'valid_10m_pixels': int(valid.sum()), 'positive_10m_pixels': int(np.count_nonzero(truth & valid))}


def main():
    cfg, root, _ = load_config('configs/pipeline.yaml')
    manifest_path = root/'results/sen12_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    calibration_events = set(manifest['calibration_events'])
    evaluation_events = set(manifest['evaluation_events'])
    if len(calibration_events) < 2 or not evaluation_events or calibration_events & evaluation_events:
        raise ValueError('Calibration needs two events and disjoint evaluation event(s)')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU unavailable')
    fraction = cfg['model']['budget_gib']*2**30/torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1, fraction))
    model = load_finetuned_model(cfg, root).eval()
    samples = {'calibration': [], 'evaluation': []}
    provenance, rejected = [], []
    for record in manifest['samples']:
        group = 'calibration' if record['event_id'] in calibration_events else 'evaluation'
        if group == 'evaluation' and record['event_id'] not in evaluation_events:
            raise ValueError(f'Unassigned event {record["event_id"]}')
        try:
            arrays, metadata = prepare_features(record, model, cfg, root)
        except UnusableTemporalPatchError as exc:
            rejected.append({'file': record['file'], 'event_id': record['event_id'], 'reason': str(exc)})
            print(f"Excluded {record['source_member']}: {exc}", flush=True)
            continue
        samples[group].append(arrays)
        provenance.append({**record, **metadata, 'evaluation_group': group})
        print(f"Processed {record['source_member']}: {metadata['valid_parent_pixels']} valid", flush=True)
    if {row['event_id'] for row in provenance if row['evaluation_group'] == 'calibration'} != calibration_events:
        raise ValueError('A calibration event has no usable temporal patch')
    if not samples['evaluation']:
        raise ValueError('No independent evaluation patch survived quality checks')
    calibration = calibrate_k(samples['calibration'], cfg['calibration']['candidate_k'],
                              parent_drop_threshold=cfg['change']['parent_drop_threshold'])
    calibration['events'] = sorted(calibration_events)
    calibration['source_manifest_sha256'] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    write_json(root/'results/calibration.json', calibration)
    held_out = score(samples['evaluation'], calibration['selected_k'],
                     cfg['change']['parent_drop_threshold'])
    result = {'dataset': manifest['dataset'], 'selected_k': calibration['selected_k'],
              'calibration': calibration, 'evaluation': held_out,
              'calibration_events': sorted(calibration_events),
              'evaluation_events': sorted(evaluation_events),
              'patch_count': len(provenance), 'patches': provenance, 'rejected_patches': rejected,
              'limitations': ['Event inventories are disjoint, but this is a small, positive-patch-enriched sample.',
                              'MASK is a 10 m landslide inventory repeated across dates; no independent 2.5 m truth exists.',
                              'The 2.5 m predictions were aggregated back to 10 m for F1.',
                              'Trusted change includes OBSERVED and INFERRED; its 10 m F1 is algebraically tied to the parent gate.']}
    write_json(root/'results/sen12_evaluation.json', result)
    print(json.dumps({'selected_k': calibration['selected_k'], 'evaluation': held_out}, indent=2))


if __name__ == '__main__':
    main()
