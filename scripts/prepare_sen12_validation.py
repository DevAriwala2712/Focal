"""Stream selected Sen12Landslides patches without downloading full archives."""
from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

import requests

from risk.common import load_config, write_json


def get(url, policy, timeout):
    last = None
    for attempt in range(policy['retries'] + 1):
        try:
            response = requests.get(url, stream=True, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last = exc
            if attempt < policy['retries']:
                import time
                time.sleep(policy['backoff_seconds']**attempt)
    raise RuntimeError(f"Download failed after {policy['retries'] + 1} attempts: {url}: {last}")


def source_record(target, root, event, row, url, payload, source_split):
    return {'event_id': event, 'file': target.relative_to(root).as_posix(),
            'published_split': source_split, 'positive_pixels': row['pixel_annotated'],
            'source_archive': url, 'source_member': target.name,
            'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload)}


def prepare():
    cfg, root, _ = load_config('configs/pipeline.yaml')
    settings = cfg['validation']
    policy = cfg['network']
    destination = root/settings['sen12_cache']
    destination.mkdir(parents=True, exist_ok=True)
    split_url = settings['sen12_split_url']
    with get(split_url, policy, settings['sen12_timeout_seconds']) as response:
        split_bytes = response.content
    published = json.loads(split_bytes)
    test_rows = {Path(row['s2']).name: row for row in published[settings['sen12_source_split']]
                 if row['pixel_annotated'] >= settings['sen12_min_positive_pixels']}
    records = []
    for event, part in settings['sen12_event_parts'].items():
        target_count = settings['sen12_patches_per_event'][event]
        url = settings['sen12_archive_template'].format(part=part)
        selected = {}
        cached = []
        for target in sorted(destination.glob(f'{event}_s2_*.nc')):
            sidecar = target.with_suffix(target.suffix + '.source.json')
            if target.name not in test_rows or not sidecar.is_file():
                continue
            record = json.loads(sidecar.read_text(encoding='utf-8'))
            if (record.get('source_archive') == url and record.get('source_member') == target.name
                    and record.get('sha256') == hashlib.sha256(target.read_bytes()).hexdigest()
                    and record.get('bytes') == target.stat().st_size):
                cached.append((target, record))
        if len(cached) >= target_count:
            for target, record in cached[:target_count]:
                selected[target.name] = record
            records.extend(selected.values())
            continue
        for attempt in range(policy['retries'] + 1):
            try:
                with get(url, policy, settings['sen12_timeout_seconds']) as response:
                    with tarfile.open(fileobj=response.raw, mode='r|gz') as archive:
                        for index, member in enumerate(archive):
                            if index >= settings['sen12_max_archive_members']:
                                break
                            row = test_rows.get(member.name)
                            if not row or row['inventory'] != event or not member.isfile():
                                continue
                            target = destination/member.name
                            payload = archive.extractfile(member).read()
                            if len(payload) != member.size:
                                raise ValueError(f'Truncated tar member {member.name} at {url}')
                            target.write_bytes(payload)
                            record = source_record(target, root, event, row, url, payload,
                                                   settings['sen12_source_split'])
                            write_json(target.with_suffix(target.suffix + '.source.json'), record)
                            selected[member.name] = record
                            print(f'Prepared {member.name}', flush=True)
                            if len(selected) == target_count:
                                break
                if len(selected) == target_count:
                    break
            except (requests.RequestException, tarfile.TarError, OSError, EOFError) as exc:
                if attempt == policy['retries']:
                    raise RuntimeError(f"Stream failed after {policy['retries'] + 1} attempts: {url}: {exc}") from exc
                import time
                time.sleep(policy['backoff_seconds']**attempt)
        if len(selected) < target_count:
            raise RuntimeError(f"Only {len(selected)} labelled {event} patches found in first {settings['sen12_max_archive_members']} members of {url}")
        records.extend(selected.values())
    manifest = {'dataset': 'Sen12Landslides harmonized S12LS-LD',
                'split_url': split_url,
                'split_sha256': hashlib.sha256(split_bytes).hexdigest(),
                'selection': 'First configured count of positive patches in publisher test split and archive order, per event; selection used no model outputs.',
                'calibration_events': list(settings['sen12_event_parts'])[:2],
                'evaluation_events': list(settings['sen12_event_parts'])[2:], 'samples': records,
                'caveat': 'Publisher test split is patch-random; this study additionally keeps calibration and evaluation event inventories disjoint. Only positive-containing patches were sampled.'}
    write_json(root/settings['sen12_manifest'], manifest)
    print(json.dumps({'samples': len(records), 'events': list(settings['sen12_event_parts'])}, indent=2))


if __name__ == '__main__':
    prepare()
