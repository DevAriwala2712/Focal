"""Count unique WorldStrat AOIs using versioned country polygons."""
from __future__ import annotations

import csv
import json
from collections import Counter

from risk.common import digest, download, request_json, run_cli, write_json


def count_countries(rows, geojson, south_asia):
    from shapely.geometry import Point, shape
    from shapely.strtree import STRtree
    features = geojson['features']
    geometries = [shape(f['geometry']) for f in features]
    tree = STRtree(geometries)
    unique = {}
    for row in rows:
        coords = (float(row['lon']), float(row['lat']))
        if row[''] in unique and unique[row['']] != coords:
            raise ValueError(f"Conflicting coordinates for AOI {row['']}")
        unique[row['']] = coords
    by_country, ambiguous, unmatched = {}, [], []
    for aoi, coords in unique.items():
        point = Point(*coords)
        matches = [int(i) for i in tree.query(point) if geometries[int(i)].covers(point)]
        if len(matches) > 1:
            ambiguous.append(aoi)
        elif not matches:
            unmatched.append(aoi)
        else:
            country = features[matches[0]]['properties']['ADMIN']
            by_country[aoi] = country
    counts = Counter(by_country.values())
    return {'unique_aoi_count': len(unique), 'country_counts': dict(sorted(counts.items())),
            'india_count': counts['India'], 'south_asia_count': sum(counts[c] for c in south_asia),
            'iran_count_separate': counts['Iran'], 'ambiguous_aoi_ids': ambiguous, 'unmatched_aoi_ids': unmatched,
            'aoi_country': by_country,
            'india_aoi_ids': [a for a, c in by_country.items() if c == 'India'],
            'south_asia_aoi_ids': [a for a, c in by_country.items() if c in south_asia]}


def probe(cfg, root):
    """Verify metadata and licences; report archive bytes and AOI-centre country counts."""
    settings, policy = cfg['worldstrat'], cfg['network']
    cache = root / cfg['paths']['cache'] / 'worldstrat'
    metadata = download(settings['metadata_url'], cache / 'metadata.csv', policy, md5=settings['metadata_md5'])
    boundaries = download(settings['boundaries_url'], cache / 'countries.geojson', policy)
    license_file = download(settings['license_url'], cache / 'LICENSE.txt', policy)
    with metadata.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    result = count_countries(rows, json.loads(boundaries.read_text(encoding='utf-8')), settings['south_asia'])
    write_json(root / cfg['paths']['results'] / 'r3_aoi_countries.json', result)
    errors, files = [], []
    try:
        record = request_json(settings['record_url'], policy)
        files = [{'name': f['key'], 'bytes': f['size'], 'checksum': f.get('checksum')} for f in record['files']]
    except Exception as exc:
        errors.append(str(exc))
    terms = license_file.read_text(encoding='utf-8')
    if 'CC BY-NC 4.0' not in terms or 'CC BY 4.0' not in terms:
        errors.append('Publisher licence text differs from expected; inspect downloaded file.')
    return {'status': 'BLOCKED' if errors else 'PASS', 'errors': errors,
            'scope': 'Dataset availability, size, metadata counts and licence; image-pair suitability is checked in R5.',
            'metadata_rows': len(rows), 'metadata_sha256': digest(metadata), 'boundaries_sha256': digest(boundaries),
            'boundaries_url': settings['boundaries_url'], 'boundary_policy': 'Natural Earth v5.1.2, AOI centre, disputed geography follows this dataset; ambiguous and offshore centres excluded',
            'unique_aois': result['unique_aoi_count'], 'india_count': result['india_count'],
            'south_asia_count': result['south_asia_count'], 'south_asia_definition': settings['south_asia'],
            'iran_count_separate': result['iran_count_separate'], 'unmatched_count': len(result['unmatched_aoi_ids']),
            'ambiguous_count': len(result['ambiguous_aoi_ids']), 'archive_files': files,
            'total_bytes': sum(f['bytes'] for f in files) if files else None,
            'license': {'hr': 'CC BY-NC 4.0', 'other_dataset_content': 'CC BY 4.0', 'code': 'BSD-3-Clause'},
            'license_sha256': digest(license_file), 'aoi_details': 'r3_aoi_countries.json'}


if __name__ == '__main__':
    run_cli('r3', probe)
