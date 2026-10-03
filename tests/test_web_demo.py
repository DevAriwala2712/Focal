"""Guards for demo/web: pages exist and are linked, shipped numbers match the run's own JSON, downloads are whitelisted,
and none of the invented claims from the original mock-ups have crept back in."""
import functools
import hashlib
import json
import re
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / 'demo' / 'web'
SITE = WEB / 'data' / 'site.json'
PAGES = ['how-it-works', 'data-dates', 'map-workspace', 'objects', 'evidence', 'experiments', 'demo', 'exports']

needs_data = pytest.mark.skipif(not SITE.exists(), reason='run scripts/build_web_data.py first')


def test_every_nav_page_exists_and_is_registered():
    shell = (WEB / 'assets' / 'shell.js').read_text(encoding='utf-8')
    nav = re.findall(r"\['([a-z-]+)', '[^']+', '[a-z_]+'\]", shell)
    assert nav == PAGES
    for p in PAGES:
        html = (WEB / f'{p}.html').read_text(encoding='utf-8')
        assert f'data-page="{p}"' in html, p


def test_no_invented_claims_from_the_mockups():
    banned = ['NCESS', 'GPG', 'OpenPGP', 'Zenodo', 'KSDMA', 'Section 65B', 'LEGAL ADMISSIBLE', 'PlanetScope', 'Dr. V. SESHAN',
              'GPS ground-truth', 'Hallucination risk', 'Cartosat', 'vault.trustsr.org', 'Baseline Bloat', '0.874']
    for p in WEB.glob('*.html'):
        text = p.read_text(encoding='utf-8')
        for b in banned:
            assert b.lower() not in text.lower(), f'{b!r} found in {p.name}'


@needs_data
def test_site_json_matches_step4_and_counts_add_up():
    site = json.loads(SITE.read_text(encoding='utf-8'))
    step4 = json.loads((ROOT / 'experiments/wayanad_evidence/outputs/step4.json').read_text(encoding='utf-8'))['measurements']
    assert site['step4']['class_counts_px'] == step4['class_counts_px']
    h, w = site['crop']['shape_2p5m']
    assert sum(step4['class_counts_px'].values()) == h * w
    assert site['labels']['model'].startswith('pretrained') and 'NOT calibrated' in site['labels']['k']
    assert site['crop']['shape_2p5m'][0] == 4 * site['crop']['shape_10m'][0]  # exact 4x4 refinement


@needs_data
def test_shipped_class_raster_reproduces_reported_counts():
    import numpy as np
    site = json.loads(SITE.read_text(encoding='utf-8'))
    cls = np.fromfile(WEB / 'data' / 'raster' / 'class_k2.u8', dtype='u1')
    exp = site['step4']['class_counts_px']
    got = {n: int((cls == c).sum()) for n, c in {'NO_CHANGE': 0, 'OBSERVED': 1, 'INFERRED': 2, 'UNSUPPORTED': 3, 'NO_DATA': 255}.items()}
    assert got == exp


@needs_data
def test_export_hashes_are_current():
    site = json.loads(SITE.read_text(encoding='utf-8'))
    for e in site['exports']:
        f = ROOT / e['path']
        assert f.stat().st_size == e['bytes'], e['path']
        assert hashlib.sha256(f.read_bytes()).hexdigest() == e['sha256'], e['path']


@needs_data
def test_server_serves_only_whitelisted_files():
    from demo import server
    exports = json.loads(SITE.read_text(encoding='utf-8'))['exports']
    allowed = {e['path']: ROOT / e['path'] for e in exports}
    srv = ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(server.Handler, allowed=allowed))
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    base = f'http://127.0.0.1:{srv.server_address[1]}'
    try:
        ok = urllib.request.urlopen(f"{base}/files/{exports[0]['path']}")
        assert ok.status == 200 and int(ok.headers['Content-Length']) == exports[0]['bytes']
        for bad in ['/files/AGENTS.md', '/files/../../etc/passwd', '/files/.env']:
            with pytest.raises(urllib.error.HTTPError) as ei:
                urllib.request.urlopen(base + bad)
            assert ei.value.code == 404
        assert urllib.request.urlopen(f'{base}/how-it-works.html').status == 200
    finally:
        srv.shutdown()
