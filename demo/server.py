"""Serve the TrustSR demo: static pages from demo/web plus whitelisted downloads under /files/.

    .venv/bin/python demo/server.py            # http://127.0.0.1:8765
    .venv/bin/python demo/server.py --port 9000

Only files listed in demo/web/data/site.json (`exports`) are downloadable; nothing else outside demo/web is exposed.
"""
from __future__ import annotations

import argparse
import functools
import json
import mimetypes
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / 'demo' / 'web'

mimetypes.add_type('image/tiff', '.tif')
mimetypes.add_type('application/octet-stream', '.i16')
mimetypes.add_type('application/octet-stream', '.u8')
mimetypes.add_type('application/octet-stream', '.u16')


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, allowed: dict[str, Path], **kw):
        self.allowed = allowed
        super().__init__(*a, directory=str(WEB), **kw)

    def do_GET(self):  # noqa: N802
        path = unquote(self.path.split('?', 1)[0])
        if path.startswith('/files/'):
            rel = path[len('/files/'):]
            p = self.allowed.get(rel)
            if p is None or not p.is_file():
                self.send_error(404, 'not an exported artifact')
                return
            self.send_response(200)
            self.send_header('Content-Type', mimetypes.guess_type(p.name)[0] or 'application/octet-stream')
            self.send_header('Content-Length', str(p.stat().st_size))
            self.send_header('Content-Disposition', f'attachment; filename="{p.name}"')
            self.end_headers()
            with open(p, 'rb') as f:
                while chunk := f.read(1 << 20):
                    self.wfile.write(chunk)
            return
        if path in ('', '/'):
            self.send_response(302)
            self.send_header('Location', '/how-it-works.html')
            self.end_headers()
            return
        super().do_GET()

    def end_headers(self):
        self.send_header('Cache-Control', 'no-cache')
        super().end_headers()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8765)
    a = ap.parse_args()
    site = WEB / 'data' / 'site.json'
    if not site.exists():
        raise SystemExit('demo/web/data/site.json missing: run  python scripts/build_web_data.py  first')
    exports = json.loads(site.read_text(encoding='utf-8'))['exports']
    allowed = {e['path']: ROOT / e['path'] for e in exports}
    handler = functools.partial(Handler, allowed=allowed)
    with ThreadingHTTPServer((a.host, a.port), handler) as srv:
        print(f'TrustSR demo on http://{a.host}:{a.port}  ({len(allowed)} downloadable artifacts)')
        srv.serve_forever()


if __name__ == '__main__':
    main()
