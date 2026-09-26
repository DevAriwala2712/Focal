import hashlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


def test_result_write_retries_transient_windows_lock(tmp_path, monkeypatch):
    from pathlib import Path
    from risk.common import write_json
    original = Path.replace
    attempts = []
    def locked_once(self, target):
        attempts.append(target)
        if len(attempts) == 1:
            raise PermissionError('OneDrive sharing violation')
        return original(self, target)
    monkeypatch.setattr(Path, 'replace', locked_once)
    path = tmp_path / 'result.json'
    write_json(path, {'status': 'PASS'})
    import json
    assert json.loads(path.read_text()) == {'status': 'PASS'}


def test_download_retries_rejects_oversize_and_checks_cache(tmp_path):
    from risk.common import download
    class Handler(BaseHTTPRequestHandler):
        attempts = 0
        def do_GET(self):
            Handler.attempts += 1
            if self.path == '/retry' and Handler.attempts < 4:
                self.send_response(503)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header('Content-Length', '6')
            self.end_headers()
            self.wfile.write(b'actual')
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    policy = dict(retries=3, backoff_seconds=0, timeout_seconds=2, max_download_bytes=20)
    url = f'http://127.0.0.1:{server.server_port}'
    try:
        target = tmp_path / 'data'
        download(url + '/retry', target, policy, sha256=hashlib.sha256(b'actual').hexdigest())
        assert target.read_bytes() == b'actual'
        assert Handler.attempts == 4
        target.write_bytes(b'corrupt')
        download(url + '/retry', target, policy, sha256=hashlib.sha256(b'actual').hexdigest())
        assert target.read_bytes() == b'actual'
        with pytest.raises(RuntimeError, match='limit'):
            download(url + '/big', tmp_path / 'big', {**policy, 'max_download_bytes': 4})
        assert not (tmp_path / 'big').exists()
    finally:
        server.shutdown()
        server.server_close()
