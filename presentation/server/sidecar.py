"""Tiny local inference sidecar for the presentation live panel.

Binds 127.0.0.1:43126 (uncommon; not 3000/5173/8080). Vite proxies /api here.
Each forecast is a NumPy forward pass of baseline_univariate.keras on the
closest real Metro Interstate test hour — not a quiet/rush blend.
"""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from infer import LiveEngine  # noqa: E402
from learning_lab import LearningLab, LabError  # noqa: E402

HOST = "127.0.0.1"
PORT = 43126
ENGINE = LiveEngine()
LAB = LearningLab(HERE.parents[1])
LOCAL_ORIGINS = {f'http://{host}:{port}' for host in ('127.0.0.1', 'localhost') for port in (43125, 43126)}


def _json_bytes(payload: dict, code: int = 200) -> tuple[int, bytes]:
    return code, json.dumps(payload).encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("[sidecar] " + (fmt % args) + "\n")

    def _cors(self) -> None:
        origin = self.headers.get("Origin")
        if origin in LOCAL_ORIGINS:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _write(self, code: int, body: bytes, content_type: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _local_request(self) -> bool:
        host = self.headers.get("Host", "").lower()
        origin = self.headers.get("Origin")
        if host not in ("127.0.0.1:43126", "localhost:43126") or (origin and origin not in LOCAL_ORIGINS):
            self._write(403, json.dumps({"ok": False, "error": "Local app requests only."}).encode())
            return False
        return True

    def _learning_response(self, operation) -> None:
        try:
            payload = operation()
            code, body = _json_bytes(payload)
            self._write(code, body)
        except (LabError, ValueError) as exc:
            self._write(getattr(exc, 'status', 400), json.dumps({"ok": False, "error": str(exc)}).encode())
        except (OSError, ImportError, RuntimeError) as exc:
            self.log_message("learning service unavailable: %s", exc)
            self._write(503, json.dumps({"ok": False, "error": str(exc)}).encode())

    def do_OPTIONS(self) -> None:  # noqa: N802
        if self._local_request():
            self._write(204, b"")

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        query = parse_qs(parsed.query)
        if path.startswith('/api/learning/'):
            if not self._local_request():
                return
            if path == '/api/learning/status':
                self._learning_response(LAB.status)
            elif path == '/api/learning/samples':
                self._learning_response(lambda: LAB.samples(
                    (query.get('split') or ['test'])[0],
                    int((query.get('offset') or ['0'])[0]), int((query.get('limit') or ['12'])[0])))
            elif path == '/api/learning/export':
                self._learning_response(lambda: LAB.export((query.get('domain') or ['street'])[0]))
            elif path.startswith('/api/learning/image/') or path.startswith('/api/learning/snapshot/'):
                try:
                    identifier = path.rsplit('/', 1)[1]
                    source = LAB.sample_path(identifier) if '/image/' in path else LAB.snapshot_image(identifier)
                    self._write(200, source.read_bytes(), 'image/jpeg' if '/image/' in path else 'image/png')
                except (LabError, OSError) as exc:
                    self._write(getattr(exc, 'status', 404), json.dumps({'ok': False, 'error': str(exc)}).encode())
            else:
                self._write(404, b'{"ok":false,"error":"not found"}')
            return
        if path in ("/health", "/api/health"):
            code, body = _json_bytes(ENGINE.health())
            self._write(code, body)
            return
        if path in ("/forecast", "/api/forecast"):
            intensity = float((query.get("intensity") or ["0"])[0])
            queue = query.get("queue", [None])[0]
            queue_val = float(queue) if queue is not None else None
            payload = ENGINE.forecast(intensity, queue=queue_val)
            code, body = _json_bytes(payload, 200 if payload.get("ok") else 503)
            self._write(code, body)
            return
        self._write(404, json.dumps({"ok": False, "error": "not found"}).encode("utf-8"))

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        if path.startswith('/api/learning/'):
            if not self._local_request():
                return
            if self.headers.get('Content-Type', '').split(';')[0].strip() != 'application/json':
                self._write(415, b'{"ok":false,"error":"Use application/json."}')
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 15 * 1024 * 1024:
                    raise LabError('Request must be between 1 byte and 15 MB.', 413)
                data = json.loads(self.rfile.read(length).decode('utf-8'))
                if not isinstance(data, dict):
                    raise LabError('Expected a JSON object.')
            except (ValueError, UnicodeError) as exc:
                self._write(getattr(exc, 'status', 400), json.dumps({'ok': False, 'error': str(exc)}).encode())
                return
            operations = {'/api/learning/analyze': LAB.analyze,
                          '/api/learning/feedback': LAB.feedback,
                          '/api/learning/train': LAB.train}
            operation = operations.get(path)
            if operation:
                self._learning_response(lambda: operation(data))
            else:
                self._write(404, b'{"ok":false,"error":"not found"}')
            return
        if path not in ("/forecast", "/api/forecast"):
            self._write(404, json.dumps({"ok": False, "error": "not found"}).encode("utf-8"))
            return
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            self._write(400, json.dumps({"ok": False, "error": "invalid json"}).encode("utf-8"))
            return
        intensity = float(data.get("intensity") or 0)
        queue = data.get("queue")
        queue_val = float(queue) if queue is not None else None
        payload = ENGINE.forecast(intensity, queue=queue_val)
        code, body = _json_bytes(payload, 200 if payload.get("ok") else 503)
        self._write(code, body)


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"inference sidecar on http://{HOST}:{PORT} (hours={len(ENGINE.hours)} weights={ENGINE.weights is not None})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
