"""Tiny local inference sidecar for the presentation live panel.

Binds 0.0.0.0:43126 (uncommon; not 3000/5173/8080). Vite proxies /api here.
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

HOST = "0.0.0.0"
PORT = 43126
ENGINE = LiveEngine()


def _json_bytes(payload: dict, code: int = 200) -> tuple[int, bytes]:
    return code, json.dumps(payload).encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("[sidecar] " + (fmt % args) + "\n")

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _write(self, code: int, body: bytes, content_type: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._write(204, b"")

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        query = parse_qs(parsed.query)
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
