"""Minimal model worker service scaffold for distributed model loading."""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer


class ModelWorkerHandler(BaseHTTPRequestHandler):
    def _send(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path == "/health":
            self._send(
                200,
                {
                    "status": "ok",
                    "service": os.environ.get("MODEL_SERVICE_NAME", "model-worker"),
                    "model": os.environ.get("MODEL_KEY", "unknown"),
                },
            )
            return
        self._send(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        if self.path != "/infer":
            self._send(404, {"error": "not found"})
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length) if content_length > 0 else b"{}"
        try:
            request = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            self._send(400, {"error": "invalid json"})
            return

        self._send(
            501,
            {
                "status": "not_implemented",
                "message": "Wire this endpoint to the FLUX adapter for full inference.",
                "received": request,
            },
        )


def main() -> None:
    host = os.environ.get("MODEL_WORKER_HOST", "0.0.0.0")
    port = int(os.environ.get("MODEL_WORKER_PORT", "8600"))
    server = HTTPServer((host, port), ModelWorkerHandler)
    print(f"Model worker listening on {host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
