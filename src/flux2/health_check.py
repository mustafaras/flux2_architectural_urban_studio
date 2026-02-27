"""Health check endpoint wrapper for FLUX.2 Streamlit application.

This provides a simple HTTP server that runs alongside Streamlit to expose
health check endpoints for load balancers and monitoring systems.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler
from http.server import HTTPServer
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class HealthCheckHandler(BaseHTTPRequestHandler):
    """HTTP handler for health check endpoints."""

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress default logging."""
        pass

    def do_GET(self) -> None:  # noqa: N802
        """Handle GET requests."""
        if self.path == "/health":
            self._handle_health()
        elif self.path == "/health/live":
            self._handle_liveness()
        elif self.path == "/health/ready":
            self._handle_readiness()
        elif self.path == "/ready":
            self._handle_readiness()
        else:
            self._send_response(404, {"error": "not found"})

    def _handle_health(self) -> None:
        """Primary health check endpoint."""
        try:
            response = {
                "status": "ok",
                "service": "flux2-ui",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "version": os.environ.get("FLUX2_VERSION", "unknown"),
                "uptime_seconds": getattr(self.server, "uptime_seconds", 0),
            }
            self._send_response(200, response)
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            self._send_response(503, {"status": "error", "message": str(e)})

    def _handle_liveness(self) -> None:
        """Kubernetes liveness probe endpoint.
        
        Indicates if the service can be restarted (always true if responding).
        """
        self._send_response(200, {"alive": True})

    def _handle_readiness(self) -> None:
        """Kubernetes readiness probe endpoint.
        
        Indicates if the service is ready to serve requests.
        """
        try:
            # Check if required dependencies are available
            checks = {
                "streamlit": self._check_streamlit(),
                "models": self._check_models(),
                "disk": self._check_disk(),
            }

            all_ready = all(checks.values())
            status_code = 200 if all_ready else 503

            response = {
                "ready": all_ready,
                "checks": checks,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }
            self._send_response(status_code, response)
        except Exception as e:
            logger.error(f"Readiness check failed: {e}")
            self._send_response(503, {"ready": False, "error": str(e)})

    @staticmethod
    def _check_streamlit() -> bool:
        """Check if Streamlit is available."""
        try:
            import streamlit  # noqa: F401
            return True
        except ImportError:
            return False

    @staticmethod
    def _check_models() -> bool:
        """Check if model weights are accessible."""
        try:
            weights_dir = Path("/opt/flux2/weights")
            return weights_dir.exists() and any(weights_dir.glob("*.safetensors"))
        except Exception:
            return False

    @staticmethod
    def _check_disk() -> bool:
        """Check if disk space is sufficient."""
        try:
            import shutil
            stat = shutil.disk_usage("/")
            # At least 1GB free
            return stat.free > 1_000_000_000
        except Exception:
            return False

    def _send_response(self, status_code: int, payload: dict[str, Any]) -> None:
        """Send JSON response."""
        try:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Service", "flux2-ui")
            self.end_headers()
            self.wfile.write(body)
        except Exception as e:
            logger.error(f"Failed to send response: {e}")


class HealthCheckServer:
    """Health check HTTP server for FLUX.2."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8888):
        """Initialize health check server.
        
        Args:
            host: Bind address
            port: Bind port
        """
        self.host = host
        self.port = port
        self.server: HTTPServer | None = None
        self.thread: threading.Thread | None = None
        self.running = False

    def start(self) -> None:
        """Start health check server in background thread."""
        if self.running:
            logger.warning("Health check server already running")
            return

        self.server = HTTPServer((self.host, self.port), HealthCheckHandler)
        self.server.uptime_seconds = 0  # type: ignore
        self.running = True

        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

        logger.info(f"Health check server started on {self.host}:{self.port}")

    def _run(self) -> None:
        """Run server loop."""
        try:
            self.server.serve_forever()
        except Exception as e:
            logger.error(f"Health check server error: {e}")
        finally:
            self.running = False

    def stop(self) -> None:
        """Stop health check server."""
        if self.server:
            self.server.shutdown()
            self.running = False
            logger.info("Health check server stopped")


# Singleton instance
_health_server: HealthCheckServer | None = None


def init_health_check(host: str = "127.0.0.1", port: int = 8888) -> HealthCheckServer:
    """Initialize and start health check server.
    
    Args:
        host: Bind address
        port: Bind port
    
    Returns:
        Health check server instance
    """
    global _health_server

    if _health_server is None:
        _health_server = HealthCheckServer(host, port)
        _health_server.start()

    return _health_server


def get_health_server() -> HealthCheckServer | None:
    """Get health check server instance."""
    return _health_server


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    server = HealthCheckServer()
    server.start()

    try:
        # Keep running
        while True:
            asyncio.sleep(1)
    except KeyboardInterrupt:
        server.stop()
