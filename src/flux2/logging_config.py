from __future__ import annotations

import json
import logging
from collections import deque
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from threading import RLock
from typing import Any

from .error_types import ErrorContext, truncate_sensitive


_LOCK = RLock()
_RECENT_OPERATIONS: deque[dict[str, Any]] = deque(maxlen=50)


def configure_flux2_logging(log_dir: str | Path = "logs", debug: bool = False) -> Path:
    root = logging.getLogger()
    if root.handlers:
        return Path(log_dir)

    level = logging.DEBUG if debug else logging.INFO
    root.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    root.addHandler(stream)

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_path / "flux2.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    return log_path


def log_operation(name: str, status: str, details: dict[str, Any] | None = None) -> None:
    details = details or {}
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "name": name,
        "status": status,
        "details": _sanitize_dict(details),
    }

    logger = logging.getLogger("flux2.operations")
    logger.info("operation=%s", json.dumps(payload, ensure_ascii=False))

    with _LOCK:
        _RECENT_OPERATIONS.append(payload)


def log_error(ctx: ErrorContext) -> None:
    logger = logging.getLogger("flux2.errors")
    payload = {
        "timestamp": ctx.timestamp,
        "category": ctx.category.value,
        "severity": ctx.severity.value,
        "message": ctx.message,
        "location": ctx.location,
        "metadata": _sanitize_dict(ctx.metadata),
        "technical_details": truncate_sensitive(ctx.technical_details),
    }
    logger.error("error=%s", json.dumps(payload, ensure_ascii=False))

    with _LOCK:
        _RECENT_OPERATIONS.append(
            {
                "timestamp": ctx.timestamp,
                "name": f"error:{ctx.category.name}",
                "status": ctx.severity.value,
                "details": {
                    "location": ctx.location,
                    "message": ctx.message,
                },
            }
        )


def get_recent_operations(limit: int = 50) -> list[dict[str, Any]]:
    with _LOCK:
        items = list(_RECENT_OPERATIONS)
    return items[-max(1, limit) :]


def _sanitize_dict(data: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in data.items():
        k = str(key).lower()
        if "token" in k or "key" in k or "secret" in k or "password" in k:
            out[str(key)] = "***"
        elif isinstance(value, str):
            out[str(key)] = truncate_sensitive(value)
        else:
            out[str(key)] = value
    return out
