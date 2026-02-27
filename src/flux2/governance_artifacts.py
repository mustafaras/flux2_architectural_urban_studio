from __future__ import annotations

import hashlib
import hmac
import json
import os
from datetime import datetime, timezone
from typing import Any


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sign_payload(payload: dict[str, Any], secret: str | None = None) -> dict[str, str]:
    key = (secret or os.environ.get("FLUX2_MANIFEST_SIGNING_KEY") or "flux2-dev-signing-key").encode("utf-8")
    canonical = _canonical_json(payload).encode("utf-8")
    digest = hmac.new(key, canonical, hashlib.sha256).hexdigest()
    return {
        "algorithm": "HMAC-SHA256",
        "signature": digest,
        "signed_at": datetime.now(timezone.utc).isoformat(),
    }


def verify_payload_signature(payload: dict[str, Any], signature: str, secret: str | None = None) -> bool:
    generated = sign_payload(payload=payload, secret=secret)
    return hmac.compare_digest(generated["signature"], str(signature))


def attach_signed_manifest(payload: dict[str, Any], *, secret: str | None = None) -> dict[str, Any]:
    payload_copy = dict(payload)
    signature_block = sign_payload(payload_copy, secret=secret)
    payload_copy["signature"] = signature_block
    return payload_copy
