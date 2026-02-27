from __future__ import annotations

import re
from typing import Any

from .governance import get_retention_profile


_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE = re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b")


def validate_project_access_boundary(
    *,
    active_project_id: str,
    target_project_id: str,
    role: str,
) -> tuple[bool, str]:
    current = str(active_project_id).strip()
    target = str(target_project_id).strip()
    role_value = str(role).strip().lower()

    if not current or not target:
        return False, "missing project scope"
    if current == target:
        return True, "ok"
    if role_value == "admin":
        return True, "admin override"
    return False, "cross-project access denied"


def _sanitize_text(value: str) -> str:
    sanitized = _EMAIL.sub("[REDACTED_EMAIL]", str(value))
    sanitized = _PHONE.sub("[REDACTED_PHONE]", sanitized)
    return sanitized[:180]


def sanitize_reference_log_payload(reference: dict[str, Any]) -> dict[str, Any]:
    tags = reference.get("tags", []) if isinstance(reference.get("tags", []), list) else []
    return {
        "reference_id": str(reference.get("reference_id", "")),
        "category": str(reference.get("category", "")),
        "mime_type": str(reference.get("mime_type", "")),
        "size_bytes": int(reference.get("size_bytes", 0) or 0),
        "source": _sanitize_text(str(reference.get("source", ""))),
        "tag_count": len(tags),
        "name_redacted": "[REDACTED]",
    }


def sanitize_comment_log_payload(*, author: str, text: str) -> dict[str, Any]:
    return {
        "author": _sanitize_text(author),
        "text": _sanitize_text(text),
        "text_length": len(str(text)),
    }


def validate_retention_deletion_behavior(
    *,
    retention_profile: str,
    artifact_age_days: int,
    workflow_state: str,
) -> dict[str, Any]:
    profile = get_retention_profile(retention_profile)
    retention_days = int(profile.get("retention_days", 0) or 0)
    deletion_policy = str(profile.get("deletion_policy", "rolling"))
    review_required = bool(profile.get("review_required", False))

    age = max(0, int(artifact_age_days))
    expired = age > retention_days

    if not expired:
        return {
            "action": "retain",
            "reason": "within retention window",
            "expired": False,
            "retention_days": retention_days,
            "deletion_policy": deletion_policy,
        }

    if review_required and workflow_state != "approved":
        return {
            "action": "hold",
            "reason": "profile requires review before deletion",
            "expired": True,
            "retention_days": retention_days,
            "deletion_policy": deletion_policy,
        }

    if deletion_policy == "immutable-window":
        return {
            "action": "hold",
            "reason": "immutable window profile",
            "expired": True,
            "retention_days": retention_days,
            "deletion_policy": deletion_policy,
        }

    return {
        "action": "delete",
        "reason": "expired and eligible for deletion",
        "expired": True,
        "retention_days": retention_days,
        "deletion_policy": deletion_policy,
    }
