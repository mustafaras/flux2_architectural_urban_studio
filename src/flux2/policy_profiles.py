from __future__ import annotations

from typing import Any


_POLICY_PROFILES: dict[str, dict[str, Any]] = {
    "commercial": {
        "name": "Commercial",
        "allow_share_links": True,
        "allow_prompt_exports": True,
        "minimum_export_state": "in-review",
        "require_signed_manifest": True,
    },
    "public_sector": {
        "name": "Public Sector",
        "allow_share_links": False,
        "allow_prompt_exports": False,
        "minimum_export_state": "approved",
        "require_signed_manifest": True,
    },
}

_STATE_ORDER = {
    "draft": 0,
    "in-review": 1,
    "approved": 2,
    "archived": 3,
}


def list_policy_profiles() -> list[str]:
    return list(_POLICY_PROFILES.keys())


def get_policy_profile(profile: str) -> dict[str, Any]:
    return dict(_POLICY_PROFILES.get(profile, _POLICY_PROFILES["commercial"]))


def _state_satisfies(current_state: str, minimum_state: str) -> bool:
    return _STATE_ORDER.get(current_state, -1) >= _STATE_ORDER.get(minimum_state, 999)


def check_export_policy(
    *,
    profile: str,
    workflow_state: str,
    include_prompts: bool = False,
) -> tuple[bool, str]:
    cfg = get_policy_profile(profile)
    minimum = str(cfg.get("minimum_export_state", "in-review"))
    if not _state_satisfies(workflow_state, minimum):
        return False, f"Policy requires state >= {minimum}"

    if include_prompts and not bool(cfg.get("allow_prompt_exports", False)):
        return False, "Policy forbids prompt export"

    return True, "ok"


def check_share_link_policy(*, profile: str) -> tuple[bool, str]:
    cfg = get_policy_profile(profile)
    if not bool(cfg.get("allow_share_links", False)):
        return False, "Policy forbids share links"
    return True, "ok"


def build_policy_conformance_dashboard(
    *,
    profile: str,
    workflow_state: str,
    recent_audit_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    cfg = get_policy_profile(profile)
    denied = [row for row in recent_audit_rows if str(row.get("status", "")) == "denied"]
    exports = [row for row in recent_audit_rows if str(row.get("event_name", "")) == "export-board"]

    export_state_ok = _state_satisfies(workflow_state, str(cfg.get("minimum_export_state", "in-review")))

    return {
        "policy_profile": profile,
        "policy": cfg,
        "workflow_state": workflow_state,
        "export_state_compliant": export_state_ok,
        "recent_denied_events": len(denied),
        "recent_export_events": len(exports),
        "auditability_status": "good" if len(denied) == 0 else "review-required",
    }
