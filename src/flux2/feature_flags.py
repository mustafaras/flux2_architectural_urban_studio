from __future__ import annotations

from typing import Any, MutableMapping


DEFAULT_FEATURE_FLAGS: dict[str, bool] = {
    "enable_explainability": True,
    "enable_connectors": False,
    "enable_policy_dashboards": True,
    "enable_rollout_playbooks": True,
}


def get_feature_flags(state: MutableMapping[str, Any]) -> dict[str, bool]:
    raw = state.get("feature_flags", {})
    flags = dict(DEFAULT_FEATURE_FLAGS)
    if isinstance(raw, dict):
        for key, value in raw.items():
            if key in flags:
                flags[key] = bool(value)
    return flags


def set_feature_flag(state: MutableMapping[str, Any], key: str, enabled: bool) -> dict[str, bool]:
    flags = get_feature_flags(state)
    if key in flags:
        flags[key] = bool(enabled)
    state["feature_flags"] = flags
    return flags


def is_feature_enabled(state: MutableMapping[str, Any], key: str) -> bool:
    return bool(get_feature_flags(state).get(key, False))
