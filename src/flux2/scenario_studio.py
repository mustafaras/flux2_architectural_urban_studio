"""Urban scenario studio helpers for lineage and delta analysis."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import uuid4


def create_scenario_from_option(
    *,
    base_option_id: str,
    base_metadata: dict[str, Any],
    density_profile: str,
    mobility_priority: str,
    greening_intensity: str,
    activation_profile: str,
    label: str,
    notes: str,
) -> dict[str, Any]:
    """Create scenario payload preserving parent lineage."""
    scenario_id = f"scn_{uuid4().hex[:10]}"
    toggles = {
        "density_profile": density_profile,
        "mobility_priority": mobility_priority,
        "greening_intensity": greening_intensity,
        "activation_profile": activation_profile,
    }
    return {
        "scenario_id": scenario_id,
        "parent_option_id": base_option_id,
        "created_at": datetime.now().isoformat(),
        "label": label or scenario_id,
        "notes": notes,
        "toggles": toggles,
        "lineage": {
            "parent_option_id": base_option_id,
            "lineage_depth": 1,
        },
        "base_metadata": base_metadata,
    }


def compute_scenario_delta(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Compute field deltas between two scenario payloads."""
    left_toggles = left.get("toggles", {}) if isinstance(left.get("toggles"), dict) else {}
    right_toggles = right.get("toggles", {}) if isinstance(right.get("toggles"), dict) else {}

    changed: dict[str, dict[str, Any]] = {}
    for key in sorted(set(left_toggles.keys()) | set(right_toggles.keys())):
        l_val = left_toggles.get(key)
        r_val = right_toggles.get(key)
        if l_val != r_val:
            changed[key] = {"left": l_val, "right": r_val}

    return {
        "left_scenario_id": left.get("scenario_id", ""),
        "right_scenario_id": right.get("scenario_id", ""),
        "changed_controls": changed,
        "changed_count": len(changed),
    }
