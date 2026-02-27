from __future__ import annotations

from typing import Any


def build_why_card(metadata: dict[str, Any]) -> dict[str, Any]:
    prompt_slots = metadata.get("prompt_slots", {}) if isinstance(metadata.get("prompt_slots"), dict) else {}
    phase2 = metadata.get("phase2_controls", {}) if isinstance(metadata.get("phase2_controls"), dict) else {}

    template = str(metadata.get("prompt_template_version", "unknown"))
    key_drivers = {
        "massing": prompt_slots.get("massing", []),
        "facade": prompt_slots.get("facade", []),
        "mobility": prompt_slots.get("mobility", []),
        "atmosphere": prompt_slots.get("atmosphere", []),
    }
    locks = {
        "camera_family": ((phase2.get("view_lock", {}) if isinstance(phase2.get("view_lock", {}), dict) else {}).get("camera_family", "")),
        "lens_mm": ((phase2.get("view_lock", {}) if isinstance(phase2.get("view_lock", {}), dict) else {}).get("lens_mm", "")),
        "perspective_lock": ((phase2.get("view_lock", {}) if isinstance(phase2.get("view_lock", {}), dict) else {}).get("perspective_lock", False)),
        "seed_group_id": str(phase2.get("seed_group_id", "")),
        "continuity_profile": str(phase2.get("continuity_profile", "")),
    }

    trade_offs: list[str] = []
    continuity = float(metadata.get("continuity_score", 0.0) or 0.0)
    steps = int(metadata.get("num_steps", 0) or 0)
    if continuity >= 75:
        trade_offs.append("High continuity favors consistency over visual novelty.")
    elif continuity > 0:
        trade_offs.append("Lower continuity may improve variety but reduce family coherence.")
    if steps <= 4:
        trade_offs.append("Low step count prioritizes speed over fine detail resolution.")
    if float(metadata.get("guidance", 0.0) or 0.0) < 2.0:
        trade_offs.append("Lower guidance encourages diversity but can reduce prompt adherence.")

    return {
        "template": template,
        "key_drivers": key_drivers,
        "locks": locks,
        "trade_offs": trade_offs,
    }


def build_confidence_labels(metadata: dict[str, Any]) -> dict[str, str]:
    continuity = float(metadata.get("continuity_score", 0.0) or 0.0)
    lint_results = metadata.get("prompt_lint_results", [])
    lint_count = len(lint_results) if isinstance(lint_results, list) else 0

    if continuity >= 80:
        continuity_label = "high"
    elif continuity >= 50:
        continuity_label = "medium"
    else:
        continuity_label = "low"

    plausibility_label = "high"
    if lint_count >= 2:
        plausibility_label = "low"
    elif lint_count == 1:
        plausibility_label = "medium"

    return {
        "continuity_confidence": continuity_label,
        "plausibility_confidence": plausibility_label,
    }


def build_lineage_graph(
    *,
    selected_options: list[dict[str, Any]],
    scenarios: list[dict[str, Any]],
) -> dict[str, Any]:
    nodes: list[dict[str, str]] = []
    edges: list[dict[str, str]] = []

    option_ids: list[str] = []
    for item in selected_options:
        meta = item.get("metadata", {}) if isinstance(item.get("metadata"), dict) else {}
        option_id = f"{meta.get('model', '?')}:{meta.get('seed', '?')}:{meta.get('timestamp', '')}"
        option_ids.append(option_id)
        nodes.append({"id": option_id, "type": "option", "label": option_id})

    for scenario in scenarios:
        scenario_id = str(scenario.get("scenario_id", ""))
        parent_option_id = str(scenario.get("parent_option_id", ""))
        if not scenario_id:
            continue
        nodes.append({"id": scenario_id, "type": "scenario", "label": scenario_id})
        if parent_option_id:
            edges.append({"from": parent_option_id, "to": scenario_id, "relation": "variant"})

    mermaid_lines = ["graph TD"]
    for node in nodes:
        safe = node["id"].replace("-", "_").replace(":", "_").replace(".", "_")
        mermaid_lines.append(f"    {safe}[\"{node['label']}\"]")
    for edge in edges:
        left = edge["from"].replace("-", "_").replace(":", "_").replace(".", "_")
        right = edge["to"].replace("-", "_").replace(":", "_").replace(".", "_")
        mermaid_lines.append(f"    {left} -->|{edge['relation']}| {right}")

    return {
        "nodes": nodes,
        "edges": edges,
        "mermaid": "\n".join(mermaid_lines),
        "option_roots": option_ids,
    }


def build_explainability_metadata(
    *,
    option_metadata: dict[str, Any],
    lineage_graph: dict[str, Any],
) -> dict[str, Any]:
    why_card = build_why_card(option_metadata)
    confidence = build_confidence_labels(option_metadata)
    return {
        "why_card": why_card,
        "confidence_labels": confidence,
        "lineage": {
            "node_count": len(lineage_graph.get("nodes", [])),
            "edge_count": len(lineage_graph.get("edges", [])),
        },
    }
