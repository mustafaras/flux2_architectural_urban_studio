"""Design quality rubric helpers for architectural option scoring."""

from __future__ import annotations

from typing import Any

RUBRIC_FIELDS = [
    "context_fit",
    "architectural_coherence",
    "human_scale_readability",
    "public_realm_narrative",
    "presentation_readiness",
]


def normalize_rubric_scores(scores: dict[str, Any]) -> dict[str, float]:
    """Normalize rubric values to 0-10 float range."""
    normalized: dict[str, float] = {}
    for field in RUBRIC_FIELDS:
        raw = scores.get(field, 0)
        value = float(raw) if isinstance(raw, (int, float)) else 0.0
        value = max(0.0, min(10.0, value))
        normalized[field] = round(value, 2)
    return normalized


def aggregate_average_score(scored_options: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate average rubric scores over options with rubric payloads."""
    if not scored_options:
        return {field: 0.0 for field in RUBRIC_FIELDS}

    totals = {field: 0.0 for field in RUBRIC_FIELDS}
    count = 0
    for item in scored_options:
        scores = item.get("rubric_scores", {}) if isinstance(item, dict) else {}
        if not isinstance(scores, dict):
            continue
        normalized = normalize_rubric_scores(scores)
        for field in RUBRIC_FIELDS:
            totals[field] += normalized[field]
        count += 1

    if count == 0:
        return {field: 0.0 for field in RUBRIC_FIELDS}

    return {field: round(totals[field] / count, 2) for field in RUBRIC_FIELDS}


def calculate_option_composite_score(scores: dict[str, Any]) -> float:
    """Calculate arithmetic mean over rubric fields (0-10)."""
    normalized = normalize_rubric_scores(scores)
    return round(sum(normalized.values()) / len(RUBRIC_FIELDS), 2)
