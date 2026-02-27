"""Baseline KPI aggregation helpers for architecture workflow telemetry."""

from __future__ import annotations

from datetime import datetime
from typing import Any


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return None


def build_baseline_kpi_report(session_state: dict[str, Any]) -> dict[str, Any]:
    """Build baseline KPI report from session metrics."""
    started_at = _parse_iso(str(session_state.get("kpi_session_started_at", "")))
    first_shortlist_at = _parse_iso(str(session_state.get("kpi_first_shortlist_ts", "")))

    time_to_first_board_s: float | None = None
    if started_at and first_shortlist_at:
        delta = (first_shortlist_at - started_at).total_seconds()
        if delta >= 0:
            time_to_first_board_s = round(delta, 2)

    latencies = session_state.get("kpi_generation_latencies", [])
    queue_waits = session_state.get("kpi_queue_wait_times_ms", [])

    valid_latencies = [float(x) for x in latencies if isinstance(x, (int, float))]
    valid_waits = [float(x) for x in queue_waits if isinstance(x, (int, float))]

    avg_latency_s = round(sum(valid_latencies) / len(valid_latencies), 3) if valid_latencies else 0.0
    avg_queue_wait_s = round((sum(valid_waits) / len(valid_waits)) / 1000.0, 3) if valid_waits else 0.0

    return {
        "session_started_at": session_state.get("kpi_session_started_at"),
        "time_to_first_board_s": time_to_first_board_s,
        "iteration_count_session": int(session_state.get("kpi_iteration_count", 0)),
        "generation_latency_avg_s": avg_latency_s,
        "queue_wait_avg_s": avg_queue_wait_s,
        "compose_prompt_count": int(session_state.get("kpi_compose_prompt_count", 0)),
        "shortlist_count": int(session_state.get("kpi_shortlist_count", 0)),
        "board_export_count": int(session_state.get("kpi_board_export_count", 0)),
    }
