from __future__ import annotations

import csv
import io
import json

import streamlit as st

from src.flux2.logging_config import get_recent_operations
from ui import error_handler
from ui import icons


def render() -> None:
    icons.heading("Debug Dashboard", icon="activity")
    st.caption("Last 50 operations and error events with sanitized details.")

    operations = get_recent_operations(limit=50)
    if not operations:
        st.info("No operations recorded yet.")
        return

    st.metric("Recorded operations", len(operations))

    error_count = sum(1 for item in operations if str(item.get("name", "")).startswith("error:"))
    st.metric("Error events", error_count)

    durations = _extract_duration_seconds(operations)
    recovery_success_rate = _calculate_recovery_success_rate(operations)

    c1, c2 = st.columns(2)
    with c1:
        avg_time = (sum(durations) / len(durations)) if durations else 0.0
        st.metric("Avg operation time", f"{avg_time:.2f}s")
    with c2:
        st.metric("Recovery success rate", f"{recovery_success_rate:.1f}%")

    errors_by_type = _error_frequency(operations)
    if errors_by_type:
        st.subheader("Error frequency by type")
        st.bar_chart(errors_by_type)

    history = error_handler.get_error_history(limit=50)
    if history:
        st.subheader("Recent user-facing errors")
        st.dataframe(history, use_container_width=True)

    try:
        from src.flux2.streamlit_adapter import get_adapter

        perf = get_adapter().get_performance_snapshot()
        runtime_samples = perf.get("runtime_samples", [])
        memory_samples = [float(row.get("gpu_memory_used_mb") or 0.0) for row in runtime_samples]
        if memory_samples:
            st.subheader("Memory usage trend")
            st.line_chart({"gpu_memory_used_mb": memory_samples[-120:]}, height=180)
    except Exception:
        pass

    st.subheader("Export logs")
    st.download_button(
        label="Export operation logs (JSON)",
        data=json.dumps(operations, indent=2, ensure_ascii=False),
        file_name="flux2_debug_operations.json",
        mime="application/json",
        use_container_width=True,
    )
    st.download_button(
        label="Export operation logs (CSV)",
        data=_operations_to_csv_bytes(operations),
        file_name="flux2_debug_operations.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.divider()
    for idx, item in enumerate(reversed(operations), start=1):
        title = f"#{idx} {item.get('timestamp', '')} — {item.get('name', 'operation')} [{item.get('status', '')}]"
        with st.expander(title, expanded=False):
            st.code(json.dumps(item.get("details", {}), indent=2, ensure_ascii=False), language="json")


def _extract_duration_seconds(operations: list[dict]) -> list[float]:
    durations: list[float] = []
    for op in operations:
        details = op.get("details", {})
        if not isinstance(details, dict):
            continue
        value = details.get("time_s")
        if value is None:
            continue
        try:
            durations.append(float(value))
        except Exception:
            continue
    return durations


def _error_frequency(operations: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for op in operations:
        name = str(op.get("name", ""))
        if not name.startswith("error:"):
            continue
        category = name.split(":", 1)[1] if ":" in name else "UNKNOWN"
        counts[category] = counts.get(category, 0) + 1
    return counts


def _calculate_recovery_success_rate(operations: list[dict]) -> float:
    starts = sum(1 for op in operations if str(op.get("name", "")).endswith(".start"))
    finishes = sum(1 for op in operations if str(op.get("name", "")).endswith(".finish"))
    if starts <= 0:
        return 100.0
    return max(0.0, min(100.0, (finishes / starts) * 100.0))


def _operations_to_csv_bytes(operations: list[dict]) -> bytes:
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=["timestamp", "name", "status", "details_json"],
    )
    writer.writeheader()
    for op in operations:
        writer.writerow(
            {
                "timestamp": op.get("timestamp", ""),
                "name": op.get("name", ""),
                "status": op.get("status", ""),
                "details_json": json.dumps(op.get("details", {}), ensure_ascii=False),
            }
        )
    return buffer.getvalue().encode("utf-8")
