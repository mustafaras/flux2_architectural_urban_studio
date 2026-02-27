"""Performance dashboard page for caching and runtime metrics."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import streamlit as st

from ui import icons


def _format_seconds(v: float | int | None) -> str:
    if v is None:
        return "n/a"
    return f"{float(v):.3f}s"


def render() -> None:
    icons.page_intro(
        "Performance Dashboard",
        "Inspect cache behavior, timing metrics, memory usage, and runtime trends.",
        icon="bolt",
    )

    from src.flux2.streamlit_adapter import get_adapter

    adapter = get_adapter()
    snapshot = adapter.get_performance_snapshot()

    summary = snapshot.get("summary", {})
    counters = snapshot.get("counters", {})
    queue = snapshot.get("queue", {})
    memory = snapshot.get("memory", {}).get("current", {})

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Requests", int(counters.get("requests", 0)))
    col2.metric("Cache Hit Rate", f"{100.0 * float(summary.get('cache_hit_rate', 0.0)):.1f}%")
    col3.metric("Avg Queue Wait", _format_seconds(summary.get("avg_wait_s", 0.0)))
    col4.metric("Errors", int(counters.get("errors", 0)))
    col5.metric("Queue Max", int(summary.get("max_queue", 0)))

    st.divider()

    phase_avg = summary.get("avg_phase_seconds", {})
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Model Load", _format_seconds(phase_avg.get("model_load")))
    p2.metric("Preprocess", _format_seconds(phase_avg.get("preprocessing")))
    p3.metric("Inference", _format_seconds(phase_avg.get("inference")))
    p4.metric("Postprocess", _format_seconds(phase_avg.get("postprocessing")))

    st.divider()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("GPU Used", f"{float(memory.get('used_mb', 0.0)):.0f} MB")
    m2.metric("GPU Total", f"{float(memory.get('total_mb', 0.0)):.0f} MB")
    used_ratio = float(memory.get("used_ratio", 0.0))
    m3.metric("GPU Utilization", f"{used_ratio * 100:.1f}%")
    m4.metric("CUDA Reserved", f"{float(memory.get('reserved_mb', 0.0)):.0f} MB")

    st.divider()

    runtime_samples = snapshot.get("runtime_samples", [])
    if runtime_samples:
        chart_cols = [
            c
            for c in [
            "gpu_utilization_percent",
            "gpu_memory_used_mb",
            "gpu_temperature_c",
            "gpu_power_w",
            "cuda_allocated_mb",
            "cuda_reserved_mb",
            ]
            if c in runtime_samples[0]
        ]

        if chart_cols:
            st.subheader("Runtime Trends")
            chart_data = {col: [row.get(col) for row in runtime_samples] for col in chart_cols}
            st.line_chart(chart_data, height=260)

    phase_timings = snapshot.get("phase_timings", [])
    if phase_timings:
        st.subheader("Recent Phase Timings")
        recent = sorted(phase_timings[-200:], key=lambda row: row.get("timestamp", 0), reverse=True)
        table_data = [
            {
                "datetime": datetime.fromtimestamp(float(row.get("timestamp", 0))).isoformat(timespec="seconds"),
                "name": row.get("name"),
                "seconds": float(row.get("seconds", 0.0)),
            }
            for row in recent
        ]
        st.dataframe(table_data, hide_index=True, use_container_width=True)

    st.divider()

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Sample Runtime Now", use_container_width=True):
            adapter.get_performance_snapshot()
            st.rerun()
    with c2:
        if st.button("Clear Generation Cache", use_container_width=True):
            adapter.clear_generation_cache()
            st.success("Generation result cache cleared.")
    with c3:
        if st.button("Export Metrics JSON", use_container_width=True):
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            out = Path("benchmarks") / f"performance_snapshot_{now}.json"
            from src.flux2.performance_metrics import get_performance_collector

            p = get_performance_collector().export_json(out)
            st.success(f"Exported metrics to {p}")

    st.caption(f"Queue waits collected: {len(queue.get('waits_s', []))}")
