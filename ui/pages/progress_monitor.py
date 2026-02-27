"""
Real-time progress monitoring for FLUX.2 generation.

Displays:
- Live progress bar and ETA countdown
- Denoise timeline visualization
- Intermediate frame previews
- GPU monitoring dashboard (VRAM, temperature, power, utilization)
- Performance metrics and trends
- Quality vs. speed tradeoffs
"""

from __future__ import annotations

import io
import time
from statistics import mean

import streamlit as st
from PIL import Image

from ui import icons
from ui import state
from ui.components_advanced import progress_gauge


def render() -> None:
    """Main entry point for progress monitor page."""
    icons.page_intro(
        "Real-time Progress Monitor",
        "Track live generation progress, previews, hardware usage, and runtime trends.",
        icon="activity",
    )

    from src.flux2.streamlit_adapter import get_adapter
    from src.flux2.gif_generator import get_gif_generator

    adapter = get_adapter()
    progress = adapter.get_progress_snapshot()
    perf = adapter.get_performance_snapshot()
    gif_gen = get_gif_generator()

    # Top-level metrics and progress gauge
    _render_top_metrics(progress)
    st.divider()

    # Denoise timeline with current step visualization
    _render_denoise_timeline(progress)
    st.divider()

    # Intermediate preview frames and GIF animation
    _render_preview_section(progress, gif_gen)
    st.divider()

    # GPU dashboard with detailed monitoring
    _render_gpu_dashboard(perf, progress)
    st.divider()

    # Network metrics if using API upsampling
    _render_network_metrics(progress)
    st.divider()

    # Performance trends and quality/speed analysis
    _render_performance_analysis(progress)
    st.divider()

    # Auto-refresh controls
    _render_auto_refresh_controls(progress)


def _render_top_metrics(progress: dict) -> None:
    """Display top-level progress metrics."""
    status = str(progress.get("status", "idle")).upper()
    mode = str(progress.get("mode", "idle")).upper()
    progress_pct = float(progress.get("progress_percent", 0.0))
    eta_s = float(progress.get("eta_s", 0.0))
    step_per_sec = float(progress.get("step_per_sec", 0.0))

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Status", status, help="Generation status: IDLE, RUNNING, COMPLETED, FAILED")
    col2.metric("Mode", mode, help="Generation mode: T2I (text) or I2I (image)")
    col3.metric("Progress", f"{progress_pct:.1f}%", help="Percentage complete")
    col4.metric("ETA", f"{_format_eta(eta_s)}", help="Estimated time remaining")
    col5.metric("Speed", f"{step_per_sec:.2f} step/s", help="Inference speed (steps/second)")

    # Progress gauge with ETA countdown
    gauge_col, info_col = st.columns([1, 3])
    with gauge_col:
        progress_gauge(
            label="Denoise\nProgress",
            value=max(0.0, min(1.0, progress_pct / 100.0)),
            subtitle=f"Step {int(progress.get('current_step', 0))}/{int(progress.get('total_steps', 0))}",
            key_prefix="progress_monitor_gauge",
        )

    with info_col:
        current_step = int(progress.get("current_step", 0))
        total_steps = int(progress.get("total_steps", 0))

        if progress.get("status") == "running":
            st.info(
                f"**{_format_eta(eta_s)}** remaining • "
                f"**{step_per_sec:.2f}** steps/sec • "
                f"**{_format_elapsed(progress.get('elapsed_s', 0.0))}** elapsed"
            )
        elif progress.get("status") == "completed":
            elapsed = float(progress.get("elapsed_s", 0.0))
            st.success(f"✓ Generation completed in {_format_elapsed(elapsed)}")
        elif progress.get("error_message"):
            st.error(f"✗ Generation failed: {progress.get('error_message')}")


def _render_denoise_timeline(progress: dict) -> None:
    """Render denoise process timeline."""
    st.subheader("Denoise Timeline")

    curr_step = int(progress.get("current_step", 0))
    total_steps = max(1, int(progress.get("total_steps", 1)))
    t_curr = float(progress.get("current_timestep", 1.0))

    col1, col2 = st.columns([3, 1])

    with col1:
        # Slider showing current position
        st.slider(
            "Current denoising step",
            min_value=0,
            max_value=total_steps,
            value=min(curr_step, total_steps),
            disabled=True,
            key="denoise_progress_slider",
        )

    with col2:
        st.caption(f"σ={t_curr:.4f}")  # Noise level indicator

    # Timeline chart showing progress metrics over time
    history = progress.get("history", [])
    if history and len(history) > 1:
        st.line_chart(
            {
                "progress %": [float(row.get("progress_percent", 0.0)) for row in history],
                "steps/sec": [
                    float(row.get("steps_per_sec", 0.0)) for row in history
                ],  # Scale down for visibility
                "ETA (s)": [float(row.get("eta_s", 0.0)) for row in history],
            },
            height=280,
        )
    elif history:
        st.caption("Waiting for more data to display timeline...")


def _render_preview_section(progress: dict, gif_gen: Any) -> None:
    """Render intermediate frame previews and GIF animation."""
    st.subheader("Intermediate Previews")

    preview_frames = progress.get("preview_frames", [])
    frame_count = int(progress.get("frame_count", 0))

    if not preview_frames:
        st.caption("⏳ Preview frames are captured during active inference every 4 steps.")
        st.info(
            "Frames will appear here as the generation progresses. "
            "Use these to monitor quality in real-time."
        )
        return

    # Show preview info
    col1, col2, col3 = st.columns(3)
    col1.metric("Frames Captured", len(preview_frames), help="Number of intermediate frames")
    col2.metric("Total Frames", frame_count, help="Total frames captured during generation")
    col3.metric(
        "Avg Capture Time",
        f"{mean([float(f.get('capture_latency_ms', 0)) for f in preview_frames]):.1f} ms"
        if preview_frames
        else "n/a",
        help="Average frame capture latency",
    )

    # Display frames in grid (most recent frames)
    st.caption("Most recent preview frames (3×3 grid)")
    recent_frames = preview_frames[-9:] if len(preview_frames) > 9 else preview_frames
    cols = st.columns(3)

    for idx, frame_data in enumerate(recent_frames):
        with cols[idx % 3]:
            try:
                frame_bytes = frame_data.get("frame_bytes")
                if isinstance(frame_bytes, bytes):
                    img = Image.open(io.BytesIO(frame_bytes))
                    progress_pct = float(frame_data.get("progress_percent", 0.0))
                    st.image(
                        img,
                        use_container_width=True,
                        caption=f"Step {int(frame_data.get('step_idx', 0))} ({progress_pct:.0f}%)",
                    )
            except Exception as e:
                st.warning(f"Could not display frame {idx}: {e}")

    # GIF animation generation
    st.divider()
    with st.expander(icons.label("Preview Animation (GIF)", "film"), expanded=False):
        col_gen, col_download = st.columns([2, 1])

        with col_gen:
            if st.button("Generate Preview GIF", key="gen_preview_gif"):
                with st.spinner("Creating animated GIF... (target: <2s)"):
                    frame_bytes_list = [
                        f.get("frame_bytes") for f in preview_frames if f.get("frame_bytes")
                    ]
                    start_time = time.time()
                    gif_data = gif_gen.generate_fast(frame_bytes_list, max_frames=12)
                    elapsed = time.time() - start_time

                    if gif_data:
                        size_mb = len(gif_data) / (1024 ** 2)
                        st.session_state["latest_preview_gif"] = gif_data
                        st.success(
                            f"✓ GIF created: {size_mb:.2f} MB in {elapsed:.2f}s"
                        )
                        st.image(gif_data, caption="Preview Animation")
                    else:
                        st.error("Failed to generate GIF")

        with col_download:
            gif_data = st.session_state.get("latest_preview_gif")
            if gif_data:
                st.download_button(
                    label=icons.label("Download GIF", "download"),
                    data=gif_data,
                    file_name="inference_preview.gif",
                    mime="image/gif",
                    key="download_preview_gif",
                )


def _render_gpu_dashboard(perf: dict, progress: dict) -> None:
    """Render detailed GPU monitoring dashboard."""
    st.subheader("System Resource Dashboard")

    runtime_samples = perf.get("runtime_samples", [])
    memory = perf.get("memory", {}).get("current", {})

    # Get latest samples for metrics
    if runtime_samples:
        latest = runtime_samples[-1]
        gpu_temp = float(latest.get("gpu_temperature_c") or 0.0)
        gpu_power = float(latest.get("gpu_power_w") or 0.0)
        gpu_util = float(latest.get("gpu_utilization_percent") or 0.0)
    else:
        gpu_temp = 0.0
        gpu_power = 0.0
        gpu_util = 0.0

    # VRAM metrics
    vram_used_mb = float(memory.get("used_mb", 0.0))
    vram_total_mb = float(memory.get("total_mb", 0.0))
    vram_pct = 100.0 * float(memory.get("used_ratio", 0.0))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "VRAM Used",
        f"{vram_used_mb:.0f} MB / {vram_total_mb:.0f} MB",
        f"{vram_pct:.1f}%",
        help="GPU memory utilization",
    )
    col2.metric(
        "GPU Util",
        f"{gpu_util:.1f}%",
        help="GPU utilization percentage",
    )
    col3.metric(
        "Temperature",
        f"{gpu_temp:.1f} °C",
        help="GPU die temperature",
    )
    col4.metric(
        "Power",
        f"{gpu_power:.1f} W",
        help="GPU power consumption",
    )

    # Warnings
    if gpu_temp >= 85.0:
        st.error(
            f"⚠️ **Critical**: GPU temperature {gpu_temp:.1f}°C - Thermal throttling likely!"
        )
    elif gpu_temp >= 80.0:
        st.warning(
            f"⚠️ **Warning**: GPU temperature {gpu_temp:.1f}°C - Risk of throttling"
        )

    if vram_pct >= 95.0:
        st.error(f"⚠️ **Critical**: VRAM at {vram_pct:.1f}% - OOM risk!")
    elif vram_pct >= 85.0:
        st.warning(f"⚠️ **Warning**: VRAM at {vram_pct:.1f}% - Monitor closely")

    # Timeline charts for GPU metrics
    if runtime_samples and len(runtime_samples) > 1:
        st.caption("GPU Metrics Timeline (last 60 samples)")

        chart_data = {
            "VRAM (MB)": [
                float(row.get("gpu_memory_used_mb") or 0.0) for row in runtime_samples[-60:]
            ],
            "GPU %": [
                float(row.get("gpu_utilization_percent") or 0.0) for row in runtime_samples[-60:]
            ],
            "Power (W)": [
                float(row.get("gpu_power_w") or 0.0) for row in runtime_samples[-60:]
            ],
        }

        st.line_chart(chart_data, height=320)

    # System resources
    st.divider()
    cpu_pct, ram_pct, net_throughput = _probe_system_resources()
    recommended_batch = _recommend_batch_size(
        used_ratio=float(memory.get("used_ratio", 0.0)),
        cpu_percent_str=cpu_pct,
        ram_percent_str=ram_pct,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("CPU Usage", cpu_pct, help="System CPU utilization")
    c2.metric("RAM Usage", ram_pct, help="System RAM utilization")
    c3.metric(
        "Recommended Batch",
        recommended_batch,
        help="Suggested batch size based on resource pressure",
    )


def _render_network_metrics(progress: dict) -> None:
    """Render API network metrics (for upsampling calls)."""
    st.subheader("API Network Metrics")

    network = progress.get("network", {}) if isinstance(progress, dict) else {}
    api_latencies = [float(v) for v in network.get("api_latency_ms", [])]

    n1, n2, n3 = st.columns(3)
    n1.metric("API Calls", str(int(network.get("request_count", 0))))
    n2.metric("Rate Limit Status", str(network.get("rate_limit_status", "unknown")).upper())

    if api_latencies:
        n3.metric(
            "Avg Latency",
            f"{mean(api_latencies):.1f} ms",
            help="Average API call latency",
        )
        st.line_chart({"API Latency (ms)": api_latencies[-60:]}, height=200)
    else:
        n3.metric("Avg Latency", "n/a")
        st.caption("No API calls recorded yet.")

    # Last API error if any
    last_error = network.get("last_error")
    if last_error:
        st.warning(f"Last API error: {last_error}")


def _render_performance_analysis(progress: dict) -> None:
    """Render performance analysis and trends."""
    st.subheader("Performance Analysis")

    quality_speed = progress.get("quality_speed_history", [])
    history = progress.get("history", [])

    col1, col2 = st.columns(2)

    with col1:
        st.caption("Quality vs. Speed Tradeoff")
        if quality_speed and len(quality_speed) > 1:
            st.line_chart(
                {
                    "Quality Proxy %": [
                        float(row.get("quality_proxy", 0.0)) for row in quality_speed
                    ],
                    "Speed (steps/sec)": [
                        float(row.get("speed_steps_per_sec", 0.0)) for row in quality_speed
                    ],
                },
                height=240,
            )
        else:
            st.caption("Data will appear during generation...")

    with col2:
        st.caption("Step Timing")
        if history and len(history) > 1:
            step_times = []
            for i in range(1, len(history)):
                prev_time = float(history[i - 1].get("timestamp", 0.0))
                curr_time = float(history[i].get("timestamp", 0.0))
                if curr_time > prev_time:
                    step_times.append(1000.0 * (curr_time - prev_time))  # ms

            if step_times:
                st.line_chart({"Step Duration (ms)": step_times}, height=240)
        else:
            st.caption("Data will appear during generation...")

    # Comparative metrics from sessions
    st.divider()
    st.caption("Session Generation History")
    generation_history = st.session_state.get("generation_history", [])
    if generation_history:
        timings = []
        for sess in generation_history:
            meta = sess.get("metadata", {})
            if isinstance(meta, dict) and meta.get("generation_time_s") is not None:
                timings.append(float(meta["generation_time_s"]))

        if timings:
            k1, k2, k3 = st.columns(3)
            k1.metric("Latest", f"{timings[0]:.2f}s")
            k2.metric("Average", f"{mean(timings):.2f}s")
            recent_trend = (timings[0] - mean(timings[-5:])) if len(timings) > 1 else 0.0
            trend = "improving" if recent_trend < 0 else "degrading"
            k3.metric("Trend", trend)

            st.line_chart({"Generation Time (s)": timings[-20:]}, height=200)

        with st.expander("Restore Settings From History", expanded=False):
            history_labels = []
            for idx, sess in enumerate(generation_history[:20]):
                meta = sess.get("metadata", {}) if isinstance(sess, dict) else {}
                label = (
                    f"#{idx + 1} · {meta.get('model', '?')} · seed {meta.get('seed', '?')} "
                    f"· {meta.get('width', '?')}x{meta.get('height', '?')}"
                )
                history_labels.append(label)

            selected_idx = st.selectbox(
                "History run",
                options=list(range(len(history_labels))),
                format_func=lambda idx: history_labels[idx],
                key="progress_restore_history_idx",
            )

            if st.button("Restore Selected Settings", use_container_width=True, key="progress_restore_history_btn"):
                selected_entry = generation_history[selected_idx]
                if state.restore_settings_from_history_entry(selected_entry):
                    st.success("Settings restored from history.")
                    st.rerun()
                else:
                    st.warning("Could not restore settings from this history entry.")
    else:
        st.caption("No generation history yet.")


def _render_auto_refresh_controls(progress: dict) -> None:
    """Render auto-refresh controls."""
    st.subheader("Live Updates")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.session_state.progress_auto_refresh = st.toggle(
            "Auto-refresh during generation",
            value=bool(st.session_state.get("progress_auto_refresh", True)),
            help="Automatically refresh page every N seconds while generation is active",
        )

    with col2:
        refresh_interval = st.number_input(
            "Refresh interval (s)",
            min_value=1,
            max_value=5,
            value=int(st.session_state.get("progress_refresh_interval", 2)),
            step=1,
            help="Update frequency: 1-5 seconds",
        )
        st.session_state.progress_refresh_interval = int(refresh_interval)

    # Auto-refresh logic
    if progress.get("status") == "running" and st.session_state.get("progress_auto_refresh", True):
        time.sleep(int(st.session_state.get("progress_refresh_interval", 2)))
        st.rerun()


# ─── Helper functions ────────────────────────────────────────────────────────


def _format_eta(seconds: float) -> str:
    """Format ETA in human-readable form."""
    s = max(0.0, float(seconds))
    if s < 60.0:
        return f"{s:.1f}s"
    elif s < 3600.0:
        m = s / 60.0
        return f"{m:.1f}m"
    else:
        h = s / 3600.0
        return f"{h:.1f}h"


def _format_elapsed(seconds: float) -> str:
    """Format elapsed time."""
    s = max(0.0, float(seconds))
    if s < 60.0:
        return f"{s:.1f}s"
    elif s < 3600.0:
        m = s / 60.0
        s_remaining = s % 60.0
        return f"{int(m)}m {s_remaining:.0f}s"
    else:
        h = int(s // 3600.0)
        m = int((s % 3600.0) // 60.0)
        return f"{h}h {m}m"


def _recommend_batch_size(used_ratio: float, cpu_percent_str: str, ram_percent_str: str) -> str:
    """Recommend batch size based on resource pressure."""
    try:
        cpu = float(cpu_percent_str.replace("%", "").strip())
        ram = float(ram_percent_str.replace("%", "").strip())
    except (ValueError, AttributeError):
        cpu = ram = 0.0

    pressure = max(used_ratio * 100.0, cpu, ram)
    if pressure >= 90.0:
        return "1 (critical)"
    if pressure >= 75.0:
        return "2 (moderate)"
    if pressure >= 50.0:
        return "4 (good)"
    return "8+ (optimal)"


def _probe_system_resources() -> tuple[str, str, str]:
    """Probe CPU, RAM, and network resources."""
    try:
        import psutil  # type: ignore

        cpu = psutil.cpu_percent(interval=0.1)
        ram = psutil.virtual_memory().percent
        net_io = psutil.net_io_counters()
        throughput_mb = (net_io.bytes_sent + net_io.bytes_recv) / (1024 * 1024)

        return f"{cpu:.1f}%", f"{ram:.1f}%", f"{throughput_mb:.1f} MB"
    except Exception:
        return "n/a", "n/a", "n/a"

