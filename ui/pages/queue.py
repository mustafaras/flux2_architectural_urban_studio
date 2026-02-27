from __future__ import annotations

import io
import json
import time
import zipfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

import streamlit as st

from src.flux2.analytics_client import EventType, get_analytics
from src.flux2.error_types import classify_exception
from src.flux2.logging_config import log_error, log_operation
from src.flux2.queue_manager import GenerationRequest, GenerationQueue, QueueLane, SchedulingType
from ui import config as cfg
from ui import error_handler, icons, utils
from ui import state


_QUEUE_STATUS_ICONS = {
    "queued": "🟡",
    "running": "🔵",
    "completed": "✅",
    "failed": "❌",
}


def format_queue_status_chip(state_name: str, count: int) -> str:
    """Return a compact status chip label for queue state display."""
    icon = _QUEUE_STATUS_ICONS.get(state_name, "•")
    return f"{icon} {state_name.title()}: {int(count)}"


# ============================================================================
# MAIN RENDER & SESSION INIT
# ============================================================================


def render() -> None:
    """Main queue page renderer"""
    icons.page_intro(
        cfg.QUEUE_TAB_TITLE,
        "Queue requests, configure scheduling/templates, and monitor queued and completed jobs.",
        icon="queue",
    )
    
    queue = _get_queue()
    _initialize_session_state()

    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        icons.tab("Queue", "queue"),
        icons.tab("Scheduling", "history"),
        icons.tab("Templates", "edit"),
        icons.tab("Stats", "activity"),
    ])
    
    with tab1:
        _render_queue_tab(queue)
    with tab2:
        _render_scheduling_tab(queue)
    with tab3:
        _render_templates_tab(queue)
    with tab4:
        _render_stats_tab(queue)


def _get_queue() -> GenerationQueue:
    """Get or create queue with persistence"""
    queue = st.session_state.get("generation_queue")
    if queue is None:
        persist_path = Path("outputs") / ".queue_state.json"
        queue = GenerationQueue(max_size=50, persist_path=persist_path)
        st.session_state.generation_queue = queue
    return queue


def _initialize_session_state() -> None:
    """Initialize all session state variables"""
    defaults = {
        "queue_auto_run": False,
        "queue_paused": False,
        "queue_last_tick_ts": 0.0,
        "queue_templates": st.session_state.get("queue_templates", {}),
        "queue_show_advanced": False,
        "queue_scheduling_enabled": False,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


# ============================================================================
# TAB 1: QUEUE MANAGEMENT
# ============================================================================


def _render_queue_tab(queue: GenerationQueue) -> None:
    """Render main queue management tab"""
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("Enqueue Request")
    with col2:
        st.session_state.queue_show_advanced = st.checkbox("Advanced", value=st.session_state.get("queue_show_advanced", False))
    
    _render_enqueue_form(queue)
    st.divider()
    _render_batch_controls(queue)
    _auto_tick(queue)
    st.divider()
    _render_status_metrics(queue)
    st.divider()
    _render_queue_items(queue)
    st.divider()
    _render_completed_section(queue)


def _render_enqueue_form(queue: GenerationQueue) -> None:
    """Render single/batch enqueue form"""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        prompt = st.text_area("Prompt", value=st.session_state.get("prompt", ""), key="queue_prompt", height=80)
    
    with col2:
        mode = st.radio("Mode", ["Single", "Batch", "Template"], horizontal=True)
    
    if mode == "Single":
        _render_single_enqueue(queue, prompt)
    elif mode == "Batch":
        _render_batch_enqueue(queue, prompt)
    else:
        _render_template_enqueue(queue)


def _render_single_enqueue(queue: GenerationQueue, prompt: str) -> None:
    """Single request enqueue form"""
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        priority = st.slider("Priority", min_value=-1, max_value=2, value=0, help="Higher = earlier execution")
    with col2:
        schedule_enabled = st.checkbox("Schedule", value=False)
    with col3:
        save_template = st.checkbox("Save template", value=False)
    with col4:
        lane = st.selectbox("Queue Lane", options=[lane.value for lane in QueueLane], index=1, key="queue_lane_single")
    
    scheduled_for = None
    if schedule_enabled:
        date_val = st.date_input("Date", value=datetime.now().date())
        time_val = st.time_input("Time", value=datetime.now().time().replace(microsecond=0))
        scheduled_for = datetime.combine(date_val, time_val).replace(tzinfo=timezone.utc)
    
    if st.button("Enqueue", type="primary", use_container_width=True):
        err = utils.validate_prompt(prompt)
        if err:
            st.error(err)
            return

        active_project = state.get_active_project()
        if not active_project:
            st.error("Create or activate a project in Project Setup before queueing requests.")
            return
        
        payload = _get_generation_settings()
        req = GenerationRequest(
            prompt=prompt,
            settings=payload,
            priority=priority,
            scheduled_for=scheduled_for,
            scheduling_type=SchedulingType.SCHEDULED if scheduled_for else SchedulingType.IMMEDIATE,
            queue_lane=lane,
        )
        queue.enqueue(req, priority=priority)
        log_operation("queue.enqueue", "success", {"request_id": req.request_id, "priority": priority})
        
        if save_template:
            _save_template(queue, req)
        
        st.success(f"{icons.label('Queued', 'check')}: {req.request_id}")
        st.session_state.prompt = ""
        time.sleep(0.5)
        st.rerun()


def _render_batch_enqueue(queue: GenerationQueue, prompt: str) -> None:
    """Batch enqueue form"""
    col1, col2, col3 = st.columns(3)
    with col1:
        batch_count = st.number_input("Number of variations", min_value=2, max_value=20, value=5)
    with col2:
        seed_mode = st.select_slider("Seed variation", options=["Same", "Increment", "Random"], value="Increment")
    with col3:
        lane = st.selectbox("Queue Lane", options=[lane.value for lane in QueueLane], index=2, key="queue_lane_batch")
    
    priority = st.slider("Priority", min_value=-1, max_value=2, value=0)
    
    if st.button("Enqueue Batch", type="primary", use_container_width=True):
        err = utils.validate_prompt(prompt)
        if err:
            st.error(err)
            return

        active_project = state.get_active_project()
        if not active_project:
            st.error("Create or activate a project in Project Setup before queueing requests.")
            return
        
        payload = _get_generation_settings()
        base_seed = payload.get("seed", 1)
        
        requests = []
        for i in range(batch_count):
            req_settings = payload.copy()
            if seed_mode == "Increment":
                req_settings["seed"] = base_seed + i
            elif seed_mode == "Random":
                req_settings["seed"] = hash(f"{prompt}{i}") % 2**31
            
            req = GenerationRequest(
                prompt=prompt,
                settings=req_settings,
                priority=priority,
                scheduling_type=SchedulingType.BATCH,
                queue_lane=lane,
            )
            requests.append(req)
        
        ids = queue.enqueue_batch(requests, priority=priority)
        st.success(f"{icons.label('Enqueued', 'check')} {len(ids)} variations")
        time.sleep(0.5)
        st.rerun()


def _render_template_enqueue(queue: GenerationQueue) -> None:
    """Enqueue from saved templates"""
    templates = st.session_state.get("queue_templates", {})
    if not templates:
        st.info("No saved templates yet. Use 'Save template' in Single mode.")
        return
    
    template_name = st.selectbox("Select template", list(templates.keys()))
    prompt = st.text_area("Prompt", value="", key="template_prompt")
    priority = st.slider("Priority", min_value=-1, max_value=2, value=0)
    lane = st.selectbox("Queue Lane", options=[lane.value for lane in QueueLane], index=1, key="queue_lane_template")
    
    if st.button("Use Template", type="primary", use_container_width=True):
        err = utils.validate_prompt(prompt)
        if err:
            st.error(err)
            return

        active_project = state.get_active_project()
        if not active_project:
            st.error("Create or activate a project in Project Setup before queueing requests.")
            return
        
        template_settings = template.copy() if (template := templates.get(template_name)) else {}
        req = GenerationRequest(
            prompt=prompt,
            settings=template_settings,
            priority=priority,
            template_name=template_name,
            queue_lane=lane,
        )
        queue.enqueue(req, priority=priority)
        st.success(f"{icons.label('Queued from template', 'check')}: {template_name}")
        time.sleep(0.5)
        st.rerun()


def _render_batch_controls(queue: GenerationQueue) -> None:
    """Batch control buttons"""
    st.subheader("Batch Controls")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if st.button(icons.label("Start", "play"), use_container_width=True):
            st.session_state.queue_auto_run = True
            st.session_state.queue_paused = False
            st.info("Auto-processing started")
    
    with col2:
        if st.button(icons.label("Pause", "pause"), use_container_width=True):
            st.session_state.queue_paused = True
            st.info("Queue paused")
    
    with col3:
        if st.button(icons.label("Resume", "resume"), use_container_width=True):
            st.session_state.queue_paused = False
            st.info("Queue resumed")
    
    with col4:
        if st.button(icons.label("Boost Wait", "boost"), use_container_width=True):
            queue.scale_priority_by_wait_time()
            st.info("Priority boosted for waiting items")
    
    with col5:
        if st.button(icons.label("Clear All", "trash"), use_container_width=True):
            if st.session_state.get("confirm_clear"):
                queue.clear()
                st.session_state.confirm_clear = False
                st.success("Queue cleared")
                st.rerun()
            else:
                st.session_state.confirm_clear = True
                st.warning("Click again to confirm")

    with st.expander("Orchestration Controls", expanded=False):
        gpu_cap = st.slider(
            "GPU Concurrency Cap",
            min_value=1,
            max_value=4,
            value=int(st.session_state.get("queue_gpu_cap", 1)),
            help="Limits concurrent GPU queue workers.",
            key="queue_gpu_cap_slider",
        )
        st.session_state.queue_gpu_cap = gpu_cap
        queue.set_gpu_concurrency_cap(gpu_cap)

        saturation_threshold = st.slider(
            "Saturation Threshold",
            min_value=5,
            max_value=50,
            value=int(st.session_state.get("queue_saturation_threshold", 20)),
            help="Apply fallback profile when queue reaches this size.",
            key="queue_saturation_threshold_slider",
        )
        st.session_state.queue_saturation_threshold = saturation_threshold
        if st.button("Apply Saturation Fallback", use_container_width=True, key="queue_apply_saturation"):
            changed = queue.apply_saturation_fallback(threshold=saturation_threshold)
            st.info(f"Applied fallback profile to {changed} queued requests.")


def _auto_tick(queue: GenerationQueue) -> None:
    """Auto-process queue items"""
    if not st.session_state.get("queue_auto_run", False):
        return
    if st.session_state.get("queue_paused", False):
        return

    now_ts = time.time()
    last = float(st.session_state.get("queue_last_tick_ts", 0.0))
    if now_ts - last < 2.0:
        return

    st.session_state.queue_last_tick_ts = now_ts
    changed = _process_one(queue)
    if changed:
        time.sleep(0.1)
        st.rerun()


def _process_one(queue: GenerationQueue) -> bool:
    """Process single queue item"""
    active = queue.process_next()
    if active is None:
        return False

    settings = active.settings
    prompt = active.prompt
    start = time.perf_counter()
    wait_time_ms = 0
    if active.started_at and active.created_at:
        wait_time_ms = max(0, int((active.started_at - active.created_at).total_seconds() * 1000))
    waits = list(st.session_state.get("kpi_queue_wait_times_ms", []))
    waits.append(wait_time_ms)
    st.session_state.kpi_queue_wait_times_ms = waits[-500:]

    try:
        from src.flux2.streamlit_adapter import get_adapter

        adapter = get_adapter()
        adapter.load(
            model_name=settings["model_name"],
            dtype_str=settings["dtype_str"],
            cpu_offloading=settings["cpu_offloading"],
            attn_slicing=settings["attn_slicing"],
        )
        img = adapter.generate(
            prompt=prompt,
            num_steps=settings["num_steps"],
            guidance=settings["guidance"],
            width=settings["width"],
            height=settings["height"],
            seed=settings["seed"],
        )
        out_path = utils.save_image(
            img,
            settings.get("output_dir", "outputs"),
            settings["model_name"],
            settings["seed"],
            settings["num_steps"],
            settings["guidance"],
        )
        duration = time.perf_counter() - start
        queue.mark_completed(active.request_id, str(out_path), duration_s=duration)
        log_operation("queue.process", "success", {"request_id": active.request_id, "duration_s": round(duration, 3)})
        st.session_state.kpi_iteration_count = int(st.session_state.get("kpi_iteration_count", 0)) + 1
        latencies = list(st.session_state.get("kpi_generation_latencies", []))
        latencies.append(float(duration))
        st.session_state.kpi_generation_latencies = latencies[-200:]
        get_analytics().log_event(
            EventType.GENERATE_OPTION,
            {
                "model": settings.get("model_name", ""),
                "duration_ms": int(duration * 1000),
                "wait_time_ms": wait_time_ms,
                "source": "queue",
            },
        )
        return True
    except Exception as exc:
        ctx = classify_exception(exc, location="queue._process_one", metadata={"request_id": active.request_id})
        log_error(ctx)
        queue.mark_failed(active.request_id, str(exc), retry=True)
        return True


def _render_status_metrics(queue: GenerationQueue) -> None:
    """Display status metrics"""
    status = queue.get_status()

    chip_cols = st.columns(4)
    chip_cols[0].caption(format_queue_status_chip("queued", status["queued"]))
    chip_cols[1].caption(format_queue_status_chip("running", status["running"]))
    chip_cols[2].caption(format_queue_status_chip("completed", status["completed"]))
    chip_cols[3].caption(format_queue_status_chip("failed", status["failed"]))

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    col1.metric("Queued", status["queued"])
    col2.metric("Running", status["running"])
    col3.metric("Completed", status["completed"])
    col4.metric("Failed", status["failed"])
    col5.metric("Success Rate", f"{status['success_rate']:.1f}%")
    col6.metric("ETA", status["eta_text"])

    lane_counts = status.get("lane_counts", {}) if isinstance(status.get("lane_counts"), dict) else {}
    lane_eta_s = status.get("lane_eta_s", {}) if isinstance(status.get("lane_eta_s"), dict) else {}
    lane_cols = st.columns(4)
    for idx, lane in enumerate([lane.value for lane in QueueLane]):
        lane_cols[idx].caption(f"{lane}: {lane_counts.get(lane, 0)} queued · ETA {lane_eta_s.get(lane, 0)}s")

    st.caption("Queue counts use the same queued/running/completed/failed status keys as the sidebar.")
    
    st.progress(float(status["progress"]), text=f"{status['progress_pct']}% complete")


def _render_queue_items(queue: GenerationQueue) -> None:
    """Render queued items with reordering"""
    st.subheader("Queue Items")
    
    if queue.active:
        with st.container(border=True):
            col1, col2, col3 = st.columns([4, 1, 1])
            col1.markdown(f"{icons.label('ACTIVE', 'refresh')} | **{queue.active.request_id}**  \n*{queue.active.prompt[:100]}*")
            confirm_active_key = f"queue_confirm_cancel_{queue.active.request_id}"
            confirm_active = bool(st.session_state.get(confirm_active_key, False))
            cancel_label = "Confirm Cancel" if confirm_active else "Cancel"
            if col2.button(cancel_label, key="cancel_active"):
                if confirm_active:
                    queue.cancel(queue.active.request_id)
                    st.session_state.pop(confirm_active_key, None)
                    st.rerun()
                else:
                    st.session_state[confirm_active_key] = True
                    st.warning("Click Confirm Cancel to stop the active request.")
    
    if not queue.queue:
        st.info("No queued items")
        return
    
    for idx, item in enumerate(queue.queue):
        with st.container(border=True):
            col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 1])
            
            # Item info
            priority_emoji = "🔴" if item.priority > 0 else "🟡" if item.priority == 0 else "🟢"
            col1.markdown(
                f"{priority_emoji} **{item.request_id}**  \n"
                f"*{item.prompt[:80]}*  \n"
                f"{icons.label('Lane', 'queue')}: {item.queue_lane}  \n"
                f"{icons.label('Due', 'clock')}: {_fmt_due(item.scheduled_for)}"
            )
            
            # Controls
            if col2.button("↑", key=f"up_{item.request_id}", help="Move up"):
                queue.reorder(item.request_id, "up")
                st.rerun()
            if col3.button("↓", key=f"down_{item.request_id}", help="Move down"):
                queue.reorder(item.request_id, "down")
                st.rerun()
            if col4.button(icons.label("Priority", "refresh"), key=f"pri_{item.request_id}", help="Change priority"):
                new_pri = st.radio(f"Priority for {item.request_id}", [-1, 0, 1, 2], key=f"pri_radio_{item.request_id}")
                queue.set_priority(item.request_id, new_pri)
                st.rerun()
            confirm_item_key = f"queue_confirm_cancel_{item.request_id}"
            confirm_item = bool(st.session_state.get(confirm_item_key, False))
            cancel_item_label = icons.label("Confirm", "cancel") if confirm_item else icons.label("Cancel", "cancel")
            if col5.button(cancel_item_label, key=f"cancel_{item.request_id}", help="Cancel queued item"):
                if confirm_item:
                    queue.cancel(item.request_id)
                    st.session_state.pop(confirm_item_key, None)
                    st.rerun()
                else:
                    st.session_state[confirm_item_key] = True
                    st.warning(f"Click Confirm to cancel {item.request_id}.")


def _render_completed_section(queue: GenerationQueue) -> None:
    """Render completed/failed results"""
    st.subheader("Results")
    
    tabs_results = st.tabs([
        icons.tab(f"Completed ({len(queue.completed)})", "check"),
        icons.tab(f"Failed ({len(queue.failed)})", "cross"),
        icons.tab(f"Canceled ({len(queue.canceled)})", "cancel"),
    ])
    
    with tabs_results[0]:
        if not queue.completed:
            st.caption("No completed items")
        else:
            for item in queue.completed[:15]:
                col1, col2, col3 = st.columns([3, 1, 1])
                col1.markdown(f"**{item.request_id}**  \n`{item.output_path or '-'}`")
                col2.caption(f"{icons.label(f'{item.duration_s:.2f}s', 'clock')}" if item.duration_s else "")
                if col3.button("View", key=f"view_completed_{item.request_id}"):
                    if item.output_path and Path(item.output_path).exists():
                        st.image(item.output_path)
            
            # Batch download
            completed_paths = [
                Path(item.output_path) for item in queue.completed 
                if item.output_path and Path(item.output_path).exists()
            ]
            if completed_paths:
                data = _zip_files(completed_paths)
                st.download_button(
                    icons.label("Download All as ZIP", "download"),
                    data=data,
                    file_name="batch_outputs.zip",
                    mime="application/zip",
                    use_container_width=True,
                )
    
    with tabs_results[1]:
        if queue.failed:
            for item in queue.failed[:10]:
                with st.expander(icons.label(item.request_id, "cross"), expanded=False):
                    st.error(item.error_message or "Unknown error")
        else:
            st.caption("No failed items")
    
    with tabs_results[2]:
        if queue.canceled:
            for item in queue.canceled[:10]:
                st.caption(icons.label(item.request_id, "cancel"))
        else:
            st.caption("No canceled items")


# ============================================================================
# TAB 2: SCHEDULING
# ============================================================================


def _render_scheduling_tab(queue: GenerationQueue) -> None:
    """Scheduling and templates tab"""
    st.subheader("Schedule & Automate")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Scheduled Generation")
        st.write("Generate at specific times")
        
        date_val = st.date_input("Date", value=(datetime.now() + timedelta(days=1)).date(), key="sch_date")
        time_val = st.time_input("Time", value=datetime.now().time().replace(hour=2, minute=0), key="sch_time")
        prompt = st.text_area("Prompt", key="sch_prompt", height=80)
        
        if st.button("Schedule", type="primary", use_container_width=True):
            scheduled_for = datetime.combine(date_val, time_val).replace(tzinfo=timezone.utc)
            payload = _get_generation_settings()
            req = GenerationRequest(
                prompt=prompt,
                settings=payload,
                scheduled_for=scheduled_for,
                scheduling_type=SchedulingType.SCHEDULED,
            )
            queue.enqueue(req)
            st.success(f"{icons.label('Scheduled for', 'check')} {scheduled_for}")
            st.rerun()
    
    with col2:
        st.markdown("#### Recurring Generation")
        st.write("Generate on a schedule")
        
        days_selected = st.multiselect(
            "Days of week",
            ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            default=["Mon", "Wed", "Fri"],
            key="recurring_days",
        )
        time_val = st.time_input("Time", value=datetime.now().time().replace(hour=9, minute=0), key="recurring_time")
        prompt = st.text_area("Prompt", key="recurring_prompt", height=80)
        
        if st.button("Setup Recurring", use_container_width=True):
            st.info(f"{icons.label('Would generate on', 'refresh')} {', '.join(days_selected)} at {time_val}")
            # TODO: Implement recurring logic


# ============================================================================
# TAB 3: TEMPLATES
# ============================================================================


def _render_templates_tab(queue: GenerationQueue) -> None:
    """Manage saved templates"""
    st.subheader("Templates & Presets")
    
    templates = st.session_state.get("queue_templates", {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Saved Templates")
        if templates:
            for name, settings in list(templates.items()):
                col_a, col_b = st.columns([3, 1])
                col_a.caption(icons.label(name, "template"))
                if col_b.button(icons.label("Remove", "trash"), key=f"delete_template_{name}"):
                    del templates[name]
                    st.session_state.queue_templates = templates
                    st.rerun()
        else:
            st.info("No templates yet")
    
    with col2:
        st.markdown("#### Create Template")
        template_name = st.text_input("Template name", key="new_template_name")
        
        if template_name:
            payload = _get_generation_settings()
            if st.button("Save as Template", use_container_width=True):
                templates[template_name] = payload
                st.session_state.queue_templates = templates
                st.success(f"{icons.label('Template saved', 'check')}: {template_name}")
                st.rerun()


# ============================================================================
# TAB 4: STATISTICS
# ============================================================================


def _render_stats_tab(queue: GenerationQueue) -> None:
    """Display queue statistics"""
    st.subheader("Queue Statistics")
    
    status = queue.get_status()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Total Processed",
            status["completed"] + status["failed"],
            delta=f"{status['success_rate']:.1f}% success"
        )
    
    with col2:
        st.metric(
            "Avg Duration",
            f"{status['avg_duration_s']:.2f}s",
            delta=f"ETA: {status['eta_text']}"
        )
    
    with col3:
        st.metric(
            "Queue Capacity",
            f"{status['current_size']}/{status['max_size']}",
            delta=f"{100 - int(status['current_size'] / status['max_size'] * 100)}% free"
        )
    
    st.divider()
    
    # History chart
    if queue.completed or queue.failed:
        recent_durations = [
            item.duration_s for item in queue.completed[-20:]
            if item.duration_s is not None
        ]
        if recent_durations:
            st.line_chart({
                "Duration (s)": recent_durations,
                "Average": [status["avg_duration_s"]] * len(recent_durations),
            })
    
    # Distribution
    st.markdown("#### Results Distribution")
    dist_col1, dist_col2, dist_col3, dist_col4 = st.columns(4)
    dist_col1.metric("Completed", status["completed"], border=True)
    dist_col2.metric("Failed", status["failed"], border=True)
    dist_col3.metric("Canceled", status["canceled"], border=True)
    dist_col4.metric("Queued", status["queued"], border=True)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def _get_generation_settings() -> dict:
    """Get current generation settings from session state"""
    return {
        "model_name": st.session_state.get("model_name", "flux.2-klein-4b"),
        "dtype_str": st.session_state.get("dtype_str", "bf16"),
        "cpu_offloading": bool(st.session_state.get("cpu_offloading", False)),
        "attn_slicing": bool(st.session_state.get("attn_slicing", True)),
        "num_steps": int(st.session_state.get("num_steps", 4)),
        "guidance": float(st.session_state.get("guidance", 1.0)),
        "width": int(st.session_state.get("width", 768)),
        "height": int(st.session_state.get("height", 768)),
        "seed": int(st.session_state.get("seed", 1)),
        "output_dir": st.session_state.get("output_dir", "outputs"),
        "project_context": state.get_active_project() or {},
        "phase2_controls": st.session_state.get("phase2_controls_payload", {}),
    }


def _save_template(queue: GenerationQueue, req: GenerationRequest) -> None:
    """Save request as template"""
    template_name = st.text_input("Template name", key="save_template_name", placeholder="My Template")
    if template_name:
        templates = st.session_state.get("queue_templates", {})
        templates[template_name] = req.settings
        st.session_state.queue_templates = templates
        st.success(f"{icons.label('Saved as template', 'check')}: {template_name}")


def _fmt_due(dt: datetime | None) -> str:
    """Format schedule time"""
    if dt is None:
        return "now"
    now = datetime.now(timezone.utc)
    diff = (dt - now).total_seconds()
    if diff < 0:
        return "overdue"
    elif diff < 60:
        return f"in {int(diff)}s"
    elif diff < 3600:
        return f"in {int(diff / 60)}m"
    else:
        return dt.astimezone(timezone.utc).strftime("%H:%M %Z")


def _zip_files(paths: list[Path]) -> bytes:
    """Create ZIP from file paths"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in paths:
            zf.write(p, arcname=p.name)
    buf.seek(0)
    return buf.getvalue()
