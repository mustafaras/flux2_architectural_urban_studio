"""Sidebar rendering for FLUX.2 professional UI.

Phase 3 introduces workflow-aware sectioning while preserving existing widget
keys and control behavior from earlier phases.
"""

from __future__ import annotations

import streamlit as st

from ui import config as cfg
from ui import icons
from ui import sidebar_copy
from ui import sidebar_quick_actions as quick_actions
from ui import state as _state


def _ensure_sidebar_sections_state() -> dict[str, bool]:
    """Initialize and return sidebar visibility state.

    State keys touched:
    - ``sidebar_sections``
    """
    defaults = {
        "workflow_mode": True,
        "project_context": True,
        "generation_controls": True,
        "operations": False,
        "session_tools": False,
        "has_project": False,
        "has_history": False,
        "has_errors": False,
    }
    existing = st.session_state.get("sidebar_sections", {})
    if not isinstance(existing, dict):
        existing = {}
    merged = {**defaults, **existing}
    st.session_state["sidebar_sections"] = merged
    return merged


def _ensure_sidebar_expander_state(queue_status: dict[str, int]) -> dict[str, bool]:
    """Initialize and return expander default state for polished sidebar UX."""
    if "sidebar_expander_state" not in st.session_state:
        st.session_state["sidebar_expander_state"] = {
            "project_context": True,
            "generation_controls": True,
            "operations": False,
            "session_tools": False,
            "advanced_settings": False,
        }

    state = st.session_state.get("sidebar_expander_state", {})
    if not isinstance(state, dict):
        state = {}
    merged = {
        "project_context": bool(state.get("project_context", True)),
        "generation_controls": bool(state.get("generation_controls", True)),
        "operations": bool(state.get("operations", False)),
        "session_tools": bool(state.get("session_tools", False)),
        "advanced_settings": bool(state.get("advanced_settings", False)),
    }
    merged["operations"] = should_expand_operations(queue_status)
    st.session_state["sidebar_expander_state"] = merged
    return merged


def _success_feedback(action: str) -> None:
    """Show concise success feedback with consistent iconography."""
    st.toast(f"✓ {action} complete", icon="✅")


def _error_feedback(action: str, reason: str) -> None:
    """Show concise persistent error feedback with consistent iconography."""
    st.error(f"✗ {action} failed: {reason}")


def _resolve_model_profile(model_key: str) -> dict[str, str]:
    """Return polished model profile copy with fallback derived from config."""
    if model_key in sidebar_copy.MODEL_PROFILES:
        return sidebar_copy.MODEL_PROFILES[model_key]

    model_cfg = cfg.MODEL_CONFIGS.get(model_key, {})
    return {
        "display_name": str(model_cfg.get("display_name", model_key)),
        "description": str(model_cfg.get("description", "Balanced")),
        "vram": str(model_cfg.get("vram_estimate", "~8 GB")),
        "icon": "🚀",
        "speed_tier": str(model_cfg.get("speed_label", "Balanced")),
        "quality_tier": str(model_cfg.get("quality_label", "Balanced")),
    }


def _get_queue_status() -> dict[str, object]:
    """Return normalized queue status counters and ETA summary."""
    try:
        status = quick_actions.update_queue_display()
        st.session_state["sidebar_last_queue_status"] = dict(status)
        st.session_state["sidebar_queue_sync_error"] = ""
        status["sync_error"] = False
        status["sync_error_message"] = ""
        return status
    except Exception as exc:
        cached = st.session_state.get("sidebar_last_queue_status", {})
        if not isinstance(cached, dict):
            cached = {}
        fallback = {
            "queued": int(cached.get("queued", 0)),
            "running": int(cached.get("running", 0)),
            "completed": int(cached.get("completed", 0)),
            "failed": int(cached.get("failed", 0)),
            "eta_s": float(cached.get("eta_s", 0.0)),
            "eta_text": str(cached.get("eta_text", "0s")),
            "avg_duration_s": float(cached.get("avg_duration_s", 0.0)),
            "auto_run": bool(st.session_state.get("queue_auto_run", False)),
            "paused": bool(st.session_state.get("queue_paused", False)),
            "sync_error": True,
            "sync_error_message": str(exc),
        }
        st.session_state["sidebar_queue_sync_error"] = str(exc)
        return fallback


def should_show_project_context() -> bool:
    """Project context section is always rendered; content is state-dependent."""
    return True


def should_enable_generation_controls() -> bool:
    """Enable generation controls when an active project exists."""
    return _state.has_active_project()


def should_expand_operations(queue_status: dict[str, object]) -> bool:
    """Expand operations panel when queue activity is present."""
    return bool(
        queue_status.get("queued", 0) > 0
        or queue_status.get("running", 0) > 0
        or st.session_state.get("queue_auto_run", False)
        or st.session_state.get("queue_paused", False)
    )


def should_show_error_badge(queue_status: dict[str, object]) -> bool:
    """Return True when sidebar should show a warning/error badge."""
    return queue_status.get("failed", 0) > 0 or bool(queue_status.get("sync_error", False))


def _render_workflow_toggles() -> None:
    """Render workflow/navigation compatibility toggles.

    State dependencies:
    - ``use_architecture_workflow``
    - ``show_legacy_navigation``
    """
    st.session_state.use_architecture_workflow = st.toggle(
        "Architecture Workflow",
        value=bool(st.session_state.get("use_architecture_workflow", True)),
        help="Use architecture/urban workflow navigation surfaces.",
        key="sidebar_architecture_workflow_toggle",
    )
    st.session_state.show_legacy_navigation = st.toggle(
        "Show Legacy Navigation",
        value=bool(st.session_state.get("show_legacy_navigation", False)),
        help="Keep previous tabs available for compatibility.",
        key="sidebar_legacy_navigation_toggle",
    )


def _render_workflow_section() -> None:
    """Render workflow mode section with status messaging."""
    with st.expander(sidebar_copy.SECTION_LABELS["workflow_mode"], expanded=True):
        _render_workflow_toggles()
        if bool(st.session_state.get("use_architecture_workflow", True)):
            st.info("Architecture Workflow is active. Guided 8-step navigation is enabled.")
        else:
            st.info("Legacy Compatibility navigation is active.")


def _render_project_context_section(expanded: bool) -> bool:
    """Render active project card or setup CTA.

    Returns:
        True when an active project exists.
    """
    active_project = _state.get_active_project()
    has_project = active_project is not None

    with st.expander(sidebar_copy.SECTION_LABELS["project_context"], expanded=expanded):
        if not has_project:
            st.warning("No active project context. Complete project setup first.")
            if st.button(sidebar_copy.ACTION_LABELS["start_project"], use_container_width=True, key="sidebar_start_new_project_cta"):
                st.session_state.use_architecture_workflow = True
                st.info("Open the Project Setup tab in the main content area to create a project.")
        else:
            with st.container(border=True):
                st.markdown(f"**{active_project.get('project_id', 'Unknown Project')}**")
                col1, col2 = st.columns(2)
                with col1:
                    st.caption(f"Type: {active_project.get('project_type', '—')}")
                    st.caption(f"Phase: {active_project.get('design_phase', '—')}")
                with col2:
                    st.caption(f"Location: {active_project.get('geography', '—')}")
                    st.caption(f"Climate: {active_project.get('climate_profile', '—')}")

    return has_project


def _render_advanced_toggle() -> None:
    """Render advanced mode toggle and warning.

    State dependencies:
    - ``advanced_mode``
    """
    advanced = st.toggle(
        cfg.SIDEBAR_ADVANCED_LABEL,
        value=bool(st.session_state.get("advanced_mode", False)),
        help=cfg.SIDEBAR_ADVANCED_HELP,
        key="sidebar_advanced_toggle",
    )
    st.session_state.advanced_mode = advanced

    if advanced:
        st.warning("Advanced Mode is ON. Recommended defaults are overridable.")


def _render_quick_controls(controls_enabled: bool, expanded: bool, advanced_expanded: bool) -> str:
    """Render model/preset/size/seed controls.

    State dependencies:
    - ``model_name``
    - ``quality_preset``
    - ``width`` / ``height``
    - ``use_random_seed`` / ``seed``

    Returns:
        The resolved current model key after validation.
    """
    model_options = [m for m in cfg.MODEL_DISPLAY_ORDER if m in cfg.MODEL_CONFIGS]
    current_model = st.session_state.get("model_name", model_options[0] if model_options else "flux.2-klein-4b")
    if current_model not in model_options and model_options:
        current_model = model_options[0]

    with st.expander(sidebar_copy.SECTION_LABELS["generation_controls"], expanded=expanded):
        if not controls_enabled:
            st.info("Generation controls are disabled until an active project is selected.")

        if model_options:
            st.caption("Model")
            selected_model = st.selectbox(
                "Model Family",
                options=model_options,
                index=model_options.index(current_model),
                format_func=lambda k: cfg.MODEL_CONFIGS[k]["display_name"],
                key="sidebar_quick_model",
                disabled=not controls_enabled,
                help="Complete project setup first to enable model changes." if not controls_enabled else None,
            )
            if selected_model != current_model:
                _state.apply_model_defaults(selected_model)
            st.session_state.model_name = selected_model
            mcfg = cfg.MODEL_CONFIGS[selected_model]
            profile = _resolve_model_profile(selected_model)
            st.write(
                f"{profile['icon']} **{profile['display_name']}** | "
                f"{profile['description']} | {profile['vram']}"
            )
            st.caption(
                f"Speed: {profile['speed_tier']} · "
                f"Quality: {profile['quality_tier']} · "
                f"Est. VRAM: {profile['vram']}"
            )

            if st.button(sidebar_copy.ACTION_LABELS["apply_recommended"], use_container_width=True, key="sidebar_apply_recommended"):
                if controls_enabled:
                    _state.apply_model_defaults(selected_model)
                    _success_feedback("Apply recommended")
                else:
                    _error_feedback("Apply recommended", "Complete project setup first")

            preset_names = _state.get_quality_preset_options(selected_model)
            current_preset = st.session_state.get("quality_preset", "— Custom —")
            if current_preset not in preset_names:
                current_preset = "— Custom —"
            chosen_preset = st.selectbox(
                sidebar_copy.STATUS_LABELS["quality_preset"],
                options=preset_names,
                index=preset_names.index(current_preset),
                key="sidebar_quick_quality_preset",
                disabled=not controls_enabled,
            )
            st.session_state.quality_preset = chosen_preset
            if st.button(sidebar_copy.ACTION_LABELS["apply_preset"], use_container_width=True, key="sidebar_apply_quality_preset"):
                if not controls_enabled:
                    _error_feedback("Apply preset", "Complete project setup first")
                elif chosen_preset == "— Custom —":
                    st.info("Select a preset to apply.")
                elif _state.apply_quality_preset(selected_model, chosen_preset):
                    _success_feedback("Preset apply")
                else:
                    _error_feedback("Preset apply", "Preset unavailable for selected model")

            ranges = mcfg["parameter_ranges"]
            st.caption(sidebar_copy.STATUS_LABELS["canvas_dimensions"])
            col_w, col_h = st.columns(2)
            with col_w:
                st.session_state.width = st.selectbox(
                    "Width (px)",
                    options=ranges["width_options"],
                    index=ranges["width_options"].index(st.session_state.get("width", mcfg["recommended"]["width"])) if st.session_state.get("width", mcfg["recommended"]["width"]) in ranges["width_options"] else 0,
                    key="sidebar_quick_width",
                    disabled=not controls_enabled,
                )
            with col_h:
                st.session_state.height = st.selectbox(
                    "Height (px)",
                    options=ranges["height_options"],
                    index=ranges["height_options"].index(st.session_state.get("height", mcfg["recommended"]["height"])) if st.session_state.get("height", mcfg["recommended"]["height"]) in ranges["height_options"] else 0,
                    key="sidebar_quick_height",
                    disabled=not controls_enabled,
                )

        st.session_state.use_random_seed = st.checkbox(
            "Use Random Generation Seed",
            value=bool(st.session_state.get("use_random_seed", False)),
            key="sidebar_quick_random_seed",
            disabled=not controls_enabled,
        )
        if not st.session_state.use_random_seed:
            st.session_state.seed = int(
                st.number_input(
                    sidebar_copy.STATUS_LABELS["generation_seed"],
                    min_value=0,
                    max_value=2_147_483_647,
                    value=int(st.session_state.get("seed", 1)),
                    step=1,
                    key="sidebar_quick_seed",
                    disabled=not controls_enabled,
                )
            )

        with st.expander(sidebar_copy.SECTION_LABELS["advanced_settings"], expanded=advanced_expanded):
            _render_advanced_toggle()

    return current_model


def _render_active_setup(current_model: str) -> None:
    """Render current model/parameter summary card.

    State dependencies:
    - ``num_steps`` / ``guidance``
    - ``width`` / ``height``
    - ``model_name``
    """
    steps = st.session_state.get("num_steps", 4)
    guidance = st.session_state.get("guidance", 1.0)
    width = st.session_state.get("width", 768)
    height = st.session_state.get("height", 768)

    with st.container(border=True):
        st.caption("Active Setup")
        st.markdown(f"**{st.session_state.get('model_name', current_model)}**")
        st.caption(f"Steps: {steps}  |  Guidance: {guidance}")
        st.caption(f"Resolution: {width} × {height} px")


def _render_generation_controls_section(controls_enabled: bool) -> None:
    """Render generation controls and active setup summary section."""
    expander_state = st.session_state.get("sidebar_expander_state", {})
    current_model = _render_quick_controls(
        controls_enabled=controls_enabled,
        expanded=bool(expander_state.get("generation_controls", True)),
        advanced_expanded=bool(expander_state.get("advanced_settings", False)),
    )
    _render_active_setup(current_model)


def _render_live_status_strip(queue_status: dict[str, object]) -> None:
    """Render compact queue + run-state summary for at-a-glance monitoring."""
    pending = int(queue_status.get("queued", 0))
    running = int(queue_status.get("running", 0))
    completed = int(queue_status.get("completed", 0))
    failed = int(queue_status.get("failed", 0))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Pending", pending)
    col2.metric("Running", running)
    col3.metric("Done", completed)
    col4.metric("Failed", failed)

    auto_run = bool(st.session_state.get("queue_auto_run", False))
    paused = bool(st.session_state.get("queue_paused", False))
    state_chip = "ON" if auto_run and not paused else ("PAUSED" if paused else "OFF")
    state_icon = "✓" if state_chip == "ON" else ("⏸" if state_chip == "PAUSED" else "⚪")
    st.caption(f"Auto-Run: {state_icon} {state_chip} · Next ETA: {queue_status.get('eta_text', '0s')}")

    generation_meta = st.session_state.get("generation_metadata")
    if isinstance(generation_meta, dict) and generation_meta:
        model_key = generation_meta.get("model", st.session_state.get("model_name", "unknown"))
        profile = _resolve_model_profile(str(model_key))
        seed_val = generation_meta.get("seed", st.session_state.get("seed", "?"))
        timestamp = generation_meta.get("timestamp", "—")
        duration = generation_meta.get("generation_time_s", st.session_state.get("generation_time_s", "?"))
        st.caption(
            "Last generation: "
            f"{profile.get('display_name', model_key)} · "
            f"seed {seed_val} · {timestamp} · {duration}s"
        )


def _render_quick_actions_panel(has_project: bool) -> None:
    """Render safe operational quick actions with confirmation and undo."""
    st.write("")
    st.markdown("**Quick Actions**")

    if bool(st.session_state.pop("clear_session_confirm_reset_pending", False)):
        st.session_state["clear_session_confirm"] = False

    project = _state.get_active_project()
    project_type = project.get("project_type", "") if project else ""

    if st.button(sidebar_copy.ACTION_LABELS["apply_recommended"], use_container_width=True, key="sidebar_quick_apply_recommended_v2"):
        if not has_project:
            _error_feedback("Apply recommended", "Complete project setup first")
        else:
            result = quick_actions.apply_recommended(project_type)
            _success_feedback("Apply recommended")
            st.caption(
                f"Applied defaults for {result.get('project_type', 'current project')}"
                f": {result.get('model_name', 'model')} / {result.get('quality_preset', 'preset')}"
            )
            st.rerun()

    quick_col1, quick_col2 = st.columns(2)
    with quick_col1:
        if st.button(sidebar_copy.ACTION_LABELS["restore_last_success"], use_container_width=True, key="sidebar_restore_last_success"):
            quick_actions.capture_action_snapshot("before_restore_last_success")
            if quick_actions.restore_last_successful():
                _success_feedback("Restore last success")
                st.rerun()
            else:
                _error_feedback("Restore last success", "No successful generation history found")

    with quick_col2:
        if st.button(sidebar_copy.ACTION_LABELS["undo_last"], use_container_width=True, key="sidebar_undo_last_action"):
            if quick_actions.undo_last_action():
                _success_feedback("Undo")
                st.rerun()
            else:
                _error_feedback("Undo", "No previous state available")

    st.checkbox(
        "I understand this clears transient results only (projects and history are preserved).",
        value=bool(st.session_state.get("clear_session_confirm", False)),
        key="clear_session_confirm",
    )
    if st.button(
        sidebar_copy.ACTION_LABELS["clear_session"],
        use_container_width=True,
        key="sidebar_clear_session_transients",
        disabled=not bool(st.session_state.get("clear_session_confirm", False)),
    ):
        quick_actions.clear_session_transients()
        _success_feedback("Clear session")
        st.rerun()


def _render_session_health(queue_status: dict[str, object]) -> None:
    """Render staleness/storage/error health indicators."""
    health = quick_actions.check_session_health()
    warnings = health.get("warnings", []) if isinstance(health, dict) else []

    if should_show_error_badge(queue_status):
        st.warning("⚠ Generation failures detected. Open Queue tab for details and retry options.")

    for item in warnings[:4]:
        st.warning(str(item))

    st.caption(
        f"Storage used by outputs: {health.get('output_usage_gb', 0.0):.2f} GB · "
        f"Disk usage: {health.get('disk_usage_pct', 0.0):.1f}% · "
        f"Free: {health.get('disk_free_gb', 0.0):.2f} GB"
    )


def _render_operations_panel(queue_status: dict[str, object], has_project: bool) -> None:
    """Render queue metrics and queue control actions.

    State dependencies:
    - ``generation_queue``
    - ``queue_auto_run`` / ``queue_paused``
    """
    queue_obj = st.session_state.get("generation_queue")
    expander_state = st.session_state.get("sidebar_expander_state", {})
    with st.expander(
        sidebar_copy.SECTION_LABELS["operations"],
        expanded=bool(expander_state.get("operations", should_expand_operations(queue_status))),
    ):
        if bool(queue_status.get("sync_error", False)):
            st.error(
                "Queue sync failed. Showing cached queue state. "
                f"Details: {queue_status.get('sync_error_message', 'Unknown error')}"
            )
            if st.button("Retry Queue Sync", use_container_width=True, key="sidebar_retry_queue_sync"):
                st.session_state["sidebar_queue_sync_error"] = ""
                st.rerun()

        if queue_obj is None:
            st.info("No active queue.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Queued", queue_status.get("queued", 0))
                st.metric("Running", queue_status.get("running", 0))
            with col2:
                st.metric("Completed", queue_status.get("completed", 0))
                st.metric("Failed", queue_status.get("failed", 0))

            _render_live_status_strip(queue_status)

            if should_show_error_badge(queue_status):
                st.warning("⚠ Queue has failed item(s). Review queue details and retry as needed.")

            if st.button(sidebar_copy.ACTION_LABELS["start_queue"], use_container_width=True, key="sidebar_queue_start"):
                st.session_state.queue_auto_run = True
                st.session_state.queue_paused = False
                _success_feedback("Auto-run start")
                st.rerun()
            if st.button(sidebar_copy.ACTION_LABELS["pause_queue"], use_container_width=True, key="sidebar_queue_pause"):
                st.session_state.queue_paused = True
                _success_feedback("Queue pause")
                st.rerun()
            if st.button(sidebar_copy.ACTION_LABELS["resume_queue"], use_container_width=True, key="sidebar_queue_resume"):
                st.session_state.queue_paused = False
                st.session_state.queue_auto_run = True
                _success_feedback("Queue resume")
                st.rerun()

            auto_run_toggle = st.toggle(
                "Auto-Run Queue",
                value=bool(st.session_state.get("queue_auto_run", False)),
                key="sidebar_queue_autorun_toggle",
            )
            st.session_state.queue_auto_run = auto_run_toggle
            if not auto_run_toggle:
                st.session_state.queue_paused = False

            _render_quick_actions_panel(has_project=has_project)
            st.divider()
            _render_session_health(queue_status)


def _render_session_tools(queue_status: dict[str, object]) -> None:
    """Render history browser and session tools.

    State dependencies:
    - ``generation_history``
    """
    history = st.session_state.get("generation_history", [])
    expander_state = st.session_state.get("sidebar_expander_state", {})
    with st.expander(
        sidebar_copy.SECTION_LABELS["session_tools"],
        expanded=bool(expander_state.get("session_tools", False)) or should_show_error_badge(queue_status),
    ):
        with st.expander(f"History ({len(history)} images)", expanded=False):
            if not history:
                st.caption("No generated images yet.")
            else:
                if st.button(sidebar_copy.ACTION_LABELS["clear_history"], use_container_width=True, key="sidebar_clear_history"):
                    st.session_state.generation_history = []
                    _success_feedback("History clear")
                    st.rerun()

                for i, entry in enumerate(history[:5]):
                    meta = entry.get("metadata", {})
                    label = f"#{i + 1} — {meta.get('model', '?')} — seed {meta.get('seed', '?')}"
                    st.image(
                        entry["image_bytes"],
                        caption=label,
                        use_container_width=True,
                    )
                    if st.button(sidebar_copy.ACTION_LABELS["restore_settings"], use_container_width=True, key=f"sidebar_restore_history_{i}"):
                        if _state.restore_settings_from_history_entry(entry):
                            _success_feedback("Restore settings")
                            st.rerun()
                        else:
                            _error_feedback("Restore settings", "Could not restore settings from this history entry")

        if st.button(sidebar_copy.ACTION_LABELS["reset_generation"], use_container_width=True, key="sidebar_reset_generation_state"):
            _state.reset_generation()
            _success_feedback("Generation reset")


def _update_sidebar_sections_state(
    *,
    has_project: bool,
    has_history: bool,
    has_errors: bool,
    operations_expanded: bool,
) -> None:
    """Persist current sidebar section visibility signals."""
    sections = _ensure_sidebar_sections_state()
    sections["workflow_mode"] = True
    sections["project_context"] = True
    sections["generation_controls"] = True
    sections["operations"] = operations_expanded
    sections["session_tools"] = has_errors
    sections["has_project"] = has_project
    sections["has_history"] = has_history
    sections["has_errors"] = has_errors
    st.session_state["sidebar_sections"] = sections


def render_sidebar() -> None:
    """Render the complete workflow-aware sidebar."""
    with st.sidebar:
        _ensure_sidebar_sections_state()
        queue_status = _get_queue_status()
        expander_state = _ensure_sidebar_expander_state(queue_status)
        has_errors = should_show_error_badge(queue_status) or bool(st.session_state.get("sidebar_queue_sync_error", ""))
        has_history = len(st.session_state.get("generation_history", [])) > 0

        icons.heading(cfg.SIDEBAR_TITLE, icon="settings", level=3)
        st.divider()

        _render_workflow_section()
        st.divider()

        has_project = False
        if should_show_project_context():
            has_project = _render_project_context_section(expanded=bool(expander_state.get("project_context", True)))
        st.divider()

        _render_generation_controls_section(controls_enabled=should_enable_generation_controls())
        quick_actions.capture_parameter_changes()
        st.divider()

        operations_expanded = should_expand_operations(queue_status)
        _render_operations_panel(queue_status, has_project=has_project)
        st.divider()

        _render_session_tools(queue_status)

        _update_sidebar_sections_state(
            has_project=has_project,
            has_history=has_history,
            has_errors=has_errors,
            operations_expanded=operations_expanded,
        )
