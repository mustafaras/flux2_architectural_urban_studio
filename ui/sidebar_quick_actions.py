"""Sidebar quick actions and live operational summaries for Phase 5."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import shutil
from typing import Any

import streamlit as st

from ui import state as _state

_TRANSIENT_KEYS: tuple[str, ...] = (
    "current_image",
    "generation_metadata",
    "generation_time_s",
    "reference_images",
    "match_image_size",
    "upsample_original_prompt",
    "upsample_result_prompt",
)

_PARAM_KEYS: tuple[str, ...] = (
    "model_name",
    "quality_preset",
    "num_steps",
    "guidance",
    "width",
    "height",
    "seed",
    "use_random_seed",
    "dtype_str",
    "prompt",
)


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _snapshot_params() -> dict[str, Any]:
    return {key: st.session_state.get(key) for key in _PARAM_KEYS}


def _push_action(action: str, params: dict[str, Any] | None = None) -> None:
    history = st.session_state.get("action_history", [])
    if not isinstance(history, list):
        history = []
    history.append(
        {
            "timestamp": _now_iso(),
            "action": action,
            "params": dict(params or _snapshot_params()),
        }
    )
    st.session_state["action_history"] = history[-10:]


def _set_params(snapshot: dict[str, Any]) -> None:
    for key in _PARAM_KEYS:
        if key in snapshot:
            st.session_state[key] = snapshot[key]


def capture_action_snapshot(action: str) -> None:
    """Persist a point-in-time parameter snapshot for undo support."""
    _push_action(action, _snapshot_params())


def capture_parameter_changes() -> None:
    """Track sidebar parameter changes without polling.

    Captures only when the parameter snapshot differs from the previous one,
    keeping a compact undo stack in session state.
    """
    current = _snapshot_params()
    previous = st.session_state.get("last_parameter_snapshot", {})
    if not isinstance(previous, dict):
        previous = {}
    if current != previous:
        _push_action("parameter_update", current)
        st.session_state["last_parameter_snapshot"] = current


def apply_recommended(project_type: str) -> dict[str, Any]:
    """Apply production defaults based on active project type."""
    normalized = str(project_type or "").strip().lower()

    model_name = "flux.2-klein-4b"
    if any(token in normalized for token in ("commercial", "mixed", "masterplan")):
        model_name = "flux.2-klein-9b"
    elif any(token in normalized for token in ("institutional", "public")):
        model_name = "flux.2-klein-base-4b"

    capture_action_snapshot("before_apply_recommended")
    _state.apply_model_defaults(model_name)

    presets = _state.get_quality_preset_options(model_name)
    preferred = "Standard (4 steps)"
    if preferred not in presets:
        preferred = "Standard (25 steps)" if "Standard (25 steps)" in presets else "— Custom —"

    if preferred != "— Custom —":
        _state.apply_quality_preset(model_name, preferred)

    result = {
        "model_name": st.session_state.get("model_name"),
        "quality_preset": st.session_state.get("quality_preset"),
        "project_type": project_type,
    }
    _push_action("apply_recommended", result)
    return result


def restore_last_successful() -> bool:
    """Restore the most recent successful generation settings from history."""
    history = st.session_state.get("generation_history", [])
    if not isinstance(history, list) or not history:
        return False

    for entry in history:
        if _state.restore_settings_from_history_entry(entry):
            _push_action("restore_last_successful")
            return True
    return False


def clear_session_transients() -> None:
    """Clear transient generation/session fields while preserving project context."""
    capture_action_snapshot("before_clear_session")

    for key in _TRANSIENT_KEYS:
        if key in ("reference_images",):
            st.session_state[key] = []
        elif key in ("match_image_size",):
            st.session_state[key] = None
        else:
            st.session_state[key] = None if "prompt" not in key else ""

    queue_obj = st.session_state.get("generation_queue")
    if queue_obj is not None and hasattr(queue_obj, "clear"):
        queue_obj.clear()

    temp_export_dir = Path.cwd() / "outputs" / "_export_tmp"
    if temp_export_dir.exists():
        for item in temp_export_dir.iterdir():
            if item.is_file():
                item.unlink(missing_ok=True)
            elif item.is_dir():
                shutil.rmtree(item, ignore_errors=True)

    st.session_state["queue_auto_run"] = False
    st.session_state["queue_paused"] = False
    st.session_state["clear_session_confirm_reset_pending"] = True
    _push_action("clear_session_transients")


def undo_last_action() -> bool:
    """Revert generation parameters to the previous snapshot in action history."""
    history = st.session_state.get("action_history", [])
    if not isinstance(history, list) or len(history) < 2:
        return False

    history.pop()
    previous = history[-1]
    params = previous.get("params", {}) if isinstance(previous, dict) else {}
    if not isinstance(params, dict):
        return False

    _set_params(params)
    st.session_state["action_history"] = history
    st.session_state["last_parameter_snapshot"] = _snapshot_params()
    return True


def update_queue_display() -> dict[str, Any]:
    """Return queue status summary synchronized with queue_auto_run/queue_paused."""
    queue_obj = st.session_state.get("generation_queue")
    if queue_obj is None:
        return {
            "queued": 0,
            "running": 0,
            "completed": 0,
            "failed": 0,
            "eta_s": 0.0,
            "eta_text": "0s",
            "avg_duration_s": float(st.session_state.get("generation_time_s") or 0.0),
            "auto_run": bool(st.session_state.get("queue_auto_run", False)),
            "paused": bool(st.session_state.get("queue_paused", False)),
        }

    raw = queue_obj.get_status()
    return {
        "queued": int(raw.get("queued", 0)),
        "running": int(raw.get("running", 0)),
        "completed": int(raw.get("completed", 0)),
        "failed": int(raw.get("failed", 0)),
        "eta_s": float(raw.get("eta_s", 0.0)),
        "eta_text": str(raw.get("eta_text", "0s")),
        "avg_duration_s": float(raw.get("avg_duration_s", 0.0)),
        "auto_run": bool(st.session_state.get("queue_auto_run", False)),
        "paused": bool(st.session_state.get("queue_paused", False)),
    }


def check_session_health() -> dict[str, Any]:
    """Return session health status and warnings for sidebar display."""
    warnings: list[str] = []

    active_project = _state.get_active_project()
    if not active_project:
        warnings.append("Complete project setup first.")

    ref_images = st.session_state.get("reference_images", [])
    if st.session_state.get("match_image_size") and not ref_images:
        warnings.append("Upload reference image before reference-based generation.")

    generation_meta = st.session_state.get("generation_metadata")
    last_ts = ""
    if isinstance(generation_meta, dict):
        last_ts = str(generation_meta.get("timestamp", "")).strip()

    stale_hours = 0.0
    if last_ts:
        try:
            stale_hours = max(0.0, (datetime.now() - datetime.fromisoformat(last_ts)).total_seconds() / 3600.0)
        except ValueError:
            stale_hours = 0.0

    is_stale = stale_hours >= 2.0
    if is_stale:
        warnings.append(f"Project appears stale ({stale_hours:.1f}h since last generation).")

    output_dir = Path(str(st.session_state.get("output_dir", "outputs")))
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir

    used_bytes = 0
    if output_dir.exists():
        used_bytes = sum(p.stat().st_size for p in output_dir.rglob("*") if p.is_file())

    usage = shutil.disk_usage(output_dir if output_dir.exists() else Path.cwd())
    usage_pct = (usage.used / usage.total) * 100 if usage.total else 0.0
    free_gb = usage.free / (1024 ** 3)

    if free_gb < 0.5:
        warnings.append(f"Insufficient disk space: 0.5 GB needed, {free_gb:.2f} GB available.")
    elif usage_pct >= 80.0:
        warnings.append(f"Storage warning: {usage_pct:.1f}% disk usage.")

    return {
        "warnings": warnings,
        "is_stale": is_stale,
        "stale_hours": stale_hours,
        "output_usage_gb": used_bytes / (1024 ** 3),
        "disk_usage_pct": usage_pct,
        "disk_free_gb": free_gb,
    }
