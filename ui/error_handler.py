from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from typing import Any

import streamlit as st

from src.flux2.error_types import ErrorCategory, ErrorContext


_RECOVERY_FILE = Path("outputs") / ".flux2_recovery_state.json"
_ERROR_HISTORY_KEY = "_flux2_error_history"


def display_recoverable_error(ctx: ErrorContext, suggestion: str | None = None) -> None:
    suggestion_text = suggestion or _default_suggestion(ctx.category)
    _record_error_history(ctx)

    with st.container(border=True):
        st.error(f"⚠️ {ctx.category.value}")
        st.markdown(f"**What happened:** {ctx.message}")
        st.markdown(f"💡 Solution: {suggestion_text}")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("Retry", use_container_width=True, key=f"retry_{ctx.timestamp}"):
                st.rerun()
        with col2:
            if st.button("Try Alternative", use_container_width=True, key=f"alt_{ctx.timestamp}"):
                switch_strategy(ctx)
                st.rerun()
        with col3:
            if st.button("View Logs", use_container_width=True, key=f"logs_{ctx.timestamp}"):
                st.session_state[f"show_logs_{ctx.timestamp}"] = True
        with col4:
            report_payload = _build_issue_report_payload(ctx)
            st.download_button(
                label="Report Issue",
                data=json.dumps(report_payload, indent=2, ensure_ascii=False),
                file_name=f"flux2_error_report_{ctx.category.name.lower()}.json",
                mime="application/json",
                use_container_width=True,
                key=f"report_{ctx.timestamp}",
            )

        if st.button("Copy Technical Details", use_container_width=True, key=f"copy_logs_{ctx.timestamp}"):
            st.session_state[f"copied_logs_{ctx.timestamp}"] = ctx.technical_details or "No technical details available."
            st.success("Technical details ready to copy.")

        if st.session_state.get(f"copied_logs_{ctx.timestamp}"):
            st.code(st.session_state.get(f"copied_logs_{ctx.timestamp}", ""))

        history_payload = get_error_history(limit=50)
        st.download_button(
            label="Download Error Log",
            data=json.dumps(history_payload, indent=2, ensure_ascii=False),
            file_name="flux2_error_history.json",
            mime="application/json",
            use_container_width=True,
            key=f"download_logs_{ctx.timestamp}",
        )

    if st.session_state.get(f"show_logs_{ctx.timestamp}", False):
        with st.expander("Technical details", expanded=True):
            st.code(ctx.technical_details or "No technical details available.")
    else:
        with st.expander("Technical details"):
            st.code(ctx.technical_details or "No technical details available.")


def switch_strategy(ctx: ErrorContext) -> None:
    if ctx.category in {ErrorCategory.INSUFFICIENT_VRAM, ErrorCategory.CONFIGURATION_ERROR}:
        apply_quality_reduction()
        maybe_switch_to_fallback_model()
        return

    if ctx.category == ErrorCategory.MODEL_NOT_FOUND:
        maybe_switch_to_fallback_model()
        return

    if ctx.category in {ErrorCategory.NETWORK_TIMEOUT, ErrorCategory.API_RATE_LIMIT}:
        st.session_state.upsample_backend = "none"


def apply_quality_reduction() -> None:
    width = int(st.session_state.get("width", 768))
    height = int(st.session_state.get("height", 768))
    steps = int(st.session_state.get("num_steps", 4))

    st.session_state.width = max(512, min(width, 768))
    st.session_state.height = max(512, min(height, 768))
    st.session_state.num_steps = max(4, min(steps, 25))


def maybe_switch_to_fallback_model() -> None:
    model_name = st.session_state.get("model_name", "flux.2-klein-4b")
    if "9b" in model_name:
        st.session_state.model_name = model_name.replace("9b", "4b")
    elif model_name == "flux.2-dev":
        st.session_state.model_name = "flux.2-klein-4b"


def save_recovery_snapshot(payload: dict[str, Any]) -> None:
    _RECOVERY_FILE.parent.mkdir(parents=True, exist_ok=True)
    _RECOVERY_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_recovery_snapshot() -> dict[str, Any] | None:
    if not _RECOVERY_FILE.exists():
        return None
    try:
        data = json.loads(_RECOVERY_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:
        return None
    return None


def clear_recovery_snapshot() -> None:
    if _RECOVERY_FILE.exists():
        _RECOVERY_FILE.unlink(missing_ok=True)


def get_error_history(limit: int = 50) -> list[dict[str, Any]]:
    payload = st.session_state.get(_ERROR_HISTORY_KEY, [])
    if not isinstance(payload, list):
        return []
    return payload[-max(1, int(limit)) :]


def clear_error_history() -> None:
    st.session_state[_ERROR_HISTORY_KEY] = []


def _record_error_history(ctx: ErrorContext) -> None:
    payload = st.session_state.get(_ERROR_HISTORY_KEY, [])
    if not isinstance(payload, list):
        payload = []

    payload.append(
        {
            "timestamp": ctx.timestamp,
            "category": ctx.category.value,
            "severity": ctx.severity.value,
            "message": ctx.message,
            "location": ctx.location,
            "metadata": ctx.metadata,
        }
    )
    st.session_state[_ERROR_HISTORY_KEY] = payload[-50:]


def _default_suggestion(category: ErrorCategory) -> str:
    if category == ErrorCategory.INSUFFICIENT_VRAM:
        return "Try a lower resolution/step count, or switch to Klein 4B."
    if category == ErrorCategory.MODEL_NOT_FOUND:
        return "Check model paths in Settings and ensure weights exist locally."
    if category == ErrorCategory.NETWORK_TIMEOUT:
        return "Retry in a few seconds or switch to a local backend."
    if category == ErrorCategory.API_RATE_LIMIT:
        return "Wait and retry, or switch backend to local mode."
    if category == ErrorCategory.CONFIGURATION_ERROR:
        return "Reset parameters to recommended defaults and retry."
    return "Retry with safe defaults."


def _build_issue_report_payload(ctx: ErrorContext) -> dict[str, Any]:
    return {
        "timestamp": ctx.timestamp,
        "category": ctx.category.value,
        "severity": ctx.severity.value,
        "message": ctx.message,
        "location": ctx.location,
        "metadata": ctx.metadata,
        "technical_details": ctx.technical_details,
        "ui": {
            "title": "FLUX.2 Professional Image Generator",
            "streamlit_version": getattr(st, "__version__", "unknown"),
        },
        "environment": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "os": platform.system(),
        },
    }
