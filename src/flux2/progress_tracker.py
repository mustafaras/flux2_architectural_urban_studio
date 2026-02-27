"""
Real-time progress tracking for FLUX.2 generation.

Tracks inference progress, calculates ETA, captures intermediate frames,
and monitors system resources during generation.
"""

from __future__ import annotations

import io
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

import torch
from PIL import Image, ImageDraw

try:
    import streamlit as st
except Exception:
    st = None  # type: ignore[assignment]

logger = logging.getLogger("flux2_progress_tracker")


@dataclass(slots=True)
class ProgressSample:
    """Single snapshot of inference progress."""

    step_idx: int
    total_steps: int
    progress_percent: float
    eta_s: float
    step_per_sec: float
    current_timestep: float
    elapsed_s: float
    timestamp: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class QualitySpeedSample:
    """Quality vs. speed tradeoff data point."""

    timestamp: float
    progress_percent: float
    quality_proxy: float
    speed_steps_per_sec: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PreviewFrame:
    """Intermediate preview frame metadata."""

    step_idx: int
    total_steps: int
    progress_percent: float
    frame_bytes: bytes
    timestamp: float
    capture_latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_idx": self.step_idx,
            "total_steps": self.total_steps,
            "progress_percent": self.progress_percent,
            "timestamp": self.timestamp,
            "capture_latency_ms": self.capture_latency_ms,
        }


class ProgressTracker:
    """
    Tracks real-time inference progress, ETA, and captures intermediate frames.

    Provides:
    - Step-by-step progress updates
    - ETA calculation with historical tuning
    - Intermediate frame capture every N steps
    - Quality vs. speed metrics
    """

    def __init__(
        self,
        session_key: str = "_flux2_progress_tracker",
        frame_interval: int = 4,
        max_history: int = 512,
        max_frames: int = 12,
    ) -> None:
        """
        Initialize progress tracker.

        Args:
            session_key: Streamlit session state key
            frame_interval: Capture frame every N steps
            max_history: Max history items to keep
            max_frames: Max intermediate frames to store
        """
        self.session_key = session_key
        self.frame_interval = max(1, frame_interval)
        self.max_history = max(10, max_history)
        self.max_frames = max(1, max_frames)
        self._started_at: float | None = None
        self._frame_capture_fn: Callable | None = None

    def _get_state(self) -> dict[str, Any]:
        """Get or initialize tracker state in session."""
        if st is None:
            if not hasattr(self, "_fallback_state"):
                self._fallback_state = self._init_state()
            return self._fallback_state
        if self.session_key not in st.session_state:
            st.session_state[self.session_key] = self._init_state()
        return st.session_state[self.session_key]

    @staticmethod
    def _init_state() -> dict[str, Any]:
        """Initialize tracker state."""
        return {
            "mode": "idle",
            "status": "idle",
            "current_step": 0,
            "total_steps": 0,
            "progress_percent": 0.0,
            "eta_s": 0.0,
            "step_per_sec": 0.0,
            "current_timestep": 0.0,
            "elapsed_s": 0.0,
            "history": [],
            "preview_frames": [],
            "quality_speed_history": [],
            "started_at": None,
            "updated_at": None,
            "error_message": None,
            "frame_count": 0,
        }

    def begin(self, mode: str, total_steps: int) -> None:
        """Start tracking a new generation."""
        self._started_at = time.perf_counter()
        state = self._get_state()
        state.update(
            {
                "mode": mode,
                "status": "running",
                "current_step": 0,
                "total_steps": max(1, total_steps),
                "progress_percent": 0.0,
                "eta_s": 0.0,
                "step_per_sec": 0.0,
                "current_timestep": 1.0,
                "elapsed_s": 0.0,
                "history": [],
                "preview_frames": [],
                "quality_speed_history": [],
                "started_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "error_message": None,
                "frame_count": 0,
            }
        )
        if st is not None:
            st.session_state[self.session_key] = state

    def set_frame_capture_fn(
        self, fn: Callable[[int, int, torch.Tensor, torch.Tensor], bytes | None] | None
    ) -> None:
        """Set callback for capturing intermediate frames."""
        self._frame_capture_fn = fn

    def update(
        self,
        step_idx: int,
        total_steps: int,
        t_curr: float,
        t_prev: float,
        latent: torch.Tensor | None = None,
        latent_ids: torch.Tensor | None = None,
    ) -> None:
        """
        Update progress for current step.

        Args:
            step_idx: Current step number (0-indexed)
            total_steps: Total steps for this generation
            t_curr: Current timestep value
            t_prev: Previous timestep value
            latent: Latent tensor (for frame capture)
            latent_ids: Latent position IDs (for frame capture)
        """
        if self._started_at is None:
            self.begin(mode="t2i", total_steps=total_steps)

        state = self._get_state()
        elapsed = max(1e-6, time.perf_counter() - self._started_at)
        step_count = max(1, step_idx + 1)
        steps_per_sec = float(step_count) / elapsed
        remain_steps = max(0, int(total_steps) - int(step_idx))
        eta_s = (remain_steps / steps_per_sec) if steps_per_sec > 0 else 0.0
        progress_percent = min(100.0, 100.0 * float(step_idx) / max(1, int(total_steps)))

        # Record history sample
        history = list(state.get("history", []))
        history.append({
            "step": int(step_idx),
            "total_steps": int(total_steps),
            "progress_percent": progress_percent,
            "eta_s": eta_s,
            "current_timestep": float(t_curr),
            "next_timestep": float(t_prev),
            "elapsed_s": elapsed,
            "steps_per_sec": steps_per_sec,
            "timestamp": time.time(),
        })
        history = history[-self.max_history :]

        # Record quality/speed tradeoff
        quality_speed_history = list(state.get("quality_speed_history", []))
        quality_speed_history.append({
            "timestamp": time.time(),
            "progress_percent": progress_percent,
            "quality_proxy": progress_percent,  # Proxy: higher progress = higher quality
            "speed_steps_per_sec": steps_per_sec,
        })
        quality_speed_history = quality_speed_history[-self.max_history :]

        # Capture intermediate frames
        preview_frames = list(state.get("preview_frames", []))
        if (step_idx % self.frame_interval == 0 or step_idx == total_steps - 1) and self._frame_capture_fn and latent is not None and latent_ids is not None:
            try:
                frame_start = time.perf_counter()
                frame_bytes = self._frame_capture_fn(step_idx, total_steps, latent, latent_ids)
                capture_latency = (time.perf_counter() - frame_start) * 1000.0

                if frame_bytes is not None:
                    preview_frames.append({
                        "step_idx": int(step_idx),
                        "total_steps": int(total_steps),
                        "progress_percent": progress_percent,
                        "timestamp": time.time(),
                        "capture_latency_ms": capture_latency,
                        "frame_bytes": frame_bytes,
                    })
                    preview_frames = preview_frames[-self.max_frames :]
                    state["frame_count"] = int(state.get("frame_count", 0)) + 1
            except Exception as e:
                logger.warning(f"Failed to capture frame at step {step_idx}: {e}")

        # Update state
        state.update({
            "status": "running",
            "current_step": int(step_idx),
            "total_steps": int(total_steps),
            "progress_percent": progress_percent,
            "eta_s": eta_s,
            "step_per_sec": steps_per_sec,
            "current_timestep": float(t_curr),
            "elapsed_s": elapsed,
            "history": history,
            "preview_frames": preview_frames,
            "quality_speed_history": quality_speed_history,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })

        if st is not None:
            st.session_state[self.session_key] = state

    def finalize(self, status: str = "completed", error_message: str | None = None) -> None:
        """Finalize tracking."""
        state = self._get_state()
        total_steps = int(state.get("total_steps", 0))

        if status == "completed" and total_steps > 0:
            state["current_step"] = total_steps
            state["progress_percent"] = 100.0
            state["eta_s"] = 0.0

        state["status"] = status
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        state["error_message"] = error_message

        if st is not None:
            st.session_state[self.session_key] = state

    def snapshot(self) -> dict[str, Any]:
        """Get current progress snapshot."""
        return self._get_state()

    def clear(self) -> None:
        """Clear all progress state."""
        if st is not None and self.session_key in st.session_state:
            del st.session_state[self.session_key]
        else:
            if hasattr(self, "_fallback_state"):
                self._fallback_state = self._init_state()


def get_progress_tracker() -> ProgressTracker:
    """Get singleton progress tracker instance."""
    if st is None:
        if not hasattr(get_progress_tracker, "_instance"):
            get_progress_tracker._instance = ProgressTracker()
        return get_progress_tracker._instance

    session_key = "_flux2_progress_tracker_instance"
    if session_key not in st.session_state:
        st.session_state[session_key] = ProgressTracker()
    return st.session_state[session_key]
