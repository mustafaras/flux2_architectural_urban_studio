"""Centralized sidebar copy and presentation metadata for Phase 4 UX polish."""

from __future__ import annotations

SECTION_LABELS: dict[str, str] = {
    "workflow_mode": "🎯 Design Workflow",
    "project_context": "📍 Active Project",
    "generation_controls": "✨ Generation Parameters",
    "operations": "⚙️ Generation Queue",
    "session_tools": "🛠 Session Control",
    "advanced_settings": "Advanced Settings",
}

MODEL_PROFILES: dict[str, dict[str, str]] = {
    "flux.2-klein-4b": {
        "display_name": "FLUX Base",
        "description": "Fast, Balanced",
        "vram": "~8 GB VRAM",
        "icon": "🚀",
        "speed_tier": "Fast",
        "quality_tier": "Balanced",
    },
    "flux.2-klein-9b": {
        "display_name": "FLUX Pro",
        "description": "Fast, Refined",
        "vram": "~16 GB VRAM",
        "icon": "⚡",
        "speed_tier": "Fast",
        "quality_tier": "High",
    },
    "flux.2-klein-base-4b": {
        "display_name": "FLUX Base+",
        "description": "Flexible, Detailed",
        "vram": "~8 GB VRAM",
        "icon": "🏛️",
        "speed_tier": "Moderate",
        "quality_tier": "High",
    },
    "flux.2-klein-base-9b": {
        "display_name": "FLUX Pro+",
        "description": "Maximum Detail",
        "vram": "~16 GB VRAM",
        "icon": "🏗️",
        "speed_tier": "Measured",
        "quality_tier": "Premium",
    },
    "flux.2-dev": {
        "display_name": "FLUX Pro Studio",
        "description": "Highest Fidelity",
        "vram": "~20 GB VRAM",
        "icon": "🧠",
        "speed_tier": "Deliberate",
        "quality_tier": "Maximum",
    },
}

ACTION_LABELS: dict[str, str] = {
    "start_queue": "▶ Start Auto-Run",
    "pause_queue": "⏸ Pause",
    "resume_queue": "▶ Resume",
    "reset_generation": "↻ Reset",
    "clear_history": "↻ Clear",
    "restore_settings": "↩ Restore Settings",
    "restore_last_success": "↩️ Restore Last Success",
    "clear_session": "↻ Clear Session",
    "undo_last": "⬅️ Undo",
    "start_project": "▶ Start New Project",
    "apply_recommended": "🎨 Apply Recommended",
    "apply_preset": "✓ Apply Preset",
}

STATUS_LABELS: dict[str, str] = {
    "quality_preset": "Output Quality",
    "canvas_dimensions": "Canvas Dimensions",
    "generation_seed": "Generation Seed",
}
