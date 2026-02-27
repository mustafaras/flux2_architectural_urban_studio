"""Utility helpers for the FLUX.2 Professional UI."""

from __future__ import annotations

import io
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from PIL import Image

logger = logging.getLogger("flux2_ui")


# ─── Device helpers ───────────────────────────────────────────────────────────

def get_device() -> str:
    """Return the best available PyTorch device string."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_device_info() -> dict[str, Any]:
    """Return a human-readable dict describing available hardware."""
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / (1024 ** 3)
        return {
            "type": "CUDA GPU",
            "name": props.name,
            "vram_gb": round(vram_gb, 1),
            "vram_label": f"{vram_gb:.1f} GB VRAM",
            "cuda": True,
        }
    return {
        "type": "CPU",
        "name": "CPU (no CUDA GPU detected)",
        "vram_gb": 0,
        "vram_label": "N/A",
        "cuda": False,
    }


# ─── dtype helpers ────────────────────────────────────────────────────────────

def dtype_from_str(s: str) -> torch.dtype:
    """Convert a dtype string to a torch.dtype."""
    mapping = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    return mapping.get(s, torch.bfloat16)


# ─── Image helpers ────────────────────────────────────────────────────────────

def pil_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    """Encode a PIL image to bytes for Streamlit download widgets."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def save_image(
    img: Image.Image,
    out_dir: str | Path,
    model_name: str,
    seed: int,
    num_steps: int,
    guidance: float,
) -> Path:
    """Save *img* to *out_dir* with a descriptive filename; return the path."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = model_name.replace(".", "-").replace("/", "_")
    fname = f"flux2_{safe_model}_{ts}_seed{seed}_s{num_steps}_g{guidance:.1f}.png"
    full_path = out_path / fname
    img.save(full_path)
    logger.info("Saved image to %s", full_path)
    return full_path


# ─── Seed helpers ─────────────────────────────────────────────────────────────

def resolve_seed(seed: int, use_random: bool) -> int:
    """If *use_random* is True, generate a random seed; otherwise return *seed*."""
    import random as _random
    if use_random:
        return _random.randint(0, 2_147_483_647)
    return int(seed)


def make_generator(seed: int, device: str) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(seed)


# ─── Validation ───────────────────────────────────────────────────────────────

def validate_prompt(prompt: str) -> str | None:
    """Return an error message string if *prompt* is invalid, else None."""
    if not prompt or not prompt.strip():
        return "Please enter a prompt before generating."
    if len(prompt.strip()) < 3:
        return "Prompt is too short. Please provide a more descriptive description."
    return None


def get_generation_blockers(
    prompt: str,
    require_reference: bool = False,
    reference_images: list[Any] | None = None,
    require_project: bool = False,
    active_project: dict[str, Any] | None = None,
) -> list[str]:
    """Return user-facing validation blockers before generation starts."""
    blockers: list[str] = []

    prompt_error = validate_prompt(prompt)
    if prompt_error:
        blockers.append(prompt_error)

    if require_reference and not (reference_images or []):
        blockers.append("Please upload at least one reference image before generating.")

    if require_project and not active_project:
        blockers.append("Create or activate a project in Project Setup before generating.")

    return blockers


def can_start_generation(
    prompt: str,
    require_reference: bool = False,
    reference_images: list[Any] | None = None,
    require_project: bool = False,
    active_project: dict[str, Any] | None = None,
) -> bool:
    """Return True when all pre-run validation checks pass."""
    return len(
        get_generation_blockers(
            prompt=prompt,
            require_reference=require_reference,
            reference_images=reference_images,
            require_project=require_project,
            active_project=active_project,
        )
    ) == 0


def validate_output_dir(path: str) -> str | None:
    """Return an error message if the output directory path is invalid."""
    if not path or not path.strip():
        return "Output directory path cannot be empty."
    try:
        resolved = Path(path)
        # Check we can create it (don't create yet – just validate)
        resolved.expanduser().resolve()
    except Exception as exc:  # noqa: BLE001
        return f"Invalid output directory path: {exc}"
    return None


def params_deviate_from_recommended(
    model_name: str,
    num_steps: int,
    guidance: float,
    width: int,
    height: int,
) -> bool:
    """Return True if the current parameters differ from the model's recommended defaults."""
    from ui.config import MODEL_CONFIGS  # late import

    cfg = MODEL_CONFIGS.get(model_name)
    if cfg is None:
        return False
    rec = cfg["recommended"]
    return (
        num_steps != rec["num_steps"]
        or abs(guidance - rec["guidance"]) > 0.05
        or width != rec["width"]
        or height != rec["height"]
    )


# ─── Metadata ─────────────────────────────────────────────────────────────────

def build_metadata(
    model_name: str,
    prompt: str,
    seed: int,
    num_steps: int,
    guidance: float,
    width: int,
    height: int,
    dtype_str: str,
    generation_time_s: float,
    upsampled_prompt: str | None = None,
    has_reference: bool = False,
    project_context: dict[str, Any] | None = None,
    prompt_taxonomy: dict[str, Any] | None = None,
    phase2_controls: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a serialisable metadata dict for a completed generation."""
    payload = {
        "model": model_name,
        "prompt": prompt,
        "upsampled_prompt": upsampled_prompt,
        "seed": seed,
        "num_steps": num_steps,
        "guidance": guidance,
        "width": width,
        "height": height,
        "dtype": dtype_str,
        "has_reference_image": has_reference,
        "generation_time_s": round(generation_time_s, 2),
        "timestamp": datetime.now().isoformat(),
    }
    if project_context:
        payload["project_context"] = {
            "project_id": project_context.get("project_id", ""),
            "project_type": project_context.get("project_type", ""),
            "geography": project_context.get("geography", ""),
            "climate_profile": project_context.get("climate_profile", ""),
            "design_phase": project_context.get("design_phase", ""),
            "team_label": project_context.get("team_label", ""),
        }
    if prompt_taxonomy:
        payload["prompt_template_version"] = prompt_taxonomy.get("template_version", "")
        payload["prompt_lint_results"] = prompt_taxonomy.get("lint_results", [])
        payload["prompt_slots"] = prompt_taxonomy.get("slots", {})
    if phase2_controls:
        payload["phase2_controls"] = phase2_controls
        if "continuity_score" in phase2_controls:
            payload["continuity_score"] = phase2_controls.get("continuity_score")
    return payload


def format_metadata_display(meta: dict[str, Any]) -> str:
    """Return a human-readable multiline string for metadata display."""
    lines = [
        f"**Model:** {meta.get('model', 'N/A')}",
        f"**Seed:** {meta.get('seed', 'N/A')}",
        f"**Steps:** {meta.get('num_steps', 'N/A')}  |  **Guidance:** {meta.get('guidance', 'N/A')}",
        f"**Resolution:** {meta.get('width', '?')} × {meta.get('height', '?')} px",
        f"**Dtype:** {meta.get('dtype', 'N/A')}",
        f"**Time:** {meta.get('generation_time_s', 'N/A')} s",
    ]
    if "safety_prompt_status" in meta:
        lines.append(f"**Safety (Prompt):** {meta.get('safety_prompt_status', 'N/A')}")
    if "safety_output_status" in meta:
        lines.append(f"**Safety (Output):** {meta.get('safety_output_status', 'N/A')}")
    if meta.get("has_reference_image"):
        lines.append("**Mode:** Image-to-Image")
    if meta.get("upsampled_prompt"):
        lines.append("**Upsampling:** enabled")
    if "continuity_score" in meta:
        lines.append(f"**Continuity Score:** {meta.get('continuity_score', 'N/A')}")
    phase2_controls = meta.get("phase2_controls", {}) if isinstance(meta.get("phase2_controls"), dict) else {}
    if phase2_controls:
        lines.append(f"**Massing Preset:** {phase2_controls.get('massing_preset', 'N/A')}")
        lines.append(f"**Façade Pack:** {phase2_controls.get('facade_style_pack', 'N/A')}")
    return "\n".join(lines)


def build_settings_copy_text(meta: dict[str, Any]) -> str:
    """Return a copy-friendly JSON string for generation settings."""
    settings = {
        "model": meta.get("model"),
        "seed": meta.get("seed"),
        "num_steps": meta.get("num_steps"),
        "guidance": meta.get("guidance"),
        "width": meta.get("width"),
        "height": meta.get("height"),
        "dtype": meta.get("dtype"),
    }
    return json.dumps(settings, indent=2)


def build_result_report(
    meta: dict[str, Any],
    output_path: str,
    prompt: str | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable report payload for result panel actions."""
    return {
        "metadata": meta,
        "prompt": prompt or meta.get("prompt", ""),
        "output_path": output_path,
        "exported_at": datetime.now().isoformat(),
    }


def build_history_compare_candidates(
    history: list[dict[str, Any]],
    max_items: int = 10,
) -> list[dict[str, Any]]:
    """Build compare candidates from history excluding the latest entry."""
    candidates: list[dict[str, Any]] = []
    if not history or len(history) < 2:
        return candidates

    for idx, entry in enumerate(history[1:max_items], start=1):
        if not isinstance(entry, dict):
            continue
        meta = entry.get("metadata", {}) if isinstance(entry.get("metadata"), dict) else {}
        label = (
            f"#{idx + 1} · {meta.get('model', '?')} · seed {meta.get('seed', '?')} "
            f"· {meta.get('width', '?')}x{meta.get('height', '?')}"
        )
        candidates.append({"label": label, "entry": entry})

    return candidates


# ─── Timing ───────────────────────────────────────────────────────────────────

class Timer:
    """Context manager that records elapsed wall-clock time in seconds."""

    def __init__(self):
        self.elapsed: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self._start


# ─── Logging setup ────────────────────────────────────────────────────────────

def configure_logging(level: int = logging.INFO) -> None:
    """Configure the module-level logger (call once from the main entry point)."""
    from src.flux2.logging_config import configure_flux2_logging

    debug_enabled = level <= logging.DEBUG
    configure_flux2_logging(log_dir="logs", debug=debug_enabled)
