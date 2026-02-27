"""Session state management for the FLUX.2 Professional UI."""

from __future__ import annotations

from collections.abc import MutableMapping
from datetime import datetime

from src.flux2.queue_manager import GenerationQueue
import streamlit as st
import torch

# ─── Default values for every session state key ───────────────────────────────

_DEFAULTS: dict[str, object] = {
    # Model selection
    "model_name": "flux.2-klein-4b",
    "dtype_str": "bf16",
    "cpu_offloading": False,
    "attn_slicing": True,

    # Generation parameters
    "prompt": "",
    "num_steps": 4,
    "guidance": 1.0,
    "width": 768,
    "height": 768,
    "seed": 1,
    "use_random_seed": False,

    # Advanced mode toggle
    "advanced_mode": False,

    # Quality preset selection
    "quality_preset": "— Custom —",

    # Output
    "output_dir": "outputs",
    "current_image": None,          # PIL Image | None
    "generation_metadata": None,    # dict | None
    "generation_time_s": None,      # float | None

    # Generation history (list of dicts with image bytes + metadata)
    "generation_history": [],

    # Image editor state
    "reference_images": [],         # list[PIL.Image]
    "match_image_size": None,       # int | None

    # Upsampler state
    "upsample_original_prompt": "",
    "upsample_result_prompt": "",
    "upsample_backend": "none",
    "ollama_profile": "quality",
    "ollama_model": "qwen3:30b",
    "ollama_temperature": 0.15,
    "openrouter_model": "mistralai/pixtral-large-2411",

    # Settings / API keys (not persisted to disk in this version)
    "openrouter_api_key": "",
    "nsfw_threshold": 0.85,
    "safety_check_prompt": False,
    "safety_check_output": False,

    # Custom model / AE weight paths (override env vars if set)
    "custom_klein_4b_path": "",
    "custom_klein_9b_path": "",
    "custom_flux2_dev_path": "",
    "custom_ae_path": "",

    # UI state
    "active_tab_index": 0,
    "show_help": False,
    "device_profile": "desktop",

    # Queue / batch processing
    "generation_queue": None,
    "queue_auto_run": False,
    "queue_paused": False,
    "queue_last_tick_ts": 0.0,
    "queue_templates": [],
    "queue_export_zip_name": "batch_outputs.zip",

    # Sidebar operational quick-actions history (Phase 5)
    "action_history": [],
    "last_parameter_snapshot": {},
    "clear_session_confirm": False,

    # Progress monitor
    "progress_auto_refresh": True,
    "progress_refresh_seconds": 2,

    # Architectural workflow mode + compatibility
    "use_architecture_workflow": True,
    "show_legacy_navigation": False,

    # Project-scoped context (Prompt 1.2)
    "projects": [],
    "active_project_id": "",

    # Site/context references by project_id (Prompt 1.3)
    "site_references_by_project": {},

    # Prompt taxonomy (Prompt 1.4)
    "prompt_taxonomy_payload": None,

    # Phase 2 domain controls
    "phase2_controls_payload": {},
    "seed_groups": {},

    # Comparison board session state (Prompt 1.5)
    "comparison_board_shortlist_ids": [],
    "comparison_board_rubric_scores": {},
    "project_rubric_scores": {},

    # Urban scenario studio state (Phase 3)
    "urban_scenarios": [],
    "urban_delta_snapshots": [],

    # Governance / collaboration state (Phase 4)
    "current_user_role": "editor",
    "review_workflow_state": "draft",
    "review_workflow_history": [],
    "retention_profile": "exploratory",
    "annotation_threads": [],

    # Intelligence / rollout state (Phase 5)
    "current_policy_profile": "commercial",
    "feature_flags": {
        "enable_explainability": True,
        "enable_connectors": False,
        "enable_policy_dashboards": True,
        "enable_rollout_playbooks": True,
    },
    "rollout_stage": "pilot",

    # Baseline KPI telemetry counters (Prompt 1.6)
    "kpi_session_started_at": "",
    "kpi_iteration_count": 0,
    "kpi_generation_latencies": [],
    "kpi_queue_wait_times_ms": [],
    "kpi_first_shortlist_ts": "",
    "kpi_compose_prompt_count": 0,
    "kpi_shortlist_count": 0,
    "kpi_board_export_count": 0,

    "_hardware_defaults_applied": False,
}


def init() -> None:
    """Initialise all session state keys with defaults if not already set.

    Call this once at the top of the main entry-point script.
    """
    for key, default in _DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default

    if not st.session_state.get("_hardware_defaults_applied", False):
        if torch.cuda.is_available():
            st.session_state.cpu_offloading = False
        st.session_state._hardware_defaults_applied = True

    if st.session_state.get("generation_queue") is None:
        st.session_state.generation_queue = GenerationQueue(max_size=50)

    if not st.session_state.get("kpi_session_started_at"):
        st.session_state.kpi_session_started_at = datetime.now().isoformat()


def sync_model_selection_state(session: MutableMapping[str, object] | None = None) -> None:
    """Synchronize model selector widget keys before widgets are instantiated.

    This must run early in the script (before sidebar/page model selectboxes are
    rendered). It resolves model changes coming from any selector and keeps all
    selector widget keys aligned without mutating widget state after creation.
    """
    from ui.config import MODEL_CONFIGS

    current_state = _state(session)
    fallback_model = "flux.2-klein-4b"
    known_models = set(MODEL_CONFIGS.keys())

    previous_model = str(current_state.get("model_name", fallback_model))
    if previous_model not in known_models:
        previous_model = fallback_model

    resolved_model = previous_model
    selector_keys = (
        "sidebar_quick_model",
        "gen_model_selector_widget",
        "edit_model_selector_widget",
    )

    for selector_key in selector_keys:
        candidate = current_state.get(selector_key)
        if isinstance(candidate, str) and candidate in known_models and candidate != previous_model:
            resolved_model = candidate
            break

    if resolved_model != previous_model:
        apply_model_defaults(resolved_model, current_state)

    current_state["model_name"] = resolved_model
    for selector_key in selector_keys:
        current_state[selector_key] = resolved_model


def reset_generation() -> None:
    """Clear the result of the last generation run."""
    st.session_state.current_image = None
    st.session_state.generation_metadata = None
    st.session_state.generation_time_s = None


def push_history(image_bytes: bytes, metadata: dict) -> None:
    """Prepend a finished generation to the in-session history (max 20 entries)."""
    entry = {"image_bytes": image_bytes, "metadata": metadata}
    history: list = st.session_state.generation_history
    history.insert(0, entry)
    st.session_state.generation_history = history[:20]


def restore_settings_from_history_entry(
    entry: dict[str, object],
    session: MutableMapping[str, object] | None = None,
) -> bool:
    """Restore generation settings from a history entry into session state."""
    current_state = _state(session)
    if not isinstance(entry, dict):
        return False

    metadata = entry.get("metadata", {})
    if not isinstance(metadata, dict):
        return False

    restored_any = False

    model_name = metadata.get("model")
    if isinstance(model_name, str) and model_name.strip():
        current_state["model_name"] = model_name
        restored_any = True

    if "num_steps" in metadata:
        current_state["num_steps"] = int(metadata["num_steps"])
        restored_any = True
    if "guidance" in metadata:
        current_state["guidance"] = float(metadata["guidance"])
        restored_any = True
    if "width" in metadata:
        current_state["width"] = int(metadata["width"])
        restored_any = True
    if "height" in metadata:
        current_state["height"] = int(metadata["height"])
        restored_any = True
    if "seed" in metadata:
        current_state["seed"] = int(metadata["seed"])
        current_state["use_random_seed"] = False
        restored_any = True

    dtype = metadata.get("dtype")
    if isinstance(dtype, str) and dtype.strip():
        current_state["dtype_str"] = dtype
        restored_any = True

    prompt = metadata.get("prompt")
    if isinstance(prompt, str):
        current_state["prompt"] = prompt
        current_state["gen_prompt"] = prompt
        current_state["editor_prompt"] = prompt
        restored_any = True

    phase2_controls = metadata.get("phase2_controls")
    if isinstance(phase2_controls, dict):
        current_state["phase2_controls_payload"] = phase2_controls
        seed_group_id = str(phase2_controls.get("seed_group_id", "")).strip()
        if seed_group_id and "seed" in metadata:
            register_seed_group_seed(seed_group_id, int(metadata["seed"]), session=current_state)
        restored_any = True

    if restored_any:
        model_for_preset = str(current_state.get("model_name", "flux.2-klein-4b"))
        num_steps = int(current_state.get("num_steps", 4))
        guidance = float(current_state.get("guidance", 1.0))
        current_state["quality_preset"] = infer_quality_preset(model_for_preset, num_steps, guidance)

    return restored_any


def restore_settings_from_snapshot(
    snapshot: dict[str, object],
    session: MutableMapping[str, object] | None = None,
) -> bool:
    """Restore a saved recovery snapshot into session state."""
    current_state = _state(session)
    if not isinstance(snapshot, dict):
        return False

    restored_any = False
    for key in (
        "model_name",
        "dtype_str",
        "cpu_offloading",
        "attn_slicing",
        "num_steps",
        "guidance",
        "width",
        "height",
        "seed",
        "prompt",
    ):
        if key in snapshot:
            current_state[key] = snapshot[key]
            restored_any = True

    prompt = current_state.get("prompt")
    if isinstance(prompt, str):
        current_state["gen_prompt"] = prompt
        current_state["editor_prompt"] = prompt

    if restored_any:
        model_name = str(current_state.get("model_name", "flux.2-klein-4b"))
        num_steps = int(current_state.get("num_steps", 4))
        guidance = float(current_state.get("guidance", 1.0))
        current_state["quality_preset"] = infer_quality_preset(model_name, num_steps, guidance)

    return restored_any


def _state(session: MutableMapping[str, object] | None = None) -> MutableMapping[str, object]:
    return st.session_state if session is None else session


def get_quality_preset_map(model_name: str) -> dict[str, dict[str, float | int]]:
    """Return the quality preset map for the given model."""
    from ui.config import MODEL_CONFIGS, QUALITY_PRESETS_BASE, QUALITY_PRESETS_DISTILLED

    model_cfg = MODEL_CONFIGS.get(model_name)
    if not model_cfg:
        return {}
    return QUALITY_PRESETS_DISTILLED if model_cfg.get("guidance_distilled") else QUALITY_PRESETS_BASE


def get_quality_preset_options(model_name: str) -> list[str]:
    """Return quality preset options including custom mode."""
    return ["— Custom —", *list(get_quality_preset_map(model_name).keys())]


def apply_quality_preset(
    model_name: str,
    preset_name: str,
    session: MutableMapping[str, object] | None = None,
) -> bool:
    """Apply a quality preset to session state; return True if preset applied."""
    current_state = _state(session)
    preset_map = get_quality_preset_map(model_name)
    preset = preset_map.get(preset_name)
    if not preset:
        current_state["quality_preset"] = "— Custom —"
        return False

    current_state["model_name"] = model_name
    current_state["quality_preset"] = preset_name
    current_state["num_steps"] = int(preset.get("num_steps", current_state.get("num_steps", 4)))
    if "guidance" in preset:
        current_state["guidance"] = float(preset["guidance"])
    return True


def infer_quality_preset(model_name: str, num_steps: int, guidance: float) -> str:
    """Infer active quality preset from current parameters, else custom."""
    preset_map = get_quality_preset_map(model_name)
    for preset_name, preset in preset_map.items():
        if int(preset.get("num_steps", num_steps)) != int(num_steps):
            continue
        expected_guidance = float(preset.get("guidance", guidance))
        if abs(float(guidance) - expected_guidance) > 1e-6:
            continue
        return preset_name
    return "— Custom —"


def apply_model_defaults(
    model_name: str,
    session: MutableMapping[str, object] | None = None,
) -> None:
    """Reset generation parameters to the recommended values for *model_name*."""
    from ui.config import MODEL_CONFIGS  # local import to avoid circular deps

    current_state = _state(session)
    model_cfg = MODEL_CONFIGS.get(model_name)
    if model_cfg is None:
        return
    rec = model_cfg["recommended"]
    current_state["num_steps"] = rec["num_steps"]
    current_state["guidance"] = rec["guidance"]
    current_state["width"] = rec["width"]
    current_state["height"] = rec["height"]
    current_state["dtype_str"] = rec["dtype"]
    current_state["model_name"] = model_name
    current_state["quality_preset"] = infer_quality_preset(
        model_name,
        num_steps=int(rec["num_steps"]),
        guidance=float(rec["guidance"]),
    )


def get(key: str, default=None):
    """Convenience getter that returns *default* if key is absent."""
    return st.session_state.get(key, default)


def list_projects(session: MutableMapping[str, object] | None = None) -> list[dict[str, str]]:
    """Return all project contexts from session state."""
    current_state = _state(session)
    projects = current_state.get("projects", [])
    if not isinstance(projects, list):
        return []
    cleaned: list[dict[str, str]] = []
    for item in projects:
        if not isinstance(item, dict):
            continue
        cleaned.append(
            {
                "project_id": str(item.get("project_id", "")).strip(),
                "project_type": str(item.get("project_type", "")).strip(),
                "geography": str(item.get("geography", "")).strip(),
                "climate_profile": str(item.get("climate_profile", "")).strip(),
                "design_phase": str(item.get("design_phase", "")).strip(),
                "team_label": str(item.get("team_label", "")).strip(),
            }
        )
    return cleaned


def create_or_update_project(
    *,
    project_type: str,
    geography: str,
    climate_profile: str,
    design_phase: str,
    team_label: str,
    project_id: str | None = None,
    session: MutableMapping[str, object] | None = None,
) -> dict[str, str]:
    """Create or update a project context and mark it active."""
    current_state = _state(session)
    projects = list_projects(current_state)

    normalized: dict[str, str] = {
        "project_id": str(project_id or "").strip(),
        "project_type": str(project_type).strip(),
        "geography": str(geography).strip(),
        "climate_profile": str(climate_profile).strip(),
        "design_phase": str(design_phase).strip(),
        "team_label": str(team_label).strip(),
    }

    if not normalized["project_id"]:
        normalized["project_id"] = f"proj-{len(projects) + 1:03d}"

    updated = False
    for idx, item in enumerate(projects):
        if item.get("project_id") == normalized["project_id"]:
            projects[idx] = normalized
            updated = True
            break
    if not updated:
        projects.append(normalized)

    current_state["projects"] = projects
    current_state["active_project_id"] = normalized["project_id"]
    return normalized


def set_active_project(project_id: str, session: MutableMapping[str, object] | None = None) -> bool:
    """Set active project by ID."""
    current_state = _state(session)
    projects = list_projects(current_state)
    wanted = str(project_id).strip()
    if not wanted:
        return False
    if not any(p.get("project_id") == wanted for p in projects):
        return False
    current_state["active_project_id"] = wanted
    return True


def get_active_project(session: MutableMapping[str, object] | None = None) -> dict[str, str] | None:
    """Return active project context or None if unavailable."""
    current_state = _state(session)
    projects = list_projects(current_state)
    active_project_id = str(current_state.get("active_project_id", "")).strip()
    if not active_project_id:
        return None
    for item in projects:
        if item.get("project_id") == active_project_id:
            return item
    return None


def has_active_project(session: MutableMapping[str, object] | None = None) -> bool:
    """Return True when an active project context exists."""
    return get_active_project(session=session) is not None


def get_seed_group_seed(
    seed_group_id: str,
    session: MutableMapping[str, object] | None = None,
) -> int | None:
    """Return stored seed for a seed-group ID, if available."""
    current_state = _state(session)
    groups = current_state.get("seed_groups", {})
    if not isinstance(groups, dict):
        return None
    raw = groups.get(str(seed_group_id).strip())
    if raw is None:
        return None
    return int(raw)


def register_seed_group_seed(
    seed_group_id: str,
    seed: int,
    session: MutableMapping[str, object] | None = None,
) -> int:
    """Persist and return seed for a seed-group ID."""
    current_state = _state(session)
    groups = current_state.get("seed_groups", {})
    if not isinstance(groups, dict):
        groups = {}
    normalized_id = str(seed_group_id).strip()
    if not normalized_id:
        return int(seed)
    if normalized_id in groups:
        return int(groups[normalized_id])
    groups[normalized_id] = int(seed)
    current_state["seed_groups"] = groups
    return int(seed)


# ─── Prompt Preset Helpers ───────────────────────────────────────────────────

def get_prompt_presets_for_active_project(
    session: MutableMapping[str, object] | None = None,
) -> dict[str, str]:
    """Return prompt presets for active project type, or empty dict if none."""
    from ui.config import PROMPT_PRESETS_BY_PROJECT_TYPE

    active_project = get_active_project(session=session)
    if not active_project:
        return {}
    project_type = active_project.get("project_type", "")
    return PROMPT_PRESETS_BY_PROJECT_TYPE.get(project_type, {})


def apply_prompt_preset(
    preset_name: str,
    session: MutableMapping[str, object] | None = None,
) -> str:
    """Apply a prompt preset to editor prompt field; return the preset text."""
    current_state = _state(session)
    presets = get_prompt_presets_for_active_project(session=current_state)
    preset_text = presets.get(preset_name, "")
    if preset_text:
        current_state["editor_prompt"] = preset_text
        current_state["prompt"] = preset_text
    return preset_text


def get_prompt_preset_options(
    session: MutableMapping[str, object] | None = None,
) -> list[str]:
    """Return list of prompt preset names for active project, including '— Custom —'."""
    presets = get_prompt_presets_for_active_project(session=session)
    return ["— Custom —", *list(presets.keys())]
