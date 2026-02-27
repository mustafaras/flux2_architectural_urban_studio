"""Generator page — text-to-image generation."""

from __future__ import annotations

import streamlit as st
import torch

from src.flux2.analytics_client import EventType, get_analytics
from src.flux2.error_types import classify_exception
from src.flux2.logging_config import log_error, log_operation
from src.flux2.prompt_taxonomy import (
    ATMOSPHERE_CONTROLS,
    CAMERA_FAMILIES,
    CLIMATE_MOOD_PROFILES,
    CONTINUITY_PROFILES,
    FACADE_STYLE_PACKS,
    FACADE_VOCABULARY,
    MASSING_DESCRIPTORS,
    MASSING_PRESET_PACKS,
    MOBILITY_PUBLIC_REALM,
    PROGRAM_OVERLAYS,
    SEASONAL_PRESETS,
    TIME_OF_DAY_PRESETS,
    apply_massing_preset,
    build_prompt_taxonomy_payload,
    build_phase2_prompt_segments,
    calculate_continuity_score,
    validate_material_controls,
)
from ui import config as cfg
from ui import error_handler
from ui import state, utils
from ui import icons
from ui.components_advanced import metadata_viewer, parameter_tuner, prompt_builder
from ui.components import (
    hardware_panel,
    help_model_comparison,
    help_prompt_tips,
    help_troubleshooting,
    image_result_panel,
    model_info_card,
    model_is_available_locally,
    model_missing_hint,
    model_selector,
    parameter_panel,
    prompt_input,
)


def render() -> None:
    """Entry point called from the main app to render the Generator tab."""
    icons.page_intro(
        "Text-to-Image Generator",
        "Write a prompt, tune parameters and hardware settings, then generate and review results.",
        icon="image",
    )

    # ── Layout: prompt | params | info ────────────────────────────────────
    col_prompt, col_params, col_info = st.columns([3, 2, 1.35], gap="large")

    with col_prompt:
        st.subheader("Inputs")
        _render_prompt_column()

    with col_params:
        st.subheader("Parameters")
        _render_params_column()

    with col_info:
        st.subheader("Hardware")
        _render_info_column()

    # ── Full-width result area ─────────────────────────────────────────────
    st.subheader("Results")
    _render_result_area()

    # ── Embedded prompt upsampler (compact) ───────────────────────────────
    st.divider()
    with st.expander(_upsampler_expander_title(), expanded=False):
        from ui.pages import upsampler
        upsampler.render_embedded()

    # ── Help ──────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Help")
    help_prompt_tips()
    help_model_comparison()
    help_troubleshooting()


# ─── Private helpers ──────────────────────────────────────────────────────────

def _render_prompt_column() -> None:
    pending_prompt = st.session_state.pop("_pending_prompt_for_generator", None)
    if pending_prompt:
        st.session_state.gen_prompt = pending_prompt
        st.session_state.prompt = pending_prompt
        st.success("Expanded prompt applied to Generator.")

    pending_massing_descriptors = st.session_state.pop("_pending_taxonomy_massing_descriptors", None)
    if isinstance(pending_massing_descriptors, list):
        st.session_state.taxonomy_massing_descriptors = pending_massing_descriptors
        st.success("Massing preset applied to taxonomy slots.")

    prompt = prompt_input(key="gen_prompt")
    st.session_state.prompt = prompt

    with st.expander("Architecture / Urban Prompt Taxonomy", expanded=True):
        st.caption("Use structured slots to compose reproducible architecture prompts.")

        massing = st.multiselect(
            "Massing Descriptors",
            options=MASSING_DESCRIPTORS,
            key="taxonomy_massing_descriptors",
        )
        facade = st.multiselect(
            "Façade Vocabulary",
            options=FACADE_VOCABULARY,
            key="taxonomy_facade_vocabulary",
        )
        mobility = st.multiselect(
            "Mobility / Public-Realm Descriptors",
            options=MOBILITY_PUBLIC_REALM,
            key="taxonomy_mobility_public_realm",
        )
        atmosphere = st.multiselect(
            "Atmosphere Controls",
            options=ATMOSPHERE_CONTROLS,
            key="taxonomy_atmosphere_controls",
        )

        taxonomy_payload = build_prompt_taxonomy_payload(
            free_text=prompt,
            massing=massing,
            facade=facade,
            mobility=mobility,
            atmosphere=atmosphere,
        )

        if taxonomy_payload["lint_results"]:
            for lint in taxonomy_payload["lint_results"]:
                st.warning(lint)
        else:
            st.caption("No taxonomy conflicts detected.")

        st.code(taxonomy_payload["assembled_prompt"] or "(empty prompt)", language="markdown")

    with st.expander("Phase 2 Domain Controls", expanded=False):
        st.caption("Massing, façade/material realism, view locks, continuity, and environment presets.")

        massing_preset_names = list(MASSING_PRESET_PACKS.keys())
        selected_massing_preset = st.selectbox(
            "Massing Preset",
            options=massing_preset_names,
            key="phase2_massing_preset",
        )
        massing_pack = apply_massing_preset(selected_massing_preset)
        st.caption(f"Preview: {massing_pack.get('preview', '')}")
        st.caption(f"Recommended context: {massing_pack.get('recommended_use', '')}")
        if st.button("Apply Massing Preset to Slots", use_container_width=True, key="phase2_apply_massing_preset"):
            st.session_state._pending_taxonomy_massing_descriptors = list(massing_pack.get("descriptors", []))
            st.rerun()

        facade_style_names = list(FACADE_STYLE_PACKS.keys())
        selected_facade_pack = st.selectbox(
            "Façade Style Pack",
            options=facade_style_names,
            key="phase2_facade_style_pack",
        )
        selected_program_overlays = st.multiselect(
            "Program Overlays",
            options=PROGRAM_OVERLAYS,
            default=["mixed-use"],
            key="phase2_program_overlays",
        )

        col_mat_1, col_mat_2, col_mat_3 = st.columns(3)
        with col_mat_1:
            glazing_ratio = st.slider("Glazing Ratio (%)", min_value=0, max_value=95, value=45, key="phase2_glazing_ratio")
        with col_mat_2:
            texture_roughness = st.slider("Texture Roughness", min_value=0.0, max_value=1.0, value=0.35, step=0.05, key="phase2_texture_roughness")
        with col_mat_3:
            reflectance_mood = st.selectbox("Reflectance Mood", options=["muted", "balanced", "reflective"], key="phase2_reflectance_mood")

        material_realism, material_warnings = validate_material_controls(
            glazing_ratio=glazing_ratio,
            texture_roughness=texture_roughness,
            reflectance_mood=reflectance_mood,
        )
        for warning in material_warnings:
            st.warning(warning)

        st.divider()
        col_view_1, col_view_2, col_view_3 = st.columns(3)
        with col_view_1:
            camera_family = st.selectbox("Camera Family", options=CAMERA_FAMILIES, key="phase2_camera_family")
        with col_view_2:
            lens_mm = st.slider("Lens (mm)", min_value=14, max_value=120, value=35, key="phase2_lens_mm")
        with col_view_3:
            perspective_lock = st.checkbox("Perspective Lock", value=True, key="phase2_perspective_lock")
        render_set_id = st.text_input("Render Set ID", value=st.session_state.get("phase2_render_set_id", "render-set-001"), key="phase2_render_set_id")

        st.divider()
        col_cont_1, col_cont_2, col_cont_3 = st.columns(3)
        with col_cont_1:
            seed_group_id = st.text_input("Seed Group ID", value=st.session_state.get("phase2_seed_group_id", ""), key="phase2_seed_group_id")
        with col_cont_2:
            replay_seed_group = st.checkbox("Replay Seed Group", value=True, key="phase2_replay_seed_group")
        with col_cont_3:
            continuity_profile = st.selectbox("Continuity Profile", options=CONTINUITY_PROFILES, key="phase2_continuity_profile")

        col_env_1, col_env_2, col_env_3 = st.columns(3)
        with col_env_1:
            time_of_day = st.selectbox("Time of Day Preset", options=TIME_OF_DAY_PRESETS, key="phase2_time_of_day")
        with col_env_2:
            seasonal_profile = st.selectbox("Seasonal Profile", options=SEASONAL_PRESETS, key="phase2_seasonal_profile")
        with col_env_3:
            climate_mood = st.selectbox("Climate Mood Profile", options=CLIMATE_MOOD_PROFILES, key="phase2_climate_mood")

        phase2_controls = {
            "massing_preset": selected_massing_preset,
            "facade_style_pack": selected_facade_pack,
            "material_realism": material_realism,
            "program_overlays": selected_program_overlays,
            "view_lock": {
                "camera_family": camera_family,
                "lens_mm": lens_mm,
                "perspective_lock": perspective_lock,
                "render_set_id": render_set_id,
            },
            "seed_group_id": seed_group_id,
            "replay_seed_group": replay_seed_group,
            "continuity_profile": continuity_profile,
            "environment": {
                "time_of_day": time_of_day,
                "seasonal_profile": seasonal_profile,
                "climate_mood": climate_mood,
            },
        }
        st.session_state.phase2_controls_payload = phase2_controls

        continuity_score = calculate_continuity_score(phase2_controls)
        st.caption(f"Continuity score summary (render set): {continuity_score}/100")

        if st.button(
            "Apply Structured Prompt",
            key="gen_apply_structured_prompt",
            use_container_width=True,
            help="Assemble prompt from taxonomy slots + free text.",
        ):
            phase2_controls = st.session_state.get("phase2_controls_payload", {})
            phase2_segments = build_phase2_prompt_segments(phase2_controls if isinstance(phase2_controls, dict) else {})
            scenario_module_contract = st.session_state.get("scenario_module_contract", {})
            if isinstance(scenario_module_contract, dict) and scenario_module_contract:
                taxonomy_payload["slots"]["scenario_module_contract"] = scenario_module_contract
                phase2_segments.append(
                    "urban modules: "
                    f"street {scenario_module_contract.get('street_hierarchy_template', '')}, "
                    f"realm {scenario_module_contract.get('public_realm_typology', '')}, "
                    f"multimodal {scenario_module_contract.get('multimodal_preset', '')}"
                )

            taxonomy_payload["slots"]["phase2_controls"] = phase2_controls
            taxonomy_payload["continuity_score"] = calculate_continuity_score(phase2_controls if isinstance(phase2_controls, dict) else {})

            final_prompt = taxonomy_payload["assembled_prompt"]
            if phase2_segments:
                final_prompt = f"{final_prompt}; {'; '.join(phase2_segments)}" if final_prompt else "; ".join(phase2_segments)

            st.session_state.prompt_taxonomy_payload = taxonomy_payload
            st.session_state.prompt = final_prompt
            st.session_state.gen_prompt = final_prompt
            st.session_state.kpi_compose_prompt_count = int(st.session_state.get("kpi_compose_prompt_count", 0)) + 1
            get_analytics().log_event(
                EventType.COMPOSE_PROMPT,
                {
                    "template_version": taxonomy_payload.get("template_version", ""),
                    "lint_count": len(taxonomy_payload.get("lint_results", [])),
                    "slot_counts": {
                        "massing": len(massing),
                        "facade": len(facade),
                        "mobility": len(mobility),
                        "atmosphere": len(atmosphere),
                    },
                    "phase2_enabled": True,
                    "continuity_score": taxonomy_payload.get("continuity_score", 0),
                },
            )
            st.success("Structured prompt applied.")
            st.rerun()

    with st.expander("Advanced Prompt Builder", expanded=False):
        built = prompt_builder(default_prompt=prompt, key_prefix="gen_adv_prompt")
        if st.button(
            "Apply Built Prompt",
            key="gen_apply_built_prompt",
            use_container_width=True,
            help="Apply the advanced prompt builder text to Generator input.",
        ):
            st.session_state._pending_prompt_for_generator = built.text
            st.success("Advanced prompt applied.")

    col_a, col_b = st.columns([1, 1])
    with col_a:
        if st.button(cfg.GEN_EXPAND_PROMPT_LABEL, help=cfg.GEN_EXPAND_PROMPT_HELP, use_container_width=True):
            st.session_state.upsample_original_prompt = prompt
            st.info("Prompt sent to the embedded Prompt Upsampler section below.")
    with col_b:
        if st.button(cfg.GEN_RESET_PARAMS_LABEL, help=cfg.GEN_RESET_PARAMS_HELP, use_container_width=True):
            state.apply_model_defaults(st.session_state.get("model_name", "flux.2-klein-4b"))
            st.success("Parameters reset to recommended defaults.")


def _upsampler_expander_title() -> str:
    compare_fast = st.session_state.get("upsample_compare_fast", "")
    compare_quality = st.session_state.get("upsample_compare_quality", "")
    result = st.session_state.get("upsample_result_prompt", "")

    if compare_fast and compare_quality:
        status = "compared"
    elif result:
        status = "ready"
    else:
        status = "idle"

    return f"Prompt Upsampler [{status}]"


def _render_params_column() -> None:
    advanced = st.session_state.get("advanced_mode", False)

    model_name = model_selector(key_prefix="gen")
    params = parameter_panel(model_name, advanced, key_prefix="gen")

    # Persist selections
    for k, v in {**params, "model_name": model_name}.items():
        st.session_state[k] = v

    st.divider()

    prompt = st.session_state.get("prompt", "")
    active_project = state.get_active_project()
    use_architecture_workflow = bool(st.session_state.get("use_architecture_workflow", True))
    selected_navigation_mode = st.session_state.get("navigation_mode_selector", "Architecture Workflow")
    require_project = use_architecture_workflow and selected_navigation_mode != "Legacy Compatibility"
    blockers = utils.get_generation_blockers(
        prompt=prompt,
        require_project=require_project,
        active_project=active_project,
    )
    generate_disabled = len(blockers) > 0

    col_gen, col_clear = st.columns(2)
    with col_gen:
        if st.button(
            cfg.GEN_BUTTON_LABEL,
            type="primary",
            use_container_width=True,
            disabled=generate_disabled,
            help=blockers[0] if blockers else None,
        ):
            st.session_state["_trigger_generate"] = True
        if blockers:
            st.caption(blockers[0])
    with col_clear:
        if st.button(cfg.GEN_CLEAR_CACHE_LABEL, use_container_width=True, help=cfg.GEN_CLEAR_CACHE_HELP):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                st.success(cfg.STATUS_CACHE_CLEARED)

    out_dir = st.text_input(
        cfg.GEN_OUTPUT_DIR_LABEL,
        value=st.session_state.get("output_dir", "outputs"),
        help=cfg.GEN_OUTPUT_DIR_HELP,
        key="gen_output_dir",
    )
    st.session_state.output_dir = out_dir

    with st.expander("Advanced Parameter Tuner", expanded=False):
        tuned = parameter_tuner(key_prefix="gen_adv_tuner")
        if st.button(
            "Apply Tuner Values",
            key="gen_apply_tuner_values",
            use_container_width=True,
            help="Apply tuner values to generation parameters without starting a run.",
        ):
            st.session_state.num_steps = int(tuned["steps"])
            st.session_state.guidance = float(tuned["guidance"])
            st.success("Tuner values applied.")


def _render_info_column() -> None:
    advanced = st.session_state.get("advanced_mode", False)
    model_name = st.session_state.get("model_name", "flux.2-klein-4b")

    st.caption("Model & hardware")
    model_info_card(model_name)
    st.divider()
    hw = hardware_panel(advanced, key_prefix="gen")
    for k, v in hw.items():
        st.session_state[k] = v

    st.divider()
    st.subheader("Recommended Defaults")
    rec = cfg.MODEL_CONFIGS[model_name]["recommended"]
    st.caption(f"Steps: {rec['num_steps']}  |  Guidance: {rec['guidance']}")
    st.caption(f"Resolution: {rec['width']} × {rec['height']} px")


def _render_result_area() -> None:
    if not st.session_state.get("_trigger_generate"):
        return
    st.session_state["_trigger_generate"] = False

    prompt = st.session_state.get("prompt", "")
    err = utils.validate_prompt(prompt)
    if err:
        st.error(err)
        return

    model_name: str = st.session_state.get("model_name", "flux.2-klein-4b")
    if not model_is_available_locally(model_name):
        st.error(model_missing_hint(model_name))
        return

    dtype_str: str = st.session_state.get("dtype_str", "bf16")
    cpu_offloading: bool = st.session_state.get("cpu_offloading", False)
    attn_slicing: bool = st.session_state.get("attn_slicing", True)
    num_steps: int = st.session_state.get("num_steps", 4)
    guidance: float = st.session_state.get("guidance", 1.0)
    width: int = st.session_state.get("width", 768)
    height: int = st.session_state.get("height", 768)
    out_dir: str = st.session_state.get("output_dir", "outputs")
    use_random: bool = st.session_state.get("use_random_seed", False)
    seed_raw: int = st.session_state.get("seed", 1)
    seed = utils.resolve_seed(seed_raw, use_random)
    phase2_controls = st.session_state.get("phase2_controls_payload", {})
    if not isinstance(phase2_controls, dict):
        phase2_controls = {}

    seed_group_id = str(phase2_controls.get("seed_group_id", "")).strip()
    replay_seed_group = bool(phase2_controls.get("replay_seed_group", False))
    if seed_group_id and replay_seed_group:
        stored_seed = state.get_seed_group_seed(seed_group_id)
        if stored_seed is not None:
            seed = int(stored_seed)
    elif seed_group_id:
        state.register_seed_group_seed(seed_group_id, seed)

    continuity_score = calculate_continuity_score(phase2_controls)
    phase2_controls["continuity_score"] = continuity_score

    device = utils.get_device()

    recovery_payload = {
        "mode": "t2i",
        "model_name": model_name,
        "dtype_str": dtype_str,
        "cpu_offloading": cpu_offloading,
        "attn_slicing": attn_slicing,
        "num_steps": num_steps,
        "guidance": guidance,
        "width": width,
        "height": height,
        "seed": seed,
        "prompt": prompt,
    }
    error_handler.save_recovery_snapshot(recovery_payload)

    prompt_safety_status = "disabled"
    output_safety_status = "disabled"

    try:
        log_operation("generation.start", "running", {"model": model_name, "width": width, "height": height})
        with st.spinner(cfg.STATUS_LOADING_MODEL):
            from src.flux2.streamlit_adapter import get_adapter
            adapter = get_adapter()
            try:
                adapter.load(
                    model_name=model_name,
                    dtype_str=dtype_str,
                    cpu_offloading=cpu_offloading,
                    attn_slicing=attn_slicing,
                )
            except Exception:
                if "9b" in model_name:
                    fallback_model = model_name.replace("9b", "4b")
                    st.session_state.model_name = fallback_model
                    model_name = fallback_model
                    log_operation("generation.fallback_model", "running", {"from": recovery_payload["model_name"], "to": fallback_model})
                    adapter.load(
                        model_name=fallback_model,
                        dtype_str=dtype_str,
                        cpu_offloading=cpu_offloading,
                        attn_slicing=attn_slicing,
                    )
                else:
                    raise

        if st.session_state.get("safety_check_prompt", False):
            prompt_safety_status = "passed"
            if not adapter.check_prompt_safety(prompt):
                prompt_safety_status = "blocked"
                st.error(cfg.ERR_SAFETY_PROMPT)
                st.info("Go to Settings → Safety to review or adjust safety checks.")
                return

        with st.spinner(cfg.STATUS_GENERATING):
            timer = utils.Timer()
            with timer:
                img = adapter.generate(
                    prompt=prompt,
                    num_steps=num_steps,
                    guidance=guidance,
                    width=width,
                    height=height,
                    seed=seed,
                )
    except torch.cuda.OutOfMemoryError:
        error_handler.apply_quality_reduction()
        error_handler.maybe_switch_to_fallback_model()
        ctx = classify_exception(
            torch.cuda.OutOfMemoryError("Out of memory during generation"),
            location="generator._render_result_area",
            metadata={"model": model_name, "width": width, "height": height, "steps": num_steps},
        )
        log_error(ctx)
        error_handler.display_recoverable_error(ctx)
        return
    except Exception as exc:  # noqa: BLE001
        ctx = classify_exception(
            exc,
            location="generator._render_result_area",
            metadata={"model": model_name, "width": width, "height": height, "steps": num_steps},
        )
        log_error(ctx)
        error_handler.display_recoverable_error(ctx, suggestion="Use Retry first. If it fails again, use Try Alternative.")
        return

    if st.session_state.get("safety_check_output", False):
        output_safety_status = "passed"
        if not adapter.check_image_safety(img):
            output_safety_status = "blocked"
            st.error(cfg.ERR_SAFETY_IMAGE)
            st.info("Go to Settings → Safety to review or adjust safety checks.")
            return

    out_path = utils.save_image(img, out_dir, model_name, seed, num_steps, guidance)
    metadata = utils.build_metadata(
        model_name=model_name,
        prompt=prompt,
        seed=seed,
        num_steps=num_steps,
        guidance=guidance,
        width=width,
        height=height,
        dtype_str=dtype_str,
        generation_time_s=timer.elapsed,
        project_context=state.get_active_project(),
        prompt_taxonomy=st.session_state.get("prompt_taxonomy_payload"),
        phase2_controls=phase2_controls,
    )
    metadata["safety_prompt_status"] = prompt_safety_status
    metadata["safety_output_status"] = output_safety_status
    metadata["continuity_score"] = continuity_score

    if seed_group_id:
        state.register_seed_group_seed(seed_group_id, seed)

    st.session_state.current_image = img
    st.session_state.generation_metadata = metadata
    st.session_state.generation_time_s = timer.elapsed

    state.push_history(utils.pil_to_bytes(img), metadata)

    st.session_state.kpi_iteration_count = int(st.session_state.get("kpi_iteration_count", 0)) + 1
    latencies = list(st.session_state.get("kpi_generation_latencies", []))
    latencies.append(float(timer.elapsed))
    st.session_state.kpi_generation_latencies = latencies[-200:]
    get_analytics().log_event(
        EventType.GENERATE_OPTION,
        {
            "model": model_name,
            "duration_ms": int(timer.elapsed * 1000),
            "has_project": bool(state.get_active_project()),
            "prompt_template_version": metadata.get("prompt_template_version", ""),
            "lint_count": len(metadata.get("prompt_lint_results", [])),
            "continuity_score": continuity_score,
        },
    )

    image_result_panel(
        img=img,
        metadata=metadata,
        generation_time_s=timer.elapsed,
        output_path=str(out_path),
        prompt=prompt,
    )

    with st.expander("Generation Metadata (Advanced)", expanded=False):
        metadata_viewer(metadata, title="Metadata Inspector", key_prefix="gen_meta_viewer")

    # Update seed display if random was used
    if use_random:
        st.session_state.seed = seed

    log_operation("generation.finish", "success", {"model": model_name, "time_s": round(timer.elapsed, 3)})
    error_handler.clear_recovery_snapshot()
