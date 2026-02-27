"""Image Editor page — image-to-image / reference-guided generation."""

from __future__ import annotations

import streamlit as st
import torch

from src.flux2.error_types import classify_exception
from src.flux2.logging_config import log_error, log_operation
from ui import config as cfg
from ui import icons
from ui import error_handler
from ui import state, utils
from ui.components import (
    hardware_panel,
    help_image_editor,
    help_troubleshooting,
    image_result_panel,
    image_uploader,
    model_info_card,
    model_is_available_locally,
    model_missing_hint,
    model_selector,
    parameter_panel,
)


def render() -> None:
    """Entry point called from the main app to render the Image Editor tab."""
    icons.page_intro(
        "Image-to-Image Editor",
        "Upload reference images, set parameters and hardware options, then generate and review results.",
        icon="edit",
    )

    advanced = st.session_state.get("advanced_mode", False)

    col_ref, col_params = st.columns([2, 2], gap="large")

    with col_ref:
        st.subheader("Inputs")
        _render_reference_column()

    with col_params:
        st.subheader("Parameters")
        _render_params_column(advanced)

    st.subheader("Results")
    _render_result_area()

    st.divider()
    st.subheader("Help")
    help_image_editor()
    help_troubleshooting()


# ─── Private helpers ──────────────────────────────────────────────────────────

def _render_reference_column() -> None:
    st.caption("Reference images")
    reference_images = image_uploader(
        label=cfg.EDITOR_UPLOAD_LABEL,
        key="editor_ref_upload",
        help_text=cfg.EDITOR_UPLOAD_HELP,
        accept_multiple=True,
    )
    st.session_state.reference_images = reference_images

    if reference_images:
        st.success(f"{len(reference_images)} image(s) loaded.")
        cols = st.columns(min(len(reference_images), 3))
        for i, img in enumerate(reference_images[:3]):
            with cols[i]:
                st.image(img, use_container_width=True, caption=f"Ref {i + 1}")

        match_size = st.checkbox(
            cfg.EDITOR_MATCH_SIZE_LABEL,
            value=False,
            help=cfg.EDITOR_MATCH_SIZE_HELP,
            key="editor_match_size",
        )
        if match_size and reference_images:
            w, h = reference_images[0].size
            st.info(f"Output will be resized to {w} × {h} px (from reference 1).")
            st.session_state.match_image_size = 0
        else:
            st.session_state.match_image_size = None
    else:
        st.info("No reference images uploaded yet.")


def _render_params_column(advanced: bool) -> None:
    model_name = model_selector(key_prefix="editor")
    model_info_card(model_name)

    # ── Prompt Preset Selector ────────────────────────────────────────────
    active_project = state.get_active_project()
    if active_project:
        st.markdown("##### Prompt Preset")
        preset_options = state.get_prompt_preset_options()
        selected_preset = st.selectbox(
            cfg.EDITOR_PROMPT_PRESET_LABEL,
            options=preset_options,
            index=0,
            help=cfg.EDITOR_PROMPT_PRESET_HELP,
            key="editor_prompt_preset_selector",
        )
        if st.button("Apply Preset", use_container_width=True, key="editor_apply_prompt_preset"):
            if selected_preset == "— Custom —":
                st.info("Select a preset to apply.")
            else:
                preset_text = state.apply_prompt_preset(selected_preset)
                if preset_text:
                    st.success(f"Preset applied: {selected_preset}")
                    st.rerun()
                else:
                    st.warning("Preset not found for the active project type.")
    else:
        st.caption("💡 Create a project in **Project Setup** to unlock prompt presets.")

    # ── Prompt Input ──────────────────────────────────────────────────────
    prompt = st.text_area(
        cfg.EDITOR_PROMPT_LABEL,
        value=st.session_state.get("prompt", ""),
        height=100,
        help=cfg.EDITOR_PROMPT_HELP,
        key="editor_prompt",
        placeholder="Describe what to generate using the reference image(s)...",
    )

    prompt_error = utils.validate_prompt(prompt)
    if prompt_error:
        st.warning(prompt_error)

    params = parameter_panel(model_name, advanced, key_prefix="editor")

    st.markdown("##### Hardware")
    hw = hardware_panel(advanced, key_prefix="editor")

    for k, v in {**params, **hw, "model_name": model_name}.items():
        st.session_state[k] = v

    reference_images = st.session_state.get("reference_images", [])
    blockers = utils.get_generation_blockers(
        prompt=prompt,
        require_reference=True,
        reference_images=reference_images,
    )
    edit_disabled = len(blockers) > 0

    if st.button(
        "Generate with Reference",
        type="primary",
        use_container_width=True,
        disabled=edit_disabled,
        help=blockers[0] if blockers else "Generate an image using current prompt and uploaded reference images.",
    ):
        st.session_state["_trigger_edit"] = True
    if blockers:
        st.caption(blockers[0])

    out_dir = st.text_input(
        cfg.GEN_OUTPUT_DIR_LABEL,
        value=st.session_state.get("output_dir", "outputs"),
        key="editor_output_dir",
    )
    st.session_state.output_dir = out_dir


def _render_result_area() -> None:
    if not st.session_state.get("_trigger_edit"):
        return
    st.session_state["_trigger_edit"] = False

    prompt = st.session_state.get("editor_prompt", "")
    err = utils.validate_prompt(prompt)
    if err:
        st.error(err)
        return

    reference_images = st.session_state.get("reference_images", [])
    if not reference_images:
        st.error("Please upload at least one reference image before generating.")
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
    out_dir: str = st.session_state.get("output_dir", "outputs")
    use_random: bool = st.session_state.get("use_random_seed", False)
    seed = utils.resolve_seed(st.session_state.get("seed", 1), use_random)

    # Match size from reference if requested
    match_size_idx = st.session_state.get("match_image_size")
    if match_size_idx is not None and match_size_idx < len(reference_images):
        width, height = reference_images[match_size_idx].size
    else:
        width = st.session_state.get("width", 768)
        height = st.session_state.get("height", 768)

    recovery_payload = {
        "mode": "i2i",
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
        "reference_count": len(reference_images),
    }
    error_handler.save_recovery_snapshot(recovery_payload)

    prompt_safety_status = "disabled"
    output_safety_status = "disabled"

    try:
        log_operation("editor.start", "running", {"model": model_name, "reference_count": len(reference_images)})
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
                    log_operation("editor.fallback_model", "running", {"from": recovery_payload["model_name"], "to": fallback_model})
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

        if st.session_state.get("safety_check_output", False):
            output_safety_status = "passed"
            for ref_img in reference_images:
                if not adapter.check_image_safety(ref_img):
                    output_safety_status = "blocked"
                    st.error(cfg.ERR_SAFETY_IMAGE)
                    st.info("Go to Settings → Safety to review or adjust safety checks.")
                    return

        with st.spinner("Encoding reference images and generating..."):
            timer = utils.Timer()
            with timer:
                img = adapter.edit(
                    prompt=prompt,
                    reference_images=reference_images,
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
            torch.cuda.OutOfMemoryError("Out of memory during image editing"),
            location="editor._render_result_area",
            metadata={"model": model_name, "width": width, "height": height, "steps": num_steps},
        )
        log_error(ctx)
        error_handler.display_recoverable_error(ctx)
        return
    except Exception as exc:  # noqa: BLE001
        ctx = classify_exception(
            exc,
            location="editor._render_result_area",
            metadata={"model": model_name, "width": width, "height": height, "steps": num_steps},
        )
        log_error(ctx)
        error_handler.display_recoverable_error(ctx, suggestion="Retry. If the issue persists, use Try Alternative to reduce load.")
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
        has_reference=True,
    )
    metadata["safety_prompt_status"] = prompt_safety_status
    metadata["safety_output_status"] = output_safety_status

    state.push_history(utils.pil_to_bytes(img), metadata)
    image_result_panel(img, metadata, timer.elapsed, str(out_path), prompt=prompt)

    if use_random:
        st.session_state.seed = seed

    log_operation("editor.finish", "success", {"model": model_name, "time_s": round(timer.elapsed, 3)})
    error_handler.clear_recovery_snapshot()
