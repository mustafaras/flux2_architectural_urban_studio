"""Reusable Streamlit widget components for the FLUX.2 Professional UI."""

from __future__ import annotations

import io
import os
import json
from pathlib import Path
from typing import Any

import streamlit as st
from PIL import Image

from ui import config as cfg
from ui import state, utils
from ui.components_advanced import image_comparison


_MODEL_ENV_KEYS = {
    "flux.2-klein-4b": "KLEIN_4B_MODEL_PATH",
    "flux.2-klein-9b": "KLEIN_9B_MODEL_PATH",
    "flux.2-klein-base-4b": "KLEIN_4B_BASE_MODEL_PATH",
    "flux.2-klein-base-9b": "KLEIN_9B_BASE_MODEL_PATH",
    "flux.2-dev": "FLUX2_MODEL_PATH",
}

_MODEL_DEFAULT_FILES = {
    "flux.2-klein-4b": Path("weights/flux-2-klein-4b.safetensors"),
    "flux.2-klein-9b": Path("weights/flux-2-klein-9b.safetensors"),
    "flux.2-klein-base-4b": Path("weights/flux-2-klein-base-4b.safetensors"),
    "flux.2-klein-base-9b": Path("weights/flux-2-klein-base-9b.safetensors"),
    "flux.2-dev": Path("weights/flux2-dev.safetensors"),
}


def _resolve_model_file_path(model_name: str) -> Path | None:
    env_key = _MODEL_ENV_KEYS.get(model_name)
    if env_key:
        path_value = os.environ.get(env_key, "").strip()
        if path_value:
            env_path = Path(path_value)
            if env_path.exists():
                return env_path

    default_file = _MODEL_DEFAULT_FILES.get(model_name)
    if default_file and default_file.exists():
        return default_file

    return None


def model_is_available_locally(model_name: str) -> bool:
    return _resolve_model_file_path(model_name) is not None


def model_missing_hint(model_name: str) -> str:
    env_key = _MODEL_ENV_KEYS.get(model_name, "MODEL_PATH")
    expected_path = _MODEL_DEFAULT_FILES.get(model_name)
    expected_text = str(expected_path) if expected_path else "weights/<model>.safetensors"
    return (
        f"Model file not found for {cfg.MODEL_CONFIGS.get(model_name, {}).get('display_name', model_name)}. "
        f"Download the weights and copy them to `{expected_text}` or set `{env_key}` in Settings > Custom Model Paths."
    )


# ─── Model Selector ───────────────────────────────────────────────────────────

def model_selector(key_prefix: str = "gen") -> str:
    """Render a model selector widget and return the selected model key."""
    options = [m for m in cfg.MODEL_DISPLAY_ORDER if m in cfg.MODEL_CONFIGS]

    current = st.session_state.get("model_name", options[0])
    try:
        idx = options.index(current)
    except ValueError:
        idx = 0

    selected_key = st.selectbox(
        cfg.GEN_MODEL_LABEL,
        options=options,
        index=idx,
        format_func=lambda model_key: (
            cfg.MODEL_CONFIGS[model_key]["display_name"]
            if model_is_available_locally(model_key)
            else f"{cfg.MODEL_CONFIGS[model_key]['display_name']} (missing)"
        ),
        help=cfg.GEN_MODEL_HELP,
        key=f"{key_prefix}_model_selector_widget",
    )

    if not model_is_available_locally(selected_key):
        st.warning(model_missing_hint(selected_key))

    # Apply recommended defaults when model changes
    if selected_key != current:
        state.apply_model_defaults(selected_key)

    return selected_key


def model_info_card(model_name: str) -> None:
    """Render a compact info card for the currently selected model."""
    mcfg = cfg.MODEL_CONFIGS.get(model_name)
    if not mcfg:
        return
    with st.container(border=True):
        st.caption(
            f"{mcfg['speed_label']}  |  {mcfg['quality_label']}  |  "
            f"VRAM: {mcfg['vram_estimate']}"
        )
        st.caption(mcfg["description"])
        if model_is_available_locally(model_name):
            model_path = _resolve_model_file_path(model_name)
            st.caption(f"Local model file: {model_path}")
        else:
            st.caption(model_missing_hint(model_name))


# ─── Parameter Panel ──────────────────────────────────────────────────────────

def parameter_panel(model_name: str, advanced: bool, key_prefix: str = "gen") -> dict[str, Any]:
    """
    Render generation parameter widgets (steps, guidance, size, seed, dtype).

    Returns a dict with the current parameter values.
    """
    mcfg = cfg.MODEL_CONFIGS[model_name]
    ranges = mcfg["parameter_ranges"]
    rec = mcfg["recommended"]
    fixed = mcfg["fixed_params"]

    params: dict[str, Any] = {}

    st.subheader("Generation Parameters")

    # ── Quality Presets ──────────────────────────────────────────────────
    preset_map = state.get_quality_preset_map(model_name)
    preset_options = state.get_quality_preset_options(model_name)
    current_preset = st.session_state.get("quality_preset", "— Custom —")
    if current_preset not in preset_options:
        current_preset = "— Custom —"

    selected_preset = st.selectbox(
        cfg.GEN_QUALITY_PRESET_LABEL,
        options=preset_options,
        index=preset_options.index(current_preset),
        help=cfg.GEN_QUALITY_PRESET_HELP,
        key=f"{key_prefix}_param_quality_preset",
    )
    st.session_state.quality_preset = selected_preset

    col_apply_preset, col_apply_rec = st.columns(2)
    with col_apply_preset:
        if st.button(
            "Apply Preset",
            key=f"{key_prefix}_apply_quality_preset",
            use_container_width=True,
            help="Apply the selected quality preset to steps and guidance.",
        ):
            if selected_preset == "— Custom —":
                st.info("Select a preset to apply.")
            elif state.apply_quality_preset(model_name, selected_preset):
                st.success(f"Preset applied: {selected_preset}")
            else:
                st.warning("Preset could not be applied for the selected model.")
    with col_apply_rec:
        if st.button(
            "Apply Recommended",
            key=f"{key_prefix}_apply_model_recommended",
            use_container_width=True,
            help="Restore model-recommended defaults for parameters and dtype.",
        ):
            state.apply_model_defaults(model_name)
            st.success("Recommended parameters applied.")

    # ── Steps ──────────────────────────────────────────────────────────────
    if "num_steps" in fixed and not advanced:
        st.metric(cfg.GEN_STEPS_LABEL, rec["num_steps"], help=cfg.GEN_STEPS_HELP)
        params["num_steps"] = rec["num_steps"]
    else:
        r = ranges["num_steps"]
        params["num_steps"] = st.slider(
            cfg.GEN_STEPS_LABEL,
            min_value=r["min"],
            max_value=r["max"],
            value=st.session_state.get("num_steps", rec["num_steps"]),
            step=r["step"],
            help=cfg.GEN_STEPS_HELP,
            key=f"{key_prefix}_param_num_steps",
        )

    # ── Guidance ────────────────────────────────────────────────────────────
    if "guidance" in fixed and not advanced:
        st.metric(cfg.GEN_GUIDANCE_LABEL, rec["guidance"], help=cfg.GEN_GUIDANCE_HELP)
        params["guidance"] = rec["guidance"]
    else:
        r = ranges["guidance"]
        params["guidance"] = st.slider(
            cfg.GEN_GUIDANCE_LABEL,
            min_value=float(r["min"]),
            max_value=float(r["max"]),
            value=float(st.session_state.get("guidance", rec["guidance"])),
            step=float(r["step"]),
            help=cfg.GEN_GUIDANCE_HELP,
            key=f"{key_prefix}_param_guidance",
        )

    # ── Resolution ─────────────────────────────────────────────────────────
    col_w, col_h = st.columns(2)
    with col_w:
        params["width"] = st.selectbox(
            cfg.GEN_WIDTH_LABEL,
            options=ranges["width_options"],
            index=ranges["width_options"].index(
                st.session_state.get("width", rec["width"])
            ),
            help="Output width in pixels.",
            key=f"{key_prefix}_param_width",
        )
    with col_h:
        params["height"] = st.selectbox(
            cfg.GEN_HEIGHT_LABEL,
            options=ranges["height_options"],
            index=ranges["height_options"].index(
                st.session_state.get("height", rec["height"])
            ),
            help="Output height in pixels.",
            key=f"{key_prefix}_param_height",
        )

    # ── Seed ────────────────────────────────────────────────────────────────
    col_seed, col_rand = st.columns([3, 1])
    with col_seed:
        params["seed"] = st.number_input(
            cfg.GEN_SEED_LABEL,
            min_value=0,
            max_value=2_147_483_647,
            value=int(st.session_state.get("seed", 1)),
            step=1,
            help=cfg.GEN_SEED_HELP,
            key=f"{key_prefix}_param_seed",
        )
    with col_rand:
        params["use_random_seed"] = st.checkbox(
            "Random",
            value=bool(st.session_state.get("use_random_seed", False)),
            help="Generate a new random seed on each run.",
            key=f"{key_prefix}_param_random_seed",
        )

    # ── Dtype (Advanced) ────────────────────────────────────────────────────
    if advanced:
        dtype_labels = list(cfg.DTYPE_OPTIONS.values())
        dtype_keys = list(cfg.DTYPE_OPTIONS.keys())
        current_dtype = st.session_state.get("dtype_str", rec["dtype"])
        dtype_idx = dtype_keys.index(current_dtype) if current_dtype in dtype_keys else 0
        selected_dtype_label = st.selectbox(
            cfg.GEN_DTYPE_LABEL,
            options=dtype_labels,
            index=dtype_idx,
            help=cfg.GEN_DTYPE_HELP,
            key=f"{key_prefix}_param_dtype",
        )
        params["dtype_str"] = dtype_keys[dtype_labels.index(selected_dtype_label)]
    else:
        params["dtype_str"] = st.session_state.get("dtype_str", rec["dtype"])

    # Warn if advanced params deviate from recommended
    if advanced and utils.params_deviate_from_recommended(
        model_name,
        params["num_steps"],
        params["guidance"],
        params["width"],
        params["height"],
    ):
        st.warning(cfg.WARN_ADVANCED_PARAMS)

    active_preset = state.infer_quality_preset(
        model_name,
        num_steps=int(params["num_steps"]),
        guidance=float(params["guidance"]),
    )
    st.session_state.quality_preset = active_preset

    return params


# ─── Hardware Panel ───────────────────────────────────────────────────────────

def hardware_panel(advanced: bool, key_prefix: str = "gen") -> dict[str, Any]:
    """Render hardware-optimisation widgets (offload, attention slicing)."""
    hw: dict[str, Any] = {}
    device_info = utils.get_device_info()

    st.subheader("Hardware")
    st.caption(f"🖥️ {device_info['name']}")
    if device_info["cuda"]:
        st.caption(f"💾 {device_info['vram_label']}")
    else:
        st.warning(cfg.ERR_NO_CUDA)

    if advanced:
        hw["cpu_offloading"] = st.checkbox(
            cfg.GEN_CPU_OFFLOAD_LABEL,
            value=bool(st.session_state.get("cpu_offloading", False)),
            help=cfg.GEN_CPU_OFFLOAD_HELP,
            key=f"{key_prefix}_hw_cpu_offload",
        )
        hw["attn_slicing"] = st.checkbox(
            cfg.GEN_ATTN_SLICING_LABEL,
            value=bool(st.session_state.get("attn_slicing", True)),
            help=cfg.GEN_ATTN_SLICING_HELP,
            key=f"{key_prefix}_hw_attn_slicing",
        )
    else:
        hw["cpu_offloading"] = st.session_state.get("cpu_offloading", False)
        hw["attn_slicing"] = st.session_state.get("attn_slicing", True)

    return hw


# ─── Prompt Input ─────────────────────────────────────────────────────────────

def prompt_input(key: str = "prompt_input") -> str:
    """Render the prompt text area with style presets and example loader."""
    # Example prompt loader
    example_options = ["— Select an example —"] + cfg.EXAMPLE_PROMPTS
    selected_example = st.selectbox(
        cfg.GEN_EXAMPLE_LABEL,
        options=example_options,
        index=0,
        key=f"{key}_example",
    )

    # Style preset append
    preset_options = ["— None —"] + list(cfg.PROMPT_STYLE_PRESETS.keys())
    selected_preset = st.selectbox(
        cfg.GEN_STYLE_PRESET_LABEL,
        options=preset_options,
        index=0,
        help=cfg.GEN_STYLE_PRESET_HELP,
        key=f"{key}_preset",
    )

    # Determine initial value
    initial = st.session_state.get(key, st.session_state.get("prompt", ""))

    # Apply example/preset selection changes into the actual text area state.
    # Streamlit keeps widget state by `key`, so `value=` alone is not enough after first render.
    last_example_key = f"{key}_last_example"
    last_preset_key = f"{key}_last_preset"
    last_example = st.session_state.get(last_example_key, "— Select an example —")
    last_preset = st.session_state.get(last_preset_key, "— None —")

    if selected_example != last_example and selected_example != "— Select an example —":
        initial = selected_example

    if selected_preset != last_preset and selected_preset != "— None —":
        suffix = cfg.PROMPT_STYLE_PRESETS[selected_preset]
        if suffix and suffix not in initial:
            initial = f"{initial.rstrip(', ')} {suffix}" if initial else suffix

    if selected_example != last_example or selected_preset != last_preset:
        st.session_state[key] = initial
        st.session_state[last_example_key] = selected_example
        st.session_state[last_preset_key] = selected_preset

    prompt = st.text_area(
        cfg.GEN_PROMPT_LABEL,
        value=initial,
        height=150,
        placeholder=cfg.GEN_PROMPT_PLACEHOLDER,
        help=cfg.GEN_PROMPT_HELP,
        key=key,
    )

    prompt_error = utils.validate_prompt(prompt)
    if prompt_error:
        st.warning(prompt_error)

    # Token count hint
    word_count = len(prompt.split()) if prompt else 0
    st.caption(f"~{word_count} words")

    return prompt


# ─── Image Result Panel ───────────────────────────────────────────────────────

def image_result_panel(
    img: Image.Image,
    metadata: dict[str, Any],
    generation_time_s: float,
    output_path: str,
    prompt: str | None = None,
) -> None:
    """Render the generated image with metadata, download, and history."""
    st.success(f"{cfg.STATUS_DONE} ({generation_time_s:.1f} s)  |  Saved: `{output_path}`")
    st.markdown("#### Preview")
    prompt_text = (prompt or str(metadata.get("prompt", ""))).strip()
    preview_caption = f"Seed {metadata.get('seed', 'N/A')} · {prompt_text[:120]}" if prompt_text else f"Seed {metadata.get('seed', 'N/A')}"
    st.image(img, use_container_width=True, caption=preview_caption)

    st.markdown("#### Key Metadata")
    metadata_cols = st.columns(4)
    with metadata_cols[0]:
        st.metric("Model", str(metadata.get("model", "N/A")))
        st.metric("Seed", str(metadata.get("seed", "N/A")))
    with metadata_cols[1]:
        st.metric("Steps", str(metadata.get("num_steps", "N/A")))
        st.metric("Guidance", str(metadata.get("guidance", "N/A")))
    with metadata_cols[2]:
        resolution = f"{metadata.get('width', '?')} × {metadata.get('height', '?')}"
        st.metric("Resolution", resolution)
        st.metric("Dtype", str(metadata.get("dtype", "N/A")))
    with metadata_cols[3]:
        st.metric("Runtime", f"{generation_time_s:.2f}s")
        st.metric("Output", Path(output_path).name)

    safety_prompt = metadata.get("safety_prompt_status")
    safety_output = metadata.get("safety_output_status")
    if safety_prompt is not None or safety_output is not None:
        safety_parts = []
        if safety_prompt is not None:
            safety_parts.append(f"Prompt: {safety_prompt}")
        if safety_output is not None:
            safety_parts.append(f"Output: {safety_output}")
        st.caption(f"Safety status — {' | '.join(safety_parts)}")

    st.caption(f"Output path: {output_path}")

    with st.expander("Generation Details", expanded=False):
        st.markdown(utils.format_metadata_display(metadata))

    st.markdown("#### Actions")
    col_copy_prompt, col_copy_settings = st.columns([1, 1])
    with col_copy_prompt:
        if st.button("Copy Prompt", use_container_width=True, key="result_copy_prompt"):
            st.session_state.result_copy_prompt = prompt or str(metadata.get("prompt", ""))
            st.success("Prompt ready to copy.")
        if st.session_state.get("result_copy_prompt"):
            st.code(st.session_state.get("result_copy_prompt", ""), language="markdown")

    with col_copy_settings:
        if st.button("Copy Settings", use_container_width=True, key="result_copy_settings"):
            st.session_state.result_copy_settings = utils.build_settings_copy_text(metadata)
            st.success("Settings ready to copy.")
        if st.session_state.get("result_copy_settings"):
            st.code(st.session_state.get("result_copy_settings", ""), language="json")

    col_dl, col_report, col_ref = st.columns([1, 1, 1])
    with col_dl:
        img_bytes = utils.pil_to_bytes(img)
        st.download_button(
            label=cfg.GEN_DOWNLOAD_LABEL,
            data=img_bytes,
            file_name=Path(output_path).name,
            mime="image/png",
            type="primary",
            key="result_download_image",
        )

    with col_report:
        report_payload = utils.build_result_report(metadata, output_path=output_path, prompt=prompt)
        st.download_button(
            label="Download Report",
            data=json.dumps(report_payload, indent=2),
            file_name=f"result_report_{Path(output_path).stem}.json",
            mime="application/json",
            key="result_download_report",
        )

    with col_ref:
        if st.button(cfg.GEN_SEND_TO_EDITOR_LABEL, use_container_width=True, key="result_send_to_editor"):
            st.session_state.reference_images = [img]
            st.session_state.editor_prompt = prompt or ""
            st.session_state.match_image_size = 0
            st.success("Reference image loaded into the Image Editor tab.")

    history = st.session_state.get("generation_history", [])
    compare_candidates = utils.build_history_compare_candidates(history)
    with st.expander("Compare (Last vs Selected)", expanded=False):
        if not compare_candidates:
            st.caption("Generate at least two results to enable comparison.")
        else:
            selected_idx = st.selectbox(
                "Compare latest result against",
                options=list(range(len(compare_candidates))),
                format_func=lambda idx: compare_candidates[idx]["label"],
                key="result_compare_target",
            )

            selected_entry = compare_candidates[selected_idx]["entry"]
            selected_meta = selected_entry.get("metadata", {}) if isinstance(selected_entry, dict) else {}
            selected_image = selected_entry.get("image_bytes") if isinstance(selected_entry, dict) else None

            image_comparison(
                left_image=selected_image,
                right_image=img,
                left_label="Selected history",
                right_label="Latest result",
                key_prefix="result_compare",
            )

            meta_col_l, meta_col_r = st.columns(2)
            with meta_col_l:
                st.markdown("**Selected Metadata**")
                st.markdown(utils.format_metadata_display(selected_meta if isinstance(selected_meta, dict) else {}))
            with meta_col_r:
                st.markdown("**Latest Metadata**")
                st.markdown(utils.format_metadata_display(metadata))


# ─── Help Expanders ───────────────────────────────────────────────────────────

def help_prompt_tips() -> None:
    with st.expander("Prompt Writing Tips"):
        st.markdown(cfg.HELP_PROMPT_TIPS)


def help_model_comparison() -> None:
    with st.expander("Model Comparison"):
        st.markdown(cfg.HELP_MODEL_COMPARISON)


def help_upsampling() -> None:
    with st.expander("About Prompt Upsampling"):
        st.markdown(cfg.HELP_UPSAMPLING)


def help_image_editor() -> None:
    with st.expander("Image Editor Tips"):
        st.markdown(cfg.HELP_IMAGE_EDITOR)


def help_troubleshooting() -> None:
    with st.expander("Troubleshooting"):
        st.markdown(cfg.HELP_TROUBLESHOOTING)


# ─── Upload helper ───────────────────────────────────────────────────────────

def image_uploader(
    label: str,
    key: str,
    help_text: str = "",
    accept_multiple: bool = False,
) -> list[Image.Image]:
    """Render an image upload widget and return a list of PIL Images."""
    uploaded = st.file_uploader(
        label,
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=accept_multiple,
        help=help_text,
        key=key,
    )
    if uploaded is None:
        return []
    if not accept_multiple:
        uploaded = [uploaded]
    images = []
    for f in uploaded:
        try:
            images.append(Image.open(io.BytesIO(f.read())).convert("RGB"))
        except Exception:  # noqa: BLE001
            st.warning(f"Could not load '{f.name}'. Skipping.")
    return images
