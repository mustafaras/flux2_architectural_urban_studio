"""Prompt Upsampler page — expand short prompts using a language model."""

from __future__ import annotations

import streamlit as st

from ui import config as cfg
from ui import icons
from ui import state
from ui.components import help_upsampling, image_uploader


def render() -> None:
    """Entry point called from the main app to render the Prompt Upsampler tab."""
    _render_layout(show_header=True, show_expanders=True)


def render_embedded() -> None:
    """Render upsampler UI embedded in another page (e.g., Generator)."""
    _render_layout(show_header=False, show_expanders=False)


def _render_layout(show_header: bool, show_expanders: bool = True) -> None:
    if show_header:
        icons.heading("Prompt Upsampler", icon="wand")
    else:
        icons.heading("Prompt Upsampler", icon="wand", level=3)

    st.caption(
        "Automatically expand a short or rough prompt into a detailed, well-structured "
        "description that gets better results from FLUX.2 models."
    )

    if show_expanders:
        help_upsampling()
    st.divider()

    col_input, col_output = st.columns(2, gap="large")

    with col_input:
        _render_input_column()

    with col_output:
        _render_output_column(show_expanders=show_expanders)


# ─── Private helpers ──────────────────────────────────────────────────────────

def _render_input_column() -> None:
    st.subheader("Input")

    original = st.text_area(
        cfg.UPS_INPUT_LABEL,
        value=st.session_state.get("upsample_original_prompt", ""),
        height=150,
        placeholder=cfg.UPS_INPUT_PLACEHOLDER,
        key="ups_input_prompt",
    )
    st.session_state.upsample_original_prompt = original

    backend_labels = list(cfg.UPSAMPLE_BACKENDS.values())
    backend_keys = list(cfg.UPSAMPLE_BACKENDS.keys())
    current_backend = st.session_state.get("upsample_backend", "none")
    try:
        backend_idx = backend_keys.index(current_backend)
    except ValueError:
        backend_idx = 0

    selected_label = st.selectbox(
        cfg.UPS_BACKEND_LABEL,
        options=backend_labels,
        index=backend_idx,
        key="ups_backend_selector",
    )
    backend = backend_keys[backend_labels.index(selected_label)]
    st.session_state.upsample_backend = backend

    st.caption(cfg.UPSAMPLE_BACKEND_HINTS.get(backend, ""))

    reference_images = image_uploader(
        label=cfg.UPS_REFERENCE_HINT,
        key="ups_ref_images",
        help_text="Attach images if you want upsampling to consider visual context.",
        accept_multiple=True,
    )
    st.session_state.upsample_reference_images = reference_images

    if backend == "openrouter":
        _render_openrouter_config()
    elif backend == "local":
        _render_ollama_config()

    st.divider()

    disabled = not original.strip()
    btn_expand, btn_compare = st.columns(2)
    with btn_expand:
        if st.button(cfg.UPS_BUTTON_LABEL, type="primary", disabled=disabled, use_container_width=True):
            st.session_state["_trigger_upsample"] = True
    with btn_compare:
        if st.button("Compare Fast vs Quality", disabled=disabled, use_container_width=True):
            st.session_state["_trigger_upsample_compare"] = True

    _run_upsampling_if_triggered(original, backend, reference_images)
    _run_compare_if_triggered(original, backend, reference_images)


def _render_openrouter_config() -> None:
    api_key = st.session_state.get("openrouter_api_key", "")
    if not api_key:
        st.warning(cfg.ERR_API_KEY_MISSING)

    selected_model = st.selectbox(
        cfg.UPS_MODEL_LABEL,
        options=cfg.OPENROUTER_MODEL_OPTIONS,
        index=0,
        key="ups_openrouter_model",
    )
    st.session_state.openrouter_model = selected_model


def _render_ollama_config() -> None:
    st.caption("Using local Ollama backend")

    profile_keys = list(cfg.UPS_OLLAMA_PROFILES.keys())
    profile_labels = [cfg.UPS_OLLAMA_PROFILES[k]["label"] for k in profile_keys]
    current_profile = st.session_state.get("ollama_profile", "quality")
    if current_profile not in profile_keys:
        current_profile = "quality"

    selected_profile_label = st.selectbox(
        cfg.UPS_OLLAMA_PROFILE_LABEL,
        options=profile_labels,
        index=profile_keys.index(current_profile),
        key="ups_ollama_profile",
    )
    selected_profile = profile_keys[profile_labels.index(selected_profile_label)]
    st.session_state.ollama_profile = selected_profile

    default_model_for_profile = cfg.UPS_OLLAMA_PROFILES[selected_profile]["default_model"]
    st.session_state.ollama_temperature = float(cfg.UPS_OLLAMA_PROFILES[selected_profile]["temperature"])

    models: list[str] = []
    try:
        from src.flux2.streamlit_adapter import get_adapter
        models = get_adapter().list_ollama_models()
    except Exception:
        models = []

    current_model = st.session_state.get("ollama_model", default_model_for_profile)
    if models:
        if default_model_for_profile in models:
            current_model = default_model_for_profile
        if current_model not in models:
            current_model = models[0]
        selected_model = st.selectbox(
            cfg.UPS_OLLAMA_MODEL_LABEL,
            options=models,
            index=models.index(current_model),
            help=cfg.UPS_OLLAMA_MODEL_HELP,
            key="ups_ollama_model",
        )
        st.session_state.ollama_model = selected_model
    else:
        manual = st.text_input(
            cfg.UPS_OLLAMA_MODEL_LABEL,
            value=default_model_for_profile,
            help="No Ollama model list received. Enter model tag manually (e.g. qwen3:30b).",
            key="ups_ollama_model_manual",
        )
        st.session_state.ollama_model = manual.strip()


def _run_upsampling_if_triggered(original: str, backend: str, reference_images: list) -> None:
    if not st.session_state.get("_trigger_upsample"):
        return
    st.session_state["_trigger_upsample"] = False

    if not original.strip():
        st.error("Please enter a prompt to expand.")
        return

    if backend == "none":
        st.info("Select a backend (Local or OpenRouter) to enable upsampling.")
        return

    if backend == "openrouter":
        api_key = st.session_state.get("openrouter_api_key", "")
        if not api_key:
            st.error(cfg.ERR_API_KEY_MISSING)
            return

    with st.spinner("Expanding prompt..."):
        try:
            from src.flux2.streamlit_adapter import get_adapter
            adapter = get_adapter()
            result = adapter.upsample_prompt(
                prompt=original,
                backend=backend,
                ollama_model=st.session_state.get("ollama_model", "qwen3:30b"),
                ollama_temperature=float(st.session_state.get("ollama_temperature", 0.15)),
                openrouter_model=st.session_state.get("openrouter_model", "mistralai/pixtral-large-2411"),
                api_key=st.session_state.get("openrouter_api_key", ""),
                reference_images=reference_images,
            )
            st.session_state.upsample_result_prompt = result
        except Exception as exc:  # noqa: BLE001
            st.error(f"Upsampling failed: {exc}")
            st.session_state.upsample_result_prompt = original


def _run_compare_if_triggered(original: str, backend: str, reference_images: list) -> None:
    if not st.session_state.get("_trigger_upsample_compare"):
        return
    st.session_state["_trigger_upsample_compare"] = False

    if not original.strip():
        st.error("Please enter a prompt to compare.")
        return

    if backend != "local":
        st.info("Fast vs Quality comparison is available for Local (Ollama) backend.")
        return

    try:
        from src.flux2.streamlit_adapter import get_adapter

        adapter = get_adapter()
        profiles = cfg.UPS_OLLAMA_PROFILES

        with st.spinner("Comparing Fast vs Quality profiles..."):
            fast = adapter.upsample_prompt(
                prompt=original,
                backend="local",
                ollama_model=profiles["fast"]["default_model"],
                ollama_temperature=float(profiles["fast"]["temperature"]),
                reference_images=reference_images,
            )
            quality = adapter.upsample_prompt(
                prompt=original,
                backend="local",
                ollama_model=profiles["quality"]["default_model"],
                ollama_temperature=float(profiles["quality"]["temperature"]),
                reference_images=reference_images,
            )

        st.session_state.upsample_compare_original = original
        st.session_state.upsample_compare_fast = fast
        st.session_state.upsample_compare_quality = quality
    except Exception as exc:  # noqa: BLE001
        st.error(f"Comparison failed: {exc}")


def _render_output_column(show_expanders: bool = True) -> None:
    st.subheader("Expanded Prompt")

    result = st.session_state.get("upsample_result_prompt", "")
    compare_fast = st.session_state.get("upsample_compare_fast", "")
    compare_quality = st.session_state.get("upsample_compare_quality", "")
    compare_original = st.session_state.get("upsample_compare_original", "")

    if result:
        edited_result = st.text_area(
            cfg.UPS_RESULT_LABEL,
            value=result,
            height=200,
            key="ups_result_text",
            help="You can edit this prompt before applying it to the generator.",
        )

        if st.button(cfg.UPS_APPLY_BUTTON_LABEL, type="primary", use_container_width=True):
            st.session_state.upsample_result_prompt = edited_result
            st.session_state["_pending_prompt_for_generator"] = edited_result
            st.rerun()

        original = st.session_state.get("upsample_original_prompt", "")
        if original and result and original.strip() != result.strip() and show_expanders:
            with st.expander(cfg.UPS_COMPARISON_HEADER):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**Before**")
                    st.info(original)
                with col_b:
                    st.markdown("**After**")
                    st.success(result)
    else:
        st.info("Enter a prompt on the left and click **Expand Prompt** to see the result here.")

    if compare_fast and compare_quality:
        st.divider()
        st.subheader("Fast vs Quality (Preview)")
        if compare_original:
            st.caption(f"Original: {compare_original}")

        col_fast, col_quality = st.columns(2)
        with col_fast:
            st.markdown("**Fast**")
            fast_text = st.text_area(
                "Fast result",
                value=compare_fast,
                height=180,
                key="ups_compare_fast_text",
            )
            if st.button("Apply Fast", use_container_width=True, key="ups_apply_fast"):
                st.session_state.upsample_result_prompt = fast_text
                st.session_state["_pending_prompt_for_generator"] = fast_text
                st.rerun()

        with col_quality:
            st.markdown("**Quality**")
            quality_text = st.text_area(
                "Quality result",
                value=compare_quality,
                height=180,
                key="ups_compare_quality_text",
            )
            if st.button("Apply Quality", use_container_width=True, key="ups_apply_quality"):
                st.session_state.upsample_result_prompt = quality_text
                st.session_state["_pending_prompt_for_generator"] = quality_text
                st.rerun()
