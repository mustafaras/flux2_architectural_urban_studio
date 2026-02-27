"""Settings & Help page."""

from __future__ import annotations

import json
import os

import streamlit as st

from src.flux2.deployment_profiles import get_profile, list_profiles, validate_startup_health_prereqs
from src.flux2.feature_flags import get_feature_flags, set_feature_flag
from src.flux2.logging_config import log_operation
from src.flux2.policy_profiles import get_policy_profile, list_policy_profiles
from src.flux2.rollout_playbooks import evaluate_go_no_go, get_rollout_playbook, list_rollout_stages
from ui import error_handler
from ui import config as cfg
from ui import icons
from ui.components import (
    help_image_editor,
    help_model_comparison,
    help_prompt_tips,
    help_troubleshooting,
    help_upsampling,
)


def render() -> None:
    """Entry point called from the main app to render the Settings & Help tab."""
    icons.heading("Settings & Help", icon="settings")

    _render_output_section()
    _render_api_keys_section()
    _render_model_paths_section()
    _render_custom_models_section()
    _render_safety_section()
    _render_deployment_hardening_section()
    _render_phase5_policy_and_rollout_section()
    _render_session_tools_section()
    _render_debug_section()
    _render_help_section()
    _render_about_section()


# ─── Sections ─────────────────────────────────────────────────────────────────

def _render_output_section() -> None:
    with st.expander(cfg.SETTINGS_OUTPUT_SECTION, expanded=True):
        out_dir = st.text_input(
            cfg.GEN_OUTPUT_DIR_LABEL,
            value=st.session_state.get("output_dir", "outputs"),
            help=cfg.GEN_OUTPUT_DIR_HELP,
            key="settings_output_dir",
        )
        err = None
        from ui.utils import validate_output_dir
        err = validate_output_dir(out_dir)
        if err:
            st.warning(err)
        else:
            st.session_state.output_dir = out_dir
            st.caption(f"Images will be saved to: `{out_dir}`")


def _render_api_keys_section() -> None:
    with st.expander(cfg.SETTINGS_API_SECTION):
        api_key = st.text_input(
            cfg.SETTINGS_OPENROUTER_KEY_LABEL,
            value=st.session_state.get("openrouter_api_key", ""),
            type="password",
            help=cfg.SETTINGS_OPENROUTER_KEY_HELP,
            key="settings_openrouter_key",
            placeholder="sk-or-...",
        )
        if api_key:
            st.session_state.openrouter_api_key = api_key
            # Also set as env var so existing client code picks it up
            os.environ["OPENROUTER_API_KEY"] = api_key
            st.success("OpenRouter API key saved for this session.")
        st.caption(
            "Your API key is stored only for the current session and is never written to disk."
        )


def _render_model_paths_section() -> None:
    with st.expander(cfg.SETTINGS_MODEL_PATHS_SECTION):
        st.markdown(
            "Set local model file paths for this repository. "
            "Local-only mode is enabled by default; automatic downloads are disabled."
        )
        st.caption(
            "Tip: Keep weights under `weights/` in this repo and point each path here."
        )
        st.caption(
            "If these default filenames exist, they are detected automatically: "
            "`weights/flux-2-klein-4b.safetensors`, `weights/flux-2-klein-9b.safetensors`, "
            "`weights/flux-2-klein-base-4b.safetensors`, `weights/flux-2-klein-base-9b.safetensors`, "
            "`weights/flux2-dev.safetensors`, `weights/ae.safetensors`."
        )

        paths: dict[str, str] = {
            "KLEIN_4B_MODEL_PATH": "Klein 4B weights (.safetensors)",
            "KLEIN_9B_MODEL_PATH": "Klein 9B weights (.safetensors)",
            "KLEIN_4B_BASE_MODEL_PATH": "Klein Base 4B weights (.safetensors)",
            "KLEIN_9B_BASE_MODEL_PATH": "Klein Base 9B weights (.safetensors)",
            "FLUX2_MODEL_PATH": "FLUX.2-Dev weights (.safetensors)",
            "AE_MODEL_PATH": "Autoencoder (ae.safetensors)",
        }

        for env_key, label in paths.items():
            current = os.environ.get(env_key, "")
            val = st.text_input(
                label,
                value=current,
                key=f"settings_path_{env_key}",
                placeholder="Set a local file path",
            )
            if val and val != current:
                os.environ[env_key] = val
                st.success(f"`{env_key}` updated.")
            if os.environ.get(env_key, ""):
                if os.path.exists(os.environ[env_key]):
                    st.caption(f"{env_key} file found")
                else:
                    st.caption(f"{env_key} file not found")


def _render_custom_models_section() -> None:
    """Allow users to register custom models and LoRA weights"""
    from src.flux2.model_registry import get_model_registry, ModelType
    
    with st.expander(icons.label("Custom Model Management", "robot")):
        st.markdown("**Register Custom Models & LoRA Weights**")
        st.caption("Add your own fine-tuned models, LoRA adapters, or custom FLUX variants.")
        
        registry = get_model_registry()
        
        # Upload new model
        st.markdown("##### Upload New Model")
        
        col1, col2 = st.columns(2)
        
        with col1:
            model_name = st.text_input(
                "Model Name",
                placeholder="My Custom Model",
                key="custom_model_name"
            )
            
            model_type = st.selectbox(
                "Model Type",
                options=[t.value for t in ModelType],
                key="custom_model_type"
            )
        
        with col2:
            model_file = st.file_uploader(
                "Model File",
                type=["safetensors", "pt", "bin"],
                key="custom_model_file"
            )
            
            description = st.text_area(
                "Description (optional)",
                placeholder="Trained on anime art...",
                key="custom_model_desc",
                height=80
            )
        
        if st.button(icons.label("Register Model", "rocket"), use_container_width=True, type="primary"):
            if not model_name:
                st.error("Please provide a model name")
            elif not model_file:
                st.error("Please upload a model file")
            else:
                # Save uploaded file
                from pathlib import Path
                import tempfile
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".safetensors") as tmp:
                    tmp.write(model_file.getvalue())
                    tmp_path = Path(tmp.name)
                
                # Register the model
                model_key = registry.add_custom_model(
                    name=model_name,
                    file_path=tmp_path,
                    model_type=ModelType(model_type),
                    description=description
                )
                
                if model_key:
                    st.success(f"{icons.label('Model registered', 'check')}: {model_key}")
                else:
                    st.error("Failed to register model")
        
        # List registered models
        st.divider()
        st.markdown("##### Registered Models")
        
        stats = registry.get_statistics()
        
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        with col_stat1:
            st.metric("Total Models", stats['total_models'])
        with col_stat2:
            st.metric("Verified", stats['verified_models'])
        with col_stat3:
            st.metric("Total Size", f"{stats['total_size_gb']:.1f} GB")
        
        # Show models in a table
        models = registry.list_available()
        
        if models:
            for model in models:
                with st.container(border=True):
                    col_a, col_b = st.columns([3, 1])
                    
                    with col_a:
                        st.markdown(f"**{model.name}**")
                        st.caption(f"Type: {model.model_type.value} | Status: {model.status.value}")
                        if model.description:
                            st.caption(icons.label(model.description, "note"))
                    
                    with col_b:
                        st.caption(f"⭐ {model.performance_score:.1f}")
                        st.caption(icons.label(f"{model.file_size_mb:.0f} MB", "save"))
                        
                        # Find model key
                        model_key = None
                        for key, m in registry.registry.items():
                            if m == model:
                                model_key = key
                                break
                        
                        if model_key and model.model_type in [ModelType.CUSTOM, ModelType.LORA]:
                            if st.button(icons.label("Remove", "trash"), key=f"remove_{model_key}", use_container_width=True):
                                if registry.remove_model(model_key, delete_file=True):
                                    st.success(f"Removed {model.name}")
                                    st.rerun()
        else:
            st.info("No custom models registered yet.")


def _render_safety_section() -> None:
    with st.expander(cfg.SETTINGS_SAFETY_SECTION):
        nsfw_threshold = st.slider(
            cfg.SETTINGS_NSFW_THRESHOLD_LABEL,
            min_value=0.0,
            max_value=1.0,
            value=float(st.session_state.get("nsfw_threshold", 0.85)),
            step=0.05,
            help=cfg.SETTINGS_NSFW_THRESHOLD_HELP,
            key="settings_nsfw_threshold",
        )
        st.session_state.nsfw_threshold = nsfw_threshold

        safety_prompt = st.checkbox(
            cfg.SETTINGS_SAFETY_PROMPT_LABEL,
            value=bool(st.session_state.get("safety_check_prompt", False)),
            help="Block generation early when a prompt fails safety checks.",
            key="settings_safety_prompt",
        )
        st.session_state.safety_check_prompt = safety_prompt

        safety_image = st.checkbox(
            cfg.SETTINGS_SAFETY_IMAGE_LABEL,
            value=bool(st.session_state.get("safety_check_output", False)),
            help="Check generated or reference images for safety policy violations.",
            key="settings_safety_image",
        )
        st.session_state.safety_check_output = safety_image


def _render_deployment_hardening_section() -> None:
    with st.expander("Deployment Hardening Profiles"):
        st.caption("Define deployment profile, validate startup health checks, and keep rollback/version pinning explicit.")

        profiles = list_profiles()
        selected_profile = st.selectbox(
            "Environment Profile",
            options=profiles,
            key="settings_deployment_profile",
            format_func=lambda x: x.replace("_", " ").title(),
        )
        profile = get_profile(selected_profile)

        st.write({
            "profile": profile.get("name", selected_profile),
            "health_endpoints": profile.get("health_endpoints", []),
            "version_pinning": profile.get("version_pinning", ""),
        })

        checks = validate_startup_health_prereqs()
        st.markdown("**Startup Health Check Validation**")
        st.write(checks)

        st.markdown("**Graceful Degradation Paths**")
        for item in profile.get("graceful_degradation", []):
            st.markdown(f"- {item}")

        st.markdown("**Rollback Procedure**")
        for idx, item in enumerate(profile.get("rollback", []), 1):
            st.markdown(f"{idx}. {item}")

        if st.button("Apply Deployment Profile", key="settings_apply_deployment_profile", use_container_width=True):
            st.session_state.deployment_profile = selected_profile
            log_operation(
                "deployment-profile-update",
                "success",
                {
                    "profile": selected_profile,
                    "health_checks": checks,
                },
            )
            st.success(f"Deployment profile applied: {selected_profile}")


def _render_phase5_policy_and_rollout_section() -> None:
    with st.expander("Policy Profiles, Feature Flags, and Rollout"):
        st.caption("Configure policy constraints, staged feature rollout, and KPI go/no-go gates.")

        profile_keys = list_policy_profiles()
        current_profile = str(st.session_state.get("current_policy_profile", "commercial"))
        profile = st.selectbox(
            "Policy Profile",
            options=profile_keys,
            index=profile_keys.index(current_profile) if current_profile in profile_keys else 0,
            key="settings_policy_profile",
        )
        st.session_state.current_policy_profile = profile
        st.write(get_policy_profile(profile))

        st.markdown("**Feature Flags**")
        flags = get_feature_flags(st.session_state)
        for key in sorted(flags.keys()):
            enabled = st.checkbox(key, value=bool(flags.get(key, False)), key=f"flag_{key}")
            set_feature_flag(st.session_state, key, enabled)

        if st.button("Save Feature Flag Configuration", key="settings_save_flags", use_container_width=True):
            log_operation("feature-flags-update", "success", {"flags": get_feature_flags(st.session_state)})
            st.success("Feature flags updated.")

        st.markdown("**Rollout Playbooks**")
        stages = list_rollout_stages()
        rollout_stage = st.selectbox(
            "Rollout Stage",
            options=stages,
            index=stages.index(str(st.session_state.get("rollout_stage", "pilot"))) if str(st.session_state.get("rollout_stage", "pilot")) in stages else 0,
            key="settings_rollout_stage",
        )
        st.session_state.rollout_stage = rollout_stage

        playbook = get_rollout_playbook(rollout_stage)
        st.write(playbook)

        queue = st.session_state.get("generation_queue")
        queue_status = queue.get_status() if queue is not None else {}
        if not isinstance(queue_status, dict):
            queue_status = {}
        success_rate = float(queue_status.get("success_rate", 0.0) or 0.0)
        latencies = st.session_state.get("kpi_generation_latencies", [])
        latency_values = [float(x) for x in latencies if isinstance(x, (int, float))]
        latency_values.sort()
        p95_index = int(0.95 * (len(latency_values) - 1)) if latency_values else 0
        p95_latency = latency_values[p95_index] if latency_values else 0.0
        backlog = int(queue_status.get("queued", 0) or 0)

        gate = evaluate_go_no_go(
            rollout_stage,
            success_rate=success_rate,
            p95_latency_s=p95_latency,
            queue_backlog=backlog,
        )
        st.write({"kpi_gate": gate})


def _render_session_tools_section() -> None:
    with st.expander("Session Tools"):
        snapshot = error_handler.load_recovery_snapshot()
        if snapshot:
            st.warning("A recoverable generation snapshot was found from a previous failed run.")
            if st.button("Restore recovery snapshot to session", use_container_width=True):
                for key, value in snapshot.items():
                    if key in {"mode", "reference_count"}:
                        continue
                    st.session_state[key] = value
                st.success("Recovery snapshot restored.")
            if st.button("Clear recovery snapshot", use_container_width=True):
                error_handler.clear_recovery_snapshot()
                st.success("Recovery snapshot cleared.")

        if st.button("Clear generation history", use_container_width=True):
            st.session_state.generation_history = []
            st.success("Generation history cleared.")

        if st.button("Clear error history", use_container_width=True):
            error_handler.clear_error_history()
            st.success("Error history cleared.")

        exported = {
            "model_name": st.session_state.get("model_name"),
            "dtype_str": st.session_state.get("dtype_str"),
            "num_steps": st.session_state.get("num_steps"),
            "guidance": st.session_state.get("guidance"),
            "width": st.session_state.get("width"),
            "height": st.session_state.get("height"),
            "advanced_mode": st.session_state.get("advanced_mode"),
            "output_dir": st.session_state.get("output_dir"),
            "safety_check_prompt": st.session_state.get("safety_check_prompt"),
            "safety_check_output": st.session_state.get("safety_check_output"),
            "nsfw_threshold": st.session_state.get("nsfw_threshold"),
            "upsample_backend": st.session_state.get("upsample_backend"),
            "openrouter_model": st.session_state.get("openrouter_model"),
        }
        st.download_button(
            label="Download session settings (JSON)",
            data=json.dumps(exported, indent=2),
            file_name="flux2_ui_session_settings.json",
            mime="application/json",
            use_container_width=True,
        )

        st.download_button(
            label="Download quick-reference cheat sheet (Markdown)",
            data=cfg.QUICK_REFERENCE_MD,
            file_name="flux2_ui_quick_reference.md",
            mime="text/markdown",
            use_container_width=True,
        )

        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.pdfgen import canvas

            import io

            pdf_buffer = io.BytesIO()
            p = canvas.Canvas(pdf_buffer, pagesize=A4)
            p.setFont("Helvetica-Bold", 14)
            p.drawString(48, 800, "FLUX.2 UI Quick Reference")
            p.setFont("Helvetica", 10)
            y = 776
            for line in cfg.QUICK_REFERENCE_MD.splitlines():
                text = line.replace("#", "").replace("*", "")
                if not text.strip():
                    y -= 8
                    continue
                p.drawString(48, y, text[:110])
                y -= 14
                if y < 48:
                    p.showPage()
                    p.setFont("Helvetica", 10)
                    y = 800
            p.save()
            pdf_buffer.seek(0)
            st.download_button(
                label="Download quick-reference cheat sheet (PDF)",
                data=pdf_buffer.getvalue(),
                file_name="flux2_ui_quick_reference.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception:
            st.caption("PDF export requires `reportlab` (optional). Markdown cheat sheet is always available.")


def _render_debug_section() -> None:
    with st.container():
        from ui.pages import debug

        debug.render()


def _render_help_section() -> None:
    icons.heading("Help & Documentation", icon="book", level=3)
    help_prompt_tips()
    help_model_comparison()
    help_upsampling()
    help_image_editor()
    help_troubleshooting()


def _render_about_section() -> None:
    with st.expander(cfg.SETTINGS_ABOUT_SECTION):
        st.markdown(
            """
**FLUX.2 Professional Image Generator**

Built on top of [FLUX.2](https://github.com/black-forest-labs/flux) by Black Forest Labs.

| Component | Details |
|---|---|
| Models | FLUX.2-Klein 4B / 9B, FLUX.2-Dev |
| Text Encoders | Qwen3 (Klein), Mistral Small (Dev) |
| Upsampling | OpenRouter API, local Mistral-Small-3.2-24B |
| UI | Streamlit |

---

*This interface is intended for research and personal use in accordance with the  
[FLUX.2 model license](https://github.com/black-forest-labs/flux/blob/main/model_licenses/).*
            """
        )
