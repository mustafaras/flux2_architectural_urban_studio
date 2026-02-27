"""
FLUX.2 Professional Image Generator — main Streamlit entry point.

Run with:
    streamlit run ui_flux2_professional.py
"""

import os
import sys
from pathlib import Path

import streamlit as st

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("FLUX2_LOCAL_ONLY", "1")

_ROOT = Path(__file__).parent
_SRC = _ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_LOCAL_AE = _ROOT / "weights" / "ae.safetensors"
if ("AE_MODEL_PATH" not in os.environ or not Path(os.environ.get("AE_MODEL_PATH", "")).exists()) and _LOCAL_AE.exists():
    os.environ["AE_MODEL_PATH"] = str(_LOCAL_AE.resolve())


def _set_env_if_missing(env_key: str, rel_glob: str) -> None:
    current = os.environ.get(env_key, "").strip()
    if current and Path(current).exists():
        return
    matches = sorted(_ROOT.glob(rel_glob))
    if matches:
        os.environ[env_key] = str(matches[0].resolve())


_set_env_if_missing(
    "KLEIN_4B_MODEL_PATH",
    "weights/flux-2-klein-4b.safetensors",
)
_set_env_if_missing(
    "KLEIN_4B_BASE_MODEL_PATH",
    "weights/flux-2-klein-base-4b.safetensors",
)
_set_env_if_missing(
    "AE_MODEL_PATH",
    "weights/ae.safetensors",
)

# ── Page config (must be first Streamlit call) ────────────────────────────────
from ui import config as cfg
from ui import error_handler
from ui import icons
from ui.sidebar import render_sidebar
from ui.theme import apply_theme

st.set_page_config(
    page_title=cfg.PAGE_TITLE,
    page_icon=cfg.PAGE_ICON or None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state init ────────────────────────────────────────────────────────
from ui import state as _state
from ui.utils import configure_logging

configure_logging()
_state.init()
_state.sync_model_selection_state()

apply_theme()

snapshot = error_handler.load_recovery_snapshot()
if snapshot:
    with st.container(border=True):
        st.warning("A recovery snapshot from a previous failed run is available.")
        col_restore, col_clear = st.columns(2)
        with col_restore:
            if st.button("Restore last run", use_container_width=True, key="startup_restore_last_run"):
                if _state.restore_settings_from_snapshot(snapshot):
                    st.success("Last run settings restored.")
                else:
                    st.warning("Recovery snapshot could not be restored.")
        with col_clear:
            if st.button("Dismiss snapshot", use_container_width=True, key="startup_clear_snapshot"):
                error_handler.clear_recovery_snapshot()
                st.rerun()

# ── Sidebar ───────────────────────────────────────────────────────────────────
render_sidebar()

# ── Main content: tabbed interface ────────────────────────────────────────────
icons.title("FLUX.2 Architectural & Urban Studio", icon="sparkles")
st.caption(
    "An integrated computational design environment for architectural and urban exploration, "
    "structured to connect site intelligence, program intent, massing logic, façade articulation, and "
    "public-realm performance within a single, traceable visual workflow. "
    "The system operationalizes design intent through project-scoped metadata, parameterized generation controls, "
    "seed-consistent iteration, and scenario-based comparison so alternatives can be evaluated with greater "
    "reproducibility and analytical clarity. "
    "By coupling controlled image synthesis with workflow governance, queue-aware execution, and artifact-ready export, "
    "the platform supports rigorous option testing from early concept through communication phases. "
    "This approach reduces ambiguity in design exploration, improves stakeholder interpretability, and increases "
    "the fidelity, consistency, and decision value of architecture and urban design outputs."
)
st.divider()

use_architecture_workflow = bool(st.session_state.get("use_architecture_workflow", True))
show_legacy_navigation = bool(st.session_state.get("show_legacy_navigation", False))

if use_architecture_workflow and show_legacy_navigation:
    st.caption("Navigation Mode")
    navigation_mode = st.radio(
        "Navigation Mode",
        options=["Architecture Workflow", "Legacy Compatibility"],
        horizontal=True,
        key="navigation_mode_selector",
        label_visibility="collapsed",
    )
elif use_architecture_workflow:
    navigation_mode = "Architecture Workflow"
else:
    navigation_mode = "Legacy Compatibility"

if navigation_mode == "Architecture Workflow":
    workflow_tabs = st.tabs([
        icons.tab("Project Setup", "folder"),
        icons.tab("Site & Context", "image"),
        icons.tab("Massing & Form", "sparkles"),
        icons.tab("Façade & Materials", "edit"),
        icons.tab("Urban Scenario Studio", "queue"),
        icons.tab("Views & Rendering Controls", "activity"),
        icons.tab("Comparison Board", "scales"),
        icons.tab("Export & Presentation", "book"),
    ])

    with workflow_tabs[0]:
        from ui.pages import project_setup
        project_setup.render()

    with workflow_tabs[1]:
        from ui.pages import site_context
        site_context.render()

    with workflow_tabs[2]:
        st.caption("Workflow: 3/8 · Compose massing intent and generate concept options.")
        from ui.pages import generator
        generator.render()

    with workflow_tabs[3]:
        st.caption("Workflow: 4/8 · Refine façade language and material expression.")
        from ui.pages import editor
        editor.render()

    with workflow_tabs[4]:
        st.caption("Workflow: 5/8 · Create and manage urban scenario alternatives with lineage.")
        from ui.pages import urban_scenario_studio
        urban_scenario_studio.render()

    with workflow_tabs[5]:
        st.caption("Workflow: 6/8 · Monitor render progress, queue lanes, and consistency controls.")
        from ui.pages import queue
        queue.render()
        st.divider()
        from ui.pages import progress_monitor
        progress_monitor.render()

    with workflow_tabs[6]:
        st.caption("Workflow: 7/8 · Compare 2–4 options and shortlist review candidates.")
        from ui.pages import comparison_board
        comparison_board.render()

    with workflow_tabs[7]:
        st.caption("Workflow: 8/8 · Export presentation-ready boards and artifacts.")
        from ui.pages import export
        export.main()

if navigation_mode == "Legacy Compatibility":
    st.divider()
    st.caption("Legacy Compatibility Navigation")
    tab_gen, tab_edit, tab_queue, tab_progress, tab_comparison, tab_safety, tab_analytics, tab_export, tab_performance, tab_settings = st.tabs([
        icons.tab(cfg.TAB_GENERATOR, "image"),
        icons.tab(cfg.TAB_EDITOR, "edit"),
        icons.tab(cfg.TAB_QUEUE, "queue"),
        icons.tab(cfg.TAB_PROGRESS, "activity"),
        icons.tab("Model Comparison", "sparkles"),
        icons.tab("Safety", "settings"),
        icons.tab("Analytics", "activity"),
        icons.tab("Export & Sharing", "book"),
        icons.tab(cfg.TAB_PERFORMANCE, "bolt"),
        icons.tab(cfg.TAB_SETTINGS, "settings"),
    ])

    with tab_gen:
        from ui.pages import generator
        generator.render()

    with tab_edit:
        from ui.pages import editor
        editor.render()

    with tab_queue:
        from ui.pages import queue
        queue.render()

    with tab_progress:
        from ui.pages import progress_monitor
        progress_monitor.render()

    with tab_comparison:
        from ui.pages import model_comparison
        model_comparison.render()

    with tab_safety:
        from ui.pages import safety
        safety.render()

    with tab_analytics:
        from ui.pages import analytics
        analytics.main()

    with tab_export:
        from ui.pages import export
        export.main()

    with tab_performance:
        from ui.pages import performance
        performance.render()

    with tab_settings:
        from ui.pages import settings
        settings.render()
