"""Project Setup page for architecture and urban workflows."""

from __future__ import annotations

import streamlit as st

from src.flux2.analytics_client import EventType, get_analytics
from src.flux2.logging_config import log_operation
from ui import icons, state

_PROJECT_TYPES = [
    "Mixed-use Development",
    "Residential",
    "Commercial",
    "Institutional",
    "Public Realm",
    "Urban Masterplan",
]

_CLIMATE_PROFILES = [
    "Temperate",
    "Arid",
    "Cold",
    "Tropical",
    "Continental",
]

_DESIGN_PHASES = [
    "Concept",
    "Schematic",
    "Design Development",
    "Planning Approval",
    "Presentation",
]


def render() -> None:
    """Render project setup and project context management."""
    icons.page_intro(
        "Project Setup",
        "Define project-scoped metadata so every generation stays traceable and reproducible.",
        icon="folder",
    )
    st.caption("Workflow: 1/8 · Start with a project brief, then continue to Site & Context.")

    active_project = state.get_active_project()
    projects = state.list_projects()

    with st.container(border=True):
        st.subheader("Create / Edit Project Context")

        default_project_type = active_project.get("project_type", _PROJECT_TYPES[0]) if active_project else _PROJECT_TYPES[0]
        default_climate = active_project.get("climate_profile", _CLIMATE_PROFILES[0]) if active_project else _CLIMATE_PROFILES[0]
        default_phase = active_project.get("design_phase", _DESIGN_PHASES[0]) if active_project else _DESIGN_PHASES[0]

        col1, col2 = st.columns(2)
        with col1:
            project_type = st.selectbox(
                "Project Type",
                options=_PROJECT_TYPES,
                index=_PROJECT_TYPES.index(default_project_type) if default_project_type in _PROJECT_TYPES else 0,
                key="project_setup_project_type",
            )
            geography = st.text_input(
                "Geography",
                value=active_project.get("geography", "") if active_project else "",
                placeholder="e.g., Toronto, CA",
                key="project_setup_geography",
            )
            climate_profile = st.selectbox(
                "Climate Profile",
                options=_CLIMATE_PROFILES,
                index=_CLIMATE_PROFILES.index(default_climate) if default_climate in _CLIMATE_PROFILES else 0,
                key="project_setup_climate_profile",
            )

        with col2:
            design_phase = st.selectbox(
                "Design Phase",
                options=_DESIGN_PHASES,
                index=_DESIGN_PHASES.index(default_phase) if default_phase in _DESIGN_PHASES else 0,
                key="project_setup_design_phase",
            )
            team_label = st.text_input(
                "Team Label",
                value=active_project.get("team_label", "") if active_project else "",
                placeholder="e.g., Studio A / Urban Pod",
                key="project_setup_team_label",
            )
            project_id = st.text_input(
                "Project ID",
                value=active_project.get("project_id", "") if active_project else "",
                placeholder="auto-generated if empty",
                key="project_setup_project_id",
            )

        if st.button("Save Project Context", type="primary", use_container_width=True, key="project_setup_save"):
            if not geography.strip() or not team_label.strip():
                st.error("Geography and Team Label are required.")
            else:
                saved = state.create_or_update_project(
                    project_type=project_type,
                    geography=geography,
                    climate_profile=climate_profile,
                    design_phase=design_phase,
                    team_label=team_label,
                    project_id=project_id or None,
                )
                get_analytics().log_event(
                    EventType.CREATE_PROJECT,
                    {
                        "project_id": saved["project_id"],
                        "project_type": saved["project_type"],
                        "design_phase": saved["design_phase"],
                    },
                )
                log_operation(
                    "create-project",
                    "success",
                    {
                        "project_id": saved["project_id"],
                        "project_type": saved["project_type"],
                        "design_phase": saved["design_phase"],
                    },
                )
                st.success(f"Active project set: {saved['project_id']}")
                st.rerun()

    with st.container(border=True):
        st.subheader("Active Project")
        active_project = state.get_active_project()
        if not active_project:
            st.warning("No active project yet. Create one above to enable downstream generation.")
        else:
            st.write(
                {
                    "project_id": active_project["project_id"],
                    "project_type": active_project["project_type"],
                    "geography": active_project["geography"],
                    "climate_profile": active_project["climate_profile"],
                    "design_phase": active_project["design_phase"],
                    "team_label": active_project["team_label"],
                }
            )

    with st.container(border=True):
        st.subheader("Project Registry")
        if not projects:
            st.caption("No saved project contexts in this session.")
            return

        labels = [f"{p['project_id']} · {p['project_type']} · {p['geography']}" for p in projects]
        selected_label = st.selectbox("Switch Active Project", options=labels, key="project_setup_registry_select")
        selected_idx = labels.index(selected_label)
        selected_project_id = projects[selected_idx]["project_id"]
        if st.button("Activate Selected Project", use_container_width=True, key="project_setup_activate"):
            if state.set_active_project(selected_project_id):
                st.success(f"Activated: {selected_project_id}")
                st.rerun()
            else:
                st.error("Unable to activate the selected project.")
