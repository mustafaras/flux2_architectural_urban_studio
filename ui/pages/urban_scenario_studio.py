"""Urban Scenario Studio core workflows (Phase 3.1/3.2/3.4/3.5)."""

from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

from src.flux2.analytics_client import EventType, get_analytics
from src.flux2.annotations import build_annotation_summary
from src.flux2.explainability import build_confidence_labels, build_why_card
from src.flux2.governance import GovernanceAction, WorkflowState, can_perform_action, get_retention_profile
from src.flux2.governance_artifacts import attach_signed_manifest
from src.flux2.logging_config import log_operation
from src.flux2.quality_rubric import RUBRIC_FIELDS, aggregate_average_score
from src.flux2.scenario_studio import compute_scenario_delta, create_scenario_from_option
from ui import icons

_DENSITY_PROFILES = ["low", "medium", "high"]
_MOBILITY_PRIORITIES = ["pedestrian-first", "transit-first", "balanced"]
_GREENING_INTENSITIES = ["low", "medium", "high"]
_ACTIVATION_PROFILES = ["quiet", "mixed", "active"]

_STREET_HIERARCHY_TEMPLATES = ["boulevard", "mixed street", "woonerf", "transit corridor"]
_PUBLIC_REALM_TYPOLOGIES = ["civic plaza", "linear park", "shared court", "waterfront promenade"]
_MULTIMODAL_PRESETS = ["transit-priority", "bike-priority", "balanced access"]


def _option_id(entry: dict[str, Any]) -> str:
    meta = entry.get("metadata", {}) if isinstance(entry.get("metadata"), dict) else {}
    return f"{meta.get('model', '?')}:{meta.get('seed', '?')}:{meta.get('timestamp', '')}"


def _option_label(index: int, entry: dict[str, Any]) -> str:
    meta = entry.get("metadata", {}) if isinstance(entry.get("metadata"), dict) else {}
    return f"#{index + 1} · {meta.get('model', '?')} · seed {meta.get('seed', '?')}"


def _get_shortlisted_options(history: list[dict[str, Any]], shortlist_ids: list[str]) -> list[dict[str, Any]]:
    shortlist = set(shortlist_ids)
    return [entry for entry in history if _option_id(entry) in shortlist]


def _build_workshop_package(
    *,
    shortlisted_options: list[dict[str, Any]],
    scenarios: list[dict[str, Any]],
    project_id: str,
    review_workflow_state: str,
    retention_profile: str,
) -> tuple[bytes, dict[str, Any]]:
    manifest = {
        "package_version": "workshop-package-v1",
        "created_at": datetime.now().isoformat(),
        "project_id": project_id,
        "scenario_count": len(scenarios),
        "shortlisted_count": len(shortlisted_options),
        "review_workflow_state": review_workflow_state,
        "retention_profile": retention_profile,
        "retention_policy": get_retention_profile(retention_profile),
        "lineage": [
            {
                "scenario_id": item.get("scenario_id", ""),
                "parent_option_id": item.get("parent_option_id", ""),
                "lineage": item.get("lineage", {}),
                "notes": item.get("notes", ""),
            }
            for item in scenarios
        ],
    }

    rubric_scores_map = st.session_state.get("comparison_board_rubric_scores", {})
    scored = []
    if isinstance(rubric_scores_map, dict):
        for option in shortlisted_options:
            oid = _option_id(option)
            scores = rubric_scores_map.get(oid, {})
            if isinstance(scores, dict) and scores:
                scored.append({"rubric_scores": scores})
    manifest["rubric_summary"] = {
        "fields": RUBRIC_FIELDS,
        "average": aggregate_average_score(scored),
    }
    manifest["annotation_summary"] = build_annotation_summary(
        st.session_state.get("annotation_threads", []) if isinstance(st.session_state.get("annotation_threads", []), list) else []
    )
    manifest["explainability"] = {
        "options": [
            {
                "option_id": _option_id(option),
                "why_card": build_why_card(option.get("metadata", {}) if isinstance(option.get("metadata"), dict) else {}),
                "confidence": build_confidence_labels(option.get("metadata", {}) if isinstance(option.get("metadata"), dict) else {}),
            }
            for option in shortlisted_options
        ]
    }
    manifest = attach_signed_manifest(manifest)

    archive = io.BytesIO()
    with zipfile.ZipFile(archive, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        zf.writestr("scenarios.json", json.dumps(scenarios, indent=2))

        options_payload = []
        for idx, option in enumerate(shortlisted_options):
            image_bytes = option.get("image_bytes", b"")
            if isinstance(image_bytes, (bytes, bytearray)):
                zf.writestr(f"boards/option_{idx + 1:02d}.png", bytes(image_bytes))
            options_payload.append(
                {
                    "option_id": _option_id(option),
                    "metadata": option.get("metadata", {}),
                }
            )
        zf.writestr("options.json", json.dumps(options_payload, indent=2))

        zf.writestr(
            "README.txt",
            "Workshop package includes PNG board assets, scenario lineage, metadata JSON, and rubric summary. PDF board output is included where external PDF rendering is available.",
        )

    archive.seek(0)
    return archive.getvalue(), manifest


def render() -> None:
    """Render urban scenario studio workflows."""
    icons.page_intro(
        "Urban Scenario Studio",
        "Create scenario alternatives from base options, compare deltas, and export workshop-ready packages.",
        icon="queue",
    )

    history = st.session_state.get("generation_history", [])
    if not isinstance(history, list):
        history = []

    if "urban_scenarios" not in st.session_state:
        st.session_state.urban_scenarios = []

    scenarios = st.session_state.get("urban_scenarios", [])
    if not isinstance(scenarios, list):
        scenarios = []

    tabs = st.tabs([
        icons.tab("Scenario Core", "plus"),
        icons.tab("Urban Modules", "target"),
        icons.tab("Scenario Delta", "activity"),
        icons.tab("Workshop Package", "zip"),
    ])

    with tabs[0]:
        st.subheader("Scenario Duplicator")
        if not history:
            st.info("Generate options first, then duplicate a base option into scenarios.")
        else:
            labels = [_option_label(i, entry) for i, entry in enumerate(history[:20])]
            selected_label = st.selectbox("Base Design Option", labels, key="scenario_base_option")
            base_idx = labels.index(selected_label)
            base_entry = history[base_idx]
            base_option_id = _option_id(base_entry)

            col1, col2 = st.columns(2)
            with col1:
                density_profile = st.selectbox("Density Profile", _DENSITY_PROFILES, key="scenario_density")
                mobility_priority = st.selectbox("Mobility Priority", _MOBILITY_PRIORITIES, key="scenario_mobility")
                greening_intensity = st.selectbox("Greening Intensity", _GREENING_INTENSITIES, key="scenario_greening")
                activation_profile = st.selectbox("Activation Profile", _ACTIVATION_PROFILES, key="scenario_activation")
            with col2:
                label = st.text_input("Scenario Label", value="", key="scenario_label")
                notes = st.text_area("Scenario Notes", value="", key="scenario_notes", height=140)

            if st.button("Create Scenario from Base Option", type="primary", use_container_width=True, key="scenario_create"):
                scenario = create_scenario_from_option(
                    base_option_id=base_option_id,
                    base_metadata=base_entry.get("metadata", {}) if isinstance(base_entry.get("metadata"), dict) else {},
                    density_profile=density_profile,
                    mobility_priority=mobility_priority,
                    greening_intensity=greening_intensity,
                    activation_profile=activation_profile,
                    label=label,
                    notes=notes,
                )
                scenarios.append(scenario)
                st.session_state.urban_scenarios = scenarios
                log_operation("scenario-create", "success", {"scenario_id": scenario["scenario_id"], "parent_option_id": base_option_id})
                st.success(f"Scenario created: {scenario['scenario_id']}")
                st.rerun()

        if scenarios:
            st.markdown("#### Current Scenarios")
            for item in scenarios:
                with st.container(border=True):
                    st.write({
                        "scenario_id": item.get("scenario_id", ""),
                        "parent_option_id": item.get("parent_option_id", ""),
                        "label": item.get("label", ""),
                        "toggles": item.get("toggles", {}),
                        "lineage": item.get("lineage", {}),
                    })

    with tabs[1]:
        st.subheader("Mobility and Public Realm Modules")
        street_template = st.selectbox("Street Hierarchy Template", _STREET_HIERARCHY_TEMPLATES, key="scenario_street_hierarchy")
        public_realm_typology = st.selectbox("Public Realm Typology", _PUBLIC_REALM_TYPOLOGIES, key="scenario_public_realm")
        multimodal_preset = st.selectbox("Multimodal Preset", _MULTIMODAL_PRESETS, key="scenario_multimodal")

        module_contract = {
            "street_hierarchy_template": street_template,
            "public_realm_typology": public_realm_typology,
            "multimodal_preset": multimodal_preset,
        }
        st.session_state.scenario_module_contract = module_contract
        st.caption("Mapped module contract is persisted for scenario prompt/metadata integration.")
        st.code(json.dumps(module_contract, indent=2), language="json")

    with tabs[2]:
        st.subheader("Scenario Delta Comparison")
        if len(scenarios) < 2:
            st.info("Create at least two scenarios to compare deltas.")
        else:
            labels = [f"{s.get('scenario_id', '')} · {s.get('label', '')}" for s in scenarios]
            left_label = st.selectbox("Left Scenario", labels, index=0, key="delta_left")
            right_label = st.selectbox("Right Scenario", labels, index=1, key="delta_right")
            left = scenarios[labels.index(left_label)]
            right = scenarios[labels.index(right_label)]

            history_lookup = {_option_id(entry): entry for entry in history}
            left_parent = history_lookup.get(str(left.get("parent_option_id", "")), {})
            right_parent = history_lookup.get(str(right.get("parent_option_id", "")), {})

            left_img = left_parent.get("image_bytes") if isinstance(left_parent, dict) else None
            right_img = right_parent.get("image_bytes") if isinstance(right_parent, dict) else None
            if isinstance(left_img, (bytes, bytearray)) and isinstance(right_img, (bytes, bytearray)):
                c1, c2 = st.columns(2)
                c1.image(left_img, caption=f"Left parent: {left.get('parent_option_id', '')}", use_container_width=True)
                c2.image(right_img, caption=f"Right parent: {right.get('parent_option_id', '')}", use_container_width=True)

            delta = compute_scenario_delta(left, right)
            st.write({"changed_controls": delta.get("changed_controls", {}), "changed_count": delta.get("changed_count", 0)})
            st.write(
                {
                    "left_metadata": left.get("base_metadata", {}),
                    "right_metadata": right.get("base_metadata", {}),
                }
            )

            if st.button("Export Delta Snapshot", use_container_width=True, key="scenario_export_delta"):
                snapshots = st.session_state.get("urban_delta_snapshots", [])
                if not isinstance(snapshots, list):
                    snapshots = []
                snapshots.append({
                    "timestamp": datetime.now().isoformat(),
                    "delta": delta,
                })
                st.session_state.urban_delta_snapshots = snapshots
                st.download_button(
                    "Download Delta JSON",
                    data=json.dumps(delta, indent=2).encode("utf-8"),
                    file_name=f"scenario_delta_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    key="scenario_delta_download",
                )

    with tabs[3]:
        st.subheader("Workshop Export Package")
        shortlist_ids = st.session_state.get("comparison_board_shortlist_ids", [])
        if not isinstance(shortlist_ids, list):
            shortlist_ids = []

        shortlisted_options = _get_shortlisted_options(history, shortlist_ids)
        st.caption(f"Shortlisted options detected: {len(shortlisted_options)}")

        role = str(st.session_state.get("current_user_role", "editor"))
        workflow_state = str(st.session_state.get("review_workflow_state", WorkflowState.DRAFT.value))
        retention_profile = str(st.session_state.get("retention_profile", "exploratory"))
        can_export = can_perform_action(role, GovernanceAction.EXPORT.value, workflow_state)

        if st.button("Generate Workshop Package", type="primary", use_container_width=True, key="workshop_generate_package", disabled=not can_export):
            if not shortlisted_options:
                st.error("No shortlisted options found. Shortlist options in Comparison Board first.")
            else:
                active_project = st.session_state.get("active_project_id", "")
                package_bytes, manifest = _build_workshop_package(
                    shortlisted_options=shortlisted_options,
                    scenarios=scenarios,
                    project_id=str(active_project),
                    review_workflow_state=workflow_state,
                    retention_profile=retention_profile,
                )
                st.session_state.last_workshop_manifest = manifest
                st.download_button(
                    "Download Workshop Package (.zip)",
                    data=package_bytes,
                    file_name=f"workshop_package_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip",
                    key="workshop_package_download",
                )
                get_analytics().log_event(
                    EventType.EXPORT_BOARD,
                    {
                        "option_count": len(shortlisted_options),
                        "scenario_count": len(scenarios),
                        "package_type": "workshop",
                    },
                )
                log_operation("workshop-export", "success", {"shortlisted": len(shortlisted_options), "scenarios": len(scenarios)})

        manifest = st.session_state.get("last_workshop_manifest")
        if isinstance(manifest, dict):
            st.json(manifest)
