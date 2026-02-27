"""Site & Context page for project-scoped reference management."""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.flux2.context_connectors import available_connectors, run_connector
from src.flux2.feature_flags import is_feature_enabled
from src.flux2.governance_validation import sanitize_reference_log_payload, validate_project_access_boundary
from src.flux2.logging_config import log_operation
from ui import icons, state

REFERENCE_CATEGORIES = [
    "site photos",
    "aerial/context image",
    "precedent board",
    "material swatches",
    "sketch overlays",
]

MAX_REFERENCE_SIZE_BYTES = 15 * 1024 * 1024
MAX_REFERENCES_PER_PROJECT = 30
_ALLOWED_MIME_PREFIX = "image/"


def validate_reference_upload(
    *,
    file_name: str,
    file_size: int,
    mime_type: str,
    existing_count: int,
) -> list[str]:
    """Return validation errors for a reference upload candidate."""
    errors: list[str] = []
    if not file_name.strip():
        errors.append("Reference file name is required.")
    if file_size <= 0:
        errors.append("Reference file is empty.")
    if file_size > MAX_REFERENCE_SIZE_BYTES:
        errors.append("Reference file exceeds 15 MB limit.")
    if not mime_type.startswith(_ALLOWED_MIME_PREFIX):
        errors.append("Only image uploads are supported for references.")
    if existing_count >= MAX_REFERENCES_PER_PROJECT:
        errors.append("Reference limit reached for this project.")
    return errors


def render() -> None:
    """Render site/context reference manager for the active project."""
    icons.page_intro(
        "Site & Context",
        "Attach and curate contextual reference imagery scoped to the active project.",
        icon="image",
    )
    st.caption("Workflow: 2/8 · Add site references before Massing & Form generation.")

    active_project = state.get_active_project()
    if not active_project:
        st.warning("Create or activate a project in Project Setup before adding references.")
        return

    project_id = active_project["project_id"]
    access_ok, access_reason = validate_project_access_boundary(
        active_project_id=str(st.session_state.get("active_project_id", "")),
        target_project_id=project_id,
        role=str(st.session_state.get("current_user_role", "editor")),
    )
    if not access_ok:
        st.error(f"Project access boundary check failed: {access_reason}")
        return

    ref_store = st.session_state.get("site_references_by_project", {})
    if not isinstance(ref_store, dict):
        ref_store = {}

    project_refs = ref_store.get(project_id, [])
    if not isinstance(project_refs, list):
        project_refs = []

    with st.container(border=True):
        st.subheader("Add Reference")
        col1, col2 = st.columns(2)
        with col1:
            category = st.selectbox("Category", options=REFERENCE_CATEGORIES, key="site_context_category")
            tags_raw = st.text_input(
                "Metadata Tags",
                value="",
                placeholder="comma-separated tags",
                key="site_context_tags",
            )
        with col2:
            upload = st.file_uploader(
                "Reference Image",
                type=["png", "jpg", "jpeg", "webp"],
                key="site_context_upload",
            )

        if st.button("Add Reference", type="primary", use_container_width=True, key="site_context_add"):
            if upload is None:
                st.error("Please choose an image to add.")
            else:
                errors = validate_reference_upload(
                    file_name=upload.name,
                    file_size=int(upload.size),
                    mime_type=str(upload.type or ""),
                    existing_count=len(project_refs),
                )
                if errors:
                    for msg in errors:
                        st.error(msg)
                else:
                    tags = [tag.strip() for tag in tags_raw.split(",") if tag.strip()]
                    ref_id = f"{project_id}-ref-{len(project_refs) + 1:03d}"
                    project_refs.append(
                        {
                            "reference_id": ref_id,
                            "name": upload.name,
                            "category": category,
                            "mime_type": str(upload.type or "image/unknown"),
                            "size_bytes": int(upload.size),
                            "tags": tags,
                            "image_bytes": upload.getvalue(),
                        }
                    )
                    ref_store[project_id] = project_refs
                    st.session_state.site_references_by_project = ref_store
                    log_operation(
                        "site-context.add-reference",
                        "success",
                        {
                            "project_id": project_id,
                            "reference": sanitize_reference_log_payload(project_refs[-1]),
                        },
                    )
                    st.success(f"Reference added: {ref_id}")
                    st.rerun()

    if is_feature_enabled(st.session_state, "enable_connectors"):
        with st.container(border=True):
            st.subheader("External Context Connectors (Optional)")
            st.caption("Phase-gated connector adapters for CAD/BIM/GIS-style context ingestion.")

            connector_map = available_connectors()
            connector_ids = list(connector_map.keys())
            connector_id = st.selectbox("Connector", options=connector_ids, key="site_context_connector_id")
            connector_query = st.text_input("Connector Query", value=active_project.get("geography", ""), key="site_context_connector_query")

            if st.button("Import Connector Context", use_container_width=True, key="site_context_connector_import"):
                records = run_connector(
                    connector_id=connector_id,
                    query=connector_query,
                    project_context=active_project,
                )
                if not records:
                    st.warning("Connector returned no references.")
                else:
                    next_index = len(project_refs) + 1
                    for rec in records:
                        project_refs.append(
                            {
                                "reference_id": f"{project_id}-ref-{next_index:03d}",
                                "name": rec.get("name", "connector-reference"),
                                "category": rec.get("category", "aerial/context image"),
                                "mime_type": "image/connector-stub",
                                "size_bytes": 0,
                                "tags": rec.get("tags", []),
                                "image_bytes": b"",
                                "source": rec.get("source", connector_id),
                                "notes": rec.get("notes", ""),
                            }
                        )
                        next_index += 1
                    ref_store[project_id] = project_refs
                    st.session_state.site_references_by_project = ref_store
                    log_operation(
                        "site-context.import-connector",
                        "success",
                        {
                            "project_id": project_id,
                            "connector_id": connector_id,
                            "imported_count": len(records),
                        },
                    )
                    st.success(f"Imported {len(records)} connector references.")
                    st.rerun()

    with st.container(border=True):
        st.subheader(f"References for {project_id}")
        if not project_refs:
            st.caption("No references attached yet.")
            return

        replace_options = [r["reference_id"] for r in project_refs]
        replace_target = st.selectbox("Replace Reference", options=replace_options, key="site_context_replace_target")
        replace_upload = st.file_uploader(
            "Replacement Image",
            type=["png", "jpg", "jpeg", "webp"],
            key="site_context_replace_upload",
        )
        if st.button("Replace Selected", use_container_width=True, key="site_context_replace"):
            if replace_upload is None:
                st.error("Please choose a replacement image.")
            else:
                target_index = next((idx for idx, item in enumerate(project_refs) if item.get("reference_id") == replace_target), None)
                if target_index is None:
                    st.error("Selected reference was not found.")
                else:
                    errors = validate_reference_upload(
                        file_name=replace_upload.name,
                        file_size=int(replace_upload.size),
                        mime_type=str(replace_upload.type or ""),
                        existing_count=max(0, len(project_refs) - 1),
                    )
                    if errors:
                        for msg in errors:
                            st.error(msg)
                    else:
                        project_refs[target_index]["name"] = replace_upload.name
                        project_refs[target_index]["mime_type"] = str(replace_upload.type or "image/unknown")
                        project_refs[target_index]["size_bytes"] = int(replace_upload.size)
                        project_refs[target_index]["image_bytes"] = replace_upload.getvalue()
                        ref_store[project_id] = project_refs
                        st.session_state.site_references_by_project = ref_store
                        log_operation(
                            "site-context.replace-reference",
                            "success",
                            {
                                "project_id": project_id,
                                "reference": sanitize_reference_log_payload(project_refs[target_index]),
                            },
                        )
                        st.success(f"Replaced: {replace_target}")
                        st.rerun()

        for idx, ref in enumerate(project_refs):
            cols = st.columns([1.2, 2.2, 0.8])
            with cols[0]:
                image_payload = ref.get("image_bytes")
                if isinstance(image_payload, (bytes, bytearray)) and image_payload:
                    st.image(image_payload, use_container_width=True)
                else:
                    st.caption("Connector reference (no thumbnail)")
            with cols[1]:
                st.markdown(f"**{ref.get('reference_id', '?')}** · {ref.get('name', 'unnamed')}")
                st.caption(f"Category: {ref.get('category', '-')}")
                st.caption(f"Tags: {', '.join(ref.get('tags', [])) or 'none'}")
                size_kb = int(ref.get("size_bytes", 0)) / 1024
                st.caption(f"Size: {size_kb:.1f} KB")
            with cols[2]:
                if st.button("Remove", key=f"site_context_remove_{idx}", use_container_width=True):
                    removed_ref = project_refs[idx] if idx < len(project_refs) else {}
                    project_refs.pop(idx)
                    ref_store[project_id] = project_refs
                    st.session_state.site_references_by_project = ref_store
                    log_operation(
                        "site-context.remove-reference",
                        "success",
                        {
                            "project_id": project_id,
                            "reference": sanitize_reference_log_payload(removed_ref if isinstance(removed_ref, dict) else {}),
                        },
                    )
                    st.rerun()


def get_project_references(project_id: str, session: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Return references list for a project from a session-like mapping."""
    source = st.session_state if session is None else session
    ref_store = source.get("site_references_by_project", {})
    if not isinstance(ref_store, dict):
        return []
    refs = ref_store.get(project_id, [])
    return refs if isinstance(refs, list) else []
