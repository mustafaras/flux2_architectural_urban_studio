"""Comparison Board v1 for 2-4 option shortlist workflows."""

from __future__ import annotations

import base64
import json
from datetime import datetime
from typing import Any

import streamlit as st

from src.flux2.analytics_client import EventType, get_analytics
from src.flux2.annotations import add_comment, build_annotation_summary, create_thread, set_resolved
from src.flux2.explainability import (
    build_confidence_labels,
    build_explainability_metadata,
    build_lineage_graph,
    build_why_card,
)
from src.flux2.feature_flags import is_feature_enabled
from src.flux2.governance import (
    GovernanceAction,
    GovernanceAuditLog,
    UserRole,
    WorkflowState,
    apply_transition,
    can_perform_action,
    get_retention_profile,
)
from src.flux2.governance_artifacts import attach_signed_manifest
from src.flux2.governance_validation import sanitize_comment_log_payload
from src.flux2.logging_config import log_operation
from src.flux2.policy_profiles import check_export_policy, get_policy_profile
from src.flux2.quality_rubric import (
    RUBRIC_FIELDS,
    aggregate_average_score,
    calculate_option_composite_score,
    normalize_rubric_scores,
)
from ui import icons
from ui import state


def _history_label(index: int, entry: dict[str, Any]) -> str:
    meta = entry.get("metadata", {}) if isinstance(entry.get("metadata"), dict) else {}
    return (
        f"#{index + 1} · {meta.get('model', '?')} · seed {meta.get('seed', '?')} "
        f"· {meta.get('width', '?')}x{meta.get('height', '?')}"
    )


def _entry_id(entry: dict[str, Any]) -> str:
    meta = entry.get("metadata", {}) if isinstance(entry.get("metadata"), dict) else {}
    model = str(meta.get("model", "?"))
    seed = str(meta.get("seed", "?"))
    ts = str(meta.get("timestamp", ""))
    return f"{model}:{seed}:{ts}"


def _build_board_package(
    options: list[dict[str, Any]],
    shortlist_ids: list[str],
    rubric_scores_by_option: dict[str, dict[str, float]] | None = None,
    render_set_rubric_avg: dict[str, float] | None = None,
    project_rubric_avg: dict[str, float] | None = None,
    annotations: list[dict[str, Any]] | None = None,
    review_workflow_state: str = WorkflowState.DRAFT.value,
    retention_profile: str = "exploratory",
    explainability: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rubric_scores_by_option = rubric_scores_by_option or {}
    render_set_rubric_avg = render_set_rubric_avg or {field: 0.0 for field in RUBRIC_FIELDS}
    project_rubric_avg = project_rubric_avg or {field: 0.0 for field in RUBRIC_FIELDS}

    packaged_options: list[dict[str, Any]] = []
    for item in options:
        image_bytes = item.get("image_bytes", b"")
        if not isinstance(image_bytes, (bytes, bytearray)):
            continue
        packaged_options.append(
            {
                "option_id": _entry_id(item),
                "metadata": item.get("metadata", {}),
                "rubric_scores": rubric_scores_by_option.get(_entry_id(item), {}),
                "rubric_composite": calculate_option_composite_score(rubric_scores_by_option.get(_entry_id(item), {})),
                "image_b64": base64.b64encode(bytes(image_bytes)).decode("utf-8"),
            }
        )

    return {
        "board_version": "comparison-board-v1",
        "created_at": datetime.now().isoformat(),
        "shortlist_ids": shortlist_ids,
        "review_workflow_state": review_workflow_state,
        "retention_profile": retention_profile,
        "retention_policy": get_retention_profile(retention_profile),
        "rubric_summary": {
            "render_set_average": render_set_rubric_avg,
            "project_average": project_rubric_avg,
        },
        "annotation_summary": build_annotation_summary(annotations or []),
        "annotations": annotations or [],
        "explainability": explainability or {},
        "options": packaged_options,
    }


def _load_board_package(raw_json: str) -> tuple[list[dict[str, Any]], list[str]]:
    payload = json.loads(raw_json)
    options: list[dict[str, Any]] = []
    shortlist_ids: list[str] = []

    for item in payload.get("options", []):
        image_b64 = item.get("image_b64", "")
        if not image_b64:
            continue
        options.append(
            {
                "image_bytes": base64.b64decode(image_b64.encode("utf-8")),
                "metadata": item.get("metadata", {}),
            }
        )

    raw_shortlist = payload.get("shortlist_ids", [])
    if isinstance(raw_shortlist, list):
        shortlist_ids = [str(x) for x in raw_shortlist]

    return options, shortlist_ids


def render() -> None:
    """Render side-by-side board for 2-4 generated options."""
    icons.page_intro(
        "Comparison Board",
        "Review 2–4 options side-by-side, shortlist candidates, and export board packages.",
        icon="scales",
    )

    history = st.session_state.get("generation_history", [])
    if not isinstance(history, list):
        history = []

    source_options = history[:20]

    with st.container(border=True):
        st.subheader("Rebuild Board from Package")
        uploaded_package = st.file_uploader(
            "Load comparison board package",
            type=["json"],
            key="comparison_board_import",
        )
        if uploaded_package is not None:
            loaded_options, loaded_shortlist = _load_board_package(uploaded_package.getvalue().decode("utf-8"))
            st.session_state.comparison_board_loaded_options = loaded_options
            st.session_state.comparison_board_shortlist_ids = loaded_shortlist
            st.success("Board package loaded.")
            st.rerun()

    loaded_options = st.session_state.get("comparison_board_loaded_options", [])
    if isinstance(loaded_options, list) and loaded_options:
        source_options = loaded_options

    if not source_options:
        st.info("Generate at least two options before building a comparison board.")
        return

    if "annotation_threads" not in st.session_state or not isinstance(st.session_state.get("annotation_threads"), list):
        st.session_state.annotation_threads = []

    role_value = str(st.session_state.get("current_user_role", UserRole.EDITOR.value))
    workflow_state_value = str(st.session_state.get("review_workflow_state", WorkflowState.DRAFT.value))
    retention_profile = str(st.session_state.get("retention_profile", "exploratory"))
    policy_profile = str(st.session_state.get("current_policy_profile", "commercial"))
    audit_log = GovernanceAuditLog()

    with st.container(border=True):
        st.subheader("Review Governance")
        col_role, col_state, col_retention = st.columns(3)
        with col_role:
            role_value = st.selectbox(
                "Role",
                options=[r.value for r in UserRole],
                index=[r.value for r in UserRole].index(role_value) if role_value in [r.value for r in UserRole] else 1,
                key="governance_role_selector",
            )
            st.session_state.current_user_role = role_value
        with col_state:
            st.caption(f"Workflow state: {workflow_state_value}")
            target_state = st.selectbox(
                "Transition Target",
                options=[s.value for s in WorkflowState],
                key="governance_target_state",
            )
            if st.button("Apply Transition", use_container_width=True, key="governance_apply_transition"):
                try:
                    new_state = apply_transition(role_value, workflow_state_value, target_state)
                    st.session_state.review_workflow_state = new_state.value
                    history = st.session_state.get("review_workflow_history", [])
                    if not isinstance(history, list):
                        history = []
                    history.append({
                        "timestamp": datetime.now().isoformat(),
                        "role": role_value,
                        "from": workflow_state_value,
                        "to": new_state.value,
                    })
                    st.session_state.review_workflow_history = history
                    audit_log.log_event(
                        event_name="review-transition",
                        actor_role=role_value,
                        status="success",
                        details={"from": workflow_state_value, "to": new_state.value},
                    )
                    log_operation("review-transition", "success", {"from": workflow_state_value, "to": new_state.value, "role": role_value})
                    st.success(f"Transitioned to {new_state.value}")
                    st.rerun()
                except Exception as exc:  # noqa: BLE001
                    audit_log.log_event(
                        event_name="review-transition",
                        actor_role=role_value,
                        status="denied",
                        details={"from": workflow_state_value, "to": target_state, "reason": str(exc)},
                    )
                    st.error(str(exc))
        with col_retention:
            retention_profile = st.selectbox(
                "Retention Profile",
                options=["exploratory", "approved"],
                index=0 if retention_profile == "exploratory" else 1,
                key="governance_retention_profile",
            )
            st.session_state.retention_profile = retention_profile
            st.json(get_retention_profile(retention_profile))

        st.caption(f"Policy Profile: {policy_profile}")
        st.write(get_policy_profile(policy_profile))

    labels = [_history_label(i, entry) for i, entry in enumerate(source_options)]
    selected_labels = st.multiselect(
        "Select 2–4 options",
        options=labels,
        default=labels[: min(2, len(labels))],
        max_selections=4,
        key="comparison_board_selection",
    )

    if len(selected_labels) < 2:
        st.warning("Select at least 2 options to render the board.")
        return

    selected_indexes = [labels.index(label) for label in selected_labels]
    selected_options = [source_options[idx] for idx in selected_indexes]

    rubric_scores_by_option = st.session_state.get("comparison_board_rubric_scores", {})
    if not isinstance(rubric_scores_by_option, dict):
        rubric_scores_by_option = {}

    shortlist_ids = st.session_state.get("comparison_board_shortlist_ids", [])
    if not isinstance(shortlist_ids, list):
        shortlist_ids = []

    cols = st.columns(len(selected_options))
    for idx, option in enumerate(selected_options):
        option_id = _entry_id(option)
        meta = option.get("metadata", {}) if isinstance(option.get("metadata"), dict) else {}

        with cols[idx]:
            st.image(option.get("image_bytes", b""), use_container_width=True)
            st.caption(f"Seed: {meta.get('seed', '?')}")
            st.caption(f"Model: {meta.get('model', '?')}")
            st.caption(f"Settings: {meta.get('num_steps', '?')} steps · g {meta.get('guidance', '?')}")
            st.caption(f"Runtime: {meta.get('generation_time_s', '?')}s")

            if is_feature_enabled(st.session_state, "enable_explainability"):
                with st.expander("Why Card", expanded=False):
                    why_card = build_why_card(meta)
                    confidence = build_confidence_labels(meta)
                    st.write(why_card)
                    st.write(confidence)

            with st.expander("Design Quality Rubric", expanded=False):
                existing_scores = rubric_scores_by_option.get(option_id, {}) if isinstance(rubric_scores_by_option.get(option_id, {}), dict) else {}
                updated_scores: dict[str, float] = {}
                for field in RUBRIC_FIELDS:
                    label = field.replace("_", " ").title()
                    slider_key = f"rubric_{field}_{option_id}"
                    updated_scores[field] = float(
                        st.slider(
                            label,
                            min_value=0.0,
                            max_value=10.0,
                            value=float(existing_scores.get(field, 5.0)),
                            step=0.5,
                            key=slider_key,
                        )
                    )
                if st.button("Save Rubric Scores", use_container_width=True, key=f"rubric_save_{option_id}"):
                    if not can_perform_action(role_value, GovernanceAction.EDIT.value, st.session_state.get("review_workflow_state", WorkflowState.DRAFT.value)):
                        st.error("Current role/workflow state does not allow edits.")
                        audit_log.log_event("rubric-save", role_value, "denied", {"option_id": option_id})
                        st.stop()
                    normalized = normalize_rubric_scores(updated_scores)
                    rubric_scores_by_option[option_id] = normalized
                    st.session_state.comparison_board_rubric_scores = rubric_scores_by_option
                    project = state.get_active_project()
                    if project:
                        project_scores = st.session_state.get("project_rubric_scores", {})
                        if not isinstance(project_scores, dict):
                            project_scores = {}
                        project_id = project.get("project_id", "")
                        bucket = project_scores.get(project_id, {}) if isinstance(project_scores.get(project_id, {}), dict) else {}
                        bucket[option_id] = normalized
                        project_scores[project_id] = bucket
                        st.session_state.project_rubric_scores = project_scores
                    audit_log.log_event("rubric-save", role_value, "success", {"option_id": option_id})
                    st.success("Rubric scores saved.")
                    st.rerun()

                composite = calculate_option_composite_score(updated_scores)
                st.caption(f"Option composite score: {composite}/10")

            shortlisted = option_id in shortlist_ids
            action_label = "Unshortlist" if shortlisted else "Shortlist"
            if st.button(action_label, use_container_width=True, key=f"comparison_board_shortlist_{idx}_{option_id}"):
                if not can_perform_action(role_value, GovernanceAction.EDIT.value, st.session_state.get("review_workflow_state", WorkflowState.DRAFT.value)):
                    st.error("Current role/workflow state does not allow shortlist edits.")
                    audit_log.log_event("shortlist-option", role_value, "denied", {"option_id": option_id})
                    st.stop()
                if shortlisted:
                    shortlist_ids = [item for item in shortlist_ids if item != option_id]
                else:
                    shortlist_ids.append(option_id)
                    if not st.session_state.get("kpi_first_shortlist_ts"):
                        st.session_state.kpi_first_shortlist_ts = datetime.now().isoformat()
                    st.session_state.kpi_shortlist_count = int(st.session_state.get("kpi_shortlist_count", 0)) + 1
                    get_analytics().log_event(
                        EventType.SHORTLIST_OPTION,
                        {
                            "option_id": option_id,
                            "model": meta.get("model", ""),
                            "seed": meta.get("seed", 0),
                        },
                    )
                    log_operation("shortlist-option", "success", {"option_id": option_id})
                audit_log.log_event("shortlist-option", role_value, "success", {"option_id": option_id, "shortlisted": not shortlisted})
                st.session_state.comparison_board_shortlist_ids = shortlist_ids
                st.rerun()

            with st.expander("Annotations & Threaded Feedback", expanded=False):
                threads = st.session_state.get("annotation_threads", [])
                if not isinstance(threads, list):
                    threads = []

                option_threads = [t for t in threads if str(t.get("option_id", "")) == option_id]
                st.caption(f"Threads on this option: {len(option_threads)}")

                px = st.slider("Pin X (%)", min_value=0.0, max_value=100.0, value=50.0, step=1.0, key=f"ann_pin_x_{option_id}")
                py = st.slider("Pin Y (%)", min_value=0.0, max_value=100.0, value=50.0, step=1.0, key=f"ann_pin_y_{option_id}")
                label = st.text_input("Pin Label", value="", key=f"ann_label_{option_id}")
                note = st.text_area("Initial Comment", value="", key=f"ann_note_{option_id}", height=80)
                author = st.text_input("Author", value="reviewer", key=f"ann_author_{option_id}")

                if st.button("Add Annotation", use_container_width=True, key=f"ann_add_{option_id}"):
                    if not can_perform_action(role_value, GovernanceAction.EDIT.value, st.session_state.get("review_workflow_state", WorkflowState.DRAFT.value)):
                        st.error("Current role/workflow state does not allow annotation edits.")
                        st.stop()
                    if not note.strip():
                        st.warning("Initial comment is required.")
                    else:
                        threads.append(
                            create_thread(
                                option_id=option_id,
                                pin_x_pct=px,
                                pin_y_pct=py,
                                label=label,
                                author=author,
                                note=note,
                                related_option_ids=shortlist_ids,
                            )
                        )
                        st.session_state.annotation_threads = threads
                        audit_log.log_event("annotation-add", role_value, "success", {"option_id": option_id})
                        log_operation(
                            "annotation.add",
                            "success",
                            {
                                "option_id": option_id,
                                "payload": sanitize_comment_log_payload(author=author, text=note),
                            },
                        )
                        st.success("Annotation added.")
                        st.rerun()

                for thread in option_threads:
                    thread_id = str(thread.get("thread_id", ""))
                    with st.container(border=True):
                        st.write({
                            "thread_id": thread_id,
                            "pin": [thread.get("pin_x_pct", 0.0), thread.get("pin_y_pct", 0.0)],
                            "label": thread.get("label", ""),
                            "status": thread.get("status", "open"),
                            "created_by": thread.get("created_by", ""),
                        })
                        comments = thread.get("comments", [])
                        if isinstance(comments, list):
                            for comment in comments:
                                st.caption(f"[{comment.get('created_at', '')}] {comment.get('author', '')}: {comment.get('text', '')}")

                        reply = st.text_input("Reply", value="", key=f"ann_reply_{thread_id}")
                        col_reply, col_resolve = st.columns(2)
                        with col_reply:
                            if st.button("Reply", key=f"ann_reply_btn_{thread_id}", use_container_width=True):
                                add_comment(thread, author=author, text=reply)
                                st.session_state.annotation_threads = threads
                                log_operation(
                                    "annotation.reply",
                                    "success",
                                    {
                                        "thread_id": thread_id,
                                        "payload": sanitize_comment_log_payload(author=author, text=reply),
                                    },
                                )
                                st.rerun()
                        with col_resolve:
                            mark_resolved = thread.get("status", "open") != "resolved"
                            resolve_label = "Resolve" if mark_resolved else "Re-open"
                            if st.button(resolve_label, key=f"ann_resolve_btn_{thread_id}", use_container_width=True):
                                set_resolved(thread, resolved=mark_resolved, actor=author)
                                st.session_state.annotation_threads = threads
                                st.rerun()

    with st.container(border=True):
        st.subheader("Board Actions")
        st.caption(f"Shortlisted options: {len(shortlist_ids)}")

        render_set_scored = [
            {"rubric_scores": rubric_scores_by_option.get(_entry_id(option), {})}
            for option in selected_options
            if isinstance(rubric_scores_by_option.get(_entry_id(option), {}), dict)
            and rubric_scores_by_option.get(_entry_id(option), {})
        ]
        render_set_avg = aggregate_average_score(render_set_scored)
        st.write({"render_set_rubric_average": render_set_avg})

        project_avg = {field: 0.0 for field in RUBRIC_FIELDS}
        project = state.get_active_project()
        if project:
            project_id = project.get("project_id", "")
            all_project_scores = st.session_state.get("project_rubric_scores", {})
            project_bucket = all_project_scores.get(project_id, {}) if isinstance(all_project_scores, dict) and isinstance(all_project_scores.get(project_id, {}), dict) else {}
            project_scored = [{"rubric_scores": score} for score in project_bucket.values() if isinstance(score, dict)]
            project_avg = aggregate_average_score(project_scored)
            st.write({"project_rubric_average": project_avg})

        package_payload = _build_board_package(
            selected_options,
            shortlist_ids,
            rubric_scores_by_option,
            render_set_avg,
            project_avg,
            annotations=st.session_state.get("annotation_threads", []),
            review_workflow_state=str(st.session_state.get("review_workflow_state", WorkflowState.DRAFT.value)),
            retention_profile=str(st.session_state.get("retention_profile", "exploratory")),
            explainability={
                "lineage_graph": build_lineage_graph(
                    selected_options=selected_options,
                    scenarios=st.session_state.get("urban_scenarios", []) if isinstance(st.session_state.get("urban_scenarios", []), list) else [],
                ),
                "options": {
                    _entry_id(option): build_explainability_metadata(
                        option_metadata=(option.get("metadata", {}) if isinstance(option.get("metadata"), dict) else {}),
                        lineage_graph=build_lineage_graph(
                            selected_options=selected_options,
                            scenarios=st.session_state.get("urban_scenarios", []) if isinstance(st.session_state.get("urban_scenarios", []), list) else [],
                        ),
                    )
                    for option in selected_options
                },
            },
        )
        package_payload = attach_signed_manifest(package_payload)
        package_json = json.dumps(package_payload, indent=2)

        lineage = package_payload.get("explainability", {}).get("lineage_graph", {})
        if isinstance(lineage, dict):
            st.markdown("**Lineage Graph (Mermaid)**")
            st.code(str(lineage.get("mermaid", "")), language="markdown")

        can_export = can_perform_action(
            role_value,
            GovernanceAction.EXPORT.value,
            st.session_state.get("review_workflow_state", WorkflowState.DRAFT.value),
        )
        policy_ok, policy_reason = check_export_policy(
            profile=policy_profile,
            workflow_state=str(st.session_state.get("review_workflow_state", WorkflowState.DRAFT.value)),
            include_prompts=False,
        )
        can_export = can_export and policy_ok
        if not policy_ok:
            st.warning(f"Policy block: {policy_reason}")

        exported = st.download_button(
            label="Export Board Package",
            data=package_json.encode("utf-8"),
            file_name=f"comparison_board_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True,
            key="comparison_board_export_package",
            disabled=not can_export,
        )
        if exported:
            st.session_state.kpi_board_export_count = int(st.session_state.get("kpi_board_export_count", 0)) + 1
            get_analytics().log_event(
                EventType.EXPORT_BOARD,
                {
                    "option_count": len(selected_options),
                    "shortlist_count": len(shortlist_ids),
                },
            )
            audit_log.log_event(
                "export-board",
                role_value,
                "success",
                {
                    "shortlist_count": len(shortlist_ids),
                    "workflow_state": st.session_state.get("review_workflow_state", WorkflowState.DRAFT.value),
                },
            )
            log_operation(
                "export-board",
                "success",
                {
                    "option_count": len(selected_options),
                    "shortlist_count": len(shortlist_ids),
                },
            )
