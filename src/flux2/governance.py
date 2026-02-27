from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import Any


class UserRole(str, Enum):
    VIEWER = "viewer"
    EDITOR = "editor"
    LEAD_REVIEWER = "lead_reviewer"
    ADMIN = "admin"


class WorkflowState(str, Enum):
    DRAFT = "draft"
    IN_REVIEW = "in-review"
    APPROVED = "approved"
    ARCHIVED = "archived"


class GovernanceAction(str, Enum):
    EDIT = "edit"
    APPROVE = "approve"
    EXPORT = "export"


_ALLOWED_ACTIONS: dict[UserRole, set[GovernanceAction]] = {
    UserRole.VIEWER: set(),
    UserRole.EDITOR: {GovernanceAction.EDIT},
    UserRole.LEAD_REVIEWER: {GovernanceAction.EDIT, GovernanceAction.APPROVE, GovernanceAction.EXPORT},
    UserRole.ADMIN: {GovernanceAction.EDIT, GovernanceAction.APPROVE, GovernanceAction.EXPORT},
}

_ALLOWED_TRANSITIONS: dict[WorkflowState, set[WorkflowState]] = {
    WorkflowState.DRAFT: {WorkflowState.IN_REVIEW},
    WorkflowState.IN_REVIEW: {WorkflowState.DRAFT, WorkflowState.APPROVED},
    WorkflowState.APPROVED: {WorkflowState.ARCHIVED},
    WorkflowState.ARCHIVED: set(),
}

_TRANSITION_ROLES: dict[tuple[WorkflowState, WorkflowState], set[UserRole]] = {
    (WorkflowState.DRAFT, WorkflowState.IN_REVIEW): {UserRole.EDITOR, UserRole.LEAD_REVIEWER, UserRole.ADMIN},
    (WorkflowState.IN_REVIEW, WorkflowState.DRAFT): {UserRole.LEAD_REVIEWER, UserRole.ADMIN},
    (WorkflowState.IN_REVIEW, WorkflowState.APPROVED): {UserRole.LEAD_REVIEWER, UserRole.ADMIN},
    (WorkflowState.APPROVED, WorkflowState.ARCHIVED): {UserRole.ADMIN},
}


@dataclass(slots=True)
class TransitionResult:
    allowed: bool
    reason: str


_RETENTION_PROFILES: dict[str, dict[str, Any]] = {
    "exploratory": {"retention_days": 30, "deletion_policy": "rolling", "review_required": False},
    "approved": {"retention_days": 365, "deletion_policy": "immutable-window", "review_required": True},
}


def parse_role(value: str | UserRole) -> UserRole:
    if isinstance(value, UserRole):
        return value
    return UserRole(str(value).strip().lower())


def parse_workflow_state(value: str | WorkflowState) -> WorkflowState:
    if isinstance(value, WorkflowState):
        return value
    return WorkflowState(str(value).strip().lower())


def can_perform_action(role: str | UserRole, action: str | GovernanceAction, state: str | WorkflowState) -> bool:
    role_enum = parse_role(role)
    state_enum = parse_workflow_state(state)
    action_enum = action if isinstance(action, GovernanceAction) else GovernanceAction(str(action).strip().lower())

    if action_enum not in _ALLOWED_ACTIONS.get(role_enum, set()):
        return False

    if action_enum == GovernanceAction.EDIT and state_enum in {WorkflowState.APPROVED, WorkflowState.ARCHIVED}:
        return False

    if action_enum == GovernanceAction.APPROVE and state_enum != WorkflowState.IN_REVIEW:
        return False

    if action_enum == GovernanceAction.EXPORT and state_enum == WorkflowState.ARCHIVED:
        return role_enum == UserRole.ADMIN

    return True


def can_transition(role: str | UserRole, current: str | WorkflowState, target: str | WorkflowState) -> TransitionResult:
    role_enum = parse_role(role)
    current_enum = parse_workflow_state(current)
    target_enum = parse_workflow_state(target)

    if target_enum == current_enum:
        return TransitionResult(False, "Already in requested state")

    allowed_targets = _ALLOWED_TRANSITIONS.get(current_enum, set())
    if target_enum not in allowed_targets:
        return TransitionResult(False, f"Transition {current_enum.value} -> {target_enum.value} is not allowed")

    allowed_roles = _TRANSITION_ROLES.get((current_enum, target_enum), set())
    if role_enum not in allowed_roles:
        return TransitionResult(False, f"Role {role_enum.value} cannot transition to {target_enum.value}")

    return TransitionResult(True, "ok")


def apply_transition(role: str | UserRole, current: str | WorkflowState, target: str | WorkflowState) -> WorkflowState:
    result = can_transition(role=role, current=current, target=target)
    if not result.allowed:
        raise PermissionError(result.reason)
    return parse_workflow_state(target)


def get_retention_profile(profile_name: str) -> dict[str, Any]:
    return dict(_RETENTION_PROFILES.get(profile_name, _RETENTION_PROFILES["exploratory"]))


class GovernanceAuditLog:
    """Structured, searchable governance event logger."""

    def __init__(self, log_path: str | Path = "logs/governance/audit.jsonl") -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()

    def log_event(self, event_name: str, actor_role: str, status: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_name": event_name,
            "actor_role": str(actor_role),
            "status": status,
            "details": details or {},
        }
        serialized = json.dumps(payload, ensure_ascii=False)
        with self._lock:
            with open(self.log_path, "a", encoding="utf-8") as handle:
                handle.write(serialized + "\n")
        return payload

    def query(self, event_name: str | None = None, status: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        if not self.log_path.exists():
            return []
        out: list[dict[str, Any]] = []
        with self._lock:
            with open(self.log_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if event_name and item.get("event_name") != event_name:
                        continue
                    if status and item.get("status") != status:
                        continue
                    out.append(item)
        return out[-max(1, int(limit)):]
