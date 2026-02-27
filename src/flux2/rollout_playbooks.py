from __future__ import annotations

from typing import Any


_ROLLOUT_PLAYBOOKS: dict[str, dict[str, Any]] = {
    "pilot": {
        "checklist": [
            "Enable explainability and governance core modules",
            "Run shortlist-to-export rehearsal with one project",
            "Validate signed manifest and retention policy outputs",
        ],
        "kpi_gates": {
            "success_rate_min": 95.0,
            "p95_latency_max_s": 10.0,
            "queue_backlog_max": 30,
        },
    },
    "expansion": {
        "checklist": [
            "Enable optional connectors in selected teams",
            "Validate policy profile enforcement by organization type",
            "Run governance dashboard review cadence weekly",
        ],
        "kpi_gates": {
            "success_rate_min": 97.0,
            "p95_latency_max_s": 8.0,
            "queue_backlog_max": 20,
        },
    },
    "production": {
        "checklist": [
            "Freeze feature flags for controlled rollout",
            "Validate rollback playbook and runbook links",
            "Approve go/no-go with KPI gate evidence",
        ],
        "kpi_gates": {
            "success_rate_min": 98.0,
            "p95_latency_max_s": 6.0,
            "queue_backlog_max": 12,
        },
    },
}


def list_rollout_stages() -> list[str]:
    return list(_ROLLOUT_PLAYBOOKS.keys())


def get_rollout_playbook(stage: str) -> dict[str, Any]:
    return dict(_ROLLOUT_PLAYBOOKS.get(stage, _ROLLOUT_PLAYBOOKS["pilot"]))


def evaluate_go_no_go(stage: str, *, success_rate: float, p95_latency_s: float, queue_backlog: int) -> dict[str, Any]:
    playbook = get_rollout_playbook(stage)
    gates = playbook.get("kpi_gates", {}) if isinstance(playbook.get("kpi_gates", {}), dict) else {}

    checks = {
        "success_rate": float(success_rate) >= float(gates.get("success_rate_min", 95.0)),
        "p95_latency": float(p95_latency_s) <= float(gates.get("p95_latency_max_s", 10.0)),
        "queue_backlog": int(queue_backlog) <= int(gates.get("queue_backlog_max", 30)),
    }

    return {
        "stage": stage,
        "checks": checks,
        "passed": all(checks.values()),
        "kpi_gates": gates,
    }
