from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class SLAThresholds:
    p95_latency_s: float = 8.0
    queue_backlog: int = 20
    error_rate_pct: float = 5.0
    min_cache_hit_ratio_pct: float = 50.0


def evaluate_sla(metrics: dict[str, Any], thresholds: SLAThresholds | None = None) -> list[dict[str, Any]]:
    cfg = thresholds or SLAThresholds()
    alerts: list[dict[str, Any]] = []

    p95 = float(metrics.get("p95_latency_s", 0.0) or 0.0)
    backlog = int(metrics.get("queue_backlog", 0) or 0)
    error_rate = float(metrics.get("error_rate_pct", 0.0) or 0.0)
    cache_hit = float(metrics.get("cache_hit_ratio_pct", 0.0) or 0.0)

    if p95 > cfg.p95_latency_s:
        alerts.append({"metric": "p95_latency_s", "value": p95, "threshold": cfg.p95_latency_s, "severity": "high"})
    if backlog > cfg.queue_backlog:
        alerts.append({"metric": "queue_backlog", "value": backlog, "threshold": cfg.queue_backlog, "severity": "high"})
    if error_rate > cfg.error_rate_pct:
        alerts.append({"metric": "error_rate_pct", "value": error_rate, "threshold": cfg.error_rate_pct, "severity": "critical"})
    if cache_hit < cfg.min_cache_hit_ratio_pct:
        alerts.append({"metric": "cache_hit_ratio_pct", "value": cache_hit, "threshold": cfg.min_cache_hit_ratio_pct, "severity": "medium"})

    return alerts


def correlate_quality_with_performance(
    *,
    quality_score: float,
    p95_latency_s: float,
    error_rate_pct: float,
) -> dict[str, Any]:
    efficiency = max(0.0, round(quality_score - (p95_latency_s * 0.25) - (error_rate_pct * 0.5), 2))
    return {
        "quality_score": round(float(quality_score), 2),
        "p95_latency_s": round(float(p95_latency_s), 2),
        "error_rate_pct": round(float(error_rate_pct), 2),
        "quality_efficiency_index": efficiency,
    }


def default_runbook_links() -> dict[str, str]:
    return {
        "latency": "docs/DEPLOYMENT.md#health-checks--monitoring",
        "queue": "docs/PERFORMANCE.md",
        "errors": "docs/ERROR_HANDLING.md",
        "cache": "docs/PERFORMANCE.md#caching",
    }
