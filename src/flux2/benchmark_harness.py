from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_prompt_suite(path: str | Path) -> dict[str, list[str]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    architecture = payload.get("architecture", [])
    urban = payload.get("urban", [])
    return {
        "architecture": [str(p).strip() for p in architecture if str(p).strip()],
        "urban": [str(p).strip() for p in urban if str(p).strip()],
    }


def validate_prompt_suite(suite: dict[str, list[str]], minimum: int = 10) -> tuple[bool, list[str]]:
    issues: list[str] = []
    arch_count = len(suite.get("architecture", []))
    urban_count = len(suite.get("urban", []))

    if arch_count < minimum:
        issues.append(f"architecture prompts below minimum: {arch_count} < {minimum}")
    if urban_count < minimum:
        issues.append(f"urban prompts below minimum: {urban_count} < {minimum}")

    return len(issues) == 0, issues


def _bounded_score(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 4)


def _score_prompt(prompt: str) -> dict[str, float]:
    text = prompt.lower()
    words = [w for w in text.split() if w]
    token_count = max(1, len(words))

    coherence_signals = sum(1 for t in ("material", "facade", "massing", "street", "public realm", "mobility") if t in text)
    plausibility_signals = sum(1 for t in ("climate", "context", "program", "human-scale", "access", "daylight") if t in text)
    readiness_signals = sum(1 for t in ("diagram", "board", "render", "option", "stakeholder", "review") if t in text)

    coherence = _bounded_score(0.45 + min(0.4, coherence_signals * 0.08) + min(0.15, token_count / 100.0))
    plausibility = _bounded_score(0.45 + min(0.4, plausibility_signals * 0.08) + min(0.15, token_count / 120.0))
    readiness = _bounded_score(0.4 + min(0.45, readiness_signals * 0.09) + min(0.15, token_count / 90.0))

    return {
        "coherence": coherence,
        "plausibility": plausibility,
        "readiness": readiness,
    }


def build_release_quality_metrics(suite: dict[str, list[str]]) -> dict[str, Any]:
    by_domain: dict[str, dict[str, float]] = {}
    prompt_count = 0

    for domain in ("architecture", "urban"):
        prompts = suite.get(domain, [])
        prompt_count += len(prompts)
        if not prompts:
            by_domain[domain] = {"coherence": 0.0, "plausibility": 0.0, "readiness": 0.0}
            continue

        rows = [_score_prompt(p) for p in prompts]
        by_domain[domain] = {
            "coherence": round(sum(r["coherence"] for r in rows) / len(rows), 4),
            "plausibility": round(sum(r["plausibility"] for r in rows) / len(rows), 4),
            "readiness": round(sum(r["readiness"] for r in rows) / len(rows), 4),
        }

    all_domains = [by_domain.get("architecture", {}), by_domain.get("urban", {})]
    quality = {
        "coherence": round(sum(float(d.get("coherence", 0.0)) for d in all_domains) / 2, 4),
        "plausibility": round(sum(float(d.get("plausibility", 0.0)) for d in all_domains) / 2, 4),
        "readiness": round(sum(float(d.get("readiness", 0.0)) for d in all_domains) / 2, 4),
    }

    return {
        "prompt_counts": {
            "architecture": len(suite.get("architecture", [])),
            "urban": len(suite.get("urban", [])),
            "total": prompt_count,
        },
        "quality": quality,
        "quality_by_domain": by_domain,
    }


def compare_quality_regression(
    baseline: dict[str, Any],
    current: dict[str, Any],
    *,
    max_drop_pct: float = 3.0,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "regression_detected": False,
        "threshold_pct": max_drop_pct,
        "details": [],
    }

    baseline_quality = baseline.get("quality", {}) if isinstance(baseline, dict) else {}
    current_quality = current.get("quality", {}) if isinstance(current, dict) else {}

    for metric in ("coherence", "plausibility", "readiness"):
        b = float(baseline_quality.get(metric, 0.0) or 0.0)
        c = float(current_quality.get(metric, 0.0) or 0.0)
        if b <= 0:
            continue
        delta_pct = ((c - b) / b) * 100.0
        item = {
            "metric": metric,
            "baseline": round(b, 4),
            "current": round(c, 4),
            "delta_pct": round(delta_pct, 2),
            "status": "stable",
        }
        if delta_pct < -abs(max_drop_pct):
            item["status"] = "regression"
            out["regression_detected"] = True
        elif delta_pct > 0:
            item["status"] = "improved"

        out["details"].append(item)

    return out


def build_release_quality_summary(
    *,
    current: dict[str, Any],
    baseline: dict[str, Any] | None,
    comparison: dict[str, Any] | None,
) -> str:
    quality = current.get("quality", {})
    counts = current.get("prompt_counts", {})

    lines = [
        "# Release Quality Summary",
        "",
        "## Prompt Suite Coverage",
        f"- Architecture prompts: {int(counts.get('architecture', 0))}",
        f"- Urban prompts: {int(counts.get('urban', 0))}",
        f"- Total prompts: {int(counts.get('total', 0))}",
        "",
        "## Quality Metrics",
        f"- Coherence: {float(quality.get('coherence', 0.0)):.4f}",
        f"- Plausibility: {float(quality.get('plausibility', 0.0)):.4f}",
        f"- Readiness: {float(quality.get('readiness', 0.0)):.4f}",
    ]

    if baseline and comparison:
        lines.append("")
        lines.append("## Regression Check")
        lines.append(f"- Threshold: {float(comparison.get('threshold_pct', 0.0)):.2f}%")
        for row in comparison.get("details", []):
            if not isinstance(row, dict):
                continue
            lines.append(
                f"- {row.get('metric', 'metric')}: baseline={row.get('baseline', 0.0)} current={row.get('current', 0.0)} delta={row.get('delta_pct', 0.0)}% status={row.get('status', 'stable')}"
            )

    return "\n".join(lines) + "\n"
