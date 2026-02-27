"""Prometheus metrics and structured observability helpers for Phase 10."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

try:
    from prometheus_client import Counter, Gauge, Histogram, generate_latest, CollectorRegistry, REGISTRY  # type: ignore
except Exception:  # pragma: no cover
    Counter = None  # type: ignore[assignment]
    Gauge = None  # type: ignore[assignment]
    Histogram = None  # type: ignore[assignment]
    generate_latest = None  # type: ignore[assignment]
    CollectorRegistry = None  # type: ignore[assignment]
    REGISTRY = None  # type: ignore[assignment]


@dataclass(slots=True)
class MetricsBundle:
    generation_duration_seconds: Any = None
    queue_length: Any = None
    gpu_memory_bytes: Any = None
    model_cache_hits: Any = None
    api_errors_total: Any = None


def create_metrics_bundle(namespace: str = "flux2") -> MetricsBundle:
    """Create metrics bundle, handling duplicate registration gracefully."""
    if Histogram is None:
        return MetricsBundle()

    # Check if metrics are already registered
    if REGISTRY is not None:
        existing_names = {name for name, _ in REGISTRY._names_to_collectors.items() if name.startswith(namespace)}
        if existing_names:
            # Metrics already registered, return empty bundle
            # This happens on Streamlit reruns
            return MetricsBundle()
    
    try:
        return MetricsBundle(
            generation_duration_seconds=Histogram(
                f"{namespace}_generation_duration_seconds",
                "Generation duration in seconds",
                buckets=(0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0),
            ),
            queue_length=Gauge(
                f"{namespace}_queue_length",
                "Current request queue length",
            ),
            gpu_memory_bytes=Gauge(
                f"{namespace}_gpu_memory_bytes",
                "Current GPU memory usage in bytes",
            ),
            model_cache_hits=Counter(
                f"{namespace}_model_cache_hits_total",
                "Model generation cache hit count",
            ),
            api_errors_total=Counter(
                f"{namespace}_api_errors_total",
                "External API error count",
                labelnames=("provider",),
            ),
        )
    except ValueError:
        # Metrics already registered (Streamlit rerun), return empty bundle
        return MetricsBundle()


_GLOBAL_METRICS = create_metrics_bundle()


def observe_generation_duration(seconds: float) -> None:
    metric = _GLOBAL_METRICS.generation_duration_seconds
    if metric is not None:
        metric.observe(max(0.0, float(seconds)))


def set_queue_length(length: int) -> None:
    metric = _GLOBAL_METRICS.queue_length
    if metric is not None:
        metric.set(max(0, int(length)))


def set_gpu_memory_bytes(used_bytes: int) -> None:
    metric = _GLOBAL_METRICS.gpu_memory_bytes
    if metric is not None:
        metric.set(max(0, int(used_bytes)))


def inc_model_cache_hits() -> None:
    metric = _GLOBAL_METRICS.model_cache_hits
    if metric is not None:
        metric.inc()


def inc_api_errors(provider: str = "unknown") -> None:
    metric = _GLOBAL_METRICS.api_errors_total
    if metric is not None:
        metric.labels(provider=provider).inc()


def export_prometheus_metrics() -> bytes:
    if generate_latest is None:
        return b""
    return generate_latest()


def log_generation_completed(
    logger: logging.Logger,
    duration_ms: int,
    model: str,
    prompt_length: int,
    user_session: str,
) -> None:
    """ELK-friendly structured log line for completed generation."""
    logger.info(
        "Generation completed",
        extra={
            "duration_ms": int(duration_ms),
            "model": str(model),
            "prompt_length": int(prompt_length),
            "user_session": str(user_session),
        },
    )
