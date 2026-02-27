"""Performance metrics collection utilities for FLUX.2 Streamlit runtime."""

from __future__ import annotations

import json
import subprocess
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

import torch

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None  # type: ignore[assignment]


_SESSION_KEY = "_flux2_performance_metrics"
_LOCK = threading.RLock()


@dataclass(slots=True)
class PhaseTiming:
    name: str
    seconds: float
    timestamp: float


@dataclass(slots=True)
class RuntimeSample:
    timestamp: float
    gpu_utilization_percent: float | None = None
    gpu_memory_used_mb: float | None = None
    gpu_memory_total_mb: float | None = None
    gpu_temperature_c: float | None = None
    gpu_power_w: float | None = None
    cuda_reserved_mb: float | None = None
    cuda_allocated_mb: float | None = None


@dataclass(slots=True)
class QueueMetrics:
    queued: int = 0
    max_queue: int = 0
    waits_s: list[float] = field(default_factory=list)

    def add_wait(self, wait_s: float) -> None:
        self.waits_s.append(float(wait_s))
        if len(self.waits_s) > 256:
            self.waits_s = self.waits_s[-256:]

    @property
    def avg_wait_s(self) -> float:
        if not self.waits_s:
            return 0.0
        return sum(self.waits_s) / len(self.waits_s)


@dataclass(slots=True)
class PerformanceState:
    phase_timings: list[dict[str, Any]] = field(default_factory=list)
    runtime_samples: list[dict[str, Any]] = field(default_factory=list)
    queue: dict[str, Any] = field(default_factory=lambda: asdict(QueueMetrics()))
    counters: dict[str, int] = field(
        default_factory=lambda: {
            "loads": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "requests": 0,
            "errors": 0,
        }
    )


class PerformanceMetricsCollector:
    """Thread-safe in-session performance metrics store."""

    def __init__(self, max_phase_entries: int = 2000, max_runtime_samples: int = 7200) -> None:
        self._max_phase_entries = max_phase_entries
        self._max_runtime_samples = max_runtime_samples

    def _state(self) -> dict[str, Any]:
        if st is None:
            if not hasattr(self, "_fallback_state"):
                self._fallback_state = asdict(PerformanceState())
            return self._fallback_state
        if _SESSION_KEY not in st.session_state:
            st.session_state[_SESSION_KEY] = asdict(PerformanceState())
        return st.session_state[_SESSION_KEY]

    @contextmanager
    def track_phase(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.record_phase(name, time.perf_counter() - started)

    def record_phase(self, name: str, seconds: float) -> None:
        payload = asdict(PhaseTiming(name=name, seconds=float(seconds), timestamp=time.time()))
        with _LOCK:
            state = self._state()
            state["phase_timings"].append(payload)
            if len(state["phase_timings"]) > self._max_phase_entries:
                state["phase_timings"] = state["phase_timings"][-self._max_phase_entries :]

    def increment(self, key: str, amount: int = 1) -> None:
        with _LOCK:
            state = self._state()
            counters = state["counters"]
            counters[key] = int(counters.get(key, 0)) + amount

    def set_queue_depth(self, depth: int) -> None:
        with _LOCK:
            state = self._state()
            queue = state["queue"]
            queue["queued"] = int(depth)
            queue["max_queue"] = max(int(queue.get("max_queue", 0)), int(depth))

    def record_wait_time(self, wait_s: float) -> None:
        with _LOCK:
            state = self._state()
            queue = QueueMetrics(**state["queue"])
            queue.add_wait(wait_s)
            state["queue"] = asdict(queue)

    def sample_runtime(self) -> dict[str, Any]:
        sample = self._probe_runtime()
        with _LOCK:
            state = self._state()
            state["runtime_samples"].append(sample)
            if len(state["runtime_samples"]) > self._max_runtime_samples:
                state["runtime_samples"] = state["runtime_samples"][-self._max_runtime_samples :]
        return sample

    def snapshot(self) -> dict[str, Any]:
        with _LOCK:
            state = self._state()
            phase_timings = list(state["phase_timings"])
            runtime_samples = list(state["runtime_samples"])
            counters = dict(state["counters"])
            queue = dict(state["queue"])

        summary = self._build_summary(phase_timings, counters, queue)
        return {
            "summary": summary,
            "phase_timings": phase_timings,
            "runtime_samples": runtime_samples,
            "counters": counters,
            "queue": queue,
        }

    def clear(self) -> None:
        with _LOCK:
            if st is not None and _SESSION_KEY in st.session_state:
                del st.session_state[_SESSION_KEY]
            if hasattr(self, "_fallback_state"):
                delattr(self, "_fallback_state")

    def export_json(self, output_path: str | Path) -> Path:
        data = self.snapshot()
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return p

    def _probe_runtime(self) -> dict[str, Any]:
        gpu_memory_total_mb = None
        gpu_memory_used_mb = None
        cuda_reserved_mb = None
        cuda_allocated_mb = None

        if torch.cuda.is_available():
            try:
                cuda_reserved_mb = torch.cuda.memory_reserved(0) / (1024 * 1024)
            except Exception:
                cuda_reserved_mb = None
            try:
                cuda_allocated_mb = torch.cuda.memory_allocated(0) / (1024 * 1024)
            except Exception:
                cuda_allocated_mb = None
            try:
                free, total = torch.cuda.mem_get_info(0)
                gpu_memory_total_mb = total / (1024 * 1024)
                gpu_memory_used_mb = (total - free) / (1024 * 1024)
            except Exception:
                gpu_memory_total_mb = None
                gpu_memory_used_mb = cuda_allocated_mb or cuda_reserved_mb

        nv = self._query_nvidia_smi()
        sample = RuntimeSample(
            timestamp=time.time(),
            gpu_utilization_percent=nv.get("utilization_gpu") if nv else None,
            gpu_memory_used_mb=(nv.get("memory_used") if nv and gpu_memory_used_mb is None else gpu_memory_used_mb),
            gpu_memory_total_mb=(nv.get("memory_total") if nv and gpu_memory_total_mb is None else gpu_memory_total_mb),
            gpu_temperature_c=nv.get("temperature_gpu") if nv else None,
            gpu_power_w=nv.get("power_draw") if nv else None,
            cuda_reserved_mb=cuda_reserved_mb,
            cuda_allocated_mb=cuda_allocated_mb,
        )
        return asdict(sample)

    @staticmethod
    def _query_nvidia_smi() -> dict[str, float] | None:
        try:
            cmd = [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1.5, check=False)
            if proc.returncode != 0 or not proc.stdout.strip():
                return None
            first = proc.stdout.strip().splitlines()[0]
            parts = [p.strip() for p in first.split(",")]
            if len(parts) < 5:
                return None
            return {
                "utilization_gpu": float(parts[0]),
                "memory_used": float(parts[1]),
                "memory_total": float(parts[2]),
                "temperature_gpu": float(parts[3]),
                "power_draw": float(parts[4]),
            }
        except Exception:
            return None

    @staticmethod
    def _build_summary(
        phase_timings: list[dict[str, Any]],
        counters: dict[str, int],
        queue: dict[str, Any],
    ) -> dict[str, Any]:
        grouped: dict[str, deque[float]] = {}
        for item in phase_timings:
            grouped.setdefault(item["name"], deque(maxlen=256)).append(float(item["seconds"]))

        phase_avg = {
            name: (sum(values) / len(values) if values else 0.0)
            for name, values in grouped.items()
        }

        hits = int(counters.get("cache_hits", 0))
        misses = int(counters.get("cache_misses", 0))
        total_cache = hits + misses
        hit_rate = (hits / total_cache) if total_cache else 0.0

        waits = queue.get("waits_s", [])
        avg_wait = (sum(waits) / len(waits)) if waits else 0.0

        return {
            "avg_phase_seconds": phase_avg,
            "cache_hit_rate": hit_rate,
            "avg_wait_s": avg_wait,
            "requests": int(counters.get("requests", 0)),
            "errors": int(counters.get("errors", 0)),
            "max_queue": int(queue.get("max_queue", 0)),
        }


def timed_phase(name: str, collector: PerformanceMetricsCollector) -> Callable:
    """Decorator for timing a function as a named phase."""

    def decorator(fn: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            with collector.track_phase(name):
                return fn(*args, **kwargs)

        return wrapper

    return decorator


class RequestQueueMonitor:
    """Small helper that tracks queue depth and wait-time per request."""

    def __init__(self, collector: PerformanceMetricsCollector) -> None:
        self._collector = collector
        self._queue = 0
        self._lock = threading.RLock()

    @contextmanager
    def request(self) -> Iterator[None]:
        entered = time.perf_counter()
        with self._lock:
            self._queue += 1
            self._collector.set_queue_depth(self._queue)
        try:
            yield
        finally:
            wait_s = max(0.0, time.perf_counter() - entered)
            self._collector.record_wait_time(wait_s)
            with self._lock:
                self._queue = max(0, self._queue - 1)
                self._collector.set_queue_depth(self._queue)


_GLOBAL_COLLECTOR: PerformanceMetricsCollector | None = None


def get_performance_collector() -> PerformanceMetricsCollector:
    global _GLOBAL_COLLECTOR
    if _GLOBAL_COLLECTOR is None:
        _GLOBAL_COLLECTOR = PerformanceMetricsCollector()
    return _GLOBAL_COLLECTOR
