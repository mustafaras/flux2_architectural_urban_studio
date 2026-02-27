"""Distributed model loading and request routing primitives for Phase 10."""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any


@dataclass(slots=True)
class CircuitBreakerPolicy:
    failure_threshold: int = 3
    recovery_timeout_s: int = 30


@dataclass(slots=True)
class ModelServiceEndpoint:
    service_name: str
    url: str
    model_key: str
    weight: int = 1
    healthy: bool = True
    failure_count: int = 0
    opened_until_ts: float = 0.0

    def is_available(self, now_ts: float | None = None) -> bool:
        now_ts = now_ts or time.time()
        if not self.healthy:
            return False
        return now_ts >= self.opened_until_ts


@dataclass(order=True)
class PrioritizedRequest:
    priority: int
    created_ts: float
    request_id: str = field(compare=False)
    model_key: str = field(compare=False)
    payload: dict[str, Any] = field(compare=False)


class RoundRobinModelRouter:
    """Model-aware round-robin router with circuit breaker support."""

    def __init__(self, policy: CircuitBreakerPolicy | None = None) -> None:
        self._policy = policy or CircuitBreakerPolicy()
        self._routes: dict[str, list[ModelServiceEndpoint]] = {}
        self._cursor: dict[str, int] = {}
        self._lock = RLock()

    def register(self, endpoint: ModelServiceEndpoint) -> None:
        with self._lock:
            self._routes.setdefault(endpoint.model_key, []).append(endpoint)
            self._cursor.setdefault(endpoint.model_key, 0)

    def list_endpoints(self, model_key: str) -> list[ModelServiceEndpoint]:
        with self._lock:
            return list(self._routes.get(model_key, []))

    def choose_endpoint(self, model_key: str) -> ModelServiceEndpoint:
        with self._lock:
            endpoints = self._routes.get(model_key, [])
            if not endpoints:
                raise RuntimeError(f"No endpoints registered for model: {model_key}")

            start = self._cursor.get(model_key, 0)
            total = len(endpoints)
            now_ts = time.time()

            for offset in range(total):
                idx = (start + offset) % total
                endpoint = endpoints[idx]
                if endpoint.is_available(now_ts):
                    self._cursor[model_key] = (idx + 1) % total
                    return endpoint

            raise RuntimeError(f"All endpoints unavailable for model: {model_key}")

    def mark_success(self, endpoint: ModelServiceEndpoint) -> None:
        with self._lock:
            endpoint.failure_count = 0
            endpoint.opened_until_ts = 0.0
            endpoint.healthy = True

    def mark_failure(self, endpoint: ModelServiceEndpoint) -> None:
        with self._lock:
            endpoint.failure_count += 1
            if endpoint.failure_count >= self._policy.failure_threshold:
                endpoint.opened_until_ts = time.time() + self._policy.recovery_timeout_s


class PriorityRequestQueue:
    """Priority queue for distributed inference requests."""

    def __init__(self) -> None:
        self._heap: list[PrioritizedRequest] = []
        self._lock = RLock()

    def enqueue(self, request_id: str, model_key: str, payload: dict[str, Any], priority: int = 0) -> None:
        item = PrioritizedRequest(
            priority=-int(priority),
            created_ts=time.time(),
            request_id=request_id,
            model_key=model_key,
            payload=payload,
        )
        with self._lock:
            heapq.heappush(self._heap, item)

    def dequeue(self) -> PrioritizedRequest | None:
        with self._lock:
            if not self._heap:
                return None
            return heapq.heappop(self._heap)

    def size(self) -> int:
        with self._lock:
            return len(self._heap)


class DistributedModelLoader:
    """High-level orchestrator combining routing and priority queueing."""

    def __init__(self, router: RoundRobinModelRouter | None = None, queue: PriorityRequestQueue | None = None) -> None:
        self.router = router or RoundRobinModelRouter()
        self.queue = queue or PriorityRequestQueue()

    def register_endpoint(self, service_name: str, model_key: str, url: str, weight: int = 1) -> ModelServiceEndpoint:
        endpoint = ModelServiceEndpoint(
            service_name=service_name,
            model_key=model_key,
            url=url,
            weight=weight,
        )
        self.router.register(endpoint)
        return endpoint

    def enqueue_request(self, request_id: str, model_key: str, payload: dict[str, Any], priority: int = 0) -> None:
        self.queue.enqueue(request_id=request_id, model_key=model_key, payload=payload, priority=priority)

    def dispatch_next(self) -> tuple[PrioritizedRequest, ModelServiceEndpoint] | None:
        request = self.queue.dequeue()
        if request is None:
            return None
        endpoint = self.router.choose_endpoint(request.model_key)
        return request, endpoint
