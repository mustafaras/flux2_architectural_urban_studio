from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4


class QueueItemStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class SchedulingType(str, Enum):
    IMMEDIATE = "immediate"
    SCHEDULED = "scheduled"
    RECURRING = "recurring"
    BATCH = "batch"


class QueueLane(str, Enum):
    P0_INTERACTIVE = "P0"
    P1_REVIEW_BATCH = "P1"
    P2_WORKSHOP_BATCH = "P2"
    P3_BACKGROUND = "P3"


_LANE_WEIGHTS: dict[str, int] = {
    QueueLane.P0_INTERACTIVE.value: 40,
    QueueLane.P1_REVIEW_BATCH.value: 25,
    QueueLane.P2_WORKSHOP_BATCH.value: 15,
    QueueLane.P3_BACKGROUND.value: 5,
}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class GenerationRequest:
    prompt: str
    settings: dict[str, Any]
    priority: int = 0
    scheduled_for: datetime | None = None
    scheduling_type: SchedulingType = SchedulingType.IMMEDIATE
    request_id: str = field(default_factory=lambda: f"req_{uuid4().hex[:10]}")
    created_at: datetime = field(default_factory=_utc_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    status: QueueItemStatus = QueueItemStatus.QUEUED
    error_message: str | None = None
    output_path: str | None = None
    duration_s: float | None = None
    retry_count: int = 0
    max_retries: int = 3
    tags: list[str] = field(default_factory=list)
    template_name: str | None = None
    queue_lane: str = QueueLane.P1_REVIEW_BATCH.value

    def is_due(self, now: datetime | None = None) -> bool:
        now = now or _utc_now()
        if self.scheduled_for is None:
            return True
        return self.scheduled_for <= now
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for JSON serialization"""
        d = asdict(self)
        d['scheduled_for'] = self.scheduled_for.isoformat() if self.scheduled_for else None
        d['created_at'] = self.created_at.isoformat()
        d['started_at'] = self.started_at.isoformat() if self.started_at else None
        d['finished_at'] = self.finished_at.isoformat() if self.finished_at else None
        d['status'] = self.status.value
        d['scheduling_type'] = self.scheduling_type.value
        return d
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GenerationRequest:
        """Create from dict (JSON deserialization)"""
        data = data.copy()
        if data.get('scheduled_for'):
            data['scheduled_for'] = datetime.fromisoformat(data['scheduled_for'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        if data.get('started_at'):
            data['started_at'] = datetime.fromisoformat(data['started_at'])
        if data.get('finished_at'):
            data['finished_at'] = datetime.fromisoformat(data['finished_at'])
        data['status'] = QueueItemStatus(data['status'])
        data['scheduling_type'] = SchedulingType(data.get('scheduling_type', 'immediate'))
        return cls(**data)


class GenerationQueue:
    """Priority-based generation queue with persistence and scheduling.
    
    Features:
    - Max 50 items
    - Priority-based ordering (-1, 0, 1)
    - Scheduled/delayed generation
    - Persistent state (JSON)
    - ETA prediction based on historical data
    - Retry logic with exponential backoff
    - Batch operations
    """
    
    def __init__(
        self,
        max_size: int = 50,
        persist_path: str | Path | None = None,
        gpu_concurrency_cap: int = 1,
    ):
        self.max_size = int(max_size)
        self.queue: list[GenerationRequest] = []
        self.active: GenerationRequest | None = None
        self.completed: list[GenerationRequest] = []
        self.failed: list[GenerationRequest] = []
        self.canceled: list[GenerationRequest] = []
        self.persist_path = Path(persist_path) if persist_path else None
        self._duration_history: list[float] = []
        self.gpu_concurrency_cap = max(1, int(gpu_concurrency_cap))
        
        if self.persist_path and self.persist_path.exists():
            self._load_state()

    def enqueue(self, request: GenerationRequest, priority: int = 0) -> str:
        """Add request to queue with optional priority override"""
        if len(self.queue) >= self.max_size:
            raise RuntimeError(f"Queue is full ({self.max_size}).")
        request.priority = int(priority)
        request.status = QueueItemStatus.QUEUED
        self.queue.append(request)
        self._save_state()
        return request.request_id
    
    def enqueue_batch(self, requests: list[GenerationRequest], priority: int = 0) -> list[str]:
        """Add multiple requests at once"""
        ids = []
        for req in requests:
            if len(self.queue) >= self.max_size:
                break
            req.priority = int(priority)
            req.status = QueueItemStatus.QUEUED
            self.queue.append(req)
            ids.append(req.request_id)
        self._save_state()
        return ids

    def process_next(self, now: datetime | None = None) -> GenerationRequest | None:
        """Get next due request for processing"""
        if self.active is not None:
            return None
        if self.gpu_concurrency_cap <= 0:
            return None
        idx = self._next_due_index(now)
        if idx is None:
            return None
        item = self.queue.pop(idx)
        item.status = QueueItemStatus.RUNNING
        item.started_at = _utc_now()
        self.active = item
        self._save_state()
        return item

    def mark_completed(self, request_id: str, output_path: str | None, duration_s: float | None = None) -> None:
        """Mark request as completed and track duration"""
        item = self._require_active(request_id)
        item.status = QueueItemStatus.COMPLETED
        item.finished_at = _utc_now()
        item.output_path = output_path
        item.duration_s = duration_s
        
        if duration_s is not None:
            self._duration_history.append(duration_s)
            if len(self._duration_history) > 1000:
                self._duration_history = self._duration_history[-500:]
        
        self.completed.insert(0, item)
        self.active = None
        self._save_state()

    def mark_failed(self, request_id: str, error_message: str, retry: bool = False) -> bool:
        """Mark request as failed with optional retry"""
        item = self._require_active(request_id)
        
        if retry and item.retry_count < item.max_retries:
            item.retry_count += 1
            item.status = QueueItemStatus.QUEUED
            item.error_message = None
            item.started_at = None
            self.queue.insert(0, item)
            self.active = None
            self._save_state()
            return True
        
        item.status = QueueItemStatus.FAILED
        item.finished_at = _utc_now()
        item.error_message = error_message
        self.failed.insert(0, item)
        self.active = None
        self._save_state()
        return False

    def cancel(self, request_id: str) -> bool:
        """Cancel a queued or running request"""
        if self.active is not None and self.active.request_id == request_id:
            self.active.status = QueueItemStatus.CANCELED
            self.active.finished_at = _utc_now()
            self.canceled.insert(0, self.active)
            self.active = None
            self._save_state()
            return True

        for idx, item in enumerate(self.queue):
            if item.request_id == request_id:
                item.status = QueueItemStatus.CANCELED
                item.finished_at = _utc_now()
                self.canceled.insert(0, item)
                self.queue.pop(idx)
                self._save_state()
                return True
        return False

    def reorder(self, request_id: str, direction: str) -> bool:
        """Move request up/down in queue"""
        idx = next((i for i, item in enumerate(self.queue) if item.request_id == request_id), None)
        if idx is None:
            return False
        if direction == "up" and idx > 0:
            self.queue[idx - 1], self.queue[idx] = self.queue[idx], self.queue[idx - 1]
            self._save_state()
            return True
        if direction == "down" and idx < len(self.queue) - 1:
            self.queue[idx + 1], self.queue[idx] = self.queue[idx], self.queue[idx + 1]
            self._save_state()
            return True
        return False
    
    def set_priority(self, request_id: str, new_priority: int) -> bool:
        """Update request priority"""
        item = next((i for i in self.queue if i.request_id == request_id), None)
        if item is None:
            return False
        item.priority = int(new_priority)
        self.queue.sort(key=lambda x: (-x.priority, x.created_at))
        self._save_state()
        return True

    def clear(self) -> None:
        """Clear all queues"""
        self.queue.clear()
        self.active = None
        self.completed.clear()
        self.failed.clear()
        self.canceled.clear()
        self._save_state()

    def get_status(self) -> dict[str, Any]:
        """Get queue status and ETA"""
        total_done = len(self.completed) + len(self.failed) + len(self.canceled)
        total = len(self.completed) + len(self.failed) + len(self.queue) + (1 if self.active else 0)
        progress = (total_done / total) if total > 0 else 0.0
        
        # Calculate average duration
        avg_duration = self._get_avg_duration()
        
        # ETA calculation: (remaining items + 50%) of average
        remaining_queued = len(self.queue)
        active_bonus = 0.5 if self.active else 0.0
        eta_s = (remaining_queued + active_bonus) * avg_duration

        lane_counts: dict[str, int] = {lane.value: 0 for lane in QueueLane}
        for item in self.queue:
            lane = str(item.queue_lane or QueueLane.P1_REVIEW_BATCH.value)
            lane_counts[lane] = lane_counts.get(lane, 0) + 1

        lane_eta_s = {
            lane: round(count * avg_duration, 2)
            for lane, count in lane_counts.items()
        }
        
        return {
            "queued": len(self.queue),
            "running": 1 if self.active else 0,
            "completed": len(self.completed),
            "failed": len(self.failed),
            "canceled": len(self.canceled),
            "progress": progress,
            "progress_pct": int(progress * 100),
            "eta_s": eta_s,
            "eta_text": self._format_eta(eta_s),
            "avg_duration_s": avg_duration,
            "max_size": self.max_size,
            "current_size": len(self.queue),
            "success_rate": self._get_success_rate(),
            "lane_counts": lane_counts,
            "lane_eta_s": lane_eta_s,
            "gpu_concurrency_cap": self.gpu_concurrency_cap,
        }

    def get_request(self, request_id: str) -> GenerationRequest | None:
        """Find request by ID in any list"""
        if self.active and self.active.request_id == request_id:
            return self.active
        for item in self.queue + self.completed + self.failed + self.canceled:
            if item.request_id == request_id:
                return item
        return None
    
    def get_templates(self) -> dict[str, dict[str, Any]]:
        """Get saved templates (from completed successful requests)"""
        templates = {}
        for item in self.completed[:50]:
            if item.template_name:
                templates[item.template_name] = item.settings
        return templates
    
    def scale_priority_by_wait_time(self) -> None:
        """Boost priority of items waiting longest"""
        now = _utc_now()
        for item in self.queue:
            wait_seconds = (now - item.created_at).total_seconds()
            # Boost by 1 for every 10 minutes waiting
            wait_bonus = int(wait_seconds / 600)
            if wait_bonus > 0:
                item.priority = min(item.priority + wait_bonus, 2)
        self.queue.sort(key=lambda x: (-x.priority, x.created_at))
        self._save_state()

    def _get_avg_duration(self) -> float:
        """Calculate average duration from history"""
        if not self._duration_history:
            return 5.0  # Default 5 seconds
        return sum(self._duration_history) / len(self._duration_history)
    
    def _get_success_rate(self) -> float:
        """Get success rate as percentage"""
        total = len(self.completed) + len(self.failed)
        if total == 0:
            return 0.0
        return (len(self.completed) / total) * 100
    
    @staticmethod
    def _format_eta(seconds: float) -> str:
        """Format seconds into human-readable ETA"""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            return f"{int(seconds / 60)}m {int(seconds % 60)}s"
        else:
            hours = int(seconds / 3600)
            minutes = int((seconds % 3600) / 60)
            return f"{hours}h {minutes}m"

    def _next_due_index(self, now: datetime | None = None) -> int | None:
        """Find next due request by priority and schedule"""
        now = now or _utc_now()
        due_items: list[tuple[int, datetime, int, float]] = []
        for idx, item in enumerate(self.queue):
            if item.is_due(now):
                scheduled = item.scheduled_for or item.created_at
                lane_weight = _LANE_WEIGHTS.get(str(item.queue_lane), _LANE_WEIGHTS[QueueLane.P1_REVIEW_BATCH.value])
                wait_seconds = max(0.0, (now - item.created_at).total_seconds())
                wait_bonus = min(20.0, wait_seconds / 120.0)
                score = lane_weight + (int(item.priority) * 10) + wait_bonus
                due_items.append((idx, scheduled, int(item.priority), score))

        if not due_items:
            return None

        due_items.sort(key=lambda row: (-row[3], row[1], -row[2]))
        return due_items[0][0]

    def set_gpu_concurrency_cap(self, cap: int) -> None:
        """Set GPU concurrency cap for queue processing."""
        self.gpu_concurrency_cap = max(1, int(cap))
        self._save_state()

    def apply_saturation_fallback(self, threshold: int = 20) -> int:
        """Tag queued requests with fallback profile under heavy saturation."""
        if len(self.queue) < max(1, int(threshold)):
            return 0
        changed = 0
        for item in self.queue:
            if not isinstance(item.settings, dict):
                continue
            if item.settings.get("fallback_profile") == "saturation-safe":
                continue
            item.settings["fallback_profile"] = "saturation-safe"
            changed += 1
        self._save_state()
        return changed

    def _require_active(self, request_id: str) -> GenerationRequest:
        """Get active request or raise"""
        if self.active is None or self.active.request_id != request_id:
            raise RuntimeError(f"Active request not found: {request_id}")
        return self.active
    
    def _save_state(self) -> None:
        """Persist queue state to JSON"""
        if not self.persist_path:
            return
        
        state = {
            "queue": [r.to_dict() for r in self.queue],
            "active": self.active.to_dict() if self.active else None,
            "completed": [r.to_dict() for r in self.completed[:100]],
            "failed": [r.to_dict() for r in self.failed[:50]],
            "canceled": [r.to_dict() for r in self.canceled[:50]],
            "duration_history": self._duration_history[-200:],
            "saved_at": _utc_now().isoformat(),
        }
        
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.persist_path, 'w') as f:
            json.dump(state, f, indent=2)
    
    def _load_state(self) -> None:
        """Load queue state from JSON"""
        if not self.persist_path or not self.persist_path.exists():
            return
        
        try:
            with open(self.persist_path, 'r') as f:
                state = json.load(f)
            
            self.queue = [GenerationRequest.from_dict(r) for r in state.get("queue", [])]
            if state.get("active"):
                self.active = GenerationRequest.from_dict(state["active"])
            self.completed = [GenerationRequest.from_dict(r) for r in state.get("completed", [])]
            self.failed = [GenerationRequest.from_dict(r) for r in state.get("failed", [])]
            self.canceled = [GenerationRequest.from_dict(r) for r in state.get("canceled", [])]
            self._duration_history = state.get("duration_history", [])
        except Exception:
            pass  # Silently fail if state file corrupted
