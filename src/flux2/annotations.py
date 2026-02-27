from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class AnnotationComment:
    comment_id: str
    author: str
    text: str
    created_at: str


@dataclass(slots=True)
class AnnotationThread:
    thread_id: str
    option_id: str
    pin_x_pct: float
    pin_y_pct: float
    label: str
    related_option_ids: list[str] = field(default_factory=list)
    status: str = "open"
    created_by: str = ""
    created_at: str = field(default_factory=_utc_now)
    resolved_at: str | None = None
    resolved_by: str | None = None
    comments: list[AnnotationComment] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "thread_id": self.thread_id,
            "option_id": self.option_id,
            "pin_x_pct": self.pin_x_pct,
            "pin_y_pct": self.pin_y_pct,
            "label": self.label,
            "related_option_ids": list(self.related_option_ids),
            "status": self.status,
            "created_by": self.created_by,
            "created_at": self.created_at,
            "resolved_at": self.resolved_at,
            "resolved_by": self.resolved_by,
            "comments": [
                {
                    "comment_id": c.comment_id,
                    "author": c.author,
                    "text": c.text,
                    "created_at": c.created_at,
                }
                for c in self.comments
            ],
        }


def create_thread(
    *,
    option_id: str,
    pin_x_pct: float,
    pin_y_pct: float,
    label: str,
    author: str,
    note: str,
    related_option_ids: list[str] | None = None,
) -> dict[str, Any]:
    thread = AnnotationThread(
        thread_id=f"ann_{uuid4().hex[:10]}",
        option_id=str(option_id),
        pin_x_pct=float(pin_x_pct),
        pin_y_pct=float(pin_y_pct),
        label=str(label).strip() or "annotation",
        related_option_ids=list(related_option_ids or []),
        created_by=str(author).strip() or "reviewer",
    )
    initial = AnnotationComment(
        comment_id=f"com_{uuid4().hex[:10]}",
        author=str(author).strip() or "reviewer",
        text=str(note).strip(),
        created_at=_utc_now(),
    )
    thread.comments.append(initial)
    return thread.to_dict()


def add_comment(thread: dict[str, Any], *, author: str, text: str) -> dict[str, Any]:
    comments = thread.get("comments", [])
    if not isinstance(comments, list):
        comments = []
    comments.append(
        {
            "comment_id": f"com_{uuid4().hex[:10]}",
            "author": str(author).strip() or "reviewer",
            "text": str(text).strip(),
            "created_at": _utc_now(),
        }
    )
    thread["comments"] = comments
    return thread


def set_resolved(thread: dict[str, Any], *, resolved: bool, actor: str) -> dict[str, Any]:
    if resolved:
        thread["status"] = "resolved"
        thread["resolved_at"] = _utc_now()
        thread["resolved_by"] = str(actor).strip() or "reviewer"
    else:
        thread["status"] = "open"
        thread["resolved_at"] = None
        thread["resolved_by"] = None
    return thread


def build_annotation_summary(threads: list[dict[str, Any]]) -> dict[str, Any]:
    by_option: dict[str, int] = {}
    open_count = 0
    resolved_count = 0

    for thread in threads:
        option_id = str(thread.get("option_id", ""))
        if option_id:
            by_option[option_id] = by_option.get(option_id, 0) + 1
        if thread.get("status") == "resolved":
            resolved_count += 1
        else:
            open_count += 1

    return {
        "total_threads": len(threads),
        "open_threads": open_count,
        "resolved_threads": resolved_count,
        "threads_by_option": by_option,
    }
