from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import torch


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ErrorCategory(str, Enum):
    MODEL_NOT_FOUND = "Model weights not found"
    INSUFFICIENT_VRAM = "GPU memory exceeded"
    INVALID_PROMPT = "Unsafe or invalid prompt"
    API_RATE_LIMIT = "Rate limit reached"
    NETWORK_TIMEOUT = "API request failed"
    FILE_CORRUPTION = "Corrupted model file"
    CONFIGURATION_ERROR = "Invalid settings"
    SAFETY_BLOCKED = "Safety policy blocked request"
    CACHE_ERROR = "Cache operation failed"
    UNKNOWN = "Unknown runtime error"


@dataclass(slots=True)
class SuggestedAction:
    button_label: str
    description: str
    action_id: str


@dataclass(slots=True)
class ErrorContext:
    category: ErrorCategory
    severity: Severity
    message: str
    location: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    technical_details: str = ""
    suggested_actions: list[SuggestedAction] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def classify_exception(exc: Exception, location: str, metadata: dict[str, Any] | None = None) -> ErrorContext:
    metadata = metadata or {}
    lower_msg = str(exc).lower()

    category = ErrorCategory.UNKNOWN
    severity = Severity.ERROR
    message = "An unexpected error occurred."

    if isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in lower_msg or "cuda" in lower_msg and "memory" in lower_msg:
        category = ErrorCategory.INSUFFICIENT_VRAM
        severity = Severity.WARNING
        message = "Not enough GPU memory for the current settings."
    elif "no such file" in lower_msg or "does not exist" in lower_msg or "not found" in lower_msg:
        category = ErrorCategory.MODEL_NOT_FOUND
        severity = Severity.ERROR
        message = "Required model files were not found on disk."
    elif "timed out" in lower_msg or "timeout" in lower_msg:
        category = ErrorCategory.NETWORK_TIMEOUT
        severity = Severity.WARNING
        message = "The request timed out."
    elif "429" in lower_msg or "rate limit" in lower_msg:
        category = ErrorCategory.API_RATE_LIMIT
        severity = Severity.WARNING
        message = "The upstream API rate limit was reached."
    elif "safety" in lower_msg or "nsfw" in lower_msg:
        category = ErrorCategory.SAFETY_BLOCKED
        severity = Severity.INFO
        message = "The request was blocked by safety checks."
    elif "config" in lower_msg or "invalid" in lower_msg or "valueerror" in lower_msg:
        category = ErrorCategory.CONFIGURATION_ERROR
        severity = Severity.WARNING
        message = "One or more settings are invalid for this run."

    return ErrorContext(
        category=category,
        severity=severity,
        message=message,
        location=location,
        technical_details=_safe_trace(exc),
        suggested_actions=get_default_suggested_actions(category),
        metadata=metadata,
    )


def get_default_suggested_actions(category: ErrorCategory) -> list[SuggestedAction]:
    mapping: dict[ErrorCategory, list[SuggestedAction]] = {
        ErrorCategory.INSUFFICIENT_VRAM: [
            SuggestedAction("Retry", "Retry once after cache cleanup.", "retry"),
            SuggestedAction("Try Alternative", "Switch to a lighter model and lower quality.", "fallback_model_quality"),
        ],
        ErrorCategory.MODEL_NOT_FOUND: [
            SuggestedAction("Retry", "Retry after confirming model paths.", "retry"),
            SuggestedAction("Try Alternative", "Switch to Klein 4B if available.", "fallback_model"),
        ],
        ErrorCategory.NETWORK_TIMEOUT: [
            SuggestedAction("Retry", "Retry with backoff.", "retry_backoff"),
            SuggestedAction("Try Alternative", "Use local backend if possible.", "fallback_backend"),
        ],
        ErrorCategory.API_RATE_LIMIT: [
            SuggestedAction("Retry", "Retry later with backoff.", "retry_backoff"),
            SuggestedAction("Try Alternative", "Switch to local backend.", "fallback_backend"),
        ],
        ErrorCategory.SAFETY_BLOCKED: [
            SuggestedAction("Retry", "Adjust prompt and retry.", "retry"),
        ],
        ErrorCategory.CONFIGURATION_ERROR: [
            SuggestedAction("Retry", "Reset parameters and retry.", "retry_defaults"),
            SuggestedAction("Try Alternative", "Use recommended model defaults.", "fallback_quality"),
        ],
        ErrorCategory.UNKNOWN: [
            SuggestedAction("Retry", "Retry the operation.", "retry"),
            SuggestedAction("Try Alternative", "Use safer defaults.", "fallback_quality"),
        ],
    }
    return mapping.get(category, mapping[ErrorCategory.UNKNOWN])


def truncate_sensitive(text: str, max_len: int = 800) -> str:
    redacted = text
    for token_name in ["OPENROUTER_API_KEY", "HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"]:
        redacted = redacted.replace(token_name, f"{token_name}=***")
    if len(redacted) > max_len:
        return redacted[:max_len] + "..."
    return redacted


def _safe_trace(exc: Exception) -> str:
    return truncate_sensitive("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
