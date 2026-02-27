# FLUX.2 Error Handling Guide

This document describes how FLUX.2 classifies errors, presents user-friendly recovery actions, and records diagnostics without exposing sensitive data.

## Goals

- Reduce error impact time with actionable UI recovery.
- Prevent raw tracebacks from being shown to end users.
- Keep a compact, sanitized operation/error history.
- Support graceful degradation under low-resource conditions.

## Architecture

## 1) Error Classification

File: `src/flux2/error_types.py`

- `Severity`: `INFO`, `WARNING`, `ERROR`, `CRITICAL`
- `ErrorCategory`:
  - `MODEL_NOT_FOUND`
  - `INSUFFICIENT_VRAM`
  - `INVALID_PROMPT`
  - `API_RATE_LIMIT`
  - `NETWORK_TIMEOUT`
  - `FILE_CORRUPTION`
  - `CONFIGURATION_ERROR`
  - `SAFETY_BLOCKED`
  - `CACHE_ERROR`
  - `UNKNOWN`

`classify_exception()` converts low-level exceptions into a structured `ErrorContext` with:

- Safe user message
- Severity and category
- Source location
- Suggested actions
- Redacted technical details

## 2) Logging and Sanitization

File: `src/flux2/logging_config.py`

- Centralized logger setup via `configure_flux2_logging()`.
- Rotating file logs:
  - Location: `logs/flux2.log`
  - `maxBytes=10MB`
  - `backupCount=5`
- Sensitive values are redacted if keys include `token`, `key`, `secret`, `password`.
- Recent operations are kept in memory (`maxlen=50`) for dashboard visibility.

## 3) User Recovery UI

File: `ui/error_handler.py`

`display_recoverable_error()` renders:

- Category-based user message
- Suggested solution text
- Action buttons:
  - `Retry`
  - `Try Alternative`
  - `View Logs`
  - `Report Issue` (JSON download)
- Technical details section (expandable)

The module also stores a session-scoped error history (last 50 user-facing errors), and provides:

- `get_error_history()`
- `clear_error_history()`

## 4) Graceful Degradation Rules

### OOM / VRAM pressure

- Reduce dimensions toward safe defaults (`1024 -> 768 -> 512` behavior via bounded reduction).
- Reduce steps to a safe range (`50 -> 25 -> 4` style bounded clamp).
- Fallback model strategy:
  - `*9b -> *4b`
  - `flux.2-dev -> flux.2-klein-4b`

### API timeout / rate limit

- Suggest retry with backoff.
- Switch upsampling backend to local/none when needed.

### Missing model

- Suggest path check and fallback model switch.

## 5) Recovery Snapshot

Recovery snapshots are written to:

- `outputs/.flux2_recovery_state.json`

Usage:

- Saved before risky generation/edit operations.
- Restorable from Settings → Session Tools.
- Clearable via session tools once recovered.

## 6) Debug Dashboard

File: `ui/pages/debug.py`

Dashboard includes:

- Last 50 operation/error records
- Error count and category frequency
- Average operation duration (`details.time_s`)
- Recovery success rate (`*.start` vs `*.finish` operations)
- Optional memory trend from runtime samples
- Export actions:
  - JSON export (`flux2_debug_operations.json`)
  - CSV export (`flux2_debug_operations.csv`)

## Integration Points

Primary page integrations:

- `ui/pages/generator.py`
- `ui/pages/editor.py`
- `ui/pages/queue.py`

Each flow:

1. Saves recovery snapshot
2. Wraps risky actions in `try/except`
3. Calls `classify_exception()`
4. Logs via `log_error()`
5. Displays `display_recoverable_error()`

## Validation Checklist

- Trigger OOM (high resolution/steps) and verify fallback actions appear.
- Trigger missing model path and verify category + suggestion.
- Trigger network timeout/rate limit and verify backend fallback suggestion.
- Verify no raw traceback is shown to end user surface.
- Verify `logs/flux2.log` rotates and sensitive keys are redacted.
- Verify Settings allows recovery snapshot restore/clear.
- Verify Debug dashboard exports JSON and CSV.

## Operational Tips

- Keep `advanced_mode` off for most users; defaults are safer.
- For constrained GPUs, prefer `flux.2-klein-4b` with lower resolution.
- Use debug dashboard exports when filing bug reports.
