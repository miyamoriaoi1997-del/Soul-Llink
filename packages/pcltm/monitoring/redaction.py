"""Deny-by-default sanitization for read-only monitoring data."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_DEFAULT_MAX_STRING_LENGTH = 256
_DEFAULT_MAX_ARRAY_LENGTH = 100
_DEFAULT_MAX_DEPTH = 6

_SOURCE_ALLOWLISTS: dict[str, frozenset[str]] = {
    "runtime": frozenset(
        {
            "status",
            "db_exists",
            "memfs_exists",
            "schema_version",
            "observed_at",
            "age_seconds",
            "duration_ms",
            "issues",
        }
    ),
    "context": frozenset(
        {
            "within_budget",
            "model_context_tokens",
            "rendered_chars",
            "limit_chars",
            "budget_tokens",
            "budget_buckets",
            "engine",
            "usage_ratio",
            "trigger_threshold",
            "omitted_chars",
            "truncated_count",
            "capsule_count",
            "stale",
            "source",
            "observed_at",
            "age_seconds",
            "issues",
        }
    ),
    "memory": frozenset(
        {
            "event_count",
            "summary_count",
            "record_count",
            "approved_count",
            "pending_count",
            "fts_count",
            "fts_consistent",
            "semantic_count",
            "semantic_query_ok",
            "memfs_counts",
            "scope_collision_count",
            "selection_drift_count",
            "observed_at",
            "age_seconds",
            "issues",
        }
    ),
    "persona": frozenset(
        {
            "mode",
            "state",
            "attempts",
            "successes",
            "timeouts",
            "errors",
            "fallbacks",
            "saturated",
            "latency_ms",
            "last_event_at",
            "observed_at",
            "age_seconds",
            "issues",
        }
    ),
    "router": frozenset(
        {
            "enabled",
            "ok",
            "route",
            "selected_model",
            "successes",
            "errors",
            "fallbacks",
            "fallback_used",
            "latency_ms",
            "upstream_status",
            "error_type",
            "last_event_at",
            "observed_at",
            "age_seconds",
            "issues",
        }
    ),
}

_NESTED_ALLOWLISTS: dict[str, frozenset[str]] = {
    "budget_buckets": frozenset(
        {"active_frame", "continuity", "tool_evidence", "recall", "reserve"}
    ),
    "issues": frozenset(
        {"severity", "code", "source", "message", "timestamp", "remediation"}
    ),
    "memfs_counts": frozenset({"pinned", "episodic", "semantic", "transient"}),
}


def _validate_limits(
    *, max_string_length: int, max_array_length: int, max_depth: int
) -> None:
    if max_string_length <= 0 or max_array_length <= 0 or max_depth <= 0:
        raise ValueError("monitoring sanitization bounds must be positive")


def _sanitize_value(
    value: Any,
    *,
    field_name: str,
    max_string_length: int,
    max_array_length: int,
    max_depth: int,
    depth: int,
) -> Any:
    if depth >= max_depth:
        return None
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:max_string_length]
    if isinstance(value, Mapping):
        allowed = _NESTED_ALLOWLISTS.get(field_name, frozenset())
        return {
            str(key): _sanitize_value(
                child,
                field_name=str(key),
                max_string_length=max_string_length,
                max_array_length=max_array_length,
                max_depth=max_depth,
                depth=depth + 1,
            )
            for key, child in value.items()
            if str(key) in allowed
        }
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        return [
            _sanitize_value(
                child,
                field_name=field_name,
                max_string_length=max_string_length,
                max_array_length=max_array_length,
                max_depth=max_depth,
                depth=depth + 1,
            )
            for child in value[:max_array_length]
        ]
    return None


def sanitize_source(
    source: str,
    payload: Mapping[str, Any],
    *,
    max_string_length: int = _DEFAULT_MAX_STRING_LENGTH,
    max_array_length: int = _DEFAULT_MAX_ARRAY_LENGTH,
    max_depth: int = _DEFAULT_MAX_DEPTH,
) -> dict[str, Any]:
    """Return only explicitly approved fields for a monitoring source."""
    _validate_limits(
        max_string_length=max_string_length,
        max_array_length=max_array_length,
        max_depth=max_depth,
    )
    allowed = _SOURCE_ALLOWLISTS.get(source)
    if allowed is None:
        raise ValueError(f"unknown monitoring source: {source}")
    return {
        str(key): _sanitize_value(
            value,
            field_name=str(key),
            max_string_length=max_string_length,
            max_array_length=max_array_length,
            max_depth=max_depth,
            depth=0,
        )
        for key, value in payload.items()
        if str(key) in allowed
    }


def sanitize_error(
    error: BaseException,
    *,
    max_string_length: int = _DEFAULT_MAX_STRING_LENGTH,
) -> dict[str, str]:
    """Expose only an exception type and a bounded message."""
    _validate_limits(
        max_string_length=max_string_length,
        max_array_length=1,
        max_depth=1,
    )
    return {
        "error_type": type(error).__name__[:max_string_length],
        "message": str(error)[:max_string_length],
    }


def sanitize_path(path: str | Path) -> dict[str, Any]:
    """Return path metadata without opening or reading the target."""
    resolved = Path(path).expanduser().resolve()
    exists = resolved.exists()
    if resolved.is_file():
        kind = "file"
    elif resolved.is_dir():
        kind = "directory"
    else:
        kind = "missing"
    return {"path": str(resolved), "exists": exists, "kind": kind}


__all__ = ["sanitize_error", "sanitize_path", "sanitize_source"]
