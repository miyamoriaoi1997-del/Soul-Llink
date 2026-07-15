from __future__ import annotations

from pathlib import Path

import pytest

from pcltm.monitoring.redaction import (
    sanitize_error,
    sanitize_path,
    sanitize_source,
)


OPAQUE_SENTINEL = "opaque-sentinel-value-7f31"


@pytest.mark.parametrize(
    ("source", "payload", "expected"),
    [
        (
            "runtime",
            {
                "status": "healthy",
                "db_exists": True,
                "schema_version": 3,
                "prompt": OPAQUE_SENTINEL,
                "unknown": OPAQUE_SENTINEL,
            },
            {"status": "healthy", "db_exists": True, "schema_version": 3},
        ),
        (
            "context",
            {
                "within_budget": True,
                "rendered_chars": 120,
                "budget_tokens": 256,
                "content": OPAQUE_SENTINEL,
            },
            {"within_budget": True, "rendered_chars": 120, "budget_tokens": 256},
        ),
        (
            "memory",
            {
                "approved_count": 4,
                "pending_count": 1,
                "fts_consistent": True,
                "memory_text": OPAQUE_SENTINEL,
            },
            {"approved_count": 4, "pending_count": 1, "fts_consistent": True},
        ),
        (
            "persona",
            {
                "mode": "work",
                "fallbacks": 2,
                "errors": 0,
                "input_text": OPAQUE_SENTINEL,
            },
            {"mode": "work", "fallbacks": 2, "errors": 0},
        ),
        (
            "router",
            {
                "ok": True,
                "latency_ms": 18,
                "fallback_used": False,
                "authorization": OPAQUE_SENTINEL,
            },
            {"ok": True, "latency_ms": 18, "fallback_used": False},
        ),
    ],
)
def test_source_allowlists_drop_unknown_and_forbidden_fields(
    source: str,
    payload: dict[str, object],
    expected: dict[str, object],
) -> None:
    sanitized = sanitize_source(source, payload)

    assert sanitized == expected
    assert OPAQUE_SENTINEL not in repr(sanitized)


def test_unknown_source_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown monitoring source"):
        sanitize_source("other", {"ok": True})


def test_nested_values_are_bounded_without_exposing_unknown_fields() -> None:
    sanitized = sanitize_source(
        "runtime",
        {
            "status": "x" * 600,
            "issues": [
                {"code": "A", "severity": "warning", "message": "m" * 600},
                {"code": "B", "severity": "info", "message": "ok"},
                {"code": "C", "severity": "info", "message": OPAQUE_SENTINEL},
            ],
        },
        max_string_length=32,
        max_array_length=2,
    )

    assert sanitized["status"] == "x" * 32
    assert len(sanitized["issues"]) == 2
    assert sanitized["issues"][0] == {
        "code": "A",
        "severity": "warning",
        "message": "m" * 32,
    }
    assert OPAQUE_SENTINEL not in repr(sanitized)


def test_sanitize_error_returns_only_type_and_bounded_message() -> None:
    error = RuntimeError(f"collector failed: {OPAQUE_SENTINEL}")

    sanitized = sanitize_error(error, max_string_length=18)

    assert sanitized == {
        "error_type": "RuntimeError",
        "message": "collector failed: ",
    }
    assert OPAQUE_SENTINEL not in repr(sanitized)


def test_sanitize_path_normalizes_without_reading_file(tmp_path: Path) -> None:
    source = tmp_path / "nested" / "runtime.db"
    source.parent.mkdir()
    source.write_text(OPAQUE_SENTINEL, encoding="utf-8")

    sanitized = sanitize_path(source)

    assert sanitized == {
        "path": str(source.resolve()),
        "exists": True,
        "kind": "file",
    }
    assert OPAQUE_SENTINEL not in repr(sanitized)


def test_invalid_bounds_fail_closed() -> None:
    with pytest.raises(ValueError, match="positive"):
        sanitize_source("runtime", {"status": "ok"}, max_string_length=0)
