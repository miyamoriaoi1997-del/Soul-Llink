from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from pcltm.monitoring.models import Issue, Snapshot


def test_snapshot_has_versioned_json_safe_contract() -> None:
    snapshot = Snapshot.empty(
        generated_at=datetime(2026, 7, 13, 22, 0, tzinfo=timezone.utc),
        duration_ms=12,
    )

    payload = snapshot.to_dict()

    assert payload == {
        "schema_version": 1,
        "ok": True,
        "generated_at": "2026-07-13T22:00:00+00:00",
        "duration_ms": 12,
        "runtime": {},
        "context": {},
        "memory": {},
        "persona": {},
        "router": {},
        "issues": [],
    }
    assert json.loads(json.dumps(payload)) == payload


def test_issue_has_stable_json_contract() -> None:
    issue = Issue(
        severity="warning",
        code="SOURCE_STALE",
        source="context",
        message="Telemetry is stale",
        timestamp="2026-07-13T22:00:00+00:00",
        remediation="Refresh the source",
    )

    assert issue.to_dict() == {
        "severity": "warning",
        "code": "SOURCE_STALE",
        "source": "context",
        "message": "Telemetry is stale",
        "timestamp": "2026-07-13T22:00:00+00:00",
        "remediation": "Refresh the source",
    }


@pytest.mark.parametrize(
    "forbidden",
    ["prompt", "content", "memory_text", "api_key", "authorization", "tool_output"],
)
def test_snapshot_rejects_forbidden_fields_at_any_depth(forbidden: str) -> None:
    with pytest.raises(ValueError, match="forbidden monitoring field"):
        Snapshot(
            runtime={"nested": {forbidden: "opaque-value"}},
        )


def test_snapshot_ok_is_false_when_issue_is_error() -> None:
    snapshot = Snapshot(
        issues=[
            Issue(
                severity="error",
                code="COLLECTOR_FAILED",
                source="runtime",
                message="Collector failed",
                timestamp="2026-07-13T22:00:00+00:00",
                remediation="Inspect collector health",
            )
        ]
    )

    assert snapshot.to_dict()["ok"] is False
