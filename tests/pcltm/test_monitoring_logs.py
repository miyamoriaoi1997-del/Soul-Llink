from __future__ import annotations

import json
from pathlib import Path

from pcltm.monitoring.logs import collect_persona_log, collect_router_log


OPAQUE = "opaque-log-value-e91c"


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_router_log_is_bounded_and_aggregated_without_raw_fields(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    _write(
        path,
        [
            {"ts": "2026-07-13T22:00:00+00:00", "route": "default", "selected_model": "m1", "ok": True, "fallback_used": False, "latency_ms": 10, "upstream_status": 200, "prompt": OPAQUE},
            {"ts": "2026-07-13T22:01:00+00:00", "route": "technical", "selected_model": "m2", "ok": False, "fallback_used": True, "latency_ms": 30, "upstream_status": 503, "error_type": "TimeoutError", "content": OPAQUE},
        ],
    )

    report = collect_router_log(path, max_records=10, max_bytes=4096)

    assert report["events"] == 2
    assert report["successes"] == 1
    assert report["errors"] == 1
    assert report["fallbacks"] == 1
    assert report["latency_ms"] == 20.0
    assert report["last_event_at"] == "2026-07-13T22:01:00+00:00"
    assert report["latest"][1]["error_type"] == "TimeoutError"
    assert OPAQUE not in repr(report)


def test_persona_log_tolerates_rotated_file_and_partial_tail(tmp_path: Path) -> None:
    path = tmp_path / "persona.jsonl"
    _write(path.with_suffix(".jsonl.1"), [{"timestamp": "2026-07-13T21:00:00+00:00", "packet": {"mode": "daily", "state": "stable"}}])
    path.write_text(json.dumps({"timestamp": "2026-07-13T22:00:00+00:00", "packet": {"mode": "work", "state": "focused"}, "extra": {"input_text": OPAQUE}}) + "\n{" , encoding="utf-8")

    report = collect_persona_log(path, max_records=10, max_bytes=4096)

    assert report["events"] == 2
    assert report["mode"] == "work"
    assert report["state"] == "focused"
    assert report["damaged_lines"] == 1
    assert report["issues"][0]["code"] == "PERSONA_LOG_DAMAGED_LINES"
    assert OPAQUE not in repr(report)


def test_missing_log_is_degraded_not_exception(tmp_path: Path) -> None:
    report = collect_router_log(tmp_path / "missing.jsonl")
    assert report["events"] == 0
    assert report["issues"][0]["code"] == "ROUTER_LOG_MISSING"
