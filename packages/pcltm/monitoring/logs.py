"""Bounded tail aggregation for persona and router JSONL audit logs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


def _issue(code: str, source: str, message: str) -> dict[str, str]:
    return {
        "severity": "warning",
        "code": code,
        "source": source,
        "message": message,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "remediation": f"Inspect the {source} audit log source.",
    }


def _tail(path: Path, *, max_bytes: int) -> bytes:
    with path.open("rb") as stream:
        stream.seek(0, 2)
        size = stream.tell()
        start = max(0, size - max_bytes)
        stream.seek(start)
        data = stream.read(max_bytes)
    if start and b"\n" in data:
        data = data.split(b"\n", 1)[1]
    return data


def _read_rows(path: Path, *, source: str, max_records: int, max_bytes: int) -> tuple[list[dict[str, Any]], int, list[dict[str, str]]]:
    files = [path.with_suffix(path.suffix + ".1"), path]
    existing = [item for item in files if item.is_file()]
    if not existing:
        return [], 0, [_issue(f"{source.upper()}_LOG_MISSING", source, "Audit log is missing.")]
    rows: list[dict[str, Any]] = []
    damaged = 0
    for item in existing:
        for raw in _tail(item, max_bytes=max_bytes).splitlines():
            if not raw.strip():
                continue
            try:
                value = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                damaged += 1
                continue
            if isinstance(value, dict):
                rows.append(value)
    rows = rows[-max_records:]
    issues = []
    if damaged:
        issues.append(_issue(f"{source.upper()}_LOG_DAMAGED_LINES", source, f"Skipped {damaged} damaged audit lines."))
    return rows, damaged, issues


def collect_router_log(path: str | Path, *, max_records: int = 100, max_bytes: int = 256 * 1024) -> dict[str, Any]:
    rows, damaged, issues = _read_rows(Path(path), source="router", max_records=max_records, max_bytes=max_bytes)
    safe = [
        {key: row.get(key) for key in ("ts", "route", "selected_model", "ok", "fallback_used", "latency_ms", "upstream_status", "error_type") if key in row}
        for row in rows
    ]
    latencies = [float(row["latency_ms"]) for row in safe if isinstance(row.get("latency_ms"), (int, float))]
    return {
        "events": len(safe),
        "successes": sum(row.get("ok") is True for row in safe),
        "errors": sum(row.get("ok") is False for row in safe),
        "fallbacks": sum(bool(row.get("fallback_used")) for row in safe),
        "latency_ms": sum(latencies) / len(latencies) if latencies else None,
        "last_event_at": next((str(row.get("ts")) for row in reversed(safe) if row.get("ts")), None),
        "damaged_lines": damaged,
        "latest": safe,
        "issues": issues,
    }


def collect_persona_log(path: str | Path, *, max_records: int = 100, max_bytes: int = 256 * 1024) -> dict[str, Any]:
    rows, damaged, issues = _read_rows(Path(path), source="persona", max_records=max_records, max_bytes=max_bytes)
    safe = []
    for row in rows:
        packet = row.get("packet") if isinstance(row.get("packet"), dict) else {}
        safe.append({key: value for key, value in {"timestamp": row.get("timestamp"), "mode": packet.get("mode"), "state": packet.get("state")}.items() if value is not None})
    latest = safe[-1] if safe else {}
    return {
        "events": len(safe),
        "mode": latest.get("mode"),
        "state": latest.get("state"),
        "last_event_at": latest.get("timestamp"),
        "damaged_lines": damaged,
        "latest": safe,
        "issues": issues,
    }


__all__ = ["collect_persona_log", "collect_router_log"]
