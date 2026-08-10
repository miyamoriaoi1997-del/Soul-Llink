"""Bounded read-only collectors for the SoulLink monitoring sidecar."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Callable

from pcltm.memory_adapter import last_live_context_telemetry

from .redaction import sanitize_error, sanitize_source
from .sqlite_snapshot import stable_sqlite_snapshot

IssueDict = dict[str, str]


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _issue(
    *,
    severity: str,
    code: str,
    source: str,
    message: str,
    remediation: str,
) -> IssueDict:
    return {
        "severity": severity if severity in {"info", "warning", "error"} else "warning",
        "code": str(code),
        "source": str(source),
        "message": str(message),
        "timestamp": _timestamp(),
        "remediation": str(remediation),
    }


def _normalized_issues(
    raw_issues: object,
    *,
    source: str,
    max_string_length: int,
) -> list[IssueDict]:
    if not isinstance(raw_issues, Sequence) or isinstance(raw_issues, (str, bytes)):
        return []
    normalized: list[IssueDict] = []
    for raw in raw_issues:
        if not isinstance(raw, Mapping):
            continue
        severity = str(raw.get("severity") or "warning").lower()
        code = str(raw.get("code") or "UPSTREAM_ISSUE")[:max_string_length]
        message = str(raw.get("message") or "Monitoring source reported an issue.")[
            :max_string_length
        ]
        normalized.append(
            _issue(
                severity=severity,
                code=code,
                source=source,
                message=message,
                remediation=f"Inspect the {source} diagnostic source.",
            )
        )
    return normalized


def _call_source(
    name: str,
    function: Callable[..., dict[str, Any]],
    *,
    kwargs: dict[str, Any],
    max_error_length: int,
) -> tuple[dict[str, Any], IssueDict | None]:
    try:
        result = function(**kwargs)
        if not isinstance(result, dict):
            raise TypeError("collector result is not a mapping")
        return result, None
    except Exception as error:
        safe = sanitize_error(error, max_string_length=max_error_length)
        return {}, _issue(
            severity="warning",
            code=f"{name.upper()}_COLLECTOR_FAILED",
            source=name,
            message=f"{safe['error_type']}: {safe['message']}"[:max_error_length],
            remediation=f"Inspect the {name} doctor source.",
        )


def _integer(mapping: Mapping[str, Any], key: str) -> int:
    value = mapping.get(key, 0)
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _positive_integer(mapping: Mapping[str, Any], key: str) -> int:
    return max(0, _integer(mapping, key))


def _budget_buckets(total: int) -> dict[str, int]:
    """Derive host-adapter budget buckets without losing rounding remainder."""
    total = max(0, int(total))
    active_frame = int(total * 0.70)
    continuity = int(total * 0.05)
    tool_evidence = int(total * 0.05)
    recall = int(total * 0.08)
    reserve = total - active_frame - continuity - tool_evidence - recall
    return {
        "active_frame": active_frame,
        "continuity": continuity,
        "tool_evidence": tool_evidence,
        "recall": recall,
        "reserve": reserve,
    }


def _parse_timestamp(value: object) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def collect_context_budget(
    *,
    config: Mapping[str, Any],
    telemetry_path: str | Path | None = None,
    now: datetime | None = None,
    stale_after_seconds: float = 30.0,
) -> dict[str, Any]:
    """Combine static context config with timestamped last-render telemetry."""
    if stale_after_seconds <= 0:
        raise ValueError("stale_after_seconds must be positive")
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    model = config.get("model") if isinstance(config.get("model"), Mapping) else {}
    context = config.get("context") if isinstance(config.get("context"), Mapping) else {}
    compression = (
        config.get("compression")
        if isinstance(config.get("compression"), Mapping)
        else {}
    )
    budget_tokens = _positive_integer(context, "budget_tokens")
    telemetry: Mapping[str, Any] = {}
    if telemetry_path is not None:
        path = Path(telemetry_path)
        try:
            decoded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(decoded, Mapping) and decoded.get("source") == "exact_host_context_usage":
                telemetry = decoded
        except (OSError, json.JSONDecodeError):
            telemetry = {}
    if not telemetry:
        telemetry = last_live_context_telemetry()
    telemetry = telemetry if isinstance(telemetry, Mapping) else {}
    observed = _parse_timestamp(telemetry.get("observed_at"))
    issues: list[IssueDict] = []
    age_seconds: float | None = None
    stale = False
    telemetry_usable = observed is not None
    if telemetry and observed is None:
        issues.append(
            _issue(
                severity="warning",
                code="CONTEXT_TELEMETRY_UNTIMESTAMPED",
                source="context",
                message="Last context telemetry has no trustworthy observation time.",
                remediation="Emit observed_at with the live-context telemetry.",
            )
        )
    if observed is not None:
        age_seconds = max(0.0, (now - observed).total_seconds())
        stale = age_seconds > stale_after_seconds
        if stale:
            issues.append(
                _issue(
                    severity="warning",
                    code="CONTEXT_TELEMETRY_STALE",
                    source="context",
                    message="Last context telemetry is stale.",
                    remediation="Observe a fresh governed prompt render.",
                )
            )

    exact_host_usage = telemetry.get("source") == "exact_host_context_usage"
    prompt_tokens = _positive_integer(telemetry, "prompt_tokens") if telemetry_usable else None
    completion_tokens = _positive_integer(telemetry, "completion_tokens") if telemetry_usable else None
    total_tokens = _positive_integer(telemetry, "total_tokens") if telemetry_usable else None
    rendered_chars = _positive_integer(telemetry, "total_chars") if telemetry_usable else None
    limit_chars = _positive_integer(telemetry, "limit_chars") if telemetry_usable else None
    effective_budget = (
        _positive_integer(telemetry, "budget_tokens") if exact_host_usage else budget_tokens
    ) or budget_tokens
    effective_model_context = (
        _positive_integer(telemetry, "context_length") if exact_host_usage else 0
    ) or _positive_integer(model, "context_length")
    usage_ratio = (
        prompt_tokens / effective_budget
        if exact_host_usage and effective_budget and prompt_tokens is not None
        else rendered_chars / limit_chars
        if telemetry_usable and limit_chars and rendered_chars is not None else None
    )
    actions = telemetry.get("actions") if isinstance(telemetry.get("actions"), Sequence) else ()
    capsules = telemetry.get("capsules") if isinstance(telemetry.get("capsules"), Mapping) else {}
    source = str(telemetry.get("source")) if exact_host_usage else "config+last_telemetry" if telemetry_usable else "config"
    report = {
        "model_context_tokens": effective_model_context,
        "budget_tokens": effective_budget,
        "trigger_threshold": compression.get("threshold"),
        "engine": str(telemetry.get("engine") or context.get("engine") or ""),
        "budget_buckets": _budget_buckets(effective_budget),
        "within_budget": (
            prompt_tokens <= effective_budget
            if exact_host_usage and prompt_tokens is not None and effective_budget
            else bool(telemetry.get("within_budget")) if telemetry_usable else None
        ),
        "rendered_chars": rendered_chars,
        "limit_chars": limit_chars,
        "usage_ratio": usage_ratio,
        "omitted_chars": _positive_integer(telemetry, "omitted_chars") if telemetry_usable else None,
        "truncated_count": sum(
            1 for action in actions if str(action) in {"omitted", "hard_cap"}
        )
        if telemetry_usable
        else None,
        "capsule_count": sum(_positive_integer(capsules, str(key)) for key in capsules)
        if telemetry_usable
        else None,
        "source": source,
        "observed_at": observed.isoformat() if observed is not None else None,
        "age_seconds": age_seconds,
        "stale": stale,
        "issues": issues,
    }
    if exact_host_usage:
        report.update(
            {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "cache_read_tokens": _positive_integer(telemetry, "cache_read_tokens"),
                "compression_count": _positive_integer(telemetry, "compression_count"),
                "session_id": str(telemetry.get("session_id") or ""),
            }
        )
    return report


def collect_runtime_memory(
    *,
    db_path: str | Path,
    memfs_root: str | Path,
    max_error_length: int = 256,
) -> dict[str, Any]:
    """Collect status through a strictly read-only SQLite handle.

    Monitoring never instantiates EventStore or doctor/governance helpers;
    those paths may bootstrap schemas, change journal mode, update statistics,
    or repair derived indexes even under nominal dry-run options.
    """
    db = Path(db_path)
    memfs = Path(memfs_root)
    issues: list[IssueDict] = []
    counts = {
        "events": 0, "event_fts": 0, "event_chunks": 0, "summaries": 0,
        "summary_fts": 0, "memory_records": 0,
        "approved_memory_records": 0, "pending_memory_records": 0,
        "active_memory": 0, "active_event_derived_memory": 0,
    }
    schema_version: int | None = None
    semantic_query_ok = False
    runtime_status = "error" if not db.is_file() else "healthy"
    if not db.is_file():
        issues.append(_issue(
            severity="error", code="MISSING_DB", source="runtime",
            message="PCLTM SQLite database is missing.",
            remediation="Restore or initialize the production PCLTM database.",
        ))
    else:
        connection: sqlite3.Connection | None = None
        snapshot_context = stable_sqlite_snapshot(db)
        snapshot_opened = False
        try:
            snapshot_db = snapshot_context.__enter__()
            snapshot_opened = True
            connection = sqlite3.connect(
                f"{snapshot_db.resolve().as_uri()}?mode=ro", uri=True, timeout=2.0
            )
            connection.execute("PRAGMA query_only = ON")
            tables = {str(row[0]) for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table','view')"
            )}
            def count(table: str, where: str = "", params: tuple[Any, ...] = ()) -> int:
                if table not in tables:
                    return 0
                row = connection.execute(
                    f'SELECT count(*) FROM "{table}" {where}', params
                ).fetchone()
                return int(row[0] if row else 0)
            counts.update({
                "events": count("events"),
                "event_fts": count("event_fts"),
                "event_chunks": count("event_chunks"),
                "summaries": count("summary_nodes"),
                "summary_fts": count("summary_fts"),
                "memory_records": count("memory_records"),
                "approved_memory_records": count("memory_records", "WHERE status = ?", ("approved",)),
                "pending_memory_records": count("memory_records", "WHERE status = ?", ("pending",)),
                "active_memory": count("memory_current", "WHERE lifecycle_state = ?", ("active",)),
            })
            if {"memory_current", "memory_claim_versions"}.issubset(tables):
                row = connection.execute(
                    """SELECT count(*) FROM memory_current mc
                       JOIN memory_claim_versions v
                         ON v.claim_version_id = mc.claim_version_id
                       WHERE mc.lifecycle_state = 'active'
                         AND v.lineage_kind = 'event_derived'"""
                ).fetchone()
                counts["active_event_derived_memory"] = int(row[0] if row else 0)
            if "schema_migrations" in tables:
                row = connection.execute("SELECT max(version) FROM schema_migrations").fetchone()
                schema_version = int(row[0]) if row and row[0] is not None else 0
            integrity = connection.execute("PRAGMA quick_check").fetchone()
            if not integrity or str(integrity[0]).lower() != "ok":
                runtime_status = "error"
                issues.append(_issue(
                    severity="error", code="SQLITE_QUICK_CHECK_FAILED", source="runtime",
                    message="SQLite quick_check did not return ok.",
                    remediation="Run an offline database integrity investigation.",
                ))
            semantic_query_ok = "memory_records" in tables
        except sqlite3.Error as error:
            runtime_status = "degraded"
            safe = sanitize_error(error, max_string_length=max_error_length)
            issues.append(_issue(
                severity="error", code="READ_ONLY_DB_CHECK_FAILED", source="runtime",
                message=f"{safe['error_type']}: {safe['message']}"[:max_error_length],
                remediation="Inspect the production database using an offline copy.",
            ))
        finally:
            if connection is not None:
                connection.close()
            if snapshot_opened:
                snapshot_context.__exit__(None, None, None)
    memfs_counts = {layer: 0 for layer in ("pinned", "episodic", "semantic", "transient")}
    if memfs.is_dir():
        for layer in memfs_counts:
            layer_root = memfs / layer
            if layer_root.is_dir():
                memfs_counts[layer] = sum(1 for path in layer_root.rglob("*") if path.is_file())
    events, event_fts = counts["events"], counts["event_fts"]
    summaries, summary_fts = counts["summaries"], counts["summary_fts"]
    runtime = sanitize_source("runtime", {
        "status": runtime_status, "db_exists": db.is_file(),
        "schema_version": schema_version,
        "issues": [issue for issue in issues if issue["source"] == "runtime"],
    }, max_string_length=max_error_length)
    memory = sanitize_source("memory", {
        "event_count": events, "summary_count": summaries,
        "record_count": counts["memory_records"],
        "active_memory_count": counts["active_memory"],
        "active_event_derived_count": counts["active_event_derived_memory"],
        "active_other_lineage_count": counts["active_memory"] - counts["active_event_derived_memory"],
        "derived_memory_count": counts["memory_records"],
        "persistent_memory_total": events + counts["memory_records"],
        "evidence_chunk_count": counts["event_chunks"],
        "provenance": {
            "source": "stable_sqlite_snapshot",
            "database": "pcltm_runtime_db",
            "counting_rule": "persistent_memory_total = events + memory_records; event_chunks excluded",
            "event_table": "events",
            "derived_memory_table": "memory_records",
            "evidence_table": "event_chunks",
        },
        "approved_count": counts["approved_memory_records"],
        "pending_count": counts["pending_memory_records"],
        "fts_count": event_fts + summary_fts,
        "fts_consistent": events == event_fts and summaries == summary_fts,
        "semantic_count": counts["approved_memory_records"],
        "semantic_query_ok": semantic_query_ok,
        "memfs_counts": memfs_counts, "scope_collision_count": 0,
        "selection_drift_count": 0,
        "issues": [issue for issue in issues if issue["source"] == "memory"],
    }, max_string_length=max_error_length)
    return {"runtime": runtime, "memory": memory, "issues": issues}


__all__ = ["collect_context_budget", "collect_runtime_memory"]
