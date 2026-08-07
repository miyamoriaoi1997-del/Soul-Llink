"""Private, localhost-only runtime content collectors.

These collectors intentionally expose bodies for the single-machine owner dashboard.
They remain bounded and use read-only file/database access.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any
from urllib.parse import quote

import yaml

from .sqlite_snapshot import stable_sqlite_snapshot


def collect_emotion_state(state_path: str | Path) -> dict[str, Any]:
    """Read the SoulLink emotion frontmatter without changing the state file."""
    path = Path(state_path)
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        raise ValueError("STATE.md has no YAML frontmatter")
    _, frontmatter, _body = text.split("---", 2)
    document = yaml.safe_load(frontmatter) or {}
    state = document.get("emotion_state") or {}
    if not isinstance(state, dict):
        raise ValueError("emotion_state is not a mapping")
    axes = {
        name: state.get(name)
        for name in ("affection", "trust", "possessiveness", "patience")
    }
    return {
        "source": "runtime_state_file",
        "state_path": str(path),
        "axes": axes,
        "emotion_score": state.get("emotion_score"),
        "current_emotion": state.get("current_emotion"),
        "previous_emotion_score": state.get("previous_emotion_score"),
        "last_trigger_type": state.get("last_trigger_type"),
        "last_raw_trigger_type": state.get("last_raw_trigger_type"),
        "last_update": state.get("last_update"),
        "baselines": state.get("baselines") if isinstance(state.get("baselines"), dict) else {},
        "inertia": state.get("inertia") if isinstance(state.get("inertia"), dict) else {},
    }


def _readonly_connection(db_path: Path) -> sqlite3.Connection:
    normalized = str(db_path.resolve()).replace("\\", "/")
    uri = f"file:{quote(normalized, safe='/:')}?mode=ro"
    connection = sqlite3.connect(uri, uri=True, timeout=0.25)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    connection.execute("PRAGMA busy_timeout=250")
    return connection


def collect_memory_bodies(db_path: str | Path, *, limit: int = 100) -> dict[str, Any]:
    """Return recent governed memory records, including private body text."""
    bounded_limit = min(250, max(1, int(limit)))
    db = Path(db_path)
    with stable_sqlite_snapshot(db) as snapshot:
        connection = _readonly_connection(snapshot)
        try:
            rows = connection.execute(
                """
                SELECT record_id, candidate_id, kind, target_file, content, confidence,
                       sensitivity, status, metadata
                FROM memory_records
                ORDER BY record_id DESC
                LIMIT ?
                """,
                (bounded_limit,),
            ).fetchall()
        finally:
            connection.close()
    records: list[dict[str, Any]] = []
    for row in rows:
        try:
            metadata = json.loads(row["metadata"] or "{}")
        except (TypeError, json.JSONDecodeError):
            metadata = {}
        if not isinstance(metadata, dict):
            metadata = {}
        sensitivity = str(row["sensitivity"] or "")
        sensitive = sensitivity != "normal"
        records.append(
            {
                "record_id": row["record_id"],
                "candidate_id": row["candidate_id"],
                "kind": row["kind"],
                "target_file": row["target_file"],
                "content": (
                    "[REDACTED_SENSITIVE_MEMORY]" if sensitive else row["content"]
                ),
                "confidence": row["confidence"],
                "sensitivity": sensitivity,
                "status": row["status"],
                "scope_key": "" if sensitive else metadata.get("scope_key") or metadata.get("scope") or "",
                "metadata": {} if sensitive else metadata,
            }
        )
    return {
        "source": "runtime_db",
        "record_count": len(records),
        "limit": bounded_limit,
        "records": records,
    }


def collect_injection_preview(
    db_path: str | Path,
    *,
    mode: str | None = None,
    query: str | None = None,
    memory_limit: int = 2200,
    user_limit: int = 1375,
) -> dict[str, Any]:
    """Build a bounded, non-mutating preview from approved records.

    This deliberately does not call ``load_prompt_context`` because that production
    path updates retrieval statistics. It is therefore labeled as reconstruction,
    not an exact capture of a host request.
    """
    db = Path(db_path)
    with stable_sqlite_snapshot(db) as snapshot:
        connection = _readonly_connection(snapshot)
        try:
            rows = connection.execute(
                """
                SELECT record_id, target_file, content
                FROM memory_records
                WHERE status = 'approved'
                  AND sensitivity = 'normal'
                  AND target_file IN ('USER.md', 'MEMORY.md')
                ORDER BY record_id ASC
                """
            ).fetchall()
        finally:
            connection.close()
    limits = {"USER.md": max(0, int(user_limit)), "MEMORY.md": max(0, int(memory_limit))}
    selected: dict[str, list[str]] = {"USER.md": [], "MEMORY.md": []}
    selected_ids: list[int] = []
    omitted = 0
    used = {"USER.md": 0, "MEMORY.md": 0}
    for row in rows:
        target = str(row["target_file"])
        content = str(row["content"] or "").strip()
        separator = 3 if selected[target] else 0
        if used[target] + separator + len(content) > limits[target]:
            omitted += 1
            continue
        selected[target].append(content)
        selected_ids.append(int(row["record_id"]))
        used[target] += separator + len(content)
    lines = [
        "<pcltm_context>",
        "【provenance】sidecar_reconstruction_preview",
        "【is_state_machine_output】false",
        f"【retrieval_scope】{mode or 'unscoped'}",
        f"【query】{query or ''}",
    ]
    for target in ("USER.md", "MEMORY.md"):
        lines.append(f"【{target}】")
        lines.extend(selected[target] or ["(empty)"])
    lines.append("</pcltm_context>")
    rendered = "\n".join(lines)
    return {
        "source": "sidecar_reconstruction_preview",
        "is_exact_host_capture": False,
        "is_state_machine_output": False,
        "retrieval_scope": mode or "unscoped",
        "query": query or "",
        "rendered": rendered,
        "rendered_chars": len(rendered),
        "selected_record_ids": selected_ids,
        "selected_count": len(selected_ids),
        "omitted_count": omitted,
        "limits": limits,
    }


def collect_runtime_turn_capture(
    capture_path: str | Path,
    *,
    router_audit_path: str | Path | None = None,
) -> dict[str, Any]:
    """Read the exact latest pre-reply turn capture written by the host provider."""
    path = Path(capture_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("source") != "exact_host_capture":
        raise ValueError("runtime turn capture is not an exact host capture")
    payload = dict(payload)
    payload["capture_path"] = str(path)
    correlation_id = str(payload.get("turn_correlation_id") or "")
    matched = None
    audit_path = Path(router_audit_path) if router_audit_path else None
    if correlation_id and audit_path and audit_path.is_file():
        for raw in reversed(audit_path.read_text(encoding="utf-8").splitlines()[-500:]):
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and str(row.get("turn_correlation_id") or "") == correlation_id:
                matched = row
                break
    payload["model_chain"] = {
        "status": "correlated" if matched else "unavailable",
        "correlation_scope": "turn_level_latest_outcome",
        "turn_correlation_id": correlation_id or None,
        "router_request_hash": matched.get("request_hash") if matched else None,
        "actual_forwarded_model": matched.get("forwarded_model") if matched else None,
        "router_ok": matched.get("ok") if matched else None,
        "source": "router_final_forward_audit" if matched else "router_audit_not_correlated",
    }
    return payload


def collect_soul_content(active_soul_path: str | Path, layers_dir: str | Path) -> dict[str, Any]:
    """Read the active Soul anchor and canonical mode-layer source files."""
    active = Path(active_soul_path)
    directory = Path(layers_dir)

    def entry(path: Path) -> dict[str, Any]:
        stat = path.stat()
        return {
            "path": str(path),
            "content": path.read_text(encoding="utf-8"),
            "mtime_ns": stat.st_mtime_ns,
            "size": stat.st_size,
        }

    return {
        "source": "runtime_soul_files",
        "active": entry(active),
        "layers": {
            name: entry(directory / f"SOUL.{name}.template.md")
            for name in ("core", "daily", "work", "sex")
        },
    }


__all__ = [
    "collect_emotion_state",
    "collect_memory_bodies",
    "collect_injection_preview",
    "collect_runtime_turn_capture",
    "collect_soul_content",
]
