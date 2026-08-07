"""Read-only governance runtime aggregation for PCLTM.

This module is the public "one report" surface for memory governance.  It
aggregates existing doctors and sidecars without approving, deleting, rewriting,
or injecting memory by itself.
"""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

from .governance import MemoryGovernanceOrchestrator
from .index_observability import index_doctor
from .memfs_store import MemFSStore
from .runtime_paths import resolve_db_path, resolve_memfs_root
from .store import EventStore


def _metadata(raw: object) -> dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if raw is None:
        return {}
    try:
        parsed = json.loads(str(raw))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def scan_scope_collisions(*, db_path: str | Path | None = None) -> dict[str, Any]:
    """Detect canonical memory keys reused across distinct scope keys.

    A repeated canonical key inside one scope may be a normal duplicate-review
    problem.  The dangerous case is the same canonical key appearing in multiple
    scopes, because a derived index that omits scope could collapse them.
    """

    db = resolve_db_path(db_path)
    if not db.exists():
        return {
            "ok": True,
            "db_path": str(db),
            "collision_count": 0,
            "collisions": [],
            "issues": [],
        }

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            """
            SELECT record_id, candidate_id, target_file, kind, status, metadata
            FROM memory_records
            ORDER BY record_id ASC
            """
        ).fetchall()
    finally:
        con.close()

    for row in rows:
        meta = _metadata(row["metadata"])
        canonical_key = str(meta.get("canonical_key") or "").strip()
        if not canonical_key:
            continue
        scope_key = str(meta.get("scope_key") or "global:default").strip() or "global:default"
        grouped[canonical_key].append(
            {
                "record_id": int(row["record_id"]),
                "candidate_id": str(row["candidate_id"]),
                "target_file": str(row["target_file"]),
                "kind": str(row["kind"]),
                "status": str(row["status"]),
                "scope_key": scope_key,
            }
        )

    collisions: list[dict[str, Any]] = []
    for canonical_key, records in grouped.items():
        scope_keys = sorted({record["scope_key"] for record in records})
        if len(scope_keys) <= 1:
            continue
        collisions.append(
            {
                "severity": "error",
                "code": "scope_canonical_key_collision",
                "message": "canonical_key is reused across multiple scope_key values",
                "canonical_key": canonical_key,
                "scope_keys": scope_keys,
                "records": records,
            }
        )

    return {
        "ok": not collisions,
        "db_path": str(db),
        "collision_count": len(collisions),
        "collisions": collisions,
        "issues": collisions,
    }


def _selection_report_for_db(
    *,
    db: Path,
    target: str,
    mode: str | None,
    emotion_axes: set[str] | None,
    budget_available: float | None,
) -> dict[str, Any]:
    return {
        "status": "retired",
        "bodyless": True,
        "reason": "legacy_memory_selection_probe_not_runtime_authority",
        "target": target,
        "mode": mode,
    }


def run_governance(
    *,
    db_path: str | Path | None = None,
    memfs_root: str | Path | None = None,
    selection_target: str = "user",
    mode: str | None = None,
    emotion_axes: set[str] | None = None,
    budget_available: float | None = None,
    dry_run: bool = True,
    rebuild_indexes: bool = False,
) -> dict[str, Any]:
    """Aggregate PCLTM governance doctors into one JSON-safe report."""

    db = resolve_db_path(db_path)
    memfs = resolve_memfs_root(memfs_root)
    index_report = index_doctor(db_path=db, memfs_root=memfs, rebuild=rebuild_indexes)
    scope_report = scan_scope_collisions(db_path=db)
    selection_report = _selection_report_for_db(
        db=db,
        target=selection_target,
        mode=mode,
        emotion_axes=emotion_axes,
        budget_available=budget_available,
    )

    memfs_store = MemFSStore(memfs)
    event_store: EventStore | None = EventStore(db) if db.exists() else None
    try:
        memory_report = MemoryGovernanceOrchestrator(
            memfs_store=memfs_store,
            event_store=event_store,
        ).analyze()
    finally:
        if event_store is not None:
            event_store.close()

    issues: list[dict[str, Any]] = []
    issues.extend(dict(issue) for issue in index_report.get("issues", []))
    issues.extend(dict(issue) for issue in scope_report.get("issues", []))
    error_count = sum(1 for issue in issues if issue.get("severity") == "error")
    warning_count = sum(1 for issue in issues if issue.get("severity") == "warning")

    return {
        "ok": error_count == 0,
        "dry_run": bool(dry_run),
        "authority_boundary": "read_only_governance_runtime",
        "db_path": str(db),
        "memfs_root": str(memfs),
        "index": index_report,
        "memory_governance": memory_report.to_dict(),
        "scope_collisions": scope_report,
        "selection_probe": selection_report,
        "summary": {
            "error_count": error_count,
            "warning_count": warning_count,
            "issue_codes": sorted({str(issue.get("code")) for issue in issues if issue.get("code")}),
            "pending_candidates": memory_report.pending_candidates,
            "governance_actions": len(memory_report.actions),
            "scope_collision_count": scope_report.get("collision_count", 0),
            "selection_drift_warnings": len(selection_report.get("drift_warnings", [])),
        },
        "issues": issues,
    }
