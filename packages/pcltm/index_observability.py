"""Read-only observability and repair helpers for PCLTM derived indexes."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from .memfs_store import MEMFS_LAYERS, MemFSStore
from .runtime_paths import resolve_db_path, resolve_memfs_root
from .semantic_index import SemanticIndex
from .store import EventStore

_OBSERVABILITY_TABLES = frozenset(
    {"events", "event_fts", "summary_nodes", "summary_fts", "memory_records"}
)


def _count(con: sqlite3.Connection, table: str) -> int:
    if table not in _OBSERVABILITY_TABLES:
        raise ValueError(f"unsupported observability table: {table}")
    try:
        row = con.execute(f"SELECT count(*) FROM {table}").fetchone()
    except sqlite3.Error:
        return 0
    return int(row[0] if row else 0)


def _db_counts(db: Path) -> dict[str, int]:
    if not db.exists():
        return {
            "events": 0,
            "event_fts": 0,
            "summaries": 0,
            "summary_fts": 0,
            "memory_records": 0,
            "approved_memory_records": 0,
        }
    con = sqlite3.connect(db)
    try:
        return {
            "events": _count(con, "events"),
            "event_fts": _count(con, "event_fts"),
            "summaries": _count(con, "summary_nodes"),
            "summary_fts": _count(con, "summary_fts"),
            "memory_records": _count(con, "memory_records"),
            "approved_memory_records": int(
                con.execute("SELECT count(*) FROM memory_records WHERE status = 'approved'").fetchone()[0]
            ),
        }
    finally:
        con.close()


def _memfs_counts(memfs_root: Path) -> dict[str, Any]:
    by_layer = {layer: 0 for layer in MEMFS_LAYERS}
    if not memfs_root.exists():
        return {"root_exists": False, "files": 0, "by_layer": by_layer, "warnings": []}
    store = MemFSStore(memfs_root)
    warnings: list[str] = []
    files = 0
    for item in store.list_tree():
        files += 1
        layer = item.path.split("/", 1)[0].replace("\\", "/")
        if layer in by_layer:
            by_layer[layer] += 1
    return {"root_exists": True, "files": files, "by_layer": by_layer, "warnings": warnings}


def index_stats(*, db_path: str | Path | None = None, memfs_root: str | Path | None = None) -> dict[str, Any]:
    """Return read-only statistics for SQLite FTS, BM25, and MemFS surfaces."""

    db = resolve_db_path(db_path)
    memfs = resolve_memfs_root(memfs_root)
    sqlite_counts = _db_counts(db)
    semantic_records = 0
    if db.exists():
        semantic = SemanticIndex(db)
        semantic_records = semantic.build()
    return {
        "ok": True,
        "authority_boundary": "read_only_index_observability",
        "db_path": str(db),
        "db_exists": db.exists(),
        "memfs_root": str(memfs),
        "sqlite": sqlite_counts,
        "semantic_index": {
            "records": semantic_records,
            "derived_from": "memory_records.status=approved",
            "rebuildable": True,
        },
        "memfs": _memfs_counts(memfs),
    }


def index_doctor(
    *,
    db_path: str | Path | None = None,
    memfs_root: str | Path | None = None,
    rebuild: bool = False,
) -> dict[str, Any]:
    """Check derived index consistency and optionally rebuild SQLite FTS tables."""

    db = resolve_db_path(db_path)
    memfs = resolve_memfs_root(memfs_root)
    issues: list[dict[str, Any]] = []
    rebuild_report: dict[str, int] | None = None
    if not db.exists():
        issues.append({"severity": "error", "code": "missing_db", "message": "PCLTM SQLite database is missing.", "path": str(db)})
    else:
        counts = _db_counts(db)
        if counts["events"] != counts["event_fts"]:
            issues.append(
                {
                    "severity": "error",
                    "code": "event_fts_mismatch",
                    "message": "event_fts row count does not match events; derived search index should be rebuilt.",
                    "events": counts["events"],
                    "event_fts": counts["event_fts"],
                }
            )
        if counts["summaries"] != counts["summary_fts"]:
            issues.append(
                {
                    "severity": "error",
                    "code": "summary_fts_mismatch",
                    "message": "summary_fts row count does not match summary_nodes; derived search index should be rebuilt.",
                    "summaries": counts["summaries"],
                    "summary_fts": counts["summary_fts"],
                }
            )
        if rebuild and any(issue["code"].endswith("_fts_mismatch") for issue in issues):
            store = EventStore(db)
            try:
                rebuild_report = store.rebuild_fts()
                store._conn.commit()
            finally:
                store.close()
            issues = [issue for issue in issues if not str(issue.get("code", "")).endswith("_fts_mismatch")]
    missing_layers = [layer for layer in MEMFS_LAYERS if not (memfs / layer).is_dir()]
    if missing_layers:
        issues.append(
            {
                "severity": "warning",
                "code": "missing_memfs_layers",
                "message": "MemFS layer directories are incomplete; run `soullink init` if this is a runtime tree.",
                "missing": missing_layers,
                "path": str(memfs),
            }
        )
    report = index_stats(db_path=db, memfs_root=memfs)
    if db.exists():
        semantic = SemanticIndex(db)
        semantic.build()
        probe = semantic.records[0].content if semantic.records else "pcltm index doctor smoke"
        try:
            results = semantic.query(probe, min_score=0.0)
            query_smoke = {"ok": True, "result_count": len(results)}
        except Exception as exc:
            query_smoke = {"ok": False, "result_count": 0, "error": str(exc)}
            issues.append({"severity": "error", "code": "semantic_query_smoke_failed", "message": f"Semantic index query smoke failed: {exc}"})
    else:
        query_smoke = {"ok": False, "result_count": 0, "error": "database is missing"}
    report["semantic_index"]["query_smoke"] = query_smoke
    report.update(
        {
            "ok": not any(issue.get("severity") == "error" for issue in issues),
            "issues": issues,
            "rebuild": rebuild_report or {},
        }
    )
    return report
