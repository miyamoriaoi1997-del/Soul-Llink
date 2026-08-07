"""Read-only legacy inventory and bodyless shadow-comparison tooling.

This module never promotes records, never changes runtime authority, and never
uses legacy results as a fallback for governed recall.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .evidence_chain import sha256_text
from .memory_contracts import PersonaMode, Sensitivity
from .memory_retrieval import (
    GovernedMemorySearchRequest,
    MemoryRetrievalStatus,
    search_governed_memories,
)
from .store import EventStore


_HEX = frozenset("0123456789abcdef")
_SENSITIVITY_RANK = {"normal": 0, "private": 1, "restricted": 2, "secret": 3}
_HISTORICAL_STATES = {"superseded", "rejected", "retired", "expired"}


@dataclass(frozen=True, slots=True)
class SqliteSnapshotReceipt:
    source_path: str
    snapshot_path: str
    source_sha256_before: str
    source_sha256_after: str
    snapshot_sha256: str
    quick_check: str
    source_query_only: bool


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _readonly_uri(path: Path) -> str:
    return f"file:{path.resolve().as_posix()}?mode=ro"


def _is_hash(value: object) -> bool:
    return type(value) is str and len(value) == 64 and all(char in _HEX for char in value)


def create_readonly_sqlite_snapshot(
    source_path: str | Path,
    snapshot_path: str | Path,
) -> SqliteSnapshotReceipt:
    """Use SQLite online backup from a query-only source into a new destination."""
    source = Path(source_path).resolve()
    destination = Path(snapshot_path).resolve()
    if source == destination:
        raise ValueError("source and snapshot must be distinct")
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    before = _file_sha256(source)
    source_conn = sqlite3.connect(_readonly_uri(source), uri=True)
    destination_conn: sqlite3.Connection | None = None
    try:
        source_conn.execute("PRAGMA query_only=ON")
        query_only = int(source_conn.execute("PRAGMA query_only").fetchone()[0]) == 1
        if not query_only:
            raise RuntimeError("source_query_only_unavailable")
        destination_conn = sqlite3.connect(destination)
        source_conn.backup(destination_conn)
        destination_conn.commit()
        quick_check = str(destination_conn.execute("PRAGMA quick_check").fetchone()[0])
        if quick_check != "ok":
            raise RuntimeError("snapshot_quick_check_failed")
    except BaseException:
        if destination_conn is not None:
            destination_conn.close()
            destination_conn = None
        source_conn.close()
        destination.unlink(missing_ok=True)
        raise
    finally:
        if destination_conn is not None:
            destination_conn.close()
        try:
            source_conn.close()
        except Exception:
            pass
    after = _file_sha256(source)
    if before != after:
        destination.unlink(missing_ok=True)
        raise RuntimeError("source_changed_during_snapshot")
    return SqliteSnapshotReceipt(
        str(source), str(destination), before, after,
        _file_sha256(destination), quick_check, query_only,
    )


def _parse_json_list(value: Any) -> list[Any] | None:
    try:
        parsed = json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, list) else None


def _metadata(value: Any) -> dict[str, Any] | None:
    try:
        parsed = json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _source_commitment(conn: sqlite3.Connection, source_ids: list[int]) -> tuple[bool, str, str | None, int]:
    if any(type(item) is not int or item <= 0 for item in source_ids):
        return False, "0" * 64, "legacy_source_ids_malformed", 0
    unique_ids = sorted(set(source_ids))
    if len(unique_ids) != len(source_ids):
        return False, "0" * 64, "legacy_source_ids_duplicated", len(source_ids)
    if not unique_ids:
        return False, "0" * 64, None, 0
    placeholders = ",".join("?" for _ in unique_ids)
    rows = conn.execute(
        f"""
        SELECT e.event_id, e.source_revision, e.payload_sha256, e.content,
               e.sensitivity,
               COALESCE((
                   SELECT g.new_state FROM event_governance g
                   WHERE g.event_id = e.event_id
                   ORDER BY g.governance_id DESC LIMIT 1
               ), e.evidence_state) AS lifecycle_state
        FROM events e WHERE e.event_id IN ({placeholders}) ORDER BY e.event_id
        """,
        tuple(unique_ids),
    ).fetchall()
    if len(rows) != len(unique_ids):
        return False, "0" * 64, "legacy_source_event_missing", len(source_ids)
    commitments = []
    for event in rows:
        payload = str(event["payload_sha256"])
        if sha256_text(str(event["content"])) != payload:
            return False, "0" * 64, "legacy_source_hash_mismatch", len(source_ids)
        if str(event["lifecycle_state"]) != "active":
            return False, "0" * 64, "legacy_source_inactive", len(source_ids)
        commitments.append({
            "event_id": int(event["event_id"]),
            "event_revision": int(event["source_revision"]),
            "payload_sha256": payload,
            "sensitivity": str(event["sensitivity"]),
        })
    return True, sha256_text(json.dumps(commitments, sort_keys=True, separators=(",", ":"))), None, len(source_ids)


def _classify_record(conn: sqlite3.Connection, row: sqlite3.Row) -> dict[str, Any]:
    source_ids = _parse_json_list(row["source_event_ids"])
    metadata = _metadata(row["metadata"])
    if source_ids is None:
        verified, source_commitment, source_reason, source_count = (
            False, "0" * 64, "legacy_source_ids_malformed", 0,
        )
    else:
        verified, source_commitment, source_reason, source_count = _source_commitment(conn, source_ids)

    status = str(row["status"])
    sensitivity = str(row["sensitivity"])
    metadata_source = "" if metadata is None else str(metadata.get("source") or "")
    provenance_version = None if metadata is None else metadata.get("provenance_version")
    reviewer = str(row["reviewer"] or "")
    decision_reason = str(row["decision_reason"] or "")
    durable_tool_provenance = (
        metadata_source == "memory_tool"
        and type(provenance_version) is int
        and provenance_version > 0
        and reviewer == "memory_tool"
        and decision_reason == "explicit_memory_tool_write"
    )
    legacy_provenance = sha256_text(json.dumps({
        "source": metadata_source,
        "provenance_version": provenance_version,
        "reviewer": reviewer,
        "decision_reason": decision_reason,
    }, sort_keys=True, separators=(",", ":")))

    if status in _HISTORICAL_STATES:
        classification, reason = "historical", "legacy_terminal_history"
    elif status != "approved":
        classification, reason = "quarantined", "legacy_state_requires_review"
    elif verified and source_count > 0:
        placeholders = ",".join("?" for _ in source_ids or [])
        max_source = conn.execute(
            f"SELECT sensitivity FROM events WHERE event_id IN ({placeholders})",
            tuple(source_ids or []),
        ).fetchall()
        source_rank = max((_SENSITIVITY_RANK.get(str(item[0]), 99) for item in max_source), default=99)
        if _SENSITIVITY_RANK.get(sensitivity, -1) < source_rank:
            classification, reason = "quarantined", "legacy_sensitivity_downgrade"
        else:
            classification, reason = "event_derived", "verified_event_lineage"
    elif source_reason is not None:
        classification, reason = "quarantined", source_reason
    elif sensitivity != "normal":
        classification, reason = "quarantined", "legacy_sensitivity_not_auto_promotable"
    elif durable_tool_provenance:
        classification, reason = "legacy_governed", "verified_memory_tool_provenance"
    elif metadata_source == "memory_tool":
        classification, reason = "quarantined", "legacy_provenance_insufficient"
    else:
        classification, reason = "quarantined", "legacy_source_unresolved"

    return {
        "record_id": int(row["record_id"]),
        "candidate_id_sha256": sha256_text(str(row["candidate_id"])),
        "content_sha256": sha256_text(str(row["content"])),
        "status": status,
        "kind": str(row["kind"]),
        "target_file": str(row["target_file"]),
        "sensitivity": sensitivity,
        "source_event_count": source_count,
        "source_events_verified": verified,
        "source_commitment_sha256": source_commitment,
        "legacy_provenance_sha256": legacy_provenance,
        "classification": classification,
        "reason_code": reason,
    }


def _commitment_digest(values: list[str]) -> str:
    if any(not _is_hash(value) for value in values):
        raise ValueError("shadow_result_commitment_invalid")
    # Recall is ranked: preserve order and duplicates in the comparison commitment.
    return sha256_text(json.dumps(values, separators=(",", ":")))


def _reason_digest(values: list[str]) -> str:
    if any(type(value) is not str or not value.strip() for value in values):
        raise ValueError("shadow_reason_invalid")
    return sha256_text(json.dumps(sorted(values), separators=(",", ":")))


def compare_shadow_recall(
    queries: list[dict[str, Any]],
    *,
    query_bindings: Mapping[str, str],
) -> dict[str, Any]:
    """Compare shared content commitments after binding each hash to its query."""
    if type(queries) is not list:
        raise TypeError("queries must be list")
    if type(query_bindings) is not dict:
        raise TypeError("query_bindings must be dict")
    expected_ids = {
        str(item.get("query_id")) for item in queries if type(item) is dict
    }
    if set(query_bindings) != expected_ids:
        raise ValueError("shadow_query_binding_invalid")
    diffs: list[dict[str, Any]] = []
    allowed_top = {"query_id", "query_sha256", "legacy", "governed"}
    allowed_side = {"status", "reason_codes", "result_commitments"}
    allowed_statuses = {"ok", "abstained", "unavailable"}
    for entry in queries:
        if type(entry) is not dict or set(entry) != allowed_top:
            raise ValueError("shadow_input_contains_body")
        query_id, query_sha256 = entry["query_id"], entry["query_sha256"]
        if (
            type(query_id) is not str or not query_id.strip()
            or len(query_id) > 128
            or any(
                not character.isascii()
                or not (character.isalnum() or character in "._:-")
                for character in query_id
            )
            or not _is_hash(query_sha256)
        ):
            raise ValueError("shadow_query_identity_invalid")
        bound_query = query_bindings.get(query_id)
        if type(bound_query) is not str:
            raise ValueError("shadow_query_binding_invalid")
        if sha256_text(bound_query) != query_sha256:
            raise ValueError("shadow_query_hash_mismatch")
        legacy, governed = entry["legacy"], entry["governed"]
        if type(legacy) is not dict or set(legacy) != allowed_side or type(governed) is not dict or set(governed) != allowed_side:
            raise ValueError("shadow_input_contains_body")
        if legacy["status"] not in allowed_statuses or governed["status"] not in allowed_statuses:
            raise ValueError("shadow_status_invalid")
        legacy_values, governed_values = legacy["result_commitments"], governed["result_commitments"]
        if type(legacy_values) is not list or type(governed_values) is not list:
            raise ValueError("shadow_result_commitment_invalid")
        legacy_digest = _commitment_digest(legacy_values)
        governed_digest = _commitment_digest(governed_values)
        same = legacy["status"] == governed["status"] and legacy_digest == governed_digest and len(legacy_values) == len(governed_values)
        diffs.append({
            "query_id": query_id,
            "query_sha256": query_sha256,
            "legacy_status": legacy["status"],
            "legacy_result_commitments_sha256": legacy_digest,
            "legacy_result_count": len(legacy_values),
            "legacy_reason_codes_sha256": _reason_digest(legacy["reason_codes"]),
            "governed_status": governed["status"],
            "governed_result_commitments_sha256": governed_digest,
            "governed_result_count": len(governed_values),
            "governed_reason_codes_sha256": _reason_digest(governed["reason_codes"]),
            "verdict": "same" if same else "different",
        })
    return {
        "schema_version": 2,
        "bodyless": True,
        "runtime_authority_changed": False,
        "fallback_used": False,
        "counts": {
            "different": sum(item["verdict"] == "different" for item in diffs),
            "same": sum(item["verdict"] == "same" for item in diffs),
        },
        "diffs": diffs,
    }


_SHADOW_QUERY_FIELDS = {
    "query_id", "query", "persona_mode", "sensitivity_ceiling", "limit",
}


def _legacy_shadow_side(
    conn: sqlite3.Connection,
    *,
    query: str,
    persona_mode: PersonaMode,
    sensitivity_ceiling: Sensitivity,
    limit: int,
) -> dict[str, Any]:
    """Replay the frozen legacy lexical-v1 selector without prompt injection."""
    table = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='memory_records'"
    ).fetchone()
    if table is None:
        return {
            "status": "unavailable",
            "reason_codes": ["legacy_memory_records_unavailable"],
            "result_commitments": [],
        }
    needle = " ".join(query.casefold().split())
    if not needle:
        return {
            "status": "abstained",
            "reason_codes": ["legacy_lexical_v1_no_answer"],
            "result_commitments": [],
        }
    rows = conn.execute(
        """
        SELECT record_id, content, sensitivity, metadata
        FROM memory_records
        WHERE status = 'approved'
        ORDER BY record_id
        """
    ).fetchall()
    ceiling = _SENSITIVITY_RANK[sensitivity_ceiling.value]
    ranked: list[tuple[int, str]] = []
    for row in rows:
        content = str(row["content"])
        normalized = " ".join(content.casefold().split())
        if needle not in normalized:
            continue
        sensitivity = str(row["sensitivity"])
        if _SENSITIVITY_RANK.get(sensitivity, 99) > ceiling:
            continue
        metadata = _metadata(row["metadata"]) or {}
        raw_scope = metadata.get("mode_scope")
        if isinstance(raw_scope, list) and raw_scope:
            if persona_mode.value not in {str(value) for value in raw_scope}:
                continue
        ranked.append((int(row["record_id"]), sha256_text(content)))
    commitments = [digest for _record_id, digest in ranked[:limit]]
    if not commitments:
        return {
            "status": "abstained",
            "reason_codes": ["legacy_lexical_v1_no_answer"],
            "result_commitments": [],
        }
    return {
        "status": "ok",
        "reason_codes": ["legacy_lexical_v1_match"],
        "result_commitments": commitments,
    }


def _governed_shadow_side(
    store: EventStore,
    *,
    query: str,
    persona_mode: PersonaMode,
    sensitivity_ceiling: Sensitivity,
    limit: int,
) -> dict[str, Any]:
    result = search_governed_memories(
        store,
        GovernedMemorySearchRequest(
            query=query,
            persona_mode=persona_mode,
            sensitivity_ceiling=sensitivity_ceiling,
            limit=limit,
        ),
    )
    if result.status is MemoryRetrievalStatus.OK:
        return {
            "status": "ok",
            "reason_codes": sorted({item.policy_reason for item in result.items}),
            "result_commitments": [item.content_sha256 for item in result.items],
        }
    return {
        "status": result.status.value,
        "reason_codes": [str(result.reason)],
        "result_commitments": [],
    }


def run_readonly_shadow_replay(
    snapshot_path: str | Path,
    queries: list[dict[str, Any]],
) -> dict[str, Any]:
    """Execute frozen legacy and canonical governed recall against one read-only DB.

    Query bodies and result bodies are consumed in-process and never emitted.
    The returned query entries are directly accepted by ``compare_shadow_recall``.
    """
    if type(queries) is not list:
        raise TypeError("queries must be list")
    path = Path(snapshot_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    parsed: list[tuple[str, str, PersonaMode, Sensitivity, int]] = []
    for entry in queries:
        if type(entry) is not dict or set(entry) != _SHADOW_QUERY_FIELDS:
            raise ValueError("shadow_query_schema_invalid")
        query_id, query = entry["query_id"], entry["query"]
        limit = entry["limit"]
        if (
            type(query_id) is not str or not query_id.strip()
            or len(query_id) > 128
            or any(
                not character.isascii()
                or not (character.isalnum() or character in "._:-")
                for character in query_id
            )
            or type(query) is not str
            or type(limit) is not int or isinstance(limit, bool)
            or limit <= 0 or limit > 100
        ):
            raise ValueError("shadow_query_schema_invalid")
        try:
            mode = PersonaMode(str(entry["persona_mode"]))
            ceiling = Sensitivity(str(entry["sensitivity_ceiling"]))
        except ValueError as exc:
            raise ValueError("shadow_query_schema_invalid") from exc
        parsed.append((query_id, query, mode, ceiling, limit))

    before = _file_sha256(path)
    conn = sqlite3.connect(_readonly_uri(path), uri=True)
    conn.row_factory = sqlite3.Row
    store: EventStore | None = None
    try:
        conn.execute("PRAGMA query_only=ON")
        query_only = int(conn.execute("PRAGMA query_only").fetchone()[0]) == 1
        if not query_only:
            raise RuntimeError("source_query_only_unavailable")
        if str(conn.execute("PRAGMA quick_check").fetchone()[0]) != "ok":
            raise RuntimeError("snapshot_quick_check_failed")
        store = EventStore(path, read_only=True)
        output_queries = []
        for query_id, query, mode, ceiling, limit in parsed:
            output_queries.append({
                "query_id": query_id,
                "query_sha256": sha256_text(query),
                "legacy": _legacy_shadow_side(
                    conn, query=query, persona_mode=mode,
                    sensitivity_ceiling=ceiling, limit=limit,
                ),
                "governed": _governed_shadow_side(
                    store, query=query, persona_mode=mode,
                    sensitivity_ceiling=ceiling, limit=limit,
                ),
            })
    finally:
        if store is not None:
            store.close()
        conn.close()
    after = _file_sha256(path)
    if before != after:
        raise RuntimeError("source_changed_during_shadow_replay")
    return {
        "schema_version": 1,
        "bodyless": True,
        "source_db_sha256": before,
        "source_query_only": True,
        "legacy_algorithm": "legacy_lexical_v1",
        "governed_algorithm": "pcltm_governed_search_v1",
        "runtime_authority_changed": False,
        "fallback_used": False,
        "queries": output_queries,
    }


def generate_bodyless_legacy_manifest(snapshot_path: str | Path) -> dict[str, Any]:
    """Classify legacy records from a query-only snapshot without emitting bodies."""
    path = Path(snapshot_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    conn = sqlite3.connect(_readonly_uri(path), uri=True)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA query_only=ON")
        if int(conn.execute("PRAGMA query_only").fetchone()[0]) != 1:
            raise RuntimeError("source_query_only_unavailable")
        quick_check = str(conn.execute("PRAGMA quick_check").fetchone()[0])
        if quick_check != "ok":
            raise RuntimeError("snapshot_quick_check_failed")
        rows = conn.execute(
            """
            SELECT record_id, candidate_id, kind, target_file, content,
                   sensitivity, source_event_ids, status, metadata,
                   reviewer, decision_reason
            FROM memory_records ORDER BY record_id
            """
        ).fetchall()
        records = [_classify_record(conn, row) for row in rows]
    finally:
        conn.close()
    counts: dict[str, int] = {}
    for item in records:
        key = str(item["classification"])
        counts[key] = counts.get(key, 0) + 1
    canonical_records = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "schema_version": 2,
        "bodyless": True,
        "source_db_sha256": _file_sha256(path),
        "quick_check": quick_check,
        "record_count": len(records),
        "counts_by_class": dict(sorted(counts.items())),
        "manifest_sha256": sha256_text(canonical_records),
        "records": records,
    }
