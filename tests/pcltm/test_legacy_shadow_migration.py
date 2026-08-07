from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pcltm.legacy_shadow_migration import (
    compare_shadow_recall,
    create_readonly_sqlite_snapshot,
    generate_bodyless_legacy_manifest,
    run_readonly_shadow_replay,
)
from pcltm.memory_contracts import PersonaMode, Sensitivity
from pcltm.memory_write_service import MemoryWriteRequest, MemoryWriteService
from pcltm.projections.memory_fts import MemoryFtsProjector
from pcltm.store import EventStore


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _add_record(
    store: EventStore,
    *,
    candidate_id: str,
    content: str,
    status: str = "approved",
    sensitivity: str = "normal",
    source_event_ids: list[int] | None = None,
    metadata: dict | None = None,
    reviewer: str | None = None,
    decision_reason: str | None = None,
) -> int:
    record_id, _ = store.add_memory_record(
        candidate_id=candidate_id,
        kind="memory_note",
        target_file="MEMORY.md",
        content=content,
        confidence=1.0,
        sensitivity=sensitivity,
        source_event_ids=source_event_ids or [],
        status=status,
        metadata=metadata or {},
        reviewer=reviewer,
        decision_reason=decision_reason,
    )
    return record_id


def test_online_snapshot_reads_source_without_changing_its_bytes(tmp_path: Path) -> None:
    source = tmp_path / "live.db"
    store = EventStore(source)
    try:
        _add_record(store, candidate_id="one", content="private body one")
    finally:
        store.close()
    before = _file_hash(source)

    receipt = create_readonly_sqlite_snapshot(source, tmp_path / "snapshot.db")

    assert _file_hash(source) == before == receipt.source_sha256_before == receipt.source_sha256_after
    assert receipt.snapshot_sha256 == _file_hash(tmp_path / "snapshot.db")
    assert receipt.quick_check == "ok"
    assert receipt.source_query_only is True
    assert receipt.source_path == str(source.resolve())
    assert receipt.snapshot_path == str((tmp_path / "snapshot.db").resolve())


def test_bodyless_manifest_uses_strict_mutually_exclusive_classification(tmp_path: Path) -> None:
    db = tmp_path / "snapshot.db"
    store = EventStore(db)
    try:
        event_id = store.append_event(
            session_id="s", conversation_id="c", platform="desktop",
            role="user", source="test", content="hard lineage source",
            sensitivity="private", persona_mode="daily",
            category="raw_conversation", subcategory="user",
            inject_policy="retrieve_only",
        )
        verified_id = _add_record(
            store, candidate_id="verified", content="verified body",
            sensitivity="private", source_event_ids=[event_id],
        )
        weak_tool_id = _add_record(
            store, candidate_id="weak-tool", content="weak tool body",
            metadata={"source": "memory_tool"},
        )
        governed_id = _add_record(
            store, candidate_id="governed-tool", content="governed tool body",
            metadata={"source": "memory_tool", "provenance_version": 1},
            reviewer="memory_tool", decision_reason="explicit_memory_tool_write",
        )
        pending_id = _add_record(
            store, candidate_id="pending", content="pending body", status="pending",
        )
        historical_id = _add_record(
            store, candidate_id="historical", content="historical body", status="rejected",
        )
    finally:
        store.close()

    manifest = generate_bodyless_legacy_manifest(db)
    rendered = json.dumps(manifest, ensure_ascii=False, sort_keys=True)
    by_id = {item["record_id"]: item for item in manifest["records"]}

    assert manifest["bodyless"] is True
    assert manifest["quick_check"] == "ok"
    assert manifest["counts_by_class"] == {
        "event_derived": 1,
        "historical": 1,
        "legacy_governed": 1,
        "quarantined": 2,
    }
    assert by_id[verified_id]["classification"] == "event_derived"
    assert by_id[verified_id]["source_events_verified"] is True
    assert by_id[verified_id]["source_commitment_sha256"] != "0" * 64
    assert by_id[weak_tool_id]["reason_code"] == "legacy_provenance_insufficient"
    assert by_id[governed_id]["classification"] == "legacy_governed"
    assert by_id[pending_id]["reason_code"] == "legacy_state_requires_review"
    assert by_id[historical_id]["classification"] == "historical"
    for secret_body in (
        "verified body", "weak tool body", "governed tool body",
        "pending body", "historical body", "hard lineage source",
    ):
        assert secret_body not in rendered
    assert all(set(item) == {
        "record_id", "candidate_id_sha256", "content_sha256", "status",
        "kind", "target_file", "sensitivity", "source_event_count",
        "source_events_verified", "source_commitment_sha256",
        "legacy_provenance_sha256", "classification", "reason_code",
    } for item in manifest["records"])


def test_event_lineage_requires_active_sources_and_no_sensitivity_downgrade(tmp_path: Path) -> None:
    db = tmp_path / "snapshot.db"
    store = EventStore(db)
    try:
        event_id = store.append_event(
            session_id="s", conversation_id="c", platform="desktop",
            role="user", source="test", content="restricted source",
            sensitivity="restricted", persona_mode="daily",
            category="raw_conversation", subcategory="user",
            inject_policy="retrieve_only",
        )
        downgraded = _add_record(
            store, candidate_id="downgraded", content="derived body",
            sensitivity="normal", source_event_ids=[event_id],
        )
        store._conn.execute(
            """
            INSERT INTO event_governance(event_id, action, previous_state, new_state, actor, reason)
            VALUES (?, 'redact', 'active', 'redacted', 'test', 'retired source')
            """,
            (event_id,),
        )
        store._conn.commit()
    finally:
        store.close()

    item = generate_bodyless_legacy_manifest(db)["records"][0]
    assert item["record_id"] == downgraded
    assert item["classification"] == "quarantined"
    assert item["reason_code"] in {"legacy_source_inactive", "legacy_sensitivity_downgrade"}


def test_manifest_rejects_missing_lineage_instead_of_guessing(tmp_path: Path) -> None:
    db = tmp_path / "snapshot.db"
    store = EventStore(db)
    try:
        record_id = _add_record(store, candidate_id="bad-lineage", content="body")
        store._conn.execute(
            "UPDATE memory_records SET source_event_ids = '[999999]' WHERE record_id = ?",
            (record_id,),
        )
        store._conn.commit()
    finally:
        store.close()

    item = generate_bodyless_legacy_manifest(db)["records"][0]
    assert item["classification"] == "quarantined"
    assert item["source_events_verified"] is False
    assert item["reason_code"] == "legacy_source_event_missing"


def test_snapshot_destination_must_be_new_and_outside_source(tmp_path: Path) -> None:
    source = tmp_path / "live.db"
    EventStore(source).close()
    existing = tmp_path / "existing.db"
    existing.write_bytes(b"do-not-overwrite")

    with pytest.raises(FileExistsError):
        create_readonly_sqlite_snapshot(source, existing)
    assert existing.read_bytes() == b"do-not-overwrite"
    with pytest.raises(ValueError, match="distinct"):
        create_readonly_sqlite_snapshot(source, source)


def test_shadow_runner_executes_both_readonly_chains_and_emits_only_commitments(tmp_path: Path) -> None:
    db = tmp_path / "shadow.db"
    store = EventStore(db)
    try:
        _add_record(
            store, candidate_id="legacy-shared", content="shared-shadow-token body",
            metadata={"mode_scope": ["work"]},
        )
        receipt = MemoryWriteService(store).write(
            MemoryWriteRequest(
                idempotency_key="shadow-shared",
                content="shared-shadow-token body",
                canonical_key="shadow:shared",
                target="profile",
                memory_type="preference",
                sensitivity=Sensitivity.NORMAL,
                mode_scope=(PersonaMode.WORK,),
                injection_policy="allow",
            )
        )
        assert receipt.success is True
        assert MemoryFtsProjector(store, worker_id="shadow").run_once(
            now="2026-07-31T01:00:00Z", lease_until="2026-07-31T01:01:00Z",
        )["applied"] == 1
    finally:
        store.close()

    replay = run_readonly_shadow_replay(
        db,
        [{
            "query_id": "shared",
            "query": "shared-shadow-token",
            "persona_mode": "work",
            "sensitivity_ceiling": "restricted",
            "limit": 8,
        }],
    )
    rendered = json.dumps(replay, ensure_ascii=False, sort_keys=True)
    comparison = compare_shadow_recall(
        replay["queries"], query_bindings={"shared": "shared-shadow-token"},
    )

    assert replay["schema_version"] == 1
    assert replay["bodyless"] is True
    assert replay["source_query_only"] is True
    assert replay["runtime_authority_changed"] is False
    assert replay["fallback_used"] is False
    assert replay["queries"][0]["legacy"]["status"] == "ok"
    assert replay["queries"][0]["governed"]["status"] == "ok"
    assert comparison["counts"] == {"different": 0, "same": 1}
    assert "shared-shadow-token" not in rendered
    assert "shared-shadow-token body" not in rendered


def test_shadow_runner_reports_governed_abstention_without_using_legacy_as_fallback(tmp_path: Path) -> None:
    db = tmp_path / "legacy-only.db"
    store = EventStore(db)
    try:
        _add_record(store, candidate_id="legacy-only", content="legacy-only-shadow-token private body")
    finally:
        store.close()

    replay = run_readonly_shadow_replay(
        db,
        [{
            "query_id": "legacy-only",
            "query": "legacy-only-shadow-token",
            "persona_mode": "work",
            "sensitivity_ceiling": "restricted",
            "limit": 3,
        }],
    )
    side = replay["queries"][0]

    assert side["legacy"]["status"] == "ok"
    assert side["governed"]["status"] == "abstained"
    assert side["governed"]["result_commitments"] == []
    assert replay["fallback_used"] is False
    assert compare_shadow_recall(
        replay["queries"],
        query_bindings={"legacy-only": "legacy-only-shadow-token"},
    )["counts"] == {"different": 1, "same": 0}


def test_shadow_runner_rejects_extra_fields_that_could_leak_bodies(tmp_path: Path) -> None:
    db = tmp_path / "empty.db"
    EventStore(db).close()

    with pytest.raises(ValueError, match="shadow_query_schema_invalid"):
        run_readonly_shadow_replay(
            db,
            [{
                "query_id": "bad", "query": "needle", "persona_mode": "work",
                "sensitivity_ceiling": "normal", "limit": 1,
                "body": "must never pass through",
            }],
        )


@pytest.mark.parametrize("query_id", ["query body leaked here", "路径/正文", "x" * 129])
def test_shadow_runner_requires_opaque_bodyless_query_id(tmp_path: Path, query_id: str) -> None:
    db = tmp_path / "empty.db"
    EventStore(db).close()

    with pytest.raises(ValueError, match="shadow_query_schema_invalid"):
        run_readonly_shadow_replay(
            db,
            [{
                "query_id": query_id, "query": "needle", "persona_mode": "work",
                "sensitivity_ceiling": "normal", "limit": 1,
            }],
        )


def test_shadow_comparator_preserves_rank_order_and_rejects_nonopaque_id() -> None:
    digest_a = "a" * 64
    digest_b = "b" * 64
    base = {
        "query_id": "opaque-1",
        "query_sha256": hashlib.sha256(b"ranked query").hexdigest(),
        "legacy": {
            "status": "ok", "reason_codes": ["legacy"],
            "result_commitments": [digest_a, digest_b],
        },
        "governed": {
            "status": "ok", "reason_codes": ["governed"],
            "result_commitments": [digest_b, digest_a],
        },
    }

    assert compare_shadow_recall(
        [base], query_bindings={"opaque-1": "ranked query"},
    )["counts"] == {"different": 1, "same": 0}
    with pytest.raises(ValueError, match="shadow_query_identity_invalid"):
        compare_shadow_recall(
            [{**base, "query_id": "query body"}],
            query_bindings={"query body": "ranked query"},
        )


def test_shadow_comparator_binds_emitted_hash_to_the_actual_query() -> None:
    base = {
        "query_id": "opaque-1",
        "query_sha256": hashlib.sha256(b"different query").hexdigest(),
        "legacy": {
            "status": "ok", "reason_codes": ["legacy"],
            "result_commitments": ["a" * 64],
        },
        "governed": {
            "status": "ok", "reason_codes": ["governed"],
            "result_commitments": ["a" * 64],
        },
    }

    with pytest.raises(ValueError, match="shadow_query_hash_mismatch"):
        compare_shadow_recall(
            [base], query_bindings={"opaque-1": "actual query"},
        )
