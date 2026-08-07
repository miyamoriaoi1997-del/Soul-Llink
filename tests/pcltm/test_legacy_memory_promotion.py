from __future__ import annotations

from pathlib import Path

import pytest
import sqlite3

from pcltm.evidence_chain import sha256_text
from pcltm.ledger_schema import ensure_evidence_ledger_schema
from pcltm.injection.governed_memory import (
    GovernedInjectionStatus,
    build_governed_memory_context,
)
from pcltm.legacy_memory_promotion import (
    LegacyMemoryPromotionRequest,
    LegacyMemoryPromotionService,
    LegacyMemoryPromotionSpec,
)
from pcltm.memory_contracts import PersonaMode
from pcltm.memory_transition_service import MemoryLifecycleRequest, MemoryTransitionService
from pcltm.memory_retrieval import (
    GovernedMemorySearchRequest,
    MemoryRetrievalStatus,
    search_governed_memories,
)
from pcltm.projections.memory_fts import MemoryFtsProjector
from pcltm.projections.memory_memfs import MemoryMemfsProjector
from pcltm.projections.transcript_chunks import TranscriptChunkProjector
from pcltm.store import EventStore


def _spec(record_id: int, **changes) -> LegacyMemoryPromotionSpec:
    values = {
        "record_id": record_id,
        "canonical_key": f"legacy:record:{record_id}",
        "target": "profile",
        "memory_type": "preference",
        "mode_scope": (PersonaMode.DAILY,),
        "injection_policy": "allow",
    }
    values.update(changes)
    return LegacyMemoryPromotionSpec(**values)


def test_approved_legacy_record_promotes_without_forged_event_and_reaches_injection(
    tmp_path: Path,
) -> None:
    store = EventStore(tmp_path / "authority.db")
    root = tmp_path / "memfs"
    try:
        record_id, inserted = store.add_memory_record(
            candidate_id="legacy-approved-001",
            kind="memory_note",
            target_file="USER.md",
            content="老师偏好 legacy-promotion-token UTC+8",
            confidence=1.0,
            sensitivity="normal",
            status="approved",
            reviewer="human-reviewer",
            decision_reason="approved before governed migration",
            metadata={"mode_scope": ["daily"]},
        )
        assert inserted is True
        legacy_before = tuple(store._conn.execute(
            "SELECT * FROM memory_records WHERE record_id = ?", (record_id,),
        ).fetchone())
        event_count_before = int(store._conn.execute(
            "SELECT count(*) FROM events",
        ).fetchone()[0])

        result = LegacyMemoryPromotionService(store).promote(
            LegacyMemoryPromotionRequest((
                LegacyMemoryPromotionSpec(
                    record_id=record_id,
                    canonical_key="legacy:user:timezone",
                    target="profile",
                    memory_type="preference",
                    mode_scope=(PersonaMode.DAILY,),
                    injection_policy="allow",
                ),
            ))
        )
        claim_id = result.items[0].claim_id

        fts = MemoryFtsProjector(store, worker_id="legacy-fts").run_once(
            now="2026-07-29T04:00:00Z",
            lease_until="2026-07-29T04:01:00Z",
        )
        memfs = MemoryMemfsProjector(
            store, memfs_root=root, worker_id="legacy-memfs",
        ).run_once(
            now="2026-07-29T04:00:00Z",
            lease_until="2026-07-29T04:01:00Z",
        )
        retrieval = search_governed_memories(
            store,
            GovernedMemorySearchRequest(
                query="legacy-promotion-token",
                persona_mode=PersonaMode.DAILY,
            ),
        )
        injection = build_governed_memory_context(
            store,
            retrieval,
            persona_mode=PersonaMode.DAILY,
            total_budget=200,
        )

        legacy_after = tuple(store._conn.execute(
            "SELECT * FROM memory_records WHERE record_id = ?", (record_id,),
        ).fetchone())
        event_count_after = int(store._conn.execute(
            "SELECT count(*) FROM events",
        ).fetchone()[0])
        source = store._conn.execute(
            """
            SELECT source_kind, legacy_record_id, legacy_content_sha256,
                   event_id, event_revision, event_payload_sha256
            FROM memory_claim_sources
            """
        ).fetchone()
        lineage = store._conn.execute(
            "SELECT lineage_kind FROM memory_claim_versions WHERE claim_id = ?",
            (claim_id,),
        ).fetchone()[0]
        jobs = store._conn.execute(
            """
            SELECT projection_kind, authority_kind, authority_id, status
            FROM projection_outbox
            WHERE aggregate_id = ? ORDER BY projection_kind
            """,
            (f"memory:{claim_id}",),
        ).fetchall()
    finally:
        store.close()

    assert result.status == "promoted"
    assert result.persisted is True
    assert len(result.items) == 1
    assert legacy_after == legacy_before
    assert event_count_after == event_count_before == 0
    assert tuple(source) == (
        "legacy_record",
        record_id,
        result.items[0].content_sha256,
        None,
        None,
        None,
    )
    assert lineage == "legacy_governed"
    assert fts == {"claimed": 1, "applied": 1, "failed": 0, "obsolete": 0}
    assert memfs == {"claimed": 1, "applied": 1, "failed": 0, "obsolete": 0}
    assert [(row["projection_kind"], row["authority_kind"], row["authority_id"], row["status"]) for row in jobs] == [
        ("memory_fts", "legacy_record", str(record_id), "applied"),
        ("memory_memfs", "legacy_record", str(record_id), "applied"),
    ]
    assert retrieval.status is MemoryRetrievalStatus.OK
    assert injection.status is GovernedInjectionStatus.OK
    assert injection.packet is not None
    assert "legacy-promotion-token" in injection.packet.render()


def test_promoted_legacy_claim_can_be_retired_with_durable_idempotency_receipt(
    tmp_path: Path,
) -> None:
    store = EventStore(tmp_path / "authority.db")
    try:
        record_id, _ = store.add_memory_record(
            candidate_id="legacy-retire", kind="memory_note", target_file="USER.md",
            content="legacy retirement token", confidence=1.0,
            sensitivity="normal", status="approved",
        )
        promoted = LegacyMemoryPromotionService(store).promote(
            LegacyMemoryPromotionRequest((_spec(record_id),))
        )
        request = MemoryLifecycleRequest(
            idempotency_key="legacy-retire-receipt",
            claim_id=promoted.items[0].claim_id,
            expected_current_version=1,
            reason_code="legacy_window_closed",
        )
        service = MemoryTransitionService(store)
        first = service.retire(request)
        second = service.retire(request)
        receipt_count = store._conn.execute(
            "SELECT count(*) FROM memory_transition_receipts WHERE idempotency_key = ?",
            (request.idempotency_key,),
        ).fetchone()[0]
    finally:
        store.close()

    assert first == second
    assert first.status == "retired"
    assert receipt_count == 1


def test_promotion_replay_requires_identical_spec_and_adds_nothing(
    tmp_path: Path,
) -> None:
    store = EventStore(tmp_path / "authority.db")
    try:
        record_id, _ = store.add_memory_record(
            candidate_id="legacy-replay",
            kind="memory_note",
            target_file="USER.md",
            content="legacy replay token",
            confidence=1.0,
            sensitivity="normal",
            status="approved",
        )
        service = LegacyMemoryPromotionService(store)
        first = service.promote(LegacyMemoryPromotionRequest((_spec(record_id),)))
        first_counts = tuple(store._conn.execute(
            """
            SELECT
              (SELECT count(*) FROM memory_claims),
              (SELECT count(*) FROM memory_claim_versions),
              (SELECT count(*) FROM memory_governance_events),
              (SELECT count(*) FROM projection_outbox)
            """
        ).fetchone())
        second = service.promote(LegacyMemoryPromotionRequest((_spec(record_id),)))
        second_counts = tuple(store._conn.execute(
            """
            SELECT
              (SELECT count(*) FROM memory_claims),
              (SELECT count(*) FROM memory_claim_versions),
              (SELECT count(*) FROM memory_governance_events),
              (SELECT count(*) FROM projection_outbox)
            """
        ).fetchone())
    finally:
        store.close()

    assert second == first
    assert second_counts == first_counts == (1, 1, 1, 2)


def test_promotion_replay_rejects_changed_spec_without_side_effects(
    tmp_path: Path,
) -> None:
    store = EventStore(tmp_path / "authority.db")
    try:
        record_id, _ = store.add_memory_record(
            candidate_id="legacy-spec-drift",
            kind="memory_note",
            target_file="USER.md",
            content="legacy spec drift token",
            confidence=1.0,
            sensitivity="normal",
            status="approved",
        )
        service = LegacyMemoryPromotionService(store)
        service.promote(LegacyMemoryPromotionRequest((_spec(record_id),)))
        with pytest.raises(ValueError, match="legacy_promotion_spec_conflict"):
            service.promote(LegacyMemoryPromotionRequest((
                _spec(record_id, injection_policy="deny"),
            )))
        authority = tuple(store._conn.execute(
            """
            SELECT c.canonical_key, c.target, c.memory_type,
                   v.injection_policy, v.mode_scope
            FROM memory_current mc
            JOIN memory_claims c ON c.claim_id = mc.claim_id
            JOIN memory_claim_versions v
              ON v.claim_version_id = mc.claim_version_id
            """
        ).fetchone())
        counts = tuple(store._conn.execute(
            """
            SELECT
              (SELECT count(*) FROM memory_claims),
              (SELECT count(*) FROM memory_claim_versions),
              (SELECT count(*) FROM memory_governance_events),
              (SELECT count(*) FROM projection_outbox)
            """
        ).fetchone())
    finally:
        store.close()

    assert authority == (
        f"legacy:record:{record_id}", "profile", "preference", "allow", '["daily"]',
    )
    assert counts == (1, 1, 1, 2)


def test_promotion_batch_rolls_back_when_any_record_is_not_approved(
    tmp_path: Path,
) -> None:
    store = EventStore(tmp_path / "authority.db")
    try:
        approved_id, _ = store.add_memory_record(
            candidate_id="legacy-batch-approved",
            kind="memory_note",
            target_file="USER.md",
            content="legacy batch approved token",
            confidence=1.0,
            sensitivity="normal",
            status="approved",
        )
        pending_id, _ = store.add_memory_record(
            candidate_id="legacy-batch-pending",
            kind="memory_note",
            target_file="USER.md",
            content="legacy batch pending token",
            confidence=1.0,
            sensitivity="normal",
            status="pending",
        )
        legacy_before = [
            tuple(row) for row in store._conn.execute(
                "SELECT * FROM memory_records ORDER BY record_id",
            ).fetchall()
        ]

        with pytest.raises(ValueError, match="legacy_record_not_approved"):
            LegacyMemoryPromotionService(store).promote(
                LegacyMemoryPromotionRequest((
                    _spec(approved_id),
                    _spec(pending_id),
                ))
            )

        legacy_after = [
            tuple(row) for row in store._conn.execute(
                "SELECT * FROM memory_records ORDER BY record_id",
            ).fetchall()
        ]
        counts = tuple(store._conn.execute(
            """
            SELECT
              (SELECT count(*) FROM events),
              (SELECT count(*) FROM memory_claims),
              (SELECT count(*) FROM memory_claim_versions),
              (SELECT count(*) FROM memory_claim_sources),
              (SELECT count(*) FROM memory_governance_events),
              (SELECT count(*) FROM memory_current),
              (SELECT count(*) FROM projection_outbox)
            """
        ).fetchone())
    finally:
        store.close()

    assert legacy_after == legacy_before
    assert counts == (0, 0, 0, 0, 0, 0, 0)


def test_schema_rejects_forged_legacy_source_commitment(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "authority.db")
    try:
        record_id, _ = store.add_memory_record(
            candidate_id="legacy-source-trigger",
            kind="memory_note",
            target_file="USER.md",
            content="legacy source trigger token",
            confidence=1.0,
            sensitivity="normal",
            status="approved",
        )
        claim_id = int(store._conn.execute(
            "INSERT INTO memory_claims(canonical_key, target, memory_type) "
            "VALUES ('legacy:forged', 'profile', 'preference')",
        ).lastrowid)
        version_id = int(store._conn.execute(
            """
            INSERT INTO memory_claim_versions(
                claim_id, version, content, content_sha256, confidence,
                sensitivity, injection_policy, mode_scope, lineage_kind,
                schema_version
            ) VALUES (?, 1, 'forged', ?, 1.0, 'normal', 'allow',
                      '["daily"]', 'legacy_governed', 1)
            """,
            (claim_id, "d" * 64),
        ).lastrowid)

        with pytest.raises(sqlite3.IntegrityError, match="legacy source commitment mismatch"):
            store._conn.execute(
                """
                INSERT INTO memory_claim_sources(
                    claim_version_id, source_kind, legacy_record_id,
                    legacy_content_sha256
                ) VALUES (?, 'legacy_record', ?, ?)
                """,
                (version_id, record_id, "0" * 64),
            )
    finally:
        store.close()


def test_superseded_legacy_source_fails_closed_before_projection(
    tmp_path: Path,
) -> None:
    store = EventStore(tmp_path / "authority.db")
    try:
        record_id, _ = store.add_memory_record(
            candidate_id="legacy-source-superseded",
            kind="memory_note",
            target_file="USER.md",
            content="legacy superseded source token",
            confidence=1.0,
            sensitivity="normal",
            status="approved",
        )
        LegacyMemoryPromotionService(store).promote(
            LegacyMemoryPromotionRequest((_spec(record_id),))
        )
        store._conn.execute(
            "UPDATE memory_records SET status = 'superseded' WHERE record_id = ?",
            (record_id,),
        )
        store._conn.commit()
        result = MemoryFtsProjector(store, worker_id="legacy-stale-source").run_once(
            now="2026-07-29T05:00:00Z",
            lease_until="2026-07-29T05:01:00Z",
        )
        job = store._conn.execute(
            """
            SELECT status, last_error FROM projection_outbox
            WHERE projection_kind = 'memory_fts'
            """
        ).fetchone()
    finally:
        store.close()

    assert result == {"claimed": 1, "applied": 0, "failed": 1, "obsolete": 0}
    assert tuple(job) == ("pending", "memory projection source commitment mismatch")


def test_schema_helper_registers_legacy_commitment_validation_on_raw_connection(
    tmp_path: Path,
) -> None:
    db = tmp_path / "raw-schema.db"
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute(
            """
            CREATE TABLE events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL, conversation_id TEXT NOT NULL,
                platform TEXT NOT NULL, role TEXT NOT NULL, source TEXT NOT NULL,
                content TEXT NOT NULL, persona_mode TEXT, route_bucket TEXT,
                model_hint TEXT, sensitivity TEXT NOT NULL DEFAULT 'normal',
                category TEXT NOT NULL DEFAULT 'unknown',
                subcategory TEXT NOT NULL DEFAULT 'unknown',
                inject_policy TEXT NOT NULL DEFAULT 'retrieve_only',
                classification_confidence REAL NOT NULL DEFAULT 0.0,
                classifier_version TEXT NOT NULL DEFAULT 'unknown',
                created_at TEXT NOT NULL DEFAULT '2026-07-29T00:00:00Z'
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE ingest_events (
                ingest_id INTEGER PRIMARY KEY, external_id TEXT UNIQUE,
                source_hash TEXT, kind TEXT, event_id INTEGER,
                attachments TEXT DEFAULT '[]', payload_metadata TEXT DEFAULT '{}',
                created_at TEXT DEFAULT '2026-07-29T00:00:00Z'
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE memory_records (
                record_id INTEGER PRIMARY KEY, candidate_id TEXT UNIQUE,
                kind TEXT, target_file TEXT, content TEXT, confidence REAL,
                sensitivity TEXT, source_event_ids TEXT, source_node_ids TEXT,
                status TEXT, reviewer TEXT, reviewed_at TEXT,
                decision_reason TEXT, patch_suggestion TEXT,
                metadata TEXT DEFAULT '{}',
                created_at TEXT DEFAULT '2026-07-29T00:00:00Z'
            )
            """
        )
        ensure_evidence_ledger_schema(conn)
        content = "raw schema legacy source"
        conn.execute(
            """
            INSERT INTO memory_records(
                record_id, candidate_id, kind, target_file, content, confidence,
                sensitivity, source_event_ids, source_node_ids, status
            ) VALUES (1, 'raw', 'memory_note', 'USER.md', ?, 1.0,
                      'normal', '[]', '[]', 'approved')
            """,
            (content,),
        )
        claim_id = int(conn.execute(
            "INSERT INTO memory_claims(canonical_key, target, memory_type) "
            "VALUES ('raw:legacy', 'profile', 'preference')",
        ).lastrowid)
        version_id = int(conn.execute(
            """
            INSERT INTO memory_claim_versions(
                claim_id, version, content, content_sha256, confidence,
                sensitivity, injection_policy, mode_scope, lineage_kind,
                schema_version
            ) VALUES (?, 1, ?, ?, 1.0, 'normal', 'allow', '["daily"]',
                      'legacy_governed', 1)
            """,
            (claim_id, content, sha256_text(content)),
        ).lastrowid)
        conn.execute(
            """
            INSERT INTO memory_claim_sources(
                claim_version_id, source_kind, legacy_record_id,
                legacy_content_sha256
            ) VALUES (?, 'legacy_record', 1, ?)
            """,
            (version_id, sha256_text(content)),
        )
    finally:
        conn.close()


def test_legacy_memfs_guard_blocks_cross_connection_source_change_until_ack(
    tmp_path: Path,
) -> None:
    db = tmp_path / "authority.db"
    store = EventStore(db)
    root = tmp_path / "memfs"
    blocked = {"source_change": False}
    try:
        record_id, _ = store.add_memory_record(
            candidate_id="legacy-memfs-final-reopen",
            kind="memory_note",
            target_file="USER.md",
            content="legacy memfs final reopen token",
            confidence=1.0,
            sensitivity="normal",
            status="approved",
        )
        promoted = LegacyMemoryPromotionService(store).promote(
            LegacyMemoryPromotionRequest((_spec(record_id),))
        )

        def supersede_source(_path: Path) -> None:
            attacker = EventStore(db)
            try:
                with pytest.raises(
                    sqlite3.IntegrityError,
                    match="legacy memory source projection guarded",
                ):
                    attacker._conn.execute(
                        "UPDATE memory_records SET status = 'superseded' WHERE record_id = ?",
                        (record_id,),
                    )
                attacker._conn.rollback()
                blocked["source_change"] = True
            finally:
                attacker.close()

        result = MemoryMemfsProjector(
            store,
            memfs_root=root,
            worker_id="legacy-final-reopen",
            before_replace=supersede_source,
        ).run_once(
            now="2026-07-29T06:00:00Z",
            lease_until="2026-07-29T06:01:00Z",
        )
        job = store._conn.execute(
            """
            SELECT status, last_error FROM projection_outbox
            WHERE projection_kind = 'memory_memfs'
            """
        ).fetchone()
        guard_count = int(store._conn.execute(
            "SELECT count(*) FROM memory_projection_guards"
        ).fetchone()[0])
        source_status = store._conn.execute(
            "SELECT status FROM memory_records WHERE record_id = ?",
            (record_id,),
        ).fetchone()[0]
    finally:
        store.close()

    path = root / "claims" / f"{promoted.items[0].claim_id:016d}.md"
    assert blocked["source_change"] is True
    assert result == {"claimed": 1, "applied": 1, "failed": 0, "obsolete": 0}
    assert tuple(job) == ("applied", None)
    assert guard_count == 0
    assert source_status == "approved"
    assert path.exists() is True


def test_promotion_rejects_outer_transaction_without_rolling_it_back(
    tmp_path: Path,
) -> None:
    store = EventStore(tmp_path / "authority.db")
    try:
        record_id, _ = store.add_memory_record(
            candidate_id="legacy-outer-transaction",
            kind="memory_note",
            target_file="USER.md",
            content="legacy outer transaction token",
            confidence=1.0,
            sensitivity="normal",
            status="approved",
        )
        store._conn.execute("BEGIN")
        store._conn.execute(
            "UPDATE memory_records SET decision_reason = 'caller-owned' "
            "WHERE record_id = ?",
            (record_id,),
        )
        with pytest.raises(RuntimeError, match="promotion_requires_transaction_ownership"):
            LegacyMemoryPromotionService(store).promote(
                LegacyMemoryPromotionRequest((_spec(record_id),))
            )
        still_owned = store._conn.in_transaction
        visible_to_caller = store._conn.execute(
            "SELECT decision_reason FROM memory_records WHERE record_id = ?",
            (record_id,),
        ).fetchone()[0]
        store._conn.rollback()
    finally:
        store.close()

    assert still_owned is True
    assert visible_to_caller == "caller-owned"


def test_transcript_projector_rejects_legacy_authority_job(
    tmp_path: Path,
) -> None:
    store = EventStore(tmp_path / "authority.db")
    try:
        event_id = store.append_event(
            session_id="s",
            conversation_id="c",
            platform="desktop",
            source="chat",
            role="user",
            content="transcript provenance isolation token",
            category="raw_conversation",
            subcategory="user",
            inject_policy="retrieve_only",
        )
        store._conn.execute(
            """
            UPDATE projection_outbox
            SET authority_kind = 'legacy_record', authority_id = '1'
            WHERE event_seq = ? AND projection_kind = 'transcript_chunks'
            """,
            (event_id,),
        )
        store._conn.commit()

        result = TranscriptChunkProjector(
            store, worker_id="transcript-authority-isolation",
        ).run_once(
            now="2026-07-29T07:00:00Z",
            lease_until="2026-07-29T07:01:00Z",
        )
        job = store._conn.execute(
            """
            SELECT status, last_error FROM projection_outbox
            WHERE event_seq = ? AND projection_kind = 'transcript_chunks'
            """,
            (event_id,),
        ).fetchone()
        chunk_count = int(store._conn.execute(
            "SELECT count(*) FROM event_chunks WHERE event_id = ?",
            (event_id,),
        ).fetchone()[0])
    finally:
        store.close()

    assert result == {"claimed": 1, "applied": 0, "failed": 1}
    assert tuple(job) == ("pending", "transcript projection authority mismatch")
    assert chunk_count == 0


@pytest.mark.parametrize("authority_id", ["", "+1", "01", " 1", "1 ", "１", "١"])
def test_fresh_outbox_schema_rejects_noncanonical_authority_identity(
    tmp_path: Path,
    authority_id: str,
) -> None:
    store = EventStore(tmp_path / "authority.db")
    try:
        with pytest.raises(sqlite3.IntegrityError):
            store._conn.execute(
                """
                INSERT INTO projection_outbox(
                    event_seq, authority_kind, authority_id, projection_kind,
                    aggregate_id, aggregate_version, payload_sha256, status
                ) VALUES (41, 'event', ?, 'transcript_chunks', ?, 1, ?, 'pending')
                """,
                (authority_id, f"invalid:{authority_id}", "a" * 64),
            )
    finally:
        store.close()


def test_fresh_outbox_schema_rejects_missing_authority_provenance(
    tmp_path: Path,
) -> None:
    store = EventStore(tmp_path / "authority.db")
    try:
        with pytest.raises(sqlite3.IntegrityError):
            store._conn.execute(
                """
                INSERT INTO projection_outbox(
                    event_seq, projection_kind, aggregate_id, aggregate_version,
                    payload_sha256, status
                ) VALUES (41, 'transcript_chunks', '41', 1, ?, 'pending')
                """,
                ("a" * 64,),
            )
    finally:
        store.close()




def test_promotion_replay_rejects_non_active_current_claim(
    tmp_path: Path,
) -> None:
    store = EventStore(tmp_path / "authority.db")
    try:
        record_id, _ = store.add_memory_record(
            candidate_id="legacy-replay-retired",
            kind="memory_note",
            target_file="USER.md",
            content="legacy replay retired token",
            confidence=1.0,
            sensitivity="normal",
            status="approved",
        )
        service = LegacyMemoryPromotionService(store)
        promoted = service.promote(LegacyMemoryPromotionRequest((_spec(record_id),)))
        item = promoted.items[0]
        version_id = int(store._conn.execute(
            "SELECT claim_version_id FROM memory_current WHERE claim_id = ?",
            (item.claim_id,),
        ).fetchone()[0])
        governance_id = int(store._conn.execute(
            """
            INSERT INTO memory_governance_events(
                claim_id, claim_version_id, action, previous_state, new_state,
                actor, reason_code, policy_version
            ) VALUES (?, ?, 'retire', 'active', 'retired',
                      'governor', 'explicit_retirement', 'memory-policy-v1')
            """,
            (item.claim_id, version_id),
        ).lastrowid)
        store._conn.execute(
            """
            UPDATE memory_current
            SET memory_governance_id = ?, lifecycle_state = 'retired'
            WHERE claim_id = ?
            """,
            (governance_id, item.claim_id),
        )
        store._conn.commit()

        with pytest.raises(ValueError, match="legacy_promotion_not_active"):
            service.promote(LegacyMemoryPromotionRequest((_spec(record_id),)))
        counts = tuple(store._conn.execute(
            """
            SELECT (SELECT count(*) FROM memory_claims),
                   (SELECT count(*) FROM memory_claim_versions),
                   (SELECT count(*) FROM memory_claim_sources)
            """
        ).fetchone())
    finally:
        store.close()

    assert counts == (1, 1, 1)


def test_promotion_replay_rejects_ambiguous_legacy_source_history(
    tmp_path: Path,
) -> None:
    store = EventStore(tmp_path / "authority.db")
    try:
        record_id, _ = store.add_memory_record(
            candidate_id="legacy-replay-ambiguous",
            kind="memory_note",
            target_file="USER.md",
            content="legacy replay ambiguous token",
            confidence=1.0,
            sensitivity="normal",
            status="approved",
        )
        service = LegacyMemoryPromotionService(store)
        service.promote(LegacyMemoryPromotionRequest((_spec(record_id),)))
        # Simulate an older/corrupt database that predates the global
        # one-legacy-record/one-claim uniqueness gate. The service must still
        # fail closed rather than hiding ambiguity with fetchone().
        store._conn.execute("DROP INDEX uq_memory_claim_sources_legacy_record")
        content_hash = sha256_text("legacy replay ambiguous token")
        second_claim_id = int(store._conn.execute(
            "INSERT INTO memory_claims(canonical_key, target, memory_type) "
            "VALUES ('legacy:ambiguous:second', 'profile', 'preference')",
        ).lastrowid)
        second_version_id = int(store._conn.execute(
            """
            INSERT INTO memory_claim_versions(
                claim_id, version, content, content_sha256, confidence,
                sensitivity, injection_policy, mode_scope, lineage_kind,
                schema_version
            ) VALUES (?, 1, 'legacy replay ambiguous token', ?, 1.0,
                      'normal', 'allow', '["daily"]', 'legacy_governed', 1)
            """,
            (second_claim_id, content_hash),
        ).lastrowid)
        store._conn.execute(
            """
            INSERT INTO memory_claim_sources(
                claim_version_id, source_kind, legacy_record_id,
                legacy_content_sha256
            ) VALUES (?, 'legacy_record', ?, ?)
            """,
            (second_version_id, record_id, content_hash),
        )
        second_governance_id = int(store._conn.execute(
            """
            INSERT INTO memory_governance_events(
                claim_id, claim_version_id, action, previous_state, new_state,
                actor, reason_code, policy_version
            ) VALUES (?, ?, 'activate', 'pending_review', 'active',
                      'legacy_memory_promotion', 'legacy_promotion_allowed',
                      'memory-policy-v1')
            """,
            (second_claim_id, second_version_id),
        ).lastrowid)
        store._conn.execute(
            """
            INSERT INTO memory_current(
                claim_id, claim_version_id, memory_governance_id,
                lifecycle_state
            ) VALUES (?, ?, ?, 'active')
            """,
            (second_claim_id, second_version_id, second_governance_id),
        )
        store._conn.commit()

        with pytest.raises(ValueError, match="legacy_promotion_ambiguous"):
            service.promote(LegacyMemoryPromotionRequest((_spec(record_id),)))
    finally:
        store.close()


def test_schema_rejects_promoting_one_legacy_record_into_two_claims(
    tmp_path: Path,
) -> None:
    store = EventStore(tmp_path / "authority.db")
    try:
        record_id, _ = store.add_memory_record(
            candidate_id="legacy-global-uniqueness",
            kind="memory_note",
            target_file="USER.md",
            content="legacy global uniqueness token",
            confidence=1.0,
            sensitivity="normal",
            status="approved",
        )
        LegacyMemoryPromotionService(store).promote(
            LegacyMemoryPromotionRequest((_spec(record_id),))
        )
        content_hash = sha256_text("legacy global uniqueness token")
        claim_id = int(store._conn.execute(
            "INSERT INTO memory_claims(canonical_key, target, memory_type) "
            "VALUES ('legacy:global:second', 'profile', 'preference')",
        ).lastrowid)
        version_id = int(store._conn.execute(
            """
            INSERT INTO memory_claim_versions(
                claim_id, version, content, content_sha256, confidence,
                sensitivity, injection_policy, mode_scope, lineage_kind,
                schema_version
            ) VALUES (?, 1, 'legacy global uniqueness token', ?, 1.0,
                      'normal', 'allow', '["daily"]', 'legacy_governed', 1)
            """,
            (claim_id, content_hash),
        ).lastrowid)
        with pytest.raises(sqlite3.IntegrityError, match="UNIQUE constraint failed"):
            store._conn.execute(
                """
                INSERT INTO memory_claim_sources(
                    claim_version_id, source_kind, legacy_record_id,
                    legacy_content_sha256
                ) VALUES (?, 'legacy_record', ?, ?)
                """,
                (version_id, record_id, content_hash),
            )
    finally:
        store.close()


def test_projected_legacy_memory_fails_closed_after_source_is_superseded(
    tmp_path: Path,
) -> None:
    store = EventStore(tmp_path / "authority.db")
    root = tmp_path / "memfs"
    try:
        record_id, _ = store.add_memory_record(
            candidate_id="legacy-post-projection-supersede",
            kind="memory_note",
            target_file="USER.md",
            content="legacy post projection supersede token",
            confidence=1.0,
            sensitivity="normal",
            status="approved",
        )
        LegacyMemoryPromotionService(store).promote(
            LegacyMemoryPromotionRequest((_spec(record_id),))
        )
        assert MemoryFtsProjector(store, worker_id="legacy-post-fts").run_once(
            now="2026-07-29T08:00:00Z",
            lease_until="2026-07-29T08:01:00Z",
        )["applied"] == 1
        assert MemoryMemfsProjector(
            store, memfs_root=root, worker_id="legacy-post-memfs",
        ).run_once(
            now="2026-07-29T08:00:00Z",
            lease_until="2026-07-29T08:01:00Z",
        )["applied"] == 1
        before = search_governed_memories(
            store,
            GovernedMemorySearchRequest(
                query="post projection supersede",
                persona_mode=PersonaMode.DAILY,
            ),
        )
        store._conn.execute(
            "UPDATE memory_records SET status = 'superseded' WHERE record_id = ?",
            (record_id,),
        )
        store._conn.commit()
        after = search_governed_memories(
            store,
            GovernedMemorySearchRequest(
                query="post projection supersede",
                persona_mode=PersonaMode.DAILY,
            ),
        )
        injection = build_governed_memory_context(
            store, before, persona_mode=PersonaMode.DAILY, total_budget=200,
        )
    finally:
        store.close()

    assert before.status is MemoryRetrievalStatus.OK
    assert after.status is MemoryRetrievalStatus.ABSTAINED
    assert after.reason == "authority_reopen_failed"
    assert injection.status is GovernedInjectionStatus.ABSTAINED
    assert injection.reason == "authority_receipt_changed"


def test_promotion_rejects_legacy_source_hidden_in_non_current_history(
    tmp_path: Path,
) -> None:
    store = EventStore(tmp_path / "authority.db")
    try:
        record_id, _ = store.add_memory_record(
            candidate_id="legacy-hidden-history",
            kind="memory_note",
            target_file="USER.md",
            content="legacy hidden history token",
            confidence=1.0,
            sensitivity="normal",
            status="approved",
        )
        service = LegacyMemoryPromotionService(store)
        promoted = service.promote(LegacyMemoryPromotionRequest((_spec(record_id),)))
        claim_id = promoted.items[0].claim_id
        store._conn.execute("DROP INDEX uq_memory_claim_sources_legacy_record")
        current = store._conn.execute(
            "SELECT claim_version_id FROM memory_current WHERE claim_id = ?",
            (claim_id,),
        ).fetchone()
        version_id = int(store._conn.execute(
            """
            INSERT INTO memory_claim_versions(
                claim_id, version, content, content_sha256, confidence,
                sensitivity, injection_policy, mode_scope, lineage_kind,
                schema_version
            ) VALUES (?, 2, 'replacement system source', ?, 1.0,
                      'normal', 'allow', '["daily"]',
                      'system_governed_invariant', 1)
            """,
            (claim_id, sha256_text("replacement system source")),
        ).lastrowid)
        store._conn.execute(
            """
            INSERT INTO memory_claim_sources(claim_version_id, source_kind)
            VALUES (?, 'system')
            """,
            (version_id,),
        )
        governance_id = int(store._conn.execute(
            """
            INSERT INTO memory_governance_events(
                claim_id, claim_version_id, action, previous_state, new_state,
                actor, reason_code, policy_version
            ) VALUES (?, ?, 'activate', 'pending_review', 'active',
                      'governor', 'replacement', 'memory-policy-v1')
            """,
            (claim_id, version_id),
        ).lastrowid)
        store._conn.execute(
            """
            UPDATE memory_current
            SET claim_version_id = ?, memory_governance_id = ?
            WHERE claim_id = ? AND claim_version_id = ?
            """,
            (version_id, governance_id, claim_id, int(current["claim_version_id"])),
        )
        store._conn.commit()

        with pytest.raises(ValueError, match="legacy_promotion_not_active"):
            service.promote(LegacyMemoryPromotionRequest((
                _spec(record_id, canonical_key="legacy:hidden:history:second"),
            )))
        claim_count = int(store._conn.execute(
            "SELECT count(*) FROM memory_claims",
        ).fetchone()[0])
        source_count = int(store._conn.execute(
            """
            SELECT count(*) FROM memory_claim_sources
            WHERE source_kind = 'legacy_record' AND legacy_record_id = ?
            """,
            (record_id,),
        ).fetchone()[0])
    finally:
        store.close()

    assert claim_count == 1
    assert source_count == 1
