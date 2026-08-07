from __future__ import annotations

from pathlib import Path

from pcltm.memory_contracts import PersonaMode, Sensitivity
from pcltm.memory_transition_service import (
    MemoryLifecycleRequest,
    MemoryReplaceRequest,
    MemoryTransitionService,
)
from pcltm.memory_write_service import MemoryWriteRequest, MemoryWriteService
from pcltm.projections.memory_runtime import drain_memory_projections
from pcltm.store import EventStore


def _create(store: EventStore, key: str = "timezone"):
    return MemoryWriteService(store).write(MemoryWriteRequest(
        idempotency_key=f"create-{key}",
        content="用户偏好 UTC+8",
        canonical_key=f"profile:{key}",
        target="profile",
        memory_type="preference",
        sensitivity=Sensitivity.NORMAL,
        mode_scope=(PersonaMode.DAILY,),
        injection_policy="allow",
    ))


def test_replace_appends_version_and_governance_then_cas_switches_current(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "authority.db")
    try:
        original = _create(store)
        receipt = MemoryTransitionService(store).replace(MemoryReplaceRequest(
            idempotency_key="replace-timezone-v2",
            claim_id=original.claim_id,
            expected_current_version=1,
            content="用户偏好 UTC+9",
            sensitivity=Sensitivity.NORMAL,
            mode_scope=(PersonaMode.DAILY,),
            injection_policy="allow",
        ))
        current = store._conn.execute(
            """
            SELECT v.version, v.content, mc.lifecycle_state,
                   g.action, g.previous_state, g.new_state
            FROM memory_current mc
            JOIN memory_claim_versions v ON v.claim_version_id = mc.claim_version_id
            JOIN memory_governance_events g ON g.memory_governance_id = mc.memory_governance_id
            WHERE mc.claim_id = ?
            """,
            (original.claim_id,),
        ).fetchone()
        governance = [tuple(row) for row in store._conn.execute(
            """
            SELECT v.version, g.action, g.previous_state, g.new_state
            FROM memory_governance_events g
            JOIN memory_claim_versions v ON v.claim_version_id = g.claim_version_id
            WHERE g.claim_id = ? ORDER BY g.memory_governance_id
            """,
            (original.claim_id,),
        ).fetchall()]
        jobs = [tuple(row) for row in store._conn.execute(
            """
            SELECT projection_kind, aggregate_version, status
            FROM projection_outbox
            WHERE aggregate_id = ? AND projection_kind LIKE 'memory_%'
            ORDER BY aggregate_version, projection_kind
            """,
            (f"memory:{original.claim_id}",),
        ).fetchall()]
    finally:
        store.close()

    assert receipt.success is True
    assert receipt.status == "active"
    assert receipt.claim_id == original.claim_id
    assert receipt.claim_version == 2
    assert tuple(current) == (2, "用户偏好 UTC+9", "active", "activate", "pending_review", "active")
    assert governance == [
        (1, "activate", "pending_review", "active"),
        (1, "supersede", "active", "superseded"),
        (2, "submit", None, "pending_review"),
        (2, "activate", "pending_review", "active"),
    ]
    assert jobs == [
        ("memory_fts", 1, "pending"),
        ("memory_memfs", 1, "pending"),
        ("memory_fts", 2, "pending"),
        ("memory_memfs", 2, "pending"),
    ]


def test_replace_stale_expected_version_is_typed_conflict_without_side_effects(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "authority.db")
    try:
        original = _create(store)
        before = tuple(store._conn.execute(
            """
            SELECT
              (SELECT count(*) FROM events),
              (SELECT count(*) FROM memory_claim_versions),
              (SELECT count(*) FROM memory_governance_events),
              (SELECT count(*) FROM projection_outbox)
            """
        ).fetchone())
        receipt = MemoryTransitionService(store).replace(MemoryReplaceRequest(
            idempotency_key="stale-replace",
            claim_id=original.claim_id,
            expected_current_version=2,
            content="不得写入",
            sensitivity=Sensitivity.NORMAL,
            mode_scope=(PersonaMode.DAILY,),
            injection_policy="allow",
        ))
        after = tuple(store._conn.execute(
            """
            SELECT
              (SELECT count(*) FROM events),
              (SELECT count(*) FROM memory_claim_versions),
              (SELECT count(*) FROM memory_governance_events),
              (SELECT count(*) FROM projection_outbox)
            """
        ).fetchone())
    finally:
        store.close()

    assert receipt.success is False
    assert receipt.status == "conflict"
    assert receipt.reason_code == "stale_expected_version"
    assert after == before


def test_derived_replace_replay_is_idempotent(tmp_path: Path) -> None:
    from pcltm.classifier import EventClassifier
    from pcltm.memory_contracts import AuthorityRef, LineageKind

    store = EventStore(tmp_path / "authority.db")
    try:
        original = _create(store)
        classification = EventClassifier().classify(
            role="user", source="chat", content="[replace:timezone] OPAQUE_V2",
            persona_mode="daily",
        )
        event_id = store.append_event(
            session_id="s", conversation_id="c", platform="test", role="user",
            source="chat", content="[replace:timezone] OPAQUE_V2", persona_mode="daily",
            category=classification.category, subcategory=classification.subcategory,
            inject_policy=classification.inject_policy,
        )
        event = store.get_event(event_id)
        request = MemoryReplaceRequest(
            idempotency_key="derived-replay", claim_id=original.claim_id,
            expected_current_version=1, content="OPAQUE_V2",
            sensitivity=Sensitivity.NORMAL, mode_scope=(PersonaMode.DAILY,),
            injection_policy="allow", lineage_kind=LineageKind.EVENT_DERIVED,
            source_refs=(AuthorityRef("event", str(event_id), int(event["source_revision"]), str(event["payload_sha256"])),),
        )
        service = MemoryTransitionService(store)
        first = service.replace(request)
        second = service.replace(request)
        versions = store._conn.execute(
            "SELECT count(*) FROM memory_claim_versions WHERE claim_id=?", (original.claim_id,),
        ).fetchone()[0]
    finally:
        store.close()
    assert first == second
    assert first.success is True
    assert versions == 2


def test_retire_switches_current_state_without_new_version_and_removes_projections(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "authority.db")
    root = tmp_path / "memfs"
    try:
        original = _create(store, "retire")
        drain_memory_projections(store, memfs_root=root)
        claim_file = root / "claims" / f"{original.claim_id:016d}.md"
        assert claim_file.exists()

        receipt = MemoryTransitionService(store).retire(MemoryLifecycleRequest(
            idempotency_key="retire-timezone",
            claim_id=original.claim_id,
            expected_current_version=1,
            reason_code="user_requested_retirement",
        ))
        current = store._conn.execute(
            """
            SELECT v.version, mc.lifecycle_state, g.action, g.previous_state, g.new_state
            FROM memory_current mc
            JOIN memory_claim_versions v ON v.claim_version_id = mc.claim_version_id
            JOIN memory_governance_events g ON g.memory_governance_id = mc.memory_governance_id
            WHERE mc.claim_id = ?
            """,
            (original.claim_id,),
        ).fetchone()
        drain_memory_projections(store, memfs_root=root)
        fts = store._conn.execute(
            "SELECT count(*) FROM memory_fts WHERE rowid = ?", (original.claim_id,),
        ).fetchone()[0]
    finally:
        store.close()

    assert receipt.success is True
    assert receipt.status == "retired"
    assert receipt.claim_version == 1
    assert tuple(current) == (1, "retired", "retire", "active", "retired")
    assert fts == 0
    assert not claim_file.exists()


def test_expire_and_lifecycle_replay_are_idempotent(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "authority.db")
    try:
        original = _create(store, "expire")
        request = MemoryLifecycleRequest(
            idempotency_key="expire-timezone",
            claim_id=original.claim_id,
            expected_current_version=1,
            reason_code="ttl_elapsed",
        )
        service = MemoryTransitionService(store)
        first = service.expire(request)
        counts = tuple(store._conn.execute(
            "SELECT count(*) FROM memory_governance_events"
        ).fetchone())
        second = service.expire(request)
        replay_counts = tuple(store._conn.execute(
            "SELECT count(*) FROM memory_governance_events"
        ).fetchone())
    finally:
        store.close()

    assert first == second
    assert first.status == "expired"
    assert replay_counts == counts
