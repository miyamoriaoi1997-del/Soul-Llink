from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from pcltm.memory_contracts import AuthorityRef, LineageKind, PersonaMode, Sensitivity
from pcltm.memory_write_service import MemoryWriteRequest, MemoryWriteService
from pcltm.store import EventStore


def _request(key: str = "case-001", **kwargs) -> MemoryWriteRequest:
    values = dict(
        idempotency_key=key,
        content="用户偏好 UTC+8",
        canonical_key="profile:timezone",
        target="profile",
        memory_type="preference",
        sensitivity=Sensitivity.NORMAL,
        mode_scope=(PersonaMode.DAILY,),
        injection_policy="allow",
    )
    values.update(kwargs)
    return MemoryWriteRequest(**values)


def _counts(conn: sqlite3.Connection) -> dict[str, int]:
    names = ("events", "event_governance", "ingest_events", "memory_claims", "memory_claim_versions", "memory_claim_sources",
             "memory_governance_events", "memory_current", "projection_outbox", "event_chain_state", "event_fts")
    return {name: int(conn.execute(f"SELECT count(*) FROM {name}").fetchone()[0]) for name in names}


def test_explicit_assertion_commits_authority_and_pending_projection_outbox(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "authority.db")
    try:
        receipt = MemoryWriteService(store).write(_request())
    finally:
        store.close()
    assert receipt.success is True
    assert receipt.status == "active"
    assert receipt.persisted is True
    assert receipt.projection_status == "pending"
    assert receipt.recall_ready is False
    assert receipt.claim_id and receipt.claim_version == 1 and receipt.governance_id
    with sqlite3.connect(tmp_path / "authority.db") as conn:
        assert conn.execute("SELECT source, role, category, subcategory FROM events").fetchone() == (
            "memory_assertion", "user", "memory_assertion", "explicit",
        )
        assert conn.execute("SELECT count(*) FROM memory_claims").fetchone()[0] == 1
        kinds = conn.execute("SELECT projection_kind FROM projection_outbox ORDER BY projection_kind").fetchall()
        assert [row[0] for row in kinds] == ["memory_fts", "memory_memfs", "transcript_chunks", "transcript_fts"]
        assert conn.execute("SELECT count(*) FROM event_fts").fetchone()[0] == 0
        assert conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name IN ('memory_fts','memory_memfs')").fetchall() == []


def test_assertion_snapshot_uses_real_event_governance_without_self_source(tmp_path: Path, monkeypatch) -> None:
    import pcltm.memory_write_service as write_module

    observed = {}
    real_admit = write_module.policy.admit_write

    def capture(command, source_snapshots=()):
        observed["snapshot"] = source_snapshots[0]
        return real_admit(command, source_snapshots)

    monkeypatch.setattr(write_module.policy, "admit_write", capture)
    store = EventStore(tmp_path / "snapshot.db")
    try:
        receipt = MemoryWriteService(store).write(_request("snapshot"))
        governance = store._conn.execute(
            "SELECT governance_id FROM event_governance WHERE event_id = 1"
        ).fetchone()
    finally:
        store.close()

    assert receipt.success is True
    assert governance is not None
    assert observed["snapshot"].governance_id == governance["governance_id"]
    assert observed["snapshot"].source_refs == ()


@pytest.mark.parametrize(
    ("event_revision", "event_payload_sha256"),
    [(2, "a" * 64), (1, "b" * 64)],
)
def test_store_bootstrap_rejects_mismatched_event_source_commitments(
    tmp_path: Path, event_revision: int, event_payload_sha256: str,
) -> None:
    store = EventStore(tmp_path / "source-integrity.db")
    try:
        receipt = MemoryWriteService(store).write(_request("source-integrity"))
        version_id = store._conn.execute(
            "SELECT claim_version_id FROM memory_claim_versions WHERE claim_id = ?",
            (receipt.claim_id,),
        ).fetchone()[0]
        with pytest.raises(sqlite3.IntegrityError, match="event source commitment mismatch"):
            store._conn.execute(
                """
                INSERT INTO memory_claim_sources(
                    claim_version_id, source_kind, event_id, event_revision,
                    event_payload_sha256
                ) VALUES (?, 'event', 1, ?, ?)
                """,
                (version_id, event_revision, event_payload_sha256),
            )
    finally:
        store.close()


def test_assertion_snapshot_uses_persisted_event_sensitivity(tmp_path: Path, monkeypatch) -> None:
    from types import SimpleNamespace
    import pcltm.memory_write_service as write_module
    import pcltm.store as store_module

    observed = {}
    real_admit = write_module.policy.admit_write

    def classify(*args, **kwargs):
        return SimpleNamespace(
            sensitivity="private", category="memory_assertion", subcategory="explicit",
            inject_policy="allow", confidence=1.0, classifier_version="test",
        )

    def capture(command, source_snapshots=()):
        observed["snapshot"] = source_snapshots[0]
        return real_admit(command, source_snapshots)

    monkeypatch.setattr(store_module.EventClassifier, "classify", classify)
    monkeypatch.setattr(write_module.policy, "admit_write", capture)
    store = EventStore(tmp_path / "persisted-sensitivity.db")
    try:
        receipt = MemoryWriteService(store).write(_request("persisted-sensitivity"))
        counts = _counts(store._conn)
    finally:
        store.close()

    assert receipt.success is False
    assert receipt.reason_code == "sensitivity_downgrade"
    assert observed["snapshot"].sensitivity is Sensitivity.PRIVATE
    assert counts == {name: 0 for name in counts}


def test_event_derived_without_source_refs_is_rejected_without_side_effects(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "reject.db")
    try:
        receipt = MemoryWriteService(store).write(_request(lineage_kind=LineageKind.EVENT_DERIVED))
        counts = _counts(store._conn)
    finally:
        store.close()
    assert receipt == type(receipt)(False, "rejected", None, None, None, False, "none", False, "source_snapshot_missing")
    assert counts == {name: 0 for name in counts}


def test_event_derived_rejects_mismatched_caller_source_without_residue(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "derived-source.db")
    try:
        event_id = store.append_event(
            session_id="s", conversation_id="c", platform="test", role="user", source="chat",
            content="[memory] OPAQUE_SOURCE_TOKEN", persona_mode="daily",
        )
        from pcltm.memory_contracts import AuthorityRef, LineageKind, Sensitivity
        from pcltm.memory_write_service import MemoryWriteRequest, MemoryWriteService
        bad_ref = AuthorityRef("event", str(event_id), 99, "0" * 64)
        receipt = MemoryWriteService(store).write(MemoryWriteRequest(
            idempotency_key="bad-derived", content="[memory] OPAQUE_SOURCE_TOKEN",
            canonical_key="opaque-derived", target="USER.md", memory_type="user_preference",
            sensitivity=Sensitivity.NORMAL, mode_scope=(PersonaMode.DAILY,), injection_policy="allow",
            lineage_kind=LineageKind.EVENT_DERIVED, source_refs=(bad_ref,),
        ))
        assert not receipt.success
        assert receipt.reason_code == "source_snapshot_mismatch"
        assert store._conn.execute("SELECT COUNT(*) AS n FROM memory_claims").fetchone()["n"] == 0
        assert store._conn.execute("SELECT COUNT(*) AS n FROM ingest_events WHERE kind = 'memory_derived'").fetchone()["n"] == 0
        assert store._conn.execute("SELECT COUNT(*) AS n FROM projection_outbox WHERE aggregate_id LIKE 'memory:%'").fetchone()["n"] == 0
    finally:
        store.close()


def test_event_derived_replay_is_idempotent_without_new_event(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "derived-replay.db")
    try:
        event_id = store.append_event(
            session_id="s", conversation_id="c", platform="test", role="user", source="chat",
            content="[memory] OPAQUE_REPLAY_TOKEN", persona_mode="daily",
        )
        source = store.get_event(event_id)
        from pcltm.memory_contracts import AuthorityRef, LineageKind, Sensitivity
        from pcltm.memory_write_service import MemoryWriteRequest, MemoryWriteService
        request = MemoryWriteRequest(
            idempotency_key="derived-replay", content="[memory] OPAQUE_REPLAY_TOKEN",
            canonical_key="opaque-derived-replay", target="USER.md", memory_type="user_preference",
            sensitivity=Sensitivity.NORMAL, mode_scope=(PersonaMode.DAILY,), injection_policy="allow",
            lineage_kind=LineageKind.EVENT_DERIVED,
            source_refs=(AuthorityRef("event", str(event_id), source["source_revision"], source["payload_sha256"]),),
        )
        service = MemoryWriteService(store)
        first = service.write(request)
        events_before = store._conn.execute("SELECT COUNT(*) AS n FROM events").fetchone()["n"]
        second = service.write(request)
        events_after = store._conn.execute("SELECT COUNT(*) AS n FROM events").fetchone()["n"]
        assert second == first
        assert events_after == events_before
    finally:
        store.close()


def test_existing_canonical_key_does_not_bypass_derived_authority(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "derived-existing.db")
    try:
        service = MemoryWriteService(store)
        assert service.write(_request("seed-existing")).success
        event_id = store.append_event(
            session_id="s", conversation_id="c", platform="test", role="user",
            source="chat", content="ordinary unrelated task", persona_mode="daily",
        )
        event = store.get_event(event_id)
        ref = AuthorityRef("event", str(event_id), int(event["source_revision"]), str(event["payload_sha256"]))
        request = _request(
            "invalid-existing", lineage_kind=LineageKind.EVENT_DERIVED,
            source_refs=(ref,),
        )
        before = _counts(store._conn)
        receipt = service.write(request)
        after = _counts(store._conn)
    finally:
        store.close()
    assert receipt.success is False
    assert receipt.reason_code == "source_snapshot_mismatch"
    assert after == before


def test_source_ref_cardinality_is_rejected_for_every_lineage(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "source-cardinality.db")
    try:
        event_id = store.append_event(
            session_id="s", conversation_id="c", platform="test", role="user",
            source="chat", content="[memory:timezone] 用户偏好 UTC+8", persona_mode="daily",
        )
        event = store.get_event(event_id)
        ref = AuthorityRef("event", str(event_id), int(event["source_revision"]), str(event["payload_sha256"]))
        service = MemoryWriteService(store)
        explicit = service.write(_request("explicit-ref", source_refs=(ref,)))
        derived = service.write(_request(
            "derived-multi", lineage_kind=LineageKind.EVENT_DERIVED,
            source_refs=(ref, ref),
        ))
    finally:
        store.close()
    assert explicit.success is False
    assert explicit.reason_code == "source_snapshot_missing"
    assert derived.success is False
    assert derived.reason_code == "source_snapshot_missing"


def test_non_numeric_derived_event_id_is_typed_rejection(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "non-numeric-source.db")
    try:
        ref = AuthorityRef("event", "not-an-integer", 1, "0" * 64)
        receipt = MemoryWriteService(store).write(_request(
            "bad-object-id", lineage_kind=LineageKind.EVENT_DERIVED,
            source_refs=(ref,),
        ))
    finally:
        store.close()
    assert receipt.success is False
    assert receipt.reason_code == "source_snapshot_mismatch"


@pytest.mark.parametrize("object_id", ["+1", "01", " 1", "1 ", "١"])
def test_noncanonical_derived_event_id_is_rejected(tmp_path: Path, object_id: str) -> None:
    store = EventStore(tmp_path / "noncanonical-source.db")
    try:
        ref = AuthorityRef("event", object_id, 1, "0" * 64)
        receipt = MemoryWriteService(store).write(_request(
            "bad-canonical-object-id", lineage_kind=LineageKind.EVENT_DERIVED,
            source_refs=(ref,),
        ))
    finally:
        store.close()
    assert receipt.success is False
    assert receipt.reason_code == "source_snapshot_mismatch"

def test_private_source_cannot_be_downgraded_to_normal(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "sensitivity.db")
    try:
        receipt = MemoryWriteService(store).write(
            _request(source_sensitivity=Sensitivity.PRIVATE)
        )
        counts = _counts(store._conn)
    finally:
        store.close()
    assert receipt.success is False
    assert receipt.reason_code == "sensitivity_downgrade"
    assert counts == {name: 0 for name in counts}


def test_idempotency_same_payload_replays_identical_receipt_and_adds_nothing(tmp_path: Path) -> None:
    db = tmp_path / "idem.db"
    store = EventStore(db)
    try:
        service = MemoryWriteService(store)
        first = service.write(_request("same"))
        counts_before = _counts(store._conn)
        second = service.write(_request("same"))
        counts_after = _counts(store._conn)
    finally:
        store.close()
    assert second == first
    assert counts_after == counts_before


def test_idempotency_replay_keeps_original_receipt_after_projection_state_changes(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "idem-status.db")
    try:
        service = MemoryWriteService(store)
        first = service.write(_request("same-status"))
        store._conn.execute(
            "UPDATE projection_outbox SET status='applied' WHERE projection_kind IN ('memory_fts', 'memory_memfs')"
        )
        store._conn.commit()
        replay = service.write(_request("same-status"))
    finally:
        store.close()
    assert replay == first


def test_idempotency_changed_payload_returns_stable_conflict_without_new_rows(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "conflict.db")
    try:
        service = MemoryWriteService(store)
        service.write(_request("same"))
        counts_before = _counts(store._conn)
        conflict = service.write(_request("same", content="changed"))
        counts_after = _counts(store._conn)
    finally:
        store.close()
    assert conflict == type(conflict)(False, "rejected", None, None, None, False, "none", False, "idempotency_conflict")
    assert counts_after == counts_before


def test_different_command_for_existing_canonical_key_returns_typed_conflict_without_side_effects(
    tmp_path: Path,
) -> None:
    store = EventStore(tmp_path / "canonical-conflict.db")
    try:
        service = MemoryWriteService(store)
        service.write(_request("first-command"))
        counts_before = _counts(store._conn)
        conflict = service.write(_request("second-command"))
        counts_after = _counts(store._conn)
    finally:
        store.close()
    assert conflict.success is False
    assert conflict.reason_code == "canonical_key_conflict"
    assert counts_after == counts_before


def test_idempotency_hash_binds_event_provenance(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "provenance.db")
    try:
        service = MemoryWriteService(store)
        service.write(_request("same-provenance"))
        counts_before = _counts(store._conn)
        conflict = service.write(_request("same-provenance", session_id="different-session"))
        counts_after = _counts(store._conn)
    finally:
        store.close()
    assert conflict.reason_code == "idempotency_conflict"
    assert counts_after == counts_before


def test_idempotency_replay_resolves_original_authority_through_source_binding(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "binding.db")
    try:
        service = MemoryWriteService(store)
        first = service.write(_request("bound"))
        store._conn.execute(
            "UPDATE ingest_events SET payload_metadata = ? WHERE external_id = ?",
            ('{"canonical_key":"wrong:key"}', "memory-assertion:bound"),
        )
        store._conn.commit()
        replay = service.write(_request("bound"))
    finally:
        store.close()
    assert replay == first


@pytest.mark.parametrize("checkpoint", ["assertion_after", "claim_version_after", "outbox_before_commit"])
def test_fault_at_each_checkpoint_rolls_back_complete_authority_transaction(tmp_path: Path, checkpoint: str) -> None:
    store = EventStore(tmp_path / f"fault-{checkpoint}.db")
    def fail(name: str) -> None:
        if name == checkpoint:
            raise RuntimeError("forced fault")
    try:
        with pytest.raises(RuntimeError, match="forced fault"):
            MemoryWriteService(store, fault_hook=fail).write(_request(checkpoint))
        counts = _counts(store._conn)
        assert store.verify_event_chain()["checked"] == 0
    finally:
        store.close()
    assert counts == {name: 0 for name in counts}
