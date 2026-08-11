from __future__ import annotations

from pathlib import Path

import pytest

from pcltm.memory_contracts import PersonaMode, Sensitivity
from pcltm.memory_write_service import MemoryWriteRequest, MemoryWriteService
from pcltm.projections.memory_fts import MemoryFtsProjector
from pcltm.store import EventStore


def _write_claim(store: EventStore, *, key: str = "fts-001"):
    return MemoryWriteService(store).write(
        MemoryWriteRequest(
            idempotency_key=key,
            content="用户偏好 UTC+8",
            canonical_key="profile:timezone",
            target="profile",
            memory_type="preference",
            sensitivity=Sensitivity.NORMAL,
            mode_scope=(PersonaMode.DAILY,),
            injection_policy="allow",
        )
    )


def _fts_count(store: EventStore) -> int:
    exists = store._conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name = 'memory_fts'"
    ).fetchone()
    if exists is None:
        return 0
    return int(store._conn.execute("SELECT count(*) FROM memory_fts").fetchone()[0])


def test_memory_fts_materialization_and_ack_commit_atomically(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "authority.db")
    try:
        receipt = _write_claim(store)
        result = MemoryFtsProjector(store, worker_id="fts-worker").run_once(
            now="2026-07-29T00:00:00Z",
            lease_until="2026-07-29T00:01:00Z",
        )
        row = store._conn.execute(
            """
            SELECT rowid, claim_version, governance_id, payload_sha256, content
                 , projection_generation, policy_version, source_refs
            FROM memory_fts WHERE rowid = ?
            """,
            (receipt.claim_id,),
        ).fetchone()
        status = store._conn.execute(
            """
            SELECT status FROM projection_outbox
            WHERE projection_kind = 'memory_fts' AND aggregate_id = ?
            """,
            (f"memory:{receipt.claim_id}",),
        ).fetchone()[0]
    finally:
        store.close()

    assert result == {"claimed": 1, "applied": 1, "failed": 0, "obsolete": 0}
    assert tuple(row) == (
        receipt.claim_id,
        receipt.claim_version,
        receipt.governance_id,
        "ba739a9c09553a438a87a9d9b2c5706401395bdeb32f48d9ae46feeff8593ed5",
        "用户偏好 UTC+8",
        1,
        "memory-policy-v1",
        '[{"authority_kind":"event","object_id":"1","object_version":1,"payload_sha256":"ba739a9c09553a438a87a9d9b2c5706401395bdeb32f48d9ae46feeff8593ed5"}]',
    )
    assert status == "applied"


@pytest.mark.parametrize(("event_seq", "authority_id"), [(True, "1"), ("01", "1"), (1, 1)])
def test_memory_fts_rejects_noncanonical_event_authority(
    tmp_path: Path, event_seq: object, authority_id: object,
) -> None:
    store = EventStore(tmp_path / "authority.db")
    try:
        _write_claim(store, key="fts-noncanonical")
        projector = MemoryFtsProjector(store, worker_id="fts-worker")
        original_claim = store.claim_projection_jobs

        def claim_with_malformed_authority(**kwargs):
            jobs = original_claim(**kwargs)
            return [{**job, "event_seq": event_seq, "authority_id": authority_id} for job in jobs]

        store.claim_projection_jobs = claim_with_malformed_authority
        result = projector.run_once(
            now="2026-07-29T00:00:00Z", lease_until="2026-07-29T00:01:00Z",
        )
    finally:
        store.close()
    assert result["applied"] == 0
    assert result["failed"] == 1


def test_memory_fts_failure_before_ack_rolls_back_projection_and_releases_for_retry(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "authority.db")
    try:
        receipt = _write_claim(store, key="fts-fault")

        def fail(name: str) -> None:
            if name == "fts_before_ack":
                raise RuntimeError("forced FTS/ACK boundary failure")

        result = MemoryFtsProjector(
            store, worker_id="fts-worker", fault_hook=fail,
        ).run_once(
            now="2026-07-29T00:00:00Z",
            lease_until="2026-07-29T00:01:00Z",
        )
        fts_count = _fts_count(store)
        job = store._conn.execute(
            """
            SELECT status, attempt_count, last_error
            FROM projection_outbox
            WHERE projection_kind = 'memory_fts' AND aggregate_id = ?
            """,
            (f"memory:{receipt.claim_id}",),
        ).fetchone()
    finally:
        store.close()

    assert result == {"claimed": 1, "applied": 0, "failed": 1, "obsolete": 0}
    assert fts_count == 0
    assert tuple(job) == ("pending", 1, "forced FTS/ACK boundary failure")


def test_memory_fts_rejects_outbox_payload_commitment_mismatch_without_materializing(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "authority.db")
    try:
        receipt = _write_claim(store, key="fts-stale")
        store._conn.execute(
            """
            UPDATE projection_outbox SET payload_sha256 = ?
            WHERE projection_kind = 'memory_fts' AND aggregate_id = ?
            """,
            ("0" * 64, f"memory:{receipt.claim_id}"),
        )
        store._conn.commit()
        result = MemoryFtsProjector(store, worker_id="fts-worker").run_once(
            now="2026-07-29T00:00:00Z",
            lease_until="2026-07-29T00:01:00Z",
        )
        fts_count = _fts_count(store)
        job = store._conn.execute(
            "SELECT status, last_error FROM projection_outbox WHERE projection_kind = 'memory_fts'"
        ).fetchone()
    finally:
        store.close()

    assert result == {"claimed": 1, "applied": 0, "failed": 1, "obsolete": 0}
    assert fts_count == 0
    assert tuple(job) == ("pending", "memory projection payload hash mismatch")


def test_memory_fts_rejects_outbox_event_source_mismatch_without_materializing(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "authority.db")
    try:
        receipt = _write_claim(store, key="fts-source-mismatch")
        other_event = store.append_event(
            session_id="s", conversation_id="c", platform="internal",
            role="user", source="test", content="unrelated event",
            category="raw_conversation", subcategory="user", inject_policy="retrieve_only",
        )
        store._conn.execute(
            """
            UPDATE projection_outbox SET event_seq = ?
            WHERE projection_kind = 'memory_fts' AND aggregate_id = ?
            """,
            (other_event, f"memory:{receipt.claim_id}"),
        )
        store._conn.commit()
        result = MemoryFtsProjector(store, worker_id="fts-worker").run_once(
            now="2026-07-29T00:00:00Z", lease_until="2026-07-29T00:01:00Z",
        )
        fts_count = _fts_count(store)
        job = store._conn.execute(
            "SELECT status, last_error FROM projection_outbox WHERE projection_kind = 'memory_fts'"
        ).fetchone()
    finally:
        store.close()

    assert result == {"claimed": 1, "applied": 0, "failed": 1, "obsolete": 0}
    assert fts_count == 0
    assert tuple(job) == ("pending", "memory projection source commitment mismatch")


def test_memory_fts_rejects_missing_event_authority_id_in_claimed_job(
    tmp_path: Path,
) -> None:
    store = EventStore(tmp_path / "authority.db")
    try:
        _write_claim(store, key="fts-missing-authority")
        job = store.claim_projection_jobs(
            worker_id="fts-worker", projection_kind="memory_fts", limit=1,
            now="2026-07-29T00:00:00Z", lease_until="2026-07-29T00:01:00Z",
        )[0]
        job["authority_id"] = ""
        with pytest.raises(
            ValueError,
            match="memory projection source commitment mismatch",
        ):
            MemoryFtsProjector(store, worker_id="fts-worker")._apply_and_ack(
                job, now="2026-07-29T00:00:30Z",
            )
        fts_count = _fts_count(store)
    finally:
        store.close()

    assert fts_count == 0


def test_memory_fts_projects_multi_source_claim(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "multi-source.db")
    try:
        for session_id in ("multi-fts-1", "multi-fts-2"):
            store.append_event(
                session_id=session_id, conversation_id=session_id, platform="test",
                role="user", source="chat", content="我长期偏好简洁报告。",
                persona_mode="work",
            )
        from pcltm.candidates import PersonaCandidateExtractor
        from pcltm.candidate_promotion import CandidatePromotionService

        candidate = PersonaCandidateExtractor(store).extract(
            scope={"session_id": "multi-fts-2"},
        )[0]
        assert CandidatePromotionService(store).promote([candidate]).activated == 1
        result = MemoryFtsProjector(store, worker_id="fts-worker").run_once(
            now="2026-07-29T00:00:00Z", lease_until="2026-07-29T00:01:00Z",
        )
        source_count = store._conn.execute(
            "SELECT count(*) FROM memory_claim_sources",
        ).fetchone()[0]
        fts_count = _fts_count(store)
    finally:
        store.close()

    assert source_count == 2
    assert result == {"claimed": 1, "applied": 1, "failed": 0, "obsolete": 0}
    assert fts_count == 1


def test_memory_fts_ownership_loss_does_not_release_reclaimed_lease(tmp_path: Path, monkeypatch) -> None:
    store = EventStore(tmp_path / "authority.db")
    try:
        _write_claim(store, key="fts-lease-loss")
        projector = MemoryFtsProjector(store, worker_id="fts-worker")
        original_apply = projector._apply_and_ack

        def lose_lease(job, *, now):
            store._conn.execute(
                """
                UPDATE projection_outbox
                SET lease_owner = 'new-owner', attempt_count = attempt_count + 1
                WHERE outbox_id = ?
                """,
                (job["outbox_id"],),
            )
            store._conn.commit()
            original_apply(job, now=now)

        monkeypatch.setattr(projector, "_apply_and_ack", lose_lease)
        result = projector.run_once(
            now="2026-07-29T00:00:00Z", lease_until="2026-07-29T00:01:00Z",
        )
        job = store._conn.execute(
            """
            SELECT status, lease_owner, attempt_count, last_error
            FROM projection_outbox WHERE projection_kind = 'memory_fts'
            """
        ).fetchone()
    finally:
        store.close()

    assert result == {"claimed": 1, "applied": 0, "failed": 1, "obsolete": 0}
    assert tuple(job) == ("processing", "new-owner", 2, None)


def test_memory_fts_stale_version_becomes_obsolete_without_overwriting_current(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "authority.db")
    try:
        _write_claim(store, key="fts-obsolete")
        store._conn.execute(
            "UPDATE projection_outbox SET aggregate_version = 2 WHERE projection_kind = 'memory_fts'"
        )
        store._conn.commit()
        result = MemoryFtsProjector(store, worker_id="fts-worker").run_once(
            now="2026-07-29T00:00:00Z", lease_until="2026-07-29T00:01:00Z",
        )
        row = store._conn.execute(
            "SELECT status, last_error, lease_owner FROM projection_outbox WHERE projection_kind = 'memory_fts'"
        ).fetchone()
        fts_count = _fts_count(store)
    finally:
        store.close()

    assert result == {"claimed": 1, "applied": 0, "failed": 0, "obsolete": 1}
    assert tuple(row) == ("obsolete", "stale_projection", None)
    assert fts_count == 0
