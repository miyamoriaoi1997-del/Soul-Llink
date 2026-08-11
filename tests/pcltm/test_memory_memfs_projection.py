from __future__ import annotations

import json
import multiprocessing
import os
import sqlite3
import time
from pathlib import Path

import yaml
import pytest

from pcltm.memory_contracts import PersonaMode, Sensitivity
from pcltm.memory_write_service import MemoryWriteRequest, MemoryWriteService
from pcltm.memfs_store import MemFSStore
from pcltm.projections.memory_memfs import MemoryMemfsProjector
from pcltm.store import EventStore


def _write_claim(store: EventStore, *, key: str = "memfs-001"):
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


def _parse_claim_file(path: Path) -> tuple[dict, str]:
    text = path.read_text(encoding="utf-8")
    _, raw, body = text.split("---\n", 2)
    return yaml.safe_load(raw), body.rstrip("\n")


def _hold_claim_lock(db_path: str, root: str, ready, release) -> None:
    store = EventStore(Path(db_path))
    try:
        projector = MemoryMemfsProjector(
            store, memfs_root=Path(root), worker_id="lock-holder",
        )
        with projector._claim_lock(1):
            ready.set()
            release.wait(5)
    finally:
        store.close()


def _enter_claim_lock(db_path: str, root: str, entered) -> None:
    store = EventStore(Path(db_path))
    try:
        projector = MemoryMemfsProjector(
            store, memfs_root=Path(root), worker_id="lock-waiter",
        )
        with projector._claim_lock(1):
            entered.set()
    finally:
        store.close()


def test_memory_memfs_writes_claim_id_locator_and_acks(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "authority.db")
    try:
        receipt = _write_claim(store)
        result = MemoryMemfsProjector(
            store, memfs_root=tmp_path / "memfs", worker_id="memfs-worker",
        ).run_once(now="2026-07-29T00:00:00Z", lease_until="2026-07-29T00:01:00Z")
        job = store._conn.execute(
            "SELECT status FROM projection_outbox WHERE projection_kind = 'memory_memfs'"
        ).fetchone()
    finally:
        store.close()

    path = tmp_path / "memfs" / "claims" / f"{receipt.claim_id:016d}.md"
    frontmatter, body = _parse_claim_file(path)
    assert result == {"claimed": 1, "applied": 1, "failed": 0, "obsolete": 0}
    assert job["status"] == "applied"
    assert path.is_file()
    assert body == "用户偏好 UTC+8"
    assert frontmatter == {
        "description": "Governed memory claim: profile:timezone",
        "schema_version": 1,
        "projection_kind": "memory_memfs",
        "projection_generation": 1,
        "claim_id": receipt.claim_id,
        "claim_version": receipt.claim_version,
        "canonical_key": "profile:timezone",
        "target": "profile",
        "memory_type": "preference",
        "lifecycle_state": "active",
        "content_sha256": "ba739a9c09553a438a87a9d9b2c5706401395bdeb32f48d9ae46feeff8593ed5",
        "governance_id": receipt.governance_id,
        "governance_state": "active",
        "policy_version": "memory-policy-v1",
        "sensitivity": "normal",
        "mode_scope": ["daily"],
        "injection_policy": "allow",
        "read_only_projection": True,
        "read_only": True,
        "authority_refs": [{
            "authority_kind": "event", "object_id": "1", "object_version": 1,
            "payload_sha256": "ba739a9c09553a438a87a9d9b2c5706401395bdeb32f48d9ae46feeff8593ed5",
        }],
        "evidence_refs": [{
            "authority_kind": "event", "object_id": "1", "object_version": 1,
            "payload_sha256": "ba739a9c09553a438a87a9d9b2c5706401395bdeb32f48d9ae46feeff8593ed5",
        }],
    }


def test_memory_memfs_claim_projection_is_readable_by_memfs_index(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "authority.db")
    root = tmp_path / "memfs"
    try:
        receipt = _write_claim(store, key="memfs-reader-contract")
        result = MemoryMemfsProjector(
            store, memfs_root=root, worker_id="memfs-worker",
        ).run_once(now="2026-07-29T00:00:00Z", lease_until="2026-07-29T00:01:00Z")
    finally:
        store.close()

    assert result["applied"] == 1
    items = MemFSStore(root).list_tree()
    item = next(item for item in items if item.path.endswith(f"{receipt.claim_id:016d}.md"))
    assert item.description == "Governed memory claim: profile:timezone"
    assert item.memory_type == "UserPreference"
    assert item.lifecycle_state == "active"
    assert item.read_only is True
    assert item.evidence_refs[0]["authority_kind"] == "event"


@pytest.mark.parametrize(("event_seq", "authority_id"), [(True, "1"), (0, "0"), (1, 1)])
def test_memory_memfs_current_check_rejects_noncanonical_event_authority(
    tmp_path: Path, event_seq: object, authority_id: object,
) -> None:
    store = EventStore(tmp_path / "authority.db")
    try:
        receipt = _write_claim(store, key="memfs-noncanonical")
        projector = MemoryMemfsProjector(
            store, memfs_root=tmp_path / "memfs", worker_id="memfs-worker",
        )
        job = store._conn.execute(
            "SELECT * FROM projection_outbox WHERE projection_kind='memory_memfs'",
        ).fetchone()
        malformed = {**dict(job), "event_seq": event_seq, "authority_id": authority_id}
        assert projector._job_is_current(malformed, int(receipt.claim_id)) is False
    finally:
        store.close()


def test_memory_memfs_file_io_does_not_hold_sqlite_write_lock(tmp_path: Path) -> None:
    db = tmp_path / "authority.db"
    store = EventStore(db)
    observed = {"second_writer_acquired": False}
    try:
        _write_claim(store, key="memfs-no-db-lock")

        def before_replace(_path: Path) -> None:
            other = sqlite3.connect(db, timeout=0.1)
            try:
                other.execute("BEGIN IMMEDIATE")
                observed["second_writer_acquired"] = True
                other.rollback()
            finally:
                other.close()

        result = MemoryMemfsProjector(
            store, memfs_root=tmp_path / "memfs", worker_id="memfs-worker",
            before_replace=before_replace,
        ).run_once(now="2026-07-29T00:00:00Z", lease_until="2026-07-29T00:01:00Z")
    finally:
        store.close()

    assert result["applied"] == 1
    assert observed["second_writer_acquired"] is True


def test_memory_memfs_replay_after_ack_loss_is_idempotent(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "authority.db")
    root = tmp_path / "memfs"
    try:
        receipt = _write_claim(store, key="memfs-ack-loss")
        first = MemoryMemfsProjector(
            store, memfs_root=root, worker_id="old-worker", simulate_ack_loss=True,
        ).run_once(now="2026-07-29T00:00:00Z", lease_until="2026-07-29T00:01:00Z")
        path = root / "claims" / f"{receipt.claim_id:016d}.md"
        first_bytes = path.read_bytes()
        store._conn.execute(
            "UPDATE projection_outbox SET lease_until = ? WHERE projection_kind = 'memory_memfs'",
            ("2026-07-28T23:59:59Z",),
        )
        store._conn.commit()
        second = MemoryMemfsProjector(
            store, memfs_root=root, worker_id="new-worker",
        ).run_once(now="2026-07-29T00:02:00Z", lease_until="2026-07-29T00:03:00Z")
        status = store._conn.execute(
            "SELECT status, attempt_count FROM projection_outbox WHERE projection_kind = 'memory_memfs'"
        ).fetchone()
        guard_count = int(store._conn.execute(
            "SELECT count(*) FROM memory_projection_guards"
        ).fetchone()[0])
    finally:
        store.close()

    assert first == {"claimed": 1, "applied": 0, "failed": 1, "obsolete": 0}
    assert second == {"claimed": 1, "applied": 1, "failed": 0, "obsolete": 0}
    assert path.read_bytes() == first_bytes
    assert tuple(status) == ("applied", 2)
    assert guard_count == 0
    assert list((root / "claims").glob("*.md")) == [path]
    assert list((root / "claims").glob("*.tmp")) == []


def test_memory_memfs_orphan_guard_is_reclaimed_by_next_locked_attempt(
    tmp_path: Path,
) -> None:
    store = EventStore(tmp_path / "authority.db")
    root = tmp_path / "memfs"
    try:
        receipt = _write_claim(store, key="memfs-orphan-guard")
        old_job = store.claim_projection_jobs(
            worker_id="dead-worker", projection_kind="memory_memfs", limit=1,
            now="2026-07-29T00:00:00Z", lease_until="2026-07-29T00:01:00Z",
        )[0]
        store._conn.execute(
            """
            INSERT INTO memory_projection_guards(
                claim_id, outbox_id, attempt_count, worker_id, memfs_root_id
            ) VALUES (?, ?, ?, 'dead-worker', ?)
            """,
            (
                receipt.claim_id, int(old_job["outbox_id"]),
                int(old_job["attempt_count"]), os.path.normcase(str(root.resolve())),
            ),
        )
        store._conn.execute(
            "UPDATE projection_outbox SET lease_until = '2026-07-28T23:59:59Z' WHERE outbox_id = ?",
            (int(old_job["outbox_id"]),),
        )
        store._conn.commit()

        result = MemoryMemfsProjector(
            store, memfs_root=root, worker_id="recovery-worker",
        ).run_once(
            now="2026-07-29T00:02:00Z",
            lease_until="2026-07-29T00:03:00Z",
        )
        status = store._conn.execute(
            "SELECT status, attempt_count FROM projection_outbox WHERE outbox_id = ?",
            (int(old_job["outbox_id"]),),
        ).fetchone()
        guard_count = int(store._conn.execute(
            "SELECT count(*) FROM memory_projection_guards"
        ).fetchone()[0])
    finally:
        store.close()

    assert result == {"claimed": 1, "applied": 1, "failed": 0, "obsolete": 0}
    assert tuple(status) == ("applied", 2)
    assert guard_count == 0


def test_memory_memfs_different_root_cannot_reclaim_live_guard(
    tmp_path: Path,
) -> None:
    store = EventStore(tmp_path / "authority.db")
    root_a = tmp_path / "memfs-a"
    root_b = tmp_path / "memfs-b"
    try:
        receipt = _write_claim(store, key="memfs-root-conflict")
        job = store.claim_projection_jobs(
            worker_id="root-a-worker", projection_kind="memory_memfs", limit=1,
            now="2026-07-29T00:00:00Z", lease_until="2026-07-29T00:01:00Z",
        )[0]
        store._conn.execute(
            """
            INSERT INTO memory_projection_guards(
                claim_id, outbox_id, attempt_count, worker_id, memfs_root_id
            ) VALUES (?, ?, ?, 'root-a-worker', ?)
            """,
            (
                receipt.claim_id, int(job["outbox_id"]),
                int(job["attempt_count"]),
                os.path.normcase(str(root_a.resolve())),
            ),
        )
        store._conn.commit()

        forged = dict(job)
        forged["lease_owner"] = "root-b-worker"
        projector = MemoryMemfsProjector(
            store, memfs_root=root_b, worker_id="root-b-worker",
        )
        with pytest.raises(RuntimeError, match="memory projection root conflict"):
            projector._apply_claim_file(forged)
        guard = store._conn.execute(
            "SELECT worker_id, memfs_root_id FROM memory_projection_guards"
        ).fetchone()
        path_b = root_b / "claims" / f"{receipt.claim_id:016d}.md"
    finally:
        store.close()

    assert tuple(guard) == (
        "root-a-worker", os.path.normcase(str(root_a.resolve())),
    )
    assert path_b.exists() is False


def test_memory_memfs_ack_and_guard_release_roll_back_together(
    tmp_path: Path,
) -> None:
    store = EventStore(tmp_path / "authority.db")
    try:
        receipt = _write_claim(store, key="memfs-ack-atomic")
        job = store.claim_projection_jobs(
            worker_id="atomic-worker", projection_kind="memory_memfs", limit=1,
            now="2026-07-29T00:00:00Z", lease_until="2026-07-29T00:01:00Z",
        )[0]
        store._conn.execute(
            """
            INSERT INTO memory_projection_guards(
                claim_id, outbox_id, attempt_count, worker_id, memfs_root_id
            ) VALUES (?, ?, ?, 'atomic-worker', 'test-root')
            """,
            (receipt.claim_id, int(job["outbox_id"]), int(job["attempt_count"])),
        )
        store._conn.execute(
            """
            CREATE TRIGGER fail_guard_release
            BEFORE DELETE ON memory_projection_guards
            BEGIN SELECT RAISE(ABORT, 'forced guard release failure'); END
            """
        )
        store._conn.commit()

        with pytest.raises(sqlite3.IntegrityError, match="forced guard release failure"):
            store.ack_memory_memfs_projection_job(
                int(job["outbox_id"]), claim_id=receipt.claim_id,
                worker_id="atomic-worker",
                expected_attempt_count=int(job["attempt_count"]),
                now="2026-07-29T00:00:30Z",
            )
        status = store._conn.execute(
            "SELECT status, lease_owner, applied_at FROM projection_outbox WHERE outbox_id = ?",
            (int(job["outbox_id"]),),
        ).fetchone()
        guard_count = int(store._conn.execute(
            "SELECT count(*) FROM memory_projection_guards"
        ).fetchone()[0])
    finally:
        store.close()

    assert tuple(status) == ("processing", "atomic-worker", None)
    assert guard_count == 1


def test_memory_memfs_rejects_missing_event_authority_id_in_claimed_job(
    tmp_path: Path,
) -> None:
    store = EventStore(tmp_path / "authority.db")
    root = tmp_path / "memfs"
    try:
        receipt = _write_claim(store, key="memfs-missing-authority")
        job = store.claim_projection_jobs(
            worker_id="memfs-worker", projection_kind="memory_memfs", limit=1,
            now="2026-07-29T00:00:00Z", lease_until="2026-07-29T00:01:00Z",
        )[0]
        job["authority_id"] = ""
        with pytest.raises(
            ValueError,
            match="memory projection source commitment mismatch",
        ):
            MemoryMemfsProjector(
                store, memfs_root=root, worker_id="memfs-worker",
            )._apply_claim_file(job)
        guard_count = int(store._conn.execute(
            "SELECT count(*) FROM memory_projection_guards"
        ).fetchone()[0])
        path = root / "claims" / f"{receipt.claim_id:016d}.md"
    finally:
        store.close()

    assert guard_count == 0
    assert path.exists() is False


def test_claim_projection_is_not_consumed_by_legacy_prompt_layers(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "authority.db")
    root = tmp_path / "memfs"
    try:
        _write_claim(store, key="memfs-not-layer")
        MemoryMemfsProjector(store, memfs_root=root, worker_id="memfs-worker").run_once(
            now="2026-07-29T00:00:00Z", lease_until="2026-07-29T00:01:00Z",
        )
    finally:
        store.close()

    legacy = MemFSStore(root)
    for layer in ("system", "pinned", "episodic", "transient"):
        assert legacy.load_layer(layer, budget_chars=10000).items == []
        assert legacy.search("UTC", layers=(layer,)) == []


def test_memory_memfs_projects_all_multi_source_authority_refs(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "multi-source.db")
    root = tmp_path / "memfs"
    try:
        for session_id in ("multi-memfs-1", "multi-memfs-2"):
            store.append_event(
                session_id=session_id, conversation_id=session_id, platform="test",
                role="user", source="chat", content="我长期偏好简洁报告。",
                persona_mode="work",
            )
        from pcltm.candidates import PersonaCandidateExtractor
        from pcltm.candidate_promotion import CandidatePromotionService

        candidate = PersonaCandidateExtractor(store).extract(
            scope={"session_id": "multi-memfs-2"},
        )[0]
        assert CandidatePromotionService(store).promote([candidate]).activated == 1
        result = MemoryMemfsProjector(
            store, memfs_root=root, worker_id="memfs-worker",
        ).run_once(now="2026-07-29T00:00:00Z", lease_until="2026-07-29T00:01:00Z")
        claim_id = store._conn.execute("SELECT claim_id FROM memory_claims").fetchone()[0]
        frontmatter = yaml.safe_load(
            (root / "claims" / f"{claim_id:016d}.md").read_text(encoding="utf-8")
            .split("---", 2)[1],
        )
    finally:
        store.close()

    assert result == {"claimed": 1, "applied": 1, "failed": 0, "obsolete": 0}
    assert len(frontmatter["authority_refs"]) == 2
    assert frontmatter["authority_refs"] == frontmatter["evidence_refs"]


def test_memory_memfs_rejects_symlinked_claims_directory(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "authority.db")
    root = tmp_path / "memfs"
    outside = tmp_path / "outside"
    outside.mkdir()
    root.mkdir()
    try:
        try:
            (root / "claims").symlink_to(outside, target_is_directory=True)
        except OSError:
            pytest.skip("directory symlink creation is unavailable")
        _write_claim(store, key="memfs-reparse")
        result = MemoryMemfsProjector(
            store, memfs_root=root, worker_id="memfs-worker",
        ).run_once(now="2026-07-29T00:00:00Z", lease_until="2026-07-29T00:01:00Z")
        job = store._conn.execute(
            "SELECT status, last_error FROM projection_outbox WHERE projection_kind = 'memory_memfs'"
        ).fetchone()
    finally:
        store.close()

    assert result == {"claimed": 1, "applied": 0, "failed": 1, "obsolete": 0}
    assert tuple(job) == ("pending", "memfs reparse path rejected")
    assert list(outside.iterdir()) == []


def test_memory_memfs_rejects_symlinked_root_directory(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "authority.db")
    root_link = tmp_path / "memfs-link"
    outside = tmp_path / "outside-root"
    outside.mkdir()
    try:
        try:
            root_link.symlink_to(outside, target_is_directory=True)
        except OSError:
            pytest.skip("directory symlink creation is unavailable")
        _write_claim(store, key="memfs-root-reparse")
        result = MemoryMemfsProjector(
            store, memfs_root=root_link, worker_id="memfs-worker",
        ).run_once(now="2026-07-29T00:00:00Z", lease_until="2026-07-29T00:01:00Z")
        job = store._conn.execute(
            "SELECT status, last_error FROM projection_outbox WHERE projection_kind = 'memory_memfs'"
        ).fetchone()
    finally:
        store.close()

    assert result == {"claimed": 1, "applied": 0, "failed": 1, "obsolete": 0}
    assert tuple(job) == ("pending", "memfs reparse path rejected")
    assert list(outside.iterdir()) == []


def test_memory_memfs_replace_failure_retries_without_temp_or_applied_status(
    tmp_path: Path, monkeypatch,
) -> None:
    root = tmp_path / "memfs"
    store = EventStore(tmp_path / "authority.db")
    try:
        _write_claim(store, key="memfs-replace-failure")

        def fail_replace(self: Path, target: Path) -> Path:
            raise OSError("replace failed")

        monkeypatch.setattr(Path, "replace", fail_replace)
        result = MemoryMemfsProjector(
            store, memfs_root=root, worker_id="memfs-worker",
        ).run_once(now="2026-07-29T00:00:00Z", lease_until="2026-07-29T00:01:00Z")
        job = store._conn.execute(
            "SELECT status, last_error FROM projection_outbox WHERE projection_kind = 'memory_memfs'"
        ).fetchone()
    finally:
        store.close()

    assert result == {"claimed": 1, "applied": 0, "failed": 1, "obsolete": 0}
    assert tuple(job) == ("pending", "replace failed")
    claims = root / "claims"
    assert not claims.exists() or list(claims.iterdir()) == []


def test_memory_memfs_stale_version_becomes_obsolete_without_writing_file(tmp_path: Path) -> None:
    root = tmp_path / "memfs"
    store = EventStore(tmp_path / "authority.db")
    try:
        receipt = _write_claim(store, key="memfs-obsolete")
        store._conn.execute(
            "UPDATE projection_outbox SET aggregate_version = 2 WHERE projection_kind = 'memory_memfs'"
        )
        store._conn.commit()
        result = MemoryMemfsProjector(
            store, memfs_root=root, worker_id="memfs-worker",
        ).run_once(now="2026-07-29T00:00:00Z", lease_until="2026-07-29T00:01:00Z")
        job = store._conn.execute(
            "SELECT status, last_error FROM projection_outbox WHERE projection_kind = 'memory_memfs'"
        ).fetchone()
    finally:
        store.close()

    assert result == {"claimed": 1, "applied": 0, "failed": 0, "obsolete": 1}
    assert tuple(job) == ("obsolete", "stale_projection")
    assert not (root / "claims" / f"{receipt.claim_id:016d}.md").exists()


def test_old_memfs_snapshot_cannot_overwrite_newer_projected_file(
    tmp_path: Path, monkeypatch,
) -> None:
    root = tmp_path / "memfs"
    store = EventStore(tmp_path / "authority.db")
    try:
        receipt = _write_claim(store, key="memfs-race")
        old = MemoryMemfsProjector(store, memfs_root=root, worker_id="old-worker")
        job = store.claim_projection_jobs(
            worker_id="old-worker", projection_kind="memory_memfs", limit=1,
            now="2026-07-29T00:00:00Z", lease_until="2026-07-29T00:10:00Z",
        )[0]
        stale_snapshot = old._load_snapshot(job)

        version = store._conn.execute(
            """
            INSERT INTO memory_claim_versions(
                claim_id, version, content, content_sha256, confidence, sensitivity,
                injection_policy, mode_scope, lineage_kind, schema_version
            ) VALUES (?, 2, ?, ?, 1.0, 'normal', 'allow', '["daily"]',
                      'explicit_user_assertion', 1)
            """,
            (receipt.claim_id, "用户偏好 UTC+9", "9ae26dcf5e6b0c771802ffe673f705d989203a0ce3bbf1dc341bf99deb3af71c"),
        )
        version_id = int(version.lastrowid)
        source = store._conn.execute(
            """
            SELECT event_id, event_revision, event_payload_sha256
            FROM memory_claim_sources LIMIT 1
            """
        ).fetchone()
        store._conn.execute(
            """
            INSERT INTO memory_claim_sources(
                claim_version_id, source_kind, event_id, event_revision, event_payload_sha256
            ) VALUES (?, 'event', ?, ?, ?)
            """,
            (version_id, source["event_id"], source["event_revision"], source["event_payload_sha256"]),
        )
        governance = store._conn.execute(
            """
            INSERT INTO memory_governance_events(
                claim_id, claim_version_id, action, previous_state, new_state,
                actor, reason_code, policy_version
            ) VALUES (?, ?, 'activate', 'pending_review', 'active', 'test', 'write_allowed', 'memory-policy-v1')
            """,
            (receipt.claim_id, version_id),
        )
        governance_id = int(governance.lastrowid)
        store._conn.execute(
            """
            UPDATE memory_current SET claim_version_id=?, memory_governance_id=?
            WHERE claim_id=?
            """,
            (version_id, governance_id, receipt.claim_id),
        )
        store._conn.execute(
            """
            INSERT INTO projection_outbox(
                event_seq, authority_kind, authority_id, projection_kind,
                aggregate_id, aggregate_version, payload_sha256, status
            ) VALUES (?, 'event', ?, 'memory_memfs', ?, 2, ?, 'pending')
            """,
            (
                source["event_id"], str(source["event_id"]),
                f"memory:{receipt.claim_id}",
                "9ae26dcf5e6b0c771802ffe673f705d989203a0ce3bbf1dc341bf99deb3af71c",
            ),
        )
        store._conn.commit()
        newer = MemoryMemfsProjector(store, memfs_root=root, worker_id="new-worker")
        newer.run_once(now="2026-07-29T00:01:00Z", lease_until="2026-07-29T00:02:00Z")

        monkeypatch.setattr(old, "_load_snapshot", lambda _job: stale_snapshot)
        outcome = old._apply_claim_file(job)
        path = root / "claims" / f"{receipt.claim_id:016d}.md"
        _, body = _parse_claim_file(path)
    finally:
        store.close()

    assert outcome == "obsolete"
    assert body == "用户偏好 UTC+9"


def test_claim_file_lock_serializes_windows_processes(tmp_path: Path) -> None:
    db = tmp_path / "authority.db"
    root = tmp_path / "memfs"
    store = EventStore(db)
    store.close()
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    release = context.Event()
    entered = context.Event()
    holder = context.Process(target=_hold_claim_lock, args=(str(db), str(root), ready, release))
    waiter = context.Process(target=_enter_claim_lock, args=(str(db), str(root), entered))
    holder.start()
    try:
        assert ready.wait(5)
        waiter.start()
        time.sleep(0.3)
        assert entered.is_set() is False
        release.set()
        assert entered.wait(5)
    finally:
        release.set()
        holder.join(5)
        waiter.join(5)
        if holder.is_alive():
            holder.terminate()
        if waiter.is_alive():
            waiter.terminate()

    assert holder.exitcode == 0
    assert waiter.exitcode == 0


def test_memory_memfs_rejects_preexisting_symlink_lock_file(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "authority.db")
    root = tmp_path / "memfs"
    outside = tmp_path / "outside.lock"
    outside.write_bytes(b"0")
    locks = root / ".claim-locks"
    locks.mkdir(parents=True)
    try:
        try:
            (locks / "0000000000000001.lock").symlink_to(outside)
        except OSError:
            pytest.skip("file symlink creation is unavailable")
        _write_claim(store, key="memfs-lock-reparse")
        result = MemoryMemfsProjector(
            store, memfs_root=root, worker_id="memfs-worker",
        ).run_once(now="2026-07-29T00:00:00Z", lease_until="2026-07-29T00:01:00Z")
        job = store._conn.execute(
            "SELECT status, last_error FROM projection_outbox WHERE projection_kind='memory_memfs'"
        ).fetchone()
    finally:
        store.close()

    assert result == {"claimed": 1, "applied": 0, "failed": 1, "obsolete": 0}
    assert tuple(job) == ("pending", "memfs reparse path rejected")
    assert outside.read_bytes() == b"0"


def test_memory_memfs_rechecks_claims_identity_immediately_before_replace(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "authority.db")
    root = tmp_path / "memfs"
    outside = tmp_path / "outside-swap"
    outside.mkdir()
    try:
        _write_claim(store, key="memfs-claims-swap")

        def swap_claims(_path: Path) -> None:
            claims = root / "claims"
            claims.rmdir()
            try:
                claims.symlink_to(outside, target_is_directory=True)
            except OSError:
                pytest.skip("directory symlink creation is unavailable")

        result = MemoryMemfsProjector(
            store, memfs_root=root, worker_id="memfs-worker", before_replace=swap_claims,
        ).run_once(now="2026-07-29T00:00:00Z", lease_until="2026-07-29T00:01:00Z")
        job = store._conn.execute(
            "SELECT status, last_error FROM projection_outbox WHERE projection_kind='memory_memfs'"
        ).fetchone()
    finally:
        store.close()

    assert result == {"claimed": 1, "applied": 0, "failed": 1, "obsolete": 0}
    assert tuple(job) == ("pending", "memfs reparse path rejected")
    assert list(outside.iterdir()) == []


def test_memory_memfs_current_change_in_before_replace_is_guarded_across_connections(
    tmp_path: Path,
) -> None:
    db = tmp_path / "authority.db"
    store = EventStore(db)
    root = tmp_path / "memfs"
    blocked = {"current_switch": False}
    try:
        receipt = _write_claim(store, key="memfs-current-swap")

        def advance_current(_path: Path) -> None:
            attacker = EventStore(db)
            try:
                attacker._conn.execute("BEGIN IMMEDIATE")
                version = attacker._conn.execute(
                    """
                    INSERT INTO memory_claim_versions(
                        claim_id, version, content, content_sha256, confidence, sensitivity,
                        injection_policy, mode_scope, lineage_kind, schema_version
                    ) VALUES (?, 2, '用户偏好 UTC+9', ?, 1.0, 'normal', 'allow',
                              '["daily"]', 'explicit_user_assertion', 1)
                    """,
                    (receipt.claim_id, "9ae26dcf5e6b0c771802ffe673f705d989203a0ce3bbf1dc341bf99deb3af71c"),
                )
                version_id = int(version.lastrowid)
                source = attacker._conn.execute(
                    """
                    SELECT event_id, event_revision, event_payload_sha256
                    FROM memory_claim_sources WHERE claim_version_id = (
                        SELECT claim_version_id FROM memory_current WHERE claim_id = ?
                    )
                    """,
                    (receipt.claim_id,),
                ).fetchone()
                attacker._conn.execute(
                    """
                    INSERT INTO memory_claim_sources(
                        claim_version_id, source_kind, event_id, event_revision,
                        event_payload_sha256
                    ) VALUES (?, 'event', ?, ?, ?)
                    """,
                    (
                        version_id, source["event_id"], source["event_revision"],
                        source["event_payload_sha256"],
                    ),
                )
                governance = attacker._conn.execute(
                    """
                    INSERT INTO memory_governance_events(
                        claim_id, claim_version_id, action, previous_state, new_state,
                        actor, reason_code, policy_version
                    ) VALUES (?, ?, 'activate', 'pending_review', 'active', 'test',
                              'write_allowed', 'memory-policy-v1')
                    """,
                    (receipt.claim_id, version_id),
                )
                try:
                    attacker._conn.execute(
                        """
                        UPDATE memory_current
                        SET claim_version_id = ?, memory_governance_id = ?
                        WHERE claim_id = ?
                        """,
                        (version_id, int(governance.lastrowid), receipt.claim_id),
                    )
                except sqlite3.IntegrityError as exc:
                    assert str(exc) == "memory current projection guarded"
                    blocked["current_switch"] = True
                    attacker._conn.rollback()
                else:
                    raise AssertionError("authority switch bypassed projection guard")
            finally:
                attacker.close()

        result = MemoryMemfsProjector(
            store, memfs_root=root, worker_id="memfs-worker",
            before_replace=advance_current,
        ).run_once(
            now="2026-07-29T00:00:00Z",
            lease_until="2026-07-29T00:01:00Z",
        )
        job = store._conn.execute(
            "SELECT status, last_error FROM projection_outbox WHERE projection_kind='memory_memfs'"
        ).fetchone()
        guard_count = int(store._conn.execute(
            "SELECT count(*) FROM memory_projection_guards"
        ).fetchone()[0])
        path = root / "claims" / f"{receipt.claim_id:016d}.md"
    finally:
        store.close()

    assert blocked["current_switch"] is True
    assert result == {"claimed": 1, "applied": 1, "failed": 0, "obsolete": 0}
    assert tuple(job) == ("applied", None)
    assert guard_count == 0
    assert path.exists() is True
