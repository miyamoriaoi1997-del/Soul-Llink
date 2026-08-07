from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from pcltm.memory_contracts import PersonaMode, Sensitivity
from pcltm.memory_projection_rebuild import rebuild_all_memory_projections
from pcltm.memory_write_service import MemoryWriteRequest, MemoryWriteService
from pcltm.projections.memory_runtime import drain_memory_projections
from pcltm.store import EventStore


def _write(store: EventStore, key: str, content: str):
    return MemoryWriteService(store).write(MemoryWriteRequest(
        idempotency_key=f"write-{key}", content=content,
        canonical_key=f"profile:{key}", target="profile",
        memory_type="preference", sensitivity=Sensitivity.NORMAL,
        mode_scope=(PersonaMode.DAILY,), injection_policy="allow",
    ))


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_delete_all_derived_memory_projections_and_rebuild_from_sqlite(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "authority.db")
    root = tmp_path / "memfs"
    try:
        one = _write(store, "one", "first rebuild token")
        two = _write(store, "two", "second rebuild token")
        drain_memory_projections(store, memfs_root=root)
        authority_before = [tuple(row) for row in store._conn.execute(
            """
            SELECT c.claim_id, v.version, v.content_sha256, mc.lifecycle_state,
                   mc.memory_governance_id
            FROM memory_current mc
            JOIN memory_claims c ON c.claim_id = mc.claim_id
            JOIN memory_claim_versions v ON v.claim_version_id = mc.claim_version_id
            ORDER BY c.claim_id
            """
        ).fetchall()]
        expected_hashes = {
            one.claim_id: _hash(root / "claims" / f"{one.claim_id:016d}.md"),
            two.claim_id: _hash(root / "claims" / f"{two.claim_id:016d}.md"),
        }

        result = rebuild_all_memory_projections(store, memfs_root=root)

        authority_after = [tuple(row) for row in store._conn.execute(
            """
            SELECT c.claim_id, v.version, v.content_sha256, mc.lifecycle_state,
                   mc.memory_governance_id
            FROM memory_current mc
            JOIN memory_claims c ON c.claim_id = mc.claim_id
            JOIN memory_claim_versions v ON v.claim_version_id = mc.claim_version_id
            ORDER BY c.claim_id
            """
        ).fetchall()]
        fts_ids = [row[0] for row in store._conn.execute(
            "SELECT rowid FROM memory_fts ORDER BY rowid"
        ).fetchall()]
    finally:
        store.close()

    assert result == {"claims": 2, "memory_fts": 2, "memory_memfs": 2}
    assert authority_after == authority_before
    assert fts_ids == [one.claim_id, two.claim_id]
    assert expected_hashes == {
        one.claim_id: _hash(root / "claims" / f"{one.claim_id:016d}.md"),
        two.claim_id: _hash(root / "claims" / f"{two.claim_id:016d}.md"),
    }


def test_rebuild_refuses_unmanaged_files_in_claims_directory(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "authority.db")
    root = tmp_path / "memfs"
    try:
        _write(store, "one", "safe token")
        drain_memory_projections(store, memfs_root=root)
        unmanaged = root / "claims" / "do-not-delete.txt"
        unmanaged.write_text("sentinel", encoding="utf-8")
        try:
            rebuild_all_memory_projections(store, memfs_root=root)
        except RuntimeError as exc:
            assert str(exc) == "unmanaged_memfs_projection_file"
        else:
            raise AssertionError("unmanaged file was accepted")
    finally:
        store.close()


def test_rebuild_refuses_reparse_or_symlink_memfs_root(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "authority.db")
    real_root = tmp_path / "real-memfs"
    linked_root = tmp_path / "linked-memfs"
    try:
        receipt = _write(store, "linked", "linked root token")
        drain_memory_projections(store, memfs_root=real_root)
        claim_file = real_root / "claims" / f"{receipt.claim_id:016d}.md"
        before = claim_file.read_bytes()
        try:
            linked_root.symlink_to(real_root, target_is_directory=True)
        except OSError as exc:
            pytest.skip(f"directory symlinks unavailable: {exc}")

        with pytest.raises(RuntimeError, match="unmanaged_memfs_projection_file"):
            rebuild_all_memory_projections(store, memfs_root=linked_root)

        assert claim_file.read_bytes() == before
    finally:
        store.close()


def test_rebuild_refuses_nonexistent_root_below_reparse_or_symlink_parent(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "authority.db")
    real_parent = tmp_path / "real-parent"
    linked_parent = tmp_path / "linked-parent"
    real_parent.mkdir()
    try:
        _write(store, "linked-parent", "linked parent token")
        try:
            linked_parent.symlink_to(real_parent, target_is_directory=True)
        except OSError as exc:
            pytest.skip(f"directory symlinks unavailable: {exc}")

        with pytest.raises(RuntimeError, match="unmanaged_memfs_projection_file"):
            rebuild_all_memory_projections(
                store, memfs_root=linked_parent / "new-root",
            )

        assert not (real_parent / "new-root").exists()
    finally:
        store.close()


def test_rebuild_active_claim_replaced_before_commit_is_reprojected_from_authority(
    tmp_path: Path,
) -> None:
    store = EventStore(tmp_path / "authority.db")
    root = tmp_path / "memfs"
    try:
        receipt = _write(store, "replaced", "original projection token")
        drain_memory_projections(store, memfs_root=root)
        claim_file = root / "claims" / f"{receipt.claim_id:016d}.md"
        expected = claim_file.read_bytes()
        replacement = b"replacement must not become authority"

        def replace_before_commit(checkpoint: str) -> None:
            if checkpoint == "authority_before_commit":
                temporary = claim_file.with_suffix(".replacement")
                temporary.write_bytes(replacement)
                temporary.replace(claim_file)

        result = rebuild_all_memory_projections(
            store, memfs_root=root, fault_hook=replace_before_commit,
        )

        assert result == {"claims": 1, "memory_fts": 1, "memory_memfs": 1}
        assert claim_file.read_bytes() == expected
    finally:
        store.close()


def test_rebuild_refuses_active_projection_guard_without_mutating_derived_state(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "authority.db")
    root = tmp_path / "memfs"
    try:
        receipt = _write(store, "guarded", "guarded token")
        drain_memory_projections(store, memfs_root=root)
        before_fts = [tuple(row) for row in store._conn.execute(
            "SELECT rowid, content, payload_sha256 FROM memory_fts ORDER BY rowid"
        )]
        before_outbox = [tuple(row) for row in store._conn.execute(
            "SELECT outbox_id, status, attempt_count FROM projection_outbox ORDER BY outbox_id"
        )]
        job = store._conn.execute(
            "SELECT outbox_id, attempt_count FROM projection_outbox WHERE aggregate_id = ? LIMIT 1",
            (f"memory:{receipt.claim_id}",),
        ).fetchone()
        store._conn.execute(
            """
            INSERT INTO memory_projection_guards(
                claim_id, outbox_id, attempt_count, worker_id, memfs_root_id
            ) VALUES (?, ?, ?, 'live-worker', ?)
            """,
            (receipt.claim_id, int(job["outbox_id"]), int(job["attempt_count"]), str(root.resolve())),
        )
        store._conn.commit()

        with pytest.raises(RuntimeError, match="projection_rebuild_guard_active"):
            rebuild_all_memory_projections(store, memfs_root=root)

        assert [tuple(row) for row in store._conn.execute(
            "SELECT rowid, content, payload_sha256 FROM memory_fts ORDER BY rowid"
        )] == before_fts
        assert [tuple(row) for row in store._conn.execute(
            "SELECT outbox_id, status, attempt_count FROM projection_outbox ORDER BY outbox_id"
        )] == before_outbox
        assert (root / "claims" / f"{receipt.claim_id:016d}.md").is_file()
    finally:
        store.close()


def test_rebuild_fault_before_authority_commit_rolls_back_all_derived_db_changes(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "authority.db")
    root = tmp_path / "memfs"
    try:
        receipt = _write(store, "rollback", "rollback token")
        drain_memory_projections(store, memfs_root=root)
        claim_file = root / "claims" / f"{receipt.claim_id:016d}.md"
        before_file = claim_file.read_bytes()
        before_fts = [tuple(row) for row in store._conn.execute(
            "SELECT rowid, content, payload_sha256 FROM memory_fts ORDER BY rowid"
        )]
        before_outbox = [tuple(row) for row in store._conn.execute(
            "SELECT outbox_id, status, attempt_count FROM projection_outbox ORDER BY outbox_id"
        )]

        def fail(checkpoint: str) -> None:
            if checkpoint == "authority_before_commit":
                raise RuntimeError("forced rebuild fault")

        with pytest.raises(RuntimeError, match="forced rebuild fault"):
            rebuild_all_memory_projections(store, memfs_root=root, fault_hook=fail)

        assert claim_file.read_bytes() == before_file
        assert [tuple(row) for row in store._conn.execute(
            "SELECT rowid, content, payload_sha256 FROM memory_fts ORDER BY rowid"
        )] == before_fts
        assert [tuple(row) for row in store._conn.execute(
            "SELECT outbox_id, status, attempt_count FROM projection_outbox ORDER BY outbox_id"
        )] == before_outbox
    finally:
        store.close()


def test_windows_stale_cleanup_deletes_opened_file_not_path_replacement(
    tmp_path: Path,
) -> None:
    if __import__("os").name != "nt":
        pytest.skip("Windows handle deletion contract")
    store = EventStore(tmp_path / "stale-handle.db")
    root = tmp_path / "memfs"
    try:
        receipt = _write(store, "stale-handle", "active handle token")
        drain_memory_projections(store, memfs_root=root)
        active = root / "claims" / f"{receipt.claim_id:016d}.md"
        expected_active = active.read_bytes()
        stale = root / "claims" / "9999999999999999.md"
        stale.write_bytes(b"old stale object")
        replacement = b"new replacement object"
        replaced = False

        def replace_path_after_open(checkpoint: str) -> None:
            nonlocal replaced
            if checkpoint == "stale_after_open" and not replaced:
                backup = stale.with_suffix(".old")
                stale.replace(backup)
                stale.write_bytes(replacement)
                replaced = True

        rebuild_all_memory_projections(
            store, memfs_root=root, fault_hook=replace_path_after_open,
        )

        assert active.read_bytes() == expected_active
        assert stale.read_bytes() == replacement
        assert not stale.with_suffix(".old").exists()
    finally:
        store.close()


def test_rebuild_stale_cleanup_failure_preserves_converged_active_files(
    tmp_path: Path,
) -> None:
    store = EventStore(tmp_path / "stale-cleanup.db")
    root = tmp_path / "memfs"
    try:
        one = _write(store, "cleanup-one", "cleanup first token")
        two = _write(store, "cleanup-two", "cleanup second token")
        drain_memory_projections(store, memfs_root=root)
        active_paths = [
            root / "claims" / f"{one.claim_id:016d}.md",
            root / "claims" / f"{two.claim_id:016d}.md",
        ]
        expected = {path: path.read_bytes() for path in active_paths}
        stale = root / "claims" / "9999999999999999.md"
        stale.write_text("stale projection", encoding="utf-8")

        def fail_stale(checkpoint: str) -> None:
            if checkpoint == "stale_after_open":
                raise OSError("stale unlink injected")

        with pytest.raises(OSError, match="stale unlink injected"):
            rebuild_all_memory_projections(
                store, memfs_root=root, fault_hook=fail_stale,
            )

        assert {path: path.read_bytes() for path in active_paths} == expected
        assert stale.is_file()
        assert [tuple(row) for row in store._conn.execute(
            "SELECT projection_kind, status FROM projection_outbox "
            "WHERE projection_kind IN ('memory_fts', 'memory_memfs') "
            "ORDER BY aggregate_id, projection_kind"
        )] == [
            ("memory_fts", "applied"), ("memory_memfs", "applied"),
            ("memory_fts", "applied"), ("memory_memfs", "applied"),
        ]
    finally:
        store.close()


def test_rebuild_commit_failure_restores_memfs_and_rolls_back_derived_db(tmp_path: Path) -> None:
    class FailCommitOnce:
        def __init__(self, conn):
            self._conn = conn
            self._failed = False

        def __getattr__(self, name):
            return getattr(self._conn, name)

        def commit(self) -> None:
            if not self._failed:
                self._failed = True
                raise RuntimeError("commit injected")
            self._conn.commit()

    store = EventStore(tmp_path / "commit-failure.db")
    root = tmp_path / "memfs"
    real_conn = store._conn
    try:
        receipt = _write(store, "commit-failure", "commit failure token")
        drain_memory_projections(store, memfs_root=root)
        claim_file = root / "claims" / f"{receipt.claim_id:016d}.md"
        before_file = claim_file.read_bytes()
        before_fts = [tuple(row) for row in real_conn.execute(
            "SELECT rowid, content, payload_sha256 FROM memory_fts ORDER BY rowid"
        )]
        before_outbox = [tuple(row) for row in real_conn.execute(
            "SELECT outbox_id, projection_kind, status, attempt_count "
            "FROM projection_outbox ORDER BY outbox_id"
        )]
        store._conn = FailCommitOnce(real_conn)

        with pytest.raises(RuntimeError, match="commit injected"):
            rebuild_all_memory_projections(store, memfs_root=root)

        assert claim_file.read_bytes() == before_file
        assert [tuple(row) for row in real_conn.execute(
            "SELECT rowid, content, payload_sha256 FROM memory_fts ORDER BY rowid"
        )] == before_fts
        assert [tuple(row) for row in real_conn.execute(
            "SELECT outbox_id, projection_kind, status, attempt_count "
            "FROM projection_outbox ORDER BY outbox_id"
        )] == before_outbox
    finally:
        store._conn = real_conn
        store.close()


def test_rebuild_commit_then_raise_converges_committed_outbox_before_reraising(tmp_path: Path) -> None:
    class CommitThenRaiseOnce:
        def __init__(self, conn):
            self._conn = conn
            self._failed = False

        def __getattr__(self, name):
            return getattr(self._conn, name)

        def commit(self) -> None:
            self._conn.commit()
            if not self._failed:
                self._failed = True
                raise RuntimeError("commit reported failure")

    store = EventStore(tmp_path / "commit-then-raise.db")
    root = tmp_path / "memfs"
    real_conn = store._conn
    try:
        receipt = _write(store, "commit-then-raise", "committed recovery token")
        drain_memory_projections(store, memfs_root=root)
        claim_file = root / "claims" / f"{receipt.claim_id:016d}.md"
        expected = claim_file.read_bytes()
        store._conn = CommitThenRaiseOnce(real_conn)

        with pytest.raises(RuntimeError, match="commit reported failure"):
            rebuild_all_memory_projections(store, memfs_root=root)

        assert claim_file.read_bytes() == expected
        assert [tuple(row) for row in real_conn.execute(
            "SELECT projection_kind, status FROM projection_outbox "
            "WHERE projection_kind IN ('memory_fts', 'memory_memfs') "
            "ORDER BY projection_kind"
        )] == [("memory_fts", "applied"), ("memory_memfs", "applied")]
        assert [int(row[0]) for row in real_conn.execute(
            "SELECT rowid FROM memory_fts ORDER BY rowid"
        )] == [receipt.claim_id]
    finally:
        store._conn = real_conn
        store.close()


def test_rebuild_commit_auto_rollback_then_raise_restores_old_projection(tmp_path: Path) -> None:
    class RollbackThenRaiseOnce:
        def __init__(self, conn):
            self._conn = conn
            self._failed = False

        def __getattr__(self, name):
            return getattr(self._conn, name)

        def commit(self) -> None:
            if not self._failed:
                self._failed = True
                self._conn.rollback()
                raise RuntimeError("commit rolled back")
            self._conn.commit()

    store = EventStore(tmp_path / "commit-auto-rollback.db")
    root = tmp_path / "memfs"
    real_conn = store._conn
    try:
        receipt = _write(store, "commit-auto-rollback", "rollback recovery token")
        drain_memory_projections(store, memfs_root=root)
        claim_file = root / "claims" / f"{receipt.claim_id:016d}.md"
        expected = claim_file.read_bytes()
        before_fts = [tuple(row) for row in real_conn.execute(
            "SELECT rowid, content, payload_sha256 FROM memory_fts ORDER BY rowid"
        )]
        before_outbox = [tuple(row) for row in real_conn.execute(
            "SELECT outbox_id, projection_kind, status, attempt_count "
            "FROM projection_outbox ORDER BY outbox_id"
        )]
        store._conn = RollbackThenRaiseOnce(real_conn)

        with pytest.raises(RuntimeError, match="commit rolled back"):
            rebuild_all_memory_projections(store, memfs_root=root)

        assert claim_file.read_bytes() == expected
        assert [tuple(row) for row in real_conn.execute(
            "SELECT rowid, content, payload_sha256 FROM memory_fts ORDER BY rowid"
        )] == before_fts
        assert [tuple(row) for row in real_conn.execute(
            "SELECT outbox_id, projection_kind, status, attempt_count "
            "FROM projection_outbox ORDER BY outbox_id"
        )] == before_outbox
    finally:
        store._conn = real_conn
        store.close()


def test_rebuild_empty_memfs_commit_then_raise_still_converges(tmp_path: Path) -> None:
    class CommitThenRaiseOnce:
        def __init__(self, conn):
            self._conn = conn
            self._failed = False

        def __getattr__(self, name):
            return getattr(self._conn, name)

        def commit(self) -> None:
            self._conn.commit()
            if not self._failed:
                self._failed = True
                raise RuntimeError("empty commit reported failure")

    store = EventStore(tmp_path / "empty-commit-then-raise.db")
    root = tmp_path / "empty-memfs"
    real_conn = store._conn
    try:
        receipt = _write(store, "empty-commit-then-raise", "empty committed recovery token")
        assert not root.exists()
        store._conn = CommitThenRaiseOnce(real_conn)

        with pytest.raises(RuntimeError, match="empty commit reported failure"):
            rebuild_all_memory_projections(store, memfs_root=root)

        claim_file = root / "claims" / f"{receipt.claim_id:016d}.md"
        assert claim_file.is_file()
        assert [tuple(row) for row in real_conn.execute(
            "SELECT projection_kind, status FROM projection_outbox "
            "WHERE projection_kind IN ('memory_fts', 'memory_memfs') "
            "ORDER BY projection_kind"
        )] == [("memory_fts", "applied"), ("memory_memfs", "applied")]
        assert [int(row[0]) for row in real_conn.execute(
            "SELECT rowid FROM memory_fts ORDER BY rowid"
        )] == [receipt.claim_id]
    finally:
        store._conn = real_conn
        store.close()


@pytest.mark.parametrize("checkpoint", ["authority_after_commit", "memfs_after_delete"])
def test_interrupted_rebuild_resumes_to_full_convergence(
    tmp_path: Path,
    checkpoint: str,
) -> None:
    store = EventStore(tmp_path / f"resume-{checkpoint}.db")
    root = tmp_path / f"memfs-{checkpoint}"
    try:
        one = _write(store, "resume-one", "resume first token")
        two = _write(store, "resume-two", "resume second token")
        drain_memory_projections(store, memfs_root=root)
        authority_before = [tuple(row) for row in store._conn.execute(
            """
            SELECT mc.claim_id, mc.claim_version_id, mc.memory_governance_id,
                   mc.lifecycle_state, v.content_sha256
            FROM memory_current mc
            JOIN memory_claim_versions v ON v.claim_version_id = mc.claim_version_id
            ORDER BY mc.claim_id
            """
        )]

        def fail(name: str) -> None:
            if name == checkpoint:
                raise RuntimeError("forced rebuild interruption")

        with pytest.raises(RuntimeError, match="forced rebuild interruption"):
            rebuild_all_memory_projections(store, memfs_root=root, fault_hook=fail)

        result = rebuild_all_memory_projections(store, memfs_root=root)
        authority_after = [tuple(row) for row in store._conn.execute(
            """
            SELECT mc.claim_id, mc.claim_version_id, mc.memory_governance_id,
                   mc.lifecycle_state, v.content_sha256
            FROM memory_current mc
            JOIN memory_claim_versions v ON v.claim_version_id = mc.claim_version_id
            ORDER BY mc.claim_id
            """
        )]
        fts_ids = [int(row[0]) for row in store._conn.execute(
            "SELECT rowid FROM memory_fts ORDER BY rowid"
        )]
    finally:
        store.close()

    assert result == {"claims": 2, "memory_fts": 2, "memory_memfs": 2}
    assert authority_after == authority_before
    assert fts_ids == [one.claim_id, two.claim_id]
    assert sorted(path.name for path in (root / "claims").glob("*.md")) == [
        f"{one.claim_id:016d}.md", f"{two.claim_id:016d}.md",
    ]


def test_rebuild_restores_into_initially_empty_memfs_directory(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "empty-restore.db")
    empty_root = tmp_path / "empty-memfs"
    try:
        receipt = _write(store, "empty", "empty directory restore token")
        result = rebuild_all_memory_projections(store, memfs_root=empty_root)
        restored = empty_root / "claims" / f"{receipt.claim_id:016d}.md"
        integrity = str(store._conn.execute("PRAGMA quick_check").fetchone()[0])
    finally:
        store.close()

    assert result == {"claims": 1, "memory_fts": 1, "memory_memfs": 1}
    assert restored.is_file()
    assert integrity == "ok"


def test_rebuild_does_not_delete_projection_applied_after_authority_commit(tmp_path: Path) -> None:
    db = tmp_path / "authority.db"
    root = tmp_path / "memfs"
    store = EventStore(db)
    peer = EventStore(db)
    try:
        receipt = _write(store, "post-commit", "post commit projection token")
        drain_memory_projections(store, memfs_root=root)
        claim_file = root / "claims" / f"{receipt.claim_id:016d}.md"

        def drain_from_peer(checkpoint: str) -> None:
            if checkpoint == "authority_after_commit":
                applied = drain_memory_projections(peer, memfs_root=root)
                assert applied == {"memory_fts": 1, "memory_memfs": 1}

        result = rebuild_all_memory_projections(
            store,
            memfs_root=root,
            fault_hook=drain_from_peer,
        )
    finally:
        peer.close()
        store.close()

    assert result == {"claims": 1, "memory_fts": 0, "memory_memfs": 0}
    assert claim_file.is_file()
