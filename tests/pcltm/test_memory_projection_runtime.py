from __future__ import annotations

from pathlib import Path

import pytest

from pcltm.memory_contracts import PersonaMode, Sensitivity
from pcltm.memory_write_service import MemoryWriteRequest, MemoryWriteService
from pcltm.projections.memory_runtime import (
    drain_memory_projections,
    require_memory_projections_applied,
)
from pcltm.store import EventStore


def _write(store: EventStore, key: str = "runtime"):
    return MemoryWriteService(store).write(
        MemoryWriteRequest(
            idempotency_key=key,
            content="用户偏好 UTC+8",
            canonical_key=f"profile:timezone:{key}",
            target="profile",
            memory_type="preference",
            sensitivity=Sensitivity.NORMAL,
            mode_scope=(PersonaMode.DAILY,),
            injection_policy="allow",
        )
    )


def test_bounded_drain_converges_both_memory_projections(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "authority.db")
    try:
        receipt = _write(store)
        result = drain_memory_projections(
            store, memfs_root=tmp_path / "memfs", worker_id="memory-drain", max_jobs=4,
        )
        snapshot = require_memory_projections_applied(
            store, memfs_root=tmp_path / "memfs", claim_id=receipt.claim_id,
        )
    finally:
        store.close()

    assert result == {"memory_fts": 1, "memory_memfs": 1}
    assert snapshot == {
        "claim_id": receipt.claim_id,
        "claim_version": 1,
        "governance_id": receipt.governance_id,
        "payload_sha256": "ba739a9c09553a438a87a9d9b2c5706401395bdeb32f48d9ae46feeff8593ed5",
        "projection_generation": 1,
        "projection_status": "applied",
    }


def test_convergence_assertion_fails_closed_on_tampered_claim_file(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "authority.db")
    try:
        receipt = _write(store, "tamper")
        drain_memory_projections(store, memfs_root=tmp_path / "memfs", worker_id="drain")
        path = tmp_path / "memfs" / "claims" / f"{receipt.claim_id:016d}.md"
        path.write_text(path.read_text(encoding="utf-8") + "tampered", encoding="utf-8")
        with pytest.raises(RuntimeError, match="memory_memfs commitment mismatch"):
            require_memory_projections_applied(
                store, memfs_root=tmp_path / "memfs", claim_id=receipt.claim_id,
            )
    finally:
        store.close()


def test_bounded_drain_reports_limit_without_claiming_extra_jobs(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "authority.db")
    try:
        _write(store, "one")
        _write(store, "two")
        with pytest.raises(RuntimeError, match="memory projection drain limit reached"):
            drain_memory_projections(
                store, memfs_root=tmp_path / "memfs", worker_id="drain", max_jobs=1,
            )
        processing = store._conn.execute(
            """
            SELECT count(*) FROM projection_outbox
            WHERE projection_kind IN ('memory_fts', 'memory_memfs') AND status = 'processing'
            """
        ).fetchone()[0]
    finally:
        store.close()

    assert processing == 0


def test_convergence_rejects_symlinked_claim_file(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "authority.db")
    try:
        receipt = _write(store, "claim-link")
        drain_memory_projections(store, memfs_root=tmp_path / "memfs", worker_id="drain")
        path = tmp_path / "memfs" / "claims" / f"{receipt.claim_id:016d}.md"
        copy = tmp_path / "valid-copy.md"
        copy.write_bytes(path.read_bytes())
        path.unlink()
        try:
            path.symlink_to(copy)
        except OSError:
            pytest.skip("file symlink creation is unavailable")
        with pytest.raises(RuntimeError, match="memory_memfs commitment mismatch"):
            require_memory_projections_applied(
                store, memfs_root=tmp_path / "memfs", claim_id=receipt.claim_id,
            )
    finally:
        store.close()


def test_convergence_rejects_tampered_fts_content_with_matching_metadata(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "authority.db")
    try:
        receipt = _write(store, "fts-content-tamper")
        drain_memory_projections(store, memfs_root=tmp_path / "memfs", worker_id="drain")
        store._conn.execute(
            "UPDATE memory_fts SET content='tampered' WHERE rowid=?", (receipt.claim_id,),
        )
        store._conn.commit()
        with pytest.raises(RuntimeError, match="memory_fts commitment mismatch"):
            require_memory_projections_applied(
                store, memfs_root=tmp_path / "memfs", claim_id=receipt.claim_id,
            )
    finally:
        store.close()
