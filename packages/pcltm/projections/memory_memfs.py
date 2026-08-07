"""Idempotent human-readable projection of governed memory claims."""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterator

import yaml

from ..evidence_chain import sha256_text
from ..memfs_store import atomic_write_text
from ..projection_outbox import require_event_projection_authority
from ..store import EventStore


class MemoryMemfsProjector:
    """Project current claims to ``claims/<claim_id>.md`` without file authority."""

    def __init__(
        self,
        store: EventStore,
        *,
        memfs_root: Path,
        worker_id: str,
        before_replace: Callable[[Path], None] | None = None,
        simulate_ack_loss: bool = False,
    ) -> None:
        if not worker_id.strip():
            raise ValueError("worker_id is required")
        self.store = store
        self.memfs_root = Path(memfs_root)
        self.worker_id = worker_id
        self._before_replace = before_replace
        self._simulate_ack_loss = simulate_ack_loss

    def run_once(self, *, now: str, lease_until: str) -> dict[str, int]:
        jobs = self.store.claim_projection_jobs(
            worker_id=self.worker_id,
            projection_kind="memory_memfs",
            limit=1,
            now=now,
            lease_until=lease_until,
        )
        result = {"claimed": len(jobs), "applied": 0, "failed": 0, "obsolete": 0}
        for job in jobs:
            try:
                outcome = self._apply_claim_file(job, ack_now=now)
                if outcome == "obsolete":
                    if self.store.obsolete_projection_job(
                        int(job["outbox_id"]), worker_id=self.worker_id,
                        expected_attempt_count=int(job["attempt_count"]),
                    ):
                        result["obsolete"] += 1
                    else:
                        result["failed"] += 1
                    continue
                result["applied"] += 1
            except Exception as exc:
                if str(exc) == "projection lease ownership lost":
                    result["failed"] += 1
                    continue
                if str(exc) == "stale_projection":
                    if self.store.obsolete_projection_job(
                        int(job["outbox_id"]), worker_id=self.worker_id,
                        expected_attempt_count=int(job["attempt_count"]),
                    ):
                        result["obsolete"] += 1
                    else:
                        result["failed"] += 1
                    continue
                retry_at = (
                    datetime.fromisoformat(now.replace("Z", "+00:00")) + timedelta(minutes=1)
                ).astimezone(UTC).isoformat().replace("+00:00", "Z")
                self.store.fail_projection_job(
                    int(job["outbox_id"]),
                    worker_id=self.worker_id,
                    expected_attempt_count=int(job["attempt_count"]),
                    error=str(exc),
                    now=now,
                    next_retry_at=retry_at,
                )
                result["failed"] += 1
        return result

    def _apply_claim_file(
        self,
        job: dict[str, Any],
        *,
        ack_now: str | None = None,
    ) -> str:
        """Serialize writers per claim and re-open authority after taking the lock."""
        claim_id = self._claim_id(job)
        with self._claim_lock(claim_id):
            with self._projection_guard(claim_id, job):
                outcome = self._apply_claim_file_guarded(job, claim_id)
                if outcome == "obsolete" or ack_now is None:
                    return outcome
                if self._simulate_ack_loss:
                    raise RuntimeError("projection lease ownership lost")
                acked = self.store.ack_memory_memfs_projection_job(
                    int(job["outbox_id"]),
                    claim_id=claim_id,
                    worker_id=self.worker_id,
                    expected_attempt_count=int(job["attempt_count"]),
                    now=ack_now,
                )
                if not acked:
                    raise RuntimeError("projection lease ownership lost")
                return "applied"

    def _apply_claim_file_guarded(
        self,
        job: dict[str, Any],
        claim_id: int,
    ) -> str:
            current_version = self.store._conn.execute(
                """
                SELECT v.version
                FROM memory_current mc
                JOIN memory_claim_versions v
                  ON v.claim_version_id = mc.claim_version_id
                 AND v.claim_id = mc.claim_id
                WHERE mc.claim_id = ?
                """,
                (claim_id,),
            ).fetchone()
            if (
                current_version is None
                or int(current_version["version"]) != int(job["aggregate_version"])
            ):
                return "obsolete"
            try:
                snapshot = self._load_snapshot(job)
            except ValueError as exc:
                if str(exc) == "stale_projection":
                    return "obsolete"
                raise
            path = self._claim_path(claim_id)
            if str(snapshot["governance_state"]) != "active":
                if path.exists() or path.is_symlink():
                    self._reject_reparse_path(path)
                    path.unlink()
                return "applied"
            path.parent.mkdir(parents=True, exist_ok=True)
            self._validate_claim_destination(path)
            if self._before_replace is not None:
                self._before_replace(path)
            self._validate_claim_destination(path)
            if not self._job_is_current(job, claim_id):
                return "obsolete"
            atomic_write_text(path, self._render(snapshot))
            return "applied"

    @contextmanager
    def _projection_guard(
        self,
        claim_id: int,
        job: dict[str, Any],
    ) -> Iterator[None]:
        """Block authority switches from final reopen through file ACK."""
        conn = self.store._conn
        if conn.in_transaction:
            raise RuntimeError("projection guard requires transaction ownership")
        try:
            conn.execute("BEGIN IMMEDIATE")
            root_id = os.path.normcase(str(self.memfs_root.resolve()))
            existing = conn.execute(
                """
                SELECT memfs_root_id FROM memory_projection_guards
                WHERE claim_id = ?
                """,
                (claim_id,),
            ).fetchone()
            if existing is not None and str(existing["memfs_root_id"]) != root_id:
                raise RuntimeError("memory projection root conflict")
            # The claim OS lock at this exact root proves no live projector
            # owns an older same-root guard.
            conn.execute(
                "DELETE FROM memory_projection_guards WHERE claim_id = ?",
                (claim_id,),
            )
            conn.execute(
                """
                INSERT INTO memory_projection_guards(
                    claim_id, outbox_id, attempt_count, worker_id, memfs_root_id
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    claim_id, int(job["outbox_id"]),
                    int(job["attempt_count"]), self.worker_id, root_id,
                ),
            )
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        try:
            yield
        finally:
            try:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute(
                    """
                    DELETE FROM memory_projection_guards
                    WHERE claim_id = ? AND outbox_id = ?
                      AND attempt_count = ? AND worker_id = ?
                    """,
                    (
                        claim_id, int(job["outbox_id"]),
                        int(job["attempt_count"]), self.worker_id,
                    ),
                )
                conn.commit()
            except BaseException:
                conn.rollback()
                raise

    def _job_is_current(self, job: dict[str, Any], claim_id: int) -> bool:
        row = self.store._conn.execute(
            """
            SELECT v.version
            FROM memory_current mc
            JOIN memory_claim_versions v
              ON v.claim_version_id = mc.claim_version_id
             AND v.claim_id = mc.claim_id
            WHERE mc.claim_id = ?
            """,
            (claim_id,),
        ).fetchone()
        if row is None or int(row["version"]) != int(job["aggregate_version"]):
            return False
        if str(job["authority_kind"]) == "event":
            try:
                require_event_projection_authority(job)
            except ValueError:
                return False
            return True
        if str(job["authority_kind"]) != "legacy_record" or job["event_seq"] is not None:
            return False
        source = self.store._conn.execute(
            """
            SELECT s.legacy_record_id, s.legacy_content_sha256,
                   r.content, r.status
            FROM memory_current mc
            JOIN memory_claim_sources s
              ON s.claim_version_id = mc.claim_version_id
             AND s.source_kind = 'legacy_record'
            JOIN memory_records r ON r.record_id = s.legacy_record_id
            WHERE mc.claim_id = ?
            """,
            (claim_id,),
        ).fetchall()
        if len(source) != 1:
            return False
        legacy = source[0]
        return (
            int(legacy["legacy_record_id"]) == int(job["authority_id"])
            and str(legacy["status"]) == "approved"
            and sha256_text(str(legacy["content"]))
            == str(legacy["legacy_content_sha256"])
        )

    @contextmanager
    def _claim_lock(self, claim_id: int) -> Iterator[None]:
        self._reject_reparse_path(self.memfs_root)
        root = self.memfs_root.resolve()
        locks = root / ".claim-locks"
        self._reject_reparse_path(root)
        root.mkdir(parents=True, exist_ok=True)
        self._reject_reparse_path(root)
        if locks.exists():
            self._reject_reparse_path(locks)
        locks.mkdir(exist_ok=True)
        self._reject_reparse_path(locks)
        lock_path = locks / f"{claim_id:016d}.lock"
        if lock_path.exists() or lock_path.is_symlink():
            self._reject_reparse_path(lock_path)
        handle = lock_path.open("a+b")
        try:
            self._reject_reparse_path(lock_path)
            handle.seek(0)
            if os.fstat(handle.fileno()).st_size == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()

    def _load_snapshot(self, job: dict[str, Any]) -> dict[str, Any]:
        claim_id = self._claim_id(job)
        row = self.store._conn.execute(
            """
            SELECT c.claim_id, c.canonical_key, c.target, c.memory_type,
                   v.claim_version_id, v.version AS claim_version, v.content,
                   v.content_sha256, v.sensitivity, v.injection_policy,
                   v.mode_scope, g.memory_governance_id AS governance_id,
                   g.policy_version, mc.lifecycle_state,
                   s.source_kind, s.event_id, s.event_revision,
                   s.event_payload_sha256, s.legacy_record_id,
                   s.legacy_content_sha256,
                   e.source_revision AS persisted_revision,
                   e.payload_sha256 AS persisted_payload_sha256,
                   r.content AS persisted_legacy_content,
                   r.status AS persisted_legacy_status
            FROM memory_current mc
            JOIN memory_claims c ON c.claim_id = mc.claim_id
            JOIN memory_claim_versions v
              ON v.claim_version_id = mc.claim_version_id AND v.claim_id = mc.claim_id
            JOIN memory_governance_events g
              ON g.memory_governance_id = mc.memory_governance_id
             AND g.claim_id = mc.claim_id
             AND g.claim_version_id = mc.claim_version_id
             AND g.new_state = mc.lifecycle_state
            JOIN memory_claim_sources s ON s.claim_version_id = v.claim_version_id
            LEFT JOIN events e ON e.event_id = s.event_id
            LEFT JOIN memory_records r ON r.record_id = s.legacy_record_id
            WHERE mc.claim_id = ?
            ORDER BY s.claim_source_id
            """,
            (claim_id,),
        ).fetchall()
        if not row:
            raise ValueError("memory projection authority missing")
        authority = row[0]
        computed_hash = sha256_text(str(authority["content"]))
        if int(authority["claim_version"]) != int(job["aggregate_version"]):
            raise ValueError("stale_projection")
        if computed_hash != str(authority["content_sha256"]) or computed_hash != str(job["payload_sha256"]):
            raise ValueError("memory projection payload hash mismatch")
        event_sources = [source for source in row if source["source_kind"] == "event"]
        legacy_sources = [
            source for source in row if source["source_kind"] == "legacy_record"
        ]
        if str(job["authority_kind"]) == "event":
            if len(event_sources) != 1 or legacy_sources:
                raise ValueError("memory projection source commitment mismatch")
            try:
                event_seq = require_event_projection_authority(job)
            except ValueError as exc:
                raise ValueError("memory projection source commitment mismatch") from exc
            source = event_sources[0]
            source_ref = {
                "authority_kind": "event",
                "object_id": str(source["event_id"]),
                "object_version": int(source["event_revision"]),
                "payload_sha256": str(source["event_payload_sha256"]),
            }
            if (
                int(source["event_id"]) != event_seq
                or source["persisted_revision"] is None
                or int(source["event_revision"]) != int(source["persisted_revision"])
                or str(source["event_payload_sha256"]) != str(source["persisted_payload_sha256"])
            ):
                raise ValueError("memory projection source commitment mismatch")
        elif str(job["authority_kind"]) == "legacy_record":
            if len(legacy_sources) != 1 or event_sources:
                raise ValueError("memory projection source commitment mismatch")
            if job["event_seq"] is not None or not str(job["authority_id"]):
                raise ValueError("memory projection source commitment mismatch")
            source = legacy_sources[0]
            source_ref = {
                "authority_kind": "legacy_record",
                "object_id": str(source["legacy_record_id"]),
                "object_version": 1,
                "payload_sha256": str(source["legacy_content_sha256"]),
            }
            if (
                int(source["legacy_record_id"]) != int(job["authority_id"])
                or source["persisted_legacy_content"] is None
                or str(source["persisted_legacy_status"]) != "approved"
                or sha256_text(str(source["persisted_legacy_content"]))
                != str(source["legacy_content_sha256"])
            ):
                raise ValueError("memory projection source commitment mismatch")
        else:
            raise ValueError("memory projection source commitment mismatch")
        return {
            # Claim projections share a filesystem with MemFS readers.  Keep
            # the governed claim schema intact while also satisfying the
            # reader's required human-facing locator contract.
            "description": f"Governed memory claim: {authority['canonical_key']}",
            "schema_version": 1,
            "projection_kind": "memory_memfs",
            "projection_generation": 1,
            "claim_id": claim_id,
            "claim_version": int(authority["claim_version"]),
            "canonical_key": str(authority["canonical_key"]),
            "target": str(authority["target"]),
            "memory_type": str(authority["memory_type"]),
            "lifecycle_state": str(authority["lifecycle_state"]),
            "content_sha256": computed_hash,
            "governance_id": int(authority["governance_id"]),
            "governance_state": str(authority["lifecycle_state"]),
            "policy_version": str(authority["policy_version"]),
            "sensitivity": str(authority["sensitivity"]),
            "mode_scope": json.loads(str(authority["mode_scope"])),
            "injection_policy": str(authority["injection_policy"]),
            "read_only_projection": True,
            "read_only": True,
            "authority_refs": [source_ref],
            "evidence_refs": [source_ref],
            "content": str(authority["content"]),
        }

    def _claim_path(self, claim_id: int) -> Path:
        self._reject_reparse_path(self.memfs_root)
        root = self.memfs_root.resolve()
        claims = root / "claims"
        self._reject_reparse_path(root)
        if claims.exists():
            self._reject_reparse_path(claims)
        path = claims / f"{claim_id:016d}.md"
        if path.resolve().parent != claims.resolve():
            raise ValueError("memfs path rejected")
        return path

    def _validate_claim_destination(self, path: Path) -> None:
        """Revalidate every existing path component at the final write seam."""
        self._reject_reparse_path(self.memfs_root)
        root = self.memfs_root.resolve()
        claims = root / "claims"
        if not claims.exists() or not claims.is_dir():
            raise ValueError("memfs reparse path rejected")
        self._reject_reparse_path(root)
        self._reject_reparse_path(claims)
        if path.exists() or path.is_symlink():
            self._reject_reparse_path(path)
        if path.parent.resolve() != claims.resolve():
            raise ValueError("memfs reparse path rejected")

    @staticmethod
    def _reject_reparse_path(path: Path) -> None:
        if path.is_symlink():
            raise ValueError("memfs reparse path rejected")
        try:
            attributes = int(getattr(path.stat(), "st_file_attributes", 0))
        except FileNotFoundError:
            return
        if attributes & 0x400:
            raise ValueError("memfs reparse path rejected")

    @staticmethod
    def _claim_id(job: dict[str, Any]) -> int:
        aggregate_id = str(job["aggregate_id"])
        if not aggregate_id.startswith("memory:"):
            raise ValueError("memory projection aggregate id mismatch")
        raw = aggregate_id[7:]
        if not raw.isascii() or not raw.isdecimal() or int(raw) <= 0:
            raise ValueError("memory projection aggregate id mismatch")
        return int(raw)

    @staticmethod
    def _render(snapshot: dict[str, Any]) -> str:
        metadata = {key: value for key, value in snapshot.items() if key != "content"}
        frontmatter = yaml.safe_dump(metadata, allow_unicode=True, sort_keys=False)
        return f"---\n{frontmatter}---\n{snapshot['content'].rstrip()}\n"
