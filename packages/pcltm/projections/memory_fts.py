"""Idempotent SQLite FTS projection for governed current memory claims."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any, Callable

from ..store import EventStore
from ..evidence_chain import sha256_text
from ..projection_outbox import require_event_projection_authority


class MemoryFtsProjector:
    def __init__(
        self,
        store: EventStore,
        *,
        worker_id: str,
        fault_hook: Callable[[str], None] | None = None,
    ) -> None:
        if not worker_id.strip():
            raise ValueError("worker_id is required")
        self.store = store
        self.worker_id = worker_id
        self._fault_hook = fault_hook

    def run_once(self, *, now: str, lease_until: str) -> dict[str, int]:
        jobs = self.store.claim_projection_jobs(
            worker_id=self.worker_id,
            projection_kind="memory_fts",
            limit=1,
            now=now,
            lease_until=lease_until,
        )
        result = {"claimed": len(jobs), "applied": 0, "failed": 0, "obsolete": 0}
        for job in jobs:
            try:
                self._apply_and_ack(job, now=now)
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

    def _apply_and_ack(self, job: dict[str, Any], *, now: str) -> None:
        conn = self.store._conn
        try:
            conn.execute("BEGIN IMMEDIATE")
            owned = conn.execute(
                """
                SELECT status, lease_owner, attempt_count
                FROM projection_outbox
                WHERE outbox_id = ? AND projection_kind = 'memory_fts'
                """,
                (int(job["outbox_id"]),),
            ).fetchone()
            if (
                owned is None
                or owned["status"] != "processing"
                or owned["lease_owner"] != self.worker_id
                or int(owned["attempt_count"]) != int(job["attempt_count"])
            ):
                raise RuntimeError("projection lease ownership lost")

            claim_id = self._claim_id(job)
            authority = conn.execute(
                """
                SELECT c.claim_id, c.canonical_key, c.target, c.memory_type,
                       v.version AS claim_version, v.content, v.content_sha256,
                       v.sensitivity, v.injection_policy, v.mode_scope,
                       g.memory_governance_id AS governance_id,
                       g.policy_version, mc.lifecycle_state
                FROM memory_current mc
                JOIN memory_claims c ON c.claim_id = mc.claim_id
                JOIN memory_claim_versions v
                  ON v.claim_version_id = mc.claim_version_id
                 AND v.claim_id = mc.claim_id
                JOIN memory_governance_events g
                  ON g.memory_governance_id = mc.memory_governance_id
                 AND g.claim_id = mc.claim_id
                 AND g.claim_version_id = mc.claim_version_id
                 AND g.new_state = mc.lifecycle_state
                WHERE mc.claim_id = ?
                """,
                (claim_id,),
            ).fetchone()
            if authority is None:
                raise ValueError("memory projection authority missing")
            if int(authority["claim_version"]) != int(job["aggregate_version"]):
                raise ValueError("stale_projection")
            computed_hash = sha256_text(str(authority["content"]))
            if (
                computed_hash != str(authority["content_sha256"])
                or computed_hash != str(job["payload_sha256"])
            ):
                raise ValueError("memory projection payload hash mismatch")
            source_rows = conn.execute(
                """
                SELECT source_kind, event_id, event_revision, event_payload_sha256,
                       legacy_record_id, legacy_content_sha256
                FROM memory_claim_sources s
                JOIN memory_claim_versions v
                  ON v.claim_version_id = s.claim_version_id
                WHERE v.claim_id = ? AND v.version = ?
                ORDER BY s.claim_source_id
                """,
                (claim_id, int(authority["claim_version"])),
            ).fetchall()
            source_refs = []
            for source in source_rows:
                if source["source_kind"] == "event":
                    source_refs.append({
                        "authority_kind": "event",
                        "object_id": str(source["event_id"]),
                        "object_version": int(source["event_revision"]),
                        "payload_sha256": str(source["event_payload_sha256"]),
                    })
                elif source["source_kind"] == "legacy_record":
                    source_refs.append({
                        "authority_kind": "legacy_record",
                        "object_id": str(source["legacy_record_id"]),
                        "object_version": 1,
                        "payload_sha256": str(source["legacy_content_sha256"]),
                    })
                else:
                    source_refs.append({
                        "authority_kind": "system",
                        "object_id": str(claim_id),
                        "object_version": int(authority["claim_version"]),
                        "payload_sha256": str(authority["content_sha256"]),
                    })
            serialized_sources = json.dumps(
                source_refs, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            )
            event_sources = [source for source in source_rows if source["source_kind"] == "event"]
            legacy_sources = [
                source for source in source_rows
                if source["source_kind"] == "legacy_record"
            ]
            if str(job["authority_kind"]) == "event":
                if not event_sources or legacy_sources:
                    raise ValueError("memory projection source commitment mismatch")
                try:
                    event_seq = require_event_projection_authority(job)
                except ValueError as exc:
                    raise ValueError("memory projection source commitment mismatch") from exc
                committed_event_ids = {int(source["event_id"]) for source in event_sources}
                if event_seq not in committed_event_ids:
                    raise ValueError("memory projection source commitment mismatch")
                for event_source in event_sources:
                    persisted_event = conn.execute(
                        """
                        SELECT source_revision, payload_sha256
                        FROM events WHERE event_id = ?
                        """,
                        (int(event_source["event_id"]),),
                    ).fetchone()
                    if (
                        persisted_event is None
                        or int(event_source["event_revision"])
                        != int(persisted_event["source_revision"])
                        or str(event_source["event_payload_sha256"])
                        != str(persisted_event["payload_sha256"])
                    ):
                        raise ValueError("memory projection source commitment mismatch")
            elif str(job["authority_kind"]) == "legacy_record":
                if len(legacy_sources) != 1 or event_sources:
                    raise ValueError("memory projection source commitment mismatch")
                if job["event_seq"] is not None or not str(job["authority_id"]):
                    raise ValueError("memory projection source commitment mismatch")
                legacy_source = legacy_sources[0]
                persisted_legacy = conn.execute(
                    "SELECT content, status FROM memory_records WHERE record_id = ?",
                    (int(job["authority_id"]),),
                ).fetchone()
                if (
                    int(legacy_source["legacy_record_id"]) != int(job["authority_id"])
                    or persisted_legacy is None
                    or str(persisted_legacy["status"]) != "approved"
                    or sha256_text(str(persisted_legacy["content"]))
                    != str(legacy_source["legacy_content_sha256"])
                ):
                    raise ValueError("memory projection source commitment mismatch")
            else:
                raise ValueError("memory projection source commitment mismatch")

            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                    tokenize='trigram',
                    content,
                    canonical_key UNINDEXED,
                    target UNINDEXED,
                    memory_type UNINDEXED,
                    sensitivity UNINDEXED,
                    injection_policy UNINDEXED,
                    mode_scope UNINDEXED,
                    lifecycle_state UNINDEXED,
                    claim_version UNINDEXED,
                    governance_id UNINDEXED,
                    payload_sha256 UNINDEXED,
                    projection_generation UNINDEXED,
                    policy_version UNINDEXED,
                    source_refs UNINDEXED
                )
                """
            )
            conn.execute("DELETE FROM memory_fts WHERE rowid = ?", (claim_id,))
            if str(authority["lifecycle_state"]) == "active":
                conn.execute(
                    """
                    INSERT INTO memory_fts(
                        rowid, content, canonical_key, target, memory_type,
                        sensitivity, injection_policy, mode_scope, lifecycle_state,
                        claim_version, governance_id, payload_sha256,
                        projection_generation, policy_version, source_refs
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        claim_id, authority["content"], authority["canonical_key"],
                        authority["target"], authority["memory_type"], authority["sensitivity"],
                        authority["injection_policy"], authority["mode_scope"],
                        authority["lifecycle_state"], int(authority["claim_version"]),
                        int(authority["governance_id"]), authority["content_sha256"],
                        1, authority["policy_version"], serialized_sources,
                    ),
                )
            if self._fault_hook is not None:
                self._fault_hook("fts_before_ack")
            ack = conn.execute(
                """
                UPDATE projection_outbox
                SET status = 'applied', applied_at = COALESCE(applied_at, ?),
                    lease_owner = NULL, lease_until = NULL, last_error = NULL
                WHERE outbox_id = ? AND status = 'processing'
                  AND lease_owner = ? AND attempt_count = ?
                """,
                (
                    now, int(job["outbox_id"]), self.worker_id,
                    int(job["attempt_count"]),
                ),
            )
            if ack.rowcount != 1:
                raise RuntimeError("projection lease ownership lost")
            conn.commit()
        except BaseException:
            conn.rollback()
            raise

    @staticmethod
    def _claim_id(job: dict[str, Any]) -> int:
        aggregate_id = str(job["aggregate_id"])
        prefix = "memory:"
        if not aggregate_id.startswith(prefix):
            raise ValueError("memory projection aggregate id mismatch")
        raw = aggregate_id[len(prefix):]
        if not raw.isascii() or not raw.isdecimal() or int(raw) <= 0:
            raise ValueError("memory projection aggregate id mismatch")
        return int(raw)
