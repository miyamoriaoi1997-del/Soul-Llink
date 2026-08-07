"""Explicit one-time promotion of reviewed legacy memory records.

This is intentionally separate from MemoryWriteService: legacy rows are reopened
as evidence and never rewritten or disguised as new user assertion events.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from .evidence_chain import sha256_text
from .memory_contracts import PersonaMode, Sensitivity
from .memory_policy import POLICY_VERSION
from .projection_outbox import enqueue_memory_projections
from .store import EventStore


@dataclass(frozen=True, slots=True)
class LegacyMemoryPromotionSpec:
    record_id: int
    canonical_key: str
    target: str
    memory_type: str
    mode_scope: tuple[PersonaMode, ...]
    injection_policy: str

    def __post_init__(self) -> None:
        if type(self.record_id) is not int or isinstance(self.record_id, bool) or self.record_id <= 0:
            raise TypeError("record_id must be a positive int")
        for name in ("canonical_key", "target", "memory_type", "injection_policy"):
            value = getattr(self, name)
            if type(value) is not str or not value.strip():
                raise ValueError(f"{name} is required")
        if type(self.mode_scope) is not tuple or not self.mode_scope or not all(
            type(mode) is PersonaMode for mode in self.mode_scope
        ):
            raise TypeError("mode_scope must be a non-empty tuple of PersonaMode")


@dataclass(frozen=True, slots=True)
class LegacyMemoryPromotionRequest:
    items: tuple[LegacyMemoryPromotionSpec, ...]

    def __post_init__(self) -> None:
        if type(self.items) is not tuple or not self.items or not all(
            type(item) is LegacyMemoryPromotionSpec for item in self.items
        ):
            raise TypeError("items must be a non-empty tuple of LegacyMemoryPromotionSpec")
        record_ids = [item.record_id for item in self.items]
        canonical_keys = [item.canonical_key for item in self.items]
        if len(set(record_ids)) != len(record_ids) or len(set(canonical_keys)) != len(canonical_keys):
            raise ValueError("promotion request contains duplicate identity")


@dataclass(frozen=True, slots=True)
class LegacyMemoryPromotionItem:
    record_id: int
    claim_id: int
    claim_version: int
    governance_id: int
    content_sha256: str


@dataclass(frozen=True, slots=True)
class LegacyMemoryPromotionResult:
    status: str
    persisted: bool
    items: tuple[LegacyMemoryPromotionItem, ...]


class LegacyMemoryPromotionService:
    """Atomically promote an explicit batch of approved legacy records."""

    def __init__(self, store: EventStore) -> None:
        self._store = store

    def promote(self, request: LegacyMemoryPromotionRequest) -> LegacyMemoryPromotionResult:
        if type(request) is not LegacyMemoryPromotionRequest:
            raise TypeError("request must be LegacyMemoryPromotionRequest")
        conn = self._store._conn
        if conn.in_transaction:
            raise RuntimeError("promotion_requires_transaction_ownership")
        try:
            conn.execute("BEGIN IMMEDIATE")
            promoted: list[LegacyMemoryPromotionItem] = []
            for spec in request.items:
                row = conn.execute(
                    """
                    SELECT record_id, content, sensitivity, status
                    FROM memory_records WHERE record_id = ?
                    """,
                    (spec.record_id,),
                ).fetchone()
                if row is None:
                    raise ValueError("legacy_record_missing")
                if str(row["status"]) != "approved":
                    raise ValueError("legacy_record_not_approved")
                try:
                    sensitivity = Sensitivity(str(row["sensitivity"]))
                except ValueError as exc:
                    raise ValueError("legacy_record_malformed") from exc
                if sensitivity is Sensitivity.SECRET:
                    raise ValueError("legacy_record_secret_forbidden")
                content = str(row["content"])
                if not content:
                    raise ValueError("legacy_record_malformed")
                content_sha256 = sha256_text(content)

                existing_rows = conn.execute(
                    """
                    SELECT c.claim_id, c.canonical_key, c.target, c.memory_type,
                           v.version, v.content_sha256, v.sensitivity,
                           v.injection_policy, v.mode_scope, v.lineage_kind,
                           g.memory_governance_id, g.new_state,
                           g.actor, g.reason_code, mc.lifecycle_state
                    FROM memory_claim_sources s
                    JOIN memory_claim_versions v
                      ON v.claim_version_id = s.claim_version_id
                    JOIN memory_claims c ON c.claim_id = v.claim_id
                    LEFT JOIN memory_current mc ON mc.claim_id = c.claim_id
                      AND mc.claim_version_id = v.claim_version_id
                    LEFT JOIN memory_governance_events g
                      ON g.memory_governance_id = mc.memory_governance_id
                     AND g.claim_id = c.claim_id
                     AND g.claim_version_id = v.claim_version_id
                    WHERE s.source_kind = 'legacy_record'
                      AND s.legacy_record_id = ?
                    ORDER BY c.claim_id
                    """,
                    (spec.record_id,),
                ).fetchall()
                if len(existing_rows) > 1:
                    raise ValueError("legacy_promotion_ambiguous")
                existing = existing_rows[0] if existing_rows else None
                if existing is not None:
                    if (
                        str(existing["lifecycle_state"]) != "active"
                        or str(existing["new_state"]) != "active"
                    ):
                        raise ValueError("legacy_promotion_not_active")
                    if (
                        str(existing["actor"]) != "legacy_memory_promotion"
                        or str(existing["reason_code"]) != "legacy_promotion_allowed"
                    ):
                        raise ValueError("legacy_promotion_receipt_conflict")
                    if str(existing["content_sha256"]) != content_sha256:
                        raise ValueError("legacy_record_drifted")
                    expected_scope = json.dumps(
                        [mode.value for mode in spec.mode_scope],
                        separators=(",", ":"),
                    )
                    if (
                        str(existing["canonical_key"]) != spec.canonical_key
                        or str(existing["target"]) != spec.target
                        or str(existing["memory_type"]) != spec.memory_type
                        or str(existing["sensitivity"]) != sensitivity.value
                        or str(existing["injection_policy"]) != spec.injection_policy
                        or str(existing["mode_scope"]) != expected_scope
                        or str(existing["lineage_kind"]) != "legacy_governed"
                    ):
                        raise ValueError("legacy_promotion_spec_conflict")
                    promoted.append(LegacyMemoryPromotionItem(
                        spec.record_id, int(existing["claim_id"]),
                        int(existing["version"]),
                        int(existing["memory_governance_id"]), content_sha256,
                    ))
                    continue

                conflict = conn.execute(
                    "SELECT claim_id FROM memory_claims WHERE canonical_key = ?",
                    (spec.canonical_key,),
                ).fetchone()
                if conflict is not None:
                    raise ValueError("canonical_key_conflict")
                claim = conn.execute(
                    """
                    INSERT INTO memory_claims(canonical_key, target, memory_type)
                    VALUES (?, ?, ?)
                    """,
                    (spec.canonical_key, spec.target, spec.memory_type),
                )
                claim_id = int(claim.lastrowid)
                version = conn.execute(
                    """
                    INSERT INTO memory_claim_versions(
                        claim_id, version, content, content_sha256, confidence,
                        sensitivity, injection_policy, mode_scope, lineage_kind,
                        schema_version
                    ) VALUES (?, 1, ?, ?, 1.0, ?, ?, ?, 'legacy_governed', 1)
                    """,
                    (
                        claim_id, content, content_sha256, sensitivity.value,
                        spec.injection_policy,
                        json.dumps([mode.value for mode in spec.mode_scope], separators=(",", ":")),
                    ),
                )
                version_id = int(version.lastrowid)
                conn.execute(
                    """
                    INSERT INTO memory_claim_sources(
                        claim_version_id, source_kind, legacy_record_id,
                        legacy_content_sha256
                    ) VALUES (?, 'legacy_record', ?, ?)
                    """,
                    (version_id, spec.record_id, content_sha256),
                )
                governance = conn.execute(
                    """
                    INSERT INTO memory_governance_events(
                        claim_id, claim_version_id, action, previous_state,
                        new_state, actor, reason_code, policy_version
                    ) VALUES (?, ?, 'activate', 'pending_review', 'active',
                              'legacy_memory_promotion',
                              'legacy_promotion_allowed', ?)
                    """,
                    (claim_id, version_id, POLICY_VERSION),
                )
                governance_id = int(governance.lastrowid)
                conn.execute(
                    """
                    INSERT INTO memory_current(
                        claim_id, claim_version_id, memory_governance_id,
                        lifecycle_state
                    ) VALUES (?, ?, ?, 'active')
                    """,
                    (claim_id, version_id, governance_id),
                )
                enqueue_memory_projections(
                    conn,
                    event_id=None,
                    authority_kind="legacy_record",
                    authority_id=str(spec.record_id),
                    aggregate_id=f"memory:{claim_id}",
                    aggregate_version=1,
                    payload_sha256=content_sha256,
                )
                promoted.append(LegacyMemoryPromotionItem(
                    spec.record_id, claim_id, 1, governance_id, content_sha256,
                ))
            conn.commit()
            return LegacyMemoryPromotionResult("promoted", True, tuple(promoted))
        except BaseException:
            conn.rollback()
            raise
