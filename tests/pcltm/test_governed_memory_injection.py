from __future__ import annotations

from pathlib import Path
from dataclasses import replace

import pytest

from pcltm.injection import CandidateType
from pcltm.injection.governed_memory import (
    GovernedInjectionStatus,
    build_governed_memory_context,
)
from pcltm.memory_contracts import PersonaMode, Sensitivity
from pcltm.memory_retrieval import (
    GovernedMemorySearchRequest,
    MemoryRetrievalStatus,
    GovernedMemoryRetrievalResult,
    search_governed_memories,
)
from pcltm.memory_write_service import MemoryWriteRequest, MemoryWriteService
from pcltm.projections.memory_fts import MemoryFtsProjector
from pcltm.store import EventStore


def _retrieval(
    store: EventStore,
    *,
    key: str = "inject",
    content: str = "老师偏好 authority-bound injection",
    injection_policy: str = "allow",
):
    receipt = MemoryWriteService(store).write(
        MemoryWriteRequest(
            idempotency_key=key,
            content=content,
            canonical_key=f"profile:{key}",
            target="profile",
            memory_type="preference",
            sensitivity=Sensitivity.NORMAL,
            mode_scope=(PersonaMode.WORK,),
            injection_policy=injection_policy,
        )
    )
    outcome = MemoryFtsProjector(store, worker_id=f"fts-{key}").run_once(
        now="2026-07-29T02:00:00Z",
        lease_until="2026-07-29T02:01:00Z",
    )
    assert outcome["applied"] == 1
    result = search_governed_memories(
        store,
        GovernedMemorySearchRequest(
            query="authority-bound", persona_mode=PersonaMode.WORK,
        ),
    )
    assert result.status is MemoryRetrievalStatus.OK
    return receipt, result


def test_governed_result_becomes_auditable_existing_context_packet(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "authority.db")
    try:
        receipt, retrieval = _retrieval(store)
        result = build_governed_memory_context(
            store,
            retrieval,
            persona_mode=PersonaMode.WORK,
            total_budget=100,
        )
    finally:
        store.close()

    assert result.status is GovernedInjectionStatus.OK
    assert result.reason is None
    assert result.packet is not None
    assert CandidateType.SEMANTIC_MEMORY in result.packet.sections
    candidate = result.packet.sections[CandidateType.SEMANTIC_MEMORY][0]
    assert candidate.content == "老师偏好 authority-bound injection"
    assert candidate.metadata["claim_id"] == receipt.claim_id
    assert candidate.metadata["claim_version"] == receipt.claim_version
    assert candidate.metadata["governance_id"] == receipt.governance_id
    assert candidate.metadata["authority_verified"] is True
    assert candidate.metadata["rank_score_is_authority"] is False
    assert result.packet.metadata["authority"] == "pcltm.memory_current"
    assert result.packet.metadata["retrieval_status"] == "ok"


def test_injection_rechecks_injection_policy_instead_of_reusing_search_admission(
    tmp_path: Path,
) -> None:
    store = EventStore(tmp_path / "authority.db")
    try:
        _, retrieval = _retrieval(store, key="retrieve-only", injection_policy="retrieve_only")
        result = build_governed_memory_context(
            store, retrieval, persona_mode=PersonaMode.WORK, total_budget=100,
        )
    finally:
        store.close()

    assert result.status is GovernedInjectionStatus.ABSTAINED
    assert result.packet is None
    assert result.reason == "injection_policy_filtered"


def test_retrieval_abstention_never_reaches_arbitrator(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "authority.db")
    try:
        result = build_governed_memory_context(
            store,
            GovernedMemoryRetrievalResult.abstained("no_answer"),
            persona_mode=PersonaMode.WORK,
            total_budget=100,
        )
    finally:
        store.close()

    assert result.status is GovernedInjectionStatus.ABSTAINED
    assert result.packet is None
    assert result.reason == "retrieval_no_answer"


def test_injection_fails_closed_when_current_changes_after_search(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "authority.db")
    try:
        receipt, retrieval = _retrieval(store, key="stale-receipt")
        store._conn.execute(
            "UPDATE memory_current SET updated_at = updated_at WHERE claim_id = ?",
            (receipt.claim_id,),
        )
        store._conn.execute(
            "UPDATE memory_fts SET governance_id = governance_id WHERE rowid = ?",
            (receipt.claim_id,),
        )
        # A changed current governance receipt must invalidate the earlier retrieval item.
        current = store._conn.execute(
            "SELECT claim_version_id FROM memory_current WHERE claim_id = ?",
            (receipt.claim_id,),
        ).fetchone()
        governance = store._conn.execute(
            """
            INSERT INTO memory_governance_events(
                claim_id, claim_version_id, action, previous_state, new_state,
                actor, reason_code, policy_version
            ) VALUES (?, ?, 'activate', 'pending_review', 'active', 'test',
                      'write_allowed', 'memory-policy-v1')
            """,
            (receipt.claim_id, current["claim_version_id"]),
        )
        store._conn.execute(
            "UPDATE memory_current SET memory_governance_id = ? WHERE claim_id = ?",
            (int(governance.lastrowid), receipt.claim_id),
        )
        store._conn.commit()
        result = build_governed_memory_context(
            store, retrieval, persona_mode=PersonaMode.WORK, total_budget=100,
        )
    finally:
        store.close()

    assert result.status is GovernedInjectionStatus.ABSTAINED
    assert result.packet is None
    assert result.reason == "authority_receipt_changed"


def test_unavailable_retrieval_never_reaches_arbitrator(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "authority.db")
    try:
        result = build_governed_memory_context(
            store,
            GovernedMemoryRetrievalResult.unavailable("authority_store_unavailable"),
            persona_mode=PersonaMode.WORK,
            total_budget=100,
        )
    finally:
        store.close()

    assert result.status is GovernedInjectionStatus.UNAVAILABLE
    assert result.packet is None
    assert result.reason == "retrieval_unavailable"


def test_unknown_memory_type_fails_closed_instead_of_becoming_semantic_memory(
    tmp_path: Path,
) -> None:
    store = EventStore(tmp_path / "authority.db")
    try:
        receipt = MemoryWriteService(store).write(
            MemoryWriteRequest(
                idempotency_key="unknown-type",
                content="unknown-type-token authority-bound",
                canonical_key="unknown:type",
                target="profile",
                memory_type="unmapped_future_type",
                sensitivity=Sensitivity.NORMAL,
                mode_scope=(PersonaMode.WORK,),
                injection_policy="allow",
            )
        )
        assert receipt.success is True
        MemoryFtsProjector(store, worker_id="fts-unknown").run_once(
            now="2026-07-29T02:00:00Z", lease_until="2026-07-29T02:01:00Z",
        )
        retrieval = search_governed_memories(
            store,
            GovernedMemorySearchRequest(
                query="unknown-type-token", persona_mode=PersonaMode.WORK,
            ),
        )
        result = build_governed_memory_context(
            store, retrieval, persona_mode=PersonaMode.WORK, total_budget=100,
        )
    finally:
        store.close()

    assert result.status is GovernedInjectionStatus.ABSTAINED
    assert result.packet is None
    assert result.reason == "memory_type_not_injectable"


def test_policy_denial_cannot_hide_unknown_memory_type(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "authority.db")
    try:
        receipt = MemoryWriteService(store).write(
            MemoryWriteRequest(
                idempotency_key="unknown-denied",
                content="unknown-denied-token authority-bound",
                canonical_key="unknown:denied",
                target="profile",
                memory_type="future_unknown_type",
                sensitivity=Sensitivity.NORMAL,
                mode_scope=(PersonaMode.WORK,),
                injection_policy="deny",
            )
        )
        assert receipt.success is True
        MemoryFtsProjector(store, worker_id="fts-unknown-denied").run_once(
            now="2026-07-29T02:00:00Z", lease_until="2026-07-29T02:01:00Z",
        )
        retrieval = search_governed_memories(
            store,
            GovernedMemorySearchRequest(
                query="unknown-denied-token", persona_mode=PersonaMode.WORK,
            ),
        )
        result = build_governed_memory_context(
            store, retrieval, persona_mode=PersonaMode.WORK, total_budget=100,
        )
    finally:
        store.close()

    assert result.status is GovernedInjectionStatus.ABSTAINED
    assert result.packet is None
    assert result.reason == "memory_type_not_injectable"


def test_budget_rejection_does_not_return_an_empty_success_packet(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "authority.db")
    try:
        _, retrieval = _retrieval(
            store,
            key="budget",
            content="authority-bound " + ("large-memory-content " * 20),
        )
        result = build_governed_memory_context(
            store, retrieval, persona_mode=PersonaMode.WORK, total_budget=1,
        )
    finally:
        store.close()

    assert result.status is GovernedInjectionStatus.ABSTAINED
    assert result.packet is None
    assert result.reason == "injection_budget_filtered"


def test_any_forged_receipt_prevents_partial_memory_packet(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "authority.db")
    try:
        _, retrieval = _retrieval(store, key="mixed-receipt")
        forged = replace(
            retrieval.items[0], governance_id=retrieval.items[0].governance_id + 100,
        )
        mixed = GovernedMemoryRetrievalResult.ok([retrieval.items[0], forged])
        result = build_governed_memory_context(
            store, mixed, persona_mode=PersonaMode.WORK, total_budget=100,
        )
    finally:
        store.close()

    assert result.status is GovernedInjectionStatus.ABSTAINED
    assert result.packet is None
    assert result.reason == "authority_receipt_changed"


def test_injection_rechecks_requested_mode_after_search(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "authority.db")
    try:
        _, retrieval = _retrieval(store, key="mode-recheck")
        result = build_governed_memory_context(
            store, retrieval, persona_mode=PersonaMode.DAILY, total_budget=100,
        )
    finally:
        store.close()

    assert result.status is GovernedInjectionStatus.ABSTAINED
    assert result.packet is None
    assert result.reason == "injection_policy_filtered"


def test_injection_authority_sqlite_failure_is_typed_unavailable(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "authority.db")
    try:
        _, retrieval = _retrieval(store, key="sqlite-failure")
        store._conn.execute("DROP TABLE memory_current")
        store._conn.commit()
        result = build_governed_memory_context(
            store, retrieval, persona_mode=PersonaMode.WORK, total_budget=100,
        )
    finally:
        store.close()

    assert result.status is GovernedInjectionStatus.UNAVAILABLE
    assert result.packet is None
    assert result.reason == "authority_store_unavailable"


def test_injection_does_not_commit_callers_existing_transaction(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "authority.db")
    try:
        _, retrieval = _retrieval(store, key="outer-transaction")
        store._conn.execute("BEGIN")
        result = build_governed_memory_context(
            store, retrieval, persona_mode=PersonaMode.WORK, total_budget=100,
        )
        still_owned_by_caller = store._conn.in_transaction
        store._conn.rollback()
    finally:
        store.close()

    assert result.status is GovernedInjectionStatus.OK
    assert still_owned_by_caller is True


def test_non_sqlite_programming_error_is_not_hidden_as_unavailable(
    tmp_path: Path, monkeypatch,
) -> None:
    store = EventStore(tmp_path / "authority.db")
    try:
        _, retrieval = _retrieval(store, key="programming-error")

        def broken_arbitration(*args, **kwargs):
            del args, kwargs
            raise RuntimeError("forced programming defect")

        monkeypatch.setattr(
            "pcltm.injection.governed_memory.InjectionArbitrator.arbitrate",
            broken_arbitration,
        )
        with pytest.raises(RuntimeError, match="forced programming defect"):
            build_governed_memory_context(
                store, retrieval, persona_mode=PersonaMode.WORK, total_budget=100,
            )
    finally:
        store.close()


def test_malformed_authority_row_fails_closed_as_receipt_change(
    tmp_path: Path, monkeypatch,
) -> None:
    store = EventStore(tmp_path / "authority.db")
    try:
        _, retrieval = _retrieval(store, key="malformed-authority")
        from pcltm.injection import governed_memory

        real_authority_row = governed_memory._authority_row

        def malformed_authority_row(active_store, claim_id):
            row = real_authority_row(active_store, claim_id)
            assert row is not None
            malformed = dict(row)
            malformed["version"] = "not-an-integer"
            return malformed

        monkeypatch.setattr(
            governed_memory, "_authority_row", malformed_authority_row,
        )
        result = build_governed_memory_context(
            store, retrieval, persona_mode=PersonaMode.WORK, total_budget=100,
        )
    finally:
        store.close()

    assert result.status is GovernedInjectionStatus.ABSTAINED
    assert result.packet is None
    assert result.reason == "authority_receipt_changed"


@pytest.mark.parametrize("total_budget", [0, -1, True, 1.5])
def test_injection_rejects_invalid_budgets(tmp_path: Path, total_budget: object) -> None:
    store = EventStore(tmp_path / "authority.db")
    try:
        with pytest.raises((TypeError, ValueError)):
            build_governed_memory_context(
                store,
                GovernedMemoryRetrievalResult.abstained("no_answer"),
                persona_mode=PersonaMode.WORK,
                total_budget=total_budget,  # type: ignore[arg-type]
            )
    finally:
        store.close()
