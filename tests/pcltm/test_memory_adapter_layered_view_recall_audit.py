from __future__ import annotations

from pcltm.memory_contracts import PersonaMode, Sensitivity
from pcltm.memory_retrieval import (
    GovernedMemorySearchRequest,
    MemoryRetrievalStatus,
    search_governed_memories,
)
from pcltm.memory_write_service import MemoryWriteRequest, MemoryWriteService
from pcltm.projections.memory_fts import MemoryFtsProjector
from pcltm.store import EventStore


def test_governed_retrieval_preserves_authority_metadata_for_audits(tmp_path) -> None:
    store = EventStore(tmp_path / "authority.db")
    try:
        receipt = MemoryWriteService(store).write(MemoryWriteRequest(
            idempotency_key="audit-metadata",
            content="audit-token 用户希望长期记忆召回要完整命中，不要只靠短词撞中。",
            canonical_key="profile:memory-quality",
            target="profile",
            memory_type="preference",
            sensitivity=Sensitivity.NORMAL,
            mode_scope=(PersonaMode.WORK,),
            injection_policy="allow",
        ))
        assert receipt.success is True
        outcome = MemoryFtsProjector(store, worker_id="audit").run_once(
            now="2026-07-31T03:00:00Z", lease_until="2026-07-31T03:01:00Z",
        )
        assert outcome["applied"] == 1

        result = search_governed_memories(store, GovernedMemorySearchRequest(
            query="audit-token", persona_mode=PersonaMode.WORK,
        ))
    finally:
        store.close()

    assert result.status is MemoryRetrievalStatus.OK
    item = result.items[0]
    assert item.claim_id == receipt.claim_id
    assert item.claim_version == receipt.claim_version
    assert item.governance_id == receipt.governance_id
    assert item.canonical_key == "profile:memory-quality"
    assert item.target == "profile"
    assert item.memory_type == "preference"
    assert item.mode_scope == (PersonaMode.WORK,)
    assert item.sensitivity is Sensitivity.NORMAL
    assert item.authority_verified is True
    assert item.rank == 1
    assert item.rank_score is not None
    assert item.rank_score_is_authority is False
    assert item.policy_reason == "access_allowed"
    assert item.content.endswith("用户希望长期记忆召回要完整命中，不要只靠短词撞中。")
