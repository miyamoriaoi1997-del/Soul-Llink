from __future__ import annotations

from pathlib import Path

import sqlite3
import pytest
import pcltm.memory_retrieval as memory_retrieval
import pcltm.semantic_claim_retrieval as semantic_claim_retrieval

from pcltm.evidence_chain import sha256_text
from pcltm.memory_contracts import PersonaMode, Sensitivity
from pcltm.memory_retrieval import (
    GovernedMemoryOpenRequest,
    GovernedMemorySearchRequest,
    MemoryRetrievalStatus,
    open_governed_memory,
    search_governed_memories,
)
from pcltm.memory_write_service import MemoryWriteRequest, MemoryWriteService
from pcltm.projections.memory_fts import MemoryFtsProjector
from pcltm.store import EventStore


def _write_and_project(
    store: EventStore,
    *,
    key: str,
    content: str,
    sensitivity: Sensitivity = Sensitivity.NORMAL,
    modes: tuple[PersonaMode, ...] = (PersonaMode.WORK,),
):
    receipt = MemoryWriteService(store).write(
        MemoryWriteRequest(
            idempotency_key=key,
            content=content,
            canonical_key=f"test:{key}",
            target="memory",
            memory_type="preference",
            sensitivity=sensitivity,
            mode_scope=modes,
            injection_policy="allow",
        )
    )
    outcome = MemoryFtsProjector(store, worker_id=f"fts-{key}").run_once(
        now="2026-07-29T01:00:00Z",
        lease_until="2026-07-29T01:01:00Z",
    )
    assert outcome["applied"] == 1
    return receipt


def test_search_reopens_fts_candidate_against_current_authority(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "authority.db")
    try:
        receipt = _write_and_project(
            store, key="search-authority", content="老师偏好权威约束 RAG 路线",
        )
        result = search_governed_memories(
            store,
            GovernedMemorySearchRequest(
                query="RAG", persona_mode=PersonaMode.WORK, limit=4,
            ),
        )
    finally:
        store.close()

    assert result.status is MemoryRetrievalStatus.OK
    assert len(result.items) == 1
    item = result.items[0]
    assert item.claim_id == receipt.claim_id
    assert item.claim_version == receipt.claim_version
    assert item.governance_id == receipt.governance_id
    assert item.content == "老师偏好权威约束 RAG 路线"
    assert item.authority_verified is True
    assert item.rank == 1
    assert item.rank_score is not None
    assert item.rank_score_is_authority is False
    assert item.policy_reason == "access_allowed"
    assert item.policy_version == "memory-policy-v1"
    assert item.source_refs[0].authority_kind == "event"


def test_search_uses_short_numeric_term_to_rank_the_exact_candidate(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "authority.db")
    try:
        _write_and_project(store, key="short-one", content="opaque marker 1")
        exact = _write_and_project(store, key="short-three", content="opaque marker 3")
        result = search_governed_memories(
            store,
            GovernedMemorySearchRequest(
                query="marker 3", persona_mode=PersonaMode.WORK, limit=1,
            ),
        )
    finally:
        store.close()

    assert result.status is MemoryRetrievalStatus.OK
    assert [item.claim_id for item in result.items] == [exact.claim_id]


def test_search_requires_every_short_term_to_match(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "authority.db")
    try:
        _write_and_project(store, key="short-one", content="opaque marker 1")
        _write_and_project(store, key="short-three", content="opaque marker 3")
        result = search_governed_memories(
            store, GovernedMemorySearchRequest(
                query="marker 9", persona_mode=PersonaMode.WORK, limit=1,
            ),
        )
    finally:
        store.close()
    assert result.status is MemoryRetrievalStatus.ABSTAINED
    assert result.items == ()


def test_semantic_fallback_reopens_paraphrase_candidate_through_authority(
    tmp_path: Path, monkeypatch,
) -> None:
    store = EventStore(tmp_path / "authority.db")
    try:
        receipt = _write_and_project(
            store, key="semantic-paraphrase",
            content="审计收尾还要检查版本控制看不到的缓存和临时备份。",
        )
        monkeypatch.setattr(
            "pcltm.semantic_claim_retrieval.semantic_claim_candidates",
            lambda _store, _query, *, limit: [(receipt.claim_id, 0.93)],
        )
        result = search_governed_memories(
            store,
            GovernedMemorySearchRequest(
                query="仓库干净是否就代表磁盘没有残留？",
                persona_mode=PersonaMode.WORK,
                limit=4,
            ),
        )
    finally:
        store.close()

    assert result.status is MemoryRetrievalStatus.OK
    assert [item.claim_id for item in result.items] == [receipt.claim_id]
    assert result.items[0].rank_score == pytest.approx(0.93)
    assert result.items[0].authority_verified is True


def test_semantic_fallback_abstains_when_provider_has_no_confident_candidate(
    tmp_path: Path, monkeypatch,
) -> None:
    store = EventStore(tmp_path / "authority.db")
    try:
        _write_and_project(store, key="semantic-negative", content="稳定的工程偏好记录。")
        monkeypatch.setattr(
            "pcltm.semantic_claim_retrieval.semantic_claim_candidates",
            lambda _store, _query, *, limit: [],
        )
        result = search_governed_memories(
            store,
            GovernedMemorySearchRequest(
                query="今天晚饭吃什么？", persona_mode=PersonaMode.WORK, limit=4,
            ),
        )
    finally:
        store.close()

    assert result.status is MemoryRetrievalStatus.ABSTAINED
    assert result.reason == "no_answer"


def test_semantic_fallback_sqlite_failure_is_typed_unavailable(
    tmp_path: Path, monkeypatch,
) -> None:
    store = EventStore(tmp_path / "authority.db")
    try:
        _write_and_project(store, key="semantic-sqlite-failure", content="稳定的工程偏好记录。")

        def fail_sqlite(_store, _query, *, limit):
            raise sqlite3.OperationalError("injected semantic authority read failure")

        monkeypatch.setattr(
            "pcltm.semantic_claim_retrieval.semantic_claim_candidates", fail_sqlite,
        )
        result = search_governed_memories(
            store,
            GovernedMemorySearchRequest(
                query="完全不匹配的改写查询", persona_mode=PersonaMode.WORK, limit=4,
            ),
        )
    finally:
        store.close()

    assert result.status is MemoryRetrievalStatus.UNAVAILABLE
    assert result.reason == "authority_store_unavailable"


def test_semantic_inference_sqlite_failure_is_typed_unavailable(
    tmp_path: Path, monkeypatch,
) -> None:
    store = EventStore(tmp_path / "authority.db")
    try:
        _write_and_project(
            store,
            key="semantic-inference-sqlite",
            content="这是一条长度足够用于语义候选推理的稳定工程偏好记录。",
        )
        monkeypatch.setenv("SOULLINK_PCLTM_E5_RETRIEVAL_ENABLED", "1")
        monkeypatch.setattr(
            "pcltm.semantic_claim_retrieval._embed",
            lambda _texts, *, query: (_ for _ in ()).throw(
                sqlite3.OperationalError("injected inference failure")
            ),
        )
        result = search_governed_memories(
            store,
            GovernedMemorySearchRequest(
                query="无词法重合的查询", persona_mode=PersonaMode.WORK, limit=4,
            ),
        )
    finally:
        store.close()

    assert result.status is MemoryRetrievalStatus.UNAVAILABLE
    assert result.reason == "authority_store_unavailable"


def test_semantic_inference_programming_error_is_not_silenced(
    tmp_path: Path, monkeypatch,
) -> None:
    store = EventStore(tmp_path / "authority.db")
    try:
        _write_and_project(
            store,
            key="semantic-inference-bug",
            content="这是一条长度足够用于语义候选推理的稳定工程偏好记录。",
        )
        monkeypatch.setenv("SOULLINK_PCLTM_E5_RETRIEVAL_ENABLED", "1")
        monkeypatch.setattr(
            "pcltm.semantic_claim_retrieval._embed",
            lambda _texts, *, query: (_ for _ in ()).throw(RuntimeError("injected bug")),
        )
        with pytest.raises(RuntimeError, match="injected bug"):
            search_governed_memories(
                store,
                GovernedMemorySearchRequest(
                    query="无词法重合的查询", persona_mode=PersonaMode.WORK, limit=4,
                ),
            )
    finally:
        store.close()


def test_semantic_malformed_claim_id_is_classified_as_database_error(
    monkeypatch,
) -> None:
    class FakeConnection:
        def execute(self, _sql):
            return self

        def fetchall(self):
            return [{
                "claim_id": "bad-id",
                "content": "这是一条长度明确超过最低限制并用于验证畸形数据库行分类的语义候选正文记录。",
            }]

    class FakeStore:
        _conn = FakeConnection()

    monkeypatch.setenv("SOULLINK_PCLTM_E5_RETRIEVAL_ENABLED", "1")

    with pytest.raises(sqlite3.DatabaseError, match="malformed semantic claim row"):
        semantic_claim_retrieval.semantic_claim_candidates(
            FakeStore(), "无词法重合查询", limit=4,
        )


@pytest.mark.parametrize("claim_id", [1.5, True, "01", b"1", [1]])
def test_semantic_rejects_coercible_malformed_claim_ids(
    monkeypatch, claim_id,
) -> None:
    class FakeConnection:
        def execute(self, _sql):
            return self

        def fetchall(self):
            return [{
                "claim_id": claim_id,
                "content": "这是一条长度明确超过最低限制并用于验证严格数据库类型的语义候选正文记录。",
            }]

    class FakeStore:
        _conn = FakeConnection()

    monkeypatch.setenv("SOULLINK_PCLTM_E5_RETRIEVAL_ENABLED", "1")

    with pytest.raises(sqlite3.DatabaseError, match="malformed semantic claim row"):
        semantic_claim_retrieval.semantic_claim_candidates(
            FakeStore(), "无词法重合查询", limit=4,
        )


@pytest.mark.parametrize("content", [b"bytes content", ["list content"], 123, None])
def test_semantic_rejects_coercible_malformed_content(
    monkeypatch, content,
) -> None:
    class FakeConnection:
        def execute(self, _sql):
            return self

        def fetchall(self):
            return [{"claim_id": 1, "content": content}]

    class FakeStore:
        _conn = FakeConnection()

    monkeypatch.setenv("SOULLINK_PCLTM_E5_RETRIEVAL_ENABLED", "1")

    with pytest.raises(sqlite3.DatabaseError, match="malformed semantic claim row"):
        semantic_claim_retrieval.semantic_claim_candidates(
            FakeStore(), "无词法重合查询", limit=4,
        )


def test_semantic_malformed_claim_id_returns_typed_unavailable(
    tmp_path: Path, monkeypatch,
) -> None:
    store = EventStore(tmp_path / "authority.db")
    try:
        _write_and_project(
            store,
            key="semantic-malformed-public",
            content="这是一条长度足够用于验证公共检索边界畸形数据库分类的工程偏好记录。",
        )
        monkeypatch.setattr(
            "pcltm.semantic_claim_retrieval.semantic_claim_candidates",
            lambda _store, _query, *, limit: (_ for _ in ()).throw(
                sqlite3.DatabaseError("malformed semantic claim row")
            ),
        )
        result = search_governed_memories(
            store,
            GovernedMemorySearchRequest(
                query="无词法重合公共查询", persona_mode=PersonaMode.WORK, limit=4,
            ),
        )
    finally:
        store.close()

    assert result.status is MemoryRetrievalStatus.UNAVAILABLE
    assert result.reason == "authority_store_unavailable"


def test_open_uses_current_authority_and_returns_full_content(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "authority.db")
    try:
        receipt = _write_and_project(
            store, key="open-authority", content="完整治理记忆正文，不由候选摘要替代。",
        )
        store._conn.execute(
            "UPDATE memory_fts SET content = '伪造候选正文' WHERE rowid = ?",
            (receipt.claim_id,),
        )
        store._conn.commit()
        result = open_governed_memory(
            store,
            GovernedMemoryOpenRequest(
                claim_id=receipt.claim_id, persona_mode=PersonaMode.WORK,
            ),
        )
    finally:
        store.close()

    assert result.status is MemoryRetrievalStatus.OK
    assert len(result.items) == 1
    assert result.items[0].content == "完整治理记忆正文，不由候选摘要替代。"
    assert result.items[0].rank is None
    assert result.items[0].rank_score is None


def test_open_fails_closed_on_historical_invalid_governance_transition(
    tmp_path: Path,
) -> None:
    store = EventStore(tmp_path / "authority.db")
    try:
        receipt = _write_and_project(
            store, key="invalid-governance", content="invalid-governance-token",
        )
        version_id = int(store._conn.execute(
            "SELECT claim_version_id FROM memory_current WHERE claim_id = ?",
            (receipt.claim_id,),
        ).fetchone()[0])
        store._conn.execute("DROP TRIGGER memory_governance_transition_valid")
        governance_id = int(store._conn.execute(
            """
            INSERT INTO memory_governance_events(
                claim_id, claim_version_id, action, previous_state, new_state,
                actor, reason_code, policy_version
            ) VALUES (?, ?, 'retire', 'pending_review', 'active', 'legacy', 'x',
                      'memory-policy-v1')
            """,
            (receipt.claim_id, version_id),
        ).lastrowid)
        store._conn.execute(
            "UPDATE memory_current SET memory_governance_id = ? WHERE claim_id = ?",
            (governance_id, receipt.claim_id),
        )
        store._conn.commit()
        result = open_governed_memory(
            store,
            GovernedMemoryOpenRequest(
                claim_id=receipt.claim_id, persona_mode=PersonaMode.WORK,
            ),
        )
    finally:
        store.close()

    assert result.status is MemoryRetrievalStatus.ABSTAINED
    assert result.reason == "policy_filtered"


def test_search_fails_closed_on_fts_content_hash_drift(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "authority.db")
    try:
        receipt = _write_and_project(
            store, key="fts-drift", content="漂移检测 needle authority",
        )
        store._conn.execute(
            "UPDATE memory_fts SET payload_sha256 = ? WHERE rowid = ?",
            ("0" * 64, receipt.claim_id),
        )
        store._conn.commit()
        result = search_governed_memories(
            store,
            GovernedMemorySearchRequest(
                query="needle", persona_mode=PersonaMode.WORK,
            ),
        )
    finally:
        store.close()

    assert result.status is MemoryRetrievalStatus.ABSTAINED
    assert result.items == ()
    assert result.reason == "authority_reopen_failed"


def test_search_applies_mode_and_sensitivity_policy_after_reopen(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "authority.db")
    try:
        _write_and_project(
            store,
            key="private-daily",
            content="private-policy-needle 私密偏好",
            sensitivity=Sensitivity.PRIVATE,
            modes=(PersonaMode.DAILY,),
        )
        denied = search_governed_memories(
            store,
            GovernedMemorySearchRequest(
                query="private-policy-needle", persona_mode=PersonaMode.WORK,
                sensitivity_ceiling=Sensitivity.PRIVATE,
            ),
        )
        allowed = search_governed_memories(
            store,
            GovernedMemorySearchRequest(
                query="private-policy-needle", persona_mode=PersonaMode.DAILY,
                sensitivity_ceiling=Sensitivity.PRIVATE,
            ),
        )
    finally:
        store.close()

    assert denied.status is MemoryRetrievalStatus.ABSTAINED
    assert denied.reason == "policy_filtered"
    assert allowed.status is MemoryRetrievalStatus.OK
    assert len(allowed.items) == 1


def test_search_does_not_overfetch_to_force_top_k(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "authority.db")
    try:
        denied = _write_and_project(
            store,
            key="rank-one-denied",
            content="bounded-top-k commonterm commonterm commonterm",
            sensitivity=Sensitivity.PRIVATE,
            modes=(PersonaMode.DAILY,),
        )
        allowed = _write_and_project(
            store,
            key="rank-two-allowed",
            content="bounded-top-k commonterm",
        )
        result = search_governed_memories(
            store,
            GovernedMemorySearchRequest(
                query="commonterm", persona_mode=PersonaMode.WORK, limit=1,
                sensitivity_ceiling=Sensitivity.PRIVATE,
            ),
        )
    finally:
        store.close()

    assert denied.claim_id != allowed.claim_id
    assert result.status is MemoryRetrievalStatus.ABSTAINED
    assert result.reason == "policy_filtered"


def test_blank_query_and_empty_authority_abstain(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "authority.db")
    try:
        blank = search_governed_memories(
            store,
            GovernedMemorySearchRequest(query="  ", persona_mode=PersonaMode.WORK),
        )
        missing = search_governed_memories(
            store,
            GovernedMemorySearchRequest(query="needle", persona_mode=PersonaMode.WORK),
        )
    finally:
        store.close()

    assert blank.status is MemoryRetrievalStatus.ABSTAINED
    assert blank.reason == "no_answer"
    assert missing.status is MemoryRetrievalStatus.ABSTAINED
    assert missing.reason == "no_answer"


def test_search_and_open_return_typed_unavailable_when_authority_store_is_broken(
    tmp_path: Path,
) -> None:
    store = EventStore(tmp_path / "authority.db")
    try:
        receipt = _write_and_project(
            store, key="broken-authority", content="broken-authority-token",
        )
        store._conn.execute("DROP TABLE memory_current")
        store._conn.commit()
        searched = search_governed_memories(
            store,
            GovernedMemorySearchRequest(
                query="broken-authority-token", persona_mode=PersonaMode.WORK,
            ),
        )
        opened = open_governed_memory(
            store,
            GovernedMemoryOpenRequest(
                claim_id=receipt.claim_id, persona_mode=PersonaMode.WORK,
            ),
        )
    finally:
        store.close()

    assert searched.status is MemoryRetrievalStatus.UNAVAILABLE
    assert searched.reason == "authority_store_unavailable"
    assert opened.status is MemoryRetrievalStatus.UNAVAILABLE
    assert opened.reason == "authority_store_unavailable"


def test_transaction_boundary_sqlite_failure_returns_typed_unavailable(
    tmp_path: Path, monkeypatch,
) -> None:
    store = EventStore(tmp_path / "authority.db")
    try:
        def broken_snapshot(_store):
            raise sqlite3.OperationalError("forced begin failure")

        monkeypatch.setattr(memory_retrieval, "_read_snapshot", broken_snapshot)
        searched = search_governed_memories(
            store,
            GovernedMemorySearchRequest(query="needle", persona_mode=PersonaMode.WORK),
        )
        opened = open_governed_memory(
            store,
            GovernedMemoryOpenRequest(claim_id=1, persona_mode=PersonaMode.WORK),
        )
    finally:
        store.close()

    assert searched.status is MemoryRetrievalStatus.UNAVAILABLE
    assert searched.reason == "authority_store_unavailable"
    assert opened.status is MemoryRetrievalStatus.UNAVAILABLE
    assert opened.reason == "authority_store_unavailable"


def test_open_missing_claim_abstains_without_legacy_fallback(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "authority.db")
    try:
        result = open_governed_memory(
            store,
            GovernedMemoryOpenRequest(claim_id=999, persona_mode=PersonaMode.WORK),
        )
    finally:
        store.close()

    assert result.status is MemoryRetrievalStatus.ABSTAINED
    assert result.reason == "authority_not_found"


def test_search_classifies_projected_source_drift_as_authority_failure(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "authority.db")
    try:
        receipt = _write_and_project(
            store, key="source-drift", content="source-drift-token authority",
        )
        store._conn.execute(
            "UPDATE memory_fts SET source_refs = '[]' WHERE rowid = ?",
            (receipt.claim_id,),
        )
        store._conn.commit()
        result = search_governed_memories(
            store,
            GovernedMemorySearchRequest(
                query="source-drift-token", persona_mode=PersonaMode.WORK,
            ),
        )
    finally:
        store.close()

    assert result.status is MemoryRetrievalStatus.ABSTAINED
    assert result.reason == "authority_reopen_failed"


def test_search_fails_closed_on_projected_policy_version_drift(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "authority.db")
    try:
        receipt = _write_and_project(
            store, key="policy-version-drift", content="policy-version-drift-token",
        )
        store._conn.execute(
            "UPDATE memory_fts SET policy_version = 'forged-policy' WHERE rowid = ?",
            (receipt.claim_id,),
        )
        store._conn.commit()
        result = search_governed_memories(
            store,
            GovernedMemorySearchRequest(
                query="policy-version-drift-token", persona_mode=PersonaMode.WORK,
            ),
        )
    finally:
        store.close()

    assert result.status is MemoryRetrievalStatus.ABSTAINED
    assert result.reason == "authority_reopen_failed"


def test_search_reopens_sources_inside_one_read_transaction(tmp_path: Path, monkeypatch) -> None:
    store = EventStore(tmp_path / "authority.db")
    observed: list[bool] = []
    try:
        _write_and_project(store, key="snapshot", content="snapshot-transaction-token")
        original = memory_retrieval._source_refs

        def observe(active_store, version_id):
            observed.append(active_store._conn.in_transaction)
            return original(active_store, version_id)

        monkeypatch.setattr(memory_retrieval, "_source_refs", observe)
        result = search_governed_memories(
            store,
            GovernedMemorySearchRequest(
                query="snapshot-transaction-token", persona_mode=PersonaMode.WORK,
            ),
        )
    finally:
        store.close()

    assert result.status is MemoryRetrievalStatus.OK
    assert observed and all(observed)


def test_retrieval_does_not_commit_callers_existing_transaction(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "authority.db")
    try:
        _write_and_project(store, key="outer-transaction", content="outer-transaction-token")
        store._conn.execute("BEGIN")
        result = search_governed_memories(
            store,
            GovernedMemorySearchRequest(
                query="outer-transaction-token", persona_mode=PersonaMode.WORK,
            ),
        )
        still_owned_by_caller = store._conn.in_transaction
        store._conn.rollback()
    finally:
        store.close()

    assert result.status is MemoryRetrievalStatus.OK
    assert still_owned_by_caller is True


def test_search_holds_stable_snapshot_across_concurrent_current_switch(
    tmp_path: Path, monkeypatch,
) -> None:
    db = tmp_path / "authority.db"
    reader = EventStore(db)
    writer = None
    switched = False
    observed_versions: list[int | None] = []
    try:
        receipt = _write_and_project(
            reader, key="concurrent-switch", content="concurrent-snapshot-token old",
        )
        writer = EventStore(db)
        original = memory_retrieval._source_refs

        def switch_current(active_store, version_id):
            nonlocal switched
            if not switched:
                switched = True
                source = writer._conn.execute(
                    """
                    SELECT s.event_id, s.event_revision, s.event_payload_sha256
                    FROM memory_current mc
                    JOIN memory_claim_sources s ON s.claim_version_id = mc.claim_version_id
                    WHERE mc.claim_id = ? AND s.source_kind = 'event'
                    """,
                    (receipt.claim_id,),
                ).fetchone()
                new_content = "concurrent-snapshot-token new"
                version = writer._conn.execute(
                    """
                    INSERT INTO memory_claim_versions(
                        claim_id, version, content, content_sha256, confidence,
                        sensitivity, injection_policy, mode_scope, lineage_kind, schema_version
                    ) VALUES (?, 2, ?, ?, 1.0, 'normal', 'allow', '[\"work\"]',
                              'explicit_user_assertion', 1)
                    """,
                    (receipt.claim_id, new_content, sha256_text(new_content)),
                )
                new_version_id = int(version.lastrowid)
                writer._conn.execute(
                    """
                    INSERT INTO memory_claim_sources(
                        claim_version_id, source_kind, event_id, event_revision,
                        event_payload_sha256
                    ) VALUES (?, 'event', ?, ?, ?)
                    """,
                    (new_version_id, source["event_id"], source["event_revision"], source["event_payload_sha256"]),
                )
                governance = writer._conn.execute(
                    """
                    INSERT INTO memory_governance_events(
                        claim_id, claim_version_id, action, previous_state, new_state,
                        actor, reason_code, policy_version
                    ) VALUES (?, ?, 'activate', 'pending_review', 'active', 'concurrency-test',
                              'write_allowed', 'memory-policy-v1')
                    """,
                    (receipt.claim_id, new_version_id),
                )
                writer._conn.execute(
                    """
                    UPDATE memory_current
                    SET claim_version_id = ?, memory_governance_id = ?, lifecycle_state = 'active'
                    WHERE claim_id = ?
                    """,
                    (new_version_id, int(governance.lastrowid), receipt.claim_id),
                )
                writer._conn.commit()
            visible = active_store._conn.execute(
                """
                SELECT v.version FROM memory_current mc
                JOIN memory_claim_versions v ON v.claim_version_id = mc.claim_version_id
                WHERE mc.claim_id = ?
                """,
                (receipt.claim_id,),
            ).fetchone()
            observed_versions.append(None if visible is None else int(visible["version"]))
            return original(active_store, version_id)

        monkeypatch.setattr(memory_retrieval, "_source_refs", switch_current)
        result = search_governed_memories(
            reader,
            GovernedMemorySearchRequest(
                query="concurrent-snapshot-token", persona_mode=PersonaMode.WORK,
            ),
        )
        reopened = open_governed_memory(
            reader,
            GovernedMemoryOpenRequest(
                claim_id=receipt.claim_id, persona_mode=PersonaMode.WORK,
            ),
        )
    finally:
        if writer is not None:
            writer.close()
        reader.close()

    assert switched is True
    assert observed_versions and observed_versions[0] == 1
    assert result.status is MemoryRetrievalStatus.OK
    assert result.items[0].claim_version == 1
    assert result.items[0].content.endswith(" old")
    assert reopened.status is MemoryRetrievalStatus.OK
    assert reopened.items[0].claim_version == 2
    assert reopened.items[0].content.endswith(" new")


@pytest.mark.parametrize("field", ["claim_version", "governance_id", "projection_generation"])
def test_search_fails_closed_on_malformed_numeric_projection_commitment(
    tmp_path: Path, field: str,
) -> None:
    store = EventStore(tmp_path / "authority.db")
    try:
        receipt = _write_and_project(
            store, key=f"numeric-drift-{field}", content=f"numeric-drift-{field}-token",
        )
        store._conn.execute(
            f"UPDATE memory_fts SET {field} = 'not-an-int' WHERE rowid = ?",
            (receipt.claim_id,),
        )
        store._conn.commit()
        result = search_governed_memories(
            store,
            GovernedMemorySearchRequest(
                query=f"numeric-drift-{field}-token", persona_mode=PersonaMode.WORK,
            ),
        )
    finally:
        store.close()

    assert result.status is MemoryRetrievalStatus.ABSTAINED
    assert result.reason == "authority_reopen_failed"


def test_search_treats_query_operators_as_literal_text(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "authority.db")
    try:
        _write_and_project(
            store, key="literal-query", content="literal-operator-token is governed",
        )
        result = search_governed_memories(
            store,
            GovernedMemorySearchRequest(
                query="literal-operator-token", persona_mode=PersonaMode.WORK,
            ),
        )
    finally:
        store.close()

    assert result.status is MemoryRetrievalStatus.OK
    assert len(result.items) == 1


def test_search_returns_fewer_than_limit_after_policy_filtering(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "authority.db")
    try:
        _write_and_project(
            store,
            key="mixed-denied",
            content="mixed-policy-token common common common",
            sensitivity=Sensitivity.PRIVATE,
            modes=(PersonaMode.DAILY,),
        )
        allowed = _write_and_project(
            store,
            key="mixed-allowed",
            content="mixed-policy-token common",
        )
        result = search_governed_memories(
            store,
            GovernedMemorySearchRequest(
                query="mixed-policy-token", persona_mode=PersonaMode.WORK, limit=2,
                sensitivity_ceiling=Sensitivity.PRIVATE,
            ),
        )
    finally:
        store.close()

    assert result.status is MemoryRetrievalStatus.OK
    assert len(result.items) == 1
    assert result.items[0].claim_id == allowed.claim_id


@pytest.mark.parametrize("corruption", ["commitment", "missing_source", "malformed_numeric"])
def test_search_mixed_valid_and_corrupt_candidates_is_unavailable(
    tmp_path: Path, corruption: str,
) -> None:
    store = EventStore(tmp_path / f"mixed-{corruption}.db")
    try:
        corrupt = _write_and_project(
            store, key=f"mixed-corrupt-{corruption}", content=f"mixed-integrity-token {corruption}",
        )
        valid = _write_and_project(
            store, key=f"mixed-valid-{corruption}", content="mixed-integrity-token valid",
        )
        if corruption == "commitment":
            store._conn.execute(
                "UPDATE memory_fts SET payload_sha256 = ? WHERE rowid = ?",
                ("0" * 64, corrupt.claim_id),
            )
        elif corruption == "missing_source":
            store._conn.execute(
                "UPDATE memory_fts SET source_refs = ? WHERE rowid = ?",
                ("[]", corrupt.claim_id),
            )
        else:
            store._conn.execute(
                "UPDATE memory_fts SET projection_generation = ? WHERE rowid = ?",
                ("not-an-int", corrupt.claim_id),
            )
        store._conn.commit()
        result = search_governed_memories(
            store,
            GovernedMemorySearchRequest(
                query="mixed-integrity-token", persona_mode=PersonaMode.WORK, limit=2,
            ),
        )
    finally:
        store.close()

    assert valid.claim_id != corrupt.claim_id
    assert result.status is MemoryRetrievalStatus.UNAVAILABLE
    assert result.items == ()
    assert result.reason == "authority_reopen_failed"


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, 101])
def test_search_request_rejects_invalid_limits(limit: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        GovernedMemorySearchRequest(
            query="needle", persona_mode=PersonaMode.WORK, limit=limit,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("claim_id", [0, -1, True, "1"])
def test_open_request_rejects_invalid_claim_ids(claim_id: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        GovernedMemoryOpenRequest(
            claim_id=claim_id, persona_mode=PersonaMode.WORK,  # type: ignore[arg-type]
        )


def test_search_multi_term_query_recalls_non_contiguous_terms(tmp_path: Path) -> None:
    """Multi-word queries must recall memories whose terms are NOT contiguous.

    Regression for the whole-string FTS phrase construction: query terms are
    now AND-combined per term, so a natural-language query like
    "状态机 语义 影子运行" hits a memory that contains those words separately.
    """
    store = EventStore(tmp_path / "multi.db")
    try:
        receipt = _write_and_project(
            store,
            key="multi-term",
            content="还记得我们的状态机切换规则系统吗，上次打开了语义系统的影子运行。",
            modes=(PersonaMode.DAILY, PersonaMode.WORK),
        )
        result = search_governed_memories(
            store,
            GovernedMemorySearchRequest(
                query="状态机 语义 影子运行",
                persona_mode=PersonaMode.DAILY,
                limit=4,
            ),
        )
    finally:
        store.close()

    assert result.status is MemoryRetrievalStatus.OK
    assert len(result.items) == 1
    item = result.items[0]
    assert item.claim_id == receipt.claim_id
    assert item.authority_verified is True
    assert item.policy_reason == "access_allowed"
