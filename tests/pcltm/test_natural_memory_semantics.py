from __future__ import annotations

from pathlib import Path

import pytest
from pcltm.candidate_promotion import CandidatePromotionService
from pcltm.candidates import PersonaCandidateExtractor
from pcltm.classifier import parse_stable_memory_assertion
from pcltm.store import EventStore


@pytest.mark.parametrize(
    ("content", "kind", "target", "semantic_key"),
    [
        ("回答直接一点就好，别铺垫太多。", "user_preference", "user", "preference:response-style:directness"),
        ("比起安慰，我更在意你把实际问题查清楚。", "user_preference", "user", "preference:response-style:evidence-over-reassurance"),
        ("长篇解释我一般看不下去。", "user_preference", "user", "preference:response-style:conciseness"),
        ("我做机械设计很多年了。", "identity_fact", "user", "identity:职业"),
        ("提交代码这种事，先跟我说一声。", "system_convention", "memory", "convention:git:commit-confirmation"),
        ("这种修改以后都要跑完测试再告诉我。", "system_convention", "memory", "convention:change:test-before-report"),
        ("我向来更喜欢深色界面。", "user_preference", "user", None),
        ("我受不了回答里一大段客套话。", "user_preference", "user", None),
        ("我是在电厂做设备工程的。", "identity_fact", "user", "identity:职业"),
        ("往后涉及删文件都先问我。", "system_convention", "memory", None),
    ],
)
def test_natural_semantic_assertions_are_recognized_without_memory_keywords(
    content: str, kind: str, target: str, semantic_key: str,
) -> None:
    assertion = parse_stable_memory_assertion(content)

    assert assertion is not None
    assert (assertion.kind, assertion.target_file) == (kind, target)
    if semantic_key is not None:
        assert assertion.semantic_key == semantic_key
    if semantic_key is not None:
        assert assertion.semantic_score > assertion.lexical_score
    assert 0.6 <= assertion.admission_confidence < 0.85


@pytest.mark.parametrize(
    "content",
    [
        "我朋友喜欢简洁回答。",
        "我不是说我喜欢咖啡。",
        "今天回答直接一点。",
        "这次先别提交代码。",
        "我这会儿更喜欢安静。",
        "也许我更喜欢长篇解释。",
        "你觉得我适合直接一点的回答吗？",
        "用户说：回答直接一点就好。",
        "把这个文件改一下。",
    ],
)
def test_semantic_admission_rejects_wrong_subject_negation_quotes_and_transience(content: str) -> None:
    assert parse_stable_memory_assertion(content) is None


def test_semantic_paraphrases_share_a_stable_identity() -> None:
    first = parse_stable_memory_assertion("回答直接一点就好，别铺垫太多。")
    second = parse_stable_memory_assertion("我更喜欢直接给结论，少些铺垫。")

    assert first is not None and second is not None
    assert first.semantic_key == second.semantic_key == "preference:response-style:directness"


@pytest.mark.parametrize(
    "content",
    [
        "你知道的，助手我喜欢最完美的代码和最干净的架构。",
        "说真的，助手，我喜欢完美的代码与干净的架构。",
        "其实我更看重代码质量和架构整洁。",
        "助手，我偏爱高质量代码、清晰而干净的架构。",
    ],
)
def test_dialogue_prefixes_and_vocatives_preserve_core_engineering_preference(
    content: str,
) -> None:
    assertion = parse_stable_memory_assertion(content)

    assert assertion is not None
    assert assertion.kind == "user_preference"
    assert assertion.target_file == "user"
    assert assertion.semantic_key == "preference:engineering:code-and-architecture-quality"
    assert assertion.content == content.rstrip("。")
    assert assertion.semantic_score > assertion.lexical_score
    assert 0.6 <= assertion.admission_confidence < 0.85


@pytest.mark.parametrize(
    "content",
    [
        "你知道的，凛，我今天喜欢把这段代码写得漂亮点。",
        "凛，我可能喜欢最干净的架构。",
        "你知道的，凛，我不是说我喜欢完美代码。",
        "凛，我朋友喜欢最干净的架构。",
        "用户说：凛，我喜欢最完美的代码和最干净的架构。",
    ],
)
def test_dialogue_prefix_normalization_does_not_bypass_safety_exclusions(content: str) -> None:
    assert parse_stable_memory_assertion(content) is None


def test_certain_natural_semantic_candidate_auto_activates(
    tmp_path: Path,
) -> None:
    store = EventStore(tmp_path / "pcltm.db")
    try:
        event_id = store.append_event(
            session_id="natural-semantic", conversation_id="natural-semantic",
            platform="test", role="user", source="chat",
            content="回答直接一点就好，别铺垫太多。", persona_mode="work",
        )
        event = store.get_event(event_id)
        assert event["inject_policy"] == "candidate_only"

        candidates = PersonaCandidateExtractor(store).extract(
            scope={"session_id": "natural-semantic"},
        )
        assert len(candidates) == 1
        candidate = candidates[0]
        assert candidate["semantic_score"] > candidate["lexical_score"]
        assert candidate["requires_human_confirmation"] is False
        assert candidate["confidence"] >= 0.85

        report = CandidatePromotionService(store).promote(candidates)
        assert report.pending == 0
        assert report.activated == 1
        assert store._conn.execute(
            "SELECT COUNT(*) FROM memory_current WHERE lifecycle_state='active'"
        ).fetchone()[0] == 1
    finally:
        store.close()


def test_bare_repository_cleanliness_preference_enters_active_claim(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "pcltm.db")
    try:
        event_id = store.append_event(
            session_id="repository-cleanliness",
            conversation_id="repository-cleanliness",
            platform="desktop",
            role="user",
            source="hermes_state_db",
            content="我喜欢干净整洁的代码仓库。",
            persona_mode="work",
        )
        event = store.get_event(event_id)
        assert event["inject_policy"] == "candidate_only"

        candidates = PersonaCandidateExtractor(store).extract(
            scope={"session_id": "repository-cleanliness"},
        )
        assert len(candidates) == 1
        candidate = candidates[0]
        assert candidate["semantic_key"] == "preference:engineering:repository-cleanliness"
        assert candidate["requires_human_confirmation"] is False

        report = CandidatePromotionService(store).promote(candidates)
        assert report.pending == 0
        assert report.activated == 1
        assert len(store.list_candidate_queue(status="pending")) == 0
        current = store._conn.execute(
            """SELECT c.canonical_key, mc.lifecycle_state
               FROM memory_current mc
               JOIN memory_claims c ON c.claim_id=mc.claim_id"""
        ).fetchone()
        assert current["canonical_key"].endswith(
            "preference:engineering:repository-cleanliness"
        )
        assert current["lifecycle_state"] == "active"
    finally:
        store.close()


@pytest.mark.parametrize(
    "content",
    [
        "我今天喜欢干净整洁的代码仓库。",
        "我可能喜欢干净整洁的代码仓库。",
        "我朋友喜欢干净整洁的代码仓库。",
        "我不是说我喜欢干净整洁的代码仓库。",
        "用户说：我喜欢干净整洁的代码仓库。",
    ],
)
def test_repository_cleanliness_rule_rejects_transient_uncertain_or_non_self_claims(
    content: str,
) -> None:
    assert parse_stable_memory_assertion(content) is None


@pytest.mark.parametrize(
    "content",
    [
        "我的名字是 Alice。",
        "我向来更喜欢深色界面。",
        "默认所有报告都必须附带验证证据。",
    ],
)
def test_explicitly_stable_natural_assertions_auto_activate(
    tmp_path: Path, content: str,
) -> None:
    store = EventStore(tmp_path / "pcltm.db")
    try:
        store.append_event(
            session_id="stable", conversation_id="stable", platform="test",
            role="user", source="chat", content=content, persona_mode="work",
        )
        candidate = PersonaCandidateExtractor(store).extract(
            scope={"session_id": "stable"},
        )[0]

        assert candidate["admission_tier"] == "auto_activate"
        assert candidate["requires_human_confirmation"] is False
        report = CandidatePromotionService(store).promote([candidate])
        assert (report.activated, report.pending) == (1, 0)
    finally:
        store.close()


def test_unmarked_preference_activates_after_independent_semantic_support(
    tmp_path: Path,
) -> None:
    store = EventStore(tmp_path / "pcltm.db")
    try:
        store.append_event(
            session_id="dark-1", conversation_id="dark-1", platform="test",
            role="user", source="chat", content="我喜欢深色界面。", persona_mode="daily",
        )
        first = PersonaCandidateExtractor(store).extract(scope={"session_id": "dark-1"})[0]
        assert first["admission_tier"] == "pending_review"

        store.append_event(
            session_id="dark-2", conversation_id="dark-2", platform="test",
            role="user", source="chat", content="我更喜欢深色界面。", persona_mode="daily",
        )
        second = PersonaCandidateExtractor(store).extract(scope={"session_id": "dark-2"})[0]
        assert second["semantic_key"] == first["semantic_key"]
        assert second["independent_session_count"] == 2
        assert second["admission_tier"] == "auto_activate"
        assert CandidatePromotionService(store).promote([second]).activated == 1
    finally:
        store.close()
