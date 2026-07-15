from datetime import UTC, datetime, timedelta

from pcltm.memory import EpisodeExtractor, EpisodicMemory, EpisodicRetriever, EpisodicStore
from pcltm.session import SessionSegment


def test_store_writes_event_stream_and_recalls_raw_refs(tmp_path):
    store = EpisodicStore(tmp_path / "episodic.sqlite")
    memory = EpisodicMemory.create(
        source_session="session-1",
        raw_refs=("turn:1", "turn:2"),
        event_summary="用户要求继续阶段3，persona开始实现 episodic memory。",
        event_type="session_event",
        importance_score=0.8,
        continuity_relevance=0.7,
        tags=("phase3",),
    )

    store.append(memory)
    recalled = store.get(memory.event_id)

    assert recalled == memory
    assert recalled is not None
    assert recalled.raw_refs == ("turn:1", "turn:2")
    assert recalled.fact_promotion_allowed is False
    assert "episodic_record_only" in recalled.fact_promotion_blockers


def test_extractor_builds_episodes_from_session_segment_without_fact_promotion():
    segment = SessionSegment(
        segment_id=3,
        time_range=(10, 14),
        raw_message_refs=("msg-10", "msg-11", "msg-12", "msg-13"),
        local_summary="用户要求继续阶段3，目标是保存真实发生过的事件和对话片段。",
        decisions=("决定先实现事件流写入，再实现检索。",),
        commitments=("完成后提交，不急着做图谱。",),
        unresolved_items=("需要补测试和文档。",),
        emotional_delta="用户有点担心任务被重复上下文带偏。",
        memory_candidates=("用户可能不喜欢一次性情绪被写成长期事实。",),
    )

    episodes = EpisodeExtractor().extract_from_segment(
        segment,
        source_session="session-phase-3",
        timestamp="2026-06-17T00:00:00+00:00",
    )

    assert {episode.event_type for episode in episodes} >= {
        "segment_summary",
        "decision",
        "commitment",
        "unresolved_item",
        "emotional_observation",
        "memory_candidate_observation",
    }
    assert all(episode.source_session == "session-phase-3" for episode in episodes)
    assert all(episode.raw_refs == segment.raw_message_refs for episode in episodes)
    assert all(episode.fact_promotion_allowed is False for episode in episodes)

    emotional = next(episode for episode in episodes if episode.event_type == "emotional_observation")
    assert emotional.emotional_salience > 0
    assert "emotional_fragment_not_stable_personality" in emotional.fact_promotion_blockers

    candidate = next(episode for episode in episodes if episode.event_type == "memory_candidate_observation")
    assert candidate.confidence_score < 0.5
    assert "low_confidence_fragment" in candidate.fact_promotion_blockers
    assert "single_event_not_permanent_preference" in candidate.fact_promotion_blockers


def test_retriever_scores_relevance_recency_importance_and_current_task(tmp_path):
    store = EpisodicStore(tmp_path / "episodic.sqlite")
    now = datetime(2026, 6, 17, tzinfo=UTC)
    older = EpisodicMemory.create(
        source_session="session-old",
        raw_refs=("old:1",),
        event_summary="讨论无关的旧任务。",
        event_type="session_event",
        timestamp=(now - timedelta(days=90)).isoformat(),
        importance_score=0.2,
        continuity_relevance=0.1,
        tags=("unrelated",),
    )
    relevant = EpisodicMemory.create(
        source_session="session-new",
        raw_refs=("new:1", "new:2"),
        event_summary="阶段3 episodic memory 保存 raw refs 并实现检索。",
        event_type="decision",
        timestamp=(now - timedelta(days=1)).isoformat(),
        importance_score=0.9,
        continuity_relevance=0.8,
        tags=("phase3", "episodic"),
    )
    task_related = EpisodicMemory.create(
        source_session="session-task",
        raw_refs=("task:1",),
        event_summary="当前任务需要补文档和测试。",
        event_type="unresolved_item",
        timestamp=(now - timedelta(days=3)).isoformat(),
        importance_score=0.5,
        continuity_relevance=0.5,
        tags=("tests", "docs"),
    )
    store.append_many((older, relevant, task_related))

    results = EpisodicRetriever(store).retrieve(
        "episodic memory raw refs",
        current_task="补测试 docs",
        now=now,
        limit=3,
    )

    assert results[0].memory.event_id == relevant.event_id
    assert any(result.memory.event_id == task_related.event_id for result in results)
    assert results[0].score > results[-1].score
    assert results[0].to_candidate_dict()["injection_note"] == "candidate_only_for_injection_arbitration"


def test_answer_what_happened_keeps_event_and_conclusion_separate(tmp_path):
    store = EpisodicStore(tmp_path / "episodic.sqlite")
    memory = EpisodicMemory.create(
        source_session="session-2",
        raw_refs=("turn:9",),
        event_summary="用户要求继续阶段3，并明确这层只记录发生过什么。",
        event_type="session_event",
        timestamp="2026-06-17T00:00:00+00:00",
        importance_score=0.8,
        continuity_relevance=0.8,
        tags=("phase3", "episodic"),
    )
    store.append(memory)

    answer = EpisodicRetriever(store).answer_what_happened(
        "阶段3 发生过什么",
        now=datetime(2026, 6, 17, tzinfo=UTC),
    )

    assert "用户要求继续阶段3" in answer
    assert "raw_refs=turn:9" in answer
    assert "不是长期事实或人格结论" in answer
