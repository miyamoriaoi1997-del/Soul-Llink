from __future__ import annotations

from pcltm.injection import (
    ArbitrationInput,
    CandidateType,
    InjectionArbitrator,
    InjectionCandidate,
    RiskFlag,
)


def _candidate(
    key: str,
    candidate_type: CandidateType,
    content: str | None = None,
    *,
    source: str = "runtime",
    confidence: float | None = 0.9,
    freshness: float = 0.8,
    relevance: float = 0.8,
    token_cost: int = 10,
    risk_flags: tuple[RiskFlag, ...] = (),
    metadata: dict[str, object] | None = None,
) -> InjectionCandidate:
    return InjectionCandidate(
        key=key,
        type=candidate_type,
        source=source,
        content=content or f"content for {key}",
        confidence=confidence,
        freshness=freshness,
        relevance=relevance,
        token_cost=token_cost,
        risk_flags=risk_flags,
        metadata=metadata or {},
    )


def test_long_term_memory_cannot_displace_active_dialogue_state() -> None:
    candidates = [
        _candidate("core", CandidateType.CORE_IDENTITY, token_cost=5),
        _candidate("ads", CandidateType.ACTIVE_DIALOGUE, "continue current Q2", token_cost=18),
        *[
            _candidate(
                f"semantic-{index}",
                CandidateType.SEMANTIC_MEMORY,
                source="semantic-store",
                token_cost=18,
                relevance=1.0,
            )
            for index in range(10)
        ],
    ]

    packet = InjectionArbitrator(total_budget=120).arbitrate(candidates)

    assert "continue current Q2" in packet.render()
    assert CandidateType.ACTIVE_DIALOGUE in packet.sections
    assert len(packet.sections.get(CandidateType.SEMANTIC_MEMORY, ())) < 10
    assert packet.audit.budget["bucket_limits"]["active_dialogue"] > 0


def test_core_soul_conflict_is_rejected_and_anchor_survives() -> None:
    candidates = [
        _candidate("core", CandidateType.CORE_IDENTITY, "I am Rio; teacher is the addressee", token_cost=10),
        _candidate(
            "bad-memory",
            CandidateType.SEMANTIC_MEMORY,
            "ordinary memory says identity is someone else",
            source="semantic-store",
            risk_flags=(RiskFlag.CORE_CONFLICT,),
            token_cost=8,
        ),
    ]

    packet = InjectionArbitrator(total_budget=80).arbitrate(candidates)

    rendered = packet.render()
    assert "I am Rio" in rendered
    assert "someone else" not in rendered
    assert any(item["reason"] == "conflicts-with-core-soul" for item in packet.audit.rejected)


def test_emotion_state_cannot_contaminate_fact_layer() -> None:
    candidates = [
        _candidate("emotion", CandidateType.PERSONA_EMOTION, "affection is warm", token_cost=8),
        _candidate(
            "emotion-as-fact",
            CandidateType.SEMANTIC_MEMORY,
            "emotion says permanent factual change",
            source="emotion-layer",
            risk_flags=(RiskFlag.EMOTION_FACT_CONTAMINATION,),
            token_cost=8,
        ),
    ]

    packet = InjectionArbitrator(total_budget=120).arbitrate(candidates)

    assert "affection is warm" in packet.render()
    assert "permanent factual change" not in packet.render()
    assert any(item["reason"] == "emotion-layer-cannot-write-facts" for item in packet.audit.rejected)


def test_semantic_memory_requires_source_or_confidence() -> None:
    packet = InjectionArbitrator(total_budget=160).arbitrate(
        [
            _candidate(
                "sourced-fact",
                CandidateType.SEMANTIC_MEMORY,
                "sourced semantic claim",
                source="semantic-store",
                confidence=None,
                token_cost=8,
            ),
            _candidate(
                "confidence-fact",
                CandidateType.SEMANTIC_MEMORY,
                "confidence-only semantic claim",
                source="",
                confidence=0.7,
                token_cost=8,
            ),
            _candidate(
                "unsourced-fact",
                CandidateType.SEMANTIC_MEMORY,
                "unsupported semantic claim",
                source="",
                confidence=None,
                token_cost=8,
            ),
        ]
    )

    rendered = packet.render()
    assert "sourced semantic claim" in rendered
    assert "confidence-only semantic claim" in rendered
    assert "unsupported semantic claim" not in rendered
    assert any(item["reason"] == "semantic-memory-needs-source-or-confidence" for item in packet.audit.rejected)


def test_episodic_memory_must_be_marked_as_event() -> None:
    packet = InjectionArbitrator(total_budget=100).arbitrate(
        [
            _candidate(
                "event-ok",
                CandidateType.EPISODIC_MEMORY,
                "On Tuesday the user approved phase 6.",
                source="episodic-store",
                metadata={"event": True},
                token_cost=10,
            ),
            _candidate(
                "event-bad",
                CandidateType.EPISODIC_MEMORY,
                "The user always wants this forever.",
                source="episodic-store",
                metadata={"event": False},
                token_cost=10,
            ),
        ]
    )

    rendered = packet.render()
    assert "On Tuesday" in rendered
    assert "always wants" not in rendered
    assert "event:event-ok" in rendered
    assert any(item["reason"] == "episodic-memory-must-be-event" for item in packet.audit.rejected)


def test_procedural_memory_only_injected_when_needed() -> None:
    packet = InjectionArbitrator(total_budget=120).arbitrate(
        [
            _candidate(
                "needed-skill",
                CandidateType.PROCEDURAL_SKILL,
                "Use the PCLTM release checklist.",
                source="skill-store",
                metadata={"needed": True},
                token_cost=8,
            ),
            _candidate(
                "idle-skill",
                CandidateType.PROCEDURAL_SKILL,
                "Unrelated drawing workflow.",
                source="skill-store",
                metadata={"needed": False},
                token_cost=8,
            ),
        ]
    )

    assert "release checklist" in packet.render()
    assert "drawing workflow" not in packet.render()
    assert any(item["reason"] == "procedural-memory-only-on-demand" for item in packet.audit.rejected)


def test_conflicting_memory_is_not_injected_unless_resolution_requested() -> None:
    conflict = _candidate(
        "conflict",
        CandidateType.SEMANTIC_MEMORY,
        "old fact conflicts with the current task",
        source="semantic-store",
        risk_flags=(RiskFlag.CURRENT_TASK_CONFLICT,),
        token_cost=8,
    )

    default_packet = InjectionArbitrator(total_budget=80).arbitrate([conflict])
    resolution_packet = InjectionArbitrator(total_budget=80).arbitrate(
        [conflict],
        allow_conflict_resolution=True,
    )

    assert "old fact" not in default_packet.render()
    assert any(item["reason"] == "conflict-not-requested" for item in default_packet.audit.rejected)
    assert "old fact" in resolution_packet.render()
    assert any(
        decision["reason"] == "conflict-included-for-resolution"
        for decision in resolution_packet.audit.conflict_decisions
    )


def test_packet_is_sectioned_and_auditable() -> None:
    candidates = [
        _candidate("system", CandidateType.SYSTEM, "tool discipline", token_cost=5),
        _candidate("core", CandidateType.CORE_IDENTITY, "core identity", token_cost=5),
        _candidate("mode", CandidateType.RUNTIME_MODE, "work mode", token_cost=5),
        _candidate("ads", CandidateType.ACTIVE_DIALOGUE, "active turn", token_cost=5),
        _candidate("task", CandidateType.CURRENT_TASK, "current task", token_cost=5),
        _candidate("persona", CandidateType.PERSONA_EMOTION, "emotion state", token_cost=5),
        _candidate("spine", CandidateType.SESSION_SPINE, "session spine", token_cost=5),
        _candidate(
            "episode",
            CandidateType.EPISODIC_MEMORY,
            "event memory",
            source="episodic-store",
            metadata={"event": True},
            token_cost=5,
        ),
        _candidate("fact", CandidateType.SEMANTIC_MEMORY, "semantic fact", source="semantic-store", token_cost=5),
        _candidate(
            "skill",
            CandidateType.PROCEDURAL_SKILL,
            "needed skill",
            source="skill-store",
            metadata={"needed": True},
            token_cost=5,
        ),
    ]

    packet = InjectionArbitrator(total_budget=200).arbitrate(candidates)
    rendered = packet.render()

    assert rendered.index("Core SOUL Identity Anchor") < rendered.index("Active Dialogue State")
    assert rendered.index("Active Dialogue State") < rendered.index("Relevant Semantic Memories")
    assert all(entry["reason"] == "selected-within-fixed-layer-budget" for entry in packet.audit.injected)
    assert packet.to_dict()["rendered"] == rendered


def test_arbitration_deduplicates_repeated_current_task_instructions() -> None:
    repeated_instruction = "现在出现问题，新架构PCLTM会重复注入指令，导致任务无法推进。"
    candidates = [
        _candidate("task-current", CandidateType.CURRENT_TASK, repeated_instruction, source="runtime", token_cost=8),
        _candidate("task-replayed", CandidateType.CURRENT_TASK, repeated_instruction, source="pcltm", token_cost=8),
        _candidate("dialogue", CandidateType.ACTIVE_DIALOGUE, "continue implementation", token_cost=8),
    ]

    packet = InjectionArbitrator(total_budget=100).arbitrate(candidates)
    rendered = packet.render()

    assert rendered.count(repeated_instruction) == 1
    assert [entry["key"] for entry in packet.audit.injected].count("task-current") == 1
    assert any(
        entry["key"] == "task-replayed" and entry["reason"] == "duplicate-injection-fingerprint"
        for entry in packet.audit.rejected
    )


def test_arbitration_decision_can_be_replayed() -> None:
    candidates = [
        _candidate("ads", CandidateType.ACTIVE_DIALOGUE, "active turn", token_cost=10),
        _candidate("fact", CandidateType.SEMANTIC_MEMORY, "semantic fact", source="semantic-store", token_cost=10),
    ]
    arbitrator = InjectionArbitrator(total_budget=100)
    original = arbitrator.arbitrate(candidates, metadata={"turn_id": "t1"})
    replayed = arbitrator.replay(
        ArbitrationInput(
            candidates=candidates,
            total_budget=100,
            metadata={"turn_id": "t1"},
        )
    )

    assert replayed.to_dict() == original.to_dict()
