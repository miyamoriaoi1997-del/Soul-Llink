"""Active rule + semantic fusion regressions for Issue #1."""

import math

import pytest

from persona_orchestrator import StateOrchestrator
from persona_orchestrator.semantic_classifier import SemanticModeClassifier
from persona_orchestrator.types import ModeDecision
from persona_orchestrator.transition_manager_v2 import TransitionManagerV2


class _SemanticStub:
    def __init__(self, result=None, error: Exception | None = None):
        self.result = result or {}
        self.error = error

    def classify(self, **kwargs):
        if self.error:
            raise self.error
        return dict(self.result)


def _semantic_daily(confidence: float = 0.94):
    return {
        "primary_mode": "daily",
        "submode": "relationship_closeness",
        "confidence": confidence,
        "safety_flags": [],
        "intent_signals": {
            "explicit_daily_intent": True,
            "continuation_of_previous_task": False,
            "technical_context": False,
        },
        "reason_codes": ["SEMANTIC_RELATIONSHIP_BONDING"],
        "backend": "test-semantic",
        "shadow_only": False,
    }


def test_local_semantic_disambiguates_maintenance_by_object():
    classifier = SemanticModeClassifier()

    relationship = classifier.classify("我们可是亲密战友啊，平时当然要和你闲聊维护感情——战友情")
    technical = classifier.classify("请维护生产数据库并检查服务状态")

    assert relationship["primary_mode"] == "daily"
    assert relationship["confidence"] >= 0.80
    assert relationship["intent"] == "relationship_bonding"
    assert relationship["intent_signals"]["technical_context"] is False
    assert technical["primary_mode"] == "work"
    assert technical["intent_signals"]["technical_context"] is True


def test_bounded_semantic_authority_exits_sticky_work_context(tmp_path):
    orchestrator = StateOrchestrator(
        ".",
        log_path=tmp_path / "fusion.jsonl",
        enable_semantic_shadow=True,
        enable_semantic_authority=True,
    )

    packet = orchestrator.analyze_turn(
        user_message="我们可是亲密战友啊，平时当然要和你闲聊维护感情——战友情",
        previous_mode="work",
        runtime_authority="active",
    )

    assert packet.mode == "daily"
    assert packet.transition == "work->daily"
    assert packet.route_metadata["decision_audit"]["semantic_fusion"]["authority"] == "semantic"


def test_bounded_semantic_authority_exits_plain_chat_with_production_transition_table(tmp_path):
    orchestrator = StateOrchestrator(
        ".",
        log_path=tmp_path / "fusion.jsonl",
        enable_semantic_shadow=True,
        enable_semantic_authority=True,
    )
    orchestrator.transitions = TransitionManagerV2(
        enable_shadow_table=True,
        enable_bounded_activation=True,
    )

    packet = orchestrator.analyze_turn(
        user_message="你喜欢吃什么",
        previous_mode="work",
        runtime_authority="active",
    )

    assert packet.mode == "daily"
    assert packet.transition == "work->daily"


def test_semantic_daily_intent_can_release_plain_chat_from_work(tmp_path):
    orchestrator = StateOrchestrator(
        ".",
        log_path=tmp_path / "fusion.jsonl",
        enable_semantic_shadow=True,
        enable_semantic_authority=True,
    )

    packet = orchestrator.analyze_turn(
        user_message="你喜欢吃什么",
        previous_mode="work",
        runtime_authority="active",
    )

    assert packet.mode == "daily"
    assert packet.transition == "work->daily"


def test_explicit_technical_rule_is_not_overridden_by_daily_semantic_candidate(tmp_path):
    orchestrator = StateOrchestrator(
        ".",
        log_path=tmp_path / "fusion.jsonl",
        enable_semantic_shadow=True,
        enable_semantic_authority=True,
    )
    orchestrator.semantic_classifier = _SemanticStub(_semantic_daily())

    packet = orchestrator.analyze_turn(
        user_message="请运行 pytest 修复这个 bug",
        previous_mode="daily",
        runtime_authority="active",
    )

    assert packet.mode == "work"
    fusion = packet.route_metadata["decision_audit"]["semantic_fusion"]
    assert fusion["authority"] == "rules"
    assert fusion["reason"] == "hard_rule_preserved"


def test_semantic_failure_falls_back_to_rules_without_blocking(tmp_path):
    orchestrator = StateOrchestrator(
        ".",
        log_path=tmp_path / "fusion.jsonl",
        enable_semantic_shadow=True,
        enable_semantic_authority=True,
    )
    orchestrator.semantic_classifier = _SemanticStub(error=TimeoutError("semantic timeout"))

    packet = orchestrator.analyze_turn(
        user_message="请运行 pytest",
        previous_mode="daily",
        runtime_authority="active",
    )

    assert packet.mode == "work"
    fusion = packet.route_metadata["decision_audit"]["semantic_fusion"]
    assert fusion["authority"] == "rules"
    assert fusion["reason"] == "semantic_error_fallback"


def test_semantic_safety_flags_are_preserved_and_block_mode_override(tmp_path):
    orchestrator = StateOrchestrator(
        ".",
        log_path=tmp_path / "fusion.jsonl",
        enable_semantic_shadow=True,
        enable_semantic_authority=True,
    )
    semantic = _semantic_daily(confidence=0.99)
    semantic["safety_flags"] = ["crisis_guard"]
    rule_decision = ModeDecision(
        mode="work",
        submode="default",
        confidence=0.72,
        reason="rule_work_context",
        safety_flags=[],
        signals={},
    )
    fused, fusion = orchestrator._fuse_semantic_decision(rule_decision, semantic)

    assert fused.mode == "work"
    assert "crisis_guard" in fused.safety_flags
    assert fusion["authority"] == "rules"
    assert fusion["reason"] == "hard_rule_preserved"


@pytest.mark.parametrize(
    "invalid_confidence",
    [True, False, "0.99", None, math.nan, math.inf, -math.inf, -0.01, 1.01],
)
def test_semantic_authority_rejects_invalid_confidence_types_and_ranges(
    tmp_path,
    invalid_confidence,
):
    orchestrator = StateOrchestrator(
        ".",
        log_path=tmp_path / "fusion.jsonl",
        enable_semantic_shadow=True,
        enable_semantic_authority=True,
    )
    semantic = _semantic_daily(confidence=invalid_confidence)
    rule_decision = ModeDecision(
        mode="work",
        submode="default",
        confidence=0.72,
        reason="rule_work_context",
        safety_flags=[],
        signals={},
    )

    fused, fusion = orchestrator._fuse_semantic_decision(rule_decision, semantic)

    assert fused is rule_decision
    assert fusion["authority"] == "rules"
    assert fusion["reason"] == "invalid_semantic_result"


@pytest.mark.parametrize("valid_confidence", [0, 0.0, 0.8, 1, 1.0])
def test_semantic_authority_accepts_only_finite_numeric_confidence(
    tmp_path,
    valid_confidence,
):
    orchestrator = StateOrchestrator(
        ".",
        log_path=tmp_path / "fusion.jsonl",
        enable_semantic_shadow=True,
        enable_semantic_authority=True,
    )
    semantic = _semantic_daily(confidence=valid_confidence)
    rule_decision = ModeDecision(
        mode="daily",
        submode="default",
        confidence=0.72,
        reason="rule_daily_context",
        safety_flags=[],
        signals={},
    )

    fused, fusion = orchestrator._fuse_semantic_decision(rule_decision, semantic)

    if float(valid_confidence) < 0.80:
        assert fused is rule_decision
        assert fusion["authority"] == "rules"
        assert fusion["reason"] == "semantic_confidence_below_threshold"
    else:
        assert fused.confidence == float(valid_confidence)
        assert fusion["authority"] == "semantic"
        assert fusion["reason"] == "bounded_high_confidence"
