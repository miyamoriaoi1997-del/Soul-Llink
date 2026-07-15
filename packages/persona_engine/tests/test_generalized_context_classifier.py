"""Regression tests for generalized intent context in the deterministic mode classifier."""

from persona_orchestrator.mode_classifier import ModeClassifier
from persona_orchestrator.transition_manager import TransitionManager
from persona_orchestrator.types import MODE_SYSTEM_MAINTENANCE, MODE_WORK


def classify(message: str, previous_mode: str | None = None):
    decision = ModeClassifier().classify(message)
    transition = TransitionManager().transition(previous_mode, decision, desire_tier="restrained", enable_active_sex=True)
    return decision, transition


def test_financial_strategy_markers_generalize_beyond_specific_quant_words():
    decision, transition = classify("帮我把套利模型的收益回撤和资金曲线一起优化一下", previous_mode="daily")

    assert decision.mode == MODE_WORK
    assert transition.active_mode == MODE_WORK


def test_practical_factual_question_uses_work_mode_without_domain_keyword():
    decision, transition = classify("飞机上能不能带充电宝")

    assert decision.mode == MODE_WORK
    assert transition.active_mode == MODE_WORK



def test_restore_missing_task_generalizes_to_system_maintenance():
    decision, transition = classify("刚才那个任务丢了，帮我恢复一下", previous_mode=MODE_WORK)

    assert decision.mode == MODE_SYSTEM_MAINTENANCE
    assert transition.active_mode == MODE_SYSTEM_MAINTENANCE


def test_polite_continuation_inherits_technical_context_without_exact_phrase():
    decision, transition = classify("嗯，接着来", previous_mode=MODE_SYSTEM_MAINTENANCE)

    assert decision.mode == "daily"
    assert transition.active_mode == MODE_SYSTEM_MAINTENANCE
    assert transition.transition == "hold_short_message"
