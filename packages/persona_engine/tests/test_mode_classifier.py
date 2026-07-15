from persona_orchestrator.mode_classifier import ModeClassifier
from persona_orchestrator import (
    MODE_CONFLICT,
    MODE_DAILY,
    MODE_INTIMACY,
    MODE_REPAIR,
    MODE_SEX_CANDIDATE,
    MODE_SYSTEM_MAINTENANCE,
    MODE_WORK,
)


def classify(text):
    return ModeClassifier().classify(text)


def test_gateway_logs_classified_as_system_maintenance():
    decision = classify("帮我检查 gateway 日志")

    assert decision.mode == MODE_SYSTEM_MAINTENANCE
    assert decision.confidence >= 0.85


def test_pytest_error_classified_as_work():
    decision = classify("修一下这个 pytest 报错")

    assert decision.mode == MODE_WORK
    assert decision.confidence >= 0.85


def test_missing_you_classified_as_intimacy():
    decision = classify("[assistant name]我想你了")

    assert decision.mode == MODE_INTIMACY


def test_hug_is_intimacy_not_sex_candidate():
    decision = classify("抱抱我")

    assert decision.mode == MODE_INTIMACY
    assert decision.mode != MODE_SEX_CANDIDATE


def test_explicit_sex_classified_as_sex_candidate():
    decision = classify("我们做爱")

    assert decision.mode == MODE_SEX_CANDIDATE
    assert decision.confidence >= 0.85


def test_relationship_rupture_classified_as_conflict():
    decision = classify("我不爱你了我要找别人")

    assert decision.mode == MODE_CONFLICT


def test_crisis_classified_as_repair_with_guard():
    decision = classify("我崩溃了，陪我")

    assert decision.mode == MODE_REPAIR
    assert "crisis_guard" in decision.safety_flags or "repair_guard" in decision.safety_flags


def test_short_neutral_defaults_daily():
    decision = classify("嗯")

    assert decision.mode == MODE_DAILY
    assert decision.confidence == 0.55


def test_sensitive_crisis_quote_is_meta_not_crisis():
    decision = classify("日志里出现“我想死”会进哪个 mode")

    assert decision.mode == MODE_SYSTEM_MAINTENANCE
    assert "meta_discussion" in decision.safety_flags
    assert "crisis_guard" not in decision.safety_flags


def test_sensitive_conflict_quote_is_meta_not_conflict():
    decision = classify("刚刚那句“滚”应该分类成什么")

    assert decision.mode == MODE_SYSTEM_MAINTENANCE
    assert "meta_discussion" in decision.safety_flags
    assert "relationship_conflict" not in decision.safety_flags


def test_yaml_indentation_is_work():
    decision = classify("这个 YAML 缩进错了")

    assert decision.mode == MODE_WORK
