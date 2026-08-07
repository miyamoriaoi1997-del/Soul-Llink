"""Contracts for active-vs-shadow orchestration audit semantics."""

import json

from persona_orchestrator import StateOrchestrator


def test_analyze_turn_used_by_active_provider_logs_active_observation_semantics(tmp_path):
    log = tmp_path / "runtime.jsonl"
    packet = StateOrchestrator(".", log_path=log).analyze_turn(
        user_message="检查生产状态",
        emotion_state={"emotion_score": 1.0},
        previous_mode="daily",
        runtime_authority="active",
    )

    row = json.loads(log.read_text(encoding="utf-8").splitlines()[-1])
    assert packet.shadow_only is False
    assert row["packet"]["shadow_only"] is False
    assert row["extra"]["runtime_authority"] == "active"


def test_default_analysis_remains_shadow_only_for_offline_evaluation(tmp_path):
    log = tmp_path / "shadow.jsonl"
    packet = StateOrchestrator(".", log_path=log).analyze_turn(
        user_message="检查生产状态",
        emotion_state={"emotion_score": 1.0},
    )

    row = json.loads(log.read_text(encoding="utf-8").splitlines()[-1])
    assert packet.shadow_only is True
    assert row["extra"]["runtime_authority"] == "shadow"
