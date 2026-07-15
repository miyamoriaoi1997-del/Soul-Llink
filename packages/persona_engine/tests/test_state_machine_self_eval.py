from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.state_machine_self_eval import (
    apply_audit_gate,
    apply_feedback_labels,
    append_manual_feedback,
    build_shadow_report,
    extract_audit_candidates,
    load_jsonl,
)


def test_human_truth_report_is_read_only_and_blocks_on_any_regression(tmp_path: Path) -> None:
    cases = [
        {
            "id": "ok",
            "message": "检查状态机日志",
            "previous_mode": "daily",
            "emotion_score": 1.0,
            "expected_mode": "work",
            "expected_transition": "daily->work",
            "expected_layers": ["work"],
            "forbidden_layers": ["sex"],
            "source": "human_labeled_real_chat",
        },
        {
            "id": "bad",
            "message": "我想你",
            "previous_mode": "daily",
            "emotion_score": 1.0,
            "expected_mode": "work",
            "expected_transition": "stay:work",
            "expected_layers": ["work"],
            "source": "human_labeled_real_chat",
        },
    ]

    report = build_shadow_report(cases)

    assert report["authority"] == "read_only_shadow_evaluation"
    assert report["dataset_kind"] == "human_truth"
    assert report["promotion"]["status"] == "blocked"
    assert report["promotion"]["automatic_rule_write"] is False
    assert report["metrics"]["total"] == 2
    assert report["metrics"]["failed"] == 1
    assert report["failures"][0]["id"] == "bad"
    assert "confusion_matrix" in report["metrics"]
    assert "high_risk_false_positive_count" in report["metrics"]


def test_audit_candidate_extraction_flags_disagreement_and_mode_jump(tmp_path: Path) -> None:
    log = tmp_path / "audit.jsonl"
    rows = [
        {
            "timestamp": "2026-07-15T09:37:45",
            "packet": {
                "mode": "sex",
                "transition": "work->sex",
                "confidence": 0.82,
                "semantic_shadow": {"primary_mode": "daily", "confidence": 0.70},
                "route_metadata": {
                    "decision_audit": {
                        "classifier": {
                            "mode": "sex",
                            "reason": "ambiguous hint",
                            "signals": {"normalized_text": "opaque-sample"},
                        },
                        "context_router": {"top_mode": "relationship"},
                    }
                },
            },
        }
    ]
    log.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    candidates = extract_audit_candidates(load_jsonl(log))

    assert len(candidates) == 1
    assert candidates[0]["needs_manual_label"] is True
    assert set(candidates[0]["reason_codes"]) >= {
        "high_risk_mode_jump",
        "semantic_disagreement",
    }
    assert "expected_mode" not in candidates[0]


def test_clean_human_truth_is_only_a_manual_review_candidate() -> None:
    cases = [
        {
            "id": "work",
            "message": "检查状态机日志",
            "previous_mode": "daily",
            "emotion_score": 1.0,
            "expected_mode": "work",
            "expected_transition": "daily->work",
            "expected_layers": ["work"],
            "forbidden_layers": ["sex"],
            "source": "human_labeled_real_chat",
        }
    ]

    report = build_shadow_report(cases)

    assert report["metrics"]["failed"] == 0
    assert report["promotion"]["status"] == "manual_review_candidate"
    assert report["promotion"]["automatic_rule_write"] is False
    assert report["promotion"]["automatic_activation"] is False


def test_unresolved_runtime_audit_candidates_block_promotion() -> None:
    report = {
        "promotion": {"status": "manual_review_candidate"},
        "metrics": {"failed": 0},
    }
    candidates = [
        {"reason_codes": ["possible_host_continuity_reset"], "needs_manual_label": True},
        {"reason_codes": ["high_risk_mode_jump"], "needs_manual_label": True},
    ]

    gated = apply_audit_gate(report, candidates)

    assert gated["promotion"]["status"] == "blocked"
    assert gated["audit_health"]["unresolved_candidate_count"] == 2
    assert gated["audit_health"]["possible_host_continuity_reset_count"] == 1
    assert gated["audit_health"]["high_risk_mode_jump_count"] == 1


def test_manual_feedback_append_is_idempotent_and_never_writes_rules(tmp_path: Path) -> None:
    feedback = tmp_path / "feedback.jsonl"
    label = {
        "candidate_id": "audit-line-60",
        "message": "opaque-sample",
        "expected_mode": "work",
        "expected_transition": "hold_context_action",
        "reviewer": "human",
    }

    assert append_manual_feedback(feedback, label) is True
    assert append_manual_feedback(feedback, label) is False
    rows = load_jsonl(feedback)
    assert rows == [{**label, "source": "manual_feedback"}]
    assert not (tmp_path / "routing_rules.yaml").exists()


def test_continuity_reset_requires_same_session_provenance() -> None:
    records = [
        {"timestamp": "t1", "extra": {}, "packet": {"mode": "work", "transition": "start:work"}},
        {"timestamp": "t2", "extra": {}, "packet": {"mode": "daily", "transition": "start:daily"}},
        {"timestamp": "t3", "extra": {"session_id": "s1", "turn_number": 1}, "packet": {"mode": "work", "transition": "start:work"}},
        {"timestamp": "t4", "extra": {"session_id": "s1", "turn_number": 2}, "packet": {"mode": "daily", "transition": "start:daily"}},
    ]

    candidates = extract_audit_candidates(records)

    unknown = [c for c in candidates if "continuity_unknown" in c["reason_codes"]]
    resets = [c for c in candidates if "possible_host_continuity_reset" in c["reason_codes"]]
    assert len(unknown) == 2
    assert len(resets) == 1
    assert resets[0]["session_id"] == "s1"
    assert resets[0]["turn_number"] == 2


def test_manual_feedback_resolves_exact_candidate_id_only() -> None:
    candidates = [
        {"candidate_id": "a", "needs_manual_label": True, "reason_codes": ["high_risk_mode_jump"]},
        {"candidate_id": "b", "needs_manual_label": True, "reason_codes": ["continuity_unknown"]},
    ]
    feedback = [{"candidate_id": "a", "expected_mode": "work", "expected_transition": "hold_context_action", "reviewer": "human", "source": "manual_feedback"}]

    resolved = apply_feedback_labels(candidates, feedback)

    assert resolved[0]["needs_manual_label"] is False
    assert resolved[0]["manual_label"]["expected_mode"] == "work"
    assert resolved[1]["needs_manual_label"] is True


def test_untrusted_feedback_source_cannot_resolve_candidate() -> None:
    candidates = [{"candidate_id": "a", "needs_manual_label": True, "reason_codes": []}]
    feedback = [{"candidate_id": "a", "expected_mode": "work", "reviewer": "claimed-human"}]

    resolved = apply_feedback_labels(candidates, feedback)

    assert resolved[0]["needs_manual_label"] is True


def test_rapid_same_session_mode_reversal_is_a_flapping_candidate() -> None:
    records = [
        {"timestamp": "t1", "extra": {"session_id": "s1", "turn_number": 1}, "packet": {"mode": "work", "transition": "start:work"}},
        {"timestamp": "t2", "extra": {"session_id": "s1", "turn_number": 2}, "packet": {"mode": "daily", "transition": "work->daily"}},
        {"timestamp": "t3", "extra": {"session_id": "s1", "turn_number": 3}, "packet": {"mode": "work", "transition": "daily->work"}},
    ]

    candidates = extract_audit_candidates(records)

    flapping = [c for c in candidates if "rapid_mode_reversal" in c["reason_codes"]]
    assert len(flapping) == 1
    assert flapping[0]["turn_number"] == 3


def test_audit_gate_ignores_resolved_feedback_candidates() -> None:
    report = {"promotion": {"status": "manual_review_candidate"}, "metrics": {"failed": 0}}
    candidates = [
        {
            "candidate_id": "a",
            "needs_manual_label": False,
            "manual_label": {"expected_mode": "work", "reviewer": "human"},
            "reason_codes": ["high_risk_mode_jump"],
        }
    ]

    gated = apply_audit_gate(report, candidates)

    assert gated["promotion"]["status"] == "manual_review_candidate"
    assert gated["audit_health"]["unresolved_candidate_count"] == 0


def test_runtime_derived_snapshot_cannot_masquerade_as_human_truth() -> None:
    cases = [
        {
            "id": "self-labeled",
            "message": "opaque",
            "previous_mode": "work",
            "expected_mode": "work",
            "expected_transition": "stay:work",
            "expected_layers": ["work"],
            "source": "runtime_generator",
            "dataset_kind": "runtime_derived_snapshot",
            "accuracy_authority": False,
        }
    ]

    try:
        build_shadow_report(cases)
    except ValueError as error:
        assert "human-labeled" in str(error)
    else:
        raise AssertionError("runtime-derived snapshots must not become accuracy truth")


def test_cli_invalid_truth_replaces_stale_candidate_report_with_blocked_sentinel(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"promotion": {"status": "manual_review_candidate"}}), encoding="utf-8")
    invalid_truth = tmp_path / "invalid.jsonl"
    invalid_truth.write_text("not-json\n", encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "scripts" / "state_machine_self_eval.py"

    result = subprocess.run(
        [sys.executable, str(script), "--truth", str(invalid_truth), "--report", str(report)],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    current = json.loads(report.read_text(encoding="utf-8"))
    assert current["promotion"]["status"] == "blocked"
    assert current["promotion"]["blocked_by"] == "evaluation_in_progress"
