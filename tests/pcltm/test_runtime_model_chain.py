"""Contract tests for one correlated turn-to-forwarded-model chain."""

from __future__ import annotations

import json

from pcltm.monitoring.private_data import collect_runtime_turn_capture


def test_runtime_capture_joins_actual_router_model_by_turn_correlation_id(tmp_path):
    capture_path = tmp_path / "latest-turn.json"
    router_path = tmp_path / "audit.jsonl"
    capture_path.write_text(json.dumps({
        "source": "exact_host_capture",
        "turn_correlation_id": "turn-abc",
        "state_machine": {
            "mode": "work",
            "selected_layers": ["core", "work"],
            "route_metadata": {"hermes_route_bucket": "task"},
        },
        "soul_mode_layer": {"mode": "work", "content": "# Work"},
    }), encoding="utf-8")
    router_path.write_text("\n".join([
        json.dumps({"turn_correlation_id": "other", "selected_model": "other-model", "forwarded_model": "other-model"}),
        json.dumps({"turn_correlation_id": "turn-abc", "selected_model": "work-model", "forwarded_model": "work-model", "ok": True}),
    ]) + "\n", encoding="utf-8")

    report = collect_runtime_turn_capture(capture_path, router_audit_path=router_path)

    assert report["model_chain"] == {
        "status": "correlated",
        "correlation_scope": "turn_level_latest_outcome",
        "turn_correlation_id": "turn-abc",
        "router_request_hash": None,
        "actual_forwarded_model": "work-model",
        "router_ok": True,
        "source": "router_final_forward_audit",
    }


def test_runtime_capture_never_guesses_actual_model_when_router_row_is_missing(tmp_path):
    capture_path = tmp_path / "latest-turn.json"
    router_path = tmp_path / "audit.jsonl"
    capture_path.write_text(json.dumps({
        "source": "exact_host_capture",
        "turn_correlation_id": "turn-missing",
        "state_machine": {"route_metadata": {"hermes_route_bucket": "task"}},
    }), encoding="utf-8")
    router_path.write_text("", encoding="utf-8")

    report = collect_runtime_turn_capture(capture_path, router_audit_path=router_path)

    assert "decision_model" not in report["model_chain"]
    assert report["model_chain"]["actual_forwarded_model"] is None
    assert report["model_chain"]["status"] == "unavailable"


def test_runtime_capture_uses_latest_router_outcome_for_same_turn_retry(tmp_path):
    capture_path = tmp_path / "latest-turn.json"
    router_path = tmp_path / "audit.jsonl"
    capture_path.write_text(json.dumps({
        "source": "exact_host_capture",
        "turn_correlation_id": "turn-retry",
        "state_machine": {"route_metadata": {"hermes_route_bucket": "task"}},
    }), encoding="utf-8")
    router_path.write_text("\n".join([
        json.dumps({"turn_correlation_id": "turn-retry", "request_hash": "first", "forwarded_model": "fallback", "ok": False}),
        json.dumps({"turn_correlation_id": "turn-retry", "request_hash": "retry", "forwarded_model": "work-model", "ok": True}),
    ]) + "\n", encoding="utf-8")

    report = collect_runtime_turn_capture(capture_path, router_audit_path=router_path)

    assert report["model_chain"]["correlation_scope"] == "turn_level_latest_outcome"
    assert report["model_chain"]["router_request_hash"] == "retry"
    assert report["model_chain"]["actual_forwarded_model"] == "work-model"
