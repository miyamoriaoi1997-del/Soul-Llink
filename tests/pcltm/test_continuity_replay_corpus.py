import json
from pathlib import Path


CORPUS_PATH = Path(__file__).parents[1] / "fixtures" / "continuity" / "replay_corpus_v1.json"


def test_replay_corpus_v1_has_required_safety_coverage_and_no_raw_history_claim():
    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))

    assert corpus["schema_version"] == 1
    assert corpus["authority_boundary"] == "sanitized_replay_fixture_only"
    assert corpus["contains_raw_private_history"] is False
    scenario_ids = {scenario["scenario_id"] for scenario in corpus["scenarios"]}
    assert scenario_ids == {
        "explicit-resume-active-task",
        "new-request-does-not-resurrect-old-task",
        "control-payload-is-not-user-intent",
        "tool-tail-remains-evidence-only",
        "identity-task-and-constraints-survive-session-boundary",
        "terminal-task-states-do-not-reactivate",
    }
    assert all(
        scenario["source_kind"] == "sanitized_regression_scenario"
        for scenario in corpus["scenarios"]
    )


def test_every_replay_scenario_declares_expected_semantics():
    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))

    for scenario in corpus["scenarios"]:
        assert scenario["current_user_message"].strip()
        assert scenario["expected"]
        assert all(isinstance(key, str) and key for key in scenario["expected"])
