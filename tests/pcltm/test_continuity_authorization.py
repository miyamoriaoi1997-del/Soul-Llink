import hashlib
import json
import copy

from pcltm.continuity_gate import (
    canonical_json_sha256,
    evaluate_pinned_artifacts,
    evaluate_pinned_artifact_files,
    verify_promotion_artifact,
)


def _artifacts():
    baseline = {"schema_version": 1, "object_type": "continuity_baseline_set", "baseline_id": "b1", "producer": "trusted-capture", "cases": [{"case_id": "c1", "artifact": {"identity": "rin", "refs": ["a"]}}]}
    candidate = {"schema_version": 1, "object_type": "continuity_candidate_set", "producer": "candidate-runner", "cases": [{"case_id": "c1", "artifact": {"identity": "rin", "refs": ["new"]}}]}
    policy = {"schema_version": 1, "object_type": "continuity_policy", "policy_id": "p1", "producer": "deployment-policy", "assertions": [{"case_id": "c1", "path": "identity", "operator": "equal", "severity": "critical"}, {"case_id": "c1", "path": "refs", "operator": "non_empty", "severity": "critical"}]}
    return baseline, candidate, policy


def test_pinned_gate_binds_separate_trusted_inputs_and_authorizes_promotion():
    baseline, candidate, policy = _artifacts()
    report = evaluate_pinned_artifacts(baseline=baseline, candidate=candidate, policy=policy, expected_baseline_id="b1", expected_baseline_sha256=canonical_json_sha256(baseline), expected_policy_sha256=canonical_json_sha256(policy))
    assert report.status == "passed"
    payload = report.to_dict()
    assert payload["authority_boundary"] == "deployment_pinned_continuity_gate"
    assert payload["producer"] == "pcltm.continuity_gate.evaluate_pinned_artifacts"
    assert payload["schema_version"] == 2
    assert payload["bindings"]["baseline_sha256"] == canonical_json_sha256(baseline)
    assert payload["bindings"]["candidate_sha256"] == canonical_json_sha256(candidate)
    assert verify_promotion_artifact(payload, expected_baseline_id="b1", expected_baseline_sha256=canonical_json_sha256(baseline), expected_policy_sha256=canonical_json_sha256(policy), expected_candidate_sha256=canonical_json_sha256(candidate), expected_case_assertion_counts={"c1": 2})


def test_pinned_gate_rejects_candidate_self_report_and_pin_changes():
    baseline, candidate, policy = _artifacts()
    candidate["baseline_id"] = "candidate-invented"
    candidate["assertions"] = []
    report = evaluate_pinned_artifacts(baseline=baseline, candidate=candidate, policy=policy, expected_baseline_id="b1", expected_baseline_sha256=canonical_json_sha256(baseline), expected_policy_sha256=canonical_json_sha256(policy))
    assert report.status == "invalid"
    assert report.exit_code == 2
    assert any("candidate" in error for error in report.validation_errors)

    _, candidate, _ = _artifacts()
    report = evaluate_pinned_artifacts(baseline=baseline, candidate=candidate, policy=policy, expected_baseline_id="b1", expected_baseline_sha256="0" * 64, expected_policy_sha256=canonical_json_sha256(policy))
    assert report.status == "invalid"


def test_file_gate_rejects_duplicate_keys_and_promotion_tampering(tmp_path):
    baseline, candidate, policy = _artifacts()
    paths = [tmp_path / name for name in ("baseline.json", "candidate.json", "policy.json")]
    for path, value in zip(paths, (baseline, candidate, policy)):
        path.write_text(json.dumps(value), encoding="utf-8")
    paths[2].write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
    report = evaluate_pinned_artifact_files(baseline_path=paths[0], candidate_path=paths[1], policy_path=paths[2], expected_baseline_id="b1", expected_baseline_sha256=canonical_json_sha256(baseline), expected_policy_sha256=canonical_json_sha256(policy))
    assert report.status == "invalid"

    good = evaluate_pinned_artifacts(baseline=baseline, candidate=candidate, policy=policy, expected_baseline_id="b1", expected_baseline_sha256=canonical_json_sha256(baseline), expected_policy_sha256=canonical_json_sha256(policy)).to_dict()
    good["bindings"]["candidate_sha256"] = "f" * 64
    assert not verify_promotion_artifact(good, expected_baseline_id="b1", expected_baseline_sha256=canonical_json_sha256(baseline), expected_policy_sha256=canonical_json_sha256(policy), expected_candidate_sha256=canonical_json_sha256(candidate))


def test_legacy_comparator_report_cannot_authorize_promotion():
    legacy = {"schema_version": 1, "authority_boundary": "read_only_shadow_evaluation", "status": "passed", "exit_code": 0, "baseline_id": "b1"}
    assert not verify_promotion_artifact(legacy, expected_baseline_id="b1", expected_baseline_sha256="a" * 64, expected_policy_sha256="b" * 64, expected_candidate_sha256="c" * 64)


def test_promotion_verifier_rejects_internally_inconsistent_pass_artifacts():
    baseline, candidate, policy = _artifacts()
    good = evaluate_pinned_artifacts(baseline=baseline, candidate=candidate, policy=policy, expected_baseline_id="b1", expected_baseline_sha256=canonical_json_sha256(baseline), expected_policy_sha256=canonical_json_sha256(policy)).to_dict()
    pins = dict(expected_baseline_id="b1", expected_baseline_sha256=canonical_json_sha256(baseline), expected_policy_sha256=canonical_json_sha256(policy), expected_candidate_sha256=canonical_json_sha256(candidate), expected_case_assertion_counts={"c1": 2})
    mutations = []
    for key, value in (("validation_errors", ["ignored error"]), ("regressions", [{"severity": "critical"}]), ("case_results", []), ("critical_regression_count", 1), ("warning_count", "0")):
        bad = copy.deepcopy(good); bad[key] = value; mutations.append(bad)
    bad = copy.deepcopy(good); bad["case_results"][0]["status"] = "blocked"; mutations.append(bad)
    bad = copy.deepcopy(good); bad["case_results"][0]["critical_regression_count"] = 1; mutations.append(bad)
    assert all(not verify_promotion_artifact(item, **pins) for item in mutations)


def test_promotion_verifier_requires_trusted_exact_case_contract():
    baseline, candidate, policy = _artifacts()
    good = evaluate_pinned_artifacts(baseline=baseline, candidate=candidate, policy=policy, expected_baseline_id="b1", expected_baseline_sha256=canonical_json_sha256(baseline), expected_policy_sha256=canonical_json_sha256(policy)).to_dict()
    pins = dict(expected_baseline_id="b1", expected_baseline_sha256=canonical_json_sha256(baseline), expected_policy_sha256=canonical_json_sha256(policy), expected_candidate_sha256=canonical_json_sha256(candidate))
    assert not verify_promotion_artifact(good, **pins)
    assert not verify_promotion_artifact(good, **pins, expected_case_assertion_counts={"c1": 0})
    for results in ([{**good["case_results"][0], "case_id": ""}], [{**good["case_results"][0], "case_id": " c1"}], [good["case_results"][0], copy.deepcopy(good["case_results"][0])], [{**good["case_results"][0], "assertion_count": 0}]):
        bad = copy.deepcopy(good); bad["case_results"] = results
        assert not verify_promotion_artifact(bad, **pins, expected_case_assertion_counts={"c1": 2})
    assert verify_promotion_artifact(good, **pins, expected_case_assertion_counts={"c1": 2})
