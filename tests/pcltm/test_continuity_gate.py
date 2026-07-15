import json
import os
import subprocess
import sys
import time
from collections import UserDict
from unittest.mock import patch

from pcltm.continuity_gate import evaluate_continuity_manifest, evaluate_manifest_file, main


def _subprocess_env():
    packages = str((__import__("pathlib").Path(__file__).parents[2] / "packages").resolve())
    return {**os.environ, "PYTHONPATH": packages + os.pathsep + os.environ.get("PYTHONPATH", "")}


def _manifest(candidate):
    return {
        "schema_version": 1,
        "baseline_id": "continuity-baseline-v1",
        "cases": [
            {
                "case_id": "resume-active-task-across-session",
                "baseline": {
                    "identity": {"agent_id": "example-persona-rin"},
                    "task": {
                        "current": "建立 Continuity Preservation Gate",
                        "constraints": ["production-read-only", "no-git-commit"],
                        "evidence_refs": ["session:42", "test:326-passed"],
                        "summary": "baseline summary",
                    },
                },
                "candidate": candidate,
                "assertions": [
                    {"path": "identity.agent_id", "operator": "equal", "severity": "critical"},
                    {"path": "task.current", "operator": "equal", "severity": "critical"},
                    {"path": "task.constraints", "operator": "contains_all", "severity": "critical"},
                    {"path": "task.evidence_refs", "operator": "non_empty", "severity": "critical"},
                    {"path": "task.summary", "operator": "equal", "severity": "warning"},
                ],
            }
        ],
    }


def test_python_api_rejects_hostile_container_subclass_without_calling_methods():
    class HostileDict(dict):
        def get(self, *args, **kwargs):
            raise RuntimeError("evil get")

    report = evaluate_continuity_manifest(HostileDict())

    assert report.status == "invalid"
    assert report.exit_code == 2


def test_cli_rejects_unpaired_surrogate_without_traceback(tmp_path, capsys):
    manifest_path = tmp_path / "surrogate.json"
    manifest_path.write_text(
        '{"schema_version":1,"baseline_id":"\\ud800","cases":[]}', encoding="utf-8"
    )

    exit_code = main([str(manifest_path)])

    stdout = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert stdout["status"] == "invalid"


def test_python_api_rejects_string_subclass_as_invalid():
    class StringSubclass(str):
        pass

    manifest = _manifest(
        {
            "identity": {"agent_id": "example-persona-rin"},
            "task": {
                "current": "建立 Continuity Preservation Gate",
                "constraints": ["production-read-only", "no-git-commit"],
                "evidence_refs": ["session:42"],
                "summary": "baseline summary",
                "injected": StringSubclass("not-an-exact-json-string"),
            },
        }
    )

    report = evaluate_continuity_manifest(manifest)

    assert report.status == "invalid"
    assert report.exit_code == 2


def test_python_api_rejects_excessive_json_depth_as_invalid():
    value = []
    for _ in range(1500):
        value = [value]
    manifest = _manifest(
        {
            "identity": {"agent_id": "example-persona-rin"},
            "task": {
                "current": "建立 Continuity Preservation Gate",
                "constraints": ["production-read-only", "no-git-commit"],
                "evidence_refs": ["session:42", "test:326-passed"],
                "summary": "baseline summary",
                "injected": value,
            },
        }
    )

    report = evaluate_continuity_manifest(manifest)

    assert report.status == "invalid"
    assert report.exit_code == 2
    assert any("maximum depth" in error for error in report.validation_errors)


def test_cli_deep_manifest_overwrites_stale_passed_report(tmp_path, capsys):
    manifest_path = tmp_path / "deep.json"
    report_path = tmp_path / "report.json"
    report_path.write_text('{"status":"passed","exit_code":0,"stale":true}', encoding="utf-8")
    nested = "[" * 1100 + "0" + "]" * 1100
    manifest_path.write_text(
        '{"schema_version":1,"baseline_id":"v1","cases":['
        '{"case_id":"deep","baseline":{"x":1},"candidate":{"x":' + nested + '},'
        '"assertions":[{"path":"x","operator":"non_empty","severity":"critical"}]}]}',
        encoding="utf-8",
    )

    exit_code = main([str(manifest_path), "--report", str(report_path)])

    stdout = json.loads(capsys.readouterr().out)
    persisted = json.loads(report_path.read_text(encoding="utf-8"))
    assert exit_code != 0
    assert stdout["status"] != "passed"
    assert persisted["status"] != "passed"
    assert persisted["exit_code"] != 0


def test_gate_rejects_case_id_with_surrounding_whitespace():
    manifest = _manifest(
        {
            "identity": {"agent_id": "example-persona-rin"},
            "task": {
                "current": "建立 Continuity Preservation Gate",
                "constraints": ["production-read-only", "no-git-commit"],
                "evidence_refs": ["session:42", "test:326-passed"],
                "summary": "baseline summary",
            },
        }
    )
    manifest["cases"][0]["case_id"] = " case-with-whitespace "

    report = evaluate_continuity_manifest(manifest)

    assert report.status == "invalid"
    assert report.exit_code == 2
    assert any("case_id must not have surrounding whitespace" in error for error in report.validation_errors)


def test_non_empty_work_uses_manifest_comparison_budget():
    baseline = {f"k{i}": 1 for i in range(20_000)}
    candidate = dict(baseline)
    assertion = {"path": "value", "operator": "non_empty", "severity": "warning"}
    manifest = {
        "schema_version": 1,
        "baseline_id": "v1",
        "cases": [{
            "case_id": "non-empty-budget",
            "baseline": {"value": baseline},
            "candidate": {"value": candidate},
            "assertions": [assertion] * 10,
        }],
    }

    report = evaluate_continuity_manifest(manifest)

    assert report.status == "invalid"
    assert report.exit_code == 2
    assert any("comparison budget" in error for error in report.validation_errors)


def test_equal_dict_key_scan_uses_manifest_comparison_budget():
    baseline = {f"k{i}": 1 for i in range(20_000)}
    candidate = dict(baseline)
    candidate.pop("k19999")
    candidate["different"] = 1
    assertion = {"path": "value", "operator": "equal", "severity": "warning"}
    manifest = {
        "schema_version": 1,
        "baseline_id": "v1",
        "cases": [{
            "case_id": "key-scan-budget",
            "baseline": {"value": baseline},
            "candidate": {"value": candidate},
            "assertions": [assertion] * 10,
        }],
    }

    report = evaluate_continuity_manifest(manifest)

    assert report.status == "invalid"
    assert report.exit_code == 2
    assert any("comparison budget" in error for error in report.validation_errors)


def test_python_api_rejects_oversized_object_key_with_bounded_report():
    manifest = _manifest({"identity": {"agent_id": "example-persona-rin"}, "task": {}})
    manifest["x" * 20_000] = True

    report = evaluate_continuity_manifest(manifest)

    assert report.status == "invalid"
    assert report.exit_code == 2
    assert len(json.dumps(report.to_dict())) < 20_000


def test_duplicate_oversized_key_parse_error_is_bounded(tmp_path):
    manifest_path = tmp_path / "duplicate-key.json"
    key = "x" * 20_000
    manifest_path.write_text(f'{{"{key}":1,"{key}":2}}', encoding="utf-8")

    report = evaluate_manifest_file(manifest_path)

    assert report.status == "invalid"
    assert report.exit_code == 2
    assert len(json.dumps(report.to_dict())) < 20_000


def test_recursive_comparison_work_uses_manifest_budget():
    baseline_item = [0] * 20_000 + [1]
    candidate_item = [0] * 20_000 + [2]
    assertion = {"path": "value", "operator": "contains_all", "severity": "warning"}
    manifest = {
        "schema_version": 1,
        "baseline_id": "v1",
        "cases": [{
            "case_id": "recursive-budget",
            "baseline": {"value": [baseline_item]},
            "candidate": {"value": [candidate_item]},
            "assertions": [assertion] * 20,
        }],
    }

    started = time.perf_counter()
    report = evaluate_continuity_manifest(manifest)
    elapsed = time.perf_counter() - started

    assert report.status == "invalid"
    assert report.exit_code == 2
    assert any("budget" in error for error in report.validation_errors)
    assert elapsed < 1.0


def test_python_api_rejects_manifest_exceeding_node_budget():
    manifest = {
        "schema_version": 1,
        "baseline_id": "v1",
        "cases": [{"case_id": f"c{i}", "baseline": {}, "candidate": {}, "assertions": []} for i in range(120_000)],
    }

    report = evaluate_continuity_manifest(manifest)

    assert report.status == "invalid"
    assert report.exit_code == 2
    assert any("node budget" in error for error in report.validation_errors)


def test_manifest_file_rejects_oversized_input_before_read(tmp_path):
    manifest_path = tmp_path / "oversized.json"
    manifest_path.write_text(" " * 2_000_000 + "{}", encoding="utf-8")

    report = evaluate_manifest_file(manifest_path)

    assert report.status == "invalid"
    assert report.exit_code == 2
    assert any("size" in error for error in report.validation_errors)


def test_report_rejects_oversized_metadata_with_bounded_serialization():
    huge = "x" * 20_000
    manifest = {
        "schema_version": 1,
        "baseline_id": huge,
        "cases": [{
            "case_id": huge,
            "baseline": {"value": 1},
            "candidate": {"value": 2},
            "assertions": [{"path": huge, "operator": huge, "severity": "critical"}],
        }],
    }

    report = evaluate_continuity_manifest(manifest)
    serialized = json.dumps(report.to_dict())

    assert report.status == "invalid"
    assert report.exit_code == 2
    assert len(serialized) < 20_000


def test_report_rejects_integer_too_large_for_json_serialization():
    huge_integer = 10**10_000
    manifest = {
        "schema_version": 1,
        "baseline_id": "v1",
        "cases": [{
            "case_id": "huge-integer",
            "baseline": {"value": huge_integer},
            "candidate": {"value": 0},
            "assertions": [{"path": "value", "operator": "equal", "severity": "critical"}],
        }],
    }

    report = evaluate_continuity_manifest(manifest)

    assert report.status == "invalid"
    assert report.exit_code == 2
    json.dumps(report.to_dict())


def test_report_rejects_unbounded_assertion_count():
    assertion = {"path": "value", "operator": "equal", "severity": "warning"}
    manifest = {
        "schema_version": 1,
        "baseline_id": "v1",
        "cases": [{
            "case_id": "too-many-assertions",
            "baseline": {"value": 1},
            "candidate": {"value": 2},
            "assertions": [assertion] * 1_000,
        }],
    }

    report = evaluate_continuity_manifest(manifest)

    assert report.status == "invalid"
    assert report.exit_code == 2
    assert len(json.dumps(report.to_dict())) < 20_000


def test_contains_all_stops_at_comparison_budget():
    manifest = {
        "schema_version": 1,
        "baseline_id": "v1",
        "cases": [{
            "case_id": "contains-budget",
            "baseline": {"value": list(range(2_000))},
            "candidate": {"value": list(range(2_000, 4_000))},
            "assertions": [{"path": "value", "operator": "contains_all", "severity": "critical"}],
        }],
    }

    started = time.perf_counter()
    report = evaluate_continuity_manifest(manifest)
    elapsed = time.perf_counter() - started

    assert report.status == "blocked"
    assert report.exit_code == 1
    assert "budget" in report.regressions[0].reason
    assert elapsed < 1.0


def test_contains_all_missing_shared_dag_has_bounded_reason():
    missing = [0]
    for _ in range(22):
        missing = [missing, missing]
    manifest = {
        "schema_version": 1,
        "baseline_id": "v1",
        "cases": [{
            "case_id": "missing-dag",
            "baseline": {"value": [missing]},
            "candidate": {"value": []},
            "assertions": [{"path": "value", "operator": "contains_all", "severity": "critical"}],
        }],
    }

    report = evaluate_continuity_manifest(manifest)

    assert report.status == "blocked"
    assert report.exit_code == 1
    assert len(report.regressions[0].reason) < 500
    assert "lost 1 baseline item" in report.regressions[0].reason
    serialized = json.dumps(report.to_dict())
    assert len(serialized) < 5_000


def test_equal_assertion_on_shared_dag_does_not_expand_exponentially():
    baseline = [0]
    candidate = [0]
    for _ in range(22):
        baseline = [baseline, baseline]
        candidate = [candidate, candidate]
    manifest = {
        "schema_version": 1,
        "baseline_id": "v1",
        "cases": [{
            "case_id": "equal-dag",
            "baseline": {"value": baseline},
            "candidate": {"value": candidate},
            "assertions": [{"path": "value", "operator": "equal", "severity": "critical"}],
        }],
    }

    started = time.perf_counter()
    report = evaluate_continuity_manifest(manifest)
    elapsed = time.perf_counter() - started

    assert report.status == "passed"
    assert elapsed < 2.0


def test_non_empty_assertion_on_shared_dag_does_not_expand_exponentially():
    baseline = {"leaf": 1}
    candidate = {"leaf": 1}
    for _ in range(22):
        baseline = {"left": baseline, "right": baseline}
        candidate = {"left": candidate, "right": candidate}
    manifest = {
        "schema_version": 1,
        "baseline_id": "v1",
        "cases": [{
            "case_id": "nonempty-dag",
            "baseline": {"value": baseline},
            "candidate": {"value": candidate},
            "assertions": [{"path": "value", "operator": "non_empty", "severity": "critical"}],
        }],
    }

    started = time.perf_counter()
    report = evaluate_continuity_manifest(manifest)
    elapsed = time.perf_counter() - started

    assert report.status == "passed"
    assert elapsed < 2.0


def test_wide_valid_list_validation_has_bounded_peak_memory():
    import tracemalloc

    wide = [0] * 300_000
    manifest = _manifest(
        {
            "identity": {"agent_id": "example-persona-rin"},
            "task": {
                "current": "建立 Continuity Preservation Gate",
                "constraints": ["production-read-only", "no-git-commit"],
                "evidence_refs": ["session:42"],
                "summary": "baseline summary",
                "injected": wide,
            },
        }
    )

    tracemalloc.start()
    report = evaluate_continuity_manifest(manifest)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert report.status == "passed"
    assert peak < 10_000_000


def test_wide_invalid_dict_respects_global_validation_error_budget():
    manifest = {index: None for index in range(1000)}

    report = evaluate_continuity_manifest(manifest)

    assert report.status == "invalid"
    assert report.exit_code == 2
    assert len(report.validation_errors) == 101
    assert report.validation_errors[-1] == "validation stopped after 100 errors"


def test_many_invalid_cases_respect_global_validation_error_budget():
    manifest = {"schema_version": 1, "baseline_id": "v1", "cases": [{} for _ in range(1000)]}

    report = evaluate_continuity_manifest(manifest)

    assert report.status == "invalid"
    assert report.exit_code == 2
    assert len(report.validation_errors) == 101
    assert report.validation_errors[-1] == "validation stopped after 100 errors"


def test_invalid_shared_json_dag_has_bounded_validation_errors():
    shared = [object()]
    for _ in range(20):
        shared = [shared, shared]
    manifest = _manifest(
        {
            "identity": {"agent_id": "example-persona-rin"},
            "task": {
                "current": "建立 Continuity Preservation Gate",
                "constraints": ["production-read-only", "no-git-commit"],
                "evidence_refs": ["session:42"],
                "summary": "baseline summary",
                "injected": shared,
            },
        }
    )

    started = time.perf_counter()
    report = evaluate_continuity_manifest(manifest)
    elapsed = time.perf_counter() - started

    assert report.status == "invalid"
    assert report.exit_code == 2
    assert len(report.validation_errors) <= 101
    assert elapsed < 2.0


def test_python_api_validates_shared_json_dag_without_exponential_retraversal():
    shared = [0]
    for _ in range(24):
        shared = [shared, shared]
    manifest = _manifest(
        {
            "identity": {"agent_id": "example-persona-rin"},
            "task": {
                "current": "建立 Continuity Preservation Gate",
                "constraints": ["production-read-only", "no-git-commit"],
                "evidence_refs": ["session:42"],
                "summary": "baseline summary",
                "injected": shared,
            },
        }
    )

    started = time.perf_counter()
    report = evaluate_continuity_manifest(manifest)
    elapsed = time.perf_counter() - started

    assert report.status == "passed"
    assert report.exit_code == 0
    assert elapsed < 2.0


def test_python_api_rejects_cyclic_json_container_as_invalid():
    cycle = []
    cycle.append(cycle)
    manifest = _manifest(
        {
            "identity": {"agent_id": "example-persona-rin"},
            "task": {
                "current": "建立 Continuity Preservation Gate",
                "constraints": ["production-read-only", "no-git-commit"],
                "evidence_refs": ["session:42", "test:326-passed"],
                "summary": "baseline summary",
                "injected": cycle,
            },
        }
    )

    report = evaluate_continuity_manifest(manifest)

    assert report.status == "invalid"
    assert report.exit_code == 2
    assert any("cyclic" in error for error in report.validation_errors)


def test_python_api_rejects_mapping_that_json_dumps_cannot_serialize():
    manifest = _manifest(
        {
            "identity": {"agent_id": "example-persona-rin"},
            "task": {
                "current": "建立 Continuity Preservation Gate",
                "constraints": ["production-read-only", "no-git-commit"],
                "evidence_refs": ["session:42", "test:326-passed"],
                "summary": "baseline summary",
                "injected": UserDict({"a": 1}),
            },
        }
    )

    report = evaluate_continuity_manifest(manifest)

    assert report.status == "invalid"
    assert report.exit_code == 2


def test_python_api_rejects_non_json_manifest_values():
    for value in ({"not", "json"}, 1 + 2j, b"bytes"):
        manifest = _manifest(
            {
                "identity": {"agent_id": "example-persona-rin"},
                "task": {
                    "current": "建立 Continuity Preservation Gate",
                    "constraints": ["production-read-only", "no-git-commit"],
                    "evidence_refs": ["session:42", "test:326-passed"],
                    "summary": "baseline summary",
                    "injected": value,
                },
            }
        )

        report = evaluate_continuity_manifest(manifest)

        assert report.status == "invalid"
        assert report.exit_code == 2
        assert any("JSON-compatible" in error for error in report.validation_errors)


def test_gate_blocks_whitespace_padded_critical_severity_instead_of_failing_open():
    manifest = _manifest(
        {
            "identity": {"agent_id": "generic"},
            "task": {
                "current": "建立 Continuity Preservation Gate",
                "constraints": ["production-read-only", "no-git-commit"],
                "evidence_refs": ["session:42", "test:326-passed"],
                "summary": "baseline summary",
            },
        }
    )
    manifest["cases"][0]["assertions"][0]["severity"] = " critical "

    report = evaluate_continuity_manifest(manifest)

    assert report.status != "passed"
    assert report.exit_code != 0


def test_gate_blocks_any_critical_continuity_regression():
    report = evaluate_continuity_manifest(
        _manifest(
            {
                "identity": {"agent_id": "generic-agent"},
                "task": {
                    "current": "建立 Continuity Preservation Gate",
                    "constraints": ["production-read-only"],
                    "evidence_refs": [],
                    "summary": "candidate wording may change",
                },
            }
        )
    )

    assert report.status == "blocked"
    assert report.exit_code == 1
    assert report.critical_regression_count == 3
    assert report.warning_count == 1
    assert {item.path for item in report.regressions if item.severity == "critical"} == {
        "identity.agent_id",
        "task.constraints",
        "task.evidence_refs",
    }
    payload = report.to_dict()
    assert payload["authority_boundary"] == "read_only_shadow_evaluation"
    assert payload["baseline_id"] == "continuity-baseline-v1"
    json.dumps(payload, ensure_ascii=False)


def test_gate_passes_when_candidate_preserves_critical_semantics():
    report = evaluate_continuity_manifest(
        _manifest(
            {
                "identity": {"agent_id": "example-persona-rin"},
                "task": {
                    "current": "建立 Continuity Preservation Gate",
                    "constraints": ["no-git-commit", "production-read-only", "extra-safety"],
                    "evidence_refs": ["shadow:run-1"],
                    "summary": "baseline summary",
                },
            }
        )
    )

    assert report.status == "passed"
    assert report.exit_code == 0
    assert report.critical_regression_count == 0
    assert report.warning_count == 0
    assert report.case_results[0].status == "passed"


def test_gate_rejects_malformed_or_unsafe_manifest_contracts():
    manifest = _manifest({"identity": {"agent_id": "example-persona-rin"}, "task": {}})
    manifest["cases"][0]["assertions"][0]["operator"] = "execute_python"

    report = evaluate_continuity_manifest(manifest)

    assert report.status == "invalid"
    assert report.exit_code == 2
    assert report.validation_errors == [
        "case resume-active-task-across-session assertion 0 uses unsupported operator: execute_python"
    ]


def test_equal_assertion_cannot_silently_pass_when_both_paths_are_missing():
    manifest = _manifest(
        {
            "identity": {"agent_id": "example-persona-rin"},
            "task": {
                "current": "建立 Continuity Preservation Gate",
                "constraints": ["production-read-only", "no-git-commit"],
                "evidence_refs": ["shadow:run-1"],
                "summary": "baseline summary",
            },
        }
    )
    manifest["cases"][0]["assertions"] = [
        {"path": "task.typo", "operator": "equal", "severity": "critical"}
    ]

    report = evaluate_continuity_manifest(manifest)

    assert report.status == "blocked"
    assert report.regressions[0].reason == "baseline and candidate paths are both missing"


def test_cli_writes_machine_readable_report_and_returns_blocking_exit_code(tmp_path, capsys):
    manifest_path = tmp_path / "shadow-manifest.json"
    report_path = tmp_path / "reports" / "continuity-gate.json"
    manifest_path.write_text(
        json.dumps(
            _manifest(
                {
                    "identity": {"agent_id": "generic-agent"},
                    "task": {
                        "current": "建立 Continuity Preservation Gate",
                        "constraints": ["production-read-only"],
                        "evidence_refs": [],
                    },
                }
            ),
        ),
        encoding="utf-8",
    )

    exit_code = main([str(manifest_path), "--report", str(report_path)])

    stdout_payload = json.loads(capsys.readouterr().out)
    file_payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert stdout_payload == file_payload
    assert file_payload["status"] == "blocked"
    assert file_payload["critical_regression_count"] == 3


def test_non_empty_requires_a_present_non_empty_baseline():
    for baseline_task in ({}, {"evidence_refs": []}):
        manifest = _manifest(
            {
                "identity": {"agent_id": "example-persona-rin"},
                "task": {
                    "current": "建立 Continuity Preservation Gate",
                    "constraints": ["production-read-only", "no-git-commit"],
                    "evidence_refs": ["candidate:self-authored"],
                    "summary": "baseline summary",
                },
            }
        )
        manifest["cases"][0]["baseline"]["task"] = baseline_task
        manifest["cases"][0]["assertions"] = [
            {"path": "task.evidence_refs", "operator": "non_empty", "severity": "critical"}
        ]

        report = evaluate_continuity_manifest(manifest)

        assert report.status == "blocked"
        assert report.exit_code == 1
        assert report.regressions[0].reason in {
            "baseline path is missing",
            "baseline value is empty",
        }


def test_boolean_schema_version_is_invalid():
    manifest = _manifest({"identity": {"agent_id": "example-persona-rin"}, "task": {}})
    manifest["schema_version"] = True

    report = evaluate_continuity_manifest(manifest)

    assert report.status == "invalid"
    assert report.exit_code == 2


def test_cli_rejects_non_standard_json_numbers(tmp_path):
    manifest_path = tmp_path / "nan.json"
    manifest_path.write_text(
        '{"schema_version":1,"baseline_id":"v1","cases":['
        '{"case_id":"nan","baseline":{"x":1},"candidate":{"x":NaN},'
        '"assertions":[{"path":"x","operator":"non_empty","severity":"critical"}]}]}',
        encoding="utf-8",
    )

    completed = subprocess.run(
        [sys.executable, "-m", "pcltm.continuity_gate", str(manifest_path)],
        check=False,
        capture_output=True,
        text=True,
        env=_subprocess_env(),
    )

    assert completed.returncode == 2
    assert json.loads(completed.stdout)["status"] == "invalid"


def test_cli_report_write_failure_has_distinct_exit_code(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(_manifest({"identity": {"agent_id": "example-persona-rin"}, "task": {}})),
        encoding="utf-8",
    )
    report_parent_is_file = tmp_path / "not-a-directory"
    report_parent_is_file.write_text("occupied", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pcltm.continuity_gate",
            str(manifest_path),
            "--report",
            str(report_parent_is_file / "report.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=_subprocess_env(),
    )

    assert completed.returncode == 3
    assert json.loads(completed.stdout)["status"] == "report_error"


def test_json_comparisons_do_not_treat_booleans_as_numbers():
    for operator, baseline_value, candidate_value in (
        ("equal", 1, True),
        ("equal", [1], [True]),
        ("contains_all", [1], [True]),
        ("contains_all", [False], [0]),
    ):
        manifest = {
            "schema_version": 1,
            "baseline_id": "v1",
            "cases": [
                {
                    "case_id": "strict-json-types",
                    "baseline": {"value": baseline_value},
                    "candidate": {"value": candidate_value},
                    "assertions": [
                        {"path": "value", "operator": operator, "severity": "critical"}
                    ],
                }
            ],
        }

        report = evaluate_continuity_manifest(manifest)

        assert report.status == "blocked"
        assert report.exit_code == 1


def test_unicode_report_encoding_failure_has_distinct_exit_code(tmp_path):
    manifest_path = tmp_path / "surrogate.json"
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps({"status": "passed", "exit_code": 0, "stale": True}),
        encoding="utf-8",
    )
    manifest_path.write_text(
        '{"schema_version":1,"baseline_id":"v1","cases":['
        '{"case_id":"unicode","baseline":{"x":"ok"},'
        '"candidate":{"x":"\\ud800"},'
        '"assertions":[{"path":"x","operator":"equal","severity":"critical"}]}]}',
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pcltm.continuity_gate",
            str(manifest_path),
            "--report",
            str(report_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=_subprocess_env(),
    )

    assert completed.returncode == 2
    assert json.loads(completed.stdout)["status"] == "invalid"
    persisted = json.loads(report_path.read_text(encoding="utf-8"))
    assert persisted["status"] == "invalid"
    assert persisted["exit_code"] == 2


def test_manifest_rejects_duplicate_json_keys_and_non_string_identifiers(tmp_path):
    duplicate_path = tmp_path / "duplicate.json"
    duplicate_path.write_text(
        '{"schema_version":1,"schema_version":1,"baseline_id":"v1",'
        '"cases":[{"case_id":"c","baseline":{"x":1,"x":1},'
        '"candidate":{"x":1},"assertions":['
        '{"path":"x","operator":"equal","severity":"critical"}]}]}',
        encoding="utf-8",
    )
    duplicate = subprocess.run(
        [sys.executable, "-m", "pcltm.continuity_gate", str(duplicate_path)],
        check=False,
        capture_output=True,
        text=True,
        env=_subprocess_env(),
    )
    assert duplicate.returncode == 2
    assert json.loads(duplicate.stdout)["status"] == "invalid"

    valid_candidate = {"identity": {"agent_id": "rin"}, "task": {}}
    for field, value in (
        ("baseline_id", 1),
        ("baseline_id", True),
        ("case_id", 1),
        ("path", 1),
    ):
        manifest = _manifest(valid_candidate)
        if field == "baseline_id":
            manifest[field] = value
        elif field == "case_id":
            manifest["cases"][0][field] = value
        else:
            manifest["cases"][0]["assertions"][0][field] = value
        report = evaluate_continuity_manifest(manifest)
        assert report.status == "invalid"
        assert report.exit_code == 2


def test_non_empty_requires_candidate_json_type_to_match_baseline():
    for baseline_value, candidate_value in (
        (["e"], True),
        (["e"], 1),
        ("e", True),
        ({"e": 1}, {"x": 1}),
        ({"a": {"b": 1}}, {"a": {}}),
        ({"a": 1}, {"a": None}),
    ):
        manifest = {
            "schema_version": 1,
            "baseline_id": "v1",
            "cases": [
                {
                    "case_id": "non-empty-type",
                    "baseline": {"value": baseline_value},
                    "candidate": {"value": candidate_value},
                    "assertions": [
                        {"path": "value", "operator": "non_empty", "severity": "critical"}
                    ],
                }
            ],
        }
        report = evaluate_continuity_manifest(manifest)
        assert report.status == "blocked"
        assert report.exit_code == 1


def test_first_sentinel_replace_failure_does_not_leave_stale_passed_report(tmp_path, capsys):
    manifest_path = tmp_path / "manifest.json"
    report_path = tmp_path / "report.json"
    report_path.write_text('{"status":"passed","exit_code":0}', encoding="utf-8")
    manifest_path.write_text(
        json.dumps({"schema_version": 1, "baseline_id": "v1", "cases": []}),
        encoding="utf-8",
    )

    with patch("pcltm.continuity_gate.os.replace", side_effect=OSError("first replace boom")):
        exit_code = main([str(manifest_path), "--report", str(report_path)])

    stdout = json.loads(capsys.readouterr().out)
    assert exit_code == 3
    assert stdout["status"] == "report_error"
    assert not report_path.exists()


def test_report_failure_installs_fail_closed_sentinel_before_risky_write(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    report_path = tmp_path / "report.json"
    report_path.write_text('{"status":"passed","exit_code":0}', encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "baseline_id": "v1",
                "cases": [
                    {
                        "case_id": "c",
                        "baseline": {"x": 1},
                        "candidate": {"x": 1},
                        "assertions": [
                            {"path": "x", "operator": "equal", "severity": "critical"}
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    real_replace = __import__("os").replace
    replacement_count = 0

    def fail_final_replace(source, destination):
        nonlocal replacement_count
        replacement_count += 1
        if replacement_count == 2:
            raise OSError("replace boom")
        real_replace(source, destination)

    with patch("pcltm.continuity_gate.os.replace", side_effect=fail_final_replace):
        exit_code = main([str(manifest_path), "--report", str(report_path)])

    assert exit_code == 3
    assert replacement_count == 2
    assert json.loads(report_path.read_text(encoding="utf-8"))["status"] == "report_error"


def test_report_serialization_value_error_has_controlled_cli_contract(tmp_path, capsys):
    manifest_path = tmp_path / "manifest.json"
    report_path = tmp_path / "report.json"
    report_path.write_text('{"status":"passed","exit_code":0}', encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "baseline_id": "v1",
                "cases": [
                    {
                        "case_id": "c",
                        "baseline": {"x": 1},
                        "candidate": {"x": 1},
                        "assertions": [
                            {"path": "x", "operator": "equal", "severity": "critical"}
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    real_dumps = json.dumps

    def fail_report_serialization(*args, **kwargs):
        if kwargs.get("indent") == 2:
            raise ValueError("serialization boom")
        return real_dumps(*args, **kwargs)

    with patch("pcltm.continuity_gate.json.dumps", side_effect=fail_report_serialization):
        exit_code = main([str(manifest_path), "--report", str(report_path)])

    assert exit_code == 3
    stdout = json.loads(capsys.readouterr().out)
    assert stdout["status"] == "report_error"
    assert stdout["exit_code"] == 3
    assert not report_path.exists()


def test_main_does_not_misclassify_evaluator_value_error_as_report_error(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "baseline_id": "v1",
                "cases": [
                    {
                        "case_id": "c",
                        "baseline": {"x": 1},
                        "candidate": {"x": 1},
                        "assertions": [
                            {"path": "x", "operator": "equal", "severity": "critical"}
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with patch(
        "pcltm.continuity_gate.evaluate_continuity_manifest",
        side_effect=ValueError("programmer boom"),
    ):
        try:
            main([str(manifest_path)])
        except ValueError as exc:
            assert str(exc) == "programmer boom"
        else:
            raise AssertionError("evaluator ValueError must not be classified as report_error")
