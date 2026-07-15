"""Read-only continuity preservation gate for baseline/candidate shadow runs.

The gate compares already-produced structured artifacts.  It deliberately does
not execute either runtime, read a production database, or mutate memory.  This
keeps migration authority outside the evaluator: a critical regression blocks
promotion, while warnings remain visible without becoming hidden policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

SUPPORTED_OPERATORS = frozenset({"equal", "contains_all", "non_empty"})
SUPPORTED_SEVERITIES = frozenset({"critical", "warning"})
AUTHORITY_BOUNDARY = "read_only_shadow_evaluation"


class ReportWriteError(Exception):
    """Raised only when the optional machine-readable report cannot be persisted."""


@dataclass(frozen=True)
class ContinuityRegression:
    case_id: str
    path: str
    operator: str
    severity: str
    baseline_value: Any
    candidate_value: Any
    reason: str


@dataclass(frozen=True)
class ContinuityCaseResult:
    case_id: str
    status: str
    assertion_count: int
    critical_regression_count: int
    warning_count: int


@dataclass(frozen=True)
class ContinuityGateReport:
    schema_version: int = 1
    authority_boundary: str = AUTHORITY_BOUNDARY
    baseline_id: str = ""
    status: str = "invalid"
    exit_code: int = 2
    case_results: tuple[ContinuityCaseResult, ...] = field(default_factory=tuple)
    regressions: tuple[ContinuityRegression, ...] = field(default_factory=tuple)
    validation_errors: list[str] = field(default_factory=list)
    producer: str = "pcltm.continuity_gate.evaluate_continuity_manifest"
    bindings: dict[str, str] = field(default_factory=dict)

    @property
    def critical_regression_count(self) -> int:
        return sum(item.severity == "critical" for item in self.regressions)

    @property
    def warning_count(self) -> int:
        return sum(item.severity == "warning" for item in self.regressions)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["critical_regression_count"] = self.critical_regression_count
        payload["warning_count"] = self.warning_count
        return payload


def _lookup(value: Any, path: str) -> tuple[bool, Any]:
    current = value
    if not path:
        return True, current
    for part in path.split("."):
        if isinstance(current, Mapping) and part in current:
            current = current[part]
            continue
        if isinstance(current, Sequence) and not isinstance(current, (str, bytes)) and part.isdigit():
            index = int(part)
            if 0 <= index < len(current):
                current = current[index]
                continue
        return False, None
    return True, current


MAX_JSON_DEPTH = 128
MAX_VALIDATION_ERRORS = 100
MAX_TEXT_LENGTH = 4_096
MAX_METADATA_LENGTH = 256
MAX_INTEGER_BITS = 13_000
MAX_TOTAL_ASSERTIONS = 256
MAX_CONTAINS_COMPARISONS = 100_000
MAX_COMPARISON_NODES = 100_000
MAX_JSON_NODES = 500_000
MAX_INPUT_BYTES = 1_000_000
MAX_ERROR_LENGTH = 512


class _ComparisonBudgetExceeded(Exception):
    pass


@dataclass
class _ComparisonBudget:
    remaining: int = MAX_COMPARISON_NODES

    def consume(self, count: int = 1) -> None:
        self.remaining -= count
        if self.remaining < 0:
            raise _ComparisonBudgetExceeded


class _ValidationErrors(list[str]):
    @property
    def exhausted(self) -> bool:
        return bool(self) and self[-1] == f"validation stopped after {MAX_VALIDATION_ERRORS} errors"

    def append(self, message: str) -> None:
        if self.exhausted:
            return
        if len(self) < MAX_VALIDATION_ERRORS:
            if len(message) > MAX_ERROR_LENGTH:
                message = message[:MAX_ERROR_LENGTH] + "…"
            super().append(message)
        if len(self) == MAX_VALIDATION_ERRORS:
            super().append(f"validation stopped after {MAX_VALIDATION_ERRORS} errors")


def _validate_json_compatible(value: Any, path: str = "manifest") -> _ValidationErrors:
    errors = _ValidationErrors()
    active_container_ids: set[int] = set()
    validated_heights: dict[int, int] = {}
    visited_nodes = 0
    stack: list[tuple[str, Any, str, int, int]] = [("enter", value, path, 0, 0)]
    while stack:
        if errors.exhausted:
            break
        action, current, current_path, depth, error_count = stack.pop()
        if action == "enter":
            visited_nodes += 1
            if visited_nodes > MAX_JSON_NODES:
                errors.append(f"manifest exceeds JSON node budget {MAX_JSON_NODES}")
                break
        if action == "iterate_dict":
            try:
                key, item = next(current)
            except StopIteration:
                continue
            stack.append(("iterate_dict", current, current_path, depth, 0))
            if type(key) is not str:
                errors.append(f"{current_path} must contain only string object keys")
                continue
            if len(key) > MAX_METADATA_LENGTH:
                errors.append(f"{current_path} object key exceeds maximum length {MAX_METADATA_LENGTH}")
                continue
            try:
                key.encode("utf-8")
            except UnicodeEncodeError:
                errors.append(f"{current_path} must contain only UTF-8 encodable object keys")
                continue
            stack.append(("enter", item, f"{current_path}.{key}", depth + 1, 0))
            continue
        if action == "iterate_list":
            try:
                index, item = next(current)
            except StopIteration:
                continue
            stack.append(("iterate_list", current, current_path, depth, 0))
            stack.append(("enter", item, f"{current_path}[{index}]", depth + 1, 0))
            continue
        if action == "exit":
            container_id = id(current)
            active_container_ids.remove(container_id)
            if len(errors) == error_count:
                child_heights = (
                    validated_heights[id(item)] + 1
                    for item in (current.values() if type(current) is dict else current)
                    if type(item) in (dict, list)
                )
                validated_heights[container_id] = max(child_heights, default=0)
            continue
        if current is None or type(current) is bool:
            continue
        if type(current) is int:
            if current.bit_length() > MAX_INTEGER_BITS:
                errors.append(f"{current_path} integer exceeds serialization limit")
            continue
        if type(current) is str:
            if len(current) > MAX_TEXT_LENGTH:
                errors.append(f"{current_path} string exceeds maximum length {MAX_TEXT_LENGTH}")
                continue
            try:
                current.encode("utf-8")
            except UnicodeEncodeError:
                errors.append(f"{current_path} must contain only UTF-8 encodable strings")
            continue
        if type(current) is float:
            if not math.isfinite(current):
                errors.append(f"{current_path} must contain only JSON-compatible values")
            continue
        if type(current) not in (dict, list):
            errors.append(f"{current_path} must contain only JSON-compatible values")
            continue
        if depth > MAX_JSON_DEPTH:
            errors.append(f"{current_path} exceeds maximum depth {MAX_JSON_DEPTH}")
            continue
        container_id = id(current)
        if container_id in active_container_ids:
            errors.append(f"{current_path} must not contain cyclic JSON containers")
            continue
        if container_id in validated_heights:
            if depth + validated_heights[container_id] > MAX_JSON_DEPTH:
                errors.append(f"{current_path} exceeds maximum depth {MAX_JSON_DEPTH}")
            continue
        active_container_ids.add(container_id)
        stack.append(("exit", current, current_path, depth, len(errors)))
        if type(current) is dict:
            stack.append(("iterate_dict", iter(current.items()), current_path, depth, 0))
        else:
            stack.append(("iterate_list", enumerate(current), current_path, depth, 0))
    return errors


def _validate_manifest(manifest: Any) -> list[str]:
    errors = _validate_json_compatible(manifest)
    if errors:
        return errors
    if type(manifest) is not dict:
        return ["manifest must be an object"]
    schema_version = manifest.get("schema_version")
    if type(schema_version) is not int or schema_version != 1:
        errors.append("schema_version must be 1")
    baseline_id = manifest.get("baseline_id")
    if not isinstance(baseline_id, str) or not baseline_id.strip():
        errors.append("baseline_id must be a non-empty string")
    elif len(baseline_id) > MAX_METADATA_LENGTH:
        errors.append(f"baseline_id exceeds maximum length {MAX_METADATA_LENGTH}")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        errors.append("cases must be a non-empty list")
        return errors
    seen_case_ids: set[str] = set()
    total_assertions = 0
    for case_index, case in enumerate(cases):
        if errors.exhausted:
            break
        if not isinstance(case, Mapping):
            errors.append(f"case {case_index} must be an object")
            continue
        raw_case_id = case.get("case_id")
        case_id = raw_case_id.strip() if isinstance(raw_case_id, str) else ""
        label = case_id or str(case_index)
        if not case_id:
            errors.append(f"case {case_index} case_id must be a non-empty string")
        elif len(case_id) > MAX_METADATA_LENGTH:
            errors.append(f"case {case_index} case_id exceeds maximum length {MAX_METADATA_LENGTH}")
        elif isinstance(raw_case_id, str) and raw_case_id != case_id:
            errors.append(f"case {label} case_id must not have surrounding whitespace")
        elif case_id in seen_case_ids:
            errors.append(f"case_id must be unique: {case_id}")
        seen_case_ids.add(case_id)
        if not isinstance(case.get("baseline"), Mapping):
            errors.append(f"case {label} baseline must be an object")
        if not isinstance(case.get("candidate"), Mapping):
            errors.append(f"case {label} candidate must be an object")
        assertions = case.get("assertions")
        if not isinstance(assertions, list) or not assertions:
            errors.append(f"case {label} assertions must be a non-empty list")
            continue
        total_assertions += len(assertions)
        if total_assertions > MAX_TOTAL_ASSERTIONS:
            errors.append(f"manifest exceeds maximum assertion count {MAX_TOTAL_ASSERTIONS}")
            break
        for assertion_index, assertion in enumerate(assertions):
            if errors.exhausted:
                break
            if not isinstance(assertion, Mapping):
                errors.append(f"case {label} assertion {assertion_index} must be an object")
                continue
            raw_path = assertion.get("path")
            raw_operator = assertion.get("operator")
            raw_severity = assertion.get("severity")
            path = raw_path.strip() if isinstance(raw_path, str) else ""
            operator = raw_operator.strip() if isinstance(raw_operator, str) else ""
            severity = raw_severity.strip() if isinstance(raw_severity, str) else ""
            if isinstance(raw_path, str) and raw_path != path:
                errors.append(
                    f"case {label} assertion {assertion_index} path must not have surrounding whitespace"
                )
            if isinstance(raw_operator, str) and raw_operator != operator:
                errors.append(
                    f"case {label} assertion {assertion_index} operator must not have surrounding whitespace"
                )
            if isinstance(raw_severity, str) and raw_severity != severity:
                errors.append(
                    f"case {label} assertion {assertion_index} severity must not have surrounding whitespace"
                )
            if not path:
                errors.append(
                    f"case {label} assertion {assertion_index} path must be a non-empty string"
                )
            elif len(path) > MAX_METADATA_LENGTH:
                errors.append(
                    f"case {label} assertion {assertion_index} path exceeds maximum length {MAX_METADATA_LENGTH}"
                )
            if len(operator) > MAX_METADATA_LENGTH:
                errors.append(
                    f"case {label} assertion {assertion_index} operator exceeds maximum length {MAX_METADATA_LENGTH}"
                )
            elif operator not in SUPPORTED_OPERATORS:
                errors.append(
                    f"case {label} assertion {assertion_index} uses unsupported operator: {operator}"
                )
            if severity not in SUPPORTED_SEVERITIES:
                errors.append(
                    f"case {label} assertion {assertion_index} uses unsupported severity: {severity}"
                )
    return errors


def _reject_non_standard_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            key_summary = key[:MAX_METADATA_LENGTH]
            if len(key) > MAX_METADATA_LENGTH:
                key_summary += "…"
            raise ValueError(f"duplicate JSON object key: {key_summary}")
        output[key] = value
    return output


def _json_equal(
    left: Any,
    right: Any,
    seen_pairs: set[tuple[int, int]] | None = None,
    budget: _ComparisonBudget | None = None,
) -> bool:
    budget = _ComparisonBudget() if budget is None else budget
    budget.consume()
    if type(left) is not type(right):
        return False
    if type(left) in (dict, list):
        seen_pairs = set() if seen_pairs is None else seen_pairs
        pair = (id(left), id(right))
        if pair in seen_pairs:
            return True
        seen_pairs.add(pair)
    if isinstance(left, Mapping):
        budget.consume(len(left) + len(right))
        return left.keys() == right.keys() and all(
            _json_equal(left[key], right[key], seen_pairs, budget) for key in left
        )
    if isinstance(left, Sequence) and not isinstance(left, (str, bytes)):
        return len(left) == len(right) and all(
            _json_equal(left_item, right_item, seen_pairs, budget)
            for left_item, right_item in zip(left, right)
        )
    return left == right


def _preserves_non_empty_structure(
    baseline: Any,
    candidate: Any,
    seen_pairs: set[tuple[int, int]] | None = None,
    budget: _ComparisonBudget | None = None,
) -> bool:
    budget = _ComparisonBudget() if budget is None else budget
    budget.consume()
    if type(candidate) is not type(baseline):
        return False
    if type(baseline) in (dict, list):
        seen_pairs = set() if seen_pairs is None else seen_pairs
        pair = (id(baseline), id(candidate))
        if pair in seen_pairs:
            return True
        seen_pairs.add(pair)
    if isinstance(baseline, Mapping):
        budget.consume(len(baseline) + len(candidate))
        return baseline.keys() <= candidate.keys() and all(
            _preserves_non_empty_structure(baseline[key], candidate[key], seen_pairs, budget)
            for key in baseline
        )
    if isinstance(baseline, Sequence) and not isinstance(baseline, (str, bytes)):
        return bool(candidate)
    return candidate is not None and bool(candidate)


def _report_value_summary(value: Any) -> Any:
    if type(value) is dict:
        return {"type": "object", "size": len(value)}
    if type(value) is list:
        return {"type": "array", "size": len(value)}
    if type(value) is str and len(value) > 256:
        return value[:256] + "…"
    return value


def _compare(
    operator: str,
    baseline: Any,
    candidate: Any,
    *,
    baseline_found: bool,
    candidate_found: bool,
    budget: _ComparisonBudget,
) -> tuple[bool, str]:
    if operator == "equal":
        if not baseline_found and not candidate_found:
            return False, "baseline and candidate paths are both missing"
        if not baseline_found:
            return False, "baseline path is missing"
        if not candidate_found:
            return False, "candidate path is missing"
        return _json_equal(baseline, candidate, budget=budget), "candidate value differs from baseline"
    if operator == "contains_all":
        if not baseline_found:
            return False, "baseline path is missing"
        if not candidate_found:
            return False, "candidate path is missing"
        if not isinstance(baseline, (list, tuple, set)) or isinstance(baseline, (str, bytes)):
            return False, "baseline value is not a collection"
        if not isinstance(candidate, (list, tuple, set)) or isinstance(candidate, (str, bytes)):
            return False, "candidate value is not a collection"
        missing_count = 0
        comparisons = 0
        for item in baseline:
            found = False
            for candidate_item in candidate:
                comparisons += 1
                if comparisons > MAX_CONTAINS_COMPARISONS:
                    return False, f"contains_all comparison budget exceeded ({MAX_CONTAINS_COMPARISONS})"
                if _json_equal(item, candidate_item, budget=budget):
                    found = True
                    break
            if not found:
                missing_count += 1
        suffix = "item" if missing_count == 1 else "items"
        return missing_count == 0, f"candidate collection lost {missing_count} baseline {suffix}"
    if operator == "non_empty":
        if not baseline_found:
            return False, "baseline path is missing"
        if not baseline:
            return False, "baseline value is empty"
        if not candidate_found:
            return False, "candidate path is missing"
        if not _preserves_non_empty_structure(baseline, candidate, budget=budget):
            return False, "candidate value lost baseline type or structure"
        return True, "candidate value is empty"
    return False, f"unsupported operator: {operator}"


def evaluate_continuity_manifest(manifest: Any) -> ContinuityGateReport:
    """Low-level compatibility comparator; this cannot authorize promotion."""
    errors = _validate_manifest(manifest)
    raw_baseline_id = manifest.get("baseline_id") if type(manifest) is dict else ""
    baseline_id = ""
    if type(raw_baseline_id) is str and len(raw_baseline_id) <= MAX_METADATA_LENGTH:
        try:
            raw_baseline_id.encode("utf-8")
        except UnicodeEncodeError:
            pass
        else:
            baseline_id = raw_baseline_id
    if errors:
        return ContinuityGateReport(
            baseline_id=baseline_id,
            status="invalid",
            exit_code=2,
            validation_errors=errors,
        )

    regressions: list[ContinuityRegression] = []
    case_results: list[ContinuityCaseResult] = []
    comparison_budget = _ComparisonBudget()
    for case in manifest["cases"]:
        case_id = str(case["case_id"])
        case_regressions: list[ContinuityRegression] = []
        for assertion in case["assertions"]:
            path = str(assertion["path"])
            operator = str(assertion["operator"])
            severity = str(assertion["severity"])
            baseline_found, baseline_value = _lookup(case["baseline"], path)
            candidate_found, candidate_value = _lookup(case["candidate"], path)
            try:
                passed, reason = _compare(
                    operator,
                    baseline_value,
                    candidate_value,
                    baseline_found=baseline_found,
                    candidate_found=candidate_found,
                    budget=comparison_budget,
                )
            except _ComparisonBudgetExceeded:
                return ContinuityGateReport(
                    baseline_id=baseline_id,
                    status="invalid",
                    exit_code=2,
                    validation_errors=[f"manifest comparison budget exceeded ({MAX_COMPARISON_NODES})"],
                )
            if not passed:
                case_regressions.append(
                    ContinuityRegression(
                        case_id=case_id,
                        path=path,
                        operator=operator,
                        severity=severity,
                        baseline_value=_report_value_summary(baseline_value),
                        candidate_value=_report_value_summary(candidate_value),
                        reason=reason,
                    )
                )
        regressions.extend(case_regressions)
        critical_count = sum(item.severity == "critical" for item in case_regressions)
        warning_count = sum(item.severity == "warning" for item in case_regressions)
        case_results.append(
            ContinuityCaseResult(
                case_id=case_id,
                status="blocked" if critical_count else ("warning" if warning_count else "passed"),
                assertion_count=len(case["assertions"]),
                critical_regression_count=critical_count,
                warning_count=warning_count,
            )
        )

    critical_count = sum(item.severity == "critical" for item in regressions)
    return ContinuityGateReport(
        baseline_id=baseline_id,
        status="blocked" if critical_count else "passed",
        exit_code=1 if critical_count else 0,
        case_results=tuple(case_results),
        regressions=tuple(regressions),
    )


PINNED_AUTHORITY_BOUNDARY = "deployment_pinned_continuity_gate"
PINNED_PRODUCER = "pcltm.continuity_gate.evaluate_pinned_artifacts"


def canonical_json_sha256(value: Any) -> str:
    errors = _validate_json_compatible(value, "artifact")
    if errors:
        raise ValueError(errors[0])
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _invalid_pinned(errors: list[str], baseline_id: str = "", bindings: dict[str, str] | None = None) -> ContinuityGateReport:
    return ContinuityGateReport(schema_version=2, authority_boundary=PINNED_AUTHORITY_BOUNDARY, producer=PINNED_PRODUCER, baseline_id=baseline_id, status="invalid", exit_code=2, validation_errors=errors, bindings={} if bindings is None else bindings)


def evaluate_pinned_artifacts(*, baseline: Any, candidate: Any, policy: Any, expected_baseline_id: str, expected_baseline_sha256: str, expected_policy_sha256: str, expected_corpus_sha256: str | None = None) -> ContinuityGateReport:
    if any(type(value) is not dict for value in (baseline, candidate, policy)):
        return _invalid_pinned(["baseline, candidate, and policy artifacts must be objects"])
    overlap = {"baseline", "baseline_id", "assertions", "policy", "policy_id"}.intersection(candidate)
    errors = (["candidate artifact contains authority fields: " + ", ".join(sorted(overlap))] if overlap else [])
    try:
        bindings = {"baseline_sha256": canonical_json_sha256(baseline), "candidate_sha256": canonical_json_sha256(candidate), "policy_sha256": canonical_json_sha256(policy)}
    except ValueError as exc:
        return _invalid_pinned([str(exc)])
    baseline_id = baseline.get("baseline_id")
    if baseline_id != expected_baseline_id: errors.append("baseline_id does not match deployment pin")
    if bindings["baseline_sha256"] != expected_baseline_sha256: errors.append("baseline digest does not match deployment pin")
    if bindings["policy_sha256"] != expected_policy_sha256: errors.append("policy digest does not match deployment pin")
    if expected_corpus_sha256 is not None:
        bindings["corpus_sha256"] = expected_corpus_sha256
        if policy.get("corpus_sha256") != expected_corpus_sha256: errors.append("corpus digest does not match deployment pin")
    for value, kind in ((baseline, "continuity_baseline_set"), (candidate, "continuity_candidate_set"), (policy, "continuity_policy")):
        if value.get("schema_version") != 1 or value.get("object_type") != kind or not isinstance(value.get("producer"), str) or not value.get("producer"):
            errors.append(f"invalid {kind} envelope")
    bc, cc, assertions = baseline.get("cases"), candidate.get("cases"), policy.get("assertions")
    if not all(isinstance(value, list) and value for value in (bc, cc, assertions)): errors.append("baseline cases, candidate cases, and policy assertions must be non-empty lists")
    if errors: return _invalid_pinned(errors, baseline_id if isinstance(baseline_id, str) else "", bindings)
    bmap = {item.get("case_id"): item.get("artifact") for item in bc if type(item) is dict}; cmap = {item.get("case_id"): item.get("artifact") for item in cc if type(item) is dict}
    if len(bmap) != len(bc) or len(cmap) != len(cc) or bmap.keys() != cmap.keys() or None in bmap: return _invalid_pinned(["case ids must be unique and identical across artifacts"], baseline_id, bindings)
    grouped: dict[str, list[Any]] = {}
    for assertion in assertions:
        if type(assertion) is not dict or assertion.get("case_id") not in bmap: return _invalid_pinned(["policy assertion references an invalid case"], baseline_id, bindings)
        grouped.setdefault(assertion["case_id"], []).append({k: v for k, v in assertion.items() if k != "case_id"})
    compared = evaluate_continuity_manifest({"schema_version": 1, "baseline_id": baseline_id, "cases": [{"case_id": key, "baseline": bmap[key], "candidate": cmap[key], "assertions": grouped.get(key, [])} for key in bmap]})
    return ContinuityGateReport(schema_version=2, authority_boundary=PINNED_AUTHORITY_BOUNDARY, producer=PINNED_PRODUCER, bindings=bindings, baseline_id=baseline_id, status=compared.status, exit_code=compared.exit_code, case_results=compared.case_results, regressions=compared.regressions, validation_errors=compared.validation_errors)


def _load_strict_json(path: str | Path) -> Any:
    data = Path(path).read_bytes()
    if len(data) > MAX_INPUT_BYTES: raise ValueError("artifact exceeds maximum size")
    return json.loads(data.decode("utf-8"), parse_constant=_reject_non_standard_json_constant, object_pairs_hook=_reject_duplicate_json_keys)


def evaluate_pinned_artifact_files(**kwargs: Any) -> ContinuityGateReport:
    try:
        baseline = _load_strict_json(kwargs.pop("baseline_path")); candidate = _load_strict_json(kwargs.pop("candidate_path")); policy = _load_strict_json(kwargs.pop("policy_path"))
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError, RecursionError) as exc:
        return _invalid_pinned([f"cannot read pinned artifact: {str(exc)[:MAX_ERROR_LENGTH]}"])
    return evaluate_pinned_artifacts(baseline=baseline, candidate=candidate, policy=policy, **kwargs)


def verify_promotion_artifact(artifact: Any, *, expected_baseline_id: str, expected_baseline_sha256: str, expected_policy_sha256: str, expected_candidate_sha256: str, expected_corpus_sha256: str | None = None, expected_case_assertion_counts: dict[str, int] | None = None) -> bool:
    if type(artifact) is not dict or artifact.get("schema_version") != 2 or artifact.get("authority_boundary") != PINNED_AUTHORITY_BOUNDARY or artifact.get("producer") != PINNED_PRODUCER or artifact.get("status") != "passed" or artifact.get("exit_code") != 0 or artifact.get("baseline_id") != expected_baseline_id: return False
    expected = {"baseline_sha256": expected_baseline_sha256, "policy_sha256": expected_policy_sha256, "candidate_sha256": expected_candidate_sha256}
    if expected_corpus_sha256 is not None: expected["corpus_sha256"] = expected_corpus_sha256
    bindings = artifact.get("bindings")
    if type(bindings) is not dict or not all(bindings.get(key) == value for key, value in expected.items()): return False
    errors, regressions, cases = artifact.get("validation_errors"), artifact.get("regressions"), artifact.get("case_results")
    if type(errors) is not list or errors or type(regressions) not in (list, tuple) or regressions or type(cases) not in (list, tuple) or not cases: return False
    if type(expected_case_assertion_counts) is not dict or not expected_case_assertion_counts or any(type(case_id) is not str or not case_id or case_id.strip() != case_id or type(count) is not int or count <= 0 for case_id, count in expected_case_assertion_counts.items()): return False
    if any(type(case) is not dict or type(case.get("case_id")) is not str or not case["case_id"] or case["case_id"].strip() != case["case_id"] or case.get("status") != "passed" or type(case.get("assertion_count")) is not int or case["assertion_count"] <= 0 or type(case.get("critical_regression_count")) is not int or case["critical_regression_count"] != 0 or type(case.get("warning_count")) is not int or case["warning_count"] != 0 for case in cases): return False
    actual_counts = {case["case_id"]: case["assertion_count"] for case in cases}
    if len(actual_counts) != len(cases) or actual_counts != expected_case_assertion_counts or sum(actual_counts.values()) != sum(expected_case_assertion_counts.values()): return False
    for key in ("critical_regression_count", "warning_count"):
        if key in artifact and (type(artifact[key]) is not int or artifact[key] != 0): return False
    return True


def _write_report_atomically(destination: Path, report: ContinuityGateReport) -> None:
    temporary_path: Path | None = None
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        report_text = json.dumps(
            report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
        ) + "\n"
        report_bytes = report_text.encode("utf-8")
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(report_bytes)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, destination)
        temporary_path = None
    except (OSError, UnicodeError, ValueError) as exc:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise ReportWriteError(str(exc)) from exc


def evaluate_manifest_file(
    input_path: str | Path, output_path: str | Path | None = None
) -> ContinuityGateReport:
    """Read a JSON manifest and optionally write its deterministic JSON report."""
    source = Path(input_path)
    destination = Path(output_path) if output_path is not None else None
    if destination is not None:
        try:
            destination.unlink(missing_ok=True)
        except OSError as exc:
            raise ReportWriteError(f"cannot clear previous report: {exc}") from exc
        sentinel = ContinuityGateReport(
            status="report_error",
            exit_code=3,
            validation_errors=["report generation did not complete"],
        )
        _write_report_atomically(destination, sentinel)
    try:
        with source.open("rb") as source_file:
            manifest_bytes = source_file.read(MAX_INPUT_BYTES + 1)
        if len(manifest_bytes) > MAX_INPUT_BYTES:
            raise ValueError(f"manifest size exceeds maximum {MAX_INPUT_BYTES} bytes")
        manifest = json.loads(
            manifest_bytes.decode("utf-8"),
            parse_constant=_reject_non_standard_json_constant,
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
        error_text = str(exc)
        if len(error_text) > MAX_ERROR_LENGTH:
            error_text = error_text[:MAX_ERROR_LENGTH] + "…"
        report = ContinuityGateReport(
            status="invalid",
            exit_code=2,
            validation_errors=[f"cannot read manifest: {error_text}"],
        )
    else:
        report = evaluate_continuity_manifest(manifest)
    if destination is not None:
        _write_report_atomically(destination, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a read-only SoulLink continuity shadow manifest")
    parser.add_argument("manifest", help="JSON manifest containing baseline, candidate, and assertions")
    parser.add_argument("--report", help="optional path for the machine-readable JSON report")
    args = parser.parse_args(argv)
    try:
        report = evaluate_manifest_file(args.manifest, args.report)
    except ReportWriteError as exc:
        report = ContinuityGateReport(
            status="report_error",
            exit_code=3,
            validation_errors=[f"cannot write report: {exc}"],
        )
    print(json.dumps(report.to_dict(), ensure_ascii=False, sort_keys=True, allow_nan=False))
    return report.exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
