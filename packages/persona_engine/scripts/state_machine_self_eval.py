#!/usr/bin/env python3
"""Read-only state-machine self-evaluation and audit candidate extraction."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
for path in (ROOT, REPO_ROOT / "packages", REPO_ROOT / "adapters"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from persona_orchestrator import StateOrchestrator

AUTHORITY = "read_only_shadow_evaluation"


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"line {line_no} is not a JSON object")
            rows.append(row)
    return rows


def _evaluate_case(orchestrator: StateOrchestrator, case: dict[str, Any]) -> dict[str, Any]:
    packet = orchestrator.analyze_turn(
        user_message=str(case["message"]),
        recent_messages=case.get("recent_context"),
        emotion_state={"emotion_score": case.get("emotion_score")},
        previous_mode=case.get("previous_mode"),
        platform=str(case.get("platform") or "cli"),
    )
    expected_layers = list(case.get("expected_layers") or [])
    forbidden_layers = list(case.get("forbidden_layers") or [])
    expected_flags = list(case.get("expected_flags") or [])
    transition_expected = str(case.get("expected_transition") or "*")
    checks = {
        "mode": packet.mode == case["expected_mode"],
        "transition": transition_expected == "*" or packet.transition == transition_expected,
        "layers": all(layer in packet.selected_layers for layer in expected_layers),
        "forbidden_layers": not any(layer in packet.selected_layers for layer in forbidden_layers),
        "flags": all(flag in packet.safety_flags for flag in expected_flags),
    }
    return {
        "id": case.get("id"),
        "source": case.get("source"),
        "expected_mode": case["expected_mode"],
        "actual_mode": packet.mode,
        "expected_transition": transition_expected,
        "actual_transition": packet.transition,
        "expected_layers": expected_layers,
        "actual_layers": list(packet.selected_layers),
        "forbidden_layers": forbidden_layers,
        "actual_flags": list(packet.safety_flags),
        "semantic_shadow": packet.semantic_shadow,
        "reason": packet.reason,
        "checks": checks,
        "ok": all(checks.values()),
    }


def build_shadow_report(cases: list[dict[str, Any]]) -> dict[str, Any]:
    if not cases:
        raise ValueError("human-labeled truth set must not be empty")
    invalid_sources = [
        case.get("id")
        for case in cases
        if not (
            str(case.get("source") or "").startswith("human_labeled")
            or case.get("source") == "manual_feedback"
        )
        or case.get("dataset_kind") == "runtime_derived_snapshot"
        or case.get("accuracy_authority") is False
    ]
    if invalid_sources:
        raise ValueError(f"human-labeled truth required; invalid cases: {invalid_sources}")
    orchestrator = StateOrchestrator(
        ROOT,
        log_path=ROOT / "logs" / "state_machine_self_eval.jsonl",
        enable_semantic_shadow=True,
        semantic_backend="local_lightweight",
        core_source="host_core",
    )
    rows = [_evaluate_case(orchestrator, case) for case in cases]
    failures = [row for row in rows if not row["ok"]]
    confusion = Counter((row["expected_mode"], row["actual_mode"]) for row in rows)
    high_risk = [
        row for row in rows
        if row["actual_mode"] == "sex" and row["expected_mode"] != "sex"
    ]
    total = len(rows)
    failed = len(failures)
    status = "manual_review_candidate" if total and failed == 0 else "blocked"
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "authority": AUTHORITY,
        "dataset_kind": "human_truth",
        "metrics": {
            "total": total,
            "passed": total - failed,
            "failed": failed,
            "accuracy": (total - failed) / total if total else 0.0,
            "confusion_matrix": {
                f"{expected}->{actual}": count
                for (expected, actual), count in sorted(confusion.items())
            },
            "high_risk_false_positive_count": len(high_risk),
        },
        "promotion": {
            "status": status,
            "automatic_rule_write": False,
            "automatic_activation": False,
            "requires_human_approval": True,
        },
        "failures": failures,
        "rows": rows,
    }


def extract_audit_candidates(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen_sessions: set[str] = set()
    session_modes: dict[str, list[str]] = {}
    for record in records:
        packet = record.get("packet") or {}
        extra = record.get("extra") or {}
        session_id = str(extra.get("session_id") or "")
        turn_number = extra.get("turn_number")
        audit = (packet.get("route_metadata") or {}).get("decision_audit") or {}
        classifier = audit.get("classifier") or {}
        semantic = packet.get("semantic_shadow") or {}
        mode = packet.get("mode")
        transition = str(packet.get("transition") or "")
        reasons: list[str] = []
        if mode == "sex" and transition.endswith("->sex"):
            reasons.append("high_risk_mode_jump")
        semantic_mode = semantic.get("primary_mode")
        if semantic_mode and semantic_mode != mode:
            reasons.append("semantic_disagreement")
        if transition.startswith("start:"):
            if not session_id:
                reasons.append("continuity_unknown")
            elif session_id in seen_sessions:
                reasons.append("possible_host_continuity_reset")
        if classifier and classifier.get("mode") != mode:
            reasons.append("classifier_final_disagreement")
        if session_id and mode:
            history = session_modes.setdefault(session_id, [])
            if len(history) >= 2 and history[-2] == mode and history[-1] != mode:
                reasons.append("rapid_mode_reversal")
        if reasons:
            identity = "|".join(str(value or "") for value in (
                record.get("timestamp"), session_id, turn_number, mode, transition,
            ))
            candidates.append({
                "candidate_id": hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20],
                "timestamp": record.get("timestamp"),
                "session_id": session_id or None,
                "turn_number": turn_number,
                "message": (classifier.get("signals") or {}).get("normalized_text"),
                "actual_mode": mode,
                "actual_transition": transition,
                "classifier_mode": classifier.get("mode"),
                "semantic_mode": semantic_mode,
                "reason_codes": reasons,
                "needs_manual_label": True,
            })
        if session_id:
            seen_sessions.add(session_id)
            if mode:
                session_modes.setdefault(session_id, []).append(mode)
    return candidates

def build_continuous_sample(
    records: Iterable[dict[str, Any]], *, sample_size: int = 12, sampling_key: str,
) -> list[dict[str, Any]]:
    """Select a stable bounded runtime sample without manufacturing labels."""
    candidates: list[tuple[str, dict[str, Any]]] = []
    for line_number, record in enumerate(records, start=1):
        packet = record.get("packet") if isinstance(record.get("packet"), dict) else {}
        extra = record.get("extra") if isinstance(record.get("extra"), dict) else {}
        route_metadata = packet.get("route_metadata") if isinstance(packet.get("route_metadata"), dict) else {}
        decision_audit = route_metadata.get("decision_audit") if isinstance(route_metadata.get("decision_audit"), dict) else {}
        previous_mode = decision_audit.get("previous_mode")
        status = "captured" if previous_mode in {"daily", "work", "sex"} else "unavailable"
        identity = "|".join(str(value or "") for value in (
            sampling_key, record.get("timestamp"), extra.get("session_id"), extra.get("turn_number"), line_number,
        ))
        candidate_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
        candidates.append((candidate_id, {
            "candidate_id": candidate_id, "source_line_number": line_number,
            "timestamp": record.get("timestamp"), "session_id": extra.get("session_id"),
            "turn_number": extra.get("turn_number"), "previous_mode": previous_mode,
            "previous_mode_status": status, "timeliness_eligible": status == "captured",
            "actual_mode": packet.get("mode"), "actual_transition": packet.get("transition"),
            "actual_layers": list(packet.get("selected_layers") or []),
            "needs_manual_label": True, "source": "production_runtime_sample",
        }))
    return [row for _key, row in sorted(candidates, key=lambda item: item[0])[:max(0, int(sample_size))]]


def build_labeled_runtime_report(sample: Iterable[dict[str, Any]], labels: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Measure correctness and same-turn timeliness from explicit human labels."""
    trusted = {str(x.get("candidate_id")): x for x in labels if x.get("candidate_id") and x.get("reviewer") and x.get("source") == "manual_feedback"}
    rows = [(item, trusted[str(item.get("candidate_id"))]) for item in sample if str(item.get("candidate_id")) in trusted]
    total = len(rows)
    eligible = [(item, label) for item, label in rows if item.get("timeliness_eligible") is True]
    decisive = [(item, label) for item, label in eligible if item.get("previous_mode") != label.get("expected_mode")]
    same_turn = [(item, label) for item, label in decisive if item.get("actual_mode") == label.get("expected_mode") and item.get("actual_transition") == label.get("expected_transition")]
    return {"authority": "human_labeled_production_sample", "metrics": {
        "labeled_count": total,
        "mode_accuracy": sum(i.get("actual_mode") == l.get("expected_mode") for i, l in rows) / total if total else None,
        "transition_accuracy": sum(i.get("actual_transition") == l.get("expected_transition") for i, l in rows) / total if total else None,
        "timeliness": {"timeliness_eligible_count": len(eligible), "timeliness_ineligible_count": total - len(eligible),
            "decisive_switch_case_count": len(decisive), "same_turn_switch_count": len(same_turn),
            "same_turn_switch_rate": len(same_turn) / len(decisive) if decisive else None,
            "stale_mode_hold_count": len(decisive) - len(same_turn)},
    }}


def apply_feedback_labels(
    candidates: list[dict[str, Any]],
    feedback: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Resolve candidates only by exact stable ID and explicit human feedback."""
    labels = {
        str(item.get("candidate_id")): item
        for item in feedback
        if item.get("candidate_id")
        and item.get("reviewer")
        and item.get("source") == "manual_feedback"
        and item.get("expected_mode") in {"daily", "work", "sex"}
        and isinstance(item.get("expected_transition"), str)
        and item.get("expected_transition")
    }
    resolved: list[dict[str, Any]] = []
    for candidate in candidates:
        row = dict(candidate)
        label = labels.get(str(row.get("candidate_id")))
        if label:
            row["needs_manual_label"] = False
            row["manual_label"] = dict(label)
        resolved.append(row)
    return resolved


def apply_audit_gate(report: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Downgrade promotion when unresolved runtime evidence exists."""
    unresolved = [item for item in candidates if item.get("needs_manual_label") is True]
    reason_counts = Counter(
        reason
        for item in unresolved
        for reason in item.get("reason_codes") or []
    )
    report["audit_health"] = {
        "unresolved_candidate_count": len(unresolved),
        "possible_host_continuity_reset_count": reason_counts["possible_host_continuity_reset"],
        "continuity_unknown_count": reason_counts["continuity_unknown"],
        "rapid_mode_reversal_count": reason_counts["rapid_mode_reversal"],
        "high_risk_mode_jump_count": reason_counts["high_risk_mode_jump"],
        "semantic_disagreement_count": reason_counts["semantic_disagreement"],
        "classifier_final_disagreement_count": reason_counts["classifier_final_disagreement"],
    }
    if unresolved:
        report.setdefault("promotion", {})["status"] = "blocked"
        report["promotion"]["blocked_by"] = "unresolved_runtime_audit_candidates"
    return report


def append_manual_feedback(path: str | Path, label: dict[str, Any]) -> bool:
    """Append one human label idempotently; never modifies routing rules."""
    required = {"candidate_id", "message", "expected_mode", "expected_transition", "reviewer"}
    missing = sorted(required.difference(label))
    if missing:
        raise ValueError(f"manual feedback missing fields: {', '.join(missing)}")
    if label["expected_mode"] not in {"daily", "work", "sex"}:
        raise ValueError("expected_mode must be daily, work, or sex")
    target = Path(path)
    existing = load_jsonl(target) if target.exists() else []
    if any(item.get("candidate_id") == label["candidate_id"] for item in existing):
        return False
    row = {**label, "source": "manual_feedback"}
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".tmp")
    temp.write_text(
        "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in [*existing, row]),
        encoding="utf-8",
    )
    temp.replace(target)
    return True


def _write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", default="tests/fixtures/human_truth_cases.jsonl")
    parser.add_argument("--audit-log")
    parser.add_argument("--report", required=True)
    parser.add_argument("--candidates")
    parser.add_argument("--feedback")
    parser.add_argument("--sample")
    parser.add_argument("--sampling-key")
    parser.add_argument("--sample-size", type=int, default=12)
    parser.add_argument("--sample-report")
    args = parser.parse_args()

    _write_json(args.report, {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "authority": AUTHORITY,
        "promotion": {
            "status": "blocked",
            "blocked_by": "evaluation_in_progress",
            "automatic_rule_write": False,
            "automatic_activation": False,
            "requires_human_approval": True,
        },
    })
    truth_path = ROOT / args.truth if not Path(args.truth).is_absolute() else Path(args.truth)
    report = build_shadow_report(load_jsonl(truth_path))
    if args.audit_log and args.candidates:
        candidates = extract_audit_candidates(load_jsonl(args.audit_log))
        if args.feedback and Path(args.feedback).exists():
            candidates = apply_feedback_labels(candidates, load_jsonl(args.feedback))
        report = apply_audit_gate(report, candidates)
        _write_json(args.candidates, {
            "authority": AUTHORITY,
            "candidates": candidates,
        })
    if args.audit_log and args.sample and args.sampling_key:
        sample = build_continuous_sample(load_jsonl(args.audit_log), sample_size=args.sample_size, sampling_key=args.sampling_key)
        _write_json(args.sample, {"authority": "production_runtime_sample_for_manual_labeling", "sampling_key": args.sampling_key, "sample": sample})
        if args.sample_report:
            feedback = load_jsonl(args.feedback) if args.feedback and Path(args.feedback).exists() else []
            _write_json(args.sample_report, build_labeled_runtime_report(sample, feedback))
    _write_json(args.report, report)
    print(json.dumps({"report": args.report, "metrics": report["metrics"], "promotion": report["promotion"]}, ensure_ascii=False))
    return 0 if report["promotion"]["status"] == "manual_review_candidate" else 2


if __name__ == "__main__":
    raise SystemExit(main())
