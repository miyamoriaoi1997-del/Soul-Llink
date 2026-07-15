#!/usr/bin/env python3
"""Analyze eval fixture coverage against classifier/transition code paths."""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_cases():
    cases = []
    path = ROOT / "tests/fixtures/state_machine_eval_cases.jsonl"
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))
    return cases


def extract_code_paths():
    """Extract all possible submodes and transition labels from source code."""
    classifier_path = ROOT / "persona_orchestrator/mode_classifier.py"
    transition_path = ROOT / "persona_orchestrator/transition_manager.py"

    with classifier_path.open() as f:
        classifier_code = f.read()
    with transition_path.open() as f:
        transition_code = f.read()

    # All submodes defined in classifier
    submodes = set(re.findall(r'submode="([^"]+)"', classifier_code))

    # All safety_flags emitted by classifier
    flag_lists = re.findall(r'safety_flags=\["([^"]+)"\]', classifier_code)
    classifier_flags = set(flag_lists)

    # Transition labels from transition_manager
    # Look for transition= assignments and _transition_label calls
    transition_literals = set(re.findall(r'transition="([^"]+)"', transition_code))
    # Also look for f-string patterns like f"{prev}->{new}"
    # and hardcoded labels
    hold_labels = set(re.findall(r'"(hold_[a-z_]+)"', transition_code))
    blocked_labels = set(re.findall(r'"(blocked:[a-z_]+)"', transition_code))
    stay_labels = set(re.findall(r'"(stay:[a-z_]+)"', transition_code))
    start_labels = set(re.findall(r'"(start:[a-z_]+)"', transition_code))

    all_transitions = transition_literals | hold_labels | blocked_labels | stay_labels | start_labels

    return submodes, classifier_flags, all_transitions


def analyze_coverage(cases, submodes, classifier_flags, code_transitions):
    """Cross-reference fixtures against code paths."""
    # What fixtures cover
    fixture_modes = Counter()
    fixture_transitions = Counter()
    fixture_flags = Counter()
    fixture_prev_modes = Counter()
    fixture_emotion_buckets = Counter()
    fixture_mode_pairs = Counter()

    for c in cases:
        fixture_modes[c["expected_mode"]] += 1
        fixture_transitions[c["expected_transition"]] += 1
        prev = c.get("previous_mode") or "None"
        fixture_prev_modes[prev] += 1
        fixture_mode_pairs[(prev, c["expected_mode"])] += 1

        for flag in c.get("expected_flags", []):
            fixture_flags[flag] += 1

        es = c.get("emotion_score", 0)
        if es <= 2:
            bucket = "low(0-2)"
        elif es <= 4:
            bucket = "mid(2-4)"
        elif es <= 6:
            bucket = "high(4-6)"
        else:
            bucket = "very_high(6+)"
        fixture_emotion_buckets[bucket] += 1

    # Gap analysis
    covered_transitions = set(fixture_transitions.keys()) - {"*"}
    uncovered_transitions = code_transitions - covered_transitions

    covered_flags = set(fixture_flags.keys())
    uncovered_flags = classifier_flags - covered_flags

    return {
        "total_cases": len(cases),
        "mode_distribution": dict(fixture_modes.most_common()),
        "emotion_distribution": dict(sorted(fixture_emotion_buckets.items())),
        "mode_pair_matrix": {
            f"{prev}->{exp}": count
            for (prev, exp), count in fixture_mode_pairs.most_common()
        },
        "top_transitions": dict(fixture_transitions.most_common(20)),
        "unique_transitions_in_fixtures": len(fixture_transitions),
        "unique_transitions_in_code": len(code_transitions),
        "flags_in_fixtures": dict(fixture_flags.most_common()),
        "flags_in_code": sorted(classifier_flags),
        "submodes_in_code": sorted(submodes),
        "gaps": {
            "uncovered_transitions": sorted(uncovered_transitions),
            "uncovered_flags": sorted(uncovered_flags),
            "missing_mode_pairs": [
                f"{prev}->{exp}"
                for prev in ["None", "daily", "work", "sex"]
                for exp in ["daily", "work", "sex"]
                if (prev, exp) not in fixture_mode_pairs
            ],
            "low_coverage_transitions": [
                f"{t} ({c})"
                for t, c in fixture_transitions.most_common()
                if c < 5 and t != "*"
            ],
        },
    }


def main():
    cases = load_cases()
    submodes, classifier_flags, code_transitions = extract_code_paths()
    report = analyze_coverage(cases, submodes, classifier_flags, code_transitions)

    if "--json" in sys.argv:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    print(f"=== Eval Coverage Report ===")
    print(f"Total cases: {report['total_cases']}")
    print()

    print("Mode distribution:")
    for k, v in report["mode_distribution"].items():
        pct = v / report["total_cases"] * 100
        print(f"  {k}: {v} ({pct:.1f}%)")

    print()
    print("Emotion score distribution:")
    for k, v in report["emotion_distribution"].items():
        print(f"  {k}: {v}")

    print()
    print("Mode pair matrix (prev->expected):")
    for k, v in report["mode_pair_matrix"].items():
        print(f"  {k}: {v}")

    print()
    print(f"Unique transitions in fixtures: {report['unique_transitions_in_fixtures']}")
    print(f"Unique transitions in code: {report['unique_transitions_in_code']}")

    print()
    print("Classifier submodes in code:")
    for s in report["submodes_in_code"]:
        print(f"  - {s}")

    print()
    print("Flags in code vs fixtures:")
    print(f"  Code: {report['flags_in_code']}")
    print(f"  Fixtures: {list(report['flags_in_fixtures'].keys())}")

    print()
    print("=== GAPS ===")
    gaps = report["gaps"]

    if gaps["uncovered_transitions"]:
        print(f"Uncovered transitions ({len(gaps['uncovered_transitions'])}):")
        for t in gaps["uncovered_transitions"]:
            print(f"  ✗ {t}")
    else:
        print("All code transitions covered ✓")

    if gaps["uncovered_flags"]:
        print(f"Uncovered flags:")
        for f in gaps["uncovered_flags"]:
            print(f"  ✗ {f}")
    else:
        print("All code flags covered ✓")

    if gaps["missing_mode_pairs"]:
        print(f"Missing mode pairs:")
        for p in gaps["missing_mode_pairs"]:
            print(f"  ✗ {p}")
    else:
        print("All mode pairs covered ✓")

    if gaps["low_coverage_transitions"]:
        print(f"Low coverage transitions (<5 cases):")
        for t in gaps["low_coverage_transitions"]:
            print(f"  ⚠ {t}")


if __name__ == "__main__":
    raise SystemExit(main() or 0)
