#!/usr/bin/env python3
"""Evaluate persona mode classification against JSONL fixtures."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
for path in (ROOT, REPO_ROOT / "packages", REPO_ROOT / "adapters"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from persona_orchestrator.mode_classifier import ModeClassifier


def load_cases(path: Path) -> list[dict]:
    cases = []
    with path.open('r', encoding='utf-8') as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            case['_line'] = line_no
            cases.append(case)
    return cases


def evaluate(cases: list[dict]) -> tuple[list[dict], Counter, dict]:
    classifier = ModeClassifier()
    failures = []
    confusion = Counter()
    flag_misses = defaultdict(list)
    for case in cases:
        decision = classifier.classify(case['message'])
        expected = case['expected_mode']
        confusion[(expected, decision.mode)] += 1
        missing_flags = [flag for flag in case.get('expected_flags', []) if flag not in decision.safety_flags]
        expected_overlay = case.get('expected_overlay')
        actual_overlay = decision.signals.get('affective_overlay')
        overlay_wrong = expected_overlay is not None and actual_overlay != expected_overlay
        if decision.mode != expected or missing_flags or overlay_wrong:
            failures.append({
                'id': case.get('id'),
                'line': case['_line'],
                'message': case['message'],
                'expected_mode': expected,
                'actual_mode': decision.mode,
                'expected_flags': case.get('expected_flags', []),
                'actual_flags': decision.safety_flags,
                'missing_flags': missing_flags,
                'expected_overlay': expected_overlay,
                'actual_overlay': actual_overlay,
                'confidence': decision.confidence,
                'reason': decision.reason,
            })
            for flag in missing_flags:
                flag_misses[flag].append(case.get('id'))
    return failures, confusion, flag_misses


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', default='tests/fixtures/mode_classifier_cases.jsonl')
    parser.add_argument('--json', action='store_true', help='emit machine-readable JSON')
    parser.add_argument('--semantic-shadow', action='store_true', help='include semantic shadow classifier output')
    args = parser.parse_args()

    path = ROOT / args.cases
    cases = load_cases(path)
    failures, confusion, flag_misses = evaluate(cases)
    semantic_rows = []
    if args.semantic_shadow:
        from persona_orchestrator.semantic_classifier import SemanticModeClassifier
        semantic = SemanticModeClassifier()
        semantic_rows = [
            {"id": case.get("id"), "semantic_shadow": semantic.classify(case["message"])}
            for case in cases
        ]
    passed = len(cases) - len(failures)

    if args.json:
        print(json.dumps({
            'total': len(cases),
            'passed': passed,
            'failed': len(failures),
            'accuracy': passed / len(cases) if cases else 0.0,
            'overlay_total': sum(1 for case in cases if case.get('expected_overlay')),
            'overlay_passed': sum(
                1 for case in cases
                if case.get('expected_overlay') and ModeClassifier().classify(case['message']).signals.get('affective_overlay') == case.get('expected_overlay')
            ),
            'failures': failures,
            'semantic_shadow': semantic_rows,
            'confusion': {f'{exp}->{act}': count for (exp, act), count in sorted(confusion.items())},
            'flag_misses': flag_misses,
        }, ensure_ascii=False, indent=2))
    else:
        print(f'total={len(cases)} passed={passed} failed={len(failures)} accuracy={passed / len(cases):.3f}')
        print('confusion:')
        for (expected, actual), count in sorted(confusion.items()):
            print(f'  {expected:20s} -> {actual:20s} {count}')
        if failures:
            print('failures:')
            for failure in failures:
                print(
                    f"  {failure['id']}: expected={failure['expected_mode']} actual={failure['actual_mode']} "
                    f"missing_flags={failure['missing_flags']} reason={failure['reason']}"
                )
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
