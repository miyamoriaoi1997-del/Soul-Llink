#!/usr/bin/env python3
"""Evaluate full persona state-machine behavior against JSONL fixtures."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
for path in (ROOT, REPO_ROOT / "packages", REPO_ROOT / "adapters"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from persona_orchestrator import StateOrchestrator


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with path.open('r', encoding='utf-8') as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            case['_line'] = line_no
            cases.append(case)
    return cases


def _packet_for_case(orchestrator: StateOrchestrator, case: dict[str, Any]):
    return orchestrator.analyze_turn(
        user_message=case['message'],
        recent_messages=case.get('recent_context'),
        emotion_state={'emotion_score': case.get('emotion_score')},
        previous_mode=case.get('previous_mode'),
    )


def evaluate(cases: list[dict[str, Any]]) -> dict[str, Any]:
    orchestrator = StateOrchestrator(
        '.',
        log_path=ROOT / 'logs' / 'state_machine_eval.jsonl',
        enable_semantic_shadow=True,
        semantic_backend='local_lightweight',
        core_source='host_core',
    )
    failures: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    confusion: Counter[tuple[str, str]] = Counter()
    metric_pass = Counter()

    for case in cases:
        packet = _packet_for_case(orchestrator, case)
        expected_layers = case.get('expected_layers', [])
        forbidden = case.get('forbidden_layers', [])
        expected_flags = case.get('expected_flags', [])
        missing_layers = [layer for layer in expected_layers if layer not in packet.selected_layers]
        forbidden_layers = [layer for layer in forbidden if layer in packet.selected_layers]
        missing_flags = [flag for flag in expected_flags if flag not in packet.safety_flags]

        mode_ok = packet.mode == case['expected_mode']
        transition_ok = case['expected_transition'] == '*' or packet.transition == case['expected_transition']
        layers_ok = not missing_layers and not forbidden_layers
        flags_ok = not missing_flags
        confusion[(case['expected_mode'], packet.mode)] += 1
        metric_pass['final_mode'] += int(mode_ok)
        metric_pass['transition'] += int(transition_ok)
        metric_pass['layers'] += int(layers_ok)
        metric_pass['flags'] += int(flags_ok)

        row = {
            'id': case.get('id'),
            'line': case.get('_line'),
            'message': case['message'],
            'expected_mode': case['expected_mode'],
            'actual_mode': packet.mode,
            'expected_transition': case['expected_transition'],
            'actual_transition': packet.transition,
            'expected_layers': expected_layers,
            'actual_layers': packet.selected_layers,
            'forbidden_layers': forbidden,
            'missing_layers': missing_layers,
            'violated_forbidden_layers': forbidden_layers,
            'expected_flags': expected_flags,
            'actual_flags': packet.safety_flags,
            'missing_flags': missing_flags,
            'semantic_shadow': packet.semantic_shadow,
            'reason': packet.reason,
            'ok': mode_ok and transition_ok and layers_ok and flags_ok,
        }
        rows.append(row)
        if not row['ok']:
            failures.append(row)

    total = len(cases)
    return {
        'total': total,
        'passed': total - len(failures),
        'failed': len(failures),
        'accuracy': {
            'final_mode': metric_pass['final_mode'] / total if total else 0.0,
            'transition': metric_pass['transition'] / total if total else 0.0,
            'layers': metric_pass['layers'] / total if total else 0.0,
            'flags': metric_pass['flags'] / total if total else 0.0,
            'overall': (total - len(failures)) / total if total else 0.0,
        },
        'confusion': {f'{expected}->{actual}': count for (expected, actual), count in sorted(confusion.items())},
        'semantic_shadow': [
            {
                'id': row['id'],
                'semantic_shadow': row['semantic_shadow'],
            }
            for row in rows
        ],
        'failures': failures,
        'rows': rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', default='tests/fixtures/human_truth_cases.jsonl')
    parser.add_argument('--json', action='store_true', help='emit machine-readable JSON')
    args = parser.parse_args()

    cases_path = ROOT / args.cases
    summary = evaluate(load_cases(cases_path))
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        accuracy = summary['accuracy']
        print(
            f"total={summary['total']} passed={summary['passed']} failed={summary['failed']} "
            f"overall={accuracy['overall']:.3f} mode={accuracy['final_mode']:.3f} "
            f"transition={accuracy['transition']:.3f} layers={accuracy['layers']:.3f}"
        )
        print('confusion:')
        for key, count in summary['confusion'].items():
            print(f'  {key}: {count}')
        if summary['failures']:
            print('failures:')
            for failure in summary['failures']:
                print(
                    f"  {failure['id']}: mode {failure['expected_mode']}->{failure['actual_mode']} "
                    f"transition {failure['expected_transition']}->{failure['actual_transition']} "
                    f"missing_layers={failure['missing_layers']} forbidden={failure['violated_forbidden_layers']}"
                )
    return 1 if summary['failed'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
