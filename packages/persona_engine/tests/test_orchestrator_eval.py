import json
import pytest
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / 'tests' / 'fixtures' / 'mode_classifier_cases.jsonl'


def _require_public_fixture() -> None:
    if not CASES.exists():
        pytest.skip(f"public OSS fixture not present: {CASES}")


def test_mode_classifier_fixture_cases_match_expected_modes():
    _require_public_fixture()
    from persona_orchestrator.mode_classifier import ModeClassifier

    classifier = ModeClassifier()
    failures = []
    for line in CASES.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        decision = classifier.classify(case['message'])
        missing_flags = [flag for flag in case.get('expected_flags', []) if flag not in decision.safety_flags]
        expected_overlay = case.get('expected_overlay')
        actual_overlay = decision.signals.get('affective_overlay')
        overlay_wrong = expected_overlay is not None and actual_overlay != expected_overlay
        if decision.mode != case['expected_mode'] or missing_flags or overlay_wrong:
            failures.append((case['id'], case['expected_mode'], decision.mode, missing_flags, expected_overlay, actual_overlay, decision.reason))

    assert failures == []


def test_orchestrator_eval_script_outputs_json_summary():
    _require_public_fixture()
    result = subprocess.run(
        [sys.executable, str(ROOT / 'scripts' / 'orchestrator_eval.py'), '--json'],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads(result.stdout)
    assert data['failed'] == 0
    assert data['accuracy'] == 1.0


def test_orchestrator_eval_script_can_emit_semantic_shadow_rows():
    _require_public_fixture()
    result = subprocess.run(
        [sys.executable, str(ROOT / 'scripts' / 'orchestrator_eval.py'), '--json', '--semantic-shadow'],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads(result.stdout)
    assert len(data['semantic_shadow']) > 0
    first = data['semantic_shadow'][0]['semantic_shadow']
    assert first['shadow_only'] is True
    assert 'primary_mode' in first
