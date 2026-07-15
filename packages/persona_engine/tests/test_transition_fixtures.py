import json
import pytest
from pathlib import Path

from persona_orchestrator import StateOrchestrator


ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / 'tests' / 'fixtures' / 'human_truth_cases.jsonl'


def _require_public_fixture() -> None:
    if not CASES.exists():
        pytest.skip(f"public OSS fixture not present: {CASES}")


def test_transition_fixture_cases_match_expected_active_modes(tmp_path):
    _require_public_fixture()
    failures = []
    orchestrator = StateOrchestrator('.', log_path=tmp_path / 'transitions.jsonl')
    for line in CASES.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        packet = orchestrator.analyze_turn(
            user_message=case['message'],
            emotion_state={'emotion_score': case.get('emotion_score')},
            previous_mode=case.get('previous_mode'),
        )
        transition_ok = case['expected_transition'] == '*' or packet.transition == case['expected_transition']
        if packet.mode != case['expected_mode'] or not transition_ok:
            failures.append((case['id'], case['expected_mode'], packet.mode, case['expected_transition'], packet.transition, packet.reason))

    assert failures == []
