import json
import pytest
from pathlib import Path

from persona_orchestrator import StateOrchestrator


ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / 'tests' / 'fixtures' / 'human_truth_cases.jsonl'


def _require_public_fixture() -> None:
    if not CASES.exists():
        pytest.skip(f"public OSS fixture not present: {CASES}")


def load_cases():
    _require_public_fixture()
    return [
        json.loads(line)
        for line in CASES.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]


def test_real_chat_eval_cases_match_expected_modes_and_layers(tmp_path):
    failures = []
    orchestrator = StateOrchestrator(
        ROOT,
        log_path=tmp_path / 'real_chat_eval.jsonl',
        enable_semantic_shadow=True,
        semantic_backend='local_lightweight',
        core_source='host_core',
    )
    for case in load_cases():
        packet = orchestrator.analyze_turn(
            user_message=case['message'],
            recent_messages=case.get('recent_context'),
            emotion_state={'emotion_score': case.get('emotion_score')},
            previous_mode=case.get('previous_mode'),
            platform=case.get('platform', 'telegram'),
        )
        expected_layers = case.get('expected_layers', [])
        forbidden_layers = case.get('forbidden_layers', [])
        expected_flags = case.get('expected_flags', [])
        transition_ok = case['expected_transition'] == '*' or packet.transition == case['expected_transition']
        missing_layers = [layer for layer in expected_layers if layer not in packet.selected_layers]
        violated_forbidden = [layer for layer in forbidden_layers if layer in packet.selected_layers]
        missing_flags = [flag for flag in expected_flags if flag not in packet.safety_flags]
        ok = (
            packet.mode == case['expected_mode']
            and transition_ok
            and not missing_layers
            and not violated_forbidden
            and not missing_flags
        )
        if not ok:
            failures.append({
                'id': case.get('id'),
                'expected_mode': case['expected_mode'],
                'actual_mode': packet.mode,
                'expected_transition': case['expected_transition'],
                'actual_transition': packet.transition,
                'expected_layers': expected_layers,
                'actual_layers': packet.selected_layers,
                'missing_layers': missing_layers,
                'violated_forbidden_layers': violated_forbidden,
                'expected_flags': expected_flags,
                'actual_flags': packet.safety_flags,
                'missing_flags': missing_flags,
                'reason': packet.reason,
            })

    assert failures == []


def test_real_chat_eval_fixture_documents_human_labeled_samples():
    cases = load_cases()
    assert len(cases) >= 12
    assert all(str(case.get('source', '')).startswith('human_labeled') for case in cases)
    assert all(case.get('rationale') for case in cases)

    ids = {case['id'] for case in cases}
    assert {
        'work_authorization_after_eval_task',
        'pcltm_sync_query',
        'high_emotion_technical_task_stays_system',
        'work_to_daily_explicit_affection',
    } <= ids
