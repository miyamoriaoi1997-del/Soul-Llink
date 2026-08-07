import json
import pytest
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / 'tests' / 'fixtures' / 'human_truth_cases.jsonl'


def _require_public_fixture() -> None:
    if not CASES.exists():
        pytest.skip(f"public OSS fixture not present: {CASES}")


def test_state_machine_eval_fixture_cases_match_expected_transitions_and_layers(tmp_path):
    _require_public_fixture()
    from persona_orchestrator import StateOrchestrator

    failures = []
    orchestrator = StateOrchestrator(
        '.',
        log_path=tmp_path / 'state_machine_eval.jsonl',
        enable_semantic_shadow=True,
        semantic_backend='local_lightweight',
        core_source='host_core',
    )
    for line in CASES.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        packet = orchestrator.analyze_turn(
            user_message=case['message'],
            recent_messages=case.get('recent_context'),
            emotion_state={'emotion_score': case.get('emotion_score')},
            previous_mode=case.get('previous_mode'),
        )
        missing_layers = [layer for layer in case.get('expected_layers', []) if layer not in packet.selected_layers]
        forbidden_layers = [layer for layer in case.get('forbidden_layers', []) if layer in packet.selected_layers]
        missing_flags = [flag for flag in case.get('expected_flags', []) if flag not in packet.safety_flags]
        transition_ok = case['expected_transition'] == '*' or packet.transition == case['expected_transition']
        if (
            packet.mode != case['expected_mode']
            or not transition_ok
            or missing_layers
            or forbidden_layers
            or missing_flags
        ):
            failures.append({
                'id': case['id'],
                'expected_mode': case['expected_mode'],
                'actual_mode': packet.mode,
                'expected_transition': case['expected_transition'],
                'actual_transition': packet.transition,
                'expected_layers': case.get('expected_layers', []),
                'actual_layers': packet.selected_layers,
                'missing_layers': missing_layers,
                'forbidden_layers': forbidden_layers,
                'missing_flags': missing_flags,
                'reason': packet.reason,
            })

    assert failures == []


def test_contextual_work_actions_keep_work_mode_after_implicit_optimize_request(tmp_path):
    from persona_orchestrator import StateOrchestrator

    orchestrator = StateOrchestrator(
        '.',
        log_path=tmp_path / 'implicit_work_action_eval.jsonl',
        enable_semantic_shadow=True,
        semantic_backend='local_lightweight',
        core_source='host_core',
    )
    cases = [
        ('可以你来优化吧', 'hold_context_action'),
        ('可以你来优化吧。', 'hold_context_action'),
        ('可以用来优化一波规则了吗', 'hold_context_action'),
        ('那你来优化规则吧', 'hold_context_action'),
    ]

    for message, expected_transition in cases:
        packet = orchestrator.analyze_turn(
            user_message=message,
            recent_messages=[],
            emotion_state={'emotion_score': 1.0},
            previous_mode='work',
        )
        assert packet.mode == 'work'
        assert packet.transition == expected_transition
        assert 'work' in packet.selected_layers
        assert 'daily' not in packet.selected_layers


def test_state_machine_eval_script_outputs_json_summary():
    _require_public_fixture()
    result = subprocess.run(
        [sys.executable, str(ROOT / 'scripts' / 'eval_state_machine.py'), '--json'],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads(result.stdout)
    assert data['total'] >= 10
    assert data['failed'] == 0
    assert data['accuracy']['final_mode'] == 1.0
    assert data['accuracy']['transition'] == 1.0
    assert data['accuracy']['layers'] == 1.0
    assert 'confusion' in data
    assert 'semantic_shadow' in data


def test_state_machine_eval_fixture_has_no_layer_self_contradictions():
    _require_public_fixture()
    contradictions = []
    for line in CASES.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        overlap = set(case.get('expected_layers', [])).intersection(case.get('forbidden_layers', []))
        if overlap:
            contradictions.append({'id': case['id'], 'overlap': sorted(overlap)})

    assert contradictions == []


def test_state_machine_eval_fixture_has_broad_regression_coverage():
    _require_public_fixture()
    cases = [
        json.loads(line)
        for line in CASES.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]
    assert len(cases) >= 10

    expected_modes = {case['expected_mode'] for case in cases}
    assert expected_modes <= {'daily', 'work', 'sex'}
    assert {'daily', 'work', 'sex'} <= expected_modes

    transition_pairs = {
        (case.get('previous_mode'), case['expected_mode'])
        for case in cases
        if case.get('previous_mode') != case['expected_mode']
    }
    assert {('daily', 'work'), ('work', 'daily'), ('daily', 'sex'), ('sex', 'daily'), ('sex', 'work')} <= transition_pairs

    required_case_ids = {
        'work_authorization_after_eval_task',
        'pcltm_sync_query',
        'technical_with_affection_marker',
        'meta_sensitive_not_sex',
        'high_emotion_technical_task_stays_system',
        'explicit_adult_invitation_enters_sex',
        'sex_scene_close_exits_daily',
        'sex_to_work_explicit_task',
        'work_status_continuation',
    }
    actual_case_ids = {case['id'] for case in cases}
    assert required_case_ids <= actual_case_ids
