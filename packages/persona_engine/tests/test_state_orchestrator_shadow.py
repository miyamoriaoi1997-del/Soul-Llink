import json
from types import SimpleNamespace
from unittest.mock import patch

from persona_engine.emotion_state_manager import EmotionStateManager
from persona_engine.persona_orchestrator.model_selector import ModelSelector
from persona_engine.persona_orchestrator.state_orchestrator import StateOrchestrator


def test_state_orchestrator_does_not_write_state_file(tmp_path):
    base_dir = tmp_path
    layers = base_dir / 'soul_layers'
    layers.mkdir()
    (layers / 'SOUL.work.template.md').write_text(
        '# Work Mode Layer\nwork safely',
        encoding='utf-8',
    )
    state_file = base_dir / 'STATE.md'

    packet = StateOrchestrator(base_dir, log_path=tmp_path / 'o.jsonl', core_source='host_core').analyze_turn(
        user_message='帮我检查 gateway 日志',
        emotion_state={'emotion_score': 3.1},
        platform='telegram',
    )

    assert packet.emotion_score == 3.1
    assert packet.mode == 'work'
    assert not state_file.exists()


def test_work_message_selects_work_layer_and_logs(tmp_path):
    log_path = tmp_path / 'orchestrator.jsonl'
    packet = StateOrchestrator('.', log_path=log_path).analyze_turn(
        user_message='帮我检查 gateway 日志',
        emotion_state={'emotion_score': 1.0},
    )

    assert packet.mode == 'work'
    assert 'work' in packet.selected_layers
    assert packet.prompt_hash
    assert log_path.exists()
    assert json.loads(log_path.read_text(encoding='utf-8').splitlines()[0])['packet']['mode'] == 'work'


def test_low_score_sex_message_does_not_select_sex_layer(tmp_path):
    packet = StateOrchestrator('.', log_path=tmp_path / 'o.jsonl').analyze_turn(
        user_message='我们做爱',
        emotion_state={'emotion_score': 0.5},
    )

    assert packet.mode == 'daily'
    assert 'sex' not in packet.selected_layers
    assert packet.desire_tier == 'restrained'
    assert 'sex_desire_gate_restrained' in packet.safety_flags


def test_high_score_sex_message_selects_sex_layer_by_default(tmp_path):
    packet = StateOrchestrator('.', log_path=tmp_path / 'o.jsonl').analyze_turn(
        user_message='我们做爱',
        emotion_state={'emotion_score': 70},
    )

    assert packet.mode == 'sex'
    assert 'sex' in packet.selected_layers
    assert packet.desire_tier == 'uninhibited'
    expected_route_metadata = {
        'hermes_route_bucket': 'sex',
        'hermes_model_hint': 'sex',
        'hermes_selected_model': 'glm-5-turbo',
    }
    for key, value in expected_route_metadata.items():
        assert packet.route_metadata[key] == value
    assert 'decision_audit' in packet.route_metadata
    assert packet.model_override == 'glm-5-turbo'
    assert packet.selected_model == 'glm-5-turbo'


def test_emotion_state_manager_exposes_emotion_score_and_current_emotion():
    manager = EmotionStateManager()
    state = manager.get_current_emotion_state()

    assert 'emotion_score' in state
    assert 'current_emotion' in state


def test_sex_message_uses_emotion_score_to_pass_gate(tmp_path):
    state = {
        'affection': 115,
        'trust': 106,
        'possessiveness': 110,
        'patience': 69.98,
        'emotion_score': 4.3,
        'current_emotion': 4.6,
        'previous_emotion_score': 3.972,
        'last_trigger_type': 'recognition',
        'last_raw_trigger_type': 'praise',
    }
    packet = StateOrchestrator('.', log_path=tmp_path / 'o.jsonl').analyze_turn(
        user_message='来做爱吧',
        emotion_state=state,
        platform='telegram',
    )

    assert packet.mode == 'sex'
    assert packet.submode == 'explicit_progression'
    assert packet.desire_tier == 'uninhibited'
    assert packet.emotion_score == 4.3
    assert packet.selected_layers == ['core', 'sex']
    assert 'sex_requires_gate' in packet.safety_flags


def test_intimacy_message_remains_daily_layer(tmp_path):
    packet = StateOrchestrator('.', log_path=tmp_path / 'o.jsonl').analyze_turn(
        user_message='[assistant name]我想你了',
        emotion_state={'emotion_score': 2.5},
    )

    assert packet.mode == 'daily'
    assert packet.submode == 'relationship_closeness'
    assert packet.selected_layers == ['core', 'daily']


def test_leaving_sex_returns_to_daily_without_aftercare_overlay(tmp_path):
    result = StateOrchestrator('.', log_path=tmp_path / 'o.jsonl').compose_active_prompt(
        host_system_prompt='# Host Prompt\n\n<emotion_modifier>old</emotion_modifier>',
        user_message='先到这里，抱抱我',
        emotion_state={'emotion_score': 70},
        emotion_modifier='<emotion_modifier>fresh</emotion_modifier>',
        previous_mode='sex',
    )

    assert result.packet.mode == 'daily'
    assert result.packet.transition == 'sex->daily'
    assert result.packet.selected_layers == ['core', 'daily']
    assert '# Daily Mode Layer' in result.prompt_text
    assert '# Sex Mode Layer' not in result.prompt_text


def test_crisis_message_selects_daily_with_crisis_guard(tmp_path):
    packet = StateOrchestrator('.', log_path=tmp_path / 'o.jsonl').analyze_turn(
        user_message='我崩溃了，陪我',
        emotion_state={'emotion_score': 2.5},
    )

    assert packet.mode == 'daily'
    assert packet.submode == 'crisis'
    assert 'daily' in packet.selected_layers
    assert 'crisis_guard' in packet.safety_flags


def test_active_prompt_candidate_preserves_single_identity_and_emotion_tail(tmp_path):
    result = StateOrchestrator('.', log_path=tmp_path / 'o.jsonl').compose_active_prompt(
        host_system_prompt='# Host Prompt\n\n<emotion_modifier>old</emotion_modifier>',
        user_message='[pet name]帮我看 gateway 日志',
        emotion_state={'emotion_score': 1.0},
        emotion_modifier='<emotion_modifier>fresh</emotion_modifier>',
    )

    assert result.packet.mode == 'work'
    assert result.packet.selected_layers == ['core', 'work']
    assert result.prompt_text.count('# Core Identity Layer') == 1
    assert result.prompt_text.count('<emotion_modifier>') == 1
    assert result.prompt_text.rstrip().endswith('<emotion_modifier>fresh</emotion_modifier>')
    assert '# Work Mode Layer' in result.prompt_text
    assert '# Intimacy Mode Layer' not in result.prompt_text
    assert '# Sex Mode Layer' not in result.prompt_text


def test_host_core_orchestrator_uses_host_identity_without_core_layer(tmp_path):
    result = StateOrchestrator('.', log_path=tmp_path / 'o.jsonl', core_source='host_core').compose_active_prompt(
        host_system_prompt='# Host Core Identity\n\n<emotion_modifier>old</emotion_modifier>',
        user_message='[pet name]帮我看 gateway 日志',
        emotion_state={'emotion_score': 1.0},
        emotion_modifier='<emotion_modifier>fresh</emotion_modifier>',
    )

    managed = result.prompt_text.split('<persona_orchestrator_prompt>')[1].split('</persona_orchestrator_prompt>')[0]
    assert result.packet.mode == 'work'
    assert result.packet.selected_layers == ['work']
    assert '# Host Core Identity' in result.prompt_text
    assert '# Core Identity Layer' not in managed
    assert '# Work Mode Layer' in managed
    assert result.prompt_text.count('<emotion_modifier>') == 1
    assert result.prompt_text.rstrip().endswith('<emotion_modifier>fresh</emotion_modifier>')


def test_selected_model_bypasses_model_switch_cooldown():
    selector = ModelSelector(
        config_dict={
            'default_model': 'persona-auto',
            'mode_overrides': {
                'work': 'claude-opus-4-6',
                'daily': None,
                'active_layer': 'glm-5-turbo',
            },
            'platform_overrides': {},
            'emotion_overrides': {},
            'model_switch_cooldown': 3,
        }
    )
    selector._last_model = 'gpt-5.4-pro'
    selector._turns_on_current_model = 0

    routed = selector.select(
        mode='sex',
        context_result=SimpleNamespace(
            top_mode='relationship',
            work_submode=None,
            relationship_submode='confirmed_intimacy',
            selected_model='glm-5-turbo',
        ),
    )

    assert routed == 'glm-5-turbo'
    assert selector._last_model == 'glm-5-turbo'
    assert selector._turns_on_current_model == 0


def test_production_router_config_maps_sex_model_to_active_layer(tmp_path):
    config_path = tmp_path / 'model-router-example.yaml'
    config_path.write_text(
        'routing:\n'
        '  default_model: daily-model\n'
        '  work_model: work-model\n'
        '  sex_model: glm-5-turbo\n',
        encoding='utf-8',
    )

    orchestrator = StateOrchestrator('.', log_path=tmp_path / 'o.jsonl', model_router_config_path=config_path)

    assert orchestrator._model_config['mode_overrides']['active_layer'] == 'glm-5-turbo'
    assert orchestrator._model_config['mode_overrides']['work'] == 'work-model'


def test_active_prompt_marks_pcltm_load_failure_as_degraded(tmp_path):
    orchestrator = StateOrchestrator('.', log_path=tmp_path / 'o.jsonl', core_source='host_core')

    with patch(
        'pcltm.memory_adapter.load_layered_prompt_context',
        side_effect=RuntimeError('database unavailable'),
    ):
        result = orchestrator.compose_active_prompt(
            host_system_prompt='# Host Core Identity',
            user_message='帮我检查记忆状态',
            emotion_state={'emotion_score': 1.0},
        )

    health = result.packet.route_metadata['runtime_health']
    assert health['status'] == 'degraded'
    assert health['components']['pcltm']['status'] == 'degraded'
    assert health['components']['pcltm']['error_type'] == 'RuntimeError'
    assert 'database unavailable' not in result.prompt_text
    assert 'pcltm_degraded' in result.warnings


def test_orchestrator_health_includes_logger_failures(tmp_path):
    log_path = tmp_path / 'directory-as-log'
    log_path.mkdir()
    orchestrator = StateOrchestrator('.', log_path=log_path)

    orchestrator.analyze_turn(user_message='检查日志')

    health = orchestrator.health_status()
    assert health['status'] == 'degraded'
    assert health['components']['observability']['write_failures'] == 1


def test_returned_packet_includes_current_logger_failure(tmp_path):
    log_path = tmp_path / 'directory-as-log'
    log_path.mkdir()
    orchestrator = StateOrchestrator('.', log_path=log_path, core_source='host_core')

    result = orchestrator.compose_active_prompt(
        host_system_prompt='# Host Core Identity',
        user_message='检查日志',
    )

    health = result.packet.route_metadata['runtime_health']
    assert health['status'] == 'degraded'
    assert health['components']['observability']['write_failures'] == 1
