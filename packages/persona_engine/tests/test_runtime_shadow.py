from pathlib import Path
import json

from persona_orchestrator.runtime_shadow import RuntimeShadowAdapter


def test_runtime_shadow_adapter_returns_candidate_without_active_takeover(tmp_path):
    base_dir = Path(__file__).resolve().parents[1]
    adapter = RuntimeShadowAdapter(base_dir, log_path=tmp_path / 'runtime.jsonl')

    result = adapter.analyze_runtime_turn(
        host_system_prompt='# Host Prompt\n\n<emotion_modifier>old</emotion_modifier>',
        user_message='[pet name]帮我看 gateway 日志',
        emotion_state={'emotion_score': 1.0},
        emotion_modifier='<emotion_modifier>fresh</emotion_modifier>',
        platform='telegram',
        session_id='s1',
    )

    assert result['active'] is False
    assert result['session_id'] == 's1'
    assert result['platform'] == 'telegram'
    assert result['packet'].mode == 'work'
    assert result['packet'].selected_layers == ['core', 'work']
    assert result['packet'].shadow_only is True
    assert result['candidate_prompt'].rstrip().endswith('<emotion_modifier>fresh</emotion_modifier>')
    assert result['candidate_prompt'].count('<emotion_modifier>') == 1
    assert result['candidate_prompt_hash'] == result['prompt_hash']
    assert result['user_message_hash'] != '[pet name]帮我看 gateway 日志'
    assert len(result['user_message_hash']) == 16


def test_runtime_shadow_adapter_candidate_prompt_uses_host_pcltm_context_and_drops_legacy_notes(tmp_path):
    base_dir = Path(__file__).resolve().parents[1]
    adapter = RuntimeShadowAdapter(base_dir, log_path=tmp_path / 'runtime.jsonl')

    result = adapter.analyze_runtime_turn(
        host_system_prompt='# Host Prompt\n\n<pcltm_context>\n[system]\nhost memory\n</pcltm_context>\n\n<memory_profile_notes>legacy notes</memory_profile_notes>',
        user_message='帮我检查 gateway 日志',
        emotion_state={'emotion_score': 1.0},
        emotion_modifier='<emotion_modifier>fresh</emotion_modifier>',
        platform='telegram',
        session_id='s4',
    )

    prompt = result['candidate_prompt']
    assert '<memory_profile_notes>' not in prompt
    assert 'legacy notes' not in prompt
    assert '<pcltm_memory_view>' not in prompt
    assert prompt.count('<pcltm_context>') == 1
    assert prompt.count('</pcltm_context>') == 1
    assert result['packet'].selected_layers == ['core', 'work']
    assert result['route_bucket'] == 'task'
    assert result['model_hint'] == 'technical'

def test_runtime_shadow_adapter_logs_real_chat_audit_fields_without_raw_text(tmp_path):
    log_path = tmp_path / 'runtime.jsonl'
    adapter = RuntimeShadowAdapter('.', log_path=log_path)

    adapter.analyze_runtime_turn(
        host_system_prompt='# Host Prompt',
        user_message='继续',
        emotion_state={'emotion_score': 3.5},
        emotion_modifier='<emotion_modifier>fresh</emotion_modifier>',
        previous_mode='work',
        platform='telegram',
        session_id='s3',
        message_timestamp=1777678044.0,
    )

    log_text = log_path.read_text(encoding='utf-8')
    record = json.loads(log_text.splitlines()[-1])
    assert record.get('raw_user_message') is None
    assert record.get('user_message') is None
    assert record.get('last_user_message') is None
    assert record['message_timestamp'] == 1777678044.0
    assert record['previous_mode'] == 'work'
    assert record['confidence'] == record['packet']['confidence']
    assert record['route_bucket'] == 'task'
    assert record['model_hint'] == 'technical'
    assert record['switch_allowed'] is False
    assert record['switch_reason'] == 'runtime_shadow_observation_only'
    assert record['packet']['shadow_only'] is True
    assert len(record['user_message_hash']) == 16
