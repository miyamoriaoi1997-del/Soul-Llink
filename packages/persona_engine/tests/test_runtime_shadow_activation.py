from persona_orchestrator.runtime_shadow import RuntimeShadowAdapter


def test_runtime_shadow_active_flag_does_not_enable_switch_for_task_mode(tmp_path):
    adapter = RuntimeShadowAdapter('.', log_path=tmp_path / 'runtime.jsonl')

    record = adapter.analyze_runtime_turn(
        host_system_prompt='# Host\n\n<emotion_modifier>old</emotion_modifier>',
        user_message='帮我检查 gateway 日志',
        emotion_state={'emotion_score': 1.0},
        emotion_modifier='<emotion_modifier>fresh</emotion_modifier>',
        previous_mode='daily',
        platform='cli',
        active=True,
        session_id='s1',
    )

    assert record['mode'] == 'work'
    assert record['route_bucket'] == 'task'
    assert record['model_hint'] == 'technical'
    assert record['switch_allowed'] is False
    assert record['switch_reason'] == 'runtime_shadow_observation_only'
    assert record['packet'].shadow_only is False


def test_runtime_shadow_routes_active_sex_candidate_to_sex_bucket_without_enabling_switch(tmp_path):
    adapter = RuntimeShadowAdapter('.', log_path=tmp_path / 'runtime.jsonl')

    record = adapter.analyze_runtime_turn(
        host_system_prompt='# Host\n\n<emotion_modifier>old</emotion_modifier>',
        user_message='我们做爱',
        emotion_state={'emotion_score': 80.0},
        emotion_modifier='<emotion_modifier>fresh</emotion_modifier>',
        previous_mode='daily',
        platform='cli',
        active=True,
        session_id='s2',
    )

    assert record['mode'] == 'sex'
    assert record['selected_layers'] == ['core', 'sex']
    assert record['route_bucket'] == 'sex'
    assert record['model_hint'] == 'sex'
    assert record['switch_allowed'] is False
    assert record['switch_reason'] == 'runtime_shadow_observation_only'
    assert record['packet'].shadow_only is False


def test_shadow_runtime_stays_audit_only_even_when_mode_matches(tmp_path):
    adapter = RuntimeShadowAdapter('.', log_path=tmp_path / 'runtime.jsonl')

    record = adapter.analyze_runtime_turn(
        host_system_prompt='# Host\n\n<emotion_modifier>old</emotion_modifier>',
        user_message='我们做爱',
        emotion_state={'emotion_score': 80.0},
        emotion_modifier='<emotion_modifier>fresh</emotion_modifier>',
        previous_mode='daily',
        platform='cli',
        active=False,
        session_id='s3',
    )

    assert record['mode'] == 'sex'
    assert record['switch_allowed'] is False
    assert record['switch_reason'] == 'runtime_shadow_observation_only'
    assert record['packet'].shadow_only is True
