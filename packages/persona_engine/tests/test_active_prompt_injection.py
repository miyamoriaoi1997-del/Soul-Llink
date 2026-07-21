from persona_orchestrator.state_orchestrator import StateOrchestrator


BASE_PROMPT = """# Existing SOUL
old identity text

<emotion_modifier>old</emotion_modifier>
"""

NEW_EMOTION = "<emotion_modifier>new</emotion_modifier>"


def test_active_injection_replaces_existing_emotion_block_and_selects_work_layer(tmp_path):
    result = StateOrchestrator('.', log_path=tmp_path / 'o.jsonl').compose_active_prompt(
        host_system_prompt=BASE_PROMPT,
        user_message='帮我检查 gateway 日志',
        emotion_state={'emotion_score': 1.0},
        emotion_modifier=NEW_EMOTION,
        previous_mode='daily',
        platform='cli',
    )

    assert result.packet.mode == 'work'
    assert result.packet.shadow_only is False
    assert result.prompt_text.count('<emotion_modifier>') == 1
    assert '<emotion_modifier>old</emotion_modifier>' not in result.prompt_text
    assert '<host_runtime_boundary>' in result.prompt_text
    assert '<soul_runtime_boundary>' in result.prompt_text
    assert result.prompt_text.index('<host_runtime_boundary>') < result.prompt_text.index('<soul_runtime_boundary>')
    assert result.prompt_text.index('<persona_orchestrator_prompt>') < result.prompt_text.index('<emotion_modifier>new</emotion_modifier>')
    assert result.prompt_text.index('</soul_runtime_boundary>') < result.prompt_text.index('<emotion_modifier>new</emotion_modifier>')
    assert '# Core Identity Layer' in result.prompt_text
    assert '# Work Mode Layer' in result.prompt_text
    assert '# Daily Mode Layer' not in result.prompt_text


def test_active_injection_loads_sex_layer_when_high_score(tmp_path):
    result = StateOrchestrator('.', log_path=tmp_path / 'o.jsonl').compose_active_prompt(
        host_system_prompt=BASE_PROMPT,
        user_message='我们做爱',
        emotion_state={'emotion_score': 80.0},
        emotion_modifier=NEW_EMOTION,
        previous_mode='daily',
        platform='cli',
    )

    assert result.packet.mode == 'sex'
    assert result.packet.shadow_only is False
    assert 'sex_shadow_only' not in result.packet.safety_flags
    assert '# Adult Boundary Layer' in result.prompt_text
    assert '# Daily Mode Layer' not in result.prompt_text


def test_active_injection_preserves_host_prompt_outside_managed_region(tmp_path):
    result = StateOrchestrator('.', log_path=tmp_path / 'o.jsonl').compose_active_prompt(
        host_system_prompt=BASE_PROMPT,
        user_message='嗯',
        emotion_state={'emotion_score': 1.0},
        emotion_modifier=NEW_EMOTION,
        previous_mode=None,
        platform='cli',
    )

    assert '# Existing SOUL' in result.prompt_text
    assert 'old identity text' in result.prompt_text
    assert result.prompt_text.index('<host_runtime_boundary>') < result.prompt_text.index('</host_runtime_boundary>')
    assert result.prompt_text.index('# Existing SOUL') < result.prompt_text.index('</host_runtime_boundary>')
    assert result.prompt_text.index('<soul_runtime_boundary>') < result.prompt_text.index('<persona_orchestrator_prompt>')
    assert '<persona_orchestrator_prompt>' in result.prompt_text
    assert '</persona_orchestrator_prompt>' in result.prompt_text


def test_active_injection_uses_host_pcltm_context_not_managed_memory_view(tmp_path):
    host = BASE_PROMPT + "\n<pcltm_context>\n[system]\nhost memory\n</pcltm_context>\n"
    result = StateOrchestrator('.', log_path=tmp_path / 'o.jsonl').compose_active_prompt(
        host_system_prompt=host,
        user_message='帮我检查 gateway 日志',
        emotion_state={'emotion_score': 1.0},
        emotion_modifier=NEW_EMOTION,
        previous_mode='daily',
        platform='cli',
    )

    assert '<pcltm_memory_view>' not in result.prompt_text
    assert '<memory_profile_notes>' not in result.prompt_text
    assert result.prompt_text.count('<pcltm_context_boundary>') == 1
    assert result.prompt_text.count('</pcltm_context_boundary>') == 1
    assert result.prompt_text.count('<pcltm_context>') == 1
    assert result.prompt_text.count('</pcltm_context>') == 1
    assert result.prompt_text.index('<host_runtime_boundary>') < result.prompt_text.index('<pcltm_context_boundary>')
    assert result.prompt_text.index('<pcltm_context_boundary>') < result.prompt_text.index('<soul_runtime_boundary>')
    assert result.packet.selected_layers == ['core', 'work']


def test_active_injection_strips_legacy_user_memory_blocks(tmp_path):
    host = BASE_PROMPT + "\n[USER MEMORY]\nlegacy memory view\n[/USER MEMORY]\n"
    result = StateOrchestrator('.', log_path=tmp_path / 'o.jsonl').compose_active_prompt(
        host_system_prompt=host,
        user_message='帮我检查 gateway 日志',
        emotion_state={'emotion_score': 1.0},
        emotion_modifier=NEW_EMOTION,
        previous_mode='daily',
        platform='cli',
    )

    assert '[USER MEMORY]' not in result.prompt_text
    assert 'legacy memory view' not in result.prompt_text
    assert result.prompt_text.count('<pcltm_context>') <= 1


def test_active_prompt_context_summary_available_via_packet(tmp_path):
    result = StateOrchestrator('.', log_path=tmp_path / 'o.jsonl').compose_active_prompt(
        host_system_prompt=BASE_PROMPT,
        user_message='帮我检查 gateway 日志',
        emotion_state={'emotion_score': 1.0},
        emotion_modifier=NEW_EMOTION,
        previous_mode='daily',
        platform='cli',
    )

    summary = result.packet.route_metadata

    assert summary['hermes_route_bucket'] == 'task'
    assert all('model' not in key for key in summary)
    assert 'memory_context_summary' in summary
    assert summary['memory_context_summary']['active_layers'] == ['system', 'pinned', 'transient']
    assert summary['memory_context_summary']['selected_layers'] == ['system', 'pinned', 'transient']
    assert summary['memory_context_summary']['selection_contract']['layers'] == ['system', 'pinned', 'episodic', 'transient']
    assert summary['memory_context_summary']['selection_contract']['archival_layers'] == ['episodic']
    assert summary['memory_context_summary']['prompt_active_layers'] == ['system', 'pinned', 'transient']
    assert 'runtime_boundary' in summary['memory_context_summary']['selected_buckets']
    assert 'project_path' in summary['memory_context_summary']['selected_buckets']
    assert summary['memory_context_summary']['compression']['is_reference_only'] is True
    assert all(layer['layer'] != 'compression' for layer in summary['memory_context_summary']['layers'])

    audit = summary['decision_audit']
    assert audit['previous_mode'] == 'daily'
    assert audit['classifier']['mode'] in {'daily', 'work', 'sex'}
    assert audit['context_router']['enabled'] is True
    assert audit['transition']['active_mode'] == result.packet.mode
    assert audit['emotion_influence']['emotion_score'] == 1.0
    assert audit['emotion_influence']['affects_tone'] is True
    assert audit['emotion_influence']['mode_authority'] == 'gated_supporting_signal'
    assert audit['memory_selection']['profile']
    assert 'reason_codes' in audit and audit['reason_codes']


def test_active_prompt_decision_audit_explains_sex_to_work_recovery(tmp_path):
    result = StateOrchestrator('.', log_path=tmp_path / 'o.jsonl').compose_active_prompt(
        host_system_prompt=BASE_PROMPT,
        user_message='继续验收状态机和动态情绪联动',
        emotion_state={'emotion_score': 4.5},
        emotion_modifier=NEW_EMOTION,
        previous_mode='sex',
        platform='cli',
    )

    audit = result.packet.route_metadata['decision_audit']

    assert result.packet.mode == 'work'
    assert audit['previous_mode'] == 'sex'
    assert audit['transition']['active_mode'] == 'work'
    assert audit['transition']['work_override'] is True
    assert 'work_override' in audit['reason_codes']
    assert audit['emotion_influence']['intensity'] == 'overwhelming'
    assert audit['memory_selection']['selected_soul_layers'][-1] == 'work'


def test_active_prompt_does_not_reintroduce_legacy_memory_blocks(tmp_path):
    result = StateOrchestrator('.', log_path=tmp_path / 'o.jsonl').compose_active_prompt(
        host_system_prompt=BASE_PROMPT,
        user_message='继续处理 PCLTM 和 DAC 的适配',
        emotion_state={'emotion_score': 1.0},
        emotion_modifier=NEW_EMOTION,
        previous_mode='work',
        platform='cli',
    )

    legacy_tag = ''.join(['<relationship_', 'moments>'])
    legacy_close_tag = ''.join(['</relationship_', 'moments>'])
    legacy_memory_kind = ''.join(['relationship_', 'moment'])
    legacy_domain = ''.join(['MO', 'MENTS', '.md'])
    assert '<memory_profile_notes>' not in result.prompt_text
    assert legacy_tag not in result.prompt_text
    assert legacy_close_tag not in result.prompt_text
    assert legacy_memory_kind not in result.prompt_text
    assert '<dac_context>' not in result.prompt_text
    assert '<handoff>' not in result.prompt_text
    assert 'CONTEXT COMPACTION' not in result.prompt_text
    assert legacy_domain not in result.prompt_text
