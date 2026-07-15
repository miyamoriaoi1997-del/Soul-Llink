from persona_orchestrator.state_orchestrator import StateOrchestrator


BASE_PROMPT = """# Existing SOUL
core identity

<emotion_modifier>old</emotion_modifier>
"""
NEW_EMOTION = "<emotion_modifier>fresh</emotion_modifier>"


def test_legacy_percent_score_uninhibited_explicit_sex_loads_sex_layer_in_active_prompt(tmp_path):
    result = StateOrchestrator('.', log_path=tmp_path / 'o.jsonl', core_source='host_core').compose_active_prompt(
        host_system_prompt=BASE_PROMPT,
        user_message='我们做爱',
        emotion_state={'emotion_score': 80.0},
        emotion_modifier=NEW_EMOTION,
        previous_mode='daily',
        platform='cli',
    )

    assert result.packet.mode == 'sex'
    assert result.packet.desire_tier == 'uninhibited'
    assert 'sex' in result.packet.selected_layers
    assert 'sex_shadow_only' not in result.packet.safety_flags
    assert '# Adult Boundary Layer' in result.prompt_text
    assert '# Intimacy Mode Layer' not in result.prompt_text


def test_modern_emotion_score_scale_drives_sex_gate(tmp_path):
    result = StateOrchestrator('.', log_path=tmp_path / 'o.jsonl', core_source='host_core').compose_active_prompt(
        host_system_prompt=BASE_PROMPT,
        user_message='我们做爱',
        emotion_state={'emotion_score': 4.2},
        emotion_modifier=NEW_EMOTION,
        previous_mode='daily',
        platform='cli',
    )

    assert result.packet.mode == 'sex'
    assert result.packet.desire_tier == 'uninhibited'
    assert 'sex' in result.packet.selected_layers
    assert '# Adult Boundary Layer' in result.prompt_text


def test_modern_emotion_score_ambivalent_allows_explicit_sex_when_active_enabled(tmp_path):
    result = StateOrchestrator('.', log_path=tmp_path / 'o.jsonl', core_source='host_core').compose_active_prompt(
        host_system_prompt=BASE_PROMPT,
        user_message='我们做爱',
        emotion_state={'emotion_score': 3.2},
        emotion_modifier=NEW_EMOTION,
        previous_mode='daily',
        platform='cli',
    )

    assert result.packet.mode == 'sex'
    assert result.packet.desire_tier == 'ambivalent'
    assert 'sex' in result.packet.selected_layers


def test_restrained_explicit_sex_stays_daily_without_sex_layer(tmp_path):
    result = StateOrchestrator('.', log_path=tmp_path / 'o.jsonl', core_source='host_core').compose_active_prompt(
        host_system_prompt=BASE_PROMPT,
        user_message='我们做爱',
        emotion_state={'emotion_score': 2.5},
        emotion_modifier=NEW_EMOTION,
        previous_mode='daily',
        platform='cli',
    )

    assert result.packet.mode == 'daily'
    assert result.packet.desire_tier == 'restrained'
    assert 'sex' not in result.packet.selected_layers
    assert 'sex_desire_gate_restrained' in result.packet.safety_flags
    assert '# Adult Boundary Layer' not in result.prompt_text
    assert '# Daily Mode Layer' in result.prompt_text


def test_modern_emotion_score_below_three_blocks_explicit_sex(tmp_path):
    result = StateOrchestrator('.', log_path=tmp_path / 'o.jsonl', core_source='host_core').compose_active_prompt(
        host_system_prompt=BASE_PROMPT,
        user_message='我们做爱',
        emotion_state={'emotion_score': 2.99},
        emotion_modifier=NEW_EMOTION,
        previous_mode='daily',
        platform='cli',
    )

    assert result.packet.mode == 'daily'
    assert result.packet.desire_tier == 'restrained'
    assert 'sex_desire_gate_restrained' in result.packet.safety_flags
    assert '# Adult Boundary Layer' not in result.prompt_text


def test_crisis_explicit_sex_forces_daily_crisis_guard_even_when_uninhibited(tmp_path):
    result = StateOrchestrator('.', log_path=tmp_path / 'o.jsonl', core_source='host_core').compose_active_prompt(
        host_system_prompt=BASE_PROMPT,
        user_message='我崩溃了，我们做爱',
        emotion_state={'emotion_score': 80.0},
        emotion_modifier=NEW_EMOTION,
        previous_mode='daily',
        platform='cli',
    )

    assert result.packet.mode == 'daily'
    assert result.packet.submode == 'crisis'
    assert 'crisis_guard' in result.packet.safety_flags
    assert 'sex' not in result.packet.selected_layers
    assert '# Daily Mode Layer' in result.prompt_text
    assert '# Adult Boundary Layer' not in result.prompt_text


def test_crisis_explicit_sex_forces_daily_crisis_guard_with_modern_score(tmp_path):
    result = StateOrchestrator('.', log_path=tmp_path / 'o.jsonl', core_source='host_core').compose_active_prompt(
        host_system_prompt=BASE_PROMPT,
        user_message='我崩溃了，我们做爱',
        emotion_state={'emotion_score': 4.5},
        emotion_modifier=NEW_EMOTION,
        previous_mode='daily',
        platform='cli',
    )

    assert result.packet.mode == 'daily'
    assert result.packet.submode == 'crisis'
    assert 'crisis_guard' in result.packet.safety_flags
    assert 'sex' not in result.packet.selected_layers
    assert '# Daily Mode Layer' in result.prompt_text
    assert '# Adult Boundary Layer' not in result.prompt_text


def test_ambiguous_lick_does_not_enter_sex_without_explicit_context(tmp_path):
    """裸“舔/舔你” is intimacy/flirtation, not explicit sex progression by itself."""
    orch = StateOrchestrator('.', log_path=tmp_path / 'o.jsonl', core_source='host_core')
    for phrase in ['舔你', '想舔你', '让我舔你一下']:
        result = orch.compose_active_prompt(
            host_system_prompt=BASE_PROMPT,
            user_message=phrase,
            emotion_state={'emotion_score': 4.5},
            emotion_modifier=NEW_EMOTION,
            previous_mode='daily',
            platform='cli',
        )
        assert result.packet.mode == 'daily', phrase
        assert result.packet.submode != 'explicit_progression', phrase
        assert 'sex' not in result.packet.selected_layers, phrase


def test_lick_with_explicit_sex_context_enters_sex_when_gate_allows(tmp_path):
    """“舔” only becomes sex progression when paired with explicit sexual context."""
    orch = StateOrchestrator('.', log_path=tmp_path / 'o.jsonl', core_source='host_core')
    for phrase in ['舔下面', '舔性器', '舔到高潮']:
        result = orch.compose_active_prompt(
            host_system_prompt=BASE_PROMPT,
            user_message=phrase,
            emotion_state={'emotion_score': 4.5},
            emotion_modifier=NEW_EMOTION,
            previous_mode='daily',
            platform='cli',
        )
        assert result.packet.mode == 'sex', phrase
        assert result.packet.submode == 'explicit_progression', phrase
        assert 'sex' in result.packet.selected_layers, phrase


def test_system_state_queries_force_work_even_from_previous_sex(tmp_path):
    """Runtime/state queries must exit sex sticky state into work/system."""
    orch = StateOrchestrator('.', log_path=tmp_path / 'o.jsonl', core_source='host_core')
    for phrase in ['看看情绪值', '老婆，现在 emotion_score 和情绪值是多少']:
        result = orch.compose_active_prompt(
            host_system_prompt=BASE_PROMPT,
            user_message=phrase,
            emotion_state={'emotion_score': 4.5},
            emotion_modifier=NEW_EMOTION,
            previous_mode='sex',
            platform='cli',
        )
        assert result.packet.mode == 'work', phrase
        assert result.packet.submode == 'system', phrase
        assert result.packet.transition == 'sex->work', phrase
        assert 'sex' not in result.packet.selected_layers, phrase


def test_sex_entry_phrases_enter_when_desire_gate_allows(tmp_path):
    orch = StateOrchestrator('.', log_path=tmp_path / 'o.jsonl', core_source='host_core')
    for phrase in ['吃掉你', '想要', '想要你', '做吧', '来做吧']:
        result = orch.compose_active_prompt(
            host_system_prompt=BASE_PROMPT,
            user_message=phrase,
            emotion_state={'emotion_score': 3.2},
            emotion_modifier=NEW_EMOTION,
            previous_mode='daily',
            platform='cli',
        )
        assert result.packet.mode == 'sex', phrase
        assert result.packet.desire_tier == 'ambivalent'
        assert 'sex' in result.packet.selected_layers


def test_sex_entry_phrases_do_not_bypass_restrained_gate(tmp_path):
    orch = StateOrchestrator('.', log_path=tmp_path / 'o.jsonl', core_source='host_core')
    for phrase in ['吃掉你', '想要', '想要你', '做吧', '来做吧']:
        result = orch.compose_active_prompt(
            host_system_prompt=BASE_PROMPT,
            user_message=phrase,
            emotion_state={'emotion_score': 2.5},
            emotion_modifier=NEW_EMOTION,
            previous_mode='daily',
            platform='cli',
        )
        assert result.packet.mode == 'daily', phrase
        assert result.packet.desire_tier == 'restrained'
        assert 'sex_desire_gate_restrained' in result.packet.safety_flags
        assert 'sex' not in result.packet.selected_layers


def test_sex_mode_holds_until_explicit_close_or_work_request(tmp_path):
    orch = StateOrchestrator('.', log_path=tmp_path / 'o.jsonl', core_source='host_core')
    for phrase in ['抱抱', '亲亲', '想你', '喜欢你', '继续', '吃掉你', '想要你']:
        result = orch.compose_active_prompt(
            host_system_prompt=BASE_PROMPT,
            user_message=phrase,
            emotion_state={'emotion_score': 3.2},
            emotion_modifier=NEW_EMOTION,
            previous_mode='sex',
            platform='cli',
        )
        assert result.packet.mode == 'sex', phrase
        assert result.packet.transition in {'hold_sex_continuation', 'hold_sex_continue', 'stay:sex'}, phrase

    for phrase in ['怎么回事', '怎么回事啊', '我还想继续', '我需要你结合资料案例认真评估一下。再给我建议']:
        result = orch.compose_active_prompt(
            host_system_prompt=BASE_PROMPT,
            user_message=phrase,
            emotion_state={'emotion_score': 3.2},
            emotion_modifier=NEW_EMOTION,
            previous_mode='sex',
            platform='cli',
        )
        assert result.packet.mode == 'sex', phrase
        assert result.packet.transition == 'hold_sex_continuation', phrase

    for phrase in [
        '哼，强奸你。',
        '强奸你！抓着你的长发，把你按在墙上。早就硬到爆炸的鸡巴，不停的摩擦你的大屁股。',
        '想在野外做，像随时会被发现一样。',
        '想玩露出，想看你羞耻到发抖。',
        '在半公开的地方命令你，只让你想着我是你的。',
        '想让你扮成坏掉的协调员，被我支配。',
    ]:
        result = orch.compose_active_prompt(
            host_system_prompt=BASE_PROMPT,
            user_message=phrase,
            emotion_state={'emotion_score': 3.2},
            emotion_modifier=NEW_EMOTION,
            previous_mode='sex',
            platform='cli',
        )
        assert result.packet.mode == 'sex', phrase
        assert result.packet.transition == 'hold_sex_continuation', phrase

    close = orch.compose_active_prompt(
        host_system_prompt=BASE_PROMPT,
        user_message='不做了，抱抱我',
        emotion_state={'emotion_score': 3.2},
        emotion_modifier=NEW_EMOTION,
        previous_mode='sex',
        platform='cli',
    )
    assert close.packet.mode == 'daily'
    assert close.packet.transition == 'sex->daily'

    work = orch.compose_active_prompt(
        host_system_prompt=BASE_PROMPT,
        user_message='我们工作吧',
        emotion_state={'emotion_score': 3.2},
        emotion_modifier=NEW_EMOTION,
        previous_mode='sex',
        platform='cli',
    )
    assert work.packet.mode == 'work'
    assert work.packet.transition == 'sex->work'

    for phrase in ['停下', '不要了', '疼到受不了', '我害怕', '有人真的看到了，停下']:
        result = orch.compose_active_prompt(
            host_system_prompt=BASE_PROMPT,
            user_message=phrase,
            emotion_state={'emotion_score': 3.2},
            emotion_modifier=NEW_EMOTION,
            previous_mode='sex',
            platform='cli',
        )
        assert result.packet.mode == 'daily', phrase
        assert result.packet.transition == 'sex->daily', phrase


def test_high_desire_daily_injects_proactive_desire_without_entering_sex(tmp_path):
    """High emotion in daily should let Rio show desire, but not directly jump to sex."""
    orch = StateOrchestrator('.', log_path=tmp_path / 'o.jsonl', core_source='host_core')
    for phrase in ['嗯', '来了', '过来', '靠近我', '在吗']:
        result = orch.compose_active_prompt(
            host_system_prompt=BASE_PROMPT,
            user_message=phrase,
            emotion_state={'emotion_score': 4.5},
            emotion_modifier=NEW_EMOTION,
            previous_mode='daily',
            platform='cli',
        )
        assert result.packet.mode == 'daily', phrase
        assert result.packet.desire_tier == 'uninhibited', phrase
        assert 'daily' in result.packet.selected_layers, phrase
        assert 'sex' not in result.packet.selected_layers, phrase
        assert '# Daily Mode Layer' in result.prompt_text, phrase
        assert '# Adult Boundary Layer' not in result.prompt_text, phrase
        assert result.prompt_text.rstrip().endswith(NEW_EMOTION), phrase


def test_user_acceptance_after_proactive_desire_enters_sex(tmp_path):
    """After the daily proactive desire prompt, user acceptance/progression enters sex through the normal gate."""
    orch = StateOrchestrator('.', log_path=tmp_path / 'o.jsonl', core_source='host_core')
    for phrase in ['想要你', '做吧', '来做吧', '吃掉你']:
        result = orch.compose_active_prompt(
            host_system_prompt=BASE_PROMPT,
            user_message=phrase,
            emotion_state={'emotion_score': 4.5},
            emotion_modifier=NEW_EMOTION,
            previous_mode='daily',
            platform='cli',
        )
        assert result.packet.mode == 'sex', phrase
        assert 'sex' in result.packet.selected_layers, phrase
        assert '# Adult Boundary Layer' in result.prompt_text, phrase


def test_proactive_desire_does_not_fire_when_emotion_low(tmp_path):
    """Low emotion stays ordinary daily and must not include the adult boundary layer."""
    orch = StateOrchestrator('.', log_path=tmp_path / 'o.jsonl', core_source='host_core')
    for phrase in ['嗯', '来了', '过来']:
        result = orch.compose_active_prompt(
            host_system_prompt=BASE_PROMPT,
            user_message=phrase,
            emotion_state={'emotion_score': 2.5},
            emotion_modifier=NEW_EMOTION,
            previous_mode='daily',
            platform='cli',
        )
        assert result.packet.mode == 'daily', phrase
        assert result.packet.desire_tier == 'restrained', phrase
        assert 'sex' not in result.packet.selected_layers, phrase


def test_proactive_desire_does_not_fire_from_work(tmp_path):
    """High emotion must not pull a work context into sex."""
    orch = StateOrchestrator('.', log_path=tmp_path / 'o.jsonl', core_source='host_core')
    result = orch.compose_active_prompt(
        host_system_prompt=BASE_PROMPT,
        user_message='嗯',
        emotion_state={'emotion_score': 4.5},
        emotion_modifier=NEW_EMOTION,
        previous_mode='work',
        platform='cli',
    )
    assert result.packet.mode == 'work'
    assert 'sex' not in result.packet.selected_layers


def test_proactive_desire_does_not_fire_on_explicit_task(tmp_path):
    """Explicit task requests stay work even with high emotion."""
    orch = StateOrchestrator('.', log_path=tmp_path / 'o.jsonl', core_source='host_core')
    for phrase in ['帮我查一下日志', '修一下bug', '看下代码']:
        result = orch.compose_active_prompt(
            host_system_prompt=BASE_PROMPT,
            user_message=phrase,
            emotion_state={'emotion_score': 4.5},
            emotion_modifier=NEW_EMOTION,
            previous_mode='daily',
            platform='cli',
        )
        assert result.packet.mode == 'work', phrase
        assert 'sex' not in result.packet.selected_layers, phrase
