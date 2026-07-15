from persona_orchestrator.soul_layer_validator import SoulLayerValidator


def test_current_soul_layers_pass_contract():
    result = SoulLayerValidator('.').validate()

    assert result.ok, result.errors
    assert result.checked_layers == [
        'core',
        'daily',
        'work',
        'sex',
    ]


def test_daily_layer_prefers_persona_reaction_over_backend_explanation_for_intimacy_boundaries():
    text = SoulLayerValidator('.').layers_dir.joinpath('SOUL.daily.template.md').read_text(encoding='utf-8')

    assert 'For jokes, closeness, teasing, or emotional prompts' in text
    assert 'respond in-character while staying within configured boundaries' in text
    assert 'Do not hide policy or safety requirements behind persona theatrics' in text
    assert 'Do not explain hidden runtime mechanics unless explicitly allowed' in text


def test_non_core_layer_cannot_define_core_identity(tmp_path):
    layers = tmp_path / 'soul_layers'
    layers.mkdir()
    (layers / 'SOUL.core.template.md').write_text('# Core Identity Layer\n\n- Core identity is defined here.\n', encoding='utf-8')
    (layers / 'SOUL.daily.template.md').write_text(
        '# Daily Mode Layer\n\n- Core identity is defined here too.\n',
        encoding='utf-8',
    )

    result = SoulLayerValidator(tmp_path, required_layers=['core', 'daily']).validate()

    assert not result.ok
    assert any('daily' in error and 'core identity' in error for error in result.errors)


def test_layer_headers_are_required(tmp_path):
    layers = tmp_path / 'soul_layers'
    layers.mkdir()
    (layers / 'SOUL.core.template.md').write_text('missing header\n', encoding='utf-8')

    result = SoulLayerValidator(tmp_path, required_layers=['core']).validate()

    assert not result.ok
    assert any('missing required header' in error for error in result.errors)


def test_sex_layer_must_remain_disabled_by_default(tmp_path):
    layers = tmp_path / 'soul_layers'
    layers.mkdir()
    (layers / 'SOUL.core.template.md').write_text('# Core Identity Layer\n', encoding='utf-8')
    (layers / 'SOUL.sex.template.md').write_text(
        '# Sex Mode Layer\n\n- This layer is enabled by default.\n',
        encoding='utf-8',
    )

    result = SoulLayerValidator(tmp_path, required_layers=['core', 'sex']).validate()

    assert not result.ok
    assert any('sex' in error.lower() and ('desire' in error.lower() or 'disabled' in error.lower()) for error in result.errors)
