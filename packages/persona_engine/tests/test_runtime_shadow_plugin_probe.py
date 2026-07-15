import json

from scripts import runtime_shadow_plugin_probe


def test_plugin_probe_dry_run_reports_config_without_writing(capsys, tmp_path):
    plugin_dir = tmp_path / 'runtime-shadow'

    code = runtime_shadow_plugin_probe.main([
        '--plugin-dir', str(plugin_dir),
        '--persona-engine', str(tmp_path),
        '--dry-run',
    ])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload['dry_run'] is True
    assert payload['would_write'] == [
        str(plugin_dir / 'plugin.yaml'),
        str(plugin_dir / '__init__.py'),
    ]
    assert not plugin_dir.exists()


def test_plugin_probe_can_write_shadow_plugin_files(capsys, tmp_path):
    plugin_dir = tmp_path / 'runtime-shadow'

    code = runtime_shadow_plugin_probe.main([
        '--plugin-dir', str(plugin_dir),
        '--persona-engine', str(tmp_path),
        '--write',
    ])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload['written'] is True
    assert (plugin_dir / 'plugin.yaml').exists()
    init_text = (plugin_dir / '__init__.py').read_text(encoding='utf-8')
    manifest_text = (plugin_dir / 'plugin.yaml').read_text(encoding='utf-8')
    assert 'pre_llm_call' in manifest_text
    assert 'RuntimeShadowAdapter' in init_text
    assert 'active prompt candidate' in init_text
    assert 'return None' in init_text


def test_plugin_probe_template_forwards_real_chat_audit_context(capsys, tmp_path):
    plugin_dir = tmp_path / 'runtime-shadow'

    code = runtime_shadow_plugin_probe.main([
        '--plugin-dir', str(plugin_dir),
        '--persona-engine', str(tmp_path),
        '--write',
    ])

    assert code == 0
    capsys.readouterr()
    init_text = (plugin_dir / '__init__.py').read_text(encoding='utf-8')
    assert 'previous_mode: str | None = None' in init_text
    assert 'message_timestamp: float | None = None' in init_text
    assert 'previous_mode=previous_mode' in init_text
    assert 'message_timestamp=message_timestamp' in init_text
    assert 'user_message=' not in init_text.split('adapter.analyze_runtime_turn(', 1)[0]
