import json

from scripts import orchestrator_probe


def test_probe_accepts_semantic_shadow_arguments(capsys, tmp_path):
    code = orchestrator_probe.main([
        '[pet name]帮我看 gateway 日志',
        '--score',
        '1.0',
        '--semantic-shadow',
        '--semantic-backend',
        'local',
        '--log-path',
        str(tmp_path / 'probe.jsonl'),
    ])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload['mode'] == 'work'
    assert payload['selected_layers'] == ['core', 'work']
    assert payload['semantic_shadow']['backend'] == 'deterministic-local-shadow'
    assert payload['semantic_shadow']['shadow_only'] is True


def test_probe_local_lightweight_backend_can_fallback_without_sentiment(capsys, tmp_path):
    code = orchestrator_probe.main([
        '继续',
        '--previous-mode',
        'work',
        '--semantic-shadow',
        '--semantic-backend',
        'local_lightweight',
        '--log-path',
        str(tmp_path / 'probe.jsonl'),
    ])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload['mode'] == 'work'
    assert payload['transition'] == 'hold_short_message'
    assert payload['semantic_shadow']['backend'] == 'rules+local-lightweight'
    assert payload['semantic_shadow']['shadow_only'] is True
