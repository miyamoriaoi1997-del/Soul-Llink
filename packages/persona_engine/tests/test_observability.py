import json

from persona_orchestrator import StatePacket
from persona_orchestrator.observability import OrchestratorLogger


def test_orchestrator_logger_writes_jsonl(tmp_path):
    log_path = tmp_path / 'shadow.jsonl'
    packet = StatePacket(
        mode='daily',
        submode='default',
        confidence=0.55,
        reason='test',
        transition='start:daily',
        selected_layers=['core', 'daily'],
        memory_profile='core_relationship',
        safety_flags=[],
        emotion_score=None,
        desire_tier='unknown',
        prompt_hash='abc123',
    )

    OrchestratorLogger(log_path).log(packet, extra={'sample': True})

    lines = log_path.read_text(encoding='utf-8').splitlines()
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data['packet']['mode'] == 'daily'
    assert data['packet']['prompt_hash'] == 'abc123'
    assert data['extra']['sample'] is True
    assert 'timestamp' in data
