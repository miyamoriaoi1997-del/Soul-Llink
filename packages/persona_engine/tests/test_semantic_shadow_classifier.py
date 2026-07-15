from persona_orchestrator import StateOrchestrator


def test_semantic_shadow_result_is_recorded_but_does_not_override_rule_mode(tmp_path):
    packet = StateOrchestrator('.', log_path=tmp_path / 'o.jsonl', enable_semantic_shadow=True).analyze_turn(
        user_message='[pet name]帮我看一下 gateway 日志',
        emotion_state={'emotion_score': 2.0},
        previous_mode='daily',
    )

    assert packet.mode == 'work'
    assert packet.semantic_shadow is not None
    assert packet.semantic_shadow['primary_mode'] == 'work'
    assert packet.semantic_shadow['affective_overlay'] == 'daily'
    assert packet.semantic_shadow['shadow_only'] is True
    assert 'semantic_shadow' in packet.reason


def test_semantic_shadow_can_disagree_without_taking_control(tmp_path):
    packet = StateOrchestrator('.', log_path=tmp_path / 'o.jsonl', enable_semantic_shadow=True).analyze_turn(
        user_message='状态机现在是不是硬编码匹配',
        emotion_state={'emotion_score': 1.0},
        previous_mode='daily',
    )

    assert packet.mode == 'work'
    assert packet.semantic_shadow is not None
    assert packet.semantic_shadow['shadow_only'] is True
    assert packet.semantic_shadow['primary_mode'] == 'work'


def test_state_orchestrator_can_use_local_lightweight_semantic_backend(tmp_path):
    class FakeSentiment:
        label = 'sad'
        label_zh = '悲伤语调'
        confidence = 0.93
        valence = -1
        inference_ms = 2.0

    class FakeAnalyzer:
        def analyze(self, text):
            return FakeSentiment()

    packet = StateOrchestrator(
        '.',
        log_path=tmp_path / 'o.jsonl',
        enable_semantic_shadow=True,
        semantic_backend='local_lightweight',
        sentiment_analyzer=FakeAnalyzer(),
    ).analyze_turn(
        user_message='我崩溃了，陪我',
        emotion_state={'emotion_score': 1.0},
        previous_mode='daily',
    )

    assert packet.mode == 'daily'
    assert packet.semantic_shadow is not None
    assert packet.semantic_shadow['backend'] == 'rules+local-lightweight'
    assert packet.semantic_shadow['local_sentiment']['label'] == 'sad'
    assert 'LOCAL_SENTIMENT:sad' in packet.semantic_shadow['reason_codes']


def test_semantic_shadow_disabled_by_default(tmp_path):
    packet = StateOrchestrator('.', log_path=tmp_path / 'o.jsonl').analyze_turn(
        user_message='[pet name]帮我看一下 gateway 日志',
        emotion_state={'emotion_score': 2.0},
    )

    assert packet.semantic_shadow is None
