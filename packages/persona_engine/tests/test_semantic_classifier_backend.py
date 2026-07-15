from persona_orchestrator.semantic_classifier import SemanticModeClassifier


def test_semantic_classifier_uses_local_shadow_by_default():
    result = SemanticModeClassifier().classify('[pet name]帮我看一下 gateway 日志')

    assert result['backend'] == 'deterministic-local-shadow'
    assert result['shadow_only'] is True
    assert result['primary_mode'] == 'work'
    assert result['affective_overlay'] == 'daily'


def test_semantic_classifier_llm_backend_is_explicit_and_does_not_call_when_disabled():
    result = SemanticModeClassifier(backend='llm', llm_enabled=False).classify('状态机是不是硬编码')

    assert result['backend'] == 'llm-disabled-fallback'
    assert result['shadow_only'] is True
    assert result['primary_mode'] == 'work'
    assert 'LLM_DISABLED' in result['reason_codes']


def test_semantic_classifier_accepts_model_config_without_requiring_network_call():
    classifier = SemanticModeClassifier(
        backend='llm',
        llm_enabled=False,
        model='gpt-5.5-mini',
        provider='custom',
        base_url='http://example.invalid/v1',
        api_key='dummy',
    )

    result = classifier.classify('没用这个词会不会触发 conflict')

    assert result['backend'] == 'llm-disabled-fallback'
    assert result['model'] == 'gpt-5.5-mini'
    assert result['provider'] == 'custom'


def test_local_lightweight_backend_adds_sentiment_signal_without_online_call():
    class FakeSentiment:
        label = 'caring'
        label_zh = '关切语调'
        confidence = 0.91
        valence = 1
        inference_ms = 3.2

    class FakeAnalyzer:
        def analyze(self, text):
            assert text == '[pet name]帮我看一下 gateway 日志'
            return FakeSentiment()

    result = SemanticModeClassifier(backend='local_lightweight', sentiment_analyzer=FakeAnalyzer()).classify(
        '[pet name]帮我看一下 gateway 日志'
    )

    assert result['backend'] == 'rules+local-lightweight'
    assert result['primary_mode'] == 'work'
    assert result['affective_overlay'] == 'daily'
    assert result['local_sentiment']['label'] == 'caring'
    assert result['local_sentiment']['confidence'] == 0.91
    assert 'LOCAL_SENTIMENT:caring' in result['reason_codes']


def test_local_lightweight_backend_falls_back_when_model_unavailable():
    class MissingAnalyzer:
        def analyze(self, text):
            return None

    result = SemanticModeClassifier(backend='local_lightweight', sentiment_analyzer=MissingAnalyzer()).classify('嗯')

    assert result['backend'] == 'rules+local-lightweight'
    assert result['local_sentiment'] is None
    assert 'LOCAL_SENTIMENT_UNAVAILABLE' in result['reason_codes']
