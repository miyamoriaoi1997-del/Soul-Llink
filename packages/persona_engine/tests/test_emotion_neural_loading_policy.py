"""Regression tests for rule-first sentiment analyzer loading policy."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
import threading
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture()
def emotion_modules(monkeypatch):
    """Reload modules so singleton state is isolated per test."""
    import sentiment_analyzer
    import emotion_detector
    import emotion_state_manager

    sentiment_analyzer.SentimentAnalyzer._instance = None
    importlib.reload(sentiment_analyzer)
    importlib.reload(emotion_detector)
    importlib.reload(emotion_state_manager)
    yield sentiment_analyzer, emotion_detector, emotion_state_manager
    sentiment_analyzer.SentimentAnalyzer._instance = None


def test_manager_init_starts_background_warmup_without_blocking(tmp_path, monkeypatch, emotion_modules):
    sentiment_analyzer, emotion_detector, emotion_state_manager = emotion_modules

    calls = {"get_instance": 0, "analyze": 0}
    warmup_started = threading.Event()
    warmup_can_finish = threading.Event()

    class FakeAnalyzer:
        def analyze(self, text, *, allow_blocking_load=True):
            calls["analyze"] += 1
            warmup_started.set()
            warmup_can_finish.wait(timeout=2.0)
            return None

    def fake_get_instance(cls, model_cache_dir=None, model_id=None):
        calls["get_instance"] += 1
        return FakeAnalyzer()

    monkeypatch.setattr(
        sentiment_analyzer.SentimentAnalyzer,
        "get_instance",
        classmethod(fake_get_instance),
    )

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "SOUL.md").write_text("# Core Identity Layer\n我是测试角色。", encoding="utf-8")
    (hermes_home / "STATE.md").write_text(
        "---\nemotion_state:\n  affection: 60\n  trust: 60\n  possessiveness: 60\n  patience: 60\n---\n",
        encoding="utf-8",
    )

    start = time.perf_counter()
    emotion_state_manager.EmotionStateManager(hermes_home=hermes_home)
    elapsed = time.perf_counter() - start

    # The warmup analyzer is deliberately blocked above.  If manager init ran
    # warmup synchronously, construction would wait for warmup_can_finish (or the
    # timeout).  This assertion verifies the behavior, not a fragile absolute
    # startup budget that can flicker on loaded Windows runners.
    assert elapsed < 1.0
    assert warmup_started.wait(timeout=1.0)
    assert calls["get_instance"] == 1
    assert calls["analyze"] == 1
    assert emotion_detector.get_shared_sentiment_analyzer() is not None
    warmup_can_finish.set()


def test_strong_rule_match_skips_neural_analyzer(monkeypatch, emotion_modules):
    sentiment_analyzer, emotion_detector, _emotion_state_manager = emotion_modules

    calls = {"get_instance": 0, "analyze": 0}

    class FakeAnalyzer:
        def analyze(self, text, *, allow_blocking_load=True):
            calls["analyze"] += 1
            raise AssertionError("strong rules must not call neural analyzer")

    def fake_get_instance(cls, model_cache_dir=None, model_id=None):
        calls["get_instance"] += 1
        return FakeAnalyzer()

    monkeypatch.setattr(
        sentiment_analyzer.SentimentAnalyzer,
        "get_instance",
        classmethod(fake_get_instance),
    )

    detector = emotion_detector.EmotionDetector(
        agent_profile={"names": ["测试角色", "Test Persona"]},
        neural_policy="uncertain",
    )
    event = detector.detect_emotion_event([{"role": "user", "content": "测试角色，我爱你"}])

    assert event is not None
    assert event.trigger_type == "intimacy"
    assert detector.last_detection_source == "rules_strong"
    assert detector.last_neural_used is False
    assert calls == {"get_instance": 0, "analyze": 0}


def test_weak_or_no_rule_can_lazy_load_neural_once(monkeypatch, emotion_modules):
    sentiment_analyzer, emotion_detector, _emotion_state_manager = emotion_modules

    calls = {"get_instance": 0, "analyze": 0, "blocking": []}

    class FakeSentiment:
        label = "sad"
        label_zh = "悲伤语调"
        confidence = 0.9
        valence = -1.0
        inference_ms = 1.0

    class FakeAnalyzer:
        def analyze(self, text, *, allow_blocking_load=True):
            calls["analyze"] += 1
            calls["blocking"].append(allow_blocking_load)
            return FakeSentiment()

        def get_fusion_scale(self, trigger_type, sentiment):
            return 1.0

        def get_fallback_trigger(self, sentiment):
            return "ignored"

    shared = FakeAnalyzer()

    def fake_get_instance(cls, model_cache_dir=None, model_id=None):
        calls["get_instance"] += 1
        return shared

    monkeypatch.setattr(
        sentiment_analyzer.SentimentAnalyzer,
        "get_instance",
        classmethod(fake_get_instance),
    )

    detector = emotion_detector.EmotionDetector(neural_policy="uncertain")
    assert detector._analyzer is None

    first = detector.detect_emotion_event([{"role": "user", "content": "算了"}])
    second = detector.detect_emotion_event([{"role": "user", "content": "嗯哼"}])

    assert first is not None
    assert second is not None
    assert calls["get_instance"] == 1
    assert calls["analyze"] >= 1
    assert all(call is False for call in calls["blocking"])
    assert detector._analyzer is shared
    assert detector.last_neural_used is True
    assert detector.last_detection_source in {"rules_weak_neural", "neural_fallback"}


def test_uncertain_neural_inference_times_out_and_falls_back(monkeypatch, emotion_modules):
    sentiment_analyzer, emotion_detector, _ = emotion_modules
    release = threading.Event()

    class HangingAnalyzer:
        def analyze(self, text, *, allow_blocking_load=True):
            release.wait(timeout=2)
            return None

    monkeypatch.setattr(sentiment_analyzer.SentimentAnalyzer, "get_instance", classmethod(
        lambda cls, model_cache_dir=None, model_id=None: HangingAnalyzer()
    ))
    detector = emotion_detector.EmotionDetector(neural_policy="uncertain", neural_timeout_seconds=0.03)
    started = time.perf_counter()
    event = detector.detect_emotion_event([{"role": "user", "content": "嗯哼"}])
    elapsed = time.perf_counter() - started
    release.set()

    assert elapsed < 0.2
    assert event is None

    assert detector.last_neural_status == "timeout"
    assert detector.neural_telemetry["timeouts"] == 1
    assert detector.neural_telemetry["fallbacks"] == 1


def test_always_strong_self_target_inference_is_bounded(monkeypatch, emotion_modules):
    sentiment_analyzer, emotion_detector, _ = emotion_modules
    release = threading.Event()

    class HangingAnalyzer:
        def analyze(self, text, *, allow_blocking_load=True):
            release.wait(timeout=2)
            return None
        def get_fusion_scale(self, trigger, result):
            return 1.0

    monkeypatch.setattr(sentiment_analyzer.SentimentAnalyzer, "get_instance", classmethod(
        lambda cls, model_cache_dir=None, model_id=None: HangingAnalyzer()
    ))
    detector = emotion_detector.EmotionDetector(
        agent_profile={"names": ["凛"]},
        neural_policy="always",
        neural_timeout_seconds=0.03,
    )
    started = time.perf_counter()
    event = detector.detect_emotion_event([{"role": "user", "content": "我喜欢你"}])
    elapsed = time.perf_counter() - started
    release.set()
    assert elapsed < 0.2
    assert event is not None and event.trigger_type == "intimacy"
    assert detector.last_neural_status == "timeout"
    assert detector.neural_telemetry["timeouts"] == 1


def test_timed_out_neural_worker_is_saturated_not_queued(monkeypatch, emotion_modules):
    sentiment_analyzer, emotion_detector, _ = emotion_modules
    release = threading.Event()
    calls = 0

    class HangingAnalyzer:
        def analyze(self, text, *, allow_blocking_load=True):
            nonlocal calls
            calls += 1
            release.wait(timeout=2)
            return None

    monkeypatch.setattr(sentiment_analyzer.SentimentAnalyzer, "get_instance", classmethod(
        lambda cls, model_cache_dir=None, model_id=None: HangingAnalyzer()
    ))
    detector = emotion_detector.EmotionDetector(neural_policy="uncertain", neural_timeout_seconds=0.02)
    detector.detect_emotion_event([{"role": "user", "content": "嗯哼"}])
    started = time.perf_counter()
    detector.detect_emotion_event([{"role": "user", "content": "也许吧"}])
    elapsed = time.perf_counter() - started
    release.set()

    assert elapsed < 0.1
    assert calls == 1
    assert detector.last_neural_status == "saturated"
    assert detector.neural_telemetry["saturated"] == 1
    assert detector.neural_telemetry["fallbacks"] == 2


def test_neural_failure_is_observable_and_safe(monkeypatch, emotion_modules):
    sentiment_analyzer, emotion_detector, _ = emotion_modules

    class BrokenAnalyzer:
        def analyze(self, text, *, allow_blocking_load=True):
            raise RuntimeError("backend broke")

    monkeypatch.setattr(sentiment_analyzer.SentimentAnalyzer, "get_instance", classmethod(
        lambda cls, model_cache_dir=None, model_id=None: BrokenAnalyzer()
    ))
    detector = emotion_detector.EmotionDetector(neural_policy="uncertain", neural_timeout_seconds=0.1)
    assert detector.detect_emotion_event([{"role": "user", "content": "嗯哼"}]) is None
    assert detector.last_neural_status == "error"
    assert detector.neural_telemetry["errors"] == 1
    assert detector.neural_telemetry["fallbacks"] == 1


def test_background_warmup_is_nonblocking_and_reuses_singleton(monkeypatch, emotion_modules):
    sentiment_analyzer, _emotion_detector, emotion_state_manager = emotion_modules

    calls = {"get_instance": 0, "analyze": 0, "blocking": []}

    class FakeAnalyzer:
        def analyze(self, text, *, allow_blocking_load=True):
            calls["analyze"] += 1
            calls["blocking"].append(allow_blocking_load)
            return None

    shared = FakeAnalyzer()

    def fake_get_instance(cls, model_cache_dir=None, model_id=None):
        calls["get_instance"] += 1
        return shared

    monkeypatch.setattr(
        sentiment_analyzer.SentimentAnalyzer,
        "get_instance",
        classmethod(fake_get_instance),
    )

    assert emotion_state_manager.warmup_shared_sentiment_analyzer(background=False) is True
    assert emotion_state_manager.warmup_shared_sentiment_analyzer(background=False) is True
    assert calls["get_instance"] == 1
    assert calls["analyze"] >= 1
    assert all(call is True for call in calls.get("blocking", []))


def test_sentiment_analyzer_retries_after_failure_cooldown(monkeypatch, emotion_modules):
    sentiment_analyzer, _emotion_detector, _emotion_state_manager = emotion_modules

    calls = {"load": 0}

    def fake_try_load_with_one_failure(self, *, blocking=True):
        with self._load_lock:
            if self._available:
                return True
            if self._load_attempted and self._last_load_failure_at is not None:
                elapsed = time.time() - self._last_load_failure_at
                if elapsed < sentiment_analyzer.LOAD_RETRY_COOLDOWN_SECONDS:
                    return False
            self._load_attempted = True
            calls["load"] += 1
            if calls["load"] == 1:
                self._available = False
                self._last_load_failure_at = time.time()
                self._last_load_error = "simulated"
                return False
            self._available = True
            self._last_load_failure_at = None
            self._last_load_error = None
            return True

    monkeypatch.setattr(sentiment_analyzer.SentimentAnalyzer, "_try_load", fake_try_load_with_one_failure)
    monkeypatch.setattr(sentiment_analyzer, "LOAD_RETRY_COOLDOWN_SECONDS", 0.01)
    analyzer = sentiment_analyzer.SentimentAnalyzer.get_instance()

    assert analyzer._try_load() is False
    assert analyzer._try_load() is False
    assert calls["load"] == 1
    time.sleep(0.02)
    assert analyzer._try_load() is True
    assert calls["load"] == 2


def test_sentiment_analyzer_load_is_thread_safe(monkeypatch, emotion_modules):
    sentiment_analyzer, _emotion_detector, _emotion_state_manager = emotion_modules

    calls = {"load": 0}
    lock = threading.Lock()

    def fake_try_load(self, *, blocking=True):
        with self._load_lock:
            if self._load_attempted:
                return self._available
            time.sleep(0.02)
            with lock:
                calls["load"] += 1
            self._load_attempted = True
            self._available = False
            return False

    monkeypatch.setattr(sentiment_analyzer.SentimentAnalyzer, "_try_load", fake_try_load)
    analyzer = sentiment_analyzer.SentimentAnalyzer.get_instance()

    threads = [threading.Thread(target=lambda: analyzer.analyze("模糊")) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert calls["load"] == 1


def test_sentiment_model_load_pins_configured_revision(monkeypatch, emotion_modules):
    sentiment_analyzer, _emotion_detector, _emotion_state_manager = emotion_modules
    calls = []

    class FakeLoadedModel:
        config = types.SimpleNamespace(id2label={}, problem_type="")

        def eval(self):
            return self

    class FakeLoader:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls.append((model_id, kwargs))
            return FakeLoadedModel()

    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            AutoTokenizer=FakeLoader,
            PreTrainedTokenizerFast=FakeLoader,
            AutoModelForSequenceClassification=FakeLoader,
        ),
    )
    analyzer = sentiment_analyzer.SentimentAnalyzer(
        model_id=sentiment_analyzer.MULTILINGUAL_MODEL_ID,
        model_revision="51d5d73525c1d5f3e599e4b94a4cd6e69e2c9d6a",
    )

    assert analyzer._try_load() is True
    assert len(calls) == 2
    assert all(call[1]["revision"] == "51d5d73525c1d5f3e599e4b94a4cd6e69e2c9d6a" for call in calls)


def test_multilingual_model_uses_tokenizer_json_without_sentencepiece(monkeypatch, emotion_modules):
    """The multilingual checkpoint has tokenizer.json and must avoid the SWIG backend."""
    sentiment_analyzer, _emotion_detector, _emotion_state_manager = emotion_modules
    tokenizer_calls = []

    class FakeLoadedModel:
        config = types.SimpleNamespace(id2label={}, problem_type="multi_label_classification")

        def eval(self):
            return self

    class SlowAutoTokenizer:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            raise AssertionError("AutoTokenizer would import the SentencePiece SWIG extension")

    class FastTokenizer:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            tokenizer_calls.append((model_id, kwargs))
            return object()

    class FakeModelLoader:
        @classmethod
        def from_pretrained(cls, _model_id, **_kwargs):
            return FakeLoadedModel()

    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            AutoTokenizer=SlowAutoTokenizer,
            PreTrainedTokenizerFast=FastTokenizer,
            AutoModelForSequenceClassification=FakeModelLoader,
        ),
    )
    analyzer = sentiment_analyzer.SentimentAnalyzer(
        model_id=sentiment_analyzer.MULTILINGUAL_MODEL_ID,
        model_revision="51d5d73525c1d5f3e599e4b94a4cd6e69e2c9d6a",
    )

    assert analyzer._try_load() is True
    assert tokenizer_calls == [(
        sentiment_analyzer.MULTILINGUAL_MODEL_ID,
        {"revision": "51d5d73525c1d5f3e599e4b94a4cd6e69e2c9d6a"},
    )]
