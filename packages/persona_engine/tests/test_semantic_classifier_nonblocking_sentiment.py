"""Regression tests for local-lightweight semantic classifier latency."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from persona_orchestrator.semantic_classifier import SemanticModeClassifier


class SlowColdAnalyzer:
    _available = False
    _load_attempted = False

    def analyze(self, text):
        time.sleep(0.25)
        raise AssertionError("cold analyzer should not be called by semantic shadow")


class ReadyAnalyzer:
    _available = True
    _load_attempted = True

    class Sentiment:
        label = "sad"
        label_zh = "悲伤语调"
        confidence = 0.9
        valence = -1
        inference_ms = 1.0

    def analyze(self, text):
        return self.Sentiment()


def test_local_lightweight_semantic_skips_cold_sentiment_without_blocking():
    classifier = SemanticModeClassifier(
        backend="local_lightweight",
        sentiment_analyzer=SlowColdAnalyzer(),
    )

    start = time.perf_counter()
    result = classifier.classify("嗯", previous_mode="daily", platform="telegram")
    elapsed = time.perf_counter() - start

    assert elapsed < 0.1
    assert result["backend"] == "rules+local-lightweight"
    assert result["local_sentiment"] is None
    assert "LOCAL_SENTIMENT_UNAVAILABLE" in result["reason_codes"]


def test_local_lightweight_semantic_uses_ready_sentiment():
    classifier = SemanticModeClassifier(
        backend="local_lightweight",
        sentiment_analyzer=ReadyAnalyzer(),
    )

    result = classifier.classify("嗯", previous_mode="daily", platform="telegram")

    assert result["backend"] == "rules+local-lightweight"
    assert result["local_sentiment"]["label"] == "sad"
    assert any(code.startswith("LOCAL_SENTIMENT:sad") for code in result["reason_codes"])
