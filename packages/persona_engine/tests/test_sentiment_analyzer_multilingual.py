"""Tests for configurable neural sentiment analyzer backends."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sentiment_analyzer import MULTILINGUAL_MODEL_ID, SentimentAnalyzer


class FakeProbVector:
    def __init__(self, values):
        self._values = values
        self.shape = (len(values),)

    def __getitem__(self, idx):
        return self._values[idx]


class FakeProb:
    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value


def _fake_probs(values):
    return FakeProbVector([FakeProb(value) for value in values])


def test_get_instance_is_keyed_by_model_id(monkeypatch):
    monkeypatch.delenv("SOULLINK_SENTIMENT_MODEL_ID", raising=False)
    SentimentAnalyzer._instances.clear()
    SentimentAnalyzer._instance = None

    default = SentimentAnalyzer.get_instance(model_cache_dir="cache")
    multilingual = SentimentAnalyzer.get_instance(
        model_cache_dir="cache",
        model_id=MULTILINGUAL_MODEL_ID,
    )

    assert default is not multilingual
    assert multilingual.model_id == MULTILINGUAL_MODEL_ID


def test_multilingual_scores_normalize_to_legacy_contract():
    analyzer = SentimentAnalyzer(model_id=MULTILINGUAL_MODEL_ID)
    analyzer._id2label = {
        0: "anger",
        1: "contempt",
        2: "disgust",
        3: "fear",
        4: "frustration",
        5: "gratitude",
        6: "joy",
        7: "love",
        8: "neutral",
        9: "sadness",
        10: "surprise",
    }

    label, label_zh, valence, confidence, all_scores, raw_label = analyzer._normalize_multilabel_scores(
        _fake_probs([0.10, 0.20, 0.30, 0.40, 0.76, 0.55, 0.12, 0.62, 0.05, 0.18, 0.09])
    )

    assert label == "angry"
    assert raw_label == "frustration"
    assert label_zh == "烦躁语调"
    assert valence == -1
    assert confidence == 0.76
    assert all_scores["caring"] == 0.62


def test_multilingual_neutral_remains_fallback_when_no_label_crosses_threshold():
    analyzer = SentimentAnalyzer(model_id=MULTILINGUAL_MODEL_ID)
    analyzer._id2label = {0: "joy", 1: "neutral", 2: "sadness"}

    label, label_zh, valence, confidence, _all_scores, raw_label = analyzer._normalize_multilabel_scores(
        _fake_probs([0.24, 0.31, 0.28])
    )

    assert label == "neutral"
    assert raw_label == "neutral"
    assert label_zh == "平淡语气"
    assert valence == 0
    assert confidence == 0.31
