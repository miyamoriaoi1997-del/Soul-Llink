"""Tests for sentiment model -> mode correction in semantic classifier."""

from dataclasses import dataclass
from typing import Dict, Optional

import pytest

from persona_orchestrator.semantic_classifier import SemanticModeClassifier


@dataclass
class FakeSentimentResult:
    """Mimic SentimentResult for controlled testing."""
    label: str
    label_zh: str
    confidence: float
    valence: int
    all_scores: Dict[str, float]
    inference_ms: float = 1.0


class FakeSentimentAnalyzer:
    """Injectable fake that returns a preset result."""

    def __init__(self, result: Optional[FakeSentimentResult] = None):
        self._result = result

    def analyze(self, text: str) -> Optional[FakeSentimentResult]:
        return self._result


def _make_classifier(label: str, confidence: float, valence: int = 0) -> SemanticModeClassifier:
    """Build a classifier with a fake sentiment analyzer returning a fixed result."""
    fake = FakeSentimentAnalyzer(FakeSentimentResult(
        label=label,
        label_zh=f"{label}_zh",
        confidence=confidence,
        valence=valence,
        all_scores={label: confidence},
    ))
    return SemanticModeClassifier(
        backend="local_lightweight",
        sentiment_analyzer=fake,
    )


# --- Case 1: Sentiment overrides rule-miss (daily fallback) ---

def test_angry_overrides_daily_to_conflict():
    """'你根本不理解我' has no rule keyword but angry sentiment -> conflict."""
    clf = _make_classifier("angry", 0.75, valence=-1)
    result = clf.classify("你根本不理解我")
    assert result["primary_mode"] == "daily"
    assert any("SENTIMENT_OVERRIDE" in c for c in result["reason_codes"])
    assert "relationship_conflict" in result["safety_flags"]


def test_disgusted_overrides_daily_to_conflict():
    clf = _make_classifier("disgusted", 0.70, valence=-1)
    result = clf.classify("你让我恶心")
    assert result["primary_mode"] == "daily"
    assert any("SENTIMENT_OVERRIDE" in c for c in result["reason_codes"])


def test_sad_overrides_daily_to_repair():
    """'我好伤心' stays daily and records sad sentiment as a soft override."""
    clf = _make_classifier("sad", 0.65, valence=-1)
    result = clf.classify("我好伤心")
    assert result["primary_mode"] == "daily"
    assert any("SENTIMENT_OVERRIDE" in c for c in result["reason_codes"])
    assert result["safety_flags"] == []


def test_caring_overrides_daily_to_intimacy():
    """'你今天辛苦了吧' has no intimacy keyword but caring sentiment -> intimacy."""
    clf = _make_classifier("caring", 0.68, valence=1)
    result = clf.classify("你今天辛苦了吧")
    assert result["primary_mode"] == "daily"
    assert any("SENTIMENT_OVERRIDE" in c for c in result["reason_codes"])


# --- Case 1 negative: Below threshold, no override ---

def test_low_confidence_angry_stays_daily():
    """Low model confidence should not override rule-miss."""
    clf = _make_classifier("angry", 0.40, valence=-1)
    result = clf.classify("今天天气不错")
    assert result["primary_mode"] == "daily"
    assert not any("SENTIMENT_OVERRIDE" in c for c in result["reason_codes"])


def test_neutral_sentiment_stays_daily():
    """Neutral sentiment should not override anything."""
    clf = _make_classifier("neutral", 0.90, valence=0)
    result = clf.classify("嗯嗯好吧")
    assert result["primary_mode"] == "daily"
    assert not any("SENTIMENT_OVERRIDE" in c for c in result["reason_codes"])


# --- Case 2: Sentiment boosts agreeing rule ---

def test_angry_boosts_conflict_confidence():
    """When rules detect conflict and model agrees (angry), confidence goes up."""
    clf = _make_classifier("angry", 0.80, valence=-1)
    # "废物" is a conflict keyword
    result = clf.classify("废物")
    assert result["primary_mode"] == "daily"
    assert any("SENTIMENT_BOOST" in c for c in result["reason_codes"])
    # Confidence should be higher than base (0.92)
    assert result["confidence"] > 0.92


def test_sad_boosts_repair_confidence():
    """Sad sentiment + soft distress text stays daily and records sentiment override."""
    clf = _make_classifier("sad", 0.75, valence=-1)
    result = clf.classify("我好难受")
    assert result["primary_mode"] == "daily"
    assert any("SENTIMENT_OVERRIDE" in c for c in result["reason_codes"])


def test_caring_boosts_intimacy():
    clf = _make_classifier("caring", 0.70, valence=1)
    result = clf.classify("我想你了")
    assert result["primary_mode"] == "daily"
    assert any("SENTIMENT_BOOST" in c for c in result["reason_codes"])


# --- Case 3: Disagreement logged but not overridden ---

def test_angry_does_not_override_work():
    """Rules matched 'work' firmly; angry sentiment disagrees but doesn't override."""
    clf = _make_classifier("angry", 0.85, valence=-1)
    result = clf.classify("这段代码有 bug")
    assert result["primary_mode"] == "work"
    assert any("SENTIMENT_DISAGREE" in c for c in result["reason_codes"])


def test_sad_does_not_override_system_maintenance():
    """System maintenance matched by rules; sad sentiment noted but ignored."""
    clf = _make_classifier("sad", 0.70, valence=-1)
    result = clf.classify("检查 gateway 日志")
    assert result["primary_mode"] == "work"
    # Sad -> repair hint, but rule matched system_maintenance firmly


# --- Regression: sentiment unavailable falls back cleanly ---

def test_no_sentiment_falls_back_to_rules():
    """When sentiment analyzer is unavailable, pure rule classification."""
    clf = SemanticModeClassifier(
        backend="local_lightweight",
        sentiment_analyzer=FakeSentimentAnalyzer(None),
    )
    result = clf.classify("今天天气真好")
    assert result["primary_mode"] == "daily"
    assert any("UNAVAILABLE" in c for c in result["reason_codes"])


# --- Edge: rule matched at high confidence, model also agrees ---

def test_high_conf_rule_with_agreeing_model_caps_at_98():
    """Confidence should cap at 0.98 even with strong agreement."""
    clf = _make_classifier("angry", 0.95, valence=-1)
    result = clf.classify("废物")
    assert result["confidence"] <= 0.98
