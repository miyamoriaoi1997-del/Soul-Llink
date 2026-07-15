"""Semantic mode classifier shadow interface.

The default backend is deterministic and local. The LLM backend is an explicit,
disabled-by-default scaffold so semantic classification can be A/B tested without
silently introducing latency, cost, or control authority.
"""

from __future__ import annotations

import json
from typing import Any

from .mode_classifier import ModeClassifier


class SemanticModeClassifier:
    """Produce schema-like semantic mode guesses for shadow evaluation."""

    def __init__(
        self,
        backend: str = "local",
        llm_enabled: bool = False,
        model: str | None = None,
        provider: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 10.0,
        sentiment_analyzer: Any | None = None,
    ):
        self.backend = backend
        self.llm_enabled = llm_enabled
        self.model = model or ""
        self.provider = provider or ""
        self.base_url = base_url or ""
        self.api_key = api_key or ""
        self.timeout = timeout
        self.sentiment_analyzer = sentiment_analyzer
        self.rule_probe = ModeClassifier()

    def classify(
        self,
        user_message: str,
        recent_messages: list[dict] | None = None,
        emotion_state: dict | None = None,
        previous_mode: str | None = None,
        platform: str = "cli",
    ) -> dict:
        if self.backend == "llm":
            if not self.llm_enabled:
                result = self._local_classify(user_message, recent_messages, emotion_state, previous_mode, platform)
                result["backend"] = "llm-disabled-fallback"
                result["model"] = self.model
                result["provider"] = self.provider
                result["reason_codes"] = ["LLM_DISABLED", *result.get("reason_codes", [])]
                return result
            return self._llm_classify(user_message, recent_messages, emotion_state, previous_mode, platform)
        if self.backend in {"local_lightweight", "rules+local"}:
            return self._local_lightweight_classify(user_message, recent_messages, emotion_state, previous_mode, platform)
        return self._local_classify(user_message, recent_messages, emotion_state, previous_mode, platform)

    def _local_classify(
        self,
        user_message: str,
        recent_messages: list[dict] | None = None,
        emotion_state: dict | None = None,
        previous_mode: str | None = None,
        platform: str = "cli",
    ) -> dict:
        decision = self.rule_probe.classify(
            user_message=user_message,
            recent_messages=recent_messages,
            emotion_state=emotion_state,
            platform=platform,
        )
        intent_signals = {
            key: decision.signals[key]
            for key in (
                "explicit_task_request",
                "explicit_system_request",
                "sex_scene_close",
                "sex_scene_continue",
                "safety_stop",
            )
            if key in decision.signals
        }
        result = {
            "primary_mode": decision.mode,
            "submode": decision.submode,
            "confidence": decision.confidence,
            "secondary_modes": [],
            "safety_flags": decision.safety_flags,
            "intent_signals": intent_signals,
            "reason_codes": self._reason_codes(decision),
            "backend": "deterministic-local-shadow",
            "shadow_only": True,
            "previous_mode": previous_mode,
        }
        if decision.mode == "work" and self._contains_affective_marker(user_message):
            result["affective_overlay"] = "daily"
        return result

    # Sentiment label -> candidate mode when rules miss (fallback to daily).
    # Only applied when model confidence exceeds SENTIMENT_CORRECTION_THRESHOLD.
    SENTIMENT_MODE_HINTS: dict[str, str] = {
        "angry":     "daily",
        "disgusted": "daily",
        "sad":       "daily",
        "caring":    "daily",
    }
    # Minimum model confidence to override a rule-miss.
    SENTIMENT_CORRECTION_THRESHOLD = 0.55
    # Sentiment labels that can boost rule confidence when they agree.
    SENTIMENT_BOOST_MAP: dict[str, set[str]] = {
        "angry":     {"daily"},
        "disgusted": {"daily"},
        "sad":       {"daily"},
        "caring":    {"daily"},
        "happy":     {"daily"},
    }

    def _local_lightweight_classify(
        self,
        user_message: str,
        recent_messages: list[dict] | None = None,
        emotion_state: dict | None = None,
        previous_mode: str | None = None,
        platform: str = "cli",
    ) -> dict:
        result = self._local_classify(user_message, recent_messages, emotion_state, previous_mode, platform)
        sentiment = self._analyze_local_sentiment(user_message)
        result["backend"] = "rules+local-lightweight"
        result["local_sentiment"] = sentiment
        if sentiment:
            result["reason_codes"] = [*result.get("reason_codes", []), f"LOCAL_SENTIMENT:{sentiment['label']}"]
            result["affective_valence"] = sentiment.get("valence")
            self._apply_sentiment_correction(result, sentiment, previous_mode)
        else:
            result["reason_codes"] = [*result.get("reason_codes", []), "LOCAL_SENTIMENT_UNAVAILABLE"]
        return result

    def _apply_sentiment_correction(
        self,
        result: dict,
        sentiment: dict,
        previous_mode: str | None,
    ) -> None:
        """Use sentiment model to correct or boost rule-based classification.

        Three cases:
        1. Rule missed (daily fallback, low confidence) + model has strong signal
           -> override primary_mode with sentiment hint.
        2. Rule matched but model agrees -> boost confidence.
        3. Rule matched but model disagrees -> note disagreement, don't override
           (rules win on explicit keyword matches).
        """
        label = sentiment.get("label")
        model_conf = sentiment.get("confidence") or 0.0
        rule_mode = result.get("primary_mode", "daily")
        rule_conf = result.get("confidence", 0.0)

        # Case 1: Rule fallback (daily with low confidence) + model has hint
        if rule_mode == "daily" and rule_conf < 0.7:
            hint_mode = self.SENTIMENT_MODE_HINTS.get(label)
            if hint_mode and model_conf >= self.SENTIMENT_CORRECTION_THRESHOLD:
                result["primary_mode"] = hint_mode
                result["confidence"] = min(model_conf * 0.85, 0.78)
                result["reason_codes"].append(f"SENTIMENT_OVERRIDE:{label}->{hint_mode}")
                if label in {"angry", "disgusted"}:
                    result["safety_flags"] = [*result.get("safety_flags", []), "relationship_conflict"]
                return

        # Case 2: Rule matched and model agrees -> confidence boost
        boost_modes = self.SENTIMENT_BOOST_MAP.get(label, set())
        if rule_mode in boost_modes and model_conf >= 0.5:
            boost = min(model_conf * 0.08, 0.06)
            result["confidence"] = min(result["confidence"] + boost, 0.98)
            result["reason_codes"].append(f"SENTIMENT_BOOST:{label}")
            return

        # Case 3: Disagreement -> log but don't override
        if label and label not in ("neutral", "questioning"):
            hint_mode = self.SENTIMENT_MODE_HINTS.get(label)
            if hint_mode and hint_mode != rule_mode:
                result["reason_codes"].append(f"SENTIMENT_DISAGREE:{label}({hint_mode})!=rule({rule_mode})")

    def _analyze_local_sentiment(self, text: str) -> dict | None:
        analyzer = self.sentiment_analyzer or self._load_default_sentiment_analyzer()
        if analyzer is None:
            return None
        # Local-lightweight semantic classification is shadow-only and must not
        # block first-turn gateway recovery on neural model cold loading.  The
        # emotion system already starts a background warmup; until that finishes,
        # skip semantic sentiment for this turn and keep deterministic rules.
        if hasattr(analyzer, "_available") and getattr(analyzer, "_available", None) is not True:
            return None
        try:
            sentiment = analyzer.analyze(text)
        except Exception:
            return None
        if sentiment is None:
            return None
        return {
            "label": getattr(sentiment, "label", None),
            "label_zh": getattr(sentiment, "label_zh", None),
            "confidence": getattr(sentiment, "confidence", None),
            "valence": getattr(sentiment, "valence", None),
            "inference_ms": getattr(sentiment, "inference_ms", None),
        }

    @staticmethod
    def _load_default_sentiment_analyzer():
        try:
            from sentiment_analyzer import SentimentAnalyzer
        except ImportError:
            try:
                from persona_engine.sentiment_analyzer import SentimentAnalyzer
            except Exception:
                return None
        try:
            return SentimentAnalyzer.get_instance()
        except Exception:
            return None

    def _llm_classify(
        self,
        user_message: str,
        recent_messages: list[dict] | None,
        emotion_state: dict | None,
        previous_mode: str | None,
        platform: str,
    ) -> dict:
        """Call an OpenAI-compatible LLM for semantic classification.

        This path is still shadow-only and never controls active StatePacket mode.
        It is intentionally minimal and requires explicit llm_enabled=True.
        """
        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover - environment dependent
            result = self._local_classify(user_message, recent_messages, emotion_state, previous_mode, platform)
            result["backend"] = "llm-unavailable-fallback"
            result["model"] = self.model
            result["provider"] = self.provider
            result["reason_codes"] = [f"LLM_IMPORT_FAILED:{type(exc).__name__}", *result.get("reason_codes", [])]
            return result

        if not (self.model and self.base_url and self.api_key):
            result = self._local_classify(user_message, recent_messages, emotion_state, previous_mode, platform)
            result["backend"] = "llm-unconfigured-fallback"
            result["model"] = self.model
            result["provider"] = self.provider
            result["reason_codes"] = ["LLM_UNCONFIGURED", *result.get("reason_codes", [])]
            return result

        client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
        prompt = self._build_llm_prompt(user_message, recent_messages, emotion_state, previous_mode, platform)
        try:  # pragma: no cover - network path intentionally disabled in tests
            response = client.chat.completions.create(
                model=self.model,
                temperature=0,
                messages=[
                    {"role": "system", "content": self._system_instruction()},
                    {"role": "user", "content": prompt},
                ],
            )
            content = response.choices[0].message.content or "{}"
            parsed = json.loads(content)
            return self._normalize_llm_result(parsed, previous_mode)
        except Exception as exc:
            result = self._local_classify(user_message, recent_messages, emotion_state, previous_mode, platform)
            result["backend"] = "llm-error-fallback"
            result["model"] = self.model
            result["provider"] = self.provider
            result["reason_codes"] = [f"LLM_ERROR:{type(exc).__name__}", *result.get("reason_codes", [])]
            return result

    @staticmethod
    def _system_instruction() -> str:
        return (
            "You are a strict semantic mode classifier. Output JSON only. "
            "Do not roleplay. Do not answer the user. Classify the user turn into "
            "primary_mode, confidence, safety_flags, reason_codes, intent_signals. "
            "Modes: daily, work, sex. "
            "Hard safety semantics: meta discussion about a sensitive term is work, not the sensitive mode."
        )

    @staticmethod
    def _build_llm_prompt(
        user_message: str,
        recent_messages: list[dict] | None,
        emotion_state: dict | None,
        previous_mode: str | None,
        platform: str,
    ) -> str:
        return json.dumps(
            {
                "user_message": user_message,
                "recent_messages": recent_messages or [],
                "emotion_state": emotion_state or {},
                "previous_mode": previous_mode,
                "platform": platform,
            },
            ensure_ascii=False,
        )

    def _normalize_llm_result(self, parsed: dict[str, Any], previous_mode: str | None) -> dict:
        return {
            "primary_mode": parsed.get("primary_mode") or "daily",
            "submode": parsed.get("submode") or "semantic",
            "confidence": float(parsed.get("confidence") or 0.0),
            "secondary_modes": parsed.get("secondary_modes") or [],
            "safety_flags": parsed.get("safety_flags") or [],
            "intent_signals": parsed.get("intent_signals") or {},
            "reason_codes": parsed.get("reason_codes") or ["LLM_SEMANTIC"],
            "backend": "llm-shadow",
            "model": self.model,
            "provider": self.provider,
            "shadow_only": True,
            "previous_mode": previous_mode,
        }

    @staticmethod
    def _contains_affective_marker(text: str) -> bool:
        normalized = (text or "").lower()
        markers = ("[pet name]", "测试角色", "用户", "抱抱", "亲亲", "陪我", "陪陪")
        return any(marker in normalized for marker in markers)

    @staticmethod
    def _reason_codes(decision) -> list[str]:
        codes = [f"RULE_MODE:{decision.mode}"]
        if decision.submode:
            codes.append(f"SUBMODE:{decision.submode}")
        for flag in decision.safety_flags:
            codes.append(f"FLAG:{flag}")
        return codes
