"""Local neural emotion analysis.

Provides local model-based emotion detection as a complement to the
rule-based EmotionDetector. Lazy-loads the model on first use and
caches it in memory for subsequent calls.

Default Chinese-Emotion-Small labels (8 classes):
    0: 平淡语气  (neutral)
    1: 关切语调  (caring)
    2: 开心语调  (happy)
    3: 愤怒语调  (angry)
    4: 悲伤语调  (sad)
    5: 疑问语调  (questioning)
    6: 惊奇语调  (surprised)
    7: 厌恶语调  (disgusted)

The analyzer can also use tabularisai/multilingual-emotion-classification,
which is a multi-label sigmoid model. Its labels are normalized back into the
legacy labels consumed by the rest of the emotion pipeline.
"""

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "Johnson8187/Chinese-Emotion-Small"
MULTILINGUAL_MODEL_ID = "tabularisai/multilingual-emotion-classification"
DEFAULT_MODEL_REVISIONS = {
    DEFAULT_MODEL_ID: "2c04ce86de44d232f0fbe31413868eb31d791aea",
    MULTILINGUAL_MODEL_ID: "51d5d73525c1d5f3e599e4b94a4cd6e69e2c9d6a",
}
MODEL_ID = os.getenv("SOULLINK_SENTIMENT_MODEL_ID", DEFAULT_MODEL_ID)
MODEL_CACHE_DIR = None  # will be set to ~/.hermes/models at runtime
LOAD_RETRY_COOLDOWN_SECONDS = 60.0
MULTILABEL_THRESHOLD = 0.5

# Label index -> (label_name, valence)
# valence: positive=1, neutral=0, negative=-1
LABEL_MAP: Dict[int, Tuple[str, int]] = {
    0: ("neutral",     0),
    1: ("caring",      1),
    2: ("happy",       1),
    3: ("angry",      -1),
    4: ("sad",        -1),
    5: ("questioning", 0),
    6: ("surprised",   1),
    7: ("disgusted",  -1),
}

# HuggingFace label -> legacy pipeline label, valence, and Chinese display text.
MULTILINGUAL_LABEL_MAP: Dict[str, Tuple[str, int, str]] = {
    "anger": ("angry", -1, "愤怒语调"),
    "contempt": ("disgusted", -1, "轻蔑语调"),
    "disgust": ("disgusted", -1, "厌恶语调"),
    "fear": ("sad", -1, "不安语调"),
    "frustration": ("angry", -1, "烦躁语调"),
    "gratitude": ("caring", 1, "感谢语调"),
    "joy": ("happy", 1, "开心语调"),
    "love": ("caring", 1, "亲近语调"),
    "neutral": ("neutral", 0, "平淡语气"),
    "sadness": ("sad", -1, "悲伤语调"),
    "surprise": ("surprised", 1, "惊奇语调"),
}

# Mapping from model label -> emotion trigger type (for rule-miss补救)
LABEL_TO_TRIGGER: Dict[str, Optional[str]] = {
    "neutral":     None,
    "caring":      "care",
    "happy":       "praise",       # happy without rule match → treat as mild praise
    "angry":       "criticism",
    "sad":         "care",         # sad user → trigger care response
    "questioning": None,
    "surprised":   None,
    "disgusted":   "criticism",
}

# Scale factors applied to rule-detected deltas based on model emotion
# (rule_trigger, model_label) -> scale
FUSION_SCALE: Dict[Tuple[str, str], float] = {
    # Intimacy + positive emotion → amplify
    ("intimacy",   "happy"):      1.4,
    ("intimacy",   "caring"):     1.2,
    ("intimacy",   "surprised"):  1.1,
    ("intimacy",   "neutral"):    0.85,
    ("intimacy",   "questioning"):0.8,
    ("intimacy",   "sad"):        1.1,   # sad + intimacy = vulnerable, still meaningful
    ("intimacy",   "angry"):      0.5,
    ("intimacy",   "disgusted"):  0.3,

    # Praise + positive → amplify
    ("praise",     "happy"):      1.3,
    ("praise",     "caring"):     1.1,
    ("praise",     "neutral"):    0.9,
    ("praise",     "questioning"):0.8,
    ("praise",     "angry"):      0.6,
    ("praise",     "disgusted"):  0.4,

    # Care + caring/sad → amplify
    ("care",       "caring"):     1.3,
    ("care",       "sad"):        1.2,
    ("care",       "happy"):      1.0,
    ("care",       "neutral"):    0.9,
    ("care",       "angry"):      0.7,

    # Criticism + angry/disgusted → amplify (more negative impact)
    ("criticism",  "angry"):      1.5,
    ("criticism",  "disgusted"):  1.4,
    ("criticism",  "neutral"):    0.9,
    ("criticism",  "happy"):      0.6,   # criticism but happy tone → maybe joking

    # Teasing + happy → amplify
    ("teasing",    "happy"):      1.3,
    ("teasing",    "surprised"):  1.1,
    ("teasing",    "neutral"):    0.9,
    ("teasing",    "angry"):      0.5,

    # Other_ai + angry/disgusted → amplify possessiveness spike
    ("other_ai_mentioned", "angry"):    1.4,
    ("other_ai_mentioned", "disgusted"):1.3,
    ("other_ai_mentioned", "happy"):    1.6,  # happy about other AI = worse
    ("other_ai_mentioned", "neutral"):  1.0,

    # Ignored + angry → amplify
    ("ignored",    "angry"):      1.4,
    ("ignored",    "disgusted"):  1.3,
    ("ignored",    "neutral"):    1.0,
    ("ignored",    "happy"):      0.7,
}

DEFAULT_SCALE = 1.0  # fallback when no specific rule defined


@dataclass
class SentimentResult:
    """Result from the sentiment model."""
    label: str           # e.g. "happy", "angry"
    label_zh: str        # e.g. "开心语调"
    confidence: float    # 0.0-1.0 softmax probability of top class
    valence: int         # +1, 0, -1
    all_scores: Dict[str, float]  # label -> probability for all 8 classes
    inference_ms: float  # inference time in milliseconds
    raw_label: str = ""  # original model label before normalization


class SentimentAnalyzer:
    """Lazy-loading wrapper around the configured sentiment model.

    The active model comes from SOULLINK_SENTIMENT_MODEL_ID (default:
    Johnson8187/Chinese-Emotion-Small). Runs fully local from the HF cache
    when HF_HUB_OFFLINE=1; falls back to rules-only on load failure.
    """

    # Thread-safety: model loading is protected so background warmup and the
    # first uncertain user message cannot load weights concurrently.

    _instance: Optional["SentimentAnalyzer"] = None
    _instances: Dict[Tuple[Optional[str], str], "SentimentAnalyzer"] = {}
    _instance_lock = threading.Lock()

    def __init__(
        self,
        model_cache_dir: Optional[str] = None,
        model_id: Optional[str] = None,
        model_revision: Optional[str] = None,
    ):
        self._model = None
        self._tokenizer = None
        self._model_cache_dir = model_cache_dir
        self.model_id = model_id or os.getenv("SOULLINK_SENTIMENT_MODEL_ID") or MODEL_ID
        self.model_revision = (
            model_revision
            or os.getenv("SOULLINK_SENTIMENT_MODEL_REVISION")
            or DEFAULT_MODEL_REVISIONS.get(self.model_id)
        )
        self._load_attempted = False
        self._available = False
        self._last_load_failure_at: Optional[float] = None
        self._last_load_error: Optional[str] = None
        self._load_lock = threading.Lock()
        self._is_multilabel = self.model_id == MULTILINGUAL_MODEL_ID
        self._id2label: Dict[int, str] = {}

    @classmethod
    def get_instance(
        cls,
        model_cache_dir: Optional[str] = None,
        model_id: Optional[str] = None,
    ) -> "SentimentAnalyzer":
        """Return singleton instance."""
        resolved_model_id = model_id or os.getenv("SOULLINK_SENTIMENT_MODEL_ID") or DEFAULT_MODEL_ID
        key = (model_cache_dir, resolved_model_id)
        with cls._instance_lock:
            if key not in cls._instances:
                cls._instances[key] = cls(
                    model_cache_dir=model_cache_dir,
                    model_id=resolved_model_id,
                )
            cls._instance = cls._instances[key]
            return cls._instances[key]

    def _try_load(self, *, blocking: bool = True) -> bool:
        """Attempt to load model. Returns True if successful."""
        if self._available:
            return True
        if not blocking:
            # Conversation hot paths must not pay model-download/model-load
            # latency.  Neural analysis is opportunistic: background warmup may
            # make the model available, otherwise rules remain authoritative.
            return False
        if self._load_attempted and self._last_load_failure_at is not None:
            elapsed = time.time() - self._last_load_failure_at
            if elapsed < LOAD_RETRY_COOLDOWN_SECONDS:
                return False

        with self._load_lock:
            if self._available:
                return True
            if self._load_attempted and self._last_load_failure_at is not None:
                elapsed = time.time() - self._last_load_failure_at
                if elapsed < LOAD_RETRY_COOLDOWN_SECONDS:
                    return False
            self._load_attempted = True
            self._last_load_failure_at = None
            self._last_load_error = None
            try:
                import torch
                from transformers import AutoModelForSequenceClassification

                if self.model_id == MULTILINGUAL_MODEL_ID:
                    # This checkpoint ships tokenizer.json, so it can use the
                    # Rust tokenizer backend directly. Importing AutoTokenizer
                    # here would eagerly import SentencePiece's legacy SWIG
                    # extension on Windows even though it is not needed.
                    from transformers import PreTrainedTokenizerFast

                    tokenizer_loader = PreTrainedTokenizerFast
                else:
                    from transformers import AutoTokenizer

                    tokenizer_loader = AutoTokenizer

                logger.info(f"Loading sentiment model: {self.model_id}...")
                t0 = time.time()

                kwargs = {}
                if self._model_cache_dir:
                    kwargs["cache_dir"] = self._model_cache_dir
                if self.model_revision:
                    kwargs["revision"] = self.model_revision

                self._tokenizer = tokenizer_loader.from_pretrained(self.model_id, **kwargs)
                self._model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_id, **kwargs
                )
                self._model.eval()
                self._torch = torch
                config = getattr(self._model, "config", None)
                id2label = getattr(config, "id2label", None) or {}
                self._id2label = {int(k): str(v) for k, v in dict(id2label).items()}
                problem_type = str(getattr(config, "problem_type", "") or "")
                self._is_multilabel = (
                    self.model_id == MULTILINGUAL_MODEL_ID
                    or problem_type == "multi_label_classification"
                )

                elapsed = time.time() - t0
                logger.info(f"Sentiment model loaded in {elapsed:.1f}s")
                self._available = True

                self._last_load_failure_at = None
                self._last_load_error = None
                return True

            except Exception as e:
                logger.warning(f"Sentiment model {self.model_id!r} unavailable: {e}. "
                               "Emotion system will use rules only.")
                self._available = False
                self._last_load_failure_at = time.time()
                self._last_load_error = str(e)
                return False

    @property
    def available(self) -> bool:
        """True if model is loaded and ready."""
        if not self._load_attempted:
            self._try_load()
        return self._available

    def analyze(self, text: str, *, allow_blocking_load: bool = True) -> Optional[SentimentResult]:
        """Run emotion classification on text.

        Returns None if model is unavailable or text is empty.

        Args:
            text: Text to classify.
            allow_blocking_load: If False, never start a synchronous model load.
                This is for live conversation paths where rule-based detection
                must remain responsive even if the neural backend is cold,
                missing dependencies, or downloading a large checkpoint.
        """
        if not text or not text.strip():
            return None

        if not self._try_load(blocking=allow_blocking_load):
            return None

        try:
            t0 = time.time()
            torch = self._torch

            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=256,
            )

            with torch.no_grad():
                outputs = self._model(**inputs)

            if self._is_multilabel:
                probs = torch.sigmoid(outputs.logits)[0]
                label_name, label_zh, valence, confidence, all_scores, raw_label = (
                    self._normalize_multilabel_scores(probs)
                )
            else:
                probs = torch.softmax(outputs.logits, dim=-1)[0]
                pred_idx = int(torch.argmax(probs).item())
                confidence = float(probs[pred_idx].item())

                label_name, valence = LABEL_MAP[pred_idx]
                label_zh_map = {
                    "neutral": "平淡语气", "caring": "关切语调", "happy": "开心语调",
                    "angry": "愤怒语调", "sad": "悲伤语调", "questioning": "疑问语调",
                    "surprised": "惊奇语调", "disgusted": "厌恶语调",
                }
                label_zh = label_zh_map.get(label_name, label_name)
                raw_label = label_name

                all_scores = {
                    LABEL_MAP[i][0]: float(probs[i].item())
                    for i in range(len(LABEL_MAP))
                }

            elapsed_ms = (time.time() - t0) * 1000

            return SentimentResult(
                label=label_name,
                label_zh=label_zh,
                confidence=confidence,
                valence=valence,
                all_scores=all_scores,
                inference_ms=elapsed_ms,
                raw_label=raw_label,
            )

        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return None

    def _normalize_multilabel_scores(self, probs) -> Tuple[str, str, int, float, Dict[str, float], str]:
        """Normalize multi-label emotion outputs into the legacy label contract."""
        raw_scores: Dict[str, float] = {}
        for idx in range(int(probs.shape[0])):
            raw_label = self._id2label.get(idx, str(idx)).lower()
            raw_scores[raw_label] = float(probs[idx].item())

        canonical_scores: Dict[str, float] = {
            "neutral": 0.0,
            "caring": 0.0,
            "happy": 0.0,
            "angry": 0.0,
            "sad": 0.0,
            "questioning": 0.0,
            "surprised": 0.0,
            "disgusted": 0.0,
        }
        label_zh_by_canonical = {
            "neutral": "平淡语气",
            "caring": "关切语调",
            "happy": "开心语调",
            "angry": "愤怒语调",
            "sad": "悲伤语调",
            "questioning": "疑问语调",
            "surprised": "惊奇语调",
            "disgusted": "厌恶语调",
        }
        valence_by_canonical = {
            "neutral": 0,
            "caring": 1,
            "happy": 1,
            "angry": -1,
            "sad": -1,
            "questioning": 0,
            "surprised": 1,
            "disgusted": -1,
        }
        raw_for_canonical: Dict[str, str] = {}

        for raw_label, score in raw_scores.items():
            mapped = MULTILINGUAL_LABEL_MAP.get(raw_label)
            if mapped is None:
                continue
            canonical, valence, label_zh = mapped
            if score > canonical_scores.get(canonical, 0.0):
                canonical_scores[canonical] = score
                raw_for_canonical[canonical] = raw_label
                valence_by_canonical[canonical] = valence
                label_zh_by_canonical[canonical] = label_zh

        active_scores = {
            label: score
            for label, score in canonical_scores.items()
            if score >= MULTILABEL_THRESHOLD or label == "neutral"
        }
        label_name = max(active_scores, key=active_scores.get) if active_scores else "neutral"
        confidence = canonical_scores.get(label_name, 0.0)
        raw_label = raw_for_canonical.get(label_name, label_name)
        return (
            label_name,
            label_zh_by_canonical.get(label_name, label_name),
            valence_by_canonical.get(label_name, 0),
            confidence,
            canonical_scores,
            raw_label,
        )

    def get_fusion_scale(self, trigger_type: str, sentiment: Optional[SentimentResult]) -> float:
        """Get delta scale factor based on rule trigger + model sentiment.

        Args:
            trigger_type: Rule-detected trigger (e.g. "intimacy", "praise")
            sentiment: Model output, or None if unavailable

        Returns:
            Scale factor to multiply emotion deltas by (0.3 - 1.6 range)
        """
        if sentiment is None:
            return DEFAULT_SCALE

        key = (trigger_type, sentiment.label)
        base_scale = FUSION_SCALE.get(key, DEFAULT_SCALE)

        # Weight by model confidence: low confidence → pull toward 1.0
        # scale = 1.0 + (base_scale - 1.0) * confidence
        confidence_weight = sentiment.confidence
        weighted_scale = 1.0 + (base_scale - 1.0) * confidence_weight

        return round(weighted_scale, 3)

    def get_fallback_trigger(self, sentiment: Optional[SentimentResult]) -> Optional[str]:
        """Get a fallback trigger type when rules found nothing but model detected emotion.

        Only fires when model confidence is high enough (>= 0.6).

        Returns:
            Trigger type string, or None
        """
        if sentiment is None:
            return None
        if sentiment.confidence < 0.6:
            return None
        if sentiment.label in ("neutral", "questioning"):
            return None
        return LABEL_TO_TRIGGER.get(sentiment.label)
