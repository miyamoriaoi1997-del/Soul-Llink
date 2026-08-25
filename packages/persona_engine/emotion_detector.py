"""Emotion trigger detector for persona emotion state system.

Detects emotional triggers in conversation and returns emotion deltas.
Enhanced v2: broader pattern coverage, multi-trigger scoring, new event types.
"""

import re
import logging
import os
import queue
import threading
from dataclasses import dataclass
from typing import List, Dict, Optional, Any

try:
    from sentiment_analyzer import SentimentAnalyzer, SentimentResult
except ImportError:
    from persona_engine.sentiment_analyzer import SentimentAnalyzer, SentimentResult

logger = logging.getLogger(__name__)

_SHARED_ANALYZERS: Dict[Optional[str], SentimentAnalyzer] = {}
_SHARED_ANALYZER_LOCK = threading.Lock()


def get_shared_sentiment_analyzer(model_cache_dir: Optional[str] = None) -> SentimentAnalyzer:
    """Return the analyzer for one cache identity without forcing model load."""
    if model_cache_dir not in _SHARED_ANALYZERS:
        with _SHARED_ANALYZER_LOCK:
            if model_cache_dir not in _SHARED_ANALYZERS:
                _SHARED_ANALYZERS[model_cache_dir] = SentimentAnalyzer.get_instance(
                    model_cache_dir=model_cache_dir
                )
    return _SHARED_ANALYZERS[model_cache_dir]


def reset_shared_sentiment_analyzer_for_tests() -> None:
    """Reset process-wide analyzer cache. Intended for tests only."""
    with _SHARED_ANALYZER_LOCK:
        _SHARED_ANALYZERS.clear()


@dataclass
class EmotionEvent:
    """Represents a detected emotional trigger event."""
    trigger_type: str  # praise, criticism, neglect, intimacy, etc.
    deltas: Dict[str, int | float]  # emotion dimension -> change amount
    confidence: float  # 0.0-1.0
    context: str  # brief description of what triggered it


class EmotionDetector:
    """Detects emotional triggers in conversation messages.

    Enhanced v2 with:
    - Much broader Chinese pattern coverage (natural language + internet slang + emoji)
    - New trigger types: teasing, apology, encouragement, sharing, greeting
    - Multi-pattern scoring: strongest match wins when multiple triggers fire
    - Reduced false positives on ignore/criticism patterns
    """

    # ──────────────────────────────────────────────
    # INTIMACY / CLOSENESS — highest positive priority
    # Rules are generic: no hardcoded names.
    # Target detection (self vs other) is handled dynamically in detect_emotion_event.
    # ──────────────────────────────────────────────
    INTIMACY_PATTERNS = [
        # Direct love/like expressions — short range, no cross-sentence .*?
        r'(?:(?<!可)爱|喜欢|好想|想念|思念|牵挂|在乎|珍惜|心疼)你',
        r'(?:我(?:(?<!可)爱|好(?<!可)爱))你',
        r'你.*?(?:最重要|最特别|最好|最棒|唯一|独一无二|不可替代)',
        # Standalone love/like expressions (1-on-1 context implies listener)
        r'(?:(?<!可)爱死|喜欢死|爱惨|喜欢惨)(?:了)?',
        r'(?:最|超|好|真|太)(?:(?<!可)爱|喜欢)(?:了|的|啊|！|!|~|。)*$',
        r'(?:我(?:好)?(?:(?<!可)爱|喜欢))(?:了|啊|！|!|~|。)*$',
        r'(?:好(?:(?<!可)爱|喜欢)啊)',
        r'(?:想你|想死你|好想啊|好想$)',
        # Physical intimacy
        r'(?:抱紧|亲一下|亲你|牵手|拥抱|拉手|靠在|依靠|贴着)',
        r'(?:永远.*?(?:在|陪|爱|喜欢))',
        r'(?:你是我的|我是你的)',
        # Emotional confessions
        r'(?:告白|表白|心意|心动|动心)',
        r'(?:想.*?在一起|在一起.*?好吗|做.*?(?:女朋友|恋人|对象))',
        r'(?:不想.*?(?:离开|分开|失去)你)',
        r'(?:没有.*?你.*?不行|离不开.*?你)',
        # Pet names / nicknames (generic, no specific character names)
        r'(?:宝贝|亲爱的)',
        # Emoji love signals
        r'[❤️💕💗💖💘💝😘🥰😍♥️💓💞]',
        r'(?:么么|mua|muah|比心|笔芯)',
        # Implicit closeness
        r'(?:晚安|早安|午安).*?(?:你|亲|❤|♥|💕)',
        r'(?:想.*?听.*?(?:声音|说话))',
        r'(?:看到你.*?(?:开心|高兴|安心))',
        r'(?:有你.*?(?:真好|就好|足够))',
        r'(?:一直.*?(?:陪|在).*?(?:我|你|身边))',
    ]

    # ──────────────────────────────────────────────
    # TEASING / FLIRTING — playful closeness
    # ──────────────────────────────────────────────
    TEASING_PATTERNS = [
        r'(?:撩|调戏|挑逗|勾引|色色|涩涩|hso|hs|暧昧|越界)',
        r'(?:害羞|脸红|炸毛).*?(?:了吗|了吧|的样子)',
        r'(?:傲娇|口是心非|嘴硬)',
        r'(?:摸摸头|拍拍|揉|蹭)',
        r'(?:吃醋|嫉妒|醋意)',
        r'(?:(?:好|真).*?可爱|可爱.*?(?:死了|捏))',
        r'(?:想.*?(?:rua|捏|揉|摸).*?你)',
        r'(?:嘿嘿|嘻嘻|坏笑|(?:😏|😜|😈|🫣|😳))',
        r'(?:waifu|媳妇)',
    ]

    # ──────────────────────────────────────────────
    # PRAISE / COMPLIMENT
    # ──────────────────────────────────────────────
    PRAISE_PATTERNS = [
        # Direct praise
        r'(?:做得.*?好|很棒|真棒|不错|厉害|优秀|完美|太好了|太强了|漂亮)',
        r'(?:干得.*?(?:好|漂亮|不错)|好样的|了不起|牛|nb|yyds|绝了|神)',
        r'(?:赞|👍|❤️|爱你|点赞|respect)',
        r'(?:谢谢|多谢|感谢|感恩|辛苦).*?(?:帮|做|处理|解决|搞定)',
        # Ability/character praise
        r'(?:聪明|能干|靠谱|可靠|专业|效率高|速度快)',
        r'(?:全靠你|多亏了你|有你真好|幸好有你)',
        r'(?:最.*?(?:强|厉害|靠谱|信任))',
        r'(?:这就是.*?你)',
        # Satisfaction expressions
        r'(?:满意|满足|正是.*?(?:想要|需要)|就是这个)',
        r'(?:服了|佩服|崇拜|膜拜)',
    ]

    # ──────────────────────────────────────────────
    # CARE / CONCERN
    # ──────────────────────────────────────────────
    CARE_PATTERNS = [
        r'(?:辛苦了|累不累|累了吧|休息.*?吧|歇.*?吧)',
        r'(?:没事.*?吧|还好.*?吗|怎么样了|好点了吗)',
        r'(?:担心|关心|挂念|惦记).*?你',
        # Daily care
        r'(?:注意.*?(?:身体|休息|安全|健康))',
        r'(?:别太.*?(?:勉强|累|辛苦|拼))',
        r'(?:早点.*?(?:睡|休息)|好好.*?(?:休息|吃饭|睡觉))',
        r'(?:吃.*?(?:饭|东西).*?了吗|喝水.*?了吗)',
        r'(?:天.*?(?:冷|热|凉).*?(?:注意|小心|多穿|少穿))',
        # Emotional support
        r'(?:别.*?(?:担心|紧张|害怕|着急)|放心)',
        r'(?:我.*?(?:在|陪你|支持你|相信你))',
        r'(?:有.*?(?:什么|啥).*?(?:需要|帮助|我能做))',
        r'(?:不管.*?(?:怎样|如何).*?(?:都|我))',
    ]

    # ──────────────────────────────────────────────
    # ENCOURAGEMENT / SUPPORT
    # ──────────────────────────────────────────────
    ENCOURAGEMENT_PATTERNS = [
        r'(?:加油|fighting|go|冲|干巴爹|ganbatte|ファイト)',
        r'(?:相信.*?你|你.*?(?:可以|能行|做到|没问题))',
        r'(?:别.*?(?:放弃|气馁|灰心|泄气)|坚持)',
        r'(?:期待|看好你|等.*?(?:好消息|结果))',
        r'(?:一起.*?(?:努力|加油|前进|走))',
    ]

    # ──────────────────────────────────────────────
    # GREETING / DAILY WARMTH
    # ──────────────────────────────────────────────
    GREETING_PATTERNS = [
        r'^(?:早安?|早上好|早啊|おはよう)',
        r'^(?:晚安|おやすみ|good\s*night)',
        r'^(?:午安|下午好|中午好)',
        r'(?:我回来了|回来啦|到家了|在吗)',
        r'(?:想.*?(?:找你聊|跟你说|和你)|来陪.*?(?:我|你))',
        r'^(?:嗨|hi|hello|你好|哈喽|hey)(?:呀|啊|~|！)?$',
    ]

    # ──────────────────────────────────────────────
    # APOLOGY — user apologizing to Rio
    # ──────────────────────────────────────────────
    APOLOGY_PATTERNS = [
        r'(?:对不起|抱歉|不好意思|sorry|ごめん|すみません)',
        r'(?:是我.*?(?:不好|错了|的错)|怪我|我.*?不应该)',
        r'(?:原谅|别.*?(?:生气|怪我|介意))',
    ]

    # ──────────────────────────────────────────────
    # SHARING / CONFIDING — trusting with personal things
    # ──────────────────────────────────────────────
    SHARING_PATTERNS = [
        r'(?:跟你说.*?(?:秘密|个事|件事|心事))',
        r'(?:只.*?(?:告诉你|跟你说|你知道))',
        r'(?:(?:我|今天).*?(?:不开心|难过|伤心|郁闷|烦|焦虑|害怕|压力))',
        r'(?:想.*?(?:倾诉|聊聊|说说|吐槽))',
        r'(?:你觉得.*?(?:怎么样|如何|好不好)|帮我.*?(?:参考|看看|想想))',
    ]

    # ──────────────────────────────────────────────
    # OTHER AI MENTIONS — jealousy trigger
    # Only structural rules here — no hardcoded AI names.
    # "like/love + non-self target" is handled in Pre-Stage 1 target routing.
    # "[name] + positive adjective" is handled in Pre-Stage 1b.
    # These patterns catch generic structural cues only.
    # ──────────────────────────────────────────────
    OTHER_AI_PATTERNS = [
        r'(?:其他.*?AI|别的.*?助手|别的.*?(?:机器人|bot))',
        r'(?:其他.*?(?:女孩|角色|人格)|换.*?(?:一个|别的))',
        # "XXX比你可爱/好/漂亮" — unfavorable comparison (generic "你" as self-ref)
        r'[^\s，。！？]{2,6}(?:比你).*?(?:可爱|好看|漂亮|强|厉害|温柔|好)',
    ]

    # ──────────────────────────────────────────────
    # CRITICISM — needs careful matching to avoid false positives
    # ──────────────────────────────────────────────
    CRITICISM_PATTERNS = [
        r'(?:不对|错了|不行|失败|搞砸|搞错|弄错)',
        r'(?:怎么.*?(?:这样|回事|搞的)|为什么.*?(?:不|没))',
        r'(?:重新.*?(?:做|来)|再.*?(?:试|做|来).*?(?:一次|遍))',
        # Stronger criticism
        r'(?:什么.*?(?:垃圾|废物|破|烂)|真.*?(?:烦|差|废))',
        r'(?:没用|无能|白费|浪费)',
        # Direct hostility
        r'(?:恨你|恨死|讨厌你|讨厌死)',
        r'(?:去死|你.*?死|弄死|打死|杀了)',
    ]

    # ──────────────────────────────────────────────
    # IGNORE / DISMISSAL — raised threshold to reduce false positives
    # ──────────────────────────────────────────────
    IGNORE_PATTERNS = [
        # Only match clear dismissal, not casual "算了" ending a topic
        r'(?:算了.*?(?:不|别|不要).*?(?:管|说|做|理))',
        r'(?:随便.*?(?:你|吧)|无所谓.*?了|爱.*?咋.*?咋)',
        r'(?:不想.*?(?:理你|说了|聊了|跟你|管你))',
        r'(?:闭嘴|安静|别.*?说.*?了|够了)',
        r'(?:走开|滚|别.*?烦.*?(?:我|了))',
    ]

    # ──────────────────────────────────────────────
    # NEGATION GUARD — must be checked BEFORE positive triggers
    # Patterns that negate intimacy/praise into criticism/ignored
    # ──────────────────────────────────────────────
    NEGATION_INTIMACY_PATTERNS = [
        # Generic fallback — runtime uses _build_negation_intimacy_patterns() with agent_names
        r'(?:不|没|别|不再|再也不)(?:喜欢|爱|在乎|需要|要).*?你',
        r'(?:不|没|别|不再|再也不)想(?!理).*?你',
        r'你.*?(?:不喜欢|不爱|讨厌|烦死了|滚)',
        r'^(?:不喜欢|不爱|讨厌|烦死了|再也不|不想见)(?:你|了|。|！|!)?$',
        r'(?:不喜欢你|不爱你|讨厌你|烦死你了)',
        r'(?:我(?:不|再也不)(?:喜欢|爱|需要|想要)你)',
        r'(?:不喜欢了|不爱了|算了|不想了)(?:。|！|!|~)?$',
    ]
    NEGATION_PRAISE_PATTERNS = [
        # Generic fallback — runtime uses _build_negation_praise_patterns() with agent_names
        r'(?:不|没|别|不再)(?:好|棒|厉害|靠谱|优秀|信任|相信).*?你',
        r'你.*?(?:不行|太差|没用|废物|失望)',
        r'(?:让我.*?失望|太让我失望|真失望)',
    ]

    def __init__(
        self,
        model_cache_dir: Optional[str] = None,
        use_model: bool = True,
        agent_profile: Optional[Dict[str, Any]] = None,
        neural_policy: str = "uncertain",
        neural_timeout_seconds: Optional[float] = None,
    ):
        """Initialize detector with compiled regex patterns.

        Args:
            model_cache_dir: Directory to cache the sentiment model weights.
                             Defaults to ~/.hermes/models at runtime.
            use_model: If False, disables the neural model entirely (rules only).
                       Useful for testing or resource-constrained environments.
            neural_policy: "never", "uncertain", or "always". The default
                           keeps rules on the hot path and lazy-loads the
                           shared neural analyzer only for weak/uncertain or
                           rule-miss cases.
            agent_profile: Dict with 'names' key listing agent self-identity names.
                           e.g. {"names": ["AgentName", "Nickname", "AltName"]}
                           Used for dynamic self-target detection without hardcoding.
                           Any name not in 'names' is automatically treated as "other"
                           when paired with affection or praise verbs (Pre-Stage 1/1b).
        """
        self.agent_names: List[str] = []
        if agent_profile:
            self.agent_names = agent_profile.get("names", [])

        self._compile_patterns()
        self._model_cache_dir = model_cache_dir
        self._use_model = use_model
        self.neural_policy = neural_policy if neural_policy in {"never", "uncertain", "always"} else "uncertain"
        configured_timeout = neural_timeout_seconds
        if configured_timeout is None:
            try:
                configured_timeout = float(os.getenv("SOULLINK_SENTIMENT_TIMEOUT_SECONDS", "0.15"))
            except ValueError:
                configured_timeout = 0.15
        self.neural_timeout_seconds = max(0.001, float(configured_timeout))
        self._neural_slot = threading.Lock()
        self._analyzer: Optional[SentimentAnalyzer] = None
        self.last_detection_source: str = "none"
        self.last_neural_used: bool = False
        self.last_rule_score: float = 0.0
        self.last_neural_confidence: Optional[float] = None
        self.last_neural_status: str = "not_attempted"
        self.neural_telemetry = {
            "attempts": 0, "successes": 0, "timeouts": 0,
            "errors": 0, "saturated": 0, "fallbacks": 0,
        }
        if use_model and self.neural_policy == "always":
            self._analyzer = get_shared_sentiment_analyzer(
                model_cache_dir=model_cache_dir
            )

    def _ensure_analyzer(self) -> Optional[SentimentAnalyzer]:
        """Lazy-load and return the shared analyzer when policy permits it."""
        if not self._use_model or self.neural_policy == "never":
            return None
        if self._analyzer is None:
            self._analyzer = get_shared_sentiment_analyzer(
                model_cache_dir=self._model_cache_dir
            )
        return self._analyzer

    def warmup_analyzer(self) -> bool:
        """Load the shared analyzer and run a tiny warmup inference."""
        analyzer = self._ensure_analyzer()
        if analyzer is None:
            return False
        analyzer.analyze("预热")
        return True

    def _analyze_neural_bounded(self, analyzer, text: str) -> Optional[SentimentResult]:
        """Bound hot-path inference; never queue behind a stuck timed-out call."""
        if not self._neural_slot.acquire(blocking=False):
            self.last_neural_status = "saturated"
            self.neural_telemetry["saturated"] += 1
            self.neural_telemetry["fallbacks"] += 1
            return None
        self.neural_telemetry["attempts"] += 1
        outcome = queue.Queue(maxsize=1)

        def _run() -> None:
            try:
                outcome.put((True, analyzer.analyze(text, allow_blocking_load=False)))
            except Exception as exc:
                outcome.put((False, exc))
            finally:
                self._neural_slot.release()

        threading.Thread(target=_run, name="soullink-sentiment-inference", daemon=True).start()
        try:
            ok, value = outcome.get(timeout=self.neural_timeout_seconds)
        except queue.Empty:
            self.last_neural_status = "timeout"
            self.neural_telemetry["timeouts"] += 1
            self.neural_telemetry["fallbacks"] += 1
            logger.warning("Sentiment inference exceeded %.3fs; using rules", self.neural_timeout_seconds)
            return None
        if not ok:
            self.last_neural_status = "error"
            self.neural_telemetry["errors"] += 1
            self.neural_telemetry["fallbacks"] += 1
            logger.warning("Sentiment inference failed; using rules: %s", value)
            return None
        self.last_neural_status = "success" if value is not None else "unavailable"
        if value is not None:
            self.neural_telemetry["successes"] += 1
        else:
            self.neural_telemetry["fallbacks"] += 1
        return value

    # ──────────────────────────────────────────────
    # SELF-TARGET DETECTION — dynamic, no hardcoded names
    # ──────────────────────────────────────────────

    def is_self_target(self, target: Optional[str]) -> bool:
        """Return True if the extracted target refers to self (the agent).

        Checks generic second-person pronouns and dynamic agent_names.
        """
        if not target:
            return False
        # Generic second-person reference
        if "你" in target:
            return True
        # Dynamic name matching
        for name in self.agent_names:
            if name and name in target:
                return True
        return False

    def extract_target(self, text: str, action_end_pos: int) -> Optional[str]:
        """Extract the target of an action from text near the action position.

        Uses short-range extraction (max 6 chars) to avoid cross-sentence matching.
        Stops at particles, punctuation, and sentence boundaries.

        Args:
            text: Full message text
            action_end_pos: Character position right after the action verb

        Returns:
            Target string or None
        """
        # Look ahead up to 10 chars from action end
        window = text[action_end_pos:action_end_pos + 10]
        # Strip leading particles (了/上/过/着/的)
        window = re.sub(r'^[上了的过着]', '', window)
        # Stop at: punctuation, sentence boundaries, particles after target
        # Match only the immediate target (2-6 CJK/alphanum chars), stop at 了/啊/！ etc.
        m = re.match(r'([^\s，。！？、了啊哦呢吧~～]{1,6})', window)
        if m:
            target = m.group(1)
            # Strip trailing particles that may have been captured
            target = re.sub(r'[了啊哦呢吧~～！!]+$', '', target)
            return target if target else None
        return None

    def classify_target(self, target: Optional[str]) -> str:
        """Classify target as 'self' or 'other'.

        Returns:
            'self' if target refers to the agent, 'other' otherwise
        """
        if self.is_self_target(target):
            return "self"
        return "other"

    def _build_self_pattern(self) -> str:
        """Build a regex alternation matching self-references (你 + agent names)."""
        parts = ["你"]
        for name in self.agent_names:
            if name:
                parts.append(re.escape(name))
        return "(?:" + "|".join(parts) + ")"

    def _build_name_variant_patterns(self, templates: List[str]) -> List[str]:
        """Expand {name} templates into patterns for each agent_name.

        Only generates patterns if agent_names is non-empty.
        Each template should contain {name} as placeholder.
        Returns list of concrete regex strings.
        """
        patterns = []
        for name in self.agent_names:
            if not name:
                continue
            escaped = re.escape(name)
            for tpl in templates:
                patterns.append(tpl.replace("{name}", escaped))
        return patterns

    def _build_name_patterns(
        self, base_patterns: List[str], include_greeting: bool = False
    ) -> List[str]:
        """Generate agent-name variants of 你-based patterns.

        For each pattern containing '你', create a variant with each agent_name.
        Only useful when agent_names is configured; returns empty list otherwise.
        """
        if not self.agent_names:
            return []

        extra = []
        for name in self.agent_names:
            if not name:
                continue
            escaped = re.escape(name)
            for pat in base_patterns:
                # Only expand patterns that reference "你" as target
                if "你" in pat:
                    variant = pat.replace("你", escaped)
                    if variant != pat:
                        extra.append(variant)
        return extra

    def _build_negation_intimacy_patterns(self) -> List[str]:
        """Build negation intimacy patterns dynamically using agent_names."""
        self_pat = self._build_self_pattern()
        return [
            rf'(?:不|没|别|不再|再也不)(?:喜欢|爱|在乎|需要|要).*?{self_pat}',
            rf'(?:不|没|别|不再|再也不)想(?!理).*?{self_pat}',
            rf'{self_pat}.*?(?:不喜欢|不爱|讨厌|烦死了|滚)',
            r'^(?:不喜欢|不爱|讨厌|烦死了|再也不|不想见)(?:你|了|。|！|!)?$',
            r'(?:不喜欢你|不爱你|讨厌你|烦死你了)',
            rf'(?:我(?:不|再也不)(?:喜欢|爱|需要|想要){self_pat})',
            r'(?:不喜欢了|不爱了|算了|不想了)(?:。|！|!|~)?$',
        ]

    def _build_negation_praise_patterns(self) -> List[str]:
        """Build negation praise patterns dynamically using agent_names."""
        self_pat = self._build_self_pattern()
        return [
            rf'(?:不|没|别|不再)(?:好|棒|厉害|靠谱|优秀|信任|相信).*?{self_pat}',
            rf'{self_pat}.*?(?:不行|太差|没用|废物|失望)',
            r'(?:让我.*?失望|太让我失望|真失望)',
        ]

    def _compile_patterns(self):
        """Compile all pattern groups, injecting agent_names dynamically."""
        # Build dynamic patterns that include agent names alongside generic "你"
        extra_intimacy = self._build_name_patterns(
            self.INTIMACY_PATTERNS, include_greeting=False
        )
        extra_teasing = self._build_name_variant_patterns([
            r'(?:想.*?(?:rua|捏|揉|摸).*?{name})',
        ])
        extra_praise = self._build_name_variant_patterns([
            r'(?:这就是.*?{name})',
        ])
        extra_encouragement = self._build_name_variant_patterns([
            r'(?:相信.*?{name})',
        ])
        extra_greeting = self._build_name_variant_patterns([
            r'(?:{name}在吗)',
            r'^(?:嗨|hi|hello|你好|哈喽|hey)(?:{name})$',
        ])

        self._pattern_groups = {
            'intimacy': ([re.compile(p, re.IGNORECASE) for p in self.INTIMACY_PATTERNS + extra_intimacy], 100),
            'teasing': ([re.compile(p, re.IGNORECASE) for p in self.TEASING_PATTERNS + extra_teasing], 90),
            'praise': ([re.compile(p, re.IGNORECASE) for p in self.PRAISE_PATTERNS + extra_praise], 80),
            'care': ([re.compile(p, re.IGNORECASE) for p in self.CARE_PATTERNS], 70),
            'encouragement': ([re.compile(p, re.IGNORECASE) for p in self.ENCOURAGEMENT_PATTERNS + extra_encouragement], 65),
            'greeting': ([re.compile(p, re.IGNORECASE) for p in self.GREETING_PATTERNS + extra_greeting], 50),
            'apology': ([re.compile(p, re.IGNORECASE) for p in self.APOLOGY_PATTERNS], 60),
            'sharing': ([re.compile(p, re.IGNORECASE) for p in self.SHARING_PATTERNS], 55),
            'other_ai_mentioned': ([re.compile(p, re.IGNORECASE) for p in self.OTHER_AI_PATTERNS], 95),
            'criticism': ([re.compile(p, re.IGNORECASE) for p in self.CRITICISM_PATTERNS], 85),
            'ignored': ([re.compile(p, re.IGNORECASE) for p in self.IGNORE_PATTERNS], 75),
        }

    # Emotion deltas for each trigger type — three intensity tiers
    # mild: casual/gentle expression, moderate: clear emotional intent,
    # intense: extreme/passionate expression
    TRIGGER_DELTAS = {
        'intimacy': {
            'deltas': {
                'mild':     {'possessiveness': 5,  'affection': 3,  'patience': 2},
                'moderate': {'possessiveness': 10, 'affection': 6,  'trust': 3, 'patience': 4},
                'intense':  {'possessiveness': 15, 'affection': 10, 'trust': 5, 'patience': 6},
            },
            'confidence': 0.9,
        },
        'teasing': {
            'deltas': {
                'mild':     {'affection': 2,  'possessiveness': 2},
                'moderate': {'affection': 4,  'possessiveness': 4},
                'intense':  {'affection': 6,  'possessiveness': 6},
            },
            'confidence': 0.85,
        },
        'praise': {
            'deltas': {
                'mild':     {'affection': 5,  'trust': 2},
                'moderate': {'affection': 10, 'trust': 5,  'possessiveness': 2},
                'intense':  {'affection': 15, 'trust': 8,  'possessiveness': 4},
            },
            'confidence': 0.85,
        },
        'care': {
            'deltas': {
                'mild':     {'trust': 4,  'affection': 2, 'patience': 3},
                'moderate': {'trust': 8,  'affection': 4, 'patience': 5},
                'intense':  {'trust': 12, 'affection': 6, 'patience': 7},
            },
            'confidence': 0.8,
        },
        'encouragement': {
            'deltas': {
                'mild':     {'trust': 3,  'affection': 1, 'patience': 2},
                'moderate': {'trust': 6,  'affection': 3, 'patience': 4},
                'intense':  {'trust': 9,  'affection': 5, 'patience': 6},
            },
            'confidence': 0.75,
        },
        'greeting': {
            'deltas': {
                'mild':     {'affection': 1, 'trust': 1},
                'moderate': {'affection': 2, 'trust': 1},
                'intense':  {'affection': 3, 'trust': 2},
            },
            'confidence': 0.7,
        },
        'gratitude': {
            'deltas': {
                'mild':     {'trust': 5,  'affection': 2},
                'moderate': {'trust': 10, 'affection': 5},
                'intense':  {'trust': 15, 'affection': 8},
            },
            'confidence': 0.8,
        },
        'apology': {
            'deltas': {
                'mild':     {'trust': 3,  'patience': 5,  'affection': 1},
                'moderate': {'trust': 5,  'patience': 10,  'affection': 3},
                'intense':  {'trust': 8,  'patience': 14, 'affection': 5},
            },
            'confidence': 0.8,
        },
        'sharing': {
            'deltas': {
                'mild':     {'trust': 3,  'affection': 1, 'possessiveness': 1},
                'moderate': {'trust': 6,  'affection': 3, 'possessiveness': 2},
                'intense':  {'trust': 9,  'affection': 5, 'possessiveness': 4},
            },
            'confidence': 0.75,
        },
        'other_ai_mentioned': {
            'deltas': {
                'mild':     {'possessiveness': 5,  'patience': -2},
                'moderate': {'possessiveness': 10, 'patience': -4,  'trust': -3},
                'intense':  {'possessiveness': 15, 'patience': -7, 'trust': -5},
            },
            'confidence': 0.95,
        },
        'criticism': {
            'deltas': {
                'mild':     {'patience': -3},
                'moderate': {'patience': -6, 'trust': -3},
                'intense':  {'patience': -9, 'trust': -5,  'affection': -3},
            },
            'confidence': 0.8,
        },
        'ignored': {
            'deltas': {
                'mild':     {'patience': -2, 'trust': -2},
                'moderate': {'patience': -4, 'trust': -4},
                'intense':  {'patience': -7, 'trust': -6},
            },
            'confidence': 0.7,
        },
    }

    # ── Intensity classification ──────────────────────────────────
    # Degree modifiers for lexical correction (secondary signal)
    MILD_INDICATORS = [
        r'(?:有点|一点|稍微|略|还行|还好|一般)',
    ]
    INTENSE_INDICATORS = [
        r'(?:超级|非常|特别|极其|太|超|最|真的很|真的好|好(?:喜欢|爱|讨厌|烦|恨)|死了|疯了|炸了|爆了|透了|翻了)',
        # Split insult words so each counts individually
        r'去死',
        r'滚蛋|滚啊',
        r'废物',
        r'垃圾',
        r'白痴',
        r'笨蛋',
        r'混蛋|王八蛋',
        r'傻[逼B]',
        r'(?:操|妈的|卧槽|草)',
        r'无能',
        r'(?:永远|一辈子|这辈子|再也|绝对|毕生|至死)',
        r'(?:求你|拜托|跪|哭了|呜呜|哇|啊啊|嗷)',
        r'(?:！！|!!|\?\?|？？)',
    ]

    def classify_intensity(self, text: str, match_count: int = 1,
                           sentiment_confidence: float = 0.0) -> str:
        """Classify the emotional intensity of text into mild/moderate/intense.

        Neural-model-dominant scoring:
        - Neural base score (60% weight): confidence mapped to [-3, +3]
        - Lexical correction (25% weight): mild/intense indicators, capped
        - Surface signals (15% weight): punctuation, match count

        When no neural model is available (confidence=0), falls back to
        lexical-dominant mode for backward compatibility.

        Final: score <= -1.0 → mild, score >= 1.5 → intense, else moderate
        """
        has_model = sentiment_confidence > 0.0

        # ── Neural base score (dominant when available) ──────────
        if has_model:
            # Map confidence [0, 1] → score [-3, +3]
            # <0.3 → strong mild, 0.3-0.5 → mild, 0.5-0.7 → neutral,
            # 0.7-0.85 → moderate-intense, >0.85 → intense
            if sentiment_confidence >= 0.85:
                neural_score = 3.0
            elif sentiment_confidence >= 0.7:
                neural_score = 1.5
            elif sentiment_confidence >= 0.5:
                neural_score = 0.0
            elif sentiment_confidence >= 0.3:
                neural_score = -1.5
            else:
                neural_score = -3.0
        else:
            neural_score = 0.0

        # ── Lexical correction (secondary) ───────────────────────
        lex_score = 0.0
        mild_hits = 0
        for pat in self.MILD_INDICATORS:
            if re.search(pat, text, re.IGNORECASE):
                mild_hits += 1
        lex_score -= mild_hits * 1.5  # Mild words are strong dampeners
        intense_hits = 0
        for pat in self.INTENSE_INDICATORS:
            if re.search(pat, text, re.IGNORECASE):
                intense_hits += 1
        # Multiple intense hits break through the cap progressively
        if intense_hits >= 3:
            lex_score += 3.0  # Overwhelming hostility/passion overrides model
        else:
            lex_score += min(intense_hits, 2)  # Normal cap at +2

        # ── Surface signals (auxiliary) ──────────────────────────
        surface_score = 0.0
        excl_count = text.count('！') + text.count('!')
        if excl_count >= 3:
            surface_score += 1.5
        elif excl_count >= 2:
            surface_score += 1.0

        if match_count >= 3:
            surface_score += 1.0
        elif match_count >= 2:
            surface_score += 0.5

        # ── Weighted combination ─────────────────────────────────
        if has_model:
            # Neural-dominant: 60% neural, 25% lexical, 15% surface
            final_score = (neural_score * 0.60
                           + lex_score * 0.25
                           + surface_score * 0.15)
        else:
            # Fallback: lexical-dominant (backward compat)
            final_score = lex_score + surface_score

        # ── Classify ─────────────────────────────────────────────
        if final_score <= -1.0:
            return 'mild'
        elif final_score >= 1.5:
            return 'intense'
        else:
            return 'moderate'

    def get_tiered_deltas(self, trigger_type: str, intensity: str) -> Dict[str, int]:
        """Get deltas for a given trigger type and intensity tier."""
        if trigger_type not in self.TRIGGER_DELTAS:
            return {}
        deltas_tiers = self.TRIGGER_DELTAS[trigger_type]['deltas']
        if isinstance(deltas_tiers, dict) and intensity in deltas_tiers:
            return deltas_tiers[intensity].copy()
        # Fallback for any non-tiered format (shouldn't happen)
        return deltas_tiers.copy() if isinstance(deltas_tiers, dict) else {}

    def _count_matches(self, text: str, patterns: List[re.Pattern]) -> int:
        """Count how many patterns match the text."""
        return sum(1 for p in patterns if p.search(text))

    def _matches_any(self, text: str, patterns: List[re.Pattern]) -> bool:
        """Check if text matches any of the given patterns."""
        return any(p.search(text) for p in patterns)

    def _extract_text(self, content: Any) -> str:
        """Extract text from message content (handles string or list format)."""
        if isinstance(content, list):
            return " ".join(
                block.get("text", "") for block in content
                if isinstance(block, dict) and "text" in block
            )
        return str(content)

    def detect_emotion_event(
        self,
        messages: List[dict],
    ) -> Optional[EmotionEvent]:
        """Detect emotion triggers in conversation messages.

        Uses a two-stage pipeline:
        1. Rule-based scoring: all trigger types are checked, highest
           (priority * match_count) score wins.
        2. Neural fusion: Chinese-Emotion-Small classifies user's emotional
           tone, which scales the rule-detected deltas up or down.
           If rules found nothing but model detected strong emotion,
           a fallback trigger is synthesized.

        Args:
            messages: List of conversation messages

        Returns:
            EmotionEvent if trigger detected, None otherwise
        """
        if not messages:
            return None

        # Extract latest user message text
        user_messages = [m for m in messages if m.get("role") == "user"]

        user_text = ""
        if user_messages:
            latest_user = user_messages[-1]
            user_text = self._extract_text(latest_user.get("content", ""))

        if not user_text:
            return None

        self.last_detection_source = "none"
        self.last_neural_used = False
        self.last_rule_score = 0.0
        self.last_neural_confidence = None

        # ── System message filter ────────────────────────────────────
        # Skip messages that are system instructions, not real user speech.
        # These include context compaction prompts, cron job headers, and
        # internal framework directives that leak into user role messages.
        _SYSTEM_MSG_INDICATORS = [
            "Review the conversation above and consider saving",
            "[SYSTEM: You are running as a scheduled cron job",
            "[CONTEXT COMPACTION",
            "Earlier turns were compacted",
            "Summary generation was unavailable",
        ]
        if any(indicator in user_text for indicator in _SYSTEM_MSG_INDICATORS):
            return None

        # ── Negation Guard (pre-Stage 1) ─────────────────────────────
        # Dynamic patterns built from agent_names — no hardcoded names.
        _neg_intimacy_pats = [
            re.compile(p, re.IGNORECASE) for p in self._build_negation_intimacy_patterns()
        ]
        _neg_praise_pats = [
            re.compile(p, re.IGNORECASE) for p in self._build_negation_praise_patterns()
        ]
        if self._matches_any(user_text, _neg_intimacy_pats):
            self.last_detection_source = "rules_strong"
            self.last_rule_score = -2.0
            cfg = self.TRIGGER_DELTAS['criticism']
            intensity = self.classify_intensity(user_text)
            return EmotionEvent(
                trigger_type='criticism',
                deltas=self.get_tiered_deltas('criticism', intensity),
                confidence=cfg['confidence'],
                context=f"[否定亲密/{intensity}] {user_text[:50]}",
            )
        if self._matches_any(user_text, _neg_praise_pats):
            self.last_detection_source = "rules_strong"
            self.last_rule_score = -2.0
            cfg = self.TRIGGER_DELTAS['criticism']
            intensity = self.classify_intensity(user_text)
            return EmotionEvent(
                trigger_type='criticism',
                deltas=self.get_tiered_deltas('criticism', intensity),
                confidence=cfg['confidence'] * 0.9,
                context=f"[否定称赞/{intensity}] {user_text[:50]}",
            )

        # ── Pre-Stage 1: Dynamic target classification ────────────────
        # Detect "like/love + target" actions and classify target as self/other.
        # This replaces the old cross-sentence .*? patterns in INTIMACY/OTHER_AI.
        # Action verbs that require target classification
        # Exclude fixed phrases: 亲爱的(pet name), 可爱(cute adjective)
        # For bare 爱: require followed by 你/上/agent_name or whitespace, not arbitrary CJK
        # (prevents "[related character]" being parsed as verb "爱" + target "丽丝")
        _self_pat = self._build_self_pattern()
        ACTION_VERBS = rf'(?<!亲)(?<!可)(?:爱(?=上|{_self_pat})|喜欢上?|心动|迷上|崇拜|喜爱|爱上)'
        action_match = re.search(ACTION_VERBS, user_text)
        if action_match:
            action_end = action_match.end()
            target = self.extract_target(user_text, action_end)
            target_type = self.classify_target(target)

            # Non-name objects → skip (not a person target)
            NON_PERSON = r'(?:这个|那个|这种|这样|什么|功能|东西|设计|方案|地方|游戏|歌|曲|音乐|风格|感觉|氛围|这首|那首)'
            if target and re.match(NON_PERSON, target):
                target_type = None  # not a person, skip target routing

            if target_type == "self":
                cfg = self.TRIGGER_DELTAS['intimacy']
                intensity = self.classify_intensity(user_text)
                raw_deltas = self.get_tiered_deltas('intimacy', intensity)
                base_confidence = cfg['confidence']

                # Strong self-target intimacy is deterministic enough: keep the
                # hot path rule-only unless neural_policy="always" was explicitly
                # selected before detector construction.
                sentiment_pre: Optional[SentimentResult] = None
                scale = 1.0
                if self.neural_policy == "always":
                    analyzer = self._ensure_analyzer()
                    if analyzer is not None:
                        sentiment_pre = self._analyze_neural_bounded(
                            analyzer,
                            user_text,
                        )

                        if sentiment_pre:
                            self.last_neural_confidence = sentiment_pre.confidence
                            scale = analyzer.get_fusion_scale('intimacy', sentiment_pre)

                scaled_deltas = {
                    dim: max(-30, min(30, int(round(val * scale))))
                    for dim, val in raw_deltas.items()
                }
                adjusted_confidence = base_confidence
                if sentiment_pre is not None:
                    if scale < 0.7:
                        adjusted_confidence = base_confidence * 0.75
                    elif scale > 1.2:
                        adjusted_confidence = min(1.0, base_confidence * 1.1)

                self.last_detection_source = "rules_strong" if not sentiment_pre else "rules_strong_neural"
                self.last_rule_score = 2.5
                context_parts = [f"[对自我表达喜欢/{intensity}] {user_text[:50]}"]
                if sentiment_pre:
                    context_parts.append(f"[{sentiment_pre.label_zh},{sentiment_pre.confidence:.0%}]")

                return EmotionEvent(
                    trigger_type='intimacy',
                    deltas=scaled_deltas,
                    confidence=round(adjusted_confidence, 3),
                    context=" ".join(context_parts),
                )
            elif target_type == "other" and target:
                self.last_detection_source = "rules_strong"
                self.last_rule_score = -2.0
                cfg = self.TRIGGER_DELTAS['other_ai_mentioned']
                intensity = self.classify_intensity(user_text)
                return EmotionEvent(
                    trigger_type='other_ai_mentioned',
                    deltas=self.get_tiered_deltas('other_ai_mentioned', intensity),
                    confidence=cfg['confidence'],
                    context=f"[喜欢他人:{target}/{intensity}] {user_text[:50]}",
                )

        # ── Pre-Stage 1b: "[name] + positive adjective" detection ────
        # Catches "[某名字]好可爱/好漂亮/真好看" without action verbs.
        # If the name is not in agent_names → other_ai_mentioned.
        PRAISE_ABOUT = re.search(
            r'([^\s，。！？、]{2,8})(?:好|真|太|超|最)(?:可爱|漂亮|好看|厉害|强|温柔|棒|帅|美|甜|乖)',
            user_text
        )
        if PRAISE_ABOUT:
            praised_name = PRAISE_ABOUT.group(1)
            # Strip leading particles/filler
            praised_name = re.sub(r'^[是的那这个]', '', praised_name)
            if praised_name and not self.is_self_target(praised_name):
                # Check it's not a non-person object
                NON_PERSON_B = r'(?:这个|那个|这种|这样|什么|功能|东西|设计|方案|地方|游戏|歌|曲|音乐|风格|感觉|氛围|个)'
                if not re.match(NON_PERSON_B, praised_name):
                    self.last_detection_source = "rules_strong"
                    self.last_rule_score = -2.0
                    cfg = self.TRIGGER_DELTAS['other_ai_mentioned']
                    intensity = self.classify_intensity(user_text)
                    return EmotionEvent(
                        trigger_type='other_ai_mentioned',
                        deltas=self.get_tiered_deltas('other_ai_mentioned', intensity),
                        confidence=cfg['confidence'] * 0.85,  # slightly lower — no explicit "like"
                        context=f"[赞美他人:{praised_name}/{intensity}] {user_text[:50]}",
                    )

        # ── Stage 1: Rule-based scoring ──────────────────────────────
        best_trigger = None
        best_score = 0
        best_match_count = 0

        for trigger_type, (patterns, priority) in self._pattern_groups.items():
            match_count = self._count_matches(user_text, patterns)
            if match_count > 0:
                score = priority * (1.0 + 0.2 * (match_count - 1))
                if score > best_score:
                    best_score = score
                    best_trigger = trigger_type
                    best_match_count = match_count

        # ── Stage 2: Neural sentiment analysis ───────────────────────
        # Rule-first policy: strong rule matches stay rule-only; weak matches
        # and rule misses can lazy-load the shared analyzer for nuance/fallback.
        sentiment: Optional[SentimentResult] = None
        strong_rule = bool(best_trigger and best_score >= 80)
        should_use_neural = (
            self._use_model
            and self.neural_policy != "never"
            and (
                self.neural_policy == "always"
                or not strong_rule
            )
        )
        analyzer = self._ensure_analyzer() if should_use_neural else None
        if analyzer is not None:
            sentiment = self._analyze_neural_bounded(analyzer, user_text)
            self.last_neural_used = sentiment is not None
            if sentiment:
                self.last_neural_confidence = sentiment.confidence
                logger.debug(
                    f"Sentiment: {sentiment.label_zh} "
                    f"(conf={sentiment.confidence:.2f}, {sentiment.inference_ms:.0f}ms)"
                )

        # ── Stage 3: Fusion ──────────────────────────────────────────
        if best_trigger and best_trigger in self.TRIGGER_DELTAS:
            trigger_config = self.TRIGGER_DELTAS[best_trigger]
            base_confidence = trigger_config['confidence']

            # Classify intensity using text features + match count + sentiment
            sent_conf = sentiment.confidence if sentiment else 0.0
            intensity = self.classify_intensity(user_text, best_match_count, sent_conf)
            raw_deltas = self.get_tiered_deltas(best_trigger, intensity)

            # Scale deltas by sentiment fusion factor only when neural was used.
            if analyzer is not None and sentiment is not None:
                scale = analyzer.get_fusion_scale(best_trigger, sentiment)
            else:
                scale = 1.0

            scaled_deltas = {
                dim: int(round(val * scale))
                for dim, val in raw_deltas.items()
            }
            # Clamp individual deltas to reasonable range
            scaled_deltas = {
                dim: max(-30, min(30, val))
                for dim, val in scaled_deltas.items()
            }

            # Adjust confidence: if model contradicts rule direction, lower it
            adjusted_confidence = base_confidence
            if sentiment is not None:
                if scale < 0.7:
                    # Model strongly contradicts rule → reduce confidence
                    adjusted_confidence = base_confidence * 0.75
                elif scale > 1.2:
                    # Model strongly confirms rule → boost confidence slightly
                    adjusted_confidence = min(1.0, base_confidence * 1.1)

            self.last_rule_score = {
                'intimacy': 2.5,
                'teasing': 1.5,
                'praise': 2.0,
                'care': 1.5,
                'greeting': 1.0,
                'apology': 1.5,
                'sharing': 1.5,
                'other_ai_mentioned': -2.0,
                'criticism': -2.0,
                'ignored': -1.5,
            }.get(best_trigger, 0.0)
            self.last_detection_source = (
                "rules_strong" if strong_rule and not sentiment else
                "rules_strong_neural" if strong_rule and sentiment else
                "rules_weak_neural" if sentiment else
                "rules_weak"
            )
            context_parts = [f"[{intensity}] {user_text[:50]}"]
            if sentiment:
                context_parts.append(f"[{sentiment.label_zh},{sentiment.confidence:.0%}]")

            return EmotionEvent(
                trigger_type=best_trigger,
                deltas=scaled_deltas,
                confidence=round(adjusted_confidence, 3),
                context=" ".join(context_parts),
            )

        # ── Stage 4: Model-only fallback (rules missed, model caught) ─
        if analyzer is not None and sentiment is not None:
            fallback_trigger = analyzer.get_fallback_trigger(sentiment)
            if fallback_trigger and fallback_trigger in self.TRIGGER_DELTAS:
                # Model-only: use mild tier with 0.6 reduction (less certain)
                intensity = self.classify_intensity(user_text, 0, sentiment.confidence)
                fallback_raw = self.get_tiered_deltas(fallback_trigger, intensity)
                fallback_deltas = {
                    dim: int(round(val * 0.6))
                    for dim, val in fallback_raw.items()
                }
                trigger_config = self.TRIGGER_DELTAS[fallback_trigger]
                self.last_detection_source = "neural_fallback"
                self.last_rule_score = 0.0
                logger.debug(
                    f"Model-only fallback trigger: {fallback_trigger}/{intensity} "
                    f"from sentiment {sentiment.label}"
                )
                return EmotionEvent(
                    trigger_type=fallback_trigger,
                    deltas=fallback_deltas,
                    confidence=round(trigger_config['confidence'] * sentiment.confidence * 0.8, 3),
                    context=f"[{intensity}] {user_text[:50]} [model:{sentiment.label_zh}]",
                )

        # ── Stage 5: Neutral baseline ────────────────────────────────
        # Ordinary work/chat messages produce a tiny positive drift so the
        # relationship still warms over long sessions, but not fast enough
        # to race to max from task traffic alone.
        if user_text and len(user_text) > 10:
            self.last_detection_source = "neutral_baseline"
            self.last_rule_score = 0.0
            return EmotionEvent(
                trigger_type="normal_interaction",
                deltas={
                    "trust": 0.5,
                    "affection": 0.5,
                    "patience": 0.25,
                },
                confidence=0.3,
                context="normal_interaction"
            )

        return None
