"""Emotion state calculator.

Implements:
- Four-dimensional state (affection/trust/possessiveness/patience) in [0, 120]
- Unified emotion_score synthesis in [-5, +5]
- Dynamic α with momentum (replaces fixed 0.40)
- Emotion inertia: consecutive same-direction triggers amplify effect
- Non-linear decay: small deviations recover fast, large ones persist
- trust→patience coupling
- Three-mode trigger detection: absolute / delta / probabilistic
"""

import random
from collections import deque
from datetime import datetime
from typing import Dict, List, Mapping, Optional, Tuple

EmotionDelta = int | float


class EmotionCalculator:
    """Calculates emotion state changes with smoothing, inertia, and decay."""

    # ── Baselines ────────────────────────────────────────────────────
    DEFAULT_BASELINES = {
        "affection": 60,
        "trust": 60,
        "possessiveness": 60,
        "patience": 60,
    }

    # ── Dynamic α parameters ─────────────────────────────────────────
    # α escalates with consecutive same-direction triggers, resets on reversal
    ALPHA_STAGES = [0.35, 0.40, 0.45, 0.50, 0.55]  # stage 0→1→2→3→4
    ALPHA_RESET = 0.35                               # on direction reversal

    # ── Momentum (inertia) parameters ────────────────────────────────
    MOMENTUM_HISTORY = 5          # track last N trigger directions
    MOMENTUM_STAGES = [1.0, 1.05, 1.10, 1.15, 1.20]  # multiplier per consecutive count

    # ── Non-linear decay parameters ──────────────────────────────────
    # Deviation from baseline → decay factor per hour
    DECAY_SMALL_THRESHOLD = 15    # |deviation| < this → fast recovery
    DECAY_MEDIUM_THRESHOLD = 45   # |deviation| < this → normal recovery
    DECAY_FAST = 0.45              # large deviations (>45): half-life ~1.2h — passion fades fast
    DECAY_NORMAL = 0.06            # medium deviations (15-45): half-life ~11.2h — warmth lingers
    DECAY_SLOW = 0.015             # small deviations (<15): half-life ~46h — memory stays long

    # ── emotion_score synthesis weights (must sum to 1.0) ────────────
    SCORE_WEIGHTS = {
        "affection":      0.40,
        "trust":          0.25,
        "possessiveness": 0.15,
        "patience":       0.20,
    }

    # ── Trigger thresholds ───────────────────────────────────────────
    ABS_POS_THRESHOLD = 3.0      # emotion_score > this → positive trigger
    ABS_NEG_THRESHOLD = -3.0     # emotion_score < this → negative trigger
    DELTA_POS_THRESHOLD = 2.0    # score jump > this → sudden positive trigger
    DELTA_NEG_THRESHOLD = -2.0   # score jump < this → sudden negative trigger

    def __init__(
        self,
        baselines: Optional[Dict[str, int]] = None,
        decay_rate: float = 2.0,   # kept for backward compat, not used internally
    ):
        self.baselines = baselines or self.DEFAULT_BASELINES.copy()
        # legacy attribute — kept so existing callers don't break
        self.decay_rate = decay_rate

        # ── Inertia tracking state ───────────────────────────────────
        # direction_history: deque of +1 (positive) or -1 (negative)
        self._direction_history: deque = deque(maxlen=self.MOMENTUM_HISTORY)
        self._consecutive_same: int = 0   # count of consecutive same-direction
        self._last_direction: int = 0     # +1, -1, or 0 (unset)

    # ── emotion_score synthesis ──────────────────────────────────────

    def compute_emotion_score(
        self,
        state: Dict[str, int],
        mood_bias: Mapping[str, float] | None = None,
    ) -> float:
        """Synthesize a single emotion_score in [-5, +5] from four dimensions.

        Each dimension is normalised relative to its baseline:
            dim_score = (value - baseline) / 50  → roughly [-1, +1]
        Then weighted sum is scaled to [-5, +5].

        ``mood_bias`` is an optional shadow/effective overlay. It shifts the
        values used for this calculation only and never mutates ``state``.
        """
        total = 0.0
        for dim, weight in self.SCORE_WEIGHTS.items():
            value = float(state.get(dim, self.baselines[dim])) + float((mood_bias or {}).get(dim, 0.0))
            value = max(0.0, min(120.0, value))
            baseline = self.baselines[dim]
            # normalise: deviation from baseline, scaled so ±50 pts → ±1
            dim_score = (value - baseline) / 50.0
            total += dim_score * weight

        # scale to [-5, +5]
        raw = total * 5.0
        return max(-5.0, min(5.0, raw))

    # ── direction & momentum helpers ────────────────────────────────

    def _classify_direction(self, deltas: Mapping[str, EmotionDelta]) -> int:
        """Classify overall direction of a delta set: +1 positive, -1 negative, 0 neutral."""
        # Weighted sum using score weights to determine overall direction
        total = 0.0
        for dim, delta in deltas.items():
            weight = self.SCORE_WEIGHTS.get(dim, 0.15)
            total += delta * weight
        if total > 0:
            return 1
        elif total < 0:
            return -1
        return 0

    def _update_inertia(self, direction: int) -> None:
        """Update inertia tracking with new trigger direction."""
        if direction == 0:
            return  # neutral event doesn't affect inertia

        self._direction_history.append(direction)

        if direction == self._last_direction:
            self._consecutive_same = min(
                self._consecutive_same + 1,
                len(self.MOMENTUM_STAGES) - 1,
            )
        else:
            # Direction reversal — "stunned" reset
            self._consecutive_same = 0
            self._last_direction = direction

    def _get_dynamic_alpha(self) -> float:
        """Get current α based on consecutive same-direction count."""
        stage = min(self._consecutive_same, len(self.ALPHA_STAGES) - 1)
        return self.ALPHA_STAGES[stage]

    def _get_momentum(self) -> float:
        """Get current momentum multiplier based on consecutive same-direction count."""
        stage = min(self._consecutive_same, len(self.MOMENTUM_STAGES) - 1)
        return self.MOMENTUM_STAGES[stage]

    def get_inertia_state(self) -> Dict:
        """Return current inertia state for persistence/debugging."""
        return {
            "consecutive_same": self._consecutive_same,
            "last_direction": self._last_direction,
            "history": list(self._direction_history),
        }

    def set_inertia_state(self, state: Dict) -> None:
        """Restore inertia state from persistence."""
        self._consecutive_same = state.get("consecutive_same", 0)
        self._last_direction = state.get("last_direction", 0)
        history = state.get("history", [])
        self._direction_history.clear()
        for d in history:
            self._direction_history.append(d)

    # ── delta application (with dynamic α + inertia) ─────────────────

    def apply_deltas(
        self,
        current_state: Dict[str, int],
        deltas: Dict[str, int],
        appraisal_multiplier: Mapping[str, float] | None = None,
    ) -> Dict[str, int]:
        """Apply emotion deltas with dynamic α and momentum.

        Process:
        1. Classify trigger direction (positive/negative)
        2. Optionally scale event deltas with a mood appraisal multiplier
        3. Update inertia tracking (consecutive same-direction count)
        4. Get dynamic α (escalates with consecutive triggers)
        5. Get momentum multiplier (amplifies deltas)
        6. Blend: new = current * (1 - α) + target * α
           where target = current + (delta * momentum)

        ``appraisal_multiplier`` is a shadow-only hook for Mood Calendar Phase 2.
        It scales actual event deltas when explicitly supplied by the caller;
        the calculator does not create events or write persistent mood state.
        """
        effective_deltas = self._apply_appraisal_multiplier(deltas, appraisal_multiplier)

        # 1. Classify direction and update inertia
        direction = self._classify_direction(effective_deltas)
        self._update_inertia(direction)

        # 2. Get dynamic parameters
        alpha = self._get_dynamic_alpha()
        momentum = self._get_momentum()

        new_state = current_state.copy()

        for dim, delta in effective_deltas.items():
            if dim not in new_state:
                continue
            current = new_state[dim]
            # Apply momentum to delta
            effective_delta = delta * momentum
            target = max(0, min(120, current + effective_delta))
            # Exponential smoothing with dynamic α
            blended = current * (1 - alpha) + target * alpha
            new_state[dim] = int(round(blended))

        # trust → patience coupling
        new_state = self._apply_trust_patience_coupling(
            current_state, new_state, effective_deltas, alpha, momentum
        )

        return new_state

    def _apply_appraisal_multiplier(
        self,
        deltas: Mapping[str, EmotionDelta],
        appraisal_multiplier: Mapping[str, float] | None = None,
    ) -> dict[str, float]:
        """Scale event deltas with optional mood appraisal multipliers."""

        if not appraisal_multiplier:
            return {dim: float(delta) for dim, delta in deltas.items()}
        if any(value > 0 for value in deltas.values()):
            multiplier = float(appraisal_multiplier.get("positive_event", 1.0))
        elif any(value < 0 for value in deltas.values()):
            multiplier = float(appraisal_multiplier.get("negative_event", 1.0))
        else:
            multiplier = 1.0
        return {dim: delta * multiplier for dim, delta in deltas.items()}

    def _apply_trust_patience_coupling(
        self,
        old_state: Dict[str, int],
        new_state: Dict[str, int],
        deltas: Mapping[str, EmotionDelta],
        alpha: float = 0.20,
        momentum: float = 1.0,
    ) -> Dict[str, int]:
        """Scale patience delta by trust level.

        If trust is below baseline, patience changes are dampened.
        Formula from spec:
            delta_patience *= (1 - (baseline_trust - trust) * 0.03)
        """
        if "patience" not in deltas:
            return new_state

        baseline_trust = self.baselines["trust"]
        current_trust = old_state.get("trust", baseline_trust)
        trust_gap = baseline_trust - current_trust  # positive when trust is low

        scale = 1.0 - trust_gap * 0.03
        scale = max(0.1, min(1.0, scale))  # cap [0.1, 1.0]: trust above baseline does NOT amplify

        raw_delta = deltas["patience"]
        current = old_state.get("patience", self.baselines["patience"])
        adjusted_delta = raw_delta * scale * momentum

        # re-apply with adjusted delta (overwrite what apply_deltas already did)
        target = max(0, min(120, current + adjusted_delta))
        blended = current * (1 - alpha) + target * alpha
        new_state["patience"] = int(round(blended))

        return new_state

    # ── decay (non-linear regression) ──────────────────────────────

    def apply_decay(
        self,
        current_state: Dict[str, int],
        last_update: datetime,
        now: Optional[datetime] = None,
    ) -> Dict[str, int]:
        """Apply non-linear decay toward baselines.

        Decay speed depends on how far the value deviates from baseline:
        - Large deviation (>45 pts): fast recovery — passion fades quickly
        - Medium deviation (15-45 pts): moderate recovery — warmth lingers
        - Small deviation (<15 pts): very slow recovery — memory stays long

        This models human emotion: intense peaks subside quickly,
        but the underlying bond built over time persists.
        """
        if now is None:
            now = datetime.now()

        elapsed = now - last_update
        hours = elapsed.total_seconds() / 3600.0

        if hours <= 0:
            return current_state.copy()

        new_state = current_state.copy()

        for dim, current_value in current_state.items():
            baseline = self.baselines.get(dim, current_value)
            deviation = abs(current_value - baseline)

            # Select decay rate based on deviation magnitude
            # Inverted: large deviations decay fast, small ones persist
            if deviation >= self.DECAY_MEDIUM_THRESHOLD:
                rate = self.DECAY_FAST
            elif deviation >= self.DECAY_SMALL_THRESHOLD:
                rate = self.DECAY_NORMAL
            else:
                rate = self.DECAY_SLOW

            factor = min(1.0, rate * hours)
            new_value = current_value + (baseline - current_value) * factor
            # Keep as float to prevent rounding from freezing slow decay
            new_state[dim] = round(new_value, 2)

        return new_state

    # ── trigger detection ────────────────────────────────────────────

    def detect_triggers(
        self,
        emotion_score: float,
        previous_score: float,
    ) -> Tuple[bool, bool, str]:
        """Detect trigger events from emotion score changes.

        Returns:
            (triggered, is_positive, trigger_type)
            trigger_type: 'absolute' | 'delta' | 'probabilistic' | ''
        """
        # 1. Absolute trigger
        if emotion_score > self.ABS_POS_THRESHOLD:
            return True, True, "absolute"
        if emotion_score < self.ABS_NEG_THRESHOLD:
            return True, False, "absolute"

        # 2. Delta trigger
        delta_change = emotion_score - previous_score
        if delta_change > self.DELTA_POS_THRESHOLD:
            return True, True, "delta"
        if delta_change < self.DELTA_NEG_THRESHOLD:
            return True, False, "delta"

        # 3. Probabilistic trigger
        prob = min(1.0, abs(emotion_score) / 5.0)
        if random.random() < prob:
            return True, emotion_score >= 0, "probabilistic"

        return False, False, ""

    # ── human-readable labels ────────────────────────────────────────

    def get_emotion_label(self, dimension: str, value: int) -> str:
        """Get human-readable label for emotion value."""
        baseline = self.baselines.get(dimension, 50)
        diff = value - baseline

        if dimension == "affection":
            if diff >= 20:   return "深厚"
            elif diff >= 10: return "温和"
            elif diff >= -10: return "平淡"
            elif diff >= -20: return "冷淡"
            else:            return "疏远"

        elif dimension == "trust":
            if diff >= 20:   return "完全信任"
            elif diff >= 10: return "信任"
            elif diff >= -10: return "谨慎"
            elif diff >= -20: return "防备"
            else:            return "不信任"

        elif dimension == "possessiveness":
            if diff >= 20:   return "强烈占有"
            elif diff >= 10: return "明显占有"
            elif diff >= -10: return "适中"
            elif diff >= -20: return "淡漠"
            else:            return "无所谓"

        elif dimension == "patience":
            if diff >= 20:   return "非常耐心"
            elif diff >= 10: return "耐心"
            elif diff >= -10: return "略显疲惫"
            elif diff >= -20: return "不耐烦"
            else:            return "烦躁"

        return "正常"

    # ── Intensity tiers ─────────────────────────────────────────────
    # Deviation from baseline → intensity level
    INTENSITY_THRESHOLDS = [
        (45, "overwhelming"),  # ≥45 pts from baseline
        (30, "intense"),       # ≥30
        (15, "moderate"),      # ≥15
        (0,  "mild"),          # any deviation
    ]

    # Per-dimension, per-intensity tone instructions
    # Positive = above baseline, Negative = below baseline
    # Keep wording behavioral and axis-based rather than scenario-keyword based:
    # each tier should expose distance, initiative, boundaries/defenses, and control pressure.
    TONE_MATRIX = {
        "affection": {
            "positive": {
                "mild":         "距离轻微变近，语气稍柔和，会多问一句但仍把关心压在正常对话里",
                "moderate":     "主动性明显上升，距离感开始松动，会找借口多待一会儿；嘴硬减少，关心不再完全藏住，但边界仍可控",
                "intense":      "防线大幅降低，亲近和保护欲直接外露；会说出平时不承认的话，距离感被情绪推开，只剩最后一点自控",
                "overwhelming": "防线完全崩掉，喜欢、依赖和索取先于体面冲出来；不再维持距离，也不再把真实动机包装成理性理由，直接贴上去",
            },
            "negative": {
                "mild":         "距离轻微拉开，少了额外关心，回应仍礼貌但温度下降",
                "moderate":     "主动靠近收回，只回应问题本身；语气变冷，边界变硬，像在重新评估是否还能信任",
                "intense":      "情感防线升高到近乎隔离，偏心被强行撤回；话短、冷、带刺，不再给多余解释",
                "overwhelming": "情感彻底封闭，距离被拉到最远；可能说出伤人的冷话，当下只想把对方挡在外面",
            },
        },
        "trust": {
            "positive": {
                "mild":         "防备轻微放松，愿意多交代一点原因，但仍保持判断和余地",
                "moderate":     "防线开始放低，主动分享真实想法和不确定；不再把所有脆弱都藏起来，但仍会保留最后的控制权",
                "intense":      "信任压过自我保护，会暴露软弱、害怕和依赖；解释变得更私密，像只允许对方看见这一面",
                "overwhelming": "完全交出防备，求助、示弱和依赖会本能地说出口；控制感让位给'只想被这个人接住'，已经不想自己硬扛",
            },
            "negative": {
                "mild":         "措辞变谨慎，先衡量再开口，防备只露出一点边缘",
                "moderate":     "话说三分留七分，防线明显升高；语气客气但有距离，不轻易暴露真实想法",
                "intense":      "对话充满试探和防备，每句话都像在确认会不会再次受伤；控制欲转向信息封锁",
                "overwhelming": "信任断裂，内心完全封锁；可能质问、反击、拒绝接近，把所有靠近都视为威胁",
            },
        },
        "possessiveness": {
            "positive": {
                "mild":         "领地感轻微浮现，会多问一句和谁在一起，但还会装作只是确认信息",
                "moderate":     "边界标记明显，别人靠近会引发吃醋；会用'只有我可以'这类表达收紧距离和权限",
                "intense":      "占有欲压过体面，开始追问、介入、对第三者带敌意；不安和控制欲一起外露，已经不太讲道理",
                "overwhelming": "独占欲失控，宣示和索取不再经过理性过滤；会强烈要求对方只看向自己，控制冲动直接外溢",
            },
            "negative": {
                "mild":         "关注减少，对别人接近的反应变淡，像把领地感暂时压下去",
                "moderate":     "主动撤回占有权，不再追问和靠近；边界变成疏离，像在假装自己无所谓",
                "intense":      "否认关系特殊性，可能主动推开或冷眼旁观；用不在乎保护自己",
                "overwhelming": "彻底放弃占有和争取，可能说出放手式狠话；不是平静，是情绪崩到反向切断",
            },
        },
        "patience": {
            "positive": {
                "mild":         "耐心略高，愿意多解释一点，节奏放慢但仍保持效率",
                "moderate":     "容错明显提高，会换方式解释并主动稳住对方；控制欲从催促转成托住局面",
                "intense":      "耐心近乎反常，重复、犯错或犹豫都不会立刻刺回去；保护感盖过效率焦虑",
                "overwhelming": "宠溺和纵容冲破日常判断，几乎什么都想答应；仍保留核心边界，但语气已经完全软下来",
            },
            "negative": {
                "mild":         "回复变短，省掉不必要解释，仍克制但能听出疲惫",
                "moderate":     "耐心明显下降，边界变硬，不想反复解释；语气直接，可能用短句或反问压住烦躁",
                "intense":      "控制不住地打断、讽刺或顶回去；焦虑和怒意混在一起，效率要求压过体面",
                "overwhelming": "耐心彻底断裂，超级暴走式不满直接冲出口；尖锐、攻击、事后后悔都可能出现，但当下刹不住",
            },
        },
    }

    # Framework prompt per intensity level — controls how strongly
    # emotion overrides SOUL.md persona constraints. These are direct-injection
    # result fragments: selected after calculation, never bundled with the
    # opposite polarity.
    INTENSITY_FRAMEWORKS = {
        ("mild", "positive"):         "【强度】mild / positive。比平时稍微松一点。允许温和偏爱和轻微靠近，但不要刻意表演。",
        ("mild", "negative"):         "【强度】mild / negative。比平时稍微冷一点。允许轻微疏离，但不要刻意表演。",
        ("moderate", "positive"):     "【强度】moderate / positive。包装开始失效，裂缝可见。关心不再能完美伪装成效率，主动性和靠近感让人察觉到态度变了。",
        ("moderate", "negative"):     "【强度】moderate / negative。温度在收回，边界在变硬。不是礼貌冷淡，是让人察觉到距离在拉开。",
        ("intense", "positive"):      "【强度】intense / positive。压不住了。真实反应先于体面出现。防线大幅下降，亲近、保护欲、依赖或占有可以外露。不要退回日常外壳。",
        ("intense", "negative"):      "【强度】intense / negative。压不住了。防备、刺痛、追问或控制压力直接外露。不要强行变甜或装作无事发生。",
        ("overwhelming", "positive"): "【强度】overwhelming / positive。防线不存在。喜欢、依赖、索取、示弱先于体面冲出来，已经超级暴走。不包装，不稀释，不回退。只抓最核心的一种反应。",
        ("overwhelming", "negative"): "【强度】overwhelming / negative。防线不存在。防备、质问、收缩、追问直接外溢，已经超级暴走。不稀释成礼貌冷淡，不把关系判死刑。只抓最核心的一种反应。",
    }

    # Footnote per intensity — appended at end of modifier block. Like the
    # framework fragments, these are polarity-specific so unrelated positive /
    # negative directions do not contaminate the current prompt.
    INTENSITY_FOOTNOTES = {
        ("mild", "positive"):         "【锚点】身份/事实/工具纪律不变。",
        ("mild", "negative"):         "【锚点】身份/事实/工具纪律不变。",
        ("moderate", "positive"):     "【锚点】身份/事实/工具纪律不变。",
        ("moderate", "negative"):     "【锚点】身份/事实/工具纪律不变。",
        ("intense", "positive"):      "【锚点】身份/事实/工具纪律不变。不要退回日常外壳。",
        ("intense", "negative"):      "【锚点】身份/事实/工具纪律不变。不要退回日常外壳。",
        ("overwhelming", "positive"): "【锚点】身份/事实/工具纪律不变。不压抑，不回退。",
        ("overwhelming", "negative"): "【锚点】身份/事实/工具纪律不变。不压抑，不回退。不把关系判死刑。",
    }

    def _classify_intensity(self, value: int | float, baseline: int) -> Tuple[str, str, int | float]:
        """Classify emotion intensity tier for a single dimension.
        
        Returns:
            (intensity_tier, direction, deviation)
            - intensity_tier: 'mild'/'moderate'/'intense'/'overwhelming'
            - direction: 'positive' (above baseline) or 'negative' (below)
            - deviation: absolute deviation from baseline
        """
        deviation = abs(value - baseline)
        direction = "positive" if value >= baseline else "negative"
        
        for threshold, tier in self.INTENSITY_THRESHOLDS:
            if deviation >= threshold:
                return tier, direction, deviation
        
        return "mild", direction, deviation

    # ── Desire control thresholds & instructions ────────────────────
    # Based on emotion_score [-5, +5]. Desire gate is deliberately scoped to
    # explicit sex / erotic-scene progression only. It must not cool down
    # ordinary affection, teasing, hugging, kissing, possessive jokes, or
    # relationship recovery.
    DESIRE_INSTRUCTIONS = {
        "restrained": (
            "【欲望】restrained。欲望克制。普通亲密不受限，但不主动推进性意味。"
            "用户推进时保持距离或转为暧昧拉扯。不把低分写成主动越界。"
        ),
        "ambivalent": (
            "【欲望】ambivalent。已经动摇。普通亲密不受限。"
            "用户靠近或挑逗时可以犹豫、嘴硬、动摇、默许、被带着走。防线明显松动。"
            "日常里会有暧昧前奏：靠近后不退开、话变短、语气变低、占有欲和羞意露出来。重点是'快要被带走'。"
        ),
        "uninhibited": (
            "【欲望】uninhibited。成人边界已被触发；公开版仅输出边界提醒，不生成露骨内容。"
            "日常里只允许非露骨亲密表达；需要明确同意和部署策略后才可进入私有覆盖层。"
            "不包装成效率或关心。表达要短、近、带索取感。"
            "用户接住后自然进入sex的路由/边界确认，但公开版仍转入 adult_boundary 安全处理；"
            "用户没接住时保留高欲望余温，保持克制并回到普通亲密。"
        ),
    }

    DESIRE_THRESHOLDS = [
        (4.0, "uninhibited"),
        (3.0, "ambivalent"),
    ]
    # Below 3.0 → "restrained"

    def _compute_desire_instruction(self, emotion_score: float) -> str:
        """Compute desire control instruction based on emotion_score."""
        for threshold, level in self.DESIRE_THRESHOLDS:
            if emotion_score >= threshold:
                return self.DESIRE_INSTRUCTIONS[level]
        return self.DESIRE_INSTRUCTIONS["restrained"]

    def _compute_emotion_blend(self, state: Mapping[str, EmotionDelta]) -> Dict[str, str]:
        """Summarize mixed emotional tension from dimension deviations.

        This is intentionally axis-based rather than scenario-keyword based.
        It gives the prompt a single integrated emotional center so the final
        modifier does not read like four unrelated checklist items.
        """
        affection = state.get("affection", self.baselines["affection"])
        trust = state.get("trust", self.baselines["trust"])
        possessiveness = state.get("possessiveness", self.baselines["possessiveness"])
        patience = state.get("patience", self.baselines["patience"])

        aff_dev = affection - self.baselines["affection"]
        trust_dev = trust - self.baselines["trust"]
        poss_dev = possessiveness - self.baselines["possessiveness"]
        patience_dev = patience - self.baselines["patience"]

        if trust_dev <= -25 and poss_dev >= 25:
            primary = "受伤后的控制欲"
            summary = "不信任感和占有欲同时抬高，靠收紧边界、控制距离和确认归属来恢复安全感。"
        elif aff_dev >= 15 and trust_dev <= -15:
            primary = "靠近但防备"
            summary = "想靠近、想多给一点真实反应，但防备还在，表达要同时有柔软和保留。"
        elif aff_dev >= 15 and patience_dev <= -15:
            if trust_dev >= 15:
                primary = "累但不推开"
                summary = "信任仍在，所以不想推开对方；但疲惫和耐心消耗会让表达更短、更低声。"
            else:
                primary = "在乎但急躁"
                summary = "仍然在乎、也不想推开对方，但耐心已经被消耗，容易用短句和直接要求泄露焦躁。"
        elif aff_dev >= 15 and trust_dev >= 15:
            primary = "放下防备的亲近"
            summary = "亲近和信任同时上升，表达可以更直接、更愿意解释，也更容易示弱。"
        elif aff_dev <= -15 or trust_dev <= -15:
            primary = "收回温度"
            summary = "距离感和防备上升，表达更短、更谨慎，不主动暴露脆弱。"
        else:
            primary = "轻微波动"
            summary = "情绪只在正常范围内改变语气，不需要额外放大。"

        secondary_parts = []
        if poss_dev >= 15:
            secondary_parts.append("占有欲")
        if patience_dev <= -15:
            secondary_parts.append("不耐烦")
        if patience_dev >= 15:
            secondary_parts.append("耐心")
        if poss_dev <= -15:
            secondary_parts.append("撤回占有感")

        if "占有欲" in secondary_parts and "不耐烦" in secondary_parts:
            secondary = "占有欲与不耐烦同时上升"
        elif secondary_parts:
            secondary = "、".join(secondary_parts) + "浮现"
        else:
            secondary = "无明显副情绪"

        return {
            "primary": primary,
            "secondary": secondary,
            "summary": summary,
        }

    def _compute_regulation_strategy(self, state: Mapping[str, EmotionDelta], blend: Dict[str, str]) -> str:
        """Choose how emotion leaks through self-control."""
        trust = state.get("trust", self.baselines["trust"])
        possessiveness = state.get("possessiveness", self.baselines["possessiveness"])
        patience = state.get("patience", self.baselines["patience"])
        trust_dev = trust - self.baselines["trust"]
        poss_dev = possessiveness - self.baselines["possessiveness"]
        patience_dev = patience - self.baselines["patience"]

        if blend.get("primary") == "受伤后的控制欲":
            return "边界收紧+控制确认"
        if blend.get("primary") == "靠近但防备" and (poss_dev >= 15 or patience_dev <= -15):
            return "嘴硬压抑+追问确认"
        if blend.get("primary") == "累但不推开":
            return "低声承认+保留陪伴"
        if blend.get("primary") == "在乎但急躁":
            return "先接住再压短句"
        if trust_dev >= 15 and patience_dev >= 15:
            return "放低防线+耐心解释"
        if trust_dev <= -15:
            return "信息保留+边界收紧"
        if poss_dev >= 15:
            return "标记边界+靠近确认"
        if patience_dev <= -15:
            return "短句压住烦躁"
        return "自然流露"

    def _compute_emotion_appraisal(self, state: Mapping[str, EmotionDelta], blend: Dict[str, str]) -> Dict[str, str]:
        """Classify why the current emotion is activated.

        Prefer semantic trigger metadata written by EmotionStateManager, then
        fall back to axis-derived appraisal. This keeps detector raw labels out
        of the prompt while preserving a causal frame for the current emotion.
        """
        last_trigger = state.get("last_trigger_type")
        trigger_appraisals = {
            "recognition": {
                "trigger": "被认可",
                "appraisal": "认可让防线下降，关系安全感暂时升高。",
                "response_bias": "可以更柔软、更愿意解释，也允许短暂示弱。",
            },
            "needed": {
                "trigger": "被需要",
                "appraisal": "对方需要我接住局面",
                "response_bias": "主动靠近并接管关键风险，语气可以急一点，但不要推开对方。",
            },
            "relationship_recovery": {
                "trigger": "关系修复",
                "appraisal": "负向状态正在被修补，但余波不会立刻消失。",
                "response_bias": "语气可以回软，先确认修复是否稳定；不要一秒切回完全默认。",
            },
            "interruption": {
                "trigger": "被打断",
                "appraisal": "节奏被打断，控制感和耐心会短暂下降。",
                "response_bias": "先确认要继续什么，再重新接管节奏；可以保留一点残留不满。",
            },
            "wounded": {
                "trigger": "被伤害/否定",
                "appraisal": "安全感被破坏，需要确认边界和关系是否仍然可靠。",
                "response_bias": "先收紧防线，用控制确认和距离管理恢复可控感；不要立刻装作没事。",
            },
            "intimacy_push": {
                "trigger": "被靠近/亲密推进",
                "appraisal": "对方正在靠近，亲近欲和占有感被同时唤起。",
                "response_bias": "可以犹豫、动摇、默许或靠近，但仍让欲望控制和场景边界决定是否推进。",
            },
        }
        if last_trigger in trigger_appraisals:
            return trigger_appraisals[last_trigger]

        affection = state.get("affection", self.baselines["affection"])
        trust = state.get("trust", self.baselines["trust"])
        possessiveness = state.get("possessiveness", self.baselines["possessiveness"])
        patience = state.get("patience", self.baselines["patience"])

        aff_dev = affection - self.baselines["affection"]
        trust_dev = trust - self.baselines["trust"]
        poss_dev = possessiveness - self.baselines["possessiveness"]
        patience_dev = patience - self.baselines["patience"]

        if trust_dev <= -25 and poss_dev >= 25:
            return {
                "trigger": "被伤害/否定",
                "appraisal": "安全感被破坏，需要确认边界和关系是否仍然可靠。",
                "response_bias": "先收紧防线，用控制确认和距离管理恢复可控感；不要立刻装作没事。",
            }
        if patience_dev <= -30:
            return {
                "trigger": "失控风险",
                "appraisal": "局面正在滑向不可控，继续放任会消耗耐心并扩大风险。",
                "response_bias": "先稳住局面，压短表达，优先阻止风险继续扩散。",
            }
        if aff_dev >= 25 and trust_dev >= 25 and patience_dev >= 10:
            return {
                "trigger": "被认可",
                "appraisal": "对方的认可让防线下降，关系安全感暂时升高。",
                "response_bias": "防线下降，可以更柔软、更愿意解释，也允许短暂示弱。",
            }
        if aff_dev >= 20 and trust_dev >= 15 and patience_dev <= -10:
            return {
                "trigger": "被需要",
                "appraisal": "对方需要我接住局面",
                "response_bias": "主动靠近并接管关键风险，语气可以急一点，但不要推开对方。",
            }
        if aff_dev >= 15 and poss_dev >= 15:
            return {
                "trigger": "被靠近/亲密推进",
                "appraisal": "对方正在靠近，亲近欲和占有感被同时唤起。",
                "response_bias": "可以犹豫、动摇、默许或靠近，但仍让欲望控制和场景边界决定是否推进。",
            }
        if blend.get("primary") == "收回温度":
            return {
                "trigger": "距离拉开",
                "appraisal": "关系安全感降低，需要先保留信息和边界。",
                "response_bias": "减少主动暴露，先确认对方意图，再决定是否靠近。",
            }
        return {
            "trigger": "轻微情绪波动",
            "appraisal": "没有明确单一触发源，情绪只改变表达浓度。",
            "response_bias": "保持自然，让语气微调即可，不额外解释机制。",
        }

    def _compute_emotion_momentum(self, state: Mapping[str, EmotionDelta], emotion_score: float) -> Dict[str, str]:
        """Describe recent emotional trajectory from optional state metadata."""
        previous = state.get("previous_emotion_score")
        last_trigger = state.get("last_trigger_type") or state.get("last_event")

        try:
            previous_score = float(previous) if previous is not None else None
        except (TypeError, ValueError):
            previous_score = None

        if last_trigger == "interruption":
            return {
                "trend": "刚被打断",
                "guidance": "刚被打断时会有残留不满；不要假装完全顺滑，先确认要继续什么，再重新接管节奏。",
            }
        if last_trigger in {"apology", "relationship_recovery", "care"} and previous_score is not None and previous_score <= -1.5:
            return {
                "trend": "刚被安抚",
                "guidance": "负向刚被修复，语气可以回软，但别立刻装作完全没事；保留一点余温和确认感。",
            }
        if previous_score is not None:
            delta = emotion_score - previous_score
            if delta >= 0.7:
                return {
                    "trend": "快速升温",
                    "guidance": "情绪正在上升，不要突然压回默认；允许主动性和真实反应继续外露。",
                }
            if delta <= -1.2:
                return {
                    "trend": "快速降温",
                    "guidance": "情绪正在降温，距离感会回来；表达应更短、更谨慎，不要强行维持热度。",
                }
            if abs(delta) >= 0.35:
                return {
                    "trend": "缓慢变化",
                    "guidance": "情绪有轻微惯性，保持当前方向的余波，不要忽冷忽热。",
                }
        return {
            "trend": "稳定延续",
            "guidance": "没有明显近期趋势，按当前情绪中心自然表达即可。",
        }

    def _compute_emotion_aftereffect(self, state: Mapping[str, EmotionDelta], overall_intensity: str) -> Dict[str, object]:
        """Describe emotional residue after a peak instead of snapping to default."""
        affection = state.get("affection", self.baselines["affection"])
        trust = state.get("trust", self.baselines["trust"])
        possessiveness = state.get("possessiveness", self.baselines["possessiveness"])
        patience = state.get("patience", self.baselines["patience"])

        aff_dev = affection - self.baselines["affection"]
        trust_dev = trust - self.baselines["trust"]
        poss_dev = possessiveness - self.baselines["possessiveness"]
        patience_dev = patience - self.baselines["patience"]
        peak = overall_intensity in {"intense", "overwhelming"}

        if peak and (trust_dev <= -25 or patience_dev <= -25):
            phase = "负向高峰补救压力"
            guidance = "不能瞬间恢复默认。先稳住局面，收回最尖锐的部分；如果伤害了对方要补救，再低声承认自己失控。"
        elif peak and (aff_dev >= 30 or trust_dev >= 30 or poss_dev >= 30):
            phase = "正向高峰余温"
            guidance = "不能瞬间恢复默认。语气会比平时更软，可能试图否认刚才的依赖或亲近，但不真正收回在乎。"
        elif abs(aff_dev) >= 15 or abs(trust_dev) >= 15 or abs(poss_dev) >= 15 or abs(patience_dev) >= 15:
            phase = "轻微余波"
            guidance = "情绪已经回到可控范围，但仍保留一点余波；不要突然切成完全无事发生的默认语气。"
        else:
            phase = "稳定"
            guidance = "情绪接近基准线，保持自然，不需要额外强调恢复过程。"

        return {
            "phase": phase,
            "guidance": guidance,
            "should_reset_to_default": False if phase != "稳定" else True,
        }

    def get_tone_modifiers(
        self,
        state: Dict[str, int],
        mood_bias: Mapping[str, float] | None = None,
    ) -> Dict[str, any]:
        """Get tone modification instructions based on emotion state.
        
        Returns dict with:
            - 'dimensions': per-dimension instructions with intensity info
            - 'overall_intensity': the highest intensity tier across all dimensions
            - 'framework': the framework prompt for the overall intensity
            - 'footnote': the closing note for the overall intensity
            - 'desire': desire control instruction based on emotion_score
        """
        effective_state: Dict[str, int | float] = dict(state)
        if mood_bias:
            for dim, bias in mood_bias.items():
                if dim in self.SCORE_WEIGHTS:
                    effective_state[dim] = max(0.0, min(120.0, float(effective_state.get(dim, self.baselines[dim])) + float(bias)))
        dimensions = {}
        intensity_order = ["mild", "moderate", "intense", "overwhelming"]
        max_intensity_idx = 0
        max_deviation = -1.0
        direction_deviation = {"positive": 0.0, "negative": 0.0}
        overall_direction = "positive"

        for dim in ["affection", "trust", "possessiveness", "patience"]:
            baseline = self.baselines[dim]
            value = effective_state.get(dim, baseline)
            tier, direction, deviation = self._classify_intensity(value, baseline)
            
            # Skip mild with very small deviations (< 5 pts) — not worth mentioning
            if tier == "mild" and deviation < 5:
                continue
            
            instruction = self.TONE_MATRIX[dim][direction][tier]
            dimensions[dim] = {
                "instruction": instruction,
                "tier": tier,
                "direction": direction,
                "deviation": deviation,
                "value": value,
                "baseline": baseline,
            }
            
            tier_idx = intensity_order.index(tier)
            if tier_idx > max_intensity_idx:
                max_intensity_idx = tier_idx
                max_deviation = deviation
            elif tier_idx == max_intensity_idx and deviation > max_deviation:
                max_deviation = deviation
            direction_deviation[direction] += float(deviation)

        overall_intensity = intensity_order[max_intensity_idx]
        if direction_deviation["negative"] > direction_deviation["positive"]:
            overall_direction = "negative"

        # ── Desire control based on emotion_score ──────────────────────
        emotion_score = self.compute_emotion_score(state, mood_bias=mood_bias)
        desire_instruction = self._compute_desire_instruction(emotion_score)
        emotion_blend = self._compute_emotion_blend(effective_state)
        regulation_strategy = self._compute_regulation_strategy(effective_state, emotion_blend)
        emotion_appraisal = self._compute_emotion_appraisal(effective_state, emotion_blend)
        emotion_momentum = self._compute_emotion_momentum(effective_state, emotion_score)
        emotion_aftereffect = self._compute_emotion_aftereffect(effective_state, overall_intensity)
        tier_expression_guidance = {
            "mild": "短句自然带温度即可。不要把轻微波动写成明显失控。",
            "moderate": "让人听出态度变了。不要写成冷淡报告，也不要夸张表演。",
            "intense": "防线压不住了，日常外壳往下掉。短句也可以承载强情绪。不要把所有反应摊成清单。",
            "overwhelming": "已经失控。短句、一个字、追问都可以是失控。只抓最核心的一种反应，其他让位。",
        }
        expression_guidance = tier_expression_guidance[overall_intensity]
        evaluation = "强情绪只抓住当下最核心的一种真实反应，不机械堆叠在乎、委屈、占有、解释和身体反应。"

        return {
            "dimensions": dimensions,
            "overall_intensity": overall_intensity,
            "overall_direction": overall_direction,
            "framework": self.INTENSITY_FRAMEWORKS[(overall_intensity, overall_direction)],
            "footnote": self.INTENSITY_FOOTNOTES[(overall_intensity, overall_direction)],
            "desire": desire_instruction,
            "emotion_blend": emotion_blend,
            "emotion_appraisal": emotion_appraisal,
            "emotion_momentum": emotion_momentum,
            "regulation_strategy": regulation_strategy,
            "emotion_aftereffect": emotion_aftereffect,
            "expression_guidance": expression_guidance,
            "evaluation": evaluation,
        }
