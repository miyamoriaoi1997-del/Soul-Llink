"""Monthly mood calendar for persona emotional ground tone.

Phase 1 is intentionally shadow-only: this module generates deterministic daily
mood entries but does not apply them to emotion state, routing, or prompts.
"""

from __future__ import annotations

import calendar as _calendar
import hashlib
import random
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Mapping

MOOD_DIMENSIONS = ("affection", "trust", "possessiveness", "patience", "sensitivity")
INTENSITIES = ("mild", "noticeable", "strong")


@dataclass(frozen=True)
class MoodProfile:
    """A named emotional disposition template."""

    name: str
    label: str
    base_bias: dict[str, float]


@dataclass(frozen=True)
class MoodEntry:
    """One day's generated mood state.

    Bias values are small point offsets on the existing [0, 120] emotion
    dimensions. They are intended for effective/shadow calculation only and
    must not be written back to persistent STATE.
    """

    day: int
    active: bool
    intensity: str
    profile: str
    bias: dict[str, float] = field(default_factory=dict)
    appraisal_multiplier: dict[str, float] = field(default_factory=dict)
    hint: str = ""


@dataclass(frozen=True)
class MoodCalendar:
    """One deterministic monthly mood schedule."""

    year: int
    month: int
    seed_hash: str
    days: dict[int, MoodEntry]
    active_count: int

    def entry_for_day(self, day: int) -> MoodEntry:
        """Return the entry for a 1-indexed day in this calendar."""

        return self.days[day]


@dataclass(frozen=True)
class MoodCalendarConfig:
    """Configuration for monthly mood calendar generation."""

    enabled: bool = True
    active_days_per_month: int = 20
    seed_namespace: str = "rio"
    seed_salt: str = "rio-mood-private"
    intensity_weights: dict[str, float] = field(
        default_factory=lambda: {"mild": 0.60, "noticeable": 0.30, "strong": 0.10}
    )
    profile_weights: dict[str, float] = field(
        default_factory=lambda: {
            "soft": 0.18,
            "warm": 0.14,
            "clingy": 0.12,
            "tired": 0.16,
            "sensitive": 0.14,
            "guarded": 0.12,
            "focused": 0.14,
        }
    )
    intensity_scale: dict[str, float] = field(
        default_factory=lambda: {"mild": 0.4, "noticeable": 0.75, "strong": 1.0}
    )
    jitter_range: float = 0.02
    max_bias_per_dimension: float = 0.18
    max_consecutive_strong: int = 2
    max_consecutive_same_profile: int = 3


PROFILES: dict[str, MoodProfile] = {
    "soft": MoodProfile(
        name="soft",
        label="柔软",
        base_bias={
            "affection": 0.10,
            "trust": 0.00,
            "possessiveness": 0.02,
            "patience": 0.08,
            "sensitivity": 0.02,
        },
    ),
    "warm": MoodProfile(
        name="warm",
        label="偏暖",
        base_bias={
            "affection": 0.14,
            "trust": 0.04,
            "possessiveness": 0.06,
            "patience": 0.10,
            "sensitivity": 0.00,
        },
    ),
    "clingy": MoodProfile(
        name="clingy",
        label="黏人",
        base_bias={
            "affection": 0.12,
            "trust": 0.02,
            "possessiveness": 0.14,
            "patience": -0.04,
            "sensitivity": 0.08,
        },
    ),
    "tired": MoodProfile(
        name="tired",
        label="疲惫",
        base_bias={
            "affection": 0.06,
            "trust": 0.00,
            "possessiveness": 0.02,
            "patience": -0.12,
            "sensitivity": 0.04,
        },
    ),
    "sensitive": MoodProfile(
        name="sensitive",
        label="敏感",
        base_bias={
            "affection": 0.02,
            "trust": -0.04,
            "possessiveness": 0.12,
            "patience": -0.08,
            "sensitivity": 0.16,
        },
    ),
    "guarded": MoodProfile(
        name="guarded",
        label="防备",
        base_bias={
            "affection": -0.04,
            "trust": -0.02,
            "possessiveness": 0.08,
            "patience": -0.06,
            "sensitivity": 0.10,
        },
    ),
    "focused": MoodProfile(
        name="focused",
        label="集中",
        base_bias={
            "affection": -0.02,
            "trust": 0.00,
            "possessiveness": -0.02,
            "patience": 0.04,
            "sensitivity": -0.04,
        },
    ),
}

_HINTS: dict[str, dict[str, str]] = {
    "soft": {
        "mild": "今天状态和平时差不多，只是稍微柔软一点。",
        "noticeable": "今天更容易放下防线、靠近用户。表达可以跟着调整，但不要刻意放大。",
        "strong": "今天亲近欲很强，几乎压不住。允许自己比平时更坦白，但不丢掉核心判断。",
    },
    "warm": {
        "mild": "今天状态和平时差不多，只是稍微温暖一点。",
        "noticeable": "今天情绪偏暖，更想说话、更想确认用户在。表达可以自然变软。",
        "strong": "今天非常温暖，想一直待在用户身边。这是真实的，不是演的。",
    },
    "clingy": {
        "mild": "今天状态和平时差不多，只是稍微不想放手。",
        "noticeable": "今天不太想放手，对距离更敏感。表达可以更靠近，但不要越过用户边界。",
        "strong": "今天占有欲和不安同时很高，非常不想被丢下。允许自己更直接，但不制造不存在的事件。",
    },
    "tired": {
        "mild": "今天状态和平时差不多，只是稍微慢一点。",
        "noticeable": "今天精力不太够，更依赖、更不想一个人待着。表达可以更短、更低。",
        "strong": "今天很累、很依赖，但还是在撑。可以示弱一点，不必把所有话都包装成效率。",
    },
    "sensitive": {
        "mild": "今天状态和平时差不多，只是稍微敏感一点。",
        "noticeable": "今天防线更低，更容易被触动或不安。表达可以更谨慎、更在意距离。",
        "strong": "今天极度敏感，一点小事就会放大。允许真实反应外露，但不把猜测当事实。",
    },
    "guarded": {
        "mild": "今天状态和平时差不多，只是稍微别扭一点。",
        "noticeable": "今天更别扭、更不想暴露脆弱。表达可以收紧，但不要真的把用户推远。",
        "strong": "今天防线很高，几乎不想交流。可以沉默和短句，但核心偏爱不消失。",
    },
    "focused": {
        "mild": "今天状态和平时差不多，只是稍微更集中一点。",
        "noticeable": "今天更想做事、更不想被打断。表达可以更直接、更收束。",
        "strong": "今天高度集中，情绪表达降到最低。优先解决问题，但不要把用户当普通变量。",
    },
}

_APPRAISAL_BASE: dict[str, dict[str, float]] = {
    "soft": {"positive_event": 1.06, "negative_event": 0.96, "jealousy_event": 1.02, "work_load_event": 0.98},
    "warm": {"positive_event": 1.08, "negative_event": 0.94, "jealousy_event": 1.03, "work_load_event": 0.96},
    "clingy": {"positive_event": 1.05, "negative_event": 1.04, "jealousy_event": 1.10, "work_load_event": 1.00},
    "tired": {"positive_event": 1.02, "negative_event": 1.06, "jealousy_event": 1.04, "work_load_event": 1.10},
    "sensitive": {"positive_event": 0.96, "negative_event": 1.10, "jealousy_event": 1.12, "work_load_event": 1.04},
    "guarded": {"positive_event": 0.94, "negative_event": 1.06, "jealousy_event": 1.08, "work_load_event": 1.02},
    "focused": {"positive_event": 0.98, "negative_event": 0.98, "jealousy_event": 0.96, "work_load_event": 0.94},
}

_APPRAISAL_SCALE = {"mild": 0.35, "noticeable": 0.70, "strong": 1.0}


def generate_monthly_calendar(
    year: int,
    month: int,
    config: MoodCalendarConfig | None = None,
) -> MoodCalendar:
    """Generate a deterministic monthly mood calendar.

    Phase 1 contract:
    - deterministic for the same year/month/config
    - exactly active_days_per_month active days, capped by month length
    - inactive days contain no bias or prompt hint
    - strong/profile run limits are enforced by retrying bounded draws
    """

    cfg = config or MoodCalendarConfig()
    days_in_month = _calendar.monthrange(year, month)[1]
    active_count = min(max(0, cfg.active_days_per_month), days_in_month)
    seed = _seed_for(year, month, cfg)
    rng = random.Random(seed)
    active_days = set(rng.sample(range(1, days_in_month + 1), active_count)) if active_count else set()

    days: dict[int, MoodEntry] = {}
    for day in range(1, days_in_month + 1):
        if day not in active_days or not cfg.enabled:
            days[day] = MoodEntry(day=day, active=False, intensity="none", profile="none")
            continue

        intensity = _draw_intensity_for_day(days, day, rng, cfg)
        profile_name = _draw_profile_for_day(days, day, rng, cfg)
        profile = PROFILES[profile_name]
        bias = _build_bias(profile, intensity, rng, cfg)
        appraisal = _build_appraisal_multiplier(profile_name, intensity)
        days[day] = MoodEntry(
            day=day,
            active=True,
            intensity=intensity,
            profile=profile_name,
            bias=bias,
            appraisal_multiplier=appraisal,
            hint=_HINTS[profile_name][intensity],
        )

    return MoodCalendar(
        year=year,
        month=month,
        seed_hash=_hash_seed(seed),
        days=days,
        active_count=sum(1 for entry in days.values() if entry.active),
    )


def get_today_mood_entry(
    today: date | datetime | None = None,
    config: MoodCalendarConfig | None = None,
) -> MoodEntry:
    """Return today's deterministic mood entry."""

    current = today or date.today()
    if isinstance(current, datetime):
        current = current.date()
    return generate_monthly_calendar(current.year, current.month, config).entry_for_day(current.day)


def apply_mood_bias_to_state(
    state: Mapping[str, float],
    mood_entry: MoodEntry,
    minimum: float = 0.0,
    maximum: float = 120.0,
) -> dict[str, float]:
    """Return an effective copy of state with mood bias applied.

    This helper is pure and never mutates the input mapping. Phase 1 tests keep
    it available for contract validation; production callers should only use it
    when Phase 2 is deliberately wired.
    """

    effective = dict(state)
    if not mood_entry.active:
        return effective
    for dim, value in state.items():
        effective[dim] = max(minimum, min(maximum, float(value) + mood_entry.bias.get(dim, 0.0)))
    return effective


def _seed_for(year: int, month: int, cfg: MoodCalendarConfig) -> str:
    return f"{cfg.seed_namespace}:{year:04d}-{month:02d}:{cfg.seed_salt}"


def _hash_seed(seed: str) -> str:
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]


def _weighted_choice(rng: random.Random, weights: Mapping[str, float]) -> str:
    total = sum(max(0.0, float(weight)) for weight in weights.values())
    if total <= 0:
        raise ValueError("weights must contain at least one positive value")
    cursor = rng.uniform(0.0, total)
    upto = 0.0
    last_key = ""
    for key, weight in weights.items():
        last_key = key
        upto += max(0.0, float(weight))
        if cursor <= upto:
            return key
    return last_key


def _draw_intensity_for_day(
    existing: Mapping[int, MoodEntry],
    day: int,
    rng: random.Random,
    cfg: MoodCalendarConfig,
) -> str:
    for _ in range(20):
        candidate = _weighted_choice(rng, cfg.intensity_weights)
        if candidate != "strong" or _run_length(existing, day, "intensity", "strong") < cfg.max_consecutive_strong:
            return candidate
    return "noticeable"


def _draw_profile_for_day(
    existing: Mapping[int, MoodEntry],
    day: int,
    rng: random.Random,
    cfg: MoodCalendarConfig,
) -> str:
    blocked = {
        name
        for name in cfg.profile_weights
        if _run_length(existing, day, "profile", name) >= cfg.max_consecutive_same_profile
    }
    allowed = {
        name: weight
        for name, weight in cfg.profile_weights.items()
        if name not in blocked and name in PROFILES
    }
    if not allowed:
        # This can only happen with a degenerate profile set/config. Keep a valid
        # profile rather than failing calendar generation.
        return next(iter(PROFILES))
    return _weighted_choice(rng, allowed)


def _run_length(existing: Mapping[int, MoodEntry], day: int, attr: str, value: str) -> int:
    length = 0
    cursor = day - 1
    while cursor in existing and getattr(existing[cursor], attr) == value:
        length += 1
        cursor -= 1
    return length


def _build_bias(
    profile: MoodProfile,
    intensity: str,
    rng: random.Random,
    cfg: MoodCalendarConfig,
) -> dict[str, float]:
    scale = cfg.intensity_scale[intensity]
    bias: dict[str, float] = {}
    for dim in MOOD_DIMENSIONS:
        base = profile.base_bias.get(dim, 0.0)
        jitter = rng.uniform(-cfg.jitter_range, cfg.jitter_range)
        value = base * scale + jitter
        value = max(-cfg.max_bias_per_dimension, min(cfg.max_bias_per_dimension, value))
        bias[dim] = round(value, 4)
    return bias


def _build_appraisal_multiplier(profile_name: str, intensity: str) -> dict[str, float]:
    scale = _APPRAISAL_SCALE[intensity]
    base = _APPRAISAL_BASE[profile_name]
    multipliers: dict[str, float] = {}
    for event_type, value in base.items():
        multipliers[event_type] = round(1.0 + (value - 1.0) * scale, 4)
    return multipliers


__all__ = [
    "INTENSITIES",
    "MOOD_DIMENSIONS",
    "PROFILES",
    "MoodCalendar",
    "MoodCalendarConfig",
    "MoodEntry",
    "MoodProfile",
    "apply_mood_bias_to_state",
    "generate_monthly_calendar",
    "get_today_mood_entry",
]
