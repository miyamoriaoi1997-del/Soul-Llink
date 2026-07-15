"""Tests for the shadow-only monthly mood calendar."""

from __future__ import annotations

import calendar as py_calendar
from collections import Counter

from persona_engine.mood_calendar import (
    INTENSITIES,
    MOOD_DIMENSIONS,
    PROFILES,
    MoodCalendarConfig,
    MoodEntry,
    apply_mood_bias_to_state,
    generate_monthly_calendar,
    get_today_mood_entry,
)


def test_same_month_same_seed_is_deterministic():
    first = generate_monthly_calendar(2026, 5)
    second = generate_monthly_calendar(2026, 5)

    assert first == second
    assert first.seed_hash == second.seed_hash


def test_different_months_differ():
    may = generate_monthly_calendar(2026, 5)
    june = generate_monthly_calendar(2026, 6)

    assert may.seed_hash != june.seed_hash
    assert may.days != june.days


def test_active_days_count_is_capped_by_month_length():
    cfg = MoodCalendarConfig(active_days_per_month=20)
    feb = generate_monthly_calendar(2026, 2, cfg)

    assert feb.active_count == 20
    assert len(feb.days) == 28

    too_many = generate_monthly_calendar(2026, 2, MoodCalendarConfig(active_days_per_month=99))
    assert too_many.active_count == 28


def test_inactive_days_have_no_bias_multiplier_or_hint():
    calendar = generate_monthly_calendar(2026, 5, MoodCalendarConfig(active_days_per_month=1))
    inactive = [entry for entry in calendar.days.values() if not entry.active]

    assert inactive
    assert all(entry.profile == "none" for entry in inactive)
    assert all(entry.intensity == "none" for entry in inactive)
    assert all(entry.bias == {} for entry in inactive)
    assert all(entry.appraisal_multiplier == {} for entry in inactive)
    assert all(entry.hint == "" for entry in inactive)


def test_active_entries_have_valid_profile_intensity_bias_hint_and_appraisal():
    calendar = generate_monthly_calendar(2026, 5)
    active = [entry for entry in calendar.days.values() if entry.active]

    assert active
    for entry in active:
        assert entry.profile in PROFILES
        assert entry.intensity in INTENSITIES
        assert set(entry.bias) == set(MOOD_DIMENSIONS)
        assert entry.hint
        assert set(entry.appraisal_multiplier) == {
            "positive_event",
            "negative_event",
            "jealousy_event",
            "work_load_event",
        }
        assert all(0.88 <= value <= 1.12 for value in entry.appraisal_multiplier.values())


def test_bias_within_bounds():
    cfg = MoodCalendarConfig(max_bias_per_dimension=0.18)
    for month in range(1, 13):
        calendar = generate_monthly_calendar(2026, month, cfg)
        for entry in calendar.days.values():
            assert all(abs(value) <= cfg.max_bias_per_dimension for value in entry.bias.values())


def test_no_more_than_two_strong_days_in_a_row():
    cfg = MoodCalendarConfig(
        active_days_per_month=31,
        intensity_weights={"mild": 0.0, "noticeable": 0.1, "strong": 0.9},
        max_consecutive_strong=2,
    )
    calendar = generate_monthly_calendar(2026, 5, cfg)

    streak = 0
    for day in range(1, 32):
        if calendar.days[day].intensity == "strong":
            streak += 1
        else:
            streak = 0
        assert streak <= 2


def test_no_more_than_three_same_profiles_in_a_row():
    cfg = MoodCalendarConfig(
        active_days_per_month=31,
        profile_weights={
            "soft": 0.94,
            "warm": 0.01,
            "clingy": 0.01,
            "tired": 0.01,
            "sensitive": 0.01,
            "guarded": 0.01,
            "focused": 0.01,
        },
        max_consecutive_same_profile=3,
    )
    calendar = generate_monthly_calendar(2026, 5, cfg)

    streak = 0
    last = None
    for day in range(1, 32):
        profile = calendar.days[day].profile
        if profile == last:
            streak += 1
        else:
            streak = 1
            last = profile
        assert streak <= 3


def test_profile_distribution_covers_all_profiles_over_year():
    seen = set()
    for month in range(1, 13):
        calendar = generate_monthly_calendar(2026, month)
        seen.update(entry.profile for entry in calendar.days.values() if entry.active)

    assert seen == set(PROFILES)


def test_intensity_distribution_has_all_intensities_over_year():
    counter = Counter()
    for month in range(1, 13):
        calendar = generate_monthly_calendar(2026, month)
        counter.update(entry.intensity for entry in calendar.days.values() if entry.active)

    assert all(counter[intensity] > 0 for intensity in INTENSITIES)
    assert counter["mild"] > counter["strong"]


def test_all_days_present_for_month_lengths():
    for year, month in [(2026, 2), (2028, 2), (2026, 4), (2026, 5)]:
        calendar = generate_monthly_calendar(year, month)
        days_in_month = py_calendar.monthrange(year, month)[1]
        assert sorted(calendar.days) == list(range(1, days_in_month + 1))


def test_get_today_mood_entry_uses_date_argument():
    entry = get_today_mood_entry(__import__("datetime").date(2026, 5, 16))
    expected = generate_monthly_calendar(2026, 5).entry_for_day(16)

    assert entry == expected


def test_apply_mood_bias_returns_effective_copy_without_mutating_real_state():
    real_state = {"affection": 72, "trust": 66, "possessiveness": 70, "patience": 58}
    mood = MoodEntry(
        day=1,
        active=True,
        intensity="noticeable",
        profile="warm",
        bias={"affection": 0.12, "trust": 0.04, "possessiveness": 0.04, "patience": 0.08},
    )

    effective = apply_mood_bias_to_state(real_state, mood)

    assert real_state == {"affection": 72, "trust": 66, "possessiveness": 70, "patience": 58}
    assert effective != real_state
    assert effective["affection"] == 72.12
    assert effective["patience"] == 58.08


def test_apply_mood_bias_inactive_day_is_noop_copy():
    real_state = {"affection": 72, "trust": 66}
    effective = apply_mood_bias_to_state(
        real_state,
        MoodEntry(day=3, active=False, intensity="none", profile="none"),
    )

    assert effective == real_state
    assert effective is not real_state
