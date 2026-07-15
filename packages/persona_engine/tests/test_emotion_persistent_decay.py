"""Regression tests for persistent emotion decay in Soul-Link runtime paths."""

from __future__ import annotations

from datetime import datetime, timedelta

import yaml

from emotion_state_manager import EmotionStateManager


class FrozenDateTime(datetime):
    _now = datetime(2026, 5, 29, 9, 30, 0)

    @classmethod
    def now(cls, tz=None):
        if tz is not None:
            return cls._now.replace(tzinfo=tz)
        return cls._now

    @classmethod
    def fromisoformat(cls, value: str):
        return datetime.fromisoformat(value)


def _write_state(path, *, last_update, emotion_score=4.4, current_emotion=4.4):
    frontmatter = {
        "emotion_state": {
            "affection": 112,
            "trust": 109,
            "possessiveness": 103.74,
            "patience": 82,
            "emotion_score": emotion_score,
            "current_emotion": current_emotion,
            "previous_emotion_score": 4.1,
            "last_trigger_type": "needed",
            "last_raw_trigger_type": "care",
            "last_update": last_update.isoformat(),
            "inertia": {"consecutive_same": 4, "last_direction": 1, "history": [1, 1, 1, 1, 1]},
        }
    }
    path.write_text(
        "---\n" + yaml.safe_dump(frontmatter, allow_unicode=True, sort_keys=False) + "---\n\nbody\n",
        encoding="utf-8",
    )


def _read_frontmatter(path):
    content = path.read_text(encoding="utf-8")
    return yaml.safe_load(content.split("---\n", 2)[1])


def test_apply_time_decay_persists_current_emotion_after_long_idle(monkeypatch, tmp_path):
    state_path = tmp_path / "STATE.md"
    _write_state(state_path, last_update=FrozenDateTime._now - timedelta(hours=6))
    monkeypatch.setattr("emotion_state_manager.datetime", FrozenDateTime)

    manager = EmotionStateManager(hermes_home=tmp_path, state_path=state_path, update_body=False)

    assert manager.apply_time_decay_if_needed() is True

    emotion_state = _read_frontmatter(state_path)["emotion_state"]
    assert emotion_state["affection"] == 60.0
    assert emotion_state["trust"] == 60.0
    assert emotion_state["possessiveness"] < 104
    assert emotion_state["patience"] < 82
    assert emotion_state["emotion_score"] < 4.4
    assert emotion_state["current_emotion"] == emotion_state["emotion_score"]
    assert emotion_state["previous_emotion_score"] == 4.4
    assert emotion_state["last_trigger_type"] == "decay"
    assert emotion_state["last_raw_trigger_type"] == "decay"
    assert emotion_state["last_update"] == FrozenDateTime._now.isoformat()


def test_apply_time_decay_updates_markdown_body_from_decayed_state(monkeypatch, tmp_path):
    state_path = tmp_path / "STATE.md"
    _write_state(state_path, last_update=FrozenDateTime._now - timedelta(hours=6))
    state_path.write_text(
        state_path.read_text(encoding="utf-8").replace(
            "body",
            "## 当前情绪状态\n\n好感度: 999/120 (stale)\n情绪分值: +9.99 / 5.00",
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("emotion_state_manager.datetime", FrozenDateTime)

    manager = EmotionStateManager(hermes_home=tmp_path, state_path=state_path, update_body=True)

    assert manager.apply_time_decay_if_needed() is True

    content = state_path.read_text(encoding="utf-8")
    emotion_state = _read_frontmatter(state_path)["emotion_state"]
    assert "999/120" not in content
    assert "+9.99 / 5.00" not in content
    assert f"好感度: {emotion_state['affection']}/120" in content
    assert f"信任度: {emotion_state['trust']}/120" in content
    assert f"占有欲: {emotion_state['possessiveness']}/120" in content
    assert f"耐心值: {emotion_state['patience']}/120" in content
    assert f"情绪分值: {emotion_state['emotion_score']:+.2f} / 5.00" in content
    assert "最近触发: (衰减更新)" in content


def test_apply_time_decay_repairs_stale_decay_body_before_threshold(monkeypatch, tmp_path):
    state_path = tmp_path / "STATE.md"
    recent_update = FrozenDateTime._now - timedelta(minutes=20)
    _write_state(state_path, last_update=recent_update, emotion_score=1.09, current_emotion=1.09)
    state_path.write_text(
        state_path.read_text(encoding="utf-8").replace(
            "body",
            "## 当前情绪状态\n\n好感度: 999/120 (stale)\n情绪分值: +9.99 / 5.00",
        ),
        encoding="utf-8",
    )
    data = _read_frontmatter(state_path)
    data["emotion_state"]["last_trigger_type"] = "decay"
    data["emotion_state"]["last_raw_trigger_type"] = "decay"
    body = state_path.read_text(encoding="utf-8").split("---\n", 2)[2]
    state_path.write_text(
        "---\n" + yaml.safe_dump(data, allow_unicode=True, sort_keys=False) + "---\n" + body,
        encoding="utf-8",
    )
    monkeypatch.setattr("emotion_state_manager.datetime", FrozenDateTime)

    manager = EmotionStateManager(hermes_home=tmp_path, state_path=state_path, update_body=True)

    assert manager.apply_time_decay_if_needed() is True
    content = state_path.read_text(encoding="utf-8")
    emotion_state = _read_frontmatter(state_path)["emotion_state"]
    assert "999/120" not in content
    assert "+9.99 / 5.00" not in content
    assert f"好感度: {emotion_state['affection']}/120" in content
    assert f"情绪分值: {emotion_state['emotion_score']:+.2f} / 5.00" in content
    assert "最近触发: (衰减更新)" in content


def test_apply_time_decay_skips_persist_before_threshold(monkeypatch, tmp_path):
    state_path = tmp_path / "STATE.md"
    recent_update = FrozenDateTime._now - timedelta(minutes=20)
    _write_state(state_path, last_update=recent_update)
    before = state_path.read_text(encoding="utf-8")
    monkeypatch.setattr("emotion_state_manager.datetime", FrozenDateTime)

    manager = EmotionStateManager(hermes_home=tmp_path, state_path=state_path, update_body=False)

    assert manager.apply_time_decay_if_needed() is False
    assert state_path.read_text(encoding="utf-8") == before
