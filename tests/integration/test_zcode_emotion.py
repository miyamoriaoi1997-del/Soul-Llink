from __future__ import annotations

import json
from pathlib import Path

import pytest

from soul_link.zcode_emotion import EmotionBridge


@pytest.fixture(autouse=True)
def _isolated_zcode_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZCODE_ROOT", str(tmp_path / "zcode"))
    monkeypatch.setenv("HERMES_PCLTM_MEMFS_ROOT", str(tmp_path / "memfs"))
    monkeypatch.setenv("HERMES_PCLTM_DB", str(tmp_path / "pcltm.db"))


def test_emotion_bridge_updates_state_and_writes_files(tmp_path: Path) -> None:
    bridge = EmotionBridge()
    state = bridge.update("你太厉害了，我爱你！")
    assert state.get("affection") is not None
    assert (tmp_path / "zcode" / "soullink" / "STATE.md").is_file()
    assert (tmp_path / "zcode" / "soullink" / "emotion-state.json").is_file()


def test_emotion_bridge_tone_modifier_has_levels(tmp_path: Path) -> None:
    bridge = EmotionBridge()
    tone = bridge.tone_modifier()
    assert "【强度】" in tone or "mild" in tone or "moderate" in tone


def test_emotion_bridge_initial_state_is_neutral(tmp_path: Path) -> None:
    bridge = EmotionBridge()
    state = bridge.emotion_state()
    assert state.get("affection") == 60
    assert state.get("trust") == 60
    assert abs(state.get("emotion_score") or 0) < 5


def test_emotion_bridge_continuation_request_is_empty_when_neutral(tmp_path: Path) -> None:
    bridge = EmotionBridge()
    assert bridge.continuation_request() == {}


def test_emotion_bridge_continuation_requests_after_strong_turns(tmp_path: Path) -> None:
    bridge = EmotionBridge()
    for _ in range(8):
        bridge.update("你太厉害了，我爱你，我永远都信任你！")
    request = bridge.continuation_request()
    if request:
        assert request.get("continue") is True


def test_emotion_bridge_continues_after_stop_bounded(tmp_path: Path) -> None:
    bridge = EmotionBridge()
    for _ in range(8):
        bridge.update("你太厉害了，我爱你，我永远都信任你！")
    assert bridge.continuation_request().get("continue") is True
    # A second Stop call in the same turn is still allowed by the bridge
    # (the hook layer enforces the 3-continuation bound); here we only
    # verify the bridge keeps reporting the strong emotion.
    assert bridge.continuation_request().get("continue") is True


def test_emotion_bridge_isolation_between_roots(tmp_path: Path) -> None:
    other = tmp_path / "other"
    other.mkdir()
    a = EmotionBridge(tmp_path / "zcode")
    b = EmotionBridge(other)
    a.update("你太厉害了")
    assert a.emotion_state().get("affection", 60) != 60 or a.emotion_state().get("emotion_score", 0) != 0
    assert b.emotion_state().get("affection") == 60


def test_emotion_disabled_writes_nothing(tmp_path: Path) -> None:
    """With emotion_enabled=false in adapter.json the bridge is inert: no
    state, no files, no continuation — matching a deployment that stripped
    the emotion layer."""
    soullink = tmp_path / "zcode" / "soullink"
    soullink.mkdir(parents=True)
    (soullink / "adapter.json").write_text(
        json.dumps({"emotion_enabled": False}), encoding="utf-8"
    )
    bridge = EmotionBridge(tmp_path / "zcode")
    assert bridge.update("你太厉害了，我爱你！") == {}
    assert bridge.emotion_state() == {}
    assert bridge.tone_modifier() == ""
    assert bridge.continuation_request() == {}
    assert not (soullink / "STATE.md").exists()
    assert not (soullink / "emotion-state.json").exists()


def test_emotion_enabled_default_when_adapter_absent(tmp_path: Path) -> None:
    """No adapter.json means enabled (the PR default keeps emotion on)."""
    bridge = EmotionBridge(tmp_path / "zcode")
    state = bridge.emotion_state()
    assert state.get("affection") == 60
