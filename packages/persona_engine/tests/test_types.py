from persona_orchestrator import MODE_DAILY, StatePacket


def test_state_packet_importable():
    packet = StatePacket(
        mode=MODE_DAILY,
        submode="",
        confidence=1.0,
        reason="test",
        transition="start:daily",
        selected_layers=["core", "daily"],
        memory_profile="core_relationship",
        safety_flags=[],
        emotion_score=None,
        desire_tier="unknown",
    )

    assert packet.mode == "daily"
    assert packet.shadow_only is True
    assert packet.selected_layers == ["core", "daily"]


def test_emotion_snapshot_contract_is_read_only_and_importable():
    from persona_orchestrator import EmotionSnapshot

    snapshot = EmotionSnapshot(
        affection=61.0,
        trust=60.0,
        possessiveness=63.0,
        patience=59.0,
        emotion_score=3.25,
        current_emotion=3.25,
        desire_tier="ambivalent",
        source="STATE.md",
    )

    assert snapshot.emotion_score == 3.25
    assert snapshot.desire_tier == "ambivalent"
    assert snapshot.source == "STATE.md"

    try:
        snapshot.emotion_score = 1.0
    except Exception as exc:
        assert exc.__class__.__name__ == "FrozenInstanceError"
    else:
        raise AssertionError("EmotionSnapshot must be immutable/read-only")
