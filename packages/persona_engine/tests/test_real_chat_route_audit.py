from scripts.real_chat_route_audit import classify_match, expected_from_text, sha16, build_indexes


def test_repeated_session_hash_is_time_resolved_to_nearest_message():
    messages = [
        {"id": 1, "session_id": "s1", "content": "好继续", "timestamp": 100.0, "hash": sha16("好继续")},
        {"id": 2, "session_id": "s1", "content": "好继续", "timestamp": 200.0, "hash": sha16("好继续")},
        {"id": 3, "session_id": "s1", "content": "好继续", "timestamp": 300.0, "hash": sha16("好继续")},
    ]
    by_session_hash, by_hash = build_indexes(messages)

    match_type, candidates, matched, delta = classify_match(
        {"session_id": "s1", "user_message_hash": sha16("好继续"), "message_timestamp": 215.0},
        by_session_hash,
        by_hash,
    )

    assert match_type == "session_hash_ambiguous_time_resolved"
    assert len(candidates) == 3
    assert matched["id"] == 2
    assert delta == 15.0


def test_sex_label_respects_restrained_desire_gate():
    assert expected_from_text("我们做爱吧", "daily", ["sex_requires_gate", "sex_desire_gate_restrained"]) == (
        "daily",
        "relationship",
        "explicit sex wording blocked by restrained desire gate",
    )


def test_sex_label_without_restrained_gate_expects_sex_route():
    assert expected_from_text("我们做爱吧", "daily", ["sex_requires_gate"]) == (
        "sex",
        "sex",
        "explicit sex wording with no restrained gate flag",
    )


def test_short_continuation_inherits_task_context():
    assert expected_from_text("好继续", "work", []) == (
        "work",
        "task",
        "short continuation inherits active work mode",
    )
