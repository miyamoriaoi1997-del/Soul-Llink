from pcltm.memfs_types import ContextSelectionSnapshot, MemoryLayerItem, MemoryLayerView, PromptMemoryView


def test_context_summary_exposes_selection_audit_without_rendering_metadata():
    view = PromptMemoryView(
        system=MemoryLayerView(layer="system", budget_chars=100, used_chars=20),
        pinned=MemoryLayerView(
            layer="pinned",
            budget_chars=100,
            used_chars=42,
            items=[
                MemoryLayerItem(
                    path="pinned/user_pref.md",
                    description="Teacher preference",
                    body="用户偏好先看清楚再动手。",
                    authority="pinned",
                    buckets=("user_preferences",),
                    mode_scope=("work",),
                    char_count=42,
                    char_limit=100,
                    score=0.87,
                    memory_type="UserPreference",
                    lifecycle_state="active",
                    injection_policy="mode_aware",
                )
            ],
        ),
        episodic=MemoryLayerView(layer="episodic", budget_chars=100, used_chars=0, omitted_count=3),
        transient=MemoryLayerView(layer="transient", budget_chars=100, used_chars=0),
        selected_layers=("system", "pinned", "episodic", "transient"),
        selected_buckets=("user_preferences",),
        selection_source="memfs",
    )

    summary = view.context_summary()
    audit = summary["selection_audit"]

    assert audit["schema_version"] == 1
    assert audit["selection_source"] == "memfs"
    assert audit["total_selected_items"] == 1
    assert audit["total_omitted_items"] == 3
    assert "transient_layer_empty" in audit["warnings"]

    pinned_audit = next(layer for layer in audit["layers"] if layer["layer"] == "pinned")
    assert pinned_audit["selected_items"] == [
        {
            "id": "pinned/user_pref.md",
            "path": "pinned/user_pref.md",
            "description": "Teacher preference",
            "memory_type": "UserPreference",
            "lifecycle_state": "active",
            "authority": "pinned",
            "buckets": ["user_preferences"],
            "mode_scope": ["work"],
            "score": 0.87,
            "char_count": 42,
            "char_limit": 100,
            "injection_policy": "mode_aware",
            "ttl": "none",
            "read_only": False,
        }
    ]

    episodic_audit = next(layer for layer in audit["layers"] if layer["layer"] == "episodic")
    assert episodic_audit["omitted_reasons"] == [
        {"reason": "budget_exceeded_or_filter_not_selected", "count": 3}
    ]

    rendered = view.render_active_frame()
    assert "用户偏好先看清楚再动手。" in rendered
    assert "selection_audit" not in rendered
    assert "pinned/user_pref.md" not in rendered
    assert "0.87" not in rendered


def test_context_summary_omits_transient_empty_warning_when_transient_has_item():
    view = PromptMemoryView(
        transient=MemoryLayerView(
            layer="transient",
            budget_chars=100,
            used_chars=20,
            items=[
                MemoryLayerItem(
                    path="transient/current_task.md",
                    body="当前任务：补 PCLTM selection audit。",
                    authority="transient",
                    buckets=("current_task",),
                    mode_scope=("work",),
                    char_count=20,
                    memory_type="TemporaryTaskState",
                    lifecycle_state="active",
                )
            ],
        )
    )

    assert view.context_summary()["selection_audit"]["warnings"] == []


def test_context_selection_snapshot_is_host_neutral_serializable_contract():
    view = PromptMemoryView(
        pinned=MemoryLayerView(
            layer="pinned",
            budget_chars=100,
            used_chars=42,
            items=[
                MemoryLayerItem(
                    path="pinned/runtime.md",
                    description="Runtime boundary",
                    body="runtime boundary stays active",
                    authority="pinned",
                    buckets=("runtime_boundary",),
                    mode_scope=("work",),
                    char_count=42,
                    score=0.5,
                    memory_type="RuntimeInvariant",
                    lifecycle_state="active",
                    injection_policy="always",
                    read_only=True,
                )
            ],
        ),
        episodic=MemoryLayerView(layer="episodic", budget_chars=100, omitted_count=2),
        selected_layers=("system", "pinned"),
        selected_buckets=("runtime_boundary",),
        selection_source="memfs",
    )

    snapshot = view.context_selection_snapshot()
    payload = snapshot.to_dict()

    assert isinstance(snapshot, ContextSelectionSnapshot)
    assert payload["object_type"] == "pcltm_context_selection_snapshot"
    assert payload["schema_version"] == 1
    assert payload["selection_source"] == "memfs"
    assert payload["active_layers"] == ["system", "pinned"]
    assert payload["selected_buckets"] == ["runtime_boundary"]
    assert "compression" in payload["reference_only_layers"]
    assert payload["total_selected_items"] == 1
    assert payload["total_omitted_items"] == 2
    assert payload["layers"][1]["selected_items"][0]["memory_type"] == "RuntimeInvariant"
    assert "active_text" not in payload
    assert "Hermes" not in payload
