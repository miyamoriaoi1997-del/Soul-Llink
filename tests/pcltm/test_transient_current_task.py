from pcltm.memory_adapter import load_layered_prompt_context, write_current_task_state


def test_write_current_task_state_feeds_transient_layer(tmp_path):
    result = write_current_task_state(
        title="PCLTM transient rollout",
        body="当前任务：把 current task 写入 transient 层。",
        mode="work",
        task_id="task-123",
        root=tmp_path,
    )

    assert result == {
        "ok": True,
        "memory_id": "transient/current-task.md",
        "layer": "transient",
        "body_chars": len("当前任务：把 current task 写入 transient 层。"),
        "overwrite_policy": "replace_current_task",
    }

    view = load_layered_prompt_context(
        mode="work",
        query="current task",
        budgets={"system": 100, "pinned": 100, "episodic": 100, "transient": 200},
        buckets=["current_task"],
        root=tmp_path,
    )

    assert view.transient.items[0].path == "transient/current-task.md"
    assert view.transient.items[0].body.strip() == "当前任务：把 current task 写入 transient 层。"
    assert view.transient.items[0].memory_type == "TemporaryTaskState"
    assert view.transient.items[0].ttl == "short"
    assert view.transient.items[0].injection_policy == "transient_only"
    assert view.context_summary()["selection_audit"]["warnings"] == []


def test_write_current_task_state_overwrites_stable_slot(tmp_path):
    first = write_current_task_state(
        title="old task",
        body="旧任务。",
        mode="work",
        task_id="old",
        root=tmp_path,
    )
    second = write_current_task_state(
        title="new task",
        body="新任务。",
        mode="work",
        task_id="new",
        root=tmp_path,
    )

    assert first["memory_id"] == second["memory_id"] == "transient/current-task.md"
    files = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*.md"))
    assert files == ["transient/current-task.md"]

    view = load_layered_prompt_context(
        mode="work",
        query="task",
        budgets={"transient": 200},
        buckets=["current_task"],
        root=tmp_path,
    )
    assert len(view.transient.items) == 1
    assert view.transient.items[0].description == "new task"
    assert view.transient.items[0].body.strip() == "新任务。"


def test_write_current_task_state_rejects_empty_body(tmp_path):
    assert write_current_task_state(title="empty", body="   ", root=tmp_path) == {
        "ok": False,
        "error": "current_task_body_required",
    }
    assert not list(tmp_path.rglob("*.md"))
