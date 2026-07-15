from __future__ import annotations

from pcltm import memory_adapter


def test_write_evidence_capsule_is_transient_reference_only_and_searchable(tmp_path, monkeypatch):
    root = tmp_path / "memfs"
    monkeypatch.setenv("HERMES_PCLTM_MEMFS_ROOT", str(root))
    monkeypatch.setattr(memory_adapter, "MEMFS_ROOT", root)

    result = memory_adapter.write_evidence_capsule(
        title="pytest failure evidence",
        body="command=pytest\nexit_code=1\nimportant traceback excerpt",
        mode="work",
        buckets=["tool_evidence", "current_task"],
        source_tool="terminal",
        evidence_id="pytest-failure-1",
        root=root,
    )

    assert result["ok"] is True
    assert result["layer"] == "transient"
    assert result["reference_only"] is True

    view = memory_adapter.load_layered_prompt_context(
        mode="work",
        query="pytest failure traceback",
        layers=["system", "pinned", "episodic", "transient"],
        active_layers=["system", "pinned"],
        buckets=["tool_evidence", "current_task"],
        budgets={"system": 500, "pinned": 500, "episodic": 500, "transient": 1000},
    )
    active = view.active_frame().active_text
    assert "important traceback excerpt" not in active
    summary = view.context_summary()
    assert "transient" in summary["reference_only_layers"]

    hits = memory_adapter.search_archival_memories(
        "pytest traceback",
        mode="work",
        layers=["transient"],
        buckets=["tool_evidence"],
    )
    assert isinstance(hits, list)
    assert len(hits) == 1
    hit = hits[0]
    assert hit["memory_id"] == result["memory_id"]
    assert hit["reference_only"] is True

    opened = memory_adapter.open_archival_memory(result["memory_id"], body_limit=500)
    assert "body" in opened
    assert opened["body"] == "command=pytest\nexit_code=1\nimportant traceback excerpt"
    assert opened["reference_only"] is True


def test_write_evidence_capsule_truncates_body(tmp_path, monkeypatch):
    root = tmp_path / "memfs"
    monkeypatch.setattr(memory_adapter, "MEMFS_ROOT", root)
    result = memory_adapter.write_evidence_capsule(
        title="large evidence",
        body="X" * 5000,
        source_tool="terminal",
        root=root,
    )
    opened = memory_adapter.open_archival_memory(result["memory_id"], body_limit=3000)
    assert "body" in opened
    assert len(opened["body"]) <= 1800


def test_write_evidence_capsule_redacts_secret_values(tmp_path, monkeypatch):
    root = tmp_path / "memfs"
    monkeypatch.setattr(memory_adapter, "MEMFS_ROOT", root)
    fake_secret = "PASSWORD=hunter2"

    result = memory_adapter.write_evidence_capsule(
        title="dirty terminal evidence",
        body=f"command output leaked {fake_secret}",
        source_tool="terminal",
        root=root,
    )

    opened = memory_adapter.open_archival_memory(result["memory_id"], body_limit=500)
    assert fake_secret not in str(opened)
    assert "[REDACTED_SECRET]" in str(opened)
