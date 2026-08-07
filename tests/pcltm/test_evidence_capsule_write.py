from __future__ import annotations

import pytest

from pcltm import memory_adapter


def test_write_evidence_capsule_persists_reference_only_but_legacy_consumers_are_retired(
    tmp_path, monkeypatch,
):
    root = tmp_path / "memfs"
    monkeypatch.setenv("HERMES_PCLTM_MEMFS_ROOT", str(root))
    monkeypatch.setattr(memory_adapter, "MEMFS_ROOT", root)
    body = "command=pytest\nexit_code=1\nimportant traceback excerpt"

    result = memory_adapter.write_evidence_capsule(
        title="pytest failure evidence", body=body, mode="work",
        buckets=["tool_evidence", "current_task"], source_tool="terminal",
        evidence_id="pytest-failure-1", root=root,
    )

    assert result["ok"] is True
    assert result["layer"] == "transient"
    assert result["reference_only"] is True
    artifact = root / result["memory_id"]
    assert artifact.is_file()
    assert body in artifact.read_text(encoding="utf-8")
    assert memory_adapter.load_layered_prompt_context(mode="work", root=root).render() == ""
    assert memory_adapter.search_archival_memories("pytest traceback") == []
    with pytest.raises(ValueError, match="legacy_memfs_archival_open_retired"):
        memory_adapter.open_archival_memory(result["memory_id"])


def test_write_evidence_capsule_truncates_persisted_body(tmp_path, monkeypatch):
    root = tmp_path / "memfs"
    monkeypatch.setattr(memory_adapter, "MEMFS_ROOT", root)
    result = memory_adapter.write_evidence_capsule(
        title="large evidence", body="X" * 5000, source_tool="terminal", root=root,
    )
    rendered = (root / result["memory_id"]).read_text(encoding="utf-8")
    assert len(rendered) < 3000


def test_write_evidence_capsule_redacts_secret_values_at_write_boundary(tmp_path, monkeypatch):
    root = tmp_path / "memfs"
    monkeypatch.setattr(memory_adapter, "MEMFS_ROOT", root)
    fake_secret = "PASSWORD=hunter2"

    result = memory_adapter.write_evidence_capsule(
        title="dirty terminal evidence", body=f"command output leaked {fake_secret}",
        source_tool="terminal", root=root,
    )
    rendered = (root / result["memory_id"]).read_text(encoding="utf-8")

    assert fake_secret not in rendered
    assert "[REDACTED_SECRET]" in rendered
