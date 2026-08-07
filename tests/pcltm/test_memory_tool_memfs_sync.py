from __future__ import annotations

import inspect

import pytest

from pcltm import memory_adapter
from pcltm.memory_transition_service import MemoryTransitionService
from pcltm.memory_write_service import MemoryWriteService


def test_legacy_memory_tool_sync_is_retired_without_fallback_signal() -> None:
    with pytest.raises(RuntimeError, match="legacy_memory_tool_sync_retired"):
        memory_adapter.sync_memory_tool_write(
            "memory", "add", content="must not enter legacy memory_records",
        )


def test_retired_sync_implementation_is_not_exported_from_package_root() -> None:
    import pcltm

    assert not hasattr(pcltm, "sync_memory_tool_write")
    assert not hasattr(pcltm, "_retired_sync_memory_tool_write_implementation")


def test_governed_replacement_services_do_not_use_like_or_transactional_memfs() -> None:
    write_source = inspect.getsource(MemoryWriteService.write)
    transition_source = inspect.getsource(MemoryTransitionService)
    assert "memory_records" not in write_source
    assert "LIKE" not in transition_source
    assert "_materialize_memfs_record" not in write_source
    assert "_materialize_memfs_record" not in transition_source
