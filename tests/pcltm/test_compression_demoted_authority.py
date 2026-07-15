"""Contract tests for retired compression authority in PromptMemoryView.

The user no longer wants context compression as an active memory source.
Compression compatibility fields may exist for old objects, but active memory
rendering and summaries are Stateful state layers only:
system -> pinned -> episodic -> transient.
"""

from __future__ import annotations

from pcltm.memfs_types import (
    MemoryAuthority,
    MemoryLayerItem,
    MemoryLayerView,
    PromptMemoryView,
)


def _item(layer: MemoryAuthority, body: str) -> MemoryLayerItem:
    return MemoryLayerItem(
        path=f"{layer.value}/contract.md",
        body=body,
        authority=layer.value,
        char_count=len(body),
    )


def test_compression_slot_remains_reference_only_for_compatibility():
    view = PromptMemoryView(
        compression=MemoryLayerView(
            layer=MemoryAuthority.COMPRESSION.value,
            items=[_item(MemoryAuthority.COMPRESSION, "old compressed context")],
            is_reference_only=True,
        )
    )

    assert view.compression.is_reference_only is True
    assert view.compression.layer == "compression"


def test_compression_not_rendered_even_when_present():
    view = PromptMemoryView(
        system=MemoryLayerView(
            layer=MemoryAuthority.SYSTEM.value,
            items=[_item(MemoryAuthority.SYSTEM, "system authority")],
        ),
        pinned=MemoryLayerView(
            layer=MemoryAuthority.PINNED.value,
            items=[_item(MemoryAuthority.PINNED, "pinned authority")],
        ),
        compression=MemoryLayerView(
            layer=MemoryAuthority.COMPRESSION.value,
            items=[_item(MemoryAuthority.COMPRESSION, "compressed reference")],
            is_reference_only=True,
        ),
    )

    rendered = view.render()

    assert rendered.index("[system]") < rendered.index("[pinned]")
    assert "[compression]" not in rendered
    assert "compressed reference" not in rendered


def test_compression_cannot_create_active_work():
    view = PromptMemoryView(
        compression=MemoryLayerView(
            layer=MemoryAuthority.COMPRESSION.value,
            items=[
                _item(
                    MemoryAuthority.COMPRESSION,
                    "Current Active Task: fix the router",
                )
            ],
            is_reference_only=True,
        )
    )

    rendered = view.render()

    assert rendered == ""
    assert "Current Active Task: fix the router" not in rendered


def test_compression_not_present_in_summary_layers():
    view = PromptMemoryView(
        compression=MemoryLayerView(
            layer=MemoryAuthority.COMPRESSION.value,
            items=[_item(MemoryAuthority.COMPRESSION, "Old task: rewrite the parser")],
            is_reference_only=True,
        )
    )

    summary = view.summary()

    assert [layer["layer"] for layer in summary["layers"]] == [
        "system",
        "pinned",
        "episodic",
        "transient",
    ]
    assert all(layer["layer"] != "compression" for layer in summary["layers"])


def test_compression_empty_view_safe():
    view = PromptMemoryView()

    rendered = view.render()

    assert rendered == ""


def test_active_authority_layers_remain_ordered_without_compression():
    view = PromptMemoryView(
        system=MemoryLayerView(
            layer=MemoryAuthority.SYSTEM.value,
            items=[_item(MemoryAuthority.SYSTEM, "system instruction")],
        ),
        pinned=MemoryLayerView(
            layer=MemoryAuthority.PINNED.value,
            items=[_item(MemoryAuthority.PINNED, "pinned instruction")],
        ),
        episodic=MemoryLayerView(
            layer=MemoryAuthority.EPISODIC.value,
            items=[_item(MemoryAuthority.EPISODIC, "episodic evidence")],
        ),
        transient=MemoryLayerView(
            layer=MemoryAuthority.TRANSIENT.value,
            items=[_item(MemoryAuthority.TRANSIENT, "current evidence")],
        ),
        compression=MemoryLayerView(
            layer=MemoryAuthority.COMPRESSION.value,
            items=[_item(MemoryAuthority.COMPRESSION, "compressed note")],
            is_reference_only=True,
        ),
    )

    rendered = view.render()

    assert rendered.index("[system]") < rendered.index("[pinned]")
    assert rendered.index("[pinned]") < rendered.index("[episodic]")
    assert rendered.index("[episodic]") < rendered.index("[transient]")
    assert "[compression]" not in rendered


def test_prompt_memory_view_authority_order_constant():
    view = PromptMemoryView(
        transient=MemoryLayerView(
            layer=MemoryAuthority.TRANSIENT.value,
            items=[_item(MemoryAuthority.TRANSIENT, "current evidence")],
        ),
        compression=MemoryLayerView(
            layer=MemoryAuthority.COMPRESSION.value,
            items=[_item(MemoryAuthority.COMPRESSION, "compressed note")],
            is_reference_only=True,
        ),
    )

    assert [layer.layer for layer in view.layers] == [
        "system",
        "pinned",
        "episodic",
        "transient",
    ]
