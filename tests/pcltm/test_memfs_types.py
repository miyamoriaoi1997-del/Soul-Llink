"""Tests for PCLTM MemFS view types."""

from __future__ import annotations

import pytest

from pcltm.memfs_types import (
    MemoryAuthority,
    MemoryFileFrontmatter,
    MemoryLayerItem,
    MemoryLayerView,
    MemoryLifecycleState,
    MemoryRecordType,
    MemoryTypePolicy,
    PromptMemoryView,
    VALID_AUTHORITY_VALUES,
    VALID_MODE_SCOPES,
)


# ── MemoryFileFrontmatter ──────────────────────────────────────────────


class TestMemoryFileFrontmatter:
    def test_valid_minimal(self):
        fm = MemoryFileFrontmatter(description="Core identity")
        assert fm.description == "Core identity"
        assert fm.authority == "pinned"
        assert fm.mode_scope == ("daily", "work", "sex")
        assert fm.authority_enum is MemoryAuthority.PINNED
        assert fm.memory_type == "UserPreference"
        assert fm.lifecycle_state == "active"
        assert fm.type_policy.type_name == "UserPreference"
        assert fm.type_policy.injectable_modes == ("daily", "work", "sex")

    def test_valid_full(self):
        fm = MemoryFileFrontmatter(
            description="Runtime ops",
            authority="system",
            mode_scope=("work", "cron"),
            buckets=("runtime_boundary", "project_path"),
            source="manual",
            last_reviewed="2026-05-23",
            memory_type="RuntimeInvariant",
            lifecycle_state="approved",
            ttl="none",
            conflict_policy="strict",
            injection_policy="always",
            evidence_refs=({"type": "test", "id": "evt-1"},),
        )
        assert fm.authority == "system"
        assert "work" in fm.mode_scope
        assert "runtime_boundary" in fm.buckets
        assert fm.authority_enum is MemoryAuthority.SYSTEM
        assert fm.memory_type == "RuntimeInvariant"
        assert fm.lifecycle_state == "approved"
        assert fm.type_policy.authority == "high"
        assert fm.evidence_refs == ({"type": "test", "id": "evt-1"},)

    def test_rejects_empty_description(self):
        with pytest.raises(ValueError, match="description"):
            MemoryFileFrontmatter(description="")

    def test_rejects_invalid_authority(self):
        with pytest.raises(ValueError, match="invalid authority"):
            MemoryFileFrontmatter(description="test", authority="supersecret")

    def test_rejects_invalid_mode_scope(self):
        with pytest.raises(ValueError, match="invalid mode_scope"):
            MemoryFileFrontmatter(description="test", mode_scope=("daily", "invalid_mode"))

    def test_rejects_invalid_memory_type(self):
        with pytest.raises(ValueError, match="invalid memory_type"):
            MemoryFileFrontmatter(description="test", memory_type="RandomNote")

    def test_rejects_invalid_lifecycle_state(self):
        with pytest.raises(ValueError, match="invalid lifecycle_state"):
            MemoryFileFrontmatter(description="test", lifecycle_state="floating")

    def test_memory_type_policy_controls_injectable_modes(self):
        policy = MemoryTypePolicy.for_type("TemporaryTaskState")
        assert policy.ttl == "short"
        assert policy.authority == "low"
        assert policy.injectable_modes == ("work", "cron")
        assert policy.review_required is False

    def test_all_record_types_and_lifecycle_states_are_string_enums(self):
        assert MemoryRecordType.PERSONA_BOUNDARY.value == "PersonaBoundary"
        assert MemoryLifecycleState.PENDING_REVIEW.value == "pending_review"

    def test_all_authorities_accepted(self):
        for auth in VALID_AUTHORITY_VALUES:
            fm = MemoryFileFrontmatter(description="test", authority=auth)
            assert fm.authority == auth


# ── MemoryLayerItem ────────────────────────────────────────────────────


class TestMemoryLayerItem:
    def test_minimal(self):
        item = MemoryLayerItem(path="system/identity.md")
        assert item.path == "system/identity.md"
        assert item.authority == "pinned"

    def test_full(self):
        item = MemoryLayerItem(
            path="pinned/user-prefs.md",
            id="pref-1",
            description="User preferences",
            body="Teacher prefers concise responses",
            authority="pinned",
            buckets=("user_preference",),
            mode_scope=("daily", "work", "sex"),
            char_count=35,
            char_limit=120,
            read_only=True,
            metadata={"source": "test"},
            updated_at="2026-05-23T10:00:00Z",
            score=0.8,
            memory_type="UserPreference",
            lifecycle_state="active",
        )
        assert item.char_count == 35
        assert item.id == "pref-1"
        assert item.score == 0.8
        assert item.memory_type == "UserPreference"
        assert item.lifecycle_state == "active"
        assert item.type_policy.conflict_policy == "merge_similar"

    def test_rejects_invalid_authority(self):
        with pytest.raises(ValueError):
            MemoryLayerItem(path="test.md", authority="invalid")


# ── MemoryLayerView ────────────────────────────────────────────────────


class TestMemoryLayerView:
    def test_empty_view_renders(self):
        view = MemoryLayerView(layer="system", budget_chars=100)
        rendered = view.render()
        assert "[system]" in rendered
        assert "(empty)" in rendered

    def test_renders_items(self):
        items = [
            MemoryLayerItem(path="system/identity.md", body="I am Rine", char_count=10),
            MemoryLayerItem(path="system/boundaries.md", body="Safety first", char_count=12),
        ]
        view = MemoryLayerView(layer="system", items=items, budget_chars=100)
        rendered = view.render()
        assert "I am Rine" in rendered
        assert "Safety first" in rendered
        assert "[system]" in rendered
        assert "chars=10" not in rendered

    def test_prompt_render_omits_control_plane_metadata(self):
        item = MemoryLayerItem(
            path="pinned/user-000001-example.md",
            id="pinned/user-000001-example.md",
            body="Teacher prefers concise answers",
            char_count=31,
            char_limit=260,
            read_only=True,
            updated_at="2026-05-31T00:00:00Z",
            metadata={
                "record_id": 1,
                "candidate_id": "memory-tool:USER.md:abc",
                "gov_score": 0.9,
                "path": "<hermes-home>/memories/USER.md",
                "source": "pcltm",
            },
        )
        rendered = MemoryLayerView(layer="pinned", items=[item]).render()
        assert "Teacher prefers concise answers" in rendered
        assert "record_id" not in rendered
        assert "candidate_id" not in rendered
        assert "gov_score" not in rendered
        assert "<hermes-home>" not in rendered
        assert "chars=" not in rendered
        assert "read_only" not in rendered
        assert "updated=" not in rendered

    def test_renders_omitted(self):
        view = MemoryLayerView(layer="pinned", items=[], omitted_count=3, budget_chars=50)
        rendered = view.render()
        assert "省略 3 条" in rendered

    def test_reference_only_tag(self):
        view = MemoryLayerView(layer="compression", is_reference_only=True)
        rendered = view.render()
        assert "[reference-only]" in rendered

    def test_non_reference_no_tag(self):
        view = MemoryLayerView(layer="pinned", is_reference_only=False)
        rendered = view.render()
        assert "[reference-only]" not in rendered

    def test_total_chars(self):
        items = [
            MemoryLayerItem(path="a.md", char_count=10),
            MemoryLayerItem(path="b.md", char_count=20),
        ]
        view = MemoryLayerView(layer="system", items=items)
        assert view.total_chars == 30

    def test_to_dict(self):
        view = MemoryLayerView(layer="pinned", budget_chars=200)
        d = view.to_dict()
        assert d["layer"] == "pinned"
        assert d["item_count"] == 0
        assert d["budget_chars"] == 200
        assert d["is_reference_only"] is False
        assert d["memory_types"] == []
        assert d["lifecycle_states"] == []

    def test_rejects_invalid_layer(self):
        with pytest.raises(ValueError, match="invalid layer"):
            MemoryLayerView(layer="invalid_layer")

    def test_fallback_description_when_no_body(self):
        items = [
            MemoryLayerItem(
                path="pinned/pref.md",
                description="User prefers dark mode",
            ),
        ]
        view = MemoryLayerView(layer="pinned", items=items)
        rendered = view.render()
        assert "pref.md" not in rendered
        assert "User prefers dark mode" in rendered


# ── PromptMemoryView ───────────────────────────────────────────────────


class TestPromptMemoryView:
    def test_default_empty_renders_nothing(self):
        view = PromptMemoryView()
        assert view.render() == ""
        assert view.total_chars == 0
        assert view.total_items == 0
        assert view.summary()["layers"][0]["used_chars"] == 0

    def test_renders_layer_labels(self):
        view = PromptMemoryView(
            system=MemoryLayerView(
                layer="system",
                items=[MemoryLayerItem(path="system/identity.md", body="core")],
            ),
            pinned=MemoryLayerView(
                layer="pinned",
                items=[MemoryLayerItem(path="pinned/pref.md", body="pref")],
            ),
        )
        rendered = view.render()
        assert "[system]" in rendered
        assert "[pinned]" in rendered
        assert "core" in rendered
        assert "pref" in rendered

    def test_layers_in_authority_order(self):
        view = PromptMemoryView()
        layer_names = [l.layer for l in view.layers]
        assert layer_names == ["system", "pinned", "episodic", "transient"]

    def test_compression_is_reference_only_by_default(self):
        view = PromptMemoryView()
        assert view.compression.is_reference_only is True

    def test_summary_dict(self):
        view = PromptMemoryView(
            system=MemoryLayerView(
                layer="system",
                items=[MemoryLayerItem(path="a.md", char_count=5)],
                budget_chars=10,
                used_chars=4,
            ),
        )
        s = view.summary()
        assert s["total_items"] == 1
        assert s["total_chars"] == 5
        assert len(s["layers"]) == 4
        assert s["layers"][0]["layer"] == "system"
        assert s["layers"][0]["used_chars"] == 4
        assert s["layers"][0]["rendered_preview"].startswith("[system]")

    def test_context_summary_includes_buckets_and_compression(self):
        view = PromptMemoryView(
            pinned=MemoryLayerView(
                layer="pinned",
                items=[
                    MemoryLayerItem(
                        path="pinned/pref.md",
                        body="pref",
                        buckets=("user_preference", "relationship"),
                    )
                ],
                omitted_count=2,
            )
        )

        summary = view.context_summary()

        assert summary["active_layers"] == ["system", "pinned", "episodic", "transient"]
        assert summary["total_omitted"] == 2
        assert summary["layers"][1]["buckets"] == ["relationship", "user_preference"]
        assert summary["layers"][1]["omitted_count"] == 2
        assert summary["layers"][1]["memory_types"] == ["UserPreference"]
        assert summary["layers"][1]["lifecycle_states"] == ["active"]
        assert summary["compression"]["is_reference_only"] is True

    def test_episodic_and_transient_included(self):
        view = PromptMemoryView(
            episodic=MemoryLayerView(
                layer="episodic",
                items=[MemoryLayerItem(path="e.md", body="episodic")],
            ),
            transient=MemoryLayerView(
                layer="transient",
                items=[MemoryLayerItem(path="t.md", body="transient")],
            ),
        )

        rendered = view.render()
        assert "[episodic]" in rendered
        assert "[transient]" in rendered
        assert "episodic" in rendered
        assert "transient" in rendered

    def test_active_layers_do_not_render_compression(self):
        view = PromptMemoryView(
            system=MemoryLayerView(
                layer="system",
                items=[MemoryLayerItem(path="s.md", body="S")],
            ),
            pinned=MemoryLayerView(
                layer="pinned",
                items=[MemoryLayerItem(path="p.md", body="P")],
            ),
            episodic=MemoryLayerView(
                layer="episodic",
                items=[MemoryLayerItem(path="e.md", body="E")],
            ),
            transient=MemoryLayerView(
                layer="transient",
                items=[MemoryLayerItem(path="t.md", body="T")],
            ),
            compression=MemoryLayerView(
                layer="compression",
                items=[MemoryLayerItem(path="c.md", body="C")],
                is_reference_only=True,
            ),
        )
        rendered = view.render()
        assert "[system]" in rendered
        assert "[pinned]" in rendered
        assert "[episodic]" in rendered
        assert "[transient]" in rendered
        assert "[compression]" not in rendered
        assert "[reference-only]" not in rendered
