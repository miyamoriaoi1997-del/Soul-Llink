"""Versioned registry of governed-memory architecture surfaces."""

from __future__ import annotations

ARCHITECTURE_SURFACES: dict[str, tuple[str, ...]] = {
    "canonical": (
        "pcltm.events",
        "pcltm.memory_claims",
        "pcltm.memory_current",
        "pcltm.memory_governance_events",
        "pcltm.projection_outbox",
        "pcltm.governed_search_open_exact_injection",
    ),
    "legacy": (
        "pcltm.memory_records",
        "pcltm.legacy_assets_quarantine",
        "pcltm.legacy_shadow_migration",
    ),
    "retired": (
        "legacy_direct_db_prompt",
        "legacy_db_memfs_fallback",
        "legacy_memory_tool_sync",
        "legacy_like_supersede",
        "legacy_transactional_memfs_write",
        "legacy_quota_ignore_fallback",
        "legacy_db_archival_open",
        "legacy_memfs_archival_search_open",
        "legacy_layered_memfs_prompt",
        "legacy_memory_records_live_entries",
        "legacy_memory_records_memfs_materialization",
    ),
}

__all__ = ["ARCHITECTURE_SURFACES"]
