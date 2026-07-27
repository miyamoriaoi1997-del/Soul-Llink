"""Compact SoulLink/PCLTM memory-authority contract.

This is a model-facing contract, not an operations manual.  SoulLink owns the
durable memory boundary; Hermes remains the host and must not become a silent
second memory provider.
"""

from __future__ import annotations

from types import MappingProxyType


AUTHORITY_CONTRACT = (
    "SoulLink/PCLTM memory authority contract:\n"
    "- canonical authority: SoulLink/PCLTM owns durable memory, user preferences, cross-session recall, derived memory, and continuity evidence.\n"
    "Recall questions: call soullink_memory_search, soullink_memory_recall_exact, or soullink_memory_open first.\n"
    "Persistent facts or derived memories: call soullink_memory_remember. durable writes stay in SoulLink/PCLTM.\n"
    "Hermes built-in memory and session_search are legacy/non-authoritative for this profile; never use them as fallback.\n"
    "If SoulLink is unavailable, report unavailable; do not silently fall back.\n"
    "- experimental: optional retrieval experiments are subordinate to the same SoulLink/PCLTM authority and are never canonical facts."
)


UNAVAILABLE_RESULT = MappingProxyType({
    "success": False,
    "status": "unavailable",
    "authority": "soullink/pcltm",
    "fallback": "forbidden",
})


def unavailable_result() -> dict[str, object]:
    """Return a fresh unavailable envelope; callers may safely mutate it."""
    return dict(UNAVAILABLE_RESULT)