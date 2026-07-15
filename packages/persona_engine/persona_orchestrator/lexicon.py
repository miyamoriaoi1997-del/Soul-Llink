"""Shared lexical groups for the three-mode persona router.

All term lists are loaded from config/routing_rules.yaml at import time.
This module re-exports the same names as before for backward compatibility.
"""
from __future__ import annotations

from .config_loader import load_routing_config

_cfg = load_routing_config()

# ── Exported term groups (same names as before) ────────────────────────────────
SYSTEM_DOMAIN_TERMS: list[str] = _cfg.lexicon("system_domain")
WORK_DOMAIN_TERMS: list[str] = _cfg.lexicon("work_domain")
TASK_ACTION_TERMS: list[str] = _cfg.lexicon("task_action")
CONTEXT_CONTINUATION_TERMS: list[str] = _cfg.lexicon("context_continuation")
CONTEXT_QUESTION_TERMS: list[str] = _cfg.lexicon("context_question")
CONTEXT_ACTION_TERMS: list[str] = _cfg.lexicon("context_action")
DESTRUCTIVE_OR_CLEANUP_ACTION_TERMS: list[str] = _cfg.lexicon("destructive_or_cleanup")


def combined(*groups: list[str]) -> list[str]:
    """Return a stable de-duplicated concatenation of term groups."""
    out: list[str] = []
    for group in groups:
        for term in group:
            if term not in out:
                out.append(term)
    return out
