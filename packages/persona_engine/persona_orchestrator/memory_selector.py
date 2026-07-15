"""Memory profile selection for persona orchestration."""

from __future__ import annotations

from .types import MODE_DAILY, MODE_SEX, MODE_WORK, MemorySelection


WORK_LAYERS = ["system", "pinned", "episodic", "transient"]
WORK_ACTIVE_LAYERS = ["system", "pinned", "transient"]
WORK_ARCHIVAL_LAYERS = ["episodic"]
WORK_BUCKETS = [
    "runtime_boundary",
    "project_path",
    "current_task",
    "active_task",
    "continuity_capsule",
    "user_preference",
    "investment",
]
WORK_BUDGETS = {
    "system": 500,
    "pinned": 1000,
    "episodic": 1500,
    "transient": 1500,
}

SEX_LAYERS = ["system", "pinned"]
SEX_ACTIVE_LAYERS = ["system", "pinned"]
SEX_ARCHIVAL_LAYERS: list[str] = []
SEX_BUCKETS = ["relationship", "emotion_boundary", "user_preference"]
SEX_BUDGETS = {
    "system": 500,
    "pinned": 1000,
}

DAILY_LAYERS = ["system", "pinned", "episodic"]
DAILY_ACTIVE_LAYERS = ["system", "pinned"]
DAILY_ARCHIVAL_LAYERS = ["episodic"]
DAILY_BUCKETS = ["relationship", "user_preference", "emotion_boundary"]
DAILY_BUDGETS = {
    "system": 500,
    "pinned": 1000,
    "episodic": 1000,
}


class MemorySelector:
    """Select lightweight memory profiles by active mode."""

    def select(self, mode: str, safety_flags: list[str] | None = None) -> MemorySelection:
        flags = list(dict.fromkeys(safety_flags or []))
        raw_mode = mode
        if mode not in {MODE_DAILY, MODE_WORK, MODE_SEX}:
            mode = MODE_DAILY

        if mode == MODE_WORK:
            return MemorySelection(
                profile="technical_plus_core_relationship",
                candidate_files=["MEMORY.md", "USER.md", "STATE.md"],
                reason="work mode needs technical facts plus stable relationship boundaries",
                safety_flags=flags,
                layers=list(WORK_LAYERS),
                buckets=list(WORK_BUCKETS),
                budgets=dict(WORK_BUDGETS),
                active_layers=list(WORK_ACTIVE_LAYERS),
                archival_layers=list(WORK_ARCHIVAL_LAYERS),
                retrieval_policy="archival_reference_only",
            )

        if mode == MODE_SEX:
            return MemorySelection(
                profile="relationship_preferences_desire_gated",
                candidate_files=["MEMORY.md", "USER.md", "STATE.md"],
                reason="sex mode uses relationship preferences, tagged MEMORY milestones, and current state",
                safety_flags=flags,
                layers=list(SEX_LAYERS),
                buckets=list(SEX_BUCKETS),
                budgets=dict(SEX_BUDGETS),
                active_layers=list(SEX_ACTIVE_LAYERS),
                archival_layers=list(SEX_ARCHIVAL_LAYERS),
                retrieval_policy="active_relationship_only",
            )

        return MemorySelection(
            profile="core_relationship",
            candidate_files=["MEMORY.md", "USER.md", "STATE.md"],
            reason="daily mode needs tagged MEMORY milestones, stable relationship preferences, and current state",
            safety_flags=flags if raw_mode in {MODE_DAILY, "intimacy", "conflict"} else list(dict.fromkeys(flags + ["unknown_mode_fallback"])),
            layers=list(DAILY_LAYERS),
            buckets=list(DAILY_BUCKETS),
            budgets=dict(DAILY_BUDGETS),
            active_layers=list(DAILY_ACTIVE_LAYERS),
            archival_layers=list(DAILY_ARCHIVAL_LAYERS),
            retrieval_policy="archival_reference_only",
        )
