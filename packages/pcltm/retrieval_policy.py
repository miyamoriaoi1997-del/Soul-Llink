"""Mode-aware retrieval policy for Persona Context Long-Term Memory.

The policy is deliberately bounded by the persona state-machine modes. It does
not introduce domain modes such as investment/a-share; those remain query terms
inside the existing work mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


_STATE_MODES = {
    "work",
    "system_maintenance",
    "daily",
    "intimacy",
    "conflict",
    "repair",
    "sex_candidate",
    "sex",
}
_RELATIONSHIP_MODES = {"daily", "intimacy", "conflict", "repair", "sex_candidate", "sex"}
_WORK_MODES = {"work", "system_maintenance"}


@dataclass(frozen=True)
class RetrievalPolicyDecision:
    mode: str
    allowed_modes: tuple[str, ...]
    denied_modes: tuple[str, ...] = ()
    allow_private: bool = False
    allow_restricted: bool = False
    allow_secret: bool = False
    limit: int = 20


class RetrievalPolicy:
    """Filter full-text retrieval by the single persona state-machine mode taxonomy."""

    def decision_for(self, mode: str) -> RetrievalPolicyDecision:
        mode = (mode or "unknown").lower()
        if mode == "system_maintenance":
            return RetrievalPolicyDecision(mode=mode, allowed_modes=("system_maintenance", "work"), denied_modes=tuple(_RELATIONSHIP_MODES))
        if mode == "work":
            return RetrievalPolicyDecision(mode=mode, allowed_modes=("work", "system_maintenance"), denied_modes=tuple(_RELATIONSHIP_MODES))
        if mode == "intimacy":
            return RetrievalPolicyDecision(mode=mode, allowed_modes=tuple(_RELATIONSHIP_MODES), denied_modes=tuple(_WORK_MODES) + ("cron",), allow_private=True)
        if mode in {"daily", "conflict", "repair", "sex_candidate", "sex"}:
            return RetrievalPolicyDecision(mode=mode, allowed_modes=tuple(_RELATIONSHIP_MODES), denied_modes=tuple(_WORK_MODES) + ("cron",), allow_private=True)
        if mode == "cron":
            # Scheduled jobs are not persona long-term memory. Retrieval should return nothing.
            return RetrievalPolicyDecision(mode="no_memory", allowed_modes=(), denied_modes=tuple(_STATE_MODES))
        return RetrievalPolicyDecision(mode="unknown", allowed_modes=("unknown", "work", "system_maintenance"), denied_modes=tuple(_RELATIONSHIP_MODES) + ("no_memory",))

    def retrieve(
        self,
        store: Any,
        *,
        mode: str,
        query: str,
        scope: dict[str, Any],
        include_sensitive: bool = False,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        decision = self.decision_for(mode)
        effective_include_sensitive = include_sensitive and (decision.allow_private or decision.allow_restricted or decision.allow_secret)
        rows = store.search_events(
            query,
            session_id=scope.get("session_id"),
            conversation_id=scope.get("conversation_id"),
            platform=scope.get("platform"),
            persona_mode=scope.get("persona_mode"),
            source=scope.get("source"),
            limit=limit or decision.limit,
            include_sensitive=effective_include_sensitive,
        )
        if not rows:
            seen: set[int] = set()
            rows = []
            for token in [query, *query.split(), *list(query)]:
                for row in store.search_events(
                    token,
                    session_id=scope.get("session_id"),
                    conversation_id=scope.get("conversation_id"),
                    platform=scope.get("platform"),
                    persona_mode=scope.get("persona_mode"),
                    source=scope.get("source"),
                    limit=limit or decision.limit,
                    include_sensitive=effective_include_sensitive,
                ):
                    event_id = row["event_id"]
                    if event_id not in seen:
                        seen.add(event_id)
                        rows.append(row)
        filtered = [row for row in rows if self._allowed(row, decision)]
        return filtered[: limit or decision.limit]

    def _allowed(self, row: dict[str, Any], decision: RetrievalPolicyDecision) -> bool:
        mode = self._row_mode(row)
        sensitivity = row.get("sensitivity") or "normal"
        if mode in decision.denied_modes:
            return False
        if mode == "no_memory":
            return False
        if sensitivity == "private" and not decision.allow_private:
            return False
        if sensitivity == "restricted" and not decision.allow_restricted:
            return False
        if sensitivity == "secret" and not decision.allow_secret:
            return False
        return mode in decision.allowed_modes

    @staticmethod
    def _row_mode(row: dict[str, Any]) -> str:
        """Normalize both new mode-only records and old fine-grained records to state-machine modes."""
        subcategory = (row.get("subcategory") or "unknown").lower()
        category = (row.get("category") or "unknown").lower()
        if subcategory == "no_memory" or category == "no_memory":
            return "no_memory"
        if subcategory in _STATE_MODES:
            return subcategory
        if category in _STATE_MODES:
            return category
        if subcategory in {"tool.output", "task.assistant_response", "task.a_share"}:
            return "work"
        if subcategory in {"memory.system_convention", "memory.project_path"}:
            return "work"
        if subcategory in {"memory.runtime_boundary", "memory.secret_boundary"}:
            return "system_maintenance"
        if subcategory in {"user.relationship", "user.preference"}:
            return "daily"
        if subcategory in {"task.cron", "noise.cron"}:
            return "no_memory"
        if category in {"tool", "task", "memory"}:
            return "work"
        if category == "user":
            return "daily"
        if category == "noise":
            return "no_memory"
        return "unknown"
