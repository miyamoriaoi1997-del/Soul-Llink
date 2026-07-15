"""Read-only memory usage feedback for PCLTM/MemFS.

This module records control-plane observations about which selected memories
appear to help a response.  It does not mutate memory records by itself; callers
can feed the report into governance/review later.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .memfs_types import MemoryLayerItem, PromptMemoryView


@dataclass(frozen=True)
class MemoryFeedbackSignal:
    """One read-only feedback signal for a selected memory item."""

    record_id: int | None
    memory_id: str
    signal: str
    mode: str
    memory_type: str
    suggested_adjustment: str
    confidence: float = 0.5
    requires_human_review: bool = False
    evidence_refs: tuple[dict[str, Any], ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "memory_id": self.memory_id,
            "signal": self.signal,
            "mode": self.mode,
            "memory_type": self.memory_type,
            "suggested_adjustment": self.suggested_adjustment,
            "confidence": self.confidence,
            "requires_human_review": self.requires_human_review,
            "evidence_refs": [dict(ref) for ref in self.evidence_refs],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class MemoryUsageFeedbackReport:
    """Read-only usage feedback report for one response turn."""

    schema_version: int = 1
    authority_boundary: str = "read_only_usage_feedback"
    selected_record_ids: tuple[int, ...] = ()
    used_record_ids: tuple[int, ...] = ()
    unused_record_ids: tuple[int, ...] = ()
    signals: tuple[MemoryFeedbackSignal, ...] = ()

    @classmethod
    def build(
        cls,
        *,
        selected_record_ids: list[int] | tuple[int, ...] = (),
        used_record_ids: list[int] | tuple[int, ...] = (),
        unused_record_ids: list[int] | tuple[int, ...] | None = None,
        signals: list[MemoryFeedbackSignal] | tuple[MemoryFeedbackSignal, ...] = (),
    ) -> "MemoryUsageFeedbackReport":
        selected = tuple(int(rid) for rid in selected_record_ids if rid is not None)
        used = tuple(int(rid) for rid in used_record_ids if rid is not None)
        if unused_record_ids is None:
            used_set = set(used)
            unused = tuple(rid for rid in selected if rid not in used_set)
        else:
            unused = tuple(int(rid) for rid in unused_record_ids if rid is not None)
        return cls(
            selected_record_ids=selected,
            used_record_ids=used,
            unused_record_ids=unused,
            signals=tuple(signals),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "authority_boundary": self.authority_boundary,
            "selected_record_ids": list(self.selected_record_ids),
            "used_record_ids": list(self.used_record_ids),
            "unused_record_ids": list(self.unused_record_ids),
            "signals": [signal.to_dict() for signal in self.signals],
        }


class MemoryUsageFeedbackRecorder:
    """Analyze memory usage in a response without changing memory state."""

    correction_markers = (
        "不是这样",
        "不对",
        "以后别",
        "别这样",
        "记错",
        "你错了",
        "不是这个",
    )

    def analyze_response(
        self,
        *,
        memory_view: PromptMemoryView,
        response_text: str,
        user_message: str = "",
        mode: str = "default",
    ) -> MemoryUsageFeedbackReport:
        items = [item for layer in memory_view.layers for item in layer.items]
        selected_ids = [rid for item in items if (rid := self._record_id(item)) is not None]
        used_ids: list[int] = []
        signals: list[MemoryFeedbackSignal] = []
        correction = self._has_correction(user_message)

        for item in items:
            record_id = self._record_id(item)
            if record_id is None:
                continue
            used = self._item_used(item, response_text)
            if used:
                used_ids.append(record_id)
                signals.append(
                    self._signal(
                        item,
                        record_id,
                        signal="used_in_response",
                        mode=mode,
                        suggested_adjustment="stabilize",
                        confidence=0.7,
                    )
                )
            else:
                signals.append(
                    self._signal(
                        item,
                        record_id,
                        signal="selected_but_unused",
                        mode=mode,
                        suggested_adjustment="observe_decay",
                        confidence=0.4,
                    )
                )
            if correction:
                signals.append(
                    self._signal(
                        item,
                        record_id,
                        signal="user_corrected",
                        mode=mode,
                        suggested_adjustment="review_or_supersede",
                        confidence=0.8,
                        requires_human_review=True,
                    )
                )
            if self._mode_mismatch(item, mode):
                signals.append(
                    self._signal(
                        item,
                        record_id,
                        signal="mode_mismatch",
                        mode=mode,
                        suggested_adjustment="restrict_mode_affinity",
                        confidence=0.75,
                        requires_human_review=True,
                    )
                )

        return MemoryUsageFeedbackReport.build(
            selected_record_ids=selected_ids,
            used_record_ids=used_ids,
            signals=signals,
        )

    def _record_id(self, item: MemoryLayerItem) -> int | None:
        raw = item.metadata.get("record_id")
        if raw is None:
            match = re.search(r"(?:^|[-_/])(\d{3,})(?:[-_.]|$)", item.id or item.path)
            raw = match.group(1) if match else None
        try:
            return int(raw) if raw is not None else None
        except (TypeError, ValueError):
            return None

    def _signal(
        self,
        item: MemoryLayerItem,
        record_id: int,
        *,
        signal: str,
        mode: str,
        suggested_adjustment: str,
        confidence: float,
        requires_human_review: bool = False,
    ) -> MemoryFeedbackSignal:
        return MemoryFeedbackSignal(
            record_id=record_id,
            memory_id=item.id or item.path,
            signal=signal,
            mode=mode,
            memory_type=item.memory_type,
            suggested_adjustment=suggested_adjustment,
            confidence=confidence,
            requires_human_review=requires_human_review,
            evidence_refs=({"type": "memfs_path", "path": item.path}, {"type": "memory_record", "id": record_id}),
            metadata={"buckets": list(item.buckets), "mode_scope": list(item.mode_scope)},
        )

    def _has_correction(self, user_message: str) -> bool:
        return any(marker in user_message for marker in self.correction_markers)

    def _item_used(self, item: MemoryLayerItem, response_text: str) -> bool:
        if not response_text:
            return False
        response_lower = response_text.lower()
        keys = self._key_phrases(item.body) + self._key_phrases(item.description)
        hits = sum(1 for key in keys if key and key in response_lower)
        return hits >= 1

    def _key_phrases(self, text: str) -> list[str]:
        if not text:
            return []
        keys: list[str] = []
        for token in re.findall(r"[a-zA-Z0-9_./+-]{4,}", text.lower()):
            keys.append(token)
        compact = "".join(text.lower().split())
        for size in (3, 4, 5, 6):
            for index in range(0, max(0, len(compact) - size + 1)):
                chunk = compact[index : index + size]
                if any("\u4e00" <= char <= "\u9fff" for char in chunk):
                    keys.append(chunk)
        return list(dict.fromkeys(keys))[:20]

    def _mode_mismatch(self, item: MemoryLayerItem, mode: str) -> bool:
        if not mode or mode == "default":
            return False
        if "default" in item.mode_scope:
            return False
        return mode not in item.mode_scope


__all__ = [
    "MemoryFeedbackSignal",
    "MemoryUsageFeedbackRecorder",
    "MemoryUsageFeedbackReport",
]
