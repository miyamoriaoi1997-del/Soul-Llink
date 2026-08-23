"""Emotion bridge: persona_engine emotion state wired into the ZCode hook.

The persona engine (``persona_engine.emotion_state_manager.EmotionStateManager``)
is host-independent and falls back to pure rules when torch/transformers are
absent, so this bridge is safe to run inside a hook. It owns:

- the emotion state file ``<zcode_root>/soullink/STATE.md`` (YAML frontmatter
  written by the persona engine, entirely separate from any Hermes profile);
- the continuation flag ``<zcode_root>/soullink/emotion-state.json`` consumed
  by the ``Stop`` hook (``continue: true`` when the current emotion is strong);
- a prompt-safe transient evidence capsule so the current emotion is
  retrievable through governed PCLTM search.

Everything fails open: any emotion-layer failure degrades to a neutral tone
and never blocks or interrupts the session.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import UTC
from pathlib import Path
from typing import Any

STATE_FILE = "STATE.md"
CONTINUATION_FILE = "emotion-state.json"
TONE_STRONG = ("intense", "overwhelming")


def _zcode_root() -> Path:
    return Path(os.environ.get("ZCODE_ROOT", Path.home() / ".zcode" / "cli")).expanduser().resolve()


def _load_continuation(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


class EmotionBridge:
    """Thin, fail-open wrapper around the persona engine emotion manager."""

    def __init__(self, zcode_root: Path | None = None) -> None:
        self._root = zcode_root or _zcode_root()
        self._manager: Any = None
        self._init_error: str | None = None

    @property
    def soullink_dir(self) -> Path:
        return self._root / "soullink"

    @property
    def state_path(self) -> Path:
        return self.soullink_dir / STATE_FILE

    @property
    def continuation_path(self) -> Path:
        return self.soullink_dir / CONTINUATION_FILE

    def _manager_or_none(self) -> Any:
        if self._manager is not None:
            return self._manager
        if not self._enabled():
            return None
        try:
            from persona_engine.emotion_state_manager import EmotionStateManager

            self._manager = EmotionStateManager(
                hermes_home=self.soullink_dir,
                state_path=self.state_path,
            )
        except Exception as exc:  # fail open: any import/init failure is neutral
            self._init_error = str(exc)
            return None
        return self._manager

    def _enabled(self) -> bool:
        """Whether the emotion layer is enabled for this deployment.

        Reads ``soullink/adapter.json``; absent or ``true`` means enabled,
        ``false`` strips the emotion layer for the deployment while leaving
        the code intact (the persona engine is never imported)."""
        try:
            adapter = json.loads((self.soullink_dir / "adapter.json").read_text(encoding="utf-8"))
            return bool(adapter.get("emotion_enabled", True))
        except (OSError, json.JSONDecodeError):
            return True

    def update(self, prompt: str, *, session_id: str = "") -> dict[str, Any]:
        """Detect emotion from the user turn, persist state, and update the
        continuation flag. Returns the resulting emotion snapshot (or an
        empty dict when the emotion layer is unavailable)."""
        try:
            manager = self._manager_or_none()
            if manager is None:
                return {}
            messages = [{"role": "user", "content": prompt}]
            manager.apply_time_decay_if_needed()
            manager.update_emotion_state(messages=messages)
            state = self.emotion_state()
            self._write_continuation(state)
            self._write_evidence(state)
            return state
        except Exception as exc:
            print(f"SoulLink emotion update unavailable: {exc}", file=sys.stderr)
            return {}

    def emotion_state(self) -> dict[str, Any]:
        try:
            manager = self._manager_or_none()
            if manager is None:
                return {}
            return dict(manager.get_current_emotion_state())
        except Exception as exc:
            print(f"SoulLink emotion state unavailable: {exc}", file=sys.stderr)
            return {}

    def tone_modifier(self, max_chars: int = 12000) -> str:
        try:
            manager = self._manager_or_none()
            if manager is None:
                return ""
            return str(manager.get_tone_modifiers())[:max_chars]
        except Exception as exc:
            print(f"SoulLink tone modifier unavailable: {exc}", file=sys.stderr)
            return ""

    def _decide_continue(self, state: dict[str, Any]) -> bool:
        """Single source of truth for the continuation decision.

        emotion_score is synthesised in [-5, +5] and can never reach a 30
        threshold; continuation is driven solely by the persona tone tier
        (``intense`` / ``overwhelming``), which maps to per-dimension
        deviation from baseline. ``update()`` persists this decision and the
        ``Stop`` hook consumes it through ``continuation_request()``.
        """
        if not state:
            return False
        tone = self.tone_modifier()
        return any(marker in tone for marker in TONE_STRONG)

    def is_emotion_strong(self) -> bool:
        return self._decide_continue(self.emotion_state())

    def continuation_request(self) -> dict[str, Any]:
        """The Stop hook reads this: continue only when the emotion decision
        persisted by ``update()`` requests it (zero recomputation at Stop)."""
        payload = _load_continuation(self.continuation_path)
        if not payload.get("continue"):
            return {}
        return {"continue": True, "reason": "SoulLink/PCLTM emotion state requests continuation"}

    def _write_continuation(self, state: dict[str, Any]) -> None:
        try:
            strong = self._decide_continue(state)
            score = int(state.get("emotion_score") or 0)
            payload = {
                "continue": bool(strong),
                "emotion_score": score,
                "updated_at": self._now_iso(),
            }
            self.continuation_path.parent.mkdir(parents=True, exist_ok=True)
            temp = self.continuation_path.with_name(self.continuation_path.name + ".tmp")
            temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(temp, self.continuation_path)
        except Exception as exc:
            print(f"SoulLink continuation write unavailable: {exc}", file=sys.stderr)

    def _write_evidence(self, state: dict[str, Any]) -> None:
        try:
            from pcltm import memory_adapter
            from pcltm.runtime_paths import resolve_memfs_root

            body = json.dumps(
                {
                    "affection": state.get("affection"),
                    "trust": state.get("trust"),
                    "possessiveness": state.get("possessiveness"),
                    "patience": state.get("patience"),
                    "emotion_score": state.get("emotion_score"),
                    "current_emotion": state.get("current_emotion"),
                    "tone": self.tone_modifier(400),
                },
                ensure_ascii=False,
            )
            memory_adapter.write_evidence_capsule(
                title="zcode emotion state",
                body=body,
                mode="default",
                buckets=["emotion_state", "current_task"],
                source_tool="emotion",
                evidence_id=f"emotion-{self._now_iso()}",
                root=resolve_memfs_root(),
            )
        except Exception as exc:
            print(f"SoulLink emotion evidence unavailable: {exc}", file=sys.stderr)

    @staticmethod
    def _now_iso() -> str:
        from datetime import datetime

        return datetime.now(UTC).isoformat()
