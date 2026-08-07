"""Runtime shadow adapter for observing multi-SOUL orchestration safely.

This module is intentionally non-invasive: it builds an active prompt candidate
and audit record for observation only. It never decides whether runtime
switching is allowed; callers must enforce any active-routing policy elsewhere.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .state_orchestrator import StateOrchestrator


class RuntimeShadowAdapter:
    """Compute runtime shadow decisions and optionally produce active prompt candidates."""

    def __init__(
        self,
        base_dir: str | Path,
        log_path: str | Path | None = None,
        enable_semantic_shadow: bool = False,
        semantic_backend: str = "local",
        sentiment_analyzer=None,
        moments_path: str | Path | None = None,
        core_source: str = "orchestrator_core",
    ):
        self.base_dir = Path(base_dir)
        self.log_path = Path(log_path) if log_path else self.base_dir / "logs" / "runtime_shadow.jsonl"
        # moments_path is accepted only for backward compatibility. The
        # independent legacy relationship-memory file domain is retired and must not be passed into
        # the active StateOrchestrator path.
        _ = moments_path
        self.orchestrator = StateOrchestrator(
            base_dir=self.base_dir,
            log_path=self.base_dir / "logs" / "persona_orchestrator_shadow.jsonl",
            enable_semantic_shadow=enable_semantic_shadow,
            semantic_backend=semantic_backend,
            sentiment_analyzer=sentiment_analyzer,
            core_source=core_source,
        )

    def analyze_runtime_turn(
        self,
        host_system_prompt: str,
        user_message: str,
        recent_messages: list[dict] | None = None,
        emotion_state: dict | None = None,
        emotion_modifier: str = "",
        previous_mode: str | None = None,
        platform: str = "cli",
        session_id: str | None = None,
        active: bool = False,
        message_timestamp: float | None = None,
    ) -> dict[str, Any]:
        """Return a shadow/active record and candidate prompt.

        When active=True, the record["active"] flag is set so the caller
        knows the candidate_prompt is intended for installation.
        """
        result = self.orchestrator.compose_active_prompt(
            host_system_prompt=host_system_prompt,
            user_message=user_message,
            recent_messages=recent_messages,
            emotion_state=emotion_state,
            emotion_modifier=emotion_modifier,
            previous_mode=previous_mode,
            platform=platform,
            runtime_authority="active" if active else "shadow",
        )
        route_bucket = self._route_bucket(result.packet.mode, result.packet.selected_layers)
        record = {
            "active": active,
            "session_id": session_id,
            "platform": platform,
            "message_timestamp": message_timestamp,
            "user_message_hash": self._hash_text(user_message),
            "previous_mode": previous_mode,
            "mode": result.packet.mode,
            "transition": result.packet.transition,
            "confidence": result.packet.confidence,
            "selected_layers": result.packet.selected_layers,
            "safety_flags": result.packet.safety_flags,
            "desire_tier": result.packet.desire_tier,
            "route_bucket": route_bucket,
            "model_hint": self._model_hint(route_bucket),
            "switch_allowed": False,
            "switch_reason": "runtime_shadow_observation_only",
            "prompt_hash": result.prompt_hash,
            "candidate_prompt_hash": self._hash_text(result.prompt_text),
            "candidate_prompt": result.prompt_text,
            "packet": result.packet,
            "warnings": result.warnings,
        }
        # Keep prompt_hash compatible with the candidate hash for external audit.
        record["candidate_prompt_hash"] = result.prompt_hash
        self._log_record(record)
        return record

    def _log_record(self, record: dict[str, Any]) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        redacted = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "active": record["active"],
            "session_id": record["session_id"],
            "platform": record["platform"],
            "message_timestamp": record["message_timestamp"],
            "user_message_hash": record["user_message_hash"],
            "previous_mode": record["previous_mode"],
            "mode": record["mode"],
            "transition": record["transition"],
            "confidence": record["confidence"],
            "selected_layers": record["selected_layers"],
            "safety_flags": record["safety_flags"],
            "desire_tier": record["desire_tier"],
            "route_bucket": record["route_bucket"],
            "model_hint": record["model_hint"],
            "switch_allowed": record["switch_allowed"],
            "switch_reason": record["switch_reason"],
            "prompt_hash": record["prompt_hash"],
            "warnings": record["warnings"],
            "packet": asdict(record["packet"]),
        }
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(redacted, ensure_ascii=False, sort_keys=True) + "\n")

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _route_bucket(mode: str | None, selected_layers: list[str] | None = None) -> str:
        # sex always routes as sex so router can use sex_model
        if mode == "sex":
            return "sex"
        if mode == "work":
            return "task"
        if mode == "daily":
            return "relationship"
        return "unknown"

    @staticmethod
    def _model_hint(route_bucket: str) -> str:
        if route_bucket == "task":
            return "technical"
        if route_bucket == "sex":
            return "sex"
        if route_bucket == "relationship":
            return "default"
        return route_bucket
