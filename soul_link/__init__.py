from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

from .contracts import SOUL_LINK_ROOT, resolve_persona_engine_base_dir
from pcltm import MemoryGovernanceOrchestrator


@dataclass(frozen=True)
class SoulLinkRequest:
    message: str
    recent_context: list[dict[str, Any]] = field(default_factory=list)
    previous_mode: str | None = None
    emotion_state: dict[str, Any] = field(default_factory=dict)
    emotion_modifier: str = ""
    platform: str = "cli"
    host_system_prompt: str = ""

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


@dataclass(frozen=True)
class SoulLinkResolution:
    mode: str = "daily"
    route_bucket: str = ""
    selected_layers: list[str] = field(default_factory=list)
    shadow_packet: dict[str, Any] = field(default_factory=dict)
    audit_packet: dict[str, Any] = field(default_factory=dict)
    prompt_candidate: dict[str, Any] | None = None


@dataclass(frozen=True)
class SoulLinkMemoryLifecycleLedgerSnapshot:
    schema_version: int = 1
    transitions: tuple[Any, ...] = ()
    authority_boundary: str = "read_only_governance_control_plane"
    risk_level: str = "medium"
    transition_count: int = 0
    blocked_count: int = 0
    status: str = "available"
    entries: tuple[Any, ...] = ()


@dataclass(frozen=True)
class SoulLinkGovernanceActionSnapshot:
    kind: str
    action_id: str
    risk_level: str = "medium"
    authority_boundary: str = "candidate_memory"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SoulLinkCandidateScoreSnapshot:
    candidate_id: str
    action_id: str
    priority: str = "medium"
    score: float = 0.75
    persona_relevance: float = 0.75
    stale_after: str = "unknown"


@dataclass(frozen=True)
class SoulLinkGovernanceSnapshot:
    dry_run: bool = True
    pending_candidates: int = 0
    reflection_drafts: int = 0
    actions: list[Any] = field(default_factory=list)
    lifecycle_ledger: SoulLinkMemoryLifecycleLedgerSnapshot = field(default_factory=SoulLinkMemoryLifecycleLedgerSnapshot)
    candidate_scores: list[SoulLinkCandidateScoreSnapshot] = field(default_factory=list)
    top_candidates: list[SoulLinkCandidateScoreSnapshot] = field(default_factory=list)
    risk_level: str = "medium"
    status: str = "available"
    report: Any = None


@dataclass(frozen=True)
class SoulLinkSnapshot:
    emotion_state: dict[str, Any]
    memory_lifecycle_ledger: SoulLinkMemoryLifecycleLedgerSnapshot
    governance: SoulLinkGovernanceSnapshot
    schema_version: str = "v1"
    runtime_target: str = ""
    source: str = "soul_link.snapshot"
    trace_id: str = ""
    warnings: list[str] = field(default_factory=list)
    generated_at: str = ""

    @property
    def governance_report(self) -> SoulLinkGovernanceSnapshot:
        return self.governance

    def to_dict(self) -> dict[str, Any]:
        def convert(value: Any) -> Any:
            if is_dataclass(value) and not isinstance(value, type):
                return {key: convert(item) for key, item in asdict(value).items()}
            if isinstance(value, dict):
                return {key: convert(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [convert(item) for item in value]
            return value

        return {
            "schema_version": self.schema_version,
            "runtime_target": self.runtime_target,
            "source": self.source,
            "trace_id": self.trace_id,
            "warnings": list(self.warnings),
            "generated_at": self.generated_at,
            "emotion_state": convert(self.emotion_state),
            "governance_report": convert(self.governance_report),
        }


@runtime_checkable
class HostAdapter(Protocol):
    def run_turn(
        self,
        link: "SoulLink",
        message: str,
        *,
        previous_mode: str | None = None,
        emotion_state: dict[str, Any] | None = None,
        emotion_modifier: str = "",
    ) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class FakeHostAdapter:
    host_system_prompt: str = ""
    context: list[dict[str, Any]] = field(default_factory=list)
    host: str = "fake-host"

    def run_turn(
        self,
        link: "SoulLink",
        message: str,
        *,
        previous_mode: str | None = None,
        emotion_state: dict[str, Any] | None = None,
        emotion_modifier: str = "",
    ) -> dict[str, Any]:
        request = link.ingest(
            message,
            recent_context=self.context,
            previous_mode=previous_mode,
            emotion_state=emotion_state,
            emotion_modifier=emotion_modifier,
            platform=self.host,
            host_system_prompt=self.host_system_prompt,
        )
        resolution = link.resolve(request)
        prompt_candidate = resolution.prompt_candidate or {}
        return {
            "host": self.host,
            "platform": self.host,
            "mode": resolution.mode,
            "route_bucket": resolution.route_bucket,
            "selected_layers": resolution.selected_layers,
            "prompt_text": prompt_candidate.get("prompt_text", ""),
            "prompt_candidate": prompt_candidate,
            "shadow_packet": resolution.shadow_packet,
            "audit_packet": resolution.audit_packet,
        }


class SoulLink:
    def __init__(self, base_dir: str | Path | None = None, **orchestrator_options: Any) -> None:
        self.base_dir = resolve_persona_engine_base_dir(base_dir)
        self.orchestrator_options = dict(orchestrator_options)
        self.__orchestrator = None

    @property
    def _orchestrator(self):
        if self.__orchestrator is None:
            self.__orchestrator = self._build_orchestrator()
        return self.__orchestrator

    def ingest(
        self,
        message: str,
        *,
        recent_context: list[dict[str, Any]] | None = None,
        previous_mode: str | None = None,
        emotion_state: dict[str, Any] | None = None,
        emotion_modifier: str = "",
        platform: str = "cli",
        host_system_prompt: str = "",
    ) -> SoulLinkRequest:
        return SoulLinkRequest(
            message=message,
            recent_context=list(recent_context or []),
            previous_mode=previous_mode,
            emotion_state=dict(emotion_state or {}),
            emotion_modifier=emotion_modifier,
            platform=platform,
            host_system_prompt=host_system_prompt,
        )

    def resolve(
        self,
        request: SoulLinkRequest | Mapping[str, Any],
        *,
        host_system_prompt: str | None = None,
    ) -> SoulLinkResolution:
        if isinstance(request, Mapping):
            request = SoulLinkRequest(
                message=str(request.get("message", "")),
                recent_context=list(request.get("recent_context") or []),
                previous_mode=request.get("previous_mode"),
                emotion_state=dict(request.get("emotion_state") or {}),
                emotion_modifier=str(request.get("emotion_modifier") or ""),
                platform=str(request.get("platform") or "cli"),
                host_system_prompt=str(request.get("host_system_prompt") or ""),
            )

        if host_system_prompt is not None:
            request = SoulLinkRequest(
                message=request.message,
                recent_context=request.recent_context,
                previous_mode=request.previous_mode,
                emotion_state=request.emotion_state,
                emotion_modifier=request.emotion_modifier,
                platform=request.platform,
                host_system_prompt=host_system_prompt,
            )

        emotion_state = request.emotion_state
        emotion_modifier = request.emotion_modifier
        if not emotion_state or not emotion_modifier:
            try:
                from persona_engine.emotion_state_manager import EmotionStateManager
            except ModuleNotFoundError:
                from emotion_state_manager import EmotionStateManager

            manager = EmotionStateManager()
            if not emotion_state:
                messages = [*request.recent_context, {"role": "user", "content": request.message}]
                if hasattr(manager, "update_emotion_state"):
                    manager.update_emotion_state(messages)
                elif hasattr(manager, "update_emotion_from_message"):
                    manager.update_emotion_from_message(request.message, platform=request.platform)
                elif hasattr(manager, "update_emotion"):
                    manager.update_emotion(request.message)
                if hasattr(manager, "get_current_emotion_state"):
                    emotion_state = dict(manager.get_current_emotion_state() or {})
                elif hasattr(manager, "get_emotion_state"):
                    emotion_state = dict(manager.get_emotion_state() or {})
            if not emotion_modifier:
                if hasattr(manager, "get_tone_modifiers"):
                    emotion_modifier = str(manager.get_tone_modifiers() or "")
                elif hasattr(manager, "get_emotion_modifier"):
                    emotion_modifier = str(manager.get_emotion_modifier() or "")
                elif hasattr(manager, "get_current_emotion_modifier"):
                    emotion_modifier = str(manager.get_current_emotion_modifier() or "")
            request = SoulLinkRequest(
                message=request.message,
                recent_context=request.recent_context,
                previous_mode=request.previous_mode,
                emotion_state=emotion_state,
                emotion_modifier=emotion_modifier,
                platform=request.platform,
                host_system_prompt=request.host_system_prompt,
            )

        orchestrator = self._orchestrator
        packet = orchestrator.analyze_turn(
            request.message,
            recent_messages=request.recent_context,
            emotion_state=request.emotion_state,
            emotion_modifier=request.emotion_modifier,
            previous_mode=request.previous_mode,
            platform=request.platform,
        )
        prompt_candidate = None
        if request.host_system_prompt:
            active = orchestrator.compose_active_prompt(
                request.host_system_prompt,
                request.message,
                recent_messages=request.recent_context,
                emotion_state=request.emotion_state,
                emotion_modifier=request.emotion_modifier,
                previous_mode=request.previous_mode,
                platform=request.platform,
                packet=packet,
            )
            prompt_text = active.prompt_text.replace("<memory_profile_notes>", "").replace("</memory_profile_notes>", "")
            prompt_hash = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()[:16]
            prompt_candidate = {
                "prompt_text": prompt_text,
                "prompt_hash": prompt_hash,
                "selected_layers": list(getattr(packet, "selected_layers", []) or []),
            }

        shadow_packet = dict(getattr(packet, "shadow_packet", {}) or {})
        if prompt_candidate and "prompt_hash" not in shadow_packet:
            shadow_packet["prompt_hash"] = prompt_candidate["prompt_hash"]
        audit_packet = dict(getattr(packet, "audit_packet", {}) or {})
        audit_packet.setdefault(
            "request",
            {
                "message": request.message,
                "recent_context": request.recent_context,
                "previous_mode": request.previous_mode,
                "emotion_state": request.emotion_state,
                "emotion_modifier": request.emotion_modifier,
                "platform": request.platform,
            },
        )
        route_bucket = getattr(packet, "route_bucket", "") or "task"
        return SoulLinkResolution(
            mode=getattr(packet, "mode", "daily"),
            route_bucket=route_bucket,
            selected_layers=list(getattr(packet, "selected_layers", []) or []),
            shadow_packet=shadow_packet,
            audit_packet=audit_packet,
            prompt_candidate=prompt_candidate,
        )

    def compose_active_prompt(
        self,
        *,
        host_system_prompt: str,
        user_message: str,
        recent_context: list[dict[str, Any]] | None = None,
        previous_mode: str | None = None,
        emotion_state: dict[str, Any] | None = None,
        emotion_modifier: str = "",
        platform: str = "cli",
    ) -> SoulLinkResolution:
        request = self.ingest(
            user_message,
            recent_context=recent_context,
            previous_mode=previous_mode,
            emotion_state=emotion_state,
            emotion_modifier=emotion_modifier,
            platform=platform,
            host_system_prompt=host_system_prompt,
        )
        return self.resolve(request)

    def snapshot(
        self,
        *,
        host_system_prompt: str | None = None,
        user_message: str = "",
        event_store: Any | None = None,
    ) -> SoulLinkSnapshot:
        emotion_state: dict[str, Any] = {}
        if user_message:
            import importlib

            try:
                manager_module = importlib.import_module("persona_engine.emotion_state_manager")
            except ModuleNotFoundError:
                manager_module = importlib.import_module("emotion_state_manager")
            manager = manager_module.EmotionStateManager()
            messages = [{"role": "user", "content": user_message}]
            if hasattr(manager, "update_emotion_state"):
                manager.update_emotion_state(messages)
            elif hasattr(manager, "update_emotion_from_message"):
                manager.update_emotion_from_message(user_message, platform="cli")
            elif hasattr(manager, "update_emotion"):
                manager.update_emotion(user_message)
            if hasattr(manager, "get_current_emotion_state"):
                emotion_state = dict(manager.get_current_emotion_state() or {})
            elif hasattr(manager, "get_emotion_state"):
                emotion_state = dict(manager.get_emotion_state() or {})
        controller = self.governance(event_store=event_store)
        report = controller.analyze()
        ledger = SoulLinkMemoryLifecycleLedgerSnapshot(
            schema_version=report.lifecycle_ledger.schema_version,
            transitions=tuple(report.lifecycle_ledger.transitions),
            authority_boundary=report.lifecycle_ledger.authority_boundary,
            risk_level="medium",
            transition_count=len(report.actions),
        )
        actions = [
            SoulLinkGovernanceActionSnapshot(
                kind=getattr(action, "kind", ""),
                action_id=f"gov_{index + 1}",
                risk_level="medium",
                authority_boundary=getattr(action, "metadata", {}).get("authority_boundary", "candidate_memory"),
                metadata=dict(getattr(action, "metadata", {}) or {}),
            )
            for index, action in enumerate(report.actions)
        ]
        candidate_scores = [
            SoulLinkCandidateScoreSnapshot(
                candidate_id=str(action.metadata.get("candidate_id", action.metadata.get("record_id", ""))),
                action_id=action.action_id,
                priority=action.risk_level,
                score=0.75,
                persona_relevance=0.75,
            )
            for action in actions
            if action.metadata.get("candidate_id") or action.metadata.get("record_id")
        ]
        governance = SoulLinkGovernanceSnapshot(
            dry_run=report.dry_run,
            pending_candidates=report.pending_candidates,
            reflection_drafts=report.reflection_drafts,
            actions=actions,
            lifecycle_ledger=ledger,
            candidate_scores=candidate_scores,
            top_candidates=candidate_scores,
            report=report,
        )
        return SoulLinkSnapshot(
            emotion_state=emotion_state,
            memory_lifecycle_ledger=ledger,
            governance=governance,
            runtime_target=str(self.base_dir),
            trace_id=f"snap_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}",
            generated_at=datetime.now(timezone.utc).isoformat(),
            warnings=[],
        )

    def governance(self, *, event_store: Any | None = None) -> MemoryGovernanceOrchestrator:
        from pcltm.memfs_store import MemFSStore

        return MemoryGovernanceOrchestrator(
            memfs_store=MemFSStore(SOUL_LINK_ROOT / "var" / "memfs"),
            event_store=event_store,
        )

    def governance_snapshot(self) -> SoulLinkGovernanceSnapshot:
        return SoulLinkGovernanceSnapshot()

    def _build_orchestrator(self):
        import sys

        base_dir = str(self.base_dir)
        if base_dir not in sys.path:
            sys.path.insert(0, base_dir)
        from persona_orchestrator.state_orchestrator import StateOrchestrator

        allowed_options = {
            "log_path",
            "enable_active_sex",
            "enable_semantic_shadow",
            "semantic_backend",
            "sentiment_analyzer",
            "core_source",
            "enable_context_router",
            "enable_model_selector",
            "context_router_config",
            "model_router_config_path",
        }
        options = {key: value for key, value in self.orchestrator_options.items() if key in allowed_options}
        return StateOrchestrator(self.base_dir, **options)


__all__ = [
    "FakeHostAdapter",
    "HostAdapter",
    "MemoryGovernanceOrchestrator",
    "SoulLink",
    "SoulLinkGovernanceSnapshot",
    "SoulLinkMemoryLifecycleLedgerSnapshot",
    "SoulLinkRequest",
    "SoulLinkResolution",
    "SoulLinkSnapshot",
]
