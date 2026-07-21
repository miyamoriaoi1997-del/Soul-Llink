"""High-level shadow pipeline for persona state orchestration.

v2: Context Router + Model Selector fully integrated.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from soul_link.contracts import resolve_persona_engine_base_dir

from .context_router import AbstractStateAdapter, ContextRouter, ContextRouterConfig, ContextRouteResult, IntegrationLevel
from .memory_selector import MemorySelector
from .mode_classifier import ModeClassifier
from .model_selector import ModelSelector
from .observability import OrchestratorLogger
from .prompt_composer import PromptComposer
from .semantic_classifier import SemanticModeClassifier
from .transition_manager import TransitionManager
from .transition_manager_v2 import TransitionManagerV2
from .types import ActivePromptResult, MODE_DAILY, MODE_SEX, MODE_WORK, MemorySelection, ModeDecision, StatePacket


class StateOrchestrator:
    """Coordinate three-state classification, transition, memory, and prompt preview.

    v2 additions:
    - Context Router as primary routing authority (DIRECT_SWITCH)
    - Dynamic Model Selector based on mode/context
    - Anti-flap state tracking
    """

    ACTIVE_LAYER_MODES = {MODE_DAILY, MODE_WORK, MODE_SEX}

    def __init__(
        self,
        base_dir: str | Path,
        log_path: str | Path | None = None,
        enable_active_sex: bool = True,
        enable_semantic_shadow: bool = False,
        semantic_backend: str = "local",
        sentiment_analyzer=None,
        core_source: str = "orchestrator_core",
        enable_context_router: bool = True,
        enable_model_selector: bool = True,
        context_router_config: ContextRouterConfig | None = None,
        model_router_config_path: str | Path | None = None,
    ):
        if core_source not in {"orchestrator_core", "host_core"}:
            raise ValueError(f"invalid core_source: {core_source}")
        self.base_dir = self._resolve_base_dir(Path(base_dir))
        self.enable_active_sex = enable_active_sex
        self.enable_semantic_shadow = enable_semantic_shadow
        self.enable_context_router = enable_context_router
        self.enable_model_selector = enable_model_selector
        self.core_source = core_source

        # Legacy components
        self.classifier = ModeClassifier()
        self.semantic_classifier = (
            SemanticModeClassifier(backend=semantic_backend, sentiment_analyzer=sentiment_analyzer)
            if enable_semantic_shadow
            else None
        )
        # Use v2 TransitionManager with shadow table support
        self.transitions = TransitionManagerV2()
        self.memory_selector = MemorySelector()
        self.composer = PromptComposer(self.base_dir, core_source=core_source)
        self.logger = OrchestratorLogger(log_path or (self.base_dir / "logs" / "persona_orchestrator_shadow.jsonl"))

        # v2: Context Router
        if enable_context_router:
            cr_config = context_router_config or ContextRouterConfig(
                enabled=True,
                integration=IntegrationLevel.DIRECT_SWITCH,
            )
            self.context_router = ContextRouter(config=cr_config)
            self.adapter = AbstractStateAdapter()
        else:
            self.context_router = None
            self.adapter = None

        # v2: Model Selector
        self._model_config = self._load_model_config(model_router_config_path)
        self.model_selector = ModelSelector(config_dict=self._model_config) if self._model_config else None

        # v2: Anti-flap state
        # Start high so the first turn is never blocked by anti-flapping
        self._turns_since_last_switch = 99
        self._last_top_mode: str | None = None
        self._last_submode: str | None = None
        self._recent_decisions: list[ModeDecision] = []
        self._component_health: dict[str, dict[str, Any]] = {
            "pcltm": {"status": "unknown"},
        }

    @staticmethod
    def _resolve_base_dir(base_dir: Path) -> Path:
        return resolve_persona_engine_base_dir(base_dir)

    def _load_model_config(self, config_path: str | Path | None) -> dict:
        """Load and normalize model router config from yaml.

        The production proxy config uses ``routing.sex_model`` while the
        in-process ModelSelector consumes ``mode_overrides.active_layer``.
        Normalize both shapes here so the state machine and proxy share one
        routing source instead of drifting.
        """
        if not self.enable_model_selector:
            return {}
        if config_path:
            candidates = [Path(config_path)]
        else:
            candidates = [
                self.base_dir.parent / "model_router" / "config.yaml",
                (Path.home() / "soul-link" / "config" / "model-router.example.yaml"),
                self.base_dir / "config" / "model-router.example.yaml",
                self.base_dir / ".." / "model_router" / "config.example.yaml",
                self.base_dir.parent / "model_router" / "config.example.yaml",
            ]
        for p in candidates:
            if p.exists():
                with open(p) as f:
                    return self._normalize_model_config(yaml.safe_load(f) or {})
        return {}

    @staticmethod
    def _normalize_model_config(raw: dict) -> dict:
        """Normalize proxy/scaffold model config into ModelSelector shape."""
        if not isinstance(raw, dict):
            return {}
        if "routing" not in raw:
            return raw

        routing = raw.get("routing") or {}
        return {
            "default_model": routing.get("default_model") or "persona-auto",
            "mode_overrides": {
                "work": routing.get("work_model"),
                "daily": None,
                "active_layer": routing.get("sex_model"),
            },
            "platform_overrides": raw.get("platform_overrides") or {},
            "emotion_overrides": raw.get("emotion_overrides") or {},
            "model_switch_cooldown": raw.get("model_switch_cooldown", 3),
        }

    # ─── Public API ─────────────────────────────────────────────────────────

    def health_status(self) -> dict[str, Any]:
        """Return non-sensitive runtime degradation telemetry."""
        components = dict(self._component_health)
        logger_status = self.logger.status()
        components["observability"] = {
            "status": "healthy" if logger_status["healthy"] else "degraded",
            **logger_status,
        }
        status = "degraded" if any(
            component.get("status") == "degraded" for component in components.values()
        ) else "healthy"
        return {"status": status, "components": components}

    def analyze_turn(
        self,
        user_message: str,
        recent_messages: list[dict] | None = None,
        emotion_state: dict | None = None,
        emotion_modifier: str = "",
        previous_mode: str | None = None,
        platform: str = "cli",
        session_id: str = "",
        turn_number: int | None = None,
        runtime_authority: str = "shadow",
    ) -> StatePacket:
        authority = "active" if runtime_authority == "active" else "shadow"
        packet, memory_notes, selected_layers = self._analyze(
            user_message=user_message,
            recent_messages=recent_messages,
            emotion_state=emotion_state,
            emotion_modifier=emotion_modifier,
            previous_mode=previous_mode,
            platform=platform,
            shadow_only=authority != "active",
        )
        composition = self.composer.compose(
            selected_layers=selected_layers,
            emotion_modifier=emotion_modifier,
            memory_notes=memory_notes,
            shadow_only=authority != "active",
        )
        packet.prompt_hash = composition.prompt_hash
        packet.selected_layers = composition.selected_layers
        self.logger.log(packet, extra={
            "warnings": composition.warnings,
            "session_id": session_id,
            "turn_number": turn_number,
            "runtime_authority": authority,
        })
        return packet

    def compose_active_prompt(
        self,
        host_system_prompt: str,
        user_message: str,
        recent_messages: list[dict] | None = None,
        emotion_state: dict | None = None,
        emotion_modifier: str = "",
        previous_mode: str | None = None,
        platform: str = "cli",
        packet: StatePacket | None = None,
        runtime_authority: str = "active",
    ) -> ActivePromptResult:
        shadow_only = runtime_authority != "active"
        if packet is None:
            analysis = self._analyze(
                user_message=user_message,
                recent_messages=recent_messages,
                emotion_state=emotion_state,
                emotion_modifier=emotion_modifier,
                previous_mode=previous_mode,
                platform=platform,
                shadow_only=shadow_only,
            )
            packet, memory_notes, selected_layers = analysis
        else:
            packet.shadow_only = shadow_only
            memory_notes = self.memory_selector.select(packet.mode, packet.safety_flags).reason
            selected_layers = list(packet.selected_layers)

        memory_view_text = ""
        memory_selection = self.memory_selector.select(packet.mode, packet.safety_flags)
        try:
            from pcltm.memory_adapter import load_layered_prompt_context

            memory_view = load_layered_prompt_context(
                mode=packet.mode,
                query=user_message,
                budgets=memory_selection.budgets,
                layers=memory_selection.layers,
                buckets=memory_selection.buckets,
                active_layers=memory_selection.active_layer_contract(),
            )
            frame = memory_view.active_frame() if hasattr(memory_view, "active_frame") else None
            if frame is not None:
                memory_view_text = str(frame.active_text or "").strip()
                if memory_view_text.startswith("<pcltm_context>") and memory_view_text.endswith("</pcltm_context>"):
                    memory_view_text = memory_view_text.removeprefix("<pcltm_context>").removesuffix("</pcltm_context>").strip()
                memory_context_summary = frame.to_dict()
                if isinstance(getattr(frame, "audit", None), dict):
                    # Preserve the historical top-level audit shape for
                    # dashboards/tests while still exposing the full
                    # ActiveContextFrame under explicit keys.
                    for key, value in frame.audit.items():
                        if key in {"active_layers", "selected_layers"}:
                            memory_context_summary[key] = value
                        else:
                            memory_context_summary.setdefault(key, value)
            else:
                render_active = getattr(memory_view, "render_active_frame", None)
                if callable(render_active):
                    memory_view_text = str(render_active() or "").strip()
                else:
                    memory_view_text = str(memory_view.render() or "").strip()
                memory_context_summary = memory_view.context_summary()
            memory_context_summary["selection_contract"] = memory_selection.contract_summary()
            self._component_health["pcltm"] = {"status": "healthy"}
        except Exception as exc:
            memory_view_text = ""
            memory_context_summary = {}
            self._component_health["pcltm"] = {
                "status": "degraded",
                "error_type": type(exc).__name__,
            }
            packet.route_metadata["runtime_health"] = self.health_status()

        composition = self.composer.compose_active(
            host_system_prompt=host_system_prompt,
            selected_layers=selected_layers,
            emotion_modifier=emotion_modifier,
            memory_notes=memory_notes,
            memory_view_text=memory_view_text,
            preserve_host_pcltm_fallback=False,
            emit_empty_pcltm_boundary=True,
        )
        packet.prompt_hash = composition.prompt_hash
        packet.selected_layers = composition.selected_layers
        if memory_context_summary:
            packet.route_metadata["memory_context_summary"] = memory_context_summary
        warnings = list(composition.warnings)
        if self._component_health["pcltm"].get("status") == "degraded":
            warnings.append("pcltm_degraded")
        packet.route_metadata["runtime_health"] = self.health_status()
        log_ok = self.logger.log(packet, extra={
            "warnings": warnings,
            "active_candidate": not shadow_only,
            "runtime_authority": "shadow" if shadow_only else "active",
            "memory_context_summary": memory_context_summary,
        })
        if not log_ok:
            packet.route_metadata["runtime_health"] = self.health_status()
        return ActivePromptResult(
            prompt_text=composition.prompt_text,
            prompt_hash=composition.prompt_hash,
            packet=packet,
            warnings=warnings,
        )

    # ─── Internal Pipeline ──────────────────────────────────────────────────

    def _analyze(
        self,
        user_message: str,
        recent_messages: list[dict] | None = None,
        emotion_state: dict | None = None,
        emotion_modifier: str = "",
        previous_mode: str | None = None,
        platform: str = "cli",
        shadow_only: bool = True,
    ) -> tuple[StatePacket, str, list[str]]:
        # Layer 1: Emotion
        emotion_score = self._extract_emotion_score(emotion_state)
        desire_tier = self._desire_tier(emotion_score)

        # Layer 2: Legacy classifier
        mode_decision = self.classifier.classify(
            user_message=user_message,
            recent_messages=recent_messages,
            emotion_state=emotion_state,
            platform=platform,
        )

        # Layer 3: Semantic shadow (optional)
        semantic_shadow = None
        if self.semantic_classifier:
            semantic_shadow = self.semantic_classifier.classify(
                user_message=user_message,
                recent_messages=recent_messages,
                emotion_state=emotion_state,
                previous_mode=previous_mode,
                platform=platform,
            )

        # Layer 4: Context Router (primary decision authority)
        context_result = self._run_context_router(
            mode_decision=mode_decision,
            previous_mode=previous_mode,
            emotion_score=emotion_score,
            desire_tier=desire_tier,
        )

        # Layer 5: Resolve final mode
        final_mode, final_submode, context_reason, extra_safety_flags = self._resolve_final_mode(
            context_result=context_result,
            mode_decision=mode_decision,
            previous_mode=previous_mode,
            desire_tier=desire_tier,
        )

        # Layer 6: Transition (using resolved mode)
        merged_safety_flags = list(mode_decision.safety_flags or []) + list(extra_safety_flags or [])
        transition = self.transitions.transition(
            previous_mode,
            ModeDecision(
                mode=final_mode,
                submode=final_submode or mode_decision.submode,
                confidence=mode_decision.confidence,
                reason=context_reason or mode_decision.reason,
                safety_flags=merged_safety_flags,
                signals=mode_decision.signals,
            ),
            desire_tier=desire_tier,
            enable_active_sex=self.enable_active_sex,
            emotion_score=emotion_score,
        )

        # Layer 7: Memory + layers
        memory = self.memory_selector.select(transition.active_mode, transition.safety_flags)
        selected_layers = self._selected_layers(
            transition.active_mode,
            transition.safety_flags,
            previous_mode=previous_mode,
        )
        safety_flags = self._merge_flags(transition.safety_flags, memory.safety_flags, extra_safety_flags)
        memory_notes = f"profile={memory.profile}; candidates={','.join(memory.candidate_files)}"

        # Layer 8: Model selector
        model_override = self._select_model(
            mode=transition.active_mode,
            submode=final_submode or mode_decision.submode,
            platform=platform,
            emotion_score=emotion_score,
            context_result=context_result,
        )

        route_metadata = self._route_metadata(transition.active_mode)
        route_metadata["decision_audit"] = self._decision_audit(
            previous_mode=previous_mode,
            mode_decision=mode_decision,
            context_result=context_result,
            transition=transition,
            memory=memory,
            emotion_score=emotion_score,
            desire_tier=desire_tier,
            emotion_modifier=emotion_modifier,
            selected_layers=selected_layers,
            extra_safety_flags=extra_safety_flags,
        )
        if model_override:
            route_metadata["hermes_selected_model"] = model_override

        # Layer 9: Build packet
        packet = StatePacket(
            mode=transition.active_mode,
            submode=final_submode or mode_decision.submode,
            confidence=transition.confidence,
            reason=self._reason_with_semantic_shadow(
                context_reason or mode_decision.reason,
                transition.reason,
                memory.reason,
                semantic_shadow,
            ),
            transition=transition.transition,
            selected_layers=selected_layers,
            memory_profile=memory.profile,
            safety_flags=safety_flags,
            emotion_score=emotion_score,
            desire_tier=desire_tier,
            prompt_hash=None,
            shadow_only=shadow_only,
            semantic_shadow=semantic_shadow,
            model_override=model_override,
            selected_model=model_override,
            route_metadata=route_metadata,
        )

        # Update anti-flap state
        self._update_anti_flap(transition.active_mode, final_submode)
        self._recent_decisions.append(mode_decision)
        if len(self._recent_decisions) > 8:
            self._recent_decisions = self._recent_decisions[-8:]

        return packet, memory_notes, selected_layers

    # ─── Layer 4: Context Router ────────────────────────────────────────────

    def _run_context_router(
        self,
        mode_decision: ModeDecision,
        previous_mode: str | None,
        emotion_score: float | None,
        desire_tier: str | None,
    ) -> ContextRouteResult | None:
        """Run context router if enabled. Returns None on error/disabled."""
        if not self.enable_context_router or not self.context_router or not self.adapter:
            return None

        try:
            abstract_input = self.adapter.build_abstract_input(
                mode_decision=mode_decision,
                previous_mode=previous_mode,
                previous_submode=self._last_submode,
                turns_since_last_switch=self._turns_since_last_switch,
                emotion_score=emotion_score,
                desire_tier=desire_tier,
                recent_decisions=self._recent_decisions,
            )
            result = self.context_router.analyze(abstract_input.to_dict())

            # Apply model selection via adapter
            if self.enable_model_selector and self._model_config:
                model_map = self._model_config.get("model_map") or self._build_model_map()
                self.adapter.apply_model_selection(result, model_map=model_map)

            return result
        except Exception:
            return None

    # ─── Layer 5: Resolve Final Mode ────────────────────────────────────────

    def _resolve_final_mode(
        self,
        context_result: ContextRouteResult | None,
        mode_decision: ModeDecision,
        previous_mode: str | None,
        desire_tier: str | None,
    ) -> tuple[str, str | None, str | None, list[str]]:
        """Resolve the final production mode from context router decision.

        Strategy:
        - If context router is disabled/errored → use legacy
        - If context router has HIGH confidence decision → use it (DIRECT_SWITCH)
        - If context router is uncertain (hold/low_context) → fall back to legacy
        - Legacy submode is always preserved when legacy and context router agree on top mode

        Returns (mode, submode, reason, extra_safety_flags).
        """
        if context_result is None:
            # Fallback to legacy classifier
            return mode_decision.mode, mode_decision.submode, None, []

        top_mode = context_result.top_mode
        submode = context_result.work_submode or context_result.relationship_submode
        transition_type = context_result.transition_type

        # ─── Uncertain / hold → fall back to legacy ─────────────────────
        # Context router explicitly says "I'm not sure, hold current state"
        # In this case, trust the legacy classifier which has keyword matching
        if transition_type == "hold" and top_mode != "work":
            # Exception: if context router found a secondary_candidate, handle it
            if context_result.secondary_candidate in ("confirmed_intimacy", "intimacy_candidate"):
                # Legacy says sex → pass through as sex so TransitionManager handles gate
                if mode_decision.mode == MODE_SEX:
                    return MODE_SEX, mode_decision.submode, f"context_router:legacy_sex_confirmed|{','.join(context_result.reasons)}", []
                # Legacy says daily but context router sees intimacy candidate with restrained desire
                if desire_tier == "restrained":
                    return mode_decision.mode, mode_decision.submode, None, ["sex_desire_gate_restrained"]

            # Default hold: trust legacy
            return mode_decision.mode, mode_decision.submode, None, []

        # ─── Work override (high confidence) ────────────────────────────
        if top_mode == "work":
            # Preserve legacy's more specific submode when it also says work
            resolved_submode = mode_decision.submode if mode_decision.mode == MODE_WORK else submode
            return MODE_WORK, resolved_submode, f"context_router:{','.join(context_result.reasons)}", []

        # ─── Boundary / cooldown ────────────────────────────────────────
        if submode == "cooldown" or transition_type == "downgrade":
            if mode_decision.signals.get("explicit_task_request") or mode_decision.signals.get("explicit_system_request"):
                return MODE_WORK, mode_decision.submode, f"context_router:work_signal_over_boundary|{','.join(context_result.reasons)}", []
            # Explicit work/meta edits can quote crisis or adult boundary text;
            # do not let the abstract boundary router override the concrete
            # maintenance intent in those cases.
            if mode_decision.mode == MODE_WORK and (
                mode_decision.signals.get("explicit_task_request") or mode_decision.signals.get("explicit_system_request")
            ):
                return MODE_WORK, mode_decision.submode, f"context_router:legacy_work_over_boundary|{','.join(context_result.reasons)}", []
            # Preserve crisis semantics when legacy classifier detected crisis_guard
            legacy_flags = getattr(mode_decision, "safety_flags", []) or []
            resolved_sub = "crisis" if "crisis_guard" in legacy_flags else "cooldown"
            return MODE_DAILY, resolved_sub, f"context_router:{','.join(context_result.reasons)}", []

        # ─── Confirmed intimacy (high confidence switch) ────────────────
        if submode in ("confirmed_intimacy", "intimacy_candidate") and transition_type == "switch":
            # If legacy classifier says work, work takes priority over sex
            if mode_decision.mode == MODE_WORK:
                resolved_submode = mode_decision.submode
                return MODE_WORK, resolved_submode, f"context_router:{','.join(context_result.reasons)}", []
            # Context router can only confirm sex if legacy ALSO says sex
            # If legacy says daily, context router cannot unilaterally upgrade
            if mode_decision.mode == MODE_SEX:
                if desire_tier in ("ambivalent", "uninhibited") and self.enable_active_sex:
                    resolved_submode = mode_decision.submode
                    return MODE_SEX, resolved_submode, f"context_router:{','.join(context_result.reasons)}", []
                else:
                    # Desire gate blocks
                    extra_flags_block: list[str] = []
                    if desire_tier == "restrained":
                        extra_flags_block.append("sex_desire_gate_restrained")
                    elif not self.enable_active_sex:
                        extra_flags_block.append("sex_requires_gate")
                    else:
                        extra_flags_block.append("sex_requires_gate")
                    return MODE_DAILY, "affectionate", f"context_router:desire_gate_blocked|{','.join(context_result.reasons)}", extra_flags_block
            # Legacy says daily — trust legacy, don't upgrade to sex
            return mode_decision.mode, mode_decision.submode, None, []

        # ─── Default: trust legacy ──────────────────────────────────────
        return mode_decision.mode, mode_decision.submode, None, []

    # ─── Layer 8: Model Selector ────────────────────────────────────────────

    def _select_model(
        self,
        mode: str,
        submode: str | None,
        platform: str,
        emotion_score: float | None,
        context_result: ContextRouteResult | None,
    ) -> str | None:
        """Dynamic model selection based on mode/context. Returns model name or None.

        Delegates to ModelSelector instance if available, otherwise falls back
        to inline config lookup for backward compatibility.
        """
        if not self.enable_model_selector or not self._model_config:
            return None

        # Prefer ModelSelector class (has cooldown logic)
        if self.model_selector:
            return self.model_selector.select(
                mode=mode,
                submode=submode,
                platform=platform,
                emotion_score=emotion_score,
                context_result=context_result,
            )

        # Fallback: inline logic (backward compat if ModelSelector not instantiated)
        if context_result and context_result.selected_model:
            return context_result.selected_model

        mode_overrides = self._model_config.get("mode_overrides", {})
        mode_key = self._build_mode_key(mode, submode)
        if mode_key in mode_overrides and mode_overrides[mode_key]:
            return mode_overrides[mode_key]

        intensity = self._emotion_intensity(emotion_score)
        emotion_overrides = self._model_config.get("emotion_overrides", {})
        if intensity and intensity in emotion_overrides and emotion_overrides[intensity]:
            return emotion_overrides[intensity]

        platform_overrides = self._model_config.get("platform_overrides", {})
        if platform in platform_overrides and platform_overrides[platform]:
            return platform_overrides[platform]

        return None

    # ─── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _route_metadata(mode: str) -> dict[str, Any]:
        if mode == MODE_SEX:
            return {"hermes_route_bucket": "sex", "hermes_model_hint": "sex"}
        if mode == MODE_WORK:
            return {"hermes_route_bucket": "task", "hermes_model_hint": "technical"}
        if mode == MODE_DAILY:
            return {"hermes_route_bucket": "relationship", "hermes_model_hint": "default"}
        return {}

    def _decision_audit(
        self,
        *,
        previous_mode: str | None,
        mode_decision: ModeDecision,
        context_result: ContextRouteResult | None,
        transition,
        memory: MemorySelection,
        emotion_score: float | None,
        desire_tier: str | None,
        emotion_modifier: str,
        selected_layers: list[str],
        extra_safety_flags: list[str],
    ) -> dict[str, Any]:
        """Return read-only reason-code telemetry for cross-system audits.

        This is control-plane metadata only: it does not participate in mode,
        emotion, memory, or model decisions. Keep it compact and JSON-safe so
        dashboard/probe tools can explain why a turn selected a given runtime
        layer without scraping prose logs.
        """
        context_dict = context_result.to_dict() if context_result is not None else {}
        context_routing = context_dict.get("routing") or {}
        signals = context_dict.get("signals") or {}
        belief = context_dict.get("belief") or {}
        emotion_intensity = self._emotion_intensity(emotion_score)
        final_mode = getattr(transition, "active_mode", None)
        requested_mode = getattr(transition, "requested_mode", None)
        transition_reason = getattr(transition, "reason", "")
        classifier_reason = getattr(mode_decision, "reason", "")
        context_reasons = list(context_dict.get("reasons") or [])
        mode_overridden_by_context = bool(context_result is not None and requested_mode != mode_decision.mode)
        sex_gate_blocked = any(flag.startswith("sex_") for flag in (extra_safety_flags or []))
        work_override = final_mode == MODE_WORK and (mode_decision.mode != MODE_WORK or previous_mode == MODE_SEX)
        emotion_affects_mode = bool(desire_tier in {"ambivalent", "uninhibited"} and requested_mode == MODE_SEX)
        if sex_gate_blocked:
            emotion_affects_mode = True
        return {
            "previous_mode": previous_mode,
            "classifier": {
                "mode": mode_decision.mode,
                "submode": mode_decision.submode,
                "confidence": mode_decision.confidence,
                "reason": classifier_reason,
                "signals": dict(mode_decision.signals or {}),
                "safety_flags": list(mode_decision.safety_flags or []),
            },
            "context_router": {
                "enabled": context_result is not None,
                "top_mode": context_routing.get("top_mode"),
                "transition_type": context_routing.get("transition_type"),
                "confidence": context_routing.get("confidence"),
                "work_submode": context_routing.get("work_submode"),
                "relationship_submode": context_routing.get("relationship_submode"),
                "secondary_candidate": context_routing.get("secondary_candidate"),
                "reasons": context_reasons,
                "signals": signals,
                "belief": belief,
            },
            "transition": {
                "requested_mode": requested_mode,
                "active_mode": final_mode,
                "transition": getattr(transition, "transition", ""),
                "confidence": getattr(transition, "confidence", 0.0),
                "reason": transition_reason,
                "mode_overridden_by_context": mode_overridden_by_context,
                "work_override": work_override,
                "sex_gate_blocked": sex_gate_blocked,
            },
            "emotion_influence": {
                "emotion_score": emotion_score,
                "intensity": emotion_intensity,
                "desire_tier": desire_tier,
                "modifier_present": bool(emotion_modifier),
                "affects_tone": bool(emotion_modifier),
                "affects_desire": desire_tier not in {None, "unknown"},
                "affects_mode": emotion_affects_mode,
                "mode_authority": "gated_supporting_signal",
            },
            "memory_selection": {
                "profile": memory.profile,
                "candidate_files": list(memory.candidate_files),
                "layers": list(memory.layers),
                "buckets": list(memory.buckets),
                "budgets": dict(memory.budgets),
                "reason": memory.reason,
                "selected_soul_layers": list(selected_layers),
            },
            "reason_codes": [
                code for code in [
                    "context_router_enabled" if context_result is not None else "legacy_classifier_only",
                    "context_override" if mode_overridden_by_context else "classifier_preserved",
                    "work_override" if work_override else "no_work_override",
                    "sex_gate_blocked" if sex_gate_blocked else "sex_gate_not_blocked",
                    f"emotion_intensity:{emotion_intensity}" if emotion_intensity else "emotion_intensity:unknown",
                    f"desire:{desire_tier or 'unknown'}",
                    f"memory_profile:{memory.profile}",
                ]
            ],
        }

    def _update_anti_flap(self, current_mode: str, current_submode: str | None) -> None:
        """Track mode switches for anti-flap logic."""
        if self._last_top_mode is not None and current_mode != self._last_top_mode:
            self._turns_since_last_switch = 0
        else:
            self._turns_since_last_switch += 1
        self._last_top_mode = current_mode
        self._last_submode = current_submode

    def _build_model_map(self) -> dict[str, str]:
        """Build model map from config for adapter.apply_model_selection.

        Delegates to ModelSelector if available; otherwise falls back to inline logic.
        """
        if self.model_selector:
            return self.model_selector.build_model_map()
        # Fallback: inline (backward compat)
        if not self._model_config:
            return {}
        mode_overrides = self._model_config.get("mode_overrides", {})
        default = self._model_config.get("default_model", "")
        result: dict[str, str] = {"default": default}
        for key, model in mode_overrides.items():
            if model:
                if key == "active_layer":
                    result["relationship:confirmed_intimacy"] = model
                    result["relationship:intimacy_candidate"] = model
                elif key == "work":
                    result["work"] = model
                elif key == "daily":
                    result["relationship"] = model
                    result["relationship:daily"] = model
                    result["relationship:affectionate"] = model
        return result

    def _selected_layers(
        self,
        active_mode: str,
        safety_flags: list[str],
        previous_mode: str | None = None,
    ) -> list[str]:
        layers = [active_mode] if active_mode in self.ACTIVE_LAYER_MODES else [MODE_DAILY]
        if self.core_source == "orchestrator_core":
            layers = ["core"] + layers
        return layers

    @staticmethod
    def _reason_with_semantic_shadow(
        mode_reason: str,
        transition_reason: str,
        memory_reason: str,
        semantic_shadow: dict | None,
    ) -> str:
        parts = [mode_reason, transition_reason, memory_reason]
        if semantic_shadow:
            parts.append(
                "semantic_shadow="
                + str(semantic_shadow.get("primary_mode"))
            )
        return " | ".join(part for part in parts if part)

    @staticmethod
    def _extract_emotion_score(emotion_state: dict | None) -> float | None:
        if not emotion_state:
            return None
        for key in ("emotion_score", "current_emotion"):
            if key in emotion_state:
                try:
                    return float(emotion_state[key])
                except (TypeError, ValueError):
                    return None
        nested = emotion_state.get("emotion_state") if isinstance(emotion_state, dict) else None
        if isinstance(nested, dict):
            return StateOrchestrator._extract_emotion_score(nested)
        return None

    @staticmethod
    def _desire_tier(emotion_score: float | None) -> str:
        """Map emotion score to the sex desire gate.

        The production emotion system writes ``emotion_score`` on the modern
        [-5, +5] scale and uses 3.0 / 4.0 as the ambivalent / uninhibited
        thresholds.  Older state-machine fixtures used a percentage-style score
        (35 / 70).  Accept both ranges so the three-state machine can consume
        the new dynamic emotion state without silently treating score 4.x as
        restrained.
        """
        if emotion_score is None:
            return "unknown"
        if -5.0 <= emotion_score <= 5.0:
            if emotion_score < 3.0:
                return "restrained"
            if emotion_score < 4.0:
                return "ambivalent"
            return "uninhibited"
        if emotion_score < 35:
            return "restrained"
        if emotion_score < 70:
            return "ambivalent"
        return "uninhibited"

    @staticmethod
    def _build_mode_key(mode: str, submode: str | None) -> str:
        """Map mode+submode to config key."""
        if mode == MODE_WORK:
            return "work"
        if mode == MODE_SEX:
            return "active_layer"
        return "daily"

    @staticmethod
    def _emotion_intensity(emotion_score: float | None) -> str | None:
        """Map emotion score to intensity band for override lookup."""
        if emotion_score is None:
            return None
        if -5.0 <= emotion_score <= 5.0:
            abs_score = abs(emotion_score)
            if abs_score >= 4.5:
                return "overwhelming"
            if abs_score >= 3.0:
                return "intense"
            if abs_score >= 1.5:
                return "moderate"
            return "mild"
        # Legacy percentage-style score
        if emotion_score >= 90:
            return "overwhelming"
        if emotion_score >= 60:
            return "intense"
        if emotion_score >= 30:
            return "moderate"
        return "mild"

    @staticmethod
    def _merge_flags(*flag_lists: list[str]) -> list[str]:
        merged: list[str] = []
        for flags in flag_lists:
            for flag in flags:
                if flag not in merged:
                    merged.append(flag)
        return merged
