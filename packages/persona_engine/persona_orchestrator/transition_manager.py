"""Mode transition rules for persona orchestration."""

from __future__ import annotations

from .config_loader import load_routing_config
from .lexicon import CONTEXT_ACTION_TERMS, CONTEXT_CONTINUATION_TERMS, CONTEXT_QUESTION_TERMS, combined
from .types import MODE_DAILY, MODE_SEX, MODE_WORK, ModeDecision, TransitionDecision

_cfg = load_routing_config()


class TransitionManager:
    """Apply anti-flap, safety, and desire-gate rules to three canonical modes."""

    _ANTI_FLAP_CONFIDENCE = _cfg.threshold("anti_flap_confidence", 0.65)
    _WORK_OVERRIDE_CONFIDENCE = _cfg.threshold("work_override_confidence", 0.75)
    _WORK_EXIT_CONFIDENCE = _cfg.threshold("work_exit_confidence", 0.75)
    _SHORT_MESSAGE_MAX_CHARS = int(_cfg.threshold("short_message_max_chars", 4))
    _MACHINE_STATUS_TERMS = _cfg.intent_terms("machine_status")
    _WORK_CONTINUATION_EXIT_TERMS = _cfg.intent_terms("contextual_work_exit")

    def transition(
        self,
        previous_mode: str | None,
        decision: ModeDecision,
        safety_flags: list[str] | None = None,
        desire_tier: str | None = None,
        enable_active_sex: bool = False,
        emotion_score: float | None = None,
    ) -> TransitionDecision:
        flags = self._merge_flags(decision.safety_flags, safety_flags or [])
        requested = decision.mode if decision.mode in {MODE_DAILY, MODE_WORK, MODE_SEX} else MODE_DAILY
        previous = previous_mode if previous_mode in {MODE_DAILY, MODE_WORK, MODE_SEX} else None

        if "crisis_guard" in flags and not (
            decision.mode == MODE_WORK
            and (decision.signals.get("explicit_task_request") or decision.signals.get("explicit_system_request"))
        ):
            return TransitionDecision(
                previous_mode=previous_mode,
                requested_mode=requested,
                active_mode=MODE_DAILY,
                transition=self._transition_label(previous, MODE_DAILY),
                confidence=decision.confidence,
                reason="crisis_guard keeps relationship response in daily mode",
                safety_flags=flags,
            )

        if requested == MODE_SEX:
            if (
                previous != MODE_SEX
                and decision.submode == "hint_progression"
                and not decision.signals.get("sex_scene_continue")
            ):
                return TransitionDecision(
                    previous_mode=previous_mode,
                    requested_mode=requested,
                    active_mode=previous or MODE_DAILY,
                    transition="hold_sex_candidate",
                    confidence=decision.confidence,
                    reason="sex hint requires explicit progression evidence before mode entry",
                    safety_flags=self._merge_flags(flags, ["sex_transition_confirmation_pending"]),
                )
            if (
                previous == MODE_WORK
                and decision.submode == "hint_progression"
                and desire_tier in {"ambivalent", "uninhibited"}
            ):
                return TransitionDecision(
                    previous_mode=previous_mode,
                    requested_mode=requested,
                    active_mode=MODE_WORK,
                    transition="hold_context_action",
                    confidence=decision.confidence,
                    reason="ambiguous sex hint cannot override an active work context without explicit progression",
                    safety_flags=self._merge_flags(flags, ["ambiguous_hint_held_by_work_context"]),
                )
            if previous == MODE_SEX and (
                decision.signals.get("sex_scene_continue") or self._is_context_continuation_message(decision)
            ):
                return TransitionDecision(
                    previous_mode=previous_mode,
                    requested_mode=requested,
                    active_mode=MODE_SEX,
                    transition="hold_sex_continue",
                    confidence=decision.confidence,
                    reason="sex_scene_continue signal holds existing sex mode",
                    safety_flags=flags,
                )
            if enable_active_sex and desire_tier in {"uninhibited", "ambivalent"}:
                return TransitionDecision(
                    previous_mode=previous_mode,
                    requested_mode=requested,
                    active_mode=MODE_SEX,
                    transition=self._transition_label(previous, MODE_SEX),
                    confidence=decision.confidence,
                    reason="sex mode accepted after desire gate",
                    safety_flags=flags,
                )
            gate_flag = "sex_desire_gate_restrained" if desire_tier == "restrained" else "sex_shadow_only"
            return TransitionDecision(
                previous_mode=previous_mode,
                requested_mode=requested,
                active_mode=MODE_DAILY,
                transition=f"blocked:{gate_flag}",
                confidence=decision.confidence,
                reason=f"blocked:{gate_flag}; sex request remains daily",
                safety_flags=self._merge_flags(flags, [gate_flag]),
            )

        if previous is None:
            return TransitionDecision(
                previous_mode=previous_mode,
                requested_mode=requested,
                active_mode=requested,
                transition=f"start:{requested}",
                confidence=decision.confidence,
                reason=decision.reason,
                safety_flags=flags,
            )

        if previous == MODE_SEX:
            if self._is_explicit_task_request(decision, requested):
                return TransitionDecision(
                    previous_mode=previous_mode,
                    requested_mode=requested,
                    active_mode=MODE_WORK,
                    transition=self._transition_label(previous, MODE_WORK),
                    confidence=decision.confidence,
                    reason="explicit work request exits sex mode",
                    safety_flags=flags,
                )
            if (
                requested == MODE_DAILY
                and not decision.signals.get("sex_scene_continue")
                and self._is_work_continuation_exit_message(decision)
            ):
                return TransitionDecision(
                    previous_mode=previous_mode,
                    requested_mode=requested,
                    active_mode=MODE_WORK,
                    transition="sex->work_context_continuation",
                    confidence=decision.confidence,
                    reason="work-context continuation exits sex mode after task handoff",
                    safety_flags=flags,
                )
            if decision.signals.get("sex_scene_close") and (
                decision.signals.get("aftercare_request") or not decision.signals.get("sex_scene_continue")
            ):
                return TransitionDecision(
                    previous_mode=previous_mode,
                    requested_mode=requested,
                    active_mode=MODE_DAILY,
                    transition=self._transition_label(previous, MODE_DAILY),
                    confidence=decision.confidence,
                    reason="sex_scene_close signal exits sex mode",
                    safety_flags=flags,
                )
            if requested == MODE_DAILY:
                return TransitionDecision(
                    previous_mode=previous_mode,
                    requested_mode=requested,
                    active_mode=MODE_SEX,
                    transition="hold_sex_continuation",
                    confidence=decision.confidence,
                    reason="sex mode held until explicit work request, work-context continuation, or close signal",
                    safety_flags=flags,
                )

        if (
            requested == MODE_DAILY
            and decision.confidence < self._ANTI_FLAP_CONFIDENCE
            and previous != MODE_SEX
            and self._is_short_hold_message(decision)
            and not (previous == MODE_WORK and self._is_context_continuation_message(decision))
        ):
            return TransitionDecision(
                previous_mode=previous_mode,
                requested_mode=requested,
                active_mode=previous,
                transition="hold_short_message",
                confidence=decision.confidence,
                reason="short neutral continuation inherits previous mode",
                safety_flags=flags,
            )

        if requested == MODE_DAILY and previous == MODE_WORK and decision.confidence < self._ANTI_FLAP_CONFIDENCE and self._is_context_continuation_message(decision):
            if self._is_context_action_message(decision):
                label = "hold_context_action"
            elif self._is_context_question_message(decision):
                label = "hold_context_question"
            else:
                label = "hold_short_message"
            return TransitionDecision(
                previous_mode=previous_mode,
                requested_mode=requested,
                active_mode=MODE_WORK,
                transition=label,
                confidence=decision.confidence,
                reason="contextual continuation inherits previous work mode",
                safety_flags=flags,
            )

        if requested == MODE_WORK and (
            decision.signals.get("explicit_task_request")
            or decision.signals.get("explicit_system_request")
            or decision.confidence >= self._WORK_OVERRIDE_CONFIDENCE
        ):
            return TransitionDecision(
                previous_mode=previous_mode,
                requested_mode=requested,
                active_mode=MODE_WORK,
                transition=self._transition_label(previous, MODE_WORK),
                confidence=decision.confidence,
                reason="high-confidence work mode overrides previous mode",
                safety_flags=flags,
            )

        if previous == MODE_WORK and requested == MODE_DAILY:
            if decision.signals.get("semantic_explicit_daily_intent"):
                return TransitionDecision(
                    previous_mode=previous_mode,
                    requested_mode=requested,
                    active_mode=MODE_DAILY,
                    transition=self._transition_label(previous, MODE_DAILY),
                    confidence=decision.confidence,
                    reason="high-confidence semantic daily intent releases work context",
                    safety_flags=flags,
                )
            if decision.signals.get("context_continuation_exhausted"):
                return TransitionDecision(
                    previous_mode=previous_mode,
                    requested_mode=requested,
                    active_mode=MODE_DAILY,
                    transition=self._transition_label(previous, MODE_DAILY),
                    confidence=decision.confidence,
                    reason="bounded ambiguous work continuation expired",
                    safety_flags=flags,
                )
            if (
                decision.signals.get("sex_scene_close")
                or decision.signals.get("safety_stop")
                or decision.signals.get("aftercare_request")
            ):
                return TransitionDecision(
                    previous_mode=previous_mode,
                    requested_mode=requested,
                    active_mode=MODE_DAILY,
                    transition=self._transition_label(previous, MODE_DAILY),
                    confidence=decision.confidence,
                    reason="explicit boundary or aftercare signal exits work context immediately",
                    safety_flags=flags,
                )
            if self._is_machine_status_output(decision):
                return TransitionDecision(
                    previous_mode=previous_mode,
                    requested_mode=requested,
                    active_mode=MODE_WORK,
                    transition="hold_low_confidence",
                    confidence=decision.confidence,
                    reason="machine/status output inherits previous work mode",
                    safety_flags=flags,
                )
            has_release_signal = bool(
                decision.signals.get("aftercare_request")
                or decision.signals.get("sex_scene_close")
                or decision.signals.get("safety_stop")
                or decision.submode in {"relationship_closeness", "relationship_rupture", "crisis"}
            )
            if decision.confidence < self._WORK_EXIT_CONFIDENCE or not has_release_signal:
                return TransitionDecision(
                    previous_mode=previous_mode,
                    requested_mode=requested,
                    active_mode=MODE_WORK,
                    transition="hold_work_context",
                    confidence=decision.confidence,
                    reason="daily candidate below configured work-exit confidence; work context held",
                    safety_flags=flags,
                )
            return TransitionDecision(
                previous_mode=previous_mode,
                requested_mode=requested,
                active_mode=MODE_DAILY,
                transition=self._transition_label(previous, MODE_DAILY),
                confidence=decision.confidence,
                reason="work mode does not hold over relationship/daily mode",
                safety_flags=flags,
            )

        if decision.confidence < self._ANTI_FLAP_CONFIDENCE and previous != MODE_SEX:
            return TransitionDecision(
                previous_mode=previous_mode,
                requested_mode=requested,
                active_mode=previous,
                transition="hold_low_confidence",
                confidence=decision.confidence,
                reason="confidence below 0.65; previous mode held to avoid flapping",
                safety_flags=flags,
            )

        return TransitionDecision(
            previous_mode=previous_mode,
            requested_mode=requested,
            active_mode=requested,
            transition=self._transition_label(previous, requested),
            confidence=decision.confidence,
            reason=decision.reason,
            safety_flags=flags,
        )

    @classmethod
    def _is_short_hold_message(cls, decision: ModeDecision) -> bool:
        message_length = decision.signals.get("message_length")
        try:
            return int(message_length) <= cls._SHORT_MESSAGE_MAX_CHARS
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _is_context_continuation_message(decision: ModeDecision) -> bool:
        text = str(decision.signals.get("normalized_text") or "")
        if not text:
            return False
        return any(term in text for term in combined(CONTEXT_CONTINUATION_TERMS, CONTEXT_QUESTION_TERMS, CONTEXT_ACTION_TERMS))

    @classmethod
    def _is_work_continuation_exit_message(cls, decision: ModeDecision) -> bool:
        text = str(decision.signals.get("normalized_text") or "")
        if not text:
            return False
        if any(term in text for term in CONTEXT_QUESTION_TERMS):
            return True
        return any(term in text for term in cls._WORK_CONTINUATION_EXIT_TERMS)

    @staticmethod
    def _is_context_action_message(decision: ModeDecision) -> bool:
        text = str(decision.signals.get("normalized_text") or "")
        return any(term in text for term in CONTEXT_ACTION_TERMS)

    @staticmethod
    def _is_context_question_message(decision: ModeDecision) -> bool:
        text = str(decision.signals.get("normalized_text") or "")
        return any(term in text for term in CONTEXT_QUESTION_TERMS)

    @classmethod
    def _is_machine_status_output(cls, decision: ModeDecision) -> bool:
        text = str(decision.signals.get("normalized_text") or "")
        if not text:
            return False
        return any(term in text for term in cls._MACHINE_STATUS_TERMS)

    @classmethod
    def _is_explicit_task_request(cls, decision: ModeDecision, requested: str) -> bool:
        if requested == MODE_WORK and (decision.signals.get("explicit_task_request") or decision.signals.get("explicit_system_request")):
            return True
        return requested == MODE_WORK and decision.confidence >= cls._WORK_OVERRIDE_CONFIDENCE

    @staticmethod
    def _transition_label(previous_mode: str | None, active_mode: str) -> str:
        if previous_mode is None:
            return f"start:{active_mode}"
        if previous_mode == active_mode:
            return f"stay:{active_mode}"
        return f"{previous_mode}->{active_mode}"

    @staticmethod
    def _merge_flags(*flag_lists: list[str]) -> list[str]:
        merged: list[str] = []
        for flags in flag_lists:
            for flag in flags:
                if flag not in merged:
                    merged.append(flag)
        return merged
