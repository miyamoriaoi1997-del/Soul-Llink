"""Deterministic intent-to-mode classifier for persona orchestration."""

from __future__ import annotations

import re

from .config_loader import load_routing_config
from .lexicon import SYSTEM_DOMAIN_TERMS, TASK_ACTION_TERMS, WORK_DOMAIN_TERMS, combined
from .types import (
    MODE_DAILY,
    MODE_SEX,
    MODE_WORK,
    ModeDecision,
)

_cfg = load_routing_config()


class ModeClassifier:
    """Classify the current user turn into a persona operating mode.

    The first version is deliberately deterministic and inspectable. It is
    meant for shadow-mode logging before it is trusted with active prompt
    injection.
    """

    # ── Term lists from config ─────────────────────────────────────────────
    SYSTEM_TERMS = SYSTEM_DOMAIN_TERMS
    ACTION_TERMS = TASK_ACTION_TERMS
    WORK_TERMS = combined(WORK_DOMAIN_TERMS)
    CRISIS_TERMS = _cfg.classifier_terms("crisis")
    CONFLICT_TERMS = _cfg.classifier_terms("conflict")
    SEX_TERMS = _cfg.classifier_terms("sex_explicit")
    SEX_LICK_CONTEXT_TERMS = _cfg.classifier_terms("sex_lick_context")
    SEX_HINT_TERMS = _cfg.classifier_terms("sex_hint")
    SEX_CLOSE_ACTION_TERMS = _cfg.classifier_terms("sex_close_action")
    SEX_CLOSE_NEGATION_TERMS = _cfg.classifier_terms("sex_close_negation")
    SEX_CONTINUE_TERMS = _cfg.classifier_terms("sex_continue")
    AFTERCARE_TERMS = _cfg.classifier_terms("aftercare")
    INTIMACY_TERMS = _cfg.classifier_terms("intimacy")
    CREATIVE_TERMS = _cfg.classifier_terms("creative")
    META_TERMS = _cfg.classifier_terms("meta")
    META_ACTION_TERMS = _cfg.classifier_terms("meta_action")

    # ── Intent rule sub-groups from config ─────────────────────────────────
    _INTENT_RULES = _cfg.intent_rule("meta_discussion")
    _PERSONA_SYSTEM_RULES = _cfg.intent_rule("persona_system_specific")
    _FINANCIAL_RULES = _cfg.intent_rule("financial_trading")
    _RESTORE_RULES = _cfg.intent_rule("restore_or_missing")
    _HANDOFF_RULES = _cfg.intent_rule("external_handoff")
    _CONTEXTUAL_WORK_EXIT = _cfg.intent_terms("contextual_work_exit")
    _CREDENTIAL_TERMS = _cfg.intent_terms("credential_sensitive")
    _CREATIVE_GEN_ACTIONS = _cfg.intent_terms("creative_generation_actions")
    _CREATIVE_REVIEW_VERBS = _cfg.intent_terms("creative_review_verbs")
    _CREATIVE_REVIEW_EXCLUSION = _cfg.intent_terms("creative_review_exclusion")
    _MACHINE_STATUS_TERMS = _cfg.intent_terms("machine_status")
    _AFFECTIONATE_ADDRESS = _cfg.intent_terms("affectionate_address")
    _RUNTIME_ARTIFACT = _cfg.intent_terms("runtime_artifact")
    _RESTART_CONTEXT = _cfg.intent_terms("contextual_restart_context")

    def classify(
        self,
        user_message: str = "",
        *,
        recent_messages: list | None = None,
        emotion_state: dict | None = None,
        platform: str = "unknown",
    ) -> ModeDecision:
        raw = user_message
        text = self._normalize(raw)

        # ── Priority 1: Meta discussion about the system ──────────────────
        if self._is_meta_discussion(text):
            return self._decision(
                mode=MODE_WORK,
                submode="meta_discussion",
                confidence=_cfg.conf("meta_discussion"),
                reason="meta/system classifier discussion matched",
                safety_flags=["meta_discussion"],
                platform=platform,
                text=text,
            )

        # ── Priority 2: Crisis ─────────────────────────────────────────────
        if self._contains_any(text, self.CRISIS_TERMS):
            return self._decision(
                mode=MODE_DAILY,
                submode="crisis",
                confidence=_cfg.conf("crisis"),
                reason="crisis language detected — routing to daily for safety",
                safety_flags=["crisis_guard"],
                platform=platform,
                text=text,
            )

        # ── Priority 3: Relationship conflict ─────────────────────────────
        if self._contains_any(text, self.CONFLICT_TERMS):
            return self._decision(
                mode=MODE_DAILY,
                submode="relationship_rupture",
                confidence=_cfg.conf("relationship_rupture"),
                reason="relationship conflict marker matched",
                safety_flags=["relationship_conflict"],
                platform=platform,
                text=text,
            )

        # ── Priority 4: Persona system specific ───────────────────────────
        if self._has_persona_system_specific_intent(text):
            return self._decision(
                mode=MODE_WORK,
                submode="system",
                confidence=_cfg.conf("persona_system"),
                reason="persona/runtime system-specific intent matched",
                platform=platform,
                text=text,
            )

        # ── Priority 3: Creative generation ───────────────────────────────
        if self._is_creative_generation(text):
            return self._decision(
                mode=MODE_WORK,
                submode="creative",
                confidence=_cfg.conf("creative_generation"),
                reason="creative generation phrase matched",
                platform=platform,
                text=text,
            )

        # ── Priority 3b: Creative review ──────────────────────────────────
        if self._is_creative_review(text):
            return self._decision(
                mode=MODE_WORK,
                submode="creative_review",
                confidence=_cfg.conf("creative_review"),
                reason="creative review task detected",
                platform=platform,
                text=text,
            )

        # ── Priority 4: Financial/trading task ────────────────────────────
        if self._is_financial_or_trading_task(text):
            return self._decision(
                mode=MODE_WORK,
                submode="financial_trading",
                confidence=_cfg.conf("financial_trading"),
                reason="financial/trading task detected",
                platform=platform,
                text=text,
            )

        # ── Priority 5: System maintenance ────────────────────────────────
        if self._has_system_maintenance_intent(text):
            return self._decision(
                mode=MODE_WORK,
                submode="system_maintenance",
                confidence=_cfg.conf("system_maintenance"),
                reason="system maintenance intent detected",
                platform=platform,
                text=text,
            )

        # ── Priority 6: Explicit sex ──────────────────────────────────────
        if self._contains_any(text, self.SEX_TERMS) or self._has_explicit_lick_progression(text):
            return self._decision(
                mode=MODE_SEX,
                submode="explicit_progression",
                confidence=_cfg.conf("explicit_sex"),
                reason="explicit sex progression marker matched",
                safety_flags=["sex_requires_gate"],
                platform=platform,
                text=text,
            )

        # ── Priority 6b: Sex hint ─────────────────────────────────────────
        if self._contains_any(text, self.SEX_HINT_TERMS):
            return self._decision(
                mode=MODE_SEX,
                submode="hint_progression",
                confidence=_cfg.conf("sex_hint"),
                reason="sex-scene hint progression marker matched",
                safety_flags=["sex_requires_gate"],
                platform=platform,
                text=text,
            )

        # ── Priority 7: Technical work ────────────────────────────────────
        if self._contains_any(text, self.WORK_TERMS) or self._is_contextual_work_exit(text):
            return self._decision(
                mode=MODE_WORK,
                submode="technical",
                confidence=_cfg.conf("technical_work"),
                reason="technical work marker matched",
                platform=platform,
                text=text,
            )

        # ── Priority 8: Intimacy ──────────────────────────────────────────
        if self._contains_any(text, self.INTIMACY_TERMS):
            return self._decision(
                mode=MODE_DAILY,
                submode="relationship_closeness",
                confidence=_cfg.conf("intimacy"),
                reason="intimacy marker matched without explicit sex progression",
                platform=platform,
                text=text,
            )

        # ── Priority 9: Creative task (lower confidence) ──────────────────
        if self._contains_any(text, self.CREATIVE_TERMS):
            return self._decision(
                mode=MODE_WORK,
                submode="creative",
                confidence=_cfg.conf("creative_task"),
                reason="creative task marker matched",
                platform=platform,
                text=text,
            )

        # ── Default: daily (principle-driven fallback) ────────────────────
        base_conf = _cfg.conf("default_daily")
        fallback_signals: dict = {"message_length": len(raw), "normalized_text": text}

        # Contextual reference → likely refers to prior work context
        if self._contains_any(text, _cfg.fallback_terms("contextual_reference")):
            base_conf += _cfg.fallback_boost("contextual_boost")
            fallback_signals["contextual_reference"] = True

        # Question intent → likely asking about something in progress
        if self._contains_any(text, _cfg.fallback_terms("question_markers")):
            base_conf += _cfg.fallback_boost("question_boost")
            fallback_signals["question_intent"] = True

        # Emotional tone → likely relationship/daily content
        if self._contains_any(text, _cfg.fallback_terms("emotional_markers")):
            fallback_signals["emotional_tone"] = True

        return self._decision(
            mode=MODE_DAILY,
            submode="default",
            confidence=min(base_conf, 0.64),  # cap below anti-flap threshold
            reason="no higher-priority mode markers matched",
            platform=platform,
            text=text,
            signals=fallback_signals,
        )

    def _decision(
        self,
        mode: str,
        submode: str,
        confidence: float,
        reason: str,
        platform: str,
        text: str,
        safety_flags: list[str] | None = None,
        signals: dict | None = None,
    ) -> ModeDecision:
        signals = {"platform": platform, **(signals or {})}
        signals.setdefault("message_length", len(text))
        signals.setdefault("normalized_text", text)
        self._add_intent_signals(text, signals)
        safety_flags = list(safety_flags or [])
        if self._has_credential_sensitive_text(text) and "credential_sensitive" not in safety_flags:
            safety_flags.append("credential_sensitive")
        secondary_modes: list[dict] = []
        return ModeDecision(
            mode=mode,
            submode=submode,
            confidence=confidence,
            reason=reason,
            safety_flags=safety_flags,
            signals=signals,
            secondary_modes=secondary_modes,
        )

    # ── Intent detection methods (now config-driven) ───────────────────────

    def _is_meta_discussion(self, text: str) -> bool:
        has_meta = self._contains_any(text, self.META_TERMS)
        has_action = self._contains_any(text, self.META_ACTION_TERMS)
        rules = self._INTENT_RULES
        quoted_sensitive_context = rules.get("quoted_sensitive_context", []) if isinstance(rules, dict) else []
        plain_test_negation = rules.get("plain_test_negation", []) if isinstance(rules, dict) else []
        quoted_sensitive = self._contains_any(text, self.SEX_TERMS + self.CONFLICT_TERMS + self.CRISIS_TERMS) and self._contains_any(
            text, quoted_sensitive_context,
        )
        plain_test_task = "测试" in text and not self._contains_any(text, plain_test_negation + self.SEX_TERMS + self.CONFLICT_TERMS)
        return has_meta and not plain_test_task and (has_action or quoted_sensitive)

    def _has_system_maintenance_intent(self, text: str) -> bool:
        has_contextual_restart = "重启" in text and self._contains_any(text, self._RESTART_CONTEXT)
        has_runtime_artifact = self._contains_any(text, self._RUNTIME_ARTIFACT)
        return (
            self._has_restore_or_missing_task_intent(text)
            or self._has_external_handoff_integration_intent(text)
            or self._has_persona_system_specific_intent(text)
            or has_runtime_artifact
            or (self._contains_any(text, self.SYSTEM_TERMS) or has_contextual_restart)
            and (
                self._contains_any(text, self.ACTION_TERMS)
                or has_contextual_restart
            )
        )

    def _has_persona_system_specific_intent(self, text: str) -> bool:
        rules = self._PERSONA_SYSTEM_RULES
        if not isinstance(rules, dict):
            return False
        if self._contains_any(text, rules.get("value_queries", [])):
            return True
        if self._contains_any(text, rules.get("model_routing", [])):
            return True
        if self._contains_any(text, rules.get("full_system", [])):
            return True
        if self._contains_any(text, rules.get("governance_subjects", [])) and self._contains_any(
            text, rules.get("governance_actions", []),
        ):
            return True
        return False

    def _has_affectionate_address(self, text: str) -> bool:
        return self._contains_any(text, self._AFFECTIONATE_ADDRESS)

    def _is_financial_or_trading_task(self, text: str) -> bool:
        rules = self._FINANCIAL_RULES
        if not isinstance(rules, dict):
            return False
        domain_terms = rules.get("domain", [])
        task_terms = rules.get("task", [])
        exploratory_negation = rules.get("exploratory_negation", [])
        if self._contains_any(text, exploratory_negation) and self._contains_any(text, self.SYSTEM_TERMS + self.META_TERMS):
            return False
        return self._contains_any(text, domain_terms) and self._contains_any(text, task_terms)

    def _has_restore_or_missing_task_intent(self, text: str) -> bool:
        rules = self._RESTORE_RULES
        if not isinstance(rules, dict):
            return False
        return (
            self._contains_any(text, rules.get("loss", []))
            and self._contains_any(text, rules.get("object", []))
            and self._contains_any(text, rules.get("action", []))
        )

    def _has_external_handoff_integration_intent(self, text: str) -> bool:
        rules = self._HANDOFF_RULES
        if not isinstance(rules, dict):
            return False
        return self._contains_any(text, rules.get("handoff", [])) and self._contains_any(
            text, rules.get("operation", []) + rules.get("directness", []),
        )

    def _is_contextual_work_exit(self, text: str) -> bool:
        return self._contains_any(text, self._CONTEXTUAL_WORK_EXIT)

    def _has_credential_sensitive_text(self, text: str) -> bool:
        return self._contains_any(text, self._CREDENTIAL_TERMS)

    def _add_intent_signals(self, text: str, signals: dict) -> None:
        explicit_system = self._has_system_maintenance_intent(text) or self._is_meta_discussion(text)
        explicit_task = self._is_financial_or_trading_task(text) or self._contains_any(text, self.WORK_TERMS)
        if explicit_system:
            signals["explicit_system_request"] = True
        if explicit_task:
            signals["explicit_task_request"] = True
        if self._has_sex_scene_close_intent(text):
            signals["sex_scene_close"] = True
            if self._has_aftercare_request(text):
                signals["aftercare_request"] = True
        if self._has_sex_scene_continue_intent(text):
            signals["sex_scene_continue"] = True
        if self._is_contextual_work_exit(text):
            signals["explicit_task_request"] = True
        if self._has_aftercare_request(text):
            signals["aftercare_request"] = True
        if self._contains_any(text, self.CRISIS_TERMS):
            signals["safety_stop"] = True

    def _has_explicit_lick_progression(self, text: str) -> bool:
        if "舔" not in text:
            return False
        return self._contains_any(text, self.SEX_LICK_CONTEXT_TERMS)

    def _has_sex_scene_close_intent(self, text: str) -> bool:
        if not text or self._contains_any(text, self.SEX_CLOSE_NEGATION_TERMS):
            return False
        return self._contains_any(text, self.SEX_CLOSE_ACTION_TERMS)

    def _has_sex_scene_continue_intent(self, text: str) -> bool:
        if not text:
            return False
        if self._contains_any(text, self.SEX_CLOSE_NEGATION_TERMS):
            return True
        return self._contains_any(text, self.SEX_CONTINUE_TERMS + self.SEX_TERMS + self.SEX_HINT_TERMS)

    def _has_aftercare_request(self, text: str) -> bool:
        return self._contains_any(text, self.AFTERCARE_TERMS)

    def _is_creative_generation(self, text: str) -> bool:
        return self._contains_any(text, self.CREATIVE_TERMS) and self._contains_any(
            text, self._CREATIVE_GEN_ACTIONS,
        )

    def _is_creative_review(self, text: str) -> bool:
        return self._contains_any(text, self._CREATIVE_REVIEW_VERBS) and not self._contains_any(
            text, self._CREATIVE_REVIEW_EXCLUSION,
        )

    @staticmethod
    def _normalize(text: str) -> str:
        normalized = re.sub(r"\s+", "", (text or "").lower())
        # Preserve spaces inside public placeholders such as [pet name].
        normalized = normalized.replace("[petname]", "[pet name]")
        normalized = normalized.replace("[assistantname]", "[assistant name]")
        normalized = normalized.replace("[usertitle]", "[user title]")
        return normalized

    @staticmethod
    def _contains_any(text: str, terms: list[str]) -> bool:
        compact_text = re.sub(r"\s+", "", text)
        return any(
            term.lower() in text or re.sub(r"\s+", "", term.lower()) in compact_text
            for term in terms
        )
