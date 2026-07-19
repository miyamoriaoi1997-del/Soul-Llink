"""Compile and validate routing rules from YAML configuration.

Converts raw YAML term lists into validated CompiledRule objects with
full metadata. Performs static analysis to detect conflicts, dead rules,
and configuration errors.
"""
from __future__ import annotations

from typing import Any

from .config_loader import RoutingConfig, load_routing_config
from .rule_schema import (
    CompiledRule,
    FeatureClass,
    MatchScope,
    Mode,
    NegationPolicy,
    QuotePolicy,
    RuleTerm,
    validate_feature_class,
    validate_mode,
    validate_priority,
    validate_weight,
)


class RuleCompilationError(Exception):
    """Raised when rule configuration is invalid."""
    pass


class RuleCompiler:
    """Compile YAML configuration into validated typed rules."""

    def __init__(self, config: RoutingConfig | None = None):
        self.config = config or load_routing_config()
        self._compiled_rules: list[CompiledRule] = []
        self._rule_ids: set[str] = set()

    def compile_classifier_group(
        self,
        group_name: str,
        *,
        feature_class: FeatureClass,
        candidate_mode: Mode | None,
        default_weight: float = 1.0,
        default_priority: int = 500,
        match_any: bool = True,
    ) -> list[CompiledRule]:
        """Compile a classifier term group into typed rules.

        Args:
            group_name: Config key like 'crisis', 'sex_explicit', etc.
            feature_class: Semantic category for evidence
            candidate_mode: Mode this evidence supports (None for veto-only)
            default_weight: Weight when not specified in term
            default_priority: Priority when not specified
            match_any: True = any term matches, False = all terms required

        Returns:
            List of compiled rules (typically one rule per group)
        """
        raw_terms = self.config.classifier_terms_typed(group_name)
        if not raw_terms:
            return []

        # Build rule ID from group name
        rule_id = f"CLASSIFIER.{group_name.upper()}.001"

        # Parse terms with metadata
        compiled_terms: list[RuleTerm] = []
        for item in raw_terms:
            if isinstance(item, str):
                compiled_terms.append(RuleTerm(term=item, weight=default_weight))
            elif isinstance(item, dict):
                term_str = item.get("term", "")
                if not term_str:
                    raise RuleCompilationError(
                        f"Rule {rule_id}: dict term missing 'term' field"
                    )

                weight = item.get("weight", default_weight)
                try:
                    validate_weight(weight, context=rule_id)
                except (ValueError, TypeError) as e:
                    raise RuleCompilationError(str(e)) from e

                requires_ctx = item.get("requires_context", [])
                forbids_ctx = item.get("forbids_context", [])

                compiled_terms.append(
                    RuleTerm(
                        term=term_str,
                        weight=weight,
                        requires_context=tuple(requires_ctx) if requires_ctx else (),
                        forbids_context=tuple(forbids_ctx) if forbids_ctx else (),
                    )
                )
            else:
                raise RuleCompilationError(
                    f"Rule {rule_id}: invalid term type {type(item)}"
                )

        if not compiled_terms:
            return []

        # Create single rule for this group
        rule = CompiledRule(
            rule_id=rule_id,
            feature_class=feature_class,
            candidate_mode=candidate_mode,
            terms=tuple(compiled_terms),
            weight=default_weight,
            priority=default_priority,
            match_any=match_any,
            activation_group=group_name,
        )

        self._register_rule(rule)
        return [rule]

    def compile_all_legacy_groups(self) -> list[CompiledRule]:
        """Compile all existing classifier groups into typed rules.

        This creates parity with the current mode_classifier.py structure.
        Each classifier term group becomes one CompiledRule.
        """
        rules: list[CompiledRule] = []

        # Hard boundary rules (priority 1000)
        rules.extend(
            self.compile_classifier_group(
                "crisis",
                feature_class=FeatureClass.HARD_BOUNDARY,
                candidate_mode=Mode.DAILY,
                default_priority=1000,
                default_weight=0.90,
            )
        )

        # Explicit meta (priority 900)
        # Meta discussion typically routes to WORK for system configuration

        # Explicit task (priority 800)
        # These would be compiled from intent_rules in full implementation

        # Protected progression (priority 700)
        rules.extend(
            self.compile_classifier_group(
                "sex_explicit",
                feature_class=FeatureClass.PROTECTED_PROGRESSION,
                candidate_mode=Mode.SEX,
                default_priority=700,
                default_weight=0.90,
            )
        )

        rules.extend(
            self.compile_classifier_group(
                "sex_hint",
                feature_class=FeatureClass.PROTECTED_PROGRESSION,
                candidate_mode=Mode.SEX,
                default_priority=650,
                default_weight=0.82,
            )
        )

        # Relationship evidence (priority 600)
        rules.extend(
            self.compile_classifier_group(
                "intimacy",
                feature_class=FeatureClass.RELATIONSHIP,
                candidate_mode=Mode.DAILY,
                default_priority=600,
                default_weight=0.78,
            )
        )

        rules.extend(
            self.compile_classifier_group(
                "aftercare",
                feature_class=FeatureClass.RELATIONSHIP,
                candidate_mode=Mode.DAILY,
                default_priority=600,
                default_weight=0.85,
            )
        )

        # Continuation/context (priority 500)
        rules.extend(
            self.compile_classifier_group(
                "sex_continue",
                feature_class=FeatureClass.CONTINUATION_BINDING,
                candidate_mode=Mode.SEX,
                default_priority=500,
                default_weight=0.75,
            )
        )

        return rules

    def _register_rule(self, rule: CompiledRule) -> None:
        """Register a rule and check for ID conflicts."""
        if rule.rule_id in self._rule_ids:
            raise RuleCompilationError(f"Duplicate rule ID: {rule.rule_id}")
        self._rule_ids.add(rule.rule_id)
        self._compiled_rules.append(rule)

    def get_all_rules(self) -> list[CompiledRule]:
        """Get all compiled rules."""
        return self._compiled_rules.copy()

    def validate_config(self) -> list[str]:
        """Run static validation on compiled rules.

        Returns list of warning/error messages.
        """
        issues: list[str] = []

        # Check for duplicate IDs (already enforced during compilation)

        # Check for rules with identical conditions
        seen_conditions: dict[tuple, str] = {}
        for rule in self._compiled_rules:
            # Create hashable condition key
            condition_key = (
                rule.feature_class,
                rule.activation_group,
                frozenset(term.term for term in rule.terms),
            )
            if condition_key in seen_conditions:
                issues.append(
                    f"Rules {seen_conditions[condition_key]} and {rule.rule_id} "
                    f"have identical conditions"
                )
            else:
                seen_conditions[condition_key] = rule.rule_id

        # Check for unreachable rules (shadowed by higher priority)
        grouped: dict[str, list[CompiledRule]] = {}
        for rule in self._compiled_rules:
            group = rule.activation_group or "_default"
            if group not in grouped:
                grouped[group] = []
            grouped[group].append(rule)

        for group, group_rules in grouped.items():
            # Sort by priority descending
            sorted_rules = sorted(group_rules, key=lambda r: r.priority, reverse=True)
            for i, rule in enumerate(sorted_rules):
                for higher_rule in sorted_rules[:i]:
                    # Check if higher_rule completely shadows this rule
                    if self._rule_shadows(higher_rule, rule):
                        issues.append(
                            f"Rule {rule.rule_id} is shadowed by {higher_rule.rule_id} "
                            f"in group {group}"
                        )

        return issues

    def _rule_shadows(self, higher: CompiledRule, lower: CompiledRule) -> bool:
        """Check if higher-priority rule shadows lower-priority rule."""
        # Simple check: if terms are identical or higher is superset
        higher_terms = {term.term for term in higher.terms}
        lower_terms = {term.term for term in lower.terms}

        # If higher contains all of lower's terms, it shadows
        return lower_terms.issubset(higher_terms)


def compile_rules(config: RoutingConfig | None = None) -> list[CompiledRule]:
    """Convenience function to compile all rules."""
    compiler = RuleCompiler(config)
    rules = compiler.compile_all_legacy_groups()

    # Validate and report issues
    issues = compiler.validate_config()
    if issues:
        # For now, just warn; later versions can fail-closed
        import warnings
        for issue in issues:
            warnings.warn(f"Rule configuration issue: {issue}")

    return rules
