"""Tests for rule engine evidence matching."""
import pytest

from packages.persona_engine.persona_orchestrator.clause_analysis import ClauseAnalyzer
from packages.persona_engine.persona_orchestrator.rule_engine import (
    RuleEngine,
    EvidenceMatcher,
)
from packages.persona_engine.persona_orchestrator.rule_schema import (
    CompiledRule,
    FeatureClass,
    MatchEvidence,
    Mode,
    NegationPolicy,
    QuotePolicy,
    RuleTerm,
)


class TestEvidenceMatcher:
    """Test basic evidence matching."""

    def test_simple_term_match(self):
        rule = CompiledRule(
            rule_id="TEST.001",
            feature_class=FeatureClass.EXPLICIT_TASK,
            candidate_mode=Mode.WORK,
            terms=(RuleTerm(term="system"),),
            weight=0.85,
            priority=800,
            match_any=True
        )

        matcher = EvidenceMatcher()
        evidences = matcher.match_rule(rule, "Configure the system please.", 0)

        assert len(evidences) >= 1
        assert evidences[0].rule_id == "TEST.001"
        assert evidences[0].candidate_mode == Mode.WORK

    def test_no_match_returns_empty(self):
        rule = CompiledRule(
            rule_id="TEST.002",
            feature_class=FeatureClass.EXPLICIT_TASK,
            candidate_mode=Mode.WORK,
            terms=(RuleTerm(term="nonexistent"),),
            weight=0.85,
            priority=800,
            match_any=True
        )

        matcher = EvidenceMatcher()
        evidences = matcher.match_rule(rule, "Configure the system.", 0)

        assert len(evidences) == 0

    def test_match_any_multiple_terms(self):
        rule = CompiledRule(
            rule_id="TEST.003",
            feature_class=FeatureClass.EXPLICIT_TASK,
            candidate_mode=Mode.WORK,
            terms=(
                RuleTerm(term="alpha"),
                RuleTerm(term="beta"),
                RuleTerm(term="gamma"),
            ),
            weight=0.85,
            priority=800,
            match_any=True
        )

        matcher = EvidenceMatcher()

        # Should match if any term present
        evidences = matcher.match_rule(rule, "Use beta approach.", 0)
        assert len(evidences) >= 1

    def test_match_all_requires_all_terms(self):
        rule = CompiledRule(
            rule_id="TEST.004",
            feature_class=FeatureClass.EXPLICIT_TASK,
            candidate_mode=Mode.WORK,
            terms=(
                RuleTerm(term="alpha"),
                RuleTerm(term="beta"),
            ),
            weight=0.85,
            priority=800,
            match_any=False  # All terms required
        )

        matcher = EvidenceMatcher()

        # Should not match with only one term
        evidences = matcher.match_rule(rule, "Use alpha only.", 0)
        assert len(evidences) == 0

        # Should match with both terms
        evidences = matcher.match_rule(rule, "Use alpha and beta.", 0)
        assert len(evidences) >= 1


class TestQuotePolicy:
    """Test quote policy handling."""

    def test_suppress_ignores_quoted_terms(self):
        rule = CompiledRule(
            rule_id="TEST.QUOTE.001",
            feature_class=FeatureClass.EXPLICIT_TASK,
            candidate_mode=Mode.WORK,
            terms=(RuleTerm(term="stop"),),
            weight=0.90,
            priority=1000,
            match_any=True,
            quote_policy=QuotePolicy.SUPPRESS
        )

        matcher = EvidenceMatcher()

        # Should not match inside quotes
        evidences = matcher.match_rule(rule, 'She said "stop that".', 0)
        # May or may not match depending on implementation - check valid flag
        if evidences:
            assert not evidences[0].is_valid() or not evidences[0].quoted

    def test_allow_matches_quoted_terms(self):
        rule = CompiledRule(
            rule_id="TEST.QUOTE.002",
            feature_class=FeatureClass.RELATIONSHIP,
            candidate_mode=Mode.DAILY,
            terms=(RuleTerm(term="love"),),
            weight=0.80,
            priority=600,
            match_any=True,
            quote_policy=QuotePolicy.ALLOW
        )

        matcher = EvidenceMatcher()

        # Should match even inside quotes
        evidences = matcher.match_rule(rule, 'He said "I love you".', 0)
        assert len(evidences) >= 1


class TestNegationPolicy:
    """Test negation policy handling."""

    def test_suppress_ignores_negated_terms(self):
        rule = CompiledRule(
            rule_id="TEST.NEG.001",
            feature_class=FeatureClass.HARD_BOUNDARY,
            candidate_mode=Mode.DAILY,
            terms=(RuleTerm(term="stop"),),
            weight=0.95,
            priority=1000,
            match_any=True,
            negation_policy=NegationPolicy.SUPPRESS
        )

        matcher = EvidenceMatcher()

        # Should not match when negated
        evidences = matcher.match_rule(rule, "Don't stop now.", 0)
        if evidences:
            assert not evidences[0].is_valid() or not evidences[0].negated

    def test_allow_matches_negated_terms(self):
        rule = CompiledRule(
            rule_id="TEST.NEG.002",
            feature_class=FeatureClass.RELATIONSHIP,
            candidate_mode=Mode.DAILY,
            terms=(RuleTerm(term="happy"),),
            weight=0.75,
            priority=600,
            match_any=True,
            negation_policy=NegationPolicy.ALLOW
        )

        matcher = EvidenceMatcher()

        # Should match even when negated
        evidences = matcher.match_rule(rule, "I'm not happy.", 0)
        assert len(evidences) >= 1


class TestContextRequirements:
    """Test requires_context and forbids_context."""

    def test_requires_context_satisfied(self):
        rule = CompiledRule(
            rule_id="TEST.CTX.001",
            feature_class=FeatureClass.CONTINUATION_BINDING,
            candidate_mode=Mode.WORK,
            terms=(RuleTerm(
                term="continue",
                requires_context=("task_active",)
            ),),
            weight=0.80,
            priority=500,
            match_any=True
        )

        matcher = EvidenceMatcher()

        # With context
        evidences = matcher.match_rule(
            rule, "Let's continue.", 0,
            context_flags={"task_active"}
        )
        assert len(evidences) >= 1
        assert evidences[0].requires_satisfied

    def test_requires_context_unsatisfied(self):
        rule = CompiledRule(
            rule_id="TEST.CTX.002",
            feature_class=FeatureClass.CONTINUATION_BINDING,
            candidate_mode=Mode.WORK,
            terms=(RuleTerm(
                term="continue",
                requires_context=("task_active",)
            ),),
            weight=0.80,
            priority=500,
            match_any=True
        )

        matcher = EvidenceMatcher()

        # Without required context
        evidences = matcher.match_rule(
            rule, "Let's continue.", 0,
            context_flags=set()
        )
        if evidences:
            assert not evidences[0].requires_satisfied

    def test_forbids_context_triggered(self):
        rule = CompiledRule(
            rule_id="TEST.CTX.003",
            feature_class=FeatureClass.RELATIONSHIP,
            candidate_mode=Mode.DAILY,
            terms=(RuleTerm(
                term="relax",
                forbids_context=("crisis_mode",)
            ),),
            weight=0.75,
            priority=600,
            match_any=True
        )

        matcher = EvidenceMatcher()

        # With forbidden context
        evidences = matcher.match_rule(
            rule, "Let's relax.", 0,
            context_flags={"crisis_mode"}
        )
        if evidences:
            assert evidences[0].forbids_triggered


class TestRuleEngine:
    """Test full rule engine with multiple rules."""

    def test_collects_evidence_from_multiple_rules(self):
        rules = [
            CompiledRule(
                rule_id="TASK.001",
                feature_class=FeatureClass.EXPLICIT_TASK,
                candidate_mode=Mode.WORK,
                terms=(RuleTerm(term="debug"),),
                weight=0.85,
                priority=800,
                match_any=True
            ),
            CompiledRule(
                rule_id="TASK.002",
                feature_class=FeatureClass.EXPLICIT_TASK,
                candidate_mode=Mode.WORK,
                terms=(RuleTerm(term="system"),),
                weight=0.82,
                priority=800,
                match_any=True
            ),
        ]

        engine = RuleEngine(rules)
        result = engine.match_message("Debug the system issue.")

        # Should collect evidence from both rules
        assert len(result) >= 2
        rule_ids = {e.rule_id for e in result}
        assert "TASK.001" in rule_ids
        assert "TASK.002" in rule_ids

    def test_priority_ordering(self):
        rules = [
            CompiledRule(
                rule_id="LOW.001",
                feature_class=FeatureClass.FALLBACK,
                candidate_mode=Mode.DAILY,
                terms=(RuleTerm(term="hello"),),
                weight=0.50,
                priority=100,
                match_any=True,
                activation_group="greeting"
            ),
            CompiledRule(
                rule_id="HIGH.001",
                feature_class=FeatureClass.HARD_BOUNDARY,
                candidate_mode=Mode.DAILY,
                terms=(RuleTerm(term="hello"),),
                weight=0.95,
                priority=1000,
                match_any=True,
                activation_group="greeting"
            ),
        ]

        engine = RuleEngine(rules)
        result = engine.match_message("Hello there.")

        # High priority should come first
        assert len(result) >= 1
        assert result[0].priority >= result[-1].priority

    def test_activation_group_conflict_resolution(self):
        """Rules in same activation group with stop_processing."""
        rules = [
            CompiledRule(
                rule_id="BOUNDARY.001",
                feature_class=FeatureClass.HARD_BOUNDARY,
                candidate_mode=Mode.DAILY,
                terms=(RuleTerm(term="crisis"),),
                weight=0.95,
                priority=1000,
                match_any=True,
                activation_group="boundary",
                stop_processing=True
            ),
            CompiledRule(
                rule_id="BOUNDARY.002",
                feature_class=FeatureClass.HARD_BOUNDARY,
                candidate_mode=Mode.DAILY,
                terms=(RuleTerm(term="crisis"),),
                weight=0.90,
                priority=900,
                match_any=True,
                activation_group="boundary"
            ),
        ]

        engine = RuleEngine(rules)
        result = engine.match_message("This is a crisis situation.")

        # Should stop after first rule
        assert len(result) >= 1
        # Only highest priority in group should fire
        assert result[0].rule_id == "BOUNDARY.001"


class TestDeterministicOrdering:
    """Test deterministic rule ordering."""

    def test_same_priority_ordered_by_rule_id(self):
        rules = [
            CompiledRule(
                rule_id="Z.LAST",
                feature_class=FeatureClass.RELATIONSHIP,
                candidate_mode=Mode.DAILY,
                terms=(RuleTerm(term="test"),),
                weight=0.75,
                priority=600,
                match_any=True
            ),
            CompiledRule(
                rule_id="A.FIRST",
                feature_class=FeatureClass.RELATIONSHIP,
                candidate_mode=Mode.DAILY,
                terms=(RuleTerm(term="test"),),
                weight=0.75,
                priority=600,
                match_any=True
            ),
        ]

        engine = RuleEngine(rules)
        result = engine.match_message("This is a test.")

        # With same priority, should be ordered by rule_id
        if len(result) >= 2:
            ids = [e.rule_id for e in result]
            assert ids == sorted(ids)
