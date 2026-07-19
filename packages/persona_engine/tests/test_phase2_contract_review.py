"""Phase 2 Contract Review: Adversarial tests for rule engine contracts.

Tests clause boundaries, quote/negation scope, context conditions, activation groups,
stop_processing, deterministic ordering, evidence immutability, and privacy.
"""
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


class TestClauseBoundaries:
    """Verify clause boundary handling and span preservation."""

    def test_clause_index_preserved_across_multiple_clauses(self):
        """Clause indices must be preserved for multi-sentence input."""
        analyzer = ClauseAnalyzer()
        result = analyzer.analyze("First sentence. Second sentence. Third.")

        assert len(result.clauses) >= 2
        # Indices should be sequential
        for i, clause in enumerate(result.clauses):
            assert clause.index == i

    def test_span_ranges_non_overlapping(self):
        """Clause spans should not overlap."""
        analyzer = ClauseAnalyzer()
        result = analyzer.analyze("Sentence one. Sentence two.")

        if len(result.clauses) >= 2:
            for i in range(len(result.clauses) - 1):
                c1 = result.clauses[i]
                c2 = result.clauses[i + 1]
                # c1 should end before or at c2 start
                assert c1.span.end <= c2.span.start


class TestQuoteNegationScopeIsolation:
    """Verify quote and negation policies respect clause scope."""

    def test_quote_policy_suppress_prevents_match(self):
        """SUPPRESS policy must completely prevent quoted matches."""
        rule = CompiledRule(
            rule_id="QUOTE.SUPPRESS.001",
            feature_class=FeatureClass.HARD_BOUNDARY,
            candidate_mode=Mode.DAILY,
            terms=(RuleTerm(term="exit"),),
            weight=0.95,
            priority=1000,
            match_any=True,
            quote_policy=QuotePolicy.SUPPRESS
        )

        matcher = EvidenceMatcher()

        # Should match unquoted
        result_plain = matcher.match_rule(rule, "Let's exit now.", 0)
        assert len(result_plain) >= 1

        # Should NOT match quoted
        result_quoted = matcher.match_rule(rule, 'She said "exit please".', 0)
        assert len(result_quoted) == 0

    def test_negation_policy_suppress_prevents_match(self):
        """SUPPRESS policy must completely prevent negated matches."""
        rule = CompiledRule(
            rule_id="NEG.SUPPRESS.001",
            feature_class=FeatureClass.HARD_BOUNDARY,
            candidate_mode=Mode.DAILY,
            terms=(RuleTerm(term="stop"),),
            weight=0.95,
            priority=1000,
            match_any=True,
            negation_policy=NegationPolicy.SUPPRESS
        )

        matcher = EvidenceMatcher()

        # Should match plain
        result_plain = matcher.match_rule(rule, "Stop this.", 0)
        assert len(result_plain) >= 1

        # Should NOT match negated
        result_negated = matcher.match_rule(rule, "Don't stop this.", 0)
        assert len(result_negated) == 0

    def test_quote_and_negation_independent(self):
        """Quote and negation flags should be independent."""
        rule = CompiledRule(
            rule_id="BOTH.001",
            feature_class=FeatureClass.RELATIONSHIP,
            candidate_mode=Mode.DAILY,
            terms=(RuleTerm(term="love"),),
            weight=0.80,
            priority=600,
            match_any=True,
            quote_policy=QuotePolicy.ALLOW,
            negation_policy=NegationPolicy.ALLOW
        )

        matcher = EvidenceMatcher()

        # Negated and quoted
        result = matcher.match_rule(rule, 'He said "I don\'t love that".', 0)
        if result:
            # Both flags should be set
            assert result[0].quoted
            assert result[0].negated


class TestContextConditions:
    """Verify requires_context and forbids_context enforcement."""

    def test_multiple_requires_all_must_be_satisfied(self):
        """All required contexts must be present."""
        rule = CompiledRule(
            rule_id="CTX.MULTI.001",
            feature_class=FeatureClass.CONTINUATION_BINDING,
            candidate_mode=Mode.WORK,
            terms=(RuleTerm(
                term="continue",
                requires_context=("task_active", "user_engaged")
            ),),
            weight=0.85,
            priority=500,
            match_any=True
        )

        matcher = EvidenceMatcher()

        # Only one context - should not be valid
        result_partial = matcher.match_rule(
            rule, "Continue please.", 0,
            context_flags={"task_active"}
        )
        if result_partial:
            assert not result_partial[0].requires_satisfied

        # Both contexts - should be valid
        result_full = matcher.match_rule(
            rule, "Continue please.", 0,
            context_flags={"task_active", "user_engaged"}
        )
        assert len(result_full) >= 1
        assert result_full[0].requires_satisfied

    def test_any_forbids_triggers_block(self):
        """Any forbidden context should trigger forbids_triggered."""
        rule = CompiledRule(
            rule_id="CTX.FORBID.001",
            feature_class=FeatureClass.RELATIONSHIP,
            candidate_mode=Mode.DAILY,
            terms=(RuleTerm(
                term="relax",
                forbids_context=("crisis_mode", "emergency_mode")
            ),),
            weight=0.75,
            priority=600,
            match_any=True
        )

        matcher = EvidenceMatcher()

        # With one forbidden context
        result = matcher.match_rule(
            rule, "Let's relax.", 0,
            context_flags={"emergency_mode"}
        )
        if result:
            assert result[0].forbids_triggered
            assert not result[0].is_valid()


class TestActivationGroupConflict:
    """Verify activation group conflict resolution."""

    def test_stop_processing_prevents_same_group_only(self):
        """stop_processing should only affect same activation group."""
        rules = [
            CompiledRule(
                rule_id="GROUP_A_HIGH",
                feature_class=FeatureClass.HARD_BOUNDARY,
                candidate_mode=Mode.DAILY,
                terms=(RuleTerm(term="test"),),
                weight=0.95,
                priority=1000,
                match_any=True,
                activation_group="group_a",
                stop_processing=True
            ),
            CompiledRule(
                rule_id="GROUP_A_LOW",
                feature_class=FeatureClass.HARD_BOUNDARY,
                candidate_mode=Mode.DAILY,
                terms=(RuleTerm(term="test"),),
                weight=0.90,
                priority=900,
                match_any=True,
                activation_group="group_a"
            ),
            CompiledRule(
                rule_id="GROUP_B_MID",
                feature_class=FeatureClass.EXPLICIT_TASK,
                candidate_mode=Mode.WORK,
                terms=(RuleTerm(term="test"),),
                weight=0.85,
                priority=800,
                match_any=True,
                activation_group="group_b"
            ),
        ]

        engine = RuleEngine(rules)
        result = engine.match_message("test case")

        rule_ids = [e.rule_id for e in result]

        # GROUP_A_HIGH should fire
        assert "GROUP_A_HIGH" in rule_ids

        # GROUP_A_LOW should NOT fire (stopped)
        assert "GROUP_A_LOW" not in rule_ids

        # GROUP_B_MID should fire (different group)
        assert "GROUP_B_MID" in rule_ids

    def test_activation_group_empty_string_vs_none(self):
        """Empty activation_group should not interfere with other groups."""
        rules = [
            CompiledRule(
                rule_id="NO_GROUP_1",
                feature_class=FeatureClass.RELATIONSHIP,
                candidate_mode=Mode.DAILY,
                terms=(RuleTerm(term="test"),),
                weight=0.75,
                priority=600,
                match_any=True,
                activation_group=""  # No group
            ),
            CompiledRule(
                rule_id="NO_GROUP_2",
                feature_class=FeatureClass.RELATIONSHIP,
                candidate_mode=Mode.DAILY,
                terms=(RuleTerm(term="test"),),
                weight=0.72,
                priority=600,
                match_any=True,
                activation_group=""  # No group
            ),
        ]

        engine = RuleEngine(rules)
        result = engine.match_message("test case")

        # Both should fire (no group means no conflict)
        rule_ids = [e.rule_id for e in result]
        assert "NO_GROUP_1" in rule_ids
        assert "NO_GROUP_2" in rule_ids


class TestDeterministicOrdering:
    """Verify deterministic evidence ordering."""

    def test_ordering_stable_across_runs(self):
        """Same rules and input should produce identical ordering."""
        rules = [
            CompiledRule(
                rule_id="C.MID",
                feature_class=FeatureClass.EXPLICIT_TASK,
                candidate_mode=Mode.WORK,
                terms=(RuleTerm(term="test"),),
                weight=0.85,
                priority=800,
                match_any=True
            ),
            CompiledRule(
                rule_id="A.MID",
                feature_class=FeatureClass.EXPLICIT_TASK,
                candidate_mode=Mode.WORK,
                terms=(RuleTerm(term="test"),),
                weight=0.85,
                priority=800,
                match_any=True
            ),
            CompiledRule(
                rule_id="B.MID",
                feature_class=FeatureClass.EXPLICIT_TASK,
                candidate_mode=Mode.WORK,
                terms=(RuleTerm(term="test"),),
                weight=0.85,
                priority=800,
                match_any=True
            ),
        ]

        engine = RuleEngine(rules)

        # Run multiple times
        results = [engine.match_message("test") for _ in range(3)]

        # All runs should produce identical ordering
        first_order = [e.rule_id for e in results[0]]
        for result in results[1:]:
            assert [e.rule_id for e in result] == first_order

        # Should be alphabetically sorted for same priority
        assert first_order == sorted(first_order)


class TestEvidenceImmutability:
    """Verify evidence objects are truly immutable."""

    def test_match_evidence_frozen(self):
        """MatchEvidence should be frozen and immutable."""
        evidence = MatchEvidence(
            rule_id="TEST.001",
            feature_class=FeatureClass.EXPLICIT_TASK,
            candidate_mode=Mode.WORK,
            weight=0.85,
            priority=800,
            clause_index=0
        )

        # Should not be able to modify
        with pytest.raises((AttributeError, TypeError)):
            evidence.weight = 0.90

        with pytest.raises((AttributeError, TypeError)):
            evidence.rule_id = "MODIFIED"

    def test_compiled_rule_frozen(self):
        """CompiledRule should be frozen and immutable."""
        rule = CompiledRule(
            rule_id="TEST.001",
            feature_class=FeatureClass.EXPLICIT_TASK,
            candidate_mode=Mode.WORK,
            terms=(RuleTerm(term="test"),),
            weight=0.85,
            priority=800,
            match_any=True
        )

        # Should not be able to modify
        with pytest.raises((AttributeError, TypeError)):
            rule.priority = 900

        with pytest.raises((AttributeError, TypeError)):
            rule.weight = 0.90


class TestPrivacyInvariant:
    """Verify sensitive content is never exposed."""

    def test_match_evidence_repr_opaque(self):
        """Evidence repr should never expose matched term."""
        evidence = MatchEvidence(
            rule_id="PRIVACY.001",
            feature_class=FeatureClass.PROTECTED_PROGRESSION,
            candidate_mode=Mode.SEX,
            weight=0.85,
            priority=700,
            clause_index=0,
            matched_term="",  # Always empty
            span_start=10,
            span_end=20
        )

        repr_str = repr(evidence)

        # Should contain rule_id
        assert "PRIVACY.001" in repr_str

        # Should NOT contain actual term or hint about content
        # Only structural information
        assert "matched_term" not in repr_str or '""' in repr_str or "=''" in repr_str

    def test_clause_analyzer_repr_opaque(self):
        """ClauseAnalyzer output repr should not expose text."""
        analyzer = ClauseAnalyzer()
        result = analyzer.analyze("Sensitive content here.")

        # Check various repr outputs
        norm_repr = repr(result.normalized)
        assert "Sensitive" not in norm_repr
        assert "[REDACTED]" in norm_repr or "..." in norm_repr

        if result.clauses:
            clause_repr = repr(result.clauses[0])
            assert "Sensitive" not in clause_repr
            assert "[REDACTED]" in clause_repr or "..." in clause_repr

    def test_evidence_never_logs_actual_term(self):
        """matched_term field should always be empty or opaque."""
        rule = CompiledRule(
            rule_id="LOG.001",
            feature_class=FeatureClass.EXPLICIT_TASK,
            candidate_mode=Mode.WORK,
            terms=(RuleTerm(term="sensitive_data"),),
            weight=0.85,
            priority=800,
            match_any=True
        )

        matcher = EvidenceMatcher()
        result = matcher.match_rule(rule, "Contains sensitive_data here.", 0)

        for evidence in result:
            # matched_term should be empty
            assert evidence.matched_term == ""


class TestStopProcessingScope:
    """Verify stop_processing scope boundaries."""

    def test_stop_processing_requires_hard_boundary(self):
        """Only hard_boundary rules can use stop_processing."""
        # This should succeed
        rule_valid = CompiledRule(
            rule_id="VALID.STOP",
            feature_class=FeatureClass.HARD_BOUNDARY,
            candidate_mode=Mode.DAILY,
            terms=(RuleTerm(term="test"),),
            weight=0.95,
            priority=1000,
            match_any=True,
            stop_processing=True
        )
        assert rule_valid.stop_processing

        # This should fail at construction
        with pytest.raises(ValueError, match="stop_processing only allowed"):
            CompiledRule(
                rule_id="INVALID.STOP",
                feature_class=FeatureClass.EXPLICIT_TASK,
                candidate_mode=Mode.WORK,
                terms=(RuleTerm(term="test"),),
                weight=0.85,
                priority=800,
                match_any=True,
                stop_processing=True
            )
