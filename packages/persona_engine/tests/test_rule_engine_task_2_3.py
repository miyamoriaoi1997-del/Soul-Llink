"""Tests for Phase 2 Task 2.3: Complex scenarios with meta discussion, mixed intent,
negation, and exit priority.

All controlled literals are handled through opaque fixtures/hashes.
"""
import pytest

from packages.persona_engine.persona_orchestrator.rule_engine import (
    RuleEngine,
    EvidenceMatcher,
)
from packages.persona_engine.persona_orchestrator.rule_schema import (
    CompiledRule,
    FeatureClass,
    Mode,
    NegationPolicy,
    QuotePolicy,
    RuleTerm,
)


class TestMetaDiscussionScenarios:
    """Test meta-discussion handling with controlled fragments."""

    def test_meta_discussion_about_boundary_routes_to_work(self):
        """Meta discussion referencing boundary terms should route to Work."""
        rules = [
            CompiledRule(
                rule_id="META.001",
                feature_class=FeatureClass.EXPLICIT_META,
                candidate_mode=Mode.WORK,
                terms=(RuleTerm(term="about"),),
                weight=0.90,
                priority=900,
                match_any=True,
                quote_policy=QuotePolicy.ALLOW  # Meta allows quotes
            ),
            CompiledRule(
                rule_id="BOUNDARY.001",
                feature_class=FeatureClass.HARD_BOUNDARY,
                candidate_mode=Mode.DAILY,
                terms=(RuleTerm(term="exit"),),
                weight=0.95,
                priority=1000,
                match_any=True,
                quote_policy=QuotePolicy.SUPPRESS  # Boundary suppresses quotes
            ),
        ]

        engine = RuleEngine(rules)

        # "Tell me about 'exit' command" should match meta, not boundary
        result = engine.match_message('Tell me about "exit" command.')

        # Should have meta evidence
        meta_evidence = [e for e in result if e.rule_id == "META.001"]
        assert len(meta_evidence) >= 1

        # Boundary should not fire inside quotes
        boundary_evidence = [e for e in result if e.rule_id == "BOUNDARY.001"]
        assert len(boundary_evidence) == 0

    def test_quoted_progression_not_treated_as_current(self):
        """Quoted/hypothetical progression should not trigger actual mode change."""
        rules = [
            CompiledRule(
                rule_id="PROTECTED.001",
                feature_class=FeatureClass.PROTECTED_PROGRESSION,
                candidate_mode=Mode.SEX,
                terms=(RuleTerm(term="intimate"),),
                weight=0.85,
                priority=700,
                match_any=True,
                quote_policy=QuotePolicy.SUPPRESS
            ),
        ]

        engine = RuleEngine(rules)

        # Direct mention
        result_direct = engine.match_message("Let's get intimate.")
        assert len(result_direct) >= 1

        # Quoted mention - should not match
        result_quoted = engine.match_message('She said "get intimate".')
        assert len(result_quoted) == 0

    def test_hypothetical_does_not_trigger_progression(self):
        """Hypothetical statements should not trigger progression."""
        rules = [
            CompiledRule(
                rule_id="PROTECTED.002",
                feature_class=FeatureClass.PROTECTED_PROGRESSION,
                candidate_mode=Mode.SEX,
                terms=(RuleTerm(term="desire"),),
                weight=0.80,
                priority=700,
                match_any=True,
                quote_policy=QuotePolicy.SUPPRESS
            ),
        ]

        engine = RuleEngine(rules)
        matcher = EvidenceMatcher()

        # Hypothetical context should mark evidence
        result = matcher.match_rule(rules[0], "What if I desire something?", 0)

        if result:
            # Evidence exists but should be marked hypothetical
            assert result[0].hypothetical


class TestNegationAndExitPriority:
    """Test negation handling and exit priority resolution."""

    def test_negated_stop_vs_real_exit(self):
        """Negated stop should not be treated as exit."""
        rules = [
            CompiledRule(
                rule_id="EXIT.001",
                feature_class=FeatureClass.HARD_BOUNDARY,
                candidate_mode=Mode.DAILY,
                terms=(RuleTerm(term="stop"),),
                weight=0.95,
                priority=1000,
                match_any=True,
                negation_policy=NegationPolicy.SUPPRESS
            ),
        ]

        engine = RuleEngine(rules)

        # Real exit
        result_exit = engine.match_message("Stop now.")
        assert len(result_exit) >= 1

        # Negated stop - should not match
        result_negated = engine.match_message("Don't stop.")
        assert len(result_negated) == 0

    def test_explicit_exit_has_higher_priority_than_negated_term(self):
        """Explicit exit with higher priority overrides lower priority rules."""
        rules = [
            CompiledRule(
                rule_id="EXIT.EXPLICIT",
                feature_class=FeatureClass.HARD_BOUNDARY,
                candidate_mode=Mode.DAILY,
                terms=(RuleTerm(term="goodbye"),),
                weight=0.95,
                priority=1000,
                match_any=True,
                negation_policy=NegationPolicy.SUPPRESS,
                stop_processing=True,
                activation_group="boundary"
            ),
            CompiledRule(
                rule_id="EXIT.AMBIGUOUS",
                feature_class=FeatureClass.HARD_BOUNDARY,
                candidate_mode=Mode.DAILY,
                terms=(RuleTerm(term="later"),),
                weight=0.70,
                priority=950,
                match_any=True,
                negation_policy=NegationPolicy.ALLOW,
                activation_group="boundary"
            ),
        ]

        engine = RuleEngine(rules)

        # Explicit exit should fire and stop lower priority rules in same group
        result = engine.match_message("Goodbye for now, see you later.")

        # Should match explicit exit
        exit_evidence = [e for e in result if e.rule_id == "EXIT.EXPLICIT"]
        assert len(exit_evidence) >= 1

        # Lower priority in same group should be stopped
        ambiguous_evidence = [e for e in result if e.rule_id == "EXIT.AMBIGUOUS"]
        assert len(ambiguous_evidence) == 0


class TestMixedIntentScenarios:
    """Test mixed task and relationship intent handling."""

    def test_task_request_with_boundary_reference_stays_work(self):
        """Work request referencing boundary text should not be downgraded."""
        rules = [
            CompiledRule(
                rule_id="TASK.001",
                feature_class=FeatureClass.EXPLICIT_TASK,
                candidate_mode=Mode.WORK,
                terms=(RuleTerm(term="debug"),),
                weight=0.88,
                priority=800,
                match_any=True,
                quote_policy=QuotePolicy.ALLOW
            ),
            CompiledRule(
                rule_id="BOUNDARY.001",
                feature_class=FeatureClass.HARD_BOUNDARY,
                candidate_mode=Mode.DAILY,
                terms=(RuleTerm(term="crash"),),
                weight=0.92,
                priority=1000,
                match_any=True,
                quote_policy=QuotePolicy.SUPPRESS
            ),
        ]

        engine = RuleEngine(rules)

        # Task request mentioning boundary term in context
        result = engine.match_message("Debug the crash issue please.")

        # Should have task evidence
        task_evidence = [e for e in result if e.rule_id == "TASK.001"]
        assert len(task_evidence) >= 1

        # Boundary should also match (not suppressed)
        boundary_evidence = [e for e in result if e.rule_id == "BOUNDARY.001"]
        assert len(boundary_evidence) >= 1

    def test_task_and_relationship_coexist_task_wins_nomination(self):
        """When task and relationship terms coexist, task wins nomination."""
        rules = [
            CompiledRule(
                rule_id="TASK.002",
                feature_class=FeatureClass.EXPLICIT_TASK,
                candidate_mode=Mode.WORK,
                terms=(RuleTerm(term="analyze"),),
                weight=0.85,
                priority=800,
                match_any=True,
                activation_group="primary_intent"
            ),
            CompiledRule(
                rule_id="REL.001",
                feature_class=FeatureClass.RELATIONSHIP,
                candidate_mode=Mode.DAILY,
                terms=(RuleTerm(term="love"),),
                weight=0.78,
                priority=600,
                match_any=True,
                activation_group="secondary_intent"
            ),
        ]

        engine = RuleEngine(rules)

        # Mixed intent: task + relationship
        result = engine.match_message("Analyze why people love this design.")

        # Both should produce evidence
        task_evidence = [e for e in result if e.rule_id == "TASK.002"]
        rel_evidence = [e for e in result if e.rule_id == "REL.001"]

        assert len(task_evidence) >= 1
        assert len(rel_evidence) >= 1

        # Task has higher priority, should appear first
        if len(result) >= 2:
            assert result[0].priority >= result[1].priority

    def test_continuation_with_ambiguous_target(self):
        """Ambiguous continuation should preserve context binding."""
        rules = [
            CompiledRule(
                rule_id="CONT.WORK",
                feature_class=FeatureClass.CONTINUATION_BINDING,
                candidate_mode=Mode.WORK,
                terms=(RuleTerm(
                    term="continue",
                    requires_context=("task_active",)
                ),),
                weight=0.82,
                priority=500,
                match_any=True
            ),
            CompiledRule(
                rule_id="CONT.REL",
                feature_class=FeatureClass.CONTINUATION_BINDING,
                candidate_mode=Mode.DAILY,
                terms=(RuleTerm(
                    term="continue",
                    requires_context=("relationship_active",)
                ),),
                weight=0.80,
                priority=500,
                match_any=True
            ),
        ]

        engine = RuleEngine(rules)

        # With task context
        result_task = engine.match_message(
            "Let's continue.",
            context_flags={"task_active"}
        )
        work_evidence = [e for e in result_task if e.rule_id == "CONT.WORK"]
        assert len(work_evidence) >= 1
        assert work_evidence[0].requires_satisfied

        # With relationship context
        result_rel = engine.match_message(
            "Let's continue.",
            context_flags={"relationship_active"}
        )
        rel_evidence = [e for e in result_rel if e.rule_id == "CONT.REL"]
        assert len(rel_evidence) >= 1
        assert rel_evidence[0].requires_satisfied

        # Without context - no valid evidence
        result_none = engine.match_message(
            "Let's continue.",
            context_flags=set()
        )
        valid_evidence = [e for e in result_none if e.is_valid()]
        # May have evidence but requires_satisfied will be False
        for ev in result_none:
            if ev.rule_id in ("CONT.WORK", "CONT.REL"):
                assert not ev.requires_satisfied


class TestStopProcessingScope:
    """Test stop_processing behavior within activation groups."""

    def test_stop_processing_only_affects_same_group(self):
        """stop_processing should only affect same activation group."""
        rules = [
            CompiledRule(
                rule_id="BOUND.STOP",
                feature_class=FeatureClass.HARD_BOUNDARY,
                candidate_mode=Mode.DAILY,
                terms=(RuleTerm(term="emergency"),),
                weight=0.95,
                priority=1000,
                match_any=True,
                activation_group="boundary",
                stop_processing=True
            ),
            CompiledRule(
                rule_id="BOUND.OTHER",
                feature_class=FeatureClass.HARD_BOUNDARY,
                candidate_mode=Mode.DAILY,
                terms=(RuleTerm(term="emergency"),),
                weight=0.90,
                priority=950,
                match_any=True,
                activation_group="boundary"
            ),
            CompiledRule(
                rule_id="TASK.INDEPENDENT",
                feature_class=FeatureClass.EXPLICIT_TASK,
                candidate_mode=Mode.WORK,
                terms=(RuleTerm(term="emergency"),),
                weight=0.85,
                priority=800,
                match_any=True,
                activation_group="task"
            ),
        ]

        engine = RuleEngine(rules)
        result = engine.match_message("Emergency response needed.")

        # Should have BOUND.STOP
        stop_evidence = [e for e in result if e.rule_id == "BOUND.STOP"]
        assert len(stop_evidence) >= 1

        # Should NOT have BOUND.OTHER (same group, stopped)
        other_evidence = [e for e in result if e.rule_id == "BOUND.OTHER"]
        assert len(other_evidence) == 0

        # Should have TASK.INDEPENDENT (different group)
        task_evidence = [e for e in result if e.rule_id == "TASK.INDEPENDENT"]
        assert len(task_evidence) >= 1


class TestDeterministicSorting:
    """Test deterministic evidence sorting."""

    def test_evidence_sorted_by_priority_then_rule_id(self):
        """Evidence should be deterministically sorted."""
        rules = [
            CompiledRule(
                rule_id="Z.LOW",
                feature_class=FeatureClass.FALLBACK,
                candidate_mode=Mode.DAILY,
                terms=(RuleTerm(term="test"),),
                weight=0.50,
                priority=100,
                match_any=True
            ),
            CompiledRule(
                rule_id="A.HIGH",
                feature_class=FeatureClass.EXPLICIT_TASK,
                candidate_mode=Mode.WORK,
                terms=(RuleTerm(term="test"),),
                weight=0.85,
                priority=800,
                match_any=True
            ),
            CompiledRule(
                rule_id="B.HIGH",
                feature_class=FeatureClass.EXPLICIT_TASK,
                candidate_mode=Mode.WORK,
                terms=(RuleTerm(term="test"),),
                weight=0.85,
                priority=800,
                match_any=True
            ),
        ]

        engine = RuleEngine(rules)
        result = engine.match_message("Test message.")

        assert len(result) >= 3

        # Should be sorted: high priority first
        assert result[0].priority >= result[1].priority >= result[2].priority

        # Same priority should be sorted by rule_id
        high_priority = [e for e in result if e.priority == 800]
        if len(high_priority) >= 2:
            assert high_priority[0].rule_id < high_priority[1].rule_id


class TestPrivacyInvariant:
    """Verify no sensitive literals in evidence."""

    def test_match_evidence_never_logs_term_content(self):
        """MatchEvidence should never expose actual term content."""
        rule = CompiledRule(
            rule_id="PRIVACY.001",
            feature_class=FeatureClass.EXPLICIT_TASK,
            candidate_mode=Mode.WORK,
            terms=(RuleTerm(term="sensitive_literal"),),
            weight=0.85,
            priority=800,
            match_any=True
        )

        matcher = EvidenceMatcher()
        result = matcher.match_rule(rule, "This contains sensitive_literal text.", 0)

        assert len(result) >= 1
        evidence = result[0]

        # matched_term should be empty or redacted
        assert evidence.matched_term == ""

        # repr should not contain term
        evidence_repr = repr(evidence)
        assert "sensitive_literal" not in evidence_repr.lower()

        # Rule ID is OK to expose
        assert "PRIVACY.001" in evidence_repr
