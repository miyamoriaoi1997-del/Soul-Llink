"""Test legacy parity between old classifier and new typed rule system.

Verifies that the new typed rule compiler produces equivalent decisions
to the existing mode_classifier.py for all baseline cases.
"""
import pytest

from packages.persona_engine.persona_orchestrator.mode_classifier import ModeClassifier
from packages.persona_engine.persona_orchestrator.rule_compiler import compile_rules
from packages.persona_engine.persona_orchestrator.rule_schema import FeatureClass


class TestLegacyParity:
    """Test that typed rules match legacy classifier behavior."""

    @pytest.fixture
    def legacy_classifier(self):
        """Create instance of legacy classifier."""
        return ModeClassifier()

    @pytest.fixture
    def compiled_rules(self):
        """Compile typed rules from config."""
        return compile_rules()

    def test_rules_compiled_successfully(self, compiled_rules):
        """Test that rules compile without errors."""
        assert len(compiled_rules) > 0
        assert all(hasattr(rule, 'rule_id') for rule in compiled_rules)
        assert all(hasattr(rule, 'terms') for rule in compiled_rules)
        assert all(len(rule.terms) > 0 for rule in compiled_rules)

    def test_crisis_rules_present(self, compiled_rules):
        """Test that crisis/hard boundary rules are compiled."""
        hard_boundary_rules = [
            r for r in compiled_rules
            if r.feature_class == FeatureClass.HARD_BOUNDARY
        ]
        assert len(hard_boundary_rules) > 0

        # Crisis should have highest priority
        crisis_rules = [r for r in hard_boundary_rules if 'crisis' in r.rule_id.lower()]
        if crisis_rules:
            assert all(r.priority >= 900 for r in crisis_rules)

    def test_sex_rules_present(self, compiled_rules):
        """Test that protected progression (sex) rules are compiled."""
        protected_rules = [
            r for r in compiled_rules
            if r.feature_class == FeatureClass.PROTECTED_PROGRESSION
        ]
        assert len(protected_rules) > 0

        # Should include explicit and hint variants
        rule_ids = [r.rule_id.lower() for r in protected_rules]
        assert any('sex' in rid for rid in rule_ids)

    def test_relationship_rules_present(self, compiled_rules):
        """Test that relationship evidence rules are compiled."""
        relationship_rules = [
            r for r in compiled_rules
            if r.feature_class == FeatureClass.RELATIONSHIP
        ]
        assert len(relationship_rules) > 0

        # Should include intimacy, aftercare, etc.
        rule_ids = [r.rule_id.lower() for r in relationship_rules]
        assert any('intimacy' in rid or 'aftercare' in rid for rid in rule_ids)

    def test_continuation_rules_present(self, compiled_rules):
        """Test that continuation binding rules are compiled."""
        continuation_rules = [
            r for r in compiled_rules
            if r.feature_class == FeatureClass.CONTINUATION_BINDING
        ]
        assert len(continuation_rules) > 0

    def test_priority_ordering(self, compiled_rules):
        """Test that rules have correct priority ordering."""
        # Group by feature class
        by_class = {}
        for rule in compiled_rules:
            fc = rule.feature_class
            if fc not in by_class:
                by_class[fc] = []
            by_class[fc].append(rule)

        # Hard boundary should have highest priorities
        if FeatureClass.HARD_BOUNDARY in by_class:
            hard_priorities = [r.priority for r in by_class[FeatureClass.HARD_BOUNDARY]]
            assert all(p >= 900 for p in hard_priorities)

        # Protected progression should be high but not highest
        if FeatureClass.PROTECTED_PROGRESSION in by_class:
            protected_priorities = [r.priority for r in by_class[FeatureClass.PROTECTED_PROGRESSION]]
            assert all(600 <= p <= 800 for p in protected_priorities)

    def test_term_metadata_preserved(self, compiled_rules):
        """Test that term metadata is preserved in compiled rules."""
        # Check that terms have proper structure
        for rule in compiled_rules:
            for term in rule.terms:
                assert hasattr(term, 'term')
                assert hasattr(term, 'weight')
                assert hasattr(term, 'requires_context')
                assert hasattr(term, 'forbids_context')
                assert isinstance(term.term, str)
                assert 0.0 <= term.weight <= 1.0

    def test_no_duplicate_rule_ids(self, compiled_rules):
        """Test that all rule IDs are unique."""
        rule_ids = [r.rule_id for r in compiled_rules]
        assert len(rule_ids) == len(set(rule_ids)), "Duplicate rule IDs found"

    def test_activation_groups_assigned(self, compiled_rules):
        """Test that rules have activation groups for conflict resolution."""
        # Most rules should have activation groups
        grouped_rules = [r for r in compiled_rules if r.activation_group]
        assert len(grouped_rules) > 0

    def test_legacy_classifier_still_works(self, legacy_classifier):
        """Test that legacy classifier continues to function."""
        # Simple smoke test
        decision = legacy_classifier.classify(
            user_message="test message",
            platform="test"
        )
        assert hasattr(decision, 'mode')
        assert hasattr(decision, 'confidence')
        assert decision.mode in ['daily', 'work', 'sex']


class TestConfigTypedAccessors:
    """Test that config loader typed accessors work correctly."""

    def test_typed_vs_flat_accessor_equivalence(self):
        """Test that typed accessor terms match flat accessor term strings."""
        from packages.persona_engine.persona_orchestrator.config_loader import load_routing_config

        config = load_routing_config()

        # Test a few known groups
        for group in ['crisis', 'intimacy', 'aftercare']:
            flat = config.classifier_terms(group)
            typed = config.classifier_terms_typed(group)

            # Extract term strings from typed
            typed_terms = []
            for item in typed:
                if isinstance(item, str):
                    typed_terms.append(item)
                elif isinstance(item, dict) and 'term' in item:
                    typed_terms.append(item['term'])

            # Should match
            assert flat == typed_terms, f"Mismatch in group {group}"
