"""Rule-based evidence matching engine.

Matches compiled rules against text using clause-aware analysis,
producing immutable evidence facts.
"""
from __future__ import annotations

from typing import Set

from .clause_analysis import ClauseAnalyzer
from .rule_schema import (
    CompiledRule,
    MatchEvidence,
    NegationPolicy,
    QuotePolicy,
)


class EvidenceMatcher:
    """Match a single rule against text and produce evidence."""

    def __init__(self):
        self.clause_analyzer = ClauseAnalyzer()

    def match_rule(
        self,
        rule: CompiledRule,
        text: str,
        clause_index: int,
        context_flags: Set[str] | None = None,
    ) -> list[MatchEvidence]:
        """Match a rule against text and produce evidence.

        Args:
            rule: Compiled rule to match
            text: Text to match against (typically one clause)
            clause_index: Index of this clause in the message
            context_flags: Active context flags for requires/forbids

        Returns:
            List of evidence (one per matching term, or empty)
        """
        if not text:
            return []

        context_flags = context_flags or set()

        # Analyze discourse structure
        analysis = self.clause_analyzer.analyze(text)
        normalized = analysis.normalized.normalized

        # Determine discourse context
        in_quote = any(c.in_quote for c in analysis.clauses)
        negated = any(c.negated for c in analysis.clauses)
        hypothetical = any(c.hypothetical for c in analysis.clauses)
        meta = any(c.meta_discussion for c in analysis.clauses)

        # Check each term
        matched_terms = []
        for term in rule.terms:
            # Simple substring match (case-insensitive via normalization)
            if term.term.lower() in normalized:
                matched_terms.append(term)

        # Apply match_any vs match_all logic
        if rule.match_any:
            terms_to_report = matched_terms if matched_terms else []
        else:
            # All terms required
            if len(matched_terms) == len(rule.terms):
                terms_to_report = matched_terms
            else:
                terms_to_report = []

        if not terms_to_report:
            return []

        # Generate evidence for each matched term
        evidences = []
        for term in terms_to_report:
            # Check quote policy
            quoted_violation = (
                in_quote and
                rule.quote_policy == QuotePolicy.SUPPRESS
            )

            # Check negation policy
            negation_violation = (
                negated and
                rule.negation_policy == NegationPolicy.SUPPRESS
            )

            # Check context requirements
            requires_satisfied = True
            if term.requires_context:
                requires_satisfied = all(
                    ctx in context_flags for ctx in term.requires_context
                )

            forbids_triggered = False
            if term.forbids_context:
                forbids_triggered = any(
                    ctx in context_flags for ctx in term.forbids_context
                )

            # Create evidence
            evidence = MatchEvidence(
                rule_id=rule.rule_id,
                feature_class=rule.feature_class,
                candidate_mode=rule.candidate_mode,
                weight=term.weight,
                priority=rule.priority,
                clause_index=clause_index,
                span_start=None,  # Simplified for now
                span_end=None,
                matched_term="",  # Never log actual term
                quoted=in_quote,
                negated=negated,
                hypothetical=hypothetical,
                meta_context=meta,
                requires_satisfied=requires_satisfied,
                forbids_triggered=forbids_triggered,
            )

            # Only include if valid or if we want to record invalid evidence
            # For now, only include valid evidence unless it's explicitly invalid
            if not quoted_violation and not negation_violation:
                evidences.append(evidence)

        return evidences


class RuleEngine:
    """Execute multiple rules and collect evidence."""

    def __init__(self, rules: list[CompiledRule]):
        self.rules = rules
        self.matcher = EvidenceMatcher()

        # Sort rules deterministically: priority desc, then rule_id asc
        self.rules = sorted(
            rules,
            key=lambda r: (-r.priority, r.rule_id)
        )

    def match_message(
        self,
        text: str,
        context_flags: Set[str] | None = None,
    ) -> list[MatchEvidence]:
        """Match all rules against a message and collect evidence.

        Args:
            text: Message text to match
            context_flags: Active context flags

        Returns:
            List of all evidence, ordered by priority
        """
        if not text:
            return []

        context_flags = context_flags or set()

        all_evidence = []
        activation_groups_stopped: set[str] = set()

        # Process rules in priority order
        for rule in self.rules:
            # Check if this activation group is stopped
            if rule.activation_group and rule.activation_group in activation_groups_stopped:
                continue

            # Match rule
            evidences = self.matcher.match_rule(
                rule, text, clause_index=0, context_flags=context_flags
            )

            # Collect valid evidence
            for evidence in evidences:
                if evidence.is_valid():
                    all_evidence.append(evidence)

                    # Handle stop_processing
                    if rule.stop_processing and rule.activation_group:
                        activation_groups_stopped.add(rule.activation_group)
                        break

        return all_evidence

    def match_clauses(
        self,
        clauses: list[str],
        context_flags: Set[str] | None = None,
    ) -> list[MatchEvidence]:
        """Match rules against multiple clauses.

        Args:
            clauses: List of clause texts
            context_flags: Active context flags

        Returns:
            List of all evidence from all clauses
        """
        all_evidence = []

        for clause_idx, clause_text in enumerate(clauses):
            evidence = self.match_message(clause_text, context_flags)
            # Update clause indices
            for ev in evidence:
                # Create new evidence with correct clause index
                updated_ev = MatchEvidence(
                    rule_id=ev.rule_id,
                    feature_class=ev.feature_class,
                    candidate_mode=ev.candidate_mode,
                    weight=ev.weight,
                    priority=ev.priority,
                    clause_index=clause_idx,
                    span_start=ev.span_start,
                    span_end=ev.span_end,
                    matched_term=ev.matched_term,
                    quoted=ev.quoted,
                    negated=ev.negated,
                    hypothetical=ev.hypothetical,
                    meta_context=ev.meta_context,
                    requires_satisfied=ev.requires_satisfied,
                    forbids_triggered=ev.forbids_triggered,
                )
                all_evidence.append(updated_ev)

        return all_evidence
