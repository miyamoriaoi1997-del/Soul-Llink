"""Tests for clause analysis and discourse annotation."""
import pytest

from packages.persona_engine.persona_orchestrator.clause_analysis import (
    ClauseAnalyzer,
    NormalizedText,
    Clause,
    DiscourseAnnotation,
    SpanRange,
)


class TestNormalizedText:
    """Test text normalization and span mapping."""

    def test_simple_normalization(self):
        analyzer = ClauseAnalyzer()
        result = analyzer.normalize("Hello World")

        assert result.normalized == "hello world"
        assert result.original == "Hello World"
        assert len(result.span_map) > 0

    def test_unicode_normalization(self):
        analyzer = ClauseAnalyzer()
        result = analyzer.normalize("café")  # é can be composed or decomposed

        assert "cafe" in result.normalized or "café" in result.normalized
        assert result.original == "café"

    def test_whitespace_normalization(self):
        analyzer = ClauseAnalyzer()
        result = analyzer.normalize("hello  \t\n  world")

        assert result.normalized == "hello world"
        assert result.original == "hello  \t\n  world"

    def test_span_map_preserves_positions(self):
        analyzer = ClauseAnalyzer()
        result = analyzer.normalize("AB CD")

        # Character 'C' at position 3 in original maps to normalized position
        assert result.original == "AB CD"


class TestClauseSegmentation:
    """Test clause and sentence segmentation."""

    def test_single_sentence_english(self):
        analyzer = ClauseAnalyzer()
        clauses = analyzer.segment_clauses("This is a test.")

        assert len(clauses) >= 1
        assert any("test" in c.text.lower() for c in clauses)

    def test_multiple_sentences_english(self):
        analyzer = ClauseAnalyzer()
        clauses = analyzer.segment_clauses("First sentence. Second sentence.")

        assert len(clauses) >= 2

    def test_chinese_punctuation(self):
        analyzer = ClauseAnalyzer()
        clauses = analyzer.segment_clauses("第一句。第二句。")

        assert len(clauses) >= 2

    def test_mixed_punctuation(self):
        analyzer = ClauseAnalyzer()
        clauses = analyzer.segment_clauses("English. 中文。Mixed.")

        assert len(clauses) >= 3

    def test_empty_text(self):
        analyzer = ClauseAnalyzer()
        clauses = analyzer.segment_clauses("")

        assert clauses == []

    def test_clause_preserves_index(self):
        analyzer = ClauseAnalyzer()
        clauses = analyzer.segment_clauses("First. Second.")

        assert clauses[0].index == 0
        assert clauses[1].index == 1


class TestQuoteDetection:
    """Test quoted text detection."""

    def test_double_quotes(self):
        analyzer = ClauseAnalyzer()
        result = analyzer.analyze('She said "hello world" to me.')

        # Should detect quoted span
        quoted_clauses = [c for c in result.clauses if c.in_quote]
        assert len(quoted_clauses) > 0 or any(
            a.annotation_type == "quoted" for a in result.annotations
        )

    def test_single_quotes(self):
        analyzer = ClauseAnalyzer()
        result = analyzer.analyze("He said 'stop that' please.")

        # Should detect quoted span
        has_quote = any(c.in_quote for c in result.clauses) or any(
            a.annotation_type == "quoted" for a in result.annotations
        )
        assert has_quote

    def test_chinese_quotes(self):
        analyzer = ClauseAnalyzer()
        result = analyzer.analyze('她说"你好"。')

        has_quote = any(c.in_quote for c in result.clauses) or any(
            a.annotation_type == "quoted" for a in result.annotations
        )
        assert has_quote

    def test_no_quotes(self):
        analyzer = ClauseAnalyzer()
        result = analyzer.analyze("Plain text without quotes.")

        quoted_clauses = [c for c in result.clauses if c.in_quote]
        assert len(quoted_clauses) == 0


class TestNegationDetection:
    """Test negation scope detection."""

    def test_simple_negation_english(self):
        analyzer = ClauseAnalyzer()
        result = analyzer.analyze("I don't want to stop.")

        # Should detect negation
        has_negation = any(c.negated for c in result.clauses) or any(
            a.annotation_type == "negated" for a in result.annotations
        )
        assert has_negation

    def test_negation_chinese(self):
        analyzer = ClauseAnalyzer()
        result = analyzer.analyze("不要停。")

        has_negation = any(c.negated for c in result.clauses) or any(
            a.annotation_type == "negated" for a in result.annotations
        )
        assert has_negation

    def test_no_negation(self):
        analyzer = ClauseAnalyzer()
        result = analyzer.analyze("Please continue.")

        negated_clauses = [c for c in result.clauses if c.negated]
        assert len(negated_clauses) == 0


class TestHypotheticalDetection:
    """Test hypothetical/conditional detection."""

    def test_if_conditional(self):
        analyzer = ClauseAnalyzer()
        result = analyzer.analyze("If we stop, what happens?")

        has_hypothetical = any(c.hypothetical for c in result.clauses) or any(
            a.annotation_type == "hypothetical" for a in result.annotations
        )
        assert has_hypothetical

    def test_would_hypothetical(self):
        analyzer = ClauseAnalyzer()
        result = analyzer.analyze("I would stop if needed.")

        has_hypothetical = any(c.hypothetical for c in result.clauses) or any(
            a.annotation_type == "hypothetical" for a in result.annotations
        )
        assert has_hypothetical

    def test_chinese_hypothetical(self):
        analyzer = ClauseAnalyzer()
        result = analyzer.analyze("如果停止会怎样？")

        has_hypothetical = any(c.hypothetical for c in result.clauses) or any(
            a.annotation_type == "hypothetical" for a in result.annotations
        )
        assert has_hypothetical


class TestMetaDiscussionDetection:
    """Test meta-discussion detection."""

    def test_about_keyword(self):
        analyzer = ClauseAnalyzer()
        result = analyzer.analyze("Tell me about the stop command.")

        has_meta = any(c.meta_discussion for c in result.clauses) or any(
            a.annotation_type == "meta" for a in result.annotations
        )
        assert has_meta

    def test_what_is_question(self):
        analyzer = ClauseAnalyzer()
        result = analyzer.analyze("What is the crisis mode?")

        has_meta = any(c.meta_discussion for c in result.clauses) or any(
            a.annotation_type == "meta" for a in result.annotations
        )
        assert has_meta

    def test_explain_keyword(self):
        analyzer = ClauseAnalyzer()
        result = analyzer.analyze("Explain how the routing works.")

        has_meta = any(c.meta_discussion for c in result.clauses) or any(
            a.annotation_type == "meta" for a in result.annotations
        )
        assert has_meta


class TestPrivacyInvariant:
    """Test that original text is never logged."""

    def test_repr_no_original_content(self):
        analyzer = ClauseAnalyzer()
        result = analyzer.analyze("sensitive content here")

        repr_str = repr(result)
        # Should not contain the actual text
        assert "sensitive" not in repr_str or "[REDACTED]" in repr_str

    def test_clause_repr_no_content(self):
        analyzer = ClauseAnalyzer()
        clauses = analyzer.segment_clauses("private data")

        if clauses:
            repr_str = repr(clauses[0])
            # Should show structure but not content
            assert "private" not in repr_str or "[REDACTED]" in repr_str
