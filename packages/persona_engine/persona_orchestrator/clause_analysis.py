"""Clause-aware text analysis and discourse annotation.

Performs normalization, segmentation, and discourse structure detection
without logging sensitive content.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SpanRange:
    """Character span in text."""
    start: int
    end: int

    def __repr__(self) -> str:
        return f"SpanRange({self.start}:{self.end})"


@dataclass(frozen=True)
class NormalizedText:
    """Text with normalization and span mapping."""
    original: str
    normalized: str
    span_map: tuple[tuple[int, int], ...]  # (norm_pos, orig_pos) pairs

    def __repr__(self) -> str:
        return f"NormalizedText(len={len(self.normalized)}, [REDACTED])"


@dataclass(frozen=True)
class Clause:
    """A single clause or sentence segment."""
    index: int
    text: str
    span: SpanRange
    in_quote: bool = False
    negated: bool = False
    hypothetical: bool = False
    meta_discussion: bool = False

    def __repr__(self) -> str:
        flags = []
        if self.in_quote:
            flags.append("quoted")
        if self.negated:
            flags.append("negated")
        if self.hypothetical:
            flags.append("hypothetical")
        if self.meta_discussion:
            flags.append("meta")
        flag_str = f" {','.join(flags)}" if flags else ""
        return f"Clause({self.index}, span={self.span}{flag_str}, [REDACTED])"


@dataclass(frozen=True)
class DiscourseAnnotation:
    """An annotation marking discourse structure."""
    annotation_type: str  # "quoted", "negated", "hypothetical", "meta"
    span: SpanRange

    def __repr__(self) -> str:
        return f"DiscourseAnnotation({self.annotation_type}, {self.span})"


@dataclass(frozen=True)
class AnalysisResult:
    """Complete analysis of a text."""
    normalized: NormalizedText
    clauses: tuple[Clause, ...]
    annotations: tuple[DiscourseAnnotation, ...]

    def __repr__(self) -> str:
        return (
            f"AnalysisResult(clauses={len(self.clauses)}, "
            f"annotations={len(self.annotations)}, [REDACTED])"
        )


class ClauseAnalyzer:
    """Analyze text into clauses with discourse annotations."""

    # Sentence-ending punctuation (English and Chinese)
    SENTENCE_ENDS = re.compile(r'[.!?。！？]+')

    # Quote characters
    QUOTE_CHARS = {'"', "'", '"', '"', ''', ''', '「', '」', '『', '』'}

    # Negation patterns (English and Chinese)
    NEGATION_PATTERNS = [
        r'\b(?:not|no|never|don\'t|doesn\'t|didn\'t|won\'t|can\'t|isn\'t|aren\'t)\b',
        r'不|没|别|勿',
    ]

    # Hypothetical/conditional patterns
    HYPOTHETICAL_PATTERNS = [
        r'\b(?:if|would|could|might|should|suppose|imagine|what if)\b',
        r'如果|假如|要是|倘若|若|假设',
    ]

    # Meta-discussion patterns
    META_PATTERNS = [
        r'\b(?:about|what is|what\'s|explain|tell me about|how does|how do)\b',
        r'什么是|怎么|如何|解释',
    ]

    def __init__(self):
        self._negation_re = re.compile('|'.join(self.NEGATION_PATTERNS), re.IGNORECASE)
        self._hypothetical_re = re.compile('|'.join(self.HYPOTHETICAL_PATTERNS), re.IGNORECASE)
        self._meta_re = re.compile('|'.join(self.META_PATTERNS), re.IGNORECASE)

    def normalize(self, text: str) -> NormalizedText:
        """Normalize text and build span map.

        Args:
            text: Original text

        Returns:
            Normalized text with span mapping
        """
        if not text:
            return NormalizedText(original="", normalized="", span_map=())

        # Unicode normalization (NFC form)
        nfc_text = unicodedata.normalize('NFC', text)

        # Lowercase
        lower_text = nfc_text.lower()

        # Normalize whitespace
        normalized = ' '.join(lower_text.split())

        # Build simple span map (for now, identity-like)
        # In full implementation, this would track every character transformation
        span_map = tuple((i, i) for i in range(min(len(normalized), len(text))))

        return NormalizedText(
            original=text,
            normalized=normalized,
            span_map=span_map
        )

    def segment_clauses(self, text: str) -> list[Clause]:
        """Segment text into clauses.

        Args:
            text: Text to segment

        Returns:
            List of clauses
        """
        if not text:
            return []

        # Normalize first
        norm = self.normalize(text)
        normalized = norm.normalized

        # Split on sentence-ending punctuation
        segments = []
        last_end = 0

        for match in self.SENTENCE_ENDS.finditer(normalized):
            segment_text = normalized[last_end:match.end()].strip()
            if segment_text:
                segments.append((last_end, match.end(), segment_text))
            last_end = match.end()

        # Capture any remaining text
        if last_end < len(normalized):
            segment_text = normalized[last_end:].strip()
            if segment_text:
                segments.append((last_end, len(normalized), segment_text))

        # If no punctuation found, treat entire text as one clause
        if not segments:
            segments = [(0, len(normalized), normalized)]

        # Create Clause objects
        clauses = []
        for idx, (start, end, seg_text) in enumerate(segments):
            clauses.append(Clause(
                index=idx,
                text=seg_text,
                span=SpanRange(start, end)
            ))

        return clauses

    def analyze(self, text: str) -> AnalysisResult:
        """Perform complete analysis of text.

        Args:
            text: Text to analyze

        Returns:
            Analysis result with clauses and annotations
        """
        if not text:
            norm = NormalizedText(original="", normalized="", span_map=())
            return AnalysisResult(normalized=norm, clauses=(), annotations=())

        # Normalize
        norm = self.normalize(text)
        normalized = norm.normalized

        # Detect discourse annotations on normalized text
        annotations = []

        # Quote detection
        quote_ranges = self._detect_quotes(normalized)
        for span in quote_ranges:
            annotations.append(DiscourseAnnotation("quoted", span))

        # Negation detection
        if self._negation_re.search(normalized):
            # Mark entire text as containing negation for simplicity
            annotations.append(DiscourseAnnotation("negated", SpanRange(0, len(normalized))))

        # Hypothetical detection
        if self._hypothetical_re.search(normalized):
            annotations.append(DiscourseAnnotation("hypothetical", SpanRange(0, len(normalized))))

        # Meta-discussion detection
        if self._meta_re.search(normalized):
            annotations.append(DiscourseAnnotation("meta", SpanRange(0, len(normalized))))

        # Segment into clauses
        base_clauses = self.segment_clauses(text)

        # Annotate clauses with discourse flags
        annotated_clauses = []
        for clause in base_clauses:
            in_quote = any(
                self._span_overlaps(clause.span, ann.span)
                for ann in annotations if ann.annotation_type == "quoted"
            )
            negated = any(
                self._span_overlaps(clause.span, ann.span)
                for ann in annotations if ann.annotation_type == "negated"
            )
            hypothetical = any(
                self._span_overlaps(clause.span, ann.span)
                for ann in annotations if ann.annotation_type == "hypothetical"
            )
            meta = any(
                self._span_overlaps(clause.span, ann.span)
                for ann in annotations if ann.annotation_type == "meta"
            )

            annotated_clauses.append(Clause(
                index=clause.index,
                text=clause.text,
                span=clause.span,
                in_quote=in_quote,
                negated=negated,
                hypothetical=hypothetical,
                meta_discussion=meta
            ))

        return AnalysisResult(
            normalized=norm,
            clauses=tuple(annotated_clauses),
            annotations=tuple(annotations)
        )

    def _detect_quotes(self, text: str) -> list[SpanRange]:
        """Detect quoted spans in text."""
        ranges = []
        in_quote = False
        quote_start = 0
        quote_char = None

        for i, char in enumerate(text):
            if char in self.QUOTE_CHARS:
                if not in_quote:
                    # Opening quote
                    in_quote = True
                    quote_start = i
                    quote_char = char
                else:
                    # Closing quote (simplified - just closes on any quote char)
                    ranges.append(SpanRange(quote_start, i + 1))
                    in_quote = False
                    quote_char = None

        return ranges

    def _span_overlaps(self, span1: SpanRange, span2: SpanRange) -> bool:
        """Check if two spans overlap."""
        return not (span1.end <= span2.start or span2.end <= span1.start)
