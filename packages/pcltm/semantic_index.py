"""Lightweight semantic index for PCLTM memory retrieval.

Uses character n-gram BM25 for Chinese text — no external tokenizer needed.
For 76 records this is instant; scales to ~10K without issues.

The index is rebuilt on each governor run and cached in-memory during
prompt assembly. No persistent vector store needed at this scale.
"""

from __future__ import annotations

import math
import re
import sqlite3
import hashlib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from .runtime_paths import DEFAULT_DB

# BM25 parameters
_K1 = 1.5
_B = 0.75
# N-gram sizes for Chinese text
_NGRAM_SIZES = (2, 3)
# Minimum score to consider a match relevant
_MIN_SCORE = 0.5


@dataclass
class IndexedRecord:
    record_id: int
    content: str
    target_file: str
    ngrams: Counter = field(default_factory=Counter)
    doc_len: int = 0


class SemanticIndex:
    """BM25-based semantic index over PCLTM memory records.

    Tokenization: character n-grams (bigrams + trigrams) for Chinese,
    plus word-level tokens for ASCII content. No external dependencies.
    """

    def __init__(self, db_path: Path | str = DEFAULT_DB):
        self.db_path = Path(db_path)
        self.records: list[IndexedRecord] = []
        self.idf: dict[str, float] = {}
        self.avg_dl: float = 0.0
        self._built = False

    def build(self) -> int:
        """Build index from all approved records. Returns record count."""
        if not self.db_path.exists():
            return 0
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT record_id, content, target_file FROM memory_records WHERE status = 'approved'"
        ).fetchall()
        con.close()

        self.records = []
        for row in rows:
            ngrams = self._tokenize(row["content"])
            self.records.append(IndexedRecord(
                record_id=row["record_id"],
                content=row["content"],
                target_file=row["target_file"],
                ngrams=ngrams,
                doc_len=sum(ngrams.values()),
            ))

        # Compute IDF
        n_docs = len(self.records)
        if n_docs == 0:
            self._built = True
            return 0

        self.avg_dl = sum(r.doc_len for r in self.records) / n_docs
        df: Counter = Counter()
        for rec in self.records:
            for term in rec.ngrams:
                df[term] += 1

        self.idf = {}
        for term, freq in df.items():
            # Standard BM25 IDF with floor at 0
            self.idf[term] = math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))

        self._built = True
        return n_docs

    def query(
        self,
        text: str,
        *,
        top_k: int = 10,
        target_file: str | None = None,
        min_score: float = _MIN_SCORE,
        metadata_terms: dict[int, Iterable[str]] | None = None,
    ) -> list[tuple[int, float]]:
        """Query the index. Returns [(record_id, score), ...] sorted by score desc.

        Args:
            text: Query text
            top_k: Max results
            target_file: Filter by USER.md or MEMORY.md
            min_score: Minimum BM25 score threshold
            metadata_terms: Optional per-record metadata terms to blend into
                lexical matching without rebuilding the persistent content index.
        """
        if not self._built:
            self.build()

        query_terms = self._tokenize(text)
        if not query_terms:
            return []

        scores: dict[int, float] = {}
        query_term_set = set(query_terms.keys())
        metadata_terms = metadata_terms or {}

        for rec in self.records:
            if target_file and rec.target_file != target_file:
                continue
            rec_terms = set(rec.ngrams.keys())
            metadata_ngram_counts = Counter()
            for term in metadata_terms.get(rec.record_id, ()):
                metadata_ngram_counts.update(self._tokenize(str(term)))
            combined_terms = rec_terms | set(metadata_ngram_counts.keys())
            # Skip if no content or metadata term overlap.
            if not (query_term_set & combined_terms):
                continue
            score = self._bm25_score(query_terms, rec)
            if metadata_ngram_counts:
                metadata_overlap = sum(query_terms[term] for term in query_term_set & set(metadata_ngram_counts.keys()))
                score += metadata_overlap * 0.25
            if score >= min_score:
                scores[rec.record_id] = score

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def _bm25_score(self, query_terms: Counter, rec: IndexedRecord) -> float:
        if self.avg_dl <= 0:
            return 0.0
        norm = _K1 * (1.0 - _B + _B * rec.doc_len / self.avg_dl)
        return sum(
            self.idf.get(term, 0.0)
            * (frequency * (_K1 + 1.0) / (frequency + norm))
            * query_frequency
            for term, query_frequency in query_terms.items()
            if (frequency := rec.ngrams.get(term, 0))
        )

    @staticmethod
    def _tokenize(text: str) -> Counter:
        """Tokenize text into character n-grams + ASCII words.

        For Chinese: bigrams and trigrams of consecutive CJK characters.
        For ASCII: lowercase word tokens (2+ chars).
        """
        tokens: Counter = Counter()

        # Extract CJK character sequences and generate n-grams
        cjk_runs = re.findall(r'[\u4e00-\u9fff\u3400-\u4dbf]+', text)
        for run in cjk_runs:
            for n in _NGRAM_SIZES:
                for i in range(len(run) - n + 1):
                    tokens[run[i:i + n]] += 1

        # Extract ASCII word tokens
        ascii_words = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]{1,}', text)
        for word in ascii_words:
            tokens[word.lower()] += 1

        return tokens


# Module-level singleton for reuse within a single prompt assembly
_INDEX: SemanticIndex | None = None
_INDEX_FINGERPRINT: tuple[object, ...] | None = None


def _db_fingerprint(path: Path) -> tuple[object, ...] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    try:
        con = sqlite3.connect(path)
        schema_version = con.execute("PRAGMA schema_version").fetchone()[0]
        user_version = con.execute("PRAGMA user_version").fetchone()[0]
        digest = hashlib.sha256()
        for row in con.execute("SELECT record_id, status, content, target_file FROM memory_records WHERE status = 'approved' ORDER BY record_id"):
            for value in row:
                encoded = str(value).encode("utf-8")
                digest.update(len(encoded).to_bytes(8, "big"))
                digest.update(encoded)
        con.close()
    except sqlite3.Error:
        return (stat.st_mtime_ns, stat.st_size, None)
    return (stat.st_mtime_ns, stat.st_size, schema_version, user_version, digest.digest())


def get_index(db_path: Path | str = DEFAULT_DB) -> SemanticIndex:
    """Get or create the module-level semantic index singleton."""
    global _INDEX, _INDEX_FINGERPRINT
    resolved = Path(db_path).expanduser().resolve()
    fingerprint = _db_fingerprint(resolved)
    if _INDEX is None or _INDEX.db_path != resolved or _INDEX_FINGERPRINT != fingerprint:
        _INDEX = SemanticIndex(resolved)
        _INDEX.build()
        _INDEX_FINGERPRINT = _db_fingerprint(resolved)
    return _INDEX


def find_related_ids(record_id: int, *, top_k: int = 3, db_path: Path | str = DEFAULT_DB) -> list[int]:
    """Find record IDs most related to a given record. For memory linking."""
    idx = get_index(db_path)
    return [rid for rid, _ in idx.related(record_id, top_k=top_k)]
