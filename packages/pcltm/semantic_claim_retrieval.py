"""Optional local E5 candidate generation for governed memory claims.

The neural model is deliberately subordinate to ``memory_current``. It only
chooses claim IDs; callers must reopen every result through the normal policy
and authority commitments before returning any content.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from pathlib import Path
from typing import Any

DEFAULT_MIN_SCORE = 0.87
DEFAULT_TOP_SCORE_MARGIN = 0.015
MIN_CONTENT_CHARS = 20
MAX_CONTENT_CHARS = 1200

_MODEL_LOCK = threading.Lock()
_MODEL_STATE: tuple[Any, Any, Any] | None = None


def _enabled() -> bool:
    return os.getenv("SOULLINK_PCLTM_E5_RETRIEVAL_ENABLED", "").lower() in {
        "1", "true", "yes", "on",
    }


def _model_path() -> Path | None:
    raw = os.getenv("SOULLINK_PCLTM_E5_MODEL_PATH", "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser().resolve()
    return path if path.is_dir() else None


def _load_model():
    global _MODEL_STATE
    if _MODEL_STATE is not None:
        return _MODEL_STATE
    path = _model_path()
    if path is None:
        return None
    with _MODEL_LOCK:
        if _MODEL_STATE is not None:
            return _MODEL_STATE
        import torch
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        model = AutoModel.from_pretrained(path, local_files_only=True)
        model.eval()
        _MODEL_STATE = (torch, tokenizer, model)
        return _MODEL_STATE


def _embed(texts: list[str], *, query: bool):
    state = _load_model()
    if state is None:
        return None
    torch, tokenizer, model = state
    prefix = "query: " if query else "passage: "
    batch = tokenizer(
        [prefix + text for text in texts],
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )
    with torch.inference_mode():
        hidden = model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return torch.nn.functional.normalize(pooled, p=2, dim=1)


def semantic_claim_candidates(
    store,
    query: str,
    *,
    limit: int,
    min_score: float | None = None,
) -> list[tuple[int, float]]:
    """Return scored claim IDs, or an empty list when optional E5 is unavailable."""
    if not _enabled() or not query.strip() or limit <= 0:
        return []
    rows = store._conn.execute(
        """
        SELECT c.claim_id, v.content
        FROM memory_current mc
        JOIN memory_claims c ON c.claim_id = mc.claim_id
        JOIN memory_claim_versions v ON v.claim_version_id = mc.claim_version_id
        WHERE mc.lifecycle_state = 'active'
        ORDER BY c.claim_id
        """
    ).fetchall()
    corpus: list[tuple[int, str]] = []
    for row in rows:
        try:
            claim_id = row["claim_id"]
            content = row["content"]
        except (KeyError, TypeError, IndexError) as exc:
            raise sqlite3.DatabaseError("malformed semantic claim row") from exc
        if type(claim_id) is not int or claim_id <= 0 or type(content) is not str:
            raise sqlite3.DatabaseError("malformed semantic claim row")
        if MIN_CONTENT_CHARS <= len(content.strip()) <= MAX_CONTENT_CHARS:
            corpus.append((claim_id, content))
    if not corpus:
        return []
    passages = _embed([content for _, content in corpus], query=False)
    query_vector = _embed([query.strip()], query=True)
    if passages is None or query_vector is None:
        return []
    scores = (query_vector @ passages.T)[0].tolist()
    threshold = min_score
    if threshold is None:
        try:
            threshold = float(os.getenv("SOULLINK_PCLTM_E5_MIN_SCORE", DEFAULT_MIN_SCORE))
        except ValueError:
            threshold = DEFAULT_MIN_SCORE
    ranked = sorted(
        (
            (claim_id, float(score))
            for (claim_id, _content), score in zip(corpus, scores)
            if float(score) >= float(threshold)
        ),
        key=lambda item: (-item[1], item[0]),
    )
    if not ranked:
        return []
    # Precision-first admission: E5 cosine scores are high and compressed for
    # short multilingual text. An absolute threshold rejects unrelated queries;
    # a tight top-score margin prevents weaker topical neighbours from filling
    # the prompt merely because they crossed that global threshold.
    try:
        margin = float(
            os.getenv("SOULLINK_PCLTM_E5_TOP_SCORE_MARGIN", DEFAULT_TOP_SCORE_MARGIN)
        )
    except ValueError:
        margin = DEFAULT_TOP_SCORE_MARGIN
    cutoff = ranked[0][1] - max(0.0, margin)
    return [item for item in ranked if item[1] >= cutoff][:limit]


__all__ = ["semantic_claim_candidates"]
