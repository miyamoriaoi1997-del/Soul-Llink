"""Deterministic transcript chunking with exact source offsets."""

from __future__ import annotations

from dataclasses import dataclass

from .evidence_chain import sha256_text


CHUNKER_VERSION = "transcript-span-v1"


@dataclass(frozen=True)
class TranscriptChunk:
    ordinal: int
    start_char: int
    end_char: int
    text: str
    sha256: str
    chunker_version: str = CHUNKER_VERSION


def _preferred_end(text: str, start: int, hard_end: int) -> int:
    if hard_end >= len(text):
        return len(text)
    window = text[start:hard_end]
    lower_bound = max(1, len(window) // 2)
    candidates = (
        window.rfind("\r\n\r\n", lower_bound),
        window.rfind("\n\n", lower_bound),
        window.rfind("\n", lower_bound),
        window.rfind("。", lower_bound),
        window.rfind("！", lower_bound),
        window.rfind("？", lower_bound),
        window.rfind(". ", lower_bound),
    )
    boundary = max(candidates)
    if boundary < 0:
        return hard_end
    if window.startswith("\r\n\r\n", boundary):
        return start + boundary + 4
    if window.startswith("\n\n", boundary):
        return start + boundary + 2
    if window.startswith("\n", boundary):
        return start + boundary + 1
    if window.startswith(". ", boundary):
        return start + boundary + 2
    return start + boundary + 1


def chunk_transcript(
    text: str,
    *,
    max_chars: int = 1200,
    overlap_chars: int = 120,
) -> list[TranscriptChunk]:
    """Split text without normalization so every chunk remains an exact span."""
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("overlap_chars must satisfy 0 <= overlap < max_chars")
    if not text:
        return []

    chunks: list[TranscriptChunk] = []
    start = 0
    ordinal = 0
    while start < len(text):
        end = _preferred_end(text, start, min(len(text), start + max_chars))
        if end <= start:
            end = min(len(text), start + max_chars)
        value = text[start:end]
        chunks.append(
            TranscriptChunk(
                ordinal=ordinal,
                start_char=start,
                end_char=end,
                text=value,
                sha256=sha256_text(value),
            )
        )
        if end >= len(text):
            break
        next_start = max(start + 1, end - overlap_chars)
        start = next_start
        ordinal += 1
    return chunks
