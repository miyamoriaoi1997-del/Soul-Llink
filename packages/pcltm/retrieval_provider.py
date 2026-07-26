from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Protocol


class RetrievalStatus(str, Enum):
    OK = "ok"
    ABSTAINED = "abstained"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class AuthorityReference:
    event_id: int
    chunk_id: int
    source_revision: int
    start_char: int
    end_char: int
    payload_sha256: str
    chain_hash: str
    chunk_sha256: str
    chunk_ordinal: int = 0
    source_hash: str | None = None
    source_created_at: str | None = None

    def __post_init__(self) -> None:
        for field in ("event_id", "chunk_id", "source_revision"):
            value = getattr(self, field)
            if type(value) is not int:
                raise TypeError(f"{field} must be int")
            if value <= 0:
                raise ValueError(f"{field} must be positive")
        for field in ("chunk_ordinal", "start_char", "end_char"):
            value = getattr(self, field)
            if type(value) is not int:
                raise TypeError(f"{field} must be int")
            if value < 0:
                raise ValueError(f"{field} must be non-negative")
        if self.end_char <= self.start_char:
            raise ValueError("end_char must be greater than start_char")
        for field in ("payload_sha256", "chain_hash", "chunk_sha256"):
            value = getattr(self, field)
            if type(value) is not str:
                raise TypeError(f"{field} must be str")
            if len(value) != 64 or any(char not in "0123456789abcdefABCDEF" for char in value):
                raise ValueError(f"{field} must be a 64-character hexadecimal digest")


@dataclass(frozen=True, slots=True)
class RetrievalRequest:
    query: str
    limit: int = 10
    session_id: str | None = None
    conversation_id: str | None = None
    platform: str | None = None
    persona_mode: str | None = None
    source: str | None = None
    include_sensitive: bool = False

    def __post_init__(self) -> None:
        if type(self.query) is not str:
            raise TypeError("query must be str")
        if type(self.limit) is not int or self.limit <= 0:
            raise ValueError("limit must be a positive integer")
        if type(self.include_sensitive) is not bool:
            raise TypeError("include_sensitive must be bool")


@dataclass(frozen=True, slots=True)
class Candidate:
    reference: AuthorityReference
    score: float
    provider: str
    quote: str = ""

    def __post_init__(self) -> None:
        if type(self.reference) is not AuthorityReference:
            raise TypeError("reference must be AuthorityReference")
        if type(self.score) not in (int, float) or isinstance(self.score, bool):
            raise TypeError("score must be numeric")
        if not math.isfinite(float(self.score)):
            raise ValueError("score must be finite")
        if type(self.provider) is not str or not self.provider:
            raise ValueError("provider must be a non-empty string")


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    status: RetrievalStatus
    candidates: tuple[Candidate, ...] = ()
    reason: str | None = None

    def __post_init__(self) -> None:
        if type(self.candidates) is not tuple or not all(
            type(candidate) is Candidate for candidate in self.candidates
        ):
            raise TypeError("candidates must be a tuple of Candidate")
        if self.status is RetrievalStatus.OK:
            if not self.candidates or self.reason is not None:
                raise ValueError("ok requires candidates and no reason")
        elif self.status in (RetrievalStatus.ABSTAINED, RetrievalStatus.UNAVAILABLE):
            if self.candidates or type(self.reason) is not str or not self.reason:
                raise ValueError("non-ok requires no candidates and a reason")
        else:
            raise ValueError("unknown retrieval status")

    @classmethod
    def ok(cls, candidates: tuple[Candidate, ...] | list[Candidate]) -> RetrievalResult:
        return cls(RetrievalStatus.OK, tuple(candidates))

    @classmethod
    def abstained(cls, reason: str) -> RetrievalResult:
        return cls(RetrievalStatus.ABSTAINED, (), reason)

    @classmethod
    def unavailable(cls, reason: str) -> RetrievalResult:
        return cls(RetrievalStatus.UNAVAILABLE, (), reason)


class RetrievalProvider(Protocol):
    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        ...


class NotConfiguredProvider:
    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        return RetrievalResult.unavailable("not_configured")