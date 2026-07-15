"""Durable episodic memory stream for PCLTM.

Episodic memory records what happened. It intentionally does not promote an
observed event into a stable user/persona fact; that decision belongs to later
reflection and injection arbitration layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable


JsonDict = dict[str, Any]


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _clean_text(value: str, *, max_chars: int = 1200) -> str:
    text = " ".join(str(value or "").strip().split())
    if len(text) > max_chars:
        return text[: max_chars - 1].rstrip() + "…"
    return text


def _tuple_of_strings(values: Iterable[str] | None) -> tuple[str, ...]:
    if not values:
        return ()
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _clean_text(str(value), max_chars=240)
        if text and text not in seen:
            cleaned.append(text)
            seen.add(text)
    return tuple(cleaned)


def _clamp_score(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _make_event_id(source_session: str, raw_refs: Iterable[str], event_summary: str) -> str:
    digest = hashlib.sha256()
    digest.update(source_session.encode("utf-8"))
    for raw_ref in raw_refs:
        digest.update(b"\0")
        digest.update(raw_ref.encode("utf-8"))
    digest.update(b"\0")
    digest.update(event_summary.encode("utf-8"))
    return "evt_" + digest.hexdigest()[:24]


@dataclass(frozen=True)
class EpisodicMemory:
    """One evidence-backed event in the episodic memory stream."""

    event_id: str
    timestamp: str
    participants: tuple[str, ...]
    source_session: str
    raw_refs: tuple[str, ...]
    event_summary: str
    event_type: str
    importance_score: float = 0.3
    emotional_salience: float = 0.0
    continuity_relevance: float = 0.0
    entities: tuple[str, ...] = field(default_factory=tuple)
    tags: tuple[str, ...] = field(default_factory=tuple)
    superseded_by: str | None = None
    privacy_level: str = "internal"
    confidence_score: float = 1.0
    fact_promotion_allowed: bool = False
    fact_promotion_blockers: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.event_id:
            raise ValueError("event_id is required")
        if not self.source_session:
            raise ValueError("source_session is required")
        if not self.raw_refs:
            raise ValueError("raw_refs are required for episodic evidence")
        if not self.event_summary:
            raise ValueError("event_summary is required")
        if not self.participants:
            raise ValueError("participants are required")

    @classmethod
    def create(
        cls,
        *,
        source_session: str,
        raw_refs: Iterable[str],
        event_summary: str,
        participants: Iterable[str] = ("user", "assistant"),
        event_type: str = "session_event",
        timestamp: str | None = None,
        importance_score: float = 0.3,
        emotional_salience: float = 0.0,
        continuity_relevance: float = 0.0,
        entities: Iterable[str] = (),
        tags: Iterable[str] = (),
        privacy_level: str = "internal",
        confidence_score: float = 1.0,
        fact_promotion_allowed: bool = False,
        fact_promotion_blockers: Iterable[str] = (),
    ) -> "EpisodicMemory":
        cleaned_refs = _tuple_of_strings(raw_refs)
        cleaned_summary = _clean_text(event_summary)
        blockers = _tuple_of_strings(fact_promotion_blockers)
        if not fact_promotion_allowed and not blockers:
            blockers = ("episodic_record_only",)
        return cls(
            event_id=_make_event_id(source_session, cleaned_refs, cleaned_summary),
            timestamp=timestamp or _now_iso(),
            participants=_tuple_of_strings(participants),
            source_session=_clean_text(source_session, max_chars=240),
            raw_refs=cleaned_refs,
            event_summary=cleaned_summary,
            event_type=_clean_text(event_type, max_chars=80) or "session_event",
            importance_score=_clamp_score(importance_score),
            emotional_salience=_clamp_score(emotional_salience),
            continuity_relevance=_clamp_score(continuity_relevance),
            entities=_tuple_of_strings(entities),
            tags=_tuple_of_strings(tags),
            privacy_level=_clean_text(privacy_level, max_chars=40) or "internal",
            confidence_score=_clamp_score(confidence_score),
            fact_promotion_allowed=bool(fact_promotion_allowed),
            fact_promotion_blockers=blockers,
        )

    def to_dict(self) -> JsonDict:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "participants": list(self.participants),
            "source_session": self.source_session,
            "raw_refs": list(self.raw_refs),
            "event_summary": self.event_summary,
            "event_type": self.event_type,
            "importance_score": self.importance_score,
            "emotional_salience": self.emotional_salience,
            "continuity_relevance": self.continuity_relevance,
            "entities": list(self.entities),
            "tags": list(self.tags),
            "superseded_by": self.superseded_by,
            "privacy_level": self.privacy_level,
            "confidence_score": self.confidence_score,
            "fact_promotion_allowed": self.fact_promotion_allowed,
            "fact_promotion_blockers": list(self.fact_promotion_blockers),
        }

    @classmethod
    def from_dict(cls, data: JsonDict) -> "EpisodicMemory":
        return cls(
            event_id=str(data["event_id"]),
            timestamp=str(data["timestamp"]),
            participants=_tuple_of_strings(data.get("participants", ())),
            source_session=str(data["source_session"]),
            raw_refs=_tuple_of_strings(data.get("raw_refs", ())),
            event_summary=str(data["event_summary"]),
            event_type=str(data.get("event_type", "session_event")),
            importance_score=_clamp_score(float(data.get("importance_score", 0.3))),
            emotional_salience=_clamp_score(float(data.get("emotional_salience", 0.0))),
            continuity_relevance=_clamp_score(float(data.get("continuity_relevance", 0.0))),
            entities=_tuple_of_strings(data.get("entities", ())),
            tags=_tuple_of_strings(data.get("tags", ())),
            superseded_by=data.get("superseded_by"),
            privacy_level=str(data.get("privacy_level", "internal")),
            confidence_score=_clamp_score(float(data.get("confidence_score", 1.0))),
            fact_promotion_allowed=bool(data.get("fact_promotion_allowed", False)),
            fact_promotion_blockers=_tuple_of_strings(data.get("fact_promotion_blockers", ())),
        )


class EpisodicStore:
    """SQLite-backed append-oriented event stream."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS episodic_memories (
                    event_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    source_session TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    importance_score REAL NOT NULL,
                    emotional_salience REAL NOT NULL,
                    continuity_relevance REAL NOT NULL,
                    privacy_level TEXT NOT NULL,
                    superseded_by TEXT,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_episodic_timestamp ON episodic_memories(timestamp)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_episodic_source ON episodic_memories(source_session)"
            )

    def append(self, memory: EpisodicMemory) -> EpisodicMemory:
        payload = json.dumps(memory.to_dict(), ensure_ascii=False, sort_keys=True)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO episodic_memories (
                    event_id, timestamp, source_session, event_type,
                    importance_score, emotional_salience, continuity_relevance,
                    privacy_level, superseded_by, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory.event_id,
                    memory.timestamp,
                    memory.source_session,
                    memory.event_type,
                    memory.importance_score,
                    memory.emotional_salience,
                    memory.continuity_relevance,
                    memory.privacy_level,
                    memory.superseded_by,
                    payload,
                ),
            )
        return memory

    def append_many(self, memories: Iterable[EpisodicMemory]) -> tuple[EpisodicMemory, ...]:
        return tuple(self.append(memory) for memory in memories)

    def get(self, event_id: str) -> EpisodicMemory | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM episodic_memories WHERE event_id = ?",
                (event_id,),
            ).fetchone()
        if row is None:
            return None
        return EpisodicMemory.from_dict(json.loads(row["payload_json"]))

    def list_events(
        self,
        *,
        source_session: str | None = None,
        include_superseded: bool = False,
        limit: int | None = None,
    ) -> tuple[EpisodicMemory, ...]:
        clauses: list[str] = []
        params: list[Any] = []
        if source_session is not None:
            clauses.append("source_session = ?")
            params.append(source_session)
        if not include_superseded:
            clauses.append("superseded_by IS NULL")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        limit_sql = " LIMIT ?" if limit is not None else ""
        if limit is not None:
            params.append(max(0, int(limit)))
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT payload_json FROM episodic_memories {where} ORDER BY timestamp DESC, event_id DESC{limit_sql}",
                tuple(params),
            ).fetchall()
        return tuple(EpisodicMemory.from_dict(json.loads(row["payload_json"])) for row in rows)

    def supersede(self, event_id: str, superseded_by: str) -> EpisodicMemory:
        current = self.get(event_id)
        if current is None:
            raise KeyError(event_id)
        updated = replace(current, superseded_by=superseded_by)
        self.append(updated)
        return updated
