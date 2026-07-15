from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any


_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b([A-Z0-9_]*(?:SECRET|TOKEN|API[_-]?KEY|PASSWORD|PASS)[A-Z0-9_]*)\s*=\s*[^\s,;]+"
)
_SEARCH_TOKEN_RE = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)


def _fts_query(query: str) -> str:
    tokens = _SEARCH_TOKEN_RE.findall(query)
    if not tokens:
        return '""'
    return " OR ".join(json.dumps(token, ensure_ascii=False) for token in tokens)


@dataclass
class DACRawMessage:
    raw_id: int = 0
    session_id: str = ""
    turn_id: str = ""
    role: str = "user"
    content: str = ""
    persona_mode: str = ""
    source_platform: str = ""
    sequence: int = 0
    token_count: int = 0
    sensitivity: str = "normal"
    inject_policy: str = "context_only"
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0


@dataclass
class DACSummaryNode:
    node_id: int = 0
    session_id: str = ""
    node_type: str = "summary"
    depth: int = 0
    summary: str = ""
    token_count: int = 0
    source_token_count: int = 0
    source_type: str = "messages"
    source_ids: list[int] = field(default_factory=list)
    persona_mode: str = ""
    inject_policy: str = "retrieve_only"
    sensitivity: str = "normal"
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0
    earliest_at: float | None = None
    latest_at: float | None = None
    expand_hint: str = ""
    status: str = "active"


class DACStore:
    """PCLTM-native DAC storage backed by the EventStore SQLite DB."""

    def __init__(self, event_store):
        self.event_store = event_store
        self.conn = event_store._conn

    def add_raw_message(self, message: DACRawMessage) -> int:
        created_at = message.created_at or time.time()
        content = _SECRET_ASSIGNMENT_RE.sub(lambda m: f"{m.group(1)}=[REDACTED]", message.content)
        cur = self.conn.execute(
            """
            INSERT INTO dac_raw_messages
                (session_id, turn_id, role, content, persona_mode, source_platform,
                 sequence, token_count, sensitivity, inject_policy, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                message.session_id,
                message.turn_id,
                message.role,
                content,
                message.persona_mode,
                message.source_platform,
                message.sequence,
                message.token_count,
                message.sensitivity,
                message.inject_policy,
                json.dumps(message.metadata, ensure_ascii=False),
                created_at,
            ),
        )
        message.raw_id = int(cur.lastrowid)
        self.conn.commit()
        return message.raw_id

    def get_raw_message(self, raw_id: int) -> DACRawMessage | None:
        row = self.conn.execute(
            "SELECT * FROM dac_raw_messages WHERE raw_id = ?",
            (raw_id,),
        ).fetchone()
        return self._row_to_raw_message(row) if row else None

    def fresh_tail(self, *, session_id: str, limit: int = 16) -> list[DACRawMessage]:
        rows = self.conn.execute(
            """
            SELECT * FROM dac_raw_messages
            WHERE session_id = ?
            ORDER BY sequence DESC, raw_id DESC
            LIMIT ?
            """,
            (session_id, max(1, min(int(limit), 200))),
        ).fetchall()
        return [self._row_to_raw_message(row) for row in reversed(rows)]

    def add_node(self, node: DACSummaryNode) -> int:
        created_at = node.created_at or time.time()
        cur = self.conn.execute(
            """
            INSERT INTO dac_summary_nodes
                (session_id, node_type, depth, summary, token_count, source_token_count,
                 source_type, source_ids, persona_mode, inject_policy, sensitivity,
                 metadata, created_at, earliest_at, latest_at, expand_hint, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                node.session_id,
                node.node_type,
                node.depth,
                node.summary,
                node.token_count,
                node.source_token_count,
                node.source_type,
                json.dumps(node.source_ids),
                node.persona_mode,
                node.inject_policy,
                node.sensitivity,
                json.dumps(node.metadata, ensure_ascii=False),
                created_at,
                node.earliest_at,
                node.latest_at,
                node.expand_hint,
                node.status,
            ),
        )
        node.node_id = int(cur.lastrowid)
        self.conn.execute(
            "INSERT INTO dac_summary_nodes_fts(rowid, summary) VALUES (?, ?)",
            (node.node_id, node.summary),
        )
        self.conn.commit()
        return node.node_id

    def get_node(self, node_id: int) -> DACSummaryNode | None:
        row = self.conn.execute(
            "SELECT * FROM dac_summary_nodes WHERE node_id = ?",
            (node_id,),
        ).fetchone()
        return self._row_to_node(row) if row else None

    def search_nodes(self, query: str, *, session_id: str | None = None, limit: int = 20) -> list[DACSummaryNode]:
        if not query.strip():
            return []
        params: list[Any] = [_fts_query(query)]
        where = "dac_summary_nodes_fts MATCH ?"
        if session_id:
            where += " AND n.session_id = ?"
            params.append(session_id)
        params.append(limit)
        try:
            rows = self.conn.execute(
                f"""
                SELECT n.*
                FROM dac_summary_nodes_fts f
                JOIN dac_summary_nodes n ON n.node_id = f.rowid
                WHERE {where}
                ORDER BY rank, n.created_at DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
        except Exception:
            rows = []
        if not rows:
            return self._search_nodes_like(query, session_id=session_id, limit=limit)
        return [self._row_to_node(row) for row in rows]

    def _search_nodes_like(self, query: str, *, session_id: str | None = None, limit: int = 20) -> list[DACSummaryNode]:
        terms = _SEARCH_TOKEN_RE.findall(query) or [query]
        clauses = []
        params: list[Any] = []
        for term in terms:
            clauses.append("summary LIKE ?")
            params.append(f"%{term}%")
        where = " OR ".join(clauses) if clauses else "summary LIKE ?"
        if session_id:
            where = f"({where}) AND session_id = ?"
            params.append(session_id)
        params.append(max(1, min(int(limit), 100)))
        rows = self.conn.execute(
            f"""
            SELECT * FROM dac_summary_nodes
            WHERE {where}
            ORDER BY created_at DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
        return [self._row_to_node(row) for row in rows]

    def expand_node(self, node_id: int, *, recursive: bool = False) -> dict[str, Any]:
        node = self.get_node(node_id)
        if node is None:
            raise KeyError(f"DAC node not found: {node_id}")
        children: list[DACSummaryNode] = []
        if node.source_type == "nodes":
            for child_id in node.source_ids:
                child = self.get_node(child_id)
                if child is not None:
                    children.append(child)
        result: dict[str, Any] = {"node": node, "children": children}
        if recursive:
            result["expanded_children"] = [
                self.expand_node(child.node_id, recursive=True) for child in children
            ]
        return result

    def add_context_snapshot(
        self,
        *,
        session_id: str,
        turn_id: str = "",
        mode: str = "",
        budget_tokens: int = 0,
        selected_node_ids: list[int] | None = None,
        selected_raw_ids: list[int] | None = None,
        fresh_tail_count: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        metadata_payload = dict(metadata or {})
        snapshot_type = str(metadata_payload.get("snapshot_type") or (f"dac_{mode}" if mode else "dac_snapshot"))
        metadata_payload.setdefault("snapshot_type", snapshot_type)
        cur = self.conn.execute(
            """
            INSERT INTO dac_context_snapshots
                (session_id, turn_id, mode, snapshot_type, budget_tokens, selected_node_ids,
                 selected_raw_ids, fresh_tail_count, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                turn_id,
                mode,
                snapshot_type,
                budget_tokens,
                json.dumps(selected_node_ids or []),
                json.dumps(selected_raw_ids or []),
                fresh_tail_count,
                json.dumps(metadata_payload, ensure_ascii=False),
                time.time(),
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def get_context_snapshot(self, snapshot_id: int) -> dict[str, Any]:
        row = self.conn.execute(
            "SELECT * FROM dac_context_snapshots WHERE snapshot_id = ?",
            (snapshot_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"DAC context snapshot not found: {snapshot_id}")
        data = dict(row)
        data["selected_node_ids"] = json.loads(data["selected_node_ids"])
        data["selected_raw_ids"] = json.loads(data["selected_raw_ids"])
        data["metadata"] = json.loads(data["metadata"])
        return data

    def all_nodes(self) -> list[DACSummaryNode]:
        rows = self.conn.execute("SELECT * FROM dac_summary_nodes ORDER BY node_id").fetchall()
        return [self._row_to_node(row) for row in rows]

    def _row_to_raw_message(self, row) -> DACRawMessage:
        return DACRawMessage(
            raw_id=int(row["raw_id"]),
            session_id=row["session_id"],
            turn_id=row["turn_id"] or "",
            role=row["role"],
            content=row["content"],
            persona_mode=row["persona_mode"] or "",
            source_platform=row["source_platform"] or "",
            sequence=int(row["sequence"] or 0),
            token_count=int(row["token_count"] or 0),
            sensitivity=row["sensitivity"],
            inject_policy=row["inject_policy"],
            metadata=json.loads(row["metadata"] or "{}"),
            created_at=float(row["created_at"]),
        )

    def _row_to_node(self, row) -> DACSummaryNode:
        return DACSummaryNode(
            node_id=int(row["node_id"]),
            session_id=row["session_id"],
            node_type=row["node_type"],
            depth=int(row["depth"]),
            summary=row["summary"],
            token_count=int(row["token_count"] or 0),
            source_token_count=int(row["source_token_count"] or 0),
            source_type=row["source_type"],
            source_ids=json.loads(row["source_ids"]),
            persona_mode=row["persona_mode"] or "",
            inject_policy=row["inject_policy"],
            sensitivity=row["sensitivity"],
            metadata=json.loads(row["metadata"] or "{}"),
            created_at=float(row["created_at"]),
            earliest_at=row["earliest_at"],
            latest_at=row["latest_at"],
            expand_hint=row["expand_hint"] or "",
            status=row["status"],
        )
