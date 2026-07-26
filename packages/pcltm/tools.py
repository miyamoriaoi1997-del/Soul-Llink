"""Bounded read-only tool facade for Persona LCM."""

from __future__ import annotations

from typing import Any

from .summarizer import redact_secret_assignments
from .candidates import PersonaCandidateExtractor
from .retrieval_policy import RetrievalPolicy
from .ingest import PCLTMIngestAdapter
from .importer import JSONLTranscriptImporter


class PersonaLCMTools:
    """Small API surface intended to become Hermes tool handlers later."""

    def __init__(self, store: Any):
        self.store = store

    def plcm_search(
        self,
        query: str,
        *,
        scope: dict[str, Any] | None = None,
        limit: int = 10,
        include_sensitive: bool = False,
        global_ok: bool = False,
    ) -> dict[str, Any]:
        if not self._has_scope(scope) and not global_ok:
            return {"ok": False, "error": "scope_required"}
        scope = scope or {}
        return {
            "ok": True,
            "results": self.store.search_events(
                query,
                session_id=scope.get("session_id"),
                conversation_id=scope.get("conversation_id"),
                platform=scope.get("platform"),
                persona_mode=scope.get("persona_mode"),
                source=scope.get("source"),
                limit=limit,
                include_sensitive=include_sensitive,
            ),
        }

    def plcm_expand(self, *, node_id: int, include_sensitive: bool = False) -> dict[str, Any]:
        expanded = self.store.expand_summary(node_id)
        events = []
        for event in expanded["events"]:
            if self._is_sensitive(event) and not include_sensitive:
                event = dict(event)
                event["content"] = redact_secret_assignments(event["content"])
            events.append(event)
        return {
            "ok": True,
            "node": expanded["node"],
            "nodes": expanded["nodes"],
            "events": events,
        }

    def plcm_timeline(
        self,
        *,
        scope: dict[str, Any] | None = None,
        limit: int = 100,
        global_ok: bool = False,
    ) -> dict[str, Any]:
        if not self._has_scope(scope) and not global_ok:
            return {"ok": False, "error": "scope_required"}
        scope = scope or {}
        events = self.store.list_events(
            session_id=scope.get("session_id"),
            conversation_id=scope.get("conversation_id"),
            platform=scope.get("platform"),
            persona_mode=scope.get("persona_mode"),
            source=scope.get("source"),
            limit=limit,
        )
        return {"ok": True, "events": events}

    def plcm_evidence(
        self,
        query: str,
        *,
        scope: dict[str, Any] | None = None,
        limit: int = 20,
        include_sensitive: bool = False,
        global_ok: bool = False,
    ) -> dict[str, Any]:
        if not self._has_scope(scope) and not global_ok:
            return {"ok": False, "error": "scope_required"}
        scope = scope or {}
        evidence: list[dict[str, Any]] = []
        for result in self.store.search_events(
            query,
            session_id=scope.get("session_id"),
            conversation_id=scope.get("conversation_id"),
            platform=scope.get("platform"),
            persona_mode=scope.get("persona_mode"),
            source=scope.get("source"),
            limit=limit,
            include_sensitive=include_sensitive,
        ):
            evidence.append(
                {
                    "kind": "event",
                    "event_id": result["event_id"],
                    "session_id": result["session_id"],
                    "conversation_id": result["conversation_id"],
                    "platform": result["platform"],
                    "role": result["role"],
                    "source": result["source"],
                    "persona_mode": result["persona_mode"],
                    "sensitivity": result["sensitivity"],
                    "created_at": result["created_at"],
                    "snippet": redact_secret_assignments(result["snippet"]),
                }
            )
        evidence.extend(self._summary_evidence(query, scope=scope, include_sensitive=include_sensitive, limit=limit))
        return {"ok": True, "evidence": evidence[:limit]}

    def _summary_evidence(
        self,
        query: str,
        *,
        scope: dict[str, Any],
        include_sensitive: bool,
        limit: int,
    ) -> list[dict[str, Any]]:
        rows = self.store._conn.execute(
            """
            SELECT * FROM summary_nodes
            WHERE summary LIKE ?
            ORDER BY node_id DESC
            LIMIT ?
            """,
            (f"%{query}%", max(1, min(int(limit), 100))),
        ).fetchall()
        output = []
        for row in rows:
            node = self.store.get_summary_node(row["node_id"])
            if not include_sensitive and self._is_sensitive(node):
                continue
            source_event_ids = node["source_event_ids"]
            if scope and not self._summary_matches_scope(source_event_ids, scope):
                continue
            output.append(
                {
                    "kind": "summary",
                    "node_id": node["node_id"],
                    "depth": node["depth"],
                    "sensitivity": node["sensitivity"],
                    "created_at": node["created_at"],
                    "snippet": redact_secret_assignments(node["summary"]),
                    "expand_path": {
                        "node_id": node["node_id"],
                        "source_event_ids": source_event_ids,
                        "source_node_ids": node["source_node_ids"],
                    },
                }
            )
        return output

    def _summary_matches_scope(self, event_ids: list[int], scope: dict[str, Any]) -> bool:
        if not event_ids:
            return True
        for event_id in event_ids:
            event = self.store.get_event(event_id)
            for key in ("session_id", "conversation_id", "platform", "persona_mode", "source"):
                if scope.get(key) is not None and event.get(key) != scope[key]:
                    return False
        return True

    def plcm_persona_candidates(
        self,
        *,
        scope: dict[str, Any] | None = None,
        limit: int = 100,
        global_ok: bool = False,
    ) -> dict[str, Any]:
        if not self._has_scope(scope) and not global_ok:
            return {"ok": False, "error": "scope_required"}
        candidates = PersonaCandidateExtractor(self.store).extract(scope=scope or {}, limit=limit)
        return {"ok": True, "candidates": candidates}

    def plcm_candidate_queue(
        self,
        *,
        scope: dict[str, Any] | None = None,
        status: str = "pending",
        limit: int = 100,
        global_ok: bool = False,
    ) -> dict[str, Any]:
        if not self._has_scope(scope) and not global_ok:
            return {"ok": False, "error": "scope_required"}
        candidates = PersonaCandidateExtractor(self.store).extract(scope=scope or {}, limit=limit)
        for candidate in candidates:
            self.store.enqueue_candidate(candidate)
        return {"ok": True, "queue": self.store.list_candidate_queue(status=status)}

    def plcm_candidate_review(
        self,
        *,
        record_id: int,
        decision: str,
        reviewer: str,
        decision_reason: str,
    ) -> dict[str, Any]:
        record = self.store.review_candidate(
            record_id,
            decision=decision,
            reviewer=reviewer,
            decision_reason=decision_reason,
        )
        return {"ok": True, "record": record}


    def plcm_retrieve_for_mode(
        self,
        *,
        mode: str,
        query: str,
        scope: dict[str, Any] | None = None,
        limit: int = 20,
        include_sensitive: bool = False,
        global_ok: bool = False,
    ) -> dict[str, Any]:
        if not self._has_scope(scope) and not global_ok:
            return {"ok": False, "error": "scope_required"}
        results = RetrievalPolicy().retrieve(
            self.store,
            mode=mode,
            query=query,
            scope=scope or {},
            include_sensitive=include_sensitive,
            limit=limit,
        )
        return {"ok": True, "mode": mode, "results": results}

    def plcm_ingest(self, payload: dict[str, Any], *, ingest_ok: bool = False) -> dict[str, Any]:
        if not ingest_ok:
            return {"ok": False, "error": "ingest_ok_required"}
        result = PCLTMIngestAdapter(self.store).ingest(payload)
        return {"ok": True, **result}

    def plcm_import_jsonl(self, path: str, *, import_ok: bool = False) -> dict[str, Any]:
        if not import_ok:
            return {"ok": False, "error": "import_ok_required"}
        report = JSONLTranscriptImporter(self.store).import_file(path)
        return {"ok": report["ok"], "report": report}

    @staticmethod
    def _has_scope(scope: dict[str, Any] | None) -> bool:
        if not scope:
            return False
        return any(scope.get(key) for key in ("session_id", "conversation_id", "platform"))

    @staticmethod
    def _is_sensitive(event: dict[str, Any]) -> bool:
        return event.get("sensitivity") in {"private", "restricted", "secret"}