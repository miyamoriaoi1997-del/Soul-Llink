"""ZCode STDIO MCP server for governed SoulLink/PCLTM tools.

ZCode mounts this server from the user-scope ``config.json``
(``mcp.servers.soullink``) managed block. The tools are the same governed
surfaces the Codex adapter exposes, but wired to the 2.2 authoritative
retrieval/write services rather than the retired legacy seams.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP
from pcltm.memory_contracts import PersonaMode, Sensitivity
from pcltm.memory_retrieval import (
    GovernedMemoryOpenRequest,
    GovernedMemorySearchRequest,
    MemoryRetrievalStatus,
    open_governed_memory,
    search_governed_memories,
)
from pcltm.memory_write_service import MemoryWriteRequest, MemoryWriteService
from pcltm.runtime_paths import resolve_db_path, resolve_memfs_root
from pcltm.store import EventStore
from pcltm.transcript_search import search_exact_evidence

mcp = FastMCP("SoulLink/PCLTM", json_response=True)


def _zcode_root() -> Path:
    return Path(os.environ.get("ZCODE_ROOT", Path.home() / ".zcode" / "cli")).expanduser().resolve()


def runtime_status() -> dict[str, Any]:
    root = _zcode_root()
    return {
        "success": True,
        "host": "zcode",
        "zcode_root": str(root),
        "db_path": str(resolve_db_path()),
        "memfs_root": str(resolve_memfs_root()),
        "final_forward_observation": "unavailable_host_boundary",
        "context_capture": "zcode_hook_additional_context",
    }


def _persona_mode(value: object) -> PersonaMode:
    try:
        return PersonaMode(str(value or "default").strip().lower())
    except ValueError:
        return PersonaMode.DEFAULT


def _serialize_item(item: Any, *, body_limit: int | None = None) -> dict[str, Any]:
    body = item.content
    truncated = False
    if body_limit is not None and len(body) > body_limit:
        body = body[:body_limit].rstrip() + "…"
        truncated = True
    return {
        "memory_id": f"claim/{item.claim_id}",
        "claim_id": item.claim_id,
        "claim_version": item.claim_version,
        "governance_id": item.governance_id,
        "canonical_key": item.canonical_key,
        "target": item.target,
        "memory_type": item.memory_type,
        "sensitivity": item.sensitivity.value,
        "mode_scope": [mode.value for mode in item.mode_scope],
        "injection_policy": item.injection_policy,
        "content_sha256": item.content_sha256,
        "authority_verified": item.authority_verified,
        "policy_reason": item.policy_reason,
        "policy_version": item.policy_version,
        "excerpt": body if body_limit is None else None,
        "body": body if body_limit is not None else None,
        "truncated": truncated if body_limit is not None else None,
        "reference_only": body_limit is None,
    }


@mcp.tool()
def soullink_memory_search(query: str, mode: str | None = None, limit: int = 8) -> dict[str, Any]:
    """Search governed SoulLink/PCLTM memory and return bounded references/excerpts."""
    store = EventStore(resolve_db_path(), read_only=True)
    try:
        result = search_governed_memories(
            store,
            GovernedMemorySearchRequest(
                query=query,
                persona_mode=_persona_mode(mode),
                limit=max(1, min(limit, 100)),
            ),
        )
    finally:
        store.close()
    if result.status is MemoryRetrievalStatus.UNAVAILABLE:
        return {"status": result.status.value, "reason": result.reason, "results": []}
    return {
        "status": result.status.value,
        "reason": result.reason,
        "results": [_serialize_item(item) for item in result.items],
    }


@mcp.tool()
def soullink_memory_open(memory_id: str, body_limit: int = 4000, mode: str | None = None) -> dict[str, Any]:
    """Open a governed SoulLink/PCLTM memory by claim id (``claim/<id>``)."""
    prefix = "claim/"
    if not memory_id.startswith(prefix):
        return {"success": False, "error": "memory_id must be claim/<id>"}
    try:
        claim_id = int(memory_id[len(prefix):])
    except ValueError:
        return {"success": False, "error": "memory_id must be claim/<id>"}
    store = EventStore(resolve_db_path(), read_only=True)
    try:
        result = open_governed_memory(
            store,
            GovernedMemoryOpenRequest(
                claim_id=claim_id,
                persona_mode=_persona_mode(mode),
                sensitivity_ceiling=Sensitivity.RESTRICTED,
            ),
        )
    finally:
        store.close()
    if result.status is MemoryRetrievalStatus.UNAVAILABLE:
        return {"status": result.status.value, "reason": result.reason, "memory": None}
    if not result.items:
        return {"status": result.status.value, "reason": result.reason, "memory": None}
    return {
        "status": result.status.value,
        "reason": result.reason,
        "memory": _serialize_item(result.items[0], body_limit=max(1, min(body_limit, 20000))),
    }


@mcp.tool()
def soullink_memory_recall_exact(query: str, limit: int = 8) -> list[dict[str, Any]]:
    """Recall exact transcript evidence with local integrity metadata."""
    store = EventStore(resolve_db_path(), read_only=True)
    try:
        return [
            {
                "evidence_level": item.evidence_level,
                "event_id": item.event_id,
                "chunk_id": item.chunk_id,
                "quote": item.quote,
                "start_char": item.start_char,
                "end_char": item.end_char,
                "source_created_at": item.source_created_at,
                "payload_sha256": item.payload_sha256,
                "verified": item.verified,
                "source_type": item.source_type,
                "integrity_scope": item.integrity_scope,
            }
            for item in search_exact_evidence(store, query, limit=max(1, min(limit, 100)))
        ]
    finally:
        store.close()


@mcp.tool()
def soullink_memory_remember(content: str, target: str = "memory") -> dict[str, Any]:
    """Persist an explicit durable user preference or memory through governed PCLTM storage."""
    if os.environ.get("SOULLINK_ZCODE_ALLOW_MEMORY_WRITES") != "1":
        return {"success": False, "error": "write_disabled", "target": target}
    if target not in {"user", "memory"}:
        return {"success": False, "error": "target must be user or memory"}
    normalized = content.strip()
    if not normalized:
        return {"success": False, "error": "content must not be empty", "target": target}
    from pcltm.projections.memory_runtime import drain_memory_projections

    digest = hashlib_hex(f"{target}\0{normalized}")
    store = EventStore(resolve_db_path())
    try:
        receipt = MemoryWriteService(store).write(MemoryWriteRequest(
            idempotency_key=f"memory-tool:{target}:{digest}",
            content=normalized,
            canonical_key=f"memory-tool:{target}:{digest}",
            target=target,
            memory_type=("user_preference" if target == "user" else "memory_note"),
            sensitivity=Sensitivity.NORMAL,
            mode_scope=(PersonaMode.DEFAULT,),
            injection_policy="allow",
            session_id="zcode-mcp",
            conversation_id="zcode-mcp",
            platform="zcode",
        ))
        if not receipt.success or receipt.claim_id is None:
            return {"success": False, "status": receipt.status, "reason": receipt.reason_code, "target": target}
        drain_memory_projections(store, memfs_root=resolve_memfs_root())
        return {"success": True, "status": receipt.status, "claim_id": receipt.claim_id, "target": target}
    finally:
        store.close()


def hashlib_hex(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@mcp.tool()
def soullink_identity_status() -> dict[str, Any]:
    """Report whether the ZCode SoulLink identity/runtime adapter is managed and available."""
    status = runtime_status()
    adapter = Path(status["zcode_root"]) / "soullink" / "adapter.json"
    status.update({"managed": adapter.is_file(), "adapter_manifest": str(adapter)})
    return status


@mcp.tool()
def soullink_runtime_status() -> dict[str, Any]:
    """Report ZCode adapter runtime paths and observation boundaries."""
    return runtime_status()


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
