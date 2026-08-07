from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from pcltm.memory_adapter import open_archival_memory, search_archival_memories, sync_memory_tool_write
from pcltm.runtime_paths import resolve_db_path, resolve_memfs_root
from pcltm.store import EventStore
from pcltm.transcript_search import search_exact_evidence

mcp = FastMCP("SoulLink/PCLTM", json_response=True)


def runtime_status() -> dict[str, Any]:
    codex_home = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")).expanduser().resolve()
    return {
        "success": True,
        "host": "codex",
        "codex_home": str(codex_home),
        "db_path": str(resolve_db_path()),
        "memfs_root": str(resolve_memfs_root()),
        "final_forward_observation": "unavailable_host_boundary",
        "context_capture": "codex_lifecycle_hook_output",
    }


@mcp.tool()
def soullink_memory_search(query: str, mode: str | None = None, limit: int = 8) -> list[dict[str, Any]]:
    """Search governed SoulLink/PCLTM memory and return bounded references/excerpts."""
    return search_archival_memories(query=query, mode=mode, layers=("pinned", "episodic"), limit=max(1, min(limit, 100)))


@mcp.tool()
def soullink_memory_open(memory_id: str, body_limit: int = 4000) -> dict[str, Any]:
    """Open a SoulLink/PCLTM memory by id after search."""
    return open_archival_memory(memory_id=memory_id, body_limit=max(1, min(body_limit, 20000)))


@mcp.tool()
def soullink_memory_recall_exact(query: str, limit: int = 8) -> list[dict[str, Any]]:
    """Recall exact transcript evidence with local integrity metadata."""
    store = EventStore(resolve_db_path())
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
    if os.environ.get("SOULLINK_CODEX_ALLOW_MEMORY_WRITES") != "1":
        return {"success": False, "error": "write_disabled", "target": target}
    if target not in {"user", "memory"}:
        return {"success": False, "error": "target must be user or memory"}
    return {"success": bool(sync_memory_tool_write(target=target, action="add", content=content)), "target": target}


@mcp.tool()
def soullink_identity_status() -> dict[str, Any]:
    """Report whether the Codex SoulLink identity/runtime adapter is managed and available."""
    status = runtime_status()
    adapter = Path(status["codex_home"]) / "soullink" / "adapter.json"
    status.update({"managed": adapter.is_file(), "adapter_manifest": str(adapter)})
    return status


@mcp.tool()
def soullink_runtime_status() -> dict[str, Any]:
    """Report Codex adapter runtime paths and observation boundaries."""
    return runtime_status()


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
