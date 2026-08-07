"""Bounded convergence service for governed memory projections."""

from __future__ import annotations

import yaml
from datetime import UTC, datetime, timedelta
from pathlib import Path

from ..evidence_chain import sha256_text
from ..store import EventStore
from .memory_fts import MemoryFtsProjector
from .memory_memfs import MemoryMemfsProjector


def _timestamp_pair() -> tuple[str, str]:
    now = datetime.now(UTC)
    return (
        now.isoformat().replace("+00:00", "Z"),
        (now + timedelta(minutes=1)).isoformat().replace("+00:00", "Z"),
    )


def drain_memory_projections(
    store: EventStore,
    *,
    memfs_root: Path,
    worker_id: str = "memory-projection-drain",
    max_jobs: int = 1000,
) -> dict[str, int]:
    if max_jobs <= 0:
        raise ValueError("max_jobs must be positive")
    result = {"memory_fts": 0, "memory_memfs": 0}
    projectors = (
        ("memory_fts", MemoryFtsProjector(store, worker_id=worker_id)),
        ("memory_memfs", MemoryMemfsProjector(
            store, memfs_root=memfs_root, worker_id=worker_id,
        )),
    )
    processed = 0
    while processed < max_jobs:
        progressed = False
        for kind, projector in projectors:
            if processed >= max_jobs:
                break
            now, lease_until = _timestamp_pair()
            outcome = projector.run_once(now=now, lease_until=lease_until)
            if outcome["failed"]:
                raise RuntimeError(f"{kind} projection failed")
            if outcome["applied"]:
                result[kind] += outcome["applied"]
                processed += outcome["applied"]
                progressed = True
        if not progressed:
            break
    incomplete = store._conn.execute(
        """
        SELECT count(*) FROM projection_outbox
        WHERE projection_kind IN ('memory_fts', 'memory_memfs')
          AND status IN ('pending', 'processing')
        """
    ).fetchone()[0]
    if incomplete:
        raise RuntimeError(
            f"memory projection drain limit reached with {incomplete} incomplete jobs"
        )
    return result


def require_memory_projections_applied(
    store: EventStore,
    *,
    memfs_root: Path,
    claim_id: int,
) -> dict[str, int | str]:
    authority = store._conn.execute(
        """
        SELECT v.version, v.content, v.content_sha256, g.memory_governance_id
        FROM memory_current mc
        JOIN memory_claim_versions v ON v.claim_version_id = mc.claim_version_id
        JOIN memory_governance_events g
          ON g.memory_governance_id = mc.memory_governance_id
         AND g.claim_id = mc.claim_id
         AND g.claim_version_id = mc.claim_version_id
         AND g.new_state = mc.lifecycle_state
        WHERE mc.claim_id = ?
        """,
        (int(claim_id),),
    ).fetchone()
    if authority is None:
        raise RuntimeError("memory projection authority missing")
    rows = store._conn.execute(
        """
        SELECT projection_kind, status, aggregate_version, payload_sha256
        FROM projection_outbox
        WHERE aggregate_id = ?
          AND projection_kind IN ('memory_fts', 'memory_memfs')
        """,
        (f"memory:{int(claim_id)}",),
    ).fetchall()
    jobs = {str(row["projection_kind"]): row for row in rows}
    for kind in ("memory_fts", "memory_memfs"):
        job = jobs.get(kind)
        if (
            job is None
            or job["status"] != "applied"
            or int(job["aggregate_version"]) != int(authority["version"])
            or str(job["payload_sha256"]) != str(authority["content_sha256"])
        ):
            raise RuntimeError(f"{kind} projection is not converged")

    fts = store._conn.execute(
        """
        SELECT content, claim_version, governance_id, payload_sha256, projection_generation
        FROM memory_fts WHERE rowid = ?
        """,
        (int(claim_id),),
    ).fetchone()
    if (
        fts is None
        or int(fts["claim_version"]) != int(authority["version"])
        or int(fts["governance_id"]) != int(authority["memory_governance_id"])
        or str(fts["payload_sha256"]) != str(authority["content_sha256"])
        or str(fts["content"]) != str(authority["content"])
        or sha256_text(str(fts["content"])) != str(authority["content_sha256"])
    ):
        raise RuntimeError("memory_fts commitment mismatch")

    path = Path(memfs_root) / "claims" / f"{int(claim_id):016d}.md"
    try:
        root = Path(memfs_root)
        claims = root / "claims"
        for candidate in (root, claims, path):
            if candidate.is_symlink():
                raise RuntimeError("memory_memfs commitment mismatch")
            attributes = int(getattr(candidate.stat(), "st_file_attributes", 0))
            if attributes & 0x400:
                raise RuntimeError("memory_memfs commitment mismatch")
        text = path.read_text(encoding="utf-8")
        marker, raw_frontmatter, body = text.split("---\n", 2)
        metadata = yaml.safe_load(raw_frontmatter)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise RuntimeError("memory_memfs commitment mismatch") from exc
    if marker != "" or not isinstance(metadata, dict):
        raise RuntimeError("memory_memfs commitment mismatch")
    body = body.rstrip("\n")
    if (
        int(metadata.get("claim_id", 0)) != int(claim_id)
        or int(metadata.get("claim_version", 0)) != int(authority["version"])
        or int(metadata.get("governance_id", 0)) != int(authority["memory_governance_id"])
        or str(metadata.get("content_sha256")) != str(authority["content_sha256"])
        or sha256_text(body) != str(authority["content_sha256"])
    ):
        raise RuntimeError("memory_memfs commitment mismatch")
    return {
        "claim_id": int(claim_id),
        "claim_version": int(authority["version"]),
        "governance_id": int(authority["memory_governance_id"]),
        "payload_sha256": str(authority["content_sha256"]),
        "projection_generation": int(fts["projection_generation"]),
        "projection_status": "applied",
    }
