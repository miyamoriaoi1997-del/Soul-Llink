"""Local opt-in observer that verifies hook-injected content reached the model.

ZCode records every model call (request body and response) as plaintext JSONL
under ``~/.zcode/cli/rollout/model-io-*.jsonl``. This observer is a local
diagnostic enhancement: when ``SOULLINK_ZCODE_OBSERVER=1`` it scans the newest
model-io records for a given session and checks whether the last
``UserPromptSubmit``-injected memory excerpts actually appear in the final
request body.

This does **not** change the default evidence boundary: ``runtime_status``
still reports ``final_forward_observation = unavailable_host_boundary`` unless
the operator enables the observer, and even then the observer only reports
what it literally observed in the official request log — it never upgrades the
adapter's default claim.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _zcode_root() -> Path:
    return Path(os.environ.get("ZCODE_ROOT", Path.home() / ".zcode" / "cli")).expanduser().resolve()


def _rollout_dir(root: Path) -> Path:
    return root / "rollout"


def _model_io_records(root: Path, *, session_id: str | None = None, limit: int = 3) -> list[dict[str, Any]]:
    rollout = _rollout_dir(root)
    if not rollout.is_dir():
        return []
    files = sorted(rollout.glob("model-io-*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    records: list[dict[str, Any]] = []
    for path in files:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if session_id and record.get("sessionId") != session_id:
                continue
            records.append(record)
        if len(records) >= limit:
            break
    return records[:limit]


def _request_bodies(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bodies: list[dict[str, Any]] = []
    for record in records:
        request = record.get("request") or {}
        body = request.get("body") or {}
        if body.get("input") is not None:
            bodies.append(body)
    return bodies


def verify_injection(
    *,
    injected: list[str],
    session_id: str | None = None,
    limit: int = 3,
    root: Path | None = None,
) -> dict[str, Any]:
    """Check whether ``injected`` strings appear in the newest model request bodies.

    Returns an honest report: which excerpts were found, which were not, and
    the observation boundary that applies. An empty injected list verifies
    nothing.
    """
    if not injected:
        return {"observed": False, "reason": "no_injected_content", "matches": [], "missing": []}
    records = _model_io_records(root or _zcode_root(), session_id=session_id, limit=limit)
    if not records:
        return {"observed": False, "reason": "no_model_io_records", "matches": [], "missing": list(injected)}
    bodies = _request_bodies(records)
    if not bodies:
        return {"observed": False, "reason": "no_request_bodies", "matches": [], "missing": list(injected)}
    combined = json.dumps(bodies, ensure_ascii=False, default=str)
    matches: list[str] = []
    missing: list[str] = []
    for excerpt in injected:
        needle = excerpt.strip()
        if needle and needle in combined:
            matches.append(excerpt)
        else:
            missing.append(excerpt)
    return {
        "observed": bool(matches),
        "reason": None,
        "matches": matches,
        "missing": missing,
        "records_scanned": len(records),
        "boundary": "model_io_log_observation",
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="soullink-zcode-observer")
    parser.add_argument("--injected", nargs="*", default=[], help="excerpts to look for")
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args(argv)
    report = verify_injection(injected=args.injected, session_id=args.session_id, limit=args.limit)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("observed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
