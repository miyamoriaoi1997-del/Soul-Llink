from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass
from typing import Any

from .memory_adapter import db_path, load_entries
from .memory_object import MemoryObject
from .memory_object_adapter import adapt_memory_object
from .memory_selection import SelectionDecision, explain_memory_selection


_TARGET_FILES = {"user": "USER.md", "memory": "MEMORY.md"}


@dataclass(frozen=True)
class MemorySelectionProbe:
    """Read-only sidecar view of a stored memory row and its selection explanation."""

    record_id: int
    target_file: str
    memory: MemoryObject
    decision: SelectionDecision

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "target_file": self.target_file,
            "memory": self.memory.to_dict(),
            "decision": self.decision.to_dict(),
        }


def explain_memory_records(
    target: str,
    *,
    mode: str | None,
    emotion_axes: set[str] | None,
    budget_available: float | None,
) -> list[MemorySelectionProbe]:
    """Return selection explanations for stored records without changing retrieval.

    The production prompt path still decides what to inject. This helper only
    reads rows, adapts them into MemoryObject, and explains their eligibility.
    """

    target_file = _TARGET_FILES.get(target)
    if target_file is None:
        return []
    path = db_path()
    if not path.exists():
        return []

    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            """
            SELECT record_id, target_file, content, status, kind, metadata
            FROM memory_records
            WHERE target_file = ?
            ORDER BY record_id ASC
            """,
            (target_file,),
        ).fetchall()
    finally:
        con.close()

    probes: list[MemorySelectionProbe] = []
    for row in rows:
        memory = adapt_memory_object(_row_to_mapping(row))
        decision = explain_memory_selection(
            memory,
            mode=mode,
            emotion_axes=emotion_axes,
            budget_available=budget_available,
        )
        probes.append(
            MemorySelectionProbe(
                record_id=int(row["record_id"]),
                target_file=str(row["target_file"]),
                memory=memory,
                decision=decision,
            )
        )
    return probes


def build_probe_report(
    target: str,
    *,
    mode: str | None,
    emotion_axes: set[str] | None,
    budget_available: float | None,
) -> dict[str, Any]:
    """Build a read-only JSON-serializable shadow report for memory selection."""

    probes = explain_memory_records(
        target,
        mode=mode,
        emotion_axes=emotion_axes,
        budget_available=budget_available,
    )
    baseline = load_entries(target)
    selected = [probe for probe in probes if probe.decision.selected]
    skipped = [
        probe
        for probe in probes
        if not probe.decision.selected
        and probe.memory.status.value not in {"quarantined", "retired"}
    ]
    quarantined = [probe for probe in probes if probe.memory.status.value == "quarantined"]
    retired = [probe for probe in probes if probe.memory.status.value == "retired"]
    selected_contents = [probe.memory.content for probe in selected]

    return {
        "target": target,
        "mode": mode,
        "emotion_axes": sorted(emotion_axes or set()),
        "budget_available": budget_available,
        "db_path": str(db_path()),
        "selected": [probe.to_dict() for probe in selected],
        "skipped": [probe.to_dict() for probe in skipped],
        "quarantined": [probe.to_dict() for probe in quarantined],
        "retired": [probe.to_dict() for probe in retired],
        "load_entries_baseline": baseline,
        "drift_warnings": _drift_warnings(selected_contents, baseline),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Read-only PCLTM memory selection shadow probe."
    )
    parser.add_argument("--target", default="user", choices=sorted(_TARGET_FILES))
    parser.add_argument("--mode", default=None)
    parser.add_argument(
        "--emotion-axis",
        dest="emotion_axes",
        action="append",
        default=[],
        help="Emotion axis hint. Repeat for multiple axes.",
    )
    parser.add_argument("--budget", type=float, default=None)
    parser.add_argument("--json", action="store_true", help="Emit JSON report.")
    args = parser.parse_args(argv)

    report = build_probe_report(
        args.target,
        mode=args.mode,
        emotion_axes=set(args.emotion_axes),
        budget_available=args.budget,
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _drift_warnings(selected_contents: list[str], baseline: list[str]) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    if selected_contents != baseline:
        warnings.append(
            {
                "kind": "selection_baseline_mismatch",
                "message": "probe selected content differs from load_entries baseline",
                "selected_count": len(selected_contents),
                "baseline_count": len(baseline),
                "missing_from_probe": [item for item in baseline if item not in selected_contents],
                "extra_in_probe": [item for item in selected_contents if item not in baseline],
            }
        )
    return warnings


def _row_to_mapping(row: sqlite3.Row) -> dict[str, Any]:
    metadata = _metadata(row["metadata"])
    kind = row["kind"]
    mapped: dict[str, Any] = {
        "canonical_key": metadata.get("canonical_key") or f"record:{row['record_id']}",
        "content": row["content"],
        "status": row["status"],
        "kind": kind,
        "object_type": metadata.get("object_type") or _object_type_from_kind(kind),
        "metadata": metadata,
    }
    state_affinity = metadata.get("state_affinity")
    if isinstance(state_affinity, dict):
        if "modes" in state_affinity:
            mapped["modes"] = state_affinity["modes"]
        if "emotion_axes" in state_affinity:
            mapped["emotion_axes"] = state_affinity["emotion_axes"]

    for field in (
        "object_type",
        "scope",
        "injection_policy",
        "stability_score",
        "confidence",
        "source",
        "modes",
        "emotion_axes",
    ):
        if field in metadata:
            mapped[field] = metadata[field]
    return mapped


def _object_type_from_kind(kind: object) -> str:
    value = str(kind or "").strip().lower()
    return {
        "identity": "identity",
        "relationship": "relationship",
        "preference": "preference",
        "procedural": "procedural",
        "procedure": "procedural",
        "state": "state_trace",
        "state_trace": "state_trace",
        "tool": "tool_evidence",
        "tool_evidence": "tool_evidence",
        "project": "project",
        "conflict": "conflict",
        "retired": "retired",
        "memory_note": "episodic",
        "note": "episodic",
        "episodic": "episodic",
    }.get(value, "episodic")


def _metadata(raw: object) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
