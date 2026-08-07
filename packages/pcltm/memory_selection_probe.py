from __future__ import annotations

import argparse
import json
from typing import Any


def explain_memory_records(
    target: str,
    *,
    mode: str | None,
    emotion_axes: set[str] | None,
    budget_available: float | None,
) -> list[object]:
    """Retired body-bearing legacy selection probe."""
    del target, mode, emotion_axes, budget_available
    return []


def build_probe_report(
    target: str,
    *,
    mode: str | None,
    emotion_axes: set[str] | None,
    budget_available: float | None,
) -> dict[str, Any]:
    """Return a bodyless retirement receipt for the legacy diagnostics surface."""
    del emotion_axes, budget_available
    return {
        "status": "retired",
        "bodyless": True,
        "reason": "legacy_memory_selection_probe_not_runtime_authority",
        "target": target,
        "mode": mode,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Retired bodyless PCLTM legacy memory-selection probe."
    )
    parser.add_argument("--target", default="user", choices=("memory", "user"))
    parser.add_argument("--mode", default=None)
    parser.add_argument("--emotion-axis", dest="emotion_axes", action="append", default=[])
    parser.add_argument("--budget", type=float, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    report = build_probe_report(
        args.target,
        mode=args.mode,
        emotion_axes=set(args.emotion_axes),
        budget_available=args.budget,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
