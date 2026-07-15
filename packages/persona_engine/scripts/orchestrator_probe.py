#!/usr/bin/env python3
"""Probe the Persona State Orchestrator in shadow mode.

Usage:
    python scripts/orchestrator_probe.py "帮我检查 gateway 日志"
    python scripts/orchestrator_probe.py "[assistant name]我想你了" --score 2.5
    python scripts/orchestrator_probe.py "我们做爱" --score 4.5
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from persona_orchestrator.state_orchestrator import StateOrchestrator


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a shadow persona-orchestrator classification probe.")
    parser.add_argument("message", help="User message to classify")
    parser.add_argument("--score", type=float, default=None, help="Optional emotion_score for desire tier calculation")
    parser.add_argument("--previous-mode", default=None, help="Optional previous mode for transition testing")
    parser.add_argument("--platform", default="cli", help="Platform label, default: cli")
    parser.add_argument("--log-path", default=None, help="Optional JSONL log path")
    parser.add_argument("--semantic-shadow", action="store_true", help="Enable semantic classifier shadow output")
    parser.add_argument(
        "--semantic-backend",
        default="local",
        choices=["local", "local_lightweight", "rules+local", "llm"],
        help="Semantic shadow backend to use when --semantic-shadow is set",
    )
    args = parser.parse_args(argv)

    emotion_state = {}
    if args.score is not None:
        emotion_state["emotion_score"] = args.score

    orchestrator = StateOrchestrator(
        base_dir=REPO_ROOT,
        log_path=args.log_path or (REPO_ROOT / "logs" / "persona_orchestrator_shadow.jsonl"),
        enable_semantic_shadow=args.semantic_shadow,
        semantic_backend=args.semantic_backend,
    )
    packet = orchestrator.analyze_turn(
        user_message=args.message,
        emotion_state=emotion_state,
        previous_mode=args.previous_mode,
        platform=args.platform,
    )
    print(json.dumps(asdict(packet), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
