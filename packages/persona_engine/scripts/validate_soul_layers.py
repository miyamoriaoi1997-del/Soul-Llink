#!/usr/bin/env python3
"""Validate layered SOUL templates for the persona orchestrator."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from persona_orchestrator.soul_layer_validator import SoulLayerValidator


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate SOUL layer template contracts.")
    parser.add_argument("--base-dir", default=str(REPO_ROOT), help="Repository root containing soul_layers/")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON result")
    args = parser.parse_args()

    result = SoulLayerValidator(args.base_dir).validate()
    if args.json:
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"checked_layers={len(result.checked_layers)} ok={result.ok}")
        for error in result.errors:
            print(f"ERROR: {error}")
        for warning in result.warnings:
            print(f"WARNING: {warning}")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
