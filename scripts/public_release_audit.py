#!/usr/bin/env python3
"""Fail-closed audit for a SoulLink public release candidate."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

TEXT_SUFFIXES = {
    ".css", ".html", ".in", ".js", ".json", ".md", ".py", ".toml",
    ".txt", ".yaml", ".yml",
}
IGNORED_DIRS = {
    ".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".venv",
    "__pycache__", "build", "dist", "node_modules",
}
FORBIDDEN_DIRS = {"backups", "logs", "var"}
FORBIDDEN_SUFFIXES = {
    ".db", ".jsonl", ".key", ".pem", ".sqlite", ".sqlite3",
}
REQUIRED_FILES = {
    ".gitignore", "CONTRIBUTING.md", "LICENSE", "MANIFEST.in", "README.md",
    "RELEASE_CHECKLIST.md", "SECURITY.md", "pyproject.toml",
}
FORBIDDEN_PRIVATE_DOCS = {
    "p12-dynamic-emotion-predeploy-acceptance-2026-05-05.md",
    "p13-dynamic-emotion-regression-closure-2026-05-05.md",
    "p14-checkpoint-review-2026-05-05.md",
    "work-soul-acceptance-20260430.md",
}

# Encoded so the audit implementation does not itself publish private markers.
PRIVATE_MARKERS = (
    "".join(map(chr, (0x4E03, 0x795E))),
    "".join(map(chr, (0x4F1A, 0x957F))),
    "Nan" + "agami",
    "miyamori" + "aoi",
)
PUBLIC_PROJECT_URLS = (
    "https://github.com/miyamoriaoi1997-del/Soul-Llink",
    "https://github.com/miyamoriaoi1997-del/Soul-Llink/issues",
)
ABSOLUTE_HOST_PATTERNS = (
    re.compile(r"(?i)[a-z]:[\\/]users[\\/](?!example(?:-user)?(?:[\\/]|$))"),
    re.compile("(?i)/" + "ho" + "me/(?!example(?:-user)?(?:/|$))"),
    re.compile("(?i)" + "app" + "data[\\\\/]local[\\\\/]" + "her" + "mes"),
)


def _walk(root: Path):
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if any(part in IGNORED_DIRS for part in relative.parts):
            continue
        yield path, relative


def audit(root: Path) -> dict[str, object]:
    private_hits: list[dict[str, object]] = []
    forbidden_files: list[str] = []

    for path, relative in _walk(root):
        if not path.is_file():
            continue
        rel = relative.as_posix()
        parts_lower = {part.lower() for part in relative.parts}
        root_state = bool(relative.parts) and relative.parts[0].lower() == "state"
        private_plan = rel.startswith("packages/persona_engine/docs/plans/")
        private_doc = path.name.lower() in FORBIDDEN_PRIVATE_DOCS
        if root_state or private_plan or private_doc or parts_lower & FORBIDDEN_DIRS or path.suffix.lower() in FORBIDDEN_SUFFIXES:
            forbidden_files.append(rel)
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        categories: set[str] = set()
        for project_url in PUBLIC_PROJECT_URLS:
            text = text.replace(project_url, "<PUBLIC_PROJECT_URL>")
        lowered = text.casefold()
        if any(marker.casefold() in lowered for marker in PRIVATE_MARKERS):
            categories.add("private-identity")
        if any(pattern.search(text) for pattern in ABSOLUTE_HOST_PATTERNS):
            categories.add("absolute-host-path")
        if categories:
            private_hits.append({"path": rel, "categories": sorted(categories)})

    missing = sorted(name for name in REQUIRED_FILES if not (root / name).is_file())
    report: dict[str, object] = {
        "ok": not private_hits and not forbidden_files and not missing,
        "root": str(root.resolve()),
        "private_marker_hits": sorted(private_hits, key=lambda item: str(item["path"])),
        "forbidden_files": sorted(forbidden_files),
        "missing_required_files": missing,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = audit(args.root.resolve())
    if args.json:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    else:
        print("PASS" if report["ok"] else "FAIL")
        for key in ("private_marker_hits", "forbidden_files", "missing_required_files"):
            if report[key]:
                print(f"{key}: {json.dumps(report[key], ensure_ascii=False)}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
