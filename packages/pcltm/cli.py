"""Public CLI for initializing and checking a SoulLink/PCLTM runtime tree."""

from __future__ import annotations

import argparse
import json
import webbrowser
from pathlib import Path
from typing import Any

from .doctor import PersonaLCMDoctor
from .index_observability import index_doctor, index_stats
from .governance_runtime import run_governance
from .hermes_history import HermesHistoryIngestor
from .live_context_evidence import build_tool_evidence_capsules
from .live_context_governor import ContextBudgetPolicy, govern_prompt_context
from .memfs_store import MEMFS_DIRECTORIES, MemFSStore
from .memory_adapter import last_live_context_telemetry, load_prompt_context
from .runtime_paths import DEFAULT_DB, DEFAULT_MEMFS_ROOT, resolve_db_path, resolve_memfs_root
from .store import EventStore



def init_runtime(
    *,
    db_path: str | Path | None = None,
    memfs_root: str | Path | None = None,
) -> dict[str, Any]:
    """Create the public runtime DB schema and MemFS directory layout.

    The operation is idempotent and non-destructive: it creates missing parents,
    bootstraps the SQLite schema through EventStore, creates MemFS directories,
    and never deletes or overwrites existing runtime data.
    """

    resolved_db = resolve_db_path(db_path)
    resolved_memfs = resolve_memfs_root(memfs_root)

    store = EventStore(resolved_db)
    try:
        schema_version = store.schema_version()
    finally:
        store.close()

    memfs = MemFSStore(resolved_memfs)
    memfs.init()

    directories = [str(resolved_memfs / name) for name in MEMFS_DIRECTORIES]
    return {
        "ok": True,
        "db_path": str(resolved_db),
        "db_exists": resolved_db.exists(),
        "schema_version": schema_version,
        "memfs_root": str(resolved_memfs),
        "memfs_directories": directories,
    }


def doctor_runtime(
    *,
    db_path: str | Path | None = None,
    memfs_root: str | Path | None = None,
    fix: bool = False,
) -> dict[str, Any]:
    """Check the runtime tree and optionally create missing runtime scaffolding."""

    if fix:
        init_runtime(db_path=db_path, memfs_root=memfs_root)

    resolved_db = resolve_db_path(db_path)
    resolved_memfs = resolve_memfs_root(memfs_root)
    issues: list[dict[str, Any]] = []

    if not resolved_db.exists():
        issues.append(
            {
                "severity": "error",
                "code": "missing_db",
                "message": "PCLTM SQLite database is missing; run `soullink init` or `soullink doctor --fix`.",
                "path": str(resolved_db),
            }
        )
        schema_version = None
    else:
        store = EventStore(resolved_db)
        try:
            schema_version = store.schema_version()
            doctor_report = PersonaLCMDoctor(store).run_checks()
        finally:
            store.close()
        issues.extend(doctor_report.get("issues", []))

    missing_dirs = [name for name in MEMFS_DIRECTORIES if not (resolved_memfs / name).is_dir()]
    if missing_dirs:
        issues.append(
            {
                "severity": "error",
                "code": "missing_memfs_directories",
                "message": "MemFS directory layout is incomplete; run `soullink init` or `soullink doctor --fix`.",
                "path": str(resolved_memfs),
                "missing": missing_dirs,
            }
        )

    return {
        "ok": not any(issue.get("severity") == "error" for issue in issues),
        "db_path": str(resolved_db),
        "db_exists": resolved_db.exists(),
        "schema_version": schema_version,
        "memfs_root": str(resolved_memfs),
        "missing_memfs_directories": missing_dirs,
        "issues": issues,
    }


def _runtime_observability() -> dict[str, Any]:
    """Return authoritative runtime paths and schema for CLI health reports."""
    resolved_db = resolve_db_path()
    resolved_memfs = resolve_memfs_root()
    schema_version = None
    if resolved_db.exists():
        store = EventStore(resolved_db)
        try:
            schema_version = store.schema_version()
        finally:
            store.close()
    return {
        "db_path": str(resolved_db),
        "schema_version": schema_version,
        "memfs_root": str(resolved_memfs),
    }


def live_context_smoke(
    *,
    mode: str | None = None,
    query: str | None = None,
    memory_limit: int = 2200,
    user_limit: int = 1375,
) -> dict[str, Any]:
    """Run a read-only smoke check for governed prompt-time PCLTM context."""
    rendered = load_prompt_context(mode=mode, query=query, memory_limit=memory_limit, user_limit=user_limit)
    telemetry = last_live_context_telemetry()
    open_count = rendered.count("<pcltm_context>")
    close_count = rendered.count("</pcltm_context>")
    has_context = bool(rendered)
    single_context = open_count == 1 and close_count == 1
    within_budget = bool(telemetry.get("within_budget")) if telemetry else False
    report = {
        "ok": has_context and single_context and within_budget,
        "mode": mode,
        "query": query,
        "has_pcltm_context": has_context,
        "single_pcltm_context": single_context,
        "open_tag_count": open_count,
        "close_tag_count": close_count,
        "rendered_chars": len(rendered),
        "telemetry": telemetry,
    }
    report.update(_runtime_observability())
    return report


def live_context_evidence_smoke() -> dict[str, Any]:
    """Run a synthetic tool-evidence smoke through the live context governor."""
    secret = "TOKEN=super-secret-value"
    capsules, evidence_telemetry = build_tool_evidence_capsules(
        [
            {
                "tool": "terminal",
                "command": "pytest -q",
                "exit_code": 0,
                "output": "312 passed in 6.91s\n" + secret,
                "affected_files": ["tests/pcltm/test_runtime_cli.py"],
            },
            {
                "tool": "read_file",
                "command": "read_file large.log",
                "exit_code": 0,
                "output": "line\n" * 500,
            },
        ],
        max_items=4,
        max_total_chars=500,
    )
    governed = govern_prompt_context(
        "<pcltm_context>\nselected memory\n</pcltm_context>",
        policy=ContextBudgetPolicy(total_chars=900, evidence_chars=500, memory_chars=220),
        tool_evidence=capsules,
        outer_tag="pcltm_context",
    )
    rendered = governed.rendered
    report = {
        "ok": evidence_telemetry.get("within_budget") is True
        and governed.telemetry.get("within_budget") is True
        and rendered.count("<pcltm_context>") == 1
        and rendered.count("</pcltm_context>") == 1
        and "super-secret-value" not in rendered,
        "single_pcltm_context": rendered.count("<pcltm_context>") == 1 and rendered.count("</pcltm_context>") == 1,
        "secret_leaked": "super-secret-value" in rendered,
        "evidence": evidence_telemetry,
        "governed": governed.telemetry,
        "rendered_chars": len(rendered),
    }
    report.update(_runtime_observability())
    return report


def _print_report(report: dict[str, Any], *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return

    status = "ok" if report.get("ok") else "error"
    print(f"status: {status}")
    print(f"db_path: {report.get('db_path')}")
    print(f"schema_version: {report.get('schema_version')}")
    print(f"memfs_root: {report.get('memfs_root')}")
    missing = report.get("missing_memfs_directories") or []
    if missing:
        print(f"missing_memfs_directories: {', '.join(missing)}")
    for issue in report.get("issues", []):
        print(f"{issue.get('severity', 'info')}: {issue.get('code', 'issue')}: {issue.get('message', '')}")



def _add_runtime_path_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--db", dest="db_path", default=None, help=f"PCLTM DB path (default: {DEFAULT_DB})")
    parser.add_argument("--memfs", dest="memfs_root", default=None, help=f"MemFS root (default: {DEFAULT_MEMFS_ROOT})")

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="soullink", description="SoulLink public runtime utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Create the runtime DB schema and MemFS directories")
    _add_runtime_path_options(init_parser)
    init_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")

    doctor_parser = subparsers.add_parser("doctor", help="Check runtime DB and MemFS readiness")
    _add_runtime_path_options(doctor_parser)
    doctor_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    doctor_parser.add_argument("--fix", action="store_true", help="Create missing DB/MemFS scaffolding before checking")

    index_parser = subparsers.add_parser("index", help="Inspect and repair derived PCLTM indexes")
    index_subparsers = index_parser.add_subparsers(dest="index_command", required=True)

    stats_parser = index_subparsers.add_parser("stats", help="Print read-only SQLite/MemFS index statistics")
    _add_runtime_path_options(stats_parser)
    stats_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")

    index_doctor_parser = index_subparsers.add_parser("doctor", help="Check derived index consistency")
    _add_runtime_path_options(index_doctor_parser)
    index_doctor_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    index_doctor_parser.add_argument("--rebuild", action="store_true", help="Rebuild SQLite FTS derived indexes when mismatched")

    live_context_parser = subparsers.add_parser("live-context", help="Inspect governed prompt-time PCLTM context")
    live_context_subparsers = live_context_parser.add_subparsers(dest="live_context_command", required=True)
    smoke_parser = live_context_subparsers.add_parser("smoke", help="Run read-only live context governance smoke check")
    smoke_parser.add_argument("--mode", default=None, help="Optional persona mode hint")
    smoke_parser.add_argument("--query", default=None, help="Optional recall query")
    smoke_parser.add_argument("--memory-limit", type=int, default=2200, help="Memory budget passed to load_prompt_context")
    smoke_parser.add_argument("--user-limit", type=int, default=1375, help="User budget passed to load_prompt_context")
    smoke_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    evidence_smoke_parser = live_context_subparsers.add_parser("evidence-smoke", help="Run synthetic tool-evidence capsule smoke check")
    evidence_smoke_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")

    governance_parser = subparsers.add_parser("governance", help="Run read-only PCLTM governance aggregation")
    governance_subparsers = governance_parser.add_subparsers(dest="governance_command", required=True)
    run_parser = governance_subparsers.add_parser("run", help="Aggregate index, scope, memory, and selection governance reports")
    _add_runtime_path_options(run_parser)
    run_parser.add_argument("--selection-target", default="user", choices=["user", "memory"], help="Memory selection probe target")
    run_parser.add_argument("--mode", default=None, help="Optional persona mode hint")
    run_parser.add_argument("--emotion-axis", dest="emotion_axes", action="append", default=[], help="Optional emotion axis hint; repeatable")
    run_parser.add_argument("--budget", dest="budget_available", type=float, default=None, help="Optional selection budget")
    run_parser.add_argument("--rebuild-indexes", action="store_true", help="Rebuild derived SQLite FTS indexes if mismatched")
    run_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")

    history_parser = subparsers.add_parser("hermes-history-ingest", help="Backfill canonical Hermes history into retrieve-only PCLTM events")
    history_parser.add_argument("--source-db", required=True, help="Canonical Hermes state.db path")
    history_parser.add_argument("--db", dest="db_path", default=None, help=f"PCLTM DB path (default: {DEFAULT_DB})")
    history_parser.add_argument("--session-id", default=None, help="Optionally ingest one Hermes session")
    history_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")

    webui_parser = subparsers.add_parser("webui", help="Run the localhost-only read-only monitoring dashboard")
    _add_runtime_path_options(webui_parser)
    webui_parser.add_argument("--host", default="127.0.0.1")
    webui_parser.add_argument("--port", type=int, default=8765)
    webui_parser.add_argument("--refresh-seconds", type=float, default=5.0)
    webui_parser.add_argument("--no-open-browser", action="store_true")
    webui_parser.add_argument("--config", dest="config_path", default=None)
    webui_parser.add_argument("--persona-log", default=None)
    webui_parser.add_argument("--router-log", default=None)
    webui_parser.add_argument("--state-path", default=None, help="SoulLink STATE.md path")
    webui_parser.add_argument("--mode", default=None, help="Optional retrieval scope for the sidecar preview; not a state-machine output")
    webui_parser.add_argument("--memory-body-limit", type=int, default=100)
    return parser


def _run_webui(args: argparse.Namespace) -> int:
    import yaml
    from .monitoring.collectors import collect_context_budget, collect_runtime_memory
    from .monitoring.logs import collect_persona_log, collect_router_log
    from .monitoring.private_data import (
        collect_emotion_state,
        collect_injection_preview,
        collect_memory_bodies,
        collect_runtime_turn_capture,
        collect_soul_content,
    )
    from .monitoring.memory_library import collect_memory_library_stats
    from .monitoring.server import create_server
    from .monitoring.snapshot import SnapshotService

    config_path = Path(args.config_path) if args.config_path else Path.home() / ".hermes" / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {} if config_path.is_file() else {}
    db = resolve_db_path(args.db_path)
    memfs = resolve_memfs_root(args.memfs_root)
    router_log = Path(args.router_log) if args.router_log else Path("packages/model_router/logs/audit.jsonl")
    state_path = Path(args.state_path) if args.state_path else Path.home() / "AppData" / "Local" / "hermes" / "STATE.md"
    hermes_home = state_path.parent
    persona_log = Path(args.persona_log) if args.persona_log else hermes_home / "logs" / "persona-orchestrator.jsonl"
    capture_path = hermes_home / "runtime" / "soullink-latest-turn.json"
    context_telemetry_path = hermes_home / "runtime" / "soullink-context-telemetry.json"
    soul_path = hermes_home / "SOUL.md"
    soul_layers = Path(__file__).resolve().parents[1] / "persona_engine" / "soul_layers"
    persona = lambda: collect_persona_log(persona_log)
    router = lambda: collect_router_log(router_log)
    service = SnapshotService(
        {
            "core": lambda: collect_runtime_memory(db_path=db, memfs_root=memfs),
            "context": lambda: {"context": collect_context_budget(config=config, telemetry_path=context_telemetry_path)},
            "persona": lambda: {"persona": persona(), "issues": persona().get("issues", [])},
            "router": lambda: {"router": router(), "issues": router().get("issues", [])},
            "emotion": lambda: {"emotion": collect_emotion_state(state_path)},
            "memory_bodies": lambda: {"memory_bodies": collect_memory_bodies(db, limit=args.memory_body_limit)},
            "memory_library": lambda: {"memory_library_stats": collect_memory_library_stats(db)},
            "injection": lambda: {"injection": collect_injection_preview(db, mode=args.mode)},
            "runtime_capture": lambda: {"runtime_capture": collect_runtime_turn_capture(
                capture_path, router_audit_path=router_log,
            )},
            "soul": lambda: {"soul": collect_soul_content(soul_path, soul_layers)},
        },
        ttl_seconds=max(2.0, args.refresh_seconds / 2),
    )
    server = create_server(args.host, args.port, service)
    url = f"http://{args.host}:{server.server_port}/"
    print(f"SoulLink monitor listening on {url}", flush=True)
    if not args.no_open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "webui":
        return _run_webui(args)

    if args.command == "init":
        report = init_runtime(db_path=args.db_path, memfs_root=args.memfs_root)
    elif args.command == "doctor":
        report = doctor_runtime(db_path=args.db_path, memfs_root=args.memfs_root, fix=args.fix)
    elif args.command == "index" and args.index_command == "stats":
        report = index_stats(db_path=args.db_path, memfs_root=args.memfs_root)
    elif args.command == "index" and args.index_command == "doctor":
        report = index_doctor(db_path=args.db_path, memfs_root=args.memfs_root, rebuild=args.rebuild)
    elif args.command == "live-context" and args.live_context_command == "smoke":
        report = live_context_smoke(
            mode=args.mode,
            query=args.query,
            memory_limit=args.memory_limit,
            user_limit=args.user_limit,
        )
    elif args.command == "live-context" and args.live_context_command == "evidence-smoke":
        report = live_context_evidence_smoke()
    elif args.command == "hermes-history-ingest":
        target = resolve_db_path(args.db_path)
        store = EventStore(target)
        try:
            report = HermesHistoryIngestor(store, args.source_db).ingest(session_id=args.session_id)
        finally:
            store.close()
        report.update({"ok": True, "source_db": str(Path(args.source_db)), "db_path": str(target)})
    elif args.command == "governance" and args.governance_command == "run":
        report = run_governance(
            db_path=args.db_path,
            memfs_root=args.memfs_root,
            selection_target=args.selection_target,
            mode=args.mode,
            emotion_axes=set(args.emotion_axes),
            budget_available=args.budget_available,
            rebuild_indexes=args.rebuild_indexes,
        )
    else:  # pragma: no cover - argparse enforces commands
        parser.error(f"unknown command: {args.command}")

    for key, value in _runtime_observability().items():
        report.setdefault(key, value)
    _print_report(report, json_output=args.json)
    return 0 if report.get("ok") else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
