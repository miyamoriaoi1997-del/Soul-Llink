#!/usr/bin/env python3
"""Prepare a Hermes runtime-shadow plugin for observational multi-SOUL auditing.

The generated plugin registers only a ``pre_llm_call`` hook and returns ``None``.
It never injects context, never edits the system prompt, never changes the
live response path, and never decides runtime switching. It only writes
redacted shadow logs through RuntimeShadowAdapter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_PLUGIN_DIR = Path.home() / ".hermes" / "plugins" / "persona-runtime-shadow"
DEFAULT_PERSONA_ENGINE = Path(__file__).resolve().parents[1]

PLUGIN_YAML = """name: persona-runtime-shadow
version: 0.1.0
description: "Shadow-only multi-SOUL runtime auditing. Does not alter live prompts or runtime switching."
author: "Hermes Persona Engine"
hooks:
  - pre_llm_call
"""

INIT_TEMPLATE = '''"""Shadow-only multi-SOUL runtime observer.

Safety contract:
- registers only pre_llm_call;
- computes an active prompt candidate for audit only;
- returns None, so Hermes receives no injected context;
- never mutates the live system prompt or message list;
- never decides whether runtime switching is allowed.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
PERSONA_ENGINE = Path({persona_engine!r})
SOUL_LINK_PACKAGES = PERSONA_ENGINE.parent
for _path in (SOUL_LINK_PACKAGES, PERSONA_ENGINE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

try:
    from persona_orchestrator import RuntimeShadowAdapter
    from emotion_state_manager import EmotionStateManager
except Exception as exc:  # pragma: no cover - runtime environment dependent
    RuntimeShadowAdapter = None
    EmotionStateManager = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

STATE_PATH = (Path.home() / "soul-link" / "state" / "STATE.md")

def _on_pre_llm_call(
    session_id: str = "",
    user_message: str = "",
    conversation_history: list[dict] | None = None,
    model: str = "",
    platform: str = "cli",
    previous_mode: str | None = None,
    message_timestamp: float | None = None,
    **_: Any,
) -> None:
    """Observe a turn in shadow mode and return no context.

    The host currently does not pass the active system prompt or emotion block
    into plugin hooks, so this runtime observer records state/layer decisions
    against an empty host prompt. That is deliberate: this is an observational
    integration step, not active prompt takeover or runtime-switch control.
    """
    if RuntimeShadowAdapter is None:
        logger.debug("persona-runtime-shadow unavailable: %s", _IMPORT_ERROR)
        return None
    try:
        emotion_state = {{}}
        emotion_modifier = ""
        if EmotionStateManager is not None:
            try:
                emotion_manager = EmotionStateManager(state_path=STATE_PATH)
                emotion_state = emotion_manager.get_current_emotion_state()
                emotion_modifier = emotion_manager.get_tone_modifiers()
            except Exception as exc:
                logger.debug("persona-runtime-shadow emotion read failed: %s", exc)
        adapter = RuntimeShadowAdapter(
            base_dir=PERSONA_ENGINE,
            log_path=Path.home() / ".hermes" / "logs" / "persona_runtime_shadow.jsonl",
            enable_semantic_shadow=True,
            semantic_backend="local",
        )
        adapter.analyze_runtime_turn(
            host_system_prompt="",
            user_message=user_message or "",
            recent_messages=conversation_history or [],
            emotion_state=emotion_state,
            emotion_modifier=emotion_modifier,
            previous_mode=previous_mode,
            platform=platform or "cli",
            session_id=session_id or None,
            message_timestamp=message_timestamp,
        )
    except Exception as exc:
        logger.debug("persona-runtime-shadow failed: %s", exc)
    return None


def register(ctx) -> None:
    ctx.register_hook("pre_llm_call", _on_pre_llm_call)
'''


def build_files(persona_engine: str | Path) -> dict[str, str]:
    engine = str(Path(persona_engine))
    return {
        "plugin.yaml": PLUGIN_YAML,
        "__init__.py": INIT_TEMPLATE.format(persona_engine=engine),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare the persona runtime shadow Hermes plugin.")
    parser.add_argument("--plugin-dir", default=str(DEFAULT_PLUGIN_DIR), help="Target plugin directory")
    parser.add_argument("--persona-engine", default=str(DEFAULT_PERSONA_ENGINE), help="persona-engine repository path")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Show planned writes without touching files")
    mode.add_argument("--write", action="store_true", help="Write plugin files")
    args = parser.parse_args(argv)

    plugin_dir = Path(args.plugin_dir)
    files = build_files(args.persona_engine)
    paths = [plugin_dir / name for name in files]
    payload = {
        "plugin_dir": str(plugin_dir),
        "persona_engine": str(Path(args.persona_engine)),
        "dry_run": not args.write,
        "would_write": [str(p) for p in paths],
        "written": False,
        "enable_command": "hermes config set plugins.enabled '[\"hermes-lcm\", \"persona-runtime-shadow\"]'",
        "restart_required": True,
    }
    if args.write:
        plugin_dir.mkdir(parents=True, exist_ok=True)
        for name, content in files.items():
            (plugin_dir / name).write_text(content, encoding="utf-8")
        payload["written"] = True
        payload["dry_run"] = False
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
