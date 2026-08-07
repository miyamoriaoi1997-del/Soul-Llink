from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path


SOUL_ROOT = Path(__file__).resolve().parents[1]
PACKAGES_ROOT = SOUL_ROOT / "packages"
FORBIDDEN_HOST_MODULES = {
    "agent",
    "gateway",
    "hermes_state",
    "model_tools",
    "run_agent",
}


def _run_probe(source: str, tmp_path: Path) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PACKAGES_ROOT)
    env.pop("HERMES_STATE_PATH", None)
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        cwd=tmp_path,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    return json.loads(proc.stdout)


def test_fake_host_adapter_runs_turn_through_public_facade_without_hermes_imports(tmp_path: Path) -> None:
    data = _run_probe(
        """
        import json
        import sys
        from pathlib import Path
        from soul_link import FakeHostAdapter, HostAdapter, SoulLink

        before = set(sys.modules)
        adapter = FakeHostAdapter(
            host_system_prompt="Non-Hermes native host prompt",
            context=[{"role": "user", "content": "上一轮：host adapter spike"}],
        )
        link = SoulLink(
            state_path=Path("native-host-state.md").resolve(),
            enable_context_router=False,
        )
        envelope = adapter.run_turn(
            link,
            "继续做 native host adapter spike",
            previous_mode="work",
            emotion_state={"emotion_score": 1.5, "current_emotion": 1.5, "mode": "work"},
            emotion_modifier="test modifier",
        )
        imported = sorted(set(sys.modules) - before)
        print(json.dumps({
            "is_host_adapter": isinstance(adapter, HostAdapter),
            "host": envelope["host"],
            "platform": envelope["platform"],
            "mode": envelope["mode"],
            "selected_layers": envelope["selected_layers"],
            "prompt_has_host": "Non-Hermes native host prompt" in envelope["prompt_text"],
            "request_platform": envelope["audit_packet"]["request"]["platform"],
            "state_files": [str(path) for path in Path.cwd().glob("*STATE*.md")],
            "host_modules": [name for name in imported if name.split(".", 1)[0] in %r],
        }, ensure_ascii=False))
        """ % sorted(FORBIDDEN_HOST_MODULES),
        tmp_path,
    )

    assert data["is_host_adapter"] is True
    assert data["host"] == "fake-host"
    assert data["platform"] == "fake-host"
    assert data["mode"] == "work"
    assert data["selected_layers"] == ["core", "work"]
    assert data["prompt_has_host"] is True
    assert data["request_platform"] == "fake-host"
    assert data["state_files"] == []
    assert data["host_modules"] == []


def test_soul_link_public_exports_include_host_adapter_contract_only() -> None:
    import soul_link

    assert "HostAdapter" in soul_link.__all__
    assert "FakeHostAdapter" in soul_link.__all__
    assert not hasattr(soul_link, "HermesHostAdapter")
