#!/usr/bin/env python3
"""User-scoped lifecycle manager for the SoulLink model router on Windows."""
from __future__ import annotations

import argparse
import contextlib
import html
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from urllib import request as urlrequest

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "config.yaml"
DEFAULT_RUNTIME = HERE / "runtime"
STARTUP = Path(os.environ.get("APPDATA", Path.home() / "AppData/Roaming")) / "Microsoft/Windows/Start Menu/Programs/Startup"
STARTUP_FILE = STARTUP / "SoulLink-Model-Router.cmd"
TASK_NAME = "SoulLink-Model-Router"


def _health(url: str = "http://127.0.0.1:18080/healthz", timeout: float = 2.0) -> dict:
    try:
        with urlrequest.urlopen(url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
            return {"running": response.status == 200 and bool(payload.get("ok")), "health": payload}
    except Exception as exc:
        return {"running": False, "error": f"{type(exc).__name__}: {exc}"}


def _command(config: Path) -> list[str]:
    return [sys.executable, str(HERE / "app.py"), "--config", str(config.resolve())]


def _pythonw() -> Path:
    candidate = Path(sys.executable).with_name("pythonw.exe")
    return candidate if candidate.is_file() else Path(sys.executable)


def _task_xml(config: Path, user_id: str) -> str:
    executable = html.escape(str(_pythonw()), quote=True)
    arguments = html.escape(
        subprocess.list2cmdline(
            [str(Path(__file__).resolve()), "run", "--config", str(config.resolve())]
        )
    )
    working = html.escape(str(HERE), quote=True)
    user = html.escape(user_id, quote=True)
    return f'''<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.4" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo><Description>SoulLink state-machine model router</Description></RegistrationInfo>
  <Triggers><LogonTrigger><Enabled>true</Enabled><UserId>{user}</UserId></LogonTrigger></Triggers>
  <Principals><Principal id="Author"><UserId>{user}</UserId><LogonType>InteractiveToken</LogonType><RunLevel>LeastPrivilege</RunLevel></Principal></Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable>
    <AllowHardTerminate>true</AllowHardTerminate>
    <RestartOnFailure><Interval>PT1M</Interval><Count>999</Count></RestartOnFailure>
    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>
    <Enabled>true</Enabled>
  </Settings>
  <Actions Context="Author"><Exec><Command>{executable}</Command><Arguments>{arguments}</Arguments><WorkingDirectory>{working}</WorkingDirectory></Exec></Actions>
</Task>
'''


def run_foreground(config: Path = DEFAULT_CONFIG, runtime: Path = DEFAULT_RUNTIME) -> int:
    """Run under Task Scheduler without a console, retaining local diagnostics."""
    runtime.mkdir(parents=True, exist_ok=True)
    pid_file = runtime / "router.pid"
    pid_file.write_text(str(os.getpid()), encoding="ascii")
    try:
        from app import main as router_main

        with (runtime / "router.stdout.log").open("a", encoding="utf-8") as stdout, (
            runtime / "router.stderr.log"
        ).open("a", encoding="utf-8") as stderr, contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            return int(router_main(["--config", str(config.resolve())]))
    except BaseException:
        with (runtime / "router.stderr.log").open("a", encoding="utf-8") as stderr:
            traceback.print_exc(file=stderr)
        return 1
    finally:
        pid_file.unlink(missing_ok=True)


def start(config: Path = DEFAULT_CONFIG, runtime: Path = DEFAULT_RUNTIME) -> dict:
    current = _health()
    if current["running"]:
        return {"started": False, **current}
    runtime.mkdir(parents=True, exist_ok=True)
    pid_file = runtime / "router.pid"
    pid_file.unlink(missing_ok=True)
    stdout = (runtime / "router.stdout.log").open("ab")
    stderr = (runtime / "router.stderr.log").open("ab")
    flags = 0
    if os.name == "nt":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    process = subprocess.Popen(
        _command(config),
        cwd=str(HERE),
        stdin=subprocess.DEVNULL,
        stdout=stdout,
        stderr=stderr,
        close_fds=True,
        creationflags=flags,
    )
    pid_file.write_text(str(process.pid), encoding="ascii")
    for _ in range(50):
        time.sleep(0.1)
        current = _health()
        if current["running"]:
            stdout.close()
            stderr.close()
            return {"started": True, "pid": process.pid, **current}
        if process.poll() is not None:
            break
    stdout.close()
    stderr.close()
    pid_file.unlink(missing_ok=True)
    return {"started": False, "pid": process.pid, **_health(), "exit_code": process.poll()}


def stop(runtime: Path = DEFAULT_RUNTIME) -> dict:
    pid_file = runtime / "router.pid"
    if not pid_file.is_file():
        return {"stopped": False, "reason": "pid_missing", **_health()}
    pid = int(pid_file.read_text(encoding="ascii").strip())
    if os.name == "nt":
        result = subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            text=True,
            encoding="mbcs",
            errors="replace",
        )
    else:
        result = subprocess.run(["kill", str(pid)], capture_output=True, text=True)
    if result.returncode == 0:
        pid_file.unlink(missing_ok=True)
    return {"stopped": result.returncode == 0, "pid": pid, "returncode": result.returncode, **_health()}


def install_autostart(config: Path = DEFAULT_CONFIG, runtime: Path = DEFAULT_RUNTIME) -> dict:
    """Install an invisible, user-scoped scheduled task and start it now."""
    if os.name != "nt":
        raise RuntimeError("Task Scheduler autostart is Windows-only")
    runtime.mkdir(parents=True, exist_ok=True)
    user = subprocess.run(
        ["whoami"], capture_output=True, text=True, encoding="mbcs", errors="replace", check=True
    ).stdout.strip()
    xml_path = runtime / "router-task.xml"
    xml_path.write_text(_task_xml(config, user), encoding="utf-16")
    created = subprocess.run(
        ["schtasks.exe", "/Create", "/TN", TASK_NAME, "/XML", str(xml_path), "/F"],
        capture_output=True,
        text=True,
        encoding="mbcs",
        errors="replace",
    )
    if created.returncode != 0:
        raise RuntimeError(f"scheduled task creation failed: {created.stderr or created.stdout}")
    # Remove the legacy Startup-folder launcher so there is one owner only.
    STARTUP_FILE.unlink(missing_ok=True)
    launched = subprocess.run(
        ["schtasks.exe", "/Run", "/TN", TASK_NAME],
        capture_output=True,
        text=True,
        encoding="mbcs",
        errors="replace",
    )
    return {
        "installed": True,
        "task_name": TASK_NAME,
        "task_xml": str(xml_path),
        "launch_returncode": launched.returncode,
        "legacy_startup_removed": not STARTUP_FILE.exists(),
    }


def remove_autostart() -> dict:
    result = subprocess.run(
        ["schtasks.exe", "/Delete", "/TN", TASK_NAME, "/F"],
        capture_output=True,
        text=True,
        encoding="mbcs",
        errors="replace",
    )
    STARTUP_FILE.unlink(missing_ok=True)
    return {"removed": result.returncode == 0, "task_name": TASK_NAME}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("run", "start", "stop", "status", "install-autostart", "remove-autostart"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    if args.action == "run":
        return run_foreground(args.config)
    if args.action == "start":
        result = start(args.config)
    elif args.action == "stop":
        result = stop()
    elif args.action == "status":
        result = _health()
    elif args.action == "install-autostart":
        result = install_autostart(args.config)
    else:
        result = remove_autostart()
    print(json.dumps(result, ensure_ascii=False))
    return 0 if (result.get("running") or result.get("installed") or result.get("removed") or result.get("stopped")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
