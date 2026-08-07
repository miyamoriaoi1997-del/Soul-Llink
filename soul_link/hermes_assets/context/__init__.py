"""Hermes general-plugin entry for the PCLTM governed context engine."""

from pathlib import Path
import os
import sys

_ROOT_FILE = Path(__file__).with_name("soullink-root.txt")
if not _ROOT_FILE.is_file():
    raise RuntimeError("PCLTM context plugin is incomplete: soullink-root.txt is missing")
_SOULLINK_ROOT = Path(_ROOT_FILE.read_text(encoding="utf-8").strip()).expanduser().resolve()
if not (_SOULLINK_ROOT / "packages" / "pcltm").is_dir():
    raise RuntimeError(f"SoulLink root is invalid: {_SOULLINK_ROOT}")
os.environ["SOULLINK_ROOT"] = str(_SOULLINK_ROOT)
for _path in (_SOULLINK_ROOT, _SOULLINK_ROOT / "packages"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from soul_link.hermes_plugin.context_engine import PCLTMContextCompressionEngine


def register(ctx) -> None:
    ctx.register_context_engine(PCLTMContextCompressionEngine())
