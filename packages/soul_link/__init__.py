from __future__ import annotations

from pathlib import Path

_ROOT_PACKAGE = Path(__file__).resolve().parents[2] / "soul_link"
__path__ = [str(_ROOT_PACKAGE), *__path__]

exec(compile((_ROOT_PACKAGE / "__init__.py").read_text(), str(_ROOT_PACKAGE / "__init__.py"), "exec"), globals())
