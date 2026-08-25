"""External dynamic emotion injection lexicon.

The lexicon is data, not routing/state logic. Missing or malformed data fails closed
by returning an empty mapping; callers retain their in-code compatibility defaults.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


_LEXICON_PATH = Path(__file__).with_name("injection_lexicon.yaml")


def load_injection_lexicon() -> dict[str, Any]:
    """Load the optional external lexicon without changing runtime state."""
    try:
        with _LEXICON_PATH.open("r", encoding="utf-8") as handle:
            value = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError):
        return {}
    return value if isinstance(value, dict) else {}


LEXICON = load_injection_lexicon()
