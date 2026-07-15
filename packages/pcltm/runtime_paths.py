"""Runtime path resolution for the public SoulLink/PCLTM distribution.

Runtime databases and MemFS data are intentionally outside package data and git.
This module centralizes the public defaults so CLI tools, audits, adapters, and
indexing code all agree on the same paths.
"""

from __future__ import annotations

import os
from pathlib import Path

SOUL_LINK_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VAR_DIR = SOUL_LINK_ROOT / "var"
DEFAULT_DB = DEFAULT_VAR_DIR / "pcltm-prod.db"
DEFAULT_MEMFS_ROOT = DEFAULT_VAR_DIR / "memfs"

DB_ENV_VAR = "HERMES_PCLTM_DB"
MEMFS_ENV_VAR = "HERMES_PCLTM_MEMFS_ROOT"


def resolve_db_path(raw_path: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the PCLTM SQLite DB path.

    Precedence:
    1. explicit argument
    2. HERMES_PCLTM_DB
    3. public distribution default: <repo-or-install-root>/var/pcltm-prod.db
    """

    value = raw_path if raw_path is not None else os.getenv(DB_ENV_VAR)
    return Path(value).expanduser() if value else DEFAULT_DB


def resolve_memfs_root(raw_root: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the MemFS root directory.

    Precedence:
    1. explicit argument
    2. HERMES_PCLTM_MEMFS_ROOT
    3. public distribution default: <repo-or-install-root>/var/memfs
    """

    value = raw_root if raw_root is not None else os.getenv(MEMFS_ENV_VAR)
    return Path(value).expanduser() if value else DEFAULT_MEMFS_ROOT
