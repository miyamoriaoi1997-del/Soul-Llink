from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for path in (ROOT, ROOT / "packages", ROOT / "packages" / "persona_engine", ROOT / "adapters" / "hermes"):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

# Host-coupled regression tests can opt into a checked-out Hermes source tree.
# The public package itself does not vendor Hermes or assume a user-specific path.
HERMES_ROOT_VALUE = os.environ.get("SOULLINK_HERMES_AGENT_ROOT") or os.environ.get("HERMES_HOST_ROOT")
if HERMES_ROOT_VALUE:
    HERMES_ROOT = Path(HERMES_ROOT_VALUE)
    if HERMES_ROOT.exists():
        hermes_text = str(HERMES_ROOT)
        if hermes_text not in sys.path:
            sys.path.append(hermes_text)

        for venv_dir in (HERMES_ROOT / "venv", HERMES_ROOT / ".venv"):
            for site_packages_root in (venv_dir / "lib", venv_dir / "Lib"):
                for site_packages in site_packages_root.glob("python*/site-packages"):
                    site_packages_text = str(site_packages)
                    if site_packages_text not in sys.path:
                        sys.path.append(site_packages_text)
                direct_site_packages = site_packages_root / "site-packages"
                if direct_site_packages.is_dir():
                    text = str(direct_site_packages)
                    if text not in sys.path:
                        sys.path.append(text)
