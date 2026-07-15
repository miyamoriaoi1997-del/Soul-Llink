from pathlib import Path

SOUL_LINK_ROOT = Path(__file__).resolve().parents[1]
PERSONA_ACTIVE = SOUL_LINK_ROOT / "packages" / "persona_engine"


def resolve_persona_engine_base_dir(base_dir=None) -> Path:
    if base_dir is None:
        return PERSONA_ACTIVE
    candidate = Path(base_dir)
    if candidate == Path("."):
        return PERSONA_ACTIVE
    if candidate.name == "persona_engine":
        return candidate
    if (candidate / "packages" / "persona_engine").exists():
        return candidate / "packages" / "persona_engine"
    if (candidate / "soul_layers").exists():
        return candidate
    return candidate
