from pathlib import Path
import re

from persona_orchestrator import (
    CANONICAL_MODES,
    LEGACY_MODE_ALIASES,
    MODE_CONFLICT,
    MODE_CREATIVE,
    MODE_DAILY,
    MODE_INTIMACY,
    MODE_REPAIR,
    MODE_SEX,
    MODE_SEX_CANDIDATE,
    MODE_SYSTEM_MAINTENANCE,
    MODE_WORK,
    normalize_mode,
)


LEGACY_EXPORTS = {
    "MODE_INTIMACY",
    "MODE_REPAIR",
    "MODE_CONFLICT",
    "MODE_SYSTEM_MAINTENANCE",
    "MODE_CREATIVE",
    "MODE_SEX_CANDIDATE",
    "LEGACY_MODE_ALIASES",
    "normalize_mode",
}


def test_public_mode_contract_is_three_canonical_states():
    assert CANONICAL_MODES == (MODE_DAILY, MODE_WORK, MODE_SEX)
    assert set(CANONICAL_MODES) == {"daily", "work", "sex"}


def test_legacy_mode_symbols_are_canonical_valued_import_aliases():
    assert MODE_INTIMACY == MODE_DAILY
    assert MODE_REPAIR == MODE_DAILY
    assert MODE_CONFLICT == MODE_DAILY
    assert MODE_SYSTEM_MAINTENANCE == MODE_WORK
    assert MODE_CREATIVE == MODE_WORK
    assert MODE_SEX_CANDIDATE == MODE_SEX


def test_normalize_mode_maps_legacy_modes_to_three_canonical_states():
    assert LEGACY_MODE_ALIASES == {
        "intimacy": MODE_DAILY,
        "repair": MODE_DAILY,
        "conflict": MODE_DAILY,
        "system_maintenance": MODE_WORK,
        "creative": MODE_WORK,
        "sex_candidate": MODE_SEX,
    }
    for legacy, canonical in LEGACY_MODE_ALIASES.items():
        assert normalize_mode(legacy) == canonical
    for canonical in CANONICAL_MODES:
        assert normalize_mode(canonical) == canonical
    assert normalize_mode("unknown") == MODE_DAILY


def test_runtime_state_machine_source_only_defines_legacy_aliases_in_types():
    source_root = Path(__file__).resolve().parents[1] / "persona_orchestrator"
    forbidden = {
        "MODE_INTIMACY",
        "MODE_REPAIR",
        "MODE_CONFLICT",
        "MODE_SYSTEM_MAINTENANCE",
        "MODE_CREATIVE",
        "MODE_SEX_CANDIDATE",
        "LEGACY_MODE_ALIASES",
        r"\bnormalize_mode\b(?!_)",
    }
    offenders = []
    for path in source_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if path.name in {"types.py", "__init__.py"}:
            continue
        hits = []
        for term in forbidden:
            if term.startswith("\\b"):
                if re.search(term, text):
                    hits.append(term)
            elif re.search(rf"(?<![A-Za-z0-9_]){re.escape(term)}(?![A-Za-z0-9_])", text):
                hits.append(term)
        hits = sorted(hits)
        if hits:
            offenders.append((path.relative_to(source_root).as_posix(), hits))
    assert offenders == []
