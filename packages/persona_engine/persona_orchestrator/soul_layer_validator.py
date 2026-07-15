"""Contract validator for layered SOUL prompt templates."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from soul_link.contracts import resolve_persona_engine_base_dir


DEFAULT_REQUIRED_LAYERS = [
    "core",
    "daily",
    "work",
    "sex",
]

EXPECTED_HEADERS = {
    "core": "# Core Identity Layer",
    "daily": "# Daily Mode Layer",
    "work": "# Work Mode Layer",
    "sex": "# Adult Boundary Layer",
}

NON_CORE_IDENTITY_PHRASES = [
    "core identity is defined here",
    "核心身份",
    "身份定义",
    "you are ",
]

# Phrases that look like identity re-definition but are actually OK in context
# (e.g., "我是[assistant name]" in conflict layer is emotional anchor, not identity redefinition)
IDENTITY_PHRASE_ALLOWLIST_CONTEXTS = {
    "我是": ["身份锚点", "身份不变", "不变的是", "锚点不变"],
}


@dataclass
class SoulLayerValidationResult:
    """Result from validating SOUL layer templates."""

    ok: bool
    checked_layers: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class SoulLayerValidator:
    """Validate layered SOUL templates before they are used for composition.

    The validator intentionally checks a small, auditable contract rather than
    interpreting prompt prose. It protects the most important invariants:
    core is unique, mode layers do not redefine identity, required headers are
    present, and the sex layer remains disabled by default in Phase 1.
    """

    def __init__(self, base_dir: str | Path, required_layers: list[str] | None = None):
        self.base_dir = self._resolve_base_dir(Path(base_dir))
        self.layers_dir = self.base_dir / "soul_layers"
        self.required_layers = list(required_layers or DEFAULT_REQUIRED_LAYERS)

    @staticmethod
    def _resolve_base_dir(base_dir: Path) -> Path:
        return resolve_persona_engine_base_dir(base_dir)

    def validate(self) -> SoulLayerValidationResult:
        errors: list[str] = []
        warnings: list[str] = []
        checked_layers: list[str] = []

        for layer in self.required_layers:
            path = self.layers_dir / f"SOUL.{layer}.template.md"
            if not path.exists():
                errors.append(f"{layer}: missing layer file {path}")
                continue

            checked_layers.append(layer)
            text = path.read_text(encoding="utf-8")
            self._validate_header(layer, text, errors)
            self._validate_identity_scope(layer, text, errors)
            self._validate_layer_specific_rules(layer, text, errors, warnings)

        return SoulLayerValidationResult(
            ok=not errors,
            checked_layers=checked_layers,
            errors=errors,
            warnings=warnings,
        )

    def _validate_header(self, layer: str, text: str, errors: list[str]) -> None:
        expected = EXPECTED_HEADERS.get(layer)
        if expected and expected not in text:
            errors.append(f"{layer}: missing required header {expected!r}")

    def _validate_identity_scope(self, layer: str, text: str, errors: list[str]) -> None:
        lower_text = text.lower()
        if layer == "core":
            if "core identity" not in lower_text and "核心身份" not in text:
                errors.append("core: must define or anchor core identity")
            return

        for phrase in NON_CORE_IDENTITY_PHRASES:
            if phrase in lower_text or phrase in text:
                errors.append(f"{layer}: non-core layer must not redefine core identity via phrase {phrase!r}")
                return

        # Check context-dependent phrases (e.g. "我是" is OK if near anchor context)
        for phrase, allowed_contexts in IDENTITY_PHRASE_ALLOWLIST_CONTEXTS.items():
            if phrase in text:
                if not any(ctx in text for ctx in allowed_contexts):
                    errors.append(f"{layer}: non-core layer must not redefine core identity via phrase {phrase!r}")
                    return

        contract_cn = "不能改写" in text or "不重新定义身份" in text or "不重定义身份" in text
        contract_en = "must not redefine identity" in lower_text or ("must not redefine" in lower_text and "identity" in lower_text)
        if not contract_cn and not contract_en:
            errors.append(f"{layer}: missing non-identity contract phrase 'must not redefine identity' or equivalent")

    def _validate_layer_specific_rules(
        self,
        layer: str,
        text: str,
        errors: list[str],
        warnings: list[str],
    ) -> None:
        lower_text = text.lower()
        if layer == "sex":
            has_disabled = (
                "disabled by default" in lower_text
                or "public template" in lower_text
                or "avoid explicit content by default" in lower_text
                or "deployment policy" in lower_text
                or "默认禁用" in text
                or "门控" in text
                or "欲望" in text
            )
            if not has_disabled:
                errors.append("sex: Adult Boundary Layer must reference desire gates, deployment policy, or disabled-by-default status")
