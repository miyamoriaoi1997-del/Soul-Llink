"""Prompt composition for layered SOUL prompts."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class _LayerTemplateCacheEntry:
    """Cached layer template keyed by filesystem fingerprint.

    Prompt composition sits on the runtime hot path.  Layer templates are
    immutable for normal turns but may be edited during operations, so cache by
    mtime/size instead of assuming process-lifetime immutability.
    """

    mtime_ns: int
    size: int
    text: str

from soul_link.contracts import resolve_persona_engine_base_dir

from .types import PromptComposition


class PromptComposer:
    """Compose prompt previews from core and selected mode layers."""

    CORE_SOURCE_ORCHESTRATOR = "orchestrator_core"
    CORE_SOURCE_HOST = "host_core"
    VALID_CORE_SOURCES = {CORE_SOURCE_ORCHESTRATOR, CORE_SOURCE_HOST}

    MANAGED_REGION_RE = re.compile(
        r"\n*<soul_runtime_boundary>.*?</soul_runtime_boundary>\n*|\n*<persona_orchestrator_prompt>.*?</persona_orchestrator_prompt>\n*",
        re.DOTALL,
    )
    HOST_RUNTIME_BOUNDARY_RE = re.compile(
        r"\n*<host_runtime_boundary>.*?</host_runtime_boundary>\n*",
        re.DOTALL,
    )
    EMOTION_BLOCK_RE = re.compile(
        r"\n*<emotion_modifier>.*?</emotion_modifier>\n*",
        re.DOTALL,
    )
    PCLTM_CONTEXT_BLOCK_RE = re.compile(
        r"\n*<pcltm_context_boundary>.*?</pcltm_context_boundary>\n*|\n*<pcltm_context>.*?</pcltm_context>\n*",
        re.DOTALL,
    )
    PCLTM_MEMORY_VIEW_BLOCK_RE = re.compile(
        r"\n*<pcltm_memory_view>.*?</pcltm_memory_view>\n*",
        re.DOTALL,
    )
    PCLTM_CONTEXT_TAG_RE = re.compile(r"</?pcltm_context>")
    PCLTM_MEMORY_VIEW_TAG_RE = re.compile(r"</?pcltm_memory_view>")
    MEMORY_PROFILE_NOTES_BLOCK_RE = re.compile(
        r"\n*<memory_profile_notes>.*?</memory_profile_notes>\n*",
        re.DOTALL,
    )
    LEGACY_MEMORY_PROFILE_NOTES_RE = re.compile(
        r"^\s*profile=[^;\n]+;\s*candidates=[^\n]+\s*$",
        re.DOTALL,
    )
    LEGACY_HERMES_MEMORY_BLOCK_RE = re.compile(
        r"(?:^|\n+)"
        r"(?:#{1,6}\s*)?(?:USER PROFILE \(who the user is\)|MEMORY \(your personal notes\))\s*\n"
        r".*?"
        r"(?="
        r"\n{2,}(?:#{1,6}\s*)?(?:USER PROFILE \(who the user is\)|MEMORY \(your personal notes\))\s*\n"
        r"|\n{2,}(?:#{1,6}\s*)?[A-Z][A-Z0-9 _/\-()]{2,}\s*\n"
        r"|\n{2,}<"
        r"|\Z"
        r")",
        re.DOTALL,
    )
    LEGACY_HERMES_MEMORY_HEADER_RE = re.compile(
        r"(?:USER PROFILE \(who the user is\)|MEMORY \(your personal notes\))"
    )
    LEGACY_USER_MEMORY_TAG_BLOCK_RE = re.compile(
        r"\n*\[USER MEMORY\].*?\[/USER MEMORY\]\n*",
        re.DOTALL,
    )

    def __init__(self, base_dir: str | Path, core_source: str = CORE_SOURCE_ORCHESTRATOR):
        if core_source not in self.VALID_CORE_SOURCES:
            raise ValueError(f"invalid core_source: {core_source}")
        self.base_dir = self._resolve_base_dir(Path(base_dir))
        self.layers_dir = self.base_dir / "soul_layers"
        self.core_source = core_source
        self._layer_template_cache: dict[Path, _LayerTemplateCacheEntry] = {}

    @staticmethod
    def _resolve_base_dir(base_dir: Path) -> Path:
        """Resolve repository-root callers to the persona_engine package root.

        Historical tests and probes instantiate ``PromptComposer('.')`` from the
        Soul-Link repository root, while production passes
        ``<soul-link-root>/packages/persona_engine`` explicitly.  Accept both so
        layer loading does not silently degrade to an empty managed prompt.
        """
        return resolve_persona_engine_base_dir(base_dir)

    def compose(
        self,
        selected_layers: list[str],
        emotion_modifier: str = "",
        memory_notes: str = "",
        shadow_only: bool = True,
    ) -> PromptComposition:
        normalized_layers = self._normalize_layers(selected_layers)
        warnings: list[str] = []
        prompt_parts: list[str] = []
        loaded_layers: list[str] = []

        for layer in normalized_layers:
            path = self.layers_dir / f"SOUL.{layer}.template.md"
            layer_text = self._read_layer_template(path)
            if layer_text is None:
                warnings.append(f"unknown layer skipped: {layer}")
                continue
            prompt_parts.append(layer_text.strip())
            loaded_layers.append(layer)

        if memory_notes.strip():
            prompt_parts.append("<memory_profile_notes>\n" + memory_notes.strip() + "\n</memory_profile_notes>")

        if emotion_modifier.strip():
            prompt_parts.append(emotion_modifier.strip())

        prompt_text = "\n\n".join(part for part in prompt_parts if part).strip() + "\n"
        prompt_hash = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()[:16]

        return PromptComposition(
            prompt_text=prompt_text,
            prompt_hash=prompt_hash,
            selected_layers=loaded_layers,
            warnings=warnings,
            shadow_only=shadow_only,
        )


    def _read_layer_template(self, path: Path) -> str | None:
        """Read a SOUL layer template with mtime/size invalidation.

        This mirrors the context-management lesson from Letta/MemGPT-style
        runtimes: active prompt construction is a scarce hot path, while durable
        layer files are cold state.  We keep the active result fresh by checking a
        cheap filesystem fingerprint and only touching file contents on cache
        miss or edit.
        """
        try:
            stat = path.stat()
        except FileNotFoundError:
            self._layer_template_cache.pop(path, None)
            return None

        cached = self._layer_template_cache.get(path)
        if cached and cached.mtime_ns == stat.st_mtime_ns and cached.size == stat.st_size:
            return cached.text

        text = path.read_text(encoding="utf-8")
        self._layer_template_cache[path] = _LayerTemplateCacheEntry(
            mtime_ns=stat.st_mtime_ns,
            size=stat.st_size,
            text=text,
        )
        return text

    def compose_with_memory_view(
        self,
        selected_layers: list[str],
        memory_view_text: str = "",
        emotion_modifier: str = "",
        shadow_only: bool = True,
        memory_notes: str = "",
    ) -> PromptComposition:
        """Compose persona layers without embedding an active PCLTM memory view.

        ``memory_view_text`` is accepted for compatibility with older callers,
        but production active memory now belongs to the host-level
        ``<pcltm_context>`` surface. Keeping a second ``<pcltm_memory_view>``
        inside the managed persona region creates duplicate active PCLTM
        surfaces and trips the watchdog.
        """
        return self.compose(
            selected_layers=selected_layers,
            emotion_modifier=emotion_modifier,
            memory_notes=memory_notes,
            shadow_only=shadow_only,
        )

    def compose_active(
        self,
        host_system_prompt: str,
        selected_layers: list[str],
        emotion_modifier: str = "",
        memory_notes: str = "",
        memory_view_text: str = "",
        preserve_host_pcltm_fallback: bool = True,
        emit_empty_pcltm_boundary: bool = False,
    ) -> PromptComposition:
        """Return an active prompt candidate with one managed region.

        The host prompt is preserved outside the managed region. Any existing
        emotion block in the host prompt is removed so the fresh modifier remains
        the final instruction in the composed prompt.

        If a live host prompt already contains the Core SOUL layer, treat it as
        the active identity owner even when this composer was initialized with
        the legacy/default ``orchestrator_core`` source. This prevents active
        gateway sessions that cached an older adapter from duplicating the core
        inside the managed region.
        """
        host_prompt = host_system_prompt or ""
        host_already_owns_core = "# Core Identity Layer" in host_prompt
        composer = self
        if self.core_source == self.CORE_SOURCE_ORCHESTRATOR and host_already_owns_core:
            composer = PromptComposer(self.base_dir, core_source=self.CORE_SOURCE_HOST)

        composition = composer.compose(
            selected_layers=selected_layers,
            emotion_modifier="",
            memory_notes=self._sanitize_active_memory_notes(memory_notes, memory_view_text=memory_view_text),
            shadow_only=False,
        )
        managed_region = (
            "<soul_runtime_boundary>\n"
            "<persona_orchestrator_prompt>\n"
            + composition.prompt_text.strip()
            + "\n</persona_orchestrator_prompt>\n"
            "</soul_runtime_boundary>"
        )

        host_without_managed = self.MANAGED_REGION_RE.sub("\n", host_prompt).strip()
        host_without_managed = self.HOST_RUNTIME_BOUNDARY_RE.sub("\n", host_without_managed).strip()
        host_context_was_stale = (
            bool(memory_view_text.strip())
            and "[system]" in memory_view_text
            and self.PCLTM_CONTEXT_BLOCK_RE.search(host_without_managed) is not None
        )
        pcltm_context = self._format_pcltm_context(memory_view_text)
        if not pcltm_context:
            pcltm_context = self._latest_pcltm_context(host_without_managed) if preserve_host_pcltm_fallback else ""
            if not pcltm_context and emit_empty_pcltm_boundary:
                pcltm_context = "<pcltm_context_boundary>\n<pcltm_context>\n</pcltm_context>\n</pcltm_context_boundary>"
            host_without_context = self.PCLTM_CONTEXT_BLOCK_RE.sub("\n", host_without_managed).strip()
        else:
            host_without_context = self.PCLTM_CONTEXT_BLOCK_RE.sub("\n", host_without_managed).strip()
        host_without_context = self.PCLTM_MEMORY_VIEW_BLOCK_RE.sub("\n", host_without_context).strip()
        host_without_emotion = self.EMOTION_BLOCK_RE.sub("\n", host_without_context).strip()
        host_without_notes = re.sub(r"\n*<memory_profile_notes>.*?</memory_profile_notes>\n*", "\n", host_without_emotion, flags=re.DOTALL).strip()
        host_without_legacy_memory = self._strip_legacy_hermes_memory_blocks(host_without_notes).strip()
        host_runtime = (
            "<host_runtime_boundary>\n" + host_without_legacy_memory + "\n</host_runtime_boundary>"
            if host_without_legacy_memory
            else ""
        )
        prompt_parts = [part for part in [host_runtime, pcltm_context, managed_region, emotion_modifier.strip()] if part]
        prompt_text = "\n\n".join(prompt_parts).strip() + "\n"
        prompt_hash = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()[:16]
        return PromptComposition(
            prompt_text=prompt_text,
            prompt_hash=prompt_hash,
            selected_layers=composition.selected_layers,
            warnings=composition.warnings,
            shadow_only=False,
        )

    def _latest_pcltm_context(self, text: str) -> str:
        """Return the newest host PCLTM block for compatibility fallback."""
        matches = list(self.PCLTM_CONTEXT_BLOCK_RE.finditer(text or ""))
        return matches[-1].group(0).strip() if matches else ""

    def _format_pcltm_context(self, memory_view_text: str) -> str:
        """Render the fresh active-memory frame as the sole tagged PCLTM surface."""
        text = (memory_view_text or "").strip()
        if not text:
            return ""
        boundary_open = "<pcltm_context_boundary>" in text
        boundary_close = "</pcltm_context_boundary>" in text
        if boundary_open and boundary_close:
            return text
        has_open = "<pcltm_context>" in text
        has_close = "</pcltm_context>" in text
        inner = self.PCLTM_CONTEXT_TAG_RE.sub("", text).strip()
        if not inner:
            return ""
        pcltm_block = text if has_open and has_close else "<pcltm_context>\n" + inner + "\n</pcltm_context>"
        return "<pcltm_context_boundary>\n" + pcltm_block.strip() + "\n</pcltm_context_boundary>"

    def _collapse_pcltm_context_blocks(self, text: str) -> str:
        """Keep at most one host-level PCLTM active-memory surface."""
        matches = list(self.PCLTM_CONTEXT_BLOCK_RE.finditer(text or ""))
        if len(matches) <= 1:
            return (text or "").strip()
        kept = matches[-1].group(0).strip()
        collapsed = self.PCLTM_CONTEXT_BLOCK_RE.sub("\n", text or "").strip()
        parts = [collapsed, kept] if collapsed else [kept]
        return "\n\n".join(part for part in parts if part).strip()

    def _strip_legacy_hermes_memory_blocks(self, text: str) -> str:
        """Remove retired Hermes USER PROFILE / MEMORY prompt surfaces.

        PCLTM/MemFS is the only active memory surface.  If an upstream host
        prompt still carries the old Hermes profile/memory sections, strip them
        before inserting the managed persona region so stale long-term memory
        cannot compete with the active ``<pcltm_context>`` frame.
        """
        if not text:
            return ""
        text = self.LEGACY_USER_MEMORY_TAG_BLOCK_RE.sub("\n", text)
        return self.LEGACY_HERMES_MEMORY_BLOCK_RE.sub("\n", text)

    def _sanitize_active_memory_notes(self, memory_notes: str, *, memory_view_text: str = "") -> str:
        """Drop legacy profile-selection notes from active prompts.

        Current active memory is owned by the host-level ``<pcltm_context>``
        surface.  Old facade/profile notes are selection metadata, not prompt
        memory, and should not be reintroduced inside the managed persona
        region.  Free-form notes used by lower-level persona tests remain valid
        and stay below the final emotion modifier.
        """
        if memory_view_text.strip():
            return ""
        if self.MEMORY_PROFILE_NOTES_BLOCK_RE.search(memory_notes or ""):
            return ""
        notes = (memory_notes or "").strip()
        if not notes:
            return ""
        if self.LEGACY_MEMORY_PROFILE_NOTES_RE.match(notes):
            return ""
        if self.LEGACY_HERMES_MEMORY_HEADER_RE.search(notes):
            return ""
        return notes

    def _normalize_layers(self, selected_layers: list[str]) -> list[str]:
        layers = list(selected_layers or [])
        layers = [layer for layer in layers if layer]
        if self.core_source == self.CORE_SOURCE_ORCHESTRATOR:
            if "core" in layers:
                layers = ["core"] + [layer for layer in layers if layer != "core"]
            else:
                layers = ["core"] + layers
        else:
            layers = [layer for layer in layers if layer != "core"]
        deduped: list[str] = []
        for layer in layers:
            if layer not in deduped:
                deduped.append(layer)
        return deduped
