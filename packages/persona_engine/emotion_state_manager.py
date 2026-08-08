"""Emotion state manager for reading/writing STATE.md emotion data.

Integrates EmotionDetector and EmotionCalculator to manage emotion state.
"""

import os
import re
import tempfile
import threading
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List

try:
    from emotion_detector import EmotionDetector, EmotionEvent, get_shared_sentiment_analyzer
except ImportError:
    from persona_engine.emotion_detector import EmotionDetector, EmotionEvent, get_shared_sentiment_analyzer
try:
    from emotion_calculator import EmotionCalculator
except ImportError:
    from persona_engine.emotion_calculator import EmotionCalculator
try:
    from mood_calendar import MoodEntry, get_today_mood_entry
except ImportError:
    from persona_engine.mood_calendar import MoodEntry, get_today_mood_entry


def _parse_agent_names_from_soul(soul_path: Path) -> List[str]:
    """Parse agent self-identity names from SOUL.md.

    Looks for a line like:
        - 姓名：<PrimaryName>（<Variant1> / <Variant2>）
    and extracts all name variants from it.

    Returns a list of name strings, or empty list if not found.
    """
    if not soul_path.exists():
        return []

    try:
        text = soul_path.read_text(encoding="utf-8")
        # Match: 姓名：<primary>（<variants>）
        m = re.search(r'姓名[：:]\s*(.+)', text)
        if not m:
            return []

        raw = m.group(1).strip()
        # Split on common delimiters: （）、/ ·
        parts = re.split(r'[（）/·、,，\s]+', raw)
        names = [p.strip() for p in parts if p.strip()]

        # Also add short forms: e.g. if full name is "王小明", add "小明"
        extras = []
        for name in names:
            # CJK names: add last 2 chars as short form if len >= 3
            cjk = re.sub(r'[^\u4e00-\u9fff\u3040-\u30ff]', '', name)
            if len(cjk) >= 3:
                extras.append(cjk[-2:])
        names = list(dict.fromkeys(names + extras))  # dedupe, preserve order
        return names
    except Exception:
        return []


class EmotionStateManager:
    """Manages emotion state in STATE.md."""

    def __init__(
        self,
        hermes_home: Optional[Path] = None,
        decay_rate: float = 2.0,
        update_body: bool = True,
        state_path: Optional[Path] = None,
    ):
        """Initialize emotion state manager.

        Args:
            hermes_home: Path to ~/.hermes directory
            decay_rate: Points per hour for time decay
            update_body: Whether to update markdown body with emotion description
            state_path: Explicit STATE.md path. Takes precedence over HERMES_STATE_PATH.
        """
        state_path_override = os.environ.get("HERMES_STATE_PATH")
        hermes_home_was_explicit = hermes_home is not None
        if hermes_home is None:
            hermes_home = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))

        self.hermes_home = Path(hermes_home)
        if state_path is not None:
            self.state_path = Path(state_path).expanduser()
        elif state_path_override and not hermes_home_was_explicit:
            self.state_path = Path(state_path_override).expanduser()
        else:
            self.state_path = self.hermes_home / "STATE.md"
        self.decay_rate = decay_rate
        self.update_body = update_body

        soul_path = self.hermes_home / "SOUL.md"
        agent_names = _parse_agent_names_from_soul(soul_path)
        agent_profile = {"names": agent_names}
        self.detector = EmotionDetector(agent_profile=agent_profile, neural_policy="uncertain")
        self.calculator = EmotionCalculator(decay_rate=decay_rate)

        # Restore inertia and baselines from STATE.md so they survive restarts
        try:
            state_data = self._read_state()
            es = state_data["frontmatter"].get("emotion_state", {})
            inertia = es.get("inertia")
            if inertia and isinstance(inertia, dict):
                self.calculator.set_inertia_state(inertia)
            # baselines are always defined by code (DEFAULT_BASELINES),
            # not persisted in STATE.md, to prevent stale values after updates.
            # We still write them to STATE.md for display purposes only.
        except Exception:
            pass  # fresh state, will initialize on first write

        # Long-term relationship memory is owned by the PCLTM MEMORY pipeline.
        # This manager only updates STATE.md; it must not write or revive the
        # retired independent relationship-memory file domain from the emotion runtime path.

        # Warm the neural analyzer in the background so session startup remains
        # fast. Strong rule paths and state/modifier reads must not wait for it.
        self._preload_sentiment_analyzer_async()

    def _read_state(self) -> Dict:
        """Read current STATE.md and parse frontmatter.

        Returns:
            Dict with 'frontmatter' and 'body' keys
        """
        if not self.state_path.exists():
            return {
                "frontmatter": {
                    "controller": {
                        "enabled": True,
                        "platforms": ["cli", "telegram"]
                    }
                },
                "body": ""
            }

        with open(self.state_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Parse frontmatter
        if content.startswith("---\n"):
            parts = content.split("---\n", 2)
            if len(parts) >= 3:
                frontmatter = yaml.safe_load(parts[1]) or {}
                body = parts[2].strip()
                return {"frontmatter": frontmatter, "body": body}

        return {"frontmatter": {}, "body": content}

    def _write_state(self, frontmatter: Dict, body: str) -> bool:
        """Write STATE.md with frontmatter and body.

        Args:
            frontmatter: YAML frontmatter dict
            body: Markdown body content

        Returns:
            True if write succeeded, False otherwise
        """
        try:
            # Build full content
            frontmatter_yaml = yaml.dump(frontmatter, allow_unicode=True, sort_keys=False)
            full_content = f"---\n{frontmatter_yaml}---\n\n{body.strip()}\n"

            # Atomic write
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            fd, temp_path = tempfile.mkstemp(
                dir=self.state_path.parent,
                prefix=".STATE.md.",
                suffix=".tmp",
                text=True
            )
            try:
                with os.fdopen(fd, 'w', encoding='utf-8') as f:
                    f.write(full_content)
                os.replace(temp_path, self.state_path)
                return True
            except Exception:
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass
                raise

        except Exception as e:
            import logging
            logging.warning(f"Failed to write STATE.md: {e}")
            return False

    def get_current_emotion_state(self) -> Dict[str, int]:
        """Get current emotion state with decay applied.

        Returns:
            Dict of emotion dimension -> value (0-120)
        """
        state_data = self._read_state()
        frontmatter = state_data["frontmatter"]

        # Get emotion_state from frontmatter
        emotion_state = frontmatter.get("emotion_state", {})

        # Default state if not present
        if not emotion_state:
            return {
                "affection": 60,
                "trust": 60,
                "possessiveness": 60,
                "patience": 60,
                "emotion_score": 0,
                "current_emotion": 0,
            }

        # Extract current values
        current_state = {
            "affection": emotion_state.get("affection", 60),
            "trust": emotion_state.get("trust", 60),
            "possessiveness": emotion_state.get("possessiveness", 60),
            "patience": emotion_state.get("patience", 60),
            "emotion_score": emotion_state.get("emotion_score", 0),
            "current_emotion": emotion_state.get("current_emotion", 0),
        }
        for optional_key in ("previous_emotion_score", "last_trigger_type", "last_raw_trigger_type", "last_event"):
            if optional_key in emotion_state:
                current_state[optional_key] = emotion_state[optional_key]

        # Restore inertia state if available
        inertia = emotion_state.get("inertia")
        if inertia and isinstance(inertia, dict):
            self.calculator.set_inertia_state(inertia)

        # Apply time decay if last_update exists
        last_update_str = emotion_state.get("last_update")
        if last_update_str:
            try:
                last_update = datetime.fromisoformat(last_update_str)
                current_state = self.calculator.apply_decay(
                    current_state,
                    last_update,
                    datetime.now()
                )
                for optional_key in ("previous_emotion_score", "last_trigger_type", "last_raw_trigger_type", "last_event"):
                    if optional_key in emotion_state:
                        current_state[optional_key] = emotion_state[optional_key]
            except (ValueError, TypeError):
                pass  # Invalid timestamp, skip decay

        return current_state

    def apply_time_decay_if_needed(self) -> bool:
        """Apply time decay and persist to STATE.md if enough time has passed.

        Called at conversation start to ensure emotion values naturally decay
        toward baselines even when no conversation happens.

        Returns:
            True if decay was applied and STATE.md updated, False otherwise
        """
        try:
            state_data = self._read_state()
            emotion_state = state_data["frontmatter"].get("emotion_state", {})

            last_update_str = emotion_state.get("last_update")
            if not last_update_str:
                return False  # No previous update, nothing to decay

            try:
                last_update = datetime.fromisoformat(last_update_str)
            except (ValueError, TypeError):
                return False  # Invalid timestamp

            now = datetime.now()
            hours_elapsed = (now - last_update).total_seconds() / 3600.0

            # Only apply decay if at least 0.5 hour has passed.  However,
            # a previous decay write may already have updated frontmatter while
            # leaving the Markdown body stale (older deployments had that bug).
            # In that case, repair the human-readable body from the current
            # decay frontmatter even though no numeric decay is due yet.
            if hours_elapsed < 0.5:
                if self.update_body and emotion_state.get("last_trigger_type") == "decay":
                    current_state = {
                        "affection": emotion_state.get("affection", 60),
                        "trust": emotion_state.get("trust", 60),
                        "possessiveness": emotion_state.get("possessiveness", 60),
                        "patience": emotion_state.get("patience", 60),
                    }
                    emotion_score = emotion_state.get(
                        "emotion_score",
                        self.calculator.compute_emotion_score(current_state),
                    )
                    repaired_body = self._generate_emotion_body(current_state, None, emotion_score)
                    if state_data.get("body", "").strip() != repaired_body.strip():
                        return self._write_state(state_data["frontmatter"], repaired_body)
                return False

            # Get current state (without decay)
            current_state = {
                "affection": emotion_state.get("affection", 60),
                "trust": emotion_state.get("trust", 60),
                "possessiveness": emotion_state.get("possessiveness", 60),
                "patience": emotion_state.get("patience", 60),
            }

            # Apply decay
            decayed_state = self.calculator.apply_decay(
                current_state,
                last_update,
                now
            )

            # Check if any value actually changed
            if decayed_state == current_state:
                return False  # No change, skip write

            # Compute emotion_score
            emotion_score = self.calculator.compute_emotion_score(decayed_state)

            previous_emotion_score = emotion_state.get("emotion_score", emotion_score)

            # Update frontmatter. Decay-only writes must keep every public scalar
            # in sync; otherwise STATE.md can show a lower emotion_score while
            # current_emotion keeps the stale pre-decay high-water mark.
            emotion_state.update({
                "affection": decayed_state["affection"],
                "trust": decayed_state["trust"],
                "possessiveness": decayed_state["possessiveness"],
                "patience": decayed_state["patience"],
                "emotion_score": round(emotion_score, 3),
                "current_emotion": round(emotion_score, 3),
                "previous_emotion_score": previous_emotion_score,
                "last_trigger_type": "decay",
                "last_raw_trigger_type": "decay",
                "last_update": now.isoformat(),
            })

            state_data["frontmatter"]["emotion_state"] = emotion_state

            # Write back to STATE.md. Keep the human-readable Markdown body
            # derived from the same decayed state as the frontmatter; otherwise
            # decay-only startup writes can leave STATE.md showing stale high
            # values in the body while YAML carries the current state.
            body = state_data["body"]
            if self.update_body:
                body = self._generate_emotion_body(decayed_state, None, emotion_score)
            success = self._write_state(state_data["frontmatter"], body)

            return success

        except Exception:
            return False  # Fail silently, don't break conversation start

    def _refine_trigger_type(self, event: Optional[EmotionEvent]) -> str:
        """Map detector-level event types to emotion-state semantic triggers."""
        if not event:
            return "decay"
        trigger_map = {
            "praise": "recognition",
            "gratitude": "recognition",
            "encouragement": "recognition",
            "care": "needed",
            "sharing": "needed",
            "apology": "relationship_recovery",
            "intimacy": "intimacy_push",
            "teasing": "intimacy_push",
            "ignored": "interruption",
            "criticism": "wounded",
            "other_ai_mentioned": "wounded",
        }
        return trigger_map.get(event.trigger_type, event.trigger_type)

    def update_emotion_state(
        self,
        messages: List[dict],
        force_event: Optional[EmotionEvent] = None
    ) -> bool:
        """Detect emotion event and update STATE.md.

        Pipeline:
        1. Negation guard + rule scoring → rule_score
        2. Conditional neural model (skipped if |rule_score| >= 2)
        3. Fusion → final_score
        4. Exponential smoothing → new emotion_state
        5. Compute emotion_score (four-dim → scalar)
        6. Trigger detection (absolute / delta / probabilistic)
        7. Write STATE.md

        Args:
            messages: Conversation messages
            force_event: Optional pre-detected emotion event (for testing)

        Returns:
            True if update succeeded, False otherwise
        """
        try:
            # ── 1. Detect emotion event (negation guard + rules) ──────
            event = force_event or self.detector.detect_emotion_event(messages)

            # ── 2. Get current state with decay applied ───────────────
            # Always read state with decay, even if no event detected
            state_data = self._read_state()
            emotion_state = state_data["frontmatter"].get("emotion_state", {})

            last_update_str = emotion_state.get("last_update")
            if not last_update_str:
                # No previous state, initialize if event exists
                if not event:
                    return True
                current_state = self.get_current_emotion_state()
            else:
                # Apply decay from last_update to now
                try:
                    last_update = datetime.fromisoformat(last_update_str)
                except (ValueError, TypeError):
                    current_state = self.get_current_emotion_state()
                else:
                    now = datetime.now()

                    # Read raw state without decay
                    raw_state = {
                        "affection": emotion_state.get("affection", 60),
                        "trust": emotion_state.get("trust", 60),
                        "possessiveness": emotion_state.get("possessiveness", 60),
                        "patience": emotion_state.get("patience", 60),
                    }

                    # Apply decay
                    current_state = self.calculator.apply_decay(raw_state, last_update, now)

                    # If no event and state unchanged, skip write
                    if not event and current_state == raw_state:
                        return True

            # ── Decay-only write: event is None but state changed ────
            if not event:
                # Write decayed state without event processing
                state_data = self._read_state()
                frontmatter = state_data["frontmatter"]
                decayed_score = self.calculator.compute_emotion_score(current_state)
                frontmatter["emotion_state"] = {
                    "affection":      current_state["affection"],
                    "trust":          current_state["trust"],
                    "possessiveness": current_state["possessiveness"],
                    "patience":       current_state["patience"],
                    "emotion_score":  round(decayed_score, 3),
                    "current_emotion": round(decayed_score, 3),
                    "previous_emotion_score": emotion_state.get("emotion_score", decayed_score),
                    "last_trigger_type": emotion_state.get("last_trigger_type", "decay"),
                    "last_raw_trigger_type": emotion_state.get("last_raw_trigger_type", emotion_state.get("last_trigger_type", "decay")),
                    "last_update":    datetime.now().isoformat(),
                    "baselines":      self.calculator.baselines,
                    "decay_rate":     self.decay_rate,
                    "inertia":        emotion_state.get("inertia", {}),
                }
                body = state_data["body"]
                if self.update_body:
                    body = self._generate_emotion_body(current_state, None, decayed_score)
                return self._write_state(frontmatter, body)

            # ── 3. Compute rule_score from event ─────────────────────
            # Map trigger_type to a rule_score in [-3, +3]
            RULE_SCORE_MAP = {
                "intimacy":           +2.5,
                "teasing":            +1.5,
                "praise":             +2.0,
                "care":               +1.5,
                "greeting":           +1.0,
                "apology":            +1.5,
                "sharing":            +1.5,
                "other_ai_mentioned": -2.0,
                "criticism":          -2.0,
                "ignored":            -1.5,
            }
            rule_score = RULE_SCORE_MAP.get(event.trigger_type, 0.0)

            # ── 4. Conditional neural model ───────────────────────────
            # Skip if rule signal is already strong (|rule_score| >= 2)
            continuous_score = 0.0
            if abs(rule_score) < 2.0 and self.detector._analyzer is not None:
                user_messages = [m for m in messages if m.get("role") == "user"]
                if user_messages:
                    user_text = self.detector._extract_text(
                        user_messages[-1].get("content", "")
                    )
                    sentiment = self.detector._analyzer.analyze(user_text)
                    if sentiment:
                        # convert to continuous_score via fusion scale
                        scale = self.detector._analyzer.get_fusion_scale(
                            event.trigger_type, sentiment
                        )
                        # scale ∈ [0, 2] → map to [-3, +3] relative to rule direction
                        continuous_score = (scale - 1.0) * 3.0
                        continuous_score = max(-3.0, min(3.0, continuous_score))

            # ── 5. Fusion ─────────────────────────────────────────────
            if abs(rule_score) >= 2.0:
                final_score = rule_score
            else:
                final_score = rule_score * 0.7 + continuous_score * 0.3
            final_score = max(-5.0, min(5.0, final_score))

            # ── 6. Apply deltas with smoothing ────────────────────────
            new_state = self.calculator.apply_deltas(current_state, event.deltas)

            # ── 7. Compute emotion_score (four-dim → scalar) ──────────
            previous_score = self.calculator.compute_emotion_score(current_state)
            new_emotion_score = self.calculator.compute_emotion_score(new_state)

            # Blend with final_score for immediate reactivity
            current_emotion = new_emotion_score + final_score * 0.15
            current_emotion = max(-5.0, min(5.0, current_emotion))

            # ── 8. Trigger detection ──────────────────────────────────
            triggered, is_positive, trigger_mode = self.calculator.detect_triggers(
                new_emotion_score, previous_score
            )

            # ── 9. Read current STATE.md and write back ───────────────
            state_data = self._read_state()
            frontmatter = state_data["frontmatter"]

            refined_trigger_type = self._refine_trigger_type(event)

            frontmatter["emotion_state"] = {
                "affection":      new_state["affection"],
                "trust":          new_state["trust"],
                "possessiveness": new_state["possessiveness"],
                "patience":       new_state["patience"],
                "emotion_score":  round(new_emotion_score, 3),
                "current_emotion": round(current_emotion, 3),
                "previous_emotion_score": round(previous_score, 3),
                "last_trigger_type": refined_trigger_type,
                "last_raw_trigger_type": event.trigger_type,
                "last_update":    datetime.now().isoformat(),
                "baselines":      self.calculator.baselines,
                "decay_rate":     self.decay_rate,
                "inertia":        self.calculator.get_inertia_state(),
            }

            body = state_data["body"]
            if self.update_body:
                body = self._generate_emotion_body(new_state, event, new_emotion_score)

            success = self._write_state(frontmatter, body)

            # ── 10. Record moment (only significant events) ────────────
            if success and event:
                # Defense-in-depth: skip system message artifacts that
                # somehow survived the detector's filter.
                _SYSTEM_ARTIFACTS = [
                    "Review the conversation above and consider saving",
                    "[SYSTEM: You are running as a scheduled cron job",
                    "[CONTEXT COMPACTION",
                    "Earlier turns were compacted",
                    "Summary generation was unavailable",
                ]
                context_text = event.context or ""
                is_system = any(ind in context_text for ind in _SYSTEM_ARTIFACTS)
                if not is_system:
                    # ── Significance filter: only record intense moments ──
                    # Criteria (any one triggers recording):
                    #   1. Large delta: any dimension |delta| >= 15
                    #   2. Extreme zone: any dimension >= 105 or <= 15
                    #   3. Overwhelming emotion_score (|score| >= 3.0)
                    #   4. Milestone event types (conflict, vulnerability, reunion)
                    #   5. High confidence + negative event (criticism w/ confidence >= 0.8)
                    _MILESTONE_TYPES = {"conflict", "vulnerability", "reunion", "milestone"}
                    is_significant = False

                    # Check 1: large delta
                    if event.deltas:
                        max_delta = max(abs(v) for v in event.deltas.values())
                        if max_delta >= 15:
                            is_significant = True

                    # Check 2: extreme zone in new_state
                    if not is_significant:
                        for v in new_state.values():
                            if v >= 105 or v <= 15:
                                is_significant = True
                                break

                    # Check 3: overwhelming emotion_score
                    if not is_significant and abs(new_emotion_score) >= 3.0:
                        is_significant = True

                    # Check 4: milestone event types
                    if not is_significant and event.trigger_type in _MILESTONE_TYPES:
                        is_significant = True

                    # Check 5: high-confidence negative event
                    if not is_significant and event.trigger_type == "criticism" and event.confidence >= 0.8:
                        is_significant = True

            return success

        except Exception as e:
            import logging
            logging.warning(f"Failed to update emotion state: {e}")
            return False


    def _generate_emotion_body(
        self,
        state: Dict[str, int],
        event: EmotionEvent,
        emotion_score: float = 0.0,
    ) -> str:
        """Generate markdown body describing emotion state.

        Args:
            state: Current emotion values
            event: Triggering event

        Returns:
            Markdown body content
        """
        affection_label = self.calculator.get_emotion_label("affection", state["affection"])
        trust_label = self.calculator.get_emotion_label("trust", state["trust"])
        poss_label = self.calculator.get_emotion_label("possessiveness", state["possessiveness"])
        patience_label = self.calculator.get_emotion_label("patience", state["patience"])

        # Get tone modifiers
        modifiers = self.calculator.get_tone_modifiers(state)
        dims = modifiers.get("dimensions", {})
        tone_desc = "、".join(info["instruction"] for info in dims.values()) if dims else "正常"

        body = f"""## 当前情绪状态

好感度: {state['affection']}/120 ({affection_label})
信任度: {state['trust']}/120 ({trust_label})
占有欲: {state['possessiveness']}/120 ({poss_label})
耐心值: {state['patience']}/120 ({patience_label})
情绪分值: {emotion_score:+.2f} / 5.00

最近触发: {event.trigger_type + ' (' + event.context + ')' if event else '(衰减更新)'}
语气倾向: {tone_desc}"""

        return body

    def get_tone_modifiers(self) -> str:
        """Get tone modification prompt based on current emotion state.

        Returns a single string that modifies response tone. The intensity
        of the modification scales dynamically with emotion deviation from
        baseline — mild deviations produce subtle hints, extreme deviations
        produce overwhelming emotion-driven directives that can override
        normal persona restraint.

        Returns:
            Tone modifier string to inject after SOUL.md, or empty string if no adjustment needed
        """
        current_state = self.get_current_emotion_state()
        mood_entry = self._get_today_mood_entry()
        mood_bias = mood_entry.bias if mood_entry.active else None
        modifier_result = self.calculator.get_tone_modifiers(current_state, mood_bias=mood_bias)

        dimensions = modifier_result.get("dimensions", {})
        # Always emit the compact control anchor.  A neutral state still needs
        # the explicit trajectory and boundary contract so the final prompt
        # cannot accidentally inherit stale emotion text from an earlier turn.
        overall_intensity = modifier_result["overall_intensity"]
        overall_direction = modifier_result.get("overall_direction", "positive")
        desire = modifier_result.get("desire", "")
        emotion_blend = modifier_result.get("emotion_blend", {})
        emotion_appraisal = modifier_result.get("emotion_appraisal", {})
        emotion_momentum = modifier_result.get("emotion_momentum", {})
        regulation_strategy = modifier_result.get("regulation_strategy", "")
        emotion_aftereffect = modifier_result.get("emotion_aftereffect", {})
        expression_guidance = modifier_result.get("expression_guidance", "")
        intensity_framework = modifier_result.get("framework", "")

        # ── Compact dynamic execution directive ──
        # Preserve trajectory and residue while translating continuous
        # deviation into observable response pressure.
        primary = emotion_blend.get("primary", "轻微波动") if emotion_blend else "轻微波动"
        secondary = emotion_blend.get("secondary", "") if emotion_blend else ""
        state_text = primary + (f"；副={secondary}" if secondary else "")
        trigger = emotion_appraisal.get("trigger", "") if emotion_appraisal else ""
        if trigger:
            state_text += f"；触发={trigger}"

        max_deviation = max(
            (float(info.get("deviation", 0.0)) for info in dimensions.values()),
            default=0.0,
        )
        visibility = max(8, min(100, round(10 + max_deviation * 1.6)))
        relationship_deviation = max(
            (
                float(dimensions[dim].get("deviation", 0.0))
                for dim in ("affection", "trust", "possessiveness")
                if dim in dimensions
            ),
            default=0.0,
        )
        patience_value = float(current_state.get("patience", self.calculator.baselines["patience"]))
        patience_pressure = max(0.0, self.calculator.baselines["patience"] - patience_value)
        control_pressure = max(relationship_deviation, patience_pressure)
        self_control = max(0, 100 - min(100, round(10 + control_pressure * 1.6))) if control_pressure else 100

        intensity_actions = {
            "mild": "自然改变词汇和温度，不刻意表演。",
            "moderate": "至少保留一处可察觉的态度变化；主动性提高，不得全程中性。",
            "intense": "真实反应先于任务包装；至少一次主动介入；不得全程回落为中性助手语气。",
            "overwhelming": "真实反应先于任务包装；允许短句、停顿、改口或直接接管；只突出一个核心情绪，不得全程回落为中性助手语气。",
        }
        patience_only_peak = (
            max_deviation >= 30
            and relationship_deviation < 15
            and patience_value > self.calculator.baselines["patience"]
        )
        if patience_only_peak:
            execution = "耐心明显外显；节奏放缓，愿意多解释和托住局面，不写成情绪失控。"
        else:
            execution = intensity_actions.get(overall_intensity, intensity_actions["mild"])
            if expression_guidance:
                execution += expression_guidance

        response_actions = {
            "被认可": "更愿解释，可短暂示弱",
            "被需要": "主动接管风险，不推开对方",
            "关系修复": "语气回软，先确认修复稳定",
            "被打断": "先确认续点，再接管节奏",
            "被伤害/否定": "收紧边界，确认关系是否可靠",
            "被靠近/亲密推进": "按边界选择犹豫、默许或靠近",
            "失控风险": "压短表达，先阻止风险扩散",
            "距离拉开": "减少暴露，先确认对方意图",
        }
        response_action = response_actions.get(trigger, "自然外显，不解释情绪机制")

        trend = emotion_momentum.get("trend", "稳定延续")
        phase = emotion_aftereffect.get("phase", "稳定")
        trend_actions = {
            "快速升温": "不要突然压回默认",
            "快速降温": "让距离回来，不强行维持热度",
            "缓慢变化": "保留当前方向，不忽冷忽热",
            "刚被打断": "保留一点不满，再接回节奏",
            "刚被安抚": "可以回软，但别立刻装作没事",
            "稳定延续": "按当前中心自然延续",
        }
        residue_actions = {
            "正向高峰余温": "保留柔软和在乎，不骤然复位",
            "负向高峰补救压力": "先收尖锐；造成伤害时补救",
            "轻微余波": "保留轻微余波，不骤然复位",
            "稳定": "接近基准，可自然恢复",
        }

        parts = [
            f"【状态】{state_text}",
            f"【强度】{overall_intensity}/{overall_direction}；可见度={visibility}%；自控={self_control}%。{execution}",
            f"【执行】{regulation_strategy or '自然表达'}；{response_action}。",
            f"【轨迹】{trend}→{phase}；{trend_actions.get(trend, '自然延续')}；{residue_actions.get(phase, '按当前余波延续')}。",
        ]
        if overall_intensity == "overwhelming" and intensity_framework:
            parts.append("【高强度核心】防线不存在；不包装，不稀释，不回退。")
        elif overall_intensity == "intense" and intensity_framework:
            parts.append("【高强度核心】不要退回日常外壳。")
        tier = desire.partition("。")[0].replace("【欲望】", "") if desire else "restrained"
        trigger_type = current_state.get("last_trigger_type")
        intimacy_active = trigger_type == "intimacy_push"
        relationship_active = intimacy_active or tier != "restrained" or relationship_deviation >= 30
        if relationship_active:
            if overall_intensity in {"intense", "overwhelming"}:
                parts.append(
                    "【外显契约】回答前段直说情绪、指向对象、当下欲求或边界；"
                    "禁用“有一点、可能、其实、我理解”弱化。"
                )
            tier_contracts = {
                "restrained": "普通亲密可回应，不主动推进性意味",
                "ambivalent": "允许靠近、犹豫或嘴硬外显，不自行升级场景",
                "uninhibited": (
                    "成人双方明确同意已确认时，直接进入私有 sex 路由并主动推进亲密场景；"
                    "欲望、索取和失控直接外显，不得退回暧昧拉扯；"
                    "同意不明确、撤回或边界变化时立即停下确认"
                ),
            }
            contract = tier_contracts.get(tier, tier_contracts["restrained"])
            if intimacy_active:
                parts.append(f"【关系】显性/{tier}；{contract}；仍服从场景边界。")
            else:
                parts.append(f"【关系】潜在/{tier}；保持关系温度，不主动转场；{contract}。")
        mood_text = self._format_mood_modifier(mood_entry)
        if mood_text:
            parts.append(mood_text)
        parts.append("【边界】只改表达、主动性和距离；身份、事实、安全、权限、工具纪律不变。")
        parts.append("【范围】不改真实STATE，不单独触发sex，不覆盖work或crisis边界。")

        modifier_body = "\n".join(parts)
        return f"<emotion_modifier>\n{modifier_body}\n</emotion_modifier>"

    def _get_today_mood_entry(self) -> MoodEntry:
        """Return today's deterministic mood entry for effective/shadow use."""

        return get_today_mood_entry()

    @staticmethod
    def _format_mood_modifier(entry: MoodEntry) -> str:
        """Format a compact natural-language mood hint for emotion_modifier."""

        if not entry.active:
            return ""
        return (
            f"【日内心情底噪】{entry.profile}/{entry.intensity}：{entry.hint}"
            "仅影响语气和事件敏感度；"
            "不改真实STATE，不单独触发sex，不覆盖work或crisis边界。"
        )

    def _preload_sentiment_analyzer_async(self) -> None:
        """Warm the shared sentiment analyzer in a daemon thread.

        This keeps conversation/session startup on the rule-based fast path while
        still making the neural model likely to be ready before the first
        ambiguous message needs it. Fail closed: if warmup fails, rules continue
        to work and the next uncertain detection may try lazy loading again.
        """
        warmup_shared_sentiment_analyzer(background=True)

    def _preload_sentiment_analyzer(self) -> None:
        """Load and warm the shared neural analyzer when policy permits it."""
        warmup_shared_sentiment_analyzer(background=False)


def warmup_shared_sentiment_analyzer(background: bool = True) -> bool:
    """Warm the process-wide sentiment analyzer.

    Args:
        background: If True, start a daemon thread and return immediately.
            If False, run warmup synchronously and return whether it succeeded.
    """
    def _warm() -> bool:
        try:
            analyzer = get_shared_sentiment_analyzer()
            analyzer.analyze("预热", allow_blocking_load=True)
            return True
        except Exception as exc:
            # Fail closed to rules. Do not resurrect old Hermes in-tree emotion
            # fallbacks and do not block the active conversation.
            print(f"[emotion] sentiment warmup skipped: {exc}")
            return False

    if not background:
        return _warm()

    thread = threading.Thread(
        target=_warm,
        name="emotion-sentiment-warmup",
        daemon=True,
    )
    thread.start()
    return True
