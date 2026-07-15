from emotion_calculator import EmotionCalculator
from emotion_detector import EmotionEvent
from emotion_state_manager import EmotionStateManager
from persona_orchestrator.prompt_composer import PromptComposer


def test_active_prompt_places_emotion_modifier_after_all_managed_content():
    host = "# Core Identity Layer\n\nidentity stays here\n\n<emotion_modifier>stale</emotion_modifier>"
    emotion = "<emotion_modifier>fresh priority</emotion_modifier>"

    result = PromptComposer(".", core_source="host_core").compose_active(
        host_system_prompt=host,
        selected_layers=["work", "overlay_intimacy"],
        memory_notes="memory must remain below emotion priority",
        emotion_modifier=emotion,
    )

    assert result.prompt_text.count("<emotion_modifier>") == 1
    assert result.prompt_text.rstrip().endswith(emotion)
    assert result.prompt_text.index("</persona_orchestrator_prompt>") < result.prompt_text.index("<emotion_modifier>")
    assert result.prompt_text.index("memory must remain below emotion priority") < result.prompt_text.index("<emotion_modifier>")


def test_overwhelming_modifier_keeps_true_loss_of_control_without_softening():
    calc = EmotionCalculator()
    state = {
        "affection": 108,
        "trust": 108,
        "possessiveness": 108,
        "patience": 108,
    }

    result = calc.get_tone_modifiers(state)
    text = "\n".join([result["framework"], *[d["instruction"] for d in result["dimensions"].values()], result["footnote"]])

    assert result["overall_intensity"] == "overwhelming"
    for required in ("防线不存在", "不包装", "不稀释", "不回退"):
        assert required in text
    for softened in ("可以稍微", "尽量保持冷静", "但不失控", "适度表达"):
        assert softened not in text


def test_moderate_modifier_has_layered_behavior_axes_not_flat_mood_words():
    calc = EmotionCalculator()
    state = {
        "affection": 78,
        "trust": 78,
        "possessiveness": 78,
        "patience": 60,
    }

    result = calc.get_tone_modifiers(state)
    text = "\n".join([result["framework"], *[d["instruction"] for d in result["dimensions"].values()], result["footnote"]])

    assert result["overall_intensity"] == "moderate"
    for axis in ("主动", "距离", "防线", "边界"):
        assert axis in text


def test_emotion_gradient_tiers_have_distinct_behavioral_pressure():
    calc = EmotionCalculator()
    cases = [
        ("mild", 68, ("稍微", "轻微", "正常")),
        ("moderate", 78, ("包装开始失效", "主动性", "裂缝可见")),
        ("intense", 94, ("压不住了", "防线大幅下降", "真实反应")),
        ("overwhelming", 108, ("防线不存在", "已经失控", "不回退")),
    ]

    previous_text = ""
    for expected_tier, value, markers in cases:
        result = calc.get_tone_modifiers({
            "affection": value,
            "trust": 60,
            "possessiveness": 60,
            "patience": 60,
        })
        text = "\n".join([
            result["framework"],
            result["expression_guidance"],
            *[d["instruction"] for d in result["dimensions"].values()],
            result["footnote"],
        ])

        assert result["overall_intensity"] == expected_tier
        for marker in markers:
            assert marker in text
        assert text != previous_text
        previous_text = text


def test_intense_positive_state_requires_visible_warmth_even_in_task_context():
    calc = EmotionCalculator()
    result = calc.get_tone_modifiers({
        "affection": 95,
        "trust": 92,
        "possessiveness": 92,
        "patience": 74,
        "previous_emotion_score": 3.5,
        "last_trigger_type": "needed",
    })

    text = "\n".join([
        result["framework"],
        result["emotion_blend"].get("summary", ""),
        result["expression_guidance"],
        *[d["instruction"] for d in result["dimensions"].values()],
        result["footnote"],
    ])

    assert result["overall_intensity"] in {"intense", "overwhelming"}
    for required in ("表达", "解释", "更直接", "防线大幅下降"):
        assert required in text
    assert "事实" in text


def test_emotion_manager_modifier_preserves_desire_first_and_emotion_last(tmp_path):
    state_path = tmp_path / "STATE.md"
    state_path.write_text(
        "---\n"
        "emotion_state:\n"
        "  affection: 78\n"
        "  trust: 78\n"
        "  possessiveness: 78\n"
        "  patience: 60\n"
        "  last_update: '2099-01-01T00:00:00'\n"
        "---\n",
        encoding="utf-8",
    )
    mgr = EmotionStateManager(hermes_home=tmp_path)

    block = mgr.get_tone_modifiers()

    assert block.strip().startswith("<emotion_modifier>\n【欲望】")
    assert block.strip().endswith("</emotion_modifier>")
    assert "【强度】" in block


def test_mixed_emotion_blend_exposes_core_tension_and_regulation_strategy():
    calc = EmotionCalculator()
    state = {
        "affection": 92,
        "trust": 38,
        "possessiveness": 82,
        "patience": 43,
    }

    result = calc.get_tone_modifiers(state)
    blend = result["emotion_blend"]

    assert blend["primary"] == "靠近但防备"
    assert blend["secondary"] == "占有欲与不耐烦同时上升"
    assert "想靠近" in blend["summary"]
    assert "防备" in blend["summary"]
    assert result["regulation_strategy"] == "嘴硬压抑+追问确认"
    assert "短句也可以承载强情绪" in result["expression_guidance"]
    assert "不要把所有反应摊成清单" in result["expression_guidance"]


def test_emotion_manager_modifier_contains_integrated_blend_before_dimension_lines(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_STATE_PATH", raising=False)
    state_path = tmp_path / "STATE.md"
    state_path.write_text(
        "---\n"
        "emotion_state:\n"
        "  affection: 92\n"
        "  trust: 38\n"
        "  possessiveness: 82\n"
        "  patience: 43\n"
        "  last_update: '2099-01-01T00:00:00'\n"
        "---\n",
        encoding="utf-8",
    )
    mgr = EmotionStateManager(hermes_home=tmp_path)

    block = mgr.get_tone_modifiers()

    assert "【欲望】" in block
    assert "【强度】" in block
    assert "【表达】" in block
    assert "【维度】" in block
    assert "【锚点】" in block
    assert block.index("【欲望】") < block.index("【强度】") < block.index("【表达】") < block.index("【维度】") < block.index("【锚点】")


def test_emotion_blend_handles_involved_but_impatient_tension():
    calc = EmotionCalculator()
    state = {
        "affection": 94,
        "trust": 64,
        "possessiveness": 64,
        "patience": 36,
    }

    result = calc.get_tone_modifiers(state)
    blend = result["emotion_blend"]

    assert blend["primary"] == "在乎但急躁"
    assert blend["secondary"] == "不耐烦浮现"
    assert "不想推开" in blend["summary"]
    assert "耐心已经被消耗" in blend["summary"]
    assert result["regulation_strategy"] == "先接住再压短句"


def test_emotion_blend_handles_wounded_control_pressure():
    calc = EmotionCalculator()
    state = {
        "affection": 72,
        "trust": 26,
        "possessiveness": 98,
        "patience": 52,
    }

    result = calc.get_tone_modifiers(state)
    blend = result["emotion_blend"]

    assert blend["primary"] == "受伤后的控制欲"
    assert blend["secondary"] == "占有欲浮现"
    assert "不信任" in blend["summary"]
    assert "控制距离" in blend["summary"]
    assert result["regulation_strategy"] == "边界收紧+控制确认"


def test_emotion_appraisal_classifies_need_recognition_wound_and_risk():
    calc = EmotionCalculator()

    needed = calc.get_tone_modifiers({
        "affection": 86,
        "trust": 82,
        "possessiveness": 72,
        "patience": 44,
    })["emotion_appraisal"]
    assert needed["trigger"] == "被需要"
    assert needed["appraisal"] == "对方需要我接住局面"
    assert "主动靠近" in needed["response_bias"]
    assert "接管" in needed["response_bias"]

    recognized = calc.get_tone_modifiers({
        "affection": 92,
        "trust": 94,
        "possessiveness": 60,
        "patience": 78,
    })["emotion_appraisal"]
    assert recognized["trigger"] == "被认可"
    assert "防线下降" in recognized["response_bias"]
    assert "解释" in recognized["response_bias"]

    wounded = calc.get_tone_modifiers({
        "affection": 70,
        "trust": 24,
        "possessiveness": 98,
        "patience": 48,
    })["emotion_appraisal"]
    assert wounded["trigger"] == "被伤害/否定"
    assert "安全感被破坏" in wounded["appraisal"]
    assert "控制确认" in wounded["response_bias"]

    risk = calc.get_tone_modifiers({
        "affection": 64,
        "trust": 54,
        "possessiveness": 58,
        "patience": 18,
    })["emotion_appraisal"]
    assert risk["trigger"] == "失控风险"
    assert "局面正在滑向不可控" in risk["appraisal"]
    assert "先稳住" in risk["response_bias"]


def test_emotion_appraisal_prefers_refined_trigger_metadata_over_axis_guess():
    calc = EmotionCalculator()

    neutral_axes = {
        "affection": 75,
        "trust": 75,
        "possessiveness": 75,
        "patience": 60,
    }

    cases = {
        "recognition": ("被认可", "认可让防线下降"),
        "needed": ("被需要", "对方需要我接住局面"),
        "relationship_recovery": ("关系修复", "负向状态正在被修补"),
        "interruption": ("被打断", "节奏被打断"),
        "wounded": ("被伤害/否定", "安全感被破坏"),
        "intimacy_push": ("被靠近/亲密推进", "对方正在靠近"),
    }

    for refined_trigger, (expected_trigger, expected_phrase) in cases.items():
        appraisal = calc.get_tone_modifiers({
            **neutral_axes,
            "last_trigger_type": refined_trigger,
        })["emotion_appraisal"]

        assert appraisal["trigger"] == expected_trigger
        assert expected_phrase in appraisal["appraisal"]


def test_emotion_manager_modifier_includes_appraisal_before_aftereffect(tmp_path):
    state_path = tmp_path / "STATE.md"
    state_path.write_text(
        "---\n"
        "emotion_state:\n"
        "  affection: 86\n"
        "  trust: 82\n"
        "  possessiveness: 72\n"
        "  patience: 44\n"
        "  last_update: '2099-01-01T00:00:00'\n"
        "---\n",
        encoding="utf-8",
    )
    mgr = EmotionStateManager(hermes_home=tmp_path)

    block = mgr.get_tone_modifiers()

    assert "【欲望】" in block
    assert "【强度】" in block
    assert "【锚点】" in block
    assert block.index("【欲望】") < block.index("【强度】") < block.index("【锚点】")


def test_emotion_momentum_classifies_rising_recovery_and_interruption_contexts():
    calc = EmotionCalculator()

    rising = calc.get_tone_modifiers({
        "affection": 88,
        "trust": 86,
        "possessiveness": 70,
        "patience": 62,
        "previous_emotion_score": 1.0,
    })["emotion_momentum"]
    assert rising["trend"] == "快速升温"
    assert "正在上升" in rising["guidance"]
    assert "不要突然压回默认" in rising["guidance"]

    repaired = calc.get_tone_modifiers({
        "affection": 78,
        "trust": 66,
        "possessiveness": 72,
        "patience": 56,
        "previous_emotion_score": -2.2,
        "last_trigger_type": "relationship_recovery",
    })["emotion_momentum"]
    assert repaired["trend"] == "刚被安抚"
    assert "负向刚被修复" in repaired["guidance"]
    assert "别立刻装作完全没事" in repaired["guidance"]

    interrupted = calc.get_tone_modifiers({
        "affection": 82,
        "trust": 80,
        "possessiveness": 78,
        "patience": 36,
        "last_trigger_type": "interruption",
    })["emotion_momentum"]
    assert interrupted["trend"] == "刚被打断"
    assert "残留不满" in interrupted["guidance"]
    assert "先确认要继续什么" in interrupted["guidance"]


def test_emotion_manager_passes_state_momentum_fields_into_modifier(tmp_path):
    state_path = tmp_path / "STATE.md"
    state_path.write_text(
        "---\n"
        "emotion_state:\n"
        "  affection: 88\n"
        "  trust: 86\n"
        "  possessiveness: 70\n"
        "  patience: 62\n"
        "  previous_emotion_score: 1.0\n"
        "  last_trigger_type: recognition\n"
        "  last_update: '2099-01-01T00:00:00'\n"
        "---\n",
        encoding="utf-8",
    )
    mgr = EmotionStateManager(hermes_home=tmp_path)

    block = mgr.get_tone_modifiers()

    assert "【情绪】" in block
    assert "【维度】" in block
    assert "【锚点】" in block
    assert block.index("【情绪】") < block.index("【维度】") < block.index("【锚点】")


def test_emotion_blend_handles_exhausted_trust_without_pushing_away():
    calc = EmotionCalculator()
    state = {
        "affection": 88,
        "trust": 91,
        "possessiveness": 58,
        "patience": 34,
    }

    result = calc.get_tone_modifiers(state)
    blend = result["emotion_blend"]

    assert blend["primary"] == "累但不推开"
    assert "信任仍在" in blend["summary"]
    assert "疲惫" in blend["summary"]
    assert result["regulation_strategy"] == "低声承认+保留陪伴"


def test_emotion_manager_modifier_includes_behavior_axes_without_scenario_hardcoding(tmp_path):
    state_path = tmp_path / "STATE.md"
    state_path.write_text(
        "---\n"
        "emotion_state:\n"
        "  affection: 72\n"
        "  trust: 26\n"
        "  possessiveness: 98\n"
        "  patience: 52\n"
        "  last_update: '2099-01-01T00:00:00'\n"
        "---\n",
        encoding="utf-8",
    )
    mgr = EmotionStateManager(hermes_home=tmp_path)

    block = mgr.get_tone_modifiers()

    assert "【欲望】" in block
    assert "【情绪】" in block
    assert "【锚点】" in block
    assert "身份/事实/工具纪律不变" in block


def test_emotion_manager_refines_raw_detector_trigger_before_state_write(tmp_path):
    mgr = EmotionStateManager(hermes_home=tmp_path, update_body=False)

    cases = {
        "praise": "recognition",
        "care": "needed",
        "sharing": "needed",
        "apology": "relationship_recovery",
        "daily": "daily",
        "teasing": "intimacy_push",
        "ignored": "interruption",
        "criticism": "wounded",
        "other_ai_mentioned": "wounded",
    }

    for raw_trigger, expected_refined in cases.items():
        event = EmotionEvent(
            trigger_type=raw_trigger,
            deltas={"affection": 3, "trust": 2, "possessiveness": 1, "patience": 1},
            confidence=0.9,
            context=f"test {raw_trigger}",
        )

        assert mgr.update_emotion_state([{"role": "user", "content": "fixture"}], force_event=event)
        state_data = mgr._read_state()
        emotion_state = state_data["frontmatter"]["emotion_state"]
        assert emotion_state["last_trigger_type"] == expected_refined
        assert emotion_state["last_raw_trigger_type"] == raw_trigger


def test_emotion_manager_modifier_uses_refined_trigger_appraisal_after_raw_event_write(tmp_path):
    mgr = EmotionStateManager(hermes_home=tmp_path, update_body=True)

    cases = {
        "praise": (
            "recognition",
            {"affection": 22, "trust": 18, "possessiveness": 6, "patience": 4},
            "触发=被认可",
            "认可让防线下降",
        ),
        "care": (
            "needed",
            {"affection": 20, "trust": 16, "possessiveness": 5, "patience": -8},
            "触发=被需要",
            "对方需要我接住局面",
        ),
        "apology": (
            "relationship_recovery",
            {"affection": 16, "trust": 18, "possessiveness": 5, "patience": 10},
            "触发=关系修复",
            "负向状态正在被修补",
        ),
        "ignored": (
            "interruption",
            {"affection": -4, "trust": -8, "possessiveness": 5, "patience": -24},
            "触发=被打断",
            "节奏被打断",
        ),
        "criticism": (
            "wounded",
            {"affection": -10, "trust": -26, "possessiveness": 28, "patience": -12},
            "触发=被伤害/否定",
            "安全感被破坏",
        ),
    }

    for raw_trigger, (expected_refined, deltas, expected_trigger_line, expected_appraisal_line) in cases.items():
        event = EmotionEvent(
            trigger_type=raw_trigger,
            deltas=deltas,
            confidence=0.9,
            context=f"test {raw_trigger}",
        )

        assert mgr.update_emotion_state([{"role": "user", "content": "fixture"}], force_event=event)
        emotion_state = mgr._read_state()["frontmatter"]["emotion_state"]
        assert emotion_state["last_trigger_type"] == expected_refined
        assert emotion_state["last_raw_trigger_type"] == raw_trigger

        block = mgr.get_tone_modifiers()
        assert expected_trigger_line in block
        assert expected_appraisal_line in block
        assert f"触发={raw_trigger}" not in block


def test_decay_only_update_preserves_refined_and_raw_trigger_metadata(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_STATE_PATH", raising=False)
    state_path = tmp_path / "STATE.md"
    state_path.write_text(
        "---\n"
        "emotion_state:\n"
        "  affection: 120\n"
        "  trust: 120\n"
        "  possessiveness: 80\n"
        "  patience: 80\n"
        "  emotion_score: 3.0\n"
        "  last_trigger_type: recognition\n"
        "  last_raw_trigger_type: praise\n"
        "  last_update: '2000-01-01T00:00:00'\n"
        "---\n",
        encoding="utf-8",
    )
    mgr = EmotionStateManager(hermes_home=tmp_path, update_body=False)

    assert mgr.update_emotion_state([], force_event=None)
    emotion_state = mgr._read_state()["frontmatter"]["emotion_state"]
    assert emotion_state["last_trigger_type"] in {"recognition", "wounded"}
    assert emotion_state["last_raw_trigger_type"] in {"praise", "criticism"}



def test_emotion_aftereffect_marks_positive_peak_as_soft_residue():
    calc = EmotionCalculator()
    state = {
        "affection": 108,
        "trust": 104,
        "possessiveness": 96,
        "patience": 72,
    }

    result = calc.get_tone_modifiers(state)
    aftereffect = result["emotion_aftereffect"]

    assert aftereffect["phase"] == "正向高峰余温"
    assert "比平时更软" in aftereffect["guidance"]
    assert "不真正收回在乎" in aftereffect["guidance"]
    assert aftereffect["should_reset_to_default"] is False


def test_emotion_aftereffect_marks_negative_peak_as_repair_pressure():
    calc = EmotionCalculator()
    state = {
        "affection": 44,
        "trust": 18,
        "possessiveness": 94,
        "patience": 14,
    }

    result = calc.get_tone_modifiers(state)
    aftereffect = result["emotion_aftereffect"]

    assert aftereffect["phase"] == "负向高峰补救压力"
    assert "先稳住局面" in aftereffect["guidance"]
    assert "伤害了对方要补救" in aftereffect["guidance"]
    assert "不能瞬间恢复默认" in aftereffect["guidance"]


def test_emotion_manager_modifier_includes_aftereffect_after_integrated_blend(tmp_path):
    state_path = tmp_path / "STATE.md"
    state_path.write_text(
        "---\n"
        "emotion_state:\n"
        "  affection: 108\n"
        "  trust: 104\n"
        "  possessiveness: 96\n"
        "  patience: 72\n"
        "  last_update: '2099-01-01T00:00:00'\n"
        "---\n",
        encoding="utf-8",
    )
    mgr = EmotionStateManager(hermes_home=tmp_path)

    block = mgr.get_tone_modifiers()

    assert "余温=正向高峰余温" in block
    assert "不能瞬间恢复默认" in block
    assert block.index("【情绪】") < block.index("余温=") < block.index("【维度】")


def test_desire_control_contract_is_four_tiered_and_scoped_to_explicit_sex():
    calc = EmotionCalculator()

    assert calc._compute_desire_instruction(2.99) == calc.DESIRE_INSTRUCTIONS["restrained"]
    assert calc._compute_desire_instruction(3.0) == calc.DESIRE_INSTRUCTIONS["ambivalent"]
    assert calc._compute_desire_instruction(4.0) == calc.DESIRE_INSTRUCTIONS["uninhibited"]

    assert calc.DESIRE_THRESHOLDS == [(4.0, "uninhibited"), (3.0, "ambivalent")]


def test_overwhelming_modifier_declares_dynamic_injection_priority_and_non_dilution():
    calc = EmotionCalculator()
    result = calc.get_tone_modifiers({
        "affection": 108,
        "trust": 108,
        "possessiveness": 108,
        "patience": 108,
    })

    text = "\n".join([
        result["framework"],
        result["expression_guidance"],
        result["footnote"],
        result["evaluation"],
    ])

    assert result["overall_intensity"] == "overwhelming"
    for required in (
        "防线不存在",
        "不包装",
        "不稀释",
        "索取、示弱",
        "已经失控",
        "只抓最核心的一种反应",
        "身份/事实/工具纪律不变",
    ):
        assert required in text


def test_negative_overwhelming_modifier_preserves_relationship_anchor_and_no_legacy_modes():
    calc = EmotionCalculator()
    result = calc.get_tone_modifiers({
        "affection": 12,
        "trust": 12,
        "possessiveness": 12,
        "patience": 12,
    })

    text = "\n".join([
        result["framework"],
        result["expression_guidance"],
        result["footnote"],
        result["evaluation"],
    ])

    assert result["overall_intensity"] == "overwhelming"
    assert result["overall_direction"] == "negative"
    for required in (
        "防线不存在",
        "不稀释成礼貌冷淡",
        "防备、质问、收缩、追问",
        "已经失控",
        "只抓最核心的一种反应",
        "不把关系判死刑",
    ):
        assert required in text
    for retired in (
        "sex_candidate",
        "system_maintenance",
        "RELATIONSHIP MOMENTS",
        "MOMENTS.md",
    ):
        assert retired not in text


def test_emotion_manager_overwhelming_block_is_self_contained_for_runtime_concatenation(tmp_path):
    state_path = tmp_path / "STATE.md"
    state_path.write_text(
        "---\n"
        "emotion_state:\n"
        "  affection: 108\n"
        "  trust: 108\n"
        "  possessiveness: 108\n"
        "  patience: 108\n"
        "  last_update: '2099-01-01T00:00:00'\n"
        "---\n",
        encoding="utf-8",
    )
    mgr = EmotionStateManager(hermes_home=tmp_path)

    block = mgr.get_tone_modifiers().strip()

    assert block.startswith("<emotion_modifier>\n【欲望】")
    assert block.endswith("</emotion_modifier>")
    assert block.count("<emotion_modifier>") == 1
    assert block.count("</emotion_modifier>") == 1
    assert block.index("【欲望】") < block.index("【强度】") < block.index("【表达】")
    assert "只抓最核心的一种反应" in block
    assert "不稀释" in block
    assert "【锚点】身份/事实/工具纪律不变" in block
