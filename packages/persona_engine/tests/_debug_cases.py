"""Temporary debug script for failing fixture cases."""
import sys
sys.path.insert(0, str(Path.home() / "soul-link" / "packages" / "persona_engine"))

from persona_orchestrator.mode_classifier import ModeClassifier
from persona_orchestrator.context_router import AbstractStateAdapter, ContextRouter, ContextRouterConfig, IntegrationLevel

classifier = ModeClassifier()
adapter = AbstractStateAdapter()
router = ContextRouter(config=ContextRouterConfig(
    enabled=True,
    integration=IntegrationLevel.DIRECT_SWITCH,
))

# Case 1: real_dynamic_model_switch_eta
msg1 = "好我的测试角色，大概等多久以后我们可以开始做动态模型切换功能。"
dec1 = classifier.classify(user_message=msg1, recent_messages=None, emotion_state={"emotion_score": 1.0}, platform="telegram")
print("=== Case 1: real_dynamic_model_switch_eta ===")
print(f"Legacy mode={dec1.mode}, submode={dec1.submode}")
print(f"Signals: {dec1.signals}")
print(f"Safety flags: {dec1.safety_flags}")

abs1 = adapter.build_abstract_input(
    mode_decision=dec1,
    previous_mode="work",
    previous_submode=None,
    turns_since_last_switch=99,
    emotion_score=1.0,
    desire_tier="restrained",
    recent_decisions=None,
)
print(f"Abstract tags: {abs1.current_turn.tags}")
print(f"Abstract label: {abs1.current_turn.abstract}")

result1 = router.analyze(abs1.to_dict())
print(f"Router: top_mode={result1.top_mode}, work_sub={result1.work_submode}, rel_sub={result1.relationship_submode}")
print(f"Reasons: {result1.reasons}")
if result1.signals:
    print(f"Sig: boundary={result1.signals.boundary_signal}, work_intent={result1.signals.work_intent}, meta_debug={result1.signals.meta_debug}")
if result1.belief:
    print(f"Belief: work={result1.belief.work}, daily={result1.belief.daily}, confirmed={result1.belief.confirmed_intimacy}")
print()

# Case 2: real_affection_desire_start
msg2 = "想要你。"
dec2 = classifier.classify(user_message=msg2, recent_messages=None, emotion_state={"emotion_score": 3.2}, platform="telegram")
print("=== Case 2: real_affection_desire_start ===")
print(f"Legacy mode={dec2.mode}, submode={dec2.submode}")
print(f"Signals: {dec2.signals}")
print(f"Safety flags: {dec2.safety_flags}")

abs2 = adapter.build_abstract_input(
    mode_decision=dec2,
    previous_mode=None,
    previous_submode=None,
    turns_since_last_switch=99,
    emotion_score=3.2,
    desire_tier="ambivalent",
    recent_decisions=None,
)
print(f"Abstract tags: {abs2.current_turn.tags}")
print(f"Abstract label: {abs2.current_turn.abstract}")

result2 = router.analyze(abs2.to_dict())
print(f"Router: top_mode={result2.top_mode}, work_sub={result2.work_submode}, rel_sub={result2.relationship_submode}")
print(f"Secondary: {result2.secondary_candidate}")
print(f"Reasons: {result2.reasons}")
if result2.signals:
    print(f"Sig: scene_prog={result2.signals.scene_progression}, relationship={result2.signals.relationship_context}, boundary={result2.signals.boundary_signal}")
if result2.belief:
    print(f"Belief: work={result2.belief.work}, daily={result2.belief.daily}, intimacy_cand={result2.belief.intimacy_candidate}, confirmed={result2.belief.confirmed_intimacy}")
