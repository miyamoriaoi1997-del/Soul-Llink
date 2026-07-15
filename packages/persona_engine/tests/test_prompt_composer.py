from pathlib import Path

from persona_orchestrator.prompt_composer import PromptComposer


def test_core_auto_prepended():
    composition = PromptComposer('.').compose(['daily'])

    assert composition.selected_layers[:2] == ['core', 'daily']
    assert '# Core Identity Layer' in composition.prompt_text
    assert '# Daily Mode Layer' in composition.prompt_text


def test_host_core_does_not_auto_prepend_or_load_orchestrator_core():
    composition = PromptComposer('.', core_source='host_core').compose(['daily'])

    assert composition.selected_layers == ['daily']
    assert '# Core Identity Layer' not in composition.prompt_text
    assert '# Daily Mode Layer' in composition.prompt_text


def test_emotion_modifier_appears_at_end():
    emotion = '<emotion_modifier>test</emotion_modifier>'
    composition = PromptComposer('.').compose(['core', 'work'], emotion_modifier=emotion)

    assert composition.prompt_text.rstrip().endswith(emotion)


def test_compose_keeps_exactly_one_emotion_modifier_block():
    emotion = '<emotion_modifier>fresh</emotion_modifier>'
    composition = PromptComposer('.').compose(['core', 'work'], emotion_modifier=emotion)

    assert composition.prompt_text.count('<emotion_modifier>') == 1
    assert composition.prompt_text.count('</emotion_modifier>') == 1


def test_unknown_layer_warning():
    composition = PromptComposer('.').compose(['core', 'missing'])

    assert 'missing' not in composition.selected_layers
    assert any('missing' in warning for warning in composition.warnings)


def test_hash_stable_across_repeated_calls():
    composer = PromptComposer('.')
    first = composer.compose(['core', 'daily'], emotion_modifier='x')
    second = composer.compose(['core', 'daily'], emotion_modifier='x')

    assert first.prompt_hash == second.prompt_hash
    assert len(first.prompt_hash) == 16


def test_sex_layer_only_loaded_when_explicitly_selected():
    daily = PromptComposer('.').compose(['core', 'daily'])
    sex = PromptComposer('.').compose(['core', 'sex'])

    assert 'Adult Boundary Layer' not in daily.prompt_text
    assert 'Adult Boundary Layer' in sex.prompt_text


def test_active_prompt_removes_old_managed_region_and_old_emotion_block():
    host = """
# Host Prompt
keep this

<persona_orchestrator_prompt>
old managed
</persona_orchestrator_prompt>

middle survives

<emotion_modifier>old</emotion_modifier>
"""
    fresh_emotion = '<emotion_modifier>fresh</emotion_modifier>'

    result = PromptComposer('.').compose_active(
        host_system_prompt=host,
        selected_layers=['core', 'work', 'overlay_intimacy'],
        emotion_modifier=fresh_emotion,
    )

    assert 'keep this' in result.prompt_text
    assert 'middle survives' in result.prompt_text
    assert 'old managed' not in result.prompt_text
    assert '<emotion_modifier>old</emotion_modifier>' not in result.prompt_text
    assert result.prompt_text.count('<persona_orchestrator_prompt>') == 1
    assert result.prompt_text.count('</persona_orchestrator_prompt>') == 1
    assert result.prompt_text.count('<emotion_modifier>') == 1
    assert fresh_emotion in result.prompt_text
    assert result.prompt_text.rstrip().endswith(fresh_emotion)


def test_active_prompt_has_core_first_inside_managed_region():
    result = PromptComposer('.').compose_active(
        host_system_prompt='# Host Prompt',
        selected_layers=['daily', 'work', 'core'],
        emotion_modifier='<emotion_modifier>fresh</emotion_modifier>',
    )

    managed = result.prompt_text.split('<persona_orchestrator_prompt>')[1].split('</persona_orchestrator_prompt>')[0]
    assert managed.index('# Core Identity Layer') < managed.index('# Daily Mode Layer')
    assert managed.index('# Core Identity Layer') < managed.index('# Work Mode Layer')


def test_host_core_active_prompt_preserves_host_identity_and_omits_orchestrator_core():
    result = PromptComposer('.', core_source='host_core').compose_active(
        host_system_prompt='# Host Core Identity\n\nKeep this identity.',
        selected_layers=['work', 'daily'],
        emotion_modifier='<emotion_modifier>fresh</emotion_modifier>',
    )

    managed = result.prompt_text.split('<persona_orchestrator_prompt>')[1].split('</persona_orchestrator_prompt>')[0]
    assert '# Host Core Identity' in result.prompt_text
    assert '# Core Identity Layer' not in managed
    assert '# Work Mode Layer' in managed
    assert '# Daily Mode Layer' in managed
    assert result.selected_layers == ['work', 'daily']


def test_compose_with_memory_view_does_not_embed_active_pcltm_surface():
    view_text = '[system]\nidentity anchor\n\n[pinned]\npreference fact'

    result = PromptComposer('.').compose_with_memory_view(
        ['core', 'work'],
        memory_view_text=view_text,
    )

    assert '<pcltm_memory_view>' not in result.prompt_text
    assert 'identity anchor' not in result.prompt_text
    assert '<memory_profile_notes>' not in result.prompt_text


def test_compose_with_memory_view_empty_falls_back():
    result = PromptComposer('.').compose_with_memory_view(
        ['core', 'work'],
        memory_view_text='  ',
        memory_notes='profile=technical_plus_core_relationship',
    )

    assert '<memory_profile_notes>\nprofile=technical_plus_core_relationship\n</memory_profile_notes>' in result.prompt_text
    assert '<pcltm_memory_view>' not in result.prompt_text


def test_compose_active_with_memory_view_omits_duplicate_pcltm_surface():
    result = PromptComposer('.').compose_active(
        host_system_prompt='# Host Prompt\n\n<pcltm_context>\n[system]\nhost active memory\n</pcltm_context>',
        selected_layers=['work'],
        memory_view_text='[transient]\ncurrent task evidence',
    )

    managed = result.prompt_text.split('<persona_orchestrator_prompt>')[1].split('</persona_orchestrator_prompt>')[0]
    assert '<pcltm_memory_view>' not in managed
    assert 'current task evidence' not in managed
    assert result.prompt_text.count('<pcltm_context>') == 1
    assert result.prompt_text.count('</pcltm_context>') == 1
    assert '<memory_profile_notes>' not in managed


def test_compose_active_memory_view_does_not_restore_legacy_notes():
    result = PromptComposer('.').compose_active(
        host_system_prompt='# Host Prompt',
        selected_layers=['work'],
        memory_notes='legacy notes should not appear',
        memory_view_text='[pinned]\nstructured fact',
    )

    assert 'legacy notes should not appear' not in result.prompt_text
    assert '<memory_profile_notes>' not in result.prompt_text
    assert '<pcltm_memory_view>' not in result.prompt_text
    assert result.prompt_text.count('<pcltm_context>') == 1
    assert '[pinned]\nstructured fact' in result.prompt_text


def test_compose_active_drops_legacy_memory_notes_from_host_when_using_memory_view():
    host = """
# Host Prompt

<memory_profile_notes>
legacy notes
</memory_profile_notes>
"""
    result = PromptComposer('.').compose_active(
        host_system_prompt=host,
        selected_layers=['work'],
        memory_view_text='[system]\nstructured memory',
    )

    assert '<memory_profile_notes>' not in result.prompt_text
    assert 'legacy notes' not in result.prompt_text
    assert '<pcltm_memory_view>' not in result.prompt_text
    assert result.prompt_text.count('<pcltm_context>') == 1
    assert 'structured memory' in result.prompt_text


def test_compose_backward_compat_memory_notes_still_works():
    result = PromptComposer('.').compose(
        ['core', 'work'],
        memory_notes='profile=technical_plus_core_relationship',
    )

    assert '<memory_profile_notes>\nprofile=technical_plus_core_relationship\n</memory_profile_notes>' in result.prompt_text
    assert '<pcltm_memory_view>' not in result.prompt_text

def test_compose_active_strips_legacy_hermes_memory_blocks_from_host():
    result = PromptComposer('.').compose_active(
        host_system_prompt=(
            '# Host Prompt\n'
            'keep host invariant\n\n'
            'USER PROFILE (who the user is)\n'
            '- stale legacy user memory\n\n'
            'MEMORY (your personal notes)\n'
            '- stale legacy assistant memory\n\n'
            '<pcltm_context>\n[system] authoritative active memory\n</pcltm_context>'
        ),
        selected_layers=['work'],
    )

    assert 'keep host invariant' in result.prompt_text
    assert 'USER PROFILE (who the user is)' not in result.prompt_text
    assert 'MEMORY (your personal notes)' not in result.prompt_text
    assert 'stale legacy user memory' not in result.prompt_text
    assert 'stale legacy assistant memory' not in result.prompt_text
    assert result.prompt_text.count('<pcltm_context>') == 1
    assert '[system] authoritative active memory' in result.prompt_text


def test_compose_active_drops_legacy_hermes_memory_notes_argument():
    result = PromptComposer('.').compose_active(
        host_system_prompt='# Host Prompt',
        selected_layers=['work'],
        memory_notes='USER PROFILE (who the user is)\n- stale legacy user memory',
    )

    assert 'USER PROFILE (who the user is)' not in result.prompt_text
    assert 'stale legacy user memory' not in result.prompt_text

def test_compose_active_strips_legacy_blocks_without_swallowing_next_host_section():
    result = PromptComposer('.').compose_active(
        host_system_prompt=(
            '# Host Prompt\n'
            'USER PROFILE (who the user is)\n'
            '- stale legacy user memory\n\n'
            'RUNTIME INVARIANTS\n'
            'keep this non-memory host rule\n\n'
            'MEMORY (your personal notes)\n'
            '- stale legacy assistant memory\n\n'
            '<pcltm_context>\n[system] authoritative active memory\n</pcltm_context>'
        ),
        selected_layers=['work'],
    )

    assert 'USER PROFILE (who the user is)' not in result.prompt_text
    assert 'MEMORY (your personal notes)' not in result.prompt_text
    assert 'stale legacy user memory' not in result.prompt_text
    assert 'stale legacy assistant memory' not in result.prompt_text
    assert 'RUNTIME INVARIANTS' in result.prompt_text
    assert 'keep this non-memory host rule' in result.prompt_text
    assert '[system] authoritative active memory' in result.prompt_text


def test_active_soul_templates_use_public_configurable_identity():
    result = PromptComposer(Path(__file__).resolve().parents[1]).compose_active(
        host_system_prompt='# Host Prompt',
        selected_layers=['daily', 'work', 'sex'],
    )

    assert 'You are a configurable persona runtime instance' in result.prompt_text
    assert 'This layer must not redefine the configured persona identity' in result.prompt_text
    assert 'This public edition does not ship explicit adult persona instructions' in result.prompt_text
    assert result.prompt_text.count('# Core Identity Layer') == 1
