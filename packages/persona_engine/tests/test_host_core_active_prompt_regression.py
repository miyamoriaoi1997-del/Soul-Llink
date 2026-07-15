from persona_orchestrator.prompt_composer import PromptComposer


def test_active_prompt_with_host_core_heading_omits_managed_core_even_for_default_composer():
    result = PromptComposer('.').compose_active(
        host_system_prompt='# Core Identity Layer\n\nhost identity\n\n<emotion_modifier>old</emotion_modifier>',
        selected_layers=['core', 'work'],
        emotion_modifier='<emotion_modifier>fresh</emotion_modifier>',
    )

    managed = result.prompt_text.split('<persona_orchestrator_prompt>')[1].split('</persona_orchestrator_prompt>')[0]
    assert result.prompt_text.count('# Core Identity Layer') == 1
    assert '# Core Identity Layer' not in managed
    assert '# Work Mode Layer' in managed
    assert result.selected_layers == ['work']
    assert result.prompt_text.count('<emotion_modifier>') == 1
    assert result.prompt_text.rstrip().endswith('<emotion_modifier>fresh</emotion_modifier>')
