from pathlib import Path


def test_host_adapter_forwards_unique_turn_identity_to_memory_provider() -> None:
    patch = (
        Path(__file__).resolve().parents[1]
        / "adapters/hermes/patches/pcltm-context-engine-host-adapter.patch"
    ).read_text(encoding="utf-8")

    assert "session_id=agent.session_id" in patch
    assert "turn_id=turn_id" in patch