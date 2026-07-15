from pcltm.memory_object import MemoryObjectType
from pcltm.memory_object_adapter import adapt_memory_object


def test_adapt_memory_object_honors_explicit_object_type_for_user_records():
    memory = adapt_memory_object(
        {
            "canonical_key": "pref.pending",
            "target_file": "USER.md",
            "content": "Pending user preference.",
            "status": "pending",
            "object_type": "preference",
            "injection_policy": "selective",
        }
    )

    assert memory.object_type is MemoryObjectType.PREFERENCE
