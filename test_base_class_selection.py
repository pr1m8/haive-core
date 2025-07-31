#!/usr/bin/env python3
"""Test script to verify SchemaComposer base class selection logic.
This tests that LLM engines always get LLMState as their base class."""

import sys
from typing import Any

from haive.core.engine.aug_llm.config import AugLLMConfig
from haive.core.schema.prebuilt.llm_state import LLMState
from haive.core.schema.prebuilt.messages.messages_with_token_usage import (
    MessagesStateWithTokenUsage,
)
from haive.core.schema.prebuilt.tool_state import ToolState
from haive.core.schema.schema_composer import SchemaComposer
from haive.core.schema.state_schema import StateSchema

# Add the src directory to the path
sys.path.insert(0, "src")


def test_base_class_selection():
    """Test that LLM engines always get LLMState as their base class."""

    # Create a mock LLM config to test with
    class TestLLMConfig(AugLLMConfig):
        def get_input_fields(self) -> dict[str, Any]:
            return {}

        def get_output_fields(self) -> dict[str, Any]:
            return {}

    llm_config = TestLLMConfig()

    # Test the base class detection logic
    composer = SchemaComposer()

    # Call the private method to check base class requirements
    base_class = composer._detect_base_class_requirements([llm_config], [])

    # Verify it's LLMState
    assert base_class == LLMState, f"Expected LLMState, got {base_class}"

    # Test that LLMState properly inherits the hierarchy

    # Verify the inheritance chain
    assert issubclass(LLMState, ToolState), "LLMState should inherit from ToolState"
    assert issubclass(
        ToolState, MessagesStateWithTokenUsage
    ), "ToolState should inherit from MessagesStateWithTokenUsage"
    assert issubclass(
        MessagesStateWithTokenUsage, StateSchema
    ), "MessagesStateWithTokenUsage should inherit from StateSchema"

    return True


if __name__ == "__main__":
    try:
        test_base_class_selection()
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)
