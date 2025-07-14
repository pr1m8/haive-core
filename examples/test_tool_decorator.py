#!/usr/bin/env python3
"""Test if the issue is with Tool constructor vs @tool decorator."""

import traceback

from langchain_core.tools import Tool, tool

from haive.core.engine.aug_llm import AugLLMConfig


@tool
def simple_add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def create_tool_with_constructor():
    """Create a tool using the Tool constructor."""

    def add_func(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    return Tool(
        name="add_constructor",
        description="Add two numbers using Tool constructor",
        func=add_func,
    )


def test_tool_decorator():
    """Test @tool decorator with AugLLMConfig."""
    print("=== Testing @tool decorator ===")

    try:
        config = AugLLMConfig(tools=[simple_add])
        print("✅ SUCCESS: @tool decorator works!")

    except Exception as e:
        print(f"❌ FAILED: {e}")
        traceback.print_exc()


def test_tool_constructor():
    """Test Tool constructor with AugLLMConfig."""
    print("\n=== Testing Tool constructor ===")

    tool = create_tool_with_constructor()

    try:
        config = AugLLMConfig(tools=[tool])
        print("✅ SUCCESS: Tool constructor works!")

    except Exception as e:
        print(f"❌ FAILED: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    test_tool_decorator()
    test_tool_constructor()
