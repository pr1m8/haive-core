#!/usr/bin/env python3
"""
Test to verify the AugLLMConfig tool duplication bug is FIXED.
"""

import pytest
from langchain_core.tools import tool

from haive.core.engine.aug_llm import AugLLMConfig


@tool
def calculator(expression: str) -> str:
    """Calculate mathematical expressions."""
    return f"Result: {eval(expression)}"


@tool
def simple_tool(text: str) -> str:
    """Simple tool for testing."""
    return f"Processed: {text}"


class TestAugLLMConfigFixed:
    """Test suite to verify the tool duplication bug is FIXED."""

    def test_single_tool_no_duplication(self):
        """Test that single tool is NOT duplicated - BUG IS FIXED."""
        config = AugLLMConfig(tools=[calculator])

        # FIXED: We now correctly get 1 tool (not 2)
        assert len(config.tools) == 1, f"Expected 1 tool, got {len(config.tools)} tools"

        # Verify it's the correct tool
        assert config.tools[0].name == "calculator"

        # Tool routes should be correct
        assert len(config.tool_routes) == 1
        assert "calculator" in config.tool_routes

    def test_multiple_tools_no_duplication(self):
        """Test that multiple tools are NOT duplicated - BUG IS FIXED."""
        config = AugLLMConfig(tools=[calculator, simple_tool])

        # FIXED: We now correctly get 2 tools (not 4)
        assert len(config.tools) == 2, f"Expected 2 tools, got {len(config.tools)} tools"

        # Verify unique tools
        tool_names = [tool.name for tool in config.tools]
        assert "calculator" in tool_names
        assert "simple_tool" in tool_names
        assert tool_names.count("calculator") == 1, "Calculator should appear exactly once"
        assert tool_names.count("simple_tool") == 1, "Simple_tool should appear exactly once"

        # Tool routes should be correct
        assert len(config.tool_routes) == 2
        assert "calculator" in config.tool_routes
        assert "simple_tool" in config.tool_routes

    def test_dynamic_tool_addition_no_duplication(self):
        """Test that dynamic tool addition does NOT cause duplication - BUG IS FIXED."""
        config = AugLLMConfig()
        assert len(config.tools) == 0

        # Add first tool
        config.add_tool(calculator)
        # FIXED: Adding 1 tool results in 1 tool (not 2)
        assert len(config.tools) == 1, f"Expected 1 tool after adding calculator, got {len(config.tools)}"

        # Add second tool
        config.add_tool(simple_tool)
        # FIXED: Adding second tool results in 2 total tools (not 6)
        assert len(config.tools) == 2, f"Expected 2 tools after adding both, got {len(config.tools)}"

        # Verify no duplicates
        tool_names = [tool.name for tool in config.tools]
        assert tool_names.count("calculator") == 1
        assert tool_names.count("simple_tool") == 1

    def test_duplicate_tool_prevention(self):
        """Test that adding the same tool twice is properly prevented."""
        config = AugLLMConfig()

        # Add tool twice
        config.add_tool(calculator)
        config.add_tool(calculator)  # Same tool again

        # Should still only have 1 tool (duplicate prevention works)
        assert len(config.tools) == 1, f"Duplicate prevention failed: {len(config.tools)} tools"
        assert config.tools[0].name == "calculator"

        # Tool routes should still be correct
        assert len(config.tool_routes) == 1
        assert "calculator" in config.tool_routes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
