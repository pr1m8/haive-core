#!/usr/bin/env python3
"""
Debug test to trace the exact source of tool duplication in AugLLMConfig.
This is a critical bug that affects all tool integration tests.
"""

import pytest
from langchain_core.tools import tool

from haive.core.engine.aug_llm import AugLLMConfig


# Test tools
@tool
def calculator(expression: str) -> str:
    """Calculate mathematical expressions."""
    return f"Result: {eval(expression)}"

@tool
def simple_tool(text: str) -> str:
    """Simple tool for testing."""
    return f"Processed: {text}"

class TestAugLLMConfigDuplicationBug:
    """Test suite to debug and document the tool duplication bug in AugLLMConfig."""

    def test_single_tool_duplication(self):
        """Test that single tool gets duplicated - THIS IS A BUG."""
        # Create AugLLMConfig with 1 tool (no LLM config to avoid validation issues)
        config = AugLLMConfig(tools=[calculator])

        # BUG: We expect 1 tool but get 2 due to duplication
        print("Tools provided: 1")
        print(f"Tools in config: {len(config.tools)}")
        print(f"Tool instances: {[f'{tool.name}(id:{id(tool)})' for tool in config.tools]}")

        # Document the bug
        assert len(config.tools) == 2, "BUG: Single tool gets duplicated"

        # Verify it's the same tool instance duplicated
        tool_ids = [id(tool) for tool in config.tools]
        unique_ids = set(tool_ids)
        assert len(unique_ids) == 1, "BUG: Same tool instance appears multiple times"

        # Tool routes should be correct (only 1 route)
        assert len(config.tool_routes) == 1
        assert "calculator" in config.tool_routes

    def test_multiple_tool_duplication(self):
        """Test that multiple tools get duplicated - THIS IS A BUG."""
        config = AugLLMConfig(tools=[calculator, simple_tool])

        print("Tools provided: 2")
        print(f"Tools in config: {len(config.tools)}")
        print(f"Tool instances: {[f'{tool.name}(id:{id(tool)})' for tool in config.tools]}")

        # BUG: We expect 2 tools but get 4 due to duplication
        assert len(config.tools) == 4, "BUG: Multiple tools get duplicated"

        # Verify each tool appears exactly twice
        tool_names = [tool.name for tool in config.tools]
        assert tool_names.count("calculator") == 2, "BUG: Calculator appears twice"
        assert tool_names.count("simple_tool") == 2, "BUG: Simple_tool appears twice"

        # Tool routes should be correct (2 routes)
        assert len(config.tool_routes) == 2
        assert "calculator" in config.tool_routes
        assert "simple_tool" in config.tool_routes

    def test_dynamic_tool_addition_exponential_duplication(self):
        """Test that dynamic tool addition causes exponential duplication - THIS IS A SEVERE BUG."""
        # Start with empty config
        config = AugLLMConfig()
        assert len(config.tools) == 0

        # Add first tool
        config.add_tool(calculator)
        print(f"After adding 1 tool: {len(config.tools)} tools")
        print(f"Tool instances: {[f'{tool.name}(id:{id(tool)})' for tool in config.tools]}")

        # BUG: Adding 1 tool results in 2 tools
        assert len(config.tools) == 2, "BUG: Adding 1 tool results in 2 tools"

        # Add second tool
        config.add_tool(simple_tool)
        print(f"After adding 2nd tool: {len(config.tools)} tools")
        print(f"Tool instances: {[f'{tool.name}(id:{id(tool)})' for tool in config.tools]}")

        # SEVERE BUG: Adding a second tool to a set with 2 tools results in 6 tools!
        # This suggests the duplication is exponential/multiplicative
        assert len(config.tools) == 6, "SEVERE BUG: Exponential tool duplication during dynamic addition"

        # Count occurrences
        tool_names = [tool.name for tool in config.tools]
        calc_count = tool_names.count("calculator")
        simple_count = tool_names.count("simple_tool")

        print(f"Calculator count: {calc_count}")
        print(f"Simple_tool count: {simple_count}")

        # This reveals the pattern of duplication
        assert calc_count >= 2, "Calculator should appear at least twice"
        assert simple_count >= 1, "Simple_tool should appear at least once"

    def test_empty_config_no_duplication(self):
        """Test that empty config has no duplication issues."""
        config = AugLLMConfig()

        assert len(config.tools) == 0
        assert len(config.tool_routes) == 0

    def test_tool_route_mixin_works_correctly(self):
        """Test that ToolRouteMixin itself doesn't have duplication issues."""
        from haive.core.common.mixins.tool_route_mixin import ToolRouteMixin

        class SimpleContainer(ToolRouteMixin):
            def __init__(self, **data):
                super().__init__(**data)

        # Test ToolRouteMixin directly
        container = SimpleContainer(tools=[calculator, simple_tool])

        print(f"Direct ToolRouteMixin - Tools: {len(container.tools)}")
        print(f"Tool instances: {[f'{tool.name}(id:{id(tool)})' for tool in container.tools]}")

        # ToolRouteMixin should work correctly (no duplication)
        assert len(container.tools) == 2, "ToolRouteMixin should not duplicate tools"

        # Should have unique instances
        tool_ids = [id(tool) for tool in container.tools]
        unique_ids = set(tool_ids)
        assert len(unique_ids) == 2, "ToolRouteMixin should have unique tool instances"

        # This proves the bug is in AugLLMConfig, not ToolRouteMixin

    @pytest.mark.skip(reason="Bug reproduction - not a real test")
    def test_trace_duplication_source(self):
        """Attempt to trace where the duplication happens."""
        # This test is for investigation purposes
        # The duplication likely happens in:
        # 1. AugLLMConfig.model_post_init()
        # 2. Some interaction between ToolRouteMixin and other mixins
        # 3. Pydantic field validation/processing

        # Evidence points to the issue being in AugLLMConfig's initialization chain
        # since ToolRouteMixin works fine in isolation


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
