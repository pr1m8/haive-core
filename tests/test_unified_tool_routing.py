"""Test the unified tool routing system without mocks.

This tests that AugLLMConfig properly inherits and uses ToolRouteMixin's
tool storage and routing capabilities.
"""

from typing import Any, Dict, List

import pytest
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from haive.core.engine.aug_llm.config import AugLLMConfig
from haive.core.models.llm.base import AzureLLMConfig


# Test tools - no mocks, real implementations
class TestOutputModel(BaseModel):
    """Test Pydantic model for structured output."""

    result: str = Field(description="The result")
    confidence: float = Field(description="Confidence score")


class ExecutableModel(BaseModel):
    """Test Pydantic model that's executable as a tool."""

    query: str = Field(description="The query to process")

    def __call__(self) -> str:
        """Make this model executable."""
        return f"Processed: {self.query}"


@tool
def test_function_tool(input_text: str) -> str:
    """Test function tool."""
    return f"Function processed: {input_text}"


class TestBaseTool(BaseTool):
    """Test BaseTool implementation."""

    name: str = "test_base_tool"
    description: str = "A test BaseTool"

    def _run(self, query: str) -> str:
        return f"BaseTool processed: {query}"


def test_aug_llm_config_inherits_tool_storage():
    """Test that AugLLMConfig properly inherits tool storage from ToolRouteMixin."""
    # Create AugLLMConfig with tools
    tools = [TestOutputModel, ExecutableModel, test_function_tool, TestBaseTool()]

    config = AugLLMConfig(
        name="test_config",
        llm_config=AzureLLMConfig(model="gpt-4"),
        tools=tools,
        structured_output_model=TestOutputModel,
        structured_output_version="v2",
    )

    # Test that tools are stored in the mixin
    assert len(config.tools) == 4
    assert len(config.tool_instances) == 4
    assert len(config.tool_routes) == 4

    print("✅ Tool storage inheritance works")


def test_tool_routing_analysis():
    """Test that tools are analyzed and routed correctly."""
    tools = [TestOutputModel, ExecutableModel, test_function_tool, TestBaseTool()]

    config = AugLLMConfig(
        name="test_config",
        llm_config=AzureLLMConfig(model="gpt-4"),
        tools=tools,
        structured_output_model=TestOutputModel,
        structured_output_version="v2",
    )

    # Check specific tool routes
    routes = config.tool_routes
    print(f"Tool routes: {routes}")

    # TestOutputModel should be structured_output_tool (v2)
    test_output_route = routes.get("TestOutputModel")
    assert (
        test_output_route == "structured_output_tool"
    ), f"Expected structured_output_tool, got {test_output_route}"

    # ExecutableModel should be pydantic_tool (has __call__)
    executable_route = routes.get("ExecutableModel")
    assert (
        executable_route == "pydantic_tool"
    ), f"Expected pydantic_tool, got {executable_route}"

    # test_function_tool should be function
    function_route = routes.get("test_function_tool")
    assert function_route == "function", f"Expected function, got {function_route}"

    # TestBaseTool should be langchain_tool
    base_tool_route = routes.get("test_base_tool")
    assert (
        base_tool_route == "langchain_tool"
    ), f"Expected langchain_tool, got {base_tool_route}"

    print("✅ Tool routing analysis works correctly")


def test_tool_metadata_enhancement():
    """Test that enhanced metadata is generated for tools."""
    config = AugLLMConfig(
        name="test_config",
        llm_config=AzureLLMConfig(model="gpt-4"),
        tools=[test_function_tool],
    )

    # Check function tool metadata
    function_metadata = config.tool_metadata.get("test_function_tool")
    assert function_metadata is not None

    # Should have callable analysis
    assert "is_async" in function_metadata
    assert "parameter_count" in function_metadata
    assert "callable_kind" in function_metadata

    print(f"Function metadata: {function_metadata}")
    print("✅ Enhanced metadata generation works")


def test_get_tools_by_route():
    """Test filtering tools by route."""
    tools = [TestOutputModel, ExecutableModel, test_function_tool, TestBaseTool()]

    config = AugLLMConfig(
        name="test_config",
        llm_config=AzureLLMConfig(model="gpt-4"),
        tools=tools,
        structured_output_model=TestOutputModel,
        structured_output_version="v2",
    )

    # Get tools by different routes
    structured_tools = config.get_tools_by_route("structured_output_tool")
    pydantic_tools = config.get_tools_by_route("pydantic_tool")
    function_tools = config.get_tools_by_route("function")
    langchain_tools = config.get_tools_by_route("langchain_tool")

    assert len(structured_tools) == 1
    assert structured_tools[0] == TestOutputModel

    assert len(pydantic_tools) == 1
    assert pydantic_tools[0] == ExecutableModel

    assert len(function_tools) == 1
    assert function_tools[0] == test_function_tool

    assert len(langchain_tools) == 1
    assert isinstance(langchain_tools[0], TestBaseTool)

    print("✅ Tool filtering by route works")


def test_dynamic_tool_management():
    """Test adding and removing tools dynamically."""
    config = AugLLMConfig(
        name="test_config", llm_config=AzureLLMConfig(model="gpt-4"), tools=[]
    )

    # Start with no tools
    assert len(config.tools) == 0

    # Add a tool dynamically
    config.add_tool(test_function_tool)
    assert len(config.tools) == 1
    assert "test_function_tool" in config.tool_routes

    # Get the tool back
    retrieved_tool = config.get_tool("test_function_tool")
    assert retrieved_tool == test_function_tool

    # Clear all tools
    config.clear_tools()
    assert len(config.tools) == 0
    assert len(config.tool_routes) == 0

    print("✅ Dynamic tool management works")


def test_structured_output_detection():
    """Test that structured output models are detected correctly."""
    config = AugLLMConfig(
        name="test_config",
        llm_config=AzureLLMConfig(model="gpt-4"),
        tools=[TestOutputModel],
        structured_output_model=TestOutputModel,
        structured_output_version="v1",  # Test v1 version
    )

    # Should be routed as parser for v1
    route = config.tool_routes.get("TestOutputModel")
    assert route == "parser", f"Expected parser for v1, got {route}"

    # Check metadata
    metadata = config.tool_metadata.get("TestOutputModel")
    assert metadata["purpose"] == "structured_output"
    assert metadata["version"] == "v1"

    print("✅ Structured output detection works")


if __name__ == "__main__":
    print("🧪 Testing unified tool routing system...")

    try:
        test_aug_llm_config_inherits_tool_storage()
        test_tool_routing_analysis()
        test_tool_metadata_enhancement()
        test_get_tools_by_route()
        test_dynamic_tool_management()
        test_structured_output_detection()

        print("\n🎉 All tests passed! Unified tool routing system works correctly.")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise
