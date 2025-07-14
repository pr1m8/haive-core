#!/usr/bin/env python3
"""Debug script to isolate the agent integration issue."""

import logging
from typing import List

from langchain_core.tools import Tool

from haive.core.engine.aug_llm import AugLLMConfig
from haive.core.persistence.store.factory import create_store
from haive.core.persistence.store.types import StoreType
from haive.core.tools.store_manager import StoreManager
from haive.core.tools.store_tools import create_memory_tools_suite

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def test_tools_isolation():
    """Test the tools in isolation."""
    print("=== Testing Tools in Isolation ===\n")

    # Create store manager
    store = create_store(store_type=StoreType.MEMORY)
    store_manager = StoreManager(store=store, default_namespace=("debug", "test"))

    # Create tools
    tools = create_memory_tools_suite(store_manager)
    print(f"Created {len(tools)} tools")

    # Test each tool
    for i, tool in enumerate(tools):
        print(f"Tool {i}: {tool.name}")
        print(f"  Type: {type(tool)}")
        print(f"  Description: {tool.description[:80]}...")
        print(f"  Has func: {hasattr(tool, 'func')}")
        print(f"  Has get: {hasattr(tool, 'get')}")
        print()

    return tools


def test_augllm_config_creation():
    """Test AugLLMConfig creation with different parameters."""
    print("=== Testing AugLLMConfig Creation ===\n")

    # Test 1: Basic config (no tools)
    try:
        config1 = AugLLMConfig()
        print("✅ Basic AugLLMConfig creation successful")
    except Exception as e:
        print(f"❌ Basic AugLLMConfig creation failed: {e}")
        return None

    # Test 2: Config with empty tools list
    try:
        config2 = AugLLMConfig(tools=[])
        print("✅ AugLLMConfig with empty tools list successful")
    except Exception as e:
        print(f"❌ AugLLMConfig with empty tools list failed: {e}")
        return None

    return config1


def test_augllm_with_tools():
    """Test AugLLMConfig with actual tools."""
    print("=== Testing AugLLMConfig with Tools ===\n")

    # Create tools
    tools = test_tools_isolation()
    if not tools:
        print("❌ No tools to test with")
        return

    # Test with single tool
    try:
        single_tool = tools[0]
        print(f"Testing with single tool: {single_tool.name}")

        config = AugLLMConfig(tools=[single_tool])
        print("✅ AugLLMConfig with single tool successful")

    except Exception as e:
        print(f"❌ AugLLMConfig with single tool failed: {e}")
        import traceback

        print("Full traceback:")
        traceback.print_exc()

        # Let's trace exactly where this is happening
        print("\n=== Detailed Error Analysis ===")
        print(f"Tool type: {type(single_tool)}")
        print(f"Tool has get method: {hasattr(single_tool, 'get')}")
        print(
            f"Tool attributes: {[attr for attr in dir(single_tool) if 'get' in attr.lower()]}"
        )
        return

    # Test with all tools
    try:
        print(f"Testing with all {len(tools)} tools")
        config = AugLLMConfig(tools=tools)
        print("✅ AugLLMConfig with all tools successful")

    except Exception as e:
        print(f"❌ AugLLMConfig with all tools failed: {e}")
        import traceback

        print("Full traceback:")
        traceback.print_exc()


def inspect_tool_attributes():
    """Inspect tool attributes to understand the issue."""
    print("=== Inspecting Tool Attributes ===\n")

    # Create a tool
    store = create_store(store_type=StoreType.MEMORY)
    store_manager = StoreManager(store=store, default_namespace=("debug",))
    tools = create_memory_tools_suite(store_manager)

    if not tools:
        print("No tools created")
        return

    tool = tools[0]
    print(f"Inspecting tool: {tool.name}")
    print(f"Tool type: {type(tool)}")
    print(f"Tool MRO: {type(tool).__mro__}")
    print()

    # Check all attributes
    print("Tool attributes:")
    for attr in dir(tool):
        if not attr.startswith("_"):
            try:
                value = getattr(tool, attr)
                print(f"  {attr}: {type(value)} = {value}")
            except Exception as e:
                print(f"  {attr}: ERROR - {e}")

    print()

    # Check if it's a proper Tool instance
    print(f"Is Tool instance: {isinstance(tool, Tool)}")

    # Check Tool class attributes
    print("\nTool class expected attributes:")
    from langchain_core.tools import Tool as ToolClass

    for attr in ["name", "description", "func", "args_schema"]:
        has_attr = hasattr(ToolClass, attr)
        print(f"  {attr}: {has_attr}")


def main():
    """Run all debug tests."""
    print("🔍 Debug Agent Integration Issue\n")

    # Test 1: Tools in isolation
    tools = test_tools_isolation()

    # Test 2: AugLLMConfig creation
    config = test_augllm_config_creation()

    # Test 3: AugLLMConfig with tools
    test_augllm_with_tools()

    # Test 4: Inspect tool attributes
    inspect_tool_attributes()

    print("\n🏁 Debug tests completed")


if __name__ == "__main__":
    main()
