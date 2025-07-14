#!/usr/bin/env python3
"""Minimal test to isolate the AugLLMConfig tool integration issue."""

import traceback

from langchain_core.tools import Tool

from haive.core.engine.aug_llm import AugLLMConfig
from haive.core.persistence.store.factory import create_store
from haive.core.persistence.store.types import StoreType
from haive.core.tools.store_manager import StoreManager
from haive.core.tools.store_tools import create_store_memory_tool


def test_minimal_case():
    """Test the minimal failing case."""
    print("=== Minimal AugLLMConfig Tool Test ===\n")

    # Create a minimal tool
    store = create_store(store_type=StoreType.MEMORY)
    store_manager = StoreManager(store=store, default_namespace=("test",))
    tool = create_store_memory_tool(store_manager)

    print(f"Tool type: {type(tool)}")
    print(f"Tool name: {tool.name}")
    print(f"Tool is LangChain Tool: {isinstance(tool, Tool)}")
    print(f"Tool has get method: {hasattr(tool, 'get')}")
    print()

    # Try to create AugLLMConfig with this tool
    print("Creating AugLLMConfig with tool...")

    try:
        config = AugLLMConfig(tools=[tool])
        print("✅ SUCCESS: AugLLMConfig created successfully!")

    except Exception as e:
        print(f"❌ FAILED: {e}")
        print("\nFull traceback:")
        traceback.print_exc()

        # Let's also try to isolate where exactly this is happening
        print(f"\nError type: {type(e)}")
        print(f"Error args: {e.args}")

        # Check the exception traceback for the specific line
        tb = traceback.extract_tb(e.__traceback__)
        print(f"\nTraceback locations:")
        for frame in tb:
            print(f"  File: {frame.filename}")
            print(f"  Line: {frame.lineno}")
            print(f"  Function: {frame.name}")
            print(f"  Code: {frame.line}")
            print()


if __name__ == "__main__":
    test_minimal_case()
