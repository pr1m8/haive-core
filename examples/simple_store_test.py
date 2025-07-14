#!/usr/bin/env python3
"""Simple test of the store system without agent integration."""

import logging

from haive.core.persistence.store.factory import create_store
from haive.core.persistence.store.types import StoreType
from haive.core.tools.store_manager import StoreManager
from haive.core.tools.store_tools import create_memory_tools_suite

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_store_system():
    """Test the basic store system functionality."""

    print("=== Store System Test ===\n")

    # Create store manager with memory store
    store = create_store(store_type=StoreType.MEMORY)
    store_manager = StoreManager(store=store, default_namespace=("test", "demo"))

    print(f"Created store manager with: {type(store).__name__}")

    # Store some memories
    memory_id1 = store_manager.store_memory(
        content="User loves coffee and prefers it in the morning",
        category="user_preference",
        importance=0.8,
        tags=["beverage", "morning"],
        metadata={"source": "conversation"},
    )

    memory_id2 = store_manager.store_memory(
        content="User is working on a Python project using FastAPI",
        category="fact",
        importance=0.7,
        tags=["work", "python", "fastapi"],
        metadata={"source": "conversation"},
    )

    print(f"Stored memory 1: {memory_id1}")
    print(f"Stored memory 2: {memory_id2}")

    # Retrieve memories
    memory1 = store_manager.retrieve_memory(memory_id1)
    memory2 = store_manager.retrieve_memory(memory_id2)

    print(f"\nRetrieved memory 1: {memory1.content}")
    print(f"Category: {memory1.category}, Importance: {memory1.importance}")

    print(f"\nRetrieved memory 2: {memory2.content}")
    print(f"Category: {memory2.category}, Tags: {memory2.tags}")

    # Test tools
    print("\n=== Testing Tools ===")

    tools = create_memory_tools_suite(store_manager)
    print(f"Created {len(tools)} tools:")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description[:50]}...")

    # Test store tool
    store_tool = tools[0]  # store_memory tool
    result = store_tool.func(
        content="User mentioned they have a meeting at 3 PM tomorrow",
        category="event",
        importance=0.9,
        tags=["meeting", "schedule"],
    )

    import json

    result_data = json.loads(result)
    print(f"\nStore tool result: {result_data}")

    # Test retrieve tool
    retrieve_tool = tools[2]  # retrieve_memory tool
    retrieve_result = retrieve_tool.func(memory_id=result_data["memory_id"])
    retrieve_data = json.loads(retrieve_result)

    print(f"Retrieve tool result: {retrieve_data['memory']['content']}")

    # Test namespace creation
    print("\n=== Testing Namespaces ===")

    user_ns = store_manager.create_user_namespace("user123")
    agent_ns = store_manager.create_agent_namespace("agent1", "user123")
    session_ns = store_manager.create_session_namespace("session1", "agent1", "user123")

    print(f"User namespace: {user_ns}")
    print(f"Agent namespace: {agent_ns}")
    print(f"Session namespace: {session_ns}")

    print("\n✅ Store system test completed successfully!")


if __name__ == "__main__":
    test_store_system()
