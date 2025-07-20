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
    # Create store manager with memory store
    store = create_store(store_type=StoreType.MEMORY)
    store_manager = StoreManager(
        store=store, default_namespace=(
            "test", "demo"))

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

    # Retrieve memories
    store_manager.retrieve_memory(memory_id1)
    store_manager.retrieve_memory(memory_id2)

    # Test tools

    tools = create_memory_tools_suite(store_manager)
    for _tool in tools:
        pass

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

    # Test retrieve tool
    retrieve_tool = tools[2]  # retrieve_memory tool
    retrieve_result = retrieve_tool.func(memory_id=result_data["memory_id"])
    json.loads(retrieve_result)

    # Test namespace creation

    store_manager.create_user_namespace("user123")
    store_manager.create_agent_namespace("agent1", "user123")
    store_manager.create_session_namespace("session1", "agent1", "user123")


if __name__ == "__main__":
    test_store_system()
