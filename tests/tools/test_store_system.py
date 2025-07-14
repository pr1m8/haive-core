"""Tests for the store management system and tools."""

import json
from datetime import datetime

import pytest

from haive.core.persistence.store.factory import create_store
from haive.core.persistence.store.types import StoreType
from haive.core.tools.store_manager import MemoryEntry, StoreManager
from haive.core.tools.store_tools import (
    create_delete_memory_tool,
    create_memory_tools_suite,
    create_retrieve_memory_tool,
    create_search_memory_tool,
    create_store_memory_tool,
    create_update_memory_tool,
)


class TestMemoryEntry:
    """Test the MemoryEntry model."""

    def test_memory_entry_creation(self):
        """Test creating a memory entry."""
        entry = MemoryEntry(
            content="Test memory",
            category="test",
            importance=0.8,
            tags=["test", "demo"],
            metadata={"source": "test"},
        )

        assert entry.content == "Test memory"
        assert entry.category == "test"
        assert entry.importance == 0.8
        assert entry.tags == ["test", "demo"]
        assert entry.metadata == {"source": "test"}
        assert isinstance(entry.id, str)
        assert isinstance(entry.created_at, datetime)

    def test_memory_entry_serialization(self):
        """Test memory entry to/from store value conversion."""
        entry = MemoryEntry(content="Test memory", category="test", importance=0.8)

        # Convert to store value
        store_value = entry.to_store_value()
        assert isinstance(store_value, dict)
        assert store_value["content"] == "Test memory"
        assert store_value["category"] == "test"
        assert store_value["importance"] == 0.8

        # Convert back from store value
        restored_entry = MemoryEntry.from_store_value(store_value)
        assert restored_entry.content == entry.content
        assert restored_entry.category == entry.category
        assert restored_entry.importance == entry.importance
        assert restored_entry.id == entry.id


class TestStoreManager:
    """Test the StoreManager class."""

    @pytest.fixture
    def store_manager(self):
        """Create a test store manager with memory store."""
        store = create_store(store_type=StoreType.MEMORY)
        return StoreManager(store=store, default_namespace=("test", "memories"))

    def test_store_manager_initialization(self, store_manager):
        """Test store manager initialization."""
        assert store_manager.store is not None
        assert store_manager.default_namespace == ("test", "memories")

    def test_store_and_retrieve_memory(self, store_manager):
        """Test storing and retrieving a memory."""
        # Store a memory
        memory_id = store_manager.store_memory(
            content="Test memory content",
            category="test",
            importance=0.7,
            tags=["test"],
            metadata={"source": "unittest"},
        )

        assert isinstance(memory_id, str)

        # Retrieve the memory
        memory = store_manager.retrieve_memory(memory_id)

        assert memory is not None
        assert memory.content == "Test memory content"
        assert memory.category == "test"
        assert memory.importance == 0.7
        assert memory.tags == ["test"]
        assert memory.metadata == {"source": "unittest"}

    def test_update_memory(self, store_manager):
        """Test updating a memory."""
        # Store a memory
        memory_id = store_manager.store_memory(
            content="Original content", category="test"
        )

        # Update the memory
        success = store_manager.update_memory(
            memory_id=memory_id,
            content="Updated content",
            importance=0.9,
            tags=["updated"],
        )

        assert success is True

        # Retrieve and verify update
        memory = store_manager.retrieve_memory(memory_id)
        assert memory.content == "Updated content"
        assert memory.importance == 0.9
        assert memory.tags == ["updated"]

    def test_delete_memory(self, store_manager):
        """Test deleting a memory."""
        # Store a memory
        memory_id = store_manager.store_memory(content="To be deleted", category="test")

        # Verify it exists
        memory = store_manager.retrieve_memory(memory_id)
        assert memory is not None

        # Delete the memory
        success = store_manager.delete_memory(memory_id)
        assert success is True

        # Verify it's gone
        memory = store_manager.retrieve_memory(memory_id)
        assert memory is None

    def test_namespace_creation(self, store_manager):
        """Test namespace creation methods."""
        # User namespace
        user_ns = store_manager.create_user_namespace("user123")
        assert user_ns == ("haive", "users", "user123", "memories")

        # Agent namespace
        agent_ns = store_manager.create_agent_namespace("agent1")
        assert agent_ns == ("haive", "agents", "agent1", "memories")

        # User-agent namespace
        user_agent_ns = store_manager.create_agent_namespace(
            agent_id="agent1", user_id="user123"
        )
        assert user_agent_ns == (
            "haive",
            "users",
            "user123",
            "agents",
            "agent1",
            "memories",
        )

        # Session namespace
        session_ns = store_manager.create_session_namespace(
            session_id="session1", agent_id="agent1", user_id="user123"
        )
        assert session_ns == (
            "haive",
            "users",
            "user123",
            "agents",
            "agent1",
            "sessions",
            "session1",
        )


class TestStoreTools:
    """Test the store tools for agents."""

    @pytest.fixture
    def store_manager(self):
        """Create a test store manager."""
        store = create_store(store_type=StoreType.MEMORY)
        return StoreManager(store=store, default_namespace=("test", "tools"))

    def test_store_memory_tool(self, store_manager):
        """Test the store memory tool."""
        tool = create_store_memory_tool(store_manager)

        # Test the tool
        result = tool.func(
            content="Test tool memory",
            category="tool_test",
            importance=0.8,
            tags=["tool", "test"],
            metadata={"source": "tool_test"},
        )

        # Parse result
        result_data = json.loads(result)

        assert result_data["success"] is True
        assert "memory_id" in result_data

        # Verify memory was stored
        memory_id = result_data["memory_id"]
        memory = store_manager.retrieve_memory(memory_id)
        assert memory is not None
        assert memory.content == "Test tool memory"

    def test_search_memory_tool(self, store_manager):
        """Test the search memory tool."""
        # Store some test memories
        memory_ids = []
        for i in range(3):
            memory_id = store_manager.store_memory(
                content=f"Test memory {i}",
                category="search_test",
                importance=0.5 + i * 0.2,
                tags=["search", f"test{i}"],
            )
            memory_ids.append(memory_id)

        # Create and test search tool
        tool = create_search_memory_tool(store_manager)

        # Search for memories
        result = tool.func(query="test memory", category="search_test", limit=5)

        # Parse result
        result_data = json.loads(result)

        assert result_data["success"] is True
        # Note: Memory store doesn't support semantic search,
        # so results might be empty. This is expected behavior.
        assert "memories" in result_data
        assert "count" in result_data

    def test_retrieve_memory_tool(self, store_manager):
        """Test the retrieve memory tool."""
        # Store a test memory
        memory_id = store_manager.store_memory(
            content="Retrieve test memory", category="retrieve_test"
        )

        # Create and test retrieve tool
        tool = create_retrieve_memory_tool(store_manager)

        # Retrieve the memory
        result = tool.func(memory_id=memory_id)

        # Parse result
        result_data = json.loads(result)

        assert result_data["success"] is True
        assert result_data["memory"]["content"] == "Retrieve test memory"
        assert result_data["memory"]["category"] == "retrieve_test"

    def test_update_memory_tool(self, store_manager):
        """Test the update memory tool."""
        # Store a test memory
        memory_id = store_manager.store_memory(
            content="Original content", category="update_test"
        )

        # Create and test update tool
        tool = create_update_memory_tool(store_manager)

        # Update the memory
        result = tool.func(
            memory_id=memory_id, content="Updated content", importance=0.9
        )

        # Parse result
        result_data = json.loads(result)

        assert result_data["success"] is True

        # Verify update
        memory = store_manager.retrieve_memory(memory_id)
        assert memory.content == "Updated content"
        assert memory.importance == 0.9

    def test_delete_memory_tool(self, store_manager):
        """Test the delete memory tool."""
        # Store a test memory
        memory_id = store_manager.store_memory(
            content="To be deleted", category="delete_test"
        )

        # Create and test delete tool
        tool = create_delete_memory_tool(store_manager)

        # Delete the memory
        result = tool.func(memory_id=memory_id)

        # Parse result
        result_data = json.loads(result)

        assert result_data["success"] is True

        # Verify deletion
        memory = store_manager.retrieve_memory(memory_id)
        assert memory is None

    def test_memory_tools_suite(self, store_manager):
        """Test creating a complete tools suite."""
        tools = create_memory_tools_suite(store_manager)

        # Should have all 5 tools by default
        assert len(tools) == 5

        tool_names = [tool.name for tool in tools]
        expected_names = [
            "store_memory",
            "search_memory",
            "retrieve_memory",
            "update_memory",
            "delete_memory",
        ]

        for expected in expected_names:
            assert expected in tool_names

    def test_partial_tools_suite(self, store_manager):
        """Test creating a partial tools suite."""
        tools = create_memory_tools_suite(
            store_manager, include_tools=["store", "search"]
        )

        assert len(tools) == 2

        tool_names = [tool.name for tool in tools]
        assert "store_memory" in tool_names
        assert "search_memory" in tool_names


class TestStoreSystemIntegration:
    """Integration tests for the complete store system."""

    @pytest.fixture
    def store_manager(self):
        """Create a test store manager."""
        store = create_store(store_type=StoreType.MEMORY)
        return StoreManager(store=store)

    def test_complete_memory_workflow(self, store_manager):
        """Test a complete memory workflow using tools."""
        # Create tools
        store_tool = create_store_memory_tool(store_manager)
        create_search_memory_tool(store_manager)
        retrieve_tool = create_retrieve_memory_tool(store_manager)
        update_tool = create_update_memory_tool(store_manager)
        delete_tool = create_delete_memory_tool(store_manager)

        # 1. Store a memory
        store_result = store_tool.func(
            content="User prefers coffee over tea",
            category="user_preference",
            importance=0.8,
            tags=["beverage", "preference"],
        )

        store_data = json.loads(store_result)
        assert store_data["success"] is True
        memory_id = store_data["memory_id"]

        # 2. Retrieve the memory
        retrieve_result = retrieve_tool.func(memory_id=memory_id)
        retrieve_data = json.loads(retrieve_result)

        assert retrieve_data["success"] is True
        assert retrieve_data["memory"]["content"] == "User prefers coffee over tea"

        # 3. Update the memory
        update_result = update_tool.func(
            memory_id=memory_id,
            content="User prefers coffee over tea, especially in the morning",
            importance=0.9,
        )

        update_data = json.loads(update_result)
        assert update_data["success"] is True

        # 4. Verify update
        retrieve_result2 = retrieve_tool.func(memory_id=memory_id)
        retrieve_data2 = json.loads(retrieve_result2)

        assert "especially in the morning" in retrieve_data2["memory"]["content"]
        assert retrieve_data2["memory"]["importance"] == 0.9

        # 5. Delete the memory
        delete_result = delete_tool.func(memory_id=memory_id)
        delete_data = json.loads(delete_result)

        assert delete_data["success"] is True

        # 6. Verify deletion
        retrieve_result3 = retrieve_tool.func(memory_id=memory_id)
        retrieve_data3 = json.loads(retrieve_result3)

        assert retrieve_data3["success"] is False

    def test_namespace_isolation(self, store_manager):
        """Test that namespaces properly isolate memories."""
        namespace1 = ("test", "user1")
        namespace2 = ("test", "user2")

        # Store same memory ID in different namespaces
        test_id = "same_id"

        memory1 = MemoryEntry(id=test_id, content="User 1 memory", category="test")

        memory2 = MemoryEntry(id=test_id, content="User 2 memory", category="test")

        # Store in different namespaces
        store_manager.store.put(namespace1, test_id, memory1.to_store_value())
        store_manager.store.put(namespace2, test_id, memory2.to_store_value())

        # Retrieve from both namespaces
        retrieved1 = store_manager.store.get(namespace1, test_id)
        retrieved2 = store_manager.store.get(namespace2, test_id)

        assert retrieved1["content"] == "User 1 memory"
        assert retrieved2["content"] == "User 2 memory"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
