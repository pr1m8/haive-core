#!/usr/bin/env python3
"""Test PostgreSQL store implementation with prepared statement fixes.

This test suite validates the PostgreSQL store wrapper's ability to handle:
1. Prepared statement conflicts with connection pooling (Supabase/pgBouncer)
2. Proper pipeline mode disabling to prevent '_pg3_X' already exists errors
3. Connection failure fallback to memory store
4. Both synchronous and asynchronous store operations
"""

import asyncio
import logging
import os
import uuid
from typing import Optional

import pytest

# Test configuration - only apply asyncio mark to specific async tests


@pytest.fixture
def postgres_connection_string() -> Optional[str]:
    """Get PostgreSQL connection string from environment."""
    return os.getenv("POSTGRES_CONNECTION_STRING")


@pytest.fixture
def test_namespace():
    """Generate unique test namespace to avoid conflicts."""
    test_id = str(uuid.uuid4())[:8]
    return ("test", "postgres_store", test_id)


class TestPostgresStoreSync:
    """Test synchronous PostgreSQL store functionality."""

    def test_postgres_store_creation_with_valid_connection(
        self, postgres_connection_string
    ):
        """Test PostgreSQL store creation with valid connection."""
        if not postgres_connection_string:
            pytest.skip("No POSTGRES_CONNECTION_STRING environment variable")

        from haive.core.persistence.store.factory import create_store
        from haive.core.persistence.store.types import StoreType

        store = create_store(
            store_type=StoreType.POSTGRES_SYNC,
            connection_string=postgres_connection_string,
        )

        assert store is not None
        assert "PostgresStoreWrapper" in type(store).__name__

        # Verify pipeline mode is disabled
        actual_store = store.get_store()
        assert hasattr(actual_store, "supports_pipeline")
        assert actual_store.supports_pipeline is False

    def test_postgres_store_basic_operations(
        self, postgres_connection_string, test_namespace
    ):
        """Test basic store operations with PostgreSQL."""
        if not postgres_connection_string:
            pytest.skip("No POSTGRES_CONNECTION_STRING environment variable")

        from haive.core.persistence.store.factory import create_store
        from haive.core.persistence.store.types import StoreType

        store = create_store(
            store_type=StoreType.POSTGRES_SYNC,
            connection_string=postgres_connection_string,
        )

        # Test data
        key = f"test_key_{uuid.uuid4().hex[:8]}"
        value = {
            "message": "Hello from PostgreSQL store!",
            "test_id": key,
            "timestamp": "2025-01-14",
        }

        # Test put operation
        store.put(test_namespace, key, value)

        # Test get operation
        retrieved = store.get(test_namespace, key)
        assert retrieved == value

        # Test delete operation
        store.delete(test_namespace, key)

        # Verify deletion
        deleted_value = store.get(test_namespace, key)
        assert deleted_value is None

    def test_postgres_store_fallback_on_invalid_connection(self):
        """Test fallback to memory store with invalid connection."""
        from haive.core.persistence.store.factory import create_store
        from haive.core.persistence.store.types import StoreType

        # Use invalid connection string
        store = create_store(
            store_type=StoreType.POSTGRES_SYNC,
            connection_string="postgresql://invalid:invalid@nonexistent:5432/invalid",
        )

        # Should have fallen back to memory store
        assert "MemoryStoreWrapper" in type(store).__name__

        # Verify it works as memory store
        test_namespace = ("test", "fallback")
        key = "fallback_test"
        value = {"status": "memory_fallback_works"}

        store.put(test_namespace, key, value)
        retrieved = store.get(test_namespace, key)
        assert retrieved == value


class TestPostgresStoreAsync:
    """Test asynchronous PostgreSQL store functionality."""

    @pytest.mark.asyncio
    async def test_async_postgres_store_creation_with_valid_connection(
        self, postgres_connection_string
    ):
        """Test async PostgreSQL store creation with valid connection."""
        if not postgres_connection_string:
            pytest.skip("No POSTGRES_CONNECTION_STRING environment variable")

        from haive.core.persistence.store.factory import create_store
        from haive.core.persistence.store.types import StoreType

        store = create_store(
            store_type=StoreType.POSTGRES_ASYNC,
            connection_string=postgres_connection_string,
        )

        assert store is not None
        assert "AsyncPostgresStoreWrapper" in type(store).__name__

        # Verify pipeline mode is disabled
        actual_store = await store.get_async_store()
        assert hasattr(actual_store, "supports_pipeline")
        assert actual_store.supports_pipeline is False

    @pytest.mark.asyncio
    async def test_async_postgres_store_basic_operations(
        self, postgres_connection_string, test_namespace
    ):
        """Test basic async store operations with PostgreSQL."""
        if not postgres_connection_string:
            pytest.skip("No POSTGRES_CONNECTION_STRING environment variable")

        from haive.core.persistence.store.factory import create_store
        from haive.core.persistence.store.types import StoreType

        store = create_store(
            store_type=StoreType.POSTGRES_ASYNC,
            connection_string=postgres_connection_string,
        )

        # Test data
        key = f"test_async_key_{uuid.uuid4().hex[:8]}"
        value = {
            "message": "Hello from async PostgreSQL store!",
            "test_id": key,
            "async": True,
            "timestamp": "2025-01-14",
        }

        # Test async operations
        await store.aput(test_namespace, key, value)

        # Test async get
        retrieved = await store.aget(test_namespace, key)
        assert retrieved == value

        # Test async delete
        await store.adelete(test_namespace, key)

        # Verify deletion
        deleted_value = await store.aget(test_namespace, key)
        assert deleted_value is None


class TestPostgresStorePipelineFix:
    """Test the specific pipeline mode fix for prepared statement conflicts."""

    def test_pipeline_mode_disabled_on_creation(self, postgres_connection_string):
        """Test that pipeline mode is explicitly disabled to prevent prepared statement conflicts."""
        if not postgres_connection_string:
            pytest.skip("No POSTGRES_CONNECTION_STRING environment variable")

        from haive.core.persistence.store.postgres import PostgresStoreWrapper
        from haive.core.persistence.store.types import StoreConfig, StoreType

        config = StoreConfig(
            type=StoreType.POSTGRES_SYNC,
            connection_params={"connection_string": postgres_connection_string},
        )

        wrapper = PostgresStoreWrapper(config=config)
        store = wrapper.get_store()

        # Verify the critical fix is applied
        assert hasattr(store, "supports_pipeline")
        assert (
            store.supports_pipeline is False
        ), "Pipeline mode should be disabled to prevent prepared statement conflicts"

    @pytest.mark.asyncio
    async def test_async_pipeline_mode_disabled_on_creation(
        self, postgres_connection_string
    ):
        """Test that async pipeline mode is explicitly disabled."""
        if not postgres_connection_string:
            pytest.skip("No POSTGRES_CONNECTION_STRING environment variable")

        from haive.core.persistence.store.postgres import AsyncPostgresStoreWrapper
        from haive.core.persistence.store.types import StoreConfig, StoreType

        config = StoreConfig(
            type=StoreType.POSTGRES_ASYNC,
            connection_params={"connection_string": postgres_connection_string},
        )

        wrapper = AsyncPostgresStoreWrapper(config=config)
        store = await wrapper.get_async_store()

        # Verify the critical fix is applied
        assert hasattr(store, "supports_pipeline")
        assert (
            store.supports_pipeline is False
        ), "Async pipeline mode should be disabled to prevent prepared statement conflicts"

    def test_no_prepared_statement_conflicts_with_multiple_operations(
        self, postgres_connection_string, test_namespace
    ):
        """Test that multiple operations don't create prepared statement conflicts."""
        if not postgres_connection_string:
            pytest.skip("No POSTGRES_CONNECTION_STRING environment variable")

        from haive.core.persistence.store.factory import create_store
        from haive.core.persistence.store.types import StoreType

        store = create_store(
            store_type=StoreType.POSTGRES_SYNC,
            connection_string=postgres_connection_string,
        )

        # Perform multiple operations that would previously cause '_pg3_X' conflicts
        for i in range(5):
            key = f"multi_op_test_{i}"
            value = {"operation": i, "data": f"test_data_{i}"}

            # These operations should not raise prepared statement errors
            store.put(test_namespace, key, value)
            retrieved = store.get(test_namespace, key)
            assert retrieved == value
            store.delete(test_namespace, key)


class TestStoreIntegration:
    """Test store integration with agents and other components."""

    @pytest.mark.asyncio
    async def test_agent_with_postgres_store(self, postgres_connection_string):
        """Test agent integration with PostgreSQL store."""
        if not postgres_connection_string:
            pytest.skip("No POSTGRES_CONNECTION_STRING environment variable")

        from haive.agents.simple import SimpleAgent

        from haive.core.engine.aug_llm import AugLLMConfig
        from haive.core.persistence.postgres_config import PostgresCheckpointerConfig
        from haive.core.persistence.types import CheckpointerMode, CheckpointStorageMode

        # Create PostgreSQL persistence config
        persistence_config = PostgresCheckpointerConfig(
            connection_string=postgres_connection_string,
            mode=CheckpointerMode.SYNC,
            storage_mode=CheckpointStorageMode.FULL,
            prepare_threshold=None,  # Disable prepared statements
            auto_commit=True,
        )

        # Create agent with PostgreSQL persistence and store enabled
        agent = SimpleAgent(
            name="test_agent_with_postgres_store",
            engine=AugLLMConfig(),
            persistence=persistence_config,
            add_store=True,  # Enable store
        )

        assert agent.store is not None
        assert "PostgresStoreWrapper" in type(agent.store).__name__

        # Test store functionality through agent
        namespace = ("agent", agent.name)
        key = "agent_memory"
        value = {"message": "Agent can remember this!", "timestamp": "2025-01-14"}

        agent.store.put(namespace, key, value)
        retrieved = agent.store.get(namespace, key)

        # Handle Item object if needed
        if hasattr(retrieved, "value"):
            retrieved_value = retrieved.value
        else:
            retrieved_value = retrieved

        assert retrieved_value == value


if __name__ == "__main__":
    # Allow running as standalone script for debugging
    pytest.main([__file__, "-v"])
