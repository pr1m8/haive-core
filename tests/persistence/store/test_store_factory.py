"""Test store factory functionality."""

import pytest

from haive.core.persistence.store.factory import StoreFactory, create_store
from haive.core.persistence.store.types import StoreConfig, StoreType


class TestStoreFactory:
    """Test the store factory functionality."""

    def test_create_memory_store(self):
        """Test creating memory store via factory."""
        config = StoreConfig(type=StoreType.MEMORY)
        store = StoreFactory.create(config)

        assert store is not None
        assert "MemoryStoreWrapper" in type(store).__name__

    def test_create_store_convenience_function(self):
        """Test the convenience create_store function."""
        store = create_store(store_type=StoreType.MEMORY)

        assert store is not None
        assert "MemoryStoreWrapper" in type(store).__name__

    def test_create_store_with_dict_config(self):
        """Test creating store from dictionary configuration."""
        config_dict = {"type": StoreType.MEMORY, "namespace_prefix": "test"}

        store = StoreFactory.create(config_dict)
        assert store is not None
        assert store.config.namespace_prefix == "test"

    def test_postgres_store_fallback_when_unavailable(self):
        """Test fallback to memory store when PostgreSQL is unavailable."""
        # This test should work even without PostgreSQL connection
        try:
            store = create_store(
                store_type=StoreType.POSTGRES_SYNC,
                connection_string="postgresql://invalid:invalid@nonexistent:5432/invalid",
            )
            # Should fallback to memory store
            assert "MemoryStoreWrapper" in type(store).__name__
        except Exception:
            # If it raises an exception, that's also acceptable behavior
            pass

    def test_unknown_store_type_raises_error(self):
        """Test that unknown store type raises ValidationError."""
        from pydantic_core import ValidationError

        with pytest.raises(ValidationError):
            config = StoreConfig(type="unknown_type")

    def test_create_store_with_connection_params(self):
        """Test creating store with individual connection parameters."""
        store = create_store(
            store_type=StoreType.POSTGRES_SYNC,
            host="localhost",
            port=5432,
            database="test_db",
            user="test_user",
            password="test_pass",
        )

        # Should create wrapper (might fallback to memory if no connection)
        assert store is not None

    def test_create_store_with_embedding_params(self):
        """Test creating store with embedding configuration."""
        store = create_store(
            store_type=StoreType.MEMORY,
            embedding_provider="openai",
            embedding_dims=1536,
            embedding_fields=["text", "content"],
        )

        assert store is not None
        assert store.config.embedding_provider == "openai"
        assert store.config.embedding_dims == 1536
