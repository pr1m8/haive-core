# src/haive/core/persistence/store/factory.py
"""Store factory for creating store instances."""

import logging
from contextlib import asynccontextmanager, contextmanager
from typing import Any

from .base import SerializableStoreWrapper
from .types import StoreConfig, StoreType
from .wrappers.memory import MemoryStoreWrapper

# Import PostgreSQL wrappers if available
try:
    from .postgres import AsyncPostgresStoreWrapper, PostgresStoreWrapper
except ImportError:
    # PostgreSQL wrappers not available yet
    AsyncPostgresStoreWrapper = None
    PostgresStoreWrapper = None

logger = logging.getLogger(__name__)


class StoreFactory:
    """Factory for creating store instances."""

    @staticmethod
    def create(config: StoreConfig | dict[str, Any]) -> SerializableStoreWrapper:
        """Create a store wrapper from configuration.

        Args:
            config: Store configuration

        Returns:
            Store wrapper instance
        """
        # Convert dict to config if needed
        if isinstance(config, dict):
            config = StoreConfig(**config)

        # Create appropriate wrapper
        if config.type == StoreType.MEMORY:
            return MemoryStoreWrapper(config=config)

        if config.type == StoreType.POSTGRES_SYNC:
            if PostgresStoreWrapper is None:
                logger.warning(
                    "PostgresStore not available, falling back to memory store. "
                    "Install with: pip install langgraph-checkpoint-postgres"
                )
                return MemoryStoreWrapper(config=config)

            # Try to create PostgreSQL wrapper, fallback to memory on connection failure
            try:
                wrapper = PostgresStoreWrapper(config=config)
                # Test the connection by attempting to get the store
                wrapper.get_store()
                return wrapper
            except Exception as e:
                logger.warning(
                    f"PostgreSQL connection failed ({e}), falling back to memory store"
                )
                return MemoryStoreWrapper(config=config)

        if config.type == StoreType.POSTGRES_ASYNC:
            if AsyncPostgresStoreWrapper is None:
                logger.warning(
                    "AsyncPostgresStore not available, falling back to memory store. "
                    "Install with: pip install langgraph-checkpoint-postgres"
                )
                return MemoryStoreWrapper(config=config)

            # For async, we can't test connection here since this is sync method
            # Connection testing will happen when store is first used
            return AsyncPostgresStoreWrapper(config=config)

        raise ValueError(f"Unknown store type: {config.type}")

    @staticmethod
    @contextmanager
    def create_with_lifecycle(config: StoreConfig | dict[str, Any]):
        """Create store with lifecycle management.

        Args:
            config: Store configuration

        Yields:
            Store wrapper instance
        """
        store = StoreFactory.create(config)
        try:
            yield store
        finally:
            # Cleanup if needed
            pass

    @staticmethod
    @asynccontextmanager
    async def create_async_with_lifecycle(config: StoreConfig | dict[str, Any]):
        """Create async store with lifecycle management.

        Args:
            config: Store configuration

        Yields:
            Store wrapper instance
        """
        store = StoreFactory.create(config)
        try:
            yield store
        finally:
            # Cleanup if needed
            pass


# Convenience functions
def create_store(
    store_type: str | StoreType = StoreType.MEMORY, **kwargs
) -> SerializableStoreWrapper:
    """Create a store with simplified parameters.

    Args:
        store_type: Type of store
        **kwargs: Additional parameters

    Returns:
        Store wrapper instance
    """
    if isinstance(store_type, str):
        store_type = StoreType(store_type)

    # Handle common parameters
    config_dict = {"type": store_type}

    # Connection parameters
    if "connection_string" in kwargs:
        # Direct connection string takes precedence
        config_dict["connection_params"] = {
            "connection_string": kwargs.pop("connection_string")
        }
    elif "host" in kwargs:
        # Individual connection parameters
        config_dict["connection_params"] = {
            "host": kwargs.pop("host"),
            "port": kwargs.pop("port", 5432),
            "database": kwargs.pop("database", "postgres"),
            "user": kwargs.pop("user", "postgres"),
            "password": kwargs.pop("password", "postgres"),
        }

    # Embedding parameters
    if "embedding_provider" in kwargs:
        config_dict["embedding_provider"] = kwargs.pop("embedding_provider")
        config_dict["embedding_dims"] = kwargs.pop("embedding_dims", 1536)
        config_dict["embedding_fields"] = kwargs.pop("embedding_fields", None)

    # Add remaining kwargs
    config_dict.update(kwargs)

    config = StoreConfig(**config_dict)
    return StoreFactory.create(config)
