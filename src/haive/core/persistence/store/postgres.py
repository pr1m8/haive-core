"""PostgreSQL store implementation for LangGraph.

This module provides synchronous and asynchronous PostgreSQL store wrappers
that integrate with LangGraph's BaseStore interface for persistent storage.
"""

import logging
from typing import Any

from core.persistence.postgres_config import PostgresCheckpointerConfig
from langgraph.store.base import BaseStore
from pydantic import Field, PrivateAttr

from .base import SerializableStoreWrapper
from .types import StoreConfig

logger = logging.getLogger(__name__)


class PostgresStoreWrapper(SerializableStoreWrapper):
    """Synchronous PostgreSQL store wrapper.

    This wrapper provides a serializable interface to LangGraph's PostgresStore
    with support for configuration management and lifecycle handling.
    """

    config: StoreConfig = Field(description="Store configuration")
    postgres_config: PostgresCheckpointerConfig | None = Field(
        default=None,
        description="Optional PostgreSQL configuration for connection details",
    )
    _store_context: Any | None = PrivateAttr(default=None)

    def _create_store(self) -> BaseStore:
        """Create the underlying PostgreSQL store.

        Returns:
            PostgresStore instance
        """
        try:
            from langgraph.store.postgres import PostgresStore
        except ImportError:
            logger.exception(
                "PostgresStore not available. Install with: pip install langgraph-checkpoint-postgres"
            )
            # Fallback to memory store
            from langgraph.store.memory import InMemoryStore

            logger.warning("Falling back to InMemoryStore")
            return InMemoryStore()

        # Build connection string
        if self.postgres_config:
            conn_string = self.postgres_config.get_connection_uri()
        elif self.config.connection_params:
            params = self.config.connection_params

            # Check for direct connection string first
            if "connection_string" in params:
                conn_string = params["connection_string"]
            else:
                # Build from individual connection params
                conn_string = (
                    f"postgresql://{params.get('user', 'postgres')}:"
                    f"{params.get('password', 'postgres')}@"
                    f"{params.get('host', 'localhost')}:"
                    f"{params.get('port', 5432)}/"
                    f"{params.get('database', 'postgres')}"
                )

                # Add SSL mode if specified
                if params.get("sslmode"):
                    conn_string += f"?sslmode={params['sslmode']}"
        else:
            raise ValueError("No connection configuration provided")

        # Create store with optional index configuration
        index_config = None
        if self.config.index_config:
            index_config = {
                "dims": self.config.index_config.get("dims", 1536),
                "fields": self.config.index_config.get("fields", ["text"]),
            }

            # Add embedding function if specified
            if "embed_model" in self.config.index_config:
                try:
                    from langchain_community.embeddings import init_embeddings

                    index_config["embed"] = init_embeddings(
                        self.config.index_config["embed_model"]
                    )
                except ImportError:
                    logger.warning("Could not initialize embeddings")

        # Create store with proper connection configuration
        try:
            # Import psycopg for connection configuration
            import psycopg
            from psycopg.rows import dict_row

            # Create connection with prepared statements disabled
            connection_kwargs = {
                "autocommit": True,
                # Completely disable prepared statements (not just 0)
                "prepare_threshold": None,
                "row_factory": dict_row,
            }

            # Create connection
            conn = psycopg.connect(conn_string, **connection_kwargs)

            # EXTRA FIX: Clear any existing prepared statements
            try:
                with conn.cursor() as cur:
                    # Deallocate all prepared statements to start fresh
                    cur.execute("DEALLOCATE ALL;")
                    logger.debug("Cleared all existing prepared statements")
            except Exception as e:
                logger.debug(
                    f"Could not clear prepared statements (this is normal): {e}"
                )

            # Create store with the configured connection
            store = (
                PostgresStore(conn, index=index_config)
                if index_config
                else PostgresStore(conn)
            )

            # CRITICAL FIX: Force disable pipeline mode BEFORE setup to prevent prepared statement conflicts
            # This solves the "prepared statement '_pg3_X' already exists" error with
            # connection pooling (especially Supabase pgBouncer in transaction
            # mode)
            store.supports_pipeline = False
            logger.debug(
                "Forced pipeline mode OFF to prevent prepared statement conflicts"
            )

            # Store connection for cleanup
            self._store_context = conn

            # Run migrations if needed (after disabling pipeline mode)
            if self.config.setup_on_init:
                try:
                    store.setup()
                    logger.info("PostgreSQL store migrations completed")
                except Exception as e:
                    logger.warning(f"Could not run store migrations: {e}")

        except Exception as e:
            logger.exception(f"Failed to create PostgresStore: {e}")
            raise

        return store

    async def _create_async_store(self) -> BaseStore:
        """Create async store (not supported for sync wrapper)."""
        raise NotImplementedError(
            "Sync wrapper does not support async store creation. Use AsyncPostgresStoreWrapper."
        )


class AsyncPostgresStoreWrapper(SerializableStoreWrapper):
    """Asynchronous PostgreSQL store wrapper.

    This wrapper provides a serializable interface to LangGraph's AsyncPostgresStore
    with support for configuration management and lifecycle handling.
    """

    config: StoreConfig = Field(description="Store configuration")
    postgres_config: PostgresCheckpointerConfig | None = Field(
        default=None,
        description="Optional PostgreSQL configuration for connection details",
    )
    _store_context: Any | None = PrivateAttr(default=None)

    def _create_store(self) -> BaseStore:
        """Create sync store (not supported for async wrapper)."""
        raise NotImplementedError(
            "Async wrapper does not support sync store creation. Use PostgresStoreWrapper."
        )

    async def _create_async_store(self) -> BaseStore:
        """Create the underlying async PostgreSQL store.

        Returns:
            AsyncPostgresStore instance
        """
        try:
            from langgraph.store.postgres.aio import AsyncPostgresStore
        except ImportError:
            logger.exception(
                "AsyncPostgresStore not available. Install with: pip install langgraph-checkpoint-postgres"
            )
            # Fallback to memory store
            from langgraph.store.memory import InMemoryStore

            logger.warning("Falling back to InMemoryStore")
            return InMemoryStore()

        # Build connection string
        if self.postgres_config:
            conn_string = self.postgres_config.get_connection_uri()
        elif self.config.connection_params:
            params = self.config.connection_params

            # Check for direct connection string first
            if "connection_string" in params:
                conn_string = params["connection_string"]
            else:
                # Build from individual connection params
                conn_string = (
                    f"postgresql://{params.get('user', 'postgres')}:"
                    f"{params.get('password', 'postgres')}@"
                    f"{params.get('host', 'localhost')}:"
                    f"{params.get('port', 5432)}/"
                    f"{params.get('database', 'postgres')}"
                )

                # Add SSL mode if specified
                if params.get("sslmode"):
                    conn_string += f"?sslmode={params['sslmode']}"
        else:
            raise ValueError("No connection configuration provided")

        # Create store with optional index configuration
        index_config = None
        if self.config.index_config:
            index_config = {
                "dims": self.config.index_config.get("dims", 1536),
                "fields": self.config.index_config.get("fields", ["text"]),
            }

            # Add embedding function if specified
            if "embed_model" in self.config.index_config:
                try:
                    from langchain_community.embeddings import init_embeddings

                    index_config["embed"] = init_embeddings(
                        self.config.index_config["embed_model"]
                    )
                except ImportError:
                    logger.warning("Could not initialize embeddings")

        # Create store with proper connection configuration
        try:
            # Import psycopg for async connection configuration
            import psycopg
            from psycopg.rows import dict_row

            # Create async connection with prepared statements disabled
            connection_kwargs = {
                "autocommit": True,
                # Completely disable prepared statements (not just 0)
                "prepare_threshold": None,
                "row_factory": dict_row,
            }

            # Create async connection
            conn = await psycopg.AsyncConnection.connect(
                conn_string, **connection_kwargs
            )

            # EXTRA FIX: Clear any existing prepared statements
            try:
                async with conn.cursor() as cur:
                    # Deallocate all prepared statements to start fresh
                    await cur.execute("DEALLOCATE ALL;")
                    logger.debug("Cleared all existing prepared statements (async)")
            except Exception as e:
                logger.debug(
                    f"Could not clear prepared statements (this is normal): {e}"
                )

            # Create store with the configured connection
            if index_config:
                store = AsyncPostgresStore(conn, index=index_config)
            else:
                store = AsyncPostgresStore(conn)

            # CRITICAL FIX: Force disable pipeline mode BEFORE setup to prevent prepared statement conflicts
            # This solves the "prepared statement '_pg3_X' already exists" error with
            # connection pooling (especially Supabase pgBouncer in transaction
            # mode)
            store.supports_pipeline = False
            logger.debug(
                "Forced async pipeline mode OFF to prevent prepared statement conflicts"
            )

            # Store connection for cleanup
            self._store_context = conn

            # Run migrations if needed (after disabling pipeline mode)
            if self.config.setup_on_init:
                try:
                    await store.setup()
                    logger.info("Async PostgreSQL store migrations completed")
                except Exception as e:
                    logger.warning(f"Could not run store migrations: {e}")

        except Exception as e:
            logger.exception(f"Failed to create AsyncPostgresStore: {e}")
            raise

        return store
