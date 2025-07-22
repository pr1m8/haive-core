"""PostgreSQL Saver with automatic thread creation.

This module provides a PostgreSQL checkpointer that automatically creates
threads before saving checkpoints, preventing foreign key constraint violations.
"""

import logging
import os
from typing import Any

from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg import Connection
from psycopg_pool import ConnectionPool

logger = logging.getLogger(__name__)


class PostgresSaverWithThreadCreation(PostgresSaver):
    """PostgreSQL checkpointer that ensures threads exist before saving checkpoints.

    This class extends the standard PostgresSaver to automatically create
    thread records before attempting to save checkpoints, preventing foreign
    key constraint violations.
    """

    def __init__(self, conn: Connection | ConnectionPool, **kwargs):
        """Initialize the PostgresSaver with thread creation capability.

        Args:
            conn: Database connection or connection pool
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(conn, **kwargs)
        self._thread_creation_cache = set()  # Cache to avoid duplicate thread creation

    def _ensure_thread_exists(self, thread_id: str, user_id: str = None) -> None:
        """Ensure a thread exists in the threads table.

        Args:
            thread_id: The thread ID to ensure exists
            user_id: Optional user ID. If None, generates/uses a default user_id
        """
        # Generate user_id if not provided (handle NOT NULL constraint)
        if user_id is None:
            # Try environment variable first
            user_id = os.getenv("DEFAULT_USER_ID") or os.getenv("USER_ID")

            if not user_id:
                # Use a real user_id from auth.users table
                # This is the first user ID we found in the database
                user_id = (
                    "5335c7e6-1d51-42d2-b958-0ad2ad2c269b"  # deloreanblack@gmail.com
                )

        # Check cache first to avoid unnecessary database calls
        cache_key = f"{thread_id}:{user_id}"
        if cache_key in self._thread_creation_cache:
            return

        try:
            # Handle both connection pool and direct connection
            if hasattr(self.conn, "connection"):
                # Using connection pool
                with self.conn.connection() as conn, conn.cursor() as cursor:
                    # Insert thread with required user_id (never NULL)
                    cursor.execute(
                        """
                        INSERT INTO threads (id, user_id, created_at, updated_at, last_access)
                        VALUES (%s, %s, NOW(), NOW(), NOW())
                        ON CONFLICT (id) DO NOTHING
                        """,
                        (thread_id, user_id),
                    )
            else:
                # Using direct connection
                with self.conn.cursor() as cursor:
                    # Insert thread with required user_id (never NULL)
                    cursor.execute(
                        """
                        INSERT INTO threads (id, user_id, created_at, updated_at, last_access)
                        VALUES (%s, %s, NOW(), NOW(), NOW())
                        ON CONFLICT (id) DO NOTHING
                        """,
                        (thread_id, user_id),
                    )

            # Add to cache with user_id context
            self._thread_creation_cache.add(cache_key)
            logger.debug(f"Thread {thread_id} ensured in database for user {user_id}")

        except Exception as e:
            logger.exception(f"Failed to ensure thread {thread_id} exists: {e}")
            raise

    def put(
        self,
        config: dict[str, Any],
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: dict[str, Any],
    ) -> dict[str, Any]:
        """Save a checkpoint after ensuring the thread exists.

        Args:
            config: Configuration containing thread_id
            checkpoint: Checkpoint data to save
            metadata: Checkpoint metadata
            new_versions: New channel versions

        Returns:
            Checkpoint configuration
        """
        # Extract thread_id from config
        thread_id = config.get("configurable", {}).get("thread_id")
        if not thread_id:
            raise ValueError("No thread_id found in checkpoint config")

        # Ensure thread exists before saving checkpoint
        self._ensure_thread_exists(thread_id)

        # Call parent put method
        return super().put(config, checkpoint, metadata, new_versions)

    def put_writes(
        self,
        config: dict[str, Any],
        writes: list[tuple[str, Any]],
        task_id: str,
    ) -> None:
        """Save writes after ensuring the thread exists.

        Args:
            config: Configuration containing thread_id
            writes: List of writes to save
            task_id: Task identifier
        """
        # Extract thread_id from config
        thread_id = config.get("configurable", {}).get("thread_id")
        if thread_id:
            # Ensure thread exists before saving writes
            self._ensure_thread_exists(thread_id)

        # Call parent put_writes method
        super().put_writes(config, writes, task_id)


class AsyncPostgresSaverWithThreadCreation:
    """Async version of PostgresSaver with thread creation.

    This class provides the same functionality as PostgresSaverWithThreadCreation
    but for async operations.
    """

    def __init__(self, conn, **kwargs) -> None:
        """Initialize the async PostgresSaver with thread creation capability.

        Args:
            conn: Async database connection or connection pool
            **kwargs: Additional arguments passed to parent class
        """
        # Import async version
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        self._base_saver = AsyncPostgresSaver(conn, **kwargs)
        self._thread_creation_cache = set()  # Cache to avoid duplicate thread creation
        self.conn = conn

    def __getattr__(self, name):
        """Delegate attribute access to the base saver."""
        return getattr(self._base_saver, name)

    async def _ensure_thread_exists(self, thread_id: str, user_id: str = None) -> None:
        """Ensure a thread exists in the threads table (async version).

        Args:
            thread_id: The thread ID to ensure exists
            user_id: Optional user ID. If None, generates/uses a default user_id
        """
        # Generate user_id if not provided (handle NOT NULL constraint)
        if user_id is None:
            # Try environment variable first
            user_id = os.getenv("DEFAULT_USER_ID") or os.getenv("USER_ID")

            if not user_id:
                # Use a real user_id from auth.users table
                # This is the first user ID we found in the database
                user_id = (
                    "5335c7e6-1d51-42d2-b958-0ad2ad2c269b"  # deloreanblack@gmail.com
                )

        # Check cache first to avoid unnecessary database calls
        cache_key = f"{thread_id}:{user_id}"
        if cache_key in self._thread_creation_cache:
            return

        try:
            # Handle both connection pool and direct connection
            if hasattr(self.conn, "connection"):
                # Using connection pool
                async with self.conn.connection() as conn, conn.cursor() as cursor:
                    # Insert thread with required user_id (never NULL)
                    await cursor.execute(
                        """
                        INSERT INTO threads (id, user_id, created_at, updated_at, last_access)
                        VALUES (%s, %s, NOW(), NOW(), NOW())
                        ON CONFLICT (id) DO NOTHING
                        """,
                        (thread_id, user_id),
                    )
            else:
                # Using direct connection
                async with self.conn.cursor() as cursor:
                    # Insert thread with required user_id (never NULL)
                    await cursor.execute(
                        """
                        INSERT INTO threads (id, user_id, created_at, updated_at, last_access)
                        VALUES (%s, %s, NOW(), NOW(), NOW())
                        ON CONFLICT (id) DO NOTHING
                        """,
                        (thread_id, user_id),
                    )

            # Add to cache with user_id context
            self._thread_creation_cache.add(cache_key)
            logger.debug(
                f"Thread {thread_id} ensured in database (async) for user {user_id}"
            )

        except Exception as e:
            logger.exception(f"Failed to ensure thread {thread_id} exists (async): {e}")
            raise

    async def put(
        self,
        config: dict[str, Any],
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: dict[str, Any],
    ) -> dict[str, Any]:
        """Save a checkpoint after ensuring the thread exists (async version).

        Args:
            config: Configuration containing thread_id
            checkpoint: Checkpoint data to save
            metadata: Checkpoint metadata
            new_versions: New channel versions

        Returns:
            Checkpoint configuration
        """
        # Extract thread_id from config
        thread_id = config.get("configurable", {}).get("thread_id")
        if not thread_id:
            raise ValueError("No thread_id found in checkpoint config")

        # Ensure thread exists before saving checkpoint
        await self._ensure_thread_exists(thread_id)

        # Call base put method
        return await self._base_saver.put(config, checkpoint, metadata, new_versions)

    async def put_writes(
        self,
        config: dict[str, Any],
        writes: list[tuple[str, Any]],
        task_id: str,
    ) -> None:
        """Save writes after ensuring the thread exists (async version).

        Args:
            config: Configuration containing thread_id
            writes: List of writes to save
            task_id: Task identifier
        """
        # Extract thread_id from config
        thread_id = config.get("configurable", {}).get("thread_id")
        if thread_id:
            # Ensure thread exists before saving writes
            await self._ensure_thread_exists(thread_id)

        # Call base put_writes method
        await self._base_saver.put_writes(config, writes, task_id)


def create_postgres_saver_with_thread_creation(
    conn: Connection | ConnectionPool, **kwargs
) -> PostgresSaverWithThreadCreation:
    """Factory function to create PostgresSaver with thread creation.

    Args:
        conn: Database connection or connection pool
        **kwargs: Additional arguments passed to PostgresSaver

    Returns:
        PostgresSaver instance with thread creation capability
    """
    return PostgresSaverWithThreadCreation(conn, **kwargs)


async def create_async_postgres_saver_with_thread_creation(
    conn, **kwargs
) -> AsyncPostgresSaverWithThreadCreation:
    """Factory function to create async PostgresSaver with thread creation.

    Args:
        conn: Async database connection or connection pool
        **kwargs: Additional arguments passed to AsyncPostgresSaver

    Returns:
        AsyncPostgresSaver instance with thread creation capability
    """
    return AsyncPostgresSaverWithThreadCreation(conn, **kwargs)
