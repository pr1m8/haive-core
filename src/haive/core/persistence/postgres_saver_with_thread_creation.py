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
        self._schema_type = None  # Will be detected on first use

    def _detect_schema_type(self, cursor) -> str:
        """Detect if using Supabase or standard LangGraph schema.

        Returns:
            'supabase' if using Supabase schema with composite constraint
            'standard' if using standard LangGraph schema with single constraint
        """
        try:
            # Check if user_id column exists in threads table
            cursor.execute(
                """
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'threads' 
                AND column_name = 'user_id'
            """
            )
            has_user_id = cursor.fetchone() is not None

            if not has_user_id:
                return "standard"

            # Check for composite unique constraint on (id, user_id)
            cursor.execute(
                """
                SELECT COUNT(*) 
                FROM information_schema.table_constraints tc
                JOIN information_schema.constraint_column_usage AS ccu 
                    USING (constraint_schema, constraint_name)
                WHERE tc.table_name = 'threads' 
                AND tc.constraint_type = 'UNIQUE'
                AND ccu.column_name IN ('id', 'user_id')
                GROUP BY tc.constraint_name
                HAVING COUNT(*) = 2
            """
            )
            has_composite = cursor.fetchone() is not None

            return "supabase" if has_composite else "standard"

        except Exception as e:
            logger.warning(f"Failed to detect schema type, assuming standard: {e}")
            return "standard"

    def _ensure_thread_exists(self, thread_id: str, user_id: str = None) -> None:
        """Ensure a thread exists in the threads table.

        Args:
            thread_id: The thread ID to ensure exists
            user_id: Optional user ID. If None, generates/uses a default user_id
        """
        # Detect schema type on first use
        if self._schema_type is None:
            if hasattr(self.conn, "connection"):
                with self.conn.connection() as conn, conn.cursor() as cursor:
                    self._schema_type = self._detect_schema_type(cursor)
            else:
                with self.conn.cursor() as cursor:
                    self._schema_type = self._detect_schema_type(cursor)

            # Also check environment variable override
            schema_override = os.getenv("POSTGRES_THREADS_SCHEMA_TYPE")
            if schema_override in ["supabase", "standard"]:
                self._schema_type = schema_override

            logger.info(f"Detected PostgreSQL threads schema type: {self._schema_type}")

        # For Supabase schema, we need a user_id
        if self._schema_type == "supabase" and user_id is None:
            # Try environment variable first
            user_id = (
                os.getenv("HAIVE_USER_ID")
                or os.getenv("DEFAULT_USER_ID")
                or os.getenv("USER_ID")
            )

            if not user_id:
                # Use your specific user_id as default
                user_id = "b9284d47-72b5-4960-a177-0788fc4b0809"
                logger.info(
                    "Using default user_id: b9284d47-72b5-4960-a177-0788fc4b0809. "
                    "To override, set HAIVE_USER_ID in your .env file."
                )

        # Check cache first to avoid unnecessary database calls
        cache_key = (
            f"{thread_id}:{user_id if self._schema_type == 'supabase' else 'none'}"
        )
        if cache_key in self._thread_creation_cache:
            return

        try:
            # Handle both connection pool and direct connection
            if hasattr(self.conn, "connection"):
                with self.conn.connection() as conn, conn.cursor() as cursor:
                    self._insert_thread(cursor, thread_id, user_id)
            else:
                with self.conn.cursor() as cursor:
                    self._insert_thread(cursor, thread_id, user_id)

            # Add to cache
            self._thread_creation_cache.add(cache_key)
            if self._schema_type == "supabase":
                logger.debug(
                    f"Thread {thread_id} ensured in database for user {user_id}"
                )
            else:
                logger.debug(f"Thread {thread_id} ensured in database")

        except Exception as e:
            logger.exception(f"Failed to ensure thread {thread_id} exists: {e}")
            raise

    def _insert_thread(self, cursor, thread_id: str, user_id: str = None) -> None:
        """Insert thread using appropriate schema."""
        if self._schema_type == "supabase":
            # Supabase schema with user_id and composite constraint
            cursor.execute(
                """
                INSERT INTO threads (id, user_id, created_at, updated_at, last_access)
                VALUES (%s, %s, NOW(), NOW(), NOW())
                ON CONFLICT (id, user_id) DO UPDATE SET
                    last_access = NOW()
                """,
                (thread_id, user_id),
            )
        else:
            # Standard LangGraph schema without user_id
            cursor.execute(
                """
                INSERT INTO threads (id, created_at, updated_at)
                VALUES (%s, NOW(), NOW())
                ON CONFLICT (id) DO UPDATE SET
                    updated_at = NOW()
                """,
                (thread_id,),
            )

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
        self._schema_type = None  # Will be detected on first use

    def __getattr__(self, name):
        """Delegate attribute access to the base saver."""
        return getattr(self._base_saver, name)

    async def _detect_schema_type(self, cursor) -> str:
        """Detect if using Supabase or standard LangGraph schema (async).

        Returns:
            'supabase' if using Supabase schema with composite constraint
            'standard' if using standard LangGraph schema with single constraint
        """
        try:
            # Check if user_id column exists in threads table
            await cursor.execute(
                """
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'threads' 
                AND column_name = 'user_id'
            """
            )
            has_user_id = await cursor.fetchone() is not None

            if not has_user_id:
                return "standard"

            # Check for composite unique constraint on (id, user_id)
            await cursor.execute(
                """
                SELECT COUNT(*) 
                FROM information_schema.table_constraints tc
                JOIN information_schema.constraint_column_usage AS ccu 
                    USING (constraint_schema, constraint_name)
                WHERE tc.table_name = 'threads' 
                AND tc.constraint_type = 'UNIQUE'
                AND ccu.column_name IN ('id', 'user_id')
                GROUP BY tc.constraint_name
                HAVING COUNT(*) = 2
            """
            )
            has_composite = await cursor.fetchone() is not None

            return "supabase" if has_composite else "standard"

        except Exception as e:
            logger.warning(f"Failed to detect schema type, assuming standard: {e}")
            return "standard"

    async def _ensure_thread_exists(self, thread_id: str, user_id: str = None) -> None:
        """Ensure a thread exists in the threads table (async version).

        Args:
            thread_id: The thread ID to ensure exists
            user_id: Optional user ID. If None, generates/uses a default user_id
        """
        # Detect schema type on first use
        if self._schema_type is None:
            if hasattr(self.conn, "connection"):
                async with self.conn.connection() as conn, conn.cursor() as cursor:
                    self._schema_type = await self._detect_schema_type(cursor)
            else:
                async with self.conn.cursor() as cursor:
                    self._schema_type = await self._detect_schema_type(cursor)

            # Also check environment variable override
            schema_override = os.getenv("POSTGRES_THREADS_SCHEMA_TYPE")
            if schema_override in ["supabase", "standard"]:
                self._schema_type = schema_override

            logger.info(f"Detected PostgreSQL threads schema type: {self._schema_type}")

        # For Supabase schema, we need a user_id
        if self._schema_type == "supabase" and user_id is None:
            # Try environment variable first
            user_id = (
                os.getenv("HAIVE_USER_ID")
                or os.getenv("DEFAULT_USER_ID")
                or os.getenv("USER_ID")
            )

            if not user_id:
                # Use your specific user_id as default
                user_id = "b9284d47-72b5-4960-a177-0788fc4b0809"
                logger.info(
                    "Using default user_id: b9284d47-72b5-4960-a177-0788fc4b0809. "
                    "To override, set HAIVE_USER_ID in your .env file."
                )

        # Check cache first to avoid unnecessary database calls
        cache_key = (
            f"{thread_id}:{user_id if self._schema_type == 'supabase' else 'none'}"
        )
        if cache_key in self._thread_creation_cache:
            return

        try:
            # Handle both connection pool and direct connection
            if hasattr(self.conn, "connection"):
                async with self.conn.connection() as conn, conn.cursor() as cursor:
                    await self._insert_thread(cursor, thread_id, user_id)
            else:
                async with self.conn.cursor() as cursor:
                    await self._insert_thread(cursor, thread_id, user_id)

            # Add to cache
            self._thread_creation_cache.add(cache_key)
            if self._schema_type == "supabase":
                logger.debug(
                    f"Thread {thread_id} ensured in database (async) for user {user_id}"
                )
            else:
                logger.debug(f"Thread {thread_id} ensured in database (async)")

        except Exception as e:
            logger.exception(f"Failed to ensure thread {thread_id} exists (async): {e}")
            raise

    async def _insert_thread(self, cursor, thread_id: str, user_id: str = None) -> None:
        """Insert thread using appropriate schema (async)."""
        if self._schema_type == "supabase":
            # Supabase schema with user_id and composite constraint
            await cursor.execute(
                """
                INSERT INTO threads (id, user_id, created_at, updated_at, last_access)
                VALUES (%s, %s, NOW(), NOW(), NOW())
                ON CONFLICT (id, user_id) DO UPDATE SET
                    last_access = NOW()
                """,
                (thread_id, user_id),
            )
        else:
            # Standard LangGraph schema without user_id
            await cursor.execute(
                """
                INSERT INTO threads (id, created_at, updated_at)
                VALUES (%s, NOW(), NOW())
                ON CONFLICT (id) DO UPDATE SET
                    updated_at = NOW()
                """,
                (thread_id,),
            )

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
