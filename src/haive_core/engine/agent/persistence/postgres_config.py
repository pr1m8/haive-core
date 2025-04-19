# src/haive/core/engine/agent/persistence/postgres_config.py

import logging
import urllib.parse
from typing import Any, Literal

from pydantic import Field

from .base import CheckpointerConfig
from .types import CheckpointerType

# Check if PostgreSQL dependencies are installed
try:
    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg_pool import ConnectionPool
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

logger = logging.getLogger(__name__)

class PostgresCheckpointerConfig(CheckpointerConfig):
    """Configuration for PostgreSQL-based checkpointing.
    
    This class handles creation and configuration of a PostgreSQL-based
    checkpointer for LangGraph agents, with thread registration and pool
    management.
    """
    type: Literal[CheckpointerType.postgres] = CheckpointerType.postgres

    # Database connection settings
    db_host: str = Field(default="localhost", description="PostgreSQL host")
    db_port: int = Field(default=5432, description="PostgreSQL port")
    db_name: str = Field(default="postgres", description="PostgreSQL database name")
    db_user: str = Field(default="postgres", description="PostgreSQL username")
    db_pass: str = Field(default="postgres", description="PostgreSQL password")
    ssl_mode: str = Field(default="disable", description="SSL mode for connection")

    # Pool settings
    min_pool_size: int = Field(default=1, description="Minimum pool size")
    max_pool_size: int = Field(default=5, description="Maximum pool size")
    auto_commit: bool = Field(default=True, description="Auto-commit transactions")
    prepare_threshold: int = Field(default=0, description="Prepare threshold")

    # Runtime settings
    setup_needed: bool = Field(default=True, description="Whether to initialize DB tables")
    use_async: bool = Field(default=False, description="Whether to use async mode")

    # Internal state (not serialized)
    _pool: Any | None = None
    _checkpointer: Any | None = None

    def create_checkpointer(self) -> Any:
        """Create a PostgreSQL checkpointer with the specified configuration.
        
        Returns:
            A PostgresSaver instance for use with LangGraph
        """
        if not POSTGRES_AVAILABLE:
            logger.warning("PostgreSQL dependencies not available, falling back to memory checkpointer")
            from langgraph.checkpoint.memory import MemorySaver
            return MemorySaver()

        try:
            # Create or reuse connection pool
            if self._pool is None:
                logger.info("Creating new PostgreSQL connection pool")
                self._pool = self._create_connection_pool()

            # Create PostgresSaver with the pool
            self._checkpointer = PostgresSaver(self._pool)

            # Initialize tables if needed
            if self.setup_needed:
                try:
                    self._checkpointer.setup()
                    logger.info("Successfully set up PostgreSQL tables")
                    self.setup_needed = False
                except Exception as e:
                    logger.warning(f"Error setting up PostgreSQL tables: {e}")

            return self._checkpointer

        except Exception as e:
            logger.error(f"Error creating PostgreSQL checkpointer: {e}")
            logger.warning("Falling back to memory checkpointer")
            from langgraph.checkpoint.memory import MemorySaver
            return MemorySaver()

    def _create_connection_pool(self) -> ConnectionPool:
        """Create and configure a new connection pool"""
        # Build connection URI
        encoded_pass = urllib.parse.quote_plus(str(self.db_pass))
        db_uri = f"postgresql://{self.db_user}:{encoded_pass}@{self.db_host}:{self.db_port}/{self.db_name}"
        if self.ssl_mode:
            db_uri += f"?sslmode={self.ssl_mode}"

        # Create pool
        pool = ConnectionPool(
            conninfo=db_uri,
            min_size=self.min_pool_size,
            max_size=self.max_pool_size,
            kwargs={
                "autocommit": self.auto_commit,
                "prepare_threshold": self.prepare_threshold,
            }
        )

        # Don't open the pool yet - we'll do that on demand
        return pool

    def register_thread(self, thread_id: str) -> None:
        """Register a thread in the PostgreSQL database.
        
        This ensures that the thread exists in the database before
        any checkpoints are created that reference it.
        
        Args:
            thread_id: The thread ID to register
        """
        if not POSTGRES_AVAILABLE or self._pool is None:
            logger.debug("PostgreSQL not available or pool not initialized")
            return

        # Ensure connection is open
        self.ensure_connection()

        try:
            # Register thread - only using thread_id and created_at
            with self._pool.connection() as conn:
                with conn.cursor() as cursor:
                    # Create threads table if needed with just thread_id and created_at
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS threads (
                            thread_id VARCHAR(255) PRIMARY KEY,
                            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                        );
                    """)

                    # Simple insert - no additional fields or updates
                    cursor.execute("""
                        INSERT INTO threads (thread_id) 
                        VALUES (%s) 
                        ON CONFLICT (thread_id) 
                        DO NOTHING
                    """, (thread_id,))

                    logger.debug(f"Thread {thread_id} registered in PostgreSQL")
        except Exception as e:
            logger.warning(f"Error registering thread: {e}")

    def ensure_connection(self) -> bool:
        """Ensure the database connection is open and ready.
        
        Returns:
            True if the connection needed to be opened, False otherwise
        """
        if not POSTGRES_AVAILABLE:
            return False

        if self._pool is None:
            try:
                self._pool = self._create_connection_pool()
                logger.debug("Created new connection pool")
                opened = True
            except Exception as e:
                logger.error(f"Failed to create connection pool: {e}")
                return False
        else:
            opened = False

        # Check if pool is open
        try:
            if hasattr(self._pool, "is_open"):
                if not self._pool.is_open():
                    self._pool.open()
                    logger.debug("Opened existing connection pool")
                    opened = True
            elif hasattr(self._pool, "_opened") and not self._pool._opened:
                # For older psycopg_pool versions
                self._pool._pool = [] if not hasattr(self._pool, "_pool") or self._pool._pool is None else self._pool._pool
                self._pool._opened = True
                logger.debug("Initialized pool for older psycopg_pool version")
                opened = True
        except Exception as e:
            logger.warning(f"Error ensuring connection: {e}")
            return False

        return opened

    def release_connection(self, close: bool = False) -> None:
        """Release a database connection.
        
        Args:
            close: Whether to actually close the connection
        """
        if self._pool is None:
            return

        if close:
            try:
                if hasattr(self._pool, "is_open") and self._pool.is_open():
                    self._pool.close()
                    logger.debug("Closed connection pool")
            except Exception as e:
                logger.warning(f"Error closing connection: {e}")
        else:
            # Just log that we're conceptually releasing it
            logger.debug("Connection pool released (not actually closed)")

    def to_dict(self) -> dict[str, Any]:
        """Convert to a serializable dictionary"""
        if hasattr(self, "model_dump"):
            # Pydantic v2
            data = self.model_dump(exclude={"_pool", "_checkpointer"})
        else:
            # Pydantic v1
            data = self.dict(exclude={"_pool", "_checkpointer"})
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PostgresCheckpointerConfig":
        """Create from a dictionary"""
        # Remove any internal fields that shouldn't be part of initialization
        for field in ["_pool", "_checkpointer"]:
            data.pop(field, None)
        return cls(**data)
