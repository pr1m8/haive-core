from typing import Optional, Dict, Any, Union, List, Type
from pydantic import BaseModel, Field, SecretStr
import logging

from haive.core.persistence.base import CheckpointerConfig
from haive.core.persistence.types import (
    CheckpointerType, 
    CheckpointerMode,
    CheckpointStorageMode, 
    ConnectionOptions
)

logger = logging.getLogger(__name__)

class PostgresCheckpointerConfig(CheckpointerConfig[Dict[str, Any]]):
    """
    Configuration for PostgreSQL-based checkpoint persistence.
    
    This implementation provides all necessary configuration options
    for connecting to and using a PostgreSQL database for agent state
    persistence.
    """
    type: CheckpointerType = CheckpointerType.POSTGRES
    
    # Database connection parameters
    db_host: str = Field(
        default="localhost",
        description="PostgreSQL server hostname"
    )
    db_port: int = Field(
        default=5432,
        description="PostgreSQL server port"
    )
    db_name: str = Field(
        default="postgres",
        description="Database name"
    )
    db_user: str = Field(
        default="postgres",
        description="Database username"
    )
    db_pass: SecretStr = Field(
        default_factory=lambda: SecretStr("postgres"),
        description="Database password"
    )
    ssl_mode: str = Field(
        default="disable",
        description="SSL mode for database connection"
    )
    
    # Connection pool configuration
    min_pool_size: int = Field(
        default=1,
        description="Minimum number of connections in the pool"
    )
    max_pool_size: int = Field(
        default=5,
        description="Maximum number of connections in the pool"
    )
    
    # Additional connection options
    auto_commit: bool = Field(
        default=True,
        description="Auto-commit transactions"
    )
    prepare_threshold: int = Field(
        default=0,
        description="Prepared statement threshold"
    )
    connection_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional connection keyword arguments"
    )
    
    def get_connection_uri(self) -> str:
        """
        Generate a connection URI for PostgreSQL.
        
        Returns:
            String connection URI
        """
        import urllib.parse
        encoded_pass = urllib.parse.quote_plus(self.db_pass.get_secret_value())
        db_uri = f"postgresql://{self.db_user}:{encoded_pass}@{self.db_host}:{self.db_port}/{self.db_name}"
        if self.ssl_mode:
            db_uri += f"?sslmode={self.ssl_mode}"
        return db_uri
    
    def get_connection_kwargs(self) -> Dict[str, Any]:
        """
        Get connection keyword arguments.
        
        Returns:
            Dictionary of connection options
        """
        kwargs = {
            "autocommit": self.auto_commit,
            "prepare_threshold": self.prepare_threshold,
        }
        # Add any additional kwargs
        kwargs.update(self.connection_kwargs)
        return kwargs
    
    def create_checkpointer(self) -> Any:
        """
        Create a synchronous PostgreSQL checkpointer.
        """
        try:
            from psycopg_pool import ConnectionPool
            from langgraph.checkpoint.postgres import PostgresSaver
            
            # Create connection pool
            pool = ConnectionPool(
                conninfo=self.get_connection_uri(),
                min_size=self.min_pool_size,
                max_size=self.max_pool_size,
                kwargs=self.get_connection_kwargs()
            )
            
            # Explicitly open the pool
            pool.open()
            
            # Create checkpointer
            checkpointer = PostgresSaver(pool)
            
            # Setup tables if needed
            if self.setup_needed:
                try:
                    checkpointer.setup()
                    logger.info("PostgreSQL tables set up successfully")
                except Exception as e:
                    logger.warning(f"Error during PostgreSQL setup: {e}")
            
            return checkpointer
            
        except Exception as e:
            logger.error(f"Failed to create PostgreSQL checkpointer: {e}")
            raise RuntimeError(f"Failed to create PostgreSQL checkpointer: {e}")
            raise RuntimeError(f"Failed to create PostgreSQL checkpointer: {e}")
    def create_async_checkpointer(self) -> Any:
        """
        Create an asynchronous PostgreSQL checkpointer with automatic resource management.
        
        Returns an async context manager that handles pool lifecycle.
        """
        try:
            from psycopg_pool import AsyncConnectionPool
            from contextlib import asynccontextmanager
            
            # Import appropriate checkpointer class
            try:
                if self.storage_mode == CheckpointStorageMode.SHALLOW:
                    from langgraph.checkpoint.postgres.aio import AsyncShallowPostgresSaver as AsyncPostgresSaver
                else:
                    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
            except ImportError:
                logger.error("langgraph-checkpoint-postgres not installed.")
                raise ImportError("Please install langgraph-checkpoint-postgres")
            
            @asynccontextmanager
            async def async_checkpointer_context():
                """Context manager that handles pool lifecycle."""
                # Create pool without opening
                pool = AsyncConnectionPool(
                    conninfo=self.get_connection_uri(),
                    min_size=self.min_pool_size,
                    max_size=self.max_pool_size,
                    kwargs=self.get_connection_kwargs(),
                    open=False  # Don't open in constructor to avoid deprecation warning
                )
                
                try:
                    # Explicitly open the pool
                    await pool.open()
                    
                    # Create checkpointer
                    checkpointer = AsyncPostgresSaver(pool)
                    
                    # Setup tables if needed
                    if self.setup_needed:
                        try:
                            await checkpointer.setup()
                            logger.info("PostgreSQL tables set up successfully (async)")
                        except Exception as e:
                            logger.warning(f"Error during PostgreSQL async setup: {e}")
                    
                    # Yield just the checkpointer for use
                    yield checkpointer
                    
                finally:
                    # Always clean up the pool
                    if hasattr(pool, '_opened') and pool._opened:
                        try:
                            await pool.close()
                            logger.debug("PostgreSQL pool closed successfully")
                        except Exception as e:
                            logger.warning(f"Error closing PostgreSQL pool: {e}")
            
            return async_checkpointer_context
            
        except Exception as e:
            logger.error(f"Failed to create async PostgreSQL checkpointer: {e}")
            raise RuntimeError(f"Failed to create async PostgreSQL checkpointer: {e}")
