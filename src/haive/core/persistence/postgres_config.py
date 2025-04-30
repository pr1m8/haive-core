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
    
    # Optional direct connection string
    connection_string: Optional[str] = Field(
        default=None,
        description="Direct connection string (overrides individual parameters)"
    )
    
    # Pipeline mode
    use_pipeline: bool = Field(
        default=False,
        description="Whether to use pipeline mode for better performance"
    )
    
    class Config:
        arbitrary_types_allowed = True
    
    def is_async_mode(self) -> bool:
        """
        Check if operating in async mode.
        
        Returns:
            True if async mode, False otherwise
        """
        return self.mode == CheckpointerMode.ASYNC
    
    def get_connection_uri(self) -> str:
        """
        Generate a connection URI for PostgreSQL.
        
        Returns:
            String connection URI
        """
        # Use direct connection string if provided
        if self.connection_string:
            return self.connection_string
            
        # Generate from individual parameters
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
        
        Returns:
            PostgresSaver instance
        """
        try:
            # Handle async mode request
            if self.is_async_mode():
                raise RuntimeError("Cannot use create_checkpointer for async mode, use create_async_checkpointer instead")
                
            from psycopg_pool import ConnectionPool
            
            # Import appropriate checkpointer class
            if self.storage_mode == CheckpointStorageMode.SHALLOW:
                try:
                    from langgraph.checkpoint.postgres import ShallowPostgresSaver as PostgresSaver
                except ImportError:
                    from langgraph.checkpoint.postgres import PostgresSaver
            else:
                from langgraph.checkpoint.postgres import PostgresSaver
            
            # Create connection pool
            pool = ConnectionPool(
                conninfo=self.get_connection_uri(),
                min_size=self.min_pool_size,
                max_size=self.max_pool_size,
                kwargs=self.get_connection_kwargs(),
                open=True  # Explicitly open the pool
            )
            
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
    
    async def create_async_checkpointer(self) -> Any:
        """
        Create an asynchronous PostgreSQL checkpointer.
        
        Returns:
            Async PostgreSQL checkpointer
        """
        try:
            # Force async mode
            self.mode = CheckpointerMode.ASYNC
            
            from psycopg_pool import AsyncConnectionPool
            
            # Import appropriate checkpointer class
            try:
                if self.storage_mode == CheckpointStorageMode.SHALLOW:
                    from langgraph.checkpoint.postgres.aio import AsyncShallowPostgresSaver as AsyncPostgresSaver
                else:
                    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
            except ImportError:
                try:
                    # Fall back to langgraph.checkpoint.postgres if aio module not available
                    from langgraph.checkpoint.postgres import AsyncPostgresSaver
                except ImportError:
                    logger.error("AsyncPostgresSaver not available. Please ensure langgraph-checkpoint-postgres is installed.")
                    raise ImportError("AsyncPostgresSaver not available")
            
            # Create connection pool
            pool = AsyncConnectionPool(
                conninfo=self.get_connection_uri(),
                min_size=self.min_pool_size,
                max_size=self.max_pool_size,
                kwargs=self.get_connection_kwargs(),
                open=False  # Don't open in constructor to avoid deprecation warning
            )
            
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
            
            return checkpointer
            
        except Exception as e:
            logger.error(f"Failed to create async PostgreSQL checkpointer: {e}")
            raise RuntimeError(f"Failed to create async PostgreSQL checkpointer: {e}")
    
    async def initialize_async_checkpointer(self) -> Any:
        """
        Initialize an async checkpointer context manager.
        
        Returns:
            Async context manager for checkpointer
        """
        from contextlib import asynccontextmanager
        
        @asynccontextmanager
        async def async_checkpointer_context():
            """Context manager for async checkpointer with proper resource management."""
            checkpointer = None
            try:
                # Create the checkpointer
                checkpointer = await self.create_async_checkpointer()
                
                # Yield it for use
                yield checkpointer
                
            finally:
                # Clean up resources
                if checkpointer and hasattr(checkpointer, 'conn') and checkpointer.conn:
                    # Close pool if available
                    try:
                        if hasattr(checkpointer.conn, 'close'):
                            await checkpointer.conn.close()
                            logger.debug("Async PostgreSQL pool closed successfully")
                    except Exception as e:
                        logger.warning(f"Error closing async PostgreSQL pool: {e}")
        
        # Return the context manager
        return async_checkpointer_context()