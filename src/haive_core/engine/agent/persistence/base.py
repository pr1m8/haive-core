# src/haive/core/engine/agent/persistence/base.py

"""
Base classes and types for agent persistence.

This module defines the configuration models for different
persistence options for agent state checkpointing.
"""

import logging
from typing import Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from enum import Enum

logger = logging.getLogger(__name__)

class CheckpointerType(str, Enum):
    """Types of checkpoint storage available."""
    memory = "memory"
    postgres = "postgres"
    mongodb = "mongodb"
    filesystem = "filesystem"

class CheckpointerConfig(BaseModel):
    """
    Base configuration for agent persistence.
    
    CheckpointerConfig provides a consistent interface for configuring
    how agent state is persisted.
    """
    type: CheckpointerType = Field(default=CheckpointerType.memory)
    setup_needed: bool = Field(default=True, description="Whether to run setup on initialization")
    
    class Config:
        arbitrary_types_allowed = True
    
    def create_checkpointer(self):
        """
        Create a checkpointer instance based on this configuration.
        
        Returns:
            The appropriate checkpointer instance
        """
        from langgraph.checkpoint.memory import MemorySaver
        return MemorySaver()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to a dictionary for serialization.
        
        Returns:
            Dictionary representation
        """
        if hasattr(self, "model_dump"):
            # Pydantic v2
            return self.model_dump()
        else:
            # Pydantic v1
            return self.dict()

class MemoryCheckpointerConfig(CheckpointerConfig):
    """
    Configuration for in-memory checkpointing.
    
    This is the simplest checkpointer, but doesn't persist across sessions.
    """
    type: CheckpointerType = Field(default=CheckpointerType.memory)
    
    def create_checkpointer(self):
        """
        Create a memory checkpointer.
        
        Returns:
            MemorySaver instance
        """
        from langgraph.checkpoint.memory import MemorySaver
        return MemorySaver()

class PostgresCheckpointerConfig(CheckpointerConfig):
    """
    Configuration for PostgreSQL-based checkpointing.
    
    Allows for persistent storage of agent state in a PostgreSQL database.
    """
    type: CheckpointerType = Field(default=CheckpointerType.postgres)
    db_host: str = Field(default="localhost", description="Database host")
    db_port: int = Field(default=5432, description="Database port")
    db_name: str = Field(default="postgres", description="Database name")
    db_user: str = Field(default="postgres", description="Database user")
    db_pass: str = Field(default="postgres", description="Database password")
    ssl_mode: str = Field(default="disable", description="SSL mode for connection")
    min_pool_size: int = Field(default=1, description="Minimum connection pool size")
    max_pool_size: int = Field(default=5, description="Maximum connection pool size")
    auto_commit: bool = Field(default=True, description="Auto-commit transactions")
    prepare_threshold: int = Field(default=0, description="Prepared statement threshold")
    setup_needed: bool = Field(default=True, description="Run setup on first connection")
    use_async: bool = Field(default=False, description="Use async connections")
    
    @property
    def is_postgres(self) -> bool:
        """Flag to easily identify postgres configuraton."""
        return True
    
    def create_checkpointer(self):
        """
        Create a PostgreSQL checkpointer.
        
        Returns:
            PostgresSaver instance
        """
        try:
            # Check if the necessary packages are available
            from langgraph.checkpoint.postgres import PostgresSaver
            from psycopg_pool import ConnectionPool
            
            # Create connection URI
            import urllib.parse
            encoded_pass = urllib.parse.quote_plus(str(self.db_pass))
            db_uri = f"postgresql://{self.db_user}:{encoded_pass}@{self.db_host}:{self.db_port}/{self.db_name}"
            if self.ssl_mode:
                db_uri += f"?sslmode={self.ssl_mode}"
            
            # Create connection pool
            pool = ConnectionPool(
                conninfo=db_uri,
                min_size=self.min_pool_size,
                max_size=self.max_pool_size,
                kwargs={
                    "autocommit": self.auto_commit,
                    "prepare_threshold": self.prepare_threshold,
                },
                open=True  # Explicitly open the pool
            )
            
            # Create and return checkpointer
            return PostgresSaver(pool)
            
        except ImportError as e:
            logger.error(f"PostgreSQL dependencies not available: {e}")
            logger.warning("Falling back to memory checkpointer")
            from langgraph.checkpoint.memory import MemorySaver
            return MemorySaver()
        except Exception as e:
            logger.error(f"Error creating PostgreSQL checkpointer: {e}")
            logger.warning("Falling back to memory checkpointer")
            from langgraph.checkpoint.memory import MemorySaver
            return MemorySaver()

class MongoDBCheckpointerConfig(CheckpointerConfig):
    """
    Configuration for MongoDB-based checkpointing.
    
    Allows for persistent storage of agent state in a MongoDB database.
    """
    type: CheckpointerType = Field(default=CheckpointerType.mongodb)
    connection_string: str = Field(default="mongodb://localhost:27017/")
    database: str = Field(default="agents")
    collection: str = Field(default="checkpoints")
    
    def create_checkpointer(self):
        """
        Create a MongoDB checkpointer.
        
        Note: This is a placeholder since LangGraph doesn't provide
        a MongoDB checkpointer out of the box.
        
        Returns:
            MemorySaver fallback
        """
        logger.warning("MongoDB checkpointer not implemented, falling back to memory")
        from langgraph.checkpoint.memory import MemorySaver
        return MemorySaver()

class FilesystemCheckpointerConfig(CheckpointerConfig):
    """
    Configuration for filesystem-based checkpointing.
    
    Allows for persistent storage of agent state in the filesystem.
    """
    type: CheckpointerType = Field(default=CheckpointerType.filesystem)
    directory: str = Field(default="./checkpoints")
    
    def create_checkpointer(self):
        """
        Create a filesystem checkpointer.
        
        Note: This is a placeholder since LangGraph doesn't provide
        a filesystem checkpointer out of the box.
        
        Returns:
            MemorySaver fallback
        """
        logger.warning("Filesystem checkpointer not implemented, falling back to memory")
        from langgraph.checkpoint.memory import MemorySaver
        return MemorySaver()

def load_checkpointer_config(data: Dict[str, Any]) -> CheckpointerConfig:
    """
    Create a checkpointer configuration from a dictionary.
    
    Args:
        data: Dictionary representation of a checkpointer configuration
        
    Returns:
        The appropriate CheckpointerConfig instance
    """
    checkpoint_type = data.get("type", "memory")
    
    if checkpoint_type == CheckpointerType.postgres:
        return PostgresCheckpointerConfig(**data)
    elif checkpoint_type == CheckpointerType.mongodb:
        return MongoDBCheckpointerConfig(**data)
    elif checkpoint_type == CheckpointerType.filesystem:
        return FilesystemCheckpointerConfig(**data)
    else:
        return MemoryCheckpointerConfig(**data)