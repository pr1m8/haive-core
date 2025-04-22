# src/haive/core/engine/agent/persistence/__init__.py

"""
Agent persistence module for storing and retrieving agent state.

This module provides configurations for different types of persistence
backends that can be used to store agent state, checkpoints, and other
data across agent runs.
"""
from typing import Dict, Optional, Any, Type, Union

from .types import CheckpointerType, CheckpointMetadata
from .base import CheckpointerConfig
from .memory import MemoryCheckpointerConfig
from .sqlite_config import SQLiteCheckpointerConfig
from .handlers import setup_checkpointer, prepare_merged_input, process_input

# Conditionally import PostgreSQL support
try:
    from .postgres_config import PostgresCheckpointerConfig
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

# Conditionally import Supabase support
try:
    from .supabase_config import SupabaseCheckpointerConfig
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False

def create_memory_checkpointer() -> MemoryCheckpointerConfig:
    """
    Create a memory-based checkpointer configuration.
    
    Returns:
        A configured MemoryCheckpointerConfig
    """
    return MemoryCheckpointerConfig()

def create_sqlite_checkpointer(db_path: str = "./checkpoints.db") -> SQLiteCheckpointerConfig:
    """
    Create a SQLite-based checkpointer configuration.
    
    Args:
        db_path: Path to the SQLite database file
        
    Returns:
        A configured SQLiteCheckpointerConfig
    """
    return SQLiteCheckpointerConfig(db_path=db_path)

def create_postgres_checkpointer(
    db_host: str = "localhost",
    db_port: int = 5432,
    db_name: str = "postgres",
    db_user: str = "postgres",
    db_pass: str = "postgres",
    ssl_mode: str = "disable",
    **kwargs
) -> CheckpointerConfig:
    """
    Create a PostgreSQL-based checkpointer configuration.
    
    Args:
        db_host: PostgreSQL host
        db_port: PostgreSQL port
        db_name: PostgreSQL database name
        db_user: PostgreSQL username
        db_pass: PostgreSQL password
        ssl_mode: SSL mode for connection
        **kwargs: Additional PostgreSQL connection options
        
    Returns:
        A configured PostgresCheckpointerConfig if dependencies are available,
        otherwise falls back to a MemoryCheckpointerConfig
    """
    if POSTGRES_AVAILABLE:
        from .postgres_config import PostgresCheckpointerConfig
        return PostgresCheckpointerConfig(
            db_host=db_host,
            db_port=db_port,
            db_name=db_name,
            db_user=db_user,
            db_pass=db_pass,
            ssl_mode=ssl_mode,
            **kwargs
        )
    else:
        import logging
        logging.getLogger(__name__).warning(
            "PostgreSQL checkpointer dependencies not available, falling back to memory checkpointer. "
            "Install with: pip install psycopg[binary] langgraph[postgres]"
        )
        return create_memory_checkpointer()

def create_supabase_checkpointer(
    supabase_url: Optional[str] = None,
    supabase_key: Optional[str] = None,
    user_id: Optional[str] = None,
    setup_needed: bool = True,
    **kwargs
) -> CheckpointerConfig:
    """
    Create a Supabase-based checkpointer configuration.
    
    Args:
        supabase_url: Supabase project URL (optional if using dataflow client)
        supabase_key: Supabase API key (optional if using dataflow client)
        user_id: Optional user ID for RLS policies
        setup_needed: Whether to initialize database schema
        **kwargs: Additional configuration parameters
        
    Returns:
        A configured SupabaseCheckpointerConfig if dependencies are available,
        otherwise falls back to a MemoryCheckpointerConfig
    """
    if SUPABASE_AVAILABLE:
        from .supabase_config import SupabaseCheckpointerConfig
        return SupabaseCheckpointerConfig(
            supabase_url=supabase_url,
            supabase_key=supabase_key,
            user_id=user_id,
            setup_needed=setup_needed,
            **kwargs
        )
    else:
        import logging
        logging.getLogger(__name__).warning(
            "Supabase dependencies not available, falling back to memory checkpointer. "
            "Install with: pip install supabase"
        )
        return create_memory_checkpointer()

def create_checkpointer(
    persistence_type: Union[CheckpointerType, str],
    **kwargs
) -> CheckpointerConfig:
    """
    Create a checkpointer of the specified type with given parameters.
    
    Args:
        persistence_type: Type of checkpointer to create
        **kwargs: Parameters specific to the checkpointer type
        
    Returns:
        A configured CheckpointerConfig of the requested type
    """
    # Convert string to enum if needed
    if isinstance(persistence_type, str):
        try:
            persistence_type = CheckpointerType(persistence_type)
        except ValueError:
            import logging
            logging.getLogger(__name__).warning(
                f"Unknown persistence type: {persistence_type}, falling back to memory checkpointer."
            )
            return create_memory_checkpointer()
    
    if persistence_type == CheckpointerType.memory:
        return create_memory_checkpointer()
    
    elif persistence_type == CheckpointerType.sqlite:
        # Extract SQLite-specific kwargs
        db_path = kwargs.get("db_path", "./checkpoints.db")
        return create_sqlite_checkpointer(db_path)
    
    elif persistence_type == CheckpointerType.postgres:
        # Extract PostgreSQL-specific kwargs
        pg_kwargs = {
            "db_host": kwargs.get("db_host", "localhost"),
            "db_port": kwargs.get("db_port", 5432),
            "db_name": kwargs.get("db_name", "postgres"),
            "db_user": kwargs.get("db_user", "postgres"),
            "db_pass": kwargs.get("db_pass", "postgres"),
            "ssl_mode": kwargs.get("ssl_mode", "disable"),
            "min_pool_size": kwargs.get("min_pool_size", 1),
            "max_pool_size": kwargs.get("max_pool_size", 5),
            "auto_commit": kwargs.get("auto_commit", True),
            "prepare_threshold": kwargs.get("prepare_threshold", 0)
        }
        return create_postgres_checkpointer(**pg_kwargs)
    
    elif persistence_type == CheckpointerType.supabase:
        # Extract Supabase-specific kwargs
        supabase_kwargs = {
            "supabase_url": kwargs.get("supabase_url"),
            "supabase_key": kwargs.get("supabase_key"),
            "user_id": kwargs.get("user_id"),
            "setup_needed": kwargs.get("setup_needed", True)
        }
        return create_supabase_checkpointer(**supabase_kwargs)
    
    else:
        import logging
        logging.getLogger(__name__).warning(
            f"Unknown checkpointer type: {persistence_type}, falling back to memory checkpointer."
        )
        return create_memory_checkpointer()

# Mapping of persistence types to their config classes
CHECKPOINTER_CONFIG_CLASSES: Dict[CheckpointerType, Type[CheckpointerConfig]] = {
    CheckpointerType.memory: MemoryCheckpointerConfig,
    CheckpointerType.sqlite: SQLiteCheckpointerConfig,
}

# Add PostgreSQL and Supabase if available
if POSTGRES_AVAILABLE:
    CHECKPOINTER_CONFIG_CLASSES[CheckpointerType.postgres] = PostgresCheckpointerConfig

if SUPABASE_AVAILABLE:
    CHECKPOINTER_CONFIG_CLASSES[CheckpointerType.supabase] = SupabaseCheckpointerConfig

# Export available checkpointer types and base classes
__all__ = [
    'CheckpointerType',
    'CheckpointerConfig',
    'CheckpointMetadata',
    'MemoryCheckpointerConfig',
    'SQLiteCheckpointerConfig',
    'create_memory_checkpointer',
    'create_sqlite_checkpointer',
    'create_postgres_checkpointer',
    'create_supabase_checkpointer',
    'create_checkpointer',
    'setup_checkpointer',
    'prepare_merged_input',
    'process_input',
    'CHECKPOINTER_CONFIG_CLASSES'
]

# Add PostgresCheckpointerConfig to exports if available
if POSTGRES_AVAILABLE:
    __all__.append('PostgresCheckpointerConfig')

# Add SupabaseCheckpointerConfig to exports if available
if SUPABASE_AVAILABLE:
    __all__.append('SupabaseCheckpointerConfig')