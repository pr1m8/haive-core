# src/haive/core/engine/agent/persistence/types.py

from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, Union

class CheckpointerType(str, Enum):
    """
    Types of supported checkpointer implementations.
    
    This enum defines the available persistence backends for agent state.
    """
    memory = "memory"
    postgres = "postgres"
    sqlite = "sqlite"
    supabase = "supabase"
    # Add other checkpointer types as needed

class CheckpointMetadata(Dict[str, Any]):
    """Type alias for checkpoint metadata dictionaries."""
    pass

class CheckpointFailedCallback(Protocol):
    """
    Protocol for callbacks when checkpoint operations fail.
    
    This can be used for monitoring, logging, or taking corrective action.
    """
    def __call__(self, ex: Exception, config: Dict[str, Any]) -> None:
        """
        Called when a checkpoint operation fails.
        
        Args:
            ex: The exception that occurred
            config: The configuration that was being used
        """
        ...

class AsyncConnectFailedCB(Protocol):
    """Protocol for async connection failed callbacks in PostgreSQL."""
    def __call__(self, pool: Any) -> Any:
        """
        Called when connections cannot be established.
        
        Args:
            pool: The connection pool that failed
        """
        ...

class AsyncConnectionCB(Protocol):
    """Protocol for async connection callbacks in PostgreSQL."""
    def __call__(self, conn: Any) -> Any:
        """
        Called to configure or check connections.
        
        Args:
            conn: The connection to configure or check
        """
        ...