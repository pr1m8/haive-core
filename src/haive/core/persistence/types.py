from enum import Enum
from typing import Optional, Dict, Any, Union, Literal

class CheckpointerType(str, Enum):
    """Types of checkpointers supported by the system."""
    MEMORY = "memory"
    POSTGRES = "postgres"
    SQLITE = "sqlite"  # For future expansion

class CheckpointerMode(str, Enum):
    """Operational modes for checkpointers."""
    SYNC = "sync"
    ASYNC = "async"

class CheckpointStorageMode(str, Enum):
    """Storage modes for checkpoints."""
    FULL = "full"      # Store complete history
    SHALLOW = "shallow"  # Store only the most recent checkpoint

# Type aliases
ConnectionOptions = Dict[str, Any]
ThreadMetadata = Dict[str, Any]