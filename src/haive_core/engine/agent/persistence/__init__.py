# src/haive/core/engine/agent/persistence/__init__.py

"""
Persistence package for agent state management.

This package provides tools and utilities for persisting agent state
across sessions, with support for different storage backends.
"""

from haive_core.engine.agent.persistence.types import CheckpointerType
from haive_core.engine.agent.persistence.base import CheckpointerConfig
from haive_core.engine.agent.persistence.memory_config import MemoryCheckpointerConfig

# Conditionally import PostgreSQL components
try:
    from haive_core.engine.agent.persistence.postgres_config import PostgresCheckpointerConfig
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

# Conditionally import MongoDB components
try:
    from haive_core.engine.agent.persistence.mongodb_config import MongoDBCheckpointerConfig
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False

# Import factory
from haive_core.engine.agent.persistence.factory import (
    load_checkpointer_config,
    CHECKPOINTER_CONFIG_MAP
)

# Update the checkpointer config map based on available implementations
if not POSTGRES_AVAILABLE:
    # Remove PostgreSQL from the config map
    CHECKPOINTER_CONFIG_MAP.pop(CheckpointerType.postgres, None)

if not MONGODB_AVAILABLE:
    # Remove MongoDB from the config map 
    CHECKPOINTER_CONFIG_MAP.pop(CheckpointerType.mongodb, None)

__all__ = [
    'CheckpointerType',
    'CheckpointerConfig',
    'MemoryCheckpointerConfig',
    'load_checkpointer_config',
]

# Add optional components to __all__ if available
if POSTGRES_AVAILABLE:
    __all__.append('PostgresCheckpointerConfig')

if MONGODB_AVAILABLE:
    __all__.append('MongoDBCheckpointerConfig')