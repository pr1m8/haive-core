"""Haive Core - Foundation for the Haive AI Agent Framework.

This package provides the core building blocks for creating AI agents:

Engine System
-------------
Universal interface for AI components:
- **InvokableEngine**: For LLMs, retrievers, and tools
- **AugLLM**: Enhanced LLM with tools and structured output
- **EngineConfig**: Runtime configuration management

Graph System
------------
Dynamic workflow builder:
- **BaseGraph**: Foundation for graph-based workflows

Schema System
-------------
Intelligent state management:
- **SchemaBuilder**: Auto-generate schemas
- **StateComposer**: Merge and manage states
- **Reducers**: Define state update logic

Persistence
-----------
Conversation and state persistence:
- **PostgreSQL/Supabase**: Auto-persistence support
- **Checkpointers**: Save/restore agent state
- **Thread management**: Conversation continuity

Quick Start
-----------
>>> from haive.core.engine import AugLLMConfig
>>> from haive.core.graph import BaseGraph
>>>
>>> # Create an enhanced LLM
>>> llm = AugLLMConfig(model="gpt-4", temperature=0.7)
>>>
>>> # Build a workflow
>>> graph = BaseGraph()
>>> # Add nodes to graph as needed

See Also:
--------
- haive.agents: Pre-built agent implementations
- haive.tools: Tool library
- haive.core.engine: Engine system documentation
- haive.core.graph: Graph building guide
"""

from haive.core.engine import (
    AugLLMConfig,
    AugLLMFactory,
    Engine,
    InvokableEngine,
    NonInvokableEngine,
)
from haive.core.graph import (
    BaseGraph,
)
from haive.core.registry import (
    DynamicRegistry,
    RegistryItem,
)

# Set version
from haive.core.schema import (  # BasicAgentState,  # Module doesn't exist
    SchemaComposer,
)

__version__ = "0.1.0"

# Suppress noisy HTTP/connection logs
import logging

logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("hpack").setLevel(logging.WARNING)

# Core imports for convenience

# Public API
__all__ = [
    "AugLLMConfig",
    "AugLLMFactory",
    # Graph system
    "BaseGraph",
    # "BasicAgentState",  # Module doesn't exist
    # Registry system
    "DynamicRegistry",
    "RegistryItem",
    # Engine system
    "Engine",
    "InvokableEngine",
    "NonInvokableEngine",
    # Schema system
    "SchemaComposer",
    # Version
    "__version__",
]
