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
---------
- haive.agents: Pre-built agent implementations
- haive.tools: Tool library
- haive.core.engine: Engine system documentation
- haive.core.graph: Graph building guide
"""

# ============================================================================
# PERFORMANCE OPTIMIZATIONS - Applied automatically on import
# ============================================================================
import logging
import os

# ============================================================================
# Lazy Core Imports - Performance Optimization
# ============================================================================

# Define lazy import mappings to avoid heavy loading at import time
_CORE_IMPORTS = {
    # Engine components (heaviest - lazy load these)
    "AugLLMConfig": ("haive.core.engine", "AugLLMConfig"),
    "AugLLMFactory": ("haive.core.engine", "AugLLMFactory"),
    "Engine": ("haive.core.engine", "Engine"),
    "InvokableEngine": ("haive.core.engine", "InvokableEngine"),
    "NonInvokableEngine": ("haive.core.engine", "NonInvokableEngine"),
    # Graph and other components (lighter weight)
    "BaseGraph": ("haive.core.graph", "BaseGraph"),
    "DynamicRegistry": ("haive.core.registry", "DynamicRegistry"),
    "RegistryItem": ("haive.core.registry", "RegistryItem"),
    "SchemaComposer": ("haive.core.schema", "SchemaComposer"),
}


def __getattr__(name: str):
    """Lazy load core components to avoid heavy import overhead."""
    if name in _CORE_IMPORTS:
        module_path, class_name = _CORE_IMPORTS[name]

        # Import module and get class only when accessed
        import importlib

        module = importlib.import_module(module_path)
        component = getattr(module, class_name)

        # Cache in globals for subsequent access
        globals()[name] = component
        return component

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# Configure reduced logging verbosity
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

# Suppress the noisiest loggers that cause slow startup
_noisy_loggers = [
    "haive.dataflow.registry.core",
    "haive.core.engine.document.loaders.strategy",
    "haive.core.engine.document.loaders.auto_registry",
    "httpcore.connection",
    "httpcore.http2",
    "hpack.hpack",
    "urllib3",
    "requests",
    "httpx",
    "supabase",
    "google.cloud.storage",
    "langchain_community.utils.user_agent",
]

for _logger_name in _noisy_loggers:
    logging.getLogger(_logger_name).setLevel(logging.ERROR)

# Set environment variables to skip heavy initialization
os.environ.setdefault("HAIVE_SKIP_HEAVY_INIT", "1")
os.environ.setdefault("HAIVE_QUIET_IMPORTS", "1")
os.environ.setdefault("HAIVE_DEBUG_CONFIG", "false")

__version__ = "0.1.0"

# Core imports for convenience

# Public API
__all__ = [
    "AugLLMConfig",
    "AugLLMFactory",
    "BaseGraph",
    "DynamicRegistry",
    "Engine",
    "InvokableEngine",
    "NonInvokableEngine",
    "RegistryItem",
    "SchemaComposer",
    "__version__",
]
