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

# ============================================================================
# PERFORMANCE OPTIMIZATIONS - Applied automatically on import
# ============================================================================

import importlib
import logging
import os
import pkgutil
from pathlib import Path

# ============================================================================
# Module Discovery - Expose all submodules
# ============================================================================

# Get the package directory
_package_dir = Path(__file__).resolve().parent

# Core modules to expose (in order of importance)
_CORE_MODULES = [
    "engine",  # Engine system - most important
    "graph",  # Graph building
    "schema",  # Schema management
    "tools",  # Tool system
    "types",  # Type definitions
    "utils",  # Utilities
    "models",  # Model configurations
    "registry",  # Component registry
    "runtime",  # Runtime system
    "persistence",  # State persistence
    "config",  # Configuration
    "common",  # Common utilities
    "errors",  # Error types
]

# ============================================================================
# Lazy Module Loading with __getattr__
# ============================================================================


def __getattr__(name: str):
    """Lazy load modules and their contents on demand."""
    # Check if it's a module name
    if name in _CORE_MODULES:
        module = importlib.import_module(f".{name}", package=__name__)
        # Cache in globals
        globals()[name] = module
        return module

    # Check if it's a specific class/function from a module
    # First try the common imports from before
    _common_imports = {
        # Engine components
        "AugLLMConfig": ("engine", "AugLLMConfig"),
        "AugLLMFactory": ("engine", "AugLLMFactory"),
        "Engine": ("engine", "Engine"),
        "InvokableEngine": ("engine", "InvokableEngine"),
        "NonInvokableEngine": ("engine", "NonInvokableEngine"),
        # Graph components
        "BaseGraph": ("graph", "BaseGraph"),
        # Registry
        "DynamicRegistry": ("registry", "DynamicRegistry"),
        "RegistryItem": ("registry", "RegistryItem"),
        # Schema
        "SchemaComposer": ("schema", "SchemaComposer"),
    }

    if name in _common_imports:
        module_name, attr_name = _common_imports[name]
        module = importlib.import_module(f".{module_name}", package=__name__)
        attr = getattr(module, attr_name)
        # Cache in globals
        globals()[name] = attr
        return attr

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

# Public API - includes both modules and common classes
__all__ = [
    # Modules
    "engine",
    "graph",
    "schema",
    "tools",
    "types",
    "utils",
    "models",
    "registry",
    "runtime",
    "persistence",
    "config",
    "common",
    "errors",
    # Common classes for convenience
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
