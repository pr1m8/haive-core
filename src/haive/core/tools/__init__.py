"""Haive Core Tools Package.

This package provides tools that agents can use, including store management
tools for memory operations similar to LangMem.
"""

from haive.core.tools.store_manager import MemoryEntry, StoreManager
from haive.core.tools.store_tools import (
    create_delete_memory_tool,
    create_manage_memory_tool,
    create_memory_tools_suite,
    create_retrieve_memory_tool,
    create_search_memory_tool,
    create_search_memory_tool_alias,
    create_store_memory_tool,
    create_update_memory_tool,
)

__all__ = [
    "MemoryEntry",
    # Store Manager
    "StoreManager",
    "create_delete_memory_tool",
    # LangMem-style aliases
    "create_manage_memory_tool",
    "create_memory_tools_suite",
    "create_retrieve_memory_tool",
    "create_search_memory_tool",
    "create_search_memory_tool_alias",
    # Store Tools
    "create_store_memory_tool",
    "create_update_memory_tool",
]
