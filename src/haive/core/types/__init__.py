"""Advanced type system module for the Haive framework.

This module provides specialized type definitions and utilities that enhance
Python's type system for use within the Haive framework. It includes dynamic
enumerations, serializable callables, advanced registries, and other type-related
utilities for flexible, extensible, and type-safe code.

The types system enables runtime extensibility while maintaining type safety,
allowing components to be registered, discovered, and validated dynamically
throughout the framework lifecycle.

Key Components:
    DynamicEnum: Runtime-extensible enumeration types
    DynamicLiteral: Dynamic type literals for improved type hinting
    SerializableCallable: Type-safe serialization of function references
    AdvancedRegistry: Enhanced registries for component management
    TreeLeaf: Tree structure utilities for type organization

Features:
    - Runtime enum extension with validation
    - Serializable function references
    - Dynamic type literal creation
    - Advanced component registries
    - Type-safe tree structures
    - Domain-specific type definitions

Examples:
    Dynamic enumeration usage::

        from haive.core.types import DynamicEnum

        class ModelProvider(DynamicEnum):
            START_VALUES = ["openai", "anthropic", "google"]

        # Use initial values
        provider = "openai"
        assert provider in ModelProvider._values

        # Register new values at runtime
        ModelProvider.register("cohere", "mistral")
        assert "mistral" in ModelProvider._values

    Serializable callable::

        from haive.core.types import SerializableCallable

        def process_data(data: dict) -> dict:
            return {"processed": data}

        # Serialize function reference
        serializable_func = SerializableCallable(
            module="__main__",
            name="process_data"
        )

        # Deserialize and call
        func = serializable_func.get_callable()
        result = func({"key": "value"})

    Dynamic literals::

        from haive.core.types import DynamicLiteral

        # Create dynamic type literal
        SupportedModels = DynamicLiteral([
            "gpt-4", "gpt-3.5-turbo", "claude-3"
        ])

        # Extend at runtime
        SupportedModels.add_values(["llama-2", "mistral-7b"])

See Also:
    - Python typing module documentation
    - Component registry system
    - Dynamic configuration guides
"""

from haive.core.types.advanced_registry import AdvancedRegistry
from haive.core.types.dynamic_enum import DynamicEnum
from haive.core.types.dynamic_literal import DynamicLiteral
from haive.core.types.general import FileTypes, ProgrammingLanguages
from haive.core.types.serializable_callable import SerializableCallable

# from haive.core.types.tree_leaf import TreeLeaf  # TODO: Fix MRO issue

__all__ = [
    # Dynamic Type System
    "DynamicEnum",
    "DynamicLiteral",
    # Serializable Components
    "SerializableCallable",
    # Advanced Structures
    "AdvancedRegistry",
    # "TreeLeaf",  # TODO: Fix MRO issue
    # General Domain Types
    "FileTypes",
    "ProgrammingLanguages",
]
