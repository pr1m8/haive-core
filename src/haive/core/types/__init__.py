"""🔍 Types Module - Intelligent Type System Revolution

**THE MOLECULAR BLUEPRINT OF AI TYPE INTELLIGENCE**

Welcome to the Types Module - the revolutionary type intelligence platform 
that transforms static type definitions into a living, adaptive type ecosystem. 
This isn't just another type system; it's a sophisticated type consciousness 
that learns, evolves, and optimizes type relationships, creating a seamless 
bridge between rigid type constraints and dynamic runtime flexibility.

⚡ REVOLUTIONARY TYPE INTELLIGENCE
---------------------------------

The Types Module represents a paradigm shift from static type definitions to 
**intelligent, adaptive type systems** that evolve with your application needs:

**🧠 Dynamic Type Evolution**: Runtime-extensible types that adapt to new requirements
**🔄 Intelligent Type Inference**: AI-powered type prediction and optimization
**⚡ Serializable Type References**: Type-safe persistence and transmission of type information
**📊 Advanced Type Registries**: Smart type organization and discovery systems
**🎯 Domain-Specific Intelligence**: Specialized types that understand their context

🌟 CORE TYPE INNOVATIONS
------------------------

For complete examples and advanced patterns, see the documentation.
"""

from haive.core.types.dynamic_enum import DynamicEnum
from haive.core.types.dynamic_literal import DynamicLiteral
from haive.core.types.general import FileTypes, ProgrammingLanguages, SerializableCallable

__all__ = [
    # Dynamic Type System
    "DynamicEnum",
    "DynamicLiteral",
    # "TreeLeaf",  # TODO: Fix MRO issue
    # General Domain Types
    "FileTypes",
    "ProgrammingLanguages",
    # Serializable Components
    "SerializableCallable",
]
