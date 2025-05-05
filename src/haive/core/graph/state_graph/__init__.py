"""
Graph serialization framework for Haive.

This package provides tools for serializing, modifying, and managing LangGraph StateGraph objects.
"""

# Base models
from .base import SerializableModel

# Builders and editors
from .builder.graph_builder import GraphBuilder
from .editor.graph_editor import GraphEditor
from .models.branch_model import BranchModel
from .models.edge_model import EdgeModel, EdgeType

# Core models
from .models.function_ref import FunctionReference
from .models.graph_model import GraphModel
from .models.node_model import NodeModel
from .models.type_ref import TypeReference

# Pattern decorators
from .registry.decorators import register_pattern

# Registries
from .registry.graph_registry import GraphRegistry
from .registry.pattern_registry import PatternDefinition, PatternRegistry

# Set up registries for the models
NodeModel._registry = GraphRegistry.get_instance()
BranchModel._registry = GraphRegistry.get_instance()
GraphModel._registry = GraphRegistry.get_instance()
PatternDefinition._registry = PatternRegistry.get_instance()

__all__ = [
    # Base
    "SerializableModel",
    # Models
    "FunctionReference",
    "TypeReference",
    "NodeModel",
    "EdgeModel",
    "EdgeType",
    "BranchModel",
    "GraphModel",
    # Registries
    "GraphRegistry",
    "PatternRegistry",
    "PatternDefinition",
    # Builders
    "GraphBuilder",
    "GraphEditor",
    # Decorators
    "register_pattern",
]
