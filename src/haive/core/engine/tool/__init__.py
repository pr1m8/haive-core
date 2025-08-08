"""Enhanced tool engine with universal typing.

This module provides the ToolEngine with comprehensive tool analysis,
universal type definitions, and factory methods for creating specialized tools.
"""

from .analyzer import ToolAnalyzer
from .engine import ToolEngine
from .types import (
    InterruptibleTool,
    StateAwareTool,
    ToolCapability,
    ToolCategory,
    ToolLike,
    ToolProperties,
    ToolType,
)

__all__ = [
    # Main engine
    "ToolEngine",
    # Types
    "ToolLike",
    "ToolType",
    "ToolCategory",
    "ToolCapability",
    "ToolProperties",
    # Protocols
    "InterruptibleTool",
    "StateAwareTool",
    # Analyzer
    "ToolAnalyzer",
]


# Convenience exports for backward compatibility
def get_tool_type():
    """Get the universal ToolLike type."""
    return ToolLike


def get_tool_analyzer():
    """Get a tool analyzer instance."""
    return ToolAnalyzer()


def get_capability_enum():
    """Get the ToolCapability enum."""
    return ToolCapability


def get_category_enum():
    """Get the ToolCategory enum."""
    return ToolCategory
