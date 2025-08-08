"""Universal tool type definitions for Haive framework.

This module defines the single source of truth for tool types and properties
that all components in Haive should use.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Callable, Protocol, TypeAlias, Union, runtime_checkable

from langchain_core.tools import BaseTool, StructuredTool
from langchain_core.tools.base import BaseToolkit
from pydantic import BaseModel, ConfigDict, Field

# Universal tool type - single source of truth for all components
ToolLike: TypeAlias = Union[
    BaseTool,  # LangChain tool instances
    StructuredTool,  # Structured tool instances
    type[BaseTool],  # Tool classes
    BaseModel,  # Pydantic model instances (callable)
    type[BaseModel],  # Pydantic model classes
    Callable[..., Any],  # Raw functions
    BaseToolkit,  # Tool collections
]


class ToolType(str, Enum):
    """Tool implementation types."""

    LANGCHAIN_TOOL = "langchain_tool"
    PYDANTIC_MODEL = "pydantic_model"
    FUNCTION = "function"
    STRUCTURED_TOOL = "structured_tool"
    TOOLKIT = "toolkit"
    RETRIEVER_TOOL = "retriever_tool"
    VALIDATION_TOOL = "validation_tool"
    STORE_TOOL = "store_tool"  # Memory/store management tools


class ToolCategory(str, Enum):
    """High-level tool categorization."""

    RETRIEVAL = "retrieval"
    COMPUTATION = "computation"
    COMMUNICATION = "communication"
    TRANSFORMATION = "transformation"
    VALIDATION = "validation"
    COORDINATION = "coordination"
    MEMORY = "memory"
    SEARCH = "search"
    GENERATION = "generation"
    UNKNOWN = "unknown"


class ToolCapability(str, Enum):
    """Fine-grained tool capabilities for routing and execution."""

    # Execution capabilities
    INTERRUPTIBLE = "interruptible"
    ASYNC_CAPABLE = "async_capable"
    STREAMING = "streaming"
    BATCH_CAPABLE = "batch_capable"

    # State interaction capabilities
    READS_STATE = "reads_state"
    WRITES_STATE = "writes_state"
    INJECTED_STATE = "injected_state"
    TO_STATE = "to_state"  # Tool that writes to state
    FROM_STATE = "from_state"  # Tool that reads from state
    STATE_AWARE = "state_aware"  # General state interaction

    # Output capabilities
    STRUCTURED_OUTPUT = "structured_output"
    VALIDATED_OUTPUT = "validated_output"

    # Special tool types
    RETRIEVER = "retriever"
    VALIDATOR = "validator"
    TRANSFORMER = "transformer"
    ROUTED = "routed"  # Tool with custom routing
    STORE = "store"  # Memory/store management tool


@runtime_checkable
class InterruptibleTool(Protocol):
    """Protocol for tools that support interruption."""

    @property
    def is_interruptible(self) -> bool:
        """Check if tool can be interrupted."""
        ...

    def interrupt(self) -> None:
        """Interrupt tool execution."""
        ...


@runtime_checkable
class StateAwareTool(Protocol):
    """Protocol for tools that interact with state."""

    @property
    def reads_state(self) -> bool:
        """Check if tool reads from state."""
        ...

    @property
    def writes_state(self) -> bool:
        """Check if tool writes to state."""
        ...

    @property
    def state_dependencies(self) -> list[str]:
        """Get state keys this tool depends on."""
        ...


class ToolProperties(BaseModel):
    """Comprehensive tool properties for analysis and routing.

    This model captures all relevant properties of a tool including
    its type, capabilities, state interaction patterns, and schemas.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Core identification
    name: str = Field(..., description="Tool name")
    tool_type: ToolType = Field(..., description="Implementation type")
    category: ToolCategory = Field(
        default=ToolCategory.UNKNOWN, description="Tool category"
    )

    # Capabilities set
    capabilities: set[ToolCapability] = Field(
        default_factory=set, description="Set of tool capabilities"
    )

    # State interaction properties
    is_state_tool: bool = Field(
        default=False, description="Tool interacts with state in any way"
    )
    to_state_tool: bool = Field(default=False, description="Tool writes to state")
    from_state_tool: bool = Field(default=False, description="Tool reads from state")
    state_dependencies: list[str] = Field(
        default_factory=list, description="State keys this tool depends on"
    )
    state_outputs: list[str] = Field(
        default_factory=list, description="State keys this tool writes to"
    )

    # Execution properties
    is_interruptible: bool = Field(
        default=False, description="Tool supports interruption"
    )
    is_async: bool = Field(default=False, description="Tool supports async execution")
    is_routed: bool = Field(default=False, description="Tool has custom routing")

    # Schema information
    input_schema: type[BaseModel] | None = Field(
        default=None, description="Input schema if available"
    )
    output_schema: type[BaseModel] | None = Field(
        default=None, description="Output schema if available"
    )
    structured_output_model: type[BaseModel] | None = Field(
        default=None,
        description="Structured output model if tool produces structured data",
    )
    is_structured_output_model: bool = Field(
        default=False, description="Tool has a structured output model"
    )

    # Metadata
    description: str | None = Field(default=None, description="Tool description")
    version: str = Field(default="1.0", description="Tool version")
    tags: list[str] = Field(default_factory=list, description="Tool tags")

    # Performance hints
    expected_duration: float | None = Field(
        default=None, description="Expected execution time in seconds"
    )
    requires_network: bool = Field(
        default=False, description="Tool requires network access"
    )

    # Helper methods
    def has_capability(self, capability: ToolCapability) -> bool:
        """Check if tool has specific capability."""
        return capability in self.capabilities

    def is_retriever(self) -> bool:
        """Check if this is a retriever tool."""
        return self.has_capability(ToolCapability.RETRIEVER)

    def has_structured_output(self) -> bool:
        """Check if tool has structured output."""
        return self.has_capability(ToolCapability.STRUCTURED_OUTPUT)

    def interacts_with_state(self) -> bool:
        """Check if tool has any state interaction."""
        return self.is_state_tool or self.to_state_tool or self.from_state_tool


# Type exports for other components to use
__all__ = [
    "ToolLike",
    "ToolType",
    "ToolCategory",
    "ToolCapability",
    "ToolProperties",
    "InterruptibleTool",
    "StateAwareTool",
]
