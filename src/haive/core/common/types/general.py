"""General type definitions and type aliases for the Haive framework.

This module provides commonly used type definitions, type aliases, and type variables
that are shared across the Haive framework. It centralizes type definitions to ensure
consistency and type safety across all components.

The types in this module include:
- Generic type variables for flexible typing
- Tool-related type aliases for LangChain integration
- State and configuration type aliases
- Protocol-based type definitions

Example:
    Use tool-like types::

        def process_tools(tools: ToolLike) -> list[BaseTool]:
            # Handle various tool input formats
            pass

See Also:
    haive.core.common.types.protocols: Protocol definitions
    langchain_core.tools: LangChain tool types
"""

from collections.abc import Callable, Sequence
from typing import Any, TypeVar, Union

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, BaseToolkit, StructuredTool, Tool
from langgraph.types import Command, Send
from pydantic import BaseModel

from haive.core.common.types.protocols.general_protocols import Nameable
from haive.core.schema.state_schema import StateSchema

# Generic type that can be string, dict with name key, or any object with
# name attribute
OptionItem = TypeVar("OptionItem", str, dict[str, Any], Nameable)
ToolLike = Sequence[
    Tool | type[BaseTool] | BaseModel | Callable | StructuredTool | BaseToolkit
]

# Core type definitions
# CHECK:
NodeLike = TypeVar("NodeLike", bound=Any)
NodeOutput = TypeVar(
    "NodeOutput",
    bound=Command
    | Send
    | list[Send]
    | str
    | type[BaseModel]
    | BaseModel
    | dict[str, Any]
    | Send,
)
StateLike = TypeVar(
    "StateLike", bound=dict[str, Any] | type[BaseModel] | BaseModel | StateSchema
)
ConfigLike = TypeVar(
    "ConfigLike", bound=RunnableConfig | dict[str, Any] | type[BaseModel]
)
NodeCallable = Callable[[StateLike, ConfigLike | None], NodeOutput]
BaseEdge = Union[tuple[str, str], tuple[NodeLike, NodeLike]]
StateType = Union[dict[str, Any], BaseModel, StateSchema]
ConfigType = Union[RunnableConfig, dict[str, Any], BaseModel]
