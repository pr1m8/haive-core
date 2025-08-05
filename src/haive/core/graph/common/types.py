# haive/core/graph/common/types.py
from collections.abc import Callable
from enum import Enum
from typing import Any, TypeVar, Union

from langchain_core.runnables import RunnableConfig
from langgraph.types import Command, Send
from pydantic import BaseModel

from haive.core.schema.state_schema import StateSchema

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
    | dict[str, Any],
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


# Node type enum - moved from BaseGraph to common types
class NodeType(str, Enum):
    """Types of nodes in a graph."""

    ENGINE = "engine"
    AGENT = "agent"
    CALLABLE = "callable"
    TOOL = "tool"
    VALIDATION = "validation"
    # BRANCH = "branch"
    # SEND = "send"
    CUSTOM = "custom"
    SUBGRAPH = "subgraph"
    PARSER = "parser"

    # Message handling nodes
    MESSAGE_TRANSFORMER = "message_transformer"

    # Coordination nodes
    COORDINATOR = "coordinator"
    TRANSFORM = "transform"

    # Output parsing
    OUTPUT_PARSER = "output_parser"
