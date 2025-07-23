from collections.abc import Callable, Sequence
from typing import (
    Annotated,
    Any,
    ClassVar,
    Dict,
    Generic,
    List,
    Literal,
    Optional,
    Tuple,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

from langchain_core.runnables import RunnableConfig
from langgraph.types import Command, Send
from pydantic import (
    BaseModel,
    Field,
    computed_field,
    field_serializer,
    field_validator,
    model_serializer,
    model_validator,
)
from typing_extensions import TypedDict

from haive.core.schema.state_schema import StateSchema

# Define a callable type alias more precisely
# Command destination types for better type checking
CommandGoto = Union[str, Literal["END"], Send, list[Send | str]]
# Type variables for better type safety
StateInput = TypeVar("StateInput", bound=BaseModel | dict[str, Any] | Any)
StateOutput = TypeVar(
    "StateOutput", bound=dict[str, Any] | Command | Send | list[Send] | Any
)
ConfigType = Union[RunnableConfig, dict[str, Any], None]

NodeCallable = Callable[
    [
        StateSchema | BaseModel | dict[str, Any] | TypedDict,
        BaseModel | dict | TypedDict | RunnableConfig,
        Send | Command | BaseModel | dict | Any,
    ]
]
NodeReturnType = TypeVar(
    "NodeReturnType", bound=BaseModel | dict | TypedDict | RunnableConfig
)
StateType = TypeVar("StateType", bound=StateSchema | BaseModel | dict)
NodeType = TypeVar("NodeType", bound=BaseModel | NodeCallable)
EdgeType = TypeVar("EdgeType", bound=BaseModel)
BranchType = TypeVar(
    "BranchType",
    bound=BaseModel
    | Callable[[StateType, ConfigType | None], bool | CommandGoto | list[str]],
)


class BaseGraph(BaseModel, Generic[NodeType, EdgeType]):
    """Base class for all graph models."""

    nodes: list[NodeType] = Field(default_factory=list)
    edges: list[EdgeType] = Field(default_factory=list)

    @computed_field
    @property
    def number_of_nodes(self) -> int:
        """Get the number of nodes in the graph."""
        return len(self.nodes)

    @computed_field
    @property
    def number_of_edges(self) -> int:
        """Get the number of edges in the graph."""
        return len(self.edges)
