from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    TypeVar,
    Union,
    runtime_checkable,
)

from langchain_core.runnables import RunnableConfig
from langgraph.types import Command, Send
from pydantic import BaseModel

from haive.core.schema.state_schema import StateSchema

# State input type - covers all possible input types
StateInput = TypeVar("StateInput", bound=Union[StateSchema, BaseModel, Dict[str, Any]])

# Output type - covers all possible return types
T = TypeVar("T", bound=Union[StateSchema, Command, Send, List[Send], Dict[str, Any]])

# Config type - covers all possible config types including BaseModel
ConfigType = Union[RunnableConfig, BaseModel, Dict[str, Any], None]


@runtime_checkable
class NodeFunction(Protocol[StateInput, T]):
    """Protocol for node functions."""

    def __call__(self, state: StateInput, config: Optional[ConfigType] = None) -> T:
        """Execute the node with the given state and configuration."""
        ...


@runtime_checkable
class AsyncNodeFunction(Protocol[StateInput, T]):
    """Protocol for async node functions."""

    async def __call__(
        self, state: StateInput, config: Optional[ConfigType] = None
    ) -> T:
        """Execute the node asynchronously."""
        ...
