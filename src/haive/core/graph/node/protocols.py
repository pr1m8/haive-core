from functools import wraps
from typing import (
    Any,
    Callable,
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


def register_node(
    command_goto: Optional[Any] = None, node_config: Optional[dict] = None
):
    """
    Decorator to register a function as a node.

    Args:
        command_goto: Optional destination for command routing
        node_config: Optional node configuration

    Returns:
        Decorated function
    """

    def decorator(func: Callable):
        @wraps(func)
        def wrapped(state, config=None):
            # Apply node_config if provided
            final_config = config
            if node_config and not config:
                final_config = node_config

            # Call function with state and config
            result = func(state, final_config)

            # Apply command_goto if not already a Command
            if command_goto is not None and not isinstance(result, Command):
                return Command(update=result, goto=command_goto)

            return result

        # Add metadata
        wrapped.__node_config__ = node_config
        wrapped.__command_goto__ = command_goto

        return wrapped

    return decorator
