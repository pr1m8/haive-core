import functools
import inspect
from typing import Any, Callable, Optional, Type, TypeVar, Union, cast, overload

from langgraph.types import Command, Send

from haive.core.graph.node.protocols import (
    AsyncNodeFunction,
    ConfigType,
    NodeFunction,
    StateInput,
    T,
)

# Function type variables
F = TypeVar("F", bound=Callable)
SyncNode = TypeVar("SyncNode", bound=NodeFunction)
AsyncNode = TypeVar("AsyncNode", bound=AsyncNodeFunction)


@overload
def register_node(
    command_goto: Optional[Any] = None, node_config: Optional[ConfigType] = None
) -> Callable[
    [F], Union[NodeFunction[StateInput, T], AsyncNodeFunction[StateInput, T]]
]: ...


@overload
def register_node(
    func: F,
) -> Union[NodeFunction[StateInput, T], AsyncNodeFunction[StateInput, T]]: ...


def register_node(
    func_or_goto: Union[F, Optional[Any]] = None,
    node_config: Optional[ConfigType] = None,
) -> Union[
    Callable[[F], Union[NodeFunction[StateInput, T], AsyncNodeFunction[StateInput, T]]],
    NodeFunction[StateInput, T],
    AsyncNodeFunction[StateInput, T],
]:
    """
    Decorator to register a function as a node that follows the NodeFunction protocol.

    Supports both styles:
        @register_node
        def my_node(state, config): ...

        @register_node(command_goto="next_node", node_config={...})
        def my_node(state, config): ...

    Args:
        func_or_goto: Either the function to decorate or command_goto value
        node_config: Optional node configuration

    Returns:
        Decorated function that complies with NodeFunction or AsyncNodeFunction protocol
    """
    # Handle case when decorator is used without arguments
    if callable(func_or_goto):
        return _create_node_wrapper(func_or_goto, None, node_config)

    # Handle case with arguments
    command_goto = func_or_goto

    def decorator(
        func: F,
    ) -> Union[NodeFunction[StateInput, T], AsyncNodeFunction[StateInput, T]]:
        return _create_node_wrapper(func, command_goto, node_config)

    return decorator


def _create_node_wrapper(
    func: F,
    command_goto: Optional[Any] = None,
    node_config: Optional[ConfigType] = None,
) -> Union[NodeFunction[StateInput, T], AsyncNodeFunction[StateInput, T]]:
    """Create appropriate wrapper based on function type."""
    is_async = inspect.iscoroutinefunction(func)

    if is_async:

        @functools.wraps(func)
        async def async_wrapper(
            state: StateInput, config: Optional[ConfigType] = None
        ) -> T:
            # Use node_config as fallback
            effective_config = config if config is not None else node_config

            # Call async function
            result = await func(state, effective_config)

            # Apply command_goto if needed
            if command_goto is not None and not isinstance(result, Command):
                return cast(T, Command(update=result, goto=command_goto))

            return result

        # Add metadata
        async_wrapper.__node_config__ = node_config
        async_wrapper.__command_goto__ = command_goto

        return cast(AsyncNodeFunction[StateInput, T], async_wrapper)
    else:

        @functools.wraps(func)
        def sync_wrapper(state: StateInput, config: Optional[ConfigType] = None) -> T:
            # Use node_config as fallback
            effective_config = config if config is not None else node_config

            # Call function
            result = func(state, effective_config)

            # Apply command_goto if needed
            if command_goto is not None and not isinstance(result, Command):
                return cast(T, Command(update=result, goto=command_goto))

            return result

        # Add metadata
        sync_wrapper.__node_config__ = node_config
        sync_wrapper.__command_goto__ = command_goto

        return cast(NodeFunction[StateInput, T], sync_wrapper)
