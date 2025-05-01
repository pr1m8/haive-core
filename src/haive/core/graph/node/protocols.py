# node/protocols.py
from typing import Protocol, TypeVar, Union, List, Any, runtime_checkable
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command, Send
from haive.core.schema.state_schema import StateSchema

T = TypeVar('T', bound=Union[StateSchema, Command, Send, List[Send]])

@runtime_checkable
class NodeFunction(Protocol[T]):
    """Protocol for node functions."""
    def __call__(self, state: StateSchema, config: RunnableConfig) -> T:
        """Execute the node with the given state and configuration."""
        ...

@runtime_checkable
class AsyncNodeFunction(Protocol[T]):
    """Protocol for async node functions."""
    async def __call__(self, state: StateSchema, config: RunnableConfig) -> T:
        """Execute the node asynchronously."""
        ...