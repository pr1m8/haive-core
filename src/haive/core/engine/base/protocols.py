# src/haive/core/engine/base/protocols.py
from typing import Any, Dict, List, Protocol, TypeVar, runtime_checkable, Optional, Union
from langchain_core.runnables import RunnableConfig
from haive.core.engine.base import Engine

# Type variables
I = TypeVar('I')  # Input type
O = TypeVar('O')  # Output type
@runtime_checkable
class EngineProtocol(Protocol):
    """
    Protocol that all engines must implement.
    
    This protocol defines the core capabilities required for
    an object to be used as an engine in the node system.
    """
    id: str
    name: str
    
    def create_runnable(self, config: Optional[Union[RunnableConfig,Engine]] = None) -> Any:
        """
        Create a runnable instance with configuration applied.
        
        Args:
            config: Optional runtime configuration
            
        Returns:
            A runnable object
        """
        ...
@runtime_checkable
class Invokable(Protocol[I, O]):
    """Protocol for objects that can be invoked."""
    def invoke(self, input_data: I, **kwargs) -> O: ...
    def stream(self, input_data: I, **kwargs) -> O: ...
    def batch(self, input_data: List[I], **kwargs) -> List[O]: ...
@runtime_checkable
class AsyncInvokable(Protocol[I, O]):
    """Protocol for objects that can be invoked asynchronously."""
    async def ainvoke(self, input_data: I, **kwargs) -> O: ...
    async def astream(self, input_data: I, **kwargs) -> O: ...
    async def abatch(self, input_data: List[I], **kwargs) -> List[O]: ...
