# src/haive/core/engine/base/protocols.py
from typing import Any, Dict, List, Protocol, TypeVar, runtime_checkable


# Type variables
I = TypeVar('I')  # Input type
O = TypeVar('O')  # Output type

@runtime_checkable
class Invokable(Protocol[I, O]):
    """Protocol for objects that can be invoked."""
    def invoke(self, input_data: I, **kwargs) -> O: ...

@runtime_checkable
class AsyncInvokable(Protocol[I, O]):
    """Protocol for objects that can be invoked asynchronously."""
    async def ainvoke(self, input_data: I, **kwargs) -> O: ...


