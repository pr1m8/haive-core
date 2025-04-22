# src/haive/core/engine/protocols.py

from typing import Any, Protocol, TypeVar, runtime_checkable

T = TypeVar('T')

@runtime_checkable
class RuntimeComponent(Protocol):
    """Protocol for runtime components created by engines."""
    
    def invoke(self, input_data: Any, **kwargs) -> Any:
        """Invoke the runtime component."""
        ...

@runtime_checkable
class AsyncRuntimeComponent(Protocol):
    """Protocol for async runtime components."""
    
    async def ainvoke(self, input_data: Any, **kwargs) -> Any:
        """Invoke the runtime component asynchronously."""
        ...

@runtime_checkable
class FactoryProtocol(Protocol[T]):
    """Protocol for factories that create runtime components."""
    
    def create(self) -> T:
        """Create a runtime component."""
        ...
    
    def invalidate_cache(self) -> None:
        """Invalidate any cached instances."""
        ...