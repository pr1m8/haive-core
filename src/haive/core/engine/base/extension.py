# src/haive/core/engine/extension.py

from typing import Any, Dict, Generic, List, Optional, Type, TypeVar, Protocol, runtime_checkable
from pydantic import BaseModel, Field
import uuid
T = TypeVar('T')  # The engine type this extension applies to

@runtime_checkable
class ExtensionPoint(Protocol):
    """Protocol for objects that can be extended."""
    
    def apply_extensions(self, extensions: List['Extension']) -> None: ...

class Extension(BaseModel, Generic[T]):
    """
    Base class for engine extensions.
    
    Extensions can modify or enhance the behavior of engines without changing
    their core implementation.
    """
    id: str = Field(
        default_factory=lambda: f"ext_{uuid.uuid4().hex[:8]}",
        description="Unique identifier for this extension"
    )
    name: str = Field(description="Name of this extension")
    description: Optional[str] = Field(
        default=None,
        description="Optional description of this extension"
    )
    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration for this extension"
    )
    
    def apply_to(self, target: T) -> T:
        """
        Apply this extension to a target object.
        
        Args:
            target: Object to apply the extension to
            
        Returns:
            Modified object
        """
        if not isinstance(target, ExtensionPoint):
            raise TypeError(f"Target {target} does not support extensions")
        
        target.apply_extensions([self])
        return target