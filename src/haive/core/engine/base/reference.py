# src/haive/core/engine/reference.py

from typing import Any, Dict, Generic, List, Optional, TypeVar, Union
from pydantic import BaseModel, Field, ConfigDict, PrivateAttr

from haive.core.engine.base import Engine, EngineType

T = TypeVar('T')  # Resolved component type

class ComponentRef(BaseModel, Generic[T]):
    """Reference to a component that can be resolved at runtime."""
    
    # Reference fields
    id: Optional[str] = Field(default=None)
    name: Optional[str] = Field(default=None)
    type: Optional[Union[str, EngineType]] = Field(default=None)
    
    # Configuration and extensions
    config_overrides: Dict[str, Any] = Field(default_factory=dict)
    extensions: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Cache for resolved instance
    _resolved: Optional[T] = PrivateAttr(default=None)
    
    # Configuration
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def resolve(self) -> Optional[T]:
        """Resolve this reference to the actual component."""
        # Implementation details...
        pass
    
    def invalidate_cache(self) -> None:
        """Clear the cached resolved component."""
        self._resolved = None
    
    @classmethod
    def from_engine(cls, engine: Engine) -> 'ComponentRef':
        """Create a reference from an engine."""
        return cls(
            id=engine.id,
            name=engine.name,
            type=engine.engine_type
        )