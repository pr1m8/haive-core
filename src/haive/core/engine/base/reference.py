
from typing import Any, Dict, Generic, List, Optional, TypeVar, Union
from pydantic import BaseModel, Field, ConfigDict, PrivateAttr

from haive.core.engine.base import EngineType

T = TypeVar('T')  # Type variable for the resolved component

class ComponentRef(BaseModel, Generic[T]):
    """
    Reference to a component that can be resolved at runtime.
    
    This allows for clean serialization while maintaining type information
    about the referenced component.
    """
    # Reference by ID (most specific)
    id: Optional[str] = Field(
        default=None,
        description="Unique ID of the referenced component"
    )
    
    # Reference by name and type (less specific)
    name: Optional[str] = Field(
        default=None,
        description="Name of the referenced component"
    )
    
    type: Optional[Union[str, EngineType]] = Field(
        default=None,
        description="Type of component (e.g., 'llm', 'agent')"
    )
    
    # Configuration overrides to apply when resolving
    config_overrides: Dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration overrides to apply when resolving"
    )
    
    # Extensions to apply when resolving
    extensions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Extensions to apply when resolving"
    )
    
    # Private attribute for runtime cache (correctly handled in Pydantic v2)
    _resolved: Optional[T] = PrivateAttr(default=None)
    
    # Configuration for model serialization
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def resolve(self) -> Optional[T]:
        """
        Resolve this reference to the actual component.
        
        Returns:
            The resolved component or None if not found
        """
        # Return cached value if available
        if self._resolved is not None:
            return self._resolved
        
        from haive.core.engine.base import EngineRegistry
        registry = EngineRegistry.get_instance()
        
        # First try by ID if available
        if self.id:
            engine = registry.find_by_id(self.id)
            if engine:
                # Apply configuration overrides if any
                if self.config_overrides and hasattr(engine, "with_config_overrides"):
                    engine = engine.with_config_overrides(self.config_overrides)
                
                # Create the runtime component
                component = engine.create_runnable(self.config_overrides)
                
                # Apply extensions if any
                if self.extensions:
                    from haive.core.engine.base.extension import Extension
                    for ext_data in self.extensions:
                        ext = Extension.model_validate(ext_data)
                        component = ext.apply_to(component)
                
                self._resolved = component
                return component
        
        # Then try by name and type
        if self.name and self.type:
            engine_type = self.type
            if isinstance(engine_type, str):
                try:
                    engine_type = EngineType(engine_type)
                except ValueError:
                    return None
            
            engine = registry.get(engine_type, self.name)
            if engine:
                # Apply configuration overrides if any
                if self.config_overrides and hasattr(engine, "with_config_overrides"):
                    engine = engine.with_config_overrides(self.config_overrides)
                
                # Create the runtime component
                component = engine.create_runnable(self.config_overrides)
                
                # Apply extensions if any
                if self.extensions:
                    from haive.core.engine.base.extension import Extension
                    for ext_data in self.extensions:
                        ext = Extension.model_validate(ext_data)
                        component = ext.apply_to(component)
                
                self._resolved = component
                return component
        
        return None
    
    def invalidate_cache(self) -> None:
        """Clear the cached resolved component."""
        self._resolved = None
    
    @classmethod
    def from_engine(cls, engine: Any) -> 'ComponentRef':
        """
        Create a reference from an engine.
        
        Args:
            engine: Engine to reference
            
        Returns:
            ComponentRef pointing to the engine
        """
        return cls(
            id=engine.id,
            name=engine.name,
            type=engine.engine_type
        )