# src/haive/core/engine/reference.py

from typing import Any, Dict, List, Optional, Type, Union
from pydantic import BaseModel, Field

class ComponentRef(BaseModel):
    """
    Reference to a component that can be resolved at runtime.
    
    This allows for clean serialization and avoids circular dependencies.
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
    
    type: Optional[str] = Field(
        default=None,
        description="Type of component (e.g., 'llm', 'agent')"
    )
    
    # Extensions to apply when resolving
    extensions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Extensions to apply when resolving"
    )
    
    # Runtime cache (excluded from serialization)
    _resolved: Optional[Any] = Field(default=None, exclude=True)
    
    def resolve(self) -> Optional[Any]:
        """
        Resolve this reference to the actual component.
        
        Returns:
            The resolved component or None if not found
        """
        # Return cached value if available
        if self._resolved is not None:
            return self._resolved
        
        # First try by ID if available
        if self.id:
            from src.haive.core.engine.registry import EngineRegistry
            config = EngineRegistry.get_instance().get_by_id(self.id)
            if config:
                # Build the component
                component = config.build()
                
                # Apply extensions if any
                if self.extensions:
                    from haive.core.engine.extension import Extension
                    for ext_data in self.extensions:
                        ext = Extension.model_validate(ext_data)
                        component = ext.apply_to(component)
                
                self._resolved = component
                return component
        
        # Then try by name and type
        if self.name and self.type:
            from src.haive.core.engine.config_base import EngineType
            try:
                from src.haive.core.engine.registry import EngineRegistry
                engine_type = EngineType(self.type)
                config = EngineRegistry.get_instance().get(engine_type, self.name)
                if config:
                    # Build the component
                    component = config.build()
                    
                    # Apply extensions if any
                    if self.extensions:
                        from src.haive.core.engine.extension import Extension
                        for ext_data in self.extensions:
                            ext = Extension.model_validate(ext_data)
                            component = ext.apply_to(component)
                    
                    self._resolved = component
                    return component
            except ValueError:
                pass
        
        return None
    
    @classmethod
    def from_config(cls, config: Any) -> 'ComponentRef':
        """
        Create a reference from a configuration.
        
        Args:
            config: Configuration to reference
            
        Returns:
            ComponentRef pointing to the configuration
        """
        if hasattr(config, "id") and hasattr(config, "name") and hasattr(config, "engine_type"):
            return cls(
                id=config.id,
                name=config.name,
                type=config.engine_type.value if hasattr(config.engine_type, "value") else config.engine_type
            )
        
        # Generic fallback
        if hasattr(config, "name"):
            return cls(name=getattr(config, "name"))
        
        return cls()