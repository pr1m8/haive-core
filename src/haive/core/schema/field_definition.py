from typing import Any, Callable, Dict, Generic, Optional, Type, TypeVar, Union,Tuple
from pydantic import Field

# Type variables for field values and reducers
TField = TypeVar('TField')
TReducer = TypeVar('TReducer', bound=Callable[[Any, Any], Any])

class FieldDefinition(Generic[TField, TReducer]):
    """
    Definition of a schema field with metadata.
    
    This class provides a clean interface for defining fields with
    associated metadata like defaults, descriptions, and reducers.
    """
    
    def __init__(
        self, 
        name: str,
        field_type: Type[TField],
        default: Optional[TField] = None,
        default_factory: Optional[Callable[[], TField]] = None,
        description: Optional[str] = None,
        shared: bool = False,
        reducer: Optional[TReducer] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a field definition.
        
        Args:
            name: Field name
            field_type: Type of the field
            default: Default value
            default_factory: Optional factory function for default value
            description: Optional field description
            shared: Whether field is shared with parent graph
            reducer: Optional reducer function
            metadata: Additional metadata
        """
        self.name = name
        self.field_type = field_type
        self.default = default
        self.default_factory = default_factory
        self.description = description
        self.shared = shared
        self.reducer = reducer
        self.metadata = metadata or {}
    
    def to_field_info(self) -> Tuple[Type[TField], Field]:
        """
        Convert to a field info tuple for Pydantic.
        
        Returns:
            Tuple of (type, field_info)
        """
        field_kwargs = {}
        if self.description:
            field_kwargs["description"] = self.description
            
        if self.default_factory is not None:
            return self.field_type, Field(default_factory=self.default_factory, **field_kwargs)
        else:
            return self.field_type, Field(default=self.default, **field_kwargs)
    
    def get_reducer_name(self) -> Optional[str]:
        """
        Get serializable name for the reducer.
        
        Returns:
            Serializable name or None if no reducer
        """
        if not self.reducer:
            return None
            
        # Handle operator module functions
        if hasattr(self.reducer, "__module__"):
            module_name = self.reducer.__module__
            # Normalize operator module name (could be _operator or operator)
            if module_name in ('operator', '_operator'):
                return f"operator.{self.reducer.__name__}"
            
        # Handle lambda functions
        if hasattr(self.reducer, "__name__") and self.reducer.__name__ == "<lambda>":
            return "<lambda>"
            
        # Handle standard functions
        if hasattr(self.reducer, "__name__"):
            # Check if it has a module for fully qualified name
            if hasattr(self.reducer, "__module__") and self.reducer.__module__ != "__main__":
                return f"{self.reducer.__module__}.{self.reducer.__name__}"
            return self.reducer.__name__
            
        # Fallback to string representation
        return str(self.reducer)