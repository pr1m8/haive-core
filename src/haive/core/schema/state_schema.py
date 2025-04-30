from __future__ import annotations
from typing import Any, Dict, List, Optional, Type, TypeVar, Generic, \
    Union, get_origin, get_args, Callable, TYPE_CHECKING
from pydantic import BaseModel, Field, create_model
import copy
import json
import logging
from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from haive.core.schema.schema_manager import StateSchemaManager

T = TypeVar('T', bound=BaseModel)

class StateSchema(BaseModel, Generic[T]):
    """
    Enhanced base class for state schemas in the Haive framework.
    """
    
    # Class variables to track field sharing and reducers
    __shared_fields__: List[str] = []
    __serializable_reducers__: Dict[str, str] = {}
    __engine_io_mappings__: Dict[str, Dict[str, List[str]]] = {}
    __input_fields__: Dict[str, List[str]] = {}
    __output_fields__: Dict[str, List[str]] = {}
    
    # Note: __reducer_fields__ is created dynamically and not part of instance properties
    
    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Override model_dump to exclude internal fields."""
        # Get the base model_dump result from Pydantic v2
        data = super().model_dump(**kwargs)
            
        # Filter out internal fields
        internal_fields = ["__shared_fields__", "__serializable_reducers__", "__reducer_fields__",
                          "__engine_io_mappings__", "__input_fields__", "__output_fields__"]
        for field in internal_fields:
            if field in data:
                data.pop(field)
                
        return data
    
    def dict(self, **kwargs) -> Dict[str, Any]:
        """Backwards compatibility alias for model_dump."""
        return self.model_dump(**kwargs)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the state to a clean dictionary."""
        return self.model_dump()
    
    def to_json(self) -> str:
        """Convert state to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'StateSchema':
        """Create state from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StateSchema':
        """Create a state from a dictionary."""
        # Filter out internal fields if present
        internal_fields = ["__shared_fields__", "__serializable_reducers__", "__reducer_fields__",
                          "__engine_io_mappings__", "__input_fields__", "__output_fields__"]
        clean_data = {k: v for k, v in data.items() if k not in internal_fields}
        
        # Use Pydantic v2 method for validation
        return cls.model_validate(clean_data)
    
    @classmethod
    def from_partial_dict(cls, data: Dict[str, Any]) -> 'StateSchema':
        """Create a state from a partial dictionary, filling in defaults."""
        # Get defaults from model fields - Pydantic v2
        full_data = {}
        for field_name, field_info in cls.model_fields.items():
            # Get default and default_factory
            default = field_info.default
            default_factory = field_info.default_factory
                
            # Apply defaults
            if default is not ...:
                full_data[field_name] = default
            elif default_factory is not None:
                full_data[field_name] = default_factory()
                
        # Update with provided data
        full_data.update(data)
        
        # Create instance with Pydantic v2 method
        return cls.model_validate(full_data)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Safely get a field value with a default."""
        if hasattr(self, key):
            return getattr(self, key)
        return default
    
    def update(self, other: Union[Dict[str, Any], 'StateSchema']) -> 'StateSchema':
        """Update the state with values from another state or dictionary."""
        if isinstance(other, StateSchema):
            data = other.model_dump()
        else:
            data = other

        # Simple update without attempting to apply reducers
        for key, value in data.items():
            setattr(self, key, value)

        return self

    def apply_reducers(self, other: Union[Dict[str, Any], 'StateSchema']) -> 'StateSchema':
        """Update state applying reducer functions where defined."""
        if isinstance(other, StateSchema):
            data = other.model_dump()
        else:
            data = other
        
        # Get reducer functions
        reducer_fields = getattr(self.__class__, "__reducer_fields__", {})
        
        # Apply updates with reducers where defined
        for key, value in data.items():
            # Skip if the field doesn't exist in this state
            if not hasattr(self, key):
                # Just add the field with simple assignment
                setattr(self, key, value)
                continue
                
            # Get current value
            current_value = getattr(self, key)
            
            # Apply reducer if available for this field
            if key in reducer_fields:
                reducer = reducer_fields[key]
                try:
                    # Apply reducer and set the result
                    reduced_value = reducer(current_value, value)
                    setattr(self, key, reduced_value)
                    continue # Skip to next field after successful reduction
                except Exception as e:
                    logger.warning(f"Error applying reducer for {key}: {e}")
                    # Fall through to special handling or simple assignment
            
            # Special handling for list values - concat them when both are lists
            if isinstance(current_value, list) and isinstance(value, list):
                merged_list = current_value + value
                setattr(self, key, merged_list)
                continue
            
            # Special handling for dictionary values - merge them instead of replacing
            if isinstance(current_value, dict) and isinstance(value, dict):
                merged_dict = current_value.copy()
                merged_dict.update(value)
                setattr(self, key, merged_dict)
                continue
            
            # Simple assignment (no reducer or reducer failed)
            setattr(self, key, value)
                    
        return self
    
    def merge_messages(self, new_messages: List[BaseMessage]) -> 'StateSchema':
        """Merge new messages with existing messages using appropriate reducer."""
        if not hasattr(self, 'messages'):
            # Create messages field if it doesn't exist
            self.messages = new_messages
            return self
        
        # Try to use add_messages reducer if available
        try:
            from langgraph.graph import add_messages
            self.messages = add_messages(self.messages, new_messages)
        except ImportError:
            # Fallback to simple list extension
            if isinstance(self.messages, list):
                self.messages.extend(new_messages)
            else:
                self.messages = new_messages
                
        return self
    
    def copy(self, **updates) -> 'StateSchema':
        """Create a copy of this state, optionally with updates."""
        # Use Pydantic v2 model_copy
        return self.model_copy(update=updates)
    
    def deep_copy(self) -> 'StateSchema':
        """Create a deep copy of this state object."""
        return copy.deepcopy(self)
    
    @classmethod
    def _get_reducer_registry(cls) -> Dict[str, Callable]:
        """Get a registry of reducer functions mapped to their names."""
        registry = {}
        
        # Add standard reducers
        try:
            from langgraph.graph import add_messages
            registry["add_messages"] = add_messages
        except ImportError:
            # Create a simple concat function as fallback
            def concat_lists(a, b):
                return (a or []) + (b or [])
            registry["concat_lists"] = concat_lists
        
        # Add common reducer functions
        def concat_strings(a, b):
            return (a or "") + (b or "")
        registry["concat_strings"] = concat_strings
        
        def sum_values(a, b):
            return (a or 0) + (b or 0)
        registry["sum_values"] = sum_values
        
        # Add common functions
        registry["max"] = max
        registry["min"] = min
        
        # Add operator module reducers
        import operator
        for op_name in dir(operator):
            if not op_name.startswith('_'):
                op_func = getattr(operator, op_name)
                if callable(op_func):
                    registry[f"operator.{op_name}"] = op_func
                    registry[op_name] = op_func  # Also store without prefix for backward compatibility
        
        # Try to get reducer functions from class if they exist
        if hasattr(cls, "__reducer_fields__"):
            registry.update(cls.__reducer_fields__)
        
        # Handle lambda functions
        if "<lambda>" in cls.__serializable_reducers__.values():
            # Can't restore lambdas from name, but we can provide a generic reducer
            def generic_lambda_reducer(a, b):
                # Simple fallback implementation
                if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
                    return a + b
                elif isinstance(a, dict) and isinstance(b, dict):
                    result = a.copy()
                    result.update(b)
                    return result
                else:
                    # Default to returning the newer value
                    return b
                
            registry["<lambda>"] = generic_lambda_reducer
        
        return registry
    
    @classmethod
    def shared_fields(cls) -> List[str]:
        """Get the list of fields shared with parent graphs."""
        return cls.__shared_fields__
    
    @classmethod
    def is_shared(cls, field_name: str) -> bool:
        """Check if a field is shared with parent graphs."""
        return field_name in cls.__shared_fields__
    
    @classmethod
    def to_manager(cls, name: Optional[str] = None) -> "StateSchemaManager":
        """Convert schema class to a StateSchemaManager for further manipulation."""
        from haive.core.schema.schema_manager import StateSchemaManager
        return StateSchemaManager(cls, name=name or cls.__name__)
    
    @classmethod
    def manager(cls) -> "StateSchemaManager":
        """Get a manager for this schema (shorthand for to_manager())."""
        return cls.to_manager()