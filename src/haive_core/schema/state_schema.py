# src/haive/core/schema/state_schema.py

from __future__ import annotations
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, get_origin, get_args, Annotated, TYPE_CHECKING
from pydantic import BaseModel, Field, create_model
import inspect
import logging
from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)

T = TypeVar('T')

if TYPE_CHECKING:
    from haive_core.schema.schema_manager import StateSchemaManager

class StateSchema(BaseModel):
    """
    Enhanced base class for state schemas in the Haive framework.
    
    Extends Pydantic's BaseModel with additional utilities for:
    - Safe attribute access with defaults
    - Dict conversion helpers
    - Support for LangGraph state operations
    - Shared state field tracking for subgraphs
    """
    
    # Class variable to track which fields are shared with parent graphs
    __shared_fields__: List[str] = []
    
    # Class variable to track fields that use reducers (string names only for serialization)
    __serializable_reducers__: Dict[str, str] = {}
    
    @classmethod
    def create(cls, **field_definitions) -> Type['StateSchema']:
        """
        Create a new StateSchema class with the given field definitions.
        
        Args:
            **field_definitions: Field definitions in the format {name: (type, default)}
            
        Returns:
            A new StateSchema subclass
        """
        # Handle special case for __name__ field
        model_name = field_definitions.pop('__name__', 'CustomStateSchema')
        
        # Create the model class with the field definitions
        model = create_model(
            model_name,
            __base__=cls,
            **field_definitions
        )
        
        # Initialize shared fields and serializable reducers
        model.__shared_fields__ = []
        model.__serializable_reducers__ = {}
        
        # Process fields for Annotated types with reducers
        for field_name, (field_type, _) in field_definitions.items():
            # Check for shared field indicator
            if hasattr(field_type, "__shared__") and field_type.__shared__:
                model.__shared_fields__.append(field_name)
            
            # Check if it's an Annotated type with a reducer
            if get_origin(field_type) is Annotated:
                args = get_args(field_type)
                # The reducer is the second argument in Annotated[T, reducer]
                if len(args) > 1 and callable(args[1]):
                    reducer = args[1]
                    model.__serializable_reducers__[field_name] = reducer.__name__
        
        # Special handling for messages field - add add_messages reducer if not already set
        if "messages" in field_definitions and "messages" not in model.__serializable_reducers__:
            try:
                from langgraph.graph import add_messages
                model.__serializable_reducers__["messages"] = "add_messages"
            except ImportError:
                logger.debug("Could not import add_messages for message reducer")
        
        return model
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Safely get a field value with a default.
        
        Args:
            key: Field name to get
            default: Default value if field doesn't exist
            
        Returns:
            Field value or default
        """
        if hasattr(self, key):
            return getattr(self, key)
        return default
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the state to a dictionary, excluding internal fields.
        
        Returns:
            Dictionary representation of the state
        """
        # Get all data first
        if hasattr(self, "model_dump"):
            # Pydantic v2
            data = self.model_dump()
        else:
            # Pydantic v1
            data = super().dict()
            
        # Filter out internal fields
        for field in ["__shared_fields__", "__serializable_reducers__"]:
            if field in data:
                data.pop(field)
                
        return data
    
    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """
        Override model_dump to exclude internal fields.
        This ensures that internal implementation details aren't leaked in serialization.
        
        Returns:
            Clean dictionary representation without internal fields
        """
        # Get the base model_dump result
        if hasattr(super(), "model_dump"):
            # Pydantic v2
            data = super().model_dump(**kwargs)
        else:
            # Pydantic v1 fallback
            data = super().dict(**kwargs)
            
        # Filter out internal fields
        for field in ["__shared_fields__", "__serializable_reducers__"]:
            if field in data:
                data.pop(field)
                
        return data
    
    def dict(self, **kwargs) -> Dict[str, Any]:
        """
        Override dict method to exclude internal fields.
        This ensures backward compatibility with Pydantic v1.
        
        Returns:
            Clean dictionary representation without internal fields
        """
        # Get the base dict result
        data = super().dict(**kwargs)
            
        # Filter out internal fields
        for field in ["__shared_fields__", "__serializable_reducers__"]:
            if field in data:
                data.pop(field)
                
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StateSchema':
        """
        Create a state from a dictionary, ignoring internal fields.
        
        Args:
            data: Dictionary to convert
            
        Returns:
            StateSchema instance
        """
        # Filter out any internal fields that might be in the data
        clean_data = {k: v for k, v in data.items() 
                      if k not in ["__shared_fields__", "__serializable_reducers__"]}
        
        # Use model_validate for Pydantic v2, parse_obj for v1
        if hasattr(cls, "model_validate"):
            return cls.model_validate(clean_data)
        else:
            return cls.parse_obj(clean_data)
    
    def update(self, other: Union[Dict[str, Any], 'StateSchema']) -> 'StateSchema':
        """
        Update the state with values from another state or dictionary.
        
        Args:
            other: State or dictionary to update from
            
        Returns:
            Self for chaining
        """
        if isinstance(other, StateSchema):
            data = other.to_dict()
        else:
            data = other

        # Simple update without attempting to apply reducers
        for key, value in data.items():
            setattr(self, key, value)

        return self
    
    def merge_messages(self, new_messages: List[BaseMessage]) -> 'StateSchema':
        """
        Merge new messages with existing messages using LangGraph's add_messages reducer.
        
        Args:
            new_messages: New messages to add
            
        Returns:
            Self for chaining
        """
        if not hasattr(self, 'messages'):
            # Create messages field if it doesn't exist
            self.messages = new_messages
            return self
        
        # Try to use add_messages reducer
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
    
    @classmethod
    def with_messages(cls) -> Type['StateSchema']:
        """
        Create a new schema with messages field using LangGraph's add_messages reducer.
        
        Returns:
            New schema class with messages field
        """
        from typing import Sequence
        
        # Create with messages field, attempting to use add_messages reducer
        try:
            from langgraph.graph import add_messages
            return cls.create(
                messages=(Annotated[Sequence[BaseMessage], add_messages], [])
            )
        except ImportError:
            # Fallback without reducer
            return cls.create(
                messages=(Sequence[BaseMessage], [])
            )
    
    @classmethod
    def shared_fields(cls) -> List[str]:
        """
        Get the list of fields shared with parent graphs.
        
        Returns:
            List of shared field names
        """
        return cls.__shared_fields__
    
    @classmethod
    def is_shared(cls, field_name: str) -> bool:
        """
        Check if a field is shared with parent graphs.
        
        Args:
            field_name: Name of the field to check
            
        Returns:
            True if field is shared, False otherwise
        """
        return field_name in cls.__shared_fields__
    
    @classmethod
    def to_manager(cls, name: Optional[str] = None) -> "StateSchemaManager":
        """Convert schema class to a StateSchemaManager for further manipulation."""
        from haive_core.schema.schema_manager import StateSchemaManager
        return StateSchemaManager(cls, name=name or cls.__name__)
    
    @classmethod
    def from_partial_dict(cls, data: Dict[str, Any]) -> 'StateSchema':
        """
        Builds a state from a partial dict, using default values for the rest.
        
        Args:
            data: Partial dictionary of values
            
        Returns:
            StateSchema instance with defaults filled in
        """
        # Get defaults from model fields
        full_data = {
            k: v.default if v.default is not None else None
            for k, v in cls.model_fields.items()
        }
        # Update with provided data
        full_data.update(data)
        
        # Create instance
        if hasattr(cls, "model_validate"):
            return cls.model_validate(full_data)
        else:
            return cls.parse_obj(full_data)
    
    def model_validate(data: Dict[str, Any], **kwargs) -> 'StateSchema':
        """
        Validates data against the schema, ignoring internal fields.
        
        Args:
            data: Dictionary to validate
            **kwargs: Additional validation arguments
            
        Returns:
            StateSchema instance
        """
        # Filter out internal fields
        if isinstance(data, dict):
            clean_data = {k: v for k, v in data.items() 
                          if k not in ["__shared_fields__", "__serializable_reducers__"]}
        else:
            clean_data = data
        
        # Use parent validation method
        if hasattr(super(), "model_validate"):
            # Pydantic v2
            return super().model_validate(clean_data, **kwargs)
        else:
            # Pydantic v1
            return super().parse_obj(clean_data)
    
    def __getstate__(self):
        """
        Custom state serialization for pickle compatibility.
        
        Returns:
            Dictionary of serializable state
        """
        # Get the normal state
        state = self.__dict__.copy()
        
        # Ensure __shared_fields__ is included
        state["__shared_fields__"] = getattr(self.__class__, "__shared_fields__", [])
        
        # Include serializable_reducers
        state["__serializable_reducers__"] = getattr(self.__class__, "__serializable_reducers__", {})
        
        return state