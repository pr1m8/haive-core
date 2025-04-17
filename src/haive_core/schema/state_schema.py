from __future__ import annotations
# src/haive/core/schema/state_schema.py

from typing import Any, Dict, List, Optional, Type, TypeVar, Union,\
    get_origin, get_args, Annotated,TYPE_CHECKING
from pydantic import BaseModel, Field, create_model
import inspect
import logging
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages

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
    
    # Class variable to track fields that use reducers
    __reducer_fields__: Dict[str, Any] = {}
    
    @classmethod
    def create(cls, **field_definitions) -> Type['StateSchema']:
        """
        Create a new StateSchema class with the given field definitions.
        
        Args:
            **field_definitions: Field definitions in the format {name: (type, default)}
            
        Returns:
            A new StateSchema subclass
        """
        field_dict = {}
        shared_fields = []
        reducer_fields = {}
        
        # Process field definitions
        for field_name, field_def in field_definitions.items():
            # Handle simple type definitions
            if isinstance(field_def, type):
                field_dict[field_name] = (field_def, None)
            
            # Handle (type, default) tuples
            elif isinstance(field_def, tuple) and len(field_def) == 2:
                field_type, default = field_def
                field_dict[field_name] = (field_type, default)
                
                # Check if it's a shared field
                if hasattr(field_type, "__shared__") and field_type.__shared__:
                    shared_fields.append(field_name)
                
                # Check if it's an Annotated type with a reducer
                if get_origin(field_type) is Annotated:
                    args = get_args(field_type)
                    # The reducer is the second argument in Annotated[T, reducer]
                    if len(args) > 1:
                        reducer = args[1]
                        reducer_fields[field_name] = reducer
            
            # Handle other types
            else:
                field_dict[field_name] = (Any, field_def)
                
        # Create the new model class
        model_name = field_definitions.pop('__name__', 'CustomStateSchema')
        model = create_model(
            model_name,
            __base__=cls,
            **field_dict
        )
        
        # Set shared fields and reducer fields
        model.__shared_fields__ = shared_fields
        model.__reducer_fields__ = reducer_fields
        
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
    
    def set(self, key: str, value: Any) -> 'StateSchema':
        """
        Set a field value, creating it if it doesn't exist.
        TODO: Implement this with a dynamic dictionary for extra, but consider alidation. 
        Args:
            key: Field name to set
            value: Value to set
        
        Returns:
            Self for chaining
        """
        # Option 1: Use model_fields to check if field exists
        if key in self.model_fields:
            setattr(self, key, value)
        else:
            # For new fields, we have several options:
            # Option 1: Raise error for non-existent fields
            raise ValueError(f"{self.__class__.__name__} has no field {key!r}")
            
            # Option 2: Only allow setting known fields (safest)
            # pass  # Do nothing for unknown fields
            
            # Option 3: Use a dynamic dictionary for extra fields
            # if not hasattr(self, "__extra_fields"):
            #     object.__setattr__(self, "__extra_fields", {})
            # self.__extra_fields[key] = value
        
        return self
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the state to a dictionary.
        
        Returns:
            Dictionary representation of the state
        """
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StateSchema':
        """
        Create a state from a dictionary.
        
        Args:
            data: Dictionary to convert
            
        Returns:
            StateSchema instance
        """
        return cls.model_validate(data)
    
        # In StateSchema.update method, ensure reducers are actually applied:
    def update(self, other: Union[Dict[str, Any], 'StateSchema']) -> 'StateSchema':
        """Update the state with values from another state or dictionary."""
        if isinstance(other, StateSchema):
            data = other.to_dict()
        else:
            data = other

        for key, value in data.items():
            if key in self.__reducer_fields__ and hasattr(self, key):
                current_val = getattr(self, key, None)
                reducer = self.__reducer_fields__[key]
                try:
                    # Apply reducer and set the result
                    setattr(self, key, reducer(current_val, value))
                except Exception as e:
                    logger.warning(f"Reducer failed for field {key}: {e}")
                    setattr(self, key, value)  # Fall back to regular assignment
            else:
                # Regular assignment for fields without reducers
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
            
        # Use add_messages reducer
        self.messages = add_messages(self.messages, new_messages)
        return self
    
    @classmethod
    def with_messages(cls) -> Type['StateSchema']:
        """
        Create a new schema with messages field using LangGraph's add_messages reducer.
        
        Returns:
            New schema class with messages field
        """
        from typing import Sequence
        
        # Create a new model with messages field
        return cls.create(
            messages=(Annotated[Sequence[BaseMessage], add_messages], [])
        )
    
    @classmethod
    def with_shared_field(cls, name: str, field_type: Type[T], default: Any = None) -> Type['StateSchema']:
        """
        Create a new schema with a shared field for parent/subgraph communication.
        
        Args:
            name: Field name
            field_type: Field type
            default: Default value
            
        Returns:
            New schema class with shared field
        """
        # Create a field definition
        field_def = {name: (field_type, default)}
        
        # Create the new model
        model = cls.create(**field_def)
        
        # Mark field as shared
        if name not in model.__shared_fields__:
            model.__shared_fields__.append(name)
            
        return model
    
    @classmethod
    def as_reducer(cls, field_name: str) -> Any:
        """
        Get the reducer for a field.
        
        Args:
            field_name: Name of the field
            
        Returns:
            Reducer function or None
        """
        return cls.__reducer_fields__.get(field_name)
    
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
    def add_field(
        cls,
        name: str,
        field_type: type,
        default: Any = None,
        *,
        default_factory: Optional[callable] = None,
        shared: bool = False,
        reducer: Optional[callable] = None,
        description: Optional[str] = None,
        optional: bool = False
    ) -> Type['StateSchema']:
        """
        Create a new StateSchema subclass with an additional field.

        Args:
            name: Name of the new field.
            field_type: Type of the field.
            default: Default value (optional).
            default_factory: A callable that returns a default value (optional).
            shared: Whether this field should be shared with parent graphs.
            reducer: Optional reducer function for merging values.
            description: Optional description for schema docs and validation.
            optional: Whether to make the field Optional if it isn't already.

        Returns:
            New StateSchema subclass with the field added.
        """
        from pydantic import Field
        from typing import Optional as OptionalType

        # Make field Optional if requested and not already Optional
        if optional and get_origin(field_type) is not OptionalType:
            field_type = OptionalType[field_type]

        # Build field metadata
        field_metadata = {}
        if description:
            field_metadata["description"] = description

        # Configure field with appropriate default handling
        if default_factory is not None:
            field_def = (field_type, Field(default_factory=default_factory, **field_metadata))
        elif default is not None:
            field_def = (field_type, Field(default=default, **field_metadata))
        else:
            if optional or get_origin(field_type) is OptionalType:
                field_def = (field_type, Field(default=None, **field_metadata))
            else:
                # For non-optional fields with no default, use ...
                field_def = (field_type, Field(default=..., **field_metadata))

        # Compose new schema fields
        fields = {}
        
        # Copy existing fields
        for f, v in cls.model_fields.items():
            fields[f] = (v.annotation, v.default)
                
        # Add the new field
        fields[name] = field_def

        new_cls = cls.create(__name__=f"{cls.__name__}_With_{name.capitalize()}", **fields)

        if shared and name not in new_cls.__shared_fields__:
            new_cls.__shared_fields__.append(name)

        if reducer:
            new_cls.__reducer_fields__[name] = reducer

        return new_cls
    @classmethod
    def to_manager(cls, name: Optional[str] = None) -> "StateSchemaManager":
        """Convert schema class to a StateSchemaManager."""
        from haive_core.schema.schema_manager import StateSchemaManager
        return StateSchemaManager(cls, name=name or cls.__name__)
    
    @classmethod
    def from_partial_dict(cls, data: Dict[str, Any]) -> 'StateSchema':
        """Builds a state from a partial dict, using default values for the rest."""
        full_data = {
            k: v.default if v.default is not None else None
            for k, v in cls.model_fields.items()
        }
        full_data.update(data)
        return cls.model_validate(full_data)