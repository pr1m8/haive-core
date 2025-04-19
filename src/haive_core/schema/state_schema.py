from __future__ import annotations
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, get_origin, get_args, Annotated, TYPE_CHECKING, cast, Callable, Sequence, Set
from pydantic import BaseModel, Field, create_model
import inspect
import logging
import json
from langchain_core.messages import BaseMessage
from copy import deepcopy
if TYPE_CHECKING:
    from haive_core.schema.schema_manager import StateSchemaManager
logger = logging.getLogger(__name__)

T = TypeVar('T')

class StateSchema(BaseModel):
    """
    Enhanced base class for state schemas in the Haive framework.
    
    A StateSchema extends Pydantic's BaseModel with additional capabilities:
    - Field sharing between parent/child graphs
    - Reducer functions for merging field values
    - Serialization support for database storage
    - LangGraph integration for stateful agents
    - Easy extension and manipulation methods
    """
    
    # Class variables to track field sharing and reducers
    __shared_fields__: List[str] = []
    __serializable_reducers__: Dict[str, str] = {}
    __reducer_fields__: Dict[str, Callable] = {}
    
    # Don't use model_serializer which causes recursion
    # Instead override the existing methods
    
    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """
        Override model_dump to exclude internal fields.
        This ensures that internal implementation details aren't leaked in serialization.
        
        Args:
            **kwargs: Arguments to pass to the parent method
            
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
        for field in ["__shared_fields__", "__serializable_reducers__", "__reducer_fields__"]:
            if field in data:
                data.pop(field)
                
        return data
    
    def dict(self, **kwargs) -> Dict[str, Any]:
        """
        Override dict method to exclude internal fields.
        This ensures backward compatibility with Pydantic v1.
        
        Args:
            **kwargs: Arguments to pass to the parent method
            
        Returns:
            Clean dictionary representation without internal fields
        """
        # Get the base dict result
        data = super().dict(**kwargs)
            
        # Filter out internal fields
        for field in ["__shared_fields__", "__serializable_reducers__", "__reducer_fields__"]:
            if field in data:
                data.pop(field)
                
        return data
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the state to a dictionary, excluding internal fields.
        
        Returns:
            Dictionary representation of the state
        """
        # Use model_dump (Pydantic v2) or dict (Pydantic v1)
        if hasattr(self, "model_dump"):
            return self.model_dump()
        else:
            return self.dict()
    
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
        model.__reducer_fields__ = {}
        
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
                    model.__reducer_fields__[field_name] = reducer
                    model.__serializable_reducers__[field_name] = reducer.__name__
        
        # Special handling for messages field - add add_messages reducer if not already set
        if "messages" in field_definitions and "messages" not in model.__serializable_reducers__:
            try:
                from langgraph.graph import add_messages
                model.__reducer_fields__["messages"] = add_messages
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
                      if k not in ["__shared_fields__", "__serializable_reducers__", "__reducer_fields__"]}
        
        # Use model_validate for Pydantic v2, parse_obj for v1
        if hasattr(cls, "model_validate"):
            return cls.model_validate(clean_data)
        else:
            return cls.parse_obj(clean_data)
    
    def update(self, other: Union[Dict[str, Any], 'StateSchema']) -> 'StateSchema':
        """
        Update the state with values from another state or dictionary.
        Simple update without applying reducers.
        
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
    
    def apply_reducers(self, other: Union[Dict[str, Any], 'StateSchema']) -> 'StateSchema':
        """
        Update state applying reducer functions where defined.
        
        Args:
            other: State or dictionary to update from with reducer application
            
        Returns:
            Self for chaining
        """
        if isinstance(other, StateSchema):
            data = other.to_dict()
        else:
            data = other
            
        # Apply simple assignment for non-reducer fields
        for key, value in data.items():
            if key in self.__class__.__reducer_fields__:
                # Use the reducer for this field
                if hasattr(self, key):
                    current_value = getattr(self, key)
                    try:
                        reducer = self.__class__.__reducer_fields__[key]
                        reduced_value = reducer(current_value, value)
                        setattr(self, key, reduced_value)
                    except Exception as e:
                        logger.warning(f"Reducer for {key} failed: {e}")
                        setattr(self, key, value)  # Fallback to simple assignment
                else:
                    # Field doesn't exist yet, just set it
                    setattr(self, key, value)
            else:
                # No reducer, use simple assignment
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
    def reducer_fields(cls) -> Dict[str, Callable]:
        """
        Get the dictionary of field reducers.
        
        Returns:
            Dictionary mapping field names to reducer functions
        """
        return cls.__reducer_fields__
    
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
    def has_reducer(cls, field_name: str) -> bool:
        """
        Check if a field has a reducer function.
        
        Args:
            field_name: Name of the field to check
            
        Returns:
            True if field has a reducer, False otherwise
        """
        return field_name in cls.__reducer_fields__
    
    @classmethod
    def to_manager(cls, name: Optional[str] = None) -> "StateSchemaManager":
        """
        Convert schema class to a StateSchemaManager for further manipulation.
        
        Args:
            name: Optional name for the manager
            
        Returns:
            StateSchemaManager instance
        """
        from haive_core.schema.schema_manager import StateSchemaManager
        return StateSchemaManager(cls, name=name or cls.__name__)
    
    @classmethod
    def from_partial_dict(cls, data: Dict[str, Any]) -> 'StateSchema':
        """
        Builds a state from a partial dict, using default values for missing fields.
        
        Args:
            data: Partial dictionary of values
            
        Returns:
            StateSchema instance with defaults filled in
        """
        # Get defaults from model fields
        full_data = {}
        for field_name, field_info in cls.model_fields.items():
            if field_info.default is not ...:
                full_data[field_name] = field_info.default
            elif field_info.default_factory is not None:
                full_data[field_name] = field_info.default_factory()
            else:
                # Required field, leave it to be filled by the data
                pass
                
        # Update with provided data
        full_data.update(data)
        
        # Create instance
        if hasattr(cls, "model_validate"):
            return cls.model_validate(full_data)
        else:
            return cls.parse_obj(full_data)
    
    @classmethod
    def model_validate(cls, data: Dict[str, Any], **kwargs) -> 'StateSchema':
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
                          if k not in ["__shared_fields__", "__serializable_reducers__", "__reducer_fields__"]}
        else:
            clean_data = data
        
        # Use parent validation method
        if hasattr(BaseModel, "model_validate"):
            # Pydantic v2
            return cast('StateSchema', super().model_validate(clean_data, **kwargs))
        else:
            # Pydantic v1
            return cast('StateSchema', super().parse_obj(clean_data))
    
    def __getstate__(self) -> Dict[str, Any]:
        """
        Custom state serialization for pickle compatibility.
        
        Returns:
            Dictionary of serializable state
        """
        # Get the normal state
        state = self.__dict__.copy()
        
        # Ensure proper serialization of class-level attributes
        # Convert reducer functions to their names for serialization
        reducer_names = {}
        for field, reducer in self.__class__.__reducer_fields__.items():
            reducer_names[field] = getattr(reducer, "__name__", str(reducer))
        
        # Add class-level attributes directly to the instance state
        state["__shared_fields__"] = self.__class__.__shared_fields__.copy()
        state["__serializable_reducers__"] = reducer_names
        
        return state
    
    def __setstate__(self, state: Dict[str, Any]) -> None:
        """
        Custom state deserialization for pickle compatibility.
        
        Args:
            state: State dictionary to restore from
        """
        # Extract special attributes
        shared_fields = state.pop("__shared_fields__", [])
        serializable_reducers = state.pop("__serializable_reducers__", {})
        
        # Set instance state
        self.__dict__.update(state)
        
        # Re-apply class attributes
        self.__class__.__shared_fields__ = shared_fields
        self.__class__.__serializable_reducers__ = serializable_reducers
    @classmethod
    def add_field(
        cls, 
        name: str, 
        field_type: Type, 
        default: Any = None, 
        default_factory: Optional[Callable[[], Any]] = None,
        description: Optional[str] = None,
        shared: bool = False,
        reducer: Optional[Callable] = None
    ) -> Type['StateSchema']:
            """
            Add a field to the schema class with support for default_factory.
            
            Args:
                name: Field name
                field_type: Field type
                default: Default value (mutually exclusive with default_factory)
                default_factory: Factory function to generate default values
                description: Optional description
                shared: Whether the field is shared with parent
                reducer: Optional reducer function
                
            Returns:
                Updated schema class
            """
            # Create field dict for the new model
            field_dict = {}
            
            # Copy existing fields
            for field_name, field_info in cls.model_fields.items():
                field_dict[field_name] = (field_info.annotation, field_info)
                
            # Create field information based on whether default or default_factory is provided
            field_kwargs = {"description": description}
            
            if default_factory is not None:
                field_kwargs["default_factory"] = default_factory
            else:
                field_kwargs["default"] = default
            
            # Add the new field
            field_dict[name] = (field_type, Field(**field_kwargs))
            
            # Create new model with all fields
            new_model = create_model(
                cls.__name__,
                __base__=cls,  # Use cls directly as base to maintain inheritance
                **field_dict
            )
            
            # Copy shared fields
            if hasattr(cls, "__shared_fields__"):
                new_model.__shared_fields__ = list(cls.__shared_fields__)
                # Add new field to shared fields if needed
                if shared and name not in new_model.__shared_fields__:
                    new_model.__shared_fields__.append(name)
                    
            # Copy reducers
            if hasattr(cls, "__serializable_reducers__"):
                new_model.__serializable_reducers__ = dict(cls.__serializable_reducers__)
                
            if hasattr(cls, "__reducer_fields__"):
                new_model.__reducer_fields__ = dict(cls.__reducer_fields__)
                
            # Add reducer if provided
            if reducer:
                if not hasattr(new_model, "__serializable_reducers__"):
                    new_model.__serializable_reducers__ = {}
                if not hasattr(new_model, "__reducer_fields__"):
                    new_model.__reducer_fields__ = {}
                    
                reducer_name = getattr(reducer, "__name__", str(reducer))
                new_model.__serializable_reducers__[name] = reducer_name
                new_model.__reducer_fields__[name] = reducer
                
            return new_model
    
    @classmethod
    def as_reducer(cls, field_name: str) -> Callable[[Any, Any], Any]:
        """
        Get the reducer function for a specific field.
        
        Args:
            field_name: Field name to get reducer for
            
        Returns:
            Reducer function
        """
        if field_name in cls.__reducer_fields__:
            return cls.__reducer_fields__[field_name]
        
        # Return simple replacement reducer as fallback
        return lambda old, new: new
    
    def pretty_print(self) -> None:
        """
        Print a formatted representation of this schema.
        """
        from haive_core.schema.schema_manager import StateSchemaManager
        StateSchemaManager(self.__class__).pretty_print()
    
    def copy(self, **updates) -> 'StateSchema':
        """
        Create a copy of this state, optionally with updates.
        
        Args:
            **updates: Field values to update in the copy
            
        Returns:
            New instance with the same values and any updates
        """
        # Use native Pydantic copy method if available
        if hasattr(self, "model_copy"):
            # Pydantic v2
            return self.model_copy(update=updates)
        else:
            # Pydantic v1
            data = self.dict()
            data.update(updates)
            return self.__class__(**data)
    
    def deep_copy(self) -> 'StateSchema':
        """
        Create a deep copy of this state object.
        
        Returns:
            New instance with deep copies of all values
        """
        return deepcopy(self)
    
    def apply_command(self, command: Any) -> 'StateSchema':
        """
        Apply a Command's update to this state.
        
        Args:
            command: Command with updates to apply
            
        Returns:
            Self for chaining
        """
        # Check if it's a Command object
        if hasattr(command, "update") and command.update is not None:
            # Apply the update
            for key, value in command.update.items():
                if key in self.__class__.__reducer_fields__:
                    # Use reducer if available
                    if hasattr(self, key):
                        current_value = getattr(self, key)
                        reducer = self.__class__.__reducer_fields__[key]
                        try:
                            reduced_value = reducer(current_value, value)
                            setattr(self, key, reduced_value)
                        except Exception as e:
                            logger.warning(f"Reducer failed for {key}: {e}")
                            setattr(self, key, value)
                    else:
                        setattr(self, key, value)
                else:
                    # No reducer, direct assignment
                    setattr(self, key, value)
        
        return self
    
    def to_json(self) -> str:
        """
        Convert to JSON string.
        
        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'StateSchema':
        """
        Create from JSON string.
        
        Args:
            json_str: JSON string representation
            
        Returns:
            StateSchema instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)
        
    @classmethod
    def combine(cls, *schemas: Type['StateSchema'], name: Optional[str] = None) -> Type['StateSchema']:
        """
        Combine multiple schemas into a new schema.
        
        Args:
            *schemas: StateSchema classes to combine
            name: Optional name for the combined schema
            
        Returns:
            New combined schema class
        """
        # Use first schema name if not provided
        if name is None and len(schemas) > 0:
            name = f"Combined{schemas[0].__name__}"
        elif name is None:
            name = "CombinedSchema"
            
        # Collect all fields
        all_fields = {}
        shared_fields = set()
        reducer_fields = {}
        
        for schema in schemas:
            # Add fields
            for field_name, field_info in schema.model_fields.items():
                # Skip if already added (first schema has priority)
                if field_name not in all_fields:
                    all_fields[field_name] = (field_info.annotation, field_info)
                    
            # Collect shared fields
            if hasattr(schema, "__shared_fields__"):
                shared_fields.update(schema.__shared_fields__)
                
            # Collect reducer fields
            if hasattr(schema, "__reducer_fields__"):
                for field, reducer in schema.__reducer_fields__.items():
                    if field not in reducer_fields:
                        reducer_fields[field] = reducer
                        
        # Create new model with all fields
        new_model = create_model(
            name,
            __base__=cls,
            **all_fields
        )
        
        # Set shared fields
        new_model.__shared_fields__ = list(shared_fields)
        
        # Set reducer fields
        new_model.__reducer_fields__ = reducer_fields
        
        # Set serializable reducer names
        serializable_reducers = {}
        for field, reducer in reducer_fields.items():
            reducer_name = getattr(reducer, "__name__", str(reducer))
            serializable_reducers[field] = reducer_name
            
        new_model.__serializable_reducers__ = serializable_reducers
        
        return new_model
        
    def get_field_names(self) -> List[str]:
        """
        Get a list of all field names in this schema.
        
        Returns:
            List of field names
        """
        return list(self.model_fields.keys())
        
    def remove_field(self, field_name: str) -> None:
        """
        Remove a field from this instance.
        
        Note: This only affects this instance, not the schema class.
        
        Args:
            field_name: Name of field to remove
        """
        if hasattr(self, field_name):
            delattr(self, field_name)