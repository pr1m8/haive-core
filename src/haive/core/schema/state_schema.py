"""
Core StateSchema foundation for the Haive framework.

This module defines the StateSchema class, which extends Pydantic's BaseModel with
capabilities for state management in graph-based systems, including field sharing,
value reduction, and serialization support.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Type, TypeVar, Generic, Union, get_origin, get_args, Annotated, Callable, Sequence, TYPE_CHECKING, Literal
from pydantic import BaseModel, Field, create_model
import copy
import inspect
import json
import logging
import operator
from langgraph.types import Command, Send
from langchain_core.messages import BaseMessage
from langgraph.graph import END
if TYPE_CHECKING:
    from haive.core.schema.schema_manager import StateSchemaManager

logger = logging.getLogger(__name__)

# TypeVar for the model type to be extended
T = TypeVar('T', bound=BaseModel)


class StateSchema(BaseModel, Generic[T]):
    """
    Enhanced base class for state schemas in the Haive framework.
    
    A StateSchema extends Pydantic's BaseModel with additional capabilities:
    - Field sharing between parent/child graphs
    - Reducer functions for merging field values
    - Serialization support for database storage
    - LangGraph integration for stateful agents
    - Engine input/output field tracking
    
    StateSchema can be parameterized with a specific model type to provide
    type hints for state fields while maintaining all state management capabilities.
    
    Example:
        class MyState(BaseModel):
            value: str
            count: int
        
        state = StateSchema[MyState](value="test", count=5)
    
    Attributes:
        __shared_fields__ (List[str]): Fields shared with parent graph.
        __serializable_reducers__ (Dict[str, str]): Mapping of field names to reducer function names.
        __engine_io_mappings__ (Dict[str, Dict[str, List[str]]]): Engine I/O field mappings.
        __input_fields__ (Dict[str, List[str]]): Input fields for each engine.
        __output_fields__ (Dict[str, List[str]]): Output fields for each engine.
    """
    
    # Class variables to track field sharing and reducers
    __shared_fields__: List[str] = []
    __serializable_reducers__: Dict[str, str] = {}
    __engine_io_mappings__: Dict[str, Dict[str, List[str]]] = {}
    __input_fields__: Dict[str, List[str]] = {}
    __output_fields__: Dict[str, List[str]] = {}
    
    # Note: __reducer_fields__ is created dynamically and not part of instance properties
    # because it causes serialization issues with storing actual callable functions
    
    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """
        Override model_dump to exclude internal fields.
        
        Args:
            **kwargs: Arguments to pass to the parent method
            
        Returns:
            Dictionary representation without internal fields
        """
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
        """
        Backwards compatibility alias for model_dump.
        
        Args:
            **kwargs: Arguments to pass to model_dump
            
        Returns:
            Dictionary representation without internal fields
        """
        return self.model_dump(**kwargs)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the state to a clean dictionary.
        
        Returns:
            Dictionary representation of the state
        """
        return self.model_dump()
    
    def to_json(self) -> str:
        """
        Convert state to JSON string.
        
        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'StateSchema':
        """
        Create state from JSON string.
        
        Args:
            json_str: JSON string representation
            
        Returns:
            StateSchema instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StateSchema':
        """
        Create a state from a dictionary.
        
        Args:
            data: Dictionary to convert
            
        Returns:
            StateSchema instance
        """
        # Filter out internal fields if present
        internal_fields = ["__shared_fields__", "__serializable_reducers__", "__reducer_fields__",
                          "__engine_io_mappings__", "__input_fields__", "__output_fields__"]
        clean_data = {k: v for k, v in data.items() if k not in internal_fields}
        
        # Use Pydantic v2 method for validation
        return cls.model_validate(clean_data)
    
    @classmethod
    def from_partial_dict(cls, data: Dict[str, Any]) -> 'StateSchema':
        """
        Create a state from a partial dictionary, filling in defaults.
        
        Args:
            data: Partial dictionary of values
            
        Returns:
            StateSchema instance with defaults filled in
        """
        # Get defaults from model fields - Pydantic v2
        full_data = {}
        fields_dict = cls.model_fields
            
        for field_name, field_info in fields_dict.items():
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
            data = other.model_dump()
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
        """
        Merge new messages with existing messages using appropriate reducer.
        
        Args:
            new_messages: New messages to add
            
        Returns:
            Self for chaining
        """
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
        """
        Create a copy of this state, optionally with updates.
        
        Args:
            **updates: Field values to update in the copy
            
        Returns:
            New instance with the same values and any updates
        """
        # Use Pydantic v2 model_copy
        return self.model_copy(update=updates)
    
    def deep_copy(self) -> 'StateSchema':
        """
        Create a deep copy of this state object.
        
        Returns:
            New instance with deep copies of all values
        """
        return copy.deepcopy(self)
    
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
            # Apply the update with reducers
            self.apply_reducers(command.update)
                
        return self
    
    @classmethod
    def _get_reducer_registry(cls) -> Dict[str, Callable]:
        """
        Get a registry of reducer functions mapped to their names.
        This approach avoids storing functions directly in the class.
        
        Returns:
            Dictionary mapping reducer names to functions
        """
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
        
        # Add custom reducers with qualified names
        for field, reducer_name in cls.__serializable_reducers__.items():
            if "." in reducer_name and reducer_name not in registry:
                # Try to import and resolve the reducer function
                try:
                    module_name, func_name = reducer_name.rsplit(".", 1)
                    module = __import__(module_name, fromlist=[func_name])
                    registry[reducer_name] = getattr(module, func_name)
                except (ImportError, AttributeError):
                    # If import fails, we can't resolve this reducer
                    pass
        
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
    def has_reducer(cls, field_name: str) -> bool:
        """
        Check if a field has a reducer function.
        
        Args:
            field_name: Name of the field to check
            
        Returns:
            True if field has a reducer, False otherwise
        """
        return field_name in cls.__serializable_reducers__
    
    @classmethod
    def as_reducer(cls, field_name: str) -> Callable[[Any, Any], Any]:
        """
        Get the reducer function for a specific field.
        
        Args:
            field_name: Field name to get reducer for
            
        Returns:
            Reducer function
        """
        # Look up the reducer from the registry
        reducer_registry = cls._get_reducer_registry()
        
        if field_name in cls.__serializable_reducers__:
            reducer_name = cls.__serializable_reducers__[field_name]
            if reducer_name in reducer_registry:
                return reducer_registry[reducer_name]
        
        # Return simple replacement reducer as fallback
        return lambda old, new: new
    
    @classmethod
    def reducer_fields(cls) -> Dict[str, Callable]:
        """
        Get a dictionary of field reducers.
        This builds a dictionary by looking up serializable reducer names in the registry.
        
        Returns:
            Dictionary mapping field names to reducer functions
        """
        reducer_registry = cls._get_reducer_registry()
        result = {}
        
        for field, reducer_name in cls.__serializable_reducers__.items():
            if reducer_name in reducer_registry:
                result[field] = reducer_registry[reducer_name]
        
        return result
    
    @classmethod
    def create(cls, **field_definitions) -> Type['StateSchema']:
        """
        Create a new StateSchema class with the given field definitions.
        
        This is a factory method for simple schema creation. For more complex schema
        manipulation, use the to_manager() method to get a StateSchemaManager.
        
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
        model.__engine_io_mappings__ = {}
        model.__input_fields__ = {}
        model.__output_fields__ = {}
        
        # Create the __reducer_fields__ dict if it doesn't exist yet
        if not hasattr(model, "__reducer_fields__"):
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
                    # Store reducer function
                    model.__reducer_fields__[field_name] = reducer
                    
                    # Store serializable reducer name with special handling for operator module
                    if hasattr(reducer, "__module__") and reducer.__module__ == 'operator':
                        model.__serializable_reducers__[field_name] = f"operator.{reducer.__name__}"
                    else:
                        model.__serializable_reducers__[field_name] = getattr(reducer, "__name__", str(reducer))
        
        # Special handling for messages field - add add_messages reducer if not already set
        if "messages" in field_definitions and "messages" not in model.__serializable_reducers__:
            try:
                from langgraph.graph import add_messages
                # Store only the name, not the function itself
                model.__serializable_reducers__["messages"] = "add_messages"
                model.__reducer_fields__["messages"] = add_messages
            except ImportError:
                # Default fallback for messages reducer
                def concat_lists(a, b):
                    return (a or []) + (b or [])
                
                model.__serializable_reducers__["messages"] = "concat_lists"
                model.__reducer_fields__["messages"] = concat_lists
        
        return model
    
    @classmethod
    def with_messages(cls) -> Type['StateSchema']:
        """
        Create a new schema with messages field using an appropriate reducer.
        
        Returns:
            New schema class with messages field
        """
        # Create schema with messages field, attempting to use add_messages reducer
        try:
            from langgraph.graph import add_messages
            from typing import Sequence
            
            schema = cls.create(
                messages=(Annotated[Sequence[BaseMessage], add_messages], [])
            )
            
            # Ensure only the name is stored, not the function
            schema.__serializable_reducers__["messages"] = "add_messages"
            
            # Create the actual reducer field dictionary if not exists
            if not hasattr(schema, "__reducer_fields__"):
                schema.__reducer_fields__ = {}
            schema.__reducer_fields__["messages"] = add_messages
            
            return schema
        except ImportError:
            # Fallback without reducer
            from typing import Sequence
            
            # Create a simple concat function as fallback
            def concat_lists(a, b):
                return (a or []) + (b or [])
            
            schema = cls.create(
                messages=(Sequence[BaseMessage], [])
            )
            
            schema.__serializable_reducers__["messages"] = "concat_lists"
            
            # Create the actual reducer field dictionary if not exists
            if not hasattr(schema, "__reducer_fields__"):
                schema.__reducer_fields__ = {}
            schema.__reducer_fields__["messages"] = concat_lists
            
            return schema
    
    @classmethod
    def to_manager(cls, name: Optional[str] = None) -> "StateSchemaManager":
        """
        Convert schema class to a StateSchemaManager for further manipulation.
        
        Args:
            name: Optional name for the manager
            
        Returns:
            StateSchemaManager instance
        """
        from haive.core.schema.schema_manager import StateSchemaManager
        return StateSchemaManager(cls, name=name or cls.__name__)
    
    @classmethod
    def manager(cls) -> "StateSchemaManager":
        """
        Get a manager for this schema (shorthand for to_manager()).
        
        Returns:
            StateSchemaManager instance
        """
        return cls.to_manager()
    
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
        # Use provided name or default
        if name is None and len(schemas) > 0:
            name = f"Combined{schemas[0].__name__}"
        elif name is None:
            name = "CombinedSchema"
            
        # Collect all fields
        all_fields = {}
        shared_fields = set()
        serializable_reducers = {}
        reducer_fields = {}
        engine_io_mappings = {}
        input_fields = {}
        output_fields = {}
        
        for schema in schemas:
            # Add fields - Pydantic v2
            fields_dict = schema.model_fields
            for field_name, field_info in fields_dict.items():
                # Skip if already added (first schema has priority)
                if field_name not in all_fields:
                    field_dict = (field_info.annotation, field_info)
                    all_fields[field_name] = field_dict
                    
            # Collect shared fields
            if hasattr(schema, "__shared_fields__"):
                shared_fields.update(schema.__shared_fields__)
                
            # Collect serializable reducers
            if hasattr(schema, "__serializable_reducers__"):
                for field, reducer_name in schema.__serializable_reducers__.items():
                    if field not in serializable_reducers:
                        serializable_reducers[field] = reducer_name
                        
            # Collect actual reducer functions
            if hasattr(schema, "__reducer_fields__"):
                for field, reducer in schema.__reducer_fields__.items():
                    if field not in reducer_fields:
                        reducer_fields[field] = reducer
                        
            # Collect engine I/O mappings
            if hasattr(schema, "__engine_io_mappings__"):
                for engine, mapping in schema.__engine_io_mappings__.items():
                    if engine not in engine_io_mappings:
                        engine_io_mappings[engine] = mapping.copy()
                        
            if hasattr(schema, "__input_fields__"):
                for engine, fields in schema.__input_fields__.items():
                    if engine not in input_fields:
                        input_fields[engine] = list(fields)
                    else:
                        input_fields[engine].extend([f for f in fields if f not in input_fields[engine]])
                        
            if hasattr(schema, "__output_fields__"):
                for engine, fields in schema.__output_fields__.items():
                    if engine not in output_fields:
                        output_fields[engine] = list(fields)
                    else:
                        output_fields[engine].extend([f for f in fields if f not in output_fields[engine]])
                        
        # Create new model with all fields
        new_model = create_model(
            name,
            __base__=cls,
            **all_fields
        )
        
        # Set shared fields
        new_model.__shared_fields__ = list(shared_fields)
        
        # Set serializable reducers (names only, not functions)
        new_model.__serializable_reducers__ = serializable_reducers
        
        # Set actual reducer functions in a separate attribute
        new_model.__reducer_fields__ = reducer_fields
        
        # Set engine I/O mappings
        new_model.__engine_io_mappings__ = engine_io_mappings
        new_model.__input_fields__ = input_fields
        new_model.__output_fields__ = output_fields
        
        return new_model
    
    @classmethod
    def pretty_print(cls) -> None:
        """
        Pretty print the schema definition.
        """
        from haive.core.schema.schema_manager import StateSchemaManager
        manager = StateSchemaManager(cls)
        manager.pretty_print()
    
    def remove_field(self, field_name: str) -> None:
        """
        Remove a field from this instance.
        
        Note: This only affects this instance, not the schema class.
        
        Args:
            field_name: Name of field to remove
        """
        if hasattr(self, field_name):
            delattr(self, field_name)
    
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
            internal_fields = ["__shared_fields__", "__serializable_reducers__", "__reducer_fields__",
                              "__engine_io_mappings__", "__input_fields__", "__output_fields__"]
            clean_data = {k: v for k, v in data.items() if k not in internal_fields}
        else:
            clean_data = data
        
        # Use parent validation method for Pydantic v2
        return super().model_validate(clean_data, **kwargs)
    
    def __getstate__(self) -> Dict[str, Any]:
        """
        Custom state serialization for pickle compatibility.
        
        Returns:
            Dictionary of serializable state
        """
        # Get the normal state
        state = self.__dict__.copy()
        
        # Add class-level attributes directly to the instance state
        state["__shared_fields__"] = self.__class__.__shared_fields__.copy()
        state["__serializable_reducers__"] = self.__class__.__serializable_reducers__.copy()
        state["__engine_io_mappings__"] = dict(getattr(self.__class__, "__engine_io_mappings__", {}))
        state["__input_fields__"] = dict(getattr(self.__class__, "__input_fields__", {}))
        state["__output_fields__"] = dict(getattr(self.__class__, "__output_fields__", {}))
        
        # Do NOT include __reducer_fields__ as it contains non-serializable functions
        
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
        engine_io_mappings = state.pop("__engine_io_mappings__", {})
        input_fields = state.pop("__input_fields__", {})
        output_fields = state.pop("__output_fields__", {})
        
        # Set instance state
        self.__dict__.update(state)
        
        # Re-apply class attributes
        self.__class__.__shared_fields__ = shared_fields
        self.__class__.__serializable_reducers__ = serializable_reducers
        self.__class__.__engine_io_mappings__ = engine_io_mappings
        self.__class__.__input_fields__ = input_fields
        self.__class__.__output_fields__ = output_fields
    
    @classmethod
    def get_input_fields(cls, engine_name: Optional[str] = None) -> List[str]:
        """
        Get input fields for a specific engine or all engines.
        
        Args:
            engine_name: Optional engine name to filter by
            
        Returns:
            List of input field names
        """
        if not hasattr(cls, "__input_fields__"):
            return []
            
        if engine_name:
            return cls.__input_fields__.get(engine_name, [])
        else:
            # Get all unique input fields across all engines
            result = []
            for fields in cls.__input_fields__.values():
                for field in fields:
                    if field not in result:
                        result.append(field)
            return result
            
    @classmethod
    def get_output_fields(cls, engine_name: Optional[str] = None) -> List[str]:
        """
        Get output fields for a specific engine or all engines.
        
        Args:
            engine_name: Optional engine name to filter by
            
        Returns:
            List of output field names
        """
        if not hasattr(cls, "__output_fields__"):
            return []
            
        if engine_name:
            return cls.__output_fields__.get(engine_name, [])
        else:
            # Get all unique output fields across all engines
            result = []
            for fields in cls.__output_fields__.values():
                for field in fields:
                    if field not in result:
                        result.append(field)
            return result
            
    @classmethod
    def get_engine_io_mappings(cls) -> Dict[str, Dict[str, List[str]]]:
        """
        Get input/output field mappings for each engine.
        
        Returns:
            Dictionary mapping engine names to their input/output field lists
        """
        if hasattr(cls, "__engine_io_mappings__"):
            return cls.__engine_io_mappings__.copy()
        return {}
