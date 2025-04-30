from __future__ import annotations
from typing import Any, Dict, List, Optional, Type, TypeVar, Generic, \
    Union, get_origin, get_args, Callable, Sequence, TYPE_CHECKING
from pydantic import BaseModel, Field, create_model
import copy
import json
import logging
import inspect
from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from haive.core.schema.schema_manager import StateSchemaManager

T = TypeVar('T', bound=BaseModel)

class StateSchema(BaseModel, Generic[T]):
    """
    Enhanced base class for state schemas in the Haive framework.
    
    StateSchema extends Pydantic's BaseModel with features for:
    - Field sharing between parent and child graphs
    - Reducer functions for combining field values
    - Input/output tracking for engines
    - Message handling utilities
    - Serialization and deserialization support
    - State manipulation utilities
    """
    
    # Class variables to track field sharing and reducers
    __shared_fields__: List[str] = []
    __serializable_reducers__: Dict[str, str] = {}
    __engine_io_mappings__: Dict[str, Dict[str, List[str]]] = {}
    __input_fields__: Dict[str, List[str]] = {}
    __output_fields__: Dict[str, List[str]] = {}
    __structured_models__: Dict[str, Type[BaseModel]] = {}
    __structured_model_fields__: Dict[str, List[str]] = {}
    
    # Note: __reducer_fields__ is created dynamically and not part of instance properties
    
    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Override model_dump to exclude internal fields."""
        # Get the base model_dump result from Pydantic v2
        data = super().model_dump(**kwargs)
            
        # Filter out internal fields
        internal_fields = ["__shared_fields__", "__serializable_reducers__", "__reducer_fields__",
                          "__engine_io_mappings__", "__input_fields__", "__output_fields__",
                          "__structured_models__", "__structured_model_fields__"]
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
                          "__engine_io_mappings__", "__input_fields__", "__output_fields__",
                          "__structured_models__", "__structured_model_fields__"]
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
        """
        Update state applying reducer functions where defined.
        
        This method processes updates with special handling for fields
        that have reducer functions defined.
        
        Args:
            other: Dictionary or StateSchema with update values
            
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
    
    def add_message(self, message: BaseMessage) -> 'StateSchema':
        """
        Add a single message to the messages field.
        
        Args:
            message: BaseMessage to add
            
        Returns:
            Self for chaining
        """
        if not hasattr(self, 'messages'):
            # Create messages field if it doesn't exist
            self.messages = [message]
            return self
        
        # Check if we're using a reducer
        reducer_fields = getattr(self.__class__, "__reducer_fields__", {})
        if 'messages' in reducer_fields:
            # Use the reducer with a single-item list
            self.messages = reducer_fields['messages'](self.messages, [message])
        else:
            # Simple append
            if isinstance(self.messages, list):
                self.messages.append(message)
            else:
                self.messages = [message]
                
        return self

    def add_messages(self, new_messages: List[BaseMessage]) -> 'StateSchema':
        """
        Add multiple messages to the messages field.
        
        Args:
            new_messages: List of messages to add
            
        Returns:
            Self for chaining
        """
        if not hasattr(self, 'messages'):
            # Create messages field if it doesn't exist
            self.messages = list(new_messages)  # Create a copy
            return self
        
        # Check if we're using a reducer
        reducer_fields = getattr(self.__class__, "__reducer_fields__", {})
        if 'messages' in reducer_fields:
            # Use the reducer
            self.messages = reducer_fields['messages'](self.messages, new_messages)
        else:
            # Simple extend
            if isinstance(self.messages, list):
                self.messages.extend(new_messages)
            else:
                self.messages = list(new_messages)
                
        return self
        
    def merge_messages(self, new_messages: List[BaseMessage]) -> 'StateSchema':
        """
        Merge new messages with existing messages using appropriate reducer.
        
        Args:
            new_messages: New messages to add
            
        Returns:
            Self for chaining
        """
        return self.add_messages(new_messages)
    
    def clear_messages(self) -> 'StateSchema':
        """
        Clear all messages in the messages field.
        
        Returns:
            Self for chaining
        """
        if hasattr(self, 'messages'):
            self.messages = []
        return self
        
    def get_last_message(self) -> Optional[BaseMessage]:
        """
        Get the last message in the messages field.
        
        Returns:
            Last message or None if no messages exist
        """
        if hasattr(self, 'messages') and self.messages:
            return self.messages[-1]
        return None
    
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
    
    @classmethod
    def create_input_schema(
        cls,
        name: str = "Input",
        message_field: str = "input",
        include_messages: bool = True,
        include_config: bool = True,
        additional_fields: Optional[Dict[str, Any]] = None
    ) -> Type[BaseModel]:
        """
        Create an input schema model based on this state schema.
        
        Args:
            name: Schema name
            message_field: Name of the primary input message field
            include_messages: Whether to include a messages field
            include_config: Whether to include a runnable_config field
            additional_fields: Any additional fields to include
            
        Returns:
            Input schema model
        """
        from typing import Optional, List, Dict, Any, Union
        
        # Set up field dict
        fields = {}
        
        # Add primary input field
        fields[message_field] = (str, Field(description="Primary input to the agent"))
        
        # Add messages field if requested
        if include_messages:
            fields["messages"] = (Optional[List[BaseMessage]], Field(
                default=None,
                description="List of messages for the conversation"
            ))
        
        # Add runnable_config field if requested
        if include_config:
            fields["runnable_config"] = (Optional[Dict[str, Any]], Field(
                default=None,
                description="Runtime configuration"
            ))
        
        # Add additional fields
        if additional_fields:
            for field_name, field_config in additional_fields.items():
                if isinstance(field_config, tuple) and len(field_config) >= 2:
                    field_type, default = field_config[0], field_config[1]
                    fields[field_name] = (
                        field_type, 
                        Field(default=default, description=f"Input field: {field_name}")
                    )
                else:
                    fields[field_name] = (
                        type(field_config),
                        Field(default=field_config, description=f"Input field: {field_name}")
                    )
        
        # Create the model
        return create_model(name, **fields)
    
    @classmethod
    def create_output_schema(
        cls,
        name: str = "Output",
        message_field: str = "output",
        include_messages: bool = False,
        include_metadata: bool = True,
        structured_output: Optional[Type[BaseModel]] = None,
        additional_fields: Optional[Dict[str, Any]] = None
    ) -> Type[BaseModel]:
        """
        Create an output schema model based on this state schema.
        
        Args:
            name: Schema name
            message_field: Name of the primary output message field
            include_messages: Whether to include messages in output
            include_metadata: Whether to include execution metadata
            structured_output: Optional structured output model
            additional_fields: Any additional fields to include
            
        Returns:
            Output schema model
        """
        from typing import Optional, List, Dict, Any, Union
        
        # Set up field dict
        fields = {}
        
        # Add primary output field
        fields[message_field] = (str, Field(description="Primary output"))
        
        # Add messages field if requested
        if include_messages:
            fields["messages"] = (Optional[List[BaseMessage]], Field(
                default=None,
                description="List of messages from the conversation"
            ))
        
        # Add metadata if requested
        if include_metadata:
            fields["metadata"] = (Dict[str, Any], Field(
                default_factory=dict,
                description="Metadata about the execution"
            ))
        
        # Add structured output field if provided
        if structured_output:
            model_name = structured_output.__name__.lower()
            fields[model_name] = (Optional[structured_output], Field(
                default=None,
                description=f"Structured output in {structured_output.__name__} format"
            ))
        
        # Add additional fields
        if additional_fields:
            for field_name, field_config in additional_fields.items():
                if isinstance(field_config, tuple) and len(field_config) >= 2:
                    field_type, default = field_config[0], field_config[1]
                    fields[field_name] = (
                        field_type, 
                        Field(default=default, description=f"Output field: {field_name}")
                    )
                else:
                    fields[field_name] = (
                        type(field_config),
                        Field(default=field_config, description=f"Output field: {field_name}")
                    )
        
        # Create the model
        return create_model(name, **fields)
    
    @classmethod
    def pretty_print(cls, include_metadata: bool = True, show_defaults: bool = True) -> str:
        """
        Generate a formatted, readable representation of the schema.
        
        Args:
            include_metadata: Whether to include field sharing, reducers, and engine I/O info
            show_defaults: Whether to show default values for fields
            
        Returns:
            Formatted string representation of the schema
        """
        output = []
        
        # Add class definition header
        output.append(f"# {cls.__name__} Schema Definition")
        output.append(f"class {cls.__name__}(StateSchema):")
        
        # Add fields by category
        fields_by_category = {
            "Standard Fields": [],
            "Shared Fields": [],
            "Fields with Reducers": []
        }
        
        # Get shared fields and reducer info
        shared_fields = getattr(cls, "__shared_fields__", [])
        reducer_names = getattr(cls, "__serializable_reducers__", {})
        
        # Organize fields by category
        for field_name, field_info in cls.model_fields.items():
            # Skip special fields
            if field_name.startswith("__"):
                continue
                
            # Determine field representation
            field_type = field_info.annotation.__name__ if hasattr(field_info.annotation, "__name__") else str(field_info.annotation)
            
            # Format default value if requested
            default_str = ""
            if show_defaults:
                if field_info.default_factory is not None:
                    default_factory_name = field_info.default_factory.__name__ if hasattr(field_info.default_factory, "__name__") else "factory"
                    default_str = f" = Field(default_factory={default_factory_name})"
                elif field_info.default is not ...:
                    if field_info.default is None:
                        default_str = " = None"
                    elif isinstance(field_info.default, str):
                        default_str = f' = "{field_info.default}"'
                    else:
                        default_str = f" = {field_info.default}"
            
            # Create field representation
            field_repr = f"    {field_name}: {field_type}{default_str}"
            
            # Add description if available
            description = getattr(field_info, "description", None)
            if description:
                field_repr = f"    # {description}\n{field_repr}"
                
            # Categorize the field
            if field_name in reducer_names:
                fields_by_category["Fields with Reducers"].append((field_name, field_repr))
            elif field_name in shared_fields:
                fields_by_category["Shared Fields"].append((field_name, field_repr))
            else:
                fields_by_category["Standard Fields"].append((field_name, field_repr))
        
        # Add fields to output by category
        for category, fields in fields_by_category.items():
            if fields:
                output.append("")
                output.append(f"    # {category}")
                # Sort fields by name within category
                for _, field_repr in sorted(fields):
                    output.append(field_repr)
        
        # Add metadata sections if requested
        if include_metadata:
            # Add input/output fields by engine
            engine_io = getattr(cls, "__engine_io_mappings__", {})
            if engine_io:
                output.append("")
                output.append("    # Engine I/O Mappings")
                for engine_name, mapping in sorted(engine_io.items()):
                    inputs = mapping.get("inputs", [])
                    outputs = mapping.get("outputs", [])
                    if inputs:
                        output.append(f"    # {engine_name} inputs: {', '.join(inputs)}")
                    if outputs:
                        output.append(f"    # {engine_name} outputs: {', '.join(outputs)}")
            
            # Add reducer information
            if reducer_names:
                output.append("")
                output.append("    # Reducer Functions")
                for field_name, reducer_name in sorted(reducer_names.items()):
                    output.append(f"    # {field_name}: {reducer_name}")
        
        # Add method list
        methods = []
        for name, attr in inspect.getmembers(cls):
            if name.startswith("__") or name in cls.model_fields:
                continue
            if inspect.isfunction(attr) or inspect.ismethod(attr):
                methods.append(name)
        
        if methods:
            output.append("")
            output.append("    # Methods")
            for method in sorted(methods):
                output.append(f"    # {method}()")
                
        return "\n".join(output)
    
    def __str__(self) -> str:
        """Enhanced string representation with proper formatting."""
        fields = []
        
        # Add fields with values
        for field_name, field_value in self:
            if field_name.startswith("__"):
                continue
                
            # Format the value
            if isinstance(field_value, str):
                value_repr = f'"{field_value}"'
            elif isinstance(field_value, list):
                if not field_value:
                    value_repr = "[]"
                else:
                    if len(field_value) > 3:
                        value_repr = f"[...{len(field_value)} items...]"
                    else:
                        value_repr = str(field_value)
            else:
                value_repr = str(field_value)
                
            # Truncate long values
            if len(value_repr) > 50:
                value_repr = value_repr[:47] + "..."
                
            fields.append(f"{field_name}={value_repr}")
        
        # Create the representation
        return f"{self.__class__.__name__}({', '.join(fields)})"