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
    def create_input_schema(cls, name: str = "Input") -> Type[BaseModel]:
        """
        Create an input schema model based on identified input fields.
        
        This method generates a Pydantic model containing only the fields
        that were identified as inputs to engines during schema composition.
        
        Args:
            name: Name for the schema class
            
        Returns:
            A Pydantic model for input validation
        """
        # Collect all input fields across all engines
        input_fields = {}
        for engine_name, mapping in cls.__engine_io_mappings__.items():
            for field_name in mapping.get("inputs", []):
                # Only include if it exists in our fields
                if field_name in cls.model_fields:
                    field_def = cls.model_fields[field_name]
                    # Extract type and field info directly
                    field_type = field_def.annotation
                    
                    # Create Field with appropriate parameters
                    if field_def.default_factory is not None:
                        field_info = Field(
                            default_factory=field_def.default_factory,
                            description=field_def.description
                        )
                    else:
                        field_info = Field(
                            default=field_def.default,
                            description=field_def.description
                        )
                    
                    input_fields[field_name] = (field_type, field_info)
        
        # Create the model
        input_schema = create_model(name, **input_fields)
        
        return input_schema
    @classmethod
    def derive_input_schema(cls, engine_name: Optional[str] = None, name: Optional[str] = None) -> Type[BaseModel]:
        """
        Derive an input schema for the given engine from this state schema.
        
        Args:
            engine_name: Optional name of the engine to target (default: all inputs)
            name: Optional name for the schema class
            
        Returns:
            A BaseModel subclass for input validation
        """
        fields = {}
        
        # Get input field names
        if engine_name is not None and hasattr(cls, "__engine_io_mappings__"):
            if engine_name in cls.__engine_io_mappings__:
                input_fields = cls.__engine_io_mappings__[engine_name].get("inputs", [])
            else:
                input_fields = []
        elif hasattr(cls, "__input_fields__"):
            # Collect input fields across all engines
            input_fields = []
            for engine_inputs in cls.__input_fields__.values():
                input_fields.extend(engine_inputs)
        else:
            input_fields = []
        
        # Add input fields to schema
        for field_name in input_fields:
            if field_name in cls.model_fields:
                field_info = cls.model_fields[field_name]
                fields[field_name] = (field_info.annotation, field_info)
        
        # Create model
        schema_name = name or f"{cls.__name__}Input"
        return create_model(schema_name, **fields)

    @classmethod
    def derive_output_schema(cls, engine_name: Optional[str] = None, name: Optional[str] = None) -> Type[BaseModel]:
        """
        Derive an output schema for the given engine from this state schema.
        
        Args:
            engine_name: Optional name of the engine to target (default: all outputs)
            name: Optional name for the schema class
            
        Returns:
            A BaseModel subclass for output validation
        """
        fields = {}
        
        # Get output field names
        if engine_name is not None and hasattr(cls, "__engine_io_mappings__"):
            if engine_name in cls.__engine_io_mappings__:
                output_fields = cls.__engine_io_mappings__[engine_name].get("outputs", [])
            else:
                output_fields = []
        elif hasattr(cls, "__output_fields__"):
            # Collect output fields across all engines
            output_fields = []
            for engine_outputs in cls.__output_fields__.values():
                output_fields.extend(engine_outputs)
        else:
            output_fields = []
        
        # Add output fields to schema
        for field_name in output_fields:
            if field_name in cls.model_fields:
                field_info = cls.model_fields[field_name]
                fields[field_name] = (field_info.annotation, field_info)
        
        # Create model
        schema_name = name or f"{cls.__name__}Output"
        return create_model(schema_name, **fields)

    @classmethod
    def derive_engine_mappings(cls, engine_name: str) -> tuple[dict[str, str], dict[str, str]]:
        """
        Derive input and output mappings for an engine.
        
        Args:
            engine_name: Name of the engine to create mappings for
            
        Returns:
            Tuple of (input_mapping, output_mapping) dictionaries
        """
        input_mapping = {}
        output_mapping = {}
        
        if hasattr(cls, "__engine_io_mappings__") and engine_name in cls.__engine_io_mappings__:
            # Create identity mappings for all inputs
            for field in cls.__engine_io_mappings__[engine_name].get("inputs", []):
                input_mapping[field] = field
            
            # Create identity mappings for all outputs
            for field in cls.__engine_io_mappings__[engine_name].get("outputs", []):
                output_mapping[field] = field
        
        return input_mapping, output_mapping
    @classmethod
    def create_output_schema(cls, name: str = "Output") -> Type[BaseModel]:
        """
        Create an output schema model based on identified output fields.
        
        This method generates a Pydantic model containing only the fields
        that were identified as outputs from engines during schema composition.
        
        Args:
            name: Name for the schema class
            
        Returns:
            A Pydantic model for output validation
        """
        # Collect all output fields across all engines
        output_fields = {}
        for engine_name, mapping in cls.__engine_io_mappings__.items():
            for field_name in mapping.get("outputs", []):
                # Only include if it exists in our fields
                if field_name in cls.model_fields:
                    field_def = cls.model_fields[field_name]
                    # Extract type and field info directly
                    field_type = field_def.field_type
                    
                    # Create Field with appropriate parameters
                    if field_def.default_factory is not None:
                        field_info = Field(
                            default_factory=field_def.default_factory,
                            description=field_def.description
                        )
                    else:
                        field_info = Field(
                            default=field_def.default,
                            description=field_def.description
                        )
                    
                    output_fields[field_name] = (field_type, field_info)
        
        # Create the model
        output_schema = create_model(name, **output_fields)
        
        return output_schema
    @classmethod
    def pretty_print(cls, include_metadata: bool = True, show_defaults: bool = True, 
                    include_reducers: bool = True, format_style: str = "detailed") -> str:
        """
        Generate a formatted, readable representation of the schema.
        
        Args:
            include_metadata: Whether to include field sharing, reducers, and engine I/O info
            show_defaults: Whether to show default values for fields
            include_reducers: Whether to show reducer functions for fields
            format_style: Output style - "detailed", "compact", or "code"
            
        Returns:
            Formatted string representation of the schema
        """
        output = []
        
        # Schema header with more information
        schema_name = cls.__name__
        output.append(f"StateSchema: {schema_name}")
        
        # Add description if available
        if hasattr(cls, "__doc__") and cls.__doc__ and cls.__doc__.strip():
            doc = cls.__doc__.strip()
            output.append(f"Description: {doc}")
        output.append("")
        
        # Get metadata information if needed
        shared_fields = getattr(cls, "__shared_fields__", [])
        reducer_functions = getattr(cls, "__reducer_fields__", {})
        reducer_names = getattr(cls, "__serializable_reducers__", {})
        engine_io_mappings = getattr(cls, "__engine_io_mappings__", {})
        structured_models = getattr(cls, "__structured_models__", {})
        
        # Process all fields
        fields_section = []
        for field_name, field_info in cls.model_fields.items():
            # Skip special fields
            if field_name.startswith("__"):
                continue
            
            # Format field with type and metadata
            field_type = field_info.annotation
            type_str = cls._format_full_type(field_type)
            
            # Get field description
            description = getattr(field_info, "description", None) or ""
            
            # Format field entry based on style
            if format_style == "compact":
                # Compact style - just field and type
                field_entry = f"{field_name}: {type_str}"
                if show_defaults:
                    if field_info.default_factory is not None:
                        field_entry += f" = Field(default_factory={field_info.default_factory.__name__})"
                    elif field_info.default is not ...:
                        field_entry += f" = {repr(field_info.default)}"
                fields_section.append(field_entry)
                
            elif format_style == "code":
                # Code style - format as Python code
                field_entry = f"{field_name}: {type_str}"
                if show_defaults:
                    if field_info.default_factory is not None:
                        field_entry += f" = Field(default_factory={field_info.default_factory.__name__})"
                    elif field_info.default is not ...:
                        if field_info.default is None:
                            field_entry += " = None"
                        else:
                            field_entry += f" = {repr(field_info.default)}"
                
                # Add metadata as comments
                metadata_parts = []
                if field_name in shared_fields:
                    metadata_parts.append("shared")
                if field_name in reducer_functions:
                    reducer = reducer_functions[field_name]
                    reducer_name = getattr(reducer, "__name__", str(reducer))
                    metadata_parts.append(f"reducer={reducer_name}")
                if description:
                    metadata_parts.append(description)
                    
                if metadata_parts and include_metadata:
                    field_entry += f"  # {', '.join(metadata_parts)}"
                    
                fields_section.append(field_entry)
                
            else:  # Default to detailed style
                # Detailed style - multi-line with metadata
                field_parts = [f"• {field_name}: {type_str}"]
                
                # Add default value if requested
                if show_defaults:
                    if field_info.default_factory is not None:
                        factory_name = field_info.default_factory.__name__
                        field_parts.append(f"  Default: Created by {factory_name}()")
                    elif field_info.default is not ...:
                        field_parts.append(f"  Default: {repr(field_info.default)}")
                
                # Add description if available
                if description:
                    field_parts.append(f"  Description: {description}")
                
                # Add metadata if requested
                if include_metadata:
                    if field_name in shared_fields:
                        field_parts.append("  Shared: Yes")
                        
                    if field_name in reducer_functions and include_reducers:
                        reducer = reducer_functions[field_name]
                        reducer_name = getattr(reducer, "__name__", str(reducer))
                        field_parts.append(f"  Reducer: {reducer_name}")
                    
                    # Check if field is part of a structured model
                    for model_name, model_path in structured_models.items():
                        if field_name == model_name:
                            field_parts.append(f"  Structured Model: {model_path}")
                    
                    # Check if field is input/output for engines
                    for engine_name, mapping in engine_io_mappings.items():
                        if field_name in mapping.get("inputs", []):
                            field_parts.append(f"  Input for: {engine_name}")
                        if field_name in mapping.get("outputs", []):
                            field_parts.append(f"  Output from: {engine_name}")
                
                fields_section.append("\n".join(field_parts))
        
        # Add fields section
        output.append("Fields:")
        output.extend(fields_section)
        
        # Add additional metadata sections if requested
        if include_metadata:
            # Add shared fields section if any exist
            if shared_fields:
                output.append("")
                output.append("Shared Fields:")
                for field in shared_fields:
                    output.append(f"• {field}")
            
            # Add reducers section if any exist and requested
            if reducer_functions and include_reducers:
                output.append("")
                output.append("Reducers:")
                for field, reducer in reducer_functions.items():
                    reducer_name = getattr(reducer, "__name__", str(reducer))
                    output.append(f"• {field}: {reducer_name}")
            
            # Add engine I/O mappings if any exist
            if engine_io_mappings:
                output.append("")
                output.append("Engine I/O Mappings:")
                for engine_name, mapping in engine_io_mappings.items():
                    inputs = mapping.get("inputs", [])
                    outputs = mapping.get("outputs", [])
                    
                    if inputs:
                        output.append(f"• {engine_name} inputs: {', '.join(inputs)}")
                    if outputs:
                        output.append(f"• {engine_name} outputs: {', '.join(outputs)}")
        
        # Join all sections
        return "\n".join(output)

    @classmethod
    def _format_full_type(cls, annotation):
        """Format a type annotation with full parameter information."""
        from typing import get_origin, get_args, Optional, List, Dict, Union, Any
        
        # Handle simple types
        if annotation is str:
            return "str"
        if annotation is int:
            return "int"
        if annotation is float:
            return "float"
        if annotation is bool:
            return "bool"
        if annotation is type(None):
            return "None"
        
        # Get origin and arguments for generic types
        origin = get_origin(annotation)
        args = get_args(annotation)
        
        if origin is None:
            # Handle direct class references
            if hasattr(annotation, "__module__") and hasattr(annotation, "__name__"):
                return f"{annotation.__module__}.{annotation.__name__}"
            return str(annotation).replace("typing.", "")
        
        # Handle specific generic types
        if origin is Optional:
            inner_type = cls._format_full_type(args[0])
            return f"Optional[{inner_type}]"
        
        # Special case: detect Union[T, None] and format as Optional[T]
        if origin is Union and len(args) == 2:
            # Check if one of the args is None or NoneType
            if args[1] is None or args[1] is type(None) or str(args[1]) == 'NoneType':
                inner_type = cls._format_full_type(args[0])
                return f"Optional[{inner_type}]"
            elif args[0] is None or args[0] is type(None) or str(args[0]) == 'NoneType':
                inner_type = cls._format_full_type(args[1])
                return f"Optional[{inner_type}]"
        
        if origin is list or origin is List:
            if args:
                inner_type = cls._format_full_type(args[0])
                return f"List[{inner_type}]"
            return "List"
        
        if origin is dict or origin is Dict:
            if len(args) == 2:
                key_type = cls._format_full_type(args[0])
                value_type = cls._format_full_type(args[1])
                return f"Dict[{key_type}, {value_type}]"
            return "Dict"
        
        if origin is Union:
            formatted_args = [cls._format_full_type(arg) for arg in args]
            return f"Union[{', '.join(formatted_args)}]"
        
        # Generic handling for other parameterized types
        if args:
            formatted_args = [cls._format_full_type(arg) for arg in args]
            origin_name = str(origin).replace("typing.", "").replace("<class '", "").replace("'>", "")
            return f"{origin_name}[{', '.join(formatted_args)}]"
            
            # Fallback
            return str(origin).replace("typing.", "").replace("<class '", "").replace("'>", "")
    
    @classmethod
    def format_schema(schema_cls):
        output = ["StateSchema:"]
        
        # Get shared fields and reducer info
        shared_fields = getattr(schema_cls, "__shared_fields__", [])
        reducer_functions = getattr(schema_cls, "__reducer_fields__", {})
        
        # Process fields
        for field_name, field_info in schema_cls.model_fields.items():
            # Skip special fields
            if field_name.startswith("__"):
                continue
                
            # Format type 
            field_type = field_info.annotation
            type_str = str(field_type).replace("typing.", "")
            
            # Check if field has reducer
            if field_name in reducer_functions:
                reducer = reducer_functions[field_name]
                reducer_name = getattr(reducer, "__name__", str(reducer))
                output.append(f"{field_name}: Annotated[{type_str}, {reducer_name}]")
            else:
                # Handle default value
                if field_info.default_factory is not None:
                    default_str = f" = Field(default_factory={field_info.default_factory.__name__})"
                elif field_info.default is not ...:
                    if field_info.default is None:
                        default_str = " = None"
                    else:
                        default_str = f" = {field_info.default}"
                else:
                    default_str = ""
                    
                output.append(f"{field_name}: {type_str}{default_str}")
        
        return "\n".join(output)


    @classmethod
    def _format_type_annotation(cls, annotation):
        """Format a type annotation properly, including type parameters."""
        from typing import get_origin, get_args, Optional, List, Dict, Union, Any, Annotated
        
        # Handle simple types
        if hasattr(annotation, "__name__"):
            return annotation.__name__
            
        # Handle origins like Optional, List, etc.
        origin = get_origin(annotation)
        if origin is None:
            # Fallback for any unhandled cases
            return str(annotation).replace("typing.", "")
            
        # Handle special cases
        if origin is Union:
            args = get_args(annotation)
            # Handle Optional[X] special case (Union[X, None])
            if len(args) == 2 and args[1] is type(None):
                inner_type = cls._format_type_annotation(args[0])
                return f"Optional[{inner_type}]"
            else:
                formatted_args = [cls._format_type_annotation(arg) for arg in args]
                return f"Union[{', '.join(formatted_args)}]"
                
        elif origin is list or origin is List:
            args = get_args(annotation)
            if args:
                inner_type = cls._format_type_annotation(args[0])
                return f"List[{inner_type}]"
            return "List"
            
        elif origin is dict or origin is Dict:
            args = get_args(annotation)
            if len(args) == 2:
                key_type = cls._format_type_annotation(args[0])
                value_type = cls._format_type_annotation(args[1])
                return f"Dict[{key_type}, {value_type}]"
            return "Dict"
            
        elif origin is Annotated:
            args = get_args(annotation)
            if args:
                base_type = cls._format_type_annotation(args[0])
                # Skip the metadata in the Annotated display
                return base_type
                
        # Generic handling for other parameterized types
        args = get_args(annotation)
        if not args:
            return str(origin).replace("typing.", "").replace("class ", "")
            
        formatted_args = [cls._format_type_annotation(arg) for arg in args]
        origin_name = str(origin).replace("typing.", "").replace("class ", "")
        
        # Clean up any extra quotes or angle brackets from the string representation
        origin_name = origin_name.replace("'", "").replace("<", "").replace(">", "")
        
        return f"{origin_name}[{', '.join(formatted_args)}]"
    
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