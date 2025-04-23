"""
StateSchemaManager for creating and manipulating state schemas.

This module provides the StateSchemaManager class which handles the creation
and modification of state schemas with fine-grained control over fields,
reducers, shared fields, and more.
"""
from __future__ import annotations
import inspect
import logging
import operator
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Set, Type,\
    TypeVar, Union, get_origin, get_args, Tuple, TYPE_CHECKING 

from pydantic import BaseModel, Field, create_model

from haive.core.schema.state_schema import StateSchema
from haive.core.schema.utils import SchemaUtils
from langgraph.types import Command, Send
from haive.core.schema.field_extractor import FieldExtractor
from haive.core.schema.field_definition import FieldDefinition

# Type variables for generic types
T = TypeVar('T', bound=BaseModel)
SchemaType = TypeVar('SchemaType', bound=Type[BaseModel])

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from haive.core.schema.schema_composer import SchemaComposer   

class StateSchemaManager:
    """
    Manager for dynamically creating and manipulating state schemas.
    
    Provides utilities for:
    - Building schemas from dicts, pydantic models, or SchemaComposer output
    - Adding, removing, or modifying fields
    - Managing field sharing, reducers, and other metadata
    - Creating node functions with schema validation
    """

    def __init__(
        self,
        data: Optional[Union[Dict[str, Any], Type[BaseModel], BaseModel, 'SchemaComposer']] = None,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a new StateSchemaManager.
        
        Args:
            data: Source data to initialize from
            name: Custom schema name
            config: Optional configuration dictionary
        """
        self.fields: Dict[str, tuple[Any, Any]] = {}
        self.validators: Dict[str, Callable] = {}
        self.properties: Dict[str, Callable] = {}
        self.computed_properties: Dict[str, tuple[Callable, Optional[Callable]]] = {}
        self.class_methods: Dict[str, Callable] = {}
        self.static_methods: Dict[str, Callable] = {}
        self.field_descriptions: Dict[str, str] = {}
        self.instance_methods: Dict[str, Callable] = {}
        self.locked: bool = False
        self.config: Dict[str, Any] = config or {}
        
        # Special field sets for state schema
        self._shared_fields: Set[str] = set()
        self._reducer_names: Dict[str, str] = {}  # Field name -> reducer name (serializable)
        self._reducer_functions: Dict[str, Callable] = {}  # Field name -> reducer function
        
        # Input/output tracking for engines
        self._input_fields: Dict[str, Set[str]] = defaultdict(set)  # Engine name -> input fields
        self._output_fields: Dict[str, Set[str]] = defaultdict(set)  # Engine name -> output fields
        self._engine_io_mappings: Dict[str, Dict[str, List[str]]] = {}  # Engine name -> IO mapping
        
        # Set default name
        if data is None:
            self.name = name or self.config.get("default_schema_name", "UnnamedSchema")
            return

        # Extract name from data if not provided
        if name is None:
            if hasattr(data, "name") and isinstance(data, type) and issubclass(data, BaseModel):
                self.name = data.__name__
            elif hasattr(data, "name"):
                self.name = data.name
            else:
                self.name = "CustomState"
        else:
            self.name = name

        # Load data based on type
        if isinstance(data, dict):
            self._load_from_dict(data)
        elif isinstance(data, type) and issubclass(data, BaseModel):
            self._load_from_model(data)
        elif isinstance(data, BaseModel):
            self._load_from_model(data.__class__)
        # Add explicit SchemaComposer check before the more generic check
        elif hasattr(data, "__class__") and data.__class__.__name__ == "SchemaComposer":
            self._load_from_composer(data)
        elif hasattr(data, "fields") and (hasattr(data, "reducer_names") or hasattr(data, "reducer_functions")):
            # This appears to be a SchemaComposer or compatible object
            self._load_from_composer(data)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

    def _load_from_model(self, model_cls: Type[BaseModel]) -> None:
        """
        Load fields and methods from a Pydantic model.
        
        Args:
            model_cls: Pydantic model class to load from
        """
        try:
            # Use the new field extractor utility
            fields, descriptions, shared_fields, reducer_names, reducer_functions, engine_io_mappings, input_fields, output_fields = \
                FieldExtractor.extract_from_model(model_cls)
            
            # Apply extracted fields and metadata
            self.fields.update(fields)
            self.field_descriptions.update(descriptions)
            self._shared_fields.update(shared_fields)
            self._reducer_names.update(reducer_names)
            self._reducer_functions.update(reducer_functions)
            self._engine_io_mappings.update(engine_io_mappings)
            
            # Update input/output fields
            for engine, fields in input_fields.items():
                self._input_fields[engine].update(fields)
                
            for engine, fields in output_fields.items():
                self._output_fields[engine].update(fields)
            
            # Log the loaded fields
            logger.debug(f"Loaded fields: {list(self.fields.keys())}")
            
            # Load methods and validators
            for name, attr in inspect.getmembers(model_cls):
                # Skip private/special methods
                if name.startswith("_") and name != "__validator__":
                    continue
                    
                if isinstance(attr, property):
                    # Handle property
                    getter = attr.fget
                    setter = attr.fset
                    if getter and setter:
                        self.computed_properties[name] = (getter, setter)
                    elif getter:
                        self.properties[name] = getter
                elif inspect.ismethod(attr) or inspect.isfunction(attr):
                    # Handle methods
                    if name.startswith("validate_"):
                        self.validators[name] = attr
                    elif getattr(attr, "__validator__", False):
                        self.validators[name] = attr
                        
        except Exception as e:
            logger.error(f"Error loading from model {model_cls.__name__}: {e}")
            # Add a placeholder field as fallback
            self.fields["placeholder"] = (str, Field(default=""))

    # In StateSchemaManager._load_from_composer method
    def _load_from_composer(self, composer: Any) -> None:
        try:
            # Existing code...
            
            # Explicitly copy engine I/O mappings - deep copy to avoid reference issues
            if hasattr(composer, "engine_io_mappings"):
                self._engine_io_mappings = {
                    k: v.copy() for k, v in composer.engine_io_mappings.items()
                }
                
            # Explicitly copy input/output fields
            if hasattr(composer, "input_fields"):
                for engine_name, fields in composer.input_fields.items():
                    self._input_fields[engine_name] = set(fields)
                    
            if hasattr(composer, "output_fields"):
                for engine_name, fields in composer.output_fields.items():
                    self._output_fields[engine_name] = set(fields)
                    
            # After copying fields, update engine_io_mappings to ensure consistency
            for engine_name in self._input_fields:
                if engine_name not in self._engine_io_mappings:
                    self._engine_io_mappings[engine_name] = {
                        "inputs": [],
                        "outputs": []
                    }
                self._engine_io_mappings[engine_name]["inputs"] = list(self._input_fields[engine_name])
                
            for engine_name in self._output_fields:
                if engine_name not in self._engine_io_mappings:
                    self._engine_io_mappings[engine_name] = {
                        "inputs": [],
                        "outputs": []
                    }
                self._engine_io_mappings[engine_name]["outputs"] = list(self._output_fields[engine_name])
                
        except Exception as e:
            logger.error(f"Error loading from composer: {e}")
            # Add a placeholder field as fallback
            self.fields["placeholder"] = (str, Field(default=""))

    def _load_from_dict(self, data: Dict[str, Any]) -> None:
        """
        Load fields from a dictionary.
        
        Args:
            data: Dictionary containing field data
        """
        try:
            # Use the field extractor utility
            fields, descriptions, shared_fields, reducer_names, reducer_functions, engine_io_mappings, input_fields, output_fields = \
                FieldExtractor.extract_from_dict(data)
            
            # Apply extracted fields and metadata
            self.fields.update(fields)
            self.field_descriptions.update(descriptions)
            self._shared_fields.update(shared_fields)
            self._reducer_names.update(reducer_names)
            self._reducer_functions.update(reducer_functions)
            self._engine_io_mappings.update(engine_io_mappings)
            
            # Update input/output fields
            for engine, fields in input_fields.items():
                self._input_fields[engine].update(fields)
                
            for engine, fields in output_fields.items():
                self._output_fields[engine].update(fields)
                
        except Exception as e:
            logger.error(f"Error loading from dict: {e}")
            # Add a placeholder field as fallback
            self.fields["placeholder"] = (str, Field(default=""))

    def add_field(
        self,
        name: str,
        field_type: Type,
        default: Any = None,
        default_factory: Optional[Callable[[], Any]] = None,
        description: Optional[str] = None,
        shared: bool = False,
        reducer: Optional[Callable] = None,
        optional: bool = True,
        **kwargs
    ) -> "StateSchemaManager":
        """
        Add a field to the schema with comprehensive options.
        
        Args:
            name: Field name
            field_type: Type of the field
            default: Default value for the field
            default_factory: Optional factory function for default value
            description: Optional field description
            shared: Whether field is shared with parent graph
            reducer: Optional reducer function for this field
            optional: Whether to make the field optional (default: True)
            **kwargs: Additional field parameters
            
        Returns:
            Self for chaining
        """
        if self.locked:
            raise ValueError("Schema is locked and cannot be modified.")

        from typing import Optional as OptionalType

        # Make field Optional if requested and not already Optional
        if optional and get_origin(field_type) is not OptionalType:
            field_type = OptionalType[field_type]

        # Build field metadata
        field_metadata = {}
        if description:
            field_metadata["description"] = description
            self.field_descriptions[name] = description

        # Configure default handling
        if default_factory is not None:
            # Use default_factory if provided
            field_info = Field(default_factory=default_factory, **field_metadata, **kwargs)
        elif default is not None:
            # Use explicit default if provided
            field_info = Field(default=default, **field_metadata, **kwargs)
        elif optional:
            # Use None for optional fields
            field_info = Field(default=None, **field_metadata, **kwargs)
        else:
            # Use ... for required fields
            field_info = Field(default=..., **field_metadata, **kwargs)

        # Debug field addition
        logger.debug(f"Adding field to schema: {name}: {field_type} with default {default} or factory {default_factory}")
        
        # Add the field
        self.fields[name] = (field_type, field_info)

        # Track additional metadata
        if shared:
            self._shared_fields.add(name)

        if reducer:
            # Store reducer function in runtime dictionary
            self._reducer_functions[name] = reducer
            
            # Store serializable name for the reducer
            field_def = FieldDefinition(name=name, field_type=field_type, reducer=reducer)
            self._reducer_names[name] = field_def.get_reducer_name()

        return self

    def add_computed_property(
        self,
        name: str,
        getter_func: Callable,
        setter_func: Optional[Callable] = None
    ) -> "StateSchemaManager":
        """
        Add a computed property with getter and optional setter.
        
        Args:
            name: Property name
            getter_func: Getter function
            setter_func: Optional setter function
            
        Returns:
            Self for chaining
        """
        if self.locked:
            raise ValueError("Schema is locked and cannot be modified.")
        self.computed_properties[name] = (getter_func, setter_func)
        return self

    def remove_field(self, name: str) -> "StateSchemaManager":
        """
        Remove a field from the schema.
        
        Args:
            name: Name of the field to remove
            
        Returns:
            Self for chaining
        """
        if self.locked:
            raise ValueError("Schema is locked and cannot be modified.")
        if name in self.fields:
            del self.fields[name]
        if name in self.field_descriptions:
            del self.field_descriptions[name]
        if name in self._shared_fields:
            self._shared_fields.remove(name)
        if name in self._reducer_names:
            del self._reducer_names[name]
        if name in self._reducer_functions:
            del self._reducer_functions[name]
        return self

    def modify_field(
        self,
        name: str,
        new_type: Optional[Type] = None,
        new_default: Any = None,
        new_description: Optional[str] = None,
        new_shared: Optional[bool] = None,
        new_reducer: Optional[Callable] = None
    ) -> "StateSchemaManager":
        """
        Modify an existing field's properties.
        
        Args:
            name: Name of the field to modify
            new_type: New type for the field
            new_default: New default value
            new_description: New description
            new_shared: Whether field should be shared
            new_reducer: New reducer function
            
        Returns:
            Self for chaining
        """
        if self.locked:
            raise ValueError("Schema is locked and cannot be modified.")
        if name in self.fields:
            current_type, current_field_info = self.fields[name]

            # Update type if provided
            field_type = new_type if new_type is not None else current_type

            # Create field metadata
            field_metadata = {}
            
            # Update description if provided
            if new_description:
                field_metadata["description"] = new_description
                self.field_descriptions[name] = new_description
            elif name in self.field_descriptions:
                field_metadata["description"] = self.field_descriptions[name]

            # Update default if provided
            if new_default is not None:
                field_info = Field(default=new_default, **field_metadata)
            elif hasattr(current_field_info, "default_factory") and current_field_info.default_factory is not None:
                # Keep existing default_factory
                field_info = Field(default_factory=current_field_info.default_factory, **field_metadata)
            else:
                # Keep existing default
                default = current_field_info.default
                field_info = Field(default=default, **field_metadata)

            # Update field with new type and info
            self.fields[name] = (field_type, field_info)
            
            # Update shared status if specified
            if new_shared is not None:
                if new_shared and name not in self._shared_fields:
                    self._shared_fields.add(name)
                elif not new_shared and name in self._shared_fields:
                    self._shared_fields.remove(name)
                    
            # Update reducer if provided
            if new_reducer is not None:
                self._reducer_functions[name] = new_reducer
                
                # Store serializable name for the reducer
                field_def = FieldDefinition(name=name, field_type=field_type, reducer=new_reducer)
                self._reducer_names[name] = field_def.get_reducer_name()

        return self

    def has_field(self, name: str) -> bool:
        """
        Check if the schema has a specific field.
        
        Args:
            name: Field name to check
            
        Returns:
            True if field exists, False otherwise
        """
        return name in self.fields

    def merge(
        self,
        other: Union["StateSchemaManager", Type[BaseModel], BaseModel, 'SchemaComposer']
    ) -> "StateSchemaManager":
        """
        Merge with another schema, preserving first occurrences.
        
        Args:
            other: Another object to merge with
            
        Returns:
            New merged StateSchemaManager
        """
        from typing import Optional

        # Create a new manager with meaningful name
        other_name = getattr(other, "name", "Other")
        merged_name = f"{self.name}_merged_with_{other_name}"
        merged = StateSchemaManager(name=merged_name, config=self.config)
        
        # Copy fields and metadata from self
        merged.fields = self.fields.copy()
        merged.field_descriptions = self.field_descriptions.copy()
        merged._shared_fields = self._shared_fields.copy()
        merged._reducer_names = self._reducer_names.copy()
        merged._reducer_functions = self._reducer_functions.copy()
        merged.validators = self.validators.copy()
        merged.properties = self.properties.copy()
        merged.computed_properties = self.computed_properties.copy()
        merged.class_methods = self.class_methods.copy()
        merged.static_methods = self.static_methods.copy()
        
        # Copy engine I/O tracking
        merged._input_fields = {k: v.copy() for k, v in self._input_fields.items()}
        merged._output_fields = {k: v.copy() for k, v in self._output_fields.items()}
        merged._engine_io_mappings = self._engine_io_mappings.copy()

        # Handle different types of 'other'
        if isinstance(other, StateSchemaManager):
            # Merge from another StateSchemaManager
            for field, (field_type, field_info) in other.fields.items():
                # Skip if field already exists to preserve first occurrence
                if field not in merged.fields:
                    # Ensure field is optional
                    if get_origin(field_type) is not Optional:
                        field_type = Optional[field_type]
                    merged.fields[field] = (field_type, field_info)

                    # Copy field description if available
                    if field in other.field_descriptions:
                        merged.field_descriptions[field] = other.field_descriptions[field]

            # Merge shared fields
            merged._shared_fields.update(other._shared_fields)
            
            # Merge reducers (don't overwrite existing)
            for field, reducer_name in other._reducer_names.items():
                if field not in merged._reducer_names:
                    merged._reducer_names[field] = reducer_name
                    
            # Merge reducer functions
            for field, reducer in other._reducer_functions.items():
                if field not in merged._reducer_functions:
                    merged._reducer_functions[field] = reducer
                    
            # Merge methods
            for name, validator in other.validators.items():
                if name not in merged.validators:
                    merged.validators[name] = validator
                    
            for name, prop in other.computed_properties.items():
                if name not in merged.computed_properties:
                    merged.computed_properties[name] = prop
                    
            for name, method in other.class_methods.items():
                if name not in merged.class_methods:
                    merged.class_methods[name] = method
                    
            for name, method in other.static_methods.items():
                if name not in merged.static_methods:
                    merged.static_methods[name] = method
                    
            # Merge engine I/O tracking (don't overwrite)
            for engine_name, fields in other._input_fields.items():
                if engine_name not in merged._input_fields:
                    merged._input_fields[engine_name] = fields.copy()
                else:
                    merged._input_fields[engine_name].update(fields)
                    
            for engine_name, fields in other._output_fields.items():
                if engine_name not in merged._output_fields:
                    merged._output_fields[engine_name] = fields.copy()
                else:
                    merged._output_fields[engine_name].update(fields)
                    
            for engine_name, mapping in other._engine_io_mappings.items():
                if engine_name not in merged._engine_io_mappings:
                    merged._engine_io_mappings[engine_name] = mapping.copy()

        elif hasattr(other, "fields") and (hasattr(other, "shared_fields") or hasattr(other, "reducer_names")):
            # Merge from a SchemaComposer-like object
            for field, (field_type, field_info) in other.fields.items():
                if field not in merged.fields:
                    # Ensure field is optional
                    if get_origin(field_type) is not Optional:
                        field_type = Optional[field_type]
                    merged.fields[field] = (field_type, field_info)
                    
                    # Copy field description if available
                    if hasattr(other, "field_descriptions") and field in other.field_descriptions:
                        merged.field_descriptions[field] = other.field_descriptions[field]
            
            # Merge shared fields if available
            if hasattr(other, "shared_fields"):
                merged._shared_fields.update(other.shared_fields)
            
            # Merge reducers if available
            if hasattr(other, "reducer_names"):
                for field, reducer_name in other.reducer_names.items():
                    if field not in merged._reducer_names:
                        merged._reducer_names[field] = reducer_name
                        
            if hasattr(other, "reducer_functions"):
                for field, reducer in other.reducer_functions.items():
                    if field not in merged._reducer_functions:
                        merged._reducer_functions[field] = reducer
                        
            # Merge engine I/O information if available
            if hasattr(other, "input_fields"):
                for engine_name, fields in other.input_fields.items():
                    if engine_name not in merged._input_fields:
                        merged._input_fields[engine_name] = set(fields)
                    else:
                        merged._input_fields[engine_name].update(fields)
                        
            if hasattr(other, "output_fields"):
                for engine_name, fields in other.output_fields.items():
                    if engine_name not in merged._output_fields:
                        merged._output_fields[engine_name] = set(fields)
                    else:
                        merged._output_fields[engine_name].update(fields)
                        
            if hasattr(other, "engine_io_mappings"):
                for engine_name, mapping in other.engine_io_mappings.items():
                    if engine_name not in merged._engine_io_mappings:
                        merged._engine_io_mappings[engine_name] = mapping.copy()

        elif isinstance(other, type) and issubclass(other, BaseModel):
            # Merge from a Pydantic model class
            # Get fields - Pydantic v2
            fields_dict = other.model_fields
            
            for field_name, field_info in fields_dict.items():
                if field_name not in merged.fields:
                    # Get field type for Pydantic v2
                    field_type = field_info.annotation
                    
                    # Ensure field is optional
                    if get_origin(field_type) is not Optional:
                        field_type = Optional[field_type]
                        
                    merged.fields[field_name] = (field_type, field_info)

                    # Copy field description
                    description = field_info.description
                    if description:
                        merged.field_descriptions[field_name] = description
            
            # Copy shared fields from StateSchema
            if hasattr(other, "__shared_fields__"):
                merged._shared_fields.update(other.__shared_fields__)

            # Copy serializable reducers
            if hasattr(other, "__serializable_reducers__"):
                for field, reducer_name in other.__serializable_reducers__.items():
                    if field not in merged._reducer_names:
                        merged._reducer_names[field] = reducer_name

            # Copy reducer functions
            if hasattr(other, "__reducer_fields__"):
                for field, reducer in other.__reducer_fields__.items():
                    if field not in merged._reducer_functions:
                        merged._reducer_functions[field] = reducer
                        
            # Copy engine I/O information
            if hasattr(other, "__engine_io_mappings__"):
                for engine_name, mapping in other.__engine_io_mappings__.items():
                    if engine_name not in merged._engine_io_mappings:
                        merged._engine_io_mappings[engine_name] = mapping.copy()
                        
            if hasattr(other, "__input_fields__"):
                for engine_name, fields in other.__input_fields__.items():
                    if engine_name not in merged._input_fields:
                        merged._input_fields[engine_name] = set(fields)
                    else:
                        merged._input_fields[engine_name].update(fields)
                        
            if hasattr(other, "__output_fields__"):
                for engine_name, fields in other.__output_fields__.items():
                    if engine_name not in merged._output_fields:
                        merged._output_fields[engine_name] = set(fields)
                    else:
                        merged._output_fields[engine_name].update(fields)

        elif isinstance(other, BaseModel):
            # Merge from a Pydantic model instance by using its class
            merged = self.merge(other.__class__)

        return merged

    def get_model(
        self,
        lock: bool = False,
        as_state_schema: bool = True,
        name: Optional[str] = None
    ) -> Type[BaseModel]:
        """
        Create a Pydantic model with all configured options.
        
        Args:
            lock: Whether to lock the schema against further modifications
            as_state_schema: Whether to use StateSchema as the base class
            name: Optional name for the schema class
            
        Returns:
            Created model class
        """
        if lock:
            self.locked = True

        # Use provided name or default
        model_name = name or self.name

        # Choose the base class
        base_class = StateSchema if as_state_schema else BaseModel

        # Create the model with fields
        model = create_model(model_name, __base__=base_class, **self.fields)
        
        # Debug model creation
        logger.debug(f"Created model {model_name} with fields: {list(getattr(model, '__annotations__', {}))}")
        logger.debug(f"Model has model_fields: {hasattr(model, 'model_fields')}")
        
        # Add shared fields metadata if using StateSchema
        if as_state_schema:
            model.__shared_fields__ = list(self._shared_fields)

            # Add reducer metadata
            model.__serializable_reducers__ = dict(self._reducer_names)
            
            # Add actual reducer functions
            if not hasattr(model, "__reducer_fields__"):
                model.__reducer_fields__ = {}
            model.__reducer_fields__.update(self._reducer_functions)
            
            # Add engine I/O mappings metadata if available
            if self._engine_io_mappings:
                model.__engine_io_mappings__ = self._engine_io_mappings
                
            # Add input/output field tracking
            if self._input_fields:
                model.__input_fields__ = {k: list(v) for k, v in self._input_fields.items()}
                
            if self._output_fields:
                model.__output_fields__ = {k: list(v) for k, v in self._output_fields.items()}

        # Add validators
        for name, validator in self.validators.items():
            setattr(model, name, validator)

        # Add properties
        for name, (getter, setter) in self.computed_properties.items():
            prop = property(getter, setter)
            setattr(model, name, prop)
            
        # Add regular properties
        for name, getter in self.properties.items():
            prop = property(getter)
            setattr(model, name, prop)

        # Add instance methods
        for name, method in self.instance_methods.items():
            setattr(model, name, method)

        # Add class methods
        for name, method in self.class_methods.items():
            setattr(model, name, classmethod(method))

        # Add static methods
        for name, method in self.static_methods.items():
            setattr(model, name, staticmethod(method))

        return model

    def pretty_print(self) -> None:
        """
        Print the schema as a Python class definition.
        """
        print(self.get_pretty_print_output())

    def get_pretty_print_output(self) -> str:
        """
        Get the schema as a Python class definition string.
        
        Returns:
            Formatted string representation
        """
        lines = [f"class {self.name}(StateSchema):"]
        lines.append('    """')
        lines.append(f"    Generated {self.name} schema")
        lines.append('    """')
        
        # Add fields
        for name, (field_type, field_info) in self.fields.items():
            # Get type as string
            type_str = str(field_type).replace("typing.", "")
            
            # Get default as string - Pydantic v2
            if field_info.default_factory is not None:
                default_str = f"Field(default_factory={field_info.default_factory.__name__})"
            else:
                default = field_info.default
                if default is ...:
                    default_str = "Field(...)"
                else:
                    default_str = f"Field(default={repr(default)})"
                    
            # Add description if available
            if name in self.field_descriptions:
                desc = self.field_descriptions[name].replace('"', '\\"')
                default_str = default_str.replace(")", f', description="{desc}")')
                
            # Format the field
            field_line = f"    {name}: {type_str} = {default_str}"
            lines.append(field_line)
            
        # Add shared fields
        if self._shared_fields:
            lines.append("")
            lines.append("    # Shared fields")
            lines.append(f"    __shared_fields__ = {list(self._shared_fields)}")
            
        # Add serializable reducers
        if self._reducer_names:
            lines.append("")
            lines.append("    # Reducers")
            lines.append(f"    __serializable_reducers__ = {self._reducer_names}")
            
        return "\n".join(lines)

    def create_node_function(
        self,
        func: Callable,
        command_goto: Optional[str] = None
    ) -> Callable:
        """
        Create a node function with schema validation.
        
        Args:
            func: Function that takes state and returns updates
            command_goto: Optional next node to go to
            
        Returns:
            Node function with validation
        """
        # Get the model with our current schema
        model_cls = self.get_model()
        
        def node_function(state: Any, config: Optional[Dict[str, Any]] = None) -> Union[Dict[str, Any], Command]:
            """Wrapper function that validates state and applies reducers."""
            # Validate input state
            if not isinstance(state, model_cls):
                try:
                    state = model_cls.model_validate(state)
                except Exception as e:
                    try:
                        state = model_cls.from_dict(state)
                    except Exception as inner_e:
                        # Last resort: just pass it through
                        pass
            
            # Call the original function
            result = func(state, config)
            
            # Handle different return types
            if isinstance(result, Command):
                # Command object - already handles control flow
                return result
            elif isinstance(result, dict):
                # Dictionary update - wrap in Command if goto specified
                if command_goto:
                    return Command(update=result, goto=command_goto)
                else:
                    return result
            else:
                # Other return type - convert to dict with result key
                update = {"result": result}
                if command_goto:
                    return Command(update=update, goto=command_goto)
                else:
                    return update
                    
        # Return the wrapped function
        return node_function
    
    def create_command(
        self,
        update: Dict[str, Any],
        goto: Optional[Union[str, "END"]] = None,
        resume: Optional[Any] = None,
        graph: Optional[str] = None,
        validate_update: bool = True
    ) -> Command:
        """
        Create a Command object with schema validation.
        TODO: Review langgraph goto, END edge.
        Args:
            update: State updates to apply
            goto: Next node to go to
            resume: Value to resume with after an interrupt
            graph: Graph to send command to (None for current, Command.PARENT for parent)
            validate_update: Whether to validate the update against schema
            
        Returns:
            Command object
        """
        # Validate update against schema if requested
        if validate_update and update:
            try:
                model_cls = self.get_model()
                # Only validate fields that exist in the schema
                valid_fields = {k: v for k, v in update.items() if k in model_cls.model_fields}
                if valid_fields:
                    # Create a minimal state with just the updates
                    temp_state = model_cls.from_partial_dict(valid_fields)
                    # Convert back to dict
                    validated_update = temp_state.to_dict()
                    # Preserve fields that weren't in the schema
                    for k, v in update.items():
                        if k not in valid_fields:
                            validated_update[k] = v
                    update = validated_update
            except Exception as e:
                logger.warning(f"Command update validation failed: {e}")
        
        # Handle special END case for goto
        if goto == "END":
            from langgraph.graph import END
            goto = END
        
        return Command(update=update, goto=goto, resume=resume, graph=graph)
    
    def create_send(
        self, 
        node: str, 
        arg: Any,
        validate_args: bool = True
    ) -> Send:
        """
        Create a Send object for routing to another node.
        
        Args:
            node: Name of the target node
            arg: State to send to the node
            validate_args: Whether to validate the arguments against schema
            
        Returns:
            Send object
        """
        # Validate arguments if requested
        if validate_args and isinstance(arg, dict):
            try:
                model_cls = self.get_model()
                # Only validate fields that exist in the schema
                valid_fields = {k: v for k, v in arg.items() if hasattr(model_cls, k)}
                if valid_fields:
                    # Create a minimal state with just the valid fields
                    temp_state = model_cls.from_partial_dict(valid_fields)
                    # Convert back to dict
                    validated_arg = temp_state.to_dict()
                    # Preserve fields that weren't in the schema
                    for k, v in arg.items():
                        if k not in valid_fields:
                            validated_arg[k] = v
                    arg = validated_arg
            except Exception as e:
                logger.warning(f"Send argument validation failed: {e}")
        
        return Send(node, arg)

    def create_sends_from_items(
        self,
        items: List[Any],
        target_node: str,
        state_factory: Callable[[Any], Dict[str, Any]],
        validate_args: bool = True
    ) -> List[Send]:
        """
        Create Send objects from a list of items using a state factory.
        Useful for map operations in map-reduce patterns.
        
        Args:
            items: List of items to process
            target_node: Node to send all items to
            state_factory: Function to convert each item to a state dict
            validate_args: Whether to validate the arguments against schema
            
        Returns:
            List of Send objects
        """
        return [self.create_send(target_node, state_factory(item), validate_args) for item in items]

    def add_method(self, name: str, method: Callable) -> "StateSchemaManager":
        """
        Add a method to the schema.
        
        Args:
            name: Method name
            method: Method implementation
            
        Returns:
            Self for chaining
        """
        if self.locked:
            raise ValueError("Schema is locked and cannot be modified.")
        
        # Store the method as an instance method
        self.instance_methods[name] = method
        
        return self

    def mark_as_input_field(self, field_name: str, engine_name: str) -> 'StateSchemaManager':
        """
        Mark a field as an input field for an engine.
        
        Args:
            field_name: Name of the field
            engine_name: Name of the engine
            
        Returns:
            Self for chaining
        """
        if field_name in self.fields:
            self._input_fields[engine_name].add(field_name)
            self._update_engine_io_mapping(engine_name)
        return self
    
    def mark_as_output_field(self, field_name: str, engine_name: str) -> 'StateSchemaManager':
        """
        Mark a field as an output field for an engine.
        
        Args:
            field_name: Name of the field
            engine_name: Name of the engine
            
        Returns:
            Self for chaining
        """
        if field_name in self.fields:
            self._output_fields[engine_name].add(field_name)
            self._update_engine_io_mapping(engine_name)
        return self
    
    def _update_engine_io_mapping(self, engine_name: str) -> None:
        """
        Update engine I/O mapping for a specific engine.
        
        Args:
            engine_name: Engine name to update mapping for
        """
        # Create mapping if it doesn't exist
        if engine_name not in self._engine_io_mappings:
            self._engine_io_mappings[engine_name] = {
                "inputs": [],
                "outputs": []
            }
            
        # Update inputs from tracked fields
        if engine_name in self._input_fields:
            self._engine_io_mappings[engine_name]["inputs"] = list(self._input_fields[engine_name])
            
        # Update outputs from tracked fields
        if engine_name in self._output_fields:
            self._engine_io_mappings[engine_name]["outputs"] = list(self._output_fields[engine_name])