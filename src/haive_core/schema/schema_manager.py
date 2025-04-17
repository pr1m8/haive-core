# src/haive/core/graph/schema/StateSchemaManager.py

from typing import Any, Dict, List, Optional, Type, Union, Callable, get_origin, get_type_hints, Tuple
from pydantic import BaseModel, Field, ValidationError, create_model
import inspect
import uuid
import typing
from langchain_core.messages import BaseMessage
import logging

from haive_core.schema.state_schema import StateSchema
from langgraph.types import Command, Send

# Set up logging
logger = logging.getLogger(__name__)

class StateSchemaManager:
    """
    A dynamic schema manager that:
    - Builds schemas from dicts, lists, or Pydantic models.
    - Merges schemas while preserving uniqueness (no duplicate fields).
    - Tracks validators, properties, computed attributes, class/static methods.
    - Supports recursive field merging for nested schemas.
    - Uses a config dictionary for settings and overrides.
    - Prints schema nicely in Python class format.
    """

    def __init__(self, 
                 data: Optional[Union[Dict[str, Any], Type[BaseModel]]] = None,
                 name: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Args:
            data: Dictionary or Pydantic BaseModel.
            name: Custom schema name (defaults to class name if BaseModel).
            config: Optional dictionary for schema customization.
        """
        self.fields = {}
        self.validators = {}
        self.properties = {}
        self.computed_properties = {}  # New: Store computed property definitions
        self.class_methods = {}
        self.static_methods = {}
        self.config = config or {}  # Store user-defined configurations
        self.locked = False  
        self.property_getters = {}  # Store property getter methods
        self.property_setters = {}  # Store property setter methods
        self.field_descriptions = {}  # Store field descriptions
        
        if data is None:
            self.name = name or self.config.get("default_schema_name", "UnnamedSchema")
            return  

        if name is None and isinstance(data, type) and issubclass(data, BaseModel):
            self.name = data.__name__  
        else:
            self.name = name or "CustomState"

        if isinstance(data, dict) or isinstance(data, typing._TypedDictMeta):
            self._load_from_dict(data)
        elif isinstance(data, type) and issubclass(data, BaseModel):
            self._load_from_model(data)
        elif isinstance(data, BaseModel):
            self._load_from_model(data.__class__)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

    def _load_from_dict(self, data: Dict[str, Any]) -> None:
        logger.debug(f"Loading from dict: {data}")
        try:
            # Handle TypedDict (which has an __annotations__ attribute)
            if hasattr(data, "__annotations__"):
                for key, type_hint in data.__annotations__.items():
                    default_value = None
                    self.fields[key] = (type_hint, Field(default=default_value))
            else:
                # Regular dict
                for key, value in data.items():
                    inferred_type = self._infer_type(value)
                    self.fields[key] = (inferred_type, Field(default=value))
        except Exception as e:
            logger.error(f"Error loading from dict: {e}")
            # Fallback approach
            try:
                # Try to treat it as a callable that returns a dict
                if callable(data):
                    result = data()
                    if isinstance(result, dict):
                        for key, value in result.items():
                            inferred_type = self._infer_type(value)
                            self.fields[key] = (inferred_type, Field(default=value))
            except Exception as inner_e:
                logger.error(f"Fallback also failed: {inner_e}")
                # Last resort: just add an empty field
                self.fields["placeholder"] = (str, Field(default=""))

    def _load_from_model(self, model_cls: Type[BaseModel]) -> None:
        try:
            # Handle Pydantic v2
            for field_name, field_info in model_cls.model_fields.items():
                self.fields[field_name] = (field_info.annotation, field_info)
                if field_info.description:
                    self.field_descriptions[field_name] = field_info.description
            
            # Load methods
            for name, method in inspect.getmembers(model_cls, predicate=inspect.isfunction):
                if hasattr(method, "__validator_config__"):
                    self.validators[name] = method
                elif isinstance(getattr(model_cls, name, None), property):
                    prop = getattr(model_cls, name)
                    self.properties[name] = method
                    if prop.fget:
                        self.property_getters[name] = prop.fget
                    if prop.fset:
                        self.property_setters[name] = prop.fset
                elif isinstance(getattr(model_cls, name, None), classmethod):
                    self.class_methods[name] = method
                elif isinstance(getattr(model_cls, name, None), staticmethod):
                    self.static_methods[name] = method
        except Exception as e:
            logger.error(f"Error loading from model: {e}")
            # Add a placeholder field to ensure we have something
            self.fields["placeholder"] = (str, Field(default=""))

    def _infer_type(self, value: Any) -> Type:
        """Infer the type of a value, with special handling for collections."""
        if isinstance(value, str):
            return str
        elif isinstance(value, int):
            return int
        elif isinstance(value, float):
            return float
        elif isinstance(value, bool):
            return bool
        elif isinstance(value, list):
            return List[Any]  # We can't guarantee homogeneity so use Any
        elif isinstance(value, dict):
            return Dict[str, Any]
        return Any
        
    def add_field(self, 
                  name: str, 
                  field_type: Type, 
                  default: Any = None, 
                  config_aware: bool = False, 
                  default_factory: Optional[Callable] = None,
                  description: Optional[str] = None,
                  shared: bool = False,
                  reducer: Optional[Callable] = None,
                  optional: bool = False,
                  **kwargs) -> 'StateSchemaManager':
        """
        Add a field to the schema, with comprehensive options.

        Args:
            name: Name of the field to add
            field_type: Type of the field
            default: Default value for the field
            config_aware: Whether the field is config-aware
            default_factory: Factory function for the default value
            description: Field description for documentation
            shared: Whether the field is shared with parent graphs
            reducer: Reducer function for merging values
            optional: Whether to make the field Optional
            **kwargs: Additional arguments for field creation
        
        Returns:
            Updated StateSchemaManager
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
        else:
            if optional or get_origin(field_type) is OptionalType:
                # Use None for optional fields
                field_info = Field(default=None, **field_metadata, **kwargs)
            else:
                # Use ... for required fields
                field_info = Field(default=..., **field_metadata, **kwargs)
        
        # Add the field
        self.fields[name] = (field_type, field_info)
        
        # Track additional metadata
        if config_aware:
            if not hasattr(self, '_config_aware_fields'):
                self._config_aware_fields = set()
            self._config_aware_fields.add(name)
        
        if shared:
            if not hasattr(self, '_shared_fields'):
                self._shared_fields = set()
            self._shared_fields.add(name)
        
        if reducer:
            if not hasattr(self, '_reducer_fields'):
                self._reducer_fields = {}
            self._reducer_fields[name] = reducer
        
        return self

    def add_computed_property(self, name: str, getter_func: Callable, setter_func: Optional[Callable] = None) -> 'StateSchemaManager':
        """Add a computed property with getter and optional setter."""
        if self.locked:
            raise ValueError("Schema is locked and cannot be modified.")
        self.computed_properties[name] = (getter_func, setter_func)
        return self

    def remove_field(self, name: str) -> 'StateSchemaManager':
        """Remove a field from the schema."""
        if self.locked:
            raise ValueError("Schema is locked and cannot be modified.")
        if name in self.fields:
            del self.fields[name]
        if name in self.field_descriptions:
            del self.field_descriptions[name]
        return self

    def modify_field(self, 
                     name: str, 
                     new_type: Optional[Type] = None, 
                     new_default: Optional[Any] = None,
                     new_description: Optional[str] = None) -> 'StateSchemaManager':
        """Modify an existing field's type, default value, or description."""
        if self.locked:
            raise ValueError("Schema is locked and cannot be modified.")
        if name in self.fields:
            current_type, current_field_info = self.fields[name]
            
            # Update type if provided
            field_type = new_type if new_type is not None else current_type
            
            # Create new field info
            field_metadata = {}
            if new_description:
                field_metadata["description"] = new_description
                self.field_descriptions[name] = new_description
            elif name in self.field_descriptions:
                field_metadata["description"] = self.field_descriptions[name]
            
            # Update default if provided, otherwise keep existing
            if new_default is not None:
                field_info = Field(default=new_default, **field_metadata)
            else:
                # Keep existing default
                field_info = Field(default=current_field_info.default, **field_metadata)
            
            self.fields[name] = (field_type, field_info)
        
        return self

    def merge(self, other: Union['StateSchemaManager', Type[BaseModel], BaseModel]) -> 'StateSchemaManager':
        """
        Enhanced merge to preserve first occurrence and handle conflict more gracefully.
        
        Args:
            other: Another schema manager or Pydantic model to merge with
            
        Returns:
            Merged StateSchemaManager
        """
        from typing import Optional as OptionalType
        
        merged = StateSchemaManager(name=f"{self.name}_Merged", config=self.config)
        merged.fields = self.fields.copy()
        merged.field_descriptions = self.field_descriptions.copy()
        
        # Copy metadata collections
        if hasattr(self, '_config_aware_fields'):
            merged._config_aware_fields = self._config_aware_fields.copy()
        if hasattr(self, '_shared_fields'):
            merged._shared_fields = self._shared_fields.copy()
        if hasattr(self, '_reducer_fields'):
            merged._reducer_fields = self._reducer_fields.copy()
        
        # Process incoming fields
        if isinstance(other, StateSchemaManager):
            # Merge from another StateSchemaManager
            for field, (field_type, field_info) in other.fields.items():
                # Skip if field already exists to preserve first occurrence
                if field not in merged.fields:
                    merged.fields[field] = (field_type, field_info)
                    
                    # Copy field description if available
                    if field in other.field_descriptions:
                        merged.field_descriptions[field] = other.field_descriptions[field]
            
            # Merge metadata collections
            if hasattr(other, '_config_aware_fields'):
                if not hasattr(merged, '_config_aware_fields'):
                    merged._config_aware_fields = set()
                merged._config_aware_fields.update(other._config_aware_fields)
            
            if hasattr(other, '_shared_fields'):
                if not hasattr(merged, '_shared_fields'):
                    merged._shared_fields = set()
                merged._shared_fields.update(other._shared_fields)
            
            if hasattr(other, '_reducer_fields'):
                if not hasattr(merged, '_reducer_fields'):
                    merged._reducer_fields = {}
                for field, reducer in other._reducer_fields.items():
                    if field not in merged._reducer_fields:
                        merged._reducer_fields[field] = reducer
        
        elif isinstance(other, type) and issubclass(other, BaseModel):
            # Merge from a Pydantic model class
            for field_name, field_info in other.model_fields.items():
                if field_name not in merged.fields:
                    # Use Optional type for added flexibility
                    optional_type = OptionalType[field_info.annotation]
                    merged.fields[field_name] = (optional_type, field_info)
                    
                    # Copy field description if available
                    if field_info.description:
                        merged.field_descriptions[field_name] = field_info.description
            
            # Copy shared fields if it's a StateSchema
            if issubclass(other, StateSchema) and hasattr(other, '__shared_fields__'):
                if not hasattr(merged, '_shared_fields'):
                    merged._shared_fields = set()
                merged._shared_fields.update(other.__shared_fields__)
            
            # Copy reducer fields if it's a StateSchema
            if issubclass(other, StateSchema) and hasattr(other, '__reducer_fields__'):
                if not hasattr(merged, '_reducer_fields'):
                    merged._reducer_fields = {}
                for field, reducer in other.__reducer_fields__.items():
                    if field not in merged._reducer_fields:
                        merged._reducer_fields[field] = reducer
        
        elif isinstance(other, BaseModel):
            # Merge from a Pydantic model instance (use its class)
            model_cls = other.__class__
            for field_name, field_info in model_cls.model_fields.items():
                if field_name not in merged.fields:
                    # Use Optional type for added flexibility
                    optional_type = OptionalType[field_info.annotation]
                    merged.fields[field_name] = (optional_type, field_info)
                    
                    # Copy field description if available
                    if field_info.description:
                        merged.field_descriptions[field_name] = field_info.description
        
        return merged
    
    def has_field(self, name: str) -> bool:
        """Check if the schema has a specific field."""
        return name in self.fields
    
    # In StateSchemaManager.get_model, add support for as_state_schema:
    def get_model(self, lock: bool = False, as_state_schema: bool = True) -> Type[BaseModel]:
        """
        Create a Pydantic model with all configured options.
        
        Args:
            lock: Whether to lock the schema after creating the model
            as_state_schema: Whether to use StateSchema as the base class
            
        Returns:
            Generated Pydantic model class
        """
        if lock:
            self.locked = True
        
        # Choose the base class
        base_class = StateSchema if as_state_schema else BaseModel
        
        # Create the model
        model = create_model(self.name, __base__=base_class, **self.fields)
        
        # Add shared fields metadata if using StateSchema
        if as_state_schema and hasattr(self, '_shared_fields'):
            model.__shared_fields__ = list(self._shared_fields)
        
        # Add reducer fields metadata if using StateSchema
        if as_state_schema and hasattr(self, '_reducer_fields'):
            model.__reducer_fields__ = self._reducer_fields
        
        return model

    def pretty_print(self) -> None:
        """
        Print the schema as if it was written as a Python class.
        """
        print(f"class {self.name}(StateSchema):")
        if not any([self.fields, self.properties, self.computed_properties, self.class_methods, self.static_methods]):
            print("    pass  # No fields defined\n")
            return

        for field_name, (field_type, field_info) in self.fields.items():
            type_str = str(field_type).replace("typing.", "")
            default_val = field_info.default if field_info.default != ... else "..."
            default_str = f" = {default_val}" if default_val != "..." else ""
            description = self.field_descriptions.get(field_name, "")
            if description:
                print(f"    # {description}")
            print(f"    {field_name}: {type_str}{default_str}")

        for prop_name in self.properties.keys():
            print(f"\n    @property")
            print(f"    def {prop_name}(self): ...  # Regular property")

        for prop_name in self.computed_properties.keys():
            print(f"\n    @property")
            print(f"    def {prop_name}(self): ...  # Computed property")

        for method_name in self.class_methods.keys():
            print(f"\n    @classmethod")
            print(f"    def {method_name}(cls): ...  # Class method")

        for method_name in self.static_methods.keys():
            print(f"\n    @staticmethod")
            print(f"    def {method_name}(): ...  # Static method")

        print()

    # --- Node creation methods ---
    def create_node_function(
        self,
        func: Callable,
        command_goto: Optional[str] = None
    ) -> Callable[[Dict[str, Any]], Any]:
        """
        Create a node function that validates state with this schema.
        
        Args:
            func: The function to wrap
            command_goto: Optional next node to route to
            
        Returns:
            Wrapped node function
        """
        Model = self.get_model()
        
        def wrapped_node(state: Dict[str, Any], config: Optional[Dict[str, Any]] = None):
            try:
                # Validate state against our schema
                valid_state = Model.model_validate(state)
                
                # Call the original function
                result = func(valid_state, config) if config else func(valid_state)
                
                # Handle different return types
                from langgraph.types import Command
                
                if isinstance(result, Command):
                    # Command already returned, preserve it
                    if command_goto and not result.goto:
                        # Add goto if not present but requested
                        return Command(
                            update=result.update,
                            goto=command_goto,
                            resume=result.resume,
                            graph=result.graph
                        )
                    return result
                elif isinstance(result, dict):
                    # Wrap dict in Command
                    return Command(update=result, goto=command_goto)
                else:
                    # Convert other results to dict update
                    return Command(update={"result": result}, goto=command_goto)
                    
            except ValidationError as e:
                logger.error(f"State validation failed: {e}")
                return Command(update={"error": str(e)}, goto=command_goto)
                
        return wrapped_node

    # --- Helper methods for Commands and Send ---
    def create_command(
        self,
        update: Optional[Dict[str, Any]] = None,
        goto: Optional[Union[str, Send, List[Union[str, Send]]]] = None,
        resume: Optional[Union[Any, Dict[str, Any]]] = None,
        graph: Optional[str] = None
    ) -> Command:
        """Create a Command object with the specified parameters."""
        return Command(
            update=update or {}, 
            goto=goto, 
            resume=resume,
            graph=graph
        )

    def create_send(self, node: str, arg: Any) -> Send:
        """Create a Send object to route to another node with specific state."""
        return Send(node, arg)