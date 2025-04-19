# src/haive/core/graph/StateSchemaManager.py

import inspect
import logging
import typing
from collections.abc import Callable
from typing import Any, Union

from langgraph.types import Command, Send
from pydantic import BaseModel, Field, ValidationError, create_model

from haive_core.engine.aug_llm.base import AugLLMConfig

# Set up logging
logger = logging.getLogger(__name__)

class StateSchemaManager:
    """A dynamic schema manager that:
    - Builds schemas from dicts, lists, or Pydantic models.
    - Merges schemas while preserving uniqueness (no duplicate `messages` field).
    - Tracks validators, properties, computed attributes, class/static methods.
    - Supports recursive field merging for nested schemas.
    - Uses a config dictionary for settings and overrides.
    - Prints schema nicely in Python class format.
    """

    def __init__(self,
                 data: dict[str, Any] | type[BaseModel] | None = None,
                 name: str | None = None,
                 config: dict[str, Any] | None = None):
        """Args:
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
        #self.input_fields =
        #self.output_fields
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
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

    def _load_from_dict(self, data: dict[str, Any]) -> None:
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

    def _load_from_model(self, model_cls: type[BaseModel]) -> None:
        try:
            # Handle Pydantic v2
            if hasattr(model_cls, "model_fields"):
                for field_name, field_info in model_cls.model_fields.items():
                    self.fields[field_name] = (field_info.annotation, field_info)
            # Handle Pydantic v1
            elif hasattr(model_cls, "__fields__"):
                for field_name, field_info in model_cls.__fields__.items():
                    self.fields[field_name] = (field_info.type_, field_info)

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

    def _infer_type(self, value: Any) -> type:
        """Infer the type of a value, with special handling for collections."""
        if isinstance(value, str):
            return str
        if isinstance(value, int):
            return int
        if isinstance(value, float):
            return float
        if isinstance(value, bool):
            return bool
        if isinstance(value, list):
            return list[Any]  # We can't guarantee homogeneity so use Any
        if isinstance(value, dict):
            return dict[str, Any]
        return Any
    def add_field(self, name: str, type_hint: type, default: Any = None,
             config_aware: bool = False, **kwargs) -> "StateSchemaManager":
        """Add a field to the schema, with optional config awareness.

        Args:
            name: Name of the field to add
            type_hint: Type of the field
            default: Default value for the field
            config_aware: Whether the field is config-aware
            **kwargs: Additional arguments for field creation
        
        Returns:
            Updated StateSchemaManager
        """
        if self.locked:
            raise ValueError("Schema is locked and cannot be modified.")

        # Simplified field handling for Pydantic
        if default is None and "default_factory" in kwargs:
            # Use default_factory if no explicit default is provided
            default = kwargs.pop("default_factory")()

        # Create the field
        field_info = Field(default=default, **kwargs)

        # Add the field
        self.fields[name] = (type_hint, field_info)

        # Track config awareness
        if config_aware:
            if not hasattr(self, "_config_aware_fields"):
                self._config_aware_fields = set()
            self._config_aware_fields.add(name)

        return self

    def add_computed_property(self, name: str, getter_func: Callable, setter_func: Callable | None = None) -> "StateSchemaManager":
        """Add a computed property with getter and optional setter."""
        if self.locked:
            raise ValueError("Schema is locked and cannot be modified.")
        self.computed_properties[name] = (getter_func, setter_func)
        return self

    def remove_field(self, name: str) -> "StateSchemaManager":
        if self.locked:
            raise ValueError("Schema is locked and cannot be modified.")
        if name in self.fields:
            del self.fields[name]
        return self

    def modify_field(self, name: str, new_type: type, new_default: Any = None) -> "StateSchemaManager":
        if self.locked:
            raise ValueError("Schema is locked and cannot be modified.")
        if name in self.fields:
            self.fields[name] = (new_type, Field(default=new_default))
        return self

    def merge(self, other: Union["StateSchemaManager", type[BaseModel]]) -> "StateSchemaManager":
        """Enhanced merge to preserve first occurrence and handle conflict more gracefully.
        
        Args:
            other: Another schema manager or Pydantic model to merge with
            
        Returns:
            Merged StateSchemaManager
        """
        merged = StateSchemaManager(name=f"{self.name}_Merged", config=self.config)
        merged.fields = self.fields.copy()

        # Process incoming fields
        if isinstance(other, StateSchemaManager):
            for field, (field_type, field_info) in other.fields.items():
                # Skip if field already exists to preserve first occurrence
                if field not in merged.fields:
                    merged.fields[field] = (field_type, field_info)
        elif isinstance(other, type) and issubclass(other, BaseModel):
            # Handle Pydantic model fields
            if hasattr(other, "model_fields"):  # Pydantic v2
                for field_name, field_info in other.model_fields.items():
                    if field_name not in merged.fields:
                        # Use Optional type for added flexibility
                        from typing import Optional
                        optional_type = Optional[field_info.annotation]
                        merged.fields[field_name] = (optional_type, field_info.default)
            elif hasattr(other, "__fields__"):  # Pydantic v1
                for field_name, field_info in other.__fields__.items():
                    if field_name not in merged.fields:
                        # Use Optional type for added flexibility
                        from typing import Optional
                        optional_type = Optional[field_info.type_]
                        merged.fields[field_name] = (optional_type, field_info.default)

        return merged
    def has_field(self, name: str) -> bool:
        return name in self.fields
    def get_model(self, lock: bool = False) -> type[BaseModel]:
        """Create a Pydantic model with config awareness."""
        if lock:
            self.locked = True

        # Create the base model
        model = create_model(self.name, **self.fields)

        # Add config awareness if tracked
        if hasattr(self, "_config_aware_fields"):
            model._config_aware_fields = self._config_aware_fields

            # Add config application method
            def apply_config(self, config):
                """Apply configuration to config-aware fields."""
                if not hasattr(config, "configurable"):
                    return self

                # Apply config to config-aware fields
                for field in getattr(self.__class__, "_config_aware_fields", set()):
                    if hasattr(self, field) and field in config["configurable"]:
                        setattr(self, field, config["configurable"][field])

                return self

            model.apply_config = apply_config

        return model

    def pretty_print(self) -> None:
        """Print the schema as if it was written as a Python class.
        """
        print(f"class {self.name}(BaseModel):")
        if not any([self.fields, self.properties, self.computed_properties, self.class_methods, self.static_methods]):
            print("    pass  # No fields defined\n")
            return

        for field_name, (field_type, field_info) in self.fields.items():
            type_str = str(field_type).replace("typing.", "").replace("NoneType", "Optional")
            default_val = field_info.default if field_info.default != ... else "..."
            default_str = f" = {default_val}" if default_val != "..." else ""
            print(f"    {field_name}: {type_str}{default_str}")

        for prop_name in self.properties.keys():
            print("\n    @property")
            print(f"    def {prop_name}(self): ...  # Regular property")

        for prop_name in self.computed_properties.keys():
            print("\n    @property")
            print(f"    def {prop_name}(self): ...  # Computed property")

        for method_name in self.class_methods.keys():
            print("\n    @classmethod")
            print(f"    def {method_name}(cls): ...  # Class method")

        for method_name in self.static_methods.keys():
            print("\n    @staticmethod")
            print(f"    def {method_name}(): ...  # Static method")

        print()

    # --- Node creation methods ---
    def create_node_from_config(
        self,
        config: AugLLMConfig,
        command_goto: str = "execute_step",
        async_mode: bool = True
    ) -> Callable[[dict[str, Any]], Any]:
        """Create a node function from an AugLLMConfig, wrapping it with state validation
        using this schema's base model.
        Returns a node function that accepts a dict, validates it, and then calls the underlying node.
        """
        from haive_core.graph.NodeFactory import create_node_function
        base_node = create_node_function(config, command_goto, async_mode)
        Model = self.get_model()
        if async_mode:
            async def wrapped_node(state: dict[str, Any]):
                try:
                    valid_state = Model.model_validate(state).model_dump()
                except ValidationError as e:
                    raise ValueError(f"State validation failed: {e}")
                return await base_node(valid_state)
            return wrapped_node
        def wrapped_node(state: dict[str, Any]):
            try:
                valid_state = Model.model_validate(state).model_dump()
            except ValidationError as e:
                raise ValueError(f"State validation failed: {e}")
            return base_node(valid_state)
        return wrapped_node

    # --- Helper methods for Commands and Send ---
    def create_default_command(
        self,
        update: dict[str, Any] | None = None,
        goto: str | Send | list[str | Send] | None = None,
        resume: Any | dict[str, Any] | None = None
    ) -> Command:
        update = update or {}
        goto = goto or ""
        resume = resume or {}
        return Command(update=update, goto=goto, resume=resume)

    def create_send(self, node: str, arg: Any) -> Send:
        return Send(node, arg)
