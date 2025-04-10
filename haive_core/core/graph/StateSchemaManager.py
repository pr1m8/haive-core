# src/haive/core/graph/StateSchemaManager.py

from typing import Any, Dict, List, Optional, Type, Union, Callable, get_type_hints, Tuple
from pydantic import BaseModel, Field, ValidationError, create_model
import inspect
import uuid
import typing
from langchain_core.messages import BaseMessage
from src.haive.core.engine.aug_llm import AugLLMConfig
import logging

from langgraph.types import Command, Send

# Set up logging
logger = logging.getLogger(__name__)

class StateSchemaManager:
    """
    A dynamic schema manager that:
    - Builds schemas from dicts, lists, or Pydantic models.
    - Merges schemas while preserving uniqueness (no duplicate `messages` field).
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

    def add_field(self, name: str, type_hint: Type, default: Any = None, required: bool = False) -> 'StateSchemaManager':
        if self.locked:
            raise ValueError("Schema is locked and cannot be modified.")
        logger.debug(f"DEBUG: var: {name}")
        logger.debug(f"DEBUG: adding field: {name}")
        field_info = Field(...) if required else Field(default=default)
        self.fields[name] = (type_hint, field_info)
        return self

    def add_computed_property(self, name: str, getter_func: Callable, setter_func: Optional[Callable] = None) -> 'StateSchemaManager':
        """Add a computed property with getter and optional setter."""
        if self.locked:
            raise ValueError("Schema is locked and cannot be modified.")
        self.computed_properties[name] = (getter_func, setter_func)
        return self

    def remove_field(self, name: str) -> 'StateSchemaManager':
        if self.locked:
            raise ValueError("Schema is locked and cannot be modified.")
        if name in self.fields:
            del self.fields[name]
        return self

    def modify_field(self, name: str, new_type: Type, new_default: Any = None) -> 'StateSchemaManager':
        if self.locked:
            raise ValueError("Schema is locked and cannot be modified.")
        if name in self.fields:
            self.fields[name] = (new_type, Field(default=new_default))
        return self

    def merge(self, other: Union['StateSchemaManager', Type[BaseModel]]) -> 'StateSchemaManager':
        if self.locked:
            raise ValueError("Schema is locked and cannot be modified.")

        merged = StateSchemaManager(name=f"{self.name}_Merged", config=self.config)
        merged.fields = self.fields.copy()
        merged.validators = self.validators.copy()
        merged.properties = self.properties.copy()
        merged.computed_properties = self.computed_properties.copy()
        merged.class_methods = self.class_methods.copy()
        merged.static_methods = self.static_methods.copy()
        merged.property_getters = self.property_getters.copy()
        merged.property_setters = self.property_setters.copy()

        if isinstance(other, StateSchemaManager):
            for field, (field_type, field_info) in other.fields.items():
                if field == "messages" and field in merged.fields:
                    continue  
                if field not in merged.fields:
                    merged.fields[field] = (field_type, field_info)

            merged.validators.update(other.validators)
            merged.properties.update(other.properties)
            merged.computed_properties.update(other.computed_properties)
            merged.class_methods.update(other.class_methods)
            merged.static_methods.update(other.static_methods)
            merged.property_getters.update(other.property_getters)
            merged.property_setters.update(other.property_setters)

        elif isinstance(other, type) and issubclass(other, BaseModel):
            return merged.merge(StateSchemaManager(other, config=self.config))

        return merged
    def has_field(self, name: str) -> bool:
        return name in self.fields
    def get_model(self, lock: bool = False) -> Type[BaseModel]:
        if lock:
            self.locked = True
        
        # Debug - print the fields before creating the model
        logger.debug(f"DEBUG: state_model.model_fields: {self.fields}")
        
        try:
            # Create the model with the fields
            model = create_model(self.name, **self.fields)
            
            # Add validators
            for name, validator in self.validators.items():
                setattr(model, name, validator)
    
            # Add properties
            for name, prop in self.properties.items():
                getter = self.property_getters.get(name)
                setter = self.property_setters.get(name)
                setattr(model, name, property(getter, setter))
    
            # Add computed properties
            for name, (getter, setter) in self.computed_properties.items():
                setattr(model, name, property(getter, setter))
    
            # Add class and static methods
            for name, method in self.class_methods.items():
                setattr(model, name, classmethod(method))
    
            for name, method in self.static_methods.items():
                setattr(model, name, staticmethod(method))
            
            return model
        except Exception as e:
            logger.error(f"Error creating model: {e}")
            # Create a minimal fallback model
            return create_model(
                self.name,
                placeholder=(str, Field(default="")),
                messages=(List[Any], Field(default_factory=list))
            )

    def pretty_print(self) -> None:
        """
        Print the schema as if it was written as a Python class.
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
    def create_node_from_config(
        self,
        config: AugLLMConfig,
        command_goto: str = "execute_step",
        async_mode: bool = True
    ) -> Callable[[Dict[str, Any]], Any]:
        """
        Create a node function from an AugLLMConfig, wrapping it with state validation
        using this schema's base model.
        Returns a node function that accepts a dict, validates it, and then calls the underlying node.
        """
        from src.haive.core.graph.NodeFactory import create_node_function
        base_node = create_node_function(config, command_goto, async_mode)
        Model = self.get_model()
        if async_mode:
            async def wrapped_node(state: Dict[str, Any]):
                try:
                    valid_state = Model.model_validate(state).model_dump()
                except ValidationError as e:
                    raise ValueError(f"State validation failed: {e}")
                return await base_node(valid_state)
            return wrapped_node
        else:
            def wrapped_node(state: Dict[str, Any]):
                try:
                    valid_state = Model.model_validate(state).model_dump()
                except ValidationError as e:
                    raise ValueError(f"State validation failed: {e}")
                return base_node(valid_state)
            return wrapped_node

    # --- Helper methods for Commands and Send ---
    def create_default_command(
        self,
        update: Optional[Dict[str, Any]] = None,
        goto: Optional[Union[str, Send, List[Union[str, Send]]]] = None,
        resume: Optional[Union[Any, Dict[str, Any]]] = None
    ) -> Command:
        update = update or {}
        goto = goto or ""
        resume = resume or {}
        return Command(update=update, goto=goto, resume=resume)

    def create_send(self, node: str, arg: Any) -> Send:
        return Send(node, arg)