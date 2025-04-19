import inspect
import logging
import typing
from collections.abc import Callable
from typing import Any, TypeVar, Union, get_origin

from langgraph.types import Command, Send
from pydantic import BaseModel, Field, create_model

from haive_core.schema.state_schema import StateSchema
from haive_core.schema.utils import SchemaUtils

# Type variables
T = TypeVar("T", bound=BaseModel)
SchemaType = TypeVar("SchemaType", bound=type[BaseModel])

# Set up logging
logger = logging.getLogger(__name__)

class StateSchemaManager:
    """Manager for dynamically creating and manipulating state schemas.
    
    Provides utilities for:
    - Building schemas from dicts, lists, or Pydantic models
    - Merging schemas while preserving uniqueness
    - Adding/modifying fields
    - Generating pretty-printed code representations
    - Creating node functions with schema validation
    """

    def __init__(
        self,
        data: dict[str, Any] | type[BaseModel] | None = None,
        name: str | None = None,
        config: dict[str, Any] | None = None
    ):
        """Initialize a new StateSchemaManager.
        
        Args:
            data: Dictionary or Pydantic BaseModel
            name: Custom schema name (defaults to class name if BaseModel)
            config: Optional dictionary for schema customization
        """
        self.fields: dict[str, tuple[Any, Any]] = {}
        self.validators: dict[str, Callable] = {}
        self.properties: dict[str, Callable] = {}
        self.computed_properties: dict[str, tuple[Callable, Callable | None]] = {}
        self.class_methods: dict[str, Callable] = {}
        self.static_methods: dict[str, Callable] = {}
        self.config: dict[str, Any] = config or {}
        self.locked: bool = False
        self.property_getters: dict[str, Callable] = {}
        self.property_setters: dict[str, Callable] = {}
        self.field_descriptions: dict[str, str] = {}

        # Set default name
        if data is None:
            self.name = name or self.config.get("default_schema_name", "UnnamedSchema")
            return

        # Extract name from data if not provided
        if name is None and isinstance(data, type) and issubclass(data, BaseModel):
            self.name = data.__name__
        else:
            self.name = name or "CustomState"

        # Load data based on type
        if isinstance(data, dict) or isinstance(data, typing._TypedDictMeta):
            self._load_from_dict(data)
        elif isinstance(data, type) and issubclass(data, BaseModel):
            self._load_from_model(data)
        elif isinstance(data, BaseModel):
            self._load_from_model(data.__class__)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

    def _load_from_dict(self, data: dict[str, Any]) -> None:
        """Load fields from a dictionary.
        
        Args:
            data: Dictionary to load fields from
        """
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
        """Load fields and methods from a Pydantic model.
        
        Args:
            model_cls: Pydantic model class to load from
        """
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

            # Load shared fields if available
            if hasattr(model_cls, "__shared_fields__"):
                self._shared_fields = set(model_cls.__shared_fields__)

            # Load reducer fields if available
            if hasattr(model_cls, "__reducer_fields__"):
                self._reducer_fields = dict(model_cls.__reducer_fields__)

        except Exception as e:
            logger.error(f"Error loading from model: {e}")
            # Add a placeholder field to ensure we have something
            self.fields["placeholder"] = (str, Field(default=""))

    def _infer_type(self, value: Any) -> type:
        """Infer the type of a value, with special handling for collections.
        
        Args:
            value: Value to infer type from
            
        Returns:
            Inferred type
        """
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

    def add_field(
        self,
        name: str,
        field_type: type,
        default: Any = None,
        config_aware: bool = False,
        default_factory: Callable | None = None,
        description: str | None = None,
        shared: bool = False,
        reducer: Callable | None = None,
        optional: bool = False,
        **kwargs
    ) -> "StateSchemaManager":
        """Add a field to the schema, with comprehensive options.

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
        elif optional or get_origin(field_type) is OptionalType:
            # Use None for optional fields
            field_info = Field(default=None, **field_metadata, **kwargs)
        else:
            # Use ... for required fields
            field_info = Field(default=..., **field_metadata, **kwargs)

        # Add the field
        self.fields[name] = (field_type, field_info)

        # Track additional metadata
        if config_aware:
            if not hasattr(self, "_config_aware_fields"):
                self._config_aware_fields = set()
            self._config_aware_fields.add(name)

        if shared:
            if not hasattr(self, "_shared_fields"):
                self._shared_fields = set()
            self._shared_fields.add(name)

        if reducer:
            if not hasattr(self, "_reducer_fields"):
                self._reducer_fields = {}
            self._reducer_fields[name] = reducer

        return self

    def add_computed_property(
        self,
        name: str,
        getter_func: Callable,
        setter_func: Callable | None = None
    ) -> "StateSchemaManager":
        """Add a computed property with getter and optional setter.
        
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
        """Remove a field from the schema.
        
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
        return self

    def modify_field(
        self,
        name: str,
        new_type: type | None = None,
        new_default: Any | None = None,
        new_description: str | None = None
    ) -> "StateSchemaManager":
        """Modify an existing field's type, default value, or description.
        
        Args:
            name: Name of the field to modify
            new_type: New type for the field
            new_default: New default value
            new_description: New description
            
        Returns:
            Self for chaining
        """
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

    def merge(
    self,
    other: Union["StateSchemaManager", type[BaseModel], BaseModel]
) -> "StateSchemaManager":
        """Merge with another schema, preserving first occurrences.
        
        Args:
            other: Another schema manager or Pydantic model to merge with
            
        Returns:
            New merged StateSchemaManager
        """
        from typing import Optional as OptionalType

        merged = StateSchemaManager(name=f"{self.name}_Merged", config=self.config)
        merged.fields = self.fields.copy()
        merged.field_descriptions = self.field_descriptions.copy()

        # Copy metadata collections
        if hasattr(self, "_config_aware_fields"):
            merged._config_aware_fields = self._config_aware_fields.copy()
        if hasattr(self, "_shared_fields"):
            merged._shared_fields = self._shared_fields.copy()
        if hasattr(self, "_reducer_fields"):
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
            if hasattr(other, "_config_aware_fields"):
                if not hasattr(merged, "_config_aware_fields"):
                    merged._config_aware_fields = set()
                merged._config_aware_fields.update(other._config_aware_fields)

            if hasattr(other, "_shared_fields"):
                if not hasattr(merged, "_shared_fields"):
                    merged._shared_fields = set()
                merged._shared_fields.update(other._shared_fields)

            if hasattr(other, "_reducer_fields"):
                if not hasattr(merged, "_reducer_fields"):
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
            if hasattr(other, "__shared_fields__"):
                if not hasattr(merged, "_shared_fields"):
                    merged._shared_fields = set()
                merged._shared_fields.update(other.__shared_fields__)

            # Copy reducer fields from source model if available
            if hasattr(other, "__reducer_fields__"):
                if not hasattr(merged, "_reducer_fields"):
                    merged._reducer_fields = {}

                # Copy actual reducer functions
                for field, reducer in other.__reducer_fields__.items():
                    if field not in merged._reducer_fields:
                        merged._reducer_fields[field] = reducer

            # Also handle serializable reducers if available
            elif hasattr(other, "__serializable_reducers__"):
                # Try to find reducer functions or create fallbacks
                for field, reducer_name in other.__serializable_reducers__.items():
                    if field not in merged._reducer_fields:
                        # Special handling for messages field with add_messages
                        if field == "messages" and reducer_name == "add_messages":
                            try:
                                from langgraph.graph import add_messages
                                if not hasattr(merged, "_reducer_fields"):
                                    merged._reducer_fields = {}
                                merged._reducer_fields[field] = add_messages
                            except ImportError:
                                # Default fallback for messages reducer
                                if not hasattr(merged, "_reducer_fields"):
                                    merged._reducer_fields = {}
                                merged._reducer_fields[field] = lambda a, b: (a or []) + (b or [])

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

            # Handle shared fields and reducers as with model class
            if hasattr(model_cls, "__shared_fields__"):
                if not hasattr(merged, "_shared_fields"):
                    merged._shared_fields = set()
                merged._shared_fields.update(model_cls.__shared_fields__)

            # Copy reducer fields if available
            if hasattr(model_cls, "__reducer_fields__"):
                if not hasattr(merged, "_reducer_fields"):
                    merged._reducer_fields = {}

                for field, reducer in model_cls.__reducer_fields__.items():
                    if field not in merged._reducer_fields:
                        merged._reducer_fields[field] = reducer

        return merged

    def has_field(self, name: str) -> bool:
        """Check if the schema has a specific field.
        
        Args:
            name: Field name to check
            
        Returns:
            True if field exists, False otherwise
        """
        return name in self.fields

    def get_model(
        self,
        lock: bool = False,
        as_state_schema: bool = True
    ) -> type[BaseModel]:
        """Create a Pydantic model with all configured options.
        
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
        if as_state_schema and hasattr(self, "_shared_fields"):
            model.__shared_fields__ = list(self._shared_fields)

        # Add reducer fields metadata if using StateSchema
        if as_state_schema and hasattr(self, "_reducer_fields"):
            # Convert to the serializable format
            serializable_reducers = {}
            for field, reducer in self._reducer_fields.items():
                reducer_name = getattr(reducer, "__name__", str(reducer))
                # Clean any special characters in the reducer name
                if "*" in reducer_name:
                    reducer_name = reducer_name.replace("*", "")
                serializable_reducers[field] = reducer_name

            model.__serializable_reducers__ = serializable_reducers
            model.__reducer_fields__ = self._reducer_fields  # Keep the original reducers too

        return model

    def _format_type_annotation(self, type_hint: Any) -> str:
        """Format a type hint for pretty printing.
        
        Args:
            type_hint: Type hint to format
            
        Returns:
            Formatted string representation
        """
        return SchemaUtils.format_type_annotation(type_hint)

    def pretty_print(self) -> None:
        """Print the schema as a Python class definition.
        """
        print(self.get_pretty_print_output())

    def get_pretty_print_output(self) -> str:
        """Get the schema as a Python class definition string.
        
        Returns:
            Formatted string representation
        """
        shared_fields = getattr(self, "_shared_fields", set())
        reducer_fields = getattr(self, "_reducer_fields", {})

        return SchemaUtils.format_schema_as_python(
            schema_name=self.name,
            fields=self.fields,
            properties=self.properties,
            computed_properties=self.computed_properties,
            class_methods=self.class_methods,
            static_methods=self.static_methods,
            field_descriptions=self.field_descriptions,
            shared_fields=shared_fields,
            reducer_fields=reducer_fields,
            base_class="StateSchema"
        )

    def create_node_function(
        self,
        func: Callable,
        command_goto: str | None = None
    ) -> Callable[[dict[str, Any]], Any]:
        """Create a node function that validates state with this schema.
        
        Args:
            func: The function to wrap
            command_goto: Optional next node to route to
            
        Returns:
            Wrapped node function
        """
        Model = self.get_model()

        def wrapped_node(state: dict[str, Any], config: dict[str, Any] | None = None):
            try:
                # Validate state against our schema
                valid_state = None
                if hasattr(Model, "model_validate"):
                    # Pydantic v2
                    valid_state = Model.model_validate(state)
                else:
                    # Pydantic v1
                    valid_state = Model.parse_obj(state)

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
                if isinstance(result, dict):
                    # Wrap dict in Command
                    return Command(update=result, goto=command_goto)
                # Convert other results to dict update
                return Command(update={"result": result}, goto=command_goto)

            except Exception as e:
                logger.error(f"State validation failed: {e}")
                return Command(update={"error": str(e)}, goto=command_goto)

        return wrapped_node

    def create_command(
        self,
        update: dict[str, Any] | None = None,
        goto: str | Send | list[str | Send] | None = None,
        resume: Any | dict[str, Any] | None = None,
        graph: str | None = None
    ) -> Command:
        """Create a Command object with specified parameters.
        
        Args:
            update: State updates
            goto: Next node to route to
            resume: Value to resume with
            graph: Target graph
            
        Returns:
            Command object
        """
        return Command(
            update=update or {},
            goto=goto,
            resume=resume,
            graph=graph
        )

    def create_send(self, node: str, arg: Any) -> Send:
        """Create a Send object to route to another node.
        
        Args:
            node: Target node name
            arg: State to send
            
        Returns:
            Send object
        """
        return Send(node, arg)
