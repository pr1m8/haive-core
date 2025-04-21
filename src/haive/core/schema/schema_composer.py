import inspect
import logging
from collections import defaultdict
from typing import Annotated, Any, Optional, get_args, get_origin

from pydantic import BaseModel, Field, create_model
from pydantic.fields import FieldInfo

logger = logging.getLogger(__name__)

class SchemaComposer:
    """A utility for dynamically composing state schemas from various components.
    
    The SchemaComposer provides methods to:
    - Extract fields from Engine objects
    - Merge schemas with conflict resolution
    - Identify shared fields
    - Create schema classes with the composed fields
    
    Important: This implementation NEVER adds __runnable_config__ to state schemas.
    """

    def __init__(self, name: str = "ComposedSchema"):
        """Initialize a new SchemaComposer.
        
        Args:
            name: Name for the composed schema class
        """
        self.name = name
        self.fields = {}  # Field name -> (type, default/field_info)
        self.shared_fields = set()  # Fields shared with parent graph
        self.reducer_names = {}  # Field name -> reducer function name
        self.reducer_functions = {}  # Field name -> reducer function
        self.field_descriptions = {}  # Field name -> description
        self.field_sources = defaultdict(set)  # Field name -> set of sources

    def add_fields_from_dict(self, fields: dict[str, tuple[Any, Any] | Any]) -> "SchemaComposer":
        """Add multiple fields from a dictionary.
        
        Args:
            fields: Dictionary mapping field names to (type, default) tuples or single type
            
        Returns:
            Self for chaining
        """
        from typing import get_origin

        for name, type_default in fields.items():
            # Skip __runnable_config__
            if name == "__runnable_config__":
                continue

            # If it's a single type, make it Optional with None default
            if not isinstance(type_default, tuple):
                self.add_field(name, Optional[type_default], default=None)
                continue

            # Unpack the tuple
            if len(type_default) == 2:
                field_type, default = type_default

                # Handle both type scenarios
                if get_origin(field_type) is not Optional and default is None:
                    # Add Optional wrapper if default is None
                    field_type = Optional[field_type]

                self.add_field(name, field_type, default=default)
            else:
                # Fallback for unexpected tuple format
                logger.warning(f"Unexpected field format for {name}: {type_default}")
                field_type = type_default[0]
                self.add_field(name, Optional[field_type], default=None)

        return self

    def add_fields_from_model(self, model: type[BaseModel]) -> "SchemaComposer":
        """Add fields from a Pydantic model.

        Args:
            model: Pydantic model class to extract fields from

        Returns:
            Self for chaining
        """
        model_name = model.__name__

        # Handle Pydantic v2
        if hasattr(model, "model_fields"):
            for field_name, field_info in model.model_fields.items():
                # Skip __runnable_config__
                if field_name == "__runnable_config__" or field_name.startswith("__") or field_name == "runnable_config":
                    continue

                field_type = field_info.annotation
                shared = False
                reducer_name = None
                reducer_func = None

                # Check for Annotated types with reducer
                if get_origin(field_type) is Annotated:
                    args = get_args(field_type)
                    if len(args) > 1:
                        field_type = args[0]
                        for arg in args[1:]:
                            if callable(arg):
                                reducer_func = arg
                                reducer_name = arg.__name__

                # Handle default or default_factory
                if field_info.default_factory is not None:
                    self.add_field(
                        name=field_name,
                        field_type=field_type,
                        default_factory=field_info.default_factory,
                        description=field_info.description,
                        shared=shared,
                        source=model_name,
                        reducer=reducer_func
                    )
                else:
                    default = field_info.default
                    if default is ... and not field_info.is_required():
                        # If required but no default, auto-wrap with Optional
                        from typing import Optional
                        field_type = Optional[field_type]
                        default = None

                    self.add_field(
                        name=field_name,
                        field_type=field_type,
                        default=default,
                        description=field_info.description,
                        shared=shared,
                        source=model_name,
                        reducer=reducer_func
                    )

        # Handle Pydantic v1 (backwards compatibility)
        elif hasattr(model, "__fields__"):
            for field_name, field_info in model.__fields__.items():
                # Skip __runnable_config__
                if field_name == "__runnable_config__" or field_name.startswith("__") or field_name == "runnable_config":
                    continue

                field_type = getattr(field_info, "type_", Any)
                shared = False
                reducer_name = None
                reducer_func = None

                # Check for Annotated types with reducer
                if get_origin(field_type) is Annotated:
                    args = get_args(field_type)
                    if len(args) > 1:
                        field_type = args[0]
                        for arg in args[1:]:
                            if callable(arg):
                                reducer_func = arg
                                reducer_name = arg.__name__

                # Handle default or default_factory
                if hasattr(field_info, "default_factory") and field_info.default_factory is not None:
                    self.add_field(
                        name=field_name,
                        field_type=field_type,
                        default_factory=field_info.default_factory,
                        description=getattr(field_info, "description", None),
                        shared=shared,
                        source=model_name,
                        reducer=reducer_func
                    )
                else:
                    default = getattr(field_info, "default", None)
                    if default is ... and not getattr(field_info, "required", False):
                        # If required but no default, auto-wrap with Optional
                        from typing import Optional
                        field_type = Optional[field_type]
                        default = None

                    self.add_field(
                        name=field_name,
                        field_type=field_type,
                        default=default,
                        description=getattr(field_info, "description", None),
                        shared=shared,
                        source=model_name,
                        reducer=reducer_func
                    )

        # Handle shared fields
        if hasattr(model, "__shared_fields__"):
            for field in model.__shared_fields__:
                if field in self.fields:
                    self.shared_fields.add(field)

        # Handle serializable reducers
        if hasattr(model, "__serializable_reducers__"):
            for field, reducer_name in model.__serializable_reducers__.items():
                if field in self.fields:
                    self.reducer_names[field] = reducer_name

        # Handle actual reducer functions
        if hasattr(model, "__reducer_fields__"):
            for field, reducer in model.__reducer_fields__.items():
                if field in self.fields:
                    self.reducer_functions[field] = reducer

        return self

    def add_fields_from_engine(self, engine) -> "SchemaComposer":
        """Extract fields from an Engine object.
        
        Args:
            engine: Engine object to extract fields from
            
        Returns:
            Self for chaining
        """
        # Use engine name for source tracking
        engine_name = getattr(engine, "name", "unnamed_engine")

        # Get schema fields directly - this is the primary method for engines
        if hasattr(engine, "get_schema_fields"):
            try:
                schema_fields = engine.get_schema_fields()

                for field_name, (field_type, field_default) in schema_fields.items():
                    # Skip __runnable_config__
                    if field_name == "__runnable_config__" or field_name == "runnable_config":
                        continue

                    # Check for Annotated types with reducers
                    reducer_name = None
                    reducer_func = None
                    if get_origin(field_type) is Annotated:
                        args = get_args(field_type)
                        # First arg is the actual type, remaining args might contain reducers
                        field_type = args[0]
                        for arg in args[1:]:
                            if callable(arg):
                                reducer_func = arg
                                reducer_name = arg.__name__

                    # Special handling for messages field
                    if field_name == "messages" and not reducer_name:
                        try:
                            from langgraph.graph import add_messages
                            reducer_func = add_messages
                            reducer_name = "add_messages"
                        except ImportError:
                            # Create a simple concat function as fallback
                            def concat_lists(a, b):
                                return (a or []) + (b or [])
                            reducer_func = concat_lists
                            reducer_name = "concat_lists"

                    self.add_field(
                        name=field_name,
                        field_type=field_type,
                        default=field_default,
                        shared=False,
                        source=f"{engine_name}.schema_fields",
                        reducer=reducer_func
                    )

                    # Record reducer name if found but not set by add_field
                    if reducer_name and field_name not in self.reducer_names:
                        self.reducer_names[field_name] = reducer_name

                    # Record reducer function if found but not set by add_field
                    if reducer_func and field_name not in self.reducer_functions:
                        self.reducer_functions[field_name] = reducer_func

            except Exception as e:
                logger.warning(f"Error getting schema fields from engine {engine_name}: {e}")

        # If the engine doesn't have get_schema_fields, try other methods
        else:
            # Try input schema
            if hasattr(engine, "derive_input_schema"):
                try:
                    input_schema = engine.derive_input_schema()
                    self._extract_schema_fields(input_schema, f"{engine_name}.input")
                except Exception as e:
                    logger.warning(f"Error deriving input schema: {e}")

            # Try output schema
            if hasattr(engine, "derive_output_schema"):
                try:
                    output_schema = engine.derive_output_schema()
                    self._extract_schema_fields(output_schema, f"{engine_name}.output")
                except Exception as e:
                    logger.warning(f"Error deriving output schema: {e}")

            # Check for structured output model
            if hasattr(engine, "structured_output_model") and engine.structured_output_model:
                try:
                    self._extract_schema_fields(engine.structured_output_model, f"{engine_name}.output_model")
                except Exception as e:
                    logger.warning(f"Error extracting structured output model: {e}")

            # Check for prompt template variables
            if hasattr(engine, "prompt_template") and engine.prompt_template:
                try:
                    self._extract_prompt_variables(engine.prompt_template, f"{engine_name}.prompt")
                except Exception as e:
                    logger.warning(f"Error extracting prompt variables: {e}")

        # Check for messages field indicator
        if hasattr(engine, "uses_messages_field") and engine.uses_messages_field:
            self._ensure_messages_field(source=f"{engine_name}.messages")

        return self

    def _ensure_messages_field(self, source: str) -> None:
        """Ensure messages field exists with proper configuration."""
        from collections.abc import Sequence

        from langchain_core.messages import BaseMessage

        # Try to get add_messages for reducer
        try:
            from langgraph.graph import add_messages
            reducer_func = add_messages
            reducer_name = "add_messages"
        except ImportError:
            # Create a simple concat function as fallback
            def concat_lists(a, b):
                return (a or []) + (b or [])
            reducer_func = concat_lists
            reducer_name = "concat_lists"

        self.add_field(
            name="messages",
            field_type=Sequence[BaseMessage],
            default_factory=list,
            description="Chat message history",
            source=source,
            reducer=reducer_func
        )

        # Ensure reducer name is set
        self.reducer_names["messages"] = reducer_name
        self.reducer_functions["messages"] = reducer_func

    def _extract_schema_fields(self, model: type[BaseModel], source: str) -> None:
        """Extract fields from a Pydantic model."""
        if not model:
            return

        # Handle both Pydantic V1 and V2
        fields_dict = getattr(model, "model_fields", None) or getattr(model, "__fields__", {})

        for field_name, field_info in fields_dict.items():
            # Skip internal fields and runnable_config
            if field_name.startswith("__") or field_name == "runnable_config" or field_name == "__runnable_config__":
                continue

            field_type = field_info.annotation if hasattr(field_info, "annotation") else getattr(field_info, "type_", Any)

            # Check for Annotated types with reducers
            reducer_name = None
            reducer_func = None
            if get_origin(field_type) is Annotated:
                args = get_args(field_type)
                # First arg is the actual type, remaining args might contain reducers
                field_type = args[0]
                for arg in args[1:]:
                    if callable(arg):
                        reducer_func = arg
                        reducer_name = arg.__name__

            # Handle shared field indication
            shared = False
            if hasattr(model, "__shared_fields__") and field_name in model.__shared_fields__:
                shared = True

            # Handle default or default_factory
            if hasattr(field_info, "default_factory") and field_info.default_factory is not None:
                self.add_field(
                    name=field_name,
                    field_type=field_type,
                    default_factory=field_info.default_factory,
                    description=getattr(field_info, "description", None),
                    shared=shared,
                    source=source,
                    reducer=reducer_func
                )
            else:
                default = getattr(field_info, "default", None)
                if hasattr(field_info, "is_required") and callable(field_info.is_required):
                    if field_info.is_required() and default is ...:
                        # Make required fields Optional with None default
                        from typing import Optional
                        field_type = Optional[field_type]
                        default = None

                self.add_field(
                    name=field_name,
                    field_type=field_type,
                    default=default,
                    description=getattr(field_info, "description", None),
                    shared=shared,
                    source=source,
                    reducer=reducer_func
                )

            # Record reducer name if found but not set by add_field
            if reducer_name and field_name not in self.reducer_names:
                self.reducer_names[field_name] = reducer_name

            # Record reducer function if found but not set by add_field
            if reducer_func and field_name not in self.reducer_functions:
                self.reducer_functions[field_name] = reducer_func

    def _extract_prompt_variables(self, prompt_template, source: str) -> None:
        """Extract variables from a prompt template."""
        # Get all variables from the template
        variables = set()

        if hasattr(prompt_template, "input_variables"):
            variables.update(prompt_template.input_variables)

        # Special handling for chat templates
        if hasattr(prompt_template, "messages"):
            # Look for MessagesPlaceholder
            for msg in prompt_template.messages:
                if hasattr(msg, "variable_name"):
                    variables.add(msg.variable_name)
                # Check message prompt templates
                if hasattr(msg, "prompt") and hasattr(msg.prompt, "input_variables"):
                    variables.update(msg.prompt.input_variables)

        # Check for partial variables and exclude them
        partial_vars = set()
        if hasattr(prompt_template, "partial_variables"):
            partial_vars.update(prompt_template.partial_variables.keys())

        # Add each variable as a field
        for var_name in variables:
            # Skip __runnable_config__
            if var_name == "__runnable_config__" or var_name == "runnable_config":
                continue

            if var_name not in partial_vars:
                if var_name == "messages":
                    # Special handling for messages
                    self._ensure_messages_field(source)
                else:
                    self.add_field(
                        name=var_name,
                        field_type=str,  # Default to string
                        default=None,
                        description=f"Prompt variable from {source}",
                        source=source
                    )

    def add_field(
        self,
        name: str,
        field_type: Any,
        default: Any = None,
        description: str | None = None,
        shared: bool = False,
        reducer: Any | None = None,
        source: str | None = None,
        default_factory: Any | None = None,
    ) -> "SchemaComposer":
        """Add a field to the schema with robust default handling.

        Args:
            name: Field name
            field_type: Field type
            default: Default value (used if default_factory not provided)
            description: Optional field description
            shared: Whether this field is shared with parent graph
            reducer: Optional reducer function for this field
            source: Optional source identifier for this field
            default_factory: Optional factory function for default value

        Returns:
            Self for chaining
        """
        # CRITICAL: Never add __runnable_config__ to state schemas
        if name == "__runnable_config__":
            logger.warning("Attempted to add __runnable_config__ to state schema. Ignoring.")
            return self


        # Check for Annotated types with reducers
        from typing import Annotated, get_args, get_origin
        if get_origin(field_type) is Annotated:
            args = get_args(field_type)
            if len(args) > 1:
                # First arg is the actual type, remaining args could be reducers
                base_type = args[0]
                for arg in args[1:]:
                    if callable(arg) and not reducer:
                        # Found a reducer
                        reducer = arg
                        break

                # Use the base type for the field
                field_type = base_type

        # Warn about fields without defaults
        if default is None and default_factory is None:
            # For primitive types, make them Optional
            origin_type = get_origin(field_type) or field_type
            if origin_type is not list and origin_type is not dict:
                # Wrap primitive types in Optional if no default
                from typing import Optional as OptionalType
                field_type = OptionalType[field_type]
                default = None
            # For collections, use empty collection as default
            elif origin_type is list or origin_type == list:
                default_factory = list
            elif origin_type is dict or origin_type == dict:
                default_factory = dict

        # Determine field configuration
        if default_factory is not None:
            # Prioritize default_factory
            field_info = Field(default_factory=default_factory)
        elif isinstance(default, FieldInfo):
            # If it's already a Pydantic Field, use as-is
            field_info = default
        elif default is not None:
            # Regular default value
            field_info = Field(default=default)
        else:
            # No default, use Field(default=None)
            field_info = Field(default=None)

        # Store the field
        self.fields[name] = (field_type, field_info)

        # Optional metadata
        if description:
            self.field_descriptions[name] = description

        if shared:
            self.shared_fields.add(name)

        if reducer:
            # Store reducer function
            self.reducer_functions[name] = reducer

            # Store reducer name for serialization
            # No cleaning/normalization of names - keeping exactly as provided
            reducer_name = getattr(reducer, "__name__", str(reducer))
            self.reducer_names[name] = reducer_name

        if source:
            self.field_sources[name].add(source)

        return self

    def compose_from_components(self, components: list[Any]) -> "SchemaComposer":
        """Compose a schema from multiple components.
        
        Args:
            components: List of components (engines, models, etc.) to extract fields from
            
        Returns:
            Self for chaining
        """
        for component in components:
            # Handle Engine objects
            if hasattr(component, "engine_type") and hasattr(component, "create_runnable"):
                self.add_fields_from_engine(component)

            # Handle Pydantic models
            elif inspect.isclass(component) and issubclass(component, BaseModel):
                self.add_fields_from_model(component)

                # Special handling for StateSchema reducers
                if hasattr(component, "__serializable_reducers__"):
                    for field, reducer_name in component.__serializable_reducers__.items():
                        self.reducer_names[field] = reducer_name

                # Special handling for actual reducer functions
                if hasattr(component, "__reducer_fields__"):
                    for field, reducer_func in component.__reducer_fields__.items():
                        self.reducer_functions[field] = reducer_func

            # Handle BaseModel instances
            elif isinstance(component, BaseModel):
                self.add_fields_from_model(component.__class__)

                # Special handling for StateSchema reducers in instance
                if hasattr(component.__class__, "__serializable_reducers__"):
                    for field, reducer_name in component.__class__.__serializable_reducers__.items():
                        self.reducer_names[field] = reducer_name

                # Special handling for actual reducer functions
                if hasattr(component.__class__, "__reducer_fields__"):
                    for field, reducer_func in component.__class__.__reducer_fields__.items():
                        self.reducer_functions[field] = reducer_func

            # Handle dictionaries of field definitions
            elif isinstance(component, dict):
                self.add_fields_from_dict(component)

        return self

    def build(self) -> type[BaseModel]:
        """Build the final model class with all composed fields.
        
        Returns:
            Created model class
        """
        # Import here to avoid circular imports
        from haive.core.schema.state_schema import StateSchema

        field_dict = {}

        # Process fields - ensure no __runnable_config__
        for field_name, (field_type, default) in self.fields.items():
            if field_name != "__runnable_config__":
                field_dict[field_name] = (field_type, default)

        # Create the model
        model = create_model(
            self.name,
            __base__=StateSchema,
            **field_dict
        )

        # Add shared fields
        model.__shared_fields__ = list(self.shared_fields)

        # Add serializable reducers - IMPORTANT: DON'T CLEAN NAMES
        serializable_reducers = {}
        for field, reducer_name in self.reducer_names.items():
            serializable_reducers[field] = reducer_name

        model.__serializable_reducers__ = serializable_reducers

        # Add actual reducer functions
        reducer_fields = {}
        for field, reducer in self.reducer_functions.items():
            reducer_fields[field] = reducer

        model.__reducer_fields__ = reducer_fields

        return model

    # Core class methods

    @classmethod
    def compose(
        cls,
        components: list[Any],
        name: str = "ComposedSchema",
        include_runnable_config: bool = False  # Parameter maintained for backwards compatibility
    ) -> type[BaseModel]:
        """Static method to quickly compose a schema from components.

        Args:
            components: List of components to extract fields from
            name: Name for the resulting schema
            include_runnable_config: Ignored parameter (kept for backwards compatibility only)
            
        Returns:
            Schema class with fields from all components
        """
        # IMPORTANT: We never add runnable_config to state schemas
        if include_runnable_config:
            logger.warning("include_runnable_config=True was specified but is ignored - runnable_config should never be in state schema")

        composer = cls(name=name)
        composer.compose_from_components(components)
        return composer.build()

    # This name is used in tests - must be included for compatibility
    @classmethod
    def compose_schema(
        cls,
        components: list[Any],
        name: str = "ComposedSchema",
        include_runnable_config: bool = False  # Parameter maintained for backwards compatibility
    ) -> type[BaseModel]:
        """Alias for compose method - used in tests.
        
        Args:
            components: List of components to extract fields from
            name: Name for the resulting schema
            include_runnable_config: Ignored parameter
            
        Returns:
            Schema class with fields from all components
        """
        return cls.compose(components, name, include_runnable_config)

    @classmethod
    def compose_as_state_schema(
        cls,
        components: list[Any],
        name: str = "ComposedSchema",
        include_messages: bool = True,
        include_runnable_config: bool = False  # Parameter maintained for backwards compatibility
    ) -> type[BaseModel]:
        """Compose components into a schema with standard fields.
        
        Args:
            components: List of components to extract fields from
            name: Name for the resulting schema
            include_messages: Whether to include messages field
            include_runnable_config: IGNORED - Never adds runnable_config to state schema
            
        Returns:
            Schema class for the composition
        """
        if include_runnable_config:
            logger.warning("include_runnable_config=True was specified but will be ignored - runnable_config should never be in state schema")

        # Create composer
        composer = cls(name=name)

        # Process components
        composer.compose_from_components(components)

        # Add messages field if needed
        if include_messages and "messages" not in composer.fields:
            from collections.abc import Sequence

            from langchain_core.messages import BaseMessage

            # Try to import add_messages reducer
            try:
                from langgraph.graph import add_messages

                # Add with reducer
                composer.add_field(
                    name="messages",
                    field_type=Sequence[BaseMessage],
                    default_factory=list,
                    description="Chat message history",
                    reducer=add_messages
                )

                # Keep exact name for test compatibility
                composer.reducer_names["messages"] = "add_messages"
                composer.reducer_functions["messages"] = add_messages

            except ImportError:
                # Create a simple concat function as fallback
                def concat_lists(a, b):
                    return (a or []) + (b or [])

                # Add without standard reducer
                composer.add_field(
                    name="messages",
                    field_type=Sequence[BaseMessage],
                    default_factory=list,
                    description="Chat message history",
                    reducer=concat_lists
                )

                # Set reducer name and function
                composer.reducer_names["messages"] = "concat_lists"
                composer.reducer_functions["messages"] = concat_lists

        # Build schema
        return composer.build()

    @classmethod
    def compose_input_schema(cls, components: list[Any], name: str = "InputSchema") -> type[BaseModel]:
        """Creates an input schema by composing from multiple components.
        
        Args:
            components: List of components to extract fields from
            name: Name for the resulting schema
            
        Returns:
            Input schema model
        """
        # Create schema composer
        composer = cls(name=name)

        # Process only input fields from engines
        for component in components:
            if hasattr(component, "engine_type") and hasattr(component, "derive_input_schema"):
                try:
                    input_schema = component.derive_input_schema()
                    composer.add_fields_from_model(input_schema)
                except Exception as e:
                    logger.warning(f"Error processing input schema: {e}")

        # Ensure a messages field exists
        if not any(field_name == "messages" for field_name in composer.fields):
            from collections.abc import Sequence

            from langchain_core.messages import BaseMessage

            # Try to import add_messages reducer
            try:
                from langgraph.graph import add_messages
                composer.add_field(
                    name="messages",
                    field_type=Sequence[BaseMessage],
                    default_factory=list,
                    description="Chat message history",
                    reducer=add_messages
                )

                # Set reducer name
                composer.reducer_names["messages"] = "add_messages"
                composer.reducer_functions["messages"] = add_messages

            except ImportError:
                # Create a simple concat function as fallback
                def concat_lists(a, b):
                    return (a or []) + (b or [])

                composer.add_field(
                    name="messages",
                    field_type=Sequence[BaseMessage],
                    default_factory=list,
                    description="Chat message history",
                    reducer=concat_lists
                )

                # Set reducer name
                composer.reducer_names["messages"] = "concat_lists"
                composer.reducer_functions["messages"] = concat_lists

        # Build and return the model
        return composer.build()

    @classmethod
    def compose_output_schema(cls, components: list[Any], name: str = "OutputSchema") -> type[BaseModel]:
        """Creates an output schema by composing from multiple components.
        
        Args:
            components: List of components to extract fields from
            name: Name for the resulting schema
            
        Returns:
            Output schema model
        """
        # Create schema composer
        composer = cls(name=name)

        # Process only output fields from engines
        for component in components:
            if hasattr(component, "engine_type") and hasattr(component, "derive_output_schema"):
                try:
                    output_schema = component.derive_output_schema()
                    composer.add_fields_from_model(output_schema)
                except Exception as e:
                    logger.warning(f"Error processing output schema: {e}")

        # Ensure a messages field exists
        if not any(field_name == "messages" for field_name in composer.fields):
            from collections.abc import Sequence

            from langchain_core.messages import BaseMessage

            # Try to import add_messages reducer
            try:
                from langgraph.graph import add_messages
                composer.add_field(
                    name="messages",
                    field_type=Sequence[BaseMessage],
                    default_factory=list,
                    description="Chat message history",
                    reducer=add_messages
                )

                # Set reducer name
                composer.reducer_names["messages"] = "add_messages"
                composer.reducer_functions["messages"] = add_messages

            except ImportError:
                # Create a simple concat function as fallback
                def concat_lists(a, b):
                    return (a or []) + (b or [])

                composer.add_field(
                    name="messages",
                    field_type=Sequence[BaseMessage],
                    default_factory=list,
                    description="Chat message history",
                    reducer=concat_lists
                )

                # Set reducer name
                composer.reducer_names["messages"] = "concat_lists"
                composer.reducer_functions["messages"] = concat_lists

        # Build and return the model
        return composer.build()

    @classmethod
    def create_schema_for_components(cls, components: list[Any], name: str = "ComposedSchema", include_messages: bool = True):
        """Creates a schema manager for components.
        
        Args:
            components: List of components to extract fields from
            name: Name for the resulting schema
            include_messages: Whether to include a messages field
            
        Returns:
            SchemaComposer instance with fields from components
        """
        composer = cls(name=name)
        composer.compose_from_components(components)

        # Add messages field if needed
        if include_messages and "messages" not in composer.fields:
            from collections.abc import Sequence

            from langchain_core.messages import BaseMessage

            # Try to import add_messages reducer
            try:
                from langgraph.graph import add_messages
                composer.add_field(
                    name="messages",
                    field_type=Sequence[BaseMessage],
                    default_factory=list,
                    description="Chat message history",
                    reducer=add_messages
                )

                # Set reducer name
                composer.reducer_names["messages"] = "add_messages"
                composer.reducer_functions["messages"] = add_messages

            except ImportError:
                # Create a simple concat function as fallback
                def concat_lists(a, b):
                    return (a or []) + (b or [])

                composer.add_field(
                    name="messages",
                    field_type=Sequence[BaseMessage],
                    default_factory=list,
                    description="Chat message history",
                    reducer=concat_lists
                )

                # Set reducer name
                composer.reducer_names["messages"] = "concat_lists"
                composer.reducer_functions["messages"] = concat_lists

        return composer

    @classmethod
    def create_message_state(cls,
                            additional_fields: dict[str, tuple[Any, Any]] | None = None,
                            name: str = "MessageState") -> type[BaseModel]:
        """Create a standard message-based state schema.
        
        Args:
            additional_fields: Optional additional fields to include
            name: Name for the schema
            
        Returns:
            Schema class with messages field and additional fields
        """
        from collections.abc import Sequence

        from langchain_core.messages import BaseMessage

        # Create composer
        composer = cls(name=name)

        # Try to import add_messages reducer
        try:
            from langgraph.graph import add_messages

            # Add with reducer
            composer.add_field(
                name="messages",
                field_type=Sequence[BaseMessage],
                default_factory=list,
                description="Chat message history",
                reducer=add_messages
            )

            # Set reducer name - exact match for test compatibility
            composer.reducer_names["messages"] = "add_messages"
            composer.reducer_functions["messages"] = add_messages

        except ImportError:
            # Create a simple concat function as fallback
            def concat_lists(a, b):
                return (a or []) + (b or [])

            composer.add_field(
                name="messages",
                field_type=Sequence[BaseMessage],
                default_factory=list,
                description="Chat message history",
                reducer=concat_lists
            )

            # Set reducer name
            composer.reducer_names["messages"] = "concat_lists"
            composer.reducer_functions["messages"] = concat_lists

        # Add any additional fields
        if additional_fields:
            composer.add_fields_from_dict(additional_fields)

        return composer.build()
