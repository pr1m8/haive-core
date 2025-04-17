# src/haive/core/schema/schema_composer.py

from typing import Dict, List, Optional, Union, Type, Any, Set, Tuple, get_origin, get_args, Annotated
from pydantic import BaseModel, Field, create_model
from pydantic.fields import FieldInfo
import inspect
import logging
from collections import defaultdict

from haive_core.schema.state_schema import StateSchema

logger = logging.getLogger(__name__)

class SchemaComposer:
    """
    A utility for dynamically composing state schemas from various components.
    
    The SchemaComposer provides methods to:
    - Extract fields from Engine objects
    - Merge schemas with conflict resolution
    - Automatically detect reducers and shared fields
    - Create StateSchema subclasses with the composed fields
    """
    
    def __init__(self, name: str = "ComposedSchema"):
        """
        Initialize a new SchemaComposer.
        
        Args:
            name: Name for the composed schema class
        """
        self.name = name
        self.fields = {}  # Field name -> (type, default/field_info)
        self.shared_fields = set()  # Fields shared with parent graph
        self.reducer_fields = {}  # Field name -> reducer function
        self.field_descriptions = {}  # Field name -> description
        self.field_sources = defaultdict(set)  # Field name -> set of sources
    
    def add_field(
        self,
        name: str,
        field_type: Any,
        default: Any = None,
        description: Optional[str] = None,
        shared: bool = False,
        reducer: Optional[Any] = None,
        source: Optional[str] = None,
        default_factory: Optional[Any] = None,
    ) -> 'SchemaComposer':
        """
        Add a field to the schema with robust default handling.

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
        from pydantic import Field
        
        # Warn about fields without defaults
        if default is None and default_factory is None:
            # For primitive types, make them Optional
            origin_type = get_origin(field_type) or field_type
            if origin_type is not list and origin_type is not dict:
                # Wrap primitive types in Optional if no default
                from typing import Optional as OptionalType
                field_type = OptionalType[field_type]
                default = None
            else:
                # For collections, use empty collection as default
                if origin_type is list or origin_type == List:
                    default_factory = list
                elif origin_type is dict or origin_type == Dict:
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
            self.reducer_fields[name] = reducer

        if source:
            self.field_sources[name].add(source)

        return self
    
    def add_fields_from_dict(self, fields: Dict[str, Union[Tuple[Any, Any], Any]]) -> 'SchemaComposer':
        """
        Add multiple fields from a dictionary.
        
        Args:
            fields: Dictionary mapping field names to (type, default) tuples or single type
            
        Returns:
            Self for chaining
        """
        from typing import get_origin, Optional

        for name, type_default in fields.items():
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
    
    def add_fields_from_model(self, model: Type[BaseModel]) -> 'SchemaComposer':
        """
        Add fields from a Pydantic model.

        Args:
            model: Pydantic model class to extract fields from

        Returns:
            Self for chaining
        """
        from pydantic.fields import FieldInfo

        model_name = model.__name__

        for field_name, field_info in model.model_fields.items():
            field_type = field_info.annotation
            shared = False
            reducer = None

            # Check for Annotated types with reducer
            if get_origin(field_type) is Annotated:
                args = get_args(field_type)
                if len(args) > 1:
                    field_type = args[0]
                    for arg in args[1:]:
                        if callable(arg):
                            reducer = arg
                            break

            # Get default value or default_factory
            default = None
            default_factory = None

            if isinstance(field_info, FieldInfo):
                if field_info.default_factory is not None:
                    default_factory = field_info.default_factory
                elif not field_info.is_required():
                    default = field_info.default
                else:
                    # If required but no default, auto-wrap with Optional
                    from typing import Optional
                    field_type = Optional[field_type]
                    default = None

            description = field_info.description if hasattr(field_info, "description") else None

            self.add_field(
                name=field_name,
                field_type=field_type,
                default=default,
                default_factory=default_factory,
                description=description,
                shared=shared,
                reducer=reducer,
                source=model_name
            )

        return self

    def add_fields_from_engine(self, engine) -> 'SchemaComposer':
        """
        Comprehensively extract fields from an Engine object.
        
        Args:
            engine: Engine object to extract fields from
            
        Returns:
            Self for chaining
        """
        # Use engine name for source tracking
        engine_name = getattr(engine, 'name', 'unnamed_engine')
        
        # Get schema fields directly - this is the primary method for engines
        if hasattr(engine, 'get_schema_fields'):
            try:
                schema_fields = engine.get_schema_fields()
                
                for field_name, (field_type, field_default) in schema_fields.items():
                    # Check for Annotated types with reducers
                    reducer = None
                    if get_origin(field_type) is Annotated:
                        args = get_args(field_type)
                        # First arg is the actual type, remaining args might contain reducers
                        field_type = args[0]
                        for arg in args[1:]:
                            if callable(arg):
                                reducer = arg
                                break
                    
                    # Special handling for messages field
                    if field_name == "messages" and not reducer:
                        from langgraph.graph import add_messages
                        reducer = add_messages
                    
                    self.add_field(
                        name=field_name,
                        field_type=field_type,
                        default=field_default,
                        shared=False,  # Default to not shared
                        reducer=reducer,
                        source=f"{engine_name}.schema_fields"
                    )
            except Exception as e:
                logger.warning(f"Error getting schema fields from engine {engine_name}: {e}")
        
        # If the engine doesn't have get_schema_fields, try other methods
        # This is for backward compatibility
        else:
            # Try input schema
            if hasattr(engine, 'derive_input_schema'):
                try:
                    input_schema = engine.derive_input_schema()
                    self._extract_schema_fields(input_schema, f"{engine_name}.input")
                except Exception as e:
                    logger.warning(f"Error deriving input schema: {e}")
                
            # Try output schema
            if hasattr(engine, 'derive_output_schema'):
                try:
                    output_schema = engine.derive_output_schema()
                    self._extract_schema_fields(output_schema, f"{engine_name}.output")
                except Exception as e:
                    logger.warning(f"Error deriving output schema: {e}")
                
            # Check for structured output model (common in AugLLM)
            if hasattr(engine, 'structured_output_model') and engine.structured_output_model:
                try:
                    self._extract_schema_fields(engine.structured_output_model, f"{engine_name}.output_model")
                except Exception as e:
                    logger.warning(f"Error extracting structured output model: {e}")
                
            # Check for prompt template variables
            if hasattr(engine, 'prompt_template') and engine.prompt_template:
                try:
                    self._extract_prompt_variables(engine.prompt_template, f"{engine_name}.prompt")
                except Exception as e:
                    logger.warning(f"Error extracting prompt variables: {e}")
        
        # Always check for messages field indicator
        if hasattr(engine, 'uses_messages_field') and engine.uses_messages_field:
            self._ensure_messages_field(source=f"{engine_name}.messages")
            
        return self

    def _ensure_messages_field(self, source: str) -> None:
        """Ensure messages field exists with proper reducer configuration."""
        from typing import Sequence
        from langchain_core.messages import BaseMessage
        from langgraph.graph import add_messages
        
        self.add_field(
            name="messages",
            field_type=Annotated[Sequence[BaseMessage], add_messages],
            default_factory=list,
            description="Chat message history",
            reducer=add_messages,
            source=source
        )

    def _extract_schema_fields(self, model: Type[BaseModel], source: str) -> None:
        """Extract fields from a Pydantic model with proper reducer detection."""
        if not model:
            return
        
        # Handle both Pydantic V1 and V2
        fields_dict = getattr(model, 'model_fields', None) or getattr(model, '__fields__', {})
        
        for field_name, field_info in fields_dict.items():
            # Skip internal fields
            if field_name.startswith("__") or field_name in ["runnable_config"]:
                continue
            
            field_type = field_info.annotation if hasattr(field_info, "annotation") else getattr(field_info, "type_", Any)
            
            # Check for Annotated types with reducers
            reducer = None
            if get_origin(field_type) is Annotated:
                args = get_args(field_type)
                # First arg is the actual type, remaining args might contain reducers
                field_type = args[0]
                for arg in args[1:]:
                    if callable(arg):
                        reducer = arg
                        break
            
            # Handle shared field indication
            shared = False
            if hasattr(model, '__shared_fields__') and field_name in model.__shared_fields__:
                shared = True
            
            # Handle default or default_factory
            if hasattr(field_info, 'default_factory') and field_info.default_factory is not None:
                self.add_field(
                    name=field_name,
                    field_type=field_type,
                    default_factory=field_info.default_factory,
                    description=getattr(field_info, 'description', None),
                    shared=shared,
                    reducer=reducer,
                    source=source
                )
            else:
                default = getattr(field_info, 'default', None)
                if hasattr(field_info, 'is_required') and callable(getattr(field_info, 'is_required')):
                    if field_info.is_required() and default is ...:
                        # Make required fields Optional with None default
                        from typing import Optional
                        field_type = Optional[field_type]
                        default = None
                
                self.add_field(
                    name=field_name,
                    field_type=field_type,
                    default=default,
                    description=getattr(field_info, 'description', None),
                    shared=shared,
                    reducer=reducer,
                    source=source
                )

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

    def compose_from_components(self, components: List[Any]) -> 'SchemaComposer':
        """
        Compose a schema from multiple components.
        
        Args:
            components: List of components (engines, models, etc.) to extract fields from
            
        Returns:
            Self for chaining
        """
        for component in components:
            # Handle Engine objects
            if hasattr(component, 'engine_type') and hasattr(component, 'create_runnable'):
                self.add_fields_from_engine(component)
                
            # Handle Pydantic models
            elif inspect.isclass(component) and issubclass(component, BaseModel):
                self.add_fields_from_model(component)
                
            # Handle StateSchema instances
            elif isinstance(component, StateSchema):
                self.add_fields_from_model(component.__class__)
                
                # Also include shared fields information
                for field in component.__class__.__shared_fields__:
                    if field in self.fields:
                        self.shared_fields.add(field)
                        
                # Include reducer information
                for field, reducer in component.__class__.__reducer_fields__.items():
                    if field in self.fields:
                        self.reducer_fields[field] = reducer
                
            # Handle dictionaries of field definitions
            elif isinstance(component, dict):
                self.add_fields_from_dict(component)
                
        return self
    
    def build(self) -> Type[StateSchema]:
        """
        Build the final StateSchema class with all composed fields.
        
        Returns:
            StateSchema subclass with all fields
        """
        field_dict = {}
        
        # Process fields to include reducers in Annotated types
        for field_name, (field_type, default) in self.fields.items():
            # If field has a reducer, wrap in Annotated
            if field_name in self.reducer_fields:
                reducer = self.reducer_fields[field_name]
                
                # Check if it's already Annotated
                if get_origin(field_type) is not Annotated:
                    field_type = Annotated[field_type, reducer]
                
            # Add to field dict
            field_dict[field_name] = (field_type, default)
            
        # Create the model
        schema_cls = StateSchema.create(__name__=self.name, **field_dict)
        
        # Set shared fields
        schema_cls.__shared_fields__ = list(self.shared_fields)
        
        # Ensure reducer fields dictionary is properly set
        schema_cls.__reducer_fields__ = dict(self.reducer_fields)
        
        # Special handling for messages reducer
        if "messages" in self.fields and "messages" not in schema_cls.__reducer_fields__:
            from langgraph.graph import add_messages
            schema_cls.__reducer_fields__["messages"] = add_messages
        
        return schema_cls
    
    @classmethod
    def compose(
        cls,
        components: List[Any],
        name: str = "ComposedSchema",
        include_runnable_config: bool = False
    ) -> Type[StateSchema]:
        """
        Static method to quickly compose a schema from components.

        Args:
            components: List of components to extract fields from
            name: Name for the resulting schema
            include_runnable_config: Whether to include a runnable_config field

        Returns:
            StateSchema subclass with fields from all components
        """
        composer = cls(name=name)
        composer.compose_from_components(components)

        # ✅ Automatically make all required fields Optional if no default
        for field_name, (field_type, field_info) in composer.fields.items():
            if isinstance(field_info, FieldInfo) and field_info.default is ...:
                logger.debug(f"Auto-assigning default=None to required field '{field_name}' during compose()")
                from typing import Optional
                composer.fields[field_name] = (
                    Optional[field_type],
                    Field(default=None)
                )

        # Optional runtime config field
        if include_runnable_config:
            composer.add_field(
                name='runnable_config',
                field_type=Dict[str, Any],
                default={},
                description="Runtime configuration for components"
            )

        return composer.build()
    
    @classmethod
    def compose_as_state_schema(
        cls,
        components: List[Any],
        name: str = "ComposedSchema",
        include_messages: bool = True,
        include_runnable_config: bool = True
    ) -> Type[StateSchema]:
        """
        Compose components into a StateSchema with standard conversions.
        
        Args:
            components: List of components to extract fields from
            name: Name for the resulting schema
            include_messages: Whether to include messages field with add_messages reducer
            include_runnable_config: Whether to include runnable_config field
            
        Returns:
            StateSchema subclass for the composition
        """
        # Create composer
        composer = cls(name=name)
        
        # Process components
        composer.compose_from_components(components)
        
        # Add messages field if needed
        if include_messages and "messages" not in composer.fields:
            from typing import Sequence
            from langchain_core.messages import BaseMessage
            from langgraph.graph import add_messages
            
            composer.add_field(
                name="messages",
                field_type=Annotated[Sequence[BaseMessage], add_messages],
                default_factory=list,
                description="Chat message history",
                reducer=add_messages
            )
        
        # Add runnable_config field if needed
        if include_runnable_config:
            composer.add_field(
                name='runnable_config',
                field_type=Dict[str, Any],
                default_factory=dict,
                description="Runtime configuration for components"
            )
        
        # Build schema
        schema_cls = composer.build()
        
        # Double-check reducer fields registration
        if include_messages and "messages" not in schema_cls.__reducer_fields__:
            from langgraph.graph import add_messages
            schema_cls.__reducer_fields__["messages"] = add_messages
        
        return schema_cls
    
    @classmethod
    def compose_schema(cls, components: List[Any], name: str = "ComposedSchema") -> Type[StateSchema]:
        """
        Creates a schema from components (compatible with older test code).
        
        Args:
            components: List of components to extract fields from
            name: Name for the resulting schema
            
        Returns:
            StateSchema subclass
        """
        return cls.compose_as_state_schema(components, name=name)
    
    @classmethod
    def compose_input_schema(cls, components: List[Any], name: str = "InputSchema") -> Type[BaseModel]:
        """
        Creates an input schema by composing from multiple components.
        
        Args:
            components: List of components to extract fields from
            name: Name for the resulting schema
            
        Returns:
            Input schema Pydantic model
        """
        # Create schema composer
        composer = cls(name=name)
        
        # Process only input fields from engines
        for component in components:
            if hasattr(component, 'engine_type') and hasattr(component, 'derive_input_schema'):
                try:
                    input_schema = component.derive_input_schema()
                    composer.add_fields_from_model(input_schema)
                except Exception as e:
                    logger.warning(f"Error processing input schema: {e}")
        
        # Ensure a messages field exists
        if not any(field_name == 'messages' for field_name in composer.fields):
            from typing import List, Sequence
            from langchain_core.messages import BaseMessage
            
            composer.add_field(
                name="messages",
                field_type=Sequence[BaseMessage],
                default_factory=list,
                description="Chat message history"
            )
        
        # Build as regular Pydantic model
        field_dict = {}
        for field_name, (field_type, default) in composer.fields.items():
            field_dict[field_name] = (field_type, default)
        
        return create_model(name, **field_dict)
    
    @classmethod
    def compose_output_schema(cls, components: List[Any], name: str = "OutputSchema") -> Type[BaseModel]:
        """
        Creates an output schema by composing from multiple components.
        
        Args:
            components: List of components to extract fields from
            name: Name for the resulting schema
            
        Returns:
            Output schema Pydantic model
        """
        # Create schema composer
        composer = cls(name=name)
        
        # Process only output fields from engines
        for component in components:
            if hasattr(component, 'engine_type') and hasattr(component, 'derive_output_schema'):
                try:
                    output_schema = component.derive_output_schema()
                    composer.add_fields_from_model(output_schema)
                except Exception as e:
                    logger.warning(f"Error processing output schema: {e}")
        
        # Ensure a messages field exists
        if not any(field_name == 'messages' for field_name in composer.fields):
            from typing import List, Sequence
            from langchain_core.messages import BaseMessage
            
            composer.add_field(
                name="messages",
                field_type=Sequence[BaseMessage],
                default_factory=list,
                description="Chat message history"
            )
        
        # Build as regular Pydantic model
        field_dict = {}
        for field_name, (field_type, default) in composer.fields.items():
            field_dict[field_name] = (field_type, default)
        
        return create_model(name, **field_dict)
    
    @classmethod
    def create_schema_for_components(cls, components: List[Any], name: str = "ComposedSchema", include_messages: bool = True):
        """
        Creates a schema manager for components (compatible with older test code).
        
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
        if include_messages and not any(field_name == 'messages' for field_name in composer.fields):
            from typing import List, Sequence
            from langchain_core.messages import BaseMessage
            from langgraph.graph import add_messages
            
            composer.add_field(
                name="messages",
                field_type=Annotated[Sequence[BaseMessage], add_messages],
                default_factory=list,
                description="Chat message history",
                reducer=add_messages
            )
        
        return composer
    
    @classmethod
    def create_message_state(cls, 
                            additional_fields: Optional[Dict[str, Tuple[Any, Any]]] = None,
                            name: str = "MessageState") -> Type[StateSchema]:
        """
        Create a standard message-based state schema.
        
        Args:
            additional_fields: Optional additional fields to include
            name: Name for the schema
            
        Returns:
            StateSchema subclass with messages field and additional fields
        """
        from typing import List, Sequence
        from langchain_core.messages import BaseMessage
        from langgraph.graph import add_messages
        
        # Create composer
        composer = cls(name=name)
        
        # Add messages field with add_messages reducer
        composer.add_field(
            name="messages",
            field_type=Sequence[BaseMessage],
            default_factory=list,
            description="Chat message history",
            reducer=add_messages
        )
        
        # Add any additional fields
        if additional_fields:
            composer.add_fields_from_dict(additional_fields)
            
        return composer.build()