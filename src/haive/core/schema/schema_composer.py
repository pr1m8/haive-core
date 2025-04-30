"""
SchemaComposer for the Haive framework.

This module provides the SchemaComposer class which dynamically composes
StateSchema classes from various components such as engines, models, and dictionaries.
It solves the common challenge of combining fields and metadata from multiple sources
into a cohesive schema definition.
"""

import inspect
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Type, get_args,\
    get_origin, Annotated, Union, Callable,TYPE_CHECKING

from pydantic import BaseModel, Field, create_model

from haive.core.schema.state_schema import StateSchema
from haive.core.schema.field_definition import FieldDefinition
from haive.core.schema.field_extractor import FieldExtractor
logger = logging.getLogger(__name__)
if TYPE_CHECKING:
    from haive.core.schema.state_schema_manager import StateSchemaManager

class SchemaComposer:
    """
    Utility for extracting field information from components and composing schemas.
    
    The SchemaComposer provides a mechanism to automatically derive state schemas
    from various components such as engines, models, and dictionaries. It handles
    field extraction, reducer setup, and proper schema composition.
    
    Attributes:
        name (str): Name for the composed schema class.
        fields (Dict[str, FieldDefinition]): Dictionary mapping field names to field definitions.
        shared_fields (Set[str]): Set of fields that are shared with parent graphs.
        field_sources (Dict[str, Set[str]]): Dictionary tracking where each field came from.
        input_fields (Dict[str, Set[str]]): Dictionary mapping engine names to input field sets.
        output_fields (Dict[str, Set[str]]): Dictionary mapping engine names to output field sets.
        engine_io_mappings (Dict[str, Dict[str, List[str]]]): Dictionary mapping engine names to
            input/output field mappings.
    """

    def __init__(self, name: str = "ComposedSchema"):
        """
        Initialize a new SchemaComposer.
        
        Args:
            name: Name for the schema to be created.
        """
        self.name = name
        self.fields: Dict[str, FieldDefinition] = {}
        self.shared_fields: Set[str] = set()
        self.field_sources: Dict[str, Set[str]] = defaultdict(set)
        
        # Track input/output mappings for engines
        self.input_fields: Dict[str, Set[str]] = defaultdict(set)
        self.output_fields: Dict[str, Set[str]] = defaultdict(set)
        self.engine_io_mappings: Dict[str, Dict[str, List[str]]] = {}

    def _collect_field(self, name: str, field_type: Type, default: Any = None, 
                      default_factory: Optional[Callable[[], Any]] = None,
                      description: Optional[str] = None, 
                      shared: bool = False,
                      reducer: Optional[Callable] = None, 
                      source: Optional[str] = None) -> None:
        """
        Internal helper to collect a field with metadata.
        
        Args:
            name: Field name
            field_type: Field type
            default: Default value
            default_factory: Factory function for default value
            description: Field description
            shared: Whether field is shared
            reducer: Reducer function
            source: Source of the field
        """
        # Create field definition
        field_def = FieldDefinition(
            name=name,
            field_type=field_type,
            default=default,
            default_factory=default_factory,
            description=description,
            shared=shared,
            reducer=reducer
        )
        
        # Store the field
        self.fields[name] = field_def
        
        # Track additional metadata
        if shared:
            self.shared_fields.add(name)
            
        if source:
            self.field_sources[name].add(source)

    def add_field(
        self,
        name: str,
        field_type: Type,
        default: Any = None,
        default_factory: Optional[Callable[[], Any]] = None,
        description: Optional[str] = None,
        shared: bool = False,
        reducer: Optional[Callable] = None,
        source: Optional[str] = None
    ) -> 'SchemaComposer':
        """
        Add a field definition to the schema.
        
        Args:
            name: Field name to add.
            field_type: Type of the field.
            default: Default value for the field.
            default_factory: Optional factory function for default value.
            description: Optional field description.
            shared: Whether field is shared with parent graph.
            reducer: Optional reducer function for the field.
            source: Optional source identifier for tracking.
            
        Returns:
            Self for method chaining.
        """
        # Skip special fields
        if name == "__runnable_config__" or name == "runnable_config":
            logger.warning(f"Skipping special field {name}")
            return self
            
        # Collect the field with metadata
        self._collect_field(
            name=name,
            field_type=field_type,
            default=default,
            default_factory=default_factory,
            description=description,
            shared=shared,
            reducer=reducer,
            source=source
        )
            
        return self

    def add_fields_from_model(self, model: Type[BaseModel]) -> 'SchemaComposer':
        """
        Extract fields from a Pydantic model and add them to the schema.
        
        Args:
            model: Pydantic model to extract fields from.
            
        Returns:
            Self for method chaining.
        """
        # Skip non-models
        if not isinstance(model, type) or not issubclass(model, BaseModel):
            logger.warning(f"Expected Pydantic model, got {type(model)}")
            return self
            
        # Get fields from Pydantic v2 model_fields
        model_name = model.__name__
        fields_dict = model.model_fields
        
        # Extract fields
        for field_name, field_info in fields_dict.items():
            # Skip internal fields
            if field_name.startswith("__") or field_name == "runnable_config":
                continue
                
            # Get field type and default
            field_type = field_info.annotation
            default = field_info.default
            default_factory = field_info.default_factory
            description = field_info.description
            
            # Check if this is a StateSchema with shared fields
            shared = False
            if issubclass(model, StateSchema) and field_name in model.__shared_fields__:
                shared = True
                
            # Check if field has a reducer in the model
            reducer = None
            if issubclass(model, StateSchema) and hasattr(model, "__reducer_fields__"):
                reducer = model.__reducer_fields__.get(field_name)
                
            # Only add field if not already present (prioritizing earlier components)
            if field_name not in self.fields:
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
                
        # Extract engine I/O mappings if available
        if issubclass(model, StateSchema):
            if hasattr(model, "__engine_io_mappings__"):
                for engine_name, mapping in model.__engine_io_mappings__.items():
                    self.engine_io_mappings[engine_name] = mapping.copy()
                    
            if hasattr(model, "__input_fields__"):
                for engine_name, fields in model.__input_fields__.items():
                    self.input_fields[engine_name].update(fields)
                    
            if hasattr(model, "__output_fields__"):
                for engine_name, fields in model.__output_fields__.items():
                    self.output_fields[engine_name].update(fields)
                
        return self

    def add_fields_from_dict(self, fields: Dict[str, Any]) -> 'SchemaComposer':
        """
        Add fields from a dictionary representation.
        
        The dictionary can contain either (type, default) tuples or direct values
        from which types will be inferred.
        
        Args:
            fields: Dictionary of field definitions.
            
        Returns:
            Self for method chaining.
        """
        # Extract all field information
        extracted_fields, descriptions, shared_fields, reducer_names, reducer_functions, engine_io_mappings, input_fields, output_fields = \
            FieldExtractor.extract_from_dict(fields)
        
        # Add fields that aren't already present
        for field_name, (field_type, field_info) in extracted_fields.items():
            if field_name not in self.fields:
                # Get default and default_factory
                default = field_info.default
                default_factory = field_info.default_factory
                
                # Determine if this field is shared
                is_shared = field_name in shared_fields
                
                # Get reducer if available
                reducer = reducer_functions.get(field_name)
                
                # Add the field with all metadata
                self.add_field(
                    name=field_name,
                    field_type=field_type,
                    default=default,
                    default_factory=default_factory,
                    description=descriptions.get(field_name),
                    shared=is_shared,
                    reducer=reducer,
                    source="dict"
                )
        
        # Update engine I/O mappings
        for engine_name, mapping in engine_io_mappings.items():
            self.engine_io_mappings[engine_name] = mapping.copy()
            
        # Update input/output field tracking
        for engine_name, fields_set in input_fields.items():
            self.input_fields[engine_name].update(fields_set)
            
        for engine_name, fields_set in output_fields.items():
            self.output_fields[engine_name].update(fields_set)
                
        return self

    def extract_fields_from_engine(self, engine: Any) -> 'SchemaComposer':
        """
        Extract fields from an Engine object (alias for add_fields_from_engine).
        
        Args:
            engine: Engine to extract fields from.
                
        Returns:
            Self for method chaining.
        """
        return self.add_fields_from_engine(engine)
    def add_fields_from_engine(self, engine: Any) -> 'SchemaComposer':
        """
        Extract fields from an Engine object.
        
        Attempts multiple methods to extract schema information from engine objects,
        including get_schema_fields(), derive_input_schema(), and derive_output_schema().
        
        Args:
            engine: Engine to extract fields from.
            
        Returns:
            Self for method chaining.
        """
        source = getattr(engine, "name", str(engine))
        logger.debug(f"Extracting fields from engine: {source}")
        
        # Get fields, descriptions, mappings, etc. from the engine
        fields, descriptions, engine_io_mappings, input_fields, output_fields = \
            FieldExtractor.extract_from_engine(engine)
        
        logger.debug(f"Extracted fields from engine {source}: {list(fields.keys())}")
        
        # Add fields that aren't already present
        for field_name, (field_type, field_info) in fields.items():
            if field_name not in self.fields:
                # Get default and default_factory
                default = field_info.default
                default_factory = field_info.default_factory
                
                # For fields from engines, we make them optional
                # unless explicitly marked as required
                from typing import Optional, get_origin
                is_required = default is ... and default_factory is None
                
                # If field is not explicitly required, make it optional
                # and ensure it has a default of None
                if not is_required:
                    if get_origin(field_type) is not Optional:
                        field_type = Optional[field_type]
                    # For fields from engines that lack a default value,
                    # we set the default to None to make them truly optional
                    if default is ... and default_factory is None:
                        default = None
                
                # Add the field with metadata
                self.add_field(
                    name=field_name,
                    field_type=field_type,
                    default=default if default is not ... else None,
                    default_factory=default_factory,
                    description=descriptions.get(field_name),
                    source=source
                )
        
        # Special handling for AugLLMConfig with structured_output_model
        if hasattr(engine, "structured_output_model") and engine.structured_output_model is not None:
            logger.debug(f"Found structured_output_model in {source}: {engine.structured_output_model.__name__}")
            
            # Add fields from the structured output model
            structured_model = engine.structured_output_model
            
            if hasattr(structured_model, "model_fields"):
                # Pydantic v2
                for field_name, field_info in structured_model.model_fields.items():
                    if field_name not in self.fields:
                        # Add field from structured output model
                        self.add_field(
                            name=field_name,
                            field_type=field_info.annotation,
                            default=field_info.default,
                            default_factory=field_info.default_factory,
                            description=field_info.description,
                            source=f"{source}.structured_output_model"
                        )
                        
                        # Add to output fields for this engine
                        self.output_fields[source].add(field_name)
                        
                        logger.debug(f"Added field {field_name} from structured_output_model")
            
            # Also add the model itself as a field with its own name in lowercase
            model_name = structured_model.__name__.lower()
            if model_name not in self.fields:
                from typing import Optional
                self.add_field(
                    name=model_name,
                    field_type=Optional[structured_model],
                    default=None,
                    description=f"Output in {structured_model.__name__} format",
                    source=f"{source}.structured_output_model"
                )
                
                # Add to output fields for this engine
                self.output_fields[source].add(model_name)
                
                logger.debug(f"Added field {model_name} from structured_output_model name")
        
        # Update engine I/O mappings
        for engine_name, mapping in engine_io_mappings.items():
            self.engine_io_mappings[engine_name] = mapping.copy()
            
        # Update input/output field tracking
        for engine_name, fields_set in input_fields.items():
            self.input_fields[engine_name].update(fields_set)
            
        for engine_name, fields_set in output_fields.items():
            self.output_fields[engine_name].update(fields_set)
            
        return self

    def update_engine_io_mapping(self, engine_name: str) -> None:
        """
        Update engine I/O mapping for a specific engine.
        
        This ensures that the mapping in engine_io_mappings is consistent
        with the current input_fields and output_fields.
        
        Args:
            engine_name: Engine name to update mapping for.
        """
        # Create mapping if it doesn't exist
        if engine_name not in self.engine_io_mappings:
            self.engine_io_mappings[engine_name] = {
                "inputs": [],
                "outputs": []
            }
            
        # Update inputs from tracked fields
        if engine_name in self.input_fields:
            self.engine_io_mappings[engine_name]["inputs"] = list(self.input_fields[engine_name])
            
        # Update outputs from tracked fields
        if engine_name in self.output_fields:
            self.engine_io_mappings[engine_name]["outputs"] = list(self.output_fields[engine_name])

    def ensure_messages_field(self, add_if_missing: bool = True) -> 'SchemaComposer':
        """
        Ensure the messages field exists with proper configuration.
        
        Adds a messages field with an appropriate reducer if one doesn't already exist.
        
        Args:
            add_if_missing: Whether to add the messages field if missing.
        
        Returns:
            Self for method chaining.
        """
        if add_if_missing and "messages" not in self.fields:
            from typing import List
            try:
                from langgraph.graph import add_messages
                from langchain_core.messages import BaseMessage
                
                # Add the field with the reducer
                self.add_field(
                    name="messages",
                    field_type=List[BaseMessage],
                    default_factory=list,
                    description="Messages for agent conversation",
                    reducer=add_messages,
                    source="default"
                )
            except ImportError:
                # Fallback if add_messages is not available
                try:
                    from langchain_core.messages import BaseMessage
                    
                    # Create simple concat lists reducer
                    def concat_lists(a, b):
                        return (a or []) + (b or [])
                    
                    self.add_field(
                        name="messages",
                        field_type=List[BaseMessage],
                        default_factory=list,
                        description="Messages for agent conversation",
                        reducer=concat_lists,
                        source="default"
                    )
                except ImportError:
                    # Last resort with Any
                    from typing import Any
                    self.add_field(
                        name="messages",
                        field_type=List[Any],
                        default_factory=list,
                        description="Messages for agent conversation",
                        source="default"
                    )
                    
        return self

    def to_manager(self) -> 'StateSchemaManager':
        """
        Convert to a StateSchemaManager for further manipulation.
        
        Returns:
            StateSchemaManager with fields from this composer.
        """
        from haive.core.schema.schema_manager import StateSchemaManager
        return StateSchemaManager(self)

    def build(self) -> Type[StateSchema]:
        """
        Build a StateSchema directly from the composer.
        
        Creates a new StateSchema subclass with all fields, reducers, and metadata
        that have been collected.
        
        Returns:
            StateSchema class with fields from all components.
        """
        # Create field definitions for the model
        field_defs = {}
        for name, field_def in self.fields.items():
            field_type, field_info = field_def.to_field_info()
            field_defs[name] = (field_type, field_info)
            
        # Create the base schema
        schema = create_model(
            self.name,
            __base__=StateSchema,
            **field_defs
        )
        
        # Add shared fields
        schema.__shared_fields__ = list(self.shared_fields)
        
        # Add reducers
        schema.__serializable_reducers__ = {}
        schema.__reducer_fields__ = {}
        
        for name, field_def in self.fields.items():
            if field_def.reducer:
                reducer_name = field_def.get_reducer_name()
                schema.__serializable_reducers__[name] = reducer_name
                schema.__reducer_fields__[name] = field_def.reducer
                
        # Make sure to deep copy the engine I/O mappings to avoid reference issues
        schema.__engine_io_mappings__ = {}
        for engine_name, mapping in self.engine_io_mappings.items():
            schema.__engine_io_mappings__[engine_name] = mapping.copy()
        
        # Same for input/output fields - convert sets to lists and deep copy
        schema.__input_fields__ = {}
        for engine_name, fields in self.input_fields.items():
            schema.__input_fields__[engine_name] = list(fields)
        
        schema.__output_fields__ = {}
        for engine_name, fields in self.output_fields.items():
            schema.__output_fields__[engine_name] = list(fields)
        
        return schema

    @classmethod
    def from_components(cls, components: List[Any], name: str = "ComposedSchema") -> Type[StateSchema]:
        """
        Create a schema from components.
        
        Convenience method to create a schema from a list of components in one step.
        
        Args:
            components: List of components to extract fields from.
            name: Name for the resulting schema.
            
        Returns:
            StateSchema class with fields from all components.
        """
        composer = cls(name=name)
        
        # Process each component
        for component in components:
            if component is None:
                continue
                
            # Process based on type
            if hasattr(component, "engine_type"):
                # Looks like an Engine
                composer.extract_fields_from_engine(component)
            elif isinstance(component, BaseModel):
                # BaseModel instance
                composer.add_fields_from_model(component.__class__)
            elif isinstance(component, type) and issubclass(component, BaseModel):
                # BaseModel class
                composer.add_fields_from_model(component)
            elif isinstance(component, dict):
                # Dictionary
                composer.add_fields_from_dict(component)
            else:
                logger.debug(f"Skipping unsupported component: {type(component)}")
                
        # Ensure messages field
        composer.ensure_messages_field()
        
        # Build the schema
        return composer.build()

    @classmethod
    def create_message_state(cls, additional_fields: Optional[Dict[str, Any]] = None, 
                            name: str = "MessageState") -> Type[StateSchema]:
        """
        Create a schema with messages field and additional fields.
        
        Creates a standard state schema with a messages field and any additional
        fields specified.
        
        Args:
            additional_fields: Additional fields to add to schema.
            name: Name for the schema.
            
        Returns:
            StateSchema with messages field and additional fields.
        """
        # Create composer
        composer = cls(name=name)
        
        # Add messages field with reducer
        from typing import List
        try:
            from langgraph.graph import add_messages
            from langchain_core.messages import BaseMessage
            
            # Add messages field with reducer
            composer.add_field(
                name="messages",
                field_type=List[BaseMessage],
                default_factory=list,
                description="Messages for conversation",
                reducer=add_messages
            )
        except ImportError:
            # Fallback if add_messages is not available
            try:
                from langchain_core.messages import BaseMessage
                
                # Create simple concat lists reducer
                def concat_lists(a, b):
                    return (a or []) + (b or [])
                
                composer.add_field(
                    name="messages",
                    field_type=List[BaseMessage],
                    default_factory=list,
                    description="Messages for conversation",
                    reducer=concat_lists
                )
            except ImportError:
                # Last resort with Any
                from typing import Any
                composer.add_field(
                    name="messages",
                    field_type=List[Any],
                    default_factory=list,
                    description="Messages for conversation"
                )
        
        # Add additional fields
        if additional_fields:
            for name, value in additional_fields.items():
                if isinstance(value, tuple) and len(value) >= 2:
                    field_type, default = value[0], value[1]
                    
                    # Check if default is a factory
                    default_factory = None
                    if callable(default) and not isinstance(default, type):
                        default_factory = default
                        default = None
                        
                    composer.add_field(
                        name=name,
                        field_type=field_type,
                        default=default,
                        default_factory=default_factory
                    )
                else:
                    # Infer type from value
                    composer.add_field(
                        name=name,
                        field_type=type(value),
                        default=value
                    )
        
        # Build schema
        return composer.build()

    @classmethod
    def create_model(cls, components: List[Any], name: str = "ComposedModel") -> Type[StateSchema]:
        """
        Create a model directly from components.
        
        Alias for from_components that emphasizes the model creation aspect.
        
        Args:
            components: List of components to extract fields from.
            name: Name for the resulting model.
            
        Returns:
            StateSchema class built from components.
        """
        # Create composer
        composer = cls(name=name)
        
        # Process each component
        for component in components:
            if component is None:
                continue
                
            # Process based on type
            if hasattr(component, "engine_type"):
                # Looks like an Engine
                composer.add_fields_from_engine(component)
            elif isinstance(component, BaseModel):
                # BaseModel instance
                composer.add_fields_from_model(component.__class__)
            elif isinstance(component, type) and issubclass(component, BaseModel):
                # BaseModel class
                composer.add_fields_from_model(component)
            elif isinstance(component, dict):
                # Dictionary
                composer.add_fields_from_dict(component)
                
        # Ensure messages field
        composer.ensure_messages_field(add_if_missing=True)
        
        # Build the schema
        return composer.build()

    @classmethod
    def compose_schema(cls, components: List[Any], name: str = "ComposedSchema",
                      include_messages: bool = True) -> Type[StateSchema]:
        """
        Compose a schema from components with options.
        
        Args:
            components: List of components to extract fields from.
            name: Name for the resulting schema.
            include_messages: Whether to ensure a messages field exists.
            
        Returns:
            StateSchema class with fields from all components.
        """
        # Create composer
        composer = cls(name=name)
        
        # Process each component
        for component in components:
            if component is None:
                continue
                
            # Process based on type
            if hasattr(component, "engine_type"):
                # Looks like an Engine
                composer.add_fields_from_engine(component)
            elif isinstance(component, BaseModel):
                # BaseModel instance
                composer.add_fields_from_model(component.__class__)
            elif isinstance(component, type) and issubclass(component, BaseModel):
                # BaseModel class
                composer.add_fields_from_model(component)
            elif isinstance(component, dict):
                # Dictionary
                composer.add_fields_from_dict(component)
                
        # Ensure messages field if requested
        if include_messages:
            composer.ensure_messages_field(add_if_missing=True)
        
        # Build the schema
        return composer.build()

    @classmethod
    def compose(cls, *args, **kwargs) -> Type[StateSchema]:
        """
        Alias for compose_schema.
        
        Args:
            *args: Positional arguments for compose_schema.
            **kwargs: Keyword arguments for compose_schema.
            
        Returns:
            StateSchema class built from components.
        """
        return cls.compose_schema(*args, **kwargs)

    @classmethod
    def compose_as_state_schema(cls, components: List[Any], name: str = "ComposedStateSchema",
                               include_messages: bool = True) -> Type[StateSchema]:
        """
        Compose a StateSchema with standard functionality.
        
        Creates a schema with standard state schema functionality, including
        messages field if requested.
        
        Args:
            components: List of components to extract fields from.
            name: Name for the resulting schema.
            include_messages: Whether to ensure a messages field exists.
            
        Returns:
            StateSchema class with fields from all components.
        """
        return cls.compose_schema(components, name, include_messages)

    @classmethod
    def compose_input_schema(cls, components: List[Any], name: str = "InputSchema") -> Type[BaseModel]:
        """
        Create an input schema from components.
        
        Creates a schema suitable for input validation, focusing on input fields
        from engines.
        
        Args:
            components: List of components to extract fields from.
            name: Name for the schema.
            
        Returns:
            BaseModel class suitable for input validation.
        """
        # Create composer
        composer = cls(name=name)
        
        # Process input schemas from components
        for component in components:
            if component is None:
                continue
                
            # Try to get input schema from engines
            if hasattr(component, "engine_type") and hasattr(component, "derive_input_schema"):
                try:
                    input_schema = component.derive_input_schema()
                    composer.add_fields_from_model(input_schema)
                except Exception as e:
                    logger.warning(f"Error deriving input schema: {e}")
            elif isinstance(component, BaseModel) or (isinstance(component, type) and issubclass(component, BaseModel)):
                # Just add all fields from the model
                model = component if isinstance(component, type) else component.__class__
                composer.add_fields_from_model(model)
                
        # Build the schema without shared fields (not a StateSchema)
        return composer.build()

    @classmethod
    def compose_output_schema(cls, components: List[Any], name: str = "OutputSchema") -> Type[BaseModel]:
        """
        Create an output schema from components.
        
        Creates a schema suitable for output validation, focusing on output fields
        from engines.
        
        Args:
            components: List of components to extract fields from.
            name: Name for the schema.
            
        Returns:
            BaseModel class suitable for output validation.
        """
        # Create composer
        composer = cls(name=name)
        
        # Process output schemas from components
        for component in components:
            if component is None:
                continue
                
            # Try to get output schema from engines
            if hasattr(component, "engine_type") and hasattr(component, "derive_output_schema"):
                try:
                    output_schema = component.derive_output_schema()
                    composer.add_fields_from_model(output_schema)
                except Exception as e:
                    logger.warning(f"Error deriving output schema: {e}")
            elif isinstance(component, BaseModel) or (isinstance(component, type) and issubclass(component, BaseModel)):
                # Just add all fields from the model
                model = component if isinstance(component, type) else component.__class__
                composer.add_fields_from_model(model)
                
        # Build the schema without shared fields (not a StateSchema)
        return composer.build()

    @classmethod
    def create_schema_for_components(cls, components: List[Any], name: str = "ComponentSchema") -> 'SchemaComposer':
        """
        Create a SchemaComposer for components.
        
        Instead of building the schema directly, this method returns the composer
        for further manipulation.
        
        Args:
            components: List of components to extract fields from.
            name: Name for the schema composer.
            
        Returns:
            SchemaComposer with fields from all components.
        """
        composer = cls(name=name)
        
        # Process each component
        for component in components:
            if component is None:
                continue
                
            # Process based on type
            if hasattr(component, "engine_type"):
                # Looks like an Engine
                composer.extract_fields_from_engine(component)
            elif isinstance(component, BaseModel):
                # BaseModel instance
                composer.add_fields_from_model(component.__class__)
            elif isinstance(component, type) and issubclass(component, BaseModel):
                # BaseModel class
                composer.add_fields_from_model(component)
            elif isinstance(component, dict):
                # Dictionary
                composer.add_fields_from_dict(component)
                
        # Return the composer without building
        return composer