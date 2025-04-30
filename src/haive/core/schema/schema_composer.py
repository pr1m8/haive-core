"""
SchemaComposer for the Haive framework.

Provides a clean implementation focused on properly handling structured output models
without duplicating their fields in the main schema.
"""
from __future__ import annotations
import inspect
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Type, Union, Callable

from pydantic import BaseModel, Field, create_model

from haive.core.schema.state_schema import StateSchema
from haive.core.schema.field_definition import FieldDefinition

logger = logging.getLogger(__name__)

class SchemaComposer:
    """
    Utility for extracting field information from components and composing schemas.
    
    The SchemaComposer provides a high-level API for:
    - Dynamically extracting fields from various components (engines, models, dictionaries)
    - Composing schemas from field definitions
    - Tracking field relationships and metadata
    - Building optimized schema classes with proper configuration
    """

    def __init__(self, name: str = "ComposedSchema"):
        """Initialize a new SchemaComposer."""
        self.name = name
        self.fields = {}
        self.shared_fields = set()
        self.field_sources = defaultdict(set)
        
        # Track input/output mappings for engines
        self.input_fields = defaultdict(set)
        self.output_fields = defaultdict(set)
        self.engine_io_mappings = {}
        
        # Track structured models separately
        self.structured_models = {}
        self.structured_model_fields = defaultdict(set)

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
            name: Field name
            field_type: Type of the field
            default: Default value for the field
            default_factory: Optional factory function for default value
            description: Optional field description
            shared: Whether field is shared with parent graph
            reducer: Optional reducer function for this field
            source: Optional source identifier (component name, etc.)
            
        Returns:
            Self for chaining
        """
        # Skip special fields
        if name == "__runnable_config__" or name == "runnable_config":
            logger.warning(f"Skipping special field {name}")
            return self
            
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
            
        return self

    def add_fields_from_engine(self, engine: Any) -> 'SchemaComposer':
        """
        Extract fields from an Engine object.
        
        Args:
            engine: Engine object to extract fields from
            
        Returns:
            Self for chaining
        """
        source = getattr(engine, "name", str(engine))
        logger.debug(f"Extracting fields from engine: {source}")
        
        # Get engine name for tracking
        engine_name = getattr(engine, "name", str(engine))
        
        # Init engine IO mapping if it doesn't exist
        if engine_name not in self.engine_io_mappings:
            self.engine_io_mappings[engine_name] = {
                "inputs": [],
                "outputs": []
            }
        
        # Extract input fields if method available
        if hasattr(engine, "get_input_fields") and callable(engine.get_input_fields):
            try:
                input_fields = engine.get_input_fields()
                
                for field_name, (field_type, field_info) in input_fields.items():
                    # Skip if already has this field
                    if field_name in self.fields:
                        continue
                        
                    # Get default and default_factory
                    if hasattr(field_info, 'default') and field_info.default is not ...:
                        default = field_info.default
                    else:
                        default = None
                        
                    default_factory = getattr(field_info, "default_factory", None)
                    description = getattr(field_info, "description", None)
                    
                    # Add the field
                    self.add_field(
                        name=field_name,
                        field_type=field_type,
                        default=default,
                        default_factory=default_factory,
                        description=description,
                        source=source
                    )
                    
                    # Track as input field
                    self.input_fields[engine_name].add(field_name)
                
                # Update engine IO mapping
                self.engine_io_mappings[engine_name]["inputs"] = list(self.input_fields[engine_name])
                
                logger.debug(f"Added {len(input_fields)} input fields from engine {engine_name}")
            except Exception as e:
                logger.warning(f"Error getting input_fields from {engine_name}: {e}")
        
        # Extract output fields if method available
        if hasattr(engine, "get_output_fields") and callable(engine.get_output_fields):
            try:
                output_fields = engine.get_output_fields()
                
                for field_name, (field_type, field_info) in output_fields.items():
                    # Don't add fields that will be handled as structured models
                    # Check if this is a structured model field name first
                    if hasattr(engine, "structured_output_model") and engine.structured_output_model:
                        model_name = engine.structured_output_model.__name__.lower()
                        
                        # If field is in the structured model, don't add it directly
                        if hasattr(engine.structured_output_model, "model_fields") and \
                           field_name in engine.structured_output_model.model_fields:
                            # Instead, track it as part of the structured model
                            self.structured_model_fields[model_name].add(field_name)
                            continue
                    
                    # Skip if already has this field
                    if field_name in self.fields:
                        # Just mark as output field if already exists
                        self.output_fields[engine_name].add(field_name)
                        continue
                        
                    # Get default and default_factory
                    if hasattr(field_info, 'default') and field_info.default is not ...:
                        default = field_info.default
                    else:
                        default = None
                        
                    default_factory = getattr(field_info, "default_factory", None)
                    description = getattr(field_info, "description", None)
                    
                    # Add the field
                    self.add_field(
                        name=field_name,
                        field_type=field_type,
                        default=default,
                        default_factory=default_factory,
                        description=description,
                        source=source
                    )
                    
                    # Track as output field
                    self.output_fields[engine_name].add(field_name)
                
                # Update engine IO mapping
                self.engine_io_mappings[engine_name]["outputs"] = list(self.output_fields[engine_name])
                
                logger.debug(f"Added {len(output_fields)} output fields from engine {engine_name}")
            except Exception as e:
                logger.warning(f"Error getting output_fields from {engine_name}: {e}")
   
        
        # Special handling for structured_output_model - add as a single field
        if hasattr(engine, "structured_output_model") and engine.structured_output_model is not None:
            model = engine.structured_output_model
            model_name = model.__name__.lower()
            
            logger.debug(f"Found structured_output_model in {source}: {model.__name__}")
            
            # Store the structured model for reference
            self.structured_models[model_name] = model
            
            # Add field just for the model itself
            if model_name not in self.fields:
                from typing import Optional
                self.add_field(
                    name=model_name,
                    field_type=Optional[model],  # Use Optional[model] for proper typing
                    default=None,
                    description=f"Output in {model.__name__} format",
                    source=f"{source}.structured_output_model"
                )
                
                # Mark as output field
                self.output_fields[engine_name].add(model_name)
                
                # Update engine mapping
                if model_name not in self.engine_io_mappings[engine_name]["outputs"]:
                    self.engine_io_mappings[engine_name]["outputs"].append(model_name)
            
            # Just log the fields for debugging - don't add them
            if hasattr(model, "model_fields"):
                for field_name, field_info in model.model_fields.items():
                    # Track field as part of this model
                    self.structured_model_fields[model_name].add(field_name)
                    logger.debug(f"  - Model field: {field_name}: {field_info.annotation}")
        
        return self

    def configure_messages_field(self, with_reducer: bool = True, force_add: bool = False) -> 'SchemaComposer':
        """
        Configure a messages field with appropriate settings if it exists or if requested.
        
        Args:
            with_reducer: Whether to add a reducer for the messages field
            force_add: Whether to add the messages field if it doesn't exist
            
        Returns:
            Self for chaining
        """
        # Only proceed if the field exists or we're forcing its addition
        if "messages" in self.fields or force_add:
            from typing import List
            
            # Try to use langgraph's add_messages if requested
            if with_reducer:
                try:
                    from langgraph.graph import add_messages
                    from langchain_core.messages import BaseMessage
                    
                    # If force_add is True and the field doesn't exist, add it
                    if force_add and "messages" not in self.fields:
                        self.add_field(
                            name="messages",
                            field_type=List[BaseMessage],
                            default_factory=list,
                            description="Messages for agent conversation",
                            reducer=add_messages
                        )
                    # Otherwise, just set the reducer if the field exists
                    elif "messages" in self.fields:
                        self.fields["messages"].reducer = add_messages
                        
                except ImportError:
                    # Fallback if add_messages is not available
                    from typing import Any
                    
                    # Create simple concat lists reducer
                    def concat_lists(a, b):
                        return (a or []) + (b or [])
                    
                    # If force_add is True and the field doesn't exist, add it
                    if force_add and "messages" not in self.fields:
                        self.add_field(
                            name="messages",
                            field_type=List[Any],
                            default_factory=list,
                            description="Messages for agent conversation",
                            reducer=concat_lists
                        )
                    # Otherwise, just set the reducer if the field exists
                    elif "messages" in self.fields:
                        self.fields["messages"].reducer = concat_lists
                
        return self

    def to_manager(self) -> 'StateSchemaManager':
        """
        Convert to a StateSchemaManager for further manipulation.
        
        Returns:
            StateSchemaManager instance
        """
        from haive.core.schema.schema_manager import StateSchemaManager
        return StateSchemaManager(self)

    def build(self) -> Type[StateSchema]:
        """
        Build a StateSchema directly from the composer with proper handling of structured models.
        
        Returns:
            Subclass of StateSchema with configured fields and metadata
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
        
        # Add structured model fields metadata
        if self.structured_model_fields:
            schema.__structured_model_fields__ = {
                k: list(v) for k, v in self.structured_model_fields.items()
            }
            
        # Add structured models themselves
        if self.structured_models:
            schema.__structured_models__ = self.structured_models
        
        return schema
    
    @classmethod
    def from_components(cls, components: List[Any], name: str = "ComposedSchema") -> Type[StateSchema]:
        """
        Create a schema from components.
        
        Args:
            components: List of components to extract fields from
            name: Name for the schema
            
        Returns:
            StateSchema subclass
        """
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
            else:
                logger.debug(f"Skipping unsupported component: {type(component)}")
        
        # Build the schema
        return composer.build()

    @classmethod
    def create_message_state(cls, additional_fields: Optional[Dict[str, Any]] = None, 
                            name: str = "MessageState") -> Type[StateSchema]:
        """
        Create a schema with messages field and additional fields.
        
        Args:
            additional_fields: Optional dictionary of additional fields to add
            name: Name for the schema
            
        Returns:
            StateSchema subclass with messages field
        """
        # Create composer
        composer = cls(name=name)
        
        # Add messages field with reducer
        from typing import List, Sequence
        try:
            from langgraph.graph import add_messages
            from langchain_core.messages import BaseMessage
            
            # Add messages field with reducer
            composer.add_field(
                name="messages",
                field_type=Sequence[BaseMessage],
                default_factory=list,
                description="Messages for conversation",
                reducer=add_messages
            )
        except ImportError:
            # Fallback if add_messages is not available
            from typing import Any
            
            # Create simple concat lists reducer
            def concat_lists(a, b):
                return (a or []) + (b or [])
            
            composer.add_field(
                name="messages",
                field_type=List[Any],
                default_factory=list,
                description="Messages for conversation",
                reducer=concat_lists
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