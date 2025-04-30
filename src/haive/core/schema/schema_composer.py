"""
SchemaComposer for the Haive framework.
"""
from __future__ import annotations
import inspect
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Type,\
    get_args, get_origin, Callable, TYPE_CHECKING
    
    
from pydantic import BaseModel, Field, create_model

from haive.core.schema.state_schema import StateSchema
from haive.core.schema.field_definition import FieldDefinition
from haive.core.schema.field_extractor import FieldExtractor

logger = logging.getLogger(__name__)
if TYPE_CHECKING:
    from haive.core.schema.schema_manager import StateSchemaManager

class SchemaComposer:
    """
    Utility for extracting field information from components and composing schemas.
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
        """Add a field definition to the schema."""
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

    def add_fields_from_model(self, model: Type[BaseModel]) -> 'SchemaComposer':
        """Extract fields from a Pydantic model and add them to the schema."""
        # Skip non-models
        if not isinstance(model, type) or not issubclass(model, BaseModel):
            logger.warning(f"Expected Pydantic model, got {type(model)}")
            return self
            
        # Get fields, descriptions, and other metadata from the model
        extracted_fields, descriptions, shared_fields, reducer_names, reducer_functions, engine_io_mappings, input_fields, output_fields = \
            FieldExtractor.extract_from_model(model)
        
        model_name = model.__name__
        
        # Add fields that aren't already present
        for field_name, (field_type, field_info) in extracted_fields.items():
            if field_name not in self.fields:
                # Get default and default_factory
                default = field_info.default
                default_factory = field_info.default_factory
                
                # Get description
                description = descriptions.get(field_name)
                
                # Determine if shared
                is_shared = field_name in shared_fields
                
                # Get reducer if available
                reducer = reducer_functions.get(field_name)
                
                # Add the field
                self.add_field(
                    name=field_name,
                    field_type=field_type,
                    default=default,
                    default_factory=default_factory,
                    description=description,
                    shared=is_shared,
                    reducer=reducer,
                    source=model_name
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

    def add_fields_from_dict(self, fields: Dict[str, Any]) -> 'SchemaComposer':
        """Add fields from a dictionary representation."""
        # Extract field information from the dictionary
        extracted_fields, descriptions, shared_fields, reducer_names, reducer_functions, engine_io_mappings, input_fields, output_fields = \
            FieldExtractor.extract_from_dict(fields)
        
        # Add fields that aren't already present
        for field_name, (field_type, field_info) in extracted_fields.items():
            if field_name not in self.fields:
                # Get default and default_factory
                default = field_info.default
                default_factory = field_info.default_factory
                
                # Get description
                description = descriptions.get(field_name)
                
                # Determine if shared
                is_shared = field_name in shared_fields
                
                # Get reducer if available
                reducer = reducer_functions.get(field_name)
                
                # Add the field
                self.add_field(
                    name=field_name,
                    field_type=field_type,
                    default=default,
                    default_factory=default_factory,
                    description=description,
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

    def add_fields_from_engine(self, engine: Any) -> 'SchemaComposer':
        """Extract fields from an Engine object."""
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
                
                # Get description
                description = descriptions.get(field_name)
                
                # Add the field with metadata
                self.add_field(
                    name=field_name,
                    field_type=field_type,
                    default=default if default is not ... else None,
                    default_factory=default_factory,
                    description=description,
                    source=source
                )
        
        # Special handling for AugLLMConfig with structured_output_model
        if hasattr(engine, "structured_output_model") and engine.structured_output_model is not None:
            logger.debug(f"Found structured_output_model in {source}: {engine.structured_output_model.__name__}")
            
            # Add model as a field with its name in lowercase
            model = engine.structured_output_model
            model_name = model.__name__.lower()
            
            from typing import Optional
            if model_name not in self.fields:
                self.add_field(
                    name=model_name,
                    field_type=Optional[model],
                    default=None,
                    description=f"Output in {model.__name__} format",
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
        """Update engine I/O mapping for a specific engine."""
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
        """Ensure the messages field exists with proper configuration."""
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
                from typing import Any
                
                # Create simple concat lists reducer
                def concat_lists(a, b):
                    return (a or []) + (b or [])
                
                self.add_field(
                    name="messages",
                    field_type=List[Any],
                    default_factory=list,
                    description="Messages for agent conversation",
                    reducer=concat_lists,
                    source="default"
                )
                    
        return self

    def to_manager(self) -> 'StateSchemaManager':
        """Convert to a StateSchemaManager for further manipulation."""
        from haive.core.schema.schema_manager import StateSchemaManager
        return StateSchemaManager(self)

    def build(self) -> Type[StateSchema]:
        """Build a StateSchema directly from the composer."""
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
        """Create a schema from components."""
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
                
        # Ensure messages field
        composer.ensure_messages_field()
        
        # Build the schema
        return composer.build()

    @classmethod
    def create_message_state(cls, additional_fields: Optional[Dict[str, Any]] = None, 
                            name: str = "MessageState") -> Type[StateSchema]:
        """Create a schema with messages field and additional fields."""
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