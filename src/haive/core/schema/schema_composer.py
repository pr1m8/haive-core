"""
SchemaComposer for the Haive framework.

The SchemaComposer provides a streamlined API for composing state schemas from
various components, with proper handling of field metadata, shared fields,
reducers, and structured output models.
"""
from __future__ import annotations
import inspect
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, \
    Set, Type, Union, Callable, Annotated, get_origin, get_args, TypeVar, TYPE_CHECKING

from pydantic import BaseModel, Field, create_model

from haive.core.schema.state_schema import StateSchema
from haive.core.schema.field_definition import FieldDefinition
from haive.core.schema.field_extractor import FieldExtractor
from haive.core.schema.field_utils import (
    create_field, create_annotated_field, extract_type_metadata,
    infer_field_type, get_common_reducers, resolve_reducer
)

logger = logging.getLogger(__name__)

T = TypeVar('T')

if TYPE_CHECKING:
    from haive.core.schema.schema_manager import StateSchemaManager

class SchemaComposer:
    """
    Utility for composing state schemas from components.
    
    The SchemaComposer extracts field information from components and composes
    them into a cohesive state schema with proper metadata handling. It provides
    a clean, high-level API for schema composition.
    """

    def __init__(self, name: str = "ComposedSchema"):
        """
        Initialize a new SchemaComposer.
        
        Args:
            name: Name for the resulting schema class
        """
        self.name = name
        self.fields: Dict[str, FieldDefinition] = {}
        self.shared_fields: Set[str] = set()
        self.field_sources: Dict[str, Set[str]] = defaultdict(set)
        
        # Track input/output mappings for engines
        self.input_fields: Dict[str, Set[str]] = defaultdict(set)
        self.output_fields: Dict[str, Set[str]] = defaultdict(set)
        self.engine_io_mappings: Dict[str, Dict[str, List[str]]] = {}
        
        # Track structured models separately
        self.structured_models: Dict[str, Type[BaseModel]] = {}
        self.structured_model_fields: Dict[str, Set[str]] = defaultdict(set)

    def add_field(
        self,
        name: str,
        field_type: Type[T],
        default: Any = None,
        default_factory: Optional[Callable[[], T]] = None,
        description: Optional[str] = None,
        shared: bool = False,
        reducer: Optional[Callable] = None,
        source: Optional[str] = None,
        input_for: Optional[List[str]] = None,
        output_from: Optional[List[str]] = None,
        structured_model: Optional[str] = None,
        **kwargs
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
            input_for: List of engines this field serves as input for
            output_from: List of engines this field is output from
            structured_model: Name of structured model this field belongs to
            **kwargs: Additional field parameters
            
        Returns:
            Self for chaining
        """
        # Skip special fields
        if name == "__runnable_config__" or name == "runnable_config":
            logger.warning(f"Skipping special field {name}")
            return self
            
        # Create field info
        field_info = None
        if "field_info" in kwargs:
            field_info = kwargs.pop("field_info")
            
        # Create field definition
        field_def = FieldDefinition(
            name=name,
            field_type=field_type,
            field_info=field_info,
            default=default,
            default_factory=default_factory,
            description=description,
            shared=shared,
            reducer=reducer,
            source=source,
            input_for=input_for,
            output_from=output_from,
            structured_model=structured_model,
            **kwargs
        )
        
        # Store the field
        self.fields[name] = field_def
        
        # Track additional metadata
        if shared:
            self.shared_fields.add(name)
            
        if source:
            self.field_sources[name].add(source)
            
        # Track engine I/O
        if input_for:
            for engine_name in input_for:
                self.input_fields[engine_name].add(name)
                
                # Update engine I/O mapping
                if engine_name not in self.engine_io_mappings:
                    self.engine_io_mappings[engine_name] = {"inputs": [], "outputs": []}
                if name not in self.engine_io_mappings[engine_name]["inputs"]:
                    self.engine_io_mappings[engine_name]["inputs"].append(name)
                    
        if output_from:
            for engine_name in output_from:
                self.output_fields[engine_name].add(name)
                
                # Update engine I/O mapping
                if engine_name not in self.engine_io_mappings:
                    self.engine_io_mappings[engine_name] = {"inputs": [], "outputs": []}
                if name not in self.engine_io_mappings[engine_name]["outputs"]:
                    self.engine_io_mappings[engine_name]["outputs"].append(name)
                    
        # Track structured model
        if structured_model:
            self.structured_model_fields[structured_model].add(name)
            
        return self

    def add_field_definition(self, field_def: FieldDefinition) -> 'SchemaComposer':
        """
        Add a pre-constructed FieldDefinition to the schema.
        
        Args:
            field_def: FieldDefinition to add
            
        Returns:
            Self for chaining
        """
        name = field_def.name
        
        # Skip special fields
        if name == "__runnable_config__" or name == "runnable_config":
            logger.warning(f"Skipping special field {name}")
            return self
            
        # Store the field
        self.fields[name] = field_def
        
        # Track additional metadata
        if field_def.shared:
            self.shared_fields.add(name)
            
        if field_def.source:
            self.field_sources[name].add(field_def.source)
            
        # Track engine I/O
        for engine_name in field_def.input_for:
            self.input_fields[engine_name].add(name)
            
            # Update engine I/O mapping
            if engine_name not in self.engine_io_mappings:
                self.engine_io_mappings[engine_name] = {"inputs": [], "outputs": []}
            if name not in self.engine_io_mappings[engine_name]["inputs"]:
                self.engine_io_mappings[engine_name]["inputs"].append(name)
                
        for engine_name in field_def.output_from:
            self.output_fields[engine_name].add(name)
            
            # Update engine I/O mapping
            if engine_name not in self.engine_io_mappings:
                self.engine_io_mappings[engine_name] = {"inputs": [], "outputs": []}
            if name not in self.engine_io_mappings[engine_name]["outputs"]:
                self.engine_io_mappings[engine_name]["outputs"].append(name)
                
        # Track structured model
        if field_def.structured_model:
            self.structured_model_fields[field_def.structured_model].add(name)
            
        return self

    def add_fields_from_dict(self, fields_dict: Dict[str, Any]) -> 'SchemaComposer':
        """
        Add fields from a dictionary definition.
        
        Args:
            fields_dict: Dictionary mapping field names to type/value information
            
        Returns:
            Self for chaining
        """
        # Check for special metadata keys
        for key in ["shared_fields", "reducer_names", "serializable_reducers", "reducer_functions",
                   "field_descriptions", "engine_io_mappings", "input_fields", "output_fields"]:
            if key in fields_dict:
                continue
                
        # Process field definitions
        for field_name, field_info in fields_dict.items():
            # Skip special fields
            if field_name == "__runnable_config__" or field_name == "runnable_config":
                logger.warning(f"Skipping special field {field_name}")
                continue
            
            if isinstance(field_info, tuple) and len(field_info) >= 2:
                # Handle (type, default) format
                field_type, default = field_info[0:2]
                
                # Check for extra metadata
                field_metadata = {}
                if len(field_info) >= 3 and isinstance(field_info[2], dict):
                    field_metadata = field_info[2]
                    
                # Extract metadata
                description = field_metadata.pop("description", None)
                shared = field_metadata.pop("shared", False)
                reducer = None
                if "reducer" in field_metadata:
                    reducer_value = field_metadata.pop("reducer")
                    if callable(reducer_value):
                        reducer = reducer_value
                    elif isinstance(reducer_value, str):
                        reducer = resolve_reducer(reducer_value)
                
                # Check if default is a factory function
                default_factory = None
                if callable(default) and not isinstance(default, type):
                    default_factory = default
                    default = None
                    
                # Add the field
                self.add_field(
                    name=field_name,
                    field_type=field_type,
                    default=default,
                    default_factory=default_factory,
                    description=description,
                    shared=shared,
                    reducer=reducer,
                    source="dictionary",
                    **field_metadata
                )
            else:
                # Infer type from value
                field_type = infer_field_type(field_info)
                
                # Add the field
                self.add_field(
                    name=field_name,
                    field_type=field_type,
                    default=field_info,
                    source="dictionary"
                )
                
        return self

    def add_fields_from_model(
        self, 
        model: Type[BaseModel],
        include_private: bool = False,
        extract_annotations: bool = True
    ) -> 'SchemaComposer':
        """
        Extract fields from a Pydantic model.
        
        Args:
            model: Pydantic model to extract fields from
            include_private: Whether to include private fields (_name)
            extract_annotations: Whether to extract metadata from Annotated types
            
        Returns:
            Self for chaining
        """
        source = model.__name__
        
        # Extract fields
        if hasattr(model, "model_fields"):
            # Pydantic v2
            for field_name, field_info in model.model_fields.items():
                # Skip special fields and private fields (if not included)
                if field_name.startswith("__") or (field_name.startswith("_") and not include_private):
                    continue
                    
                # Skip runnable_config
                if field_name == "runnable_config":
                    continue
                    
                # Get field type and extract any annotations
                field_type = field_info.annotation
                
                # Create field definition
                field_def = FieldDefinition.extract_from_model_field(
                    name=field_name,
                    field_type=field_type,
                    field_info=field_info,
                    include_annotations=extract_annotations
                )
                
                # Set source
                field_def.source = source
                
                # Add to composer
                self.add_field_definition(field_def)
        
        # Extract shared fields
        if hasattr(model, "__shared_fields__"):
            for field_name in model.__shared_fields__:
                if field_name in self.fields:
                    self.fields[field_name].shared = True
                    self.shared_fields.add(field_name)
                    
        # Extract reducers
        if hasattr(model, "__reducer_fields__"):
            for field_name, reducer in model.__reducer_fields__.items():
                if field_name in self.fields:
                    self.fields[field_name].reducer = reducer
                    
        # Extract engine I/O mappings
        if hasattr(model, "__engine_io_mappings__"):
            for engine_name, mapping in model.__engine_io_mappings__.items():
                # Create mapping if it doesn't exist
                if engine_name not in self.engine_io_mappings:
                    self.engine_io_mappings[engine_name] = {"inputs": [], "outputs": []}
                    
                # Add inputs
                for field_name in mapping.get("inputs", []):
                    if field_name in self.fields and field_name not in self.engine_io_mappings[engine_name]["inputs"]:
                        self.engine_io_mappings[engine_name]["inputs"].append(field_name)
                        
                        # Update field definition
                        if engine_name not in self.fields[field_name].input_for:
                            self.fields[field_name].input_for.append(engine_name)
                            
                        # Update input_fields tracking
                        self.input_fields[engine_name].add(field_name)
                        
                # Add outputs
                for field_name in mapping.get("outputs", []):
                    if field_name in self.fields and field_name not in self.engine_io_mappings[engine_name]["outputs"]:
                        self.engine_io_mappings[engine_name]["outputs"].append(field_name)
                        
                        # Update field definition
                        if engine_name not in self.fields[field_name].output_from:
                            self.fields[field_name].output_from.append(engine_name)
                            
                        # Update output_fields tracking
                        self.output_fields[engine_name].add(field_name)
                        
        return self

    def add_fields_from_components(
        self,
        components: List[Any],
        include_messages_field: bool = True
    ) -> 'SchemaComposer':
        """
        Extract field definitions from a list of components.
        
        Args:
            components: List of components to extract from
            include_messages_field: Whether to ensure a messages field exists
            
        Returns:
            Self for chaining
        """
        # Use FieldExtractor to get field definitions
        field_defs, engine_io_mappings, structured_model_fields, structured_models = (
            FieldExtractor.extract_from_components(
                components, 
                include_messages_field=include_messages_field
            )
        )
        
        # Add field definitions
        for field_name, field_def in field_defs.items():
            self.add_field_definition(field_def)
            
        # Update structured models
        self.structured_models.update(structured_models)
        
        # Update structured model fields
        for model_name, fields in structured_model_fields.items():
            self.structured_model_fields[model_name].update(fields)
            
        return self

    def add_fields_from_engine(self, engine: Any) -> 'SchemaComposer':
        """
        Extract fields from an Engine object.
        
        Args:
            engine: Engine object to extract fields from
            
        Returns:
            Self for chaining
        """
        # Use FieldExtractor to get field definitions
        fields, descriptions, io_mappings, in_fields, out_fields = (
            FieldExtractor.extract_from_engine(engine)
        )
        
        # Extract engine name
        engine_name = getattr(engine, "name", str(engine))
        
        # Convert to FieldDefinition objects and add to composer
        for field_name, (field_type, field_info) in fields.items():
            # Create field definition
            field_def = FieldDefinition(
                name=field_name,
                field_type=field_type,
                field_info=field_info,
                source=engine_name
            )
            
            # Add input_for and output_from
            for in_engine, field_set in in_fields.items():
                if field_name in field_set:
                    field_def.input_for.append(in_engine)
                    
            for out_engine, field_set in out_fields.items():
                if field_name in field_set:
                    field_def.output_from.append(out_engine)
                    
            # Add to composer
            self.add_field_definition(field_def)
            
        # Check for structured output model
        if hasattr(engine, "structured_output_model") and engine.structured_output_model:
            model = engine.structured_output_model
            model_name = model.__name__.lower()
            
            # Store the model
            self.structured_models[model_name] = model
            
            # Extract model fields
            if hasattr(model, "model_fields"):
                for field_name in model.model_fields:
                    self.structured_model_fields[model_name].add(field_name)
                    
        return self

    def configure_messages_field(
        self, 
        with_reducer: bool = True, 
        force_add: bool = False
    ) -> 'SchemaComposer':
        """
        Configure a messages field with appropriate settings.
        
        Args:
            with_reducer: Whether to add a reducer for the messages field
            force_add: Whether to add the messages field if it doesn't exist
            
        Returns:
            Self for chaining
        """
        # Only proceed if the field exists or we're forcing its addition
        if "messages" in self.fields or force_add:
            from typing import List
            from langchain_core.messages import BaseMessage
            
            # Try to use langgraph's add_messages if requested
            if with_reducer:
                try:
                    from langgraph.graph import add_messages
                    reducer = add_messages
                except ImportError:
                    # Fallback to a simple list concatenation
                    def concat_lists(a, b):
                        return (a or []) + (b or [])
                    reducer = concat_lists
                    
                # If force_add is True and the field doesn't exist, add it
                if force_add and "messages" not in self.fields:
                    self.add_field(
                        name="messages",
                        field_type=List[BaseMessage],
                        default_factory=list,
                        description="Messages for conversation",
                        reducer=reducer
                    )
                # Otherwise, just set the reducer if the field exists
                elif "messages" in self.fields:
                    self.fields["messages"].reducer = reducer
                    
        return self

    def build(self, use_annotated: bool = True) -> Type[StateSchema]:
        """
        Build a StateSchema from the composed fields.
        
        Args:
            use_annotated: Whether to use Annotated types for metadata
            
        Returns:
            StateSchema subclass with the composed fields
        """
        # Create field definitions for the model
        field_defs = {}
        
        for name, field_def in self.fields.items():
            if use_annotated:
                field_type, field_info = field_def.to_annotated_field()
            else:
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
        
        # Add structured model fields metadata safely - use field names instead of class references
        if self.structured_model_fields:
            schema.__structured_model_fields__ = {
                k: list(v) for k, v in self.structured_model_fields.items()
            }
            
        # Add structured models safely - use string identifiers instead of class references
        if self.structured_models:
            schema.__structured_models__ = {
                k: f"{v.__module__}.{v.__name__}" 
                for k, v in self.structured_models.items()
            }
        
        return schema

    def to_manager(self) -> 'StateSchemaManager':
        """
        Convert to a StateSchemaManager for further manipulation.
        
        Returns:
            StateSchemaManager instance
        """
        from haive.core.schema.schema_manager import StateSchemaManager
        
        # Create a manager with our fields
        manager = StateSchemaManager(name=self.name)
        
        # Transfer field definitions
        for name, field_def in self.fields.items():
            field_type, field_info = field_def.to_field_info()
            manager.fields[name] = (field_type, field_info)
            
            # Transfer metadata
            if field_def.description:
                manager.field_descriptions[name] = field_def.description
                
            if field_def.shared:
                manager._shared_fields.add(name)
                
            if field_def.reducer:
                manager._reducer_names[name] = field_def.get_reducer_name()
                manager._reducer_functions[name] = field_def.reducer
                
        # Transfer engine I/O mappings
        manager._engine_io_mappings = self.engine_io_mappings.copy()
        
        # Transfer input/output fields
        for engine_name, fields in self.input_fields.items():
            manager._input_fields[engine_name] = fields.copy()
            
        for engine_name, fields in self.output_fields.items():
            manager._output_fields[engine_name] = fields.copy()
            
        return manager
        
    @staticmethod
    def merge(first: 'SchemaComposer', second: 'SchemaComposer', name: Optional[str] = None) -> 'SchemaComposer':
        """
        Merge two SchemaComposers into a new one.
        
        Args:
            first: First SchemaComposer
            second: Second SchemaComposer
            name: Optional name for the merged schema
            
        Returns:
            New merged SchemaComposer
        """
        # Create a new composer with a meaningful name
        merged_name = name or f"{first.name}_merged_with_{second.name}"
        merged = SchemaComposer(name=merged_name)
        
        # Add fields from first composer (preserving all metadata)
        for field_name, field_def in first.fields.items():
            merged.add_field_definition(field_def)
            
        # Add fields from second composer (skipping duplicates)
        for field_name, field_def in second.fields.items():
            if field_name not in merged.fields:
                merged.add_field_definition(field_def)
            else:
                # For duplicates, preserve existing field but merge metadata
                existing_field = merged.fields[field_name]
                
                # Update source if different
                if field_def.source and field_def.source != existing_field.source:
                    merged.field_sources[field_name].add(field_def.source)
                
                # Merge input_for and output_from lists
                for engine_name in field_def.input_for:
                    if engine_name not in existing_field.input_for:
                        existing_field.input_for.append(engine_name)
                        merged.input_fields[engine_name].add(field_name)
                        
                for engine_name in field_def.output_from:
                    if engine_name not in existing_field.output_from:
                        existing_field.output_from.append(engine_name)
                        merged.output_fields[engine_name].add(field_name)
                        
                # Use shared flag from either source
                if field_def.shared:
                    existing_field.shared = True
                    merged.shared_fields.add(field_name)
                    
                # Prefer second's reducer if present
                if field_def.reducer:
                    existing_field.reducer = field_def.reducer
        
        # Merge structured models
        merged.structured_models.update(first.structured_models)
        merged.structured_models.update(second.structured_models)
        
        # Merge structured model fields
        for model_name, fields in first.structured_model_fields.items():
            merged.structured_model_fields[model_name].update(fields)
            
        for model_name, fields in second.structured_model_fields.items():
            merged.structured_model_fields[model_name].update(fields)
            
        # Merge engine IO mappings
        for engine_name, mapping in first.engine_io_mappings.items():
            if engine_name not in merged.engine_io_mappings:
                merged.engine_io_mappings[engine_name] = {"inputs": [], "outputs": []}
                
            for field_name in mapping.get("inputs", []):
                if field_name not in merged.engine_io_mappings[engine_name]["inputs"]:
                    merged.engine_io_mappings[engine_name]["inputs"].append(field_name)
                    
            for field_name in mapping.get("outputs", []):
                if field_name not in merged.engine_io_mappings[engine_name]["outputs"]:
                    merged.engine_io_mappings[engine_name]["outputs"].append(field_name)
                    
        for engine_name, mapping in second.engine_io_mappings.items():
            if engine_name not in merged.engine_io_mappings:
                merged.engine_io_mappings[engine_name] = {"inputs": [], "outputs": []}
                
            for field_name in mapping.get("inputs", []):
                if field_name not in merged.engine_io_mappings[engine_name]["inputs"]:
                    merged.engine_io_mappings[engine_name]["inputs"].append(field_name)
                    
            for field_name in mapping.get("outputs", []):
                if field_name not in merged.engine_io_mappings[engine_name]["outputs"]:
                    merged.engine_io_mappings[engine_name]["outputs"].append(field_name)
                    
        return merged
        
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
        composer.add_fields_from_components(components)
        
        # Ensure messages field is properly configured
        composer.configure_messages_field(with_reducer=True, force_add=True)
        
        # Build and return the schema
        return composer.build()

    @classmethod
    def compose_input_schema(cls, components: List[Any], name: str = "InputSchema") -> Type[BaseModel]:
        """
        Create an input schema from components, focusing on input fields.
        
        Args:
            components: List of components to extract fields from
            name: Name for the schema
            
        Returns:
            BaseModel subclass optimized for input
        """
        # Create composer
        composer = cls(name=name)
        
        # Extract fields from components
        field_defs, engine_io_mappings, structured_model_fields, structured_models = (
            FieldExtractor.extract_from_components(
                components, 
                include_messages_field=True
            )
        )
        
        # Filter to include only input fields
        input_fields = {}
        for field_name, field_def in field_defs.items():
            # Include if it's an input field for any engine or a messages field
            if field_def.input_for or field_name == "messages":
                input_fields[field_name] = field_def
                
        # Add fields to composer
        for field_name, field_def in input_fields.items():
            composer.add_field_definition(field_def)
            
        # Create field definitions for model
        field_defs = {}
        for name, field_def in composer.fields.items():
            field_type, field_info = field_def.to_field_info()
            field_defs[name] = (field_type, field_info)
                
        # Create regular model (not StateSchema)
        return create_model(name, **field_defs)

    @classmethod
    def compose_output_schema(cls, components: List[Any], name: str = "OutputSchema") -> Type[BaseModel]:
        """
        Create an output schema from components, focusing on output fields.
        
        Args:
            components: List of components to extract fields from
            name: Name for the schema
            
        Returns:
            BaseModel subclass optimized for output
        """
        # Create composer
        composer = cls(name=name)
        
        # Extract fields from components
        field_defs, engine_io_mappings, structured_model_fields, structured_models = (
            FieldExtractor.extract_from_components(
                components, 
                include_messages_field=True
            )
        )
        
        # Filter to include only output fields and messages
        output_fields = {}
        for field_name, field_def in field_defs.items():
            # Include if it's an output field for any engine or a messages field
            if field_def.output_from or field_name == "messages":
                output_fields[field_name] = field_def
                
        # Add fields to composer
        for field_name, field_def in output_fields.items():
            composer.add_field_definition(field_def)
            
        # Add an output field if no field has "output" in the name
        has_output_field = any("output" in name.lower() for name in composer.fields)
        if not has_output_field:
            composer.add_field(
                name="output",
                field_type=str,
                default="",
                description="Agent output"
            )
            
        # Create field definitions for model
        field_defs = {}
        for name, field_def in composer.fields.items():
            field_type, field_info = field_def.to_field_info()
            field_defs[name] = (field_type, field_info)
                
        # Create regular model (not StateSchema)
        return create_model(name, **field_defs)

    @classmethod
    def create_message_state(
        cls, 
        additional_fields: Optional[Dict[str, Any]] = None, 
        name: str = "MessageState"
    ) -> Type[StateSchema]:
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
        from typing import List
        from langchain_core.messages import BaseMessage
        
        # Try to use langgraph's add_messages
        try:
            from langgraph.graph import add_messages
            reducer = add_messages
        except ImportError:
            # Fallback to a simple list concatenation
            def concat_lists(a, b):
                return (a or []) + (b or [])
            reducer = concat_lists
            
        # Add messages field
        composer.add_field(
            name="messages",
            field_type=List[BaseMessage],
            default_factory=list,
            description="Messages for conversation",
            reducer=reducer
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
                        
                    # Check for additional metadata
                    kwargs = {}
                    if len(value) >= 3 and isinstance(value[2], dict):
                        kwargs = value[2]
                        
                    composer.add_field(
                        name=name,
                        field_type=field_type,
                        default=default,
                        default_factory=default_factory,
                        **kwargs
                    )
                else:
                    # Infer type from value
                    field_type = infer_field_type(value)
                    
                    composer.add_field(
                        name=name,
                        field_type=field_type,
                        default=value
                    )
        
        # Build schema
        return composer.build()