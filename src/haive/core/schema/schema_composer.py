"""
SchemaComposer for automatically extracting field information from components.

This module provides the SchemaComposer class which extracts field information
from various components (engines, models, etc.) and builds a field collection
that can be passed to the StateSchemaManager for schema creation.

The primary entry point is the from_components() class method.
"""
from __future__ import annotations
import inspect
import logging
import operator
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Set, \
    Type, get_args, get_origin, Annotated, get_type_hints, Union, TYPE_CHECKING

from pydantic import BaseModel, Field

from haive.core.schema.state_schema import StateSchema
from haive.core.schema.utils import SchemaUtils
from langgraph.types import Command, Send

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from haive.core.schema.schema_manager import StateSchemaManager

class SchemaComposer:
    """
    A utility for extracting field information from components.
    
    SchemaComposer extracts fields from engines, models, and dictionaries and
    prepares field information that can be passed to StateSchemaManager for
    creating a schema.
    
    The primary entry point is the from_components() class method, which extracts
    fields from components and returns a configured StateSchemaManager instance.
    
    Example:
        manager = SchemaComposer.from_components([llm_engine, retriever_engine])
        schema = manager.get_model()
    """

    def __init__(self, name: str = "ComposedSchema"):
        """
        Initialize a new SchemaComposer.
        
        Args:
            name: Name for the field collection
        """
        self.name = name
        self.fields: Dict[str, tuple[Any, Any]] = {}
        self.shared_fields: Set[str] = set()
        self.reducer_names: Dict[str, str] = {}  # Field name -> reducer name (serializable)
        self.reducer_functions: Dict[str, Callable] = {}  # Field name -> reducer function
        self.field_descriptions: Dict[str, str] = {}
        self.field_sources: Dict[str, Set[str]] = defaultdict(set)  # Field name -> sources
        
        # Track input/output mappings for engines
        self.input_fields: Dict[str, Set[str]] = defaultdict(set)  # Engine name -> input fields
        self.output_fields: Dict[str, Set[str]] = defaultdict(set)  # Engine name -> output fields
        self.engine_io_mappings: Dict[str, Dict[str, List[str]]] = {}  # Engine name -> {"inputs": [...], "outputs": [...]}

    def _collect_field(
        self,
        name: str,
        field_type: Any,
        default: Any = None,
        default_factory: Optional[Callable[[], Any]] = None,
        description: Optional[str] = None,
        shared: bool = False,
        reducer: Optional[Callable] = None,
        source: Optional[str] = None
    ) -> None:
        """
        Collect field information internally (not exposed as public API).

        Args:
            name: Field name
            field_type: Field type
            default: Default value
            default_factory: Optional factory function for default value
            description: Optional field description
            shared: Whether field is shared with parent graph
            reducer: Optional reducer function for this field
            source: Optional source identifier for tracking
        """
        # CRITICAL: Never add __runnable_config__ to state schemas
        if name == "__runnable_config__" or name == "runnable_config":
            logger.warning(f"Attempted to add {name} to state schema. Ignoring.")
            return
            
        # Make field optional by default for Pydantic v2
        from typing import Optional as OptionalType
        if get_origin(field_type) is not OptionalType:
            field_type = OptionalType[field_type]
            
        # Check for Annotated types with reducers
        if get_origin(field_type) is Annotated:
            args = get_args(field_type)
            if len(args) > 1:
                # First arg is the actual type, remaining args could be reducers
                base_type = args[0]
                for arg in args[1:]:
                    if callable(arg) and not reducer:
                        # Found a reducer
                        reducer = arg
                        
                        # Fix: Handle the reducer name correctly here too
                        self.reducer_names[name] = SchemaUtils.get_reducer_name(arg)
                        
                        # Store the function itself
                        self.reducer_functions[name] = arg
                        break

                # Use the base type for the field and make it Optional
                field_type = OptionalType[base_type]

        # For collections, ensure sensible defaults
        origin_type = get_origin(field_type) or field_type
        if default is None and default_factory is None:
            if origin_type in (list, List):
                default_factory = list
            elif origin_type in (dict, Dict):
                default_factory = dict
            elif origin_type in (set, Set):
                default_factory = set
            elif origin_type is str:
                default = ""  # Empty string for str types
            elif origin_type is int:
                default = 0   # 0 for int types
            elif origin_type is bool:
                default = False  # False for bool types

        # Create the field info
        if default_factory is not None:
            field_info = Field(default_factory=default_factory)
        else:
            field_info = Field(default=default)

        # Store the field
        self.fields[name] = (field_type, field_info)

        # Store metadata
        if description:
            self.field_descriptions[name] = description

        if shared:
            self.shared_fields.add(name)

        if reducer and name not in self.reducer_names:  # Only add if not already handled by Annotated
            # Store reducer function in runtime dictionary
            self.reducer_functions[name] = reducer
            
            # Store serializable name for the reducer
            self.reducer_names[name] = SchemaUtils.get_reducer_name(reducer)

        if source:
            self.field_sources[name].add(source)

    def update_engine_io_mapping(self, engine_name: str) -> None:
        """
        Update the engine I/O mapping for a specific engine.
        
        Args:
            engine_name: Name of the engine to update mapping for
        """
        if engine_name not in self.engine_io_mappings:
            self.engine_io_mappings[engine_name] = {
                "inputs": [],
                "outputs": []
            }
        
        # Update inputs and outputs
        self.engine_io_mappings[engine_name]["inputs"] = list(self.input_fields.get(engine_name, set()))
        self.engine_io_mappings[engine_name]["outputs"] = list(self.output_fields.get(engine_name, set()))

    def _extract_input_output_patterns(self, component: Any, source: str) -> None:
        """
        Extract input and output patterns from components for better schema organization.
        
        Args:
            component: Component to extract patterns from
            source: Source identifier for tracking
        """
        # Try to identify input/output structure from engine types
        input_fields = {}
        output_fields = {}
        engine_name = getattr(component, "name", source)
        
        # Check if component is an engine with identifiable input/output patterns
        if hasattr(component, "engine_type"):
            engine_type = getattr(component, "engine_type")
            
            # Handle different engine types
            if str(engine_type) == "llm" or str(engine_type) == "LLM":
                # LLMs typically take input/prompt and return output/content
                input_fields["input"] = (str, "")
                input_fields["question"] = (str, "")  # Common alias
                input_fields["content"] = (str, "")  # Another alias
                output_fields["output"] = (str, "")
                output_fields["content"] = (str, "")  # Dual purpose field
                
                # Track standard message field for LLMs
                input_fields["messages"] = (None, None)  # Type will be handled by _ensure_messages_field
                output_fields["messages"] = (None, None)  # Messages can be both input and output
                
            elif str(engine_type) == "retriever" or str(engine_type) == "RETRIEVER":
                # Retrievers take query and return documents
                from typing import List
                try:
                    from langchain_core.documents import Document
                    input_fields["query"] = (str, "")
                    output_fields["documents"] = (List[Document], list)
                except ImportError:
                    # Fall back to generic if langchain imports fail
                    input_fields["query"] = (str, "")
                    output_fields["context"] = (List[str], list)
                    
            elif str(engine_type) == "embeddings" or str(engine_type) == "EMBEDDINGS":
                # Embedding engines convert text to vectors
                from typing import List, Union
                input_fields["text"] = (Union[str, List[str]], "")
                output_fields["embeddings"] = (List[List[float]], list)
                
            elif str(engine_type) == "vectorstore" or str(engine_type) == "VECTOR_STORE":
                # Vector stores typically take queries and return documents
                from typing import List, Dict
                try:
                    from langchain_core.documents import Document
                    input_fields["query"] = (str, "")
                    input_fields["filter"] = (Dict[str, Any], None)
                    output_fields["documents"] = (List[Document], list)
                except ImportError:
                    input_fields["query"] = (str, "")
                    output_fields["results"] = (List[Dict[str, Any]], list)
                    
            elif str(engine_type) == "tool" or str(engine_type) == "TOOL":
                # Tools typically have an input/output pattern
                input_fields["tool_input"] = (str, "")
                output_fields["tool_output"] = (str, "")
        
        # Extract and store the fields
        for name, (field_type, default_value) in input_fields.items():
            # Skip if type is None (handled elsewhere, like messages)
            if field_type is None:
                continue
                
            if name not in self.fields:
                if callable(default_value) and not isinstance(default_value, type):
                    # This is a default_factory
                    self._collect_field(
                        name=name,
                        field_type=field_type,
                        default_factory=default_value,
                        description=f"Input field for {source}",
                        source=source
                    )
                else:
                    # Regular default value
                    self._collect_field(
                        name=name,
                        field_type=field_type,
                        default=default_value,
                        description=f"Input field for {source}",
                        source=source
                    )
            
            # Track as input field for this engine
            self.input_fields[engine_name].add(name)
                
        for name, (field_type, default_value) in output_fields.items():
            # Skip if type is None
            if field_type is None:
                continue
                
            if name not in self.fields:
                if callable(default_value) and not isinstance(default_value, type):
                    # This is a default_factory
                    self._collect_field(
                        name=name,
                        field_type=field_type,
                        default_factory=default_value,
                        description=f"Output field for {source}",
                        source=source
                    )
                else:
                    # Regular default value
                    self._collect_field(
                        name=name,
                        field_type=field_type,
                        default=default_value,
                        description=f"Output field for {source}",
                        source=source
                    )
                
            # Track as output field for this engine
            self.output_fields[engine_name].add(name)
            
        # Update the engine I/O mappings
        self.update_engine_io_mapping(engine_name)

    def _extract_schema_fields(self, model: Type[BaseModel], source: str) -> None:
        """
        Extract fields from a Pydantic model.
        
        Args:
            model: Pydantic model to extract fields from
            source: Source identifier for tracking
        """
        # Check if model is a Pydantic model
        if not isinstance(model, type) or not issubclass(model, BaseModel):
            logger.warning(f"Expected Pydantic model, got {type(model)} from {source}")
            return
            
        # Get fields from Pydantic v2 model_fields
        fields_dict = model.model_fields
        
        # Extract fields
        for field_name, field_info in fields_dict.items():
            # Skip internal fields
            if field_name.startswith("__") or field_name == "runnable_config":
                continue
                
            # Extract field type and default (Pydantic v2)
            field_type = field_info.annotation
            default = field_info.default
            default_factory = field_info.default_factory
            description = field_info.description
                
            # Check if this is a StateSchema with shared fields
            shared = False
            if hasattr(model, "__shared_fields__") and field_name in model.__shared_fields__:
                shared = True
                
            # Check if field has a reducer in the model
            reducer = None
            if hasattr(model, "__reducer_fields__") and field_name in model.__reducer_fields__:
                reducer = model.__reducer_fields__[field_name]
                
            # Only add field if not already present (prioritizing earlier components)
            if field_name not in self.fields:
                self._collect_field(
                    name=field_name,
                    field_type=field_type,
                    default=default,
                    default_factory=default_factory,
                    description=description,
                    shared=shared,
                    reducer=reducer,
                    source=source
                )

    def _extract_prompt_variables(self, prompt_template: Any, source: str) -> None:
        """
        Extract variables from a prompt template.
        
        Args:
            prompt_template: Prompt template to extract variables from
            source: Source identifier for tracking
        """
        # Check if this is a prompt template with input_variables
        if hasattr(prompt_template, "input_variables"):
            variables = getattr(prompt_template, "input_variables")
            
            # Add each variable as a field
            for variable in variables:
                if variable not in self.fields:
                    self._collect_field(
                        name=variable,
                        field_type=str,
                        default="",
                        description=f"Prompt variable from {source}",
                        source=source
                    )
                    
            # Track variables as input fields for the engine
            engine_name = getattr(prompt_template, "name", source)
            for variable in variables:
                self.input_fields[engine_name].add(variable)
            
            # Update engine I/O mappings
            self.update_engine_io_mapping(engine_name)

    def _extract_engine_schema(self, engine: Any, source: str) -> None:
        """
        Extract schema from an engine.
        
        Args:
            engine: Engine to extract schema from
            source: Source identifier for tracking
        """
        # Try multiple methods to get schema information from engines
        
        # Method 1: Check for get_schema_fields method
        if hasattr(engine, "get_schema_fields") and callable(engine.get_schema_fields):
            try:
                schema_fields = engine.get_schema_fields()
                
                # Add fields from schema_fields dictionary
                for field_name, (field_type, default) in schema_fields.items():
                    # Skip __runnable_config__
                    if field_name == "__runnable_config__" or field_name == "runnable_config":
                        continue
                        
                    # Determine if this is a default or default_factory
                    default_factory = None
                    if callable(default) and not isinstance(default, type):
                        default_factory = default
                        default = None
                        
                    if field_name not in self.fields:
                        self._collect_field(
                            name=field_name,
                            field_type=field_type,
                            default=default,
                            default_factory=default_factory,
                            description=f"Field from {source}",
                            source=source
                        )
            except Exception as e:
                logger.warning(f"Error extracting schema fields from {source}: {e}")
                
        # Method 2: Check for derive_input_schema or derive_output_schema methods
        for schema_method, field_type in [
            ("derive_input_schema", "input"),
            ("derive_output_schema", "output")
        ]:
            if hasattr(engine, schema_method) and callable(getattr(engine, schema_method)):
                try:
                    schema = getattr(engine, schema_method)()
                    if isinstance(schema, type) and issubclass(schema, BaseModel):
                        # Extract fields from the schema
                        self._extract_schema_fields(schema, f"{source}_{field_type}")
                        
                        # Track input/output fields
                        engine_name = getattr(engine, "name", source)
                        
                        # Get all field names from the schema (Pydantic v2)
                        fields_dict = schema.model_fields
                        field_names = list(fields_dict.keys())
                        
                        # Track as input or output fields
                        if field_type == "input":
                            for field_name in field_names:
                                if field_name not in ["__runnable_config__", "runnable_config"]:
                                    self.input_fields[engine_name].add(field_name)
                        else:  # output
                            for field_name in field_names:
                                if field_name not in ["__runnable_config__", "runnable_config"]:
                                    self.output_fields[engine_name].add(field_name)
                                    
                        # Update engine I/O mappings
                        self.update_engine_io_mapping(engine_name)
                except Exception as e:
                    logger.warning(f"Error deriving {field_type} schema from {source}: {e}")
        
        # Method 3: Extract prompt variables if engine has a prompt attribute
        if hasattr(engine, "prompt") and engine.prompt is not None:
            prompt_template = engine.prompt
            self._extract_prompt_variables(prompt_template, source)
        
        # Make all fields optional by default for Pydantic v2
        for name, (field_type, field_info) in list(self.fields.items()):
            from typing import Optional
            if get_origin(field_type) is not Optional:
                # Create a new optional type
                optional_type = Optional[field_type]
                # Fix: Ensure default values are not PydanticUndefined
                if hasattr(field_info, "default") and (field_info.default is ... or str(field_info.default) == "PydanticUndefined"):
                    # For required fields, provide a sensible default
                    if get_origin(field_type) in (list, List):
                        new_field_info = Field(default_factory=list)
                    elif get_origin(field_type) in (dict, Dict):
                        new_field_info = Field(default_factory=dict)
                    elif field_type is str:
                        new_field_info = Field(default="")
                    elif field_type is int:
                        new_field_info = Field(default=0)
                    elif field_type is bool:
                        new_field_info = Field(default=False)
                    else:
                        new_field_info = Field(default=None)
                    self.fields[name] = (optional_type, new_field_info)
                else:
                    # Keep the same field info for fields with defaults
                    self.fields[name] = (optional_type, field_info)

    def _ensure_messages_field(self, source: str) -> None:
        """
        Ensure the messages field exists with proper configuration.
        
        Args:
            source: Source identifier for tracking
        """
        if "messages" not in self.fields:
            # Try to create with add_messages reducer
            try:
                from langgraph.graph import add_messages
                from typing import List, Sequence
                from langchain_core.messages import BaseMessage
                
                # Add the field with the reducer - Fix: use proper default_factory
                self._collect_field(
                    name="messages",
                    field_type=List[BaseMessage],  # Use List instead of Annotated to avoid serialization issues
                    default_factory=list,  # Ensure this is set to list function
                    description="Messages for agent conversation",
                    shared=False,
                    reducer=add_messages,
                    source=source
                )
            except ImportError:
                # Fallback if add_messages is not available
                from typing import List
                try:
                    from langchain_core.messages import BaseMessage
                    
                    # Create simple concat lists reducer
                    def concat_lists(a, b):
                        return (a or []) + (b or [])
                    
                    self._collect_field(
                        name="messages",
                        field_type=List[BaseMessage],
                        default_factory=list,
                        description="Messages for agent conversation",
                        shared=False,
                        reducer=concat_lists,
                        source=source
                    )
                except ImportError:
                    # Last resort with generic list
                    self._collect_field(
                        name="messages",
                        field_type=List[Any],
                        default_factory=list,
                        description="Messages for agent conversation",
                        source=source
                    )

    def add_fields_from_model(self, model: Type[BaseModel]) -> "SchemaComposer":
        """
        Extract fields from a Pydantic model.

        Args:
            model: Pydantic model to extract fields from
            
        Returns:
            Self for chaining
        """
        source = model.__name__ if hasattr(model, "__name__") else str(model)
        self._extract_schema_fields(model, source)
        return self

    def add_fields_from_dict(self, fields: Dict[str, Any]) -> "SchemaComposer":
        """
        Add fields from a dictionary.

        Args:
            fields: Dictionary of field definitions
            
        Returns:
            Self for chaining
        """
        for name, value in fields.items():
            if isinstance(value, tuple) and len(value) == 2:
                # Handle (type, default) format
                field_type, default = value
                
                # Determine if this is a factory or simple default
                default_factory = None
                if callable(default) and not isinstance(default, type):
                    default_factory = default
                    default = None
                    
                self._collect_field(
                    name=name,
                    field_type=field_type,
                    default=default,
                    default_factory=default_factory,
                    source="dict"
                )
            else:
                # Infer type from value
                self._collect_field(
                    name=name,
                    field_type=type(value),
                    default=value,
                    source="dict"
                )
        return self

    def add_fields_from_engine(self, engine: Any) -> "SchemaComposer":
        """
        Extract fields from an Engine object.
        
        Args:
            engine: Engine object to extract fields from
            
        Returns:
            Self for chaining
        """
        source = getattr(engine, "name", str(engine))
        
        # Extract from schema methods
        self._extract_engine_schema(engine, source)
        
        # Extract input/output patterns
        self._extract_input_output_patterns(engine, source)
        
        # Try to extract from prompt if available
        if hasattr(engine, "prompt") and engine.prompt is not None:
            self._extract_prompt_variables(engine.prompt, source)
            
        # Make all fields optional by default for Pydantic v2
        for name, (field_type, field_info) in list(self.fields.items()):
            from typing import Optional
            if get_origin(field_type) is not Optional:
                # Create a new optional type
                optional_type = Optional[field_type]
                # Fix: Ensure default values are not PydanticUndefined
                if hasattr(field_info, "default") and (field_info.default is ... or str(field_info.default) == "PydanticUndefined"):
                    # For required fields, provide a sensible default
                    if get_origin(field_type) in (list, List):
                        new_field_info = Field(default_factory=list)
                    elif get_origin(field_type) in (dict, Dict):
                        new_field_info = Field(default_factory=dict)
                    elif field_type is str:
                        new_field_info = Field(default="")
                    elif field_type is int:
                        new_field_info = Field(default=0)
                    elif field_type is bool:
                        new_field_info = Field(default=False)
                    else:
                        new_field_info = Field(default=None)
                    self.fields[name] = (optional_type, new_field_info)
                else:
                    # Keep the same field info for fields with defaults
                    self.fields[name] = (optional_type, field_info)
            
        return self

    def compose_from_components(self, components: List[Any]) -> "SchemaComposer":
        """
        Extract field information from multiple components.
        
        Args:
            components: List of components to extract fields from
            
        Returns:
            Self for chaining
        """
        for component in components:
            if component is None:
                continue
                
            # Handle engine components
            if hasattr(component, "engine_type"):
                self.add_fields_from_engine(component)
                
            # Handle BaseModel components
            elif isinstance(component, type) and issubclass(component, BaseModel):
                self.add_fields_from_model(component)
                
            # Handle dictionaries
            elif isinstance(component, dict):
                self.add_fields_from_dict(component)
                
            # Handle instances of BaseModel
            elif isinstance(component, BaseModel):
                self.add_fields_from_model(component.__class__)
                
            # Handle prompt templates directly
            elif hasattr(component, "input_variables"):
                source = getattr(component, "name", str(component))
                # Extract prompt variables directly (no separate method call)
                variables = getattr(component, "input_variables")
                
                # Add each variable as a field
                for variable in variables:
                    if variable not in self.fields:
                        self._collect_field(
                            name=variable,
                            field_type=str,
                            default="",
                            description=f"Prompt variable from {source}",
                            source=source
                        )
                
            else:
                logger.debug(f"Skipping unsupported component: {type(component)}")
                
        return self

    def to_manager(self) -> "StateSchemaManager":
        """
        Convert the extracted fields to a StateSchemaManager for further manipulation.
        
        Returns:
            StateSchemaManager instance
        """
        from haive.core.schema.schema_manager import StateSchemaManager
        
        # Create manager
        manager = StateSchemaManager(name=self.name)
        
        # Add fields
        for name, (field_type, field_info) in self.fields.items():
            description = self.field_descriptions.get(name)
            shared = name in self.shared_fields
            reducer = self.reducer_functions.get(name)
            
            # Get default and default_factory from field_info
            if hasattr(field_info, "default_factory") and field_info.default_factory is not None:
                default_factory = field_info.default_factory
                manager.add_field(
                    name=name,
                    field_type=field_type,
                    default_factory=default_factory,
                    description=description,
                    shared=shared,
                    reducer=reducer
                )
            else:
                default = field_info.default
                manager.add_field(
                    name=name,
                    field_type=field_type,
                    default=default,
                    description=description,
                    shared=shared,
                    reducer=reducer
                )
                
        # Add engine I/O tracking
        manager._input_fields = {k: set(v) for k, v in self.input_fields.items()}
        manager._output_fields = {k: set(v) for k, v in self.output_fields.items()}
        manager._engine_io_mappings = self.engine_io_mappings.copy()
        
        return manager

    def get_engine_io_mappings(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Get input/output field mappings for each engine.
        
        Returns:
            Dictionary mapping engine names to their input/output field lists
        """
        return self.engine_io_mappings.copy()
        
    def get_input_fields(self, engine_name: Optional[str] = None) -> List[str]:
        """
        Get input fields for a specific engine or all engines.
        
        Args:
            engine_name: Optional engine name to filter by
            
        Returns:
            List of input field names
        """
        if engine_name:
            return list(self.input_fields.get(engine_name, set()))
        else:
            # Get all unique input fields across all engines
            result = set()
            for fields in self.input_fields.values():
                result.update(fields)
            return list(result)
            
    def get_output_fields(self, engine_name: Optional[str] = None) -> List[str]:
        """
        Get output fields for a specific engine or all engines.
        
        Args:
            engine_name: Optional engine name to filter by
            
        Returns:
            List of output field names
        """
        if engine_name:
            return list(self.output_fields.get(engine_name, set()))
        else:
            # Get all unique output fields across all engines
            result = set()
            for fields in self.output_fields.values():
                result.update(fields)
            return list(result)

    def build(self) -> Type[StateSchema]:
        """
        Build a StateSchema directly from the composer.
        
        Returns:
            StateSchema class with fields from all components
        """
        from haive.core.schema.schema_manager import StateSchemaManager
        
        # Convert all fields to be optional with proper defaults
        updated_fields = {}
        for name, (field_type, field_info) in self.fields.items():
            from typing import Optional
            # Make field optional if not already
            if get_origin(field_type) is not Optional:
                field_type = Optional[field_type]
                
            # Ensure default values are provided
            if hasattr(field_info, "default_factory") and field_info.default_factory is not None:
                updated_field = Field(
                    default_factory=field_info.default_factory, 
                    description=self.field_descriptions.get(name)
                )
            else:
                default = field_info.default
                if default is ... or default == "PydanticUndefined":
                    default = None
                updated_field = Field(
                    default=default, 
                    description=self.field_descriptions.get(name)
                )
                
            updated_fields[name] = (field_type, updated_field)
            
        # Replace fields with updated ones
        self.fields = updated_fields
        
        # Create the model through StateSchemaManager
        manager = self.to_manager()
        return manager.get_model(name=self.name)
    
    @classmethod
    def from_components(
        cls, 
        components: List[Any], 
        name: str = "ComposedSchema",
        include_messages: bool = True
    ) -> "StateSchemaManager":
        """
        Extract fields from components and create a StateSchemaManager.
        
        This is the primary entry point for creating schemas from components.
        
        Args:
            components: List of components to extract fields from
            name: Name for the schema
            include_messages: Whether to ensure a messages field with reducer
            
        Returns:
            StateSchemaManager instance
        """
        # Create composer and extract fields
        composer = cls(name=name)
        composer.compose_from_components(components)
        
        # Ensure messages field if requested
        if include_messages:
            composer._ensure_messages_field(source="default")
            
        # Convert to manager
        return composer.to_manager()
    
    @classmethod
    def create_model(
        cls,
        components: List[Any],
        name: str = "ComposedSchema",
        include_messages: bool = True,
        as_state_schema: bool = True
    ) -> Type[BaseModel]:
        """
        Create a StateSchema directly from components in one step.
        
        This is the most concise way to create a schema from components.
        
        Args:
            components: List of components to extract fields from
            name: Name for the schema
            include_messages: Whether to ensure a messages field with reducer
            as_state_schema: Whether to use StateSchema as base (True) or BaseModel (False)
            
        Returns:
            StateSchema class with fields from all components
        """
        # Create composer and extract fields
        composer = cls(name=name)
        composer.compose_from_components(components)
        
        # Ensure messages field if requested
        if include_messages:
            composer._ensure_messages_field(source="default")
        
        # Make all fields optional for compatibility
        for field_name, (field_type, field_info) in list(composer.fields.items()):
            from typing import Optional
            
            # Make field optional if not already
            if get_origin(field_type) is not Optional:
                new_type = Optional[field_type]
                
                # Get default or default_factory
                if hasattr(field_info, "default_factory") and field_info.default_factory is not None:
                    new_field = Field(
                        default_factory=field_info.default_factory,
                        description=composer.field_descriptions.get(field_name)
                    )
                else:
                    default = field_info.default
                    if default is ... or default == "PydanticUndefined":
                        default = None
                    new_field = Field(
                        default=default,
                        description=composer.field_descriptions.get(field_name)
                    )
                
                # Replace field with optional version
                composer.fields[field_name] = (new_type, new_field)
        
        # Build and return model
        manager = composer.to_manager()
        return manager.get_model(name=name, as_state_schema=as_state_schema)

    @classmethod
    def compose(
        cls,
        components: List[Any],
        name: str = "ComposedSchema",
        include_messages: bool = True
    ) -> Type[BaseModel]:
        """
        Create a StateSchema directly from components (alias for create_model).
        
        This is the most concise way to create a schema from components.
        
        Args:
            components: List of components to extract fields from
            name: Name for the schema
            include_messages: Whether to ensure a messages field with reducer
            
        Returns:
            StateSchema class with fields from all components
        """
        return cls.create_model(components, name, include_messages)
    
    @classmethod
    def compose_as_state_schema(
        cls,
        components: List[Any],
        name: str = "ComposedSchema",
        include_messages: bool = True
    ) -> Type[StateSchema]:
        """
        Create a StateSchema from components.
        
        This method ensures the return type is explicitly StateSchema.
        
        Args:
            components: List of components to extract fields from
            name: Name for the schema
            include_messages: Whether to ensure a messages field with reducer
            
        Returns:
            StateSchema class with fields from all components
        """
        model = cls.create_model(components, name, include_messages, as_state_schema=True)
        return model  # Type is guaranteed to be StateSchema
    
    @classmethod
    def compose_input_schema(
        cls,
        components: List[Any],
        name: str = "InputSchema"
    ) -> Type[BaseModel]:
        """
        Create an input schema from components.
        
        Args:
            components: List of components to extract fields from
            name: Name for the schema
            
        Returns:
            BaseModel class suitable for input validation
        """
        manager = cls.from_components(components, name, include_messages=False)
        return manager.get_model(name=name, as_state_schema=False)
    
    @classmethod
    def compose_output_schema(
        cls,
        components: List[Any],
        name: str = "OutputSchema"
    ) -> Type[BaseModel]:
        """
        Create an output schema from components.
        
        Args:
            components: List of components to extract fields from
            name: Name for the schema
            
        Returns:
            BaseModel class suitable for output validation
        """
        manager = cls.from_components(components, name, include_messages=False)
        return manager.get_model(name=name, as_state_schema=False)
    
    @classmethod
    def create_message_state(
        cls,
        additional_fields: Optional[Dict[str, Any]] = None,
        name: str = "MessageState"
    ) -> Type[StateSchema]:
        """
        Create a standard message-based schema.
        
        Args:
            additional_fields: Additional fields to add
            name: Name for the schema
            
        Returns:
            StateSchema with messages field and specified additional fields
        """
        # Create composer with messages field
        composer = cls(name=name)
        composer._ensure_messages_field(source="default")
        
        # Add additional fields if provided
        if additional_fields:
            composer.add_fields_from_dict(additional_fields)
            
        # Create and return schema
        return composer.to_manager().get_model()

    def build_with_command_support(self) -> Type[StateSchema]:
        """
        Build a StateSchema with Command/Send support methods.
        
        Returns:
            StateSchema class with Command/Send helper methods
        """
        # Create the base schema
        schema_cls = self.build()
        
        # Add Command creation class method
        def create_command(cls, update: Dict[str, Any], goto: Optional[str] = None, 
                          resume: Optional[Any] = None, graph: Optional[str] = None) -> Command:
            # Handle special END case
            if goto == "END":
                from langgraph.graph import END
                goto = END
            return Command(update=update, goto=goto, resume=resume, graph=graph)
        
        schema_cls.create_command = classmethod(create_command)
        
        # Add Send creation class method
        def create_send(cls, node: str, arg: Any) -> Send:
            return Send(node, arg)
        
        schema_cls.create_send = classmethod(create_send)
        
        # Add instance method to convert state to Command
        def to_command(self, goto: Optional[str] = None, resume: Optional[Any] = None,
                      graph: Optional[str] = None) -> Command:
            # Handle special END case
            if goto == "END":
                from langgraph.graph import END
                goto = END
            return Command(update=self.to_dict(), goto=goto, resume=resume, graph=graph)
        
        schema_cls.to_command = to_command
        
        # Add instance method to convert state to Send
        def to_send(self, node: str) -> Send:
            return Send(node, self.to_dict())
        
        schema_cls.to_send = to_send
        
        return schema_cls