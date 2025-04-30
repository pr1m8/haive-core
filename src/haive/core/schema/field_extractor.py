"""
Field extractor utility for extracting field information from various sources.
"""
from typing import (
    Any, Dict, List, Optional, Set, Type, Tuple, get_origin, get_args,
    Callable, TypeVar, Union
)
from pydantic import BaseModel, Field
import inspect
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

# Type variable for return types
T = TypeVar('T')

class FieldExtractor:
    """Unified utility for extracting fields from various sources."""

    @staticmethod
    def extract_from_model(
        model_cls: Type[BaseModel]
    ) -> Tuple[
        Dict[str, Tuple[Any, Any]],
        Dict[str, str],
        Set[str],
        Dict[str, str],
        Dict[str, Callable],
        Dict[str, Dict[str, List[str]]],
        Dict[str, Set[str]],
        Dict[str, Set[str]],
    ]:
        """
        Extract all field information from a Pydantic model.

        Args:
            model_cls: Pydantic model class

        Returns:
            Tuple of (fields, descriptions, shared_fields, reducer_names,
                     reducer_functions, engine_io_mappings, input_fields,
                     output_fields)
        """
        fields: Dict[str, Tuple[Any, Any]] = {}
        descriptions: Dict[str, str] = {}
        shared_fields: Set[str] = set()
        reducer_names: Dict[str, str] = {}
        reducer_functions: Dict[str, Callable] = {}
        engine_io_mappings: Dict[str, Dict[str, List[str]]] = {}
        input_fields: Dict[str, Set[str]] = defaultdict(set)
        output_fields: Dict[str, Set[str]] = defaultdict(set)

        # Check if it's a Pydantic model
        if not (isinstance(model_cls, type) and issubclass(model_cls, BaseModel)):
            logger.warning(f"Not a Pydantic model: {model_cls}")
            return (fields, descriptions, shared_fields, reducer_names,
                    reducer_functions, engine_io_mappings, input_fields,
                    output_fields)

        # Get all annotations
        annotations = getattr(model_cls, "__annotations__", {})

        # Process each field
        for field_name, field_type in annotations.items():
            # Skip internal fields
            if field_name.startswith("__") or field_name == "runnable_config":
                continue

            # Get field info from model_fields (Pydantic v2)
            if (hasattr(model_cls, "model_fields") and
                    field_name in model_cls.model_fields):
                field_info = model_cls.model_fields[field_name]

                # Make field Optional
                from typing import Optional
                if get_origin(field_type) is not Optional:
                    field_type = Optional[field_type]

                # Extract default and default_factory
                default = field_info.default
                default_factory = field_info.default_factory

                # Extract description
                description = getattr(field_info, "description", None)
                if description:
                    descriptions[field_name] = description

                # Handle required fields (PydanticUndefined or ...)
                if default is ... or str(default) == "PydanticUndefined":
                    default = None

                # Create field with proper defaults
                if default_factory is not None:
                    # Create a fresh Field, not reusing field_info directly
                    field_def = Field(
                        default_factory=default_factory,
                        description=description
                    )
                    fields[field_name] = (field_type, field_def)
                else:
                    field_def = Field(
                        default=default,
                        description=description
                    )
                    fields[field_name] = (field_type, field_def)
            else:
                # Fallback for models without model_fields
                try:
                    # Try to get default from class attribute
                    if hasattr(model_cls, field_name):
                        default = getattr(model_cls, field_name)

                        # If it's a factory function, handle differently
                        if callable(default) and not isinstance(default, type):
                            field_def = Field(default_factory=default)
                            fields[field_name] = (field_type, field_def)
                        else:
                            field_def = Field(default=default)
                            fields[field_name] = (field_type, field_def)
                    else:
                        # No default found, use None
                        field_def = Field(default=None)
                        fields[field_name] = (field_type, field_def)
                except Exception as e:
                    logger.warning(f"Error processing field {field_name}: {e}")
                    # Skip problematic fields
                    continue

        # Extract shared fields
        if hasattr(model_cls, "__shared_fields__"):
            shared_fields.update(model_cls.__shared_fields__)

        # Extract reducer information
        if hasattr(model_cls, "__serializable_reducers__"):
            reducer_names.update(model_cls.__serializable_reducers__)

        if hasattr(model_cls, "__reducer_fields__"):
            reducer_functions.update(model_cls.__reducer_fields__)

            # Ensure all reducer functions have names
            for field_name, reducer in model_cls.__reducer_fields__.items():
                if field_name not in reducer_names:
                    # Add missing reducer name
                    module = getattr(reducer, "__module__", "")
                    if module == 'operator':
                        name = reducer.__name__
                        reducer_names[field_name] = f"operator.{name}"
                    elif hasattr(reducer, "__name__"):
                        reducer_names[field_name] = reducer.__name__
                    else:
                        reducer_names[field_name] = str(reducer)

        # Extract engine I/O mappings
        if hasattr(model_cls, "__engine_io_mappings__"):
            engine_io_mappings = model_cls.__engine_io_mappings__.copy()

        if hasattr(model_cls, "__input_fields__"):
            for engine, fields in model_cls.__input_fields__.items():
                input_fields[engine].update(fields)

        if hasattr(model_cls, "__output_fields__"):
            for engine, fields in model_cls.__output_fields__.items():
                output_fields[engine].update(fields)

        return (fields, descriptions, shared_fields, reducer_names,
                reducer_functions, engine_io_mappings, input_fields, output_fields)

    @staticmethod
    def extract_from_engine(engine: Any) -> Tuple[
        Dict[str, Tuple[Any, Any]],  # fields
        Dict[str, str],  # descriptions
        Dict[str, Dict[str, List[str]]],  # engine_io_mappings
        Dict[str, Set[str]],  # input_fields
        Dict[str, Set[str]]  # output_fields
    ]:
        """
        Extract all field information from an engine.

        Args:
            engine: Engine object to extract fields from

        Returns:
            Tuple of (fields, descriptions, engine_io_mappings, input_fields, output_fields)
        """
        from typing import Optional
        
        fields = {}
        descriptions = {}
        engine_io_mappings = {}
        input_fields = defaultdict(set)
        output_fields = defaultdict(set)
        
        # Extract engine name for tracking
        if hasattr(engine, "name") and engine.name:
            engine_name = engine.name
        elif hasattr(engine, "id") and engine.id:
            engine_id = getattr(engine, "id")
            engine_name = engine.id
        else:
            engine_name = str(engine)

        # Create an initial empty mapping for this engine even if no fields are found
        engine_io_mappings[engine_name] = {
            "inputs": [],
            "outputs": []
        }
        
        # Try different methods to extract field information
        
        # Method 1: Check for get_schema_fields method
        if hasattr(engine, "get_schema_fields") and callable(engine.get_schema_fields):
            try:
                schema_fields = engine.get_schema_fields()
                for field_name, (field_type, default) in schema_fields.items():
                    # Skip internal or special fields
                    if field_name.startswith("__") or field_name == "runnable_config":
                        continue
                    
                    # Create field_info
                    if callable(default) and not isinstance(default, type):
                        # It's a factory function
                        field_info = Field(default_factory=default)
                    else:
                        field_info = Field(default=default)
                    
                    # Make type Optional if not already
                    if get_origin(field_type) is not Optional:
                        field_type = Optional[field_type]
                    
                    fields[field_name] = (field_type, field_info)
                    
                    # Track I/O fields - assume all schema fields are potential inputs
                    input_fields[engine_name].add(field_name)
            except Exception as e:
                logger.warning(f"Error getting schema_fields from {engine_name}: {e}")
        
        # Method 2: Check for structured_output_model in AugLLMConfig
        if hasattr(engine, "structured_output_model") and engine.structured_output_model is not None:
            try:
                output_model = engine.structured_output_model
                
                # Extract fields from the structured output model
                if hasattr(output_model, "model_fields"):
                    # Pydantic v2
                    for field_name, field_info in output_model.model_fields.items():
                        # Skip internal or duplicate fields
                        if field_name.startswith("__") or field_name in fields:
                            continue
                        
                        field_type = field_info.annotation
                        
                        # Make type Optional if not already
                        if get_origin(field_type) is not Optional:
                            field_type = Optional[field_type]
                        
                        # Create field with defaults
                        fields[field_name] = (field_type, field_info)
                        
                        # Get description if available
                        description = getattr(field_info, "description", None)
                        if description:
                            descriptions[field_name] = description
                        
                        # Mark as output field
                        output_fields[engine_name].add(field_name)
                
                # Also add the model name as a field (common pattern)
                model_name = output_model.__name__.lower()
                if model_name and model_name not in fields:
                    fields[model_name] = (Optional[output_model], Field(default=None))
                    output_fields[engine_name].add(model_name)
                    
            except Exception as e:
                logger.warning(f"Error extracting from structured_output_model in {engine_name}: {e}")
        
        # Method 3: Try derive_input_schema and derive_output_schema methods
        if hasattr(engine, "derive_input_schema") and callable(engine.derive_input_schema):
            try:
                input_schema = engine.derive_input_schema()
                
                # Extract fields from input schema
                if hasattr(input_schema, "model_fields"):
                    # Pydantic v2
                    for field_name, field_info in input_schema.model_fields.items():
                        # Skip internal, special, or duplicate fields
                        if field_name.startswith("__") or field_name == "runnable_config" or field_name in fields:
                            continue
                        
                        field_type = field_info.annotation
                        
                        # Make type Optional if not already
                        if get_origin(field_type) is not Optional:
                            field_type = Optional[field_type]
                        
                        # Create field with defaults
                        fields[field_name] = (field_type, field_info)
                        
                        # Get description if available
                        description = getattr(field_info, "description", None)
                        if description:
                            descriptions[field_name] = description
                        
                        # Mark as input field
                        input_fields[engine_name].add(field_name)
            except Exception as e:
                logger.warning(f"Error deriving input schema from {engine_name}: {e}")
        
        if hasattr(engine, "derive_output_schema") and callable(engine.derive_output_schema):
            try:
                output_schema = engine.derive_output_schema()
                
                # Extract fields from output schema
                if hasattr(output_schema, "model_fields"):
                    # Pydantic v2
                    for field_name, field_info in output_schema.model_fields.items():
                        # Skip internal, special, or duplicate fields
                        if field_name.startswith("__") or field_name == "runnable_config":
                            continue
                        
                        field_type = field_info.annotation
                        
                        # Make type Optional if not already
                        if get_origin(field_type) is not Optional:
                            field_type = Optional[field_type]
                        
                        # Create or update field with defaults
                        if field_name not in fields:
                            fields[field_name] = (field_type, field_info)
                        
                        # Get description if available
                        description = getattr(field_info, "description", None)
                        if description and field_name not in descriptions:
                            descriptions[field_name] = description
                        
                        # Mark as output field
                        output_fields[engine_name].add(field_name)
            except Exception as e:
                logger.warning(f"Error deriving output schema from {engine_name}: {e}")
        
        # Method 4: Extract I/O mappings if available
        if hasattr(engine, "input_schema") and engine.input_schema:
            try:
                input_schema = engine.input_schema
                
                # Extract fields from input schema
                if hasattr(input_schema, "model_fields"):
                    # Pydantic v2
                    for field_name in input_schema.model_fields:
                        # Mark as input field
                        input_fields[engine_name].add(field_name)
            except Exception as e:
                logger.warning(f"Error extracting from input_schema in {engine_name}: {e}")
        
        if hasattr(engine, "output_schema") and engine.output_schema:
            try:
                output_schema = engine.output_schema
                
                # Extract fields from output schema
                if hasattr(output_schema, "model_fields"):
                    # Pydantic v2
                    for field_name in output_schema.model_fields:
                        # Mark as output field
                        output_fields[engine_name].add(field_name)
            except Exception as e:
                logger.warning(f"Error extracting from output_schema in {engine_name}: {e}")
        
        # Method 5: For AugLLMConfig, handle special fields
        if hasattr(engine, "uses_messages_field") and engine.uses_messages_field:
            # LLM engines typically use messages
            input_fields[engine_name].add("messages")
            
            # Additional fields typically used
            common_fields = ["content", "query", "question"]
            for field in common_fields:
                input_fields[engine_name].add(field)
        
        # Always include messages field as a likely input if no input fields found
        if not input_fields[engine_name]:
            input_fields[engine_name].add("messages")
        
        # Always ensure content is an input field for LLMs
        if hasattr(engine, "engine_type") and str(engine.engine_type).lower() == "llm" and "content" not in input_fields[engine_name]:
            input_fields[engine_name].add("content")
        
        # Update engine I/O mappings
        engine_io_mappings[engine_name]["inputs"] = list(input_fields[engine_name])
        engine_io_mappings[engine_name]["outputs"] = list(output_fields[engine_name])
        
        return fields, descriptions, engine_io_mappings, input_fields, output_fields

    @staticmethod
    def extract_from_dict(data: Dict[str, Any]) -> Tuple[
        Dict[str, Tuple[Any, Any]],
        Dict[str, str],
        Set[str],
        Dict[str, str],
        Dict[str, Callable],
        Dict[str, Dict[str, List[str]]],
        Dict[str, Set[str]],
        Dict[str, Set[str]],
    ]:
        """
        Extract fields from a dictionary.

        Args:
            data: Dictionary with field definitions

        Returns:
            Tuple of (fields, descriptions, shared_fields, reducer_names,
                    reducer_functions, engine_io_mappings, input_fields,
                    output_fields)
        """
        from typing import Optional
        
        fields: Dict[str, Tuple[Any, Any]] = {}
        descriptions: Dict[str, str] = {}
        shared_fields: Set[str] = set()
        reducer_names: Dict[str, str] = {}
        reducer_functions: Dict[str, Callable] = {}
        engine_io_mappings: Dict[str, Dict[str, List[str]]] = {}
        input_fields: Dict[str, Set[str]] = defaultdict(set)
        output_fields: Dict[str, Set[str]] = defaultdict(set)

        for key, value in data.items():
            # Handle special keys
            if key == "shared_fields":
                shared_fields.update(value)
            elif key == "reducer_names" or key == "serializable_reducers":
                reducer_names.update(value)
            elif key == "reducer_functions":
                reducer_functions.update(value)
            elif key == "field_descriptions":
                descriptions.update(value)
            elif key == "engine_io_mappings":
                engine_io_mappings.update(value)
            elif key == "input_fields":
                for engine, fields_list in value.items():
                    input_fields[engine].update(fields_list)
            elif key == "output_fields":
                for engine, fields_list in value.items():
                    output_fields[engine].update(fields_list)
            elif isinstance(value, tuple) and len(value) == 2:
                # Handle (type, default) format
                field_type, default = value

                # Make field Optional
                if get_origin(field_type) is not Optional:
                    field_type = Optional[field_type]

                # Determine if default is a factory
                if callable(default) and not isinstance(default, type):
                    field_def = Field(default_factory=default)
                else:
                    field_def = Field(default=default)

                fields[key] = (field_type, field_def)
            else:
                # Infer type from value
                field_type = FieldExtractor._infer_type(value)

                field_def = Field(default=value)
                fields[key] = (field_type, field_def)

        # Return all extracted data
        return (fields, descriptions, shared_fields, reducer_names,
                reducer_functions, engine_io_mappings, input_fields, output_fields)

    @staticmethod
    def _infer_type(value: Any) -> Any:
        """
        Infer the type of a value with special handling for collections.

        Args:
            value: Value to infer type from

        Returns:
            Inferred type
        """
        from typing import Optional, Dict, List, Set, Any
        
        if isinstance(value, str):
            return Optional[str]
        if isinstance(value, int):
            return Optional[int]
        if isinstance(value, float):
            return Optional[float]
        if isinstance(value, bool):
            return Optional[bool]
        if isinstance(value, list):
            return Optional[List[Any]]
        if isinstance(value, dict):
            return Optional[Dict[str, Any]]
        if isinstance(value, set):
            return Optional[Set[Any]]
        return Optional[Any]