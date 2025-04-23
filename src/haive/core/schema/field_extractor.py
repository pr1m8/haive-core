"""
Field extractor utility for extracting field information from various sources.
"""
from typing import (
    Any, Dict, List, Optional, Set, Type, Tuple, get_origin,
    Callable, TypeVar
)
from pydantic import BaseModel, Field
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

        # Return all extracted data
        return (fields, descriptions, shared_fields, reducer_names,
                reducer_functions, engine_io_mappings, input_fields, output_fields)
    
    def extract_from_engine(engine: Any) -> Tuple[Dict[str, Any], Dict[str, str], Dict[str, Dict[str, List[str]]], Dict[str, Set[str]], Dict[str, Set[str]]]:
        fields = {}
        descriptions = {}
        engine_io_mappings = {}
        input_fields = defaultdict(set)
        output_fields = defaultdict(set)
        
        # Extract engine name for tracking - be more robust about getting a usable name
        if hasattr(engine, "name") and engine.name:
            engine_name = engine.name
        elif hasattr(engine, "id") and engine.id:
            engine_name = engine.id
        else:
            engine_name = str(engine)
        
        # Create an initial empty mapping for this engine even if no fields are found
        # This ensures the engine is at least represented in the mappings
        engine_io_mappings[engine_name] = {
            "inputs": [],
            "outputs": []
        }
        
        # Rest of field extraction logic...
        
        # Populate the mappings
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