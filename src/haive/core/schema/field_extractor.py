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
    """
    Unified utility for extracting fields from various sources.
    
    This class provides methods for extracting field information from:
    - Pydantic models
    - Engine objects with get_input_fields/get_output_fields methods
    - Dictionaries with field definitions
    
    It standardizes the field extraction process across the schema system.
    """

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
            model_cls: Pydantic model class to extract from
            
        Returns:
            Tuple of (fields, descriptions, shared_fields, reducer_names,
                    reducer_functions, engine_io_mappings, input_fields,
                    output_fields)
        """
        fields = {}
        descriptions = {}
        shared_fields = set()
        reducer_names = {}
        reducer_functions = {}
        engine_io_mappings = {}
        input_fields = defaultdict(set)
        output_fields = defaultdict(set)

        # Check if it's a Pydantic model
        if not (isinstance(model_cls, type) and issubclass(model_cls, BaseModel)):
            logger.warning(f"Not a Pydantic model: {model_cls}")
            return (fields, descriptions, shared_fields, reducer_names,
                    reducer_functions, engine_io_mappings, input_fields,
                    output_fields)

        # Get all fields from model_fields (Pydantic v2)
        for field_name, field_info in model_cls.model_fields.items():
            # Skip internal fields
            if field_name.startswith("__") or field_name == "runnable_config":
                continue

            # Get field type and info
            field_type = field_info.annotation
            
            # Make field Optional if not already
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

            # Create field definition
            if default_factory is not None:
                fields[field_name] = (field_type, Field(
                    default_factory=default_factory,
                    description=description
                ))
            else:
                fields[field_name] = (field_type, Field(
                    default=default,
                    description=description
                ))

        # Extract shared fields
        if hasattr(model_cls, "__shared_fields__"):
            shared_fields.update(model_cls.__shared_fields__)

        # Extract reducer information
        if hasattr(model_cls, "__serializable_reducers__"):
            reducer_names.update(model_cls.__serializable_reducers__)

        if hasattr(model_cls, "__reducer_fields__"):
            reducer_functions.update(model_cls.__reducer_fields__)

        # Extract engine I/O mappings
        if hasattr(model_cls, "__engine_io_mappings__"):
            engine_io_mappings = model_cls.__engine_io_mappings__.copy()

        if hasattr(model_cls, "__input_fields__"):
            for engine, fields_list in model_cls.__input_fields__.items():
                input_fields[engine].update(fields_list)

        if hasattr(model_cls, "__output_fields__"):
            for engine, fields_list in model_cls.__output_fields__.items():
                output_fields[engine].update(fields_list)

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
        
        This method prioritizes using the engine's explicit get_input_fields
        and get_output_fields methods if available.
        
        Args:
            engine: Engine object to extract from
            
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
        engine_name = getattr(engine, "name", str(engine))

        # Create an initial empty mapping for this engine
        engine_io_mappings[engine_name] = {
            "inputs": [],
            "outputs": []
        }
        
        # Primary approach: Use engine's get_input_fields and get_output_fields methods
        input_fields_extracted = False
        output_fields_extracted = False
        
        # Try to get input fields directly from the engine
        if hasattr(engine, "get_input_fields") and callable(engine.get_input_fields):
            try:
                input_fields_dict = engine.get_input_fields()
                
                # Add input fields to fields dictionary
                for field_name, (field_type, field_info) in input_fields_dict.items():
                    fields[field_name] = (field_type, field_info)
                    
                    # Extract description if available
                    description = getattr(field_info, "description", None)
                    if description:
                        descriptions[field_name] = description
                    
                    # Add to input fields
                    input_fields[engine_name].add(field_name)
                
                # Mark as successful
                input_fields_extracted = True
                logger.debug(f"Extracted {len(input_fields_dict)} input fields from {engine_name}")
            except Exception as e:
                logger.warning(f"Error getting input_fields from {engine_name}: {e}")
        
        # Try to get output fields directly from the engine
        if hasattr(engine, "get_output_fields") and callable(engine.get_output_fields):
            try:
                output_fields_dict = engine.get_output_fields()
                
                # Add output fields to fields dictionary
                for field_name, (field_type, field_info) in output_fields_dict.items():
                    # Only add if not already added from input fields
                    if field_name not in fields:
                        fields[field_name] = (field_type, field_info)
                        
                        # Extract description if available
                        description = getattr(field_info, "description", None)
                        if description:
                            descriptions[field_name] = description
                    
                    # Add to output fields
                    output_fields[engine_name].add(field_name)
                
                # Mark as successful
                output_fields_extracted = True
                logger.debug(f"Extracted {len(output_fields_dict)} output fields from {engine_name}")
            except Exception as e:
                logger.warning(f"Error getting output_fields from {engine_name}: {e}")
        
        # Secondary approaches if primary extraction methods failed
        
        # Method 2: Check for structured_output_model in AugLLMConfig
        if hasattr(engine, "structured_output_model") and engine.structured_output_model is not None:
            try:
                model = engine.structured_output_model
                model_name = model.__name__.lower()
                
                # Add the model itself as a field
                fields[model_name] = (Optional[model], Field(
                    default=None,
                    description=f"Output in {model.__name__} format"
                ))
                
                # Track as output field
                output_fields[engine_name].add(model_name)
                    
            except Exception as e:
                logger.warning(f"Error extracting from structured_output_model in {engine_name}: {e}")
        
        # Method 3: Try to detect field requirements via get_in_out_variables method
        if hasattr(engine, "get_in_out_variables") and callable(engine.get_in_out_variables):
            try:
                in_vars, out_vars = engine.get_in_out_variables()
                
                # Process input variables
                if in_vars and not input_fields_extracted:
                    from typing import Any
                    
                    for var_name in in_vars:
                        if var_name not in fields:
                            # Create a generic field
                            fields[var_name] = (Optional[Any], Field(
                                default=None,
                                description=f"Input variable for {engine_name}"
                            ))
                            
                            # Track as input field
                            input_fields[engine_name].add(var_name)
                
                # Process output variables
                if out_vars and not output_fields_extracted:
                    from typing import Any
                    
                    for var_name in out_vars:
                        if var_name not in fields:
                            # Create a generic field
                            fields[var_name] = (Optional[Any], Field(
                                default=None,
                                description=f"Output variable from {engine_name}"
                            ))
                            
                            # Track as output field
                            output_fields[engine_name].add(var_name)
                
                logger.debug(f"Extracted variables from get_in_out_variables in {engine_name}")
            except Exception as e:
                logger.warning(f"Error getting variables from {engine_name}: {e}")
        
        # Method 4: Check for schema_fields method
        if hasattr(engine, "get_schema_fields") and callable(engine.get_schema_fields):
            try:
                schema_fields = engine.get_schema_fields()
                for field_name, (field_type, default) in schema_fields.items():
                    # Skip internal or special fields
                    if field_name.startswith("__") or field_name == "runnable_config":
                        continue
                    
                    # Skip if already extracted
                    if field_name in fields:
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
                    
                    # Track as generic field - we don't know if it's input or output
                    # For schema_fields, assume both for compatibility
                    input_fields[engine_name].add(field_name)
                    output_fields[engine_name].add(field_name)
                    
                logger.debug(f"Extracted fields from get_schema_fields in {engine_name}")
            except Exception as e:
                logger.warning(f"Error getting schema_fields from {engine_name}: {e}")
        
        # Method 5: Try derive_input_schema and derive_output_schema methods
        if hasattr(engine, "derive_input_schema") and callable(engine.derive_input_schema):
            try:
                input_schema = engine.derive_input_schema()
                
                # Extract fields from input schema
                if hasattr(input_schema, "model_fields"):
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
                        
                        # Track as input field
                        input_fields[engine_name].add(field_name)
                
                logger.debug(f"Extracted fields from derive_input_schema in {engine_name}")
            except Exception as e:
                logger.warning(f"Error deriving input schema from {engine_name}: {e}")
        
        if hasattr(engine, "derive_output_schema") and callable(engine.derive_output_schema):
            try:
                output_schema = engine.derive_output_schema()
                
                # Extract fields from output schema
                if hasattr(output_schema, "model_fields"):
                    for field_name, field_info in output_schema.model_fields.items():
                        # Skip internal, special fields
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
                        
                        # Track as output field
                        output_fields[engine_name].add(field_name)
                
                logger.debug(f"Extracted fields from derive_output_schema in {engine_name}")
            except Exception as e:
                logger.warning(f"Error deriving output schema from {engine_name}: {e}")
        
        # Update engine I/O mappings based on extracted fields
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
            data: Dictionary containing field data
            
        Returns:
            Tuple of (fields, descriptions, shared_fields, reducer_names,
                    reducer_functions, engine_io_mappings, input_fields,
                    output_fields)
        """
        from typing import Optional
        
        fields = {}
        descriptions = {}
        shared_fields = set()
        reducer_names = {}
        reducer_functions = {}
        engine_io_mappings = {}
        input_fields = defaultdict(set)
        output_fields = defaultdict(set)

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
            Inferred type annotation
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
            # Try to infer element type for homogeneous lists
            if value and all(isinstance(item, type(value[0])) for item in value):
                elem_type = FieldExtractor._infer_type(value[0])
                # Extract inner type from Optional
                if get_origin(elem_type) is Optional:
                    inner_type = get_args(elem_type)[0]
                    return Optional[List[inner_type]]
            return Optional[List[Any]]
        if isinstance(value, dict):
            # Try to infer key/value types for homogeneous dicts
            if value:
                keys = list(value.keys())
                values = list(value.values())
                if all(isinstance(k, type(keys[0])) for k in keys) and all(isinstance(v, type(values[0])) for v in values):
                    key_type = FieldExtractor._infer_type(keys[0])
                    val_type = FieldExtractor._infer_type(values[0])
                    # Extract inner types from Optional
                    if get_origin(key_type) is Optional and get_origin(val_type) is Optional:
                        inner_key = get_args(key_type)[0]
                        inner_val = get_args(val_type)[0]
                        return Optional[Dict[inner_key, inner_val]]
            return Optional[Dict[str, Any]]
        if isinstance(value, set):
            return Optional[Set[Any]]
        
        # Special handling for common objects
        try:
            from langchain_core.messages import BaseMessage
            if isinstance(value, BaseMessage):
                return Optional[BaseMessage]
        except ImportError:
            pass
            
        return Optional[Any]