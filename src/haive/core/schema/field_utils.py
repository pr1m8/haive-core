"""
Field utilities for standardized Pydantic field handling in the Haive framework.

This module provides consistent utilities for creating, extracting, and manipulating
Pydantic fields across the entire framework, ensuring proper handling of field
metadata, types, and defaults.
"""
from typing import (
    Any, Dict, List, Optional, Set, Type, Tuple, get_origin, get_args,
    Callable, TypeVar, Union, Annotated, cast, ForwardRef
)
import inspect
import logging
from pydantic import BaseModel, Field, create_model
from collections import defaultdict
import operator
import sys

logger = logging.getLogger(__name__)

# Type variables
T = TypeVar('T')
FieldType = TypeVar('FieldType')
DefaultType = TypeVar('DefaultType')
ReducerType = TypeVar('ReducerType', bound=Callable[[Any, Any], Any])


class FieldMetadata:
    """
    Standardized container for field metadata in the Haive framework.
    
    This class encapsulates all metadata associated with a field:
    - Basic field properties (description, title, etc.)
    - Haive-specific properties (shared, reducer, etc.)
    - Engine I/O tracking (input/output relationships)
    
    It serves as a single source of truth for field metadata throughout the framework.
    """
    
    def __init__(
        self,
        description: Optional[str] = None,
        shared: bool = False,
        reducer: Optional[Callable] = None,
        source: Optional[str] = None,
        input_for: Optional[List[str]] = None,
        output_from: Optional[List[str]] = None,
        structured_model: Optional[str] = None,
        title: Optional[str] = None,
        **extra
    ):
        """
        Initialize field metadata with comprehensive properties.
        
        Args:
            description: Human-readable description of the field
            shared: Whether field is shared with parent graphs
            reducer: Function to combine field values during state updates
            source: Component that provided this field
            input_for: List of engines this field serves as input for
            output_from: List of engines this field is output from
            structured_model: Name of structured model this field belongs to
            title: Optional field title (for OpenAPI/Schema generation)
            **extra: Additional metadata properties
        """
        self.description = description
        self.shared = shared
        self.reducer = reducer
        self.source = source
        self.input_for = input_for or []
        self.output_from = output_from or []
        self.structured_model = structured_model
        self.title = title
        self.extra = extra
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert metadata to dictionary for Field instantiation.
        
        Returns:
            Dictionary of metadata suitable for pydantic.Field constructor
        """
        result = {}
        if self.description:
            result["description"] = self.description
        if self.title:
            result["title"] = self.title
            
        # Add standard Field parameters from extra
        standard_params = {
            "gt", "ge", "lt", "le", "multiple_of", "min_items", "max_items",
            "min_length", "max_length", "pattern", "discriminator", "json_schema_extra",
            "validation_alias", "serialization_alias", "validation_alias_priority",
            "serialization_alias_priority", "deprecated", "examples"
        }
        
        for k, v in self.extra.items():
            if k in standard_params:
                result[k] = v
        
        return result
    
    def to_annotation_metadata(self) -> List[Any]:
        """
        Convert to a list of metadata objects for Annotated types.
        
        Returns:
            List of metadata objects for use in Annotated[Type, ...]
        """
        metadata = []
        
        # Add reducer as first metadata if present
        if self.reducer:
            metadata.append(self.reducer)
            
        # Create a dictionary for other metadata
        other_meta = {}
        if self.shared:
            other_meta['shared'] = True
        if self.source:
            other_meta['source'] = self.source
        if self.input_for:
            other_meta['input_for'] = self.input_for
        if self.output_from:
            other_meta['output_from'] = self.output_from
        if self.structured_model:
            other_meta['structured_model'] = self.structured_model
        
        # Add to metadata list if we have any other metadata
        if other_meta:
            metadata.append(other_meta)
            
        return metadata
    
    def merge(self, other: 'FieldMetadata') -> 'FieldMetadata':
        """
        Merge with another FieldMetadata instance.
        
        Args:
            other: FieldMetadata instance to merge with
            
        Returns:
            New FieldMetadata instance with merged data
        """
        # Start with a copy of the current metadata
        result = FieldMetadata(
            description=other.description or self.description,
            shared=other.shared or self.shared,
            reducer=other.reducer or self.reducer,
            source=other.source or self.source,
            title=other.title or self.title
        )
        
        # Merge lists
        result.input_for = list(set(self.input_for + other.input_for))
        result.output_from = list(set(self.output_from + other.output_from))
        
        # Structured model takes the newer value
        result.structured_model = other.structured_model or self.structured_model
        
        # Merge extra attributes, with other taking precedence
        result.extra = self.extra.copy()
        for k, v in other.extra.items():
            result.extra[k] = v
            
        return result
    
    def get_reducer_name(self) -> Optional[str]:
        """
        Get serializable name for the reducer function.
        
        Returns:
            String name of the reducer function or None
        """
        if not self.reducer:
            return None
            
        # Handle operator module functions
        if hasattr(self.reducer, "__module__") and self.reducer.__module__ in ('operator', '_operator'):
            return f"operator.{self.reducer.__name__}"
        
        # Handle lambda functions
        if hasattr(self.reducer, "__name__") and self.reducer.__name__ == "<lambda>":
            return "<lambda>"
            
        # Handle standard functions
        if hasattr(self.reducer, "__name__"):
            if hasattr(self.reducer, "__module__") and self.reducer.__module__ != "__main__":
                # Use fully qualified name for imported functions
                return f"{self.reducer.__module__}.{self.reducer.__name__}"
            return self.reducer.__name__
            
        # Last resort: string representation
        return str(self.reducer)
    
    @classmethod
    def from_annotation(cls, annotation: Type) -> Optional['FieldMetadata']:
        """
        Extract field metadata from an annotated type.
        
        Args:
            annotation: Type annotation to extract metadata from
            
        Returns:
            FieldMetadata if metadata was found, None otherwise
        """
        # Check if this is an Annotated type
        if get_origin(annotation) is not Annotated:
            return None
            
        # Get args from Annotated[Type, ...]
        args = get_args(annotation)
        if len(args) < 2:
            return None
            
        # Extract metadata from annotation
        metadata = FieldMetadata()
        
        # Process each metadata object
        for meta in args[1:]:
            if callable(meta) and not isinstance(meta, type):
                # Found a callable, assume it's a reducer
                metadata.reducer = meta
            elif isinstance(meta, dict):
                # Dictionary of metadata
                if 'shared' in meta:
                    metadata.shared = meta['shared']
                if 'source' in meta:
                    metadata.source = meta['source']
                if 'input_for' in meta:
                    metadata.input_for = meta['input_for']
                if 'output_from' in meta:
                    metadata.output_from = meta['output_from']
                if 'structured_model' in meta:
                    metadata.structured_model = meta['structured_model']
                if 'description' in meta:
                    metadata.description = meta['description']
                if 'title' in meta:
                    metadata.title = meta['title']
                
                # Add any other keys to extra
                for k, v in meta.items():
                    if k not in {'shared', 'source', 'input_for', 'output_from', 
                                'structured_model', 'description', 'title'}:
                        metadata.extra[k] = v
        
        return metadata


def create_field(
    field_type: Type[T],
    default: Any = None,
    default_factory: Optional[Callable[[], T]] = None,
    metadata: Optional[FieldMetadata] = None,
    description: Optional[str] = None,
    shared: bool = False,
    reducer: Optional[Callable] = None,
    make_optional: bool = True,
    **kwargs
) -> Tuple[Type, Field]:
    """
    Create a standardized Pydantic field with consistent metadata handling.
    
    Args:
        field_type: The type of the field
        default: Default value (used if default_factory is None)
        default_factory: Optional factory function for default value
        metadata: Optional FieldMetadata object for comprehensive metadata
        description: Optional field description (ignored if metadata is provided)
        shared: Whether field is shared with parent (ignored if metadata is provided)
        reducer: Optional reducer function (ignored if metadata is provided)
        make_optional: Whether to make the field Optional if it's not already
        **kwargs: Additional field parameters
        
    Returns:
        Tuple of (field_type, field_info) ready for Pydantic model creation
    """
    from typing import Optional as OptionalType
    
    # Process field type
    if make_optional and field_type is not type(None) and get_origin(field_type) is not OptionalType:
        field_type = OptionalType[field_type]
    
    # Prepare field metadata
    meta = metadata or FieldMetadata(
        description=description,
        shared=shared,
        reducer=reducer,
        **kwargs
    )
    
    # Build field parameters
    field_params = meta.to_dict()
    
    # Handle default value or factory
    if default_factory is not None:
        field_info = Field(default_factory=default_factory, **field_params)
    else:
        field_info = Field(default=default, **field_params)
    
    # Return the field definition
    return field_type, field_info


def create_annotated_field(
    field_type: Type[T],
    default: Any = None,
    default_factory: Optional[Callable[[], T]] = None,
    metadata: Optional[FieldMetadata] = None,
    description: Optional[str] = None,
    shared: bool = False,
    reducer: Optional[Callable] = None,
    make_optional: bool = True,
    **kwargs
) -> Tuple[Type, Field]:
    """
    Create a field using Annotated type for metadata.
    
    This version uses Annotated[Type, ...] to store metadata within the type itself,
    which is more aligned with Pydantic v2's design.
    
    Args:
        field_type: The type of the field
        default: Default value (used if default_factory is None)
        default_factory: Optional factory function for default value
        metadata: Optional FieldMetadata object for comprehensive metadata
        description: Optional field description (ignored if metadata is provided)
        shared: Whether field is shared with parent (ignored if metadata is provided)
        reducer: Optional reducer function (ignored if metadata is provided)
        make_optional: Whether to make the field Optional if it's not already
        **kwargs: Additional field parameters
        
    Returns:
        Tuple of (field_type, field_info) ready for Pydantic model creation
    """
    from typing import Optional as OptionalType
    
    # Prepare field metadata
    meta = metadata or FieldMetadata(
        description=description,
        shared=shared,
        reducer=reducer,
        **kwargs
    )
    
    # Get base type (unwrap Optional if needed)
    base_type = field_type
    if get_origin(field_type) is OptionalType:
        base_type = get_args(field_type)[0]
    
    # Create Annotated type with metadata
    meta_objects = meta.to_annotation_metadata()
    if meta_objects:
        annotated_type = Annotated[(base_type, *meta_objects)]
        
        # Re-wrap in Optional if needed
        if make_optional or get_origin(field_type) is OptionalType:
            field_type = OptionalType[annotated_type]
        else:
            field_type = annotated_type
    elif make_optional and get_origin(field_type) is not OptionalType:
        field_type = OptionalType[field_type]
    
    # Build field parameters (only basic ones for Field constructor)
    field_params = {}
    if meta.description:
        field_params["description"] = meta.description
    if meta.title:
        field_params["title"] = meta.title
        
    # Add standard Field parameters from extra
    standard_params = {
        "gt", "ge", "lt", "le", "multiple_of", "min_items", "max_items",
        "min_length", "max_length", "pattern", "discriminator", "json_schema_extra",
        "validation_alias", "serialization_alias", "validation_alias_priority",
        "serialization_alias_priority", "deprecated", "examples"
    }
    
    for k, v in meta.extra.items():
        if k in standard_params:
            field_params[k] = v
    
    # Handle default value or factory
    if default_factory is not None:
        field_info = Field(default_factory=default_factory, **field_params)
    else:
        field_info = Field(default=default, **field_params)
    
    # Return the field definition
    return field_type, field_info


def extract_field_info(
    field_info: Field
) -> Tuple[Any, Optional[Callable], Dict[str, Any]]:
    """
    Extract useful information from a Pydantic Field object.
    
    Args:
        field_info: Pydantic Field object
        
    Returns:
        Tuple of (default_value, default_factory, metadata_dict)
    """
    # Extract default or default_factory
    default = getattr(field_info, "default", ...)
    default_factory = getattr(field_info, "default_factory", None)
    
    # Extract metadata
    metadata = {}
    for key in ["description", "title"]:
        value = getattr(field_info, key, None)
        if value is not None:
            metadata[key] = value
            
    # Extract extra (varies between Pydantic v1 and v2)
    if hasattr(field_info, "json_schema_extra") and field_info.json_schema_extra:
        metadata.update(field_info.json_schema_extra)
    elif hasattr(field_info, "extra") and field_info.extra:
        metadata.update(field_info.extra)
    
    return default, default_factory, metadata


def extract_type_metadata(type_annotation: Type) -> Tuple[Type, Optional[FieldMetadata]]:
    """
    Extract base type and metadata from a type annotation.
    
    Args:
        type_annotation: Type annotation to extract from
        
    Returns:
        Tuple of (base_type, field_metadata)
    """
    # For Annotated types, extract metadata
    if get_origin(type_annotation) is Annotated:
        args = get_args(type_annotation)
        base_type = args[0]
        metadata = FieldMetadata.from_annotation(type_annotation)
        return base_type, metadata
    
    # For Optional[Annotated[...]] types
    from typing import Optional as OptionalType
    if get_origin(type_annotation) is OptionalType:
        inner_type = get_args(type_annotation)[0]
        if get_origin(inner_type) is Annotated:
            base_type, metadata = extract_type_metadata(inner_type)
            return OptionalType[base_type], metadata
    
    # For other types, return as is with no metadata
    return type_annotation, None


def format_type_annotation(type_annotation: Type) -> str:
    """
    Format a type annotation for display or documentation.
    
    Args:
        type_annotation: Type annotation to format
        
    Returns:
        Formatted string representation of the type
    """
    # Handle primitive types
    if type_annotation is str:
        return "str"
    if type_annotation is int:
        return "int"
    if type_annotation is float:
        return "float"
    if type_annotation is bool:
        return "bool"
    if type_annotation is list or type_annotation is List:
        return "List"
    if type_annotation is dict or type_annotation is Dict:
        return "Dict"
    if type_annotation is None or type_annotation is type(None):
        return "None"

    # Get origin and arguments
    origin = get_origin(type_annotation)
    args = get_args(type_annotation)
    
    # Handle None origin
    if origin is None:
        if isinstance(type_annotation, ForwardRef):
            return f'"{type_annotation.__forward_arg__}"'
        if hasattr(type_annotation, "__name__"):
            if hasattr(type_annotation, "__module__") and type_annotation.__module__ != "builtins":
                return f"{type_annotation.__module__}.{type_annotation.__name__}"
            return type_annotation.__name__
        return str(type_annotation).replace("typing.", "")
    
    # Extract base type name
    if hasattr(origin, "__name__"):
        origin_name = origin.__name__
    else:
        origin_name = str(origin).replace("typing.", "")
    
    # Handle special cases
    from typing import Optional as OptionalType
    from typing import Union
    
    if origin is OptionalType:
        inner_type = format_type_annotation(args[0])
        return f"Optional[{inner_type}]"
    elif origin is Union:
        # Check if it's Optional (Union with None)
        if len(args) == 2 and (args[1] is None or args[1] is type(None)):
            inner_type = format_type_annotation(args[0])
            return f"Optional[{inner_type}]"
        elif len(args) == 2 and (args[0] is None or args[0] is type(None)):
            inner_type = format_type_annotation(args[1])
            return f"Optional[{inner_type}]"
        else:
            inner_types = [format_type_annotation(arg) for arg in args]
            return f"Union[{', '.join(inner_types)}]"
    elif origin is list or origin is List:
        if args:
            inner_type = format_type_annotation(args[0])
            return f"List[{inner_type}]"
        return "List"
    elif origin is dict or origin is Dict:
        if len(args) == 2:
            key_type = format_type_annotation(args[0])
            value_type = format_type_annotation(args[1])
            return f"Dict[{key_type}, {value_type}]"
        return "Dict"
    elif origin is Annotated:
        # For Annotated types, format the base type but ignore metadata
        return format_type_annotation(args[0])
    
    # Generic handling
    if args:
        inner_types = [format_type_annotation(arg) for arg in args]
        return f"{origin_name}[{', '.join(inner_types)}]"
    
    return origin_name


def get_common_reducers() -> Dict[str, Callable]:
    """
    Get a registry of common reducer functions.
    
    Returns:
        Dictionary of reducer name -> reducer function
    """
    reducers = {}
    
    # Add operator module reducers
    for name in dir(operator):
        if not name.startswith('_'):
            func = getattr(operator, name)
            if callable(func):
                reducers[f"operator.{name}"] = func
                reducers[name] = func  # Also store without prefix for backward compatibility
    
    # Add common list operations
    def add_lists(a, b):
        return (a or []) + (b or [])
    reducers["add_lists"] = add_lists
    
    def concat_lists(a, b):
        return (a or []) + (b or [])
    reducers["concat_lists"] = concat_lists
    
    # Try to import langgraph's add_messages if available
    try:
        from langgraph.graph import add_messages
        reducers["add_messages"] = add_messages
    except ImportError:
        # Fallback implementation
        def add_messages(a, b):
            return (a or []) + (b or [])
        reducers["add_messages"] = add_messages
    
    # Add string operations
    def concat_strings(a, b):
        return (a or "") + (b or "")
    reducers["concat_strings"] = concat_strings
    
    # Add numeric operations
    def sum_values(a, b):
        return (a or 0) + (b or 0)
    reducers["sum_values"] = sum_values
    
    # Add other common functions
    reducers["max"] = max
    reducers["min"] = min
    
    return reducers


def resolve_reducer(reducer_name: str) -> Optional[Callable]:
    """
    Resolve a reducer function from its name.
    
    Args:
        reducer_name: Name of the reducer to resolve
        
    Returns:
        Callable reducer function or None if not found
    """
    # Check common reducers first
    common_reducers = get_common_reducers()
    if reducer_name in common_reducers:
        return common_reducers[reducer_name]
    
    # Handle operator module functions
    if reducer_name.startswith("operator."):
        func_name = reducer_name.split(".", 1)[1]
        if hasattr(operator, func_name):
            return getattr(operator, func_name)
    
    # Try to import by module.function_name
    if "." in reducer_name:
        try:
            module_name, func_name = reducer_name.rsplit(".", 1)
            module = __import__(module_name, fromlist=[func_name])
            return getattr(module, func_name)
        except (ImportError, AttributeError):
            logger.warning(f"Could not resolve reducer: {reducer_name}")
            pass
    
    # Special case for lambda
    if reducer_name == "<lambda>":
        def generic_lambda_reducer(a, b):
            # Simple fallback that concatenates lists or dicts, otherwise takes b
            if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
                return a + b
            elif isinstance(a, dict) and isinstance(b, dict):
                result = a.copy()
                result.update(b)
                return result
            return b
        return generic_lambda_reducer
    
    # Could not resolve
    return None


def infer_field_type(value: Any) -> Type:
    """
    Infer the field type from a value.
    
    Args:
        value: Value to infer type from
        
    Returns:
        Inferred type
    """
    from typing import Optional, Dict, List, Any as AnyType
    
    if value is None:
        return Optional[AnyType]
    elif isinstance(value, str):
        return Optional[str]
    elif isinstance(value, int):
        return Optional[int]
    elif isinstance(value, float):
        return Optional[float]
    elif isinstance(value, bool):
        return Optional[bool]
    elif isinstance(value, list):
        if value and all(isinstance(x, type(value[0])) for x in value):
            item_type = infer_field_type(value[0])
            # Strip Optional from item_type
            if get_origin(item_type) is Optional:
                item_type = get_args(item_type)[0]
            return Optional[List[item_type]]
        return Optional[List[AnyType]]
    elif isinstance(value, dict):
        if value and all(isinstance(k, str) for k in value.keys()):
            return Optional[Dict[str, AnyType]]
        return Optional[Dict[AnyType, AnyType]]
    return Optional[AnyType]