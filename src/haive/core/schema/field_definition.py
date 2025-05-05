"""
FieldDefinition for the Haive framework.

The FieldDefinition class represents a complete field definition including type,
default value, metadata, and additional properties.
"""

import inspect
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from pydantic import BaseModel, Field

from haive.core.schema.field_utils import create_annotated_field, create_field

logger = logging.getLogger(__name__)


class FieldDefinition:
    """
    Complete field definition with metadata.

    The FieldDefinition encapsulates all information about a field, including
    its type, default value, description, and relationships to engines, making
    it the core building block for schema composition.
    """

    def __init__(
        self,
        name: str,
        field_type: Type[Any],
        field_info: Any = None,
        default: Any = None,
        default_factory: Optional[Callable[[], Any]] = None,
        description: Optional[str] = None,
        shared: bool = False,
        reducer: Optional[Callable] = None,
        source: Optional[str] = None,
        input_for: Optional[List[str]] = None,
        output_from: Optional[List[str]] = None,
        structured_model: Optional[str] = None,
        **kwargs,
    ):
        """Initialize a field definition.

        Args:
            name: Field name
            field_type: Type of the field
            field_info: Pydantic FieldInfo object
            default: Default value
            default_factory: Factory function for default value
            description: Field description
            shared: Whether field is shared with parent graph
            reducer: Reducer function for this field
            source: Source component name
            input_for: List of engines this field serves as input for
            output_from: List of engines this field is output from
            structured_model: Name of structured model this field belongs to
            **kwargs: Additional metadata
        """
        self.name = name
        self.field_type = field_type
        self.field_info = field_info
        self.default = default
        self.default_factory = default_factory
        self.description = description
        self.shared = shared
        self.reducer = reducer
        self.source = source
        self.input_for = input_for or []
        self.output_from = output_from or []
        self.structured_model = structured_model
        self.metadata = kwargs

    @classmethod
    def extract_from_model_field(
        cls,
        name: str,
        field_type: Type[Any],
        field_info: Any,
        include_annotations: bool = True,
    ) -> "FieldDefinition":
        """
        Extract a FieldDefinition from a Pydantic model field.

        Args:
            name: Field name
            field_type: Field type
            field_info: Pydantic FieldInfo object
            include_annotations: Whether to extract metadata from annotations

        Returns:
            FieldDefinition instance
        """
        # Extract basic properties
        default = field_info.default if hasattr(field_info, "default") else None
        default_factory = (
            field_info.default_factory
            if hasattr(field_info, "default_factory")
            else None
        )
        description = (
            field_info.description if hasattr(field_info, "description") else None
        )

        # If no default or factory and field is required (default is ...), set appropriate default
        if default is ... and default_factory is None:
            # Set to None to allow proper state management
            default = None

        # Create field definition
        field_def = cls(
            name=name,
            field_type=field_type,
            field_info=field_info,
            default=default,
            default_factory=default_factory,
            description=description,
        )

        # Extract additional metadata from annotations if available
        if include_annotations and hasattr(field_type, "__metadata__"):
            for annotation in field_type.__metadata__:
                if hasattr(annotation, "__dict__"):
                    for key, value in annotation.__dict__.items():
                        if key.startswith("_") or key in ["__module__", "__doc__"]:
                            continue
                        field_def.metadata[key] = value

        return field_def

    def to_field_info(self) -> Tuple[Type[Any], Any]:
        """
        Convert to a field type and info pair for model creation.

        Returns:
            Tuple of (field_type, field_info)
        """
        if self.field_info:
            # Use existing field info if available
            return self.field_type, self.field_info

        # Create field using utility function
        field_type, field_info = create_field(
            field_type=self.field_type,
            default=self.default,
            default_factory=self.default_factory,
            description=self.description,
            shared=self.shared,
            reducer=self.reducer,
            **self.metadata,
        )

        return field_type, field_info

    def to_annotated_field(self) -> Tuple[Type[Any], Any]:
        """
        Convert to an annotated field type and info pair.

        Returns:
            Tuple of (annotated_field_type, field_info)
        """
        # Create annotated field using utility function
        field_type, field_info = create_annotated_field(
            field_type=self.field_type,
            default=self.default,
            default_factory=self.default_factory,
            description=self.description,
            shared=self.shared,
            reducer=self.reducer,
            **self.metadata,
        )

        return field_type, field_info

    def get_reducer_name(self) -> Optional[str]:
        """
        Get the reducer function name for serialization.

        Returns:
            String representation of the reducer function
        """
        if not self.reducer:
            return None

        # Try to get a meaningful name
        if hasattr(self.reducer, "__name__"):
            name = self.reducer.__name__
        elif hasattr(self.reducer, "__qualname__"):
            name = self.reducer.__qualname__
        else:
            name = str(self.reducer)

        # Add module path if available
        if hasattr(self.reducer, "__module__"):
            return f"{self.reducer.__module__}.{name}"
        else:
            return name

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the field definition to a dictionary.

        Returns:
            Dictionary representation of the field definition
        """
        result = {
            "name": self.name,
            "field_type": str(self.field_type),
            "default": self.default,
            "description": self.description,
            "shared": self.shared,
            "source": self.source,
            "input_for": self.input_for,
            "output_from": self.output_from,
            "structured_model": self.structured_model,
        }

        if self.default_factory:
            result["default_factory"] = str(self.default_factory)

        if self.reducer:
            result["reducer"] = self.get_reducer_name()

        result.update(self.metadata)

        return result
