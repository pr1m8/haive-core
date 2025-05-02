"""
Field definition class for the Haive framework.

This module provides the FieldDefinition class, which represents a complete
definition of a field including its type, default value, and metadata.
"""

import logging
from typing import (
    Annotated,
    Any,
    Callable,
    Dict,
    Generic,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

from pydantic import Field  # , FieldInfo
from pydantic.fields import FieldInfo

logger = logging.getLogger(__name__)

# Type variables for field values and reducers
TField = TypeVar("TField")
TReducer = TypeVar("TReducer", bound=Callable[[Any, Any], Any])


class FieldDefinition(Generic[TField, TReducer]):
    """
    Definition of a schema field directly utilizing Pydantic's FieldInfo.

    This class encapsulates a field type and its associated Pydantic FieldInfo,
    along with Haive-specific metadata that extends beyond standard Pydantic capabilities.
    It serves as an intermediary representation during schema construction.
    """

    def __init__(
        self,
        name: str,
        field_type: Type[TField],
        field_info: Optional[FieldInfo] = None,
        default: Any = None,
        default_factory: Optional[Callable[[], TField]] = None,
        description: Optional[str] = None,
        shared: bool = False,
        reducer: Optional[TReducer] = None,
        source: Optional[str] = None,
        input_for: Optional[list[str]] = None,
        output_from: Optional[list[str]] = None,
        structured_model: Optional[str] = None,
        **extra,
    ):
        """
        Initialize a field definition with Pydantic FieldInfo.

        Args:
            name: Field name
            field_type: Type of the field
            field_info: Optional pre-constructed Pydantic FieldInfo
            default: Default value (ignored if field_info provided)
            default_factory: Default factory (ignored if field_info provided)
            description: Description (ignored if field_info provided)
            shared: Whether field is shared with parent graph
            reducer: Optional reducer function for this field
            source: Component that provided this field
            input_for: List of engines this field serves as input for
            output_from: List of engines this field is output from
            structured_model: Name of structured model this field belongs to
            **extra: Additional field parameters for Field constructor
        """
        self.name = name
        self.field_type = field_type

        # Haive-specific metadata (not part of standard Pydantic FieldInfo)
        self.shared = shared
        self.reducer = reducer
        self.source = source
        self.input_for = input_for or []
        self.output_from = output_from or []
        self.structured_model = structured_model

        # Use provided FieldInfo or create one
        if field_info is not None:
            self.field_info = field_info
        else:
            # Create field kwargs
            field_kwargs = {}
            if description:
                field_kwargs["description"] = description

            # Add extra parameters
            field_kwargs.update(extra)

            # Create FieldInfo
            if default_factory is not None:
                self.field_info = Field(default_factory=default_factory, **field_kwargs)
            else:
                self.field_info = Field(default=default, **field_kwargs)

    def to_field_info(self) -> Tuple[Type[TField], FieldInfo]:
        """
        Get the field type and FieldInfo for model creation.

        Returns:
            Tuple of (field_type, field_info) ready for model creation
        """
        # Return the field information directly
        return self.field_type, self.field_info

    def to_annotated_field(self) -> Tuple[Type[TField], FieldInfo]:
        """
        Create an Annotated type with embedded metadata.

        Returns:
            Tuple of (field_type, field_info) with metadata in Annotated
        """
        # Create a list for annotation metadata
        metadata = []

        # Add reducer as first metadata if present
        if self.reducer:
            metadata.append(self.reducer)

        # Create a dictionary for other Haive metadata
        haive_meta = {}
        if self.shared:
            haive_meta["shared"] = True
        if self.source:
            haive_meta["source"] = self.source
        if self.input_for:
            haive_meta["input_for"] = self.input_for
        if self.output_from:
            haive_meta["output_from"] = self.output_from
        if self.structured_model:
            haive_meta["structured_model"] = self.structured_model

        # Add metadata dict if not empty
        if haive_meta:
            metadata.append(haive_meta)

        # If we have metadata to add, create Annotated type
        if metadata:
            # Determine if field is already Optional
            from typing import Optional

            is_optional = False
            actual_type = self.field_type

            if get_origin(self.field_type) is Union:
                args = get_args(self.field_type)
                if len(args) == 2 and (args[1] is type(None) or args[1] is None):
                    is_optional = True
                    actual_type = args[0]
                elif len(args) == 2 and (args[0] is type(None) or args[0] is None):
                    is_optional = True
                    actual_type = args[1]

            # Create the annotated type
            annotated_type = Annotated[(actual_type, *metadata)]

            # Wrap in Optional if it was originally optional
            if is_optional:
                field_type = Optional[annotated_type]
            else:
                field_type = annotated_type

            return field_type, self.field_info

        # If no metadata to add, return the original field
        return self.field_type, self.field_info

    def get_reducer_name(self) -> Optional[str]:
        """
        Get serializable name for the reducer.

        Returns:
            String identifier for the reducer or None
        """
        if not self.reducer:
            return None

        # Handle operator module functions
        if hasattr(self.reducer, "__module__") and self.reducer.__module__ in (
            "operator",
            "_operator",
        ):
            return f"operator.{self.reducer.__name__}"

        # Handle lambda functions
        if hasattr(self.reducer, "__name__") and self.reducer.__name__ == "<lambda>":
            return "<lambda>"

        # Handle standard functions
        if hasattr(self.reducer, "__name__"):
            if (
                hasattr(self.reducer, "__module__")
                and self.reducer.__module__ != "__main__"
            ):
                # Use fully qualified name for imported functions
                return f"{self.reducer.__module__}.{self.reducer.__name__}"
            return self.reducer.__name__

        # Last resort: string representation
        return str(self.reducer)

    @property
    def description(self) -> Optional[str]:
        """Get field description from FieldInfo."""
        return getattr(self.field_info, "description", None)

    @description.setter
    def description(self, value: str) -> None:
        """Set field description in FieldInfo."""
        # This is not directly supported in Pydantic v2 as FieldInfo is immutable
        # We'll need to create a new FieldInfo
        field_kwargs = self._extract_field_kwargs()
        field_kwargs["description"] = value

        # Recreate the field info
        if getattr(self.field_info, "default_factory", None) is not None:
            self.field_info = Field(
                default_factory=self.field_info.default_factory, **field_kwargs
            )
        else:
            self.field_info = Field(default=self.field_info.default, **field_kwargs)

    @property
    def default(self) -> Any:
        """Get field default value."""
        return self.field_info.default if self.field_info.default is not ... else None

    @property
    def default_factory(self) -> Optional[Callable[[], Any]]:
        """Get field default factory."""
        return getattr(self.field_info, "default_factory", None)

    def _extract_field_kwargs(self) -> Dict[str, Any]:
        """Extract keyword arguments from field_info for creating a new Field."""
        kwargs = {}

        # Common Field parameters in Pydantic v2
        field_params = [
            "title",
            "description",
            "examples",
            "json_schema_extra",
            "deprecated",
            "validation_alias",
            "serialization_alias",
            "validation_alias_priority",
            "serialization_alias_priority",
            "gt",
            "ge",
            "lt",
            "le",
            "min_length",
            "max_length",
            "pattern",
            "discriminator",
        ]

        # Extract parameters if they exist
        for param in field_params:
            value = getattr(self.field_info, param, None)
            if value is not None and value is not ...:
                kwargs[param] = value

        return kwargs

    @classmethod
    def from_field_info(
        cls,
        name: str,
        field_type: Type[TField],
        field_info: FieldInfo,
        shared: bool = False,
        reducer: Optional[TReducer] = None,
        source: Optional[str] = None,
        input_for: Optional[list[str]] = None,
        output_from: Optional[list[str]] = None,
        structured_model: Optional[str] = None,
    ) -> "FieldDefinition[TField, TReducer]":
        """
        Create a FieldDefinition from existing field information.

        Args:
            name: Field name
            field_type: Field type
            field_info: Pydantic FieldInfo object
            shared: Whether field is shared with parent graph
            reducer: Optional reducer function
            source: Component that provided this field
            input_for: List of engines this field serves as input for
            output_from: List of engines this field is output from
            structured_model: Name of structured model this field belongs to

        Returns:
            New FieldDefinition instance
        """
        return cls(
            name=name,
            field_type=field_type,
            field_info=field_info,
            shared=shared,
            reducer=reducer,
            source=source,
            input_for=input_for,
            output_from=output_from,
            structured_model=structured_model,
        )

    @classmethod
    def extract_from_model_field(
        cls,
        name: str,
        field_type: Type[TField],
        field_info: FieldInfo,
        include_annotations: bool = True,
    ) -> "FieldDefinition[TField, Any]":
        """
        Extract a FieldDefinition from a model field, including annotations.

        Args:
            name: Field name
            field_type: Field type annotation
            field_info: Pydantic FieldInfo object
            include_annotations: Whether to extract metadata from Annotated types

        Returns:
            New FieldDefinition with extracted metadata
        """
        # Initialize with basic field info
        result = cls(name=name, field_type=field_type, field_info=field_info)

        # Extract metadata from Annotated type if requested
        if include_annotations and get_origin(field_type) is Annotated:
            args = get_args(field_type)
            if len(args) > 1:
                # First arg is the actual type
                base_type = args[0]
                # Rest are metadata
                for meta in args[1:]:
                    if callable(meta) and not isinstance(meta, type):
                        # Callable metadata is assumed to be a reducer
                        result.reducer = meta
                    elif isinstance(meta, dict):
                        # Dictionary metadata
                        if "shared" in meta:
                            result.shared = meta["shared"]
                        if "source" in meta:
                            result.source = meta["source"]
                        if "input_for" in meta and isinstance(meta["input_for"], list):
                            result.input_for = meta["input_for"]
                        if "output_from" in meta and isinstance(
                            meta["output_from"], list
                        ):
                            result.output_from = meta["output_from"]
                        if "structured_model" in meta:
                            result.structured_model = meta["structured_model"]

                # Update field type to base type
                result.field_type = base_type

        return result
