from __future__ import annotations

"""Core type definitions for the schema compatibility module.
"""


from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class CompatibilityLevel(str, Enum):
    """Levels of type compatibility between schemas."""

    EXACT = "exact"  # Types are identical
    SUBTYPE = "subtype"  # Source is subtype of target
    PROTOCOL = "protocol"  # Source implements target protocol
    CONVERTIBLE = "convertible"  # Can convert with registered converter
    COERCIBLE = "coercible"  # Can coerce with potential data loss
    PARTIAL = "partial"  # Partially compatible (subset of fields)
    INCOMPATIBLE = "incompatible"  # Cannot convert


class ConversionQuality(str, Enum):
    """Quality levels for type conversions."""

    LOSSLESS = "lossless"  # No information lost
    SAFE = "safe"  # Minor formatting changes only
    LOSSY = "lossy"  # Some information lost
    UNSAFE = "unsafe"  # Significant information lost
    DESTRUCTIVE = "destructive"  # Major data loss


class MergeStrategy(str, Enum):
    """Strategies for merging schemas."""

    UNION = "union"  # Include all fields from all schemas
    INTERSECTION = "intersection"  # Only common fields
    OVERRIDE = "override"  # Last schema wins for conflicts
    DEEP = "deep"  # Deep merge nested structures
    CUSTOM = "custom"  # User-defined merge function


@dataclass
class TypeInfo:
    """Detailed information about a type."""

    type_hint: type[Any]
    origin: type[Any] | None = None
    args: tuple[type[Any], ...] = field(default_factory=tuple)
    is_generic: bool = False
    is_union: bool = False
    is_optional: bool = False
    is_protocol: bool = False
    is_typeddict: bool = False
    is_literal: bool = False
    is_forward_ref: bool = False
    is_basemodel: bool = False
    module: str | None = None
    qualname: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def full_name(self) -> str:
        """Get fully qualified type name."""
        if self.module and self.qualname:
            return f"{self.module}.{self.qualname}"
        if self.qualname:
            return self.qualname
        return str(self.type_hint)


@dataclass
class FieldInfo:
    """Information about a schema field."""

    name: str
    type_info: TypeInfo
    is_required: bool = True
    has_default: bool = False
    default_value: Any = None
    default_factory: Callable[[], Any] | None = None
    description: str | None = None
    alias: str | None = None
    validators: list[Callable[[Any], Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Haive-specific metadata
    is_shared: bool = False
    reducer: str | Callable | None = None
    input_for_engines: set[str] = field(default_factory=set)
    output_from_engines: set[str] = field(default_factory=set)

    @property
    def field_path(self) -> str:
        """Get field path including alias."""
        return self.alias or self.name


@dataclass
class SchemaInfo:
    """Complete information about a schema/model."""

    name: str
    type_info: TypeInfo
    fields: dict[str, FieldInfo] = field(default_factory=dict)
    methods: dict[str, Callable] = field(default_factory=dict)
    base_classes: list[type[Any]] = field(default_factory=list)

    # Haive StateSchema specific
    shared_fields: set[str] = field(default_factory=set)
    reducer_fields: dict[str, str | Callable] = field(default_factory=dict)
    engine_io_mappings: dict[str, dict[str, list[str]]] = field(default_factory=dict)

    # Validation and conversion
    validators: list[Callable] = field(default_factory=list)
    computed_fields: dict[str, Callable] = field(default_factory=dict)

    # Metadata
    version: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_required_fields(self) -> list[FieldInfo]:
        """Get all required fields."""
        return [f for f in self.fields.values() if f.is_required]

    def get_optional_fields(self) -> list[FieldInfo]:
        """Get all optional fields."""
        return [f for f in self.fields.values() if not f.is_required]


@dataclass
class ConversionPath:
    """Represents a conversion path between types."""

    source_type: type[Any]
    target_type: type[Any]
    steps: list[ConversionStep] = field(default_factory=list)
    total_quality: ConversionQuality = ConversionQuality.LOSSLESS
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_step(self, step: ConversionStep) -> None:
        """Add a conversion step and update quality."""
        self.steps.append(step)
        # Update total quality (worst quality in chain)
        if step.quality.value > self.total_quality.value:
            self.total_quality = step.quality

    @property
    def step_count(self) -> int:
        """Number of conversion steps."""
        return len(self.steps)


@dataclass
class ConversionStep:
    """Single step in a conversion path."""

    from_type: type[Any]
    to_type: type[Any]
    converter_name: str
    quality: ConversionQuality
    metadata: dict[str, Any] = field(default_factory=dict)


class ConversionContext(BaseModel):
    """Context passed through conversion pipeline."""

    source_type: str
    target_type: str
    quality: ConversionQuality = ConversionQuality.LOSSLESS
    conversion_path: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    lost_fields: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Performance tracking
    conversion_time_ms: float | None = None
    memory_usage_bytes: int | None = None

    def add_warning(self, warning: str) -> None:
        """Add a conversion warning."""
        self.warnings.append(warning)

    def add_error(self, error: str) -> None:
        """Add a conversion error."""
        self.errors.append(error)

    def track_lost_field(self, field_name: str, value: Any) -> None:
        """Track fields that will be lost in conversion."""
        self.lost_fields[field_name] = value

    def add_step(self, step: str) -> None:
        """Add a step to conversion path."""
        self.conversion_path.append(step)

    class Config:
        """Pydantic config."""

        extra = "allow"
        arbitrary_types_allowed = True


@dataclass
class ValidationResult:
    """Result of a validation operation."""

    is_valid: bool
    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[ValidationWarning] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_error(self, error: ValidationError) -> None:
        """Add validation error."""
        self.errors.append(error)
        self.is_valid = False

    def add_warning(self, warning: ValidationWarning) -> None:
        """Add validation warning."""
        self.warnings.append(warning)


@dataclass
class ValidationError:
    """Validation error details."""

    field: str | None
    message: str
    error_type: str = "validation_error"
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationWarning:
    """Validation warning details."""

    field: str | None
    message: str
    warning_type: str = "validation_warning"
    context: dict[str, Any] = field(default_factory=dict)


# Type aliases for better readability
FieldName = str
EngineName = str
TypeString = str
ConverterFunc = Callable[[Any, ConversionContext], Any]
ValidatorFunc = Callable[[Any], bool]
ReducerFunc = Callable[[Any, Any], Any]
