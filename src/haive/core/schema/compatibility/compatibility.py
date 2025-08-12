"""Type compatibility checking system."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from haive.core.schema.compatibility.converters import get_converter_registry
from haive.core.schema.compatibility.types import (
    CompatibilityLevel,
    ConversionPath,
)


class CompatibilityChecker:
    """Check compatibility between types and schemas."""

    def __init__(self) -> None:
        """Initialize compatibility checker."""
        self._converter_registry = get_converter_registry()
        self._compatibility_cache: dict[
            tuple[type[Any], type[Any]], CompatibilityLevel
        ] = {}

    def check_compatibility(
        self, source_type: type[Any], target_type: type[Any]
    ) -> CompatibilityLevel:
        """Check compatibility level between two types."""
        # Check cache first
        cache_key = (source_type, target_type)
        if cache_key in self._compatibility_cache:
            return self._compatibility_cache[cache_key]

        level = self._determine_compatibility(source_type, target_type)
        self._compatibility_cache[cache_key] = level
        return level

    def _determine_compatibility(
        self, source_type: type[Any], target_type: type[Any]
    ) -> CompatibilityLevel:
        """Determine compatibility level between types."""
        # Exact match
        if source_type == target_type:
            return CompatibilityLevel.EXACT

        # Check subtype relationship
        try:
            if issubclass(source_type, target_type):
                return CompatibilityLevel.SUBTYPE
        except TypeError:
            # Not class types
            pass

        # Check if types share a protocol
        if self._check_protocol_compatibility(source_type, target_type):
            return CompatibilityLevel.PROTOCOL

        # Check if converter exists
        if self._converter_registry.find_converter(source_type, target_type):
            return CompatibilityLevel.CONVERTIBLE

        # Check if types can be coerced
        if self._check_coercible(source_type, target_type):
            return CompatibilityLevel.COERCIBLE

        # Check partial compatibility (for structured types)
        if self._check_partial_compatibility(source_type, target_type):
            return CompatibilityLevel.PARTIAL

        return CompatibilityLevel.INCOMPATIBLE

    def _check_protocol_compatibility(
        self, source_type: type[Any], target_type: type[Any]
    ) -> bool:
        """Check if types are compatible via protocol."""
        # Check if target is a Protocol
        if hasattr(target_type, "_is_protocol") and target_type._is_protocol:
            # Check if source implements the protocol
            try:
                # Simple check - more sophisticated implementation needed
                source_attrs = set(dir(source_type))
                target_attrs = {
                    attr
                    for attr in dir(target_type)
                    if not attr.startswith("_") or attr == "__init__"
                }
                return target_attrs.issubset(source_attrs)
            except Exception:
                return False
        return False

    def _check_coercible(self, source_type: type[Any], target_type: type[Any]) -> bool:
        """Check if types can be coerced (with potential data loss)."""
        # Define coercible type pairs
        coercible_pairs = [
            (int, float),
            (int, str),
            (float, str),
            (bool, int),
            (bool, str),
            (list, tuple),
            (tuple, list),
            (set, list),
            (list, set),
        ]

        for src, tgt in coercible_pairs:
            try:
                if issubclass(source_type, src) and issubclass(target_type, tgt):
                    return True
            except TypeError:
                continue

        return False

    def _check_partial_compatibility(
        self, source_type: type[Any], target_type: type[Any]
    ) -> bool:
        """Check if types are partially compatible (share some fields)."""
        # Check if both are Pydantic models
        if hasattr(source_type, "__fields__") and hasattr(target_type, "__fields__"):
            source_fields = set(source_type.__fields__.keys())
            target_fields = set(target_type.__fields__.keys())
            # Partial if they share at least one field
            return bool(source_fields & target_fields)

        # Check if both are TypedDicts
        if hasattr(source_type, "__annotations__") and hasattr(
            target_type, "__annotations__"
        ):
            source_fields = set(source_type.__annotations__.keys())
            target_fields = set(target_type.__annotations__.keys())
            return bool(source_fields & target_fields)

        return False

    def get_conversion_path(
        self, source_type: type[Any], target_type: type[Any]
    ) -> ConversionPath | None:
        """Get conversion path between types if available."""
        return self._converter_registry.get_conversion_path(source_type, target_type)

    def clear_cache(self) -> None:
        """Clear compatibility cache."""
        self._compatibility_cache.clear()


# Global compatibility checker instance
_global_checker = CompatibilityChecker()


def check_compatibility(
    source_type: type[Any], target_type: type[Any]
) -> CompatibilityLevel:
    """Check compatibility between two types using global checker."""
    return _global_checker.check_compatibility(source_type, target_type)


@dataclass
class SchemaCompatibility:
    """Result of schema compatibility check."""

    level: CompatibilityLevel
    compatible_fields: dict[str, CompatibilityLevel] = field(default_factory=dict)
    incompatible_fields: dict[str, CompatibilityLevel] = field(default_factory=dict)
    missing_fields: set[str] = field(default_factory=set)
    extra_fields: set[str] = field(default_factory=set)
    conversion_paths: dict[str, ConversionPath] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def is_compatible(self) -> bool:
        """Check if schemas are compatible."""
        return self.level not in (CompatibilityLevel.INCOMPATIBLE,)

    @property
    def requires_conversion(self) -> bool:
        """Check if conversion is required."""
        return self.level in (
            CompatibilityLevel.CONVERTIBLE,
            CompatibilityLevel.COERCIBLE,
            CompatibilityLevel.PARTIAL,
        )
