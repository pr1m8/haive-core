"""Type conversion system for schema compatibility."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type

from haive.core.schema.compatibility.types import ConversionContext, ConversionQuality


class TypeConverter(ABC):
    """Abstract base class for type converters."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this converter."""
        pass

    @property
    def priority(self) -> int:
        """Priority for converter selection (higher = preferred)."""
        return 0

    @abstractmethod
    def can_convert(self, source_type: Type[Any], target_type: Type[Any]) -> bool:
        """Check if this converter can handle the conversion."""
        pass

    @abstractmethod
    def convert(self, value: Any, context: ConversionContext) -> Any:
        """Convert a value from source to target type."""
        pass

    def estimate_quality(
        self, source_type: Type[Any], target_type: Type[Any]
    ) -> ConversionQuality:
        """Estimate the quality of conversion."""
        return ConversionQuality.SAFE


class ConverterRegistry:
    """Registry for type converters."""

    def __init__(self):
        self._converters: Dict[str, TypeConverter] = {}
        self._type_cache: Dict[Tuple[Type[Any], Type[Any]], Optional[TypeConverter]] = (
            {}
        )

    def register(self, converter: TypeConverter) -> None:
        """Register a type converter."""
        self._converters[converter.name] = converter
        # Clear cache when new converter is registered
        self._type_cache.clear()

    def unregister(self, name: str) -> None:
        """Unregister a type converter."""
        if name in self._converters:
            del self._converters[name]
            self._type_cache.clear()

    def get_converter(
        self, source_type: Type[Any], target_type: Type[Any]
    ) -> Optional[TypeConverter]:
        """Get the best converter for a type pair."""
        # Check cache first
        cache_key = (source_type, target_type)
        if cache_key in self._type_cache:
            return self._type_cache[cache_key]

        # Find compatible converters
        compatible_converters: List[TypeConverter] = []
        for converter in self._converters.values():
            if converter.can_convert(source_type, target_type):
                compatible_converters.append(converter)

        # Select best converter by priority
        if compatible_converters:
            best_converter = max(compatible_converters, key=lambda c: c.priority)
            self._type_cache[cache_key] = best_converter
            return best_converter

        self._type_cache[cache_key] = None
        return None

    def can_convert(self, source_type: Type[Any], target_type: Type[Any]) -> bool:
        """Check if conversion is possible."""
        return self.get_converter(source_type, target_type) is not None

    def convert(
        self, value: Any, source_type: Type[Any], target_type: Type[Any]
    ) -> Any:
        """Convert a value using the best available converter."""
        converter = self.get_converter(source_type, target_type)
        if not converter:
            raise ValueError(f"No converter found for {source_type} -> {target_type}")

        context = ConversionContext(
            source_type=str(source_type), target_type=str(target_type)
        )
        context.add_step(f"Using converter: {converter.name}")

        return converter.convert(value, context)

    def list_converters(self) -> List[str]:
        """List all registered converter names."""
        return list(self._converters.keys())


# Global converter registry
_global_registry = ConverterRegistry()


def register_converter(converter: TypeConverter) -> None:
    """Register a converter in the global registry."""
    _global_registry.register(converter)


def get_converter_registry() -> ConverterRegistry:
    """Get the global converter registry."""
    return _global_registry
