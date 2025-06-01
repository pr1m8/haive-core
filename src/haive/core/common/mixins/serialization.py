# src/haive/core/mixins/serialization.py

"""
Serialization mixin for enhanced JSON and dictionary conversion.

Uses Pydantic v2 patterns with model_validator, model_dump, and proper serialization handling.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Set, Type, TypeVar, Union
from uuid import UUID

from pydantic import BaseModel, ValidationError, field_serializer, model_validator

T = TypeVar("T", bound="SerializationMixin")


class SerializationMixin(BaseModel):
    """
    Mixin that adds enhanced serialization capabilities to any Pydantic model.

    Uses Pydantic v2 patterns for robust JSON serialization with proper handling
    of complex types, validation, and error recovery.
    """

    @model_validator(mode="after")
    def validate_serializable_state(self) -> "SerializationMixin":
        """Validate that the model state is serializable."""
        try:
            # Test serialization during validation to catch issues early
            self._test_serialization()
        except Exception as e:
            # Log warning but don't fail validation
            if hasattr(self, "logger"):
                self.logger.warning(f"Serialization test failed during validation: {e}")
        return self

    def _test_serialization(self) -> None:
        """Test that the model can be serialized (used during validation)."""
        try:
            # Quick serialization test
            self.model_dump(mode="json")
        except Exception as e:
            raise ValueError(f"Model contains non-serializable data: {e}")

    @field_serializer("*", when_used="json")
    def serialize_any_field(self, value: Any) -> Any:
        """
        Custom serializer for any field when serializing to JSON.

        Handles complex types that Pydantic might not handle by default.
        """
        return self._make_json_serializable(value)

    def model_dump_enhanced(
        self,
        *,
        mode: str = "python",
        include: Optional[Union[Set[str], Dict[str, Any]]] = None,
        exclude: Optional[Union[Set[str], Dict[str, Any]]] = None,
        by_alias: bool = False,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
        round_trip: bool = False,
        warnings: bool = True,
        include_private: bool = False,
        include_computed: bool = True,
        serialize_as_any: bool = False,
    ) -> Dict[str, Any]:
        """
        Enhanced model dump with additional options for Haive framework.

        Args:
            mode: Serialization mode ('python' or 'json')
            include: Fields to include
            exclude: Fields to exclude
            by_alias: Use field aliases
            exclude_unset: Exclude unset fields
            exclude_defaults: Exclude default values
            exclude_none: Exclude None values
            round_trip: Enable round-trip serialization
            warnings: Show warnings
            include_private: Include private attributes
            include_computed: Include computed fields
            serialize_as_any: Use Any serialization

        Returns:
            Dictionary representation of the model
        """
        # Get base model dump
        data = self.model_dump(
            mode=mode,
            include=include,
            exclude=exclude,
            by_alias=by_alias,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            exclude_none=exclude_none,
            round_trip=round_trip,
            warnings=warnings,
            serialize_as_any=serialize_as_any,
        )

        # Add private attributes if requested
        if include_private:
            private_data = self._get_private_attributes_dict()
            data.update(private_data)

        # Add computed fields if requested and not already included
        if include_computed and mode != "json":
            computed_data = self._get_computed_fields_dict()
            for key, value in computed_data.items():
                if key not in data:
                    data[key] = value

        # Ensure JSON serializable if in JSON mode
        if mode == "json":
            data = self._make_json_serializable(data)

        return data

    def to_dict(
        self,
        *,
        exclude_private: bool = True,
        exclude_none: bool = False,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        include_computed: bool = True,
        by_alias: bool = False,
        json_compatible: bool = False,
    ) -> Dict[str, Any]:
        """
        Convert to dictionary with comprehensive options.

        Args:
            exclude_private: Whether to exclude private attributes
            exclude_none: Whether to exclude None values
            exclude_unset: Whether to exclude unset values
            exclude_defaults: Whether to exclude default values
            include_computed: Whether to include computed fields
            by_alias: Whether to use field aliases
            json_compatible: Whether to ensure JSON compatibility

        Returns:
            Dictionary representation
        """
        return self.model_dump_enhanced(
            mode="json" if json_compatible else "python",
            exclude_none=exclude_none,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            by_alias=by_alias,
            include_private=not exclude_private,
            include_computed=include_computed,
        )

    def to_json(
        self,
        *,
        exclude_private: bool = True,
        exclude_none: bool = False,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        include_computed: bool = True,
        by_alias: bool = False,
        indent: Optional[int] = 2,
        ensure_ascii: bool = False,
        sort_keys: bool = True,
        validate_json: bool = True,
    ) -> str:
        """
        Convert to JSON string with comprehensive validation.

        Args:
            exclude_private: Whether to exclude private attributes
            exclude_none: Whether to exclude None values
            exclude_unset: Whether to exclude unset values
            exclude_defaults: Whether to exclude default values
            include_computed: Whether to include computed fields
            by_alias: Whether to use field aliases
            indent: JSON indentation level (None for compact)
            ensure_ascii: Whether to escape non-ASCII characters
            sort_keys: Whether to sort dictionary keys
            validate_json: Whether to validate the generated JSON

        Returns:
            JSON string representation

        Raises:
            ValueError: If JSON validation fails and validate_json=True
        """
        # Get data using enhanced model dump
        data = self.model_dump_enhanced(
            mode="json",
            exclude_none=exclude_none,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            by_alias=by_alias,
            include_private=not exclude_private,
            include_computed=include_computed,
        )

        # Generate JSON
        json_str = json.dumps(
            data,
            indent=indent,
            ensure_ascii=ensure_ascii,
            sort_keys=sort_keys,
            default=self._json_default_handler,
        )

        # Validate JSON if requested
        if validate_json:
            try:
                json.loads(json_str)
            except json.JSONDecodeError as e:
                raise ValueError(f"Generated invalid JSON: {e}")

        return json_str

    def to_json_compact(self, **kwargs) -> str:
        """Convert to compact JSON string (no indentation)."""
        return self.to_json(indent=None, **kwargs)

    def to_json_pretty(self, indent: int = 2, **kwargs) -> str:
        """Convert to pretty-printed JSON string."""
        return self.to_json(indent=indent, **kwargs)

    @classmethod
    def from_dict(
        cls: Type[T],
        data: Dict[str, Any],
        *,
        strict: bool = True,
        validate_assignment: bool = True,
        context: Optional[Dict[str, Any]] = None,
    ) -> T:
        """
        Create instance from dictionary with enhanced validation.

        Args:
            data: Dictionary data
            strict: Whether to use strict validation
            validate_assignment: Whether to validate assignments
            context: Additional validation context

        Returns:
            New instance of the class

        Raises:
            ValidationError: If validation fails and strict=True
        """
        try:
            return cls.model_validate(data, strict=strict, context=context)
        except ValidationError:
            if strict:
                raise

            # Try with relaxed validation
            try:
                return cls.model_validate(data, strict=False, context=context)
            except ValidationError:
                # Last resort: filter out problematic fields
                cleaned_data = cls._clean_dict_for_validation(data)
                return cls.model_validate(cleaned_data, strict=False, context=context)

    @classmethod
    def from_json(
        cls: Type[T],
        json_data: Union[str, bytes],
        *,
        strict: bool = True,
        validate_assignment: bool = True,
        context: Optional[Dict[str, Any]] = None,
    ) -> T:
        """
        Create instance from JSON string with enhanced validation.

        Args:
            json_data: JSON string or bytes
            strict: Whether to use strict validation
            validate_assignment: Whether to validate assignments
            context: Additional validation context

        Returns:
            New instance of the class

        Raises:
            ValidationError: If validation fails and strict=True
            json.JSONDecodeError: If JSON is invalid
        """
        # Parse JSON
        if isinstance(json_data, bytes):
            json_data = json_data.decode("utf-8")

        try:
            data = json.loads(json_data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")

        # Validate and create instance
        return cls.from_dict(
            data,
            strict=strict,
            validate_assignment=validate_assignment,
            context=context,
        )

    @classmethod
    def try_from_dict(cls: Type[T], data: Dict[str, Any]) -> Optional[T]:
        """
        Try to create instance from dictionary, return None on failure.

        Args:
            data: Dictionary data

        Returns:
            New instance or None if validation fails
        """
        try:
            return cls.from_dict(data, strict=False)
        except Exception:
            return None

    @classmethod
    def try_from_json(cls: Type[T], json_data: Union[str, bytes]) -> Optional[T]:
        """
        Try to create instance from JSON string, return None on failure.

        Args:
            json_data: JSON string or bytes

        Returns:
            New instance or None if parsing/validation fails
        """
        try:
            return cls.from_json(json_data, strict=False)
        except Exception:
            return None

    def clone(self: T, **updates) -> T:
        """
        Create a deep copy of this object with optional field updates.

        Args:
            **updates: Field updates to apply to the clone

        Returns:
            New instance with updates applied
        """
        # Get current data (excluding computed fields to avoid duplication)
        data = self.model_dump_enhanced(
            include_private=False, include_computed=False, exclude_none=True
        )

        # Apply updates
        data.update(updates)

        # Create new instance
        return self.__class__.model_validate(data)

    def update_from_dict(
        self,
        data: Dict[str, Any],
        *,
        strict: bool = False,
        validate_fields: bool = True,
    ) -> None:
        """
        Update this object from a dictionary with validation.

        Args:
            data: Dictionary with updates
            strict: Whether to use strict validation
            validate_fields: Whether to validate individual fields
        """
        # Get current data
        current_data = self.model_dump_enhanced(
            include_private=False, include_computed=False
        )

        # Merge updates
        current_data.update(data)

        # Validate the updated data
        updated = self.__class__.model_validate(current_data, strict=strict)

        # Update this instance's fields
        for field_name, field_value in updated.model_dump().items():
            if hasattr(self, field_name):
                setattr(self, field_name, field_value)

    def _get_private_attributes_dict(self) -> Dict[str, Any]:
        """Get private attributes as a serializable dictionary."""
        private_attrs = {}

        # Get private attributes from the model
        if hasattr(self, "__pydantic_private__"):
            private_data = self.__pydantic_private__

            for attr_name, attr_value in private_data.items():
                if attr_value is not None:
                    serializable_value = self._make_json_serializable(attr_value)
                    private_attrs[attr_name] = serializable_value

        return private_attrs

    def _get_computed_fields_dict(self) -> Dict[str, Any]:
        """Get computed field values as a dictionary."""
        computed_values = {}

        # Get computed fields from the model
        if hasattr(self.__class__, "model_computed_fields"):
            for field_name in self.__class__.model_computed_fields:
                try:
                    value = getattr(self, field_name)
                    serializable_value = self._make_json_serializable(value)
                    computed_values[field_name] = serializable_value
                except AttributeError:
                    continue

        return computed_values

    def _make_json_serializable(self, obj: Any) -> Any:
        """
        Recursively ensure an object is JSON serializable.

        Args:
            obj: Object to make serializable

        Returns:
            JSON-serializable version of the object
        """
        if obj is None:
            return None
        elif isinstance(obj, (str, int, float, bool)):
            return obj
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, UUID):
            return str(obj)
        elif isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, bytes):
            try:
                return obj.decode("utf-8")
            except UnicodeDecodeError:
                return obj.hex()
        elif isinstance(obj, (set, frozenset)):
            return sorted([self._make_json_serializable(item) for item in obj])
        elif isinstance(obj, tuple):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {str(k): self._make_json_serializable(v) for k, v in obj.items()}
        elif hasattr(obj, "model_dump"):
            # Pydantic model
            return obj.model_dump(mode="json")
        elif hasattr(obj, "__dict__"):
            # Regular object with __dict__
            return {
                k: self._make_json_serializable(v)
                for k, v in obj.__dict__.items()
                if not k.startswith("_")
            }
        elif hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes)):
            # Iterable (but not string/bytes)
            try:
                return [self._make_json_serializable(item) for item in obj]
            except Exception:
                return str(obj)
        else:
            # Fallback to string representation
            return str(obj)

    def _json_default_handler(self, obj: Any) -> Any:
        """Default handler for JSON serialization of non-standard types."""
        return self._make_json_serializable(obj)

    @classmethod
    def _clean_dict_for_validation(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean a dictionary for validation by removing problematic fields.

        Args:
            data: Dictionary to clean

        Returns:
            Cleaned dictionary
        """
        cleaned = {}

        # Get model fields for reference
        model_fields = getattr(cls, "model_fields", {})

        for key, value in data.items():
            # Skip private attributes
            if key.startswith("_"):
                continue

            # Skip computed fields (they'll be computed)
            if (
                hasattr(cls, "model_computed_fields")
                and key in cls.model_computed_fields
            ):
                continue

            # Only include known fields or be permissive
            if key in model_fields or not model_fields:
                cleaned[key] = value

        return cleaned

    def get_serialization_info(self) -> Dict[str, Any]:
        """Get comprehensive information about serialization capabilities."""
        info = {
            "class_name": self.__class__.__name__,
            "module": self.__class__.__module__,
            "model_fields_count": len(getattr(self, "model_fields", {})),
            "computed_fields_count": len(
                getattr(self.__class__, "model_computed_fields", {})
            ),
            "private_attrs_count": len(self._get_private_attributes_dict()),
            "is_json_serializable": False,
            "serialization_size_bytes": 0,
            "serialization_errors": [],
        }

        # Test JSON serialization
        try:
            json_str = self.to_json_compact()
            info["is_json_serializable"] = True
            info["serialization_size_bytes"] = len(json_str.encode("utf-8"))
        except Exception as e:
            info["serialization_errors"].append(str(e))

        # Test round-trip serialization
        try:
            json_str = self.to_json_compact()
            self.__class__.from_json(json_str)
            info["round_trip_successful"] = True
        except Exception as e:
            info["round_trip_successful"] = False
            info["serialization_errors"].append(f"Round-trip failed: {e}")

        return info

    def validate_serialization(self) -> Dict[str, Any]:
        """
        Validate that this object can be properly serialized and deserialized.

        Returns:
            Dictionary with validation results
        """
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "tests_passed": [],
            "serialization_info": {},
        }

        # Test 1: Basic model dump
        try:
            self.model_dump()
            results["tests_passed"].append("basic_model_dump")
        except Exception as e:
            results["valid"] = False
            results["errors"].append(f"Basic model dump failed: {e}")

        # Test 2: JSON model dump
        try:
            self.model_dump(mode="json")
            results["tests_passed"].append("json_model_dump")
        except Exception as e:
            results["valid"] = False
            results["errors"].append(f"JSON model dump failed: {e}")

        # Test 3: to_json conversion
        try:
            json_str = self.to_json()
            results["tests_passed"].append("to_json")
        except Exception as e:
            results["valid"] = False
            results["errors"].append(f"to_json failed: {e}")

        # Test 4: Round-trip serialization
        try:
            json_str = self.to_json()
            self.__class__.from_json(json_str)
            results["tests_passed"].append("round_trip_serialization")
        except Exception as e:
            results["warnings"].append(f"Round-trip serialization failed: {e}")

        # Get serialization info
        results["serialization_info"] = self.get_serialization_info()

        return results
