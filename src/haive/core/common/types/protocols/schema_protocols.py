"""Schema protocol definitions for type-safe schema interactions.

This module defines protocol interfaces for objects that work with Pydantic schemas and state
management in the Haive framework. These protocols enable type-safe schema composition and
validation across different components.

The protocols in this module support:
- Input/output schema awareness for structured data flow
- State schema protocols for stateful components
- Schema validation and composition patterns

Example:
    Create a schema-aware component::

        class MyProcessor(IOSchemaAware):
            input_schema = InputModel
            output_schema = OutputModel

See Also:
    haive.core.schema.state_schema: Core state schema implementations
    haive.core.schema.composer: Schema composition utilities
"""

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

# ============================================================================
# Group 1: Input and Output Schema Protocol
# ============================================================================


@runtime_checkable
class IOSchemaAware(Protocol):
    """Protocol for objects that have input/output schemas."""

    input_schema: type[BaseModel] | None
    output_schema: type[BaseModel] | None


# ============================================================================
# Group 2: State Schema Protocol
# ============================================================================


@runtime_checkable
class StateSchemaAware(Protocol):
    """Protocol for objects that have state schema."""

    state_schema: type[BaseModel] | None


# ============================================================================
# Group 3: Combined Schema Protocol (inherits from both)
# ============================================================================


@runtime_checkable
class FullSchemaAware(IOSchemaAware, StateSchemaAware, Protocol):
    """Protocol for objects that have all schema types."""

    # Inherits:
    # - input_schema: Optional[Type[BaseModel]]
    # - output_schema: Optional[Type[BaseModel]]
    # - state_schema: Optional[Type[BaseModel]]


# ============================================================================
# Group 4: Input and Output Fields Protocol
# ============================================================================


@runtime_checkable
class IOFieldAware(Protocol):
    """Protocol for objects that can provide input/output field definitions."""

    def get_input_fields(self) -> dict[str, Any]:
        """Get input field definitions."""
        ...

    def get_output_fields(self) -> dict[str, Any]:
        """Get output field definitions."""
        ...


# ============================================================================
# Group 5: Complete Protocol (combines schemas + fields)
# ============================================================================


@runtime_checkable
class CompleteSchemaAware(FullSchemaAware, IOFieldAware, Protocol):
    """Protocol for objects that have both schemas and field methods."""

    # Inherits from FullSchemaAware:
    # - input_schema: Optional[Type[BaseModel]]
    # - output_schema: Optional[Type[BaseModel]]
    # - state_schema: Optional[Type[BaseModel]]

    # Inherits from IOFieldAware:
    # - get_input_fields(self) -> Dict[str, Any]
    # - get_output_fields(self) -> Dict[str, Any]
