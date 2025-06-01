# src/haive/core/mixins/identifier.py

"""
Identifier mixin for unique identification of objects.

Uses Pydantic v2 patterns with field_validator and computed fields.
"""

import uuid
from typing import Optional
from uuid import UUID

from pydantic import (
    BaseModel,
    Field,
    PrivateAttr,
    computed_field,
    field_validator,
    model_validator,
)


class IdentifierMixin(BaseModel):
    """
    Mixin that adds unique identification to any Pydantic model.

    Provides both UUID and human-readable name identification.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this object",
    )
    name: Optional[str] = Field(
        default=None, description="Human-readable name for this object"
    )

    # Private attribute for UUID object
    _uuid_obj: Optional[UUID] = PrivateAttr(default=None)

    @field_validator("id")
    @classmethod
    def validate_id(cls, v: str) -> str:
        """Ensure ID is a valid UUID string."""
        try:
            # Validate that it's a proper UUID
            UUID(v)
            return v
        except ValueError:
            # If not valid, generate a new UUID
            return str(uuid.uuid4())

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: Optional[str]) -> Optional[str]:
        """Validate and clean the name field."""
        if v is not None:
            # Strip whitespace and ensure non-empty
            cleaned = v.strip()
            return cleaned if cleaned else None
        return v

    @model_validator(mode="after")
    def initialize_uuid_obj(self) -> "IdentifierMixin":
        """Initialize UUID object after model validation."""
        try:
            self._uuid_obj = UUID(self.id)
        except ValueError:
            # If id is somehow not a valid UUID, regenerate
            self._uuid_obj = uuid.uuid4()
            self.id = str(self._uuid_obj)
        return self

    @computed_field
    @property
    def short_id(self) -> str:
        """Short version of the ID (first 8 characters)."""
        return self.id[:8]

    @computed_field
    @property
    def display_name(self) -> str:
        """Display name (uses name if available, otherwise short_id)."""
        return self.name or f"Object-{self.short_id}"

    @computed_field
    @property
    def uuid_obj(self) -> UUID:
        """UUID object representation of the ID."""
        if self._uuid_obj is None:
            self._uuid_obj = UUID(self.id)
        return self._uuid_obj

    @computed_field
    @property
    def has_custom_name(self) -> bool:
        """Whether this object has a custom name (not auto-generated)."""
        return self.name is not None and self.name.strip() != ""

    def regenerate_id(self) -> str:
        """Generate a new ID and return it."""
        self._uuid_obj = uuid.uuid4()
        self.id = str(self._uuid_obj)
        return self.id

    def set_name(self, name: str) -> None:
        """Set the name with validation."""
        if name and name.strip():
            self.name = name.strip()
        else:
            self.name = None

    def clear_name(self) -> None:
        """Clear the name."""
        self.name = None

    def matches_id(self, id_or_name: str) -> bool:
        """Check if this object matches the given ID or name."""
        if not id_or_name:
            return False

        # Check full ID
        if self.id == id_or_name:
            return True

        # Check short ID
        if self.short_id == id_or_name:
            return True

        # Check name (case-insensitive)
        if self.name and self.name.lower() == id_or_name.lower():
            return True

        return False

    def identifier_info(self) -> dict[str, str]:
        """Get comprehensive identifier information."""
        return {
            "id": self.id,
            "short_id": self.short_id,
            "name": self.name or "unnamed",
            "display_name": self.display_name,
            "has_custom_name": self.has_custom_name,
        }
