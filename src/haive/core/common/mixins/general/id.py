import uuid

from pydantic import BaseModel, Field


class IDMixin(BaseModel):
    """Mixin for adding ID generation and management."""

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Unique identifier"
    )

    def regenerate_id(self) -> str:
        """Generate a new UUID and return it."""
        self.id = str(uuid.uuid4())
        return self.id

    @classmethod
    def with_id(cls, id_value: str, **kwargs):
        """Create instance with specific ID."""
        return cls(id=id_value, **kwargs)
