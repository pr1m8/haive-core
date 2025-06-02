from typing import Any, Dict

from pydantic import BaseModel, Field


class MetadataMixin(BaseModel):
    """Mixin for basic metadata operations."""

    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary metadata"
    )

    def add_metadata(self, key: str, value: Any) -> None:
        """Add a metadata key-value pair."""
        self.metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value by key."""
        return self.metadata.get(key, default)

    def has_metadata(self, key: str) -> bool:
        """Check if metadata key exists."""
        return key in self.metadata

    def remove_metadata(self, key: str) -> Any:
        """Remove and return metadata value."""
        return self.metadata.pop(key, None)

    def update_metadata(self, updates: Dict[str, Any]) -> None:
        """Update multiple metadata fields."""
        self.metadata.update(updates)

    def clear_metadata(self) -> None:
        """Clear all metadata."""
        self.metadata.clear()
