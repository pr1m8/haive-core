# src/haive/core/mixins/metadata.py

"""
Metadata mixin for flexible metadata and tag storage.

Uses Pydantic v2 patterns with field_validator and computed fields.
"""

from typing import Any, Dict, List, Set

from pydantic import BaseModel, Field, computed_field, field_validator


class MetadataMixin(BaseModel):
    """
    Mixin that adds flexible metadata storage to any Pydantic model.

    Provides a standardized way to store additional information and tags.
    """

    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata for this object"
    )
    tags: List[str] = Field(
        default_factory=list, description="Tags for categorizing this object"
    )

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate metadata dictionary."""
        if not isinstance(v, dict):
            return {}

        # Ensure all keys are strings
        cleaned = {}
        for key, value in v.items():
            if isinstance(key, str) and key.strip():
                cleaned[key.strip()] = value

        return cleaned

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v: List[str]) -> List[str]:
        """Validate and clean tags list."""
        if not isinstance(v, list):
            return []

        # Clean tags: remove duplicates, empty strings, strip whitespace
        cleaned_tags = []
        seen = set()

        for tag in v:
            if isinstance(tag, str):
                cleaned_tag = tag.strip().lower()
                if cleaned_tag and cleaned_tag not in seen:
                    cleaned_tags.append(cleaned_tag)
                    seen.add(cleaned_tag)

        return cleaned_tags

    @computed_field
    @property
    def tag_count(self) -> int:
        """Number of tags on this object."""
        return len(self.tags)

    @computed_field
    @property
    def metadata_count(self) -> int:
        """Number of metadata entries on this object."""
        return len(self.metadata)

    @computed_field
    @property
    def has_metadata(self) -> bool:
        """Whether this object has any metadata."""
        return len(self.metadata) > 0

    @computed_field
    @property
    def has_tags(self) -> bool:
        """Whether this object has any tags."""
        return len(self.tags) > 0

    @computed_field
    @property
    def tag_set(self) -> Set[str]:
        """Set of tags for efficient lookup."""
        return set(self.tags)

    def add_metadata(self, key: str, value: Any) -> None:
        """Add a metadata entry."""
        if key and isinstance(key, str):
            self.metadata[key.strip()] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get a metadata value with optional default."""
        return self.metadata.get(key, default)

    def remove_metadata(self, key: str) -> Any:
        """Remove and return a metadata entry."""
        return self.metadata.pop(key, None)

    def update_metadata(self, updates: Dict[str, Any]) -> None:
        """Update multiple metadata entries at once."""
        if isinstance(updates, dict):
            for key, value in updates.items():
                if isinstance(key, str) and key.strip():
                    self.metadata[key.strip()] = value

    def clear_metadata(self) -> None:
        """Clear all metadata."""
        self.metadata.clear()

    def add_tag(self, tag: str) -> bool:
        """
        Add a tag if it doesn't already exist.

        Returns:
            True if tag was added, False if it already existed
        """
        if not isinstance(tag, str):
            return False

        cleaned_tag = tag.strip().lower()
        if cleaned_tag and cleaned_tag not in self.tag_set:
            self.tags.append(cleaned_tag)
            return True
        return False

    def add_tags(self, tags: List[str]) -> int:
        """
        Add multiple tags.

        Returns:
            Number of tags that were actually added (not duplicates)
        """
        added_count = 0
        for tag in tags:
            if self.add_tag(tag):
                added_count += 1
        return added_count

    def remove_tag(self, tag: str) -> bool:
        """
        Remove a tag.

        Returns:
            True if tag was found and removed, False otherwise
        """
        if not isinstance(tag, str):
            return False

        cleaned_tag = tag.strip().lower()
        try:
            self.tags.remove(cleaned_tag)
            return True
        except ValueError:
            return False

    def remove_tags(self, tags: List[str]) -> int:
        """
        Remove multiple tags.

        Returns:
            Number of tags that were actually removed
        """
        removed_count = 0
        for tag in tags:
            if self.remove_tag(tag):
                removed_count += 1
        return removed_count

    def has_tag(self, tag: str) -> bool:
        """Check if object has a specific tag."""
        if not isinstance(tag, str):
            return False
        return tag.strip().lower() in self.tag_set

    def has_any_tags(self, tags: List[str]) -> bool:
        """Check if object has any of the specified tags."""
        for tag in tags:
            if self.has_tag(tag):
                return True
        return False

    def has_all_tags(self, tags: List[str]) -> bool:
        """Check if object has all of the specified tags."""
        for tag in tags:
            if not self.has_tag(tag):
                return False
        return True

    def clear_tags(self) -> None:
        """Clear all tags."""
        self.tags.clear()

    def filter_metadata(self, prefix: str) -> Dict[str, Any]:
        """Get all metadata entries with keys starting with prefix."""
        if not isinstance(prefix, str):
            return {}
        return {k: v for k, v in self.metadata.items() if k.startswith(prefix)}

    def filter_metadata_by_type(self, value_type: type) -> Dict[str, Any]:
        """Get all metadata entries where values are of specified type."""
        return {k: v for k, v in self.metadata.items() if isinstance(v, value_type)}

    def search_metadata(
        self, search_term: str, case_sensitive: bool = False
    ) -> Dict[str, Any]:
        """
        Search metadata keys and string values for a term.

        Args:
            search_term: Term to search for
            case_sensitive: Whether search should be case sensitive

        Returns:
            Dictionary of matching metadata entries
        """
        if not isinstance(search_term, str):
            return {}

        if not case_sensitive:
            search_term = search_term.lower()

        matches = {}
        for key, value in self.metadata.items():
            # Check key
            check_key = key if case_sensitive else key.lower()
            if search_term in check_key:
                matches[key] = value
                continue

            # Check string values
            if isinstance(value, str):
                check_value = value if case_sensitive else value.lower()
                if search_term in check_value:
                    matches[key] = value

        return matches

    def metadata_summary(self) -> Dict[str, Any]:
        """Get summary information about metadata and tags."""
        return {
            "metadata_count": self.metadata_count,
            "metadata_keys": list(self.metadata.keys()),
            "tag_count": self.tag_count,
            "tags": self.tags.copy(),
            "has_metadata": self.has_metadata,
            "has_tags": self.has_tags,
        }
