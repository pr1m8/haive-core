# src/haive/core/mixins/timestamp.py

"""
Timestamp mixin for tracking creation and update times.

Uses Pydantic v2 patterns with model_validator and computed fields.
"""
from datetime import datetime, timezone
from typing import Any, Dict

from pydantic import BaseModel, PrivateAttr, computed_field, model_validator


class TimestampMixin(BaseModel):
    """
    Mixin that adds timestamp tracking to any Pydantic model.

    Provides creation time, last updated time, and utilities for time-based operations.
    Uses private attributes to avoid field name conflicts in Pydantic v2.
    """

    # Private attributes for internal timestamp tracking
    _created_at: datetime = PrivateAttr(default=None)
    _updated_at: datetime = PrivateAttr(default=None)
    _version: int = PrivateAttr(default=1)

    @model_validator(mode="after")
    def initialize_timestamps(self) -> "TimestampMixin":
        """Initialize timestamps after model validation."""
        now = datetime.now(timezone.utc)

        # Only set if not already set (allows for explicit initialization)
        if self._created_at is None:
            self._created_at = now
        if self._updated_at is None:
            self._updated_at = now

        return self

    @computed_field
    @property
    def created_at(self) -> datetime:
        """When this object was created."""
        if self._created_at is None:
            self._created_at = datetime.now(timezone.utc)
        return self._created_at

    @computed_field
    @property
    def updated_at(self) -> datetime:
        """When this object was last updated."""
        if self._updated_at is None:
            self._updated_at = datetime.now(timezone.utc)
        return self._updated_at

    @computed_field
    @property
    def version(self) -> int:
        """Version number of this object (increments on updates)."""
        return self._version

    @computed_field
    @property
    def age_seconds(self) -> float:
        """Age of this object in seconds."""
        return (datetime.now(timezone.utc) - self.created_at).total_seconds()

    @computed_field
    @property
    def time_since_update_seconds(self) -> float:
        """Seconds since last update."""
        return (datetime.now(timezone.utc) - self.updated_at).total_seconds()

    @computed_field
    @property
    def age_formatted(self) -> str:
        """Human-readable age of this object."""
        return self._format_duration(self.age_seconds)

    @computed_field
    @property
    def time_since_update_formatted(self) -> str:
        """Human-readable time since last update."""
        return self._format_duration(self.time_since_update_seconds)

    def touch(self) -> None:
        """Update the timestamp and increment version."""
        self._updated_at = datetime.now(timezone.utc)
        self._version += 1

    def reset_timestamps(self) -> None:
        """Reset all timestamps to current time and version to 1."""
        now = datetime.now(timezone.utc)
        self._created_at = now
        self._updated_at = now
        self._version = 1

    def set_creation_time(self, timestamp: datetime) -> None:
        """Set creation time explicitly (useful for deserialization)."""
        self._created_at = timestamp
        if self._updated_at is None or self._updated_at < timestamp:
            self._updated_at = timestamp

    def set_update_time(self, timestamp: datetime) -> None:
        """Set update time explicitly (useful for deserialization)."""
        self._updated_at = timestamp

    def timestamp_info(self) -> Dict[str, Any]:
        """Get comprehensive timestamp information."""
        return {
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "version": self.version,
            "age_seconds": self.age_seconds,
            "time_since_update_seconds": self.time_since_update_seconds,
            "age_formatted": self.age_formatted,
            "time_since_update_formatted": self.time_since_update_formatted,
        }

    def is_newer_than(self, other: "TimestampMixin") -> bool:
        """Check if this object is newer than another."""
        return self.created_at > other.created_at

    def was_updated_after(self, other: "TimestampMixin") -> bool:
        """Check if this object was updated after another."""
        return self.updated_at > other.updated_at

    def is_stale(self, max_age_seconds: float) -> bool:
        """Check if object is older than specified age."""
        return self.age_seconds > max_age_seconds

    def needs_update(self, max_update_age_seconds: float) -> bool:
        """Check if object hasn't been updated in specified time."""
        return self.time_since_update_seconds > max_update_age_seconds

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format duration in seconds to human-readable string."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        elif seconds < 86400:
            hours = seconds / 3600
            return f"{hours:.1f}h"
        else:
            days = seconds / 86400
            return f"{days:.1f}d"
