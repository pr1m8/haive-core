# src/haive/core/common/mixins.py

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, PrivateAttr


class TimestampMixin(BaseModel):
    """Mixin for adding timestamp tracking to models."""

    created_at: datetime = Field(
        default_factory=datetime.now, description="When this object was created"
    )
    updated_at: datetime = Field(
        default_factory=datetime.now, description="When this object was last updated"
    )

    def update_timestamp(self) -> None:
        """Update the updated_at timestamp to now."""
        self.updated_at = datetime.now()

    def age_in_seconds(self) -> float:
        """Get age of this object in seconds."""
        return (datetime.now() - self.created_at).total_seconds()

    def time_since_update(self) -> float:
        """Get time since last update in seconds."""
        return (datetime.now() - self.updated_at).total_seconds()
