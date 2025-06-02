from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class StateMixin(BaseModel):
    """Mixin for state tracking with validation."""

    state: str = Field(default="active", description="Current state of the object")
    state_history: list[Dict[str, Any]] = Field(
        default_factory=list, description="History of state changes"
    )

    def change_state(self, new_state: str, reason: Optional[str] = None) -> None:
        """Change state and track the change."""
        old_state = self.state
        self.state = new_state

        change_record = {
            "from_state": old_state,
            "to_state": new_state,
            "timestamp": datetime.now(),
            "reason": reason,
        }
        self.state_history.append(change_record)

    def get_state_changes(self) -> list[Dict[str, Any]]:
        """Get history of all state changes."""
        return self.state_history.copy()

    def is_in_state(self, state: str) -> bool:
        """Check if currently in specified state."""
        return self.state == state
