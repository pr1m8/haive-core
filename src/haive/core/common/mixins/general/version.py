from typing import List

from pydantic import BaseModel, Field


class VersionMixin(BaseModel):
    """Mixin for version tracking."""

    version: str = Field(default="1.0.0", description="Version string")
    version_history: list[str] = Field(
        default_factory=list, description="History of version changes"
    )

    def bump_version(self, new_version: str) -> None:
        """Update version and track in history."""
        self.version_history.append(self.version)
        self.version = new_version

    def get_version_history(self) -> List[str]:
        """Get complete version history including current."""
        return self.version_history + [self.version]
