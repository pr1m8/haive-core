from typing import Any, Dict

from pydantic import BaseModel, Field

from haive.core.common.mixins.general.id import IDMixin
from haive.core.common.mixins.general.metadata import MetadataMixin
from haive.core.common.mixins.general.timestamp import TimestampMixin
from haive.core.common.mixins.general.version import VersionMixin


class SerializationMixin(BaseModel):
    """Mixin for enhanced serialization capabilities."""

    def to_dict(self, exclude_private: bool = True) -> Dict[str, Any]:
        """Convert to dictionary with options."""
        exclude_set = set()
        if exclude_private:
            # Exclude private attributes (those starting with _)
            exclude_set.update(
                field_name
                for field_name in self.model_fields
                if field_name.startswith("_")
            )

        return self.model_dump(exclude=exclude_set)

    def to_json(self, exclude_private: bool = True, **kwargs) -> str:
        """Convert to JSON string."""
        data = self.to_dict(exclude_private=exclude_private)
        import json

        return json.dumps(data, default=str, **kwargs)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create instance from dictionary."""
        return cls.model_validate(data)

    @classmethod
    def from_json(cls, json_str: str):
        """Create instance from JSON string."""
        import json

        data = json.loads(json_str)
        return cls.from_dict(data)
