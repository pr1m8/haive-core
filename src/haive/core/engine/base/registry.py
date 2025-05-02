import logging
from typing import Dict, List, Optional

from haive.core.engine.base.base import Engine
from haive.core.engine.base.types import EngineType
from haive.core.registry.base import AbstractRegistry

logger = logging.getLogger(__name__)


class EngineRegistry(AbstractRegistry[Engine]):
    """Central registry for all engines in the system."""

    _instance = None

    @classmethod
    def get_instance(cls) -> "EngineRegistry":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        """Initialize the registry with empty dictionaries."""
        self.engines = {engine_type: {} for engine_type in EngineType}
        self.engine_ids = {}  # id -> engine mapping

    def register(self, item: Engine) -> Engine:
        """Register an engine."""
        self.engines[item.engine_type][item.name] = item
        self.engine_ids[item.id] = item
        logger.debug(
            f"Registered engine {item.name} (id: {item.id}) of type {item.engine_type}"
        )
        return item

    def get(self, item_type: EngineType, name: str) -> Optional[Engine]:
        """Get an engine by type and name."""
        return self.engines[item_type].get(name)

    def find_by_id(self, id: str) -> Optional[Engine]:
        """Find an engine by its unique ID."""
        return self.engine_ids.get(id)

    def find(self, name_or_id: str) -> Optional[Engine]:
        """Find an engine by name or ID across all engine types."""
        # Check ID first (faster lookup)
        if engine := self.engine_ids.get(name_or_id):
            return engine

        # Search through all engine types by name
        for engine_type in EngineType:
            if engine := self.get(engine_type, name_or_id):
                return engine

        return None

    def list(self, item_type: EngineType) -> List[str]:
        """List all engines of a type."""
        return list(self.engines[item_type].keys())

    def get_all(self, item_type: EngineType) -> Dict[str, Engine]:
        """Get all engines of a type."""
        return self.engines[item_type]

    def clear(self) -> None:
        """Clear the registry."""
        self.engines = {engine_type: {} for engine_type in EngineType}
        self.engine_ids = {}
