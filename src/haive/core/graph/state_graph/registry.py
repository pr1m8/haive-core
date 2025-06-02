from typing import Any, ClassVar, Dict, List, Optional

from haive.core.registry.base import AbstractRegistry

from ..models.graph_model import GraphModel


class GraphRegistry(AbstractRegistry[GraphModel]):
    """Registry for graph models."""

    _instance: ClassVar[Optional["GraphRegistry"]] = None

    @classmethod
    def get_instance(cls) -> "GraphRegistry":
        """Get the singleton instance of the registry."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        """Initialize the registry."""
        self.graphs: Dict[str, GraphModel] = {}
        self.graph_ids: Dict[str, GraphModel] = {}

    def register(self, item: GraphModel) -> GraphModel:
        """Register a graph in the registry."""
        self.graphs[item.name] = item
        self.graph_ids[item.id] = item
        return item

    def get(self, item_type: Any, name: str) -> Optional[GraphModel]:
        """Get a graph by name."""
        return self.graphs.get(name)

    def find_by_id(self, id: str) -> Optional[GraphModel]:
        """Find a graph by ID."""
        return self.graph_ids.get(id)

    def list(self, item_type: Any) -> List[str]:
        """List all graph names."""
        return list(self.graphs.keys())

    def get_all(self, item_type: Any) -> Dict[str, GraphModel]:
        """Get all graphs."""
        return self.graphs

    def clear(self) -> None:
        """Clear the registry."""
        self.graphs.clear()
        self.graph_ids.clear()
