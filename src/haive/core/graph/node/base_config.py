# src/haive/core/graph/node/config.py
import logging
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from langgraph.graph import END
from pydantic import BaseModel, Field

from haive.core.graph.node.types import CommandGoto, NodeType

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class NodeConfig(ABC, BaseModel):
    """
    Base configuration for a node in a graph.

    This is a simplified base class that only contains core identification
    and shared fields. Specific node types should extend this class.
    """

    # Core identification
    id: str = Field(
        default_factory=lambda: f"node_{uuid.uuid4().hex[:8]}",
        description="Unique identifier for this node",
    )
    name: str = Field(description="Name of the node in the graph")
    node_type: NodeType = Field(description="Type of node")

    # Control flow
    command_goto: Optional[CommandGoto] = Field(
        default=END, description="Next node to go to after this node (or END)"
    )

    # Runtime configuration
    config_overrides: Dict[str, Any] = Field(
        default_factory=dict, description="Configuration overrides for this node"
    )

    # Metadata
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata for this node"
    )

    model_config = {"arbitrary_types_allowed": True}

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert this node config to a dictionary representation.
        """
        data = self.model_dump()

        # Convert CommandGoto.END to string representation
        if self.command_goto == END:
            data["command_goto"] = "END"

        return data

    @abstractmethod
    def __call__(
        self, state: Dict[str, Any], config: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Base implementation - subclasses should override this.
        """
        raise NotImplementedError(
            f"Node type {self.node_type} does not implement __call__"
        )
