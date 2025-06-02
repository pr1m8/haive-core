"""
Node component implementation for the Haive graph system.

This module defines the Node class which represents a node in a graph.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, Generic, List, Optional, TypeVar, Union

from langgraph.types import RetryPolicy
from pydantic import BaseModel, Field

from haive.core.graph.common.types import ConfigLike, NodeOutput, NodeType, StateLike

# Type variables for generic parameters
T = TypeVar("T", bound=StateLike)
C = TypeVar("C", bound=ConfigLike)
O = TypeVar("O", bound=NodeOutput)


class Node(BaseModel, Generic[T, C, O]):
    """
    Base node in a graph system.

    A node represents a processing unit in a graph that can receive a state,
    optionally a configuration, and produce an output.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    node_type: NodeType

    # Configuration
    input_mapping: Optional[Dict[str, str]] = None
    output_mapping: Optional[Dict[str, str]] = None
    command_goto: Optional[Union[str, List[str]]] = None
    retry_policy: Optional[RetryPolicy] = None

    # Metadata
    description: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)

    model_config = {"arbitrary_types_allowed": True}

    def process(self, state: T, config: Optional[C] = None) -> O:
        """
        Process state and return output.

        Args:
            state: Input state
            config: Optional configuration

        Returns:
            Process output

        Raises:
            NotImplementedError: Subclasses must implement this method
        """
        raise NotImplementedError("Subclasses must implement process method")

    @property
    def display_name(self) -> str:
        """
        Get a human-readable display name.

        Returns:
            Node display name (defaults to node name)
        """
        return self.name

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to serializable dictionary.

        Returns:
            Dictionary representation of the node
        """
        return {
            "id": self.id,
            "name": self.name,
            "node_type": self.node_type,
            "input_mapping": self.input_mapping,
            "output_mapping": self.output_mapping,
            "command_goto": self.command_goto,
            "description": self.description,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
