"""
Base graph implementation for the Haive framework.

Provides a comprehensive system for building, manipulating, and executing
graphs with consistent interfaces, serialization support, and dynamic composition.
"""

import logging
import uuid
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START
from langgraph.types import Command, RetryPolicy, Send
from pydantic import BaseModel, Field, model_validator

# Import Branch implementation
from haive.core.graph.branches.branch import Branch
from haive.core.graph.branches.types import BranchMode, BranchResult, ComparisonType
from haive.core.graph.common.references import CallableReference
from haive.core.graph.common.types import ConfigLike, NodeOutput, NodeType, StateLike
from haive.core.graph.node.config import NodeConfig
from haive.core.graph.node.decorators import send_node
from haive.core.graph.state_graph.graph_path import GraphPath
from haive.core.schema.state_schema import StateSchema

# Setup logging
logger = logging.getLogger(__name__)


class BranchType(str, Enum):
    """Types of branches for conditional routing."""

    FUNCTION = "function"  # Uses a callable function
    KEY_VALUE = "key_value"  # Compares a state key with a value
    SEND = "send"  # Uses Send objects for routing
    COMMAND = "command"  # Uses Command objects for routing


class EdgeType(str, Enum):
    """Types of edges in a graph."""

    DIRECT = "direct"  # Simple direct connection
    CONDITIONAL = "conditional"  # Edge with condition
    DYNAMIC = "dynamic"  # Created dynamically at runtime


# Simple edge: (source_node_name, target_node_name)
SimpleEdge = Tuple[str, str]

# Complete edge type - now just simple edges
Edge = SimpleEdge


class Node(BaseModel, Generic[StateLike, ConfigLike, NodeOutput]):
    """Base node in a graph system."""

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

    def process(
        self, state: StateLike, config: Optional[ConfigLike] = None
    ) -> NodeOutput:
        """Process state and return output."""
        raise NotImplementedError("Subclasses must implement process method")

    @property
    def display_name(self) -> str:
        """Get a human-readable display name."""
        return self.name

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
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


class BaseGraph(BaseModel):
    """
    Base class for graph management in the Haive framework.

    Provides comprehensive graph management capabilities including:
    - Node management (add, remove, update)
    - Edge management (direct and branch-based)
    - Branch management
    - Graph validation
    - Serialization
    """

    # Unique identifier and metadata
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Core graph components - branches now handle conditional routing
    nodes: Dict[str, Optional[Union[Node, NodeConfig, Any]]] = Field(
        default_factory=dict
    )
    edges: List[Edge] = Field(default_factory=list)
    branches: Dict[str, Branch] = Field(default_factory=dict)

    # Configuration
    state_schema: Optional[Any] = None
    default_config: Optional[RunnableConfig] = None

    # Additional components for advanced functionality
    subgraphs: Dict[str, Any] = Field(default_factory=dict)
    node_types: Dict[str, NodeType] = Field(default_factory=dict)

    # Tracking fields
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def validate_graph(self) -> "BaseGraph":
        """Validate the graph structure."""
        # Filter out None values from nodes and check uniqueness
        non_none_nodes = {k: v for k, v in self.nodes.items() if v is not None}

        # Check that node names are unique for non-None nodes
        node_names = set()
        for name, node in non_none_nodes.items():
            if hasattr(node, "name"):
                node_names.add(node.name)
            else:
                # For nodes without a name property (like callables), use the key
                node_names.add(name)

        if len(node_names) != len(non_none_nodes):
            raise ValueError("Node names must be unique")

        # Initialize additional structures if not present
        if not hasattr(self, "subgraphs"):
            self.subgraphs = {}
        if not hasattr(self, "node_types"):
            self.node_types = {}

        # Track node types for non-None nodes
        for name, node in non_none_nodes.items():
            if name not in self.node_types:
                node_type = self._infer_node_type(node)
                self._track_node_type(name, node_type, getattr(node, "metadata", {}))

        # Validate edges
        for source, target in self.edges:
            if source != START and source not in self.nodes:
                raise ValueError(f"Edge source '{source}' not found in nodes")
            if target != END and target not in self.nodes:
                raise ValueError(f"Edge target '{target}' not found in nodes")

        # Validate branches
        for branch_id, branch in self.branches.items():
            if branch.source_node != START and branch.source_node not in self.nodes:
                raise ValueError(
                    f"Branch source '{branch.source_node}' not found in nodes"
                )

            # Validate targets in destinations
            for target in branch.destinations.values():
                if target != END and target not in self.nodes:
                    raise ValueError(f"Branch target '{target}' not found in nodes")

        return self

    def _infer_node_type(self, node: Any) -> NodeType:
        """
        Infer the node type from a node object.

        Args:
            node: Node object to infer type from

        Returns:
            Inferred NodeType
        """
        # Check if node has an explicit node_type attribute
        if hasattr(node, "node_type"):
            return node.node_type

        # Check for NodeConfig instances
        if hasattr(node, "__class__") and "NodeConfig" in node.__class__.__name__:
            # Different NodeConfig classes map to different node types
            if "EngineNodeConfig" in node.__class__.__name__:
                return NodeType.ENGINE
            elif "ToolNodeConfig" in node.__class__.__name__:
                return NodeType.TOOL
            elif "ValidationNodeConfig" in node.__class__.__name__:
                return NodeType.VALIDATION
            else:
                return NodeType.CALLABLE

        # Check for engine objects
        if hasattr(node, "engine_type") and hasattr(node, "create_runnable"):
            return NodeType.ENGINE

        # Check for subgraphs
        if isinstance(node, BaseGraph):
            return NodeType.SUBGRAPH

        # Check for callable objects
        if callable(node):
            return NodeType.CALLABLE

        # Default to callable for other objects
        return NodeType.CALLABLE

    def add_node(
        self,
        node_or_name: Union[Node, Dict[str, Any], str, NodeConfig],
        node_like: Optional[Any] = None,
        **kwargs,
    ) -> "BaseGraph":
        """
        Add a node to the graph with flexible input options.

        Args:
            node_or_name: Node object, dictionary, node name, or NodeConfig
            node_like: If node_or_name is a string, this is the node object, callable, or engine
            **kwargs: Additional properties when creating a node from name

        Returns:
            Self for method chaining
        """
        node_obj = None

        # Handle different input formats
        if isinstance(node_or_name, str):
            # String name with node_like object
            name = node_or_name

            # If node_like is None, store it directly (useful for pattern placeholders)
            if node_like is None and not kwargs:
                self.nodes[name] = None
                return self

            # Prepare node data
            node_data = {"name": name}

            # Determine node type and prepare data
            if node_like is None:
                # Use any provided kwargs
                node_data.update(kwargs)
                node_data.setdefault("node_type", NodeType.CALLABLE)
            elif (
                hasattr(node_like, "__class__")
                and "NodeConfig" in node_like.__class__.__name__
            ):
                # Handle NodeConfig objects - store directly
                self.nodes[name] = node_like

                # Track node type
                node_type = self._infer_node_type(node_like)
                self._track_node_type(
                    name, node_type, getattr(node_like, "metadata", {})
                )

                self.updated_at = datetime.now()
                return self
            elif isinstance(node_like, Node):
                # Use existing node with new name
                node_data = node_like.model_dump()
                node_data["name"] = name
            elif callable(node_like):
                # Callable function
                node_data.update(
                    {
                        "node_type": NodeType.CALLABLE,
                        "metadata": {"callable": node_like},
                        **kwargs,
                    }
                )
            elif hasattr(node_like, "engine_type") and hasattr(
                node_like, "create_runnable"
            ):
                # Looks like an Engine object
                node_data.update(
                    {
                        "node_type": NodeType.ENGINE,
                        "metadata": {"engine": node_like, "callable": node_like},
                        **kwargs,
                    }
                )
            elif isinstance(node_like, BaseGraph):
                # Subgraph
                node_data.update(
                    {
                        "node_type": NodeType.SUBGRAPH,
                        "metadata": {"subgraph": node_like.name},
                        **kwargs,
                    }
                )
                # Add to subgraphs
                self.subgraphs[name] = node_like
            else:
                # Generic object
                node_data.update(
                    {
                        "node_type": (
                            NodeType.CALLABLE
                            if callable(node_like)
                            else NodeType.ENGINE
                        ),
                        "metadata": {
                            "object": node_like,
                            "callable": node_like if callable(node_like) else None,
                        },
                        **kwargs,
                    }
                )

            # Create node
            node_obj = Node(**node_data)

        elif isinstance(node_or_name, dict):
            # Dictionary representation
            if "name" not in node_or_name:
                raise ValueError("Node dictionary must have a 'name' field")

            # Merge with kwargs
            node_data = {**node_or_name, **kwargs}
            node_obj = Node(**node_data)

        elif isinstance(node_or_name, Node):
            # Node object
            node_obj = node_or_name

            # Apply any kwargs as updates
            if kwargs:
                for key, value in kwargs.items():
                    if hasattr(node_obj, key):
                        setattr(node_obj, key, value)
        elif (
            hasattr(node_or_name, "__class__")
            and "NodeConfig" in node_or_name.__class__.__name__
        ):
            # Handle NodeConfig objects directly
            config = node_or_name
            name = getattr(config, "name", f"node_{uuid.uuid4().hex[:8]}")

            # Store directly
            self.nodes[name] = config

            # Track node type
            node_type = self._infer_node_type(config)
            self._track_node_type(name, node_type, getattr(config, "metadata", {}))

            logger.debug(f"Added NodeConfig '{name}' to graph '{self.name}'")
            self.updated_at = datetime.now()
            return self
        else:
            raise TypeError(f"Unsupported node type: {type(node_or_name)}")

        # Check for existing node
        if node_obj.name in self.nodes:
            raise ValueError(f"Node '{node_obj.name}' already exists in the graph")

        # Add the node
        self.nodes[node_obj.name] = node_obj

        # Track node type
        self._track_node_type(node_obj.name, node_obj.node_type, node_obj.metadata)

        logger.debug(f"Added node '{node_obj.name}' to graph '{self.name}'")

        self.updated_at = datetime.now()
        return self

    def _track_node_type(
        self,
        node_name: str,
        node_type: Optional[NodeType] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Track node type for a node.

        Args:
            node_name: Name of the node
            node_type: Type of node (if known)
            metadata: Node metadata (used to infer type if not provided)
        """
        # Determine type from node or metadata
        determined_type = node_type

        if not determined_type and metadata:
            if "engine" in metadata:
                determined_type = NodeType.ENGINE
            elif "tools" in metadata:
                determined_type = NodeType.TOOL
            elif "validation" in metadata:
                determined_type = NodeType.VALIDATION
            elif "subgraph" in metadata:
                determined_type = NodeType.SUBGRAPH
            elif "callable" in metadata:
                determined_type = NodeType.CALLABLE

        # Default to CALLABLE if still unknown
        if not determined_type:
            determined_type = NodeType.CALLABLE

        # Store the type
        self.node_types[node_name] = determined_type

    def add_tool_node(
        self, node_name: str, node_type: NodeType = NodeType.TOOL, **kwargs
    ) -> "BaseGraph":
        """
        Add a tool node to the graph.

        Args:
            node_name: Name of the node
            node_type: Type of node (defaults to TOOL)
            **kwargs: Additional properties for the node

        Returns:
            Self for method chaining
        """
        node_obj = Node(name=node_name, node_type=node_type, **kwargs)
        self.add_node(node_obj)
        return self

    def add_subgraph(self, name: str, subgraph: "BaseGraph", **kwargs) -> "BaseGraph":
        """
        Add a subgraph as a node.

        Args:
            name: Name for the subgraph node
            subgraph: Subgraph object to add
            **kwargs: Additional node properties

        Returns:
            Self for method chaining
        """
        if name in self.nodes:
            raise ValueError(f"Node '{name}' already exists in graph")

        # Store subgraph
        self.subgraphs[name] = subgraph

        # Create node
        node_obj = Node(
            name=name,
            node_type=NodeType.SUBGRAPH,
            metadata={"subgraph": subgraph.name},
            **kwargs,
        )

        # Add node
        self.nodes[name] = node_obj

        # Track node type
        self.node_types[name] = NodeType.SUBGRAPH

        logger.debug(f"Added subgraph '{subgraph.name}' as node '{name}'")
        self.updated_at = datetime.now()
        return self

    def remove_node(self, node_name: str) -> "BaseGraph":
        """
        Remove a node from the graph.

        Args:
            node_name: Name of the node to remove

        Returns:
            Self for method chaining
        """
        # Check node exists
        if node_name not in self.nodes:
            raise ValueError(f"Node '{node_name}' not found in graph")

        # Remove node
        del self.nodes[node_name]

        # Remove from node_types if tracked
        if node_name in self.node_types:
            del self.node_types[node_name]

        # Remove from subgraphs if it's a subgraph
        if node_name in self.subgraphs:
            del self.subgraphs[node_name]

        # Remove associated direct edges
        self.edges = [
            edge for edge in self.edges if edge[0] != node_name and edge[1] != node_name
        ]

        # Remove associated branches
        branch_ids_to_remove = []
        for branch_id, branch in self.branches.items():
            if branch.source_node == node_name:
                branch_ids_to_remove.append(branch_id)

        for branch_id in branch_ids_to_remove:
            del self.branches[branch_id]

        # Update any branch destinations that pointed to this node
        for branch in self.branches.values():
            for condition, target in list(branch.destinations.items()):
                if target == node_name:
                    # Remove or set to default
                    del branch.destinations[condition]

            # Update default if needed
            if branch.default == node_name:
                branch.default = END

        logger.debug(f"Removed node '{node_name}' from graph '{self.name}'")
        self.updated_at = datetime.now()
        return self

    def get_node(self, node_name: str) -> Optional[Any]:
        """
        Get a node by name.

        Args:
            node_name: Name of the node to retrieve

        Returns:
            Node object if found, None otherwise
        """
        return self.nodes.get(node_name)

    def update_node(self, node_name: str, **updates) -> "BaseGraph":
        """
        Update a node's properties.

        Args:
            node_name: Name of the node to update
            **updates: Properties to update

        Returns:
            Self for method chaining
        """
        if node_name not in self.nodes:
            raise ValueError(f"Node '{node_name}' not found in graph")

        # Get current node
        node = self.nodes[node_name]

        # Cannot update None nodes
        if node is None:
            raise ValueError(f"Cannot update None node '{node_name}'")

        # Apply updates
        for key, value in updates.items():
            if hasattr(node, key):
                setattr(node, key, value)

        # Update node type if changed
        if "node_type" in updates and node_name in self.node_types:
            self.node_types[node_name] = updates["node_type"]

        self.updated_at = datetime.now()
        return self

    def replace_node(
        self,
        node_name: str,
        new_node: Union[Node, Dict, Any],
        preserve_connections: bool = True,
    ) -> "BaseGraph":
        """
        Replace a node while optionally preserving its connections.

        Args:
            node_name: Name of the node to replace
            new_node: New node to insert
            preserve_connections: Whether to preserve existing connections

        Returns:
            Self for method chaining
        """
        if node_name not in self.nodes:
            raise ValueError(f"Node '{node_name}' not found in graph")

        # Store existing connections if needed
        incoming_edges = []
        outgoing_edges = []
        source_branches = []

        if preserve_connections:
            # Get direct edges
            for source, target in self.edges:
                if source == node_name:
                    outgoing_edges.append((source, target))
                elif target == node_name:
                    incoming_edges.append((source, target))

            # Get branches where this is the source
            source_branches = [
                branch
                for branch in self.branches.values()
                if branch.source_node == node_name
            ]

        # Remove the old node
        self.remove_node(node_name)

        # Add the new node
        # If it's a Node object, make sure it has the right name
        if isinstance(new_node, Node):
            if new_node.name != node_name:
                new_node_copy = new_node.model_copy(deep=True)
                new_node_copy.name = node_name
                self.add_node(new_node_copy)
            else:
                self.add_node(new_node)
        elif (
            hasattr(new_node, "__class__")
            and "NodeConfig" in new_node.__class__.__name__
        ):
            # If it's a NodeConfig, make sure it has the right name
            if hasattr(new_node, "name") and new_node.name != node_name:
                # Create a copy with the updated name
                if hasattr(new_node, "model_copy"):
                    new_node_copy = new_node.model_copy(deep=True)
                    new_node_copy.name = node_name
                    self.nodes[node_name] = new_node_copy
                else:
                    # Store with original name as fallback
                    self.nodes[node_name] = new_node
            else:
                self.nodes[node_name] = new_node
        elif callable(new_node):
            # Direct callable
            self.add_node(node_name, new_node)
            # Ensure the callable is in metadata
            if hasattr(self.nodes[node_name], "metadata"):
                self.nodes[node_name].metadata["callable"] = new_node
        else:
            # For other types, use the specified name
            self.add_node(node_name, new_node)

        # Restore connections if needed
        if preserve_connections:
            # Restore direct edges
            for source, target in incoming_edges:
                self.add_edge(source, node_name)

            for source, target in outgoing_edges:
                self.add_edge(node_name, target)

            # Restore branches
            for branch in source_branches:
                branch.source_node = node_name
                self.add_branch(branch)

        logger.debug(f"Replaced node '{node_name}' in graph '{self.name}'")
        return self

    def insert_node_after(
        self,
        target_node: str,
        new_node: Union[str, Node, Dict, Any],
        new_node_obj: Optional[Any] = None,
        **kwargs,
    ) -> "BaseGraph":
        """
        Insert a new node after an existing node, redirecting all outgoing connections.

        Args:
            target_node: Name of the existing node
            new_node: New node name, object, or dictionary
            new_node_obj: If new_node is a string, this is the node object/callable
            **kwargs: Additional properties if creating from name

        Returns:
            Self for method chaining
        """
        if target_node not in self.nodes and target_node != START:
            raise ValueError(f"Target node '{target_node}' not found in graph")

        # Get outgoing edges from target node
        outgoing_edges = []
        outgoing_branches = []

        for source, target in self.edges:
            if source == target_node:
                outgoing_edges.append((source, target))

        # Add the new node
        if isinstance(new_node, str):
            self.add_node(new_node, new_node_obj, **kwargs)
            new_node_name = new_node
        elif isinstance(new_node, Node):
            self.add_node(new_node)
            new_node_name = new_node.name
        elif isinstance(new_node, dict):
            if "name" not in new_node:
                raise ValueError("Node dictionary must have a 'name' field")
            self.add_node(new_node)
            new_node_name = new_node["name"]
        elif (
            hasattr(new_node, "__class__")
            and "NodeConfig" in new_node.__class__.__name__
        ):
            # NodeConfig case
            new_node_name = getattr(
                new_node, "name", f"{target_node}_after_{uuid.uuid4().hex[:6]}"
            )
            # If name not set in NodeConfig, set it
            if not hasattr(new_node, "name") or not getattr(new_node, "name"):
                if hasattr(new_node, "model_copy"):
                    node_copy = new_node.model_copy(deep=True)
                    node_copy.name = new_node_name
                    self.nodes[new_node_name] = node_copy
                else:
                    # Fall back to direct assignment
                    self.nodes[new_node_name] = new_node
            else:
                self.nodes[new_node_name] = new_node

            # Track node type
            node_type = self._infer_node_type(new_node)
            self._track_node_type(
                new_node_name, node_type, getattr(new_node, "metadata", {})
            )
        else:
            # Generate a name
            new_node_name = f"{target_node}_after_{uuid.uuid4().hex[:6]}"
            self.add_node(new_node_name, new_node, **kwargs)

        # Get any branches from the target node
        branches_to_update = []
        for branch_id, branch in self.branches.items():
            if branch.source_node == target_node:
                branches_to_update.append(branch)

        # Remove original outgoing edges
        for edge in outgoing_edges:
            self.remove_edge(edge[0], edge[1])

        # Add edge from target to new node
        self.add_edge(target_node, new_node_name)

        # Add edges from new node to original targets
        for _, target in outgoing_edges:
            self.add_edge(new_node_name, target)

        # Update branches - create new branches from the new node
        for branch in branches_to_update:
            # Remove old branch
            self.remove_branch(branch.id)

            # Create new branch from new node
            new_branch = branch.model_copy(deep=True)
            new_branch.id = str(uuid.uuid4())  # New ID for new branch
            new_branch.source_node = new_node_name

            # Add the new branch
            self.add_branch(new_branch)

        logger.debug(
            f"Inserted node '{new_node_name}' after '{target_node}' in graph '{self.name}'"
        )
        return self

    def insert_node_before(
        self,
        target_node: str,
        new_node: Union[str, Node, Dict, Any],
        new_node_obj: Optional[Any] = None,
        **kwargs,
    ) -> "BaseGraph":
        """
        Insert a new node before an existing node, redirecting all incoming connections.

        Args:
            target_node: Name of the existing node
            new_node: New node name, object, or dictionary
            new_node_obj: If new_node is a string, this is the node object/callable
            **kwargs: Additional properties if creating from name

        Returns:
            Self for method chaining
        """
        if target_node not in self.nodes:
            raise ValueError(f"Target node '{target_node}' not found in graph")

        # Get incoming direct edges to target node
        incoming_edges = []

        for source, target in self.edges:
            if target == target_node:
                incoming_edges.append((source, target))

        # Get incoming branch connections
        incoming_branches = []
        for branch_id, branch in self.branches.items():
            for condition, dest in branch.destinations.items():
                if dest == target_node:
                    incoming_branches.append((branch, condition))

            # Check default too
            if branch.default == target_node:
                incoming_branches.append((branch, "default"))

        # Add the new node
        if isinstance(new_node, str):
            self.add_node(new_node, new_node_obj, **kwargs)
            new_node_name = new_node
        elif isinstance(new_node, Node):
            self.add_node(new_node)
            new_node_name = new_node.name
        elif isinstance(new_node, dict):
            if "name" not in new_node:
                raise ValueError("Node dictionary must have a 'name' field")
            self.add_node(new_node)
            new_node_name = new_node["name"]
        elif (
            hasattr(new_node, "__class__")
            and "NodeConfig" in new_node.__class__.__name__
        ):
            # NodeConfig case
            new_node_name = getattr(
                new_node, "name", f"{target_node}_before_{uuid.uuid4().hex[:6]}"
            )
            # If name not set in NodeConfig, set it
            if not hasattr(new_node, "name") or not getattr(new_node, "name"):
                if hasattr(new_node, "model_copy"):
                    node_copy = new_node.model_copy(deep=True)
                    node_copy.name = new_node_name
                    self.nodes[new_node_name] = node_copy
                else:
                    # Fall back to direct assignment
                    self.nodes[new_node_name] = new_node
            else:
                self.nodes[new_node_name] = new_node

            # Track node type
            node_type = self._infer_node_type(new_node)
            self._track_node_type(
                new_node_name, node_type, getattr(new_node, "metadata", {})
            )
        else:
            # Generate a name
            new_node_name = f"{target_node}_before_{uuid.uuid4().hex[:6]}"
            self.add_node(new_node_name, new_node, **kwargs)

        # Remove original incoming edges
        for edge in incoming_edges:
            self.remove_edge(edge[0], edge[1])

        # Add edges from sources to new node
        for source, _ in incoming_edges:
            self.add_edge(source, new_node_name)

        # Add edge from new node to target
        self.add_edge(new_node_name, target_node)

        # Update branch destinations to point to the new node
        for branch, condition in incoming_branches:
            if condition == "default":
                branch.default = new_node_name
            else:
                branch.destinations[condition] = new_node_name

        logger.debug(
            f"Inserted node '{new_node_name}' before '{target_node}' in graph '{self.name}'"
        )
        return self

    # Advanced node operations
    def add_prelude_node(
        self,
        prelude_node: Union[str, Node, Dict, Any],
        node_obj: Optional[Any] = None,
        **kwargs,
    ) -> "BaseGraph":
        """
        Add a node at the beginning of the graph (after START).

        Args:
            prelude_node: Node to add at the start
            node_obj: If prelude_node is a string, this is the node object/callable
            **kwargs: Additional properties if creating from name

        Returns:
            Self for method chaining
        """
        # Get all nodes connected directly from START
        start_edges = [
            (source, target) for source, target in self.edges if source == START
        ]

        # Get all nodes connected from START via branches
        start_branches = [
            branch for branch in self.branches.values() if branch.source_node == START
        ]

        # Add the prelude node
        if isinstance(prelude_node, str):
            self.add_node(prelude_node, node_obj, **kwargs)
            prelude_name = prelude_node
        elif isinstance(prelude_node, Node):
            self.add_node(prelude_node)
            prelude_name = prelude_node.name
        elif isinstance(prelude_node, dict):
            if "name" not in prelude_node:
                raise ValueError("Node dictionary must have a 'name' field")
            self.add_node(prelude_node)
            prelude_name = prelude_node["name"]
        elif (
            hasattr(prelude_node, "__class__")
            and "NodeConfig" in prelude_node.__class__.__name__
        ):
            # NodeConfig case
            prelude_name = getattr(
                prelude_node, "name", f"prelude_{uuid.uuid4().hex[:6]}"
            )
            # Store the NodeConfig
            self.nodes[prelude_name] = prelude_node

            # Track node type
            node_type = self._infer_node_type(prelude_node)
            self._track_node_type(
                prelude_name, node_type, getattr(prelude_node, "metadata", {})
            )
        else:
            # Generate a name
            prelude_name = f"prelude_{uuid.uuid4().hex[:6]}"
            self.add_node(prelude_name, prelude_node, **kwargs)

        # Remove existing START direct edges
        for edge in start_edges:
            self.remove_edge(START, edge[1])

        # Add edge from START to prelude
        self.add_edge(START, prelude_name)

        # Add edges from prelude to original start nodes
        for _, target in start_edges:
            self.add_edge(prelude_name, target)

        # Handle branches from START
        for branch in start_branches:
            # Create new branch from prelude node with same destinations
            new_branch = branch.model_copy(deep=True)
            new_branch.id = str(uuid.uuid4())
            new_branch.source_node = prelude_name

            # Add the new branch
            self.add_branch(new_branch)

            # Remove the old branch
            self.remove_branch(branch.id)

        logger.debug(f"Added prelude node '{prelude_name}' to graph '{self.name}'")
        return self

    def add_postlude_node(
        self,
        postlude_node: Union[str, Node, Dict, Any],
        node_obj: Optional[Any] = None,
        **kwargs,
    ) -> "BaseGraph":
        """
        Add a node at the end of the graph (before END).

        Args:
            postlude_node: Node to add at the end
            node_obj: If postlude_node is a string, this is the node object/callable
            **kwargs: Additional properties if creating from name

        Returns:
            Self for method chaining
        """
        # Get all nodes connecting directly to END
        end_edges = [(source, target) for source, target in self.edges if target == END]

        # Get all branches pointing to END
        end_branch_destinations = []
        for branch in self.branches.values():
            for condition, target in branch.destinations.items():
                if target == END:
                    end_branch_destinations.append((branch, condition))

            # Check default too
            if branch.default == END:
                end_branch_destinations.append((branch, "default"))

        # Add the postlude node
        if isinstance(postlude_node, str):
            self.add_node(postlude_node, node_obj, **kwargs)
            postlude_name = postlude_node
        elif isinstance(postlude_node, Node):
            self.add_node(postlude_node)
            postlude_name = postlude_node.name
        elif isinstance(postlude_node, dict):
            if "name" not in postlude_node:
                raise ValueError("Node dictionary must have a 'name' field")
            self.add_node(postlude_node)
            postlude_name = postlude_node["name"]
        elif (
            hasattr(postlude_node, "__class__")
            and "NodeConfig" in postlude_node.__class__.__name__
        ):
            # NodeConfig case
            postlude_name = getattr(
                postlude_node, "name", f"postlude_{uuid.uuid4().hex[:6]}"
            )
            # Store the NodeConfig
            self.nodes[postlude_name] = postlude_node

            # Track node type
            node_type = self._infer_node_type(postlude_node)
            self._track_node_type(
                postlude_name, node_type, getattr(postlude_node, "metadata", {})
            )
        else:
            # Generate a name
            postlude_name = f"postlude_{uuid.uuid4().hex[:6]}"
            self.add_node(postlude_name, postlude_node, **kwargs)

        # Remove existing END edges
        for edge in end_edges:
            self.remove_edge(edge[0], END)

        # Add edges from original end nodes to postlude
        for source, _ in end_edges:
            self.add_edge(source, postlude_name)

        # Add edge from postlude to END
        self.add_edge(postlude_name, END)

        # Update branch destinations
        for branch, condition in end_branch_destinations:
            if condition == "default":
                branch.default = postlude_name
            else:
                branch.destinations[condition] = postlude_name

        logger.debug(f"Added postlude node '{postlude_name}' to graph '{self.name}'")
        return self

    def add_sequence(
        self,
        nodes: List[Union[str, Node, Dict, Any]],
        node_objects: Optional[List[Any]] = None,
        connect_start: bool = False,
        connect_end: bool = False,
        **kwargs,
    ) -> "BaseGraph":
        """
        Add a sequence of nodes and connect them in order.

        Args:
            nodes: List of nodes to add (names, objects, or dictionaries)
            node_objects: If nodes contains strings, these are the corresponding objects/callables
            connect_start: Whether to connect the first node to START
            connect_end: Whether to connect the last node to END
            **kwargs: Additional properties to apply to all nodes

        Returns:
            Self for method chaining
        """
        if not nodes:
            return self

        # Prepare node_objects if needed
        if node_objects and len(node_objects) < len(nodes):
            # Extend node_objects to match nodes length
            node_objects.extend([None] * (len(nodes) - len(node_objects)))
        elif not node_objects:
            node_objects = [None] * len(nodes)

        # Add nodes
        node_names = []
        for i, node in enumerate(nodes):
            node_obj = node_objects[i] if node_objects else None

            # Add the node
            if isinstance(node, str):
                self.add_node(node, node_obj, **kwargs)
                node_names.append(node)
            elif isinstance(node, Node):
                self.add_node(node, **kwargs)
                node_names.append(node.name)
            elif isinstance(node, dict):
                if "name" not in node:
                    node["name"] = f"seq_node_{i}_{uuid.uuid4().hex[:6]}"
                self.add_node(node, **kwargs)
                node_names.append(node["name"])
            elif hasattr(node, "__class__") and "NodeConfig" in node.__class__.__name__:
                # NodeConfig case
                node_name = getattr(
                    node, "name", f"seq_node_{i}_{uuid.uuid4().hex[:6]}"
                )
                # Store the NodeConfig
                self.nodes[node_name] = node

                # Track node type
                node_type = self._infer_node_type(node)
                self._track_node_type(
                    node_name, node_type, getattr(node, "metadata", {})
                )

                node_names.append(node_name)
            else:
                # Generate a name
                node_name = f"seq_node_{i}_{uuid.uuid4().hex[:6]}"
                self.add_node(node_name, node, **kwargs)
                node_names.append(node_name)

        # Connect START if requested
        if connect_start and node_names:
            self.add_edge(START, node_names[0])

        # Connect nodes in sequence
        for i in range(len(node_names) - 1):
            self.add_edge(node_names[i], node_names[i + 1])

        # Connect END if requested
        if connect_end and node_names:
            self.add_edge(node_names[-1], END)

        logger.debug(
            f"Added sequence of {len(node_names)} nodes to graph '{self.name}'"
        )
        return self

    def add_parallel_branches(
        self,
        source_node: str,
        branches: List[Union[List[str], List[Node], List[Dict], List[Any]]],
        branch_names: Optional[List[str]] = None,
        join_node: Optional[Union[str, Node, Dict, Any]] = None,
        join_node_obj: Optional[Any] = None,
        **kwargs,
    ) -> "BaseGraph":
        """
        Add parallel branches from a source node, optionally joining at a common node.

        Args:
            source_node: Name of the source node
            branches: List of node sequences (each sequence is a branch)
            branch_names: Optional names for the branches (used in branch creation)
            join_node: Optional node to join all branches
            join_node_obj: If join_node is a string, this is the node object/callable
            **kwargs: Additional properties for nodes

        Returns:
            Self for method chaining
        """
        if source_node not in self.nodes and source_node != START:
            raise ValueError(f"Source node '{source_node}' not found in graph")

        # Generate branch names if not provided
        if not branch_names:
            branch_names = [
                f"branch_{i}_{uuid.uuid4().hex[:6]}" for i in range(len(branches))
            ]
        elif len(branch_names) < len(branches):
            # Extend branch_names to match branches length
            branch_names.extend(
                [
                    f"branch_{i}_{uuid.uuid4().hex[:6]}"
                    for i in range(len(branch_names), len(branches))
                ]
            )

        # Track the end nodes of each branch
        branch_ends = []

        # Add each branch
        for i, branch_sequence in enumerate(branches):
            # Add the sequence
            self.add_sequence(branch_sequence, **kwargs)

            # Get first and last node names
            if branch_sequence:
                first_node = None
                last_node = None

                if isinstance(branch_sequence[0], str):
                    first_node = branch_sequence[0]
                elif isinstance(branch_sequence[0], Node):
                    first_node = branch_sequence[0].name
                elif (
                    isinstance(branch_sequence[0], dict)
                    and "name" in branch_sequence[0]
                ):
                    first_node = branch_sequence[0]["name"]
                elif (
                    hasattr(branch_sequence[0], "__class__")
                    and "NodeConfig" in branch_sequence[0].__class__.__name__
                ):
                    first_node = getattr(
                        branch_sequence[0], "name", f"branch_{i}_node_0"
                    )

                if isinstance(branch_sequence[-1], str):
                    last_node = branch_sequence[-1]
                elif isinstance(branch_sequence[-1], Node):
                    last_node = branch_sequence[-1].name
                elif (
                    isinstance(branch_sequence[-1], dict)
                    and "name" in branch_sequence[-1]
                ):
                    last_node = branch_sequence[-1]["name"]
                elif (
                    hasattr(branch_sequence[-1], "__class__")
                    and "NodeConfig" in branch_sequence[-1].__class__.__name__
                ):
                    last_node = getattr(
                        branch_sequence[-1], "name", f"branch_{i}_node_last"
                    )

                # Connect source to first node of this branch
                if first_node and first_node in self.nodes:
                    self.add_edge(source_node, first_node)

                # Track the last node
                if last_node and last_node in self.nodes:
                    branch_ends.append(last_node)

        # Add join node if provided
        if join_node is not None:
            # Add the join node
            if isinstance(join_node, str):
                if join_node not in self.nodes:
                    self.add_node(join_node, join_node_obj, **kwargs)
                join_name = join_node
            elif isinstance(join_node, Node):
                self.add_node(join_node, **kwargs)
                join_name = join_node.name
            elif isinstance(join_node, dict):
                if "name" not in join_node:
                    join_node["name"] = f"join_{uuid.uuid4().hex[:6]}"
                self.add_node(join_node, **kwargs)
                join_name = join_node["name"]
            elif (
                hasattr(join_node, "__class__")
                and "NodeConfig" in join_node.__class__.__name__
            ):
                # NodeConfig case
                join_name = getattr(join_node, "name", f"join_{uuid.uuid4().hex[:6]}")
                # Store the NodeConfig
                self.nodes[join_name] = join_node

                # Track node type
                node_type = self._infer_node_type(join_node)
                self._track_node_type(
                    join_name, node_type, getattr(join_node, "metadata", {})
                )
            else:
                # Generate a name
                join_name = f"join_{uuid.uuid4().hex[:6]}"
                self.add_node(join_name, join_node, **kwargs)

            # Connect all branch ends to the join node
            for end_node in branch_ends:
                self.add_edge(end_node, join_name)

        logger.debug(f"Added {len(branches)} parallel branches to graph '{self.name}'")
        return self

    # Edge management methods
    def add_edge(self, source: str, target: str) -> "BaseGraph":
        """
        Add a direct edge to the graph.

        Args:
            source: Source node name
            target: Target node name

        Returns:
            Self for method chaining
        """
        # Validate source and target
        if source != START and source not in self.nodes:
            raise ValueError(f"Source node '{source}' not found in graph")
        if target != END and target not in self.nodes:
            raise ValueError(f"Target node '{target}' not found in graph")

        # Create edge
        edge = (source, target)

        # Check if edge already exists
        if edge in self.edges:
            logger.warning(f"Edge {source} -> {target} already exists in graph")
            return self

        # Add edge
        self.edges.append(edge)
        logger.debug(f"Added edge {source} -> {target} to graph '{self.name}'")

        self.updated_at = datetime.now()
        return self

    def remove_edge(self, source: str, target: Optional[str] = None) -> "BaseGraph":
        """
        Remove an edge from the graph.

        Args:
            source: Source node name
            target: Target node name (if None, removes all edges from source)

        Returns:
            Self for method chaining
        """
        if target:
            # Remove specific direct edge
            self.edges = [
                edge
                for edge in self.edges
                if not (edge[0] == source and edge[1] == target)
            ]
            logger.debug(f"Removed edge {source} -> {target} from graph '{self.name}'")
        else:
            # Remove all edges from source
            self.edges = [edge for edge in self.edges if edge[0] != source]
            logger.debug(f"Removed all edges from {source} in graph '{self.name}'")

        self.updated_at = datetime.now()
        return self

    def get_edges(
        self,
        source: Optional[str] = None,
        target: Optional[str] = None,
        include_branches: bool = True,
    ) -> List[Tuple[str, str]]:
        """
        Get edges matching criteria.

        Args:
            source: Filter by source node
            target: Filter by target node
            include_branches: Include edges from branches

        Returns:
            List of matching edges as (source, target) tuples
        """
        result = []

        # Direct edges
        for edge_source, edge_target in self.edges:
            # Apply filters
            source_match = source is None or edge_source == source
            target_match = target is None or edge_target == target

            if source_match and target_match:
                result.append((edge_source, edge_target))

        # Branch-based edges
        if include_branches:
            for branch in self.branches.values():
                # Check source filter
                if source is not None and branch.source_node != source:
                    continue

                # Add all destinations that match target filter
                for dest in branch.destinations.values():
                    if target is None or dest == target:
                        result.append((branch.source_node, dest))

                # Check default destination
                if branch.default and (target is None or branch.default == target):
                    result.append((branch.source_node, branch.default))

        return result

    def find_all_paths(
        self,
        start_node=START,
        end_node=END,
        max_depth=100,
        include_loops=False,
        debug=False,
    ):
        """
        Find all possible paths between two nodes.

        Args:
            start_node: Starting node (defaults to START)
            end_node: Ending node (defaults to END)
            max_depth: Maximum path depth to prevent infinite loops
            include_loops: Whether to include paths with loops/cycles
            debug: Whether to show detailed debug logging

        Returns:
            List of GraphPath objects
        """
        # Track exploration statistics
        stats = {
            "nodes_visited": 0,
            "paths_found": 0,
            "max_depth_reached": 0,
            "branches_explored": 0,
        }

        if debug:
            logger.info(
                f"Finding paths from {start_node} to {end_node} (max_depth={max_depth}, include_loops={include_loops})"
            )

        paths = []

        def dfs(current, path_obj, visited=None, depth=0, parent_branch=None):
            """Depth-first search to find all paths."""
            # Initialize visited set for this path
            if visited is None:
                visited = set()

            # Update stats
            stats["nodes_visited"] += 1
            stats["max_depth_reached"] = max(stats["max_depth_reached"], depth)

            # Check depth limit
            if depth > max_depth:
                if debug:
                    logger.debug(f"Max depth reached at node {current} (depth={depth})")
                return

            # Display current exploration state
            if debug:
                prefix = "│   " * depth
                if parent_branch:
                    logger.debug(
                        f"{prefix}├── Exploring node: {current} (via {parent_branch})"
                    )
                else:
                    logger.debug(f"{prefix}├── Exploring node: {current}")
                logger.debug(f"{prefix}│   Path so far: {' → '.join(path_obj.nodes)}")
                logger.debug(f"{prefix}│   Depth: {depth}, Visited: {visited}")

            # Check if we reached the target
            target_found = current == end_node
            if target_found:
                # Create a new path with the end node and mark as reaching end
                new_path = path_obj.append(current, is_conditional=False, is_end=True)
                paths.append(new_path)
                stats["paths_found"] += 1

                if debug:
                    logger.debug(f"{prefix}│   ✓ Found path to target!")
                    logger.debug(f"{prefix}│   Path: {' → '.join(new_path.nodes)}")

                # Don't immediately return when include_loops is True and this isn't the END node
                # This allows us to find loops that pass through the target node
                if not include_loops or end_node == END:
                    if debug:
                        logger.debug(
                            f"{prefix}│   Stopping exploration of this branch (target found)"
                        )
                    return
                elif debug:
                    logger.debug(
                        f"{prefix}│   Continuing exploration past target to find loops"
                    )

            # Skip already visited nodes to prevent infinite loops (unless include_loops is True)
            if current in visited and not include_loops:
                if debug:
                    logger.debug(
                        f"{prefix}│   Already visited {current}, skipping (loops disabled)"
                    )
                return

            # Create a copy of visited to avoid modifying the original
            new_visited = visited.copy()
            new_visited.add(current)

            # Try direct edges first
            for src, dst in self.edges:
                if src == current:
                    # Skip if we've seen this node before (unless including loops)
                    if dst in visited and not include_loops:
                        if debug:
                            logger.debug(
                                f"{prefix}│   Skipping edge to {dst} (already visited)"
                            )
                        continue

                    if debug:
                        logger.debug(f"{prefix}│   → Following direct edge to {dst}")

                    # Create new path with this node
                    is_end = dst == end_node
                    new_path = path_obj.append(dst, is_conditional=False, is_end=is_end)

                    # Continue search from this node
                    dfs(
                        dst, new_path, new_visited, depth + 1, f"direct edge from {src}"
                    )

            # Try branches/conditional edges
            for branch_id, branch in self.branches.items():
                if branch.source_node == current:
                    stats["branches_explored"] += 1

                    if debug:
                        logger.debug(f"{prefix}│   Exploring branch: {branch.name}")

                    # Check all possible destinations
                    for condition, target in branch.destinations.items():
                        # Skip if we've seen this node before (unless including loops)
                        if target in visited and not include_loops:
                            if debug:
                                logger.debug(
                                    f"{prefix}│   Skipping branch to {target} (already visited)"
                                )
                            continue

                        if debug:
                            logger.debug(
                                f"{prefix}│   → Following branch ({condition}) to {target}"
                            )

                        # Create new path with this node
                        is_end = target == end_node
                        new_path = path_obj.append(
                            target, is_conditional=True, is_end=is_end
                        )

                        # Continue search from this node
                        dfs(
                            target,
                            new_path,
                            new_visited,
                            depth + 1,
                            f"branch condition '{condition}'",
                        )

                    # Check default destination too
                    if (
                        branch.default and branch.default != END
                    ):  # Skip checking END for now
                        # Skip if we've seen this node before (unless including loops)
                        if branch.default in visited and not include_loops:
                            if debug:
                                logger.debug(
                                    f"{prefix}│   Skipping default branch to {branch.default} (already visited)"
                                )
                            continue

                        if debug:
                            logger.debug(
                                f"{prefix}│   → Following default branch to {branch.default}"
                            )

                        # Create new path with this node
                        is_end = branch.default == end_node
                        new_path = path_obj.append(
                            branch.default, is_conditional=True, is_end=is_end
                        )

                        # Continue search from this node
                        dfs(
                            branch.default,
                            new_path,
                            new_visited,
                            depth + 1,
                            "default branch",
                        )

                    # Special handling for END default
                    if branch.default == END:
                        if debug:
                            logger.debug(
                                f"{prefix}│   → Following default branch to END"
                            )

                        new_path = path_obj.append(
                            END, is_conditional=True, is_end=True
                        )
                        paths.append(new_path)
                        stats["paths_found"] += 1

        # Start DFS from the start node with initial path
        initial_path = GraphPath(nodes=[start_node])
        dfs(start_node, initial_path)

        # Filter the final results
        result_paths = []

        # For non-END targets, include paths that contain the target node
        if end_node != END:
            for path in paths:
                if debug:
                    logger.debug(f"Checking path: {' → '.join(path.nodes)}")

                # Check if path reaches end_node (might be in the middle of the path)
                # Add the path if the target node is in the path
                if end_node in path.nodes:
                    result_paths.append(path)
                    if debug:
                        logger.debug(f"✓ Path contains target {end_node}")

                    # For looping paths, check if the target appears more than once
                    if path.nodes.count(end_node) > 1 and debug:
                        logger.debug(
                            f"  This path contains multiple occurrences of {end_node} (loop)"
                        )
                elif debug:
                    logger.debug(f"✗ Path does not contain target {end_node}")
        else:
            # For END node, only include paths marked as reaching end
            result_paths = [path for path in paths if path.reaches_end]

        if debug:
            logger.info(
                f"Found {len(result_paths)} paths from {start_node} to {end_node}"
            )

        return result_paths

    def check_graph_validity(self):
        """
        Validate the graph structure.

        Returns:
            List of validation issues (empty if graph is valid)
        """
        issues = []

        # Check for connectivity from START
        if not self.get_edges(source=START):
            issues.append("No connections from START node")

        # Check for connectivity to END
        if not self.get_edges(target=END):
            issues.append("No connections to END node")

        # Check for orphaned nodes
        orphaned = self.find_unreachable_nodes()
        if orphaned:
            issues.append(f"Orphaned nodes: {orphaned}")

        # Check for nodes without path to END
        no_end_path = self.find_nodes_without_end_path()
        if no_end_path:
            issues.append(f"Nodes without path to END: {no_end_path}")

        return issues

    def find_unreachable_nodes(self):
        """
        Find nodes that can't be reached from START.

        Returns:
            List of unreachable node names
        """
        # Get all nodes reachable from START
        reachable = set()

        def dfs(node):
            if node in reachable:
                return

            reachable.add(node)

            # Follow direct edges
            for src, dst in self.get_edges(source=node):
                dfs(dst)

            # Follow conditional edges
            for branch in self.branches.values():
                if branch.source_node == node:
                    for dst in branch.destinations.values():
                        dfs(dst)

                    if branch.default:
                        dfs(branch.default)

        # Start from START node
        dfs(START)

        # Return unreachable nodes (excluding special nodes)
        return [
            node
            for node in self.nodes
            if node not in reachable and self.nodes[node] is not None
        ]

    def find_nodes_without_end_path(self):
        """
        Find nodes that can't reach END.

        Returns:
            List of node names that can't reach END
        """
        # For each node, check if there's a path to END
        no_end_path = []

        for node_name in self.nodes:
            # Skip None nodes and nodes that are already unreachable
            if self.nodes[node_name] is None:
                continue

            if not self.has_path(node_name, END):
                no_end_path.append(node_name)

        return no_end_path

    def get_source_nodes(self):
        """
        Get nodes that have no incoming edges (other than START).

        Returns:
            List of source node names
        """
        # Get all nodes with incoming edges
        has_incoming = set()

        for _, dst in self.edges:
            has_incoming.add(dst)

        for branch in self.branches.values():
            for dst in branch.destinations.values():
                has_incoming.add(dst)

            if branch.default:
                has_incoming.add(branch.default)

        # Return nodes without incoming edges (except START)
        return [
            node
            for node in self.nodes
            if node not in has_incoming
            and node != START
            and self.nodes[node] is not None
        ]

    def get_sink_nodes(self):
        """
        Get nodes that have no outgoing edges (other than to END).

        Returns:
            List of sink node names
        """
        # Get all nodes with outgoing edges
        has_outgoing = set()

        for src, dst in self.edges:
            if dst != END:
                has_outgoing.add(src)

        for branch in self.branches.values():
            has_outgoing.add(branch.source_node)

        # Return nodes without outgoing edges (except to END)
        return [
            node
            for node in self.nodes
            if node not in has_outgoing and self.nodes[node] is not None
        ]

    def add_branch(
        self,
        branch_or_name: Union[Branch, str],
        source_node: Optional[str] = None,
        condition: Optional[Any] = None,
        routes: Optional[Dict[Union[bool, str], str]] = None,
        branch_type: Optional[BranchType] = None,
        **kwargs,
    ) -> "BaseGraph":
        """
        Add a branch to the graph with flexible input options.

        Args:
            branch_or_name: Branch object or branch name
            source_node: Source node for the branch (required if branch_or_name is a string)
            condition: Condition function or key/value for evaluation
            routes: Mapping of condition results to target nodes
            branch_type: Type of branch (determined automatically if not provided)
            **kwargs: Additional parameters for branch creation

        Returns:
            Self for method chaining
        """
        if isinstance(branch_or_name, Branch):
            # Branch object
            branch = branch_or_name

            # Validate source node
            if branch.source_node is None:
                if source_node is None:
                    raise ValueError("Branch must have a source_node")
                branch.source_node = source_node

            if branch.source_node != START and branch.source_node not in self.nodes:
                raise ValueError(
                    f"Source node '{branch.source_node}' not found in graph"
                )

            # Validate destination nodes
            for dest in branch.destinations.values():
                if dest != END and dest not in self.nodes:
                    raise ValueError(f"Destination node '{dest}' not found in graph")

            # Validate default node
            if branch.default != END and branch.default not in self.nodes:
                raise ValueError(f"Default node '{branch.default}' not found in graph")

            # Add the branch
            self.branches[branch.id] = branch

        else:
            # String name
            branch_name = branch_or_name

            # Ensure source node is provided
            if not source_node:
                raise ValueError("source_node is required when adding a branch by name")

            # Validate source node
            if source_node != START and source_node not in self.nodes:
                raise ValueError(f"Source node '{source_node}' not found in graph")

            # Create branch data
            branch_data = {"name": branch_name, "source_node": source_node, **kwargs}

            # Handle condition based on type
            if callable(condition):
                branch_data["function"] = condition
                branch_data["function_ref"] = CallableReference.from_callable(condition)
                branch_data["mode"] = BranchMode.FUNCTION
            elif isinstance(condition, tuple) and len(condition) >= 2:
                # Key-value branch: (key, value, [comparison])
                branch_data["key"] = condition[0]
                branch_data["value"] = condition[1]
                if len(condition) >= 3:
                    branch_data["comparison"] = condition[2]
                else:
                    branch_data["comparison"] = ComparisonType.EQUALS
            elif condition:
                # Direct condition value
                branch_data["key"] = kwargs.get("key")
                branch_data["value"] = condition
                branch_data["comparison"] = kwargs.get(
                    "comparison", ComparisonType.EQUALS
                )

            # Set routes
            if routes:
                branch_data["destinations"] = routes

            # Validate routes
            if "destinations" in branch_data:
                for dest in branch_data["destinations"].values():
                    if dest != END and dest not in self.nodes:
                        raise ValueError(
                            f"Destination node '{dest}' not found in graph"
                        )

            # Create branch
            branch = Branch(**branch_data)
            self.branches[branch.id] = branch

        logger.debug(
            f"Added branch '{branch.name}' from node '{branch.source_node}' to graph '{self.name}'"
        )
        self.updated_at = datetime.now()
        return self

    # Compatibility method for tests
    def add_conditional_edges(
        self, source_node, condition, destinations=None, default="END"
    ):
        """
        Add conditional edges from a source node based on a condition.

        Args:
            source_node: Source node name
            condition: Function, Branch object, or condition value
            destinations: Mapping of condition results to target nodes (optional if condition is a Branch)
            default: Default destination (optional if condition is a Branch)

        Returns:
            Self for method chaining
        """
        # Validate source node
        if source_node != START and source_node not in self.nodes:
            raise ValueError(f"Source node '{source_node}' not found in graph")

        # Handle Branch objects
        if isinstance(condition, Branch):
            branch = condition

            # Set source node on the branch
            branch.source_node = source_node

            # Use branch's destinations if none provided
            if destinations is None:
                destinations = branch.destinations
            else:
                # Update branch with provided destinations
                branch.destinations = destinations

            # Use provided default or keep branch's default
            if default != "END" or branch.default is None:
                branch.default = default

            # Add the branch to the graph
            branch_id = branch.id
            self.branches[branch_id] = branch

            logger.debug(
                f"Added conditional edges from {source_node} with {len(destinations)} destinations"
            )
            self.updated_at = datetime.now()
            return self

        # Handle function or value conditions (create a new Branch)
        branch_id = str(uuid.uuid4())
        branch_name = f"branch_{branch_id[:8]}"

        # Determine branch mode and parameters
        if callable(condition):
            # Function-based branch
            branch = Branch(
                id=branch_id,
                name=branch_name,
                source_node=source_node,
                function=condition,
                destinations=destinations or {},
                default=default,
                mode=BranchMode.FUNCTION,
            )
        else:
            # Value-based branch
            branch = Branch(
                id=branch_id,
                name=branch_name,
                source_node=source_node,
                value=condition,
                destinations=destinations or {},
                default=default,
                mode=BranchMode.DIRECT,
            )

        # Add the branch to the graph
        self.branches[branch_id] = branch

        logger.debug(
            f"Added conditional edges from {source_node} with {len(destinations or {})} destinations"
        )
        self.updated_at = datetime.now()
        return self

    @property
    def conditional_edges(self):
        """
        Property for accessing branches as conditional edges (compatibility).

        Returns:
            Dictionary of branches indexed by ID
        """
        return self.branches

    def add_function_branch(
        self,
        source_node: str,
        condition: Callable[[Any], Any],
        routes: Dict[Union[bool, str], str],
        default_route: str = END,
        name: Optional[str] = None,
    ) -> "BaseGraph":
        """
        Add a function-based branch.

        Args:
            source_node: Source node name
            condition: Condition function
            routes: Mapping of condition results to target nodes
            default_route: Default destination
            name: Optional branch name

        Returns:
            Self for method chaining
        """
        # Validate source node
        if source_node != START and source_node not in self.nodes:
            raise ValueError(f"Source node '{source_node}' not found in graph")

        # Validate target nodes
        for dest in routes.values():
            if dest != END and dest not in self.nodes:
                raise ValueError(f"Destination node '{dest}' not found in graph")

        # Validate default route
        if default_route != END and default_route not in self.nodes:
            raise ValueError(f"Default node '{default_route}' not found in graph")

        # Create branch
        branch = Branch(
            id=str(uuid.uuid4()),
            name=name or f"func_branch_{uuid.uuid4().hex[:6]}",
            source_node=source_node,
            function=condition,
            function_ref=CallableReference.from_callable(condition),
            mode=BranchMode.FUNCTION,
            destinations=routes,
            default=default_route,
        )

        # Add to graph
        self.branches[branch.id] = branch

        logger.debug(
            f"Added function branch '{branch.name}' from {source_node} to graph '{self.name}'"
        )
        return self

    def add_key_value_branch(
        self,
        source_node: str,
        key: str,
        value: Any,
        comparison: Union[ComparisonType, str] = ComparisonType.EQUALS,
        true_dest: str = "continue",
        false_dest: str = END,
        name: Optional[str] = None,
    ) -> "BaseGraph":
        """
        Add a key-value comparison branch.

        Args:
            source_node: Source node name
            key: State key to check
            value: Value to compare against
            comparison: Type of comparison
            true_dest: Destination if true
            false_dest: Destination if false
            name: Optional branch name

        Returns:
            Self for method chaining
        """
        # Validate source node
        if source_node != START and source_node not in self.nodes:
            raise ValueError(f"Source node '{source_node}' not found in graph")

        # Validate target nodes
        if true_dest != END and true_dest not in self.nodes:
            raise ValueError(f"True destination node '{true_dest}' not found in graph")

        if false_dest != END and false_dest not in self.nodes:
            raise ValueError(
                f"False destination node '{false_dest}' not found in graph"
            )

        # Create branch
        branch = Branch(
            id=str(uuid.uuid4()),
            name=name or f"kv_branch_{uuid.uuid4().hex[:6]}",
            source_node=source_node,
            key=key,
            value=value,
            comparison=comparison,
            destinations={True: true_dest, False: false_dest},
            default=false_dest,
            mode=BranchMode.DIRECT,
        )

        # Add to graph
        self.branches[branch.id] = branch

        logger.debug(
            f"Added key-value branch '{branch.name}' from {source_node} to graph '{self.name}'"
        )
        return self

    def remove_branch(self, branch_id: str) -> "BaseGraph":
        """
        Remove a branch from the graph.

        Args:
            branch_id: ID of the branch to remove

        Returns:
            Self for method chaining
        """
        if branch_id not in self.branches:
            raise ValueError(f"Branch '{branch_id}' not found in graph")

        # Remove branch
        del self.branches[branch_id]

        logger.debug(f"Removed branch '{branch_id}' from graph '{self.name}'")
        self.updated_at = datetime.now()
        return self

    def update_branch(self, branch_id: str, **updates) -> "BaseGraph":
        """
        Update a branch's properties.

        Args:
            branch_id: ID of the branch to update
            **updates: Properties to update

        Returns:
            Self for method chaining
        """
        if branch_id not in self.branches:
            raise ValueError(f"Branch '{branch_id}' not found in graph")

        # Get current branch
        branch = self.branches[branch_id]

        # Apply updates
        for key, value in updates.items():
            if hasattr(branch, key):
                setattr(branch, key, value)

        # Validate connections if source/destinations were updated
        if "source_node" in updates:
            source = updates["source_node"]
            if source != START and source not in self.nodes:
                raise ValueError(f"Updated source node '{source}' not found in graph")

        if "destinations" in updates:
            for dest in updates["destinations"].values():
                if dest != END and dest not in self.nodes:
                    raise ValueError(
                        f"Updated destination node '{dest}' not found in graph"
                    )

        if "default" in updates:
            default = updates["default"]
            if default != END and default not in self.nodes:
                raise ValueError(f"Updated default node '{default}' not found in graph")

        self.updated_at = datetime.now()
        return self

    def replace_branch(self, branch_id: str, new_branch: Branch) -> "BaseGraph":
        """
        Replace a branch with a new one.

        Args:
            branch_id: ID of the branch to replace
            new_branch: New branch to insert

        Returns:
            Self for method chaining
        """
        if branch_id not in self.branches:
            raise ValueError(f"Branch '{branch_id}' not found in graph")

        # Validate new branch
        if new_branch.source_node != START and new_branch.source_node not in self.nodes:
            raise ValueError(
                f"Source node '{new_branch.source_node}' not found in graph"
            )

        for dest in new_branch.destinations.values():
            if dest != END and dest not in self.nodes:
                raise ValueError(f"Destination node '{dest}' not found in graph")

        if new_branch.default != END and new_branch.default not in self.nodes:
            raise ValueError(f"Default node '{new_branch.default}' not found in graph")

        # Update the branch ID to match if needed
        if new_branch.id != branch_id:
            new_branch_copy = new_branch.model_copy(deep=True)
            new_branch_copy.id = branch_id
            self.branches[branch_id] = new_branch_copy
        else:
            self.branches[branch_id] = new_branch

        logger.debug(f"Replaced branch '{branch_id}' in graph '{self.name}'")
        self.updated_at = datetime.now()
        return self

    def get_branches_for_node(self, node_name: str) -> List[Branch]:
        """
        Get all branches with a given source node.

        Args:
            node_name: Name of the source node

        Returns:
            List of branch objects
        """
        return [
            branch
            for branch in self.branches.values()
            if branch.source_node == node_name
        ]

    def get_branch(self, branch_id: str) -> Optional[Branch]:
        """
        Get a branch by ID.

        Args:
            branch_id: ID of the branch to retrieve

        Returns:
            Branch object if found, None otherwise
        """
        return self.branches.get(branch_id)

    def get_branch_by_name(self, name: str) -> Optional[Branch]:
        """
        Get a branch by name.

        Args:
            name: Name of the branch to retrieve

        Returns:
            First matching branch or None if not found
        """
        for branch in self.branches.values():
            if branch.name == name:
                return branch
        return None

    # Graph traversal and analysis
    def has_path(self, source: str, target: str) -> bool:
        """
        Check if there is a path between source and target nodes.

        Args:
            source: Source node name
            target: Target node name

        Returns:
            True if path exists, False otherwise
        """
        # Simple BFS implementation
        visited = set()
        queue = [source]

        while queue:
            current = queue.pop(0)

            if current == target:
                return True

            if current in visited:
                continue

            visited.add(current)

            # Find all outgoing connections including branches
            for src, dest in self.get_edges(source=current, include_branches=True):
                if dest not in visited:
                    queue.append(dest)

        return False

    def get_node_pattern(self, pattern: str) -> List[Node]:
        """
        Get nodes matching a pattern.

        Args:
            pattern: Pattern to match (supports * wildcard)

        Returns:
            List of matching nodes
        """
        import fnmatch

        return [
            node
            for name, node in self.nodes.items()
            if node is not None and fnmatch.fnmatch(name, pattern)
        ]

    # Extension methods
    def extend_from(self, other_graph, prefix=""):
        """
        Extend this graph with nodes and edges from another graph.

        Args:
            other_graph: Graph to extend from
            prefix: Optional prefix for imported node names

        Returns:
            Self for method chaining
        """
        # Copy nodes
        for name, node in other_graph.nodes.items():
            if node is None:
                # Skip None nodes or add them as placeholders
                new_name = f"{prefix}_{name}" if prefix else name
                self.nodes[new_name] = None
                continue

            new_name = f"{prefix}_{name}" if prefix else name

            if new_name not in self.nodes:
                # Clone the node
                if hasattr(node, "model_copy"):
                    # Pydantic v2
                    new_node = node.model_copy(deep=True)
                    new_node.name = new_name
                    self.nodes[new_name] = new_node
                elif (
                    hasattr(node, "__class__")
                    and "NodeConfig" in node.__class__.__name__
                ):
                    # NodeConfig - just copy reference
                    self.nodes[new_name] = node
                else:
                    # Create new node with same properties
                    new_node = Node(
                        name=new_name,
                        node_type=getattr(node, "node_type", NodeType.CALLABLE),
                        metadata=getattr(node, "metadata", {}).copy(),
                        input_mapping=getattr(node, "input_mapping", None),
                        output_mapping=getattr(node, "output_mapping", None),
                        command_goto=getattr(node, "command_goto", None),
                        retry_policy=getattr(node, "retry_policy", None),
                        description=getattr(node, "description", None),
                    )
                    self.nodes[new_name] = new_node

                # Track node type if available
                if (
                    hasattr(other_graph, "node_types")
                    and name in other_graph.node_types
                ):
                    self._track_node_type(new_name, other_graph.node_types[name])

                # Handle subgraphs
                if hasattr(other_graph, "subgraphs") and name in other_graph.subgraphs:
                    if not hasattr(self, "subgraphs"):
                        self.subgraphs = {}
                    self.subgraphs[new_name] = other_graph.subgraphs[name]

        # Copy edges
        for src, dst in other_graph.edges:
            # Convert node names
            new_src = f"{prefix}_{src}" if prefix and src not in (START, END) else src
            new_dst = f"{prefix}_{dst}" if prefix and dst not in (START, END) else dst

            # Add edge if both nodes exist
            if (new_src == START or new_src in self.nodes) and (
                new_dst == END or new_dst in self.nodes
            ):
                self.add_edge(new_src, new_dst)

        # Copy branches/conditional edges
        if hasattr(other_graph, "branches"):
            for branch_id, branch in other_graph.branches.items():
                # Convert node names
                new_src = (
                    f"{prefix}_{branch.source_node}"
                    if prefix and branch.source_node != START
                    else branch.source_node
                )

                # Skip if source node doesn't exist
                if new_src != START and new_src not in self.nodes:
                    continue

                # Convert destinations
                new_destinations = {}
                for cond, dest in branch.destinations.items():
                    new_dest = f"{prefix}_{dest}" if prefix and dest != END else dest
                    if new_dest == END or new_dest in self.nodes:
                        new_destinations[cond] = new_dest

                # Convert default
                new_default = (
                    f"{prefix}_{branch.default}"
                    if prefix and branch.default != END
                    else branch.default
                )
                if new_default != END and new_default not in self.nodes:
                    new_default = END

                # Create new branch
                if hasattr(branch, "model_copy"):
                    # Pydantic v2
                    new_branch = branch.model_copy(deep=True)
                    new_branch.id = str(uuid.uuid4())
                    new_branch.source_node = new_src
                    new_branch.destinations = new_destinations
                    new_branch.default = new_default
                else:
                    # Create new branch with same properties
                    new_branch = Branch(
                        id=str(uuid.uuid4()),
                        name=f"{prefix}_{branch.name}" if prefix else branch.name,
                        source_node=new_src,
                        function=getattr(branch, "function", None),
                        key=getattr(branch, "key", None),
                        value=getattr(branch, "value", None),
                        comparison=getattr(branch, "comparison", None),
                        destinations=new_destinations,
                        default=new_default,
                        mode=getattr(branch, "mode", "FUNCTION"),
                    )

                # Add to this graph
                self.branches[new_branch.id] = new_branch

        logger.info(
            f"Extended graph with {len(other_graph.nodes)} nodes from {other_graph.name}"
        )
        self.updated_at = datetime.now()
        return self

    # Integration with LangGraph StateGraph
    def to_langgraph(self, state_schema: Any = None) -> Any:
        """
        Convert this graph to a LangGraph StateGraph.

        Args:
            state_schema: Optional schema for the StateGraph (dict or StateSchema)

        Returns:
            StateGraph instance
        """
        try:
            from langgraph.graph import StateGraph

            # Create StateGraph
            graph_builder = StateGraph(state_schema or self.state_schema or dict)

            # Add nodes for non-None nodes
            for name, node in self.nodes.items():
                if node is None:
                    continue

                # Extract action from metadata
                action = None

                # Handle NodeConfig directly
                if (
                    hasattr(node, "__class__")
                    and "NodeConfig" in node.__class__.__name__
                ):
                    # Use NodeConfig directly
                    if hasattr(node, "__call__"):
                        action = node
                    # Try to find action in metadata or attributes
                    elif hasattr(node, "metadata") and "callable" in node.metadata:
                        action = node.metadata["callable"]
                    elif hasattr(node, "engine") and node.engine:
                        action = node.engine
                else:
                    # Node object - look for action in metadata
                    if hasattr(node, "metadata"):
                        if "callable" in node.metadata:
                            action = node.metadata["callable"]
                        elif "engine" in node.metadata:
                            engine = node.metadata["engine"]
                            if hasattr(engine, "create_runnable"):
                                action = engine.create_runnable()
                            else:
                                action = engine
                        elif "object" in node.metadata:
                            obj = node.metadata["object"]
                            if callable(obj):
                                action = obj

                # Convert retry policy if present
                retry = None
                if hasattr(node, "retry_policy") and node.retry_policy:
                    retry = node.retry_policy

                # Convert command_goto if present
                destinations = None
                if hasattr(node, "command_goto") and node.command_goto:
                    if isinstance(node.command_goto, str):
                        destinations = (node.command_goto,)
                    elif isinstance(node.command_goto, list):
                        destinations = tuple(node.command_goto)

                # Add the node
                if action:
                    metadata = getattr(node, "metadata", {})
                    graph_builder.add_node(
                        name,
                        action,
                        metadata=metadata,
                        retry=retry,
                        destinations=destinations,
                    )
                else:
                    # Placeholder node (warning: will not work in final graph)
                    logger.warning(
                        f"No action found for node '{name}', adding placeholder"
                    )
                    graph_builder.add_node(name, lambda state: state)

            # Add direct edges
            for source, target in self.edges:
                graph_builder.add_edge(source, target)

            # Add branches
            for branch_id, branch in self.branches.items():
                # Create the edge function
                def create_edge_func(branch_obj):
                    def edge_func(state):
                        return branch_obj.evaluate(state)

                    return edge_func

                # Add the conditional edge
                source_node = branch.source_node
                if (
                    source_node != START
                    and source_node in self.nodes
                    and self.nodes[source_node] is not None
                ):
                    graph_builder.add_conditional_edges(
                        source_node, create_edge_func(branch), branch.destinations
                    )

            return graph_builder

        except ImportError:
            logger.error("LangGraph not installed or not found")
            raise ImportError("LangGraph must be installed to convert to StateGraph")

    @classmethod
    def from_langgraph(
        cls, state_graph: Any, name: Optional[str] = None
    ) -> "BaseGraph":
        """
        Create a BaseGraph from a LangGraph StateGraph.

        Args:
            state_graph: LangGraph StateGraph instance
            name: Optional name for the created graph

        Returns:
            BaseGraph instance
        """
        try:
            # Check if the input is a StateGraph
            if not hasattr(state_graph, "nodes") or not hasattr(state_graph, "edges"):
                raise ValueError("Input must be a LangGraph StateGraph")

            # Create new graph
            graph = cls(
                name=name
                or getattr(state_graph, "name", f"graph_{uuid.uuid4().hex[:8]}"),
                state_schema=getattr(state_graph, "schema", None),
            )

            # Convert nodes
            for node_name, node_spec in state_graph.nodes.items():
                # Skip special nodes
                if node_name in ["__start__", "__end__"]:
                    continue

                # Extract node properties
                action = getattr(node_spec, "action", None)
                metadata = getattr(node_spec, "metadata", {})
                retry_policy = getattr(node_spec, "retry_policy", None)

                # Create node
                node = Node(
                    name=node_name,
                    node_type=(
                        NodeType.CALLABLE if callable(action) else NodeType.ENGINE
                    ),
                    metadata=metadata or {},
                    retry_policy=retry_policy,
                )

                # Add action to metadata
                if action:
                    node.metadata["callable"] = action

                # Add the node
                graph.add_node(node)

            # Convert edges
            if hasattr(state_graph, "edges"):
                for source, target in state_graph.edges:
                    # Add the edge
                    graph.add_edge(source, target)

            # Convert branches/conditional edges
            if hasattr(state_graph, "branches"):
                for source_node, conditions in state_graph.branches.items():
                    for condition_name, branch_obj in conditions.items():
                        # Extract routes
                        destinations = {}
                        if hasattr(branch_obj, "ends"):
                            for condition, target in branch_obj.ends.items():
                                destinations[condition] = target

                        # Create branch function
                        branch_func = None
                        if hasattr(branch_obj, "condition"):
                            branch_func = branch_obj.condition

                        # Create Branch object
                        branch_id = str(uuid.uuid4())
                        branch = Branch(
                            id=branch_id,
                            name=f"branch_{condition_name}_{uuid.uuid4().hex[:6]}",
                            source_node=source_node,
                            function=branch_func,
                            function_ref=(
                                CallableReference.from_callable(branch_func)
                                if branch_func
                                else None
                            ),
                            destinations=destinations,
                            default=END,
                            mode=(
                                BranchMode.FUNCTION
                                if branch_func
                                else BranchMode.DIRECT
                            ),
                        )

                        # Add the branch
                        graph.branches[branch_id] = branch

            return graph

        except ImportError:
            logger.error("LangGraph not installed or not found")
            raise ImportError("LangGraph must be installed to convert from StateGraph")

    # Serialization and conversion
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the graph to a serializable dictionary.

        Returns:
            Dictionary representation of the graph
        """
        # Import here to avoid circular imports
        from haive.core.graph.state_graph.serializable import SerializableGraph

        # Convert to serializable representation
        serializable = SerializableGraph.from_graph(self)

        # Convert to dictionary
        return serializable.to_dict()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseGraph":
        """
        Create a graph from a dictionary.

        Args:
            data: Dictionary representation of the graph

        Returns:
            Instantiated graph
        """
        # Import here to avoid circular imports
        from haive.core.graph.state_graph.serializable import SerializableGraph

        # Create serializable representation
        serializable = SerializableGraph.from_dict(data)

        # Convert to graph
        return serializable.to_graph()

    def to_json(self, **kwargs) -> str:
        """
        Convert graph to JSON string.

        Args:
            **kwargs: Additional parameters for JSON serialization

        Returns:
            JSON string representation of the graph
        """
        # Import here to avoid circular imports
        from haive.core.graph.state_graph.serializable import SerializableGraph

        # Convert to serializable representation
        serializable = SerializableGraph.from_graph(self)

        # Convert to JSON
        return serializable.to_json(**kwargs)

    @classmethod
    def from_json(cls, json_str: str) -> "BaseGraph":
        """
        Create a graph from a JSON string.

        Args:
            json_str: JSON string representation of the graph

        Returns:
            Instantiated graph
        """
        # Import here to avoid circular imports
        from haive.core.graph.state_graph.serializable import SerializableGraph

        # Create serializable representation
        serializable = SerializableGraph.from_json(json_str)

        # Convert to graph
        return serializable.to_graph()

    def to_mermaid(self) -> str:
        """
        Generate a Mermaid graph diagram string.

        Returns:
            Mermaid diagram as string
        """
        lines = ["graph TD;"]

        # Add nodes
        lines.append("    %% Nodes")
        for name, node in self.nodes.items():
            # Skip None nodes
            if node is None:
                continue

            # Determine node color based on type
            node_type = self.node_types.get(name, NodeType.CALLABLE)
            color = "#FFFFFF"  # Default white

            if node_type == NodeType.ENGINE:
                color = "#90EE90"  # Light green for engine nodes
            elif node_type == NodeType.TOOL:
                color = "#FFD700"  # Gold for tool nodes
            elif node_type == NodeType.VALIDATION:
                color = "#B0E0E6"  # Light blue for validation nodes
            elif node_type == NodeType.SUBGRAPH:
                color = "#FFA07A"  # Light salmon for subgraphs
            elif node_type == NodeType.CALLABLE:
                color = "#F5F5DC"  # Beige for callable nodes

            # Add style
            lines.append(
                f'    {name}["{name} ({node_type.value})"] style fill:{color};'
            )

        # Add special nodes
        lines.append(
            f'    {START}["{START}"] style fill:#5D8AA8,color:white,font-weight:bold;'
        )
        lines.append(
            f'    {END}["{END}"] style fill:#FF6347,color:white,font-weight:bold;'
        )

        # Add direct edges
        lines.append("    %% Direct edges")
        for source, target in self.edges:
            lines.append(f"    {source} --> {target};")

        # Add branch edges
        if self.branches:
            lines.append("    %% Branch connections")

            for branch_id, branch in self.branches.items():
                source = branch.source_node

                if branch.mode == BranchMode.FUNCTION:
                    lines.append(f"    %% Function Branch: {branch.name}")
                elif branch.mode == BranchMode.DIRECT:
                    lines.append(f"    %% Direct Branch: {branch.name}")
                elif branch.mode == BranchMode.SEND_MAPPER:
                    lines.append(f"    %% Send Mapper Branch: {branch.name}")
                else:
                    lines.append(f"    %% Branch: {branch.name}")

                # Draw connections for destinations
                for condition, target in branch.destinations.items():
                    condition_str = str(condition)
                    if len(condition_str) > 15:
                        condition_str = condition_str[:12] + "..."
                    lines.append(f'    {source} -->|"{condition_str}"| {target};')

                # Draw default connection if different from destinations
                if branch.default not in branch.destinations.values():
                    lines.append(f'    {source} -->|"default"| {branch.default};')

        return "\n".join(lines)
