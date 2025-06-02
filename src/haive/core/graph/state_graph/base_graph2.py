"""
Base graph implementation for the Haive framework.

Provides a comprehensive system for building, manipulating, and executing
graphs with consistent interfaces, serialization support, and dynamic composition.
"""

import logging
import uuid
from datetime import datetime
from enum import Enum

# NEED TO CLEAN UP
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    Union,
)

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START  # Import the actual constants
from langgraph.types import Command, RetryPolicy, Send
from pydantic import BaseModel, Field, model_validator

# Import Branch implementation
from haive.core.graph.branches.branch import Branch
from haive.core.graph.branches.types import BranchMode, ComparisonType
from haive.core.graph.common.references import CallableReference
from haive.core.graph.common.types import ConfigLike, NodeOutput, NodeType, StateLike
from haive.core.graph.node.config import NodeConfig
from haive.core.graph.state_graph.graph_path import GraphPath

# Import mixins
from haive.core.graph.state_graph.validation_mixin import ValidationMixin

# Define a type for branch result types
BranchResultType = Union[
    str,  # Node name
    bool,  # Boolean condition
    List[str],  # List of node names
    List[Send],  # List of Send objects
    Send,  # Single Send object
    Command,  # Command object
    None,  # Default case
]
# Setup logging
logger = logging.getLogger(__name__)
import inspect


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


class BaseGraph(BaseModel, ValidationMixin):
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

    # Entry and finish points - consistent plurals
    entry_points: List[str] = Field(default_factory=list)
    finish_points: List[str] = Field(default_factory=list)
    conditional_entries: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    conditional_exits: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    # Keep backward compatibility fields for singular points
    entry_point: Optional[str] = Field(
        default=None, description="Deprecated: Use entry_points instead"
    )
    finish_point: Optional[str] = Field(
        default=None, description="Deprecated: Use finish_points instead"
    )

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

    # Validation configuration - required by ValidationMixin
    allow_cycles: bool = Field(
        default=False, description="Whether to allow cycles in the graph"
    )
    require_end_path: bool = Field(
        default=True, description="Whether all nodes must have a path to END"
    )

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

        # Validate entry/end points
        if self.entry_points and self.entry_points[0] not in self.nodes:
            raise ValueError(f"Entry point '{self.entry_points[0]}' not found in nodes")

        if self.finish_points and self.finish_points[0] not in self.nodes:
            raise ValueError(
                f"Finish point '{self.finish_points[0]}' not found in nodes"
            )

        # Initialize additional structures if not present
        if not hasattr(self, "subgraphs"):
            self.subgraphs = {}
        if not hasattr(self, "node_types"):
            self.node_types = {}
        if not hasattr(self, "conditional_entries"):
            self.conditional_entries = {}
        if not hasattr(self, "conditional_exits"):
            self.conditional_exits = {}

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
        for _branch_id, branch in self.branches.items():
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
                # Subgraph - use add_subgraph method for proper handling
                return self.add_subgraph(name, node_like, **kwargs)
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

    def set_entry_point(self, node_name: str) -> "BaseGraph":
        """
        Set an entry point of the graph.

        Args:
            node_name: Name of the node to set as an entry point

        Returns:
            Self for method chaining
        """
        if node_name not in self.nodes:
            raise ValueError(f"Node '{node_name}' not found in graph")

        # Store the entry point in the list if not already there
        if node_name not in self.entry_points:
            self.entry_points.append(node_name)

        # For backward compatibility
        self.entry_point = node_name

        # Add edge from START to entry point if not already present
        if not any(src == START and dst == node_name for src, dst in self.edges):
            self.add_edge(START, node_name)

        logger.debug(f"Set entry point to '{node_name}' in graph '{self.name}'")
        self.updated_at = datetime.now()
        return self

    # Rename for consistency
    def set_finish_point(self, node_name: str) -> "BaseGraph":
        """
        Set a finish point of the graph.

        Args:
            node_name: Name of the node to set as a finish point

        Returns:
            Self for method chaining
        """
        if node_name not in self.nodes:
            raise ValueError(f"Node '{node_name}' not found in graph")

        # Store the finish point in the list if not already there
        if node_name not in self.finish_points:
            self.finish_points.append(node_name)

        # For backward compatibility
        self.finish_point = node_name

        # Add edge from finish point to END if not already present
        if not any(src == node_name and dst == END for src, dst in self.edges):
            self.add_edge(node_name, END)

        logger.debug(f"Set finish point to '{node_name}' in graph '{self.name}'")
        self.updated_at = datetime.now()
        return self

    # Backward compatibility alias
    def set_end_point(self, node_name: str) -> "BaseGraph":
        """
        Deprecated: Use set_finish_point instead.

        Set a finish point of the graph.

        Args:
            node_name: Name of the node to set as a finish point

        Returns:
            Self for method chaining
        """
        return self.set_finish_point(node_name)

    def set_conditional_entry(
        self,
        condition: Callable[[StateLike, Optional[ConfigLike]], bool],
        entry_node: str,
        default_entry: Optional[str] = None,
    ) -> "BaseGraph":
        """
        Set a conditional entry point for the graph.

        Args:
            condition: Function that takes state and config, returns boolean
            entry_node: Node to enter if condition is True
            default_entry: Node to enter if condition is False (uses self.entry_point if None)

        Returns:
            Self for method chaining
        """
        if entry_node not in self.nodes:
            raise ValueError(f"Entry node '{entry_node}' not found in graph")

        if default_entry and default_entry not in self.nodes:
            raise ValueError(f"Default entry node '{default_entry}' not found in graph")

        # Generate ID for this conditional entry
        entry_id = str(uuid.uuid4())

        # Store condition and nodes
        self.conditional_entries[entry_id] = {
            "condition": condition,
            "true_entry": entry_node,
            "false_entry": default_entry or self.entry_point,
            "function_ref": CallableReference.from_callable(condition),
        }

        # Add edges from START to both entry nodes
        if not any(src == START and dst == entry_node for src, dst in self.edges):
            self.add_edge(START, entry_node)

        false_node = default_entry or self.entry_point
        if false_node and not any(
            src == START and dst == false_node for src, dst in self.edges
        ):
            self.add_edge(START, false_node)

        logger.debug(
            f"Added conditional entry point to '{entry_node}' in graph '{self.name}'"
        )
        self.updated_at = datetime.now()
        return self

    def set_conditional_exit(
        self,
        node_name: str,
        condition: Callable[[StateLike, Optional[ConfigLike]], bool],
        exit_if_true: bool = True,
    ) -> "BaseGraph":
        """
        Set a conditional exit point for the graph.

        Args:
            node_name: Name of the node to set as conditional exit
            condition: Function that takes state and config, returns boolean
            exit_if_true: Whether to exit when condition is True (default) or False

        Returns:
            Self for method chaining
        """
        if node_name not in self.nodes:
            raise ValueError(f"Node '{node_name}' not found in graph")

        # Generate ID for this conditional exit
        exit_id = str(uuid.uuid4())

        # Store condition and configuration
        self.conditional_exits[exit_id] = {
            "node": node_name,
            "condition": condition,
            "exit_if_true": exit_if_true,
            "function_ref": CallableReference.from_callable(condition),
        }

        # Add edge from node to END if not already present
        if not any(src == node_name and dst == END for src, dst in self.edges):
            self.add_edge(node_name, END)

        logger.debug(
            f"Added conditional exit point at '{node_name}' in graph '{self.name}'"
        )
        self.updated_at = datetime.now()
        return self

    def remove_conditional_entry(self, entry_id: str) -> "BaseGraph":
        """
        Remove a conditional entry point.

        Args:
            entry_id: ID of the conditional entry to remove

        Returns:
            Self for method chaining
        """
        if entry_id not in self.conditional_entries:
            raise ValueError(f"Conditional entry '{entry_id}' not found")

        # Remove the conditional entry
        del self.conditional_entries[entry_id]

        logger.debug(
            f"Removed conditional entry point '{entry_id}' from graph '{self.name}'"
        )
        self.updated_at = datetime.now()
        return self

    def remove_conditional_exit(self, exit_id: str) -> "BaseGraph":
        """
        Remove a conditional exit point.

        Args:
            exit_id: ID of the conditional exit to remove

        Returns:
            Self for method chaining
        """
        if exit_id not in self.conditional_exits:
            raise ValueError(f"Conditional exit '{exit_id}' not found")

        # Remove the conditional exit
        del self.conditional_exits[exit_id]

        logger.debug(
            f"Removed conditional exit point '{exit_id}' from graph '{self.name}'"
        )
        self.updated_at = datetime.now()
        return self

    def get_conditional_entries(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all conditional entry points.

        Returns:
            Dictionary of conditional entries indexed by ID
        """
        return self.conditional_entries

    def get_conditional_exits(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all conditional exit points.

        Returns:
            Dictionary of conditional exits indexed by ID
        """
        return self.conditional_exits

    @property
    def all_entry_points(self) -> Dict[str, Any]:
        """
        Property that returns all entry points (regular and conditional).

        Returns:
            Dictionary containing all entry points information
        """
        result = {"primary": self.entry_points, "conditional": self.conditional_entries}

        # Get all nodes with edges from START
        start_connections = []
        for src, dst in self.edges:
            if src == START and dst not in start_connections:
                start_connections.append(dst)

        result["all_start_connections"] = start_connections

        return result

    # Rename for consistency
    @property
    def all_finish_points(self) -> Dict[str, Any]:
        """
        Property that returns all finish points (regular and conditional).

        Returns:
            Dictionary containing all finish points information
        """
        result = {"primary": self.finish_points, "conditional": self.conditional_exits}

        # Get all nodes with edges to END
        end_connections = []
        for src, dst in self.edges:
            if dst == END and src not in end_connections:
                end_connections.append(src)

        result["all_end_connections"] = end_connections

        return result

    # Backward compatibility aliases
    @property
    def entry_points_data(self) -> Dict[str, Any]:
        """Deprecated: Use all_entry_points instead."""
        return self.all_entry_points

    @property
    def exit_points(self) -> Dict[str, Any]:
        """Deprecated: Use all_finish_points instead."""
        return self.all_finish_points

    @property
    def all_exit_points(self) -> Dict[str, Any]:
        """Deprecated: Use all_finish_points instead."""
        return self.all_finish_points

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

        # Create node - this represents the subgraph as a single node in the main graph
        node_obj = Node(
            name=name,
            node_type=NodeType.SUBGRAPH,
            metadata={"subgraph": subgraph.name, "subgraph_id": subgraph.id},
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
            if not hasattr(new_node, "name") or not new_node.name:
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
        for _branch_id, branch in self.branches.items():
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
        for _branch_id, branch in self.branches.items():
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
            if not hasattr(new_node, "name") or not new_node.name:
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
            for _branch_id, branch in self.branches.items():
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
            for _src, dst in self.get_edges(source=node):
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

    def find_nodes_without_finish_path(self):
        """
        Find nodes that can't reach a finish point.
        Alias for find_nodes_without_end_path for API consistency.

        Returns:
            List of node names that can't reach a finish point
        """
        return self.find_nodes_without_end_path()

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

    def add_conditional_edges(
        self,
        source_node: str,
        condition: Union[
            Branch, Callable[[StateLike, Optional[ConfigLike]], BranchResultType], Any
        ],
        destinations: Optional[
            Union[str, List[str], Dict[Union[bool, str, int], str]]
        ] = None,
        default: Union[str, Literal["END"], None] = END,
        create_missing_nodes: bool = False,
    ) -> "BaseGraph":
        """
        Add conditional edges from a source node based on a condition.

        This method supports multiple ways to handle True/False routing:

        1. **Boolean destinations**: Use True/False as keys
           ```python
           graph.add_conditional_edges(
               'agent_node',
               has_tool_calls,
               {True: 'validation', False: END}
           )
           ```

        2. **String destinations with optional boolean fallbacks**:
           String keys like 'has_tool_calls'/'no_tool_calls' can optionally
           get True/False fallbacks added by setting add_boolean_fallbacks=True
           ```python
           graph.add_conditional_edges(
               'agent_node',
               has_tool_calls,
               {'has_tool_calls': 'validation', 'no_tool_calls': END},
               add_boolean_fallbacks=True
           )
           # With add_boolean_fallbacks=True, adds: {True: 'validation', False: END}
           ```

        3. **List format**: First item = True destination, Second item = False destination
           ```python
           graph.add_conditional_edges(
               'agent_node',
               has_tool_calls,
               ['validation', END]  # validation when True, END when False
           )
           ```

        **Alternative**: For simple boolean routing, consider using `add_boolean_conditional_edges()`
        which provides cleaner syntax for True/False conditions.

        Args:
            source_node: Source node name
            condition: A function, Branch, NodeConfig or any object that can determine branching.
                      For callables, takes (state, optional config) and returns a node name,
                      boolean, list of nodes, list of Send objects, Send object, Command object,
                      or a Branch object itself
            destinations: Target node(s) - can be:
                - A single node name string (which will be mapped to True)
                - A list of node names (which will be mapped by index)
                - A dictionary mapping condition results to target nodes
            default: Default destination if no condition matches (defaults to END).
                    IMPORTANT: When destinations is a dictionary, no default is ever added.
            create_missing_nodes: Whether to create missing destination nodes automatically (defaults to False)
            add_boolean_fallbacks: Whether to automatically add True/False keys for string-keyed destinations (defaults to False)

        Returns:
            Self for method chaining
        """
        import logging

        logger = logging.getLogger(__name__)
        from langgraph.graph import END, START

        # Validate source node
        if source_node != START and source_node not in self.nodes:
            raise ValueError(f"Source node '{source_node}' not found in graph")

        # Handle Branch objects
        if isinstance(condition, Branch):
            branch = condition
            branch.source_node = source_node

            # Process destinations based on type
            if destinations is not None:
                if isinstance(destinations, str):
                    # Single string maps to True
                    branch.destinations = {True: destinations}
                elif isinstance(destinations, list):
                    if len(destinations) >= 2:
                        # Two or more destinations map to True/False
                        branch.destinations = {
                            True: destinations[0],
                            False: destinations[1],
                        }
                    elif len(destinations) == 1:
                        # Single destination in list maps True to destination, False to END
                        branch.destinations = {True: destinations[0], False: END}
                    else:
                        # Empty list uses default {True: "continue", False: END}
                        branch.destinations = {True: "continue", False: END}
                elif isinstance(destinations, dict):
                    # Dictionary maps directly - NO MODIFICATION
                    branch.destinations = destinations

                    # NO DEFAULT - EVER
                    branch.default = None
            else:
                # No destinations provided - use default {True: "continue", False: END}
                branch.destinations = {True: "continue", False: END}
                branch.default = default

            # Validate destination nodes exist or create them if requested
            if not create_missing_nodes:
                for dest_name in branch.destinations.values():
                    if dest_name != END and dest_name not in self.nodes:
                        raise ValueError(
                            f"Destination node '{dest_name}' not found in graph. "
                            f"Use create_missing_nodes=True to create it automatically."
                        )

            self.branches[branch.id] = branch
            logger.debug(f"Added branch with ID: {branch.id}")
            return self

        # Extract function name or condition info for debugging
        if (
            hasattr(condition, "__class__")
            and "NodeConfig" in condition.__class__.__name__
        ):
            condition_name = getattr(
                condition, "name", f"branch_condition_{uuid.uuid4().hex[:8]}"
            )
            logger.debug(f"Using NodeConfig '{condition_name}' as branch controller")
        else:
            condition_name = getattr(
                condition, "__name__", f"branch_condition_{uuid.uuid4().hex[:8]}"
            )
            logger.debug(f"Using function '{condition_name}' as branch controller")

        # Create destination mapping based on input type
        destination_map = {}

        # Set default based on destination type
        if isinstance(destinations, dict):
            # NEVER use a default with dictionary destinations
            use_default = None
        else:
            use_default = default

        # Build destination map based on input type
        if isinstance(destinations, str):
            destination_map = {True: destinations, False: default}
        elif isinstance(destinations, list):
            if len(destinations) >= 2:
                # Map first element to True, second to False
                destination_map = {True: destinations[0], False: destinations[1]}
            elif destinations:
                # Single element maps True, False gets default
                destination_map = {True: destinations[0], False: default}
            else:
                # Empty list uses general default
                destination_map = {True: "continue", False: default}
        elif isinstance(destinations, dict):
            # Use dictionary exactly as provided - NO MODIFICATIONS
            destination_map = destinations
        elif destinations is None:
            # Default mapping
            destination_map = {True: "continue", False: default}

        # Validate destination nodes exist or create them if requested
        if not create_missing_nodes:
            for dest_key, dest_name in destination_map.items():
                if (
                    dest_name != END
                    and dest_name != "continue"
                    and dest_name not in self.nodes
                ):
                    raise ValueError(
                        f"Destination node '{dest_name}' for condition '{dest_key}' not found in graph. "
                        f"Use create_missing_nodes=True to create it automatically."
                    )

        # Create a new Branch
        branch_id = str(uuid.uuid4())
        branch_name = f"branch_{branch_id[:8]}"

        metadata = {"condition_object": condition} if not callable(condition) else {}

        # Create wrapper for callable conditions to log results and handle None returns
        if callable(condition):
            function_to_use = self._create_branch_wrapper(
                condition, destination_map, use_default
            )
            function_ref = CallableReference.from_callable(function_to_use)
        else:
            function_to_use = condition
            function_ref = None

        # Log the key information for debugging
        if callable(condition) and hasattr(condition, "__name__"):
            logging.info(f"Using branch function: {condition.__name__}")
        elif hasattr(condition, "__class__"):
            logging.info(f"Using condition of type: {condition.__class__.__name__}")

        # Special handling for ValidationNodeConfig
        if (
            hasattr(condition, "__class__")
            and "ValidationNodeConfig" in condition.__class__.__name__
        ):
            logging.info("Detected ValidationNodeConfig - ensuring correct routing")
            metadata["is_validation_node"] = True

            # Ensure ValidationNodeConfig function knows to return exact routing keys
            function_to_use = self._create_validation_wrapper(
                condition, destination_map
            )
            function_ref = CallableReference.from_callable(function_to_use)

        # Create branch with the mapped destinations
        branch = Branch(
            id=branch_id,
            name=branch_name,
            source_node=source_node,
            function=function_to_use,
            function_ref=function_ref,
            metadata=metadata,
            mode=BranchMode.FUNCTION if callable(condition) else BranchMode.DIRECT,
            destinations=destination_map,
            default=use_default,
        )

        # Log the finalized branch configuration for debugging
        logger.debug(
            f"Branch configuration: source={source_node}, destinations={destination_map}, default={use_default}"
        )

        # Add the branch
        self.branches[branch_id] = branch

        logger.debug(f"Branch '{branch_name}' added successfully!")
        self.updated_at = datetime.now()
        return self

    def _create_validation_wrapper(self, validation_config, destination_map):
        """Special wrapper for ValidationNodeConfig to ensure correct routing.

        ValidationNodeConfig is a common source of routing issues because it can return
        complex results that don't directly match routing keys.
        """
        from langgraph.types import Send

        # Get a list of valid string keys for routing
        valid_keys = list(destination_map.keys())
        logging.info(f"Valid routing keys: {valid_keys}")

        def validation_wrapper(state, config=None):
            try:
                # ValidationNodeConfig uses __call__ method, not process_validation
                if callable(validation_config):
                    # Call the ValidationNodeConfig directly
                    result = validation_config(state, config)
                    logging.info(
                        f"ValidationNodeConfig result: {type(result).__name__}"
                    )

                    # Handle Send objects directly - this is the primary return type
                    if isinstance(result, list) and all(
                        isinstance(item, Send) for item in result
                    ):
                        logging.info(
                            f"ValidationNodeConfig returned {len(result)} Send objects"
                        )
                        return result
                    elif isinstance(result, Send):
                        logging.info(
                            f"ValidationNodeConfig returned single Send object to {result.node}"
                        )
                        return result

                    # Handle Command objects
                    if (
                        hasattr(result, "__class__")
                        and "Command" in result.__class__.__name__
                    ):
                        logging.info("ValidationNodeConfig returned Command object")
                        return result

                    # If result is already a string in our routing map, use it directly
                    if isinstance(result, str) and result in destination_map:
                        logging.info(
                            f"ValidationNodeConfig returned string key: {result}"
                        )
                        return result

                    # Handle special case where ValidationNodeConfig returns "no_tool_calls"
                    if isinstance(result, str) and result == "no_tool_calls":
                        # Look for a key that maps to END or similar
                        for key, dest in destination_map.items():
                            if dest == "END" or str(dest).upper() == "END":
                                logging.info(
                                    f"Converting 'no_tool_calls' to routing key: {key}"
                                )
                                return key
                        # If no END destination found, return the string as-is
                        logging.info(
                            "ValidationNodeConfig returned no_tool_calls, returning as-is"
                        )
                        return result

                    # Check for validation results dictionary
                    if isinstance(result, dict):
                        # Look for has_errors, has_tools, parse_output keys
                        for key in ["has_errors", "has_tools", "parse_output"]:
                            if key in result and result[key] and key in destination_map:
                                logging.info(f"Found validation key: {key}")
                                return key

                        # Look for any True values that match routing keys
                        for key, value in result.items():
                            if value is True and key in destination_map:
                                logging.info(f"Found True key: {key}")
                                return key

                    # For any validation failure, default to 'has_errors' if available
                    if "has_errors" in destination_map:
                        logging.info("Defaulting to has_errors")
                        return "has_errors"

                    # Last resort - return first routing key
                    if valid_keys:
                        logging.warning(
                            f"No routing match found - using first key: {valid_keys[0]}"
                        )
                        return valid_keys[0]

                    return False

                # If validation_config doesn't have __call__, just return first key
                if valid_keys:
                    logging.warning("ValidationNodeConfig doesn't have __call__ method")
                    return valid_keys[0]
                return False

            except Exception as e:
                logging.error(f"Error in validation function: {e}")
                import traceback

                traceback.print_exc()
                # For errors, route to has_errors if available
                if "has_errors" in destination_map:
                    return "has_errors"
                return False

        return validation_wrapper

    def _create_branch_wrapper(self, func, destination_map, default_dest):
        """Wrapper for branch functions that handles boolean to string conversion."""

        param_count = len(inspect.signature(func).parameters)

        # Check if we have a boolean function with string destinations
        has_boolean_keys = any(isinstance(k, bool) for k in destination_map.keys())
        has_string_keys = any(isinstance(k, str) for k in destination_map.keys())

        def wrapper(state, config=None):
            try:
                # Call with appropriate number of parameters
                if param_count == 1:
                    result = func(state)
                else:
                    result = func(state, config)

                # Log the raw result for debugging
                logger.debug(
                    f"Branch function returned: {result} (type: {type(result).__name__})"
                )

                # CRITICAL FIX: Add detailed debugging for has_tool_calls function
                if hasattr(func, "__name__") and "has_tool_calls" in func.__name__:
                    logger.info("=== DEBUGGING has_tool_calls function ===")
                    logger.info(f"Function result: {result} (type: {type(result)})")
                    logger.info(
                        f"Available routing keys: {list(destination_map.keys())}"
                    )
                    logger.info(f"State type: {type(state)}")

                    # Debug the state content
                    if hasattr(state, "messages"):
                        messages = state.messages
                        logger.info(f"State has {len(messages)} messages")
                        if messages:
                            last_msg = messages[-1]
                            logger.info(f"Last message type: {type(last_msg)}")
                            logger.info(f"Last message: {last_msg}")

                            # Check for tool_calls specifically
                            if hasattr(last_msg, "tool_calls"):
                                tool_calls = getattr(last_msg, "tool_calls", None)
                                logger.info(f"tool_calls attribute: {tool_calls}")

                            if hasattr(last_msg, "additional_kwargs"):
                                additional_kwargs = getattr(
                                    last_msg, "additional_kwargs", {}
                                )
                                logger.info(f"additional_kwargs: {additional_kwargs}")
                                if "tool_calls" in additional_kwargs:
                                    logger.info(
                                        f"tool_calls in additional_kwargs: {additional_kwargs['tool_calls']}"
                                    )

                    logger.info("=== END DEBUGGING ===")

                # Handle boolean to string conversion if needed
                if (
                    isinstance(result, bool)
                    and has_string_keys
                    and not has_boolean_keys
                ):
                    # Convert boolean to string for routing
                    if result is True:
                        # Look for common "true" patterns
                        for key in destination_map.keys():
                            if key in [
                                "has_tool_calls",
                                "has_tools",
                                "true",
                                "yes",
                                "continue",
                            ]:
                                logger.debug(f"Converting True to string key: {key}")
                                return key
                        # Fallback: use first key
                        first_key = list(destination_map.keys())[0]
                        logger.debug(f"Converting True to first key: {first_key}")
                        return first_key
                    else:  # result is False
                        # Look for common "false" patterns
                        for key in destination_map.keys():
                            if key in [
                                "no_tool_calls",
                                "no_tools",
                                "false",
                                "no",
                                "end",
                            ]:
                                logger.debug(f"Converting False to string key: {key}")
                                return key
                        # Fallback: use last key or default
                        if default_dest and default_dest in destination_map.values():
                            # Find the key that maps to default_dest
                            for k, v in destination_map.items():
                                if v == default_dest:
                                    logger.debug(
                                        f"Converting False to default key: {k}"
                                    )
                                    return k
                        # Last resort: use last key
                        last_key = list(destination_map.keys())[-1]
                        logger.debug(f"Converting False to last key: {last_key}")
                        return last_key

                # Return result as-is if no conversion needed
                return result
            except Exception as e:
                logging.error(f"Error in branch function: {e}")
                import traceback

                traceback.print_exc()
                return False

        return wrapper

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
            for _src, dest in self.get_edges(source=current, include_branches=True):
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
            for _branch_id, branch in other_graph.branches.items():
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
    def to_langgraph(
        self,
        state_schema: Optional[Type[BaseModel]] = None,
        input_schema: Optional[Type[BaseModel]] = None,
        output_schema: Optional[Type[BaseModel]] = None,
        config_schema: Optional[Type[BaseModel]] = None,
        **kwargs,
    ) -> Any:
        """
        Convert to LangGraph StateGraph with proper schema handling.

        Schema Resolution Logic:
        1. If state_schema provided: use it, default input/output to state_schema
        2. If input_schema and output_schema provided: use them, create PassThroughState for state_schema
        3. If only input_schema provided: use it for both input and state, output defaults to state
        4. If only output_schema provided: use it for both output and state, input defaults to state
        5. If none provided: use self.state_schema or dict
        """
        try:
            from langgraph.graph import StateGraph
            from rich.console import Console
            from rich.panel import Panel

            console = Console()
            console.print(
                Panel.fit(
                    "[bold blue]Converting to LangGraph StateGraph[/bold blue]",
                    border_style="blue",
                )
            )

            # Schema resolution logic
            resolved_state_schema = None
            resolved_input_schema = None
            resolved_output_schema = None
            resolved_config_schema = config_schema  # Optional, can be None

            # Case 1: state_schema is provided
            if state_schema is not None:
                resolved_state_schema = state_schema
                resolved_input_schema = input_schema or state_schema  # Default to state
                resolved_output_schema = (
                    output_schema or state_schema
                )  # Default to state
                console.print(
                    f"[green]Using provided state_schema: {state_schema.__name__}[/green]"
                )

            # Case 2: Both input and output provided, but no state
            elif input_schema is not None and output_schema is not None:
                resolved_input_schema = input_schema
                resolved_output_schema = output_schema

                # Create pass-through state schema
                class PassThroughState(BaseModel):
                    model_config = ConfigDict(arbitrary_types_allowed=True)

                resolved_state_schema = PassThroughState
                console.print(
                    f"[yellow]Created PassThroughState, input: {input_schema.__name__}, output: {output_schema.__name__}[/yellow]"
                )

            # Case 3: Only input_schema provided
            elif input_schema is not None:
                resolved_input_schema = input_schema
                resolved_state_schema = input_schema  # Use input as state
                resolved_output_schema = (
                    output_schema or input_schema
                )  # Default output to input/state
                console.print(
                    f"[cyan]Using input_schema as state: {input_schema.__name__}[/cyan]"
                )

            # Case 4: Only output_schema provided
            elif output_schema is not None:
                resolved_output_schema = output_schema
                resolved_state_schema = output_schema  # Use output as state
                resolved_input_schema = (
                    input_schema or output_schema
                )  # Default input to output/state
                console.print(
                    f"[cyan]Using output_schema as state: {output_schema.__name__}[/cyan]"
                )

            # Case 5: Nothing provided, use graph's state_schema or dict
            else:
                resolved_state_schema = getattr(self, "state_schema", dict)
                resolved_input_schema = resolved_state_schema
                resolved_output_schema = resolved_state_schema
                schema_name = (
                    resolved_state_schema.__name__
                    if hasattr(resolved_state_schema, "__name__")
                    else str(resolved_state_schema)
                )
                console.print(f"[dim]Using default schema: {schema_name}[/dim]")

            # Log final schema resolution
            console.print("[bold]Final schemas:[/bold]")
            console.print(
                f"  State: {resolved_state_schema.__name__ if hasattr(resolved_state_schema, '__name__') else resolved_state_schema}"
            )
            console.print(
                f"  Input: {resolved_input_schema.__name__ if hasattr(resolved_input_schema, '__name__') else resolved_input_schema}"
            )
            console.print(
                f"  Output: {resolved_output_schema.__name__ if hasattr(resolved_output_schema, '__name__') else resolved_output_schema}"
            )
            if resolved_config_schema:
                console.print(f"  Config: {resolved_config_schema.__name__}")

            # Create StateGraph with resolved state schema
            graph_builder = StateGraph(resolved_state_schema)
            console.print("[green]✓[/green] Created StateGraph")

            # Direct debug function - doesn't wrap, just adds a print
            def log_function_call(func, name):
                import inspect

                # Check the function signature to see how many parameters it accepts
                sig = inspect.signature(func)
                param_count = len(sig.parameters)

                console.print(
                    f"Node [yellow]{name}[/yellow]: Function accepts {param_count} parameter(s)"
                )

                def inner(state, config=None):
                    try:
                        # Call with appropriate number of parameters
                        if param_count == 1:
                            # Function only accepts state (one parameter)
                            console.print(
                                f"[bold]Calling {name}[/bold] with 1 parameter (state only)"
                            )
                            result = func(state)
                        else:
                            # Function accepts both state and config
                            console.print(
                                f"[bold]Calling {name}[/bold] with 2 parameters (state and config)"
                            )
                            result = func(state, config)

                        # Log the result
                        console.print(
                            f"[bold cyan]Node {name} returned:[/bold cyan] [yellow]{type(result).__name__}[/yellow]"
                        )

                        # Special debug for Command objects
                        from langgraph.types import Command

                        if isinstance(result, Command):
                            console.print(
                                Panel.fit(
                                    "[bold yellow]Command Details:[/bold yellow]\n"
                                    + f"Type: {type(result).__name__}\n"
                                    + f"Update: {getattr(result, 'update', None)}\n"
                                    + f"Branch: {getattr(result, 'branch', None)}\n"
                                    + f"Raw: {result}",
                                    border_style="yellow",
                                )
                            )

                        return result
                    except Exception as e:
                        console.print(f"[bold red]Error in {name}:[/bold red] {str(e)}")
                        raise

                return inner

            # SIMPLE: Add nodes with direct callable functions - no complex extraction
            console.print("\n[bold]Adding Nodes:[/bold]")
            for node_name, node in self.nodes.items():
                # Skip special nodes and None nodes
                if node_name in [START, END] or node is None:
                    continue

                action = None

                # Extract the callable with simple priority rules
                if callable(node):
                    # 1. Node is directly callable
                    action = node
                    console.print(
                        f"Node [yellow]{node_name}[/yellow]: Using direct callable"
                    )
                elif (
                    hasattr(node, "metadata")
                    and "callable" in node.metadata
                    and callable(node.metadata["callable"])
                ):
                    # 2. Node has callable in metadata
                    action = node.metadata["callable"]
                    console.print(
                        f"Node [yellow]{node_name}[/yellow]: Using metadata callable"
                    )
                elif callable(node) and callable(node.__call__):
                    # 3. Node has __call__ method
                    action = node
                    console.print(
                        f"Node [yellow]{node_name}[/yellow]: Using __call__ method"
                    )
                else:
                    # Fallback
                    console.print(
                        f"Node [yellow]{node_name}[/yellow]: No callable found, using pass-through"
                    )

                    def action(state, config=None):
                        return state

                # Add logger for debugging if not in production
                import os

                if os.environ.get("PRODUCTION") != "true":
                    action = log_function_call(action, node_name)

                # Add node directly to graph
                graph_builder.add_node(node_name, action)

            # SIMPLE: Add direct edges
            console.print("\n[bold]Adding Edges:[/bold]")
            for source, target in self.edges:
                graph_builder.add_edge(source, target)
                console.print(f"[green]→[/green] {source} → {target}")

            # SIMPLE: Add branches
            console.print("\n[bold]Adding Branches:[/bold]")
            for _branch_id, branch in self.branches.items():
                source = branch.source_node

                # Extract destinations
                destinations = {}
                for key, value in branch.destinations.items():
                    destinations[key] = value

                console.print(
                    f"Branch from [yellow]{source}[/yellow] with conditions: {list(destinations.keys())}"
                )
                console.print(f"[dim]Destinations dict: {destinations}[/dim]")

                # Check branch function and add parameter-aware wrapper if needed
                if branch.mode == BranchMode.FUNCTION and branch.function:
                    # Check branch function signature
                    import inspect

                    try:
                        sig = inspect.signature(branch.function)
                        param_count = len(sig.parameters)

                        # Check if this is a ValidationNodeConfig for special handling
                        is_validation_node = getattr(branch, "metadata", {}).get(
                            "is_validation_node", False
                        )
                        if is_validation_node:
                            console.print(
                                f"[bold magenta]Special handling for ValidationNodeConfig in '{branch.name}'[/bold magenta]"
                            )

                        # Create parameter-aware branch function
                        def branch_wrapper(
                            branch_func, param_count, branch_name, dest_dict
                        ):
                            def wrapper(state, config=None):
                                try:
                                    # Call with appropriate parameter count
                                    if param_count == 1:
                                        result = branch_func(state)
                                    else:
                                        result = branch_func(state, config)

                                    # Handle Send objects correctly
                                    if isinstance(result, list) and all(
                                        isinstance(item, Send) for item in result
                                    ):
                                        console.print(
                                            f"[cyan]Branch returning list of {len(result)} Send objects[/cyan]"
                                        )
                                        return result
                                    elif isinstance(result, Send):
                                        console.print(
                                            f"[cyan]Branch returning Send object to {result.target}[/cyan]"
                                        )
                                        return result

                                    # Special handling for ValidationNodeConfig results
                                    if is_validation_node and isinstance(result, dict):
                                        # Try to extract routing keys
                                        for key in [
                                            "has_errors",
                                            "has_tools",
                                            "parse_output",
                                        ]:
                                            if (
                                                key in result
                                                and result[key]
                                                and key in dest_dict
                                            ):
                                                console.print(
                                                    f"[green]Found validation key: {key}[/green]"
                                                )
                                                return key

                                        # Look for any True values that match routing keys
                                        for key, value in result.items():
                                            if value is True and key in dest_dict:
                                                console.print(
                                                    f"[green]Found True key: {key}[/green]"
                                                )
                                                return key

                                        # For validation failure, use has_errors if available
                                        if "has_errors" in dest_dict:
                                            console.print(
                                                "[yellow]Using has_errors for validation result[/yellow]"
                                            )
                                            return "has_errors"

                                    # Return other results directly
                                    return result
                                except Exception as e:
                                    console.print(
                                        f"[bold red]Error in branch: {str(e)}[/bold red]"
                                    )
                                    return False

                            return wrapper

                        # Create wrapped branch function
                        branch_func = branch_wrapper(
                            branch.function, param_count, branch.name, destinations
                        )

                        # Add conditional edges with the wrapped function
                        # SIMPLIFIED: Just use the basic LangGraph API without complications
                        try:
                            console.print(
                                f"[dim]Adding conditional edges: {source} -> {destinations}[/dim]"
                            )

                            # Use the simplest possible call to LangGraph
                            graph_builder.add_conditional_edges(
                                source, branch_func, destinations
                            )
                            console.print(
                                f"[green]✓ Successfully added conditional edges for {source}[/green]"
                            )

                        except Exception as e:
                            console.print(
                                f"[red]✗ Error adding conditional edges for {source}: {e}[/red]"
                            )
                            console.print(f"[red]  Destinations: {destinations}[/red]")
                            console.print(f"[red]  Function: {branch_func}[/red]")
                            raise
                    except Exception as e:
                        # If anything goes wrong with signature inspection, use original function
                        console.print(
                            f"[yellow]Warning: Could not inspect branch function: {str(e)}[/yellow]"
                        )
                        console.print(
                            "[yellow]Falling back to original function[/yellow]"
                        )

                        try:
                            graph_builder.add_conditional_edges(
                                source, branch.function, destinations
                            )
                            console.print(
                                f"[green]✓ Successfully added conditional edges for {source} (fallback)[/green]"
                            )
                        except Exception as fallback_error:
                            console.print(
                                f"[red]✗ Fallback also failed for {source}: {fallback_error}[/red]"
                            )
                            console.print(f"[red]  Destinations: {destinations}[/red]")
                            console.print(
                                f"[red]  Original function: {branch.function}[/red]"
                            )
                            raise
                else:
                    # Use branch object's __call__ method
                    console.print(f"Using branch object directly for {branch.name}")

                    try:
                        graph_builder.add_conditional_edges(
                            source, branch, destinations
                        )
                        console.print(
                            f"[green]✓ Successfully added conditional edges for {source} (branch object)[/green]"
                        )
                    except Exception as e:
                        console.print(
                            f"[red]✗ Error using branch object for {source}: {e}[/red]"
                        )
                        console.print(f"[red]  Destinations: {destinations}[/red]")
                        console.print(f"[red]  Branch: {branch}[/red]")
                        raise

            console.print("\n[bold green]LangGraph conversion complete![/bold green]")
            return graph_builder

        except ImportError:
            raise ImportError(
                "LangGraph not installed. Install with: pip install langgraph"
            )

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
                if node_name in [START, END]:
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

    def to_mermaid(
        self,
        include_subgraphs: bool = True,
        theme: str = "default",
        subgraph_mode: str = "cluster",
        show_default_branches: bool = False,
    ) -> str:
        """
        Generate a Mermaid graph diagram string.

        Args:
            include_subgraphs: Whether to visualize subgraphs as clusters
            theme: Mermaid theme name (default, forest, dark, neutral)
            subgraph_mode: How to render subgraphs ("cluster", "inline", or "separate")
            show_default_branches: Whether to show default branches

        Returns:
            Mermaid diagram as string
        """
        from haive.core.graph.state_graph.graph_visualizer import GraphVisualizer

        return GraphVisualizer.generate_mermaid(
            self,
            include_subgraphs=include_subgraphs,
            theme=theme,
            subgraph_mode=subgraph_mode,
            show_default_branches=show_default_branches,
        )

    def visualize(
        self,
        output_path: Optional[str] = None,
        include_subgraphs: bool = True,
        highlight_nodes: Optional[List[str]] = None,
        highlight_paths: Optional[List[List[str]]] = None,
        save_png: bool = True,
        width: str = "100%",
        theme: str = "default",
        subgraph_mode: str = "cluster",
        show_default_branches: bool = False,
    ) -> str:
        """
        Generate and display a visualization of the graph.

        This method attempts multiple rendering approaches based on the environment,
        with fallbacks to ensure something is always displayed.

        Args:
            output_path: Optional path to save the diagram
            include_subgraphs: Whether to visualize subgraphs as clusters
            highlight_nodes: List of node names to highlight
            highlight_paths: List of paths to highlight (each path is a list of node names)
            save_png: Whether to save the diagram as PNG
            width: Width of the displayed diagram
            theme: Mermaid theme to use (e.g., "default", "forest", "dark", "neutral")
            subgraph_mode: How to render subgraphs ("cluster", "inline", or "separate")
            show_default_branches: Whether to show default branches

        Returns:
            The generated Mermaid code
        """
        from haive.core.graph.state_graph.graph_visualizer import GraphVisualizer
        from haive.core.utils.mermaid_utils import Environment, detect_environment

        logger = logging.getLogger(__name__)
        logger.debug(
            f"Visualizing graph: {self.name} (nodes: {len(self.nodes)}, edges: {len(self.edges)})"
        )

        # Log some debug info about subgraphs if they exist
        if include_subgraphs and self.subgraphs:
            subgraph_info = ", ".join(
                [
                    f"{name} ({len(sg.nodes)} nodes)"
                    for name, sg in self.subgraphs.items()
                ]
            )
            logger.debug(f"Including {len(self.subgraphs)} subgraphs: {subgraph_info}")

        # Combine nodes from highlight_paths with highlight_nodes
        all_highlight_nodes = highlight_nodes or []
        if highlight_paths:
            for path in highlight_paths:
                all_highlight_nodes.extend(path)
            logger.debug(f"Highlighting {len(all_highlight_nodes)} nodes")

        # Generate the Mermaid code
        try:
            mermaid_code = GraphVisualizer.generate_mermaid(
                self,
                include_subgraphs=include_subgraphs,
                highlight_nodes=all_highlight_nodes if all_highlight_nodes else None,
                theme=theme,
            )
            logger.debug(f"Generated Mermaid code: {len(mermaid_code)} characters")
        except Exception as e:
            logger.error(f"Error generating Mermaid code: {str(e)}")
            # Try again with subgraphs disabled as a fallback
            if include_subgraphs and self.subgraphs:
                logger.warning("Retrying visualization with subgraphs disabled")
                try:
                    mermaid_code = GraphVisualizer.generate_mermaid(
                        self,
                        include_subgraphs=False,
                        highlight_nodes=(
                            all_highlight_nodes if all_highlight_nodes else None
                        ),
                        theme=theme,
                    )
                except Exception as e2:
                    logger.error(
                        f"Failed to generate Mermaid code even without subgraphs: {str(e2)}"
                    )
                    return f"Error generating graph visualization: {str(e)}"
            else:
                return f"Error generating graph visualization: {str(e)}"

        try:
            # Display using the visualizer
            GraphVisualizer.display_graph(
                self,
                output_path=output_path,
                include_subgraphs=include_subgraphs,
                highlight_nodes=highlight_nodes,
                highlight_paths=highlight_paths,
                save_png=save_png,
                width=width,
                theme=theme,  # Pass the string value directly
                subgraph_mode=subgraph_mode,
                show_default_branches=show_default_branches,
            )
            if output_path and save_png:
                logger.info(f"Graph visualization saved to: {output_path}")
        except Exception as e:
            # Get the current environment
            env = detect_environment()

            # Provide helpful error message based on environment
            logger.error(f"Error displaying graph: {e}")
            print(f"Error displaying graph: {e}")
            print(f"Detected environment: {env}")

            if env == Environment.JUPYTER_LAB:
                print("\nTry these options:")
                print(
                    "1. Install JupyterLab Mermaid extension: jupyter labextension install @jupyterlab/mermaid"
                )
                print("2. Run this in a cell to display using HTML:")
                print("   from IPython.display import HTML")
                print(
                    "   HTML(f'''<div class=\"mermaid\">{my_graph.to_mermaid()}</div>''')"
                )
                print(
                    '   <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>'
                )
                print("   <script>mermaid.initialize({startOnLoad:true});</script>''')")
            elif env == Environment.JUPYTER_NOTEBOOK:
                print("\nTry these options:")
                print("1. Run this in a cell to display using HTML:")
                print("   from IPython.display import HTML")
                print(
                    "   HTML(f'''<div class=\"mermaid\">{my_graph.to_mermaid()}</div>"
                )
                print(
                    '   <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>'
                )
                print("   <script>mermaid.initialize({startOnLoad:true});</script>''')")
            elif env == Environment.VSCODE_NOTEBOOK:
                print("\nTry these options:")
                print("1. Install Mermaid Preview extension for VSCode")
                print(
                    "2. Save the diagram and view manually: my_graph.save_visualization('diagram.png')"
                )

        # Always return the Mermaid code for reference
        return mermaid_code

    # Implementation of ValidationMixin required methods
    def analyze_cycles(self) -> List[List[str]]:
        """Find all cycles in the graph."""
        cycles = []
        visited = set()
        path = []
        path_set = set()

        def dfs(node):
            if node in path_set:
                # Found a cycle, extract it
                cycle_start = path.index(node)
                cycles.append(path[cycle_start:] + [node])
                return

            if node in visited:
                return

            visited.add(node)
            path.append(node)
            path_set.add(node)

            # Follow direct edges
            for src, dst in self.edges:
                if src == node and dst != END:
                    dfs(dst)

            # Follow branch destinations
            for branch in self.branches.values():
                if branch.source_node == node:
                    for dest in branch.destinations.values():
                        if dest != END:
                            dfs(dest)

                    if branch.default and branch.default != END:
                        dfs(branch.default)

            path.pop()
            path_set.remove(node)

        # Start DFS from each node
        for node in self.nodes:
            if node not in visited:
                dfs(node)

        return cycles

    def find_orphan_nodes(self) -> List[str]:
        """Find nodes with no incoming or outgoing edges."""
        orphans = []

        for node_name in self.nodes:
            # Skip special nodes and None nodes
            if node_name in [START, END] or self.nodes[node_name] is None:
                continue

            # Check if node has any incoming edges
            has_incoming = False
            for _, dst in self.edges:
                if dst == node_name:
                    has_incoming = True
                    break

            for branch in self.branches.values():
                for dest in branch.destinations.values():
                    if dest == node_name:
                        has_incoming = True
                        break
                if branch.default == node_name:
                    has_incoming = True
                    break

            # Check if node has any outgoing edges
            has_outgoing = False
            for src, _ in self.edges:
                if src == node_name:
                    has_outgoing = True
                    break

            for branch in self.branches.values():
                if branch.source_node == node_name:
                    has_outgoing = True
                    break

            # If node has neither incoming nor outgoing edges, it's an orphan
            if not has_incoming and not has_outgoing:
                orphans.append(node_name)

        return orphans

    def find_dangling_edges(self) -> List[Tuple[str, str]]:
        """Find edges pointing to non-existent nodes."""
        dangling = []

        # Check direct edges
        for src, dst in self.edges:
            if src != START and src not in self.nodes:
                dangling.append((src, dst))
            if dst != END and dst not in self.nodes:
                dangling.append((src, dst))

        # Check branch destinations
        for branch in self.branches.values():
            src = branch.source_node
            if src != START and src not in self.nodes:
                for dest in branch.destinations.values():
                    dangling.append((src, dest))
            else:
                for dest in branch.destinations.values():
                    if dest != END and dest not in self.nodes:
                        dangling.append((src, dest))

                if (
                    branch.default
                    and branch.default != END
                    and branch.default not in self.nodes
                ):
                    dangling.append((src, branch.default))

        return dangling

    def has_entry_point(self) -> bool:
        """Check if the graph has an entry point."""
        # Check for direct edges from START
        for src, _ in self.edges:
            if src == START:
                return True

        # Check for branches from START
        for branch in self.branches.values():
            if branch.source_node == START:
                return True

        return False

    # Add compile method that validates first
    def compile(self, raise_on_validation_error: bool = False) -> Any:
        """
        Validate and compile the graph to a runnable LangGraph StateGraph.

        Args:
            raise_on_validation_error: Whether to raise an exception on validation errors

        Returns:
            Compiled LangGraph StateGraph

        Raises:
            ValueError: If validation fails and raise_on_validation_error is True
        """
        # Run graph validation
        issues = self.validate_graph()

        if issues:
            from rich.console import Console

            console = Console()

            console.print("\n[bold red]Graph Validation Issues:[/bold red]")
            for issue in issues:
                console.print(f"[red]- {issue}[/red]")

            if raise_on_validation_error:
                raise ValueError(f"Graph validation failed with {len(issues)} issues")

            console.print(
                "\n[yellow]Proceeding with compilation despite validation issues[/yellow]"
            )

        # Convert to LangGraph
        graph = self.to_langgraph()

        # Compile the graph
        compiled_graph = graph.compile()

        return compiled_graph

    def add_boolean_conditional_edges(
        self,
        source_node: str,
        condition: Callable[[Any], bool],
        true_destination: str,
        false_destination: str = END,
        also_accept_strings: bool = True,
    ) -> "BaseGraph":
        """
        Add conditional edges that explicitly handle boolean results.

        This is a convenience method for the common case where you have a condition
        that returns True/False and you want clear routing.

        Args:
            source_node: Source node name
            condition: Function that returns True or False
            true_destination: Where to go when condition returns True
            false_destination: Where to go when condition returns False
            also_accept_strings: Whether to also accept string equivalents like 'has_tool_calls'/'no_tool_calls'

        Returns:
            Self for method chaining

        Example:
            ```python
            graph.add_boolean_conditional_edges(
                'agent_node',
                has_tool_calls,  # Function that returns True/False
                'validation',    # Go here when True
                END             # Go here when False
            )
            ```
        """
        # ALWAYS start with boolean keys as primary routing
        destinations = {True: true_destination, False: false_destination}

        # If requested, also add string-based keys for common patterns
        if also_accept_strings:
            # Add common string patterns that might be returned instead of booleans
            if (
                "tool" in true_destination.lower()
                or "validation" in true_destination.lower()
            ):
                destinations["has_tool_calls"] = true_destination
                destinations["no_tool_calls"] = false_destination
                destinations["has_tools"] = true_destination
                destinations["no_tools"] = false_destination

            # Add generic positive/negative strings
            destinations["yes"] = true_destination
            destinations["no"] = false_destination
            destinations["true"] = true_destination
            destinations["false"] = false_destination

        logger.debug(f"Boolean conditional edges destinations: {destinations}")

        return self.add_conditional_edges(
            source_node=source_node,
            condition=condition,
            destinations=destinations,
            default=None,  # No default since we have explicit True/False handling
        )

    def debug_conditional_routing(self, source_node: str) -> None:
        """
        Debug conditional routing for a specific node.

        This shows how boolean and string results will be routed for debugging purposes.

        Args:
            source_node: The node to debug routing for
        """
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table

        console = Console()

        # Find all branches for this node
        node_branches = self.get_branches_for_node(source_node)

        if not node_branches:
            console.print(
                f"[yellow]No conditional routing found for node '{source_node}'[/yellow]"
            )
            return

        console.print(
            f"\n[bold blue]Conditional Routing Debug for '{source_node}'[/bold blue]"
        )

        for i, branch in enumerate(node_branches):
            console.print(f"\n[cyan]Branch {i+1}: {branch.name}[/cyan]")

            # Create a table showing the routing
            table = Table(
                title="Routing Map", show_header=True, header_style="bold magenta"
            )
            table.add_column("Condition Result", style="cyan")
            table.add_column("Destination", style="green")
            table.add_column("Type", style="yellow")

            # Show all destinations
            for condition_result, destination in branch.destinations.items():
                result_type = (
                    "Boolean" if isinstance(condition_result, bool) else "String"
                )
                table.add_row(str(condition_result), destination, result_type)

            # Show default if exists
            if branch.default:
                table.add_row("(default)", branch.default, "Default")

            console.print(table)

            # Show function info if available
            if branch.function:
                func_name = getattr(branch.function, "__name__", "Unknown")
                console.print(f"[dim]Condition function: {func_name}[/dim]")

        # Show helpful tips
        tips_panel = Panel.fit(
            "[bold]Tips for Boolean Routing:[/bold]\n"
            "• Functions returning True/False will use boolean keys (True, False)\n"
            "• String-based keys automatically get boolean fallbacks\n"
            "• Use graph.add_boolean_conditional_edges() for explicit True/False routing\n"
            "• Enable debug logging: logger.setLevel(logging.DEBUG)",
            title="💡 Routing Tips",
            border_style="blue",
        )
        console.print(tips_panel)


# Utility functions for common graph operations
def has_tool_calls_fixed(state) -> bool:
    """
    FIXED VERSION: Check if the last AI message has tool calls.

    This function properly checks for tool calls in various message formats
    and handles edge cases that the original function missed.

    Args:
        state: The state object containing messages

    Returns:
        bool: True if the last AI message has tool calls, False otherwise
    """
    from langchain_core.messages import AIMessage

    # Get messages from state
    messages = None
    if hasattr(state, "messages"):
        messages = state.messages
    elif isinstance(state, dict) and "messages" in state:
        messages = state["messages"]
    else:
        logger.debug("No messages found in state")
        return False

    # Check if messages exist and are not empty
    if not messages:
        logger.debug("Messages list is empty")
        return False

    # Get the last message
    last_msg = messages[-1]
    logger.debug(f"Last message type: {type(last_msg)}")

    # Check if it's an AIMessage
    if not isinstance(last_msg, AIMessage):
        logger.debug(f"Last message is not AIMessage, it's {type(last_msg)}")
        return False

    # Check for tool_calls attribute first (most common case)
    if hasattr(last_msg, "tool_calls"):
        tool_calls = getattr(last_msg, "tool_calls", None)
        logger.debug(f"tool_calls attribute: {tool_calls}")

        # Check if tool_calls exists and is non-empty
        if tool_calls:
            logger.debug(f"Found {len(tool_calls)} tool calls")
            return True
        else:
            logger.debug("tool_calls attribute exists but is empty/None")

    # Check additional_kwargs as fallback
    if hasattr(last_msg, "additional_kwargs"):
        additional_kwargs = getattr(last_msg, "additional_kwargs", {})
        if isinstance(additional_kwargs, dict) and "tool_calls" in additional_kwargs:
            tool_calls_in_kwargs = additional_kwargs["tool_calls"]
            logger.debug(f"tool_calls in additional_kwargs: {tool_calls_in_kwargs}")

            if tool_calls_in_kwargs:
                logger.debug(
                    f"Found {len(tool_calls_in_kwargs)} tool calls in additional_kwargs"
                )
                return True
            else:
                logger.debug("tool_calls in additional_kwargs is empty/None")

    logger.debug("No tool calls found")
    return False


def create_debug_has_tool_calls(original_func):
    """
    Create a debug wrapper around a has_tool_calls function to help diagnose issues.

    Args:
        original_func: The original has_tool_calls function

    Returns:
        A wrapped function with detailed debugging
    """

    def debug_wrapper(state):
        from langchain_core.messages import AIMessage

        print("\n=== DEBUGGING has_tool_calls ===")
        print(f"State type: {type(state)}")

        # Check state structure
        if hasattr(state, "messages"):
            messages = state.messages
            print(f"State.messages: {len(messages)} messages")
        elif isinstance(state, dict) and "messages" in state:
            messages = state["messages"]
            print(f"State['messages']: {len(messages)} messages")
        else:
            print("ERROR: No messages found in state!")
            return False

        if not messages:
            print("ERROR: Messages list is empty!")
            return False

        # Examine the last message
        last_msg = messages[-1]
        print(f"Last message type: {type(last_msg)}")
        print(f"Last message content preview: {str(last_msg)[:200]}...")

        if isinstance(last_msg, AIMessage):
            print("✓ Last message is AIMessage")

            # Check tool_calls attribute
            if hasattr(last_msg, "tool_calls"):
                tool_calls = getattr(last_msg, "tool_calls", None)
                print(f"tool_calls attribute: {tool_calls}")
                print(f"tool_calls type: {type(tool_calls)}")
                print(f"tool_calls bool: {bool(tool_calls)}")

                if tool_calls:
                    print(f"✓ Found {len(tool_calls)} tool calls")
                    for i, call in enumerate(tool_calls):
                        print(f"  Tool call {i}: {call}")
                else:
                    print("✗ tool_calls is empty/None")
            else:
                print("✗ No tool_calls attribute")

            # Check additional_kwargs
            if hasattr(last_msg, "additional_kwargs"):
                additional_kwargs = getattr(last_msg, "additional_kwargs", {})
                print(f"additional_kwargs: {additional_kwargs}")

                if "tool_calls" in additional_kwargs:
                    tool_calls_kwargs = additional_kwargs["tool_calls"]
                    print(f"tool_calls in additional_kwargs: {tool_calls_kwargs}")
                    print(f"tool_calls_kwargs bool: {bool(tool_calls_kwargs)}")
                else:
                    print("✗ No tool_calls in additional_kwargs")
            else:
                print("✗ No additional_kwargs")
        else:
            print(f"✗ Last message is not AIMessage: {type(last_msg)}")

        # Call original function and compare
        original_result = original_func(state)
        fixed_result = has_tool_calls_fixed(state)

        print(f"Original function result: {original_result}")
        print(f"Fixed function result: {fixed_result}")

        if original_result != fixed_result:
            print(f"⚠️  MISMATCH! Original: {original_result}, Fixed: {fixed_result}")
        else:
            print(f"✓ Results match: {original_result}")

        print("=== END DEBUGGING ===\n")

        return original_result

    return debug_wrapper
