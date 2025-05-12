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
from langgraph.types import Command, RetryPolicy, Send
from pydantic import BaseModel, Field, model_validator

# Import Branch implementation
from haive.core.graph.branches.branch import Branch
from haive.core.graph.branches.types import BranchMode, BranchResult, ComparisonType
from haive.core.graph.common.references import CallableReference
from haive.core.graph.common.types import ConfigLike, NodeOutput, StateLike
from haive.core.schema.state_schema import StateSchema

# Setup logging
logger = logging.getLogger(__name__)

# Special node names
START_NODE = "START"
END_NODE = "END"


# Node and branch types
class NodeType(str, Enum):
    """Types of nodes in a graph."""

    ENGINE = "engine"
    CALLABLE = "callable"
    TOOL_NODE = "tool_node"
    VALIDATION = "validation"
    SUBGRAPH = "subgraph"


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
    nodes: Dict[str, Node] = Field(default_factory=dict)
    edges: List[Edge] = Field(default_factory=list)
    branches: Dict[str, Branch] = Field(default_factory=dict)

    # Configuration
    state_schema: Optional[Any] = None
    default_config: Optional[RunnableConfig] = None

    # Tracking fields
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def validate_graph(self) -> "BaseGraph":
        """Validate the graph structure."""
        # Check that node names are unique
        node_names = {node.name for node in self.nodes.values()}
        if len(node_names) != len(self.nodes):
            raise ValueError("Node names must be unique")

        # Update node dict if needed (ensure name-based keys)
        nodes_by_name = {}
        for node in self.nodes.values():
            nodes_by_name[node.name] = node
        self.nodes = nodes_by_name

        # Validate edges
        for source, target in self.edges:
            if source not in self.nodes and source != START_NODE:
                raise ValueError(f"Edge source '{source}' not found in nodes")
            if target not in self.nodes and target != END_NODE:
                raise ValueError(f"Edge target '{target}' not found in nodes")

        # Validate branches
        for branch_id, branch in self.branches.items():
            if (
                branch.source_node not in self.nodes
                and branch.source_node != START_NODE
            ):
                raise ValueError(
                    f"Branch source '{branch.source_node}' not found in nodes"
                )

            # Validate targets in destinations
            for target in branch.destinations.values():
                if target != END_NODE and target not in self.nodes:
                    raise ValueError(f"Branch target '{target}' not found in nodes")

        return self

    # Node management methods
    def add_node(
        self,
        node_or_name: Union[Node, Dict[str, Any], str],
        node_like: Optional[Any] = None,
        **kwargs,
    ) -> "BaseGraph":
        """
        Add a node to the graph with flexible input options.

        Args:
            node_or_name: Node object, dictionary, or node name
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

            # Prepare node data
            node_data = {"name": name}

            # Determine node type and prepare data
            if node_like is None:
                # Use any provided kwargs
                node_data.update(kwargs)
                node_data.setdefault("node_type", NodeType.CALLABLE)
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
        else:
            raise TypeError(f"Unsupported node type: {type(node_or_name)}")

        # Check for existing node
        if node_obj.name in self.nodes:
            raise ValueError(f"Node '{node_obj.name}' already exists in the graph")

        # Add the node
        self.nodes[node_obj.name] = node_obj
        logger.debug(f"Added node '{node_obj.name}' to graph '{self.name}'")

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
                branch.default = END_NODE

        logger.debug(f"Removed node '{node_name}' from graph '{self.name}'")
        self.updated_at = datetime.now()
        return self

    def get_node(self, node_name: str) -> Optional[Node]:
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

        # Apply updates
        for key, value in updates.items():
            if hasattr(node, key):
                setattr(node, key, value)

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
        elif callable(new_node):
            # Direct callable
            self.add_node(node_name, new_node)
            # Ensure the callable is in metadata
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
        if target_node not in self.nodes and target_node != START_NODE:
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
            (source, target) for source, target in self.edges if source == START_NODE
        ]

        # Get all nodes connected from START via branches
        start_branches = [
            branch
            for branch in self.branches.values()
            if branch.source_node == START_NODE
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
        else:
            # Generate a name
            prelude_name = f"prelude_{uuid.uuid4().hex[:6]}"
            self.add_node(prelude_name, prelude_node, **kwargs)

        # Remove existing START direct edges
        for edge in start_edges:
            self.remove_edge(START_NODE, edge[1])

        # Add edge from START to prelude
        self.add_edge(START_NODE, prelude_name)

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
        end_edges = [
            (source, target) for source, target in self.edges if target == END_NODE
        ]

        # Get all branches pointing to END
        end_branch_destinations = []
        for branch in self.branches.values():
            for condition, target in branch.destinations.items():
                if target == END_NODE:
                    end_branch_destinations.append((branch, condition))

            # Check default too
            if branch.default == END_NODE:
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
        else:
            # Generate a name
            postlude_name = f"postlude_{uuid.uuid4().hex[:6]}"
            self.add_node(postlude_name, postlude_node, **kwargs)

        # Remove existing END edges
        for edge in end_edges:
            self.remove_edge(edge[0], END_NODE)

        # Add edges from original end nodes to postlude
        for source, _ in end_edges:
            self.add_edge(source, postlude_name)

        # Add edge from postlude to END
        self.add_edge(postlude_name, END_NODE)

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
            else:
                # Generate a name
                node_name = f"seq_node_{i}_{uuid.uuid4().hex[:6]}"
                self.add_node(node_name, node, **kwargs)
                node_names.append(node_name)

        # Connect START if requested
        if connect_start and node_names:
            self.add_edge(START_NODE, node_names[0])

        # Connect nodes in sequence
        for i in range(len(node_names) - 1):
            self.add_edge(node_names[i], node_names[i + 1])

        # Connect END if requested
        if connect_end and node_names:
            self.add_edge(node_names[-1], END_NODE)

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
        if source_node not in self.nodes and source_node != START_NODE:
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

                if isinstance(branch_sequence[-1], str):
                    last_node = branch_sequence[-1]
                elif isinstance(branch_sequence[-1], Node):
                    last_node = branch_sequence[-1].name
                elif (
                    isinstance(branch_sequence[-1], dict)
                    and "name" in branch_sequence[-1]
                ):
                    last_node = branch_sequence[-1]["name"]

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
        if source != START_NODE and source not in self.nodes:
            raise ValueError(f"Source node '{source}' not found in graph")
        if target != END_NODE and target not in self.nodes:
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

    # Branch management methods
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

            if (
                branch.source_node != START_NODE
                and branch.source_node not in self.nodes
            ):
                raise ValueError(
                    f"Source node '{branch.source_node}' not found in graph"
                )

            # Validate destination nodes
            for dest in branch.destinations.values():
                if dest != END_NODE and dest not in self.nodes:
                    raise ValueError(f"Destination node '{dest}' not found in graph")

            # Validate default node
            if branch.default != END_NODE and branch.default not in self.nodes:
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
            if source_node != START_NODE and source_node not in self.nodes:
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
                    if dest != END_NODE and dest not in self.nodes:
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

    def add_function_branch(
        self,
        source_node: str,
        condition: Callable[[Any], Any],
        routes: Dict[Union[bool, str], str],
        default_route: str = "END",
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
        if source_node != START_NODE and source_node not in self.nodes:
            raise ValueError(f"Source node '{source_node}' not found in graph")

        # Validate target nodes
        for dest in routes.values():
            if dest != END_NODE and dest not in self.nodes:
                raise ValueError(f"Destination node '{dest}' not found in graph")

        # Validate default route
        if default_route != END_NODE and default_route not in self.nodes:
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
        false_dest: str = "END",
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
        if source_node != START_NODE and source_node not in self.nodes:
            raise ValueError(f"Source node '{source_node}' not found in graph")

        # Validate target nodes
        if true_dest != END_NODE and true_dest not in self.nodes:
            raise ValueError(f"True destination node '{true_dest}' not found in graph")

        if false_dest != END_NODE and false_dest not in self.nodes:
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
            if source != START_NODE and source not in self.nodes:
                raise ValueError(f"Updated source node '{source}' not found in graph")

        if "destinations" in updates:
            for dest in updates["destinations"].values():
                if dest != END_NODE and dest not in self.nodes:
                    raise ValueError(
                        f"Updated destination node '{dest}' not found in graph"
                    )

        if "default" in updates:
            default = updates["default"]
            if default != END_NODE and default not in self.nodes:
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
        if (
            new_branch.source_node != START_NODE
            and new_branch.source_node not in self.nodes
        ):
            raise ValueError(
                f"Source node '{new_branch.source_node}' not found in graph"
            )

        for dest in new_branch.destinations.values():
            if dest != END_NODE and dest not in self.nodes:
                raise ValueError(f"Destination node '{dest}' not found in graph")

        if new_branch.default != END_NODE and new_branch.default not in self.nodes:
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

    # Graph validation and metadata methods
    def validate(self) -> bool:
        """
        Validate the graph is properly connected.

        Returns:
            True if valid, False otherwise
        """
        try:
            # Check for START node connections
            start_edges = self.get_edges(source=START_NODE, include_branches=True)

            if not start_edges:
                logger.warning(
                    f"Graph '{self.name}' has no connections from START node"
                )
                return False

            # Check for END node connections
            end_edges = self.get_edges(target=END_NODE, include_branches=True)

            if not end_edges:
                logger.warning(f"Graph '{self.name}' has no connections to END node")
                return False

            # Check for cycles (advanced feature)
            # TODO: Implement cycle detection

            return True

        except Exception as e:
            logger.error(f"Error validating graph '{self.name}': {str(e)}")
            return False

    def get_node_dependencies(self, node_name: str) -> Dict[str, List[str]]:
        """
        Get dependencies for a node.

        Args:
            node_name: Name of the node

        Returns:
            Dictionary with 'in' and 'out' lists of connected nodes
        """
        # Get all edges involving this node
        all_edges = self.get_edges(include_branches=True)

        incoming = []
        outgoing = []

        # Process each edge
        for source, target in all_edges:
            if target == node_name:
                incoming.append(source)
            if source == node_name:
                outgoing.append(target)

        return {"in": list(set(incoming)), "out": list(set(outgoing))}

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

    # Graph traversal and analysis
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
            node for name, node in self.nodes.items() if fnmatch.fnmatch(name, pattern)
        ]

    def get_execution_paths(self) -> List[List[str]]:
        """
        Get all possible execution paths in the graph.

        Returns:
            List of node name sequences representing possible paths
        """
        # Use DFS to find all paths from START to END
        paths = []
        visited = set()

        def dfs(node: str, path: List[str]):
            if node == END_NODE:
                paths.append(path.copy())
                return

            visited.add(node)

            # Get all outgoing connections including branches
            for src, dest in self.get_edges(source=node, include_branches=True):
                if dest not in visited:
                    path.append(dest)
                    dfs(dest, path)
                    path.pop()

            visited.remove(node)

        # Start from nodes connected to START
        start_nodes = self.get_start_nodes()
        for start_node in start_nodes:
            dfs(start_node, [start_node])

        return paths

    # Utility methods
    def get_start_nodes(self) -> List[str]:
        """
        Get all nodes directly connected from START.

        Returns:
            List of node names
        """
        # Get all outgoing connections from START
        start_edges = self.get_edges(source=START_NODE, include_branches=True)
        return list({edge[1] for edge in start_edges})

    def get_end_nodes(self) -> List[str]:
        """
        Get all nodes directly connecting to END.

        Returns:
            List of node names
        """
        # Get all incoming connections to END
        end_edges = self.get_edges(target=END_NODE, include_branches=True)
        return list({edge[0] for edge in end_edges})

    def get_orphan_nodes(self) -> List[str]:
        """
        Get nodes with no incoming or outgoing connections.

        Returns:
            List of orphaned node names
        """
        orphans = []

        for name in self.nodes:
            deps = self.get_node_dependencies(name)
            if not deps["in"] and not deps["out"]:
                orphans.append(name)

        return orphans

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
            from langgraph.graph import END, START, StateGraph

            # Create StateGraph
            graph_builder = StateGraph(state_schema or self.state_schema or dict)

            # Add nodes
            for name, node in self.nodes.items():
                # Extract action from metadata
                action = None
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
                if node.retry_policy:
                    retry = node.retry_policy

                # Convert command_goto if present
                destinations = None
                if node.command_goto:
                    if isinstance(node.command_goto, str):
                        destinations = (node.command_goto,)
                    elif isinstance(node.command_goto, list):
                        destinations = tuple(node.command_goto)

                # Add the node
                if action:
                    graph_builder.add_node(
                        name,
                        action,
                        metadata=node.metadata,
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
                # Convert START/END constants
                source_actual = START if source == START_NODE else source
                target_actual = END if target == END_NODE else target

                graph_builder.add_edge(source_actual, target_actual)

            # Add branches
            for branch_id, branch in self.branches.items():
                # Create the edge function
                def create_edge_func(branch_obj):
                    def edge_func(state):
                        return branch_obj.evaluate(state)

                    return edge_func

                # Add the conditional edge
                source_node = branch.source_node
                if source_node != START_NODE and source_node in self.nodes:
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
                    # Convert special nodes
                    source_actual = START_NODE if source == "__start__" else source
                    target_actual = END_NODE if target == "__end__" else target

                    # Add the edge
                    graph.add_edge(source_actual, target_actual)

            # Convert branches/conditional edges
            if hasattr(state_graph, "branches"):
                for source_node, conditions in state_graph.branches.items():
                    for condition_name, branch_obj in conditions.items():
                        # Extract routes
                        destinations = {}
                        if hasattr(branch_obj, "ends"):
                            for condition, target in branch_obj.ends.items():
                                # Convert __end__ to END_NODE
                                target_actual = (
                                    END_NODE if target == "__end__" else target
                                )
                                destinations[condition] = target_actual

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
                            default=END_NODE,
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
            node_type = (
                node.node_type.value if hasattr(node, "node_type") else "unknown"
            )
            lines.append(f'    {name}["{name} ({node_type})"];')

        # Add special nodes
        lines.append(f'    {START_NODE}["{START_NODE}"] style fill:#5D8AA8;')
        lines.append(f'    {END_NODE}["{END_NODE}"] style fill:#FF6347;')

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
