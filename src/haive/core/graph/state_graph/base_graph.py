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
from haive.core.graph.branches import Branch
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


# Edge types
# Simple edge: (source_node_name, target_node_name)
SimpleEdge = Tuple[str, str]


# Conditional edge with branch reference
class ConditionalEdge(BaseModel):
    """Conditional edge that uses a branch for routing."""

    source: str
    branch_id: str
    targets: Dict[str, str]  # condition result -> target node

    model_config = {"frozen": True}


# Complete edge type
Edge = Union[SimpleEdge, ConditionalEdge]


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
    - Edge management (direct and conditional)
    - Branch management
    - Graph validation
    - Serialization
    """

    # Unique identifier and metadata
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Core graph components
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
        for edge in self.edges:
            if isinstance(edge, tuple):
                source, target = edge
                if source not in self.nodes and source != START_NODE:
                    raise ValueError(f"Edge source '{source}' not found in nodes")
                if target not in self.nodes and target != END_NODE:
                    raise ValueError(f"Edge target '{target}' not found in nodes")
            elif isinstance(edge, ConditionalEdge):
                source = edge.source
                branch_id = edge.branch_id

                if source not in self.nodes and source != START_NODE:
                    raise ValueError(f"Edge source '{source}' not found in nodes")
                if branch_id not in self.branches:
                    raise ValueError(f"Branch '{branch_id}' not found in branches")

                # Validate targets
                for target in edge.targets.values():
                    if target not in self.nodes and target != END_NODE:
                        raise ValueError(f"Edge target '{target}' not found in nodes")

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
                        "metadata": {"engine": node_like},
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
                        "metadata": {"object": node_like},
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

        # Remove associated edges
        self.edges = [
            edge
            for edge in self.edges
            if (
                isinstance(edge, tuple)
                and edge[0] != node_name
                and edge[1] != node_name
            )
            or (isinstance(edge, ConditionalEdge) and edge.source != node_name)
        ]

        # Remove associated branches
        branch_ids_to_remove = []
        for branch_id, branch in self.branches.items():
            if branch.source_node == node_name:
                branch_ids_to_remove.append(branch_id)

        for branch_id in branch_ids_to_remove:
            del self.branches[branch_id]

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
        preserve_edges: bool = True,
    ) -> "BaseGraph":
        """
        Replace a node while optionally preserving its edges.

        Args:
            node_name: Name of the node to replace
            new_node: New node to insert
            preserve_edges: Whether to preserve existing edges

        Returns:
            Self for method chaining
        """
        if node_name not in self.nodes:
            raise ValueError(f"Node '{node_name}' not found in graph")

        # Store existing edges if needed
        incoming_edges = []
        outgoing_edges = []
        conditional_edges = []

        if preserve_edges:
            for edge in self.edges:
                if isinstance(edge, tuple):
                    source, target = edge
                    if source == node_name:
                        outgoing_edges.append((source, target))
                    elif target == node_name:
                        incoming_edges.append((source, target))
                elif isinstance(edge, ConditionalEdge):
                    if edge.source == node_name:
                        conditional_edges.append(edge)

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
        else:
            # For other types, use the specified name
            self.add_node(node_name, new_node)

        # Restore edges if needed
        if preserve_edges:
            # Restore direct edges
            for source, target in incoming_edges:
                self.add_edge(source, node_name)

            for source, target in outgoing_edges:
                self.add_edge(node_name, target)

            # Restore conditional edges
            for edge in conditional_edges:
                branch = self.branches.get(edge.branch_id)
                if branch:
                    self.add_conditional_edge(node_name, branch, edge.targets)

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
        Insert a new node after an existing node, redirecting all outgoing edges.

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
        conditional_edges = []

        for edge in self.edges:
            if isinstance(edge, tuple):
                source, target = edge
                if source == target_node:
                    outgoing_edges.append((source, target))
            elif isinstance(edge, ConditionalEdge):
                if edge.source == target_node:
                    conditional_edges.append(edge)

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

        # Remove original outgoing edges
        for edge in outgoing_edges:
            self.remove_edge(edge[0], edge[1])

        # Add edge from target to new node
        self.add_edge(target_node, new_node_name)

        # Add edges from new node to original targets
        for _, target in outgoing_edges:
            self.add_edge(new_node_name, target)

        # Handle conditional edges
        for edge in conditional_edges:
            # Remove the original conditional edge
            self.edges = [e for e in self.edges if e != edge]

            # Add a new edge from new node to targets
            branch = self.branches.get(edge.branch_id)
            if branch:
                self.add_conditional_edge(new_node_name, branch, edge.targets)

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
        Insert a new node before an existing node, redirecting all incoming edges.

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

        # Get incoming edges to target node
        incoming_edges = []

        for edge in self.edges:
            if isinstance(edge, tuple):
                source, target = edge
                if target == target_node:
                    incoming_edges.append((source, target))

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
        # Get all nodes connected from START
        start_edges = [
            edge
            for edge in self.edges
            if (isinstance(edge, tuple) and edge[0] == START_NODE)
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

        # Remove existing START edges
        for edge in start_edges:
            self.remove_edge(START_NODE, edge[1])

        # Add edge from START to prelude
        self.add_edge(START_NODE, prelude_name)

        # Add edges from prelude to original start nodes
        for _, target in start_edges:
            self.add_edge(prelude_name, target)

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
        # Get all nodes connecting to END
        end_edges = [
            edge
            for edge in self.edges
            if (isinstance(edge, tuple) and edge[1] == END_NODE)
        ]

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

    def add_conditional_edge(
        self, source: str, branch: Union[Branch, str], targets: Dict[str, str]
    ) -> "BaseGraph":
        """
        Add a conditional edge to the graph.

        Args:
            source: Source node name
            branch: Branch object or branch ID
            targets: Mapping of condition results to target nodes

        Returns:
            Self for method chaining
        """
        # Validate source
        if source != START_NODE and source not in self.nodes:
            raise ValueError(f"Source node '{source}' not found in graph")

        # Get or create branch
        if isinstance(branch, Branch):
            # Add the branch if not already present
            if branch.id not in self.branches:
                self.branches[branch.id] = branch
            branch_id = branch.id
        else:
            # Use existing branch
            branch_id = branch
            if branch_id not in self.branches:
                raise ValueError(f"Branch '{branch_id}' not found in graph")

        # Validate targets
        for condition, target in targets.items():
            if target != END_NODE and target not in self.nodes:
                raise ValueError(f"Target node '{target}' not found in graph")

        # Create edge
        edge = ConditionalEdge(source=source, branch_id=branch_id, targets=targets)

        # Add edge
        self.edges.append(edge)
        logger.debug(f"Added conditional edge from {source} using branch '{branch_id}'")

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
        # Handle direct edges
        if target:
            # Remove specific direct edge
            self.edges = [
                edge
                for edge in self.edges
                if not (
                    isinstance(edge, tuple) and edge[0] == source and edge[1] == target
                )
            ]
            logger.debug(f"Removed edge {source} -> {target} from graph '{self.name}'")
        else:
            # Remove all edges from source
            self.edges = [
                edge
                for edge in self.edges
                if (isinstance(edge, tuple) and edge[0] != source)
                or (isinstance(edge, ConditionalEdge) and edge.source != source)
            ]
            logger.debug(f"Removed all edges from {source} in graph '{self.name}'")

        self.updated_at = datetime.now()
        return self

    def get_edges(
        self, source: Optional[str] = None, target: Optional[str] = None
    ) -> List[Edge]:
        """
        Get edges matching criteria.

        Args:
            source: Filter by source node
            target: Filter by target node

        Returns:
            List of matching edges
        """
        result = []

        for edge in self.edges:
            if isinstance(edge, tuple):
                edge_source, edge_target = edge

                # Apply filters
                source_match = source is None or edge_source == source
                target_match = target is None or edge_target == target

                if source_match and target_match:
                    result.append(edge)

            elif isinstance(edge, ConditionalEdge) and source is not None:
                if edge.source == source:
                    result.append(edge)

        return result

    # Branch management methods
    def add_branch(
        self,
        branch_or_name: Union[Branch, str],
        source_node: Optional[str] = None,
        condition: Optional[Any] = None,
        routes: Optional[Dict[str, str]] = None,
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
            if (
                branch.source_node != START_NODE
                and branch.source_node not in self.nodes
            ):
                raise ValueError(
                    f"Source node '{branch.source_node}' not found in graph"
                )

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

            # Create branch with original Branch class
            from haive.core.graph.branches import Branch

            # Create the branch
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

            # Create branch
            branch = Branch(**branch_data)
            self.branches[branch.id] = branch

        logger.debug(f"Added branch '{branch.name}' to graph '{self.name}'")
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
        from haive.core.graph.branches import Branch

        # Create branch
        branch = Branch(
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

        logger.debug(f"Added function branch '{branch.name}' to graph '{self.name}'")
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
        from haive.core.graph.branches import Branch

        # Create branch
        branch = Branch(
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

        logger.debug(f"Added key-value branch '{branch.name}' to graph '{self.name}'")
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

        # Remove associated conditional edges
        self.edges = [
            edge
            for edge in self.edges
            if not (isinstance(edge, ConditionalEdge) and edge.branch_id == branch_id)
        ]

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

        self.updated_at = datetime.now()
        return self

    def replace_branch(self, branch_id: str, new_branch: Branch) -> "BaseGraph":
        """
        Replace a branch while preserving connections.

        Args:
            branch_id: ID of the branch to replace
            new_branch: New branch to insert

        Returns:
            Self for method chaining
        """
        if branch_id not in self.branches:
            raise ValueError(f"Branch '{branch_id}' not found in graph")

        # Get the original branch
        original_branch = self.branches[branch_id]

        # Store any edges using this branch
        conditional_edges = [
            edge
            for edge in self.edges
            if isinstance(edge, ConditionalEdge) and edge.branch_id == branch_id
        ]

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
        Get all branches associated with a node.

        Args:
            node_name: Name of the node

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
            start_connections = [
                edge
                for edge in self.edges
                if (isinstance(edge, tuple) and edge[0] == START_NODE)
                or (isinstance(edge, ConditionalEdge) and edge.source == START_NODE)
            ]

            if not start_connections:
                logger.warning(
                    f"Graph '{self.name}' has no connections from START node"
                )
                return False

            # Check for END node connections
            end_connections = [
                edge
                for edge in self.edges
                if (isinstance(edge, tuple) and edge[1] == END_NODE)
            ]

            end_in_branches = False
            for branch in self.branches.values():
                if (
                    END_NODE in branch.destinations.values()
                    or branch.default == END_NODE
                ):
                    end_in_branches = True
                    break

            if not end_connections and not end_in_branches:
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
        incoming = []
        outgoing = []

        # Check direct edges
        for edge in self.edges:
            if isinstance(edge, tuple):
                source, target = edge
                if target == node_name:
                    incoming.append(source)
                if source == node_name:
                    outgoing.append(target)

        # Check conditional edges
        for edge in self.edges:
            if isinstance(edge, ConditionalEdge):
                if edge.source == node_name:
                    outgoing.extend(list(set(edge.targets.values())))

                # Check if node is a target
                for target in edge.targets.values():
                    if target == node_name:
                        incoming.append(edge.source)

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

            # Find all outgoing connections
            for edge in self.edges:
                if isinstance(edge, tuple) and edge[0] == current:
                    queue.append(edge[1])
                elif isinstance(edge, ConditionalEdge) and edge.source == current:
                    queue.extend(edge.targets.values())

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

            # Get all outgoing connections
            for edge in self.edges:
                if isinstance(edge, tuple) and edge[0] == node:
                    next_node = edge[1]
                    if next_node not in visited:
                        path.append(next_node)
                        dfs(next_node, path)
                        path.pop()
                elif isinstance(edge, ConditionalEdge) and edge.source == node:
                    for target in edge.targets.values():
                        if target not in visited:
                            path.append(target)
                            dfs(target, path)
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
        nodes = []

        for edge in self.edges:
            if isinstance(edge, tuple) and edge[0] == START_NODE:
                nodes.append(edge[1])
            elif isinstance(edge, ConditionalEdge) and edge.source == START_NODE:
                nodes.extend(edge.targets.values())

        return list(set(nodes))

    def get_end_nodes(self) -> List[str]:
        """
        Get all nodes directly connecting to END.

        Returns:
            List of node names
        """
        nodes = []

        for edge in self.edges:
            if isinstance(edge, tuple) and edge[1] == END_NODE:
                nodes.append(edge[0])

        # Check branches too
        for branch in self.branches.values():
            if END_NODE in branch.destinations.values() or branch.default == END_NODE:
                nodes.append(branch.source_node)

        return list(set(nodes))

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

            # Add edges
            for edge in self.edges:
                if isinstance(edge, tuple):
                    source, target = edge
                    # Convert START/END constants
                    source_actual = START if source == START_NODE else source
                    target_actual = END if target == END_NODE else target

                    graph_builder.add_edge(source_actual, target_actual)

            # Add conditional edges
            for branch_id, branch in self.branches.items():
                # Find conditional edges using this branch
                for edge in self.edges:
                    if (
                        isinstance(edge, ConditionalEdge)
                        and edge.branch_id == branch_id
                    ):
                        # Create the edge function
                        def create_edge_func(branch_obj):
                            def edge_func(state):
                                return branch_obj.evaluate(state)

                            return edge_func

                        # Add the conditional edge
                        graph_builder.add_conditional_edges(
                            branch.source_node,
                            create_edge_func(branch),
                            branch.destinations,
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
                        routes = {}
                        if hasattr(branch_obj, "ends"):
                            for condition, target in branch_obj.ends.items():
                                # Convert __end__ to END_NODE
                                target_actual = (
                                    END_NODE if target == "__end__" else target
                                )
                                routes[condition] = target_actual

                        # Create branch
                        from haive.core.graph.branches import Branch

                        branch = Branch(
                            name=f"branch_{condition_name}_{uuid.uuid4().hex[:6]}",
                            source_node=source_node,
                            destinations=routes,
                            default="END",
                        )

                        # Add the branch and conditional edge
                        graph.add_branch(branch)
                        graph.add_conditional_edge(source_node, branch, routes)

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
        for edge in self.edges:
            if isinstance(edge, tuple):
                source, target = edge
                lines.append(f"    {source} --> {target};")

        # Add conditional edges
        if self.branches:
            lines.append("    %% Conditional edges")

            for edge in self.edges:
                if isinstance(edge, ConditionalEdge):
                    source = edge.source
                    branch_id = edge.branch_id
                    branch = self.branches.get(branch_id)

                    if branch:
                        lines.append(f"    %% Branch: {branch.name}")
                        for condition, target in edge.targets.items():
                            lines.append(f'    {source} -->|"{condition}"| {target};')

        return "\n".join(lines)
