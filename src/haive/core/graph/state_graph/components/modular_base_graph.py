"""Modular BaseGraph implementation using composition over inheritance.

This module provides a refactored BaseGraph that uses composition to organize
functionality into focused, testable components following the coding style guide.
"""

import logging
import uuid
from collections.abc import Callable
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from haive.core.graph.common.types import NodeType
from haive.core.graph.state_graph.components.base_component import ComponentRegistry
from haive.core.graph.state_graph.components.branch_manager import BranchManager
from haive.core.graph.state_graph.components.edge_manager import EdgeManager
from haive.core.graph.state_graph.components.node_manager import NodeManager
from haive.core.graph.state_graph.validation_mixin import ValidationMixin

logger = logging.getLogger(__name__)


class ModularBaseGraph(BaseModel, ValidationMixin):
    """Modular BaseGraph implementation using composition over inheritance.

    This class refactors the original BaseGraph2 by breaking down its 4,517 lines
    into focused, testable components. Each component handles a specific aspect
    of graph management, following the single responsibility principle.

    Architecture:
        - NodeManager: Handles all node operations
        - EdgeManager: Handles direct edge operations
        - BranchManager: Handles conditional routing and branches
        - ComponentRegistry: Manages component lifecycle

    Args:
        name: Unique identifier for the graph
        description: Optional description of the graph's purpose
        state_schema: Schema for graph state
        metadata: Additional graph metadata

    Example:
        Creating and using a modular graph::

            graph = ModularBaseGraph(name="my_workflow")

            # Add nodes
            graph.add_node("start", start_function)
            graph.add_node("process", process_function)

            # Add edges
            graph.add_edge("start", "process")

            # Add conditional routing
            graph.add_conditional_edges("process", router_function, {
                "success": "finish",
                "error": "retry"
            })

            # Compile and use
            compiled = graph.compile()
    """

    # Core graph data
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., description="Unique graph identifier")
    description: str | None = Field(None, description="Graph description")

    # Graph structure
    nodes: dict[str, Any] = Field(default_factory=dict, description="Graph nodes")
    edges: list[tuple[str, str]] = Field(
        default_factory=list, description="Direct edges"
    )
    branches: dict[str, Any] = Field(
        default_factory=dict, description="Conditional branches"
    )

    # Entry and exit points
    entry_point: str | None = Field(None, description="Main entry point")
    finish_point: str | None = Field(None, description="Main finish point")
    conditional_entries: dict[str, dict[str, Any]] = Field(default_factory=dict)
    conditional_exits: dict[str, dict[str, Any]] = Field(default_factory=dict)

    # Schema and metadata
    state_schema: type | None = Field(None, description="Graph state schema")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    # Pydantic v2 configuration
    model_config = {
        "extra": "forbid",
        "validate_assignment": True,
        "arbitrary_types_allowed": True,
    }

    # Private attributes (not Pydantic fields)
    _component_registry: ComponentRegistry | None = None
    _node_manager: NodeManager | None = None
    _edge_manager: EdgeManager | None = None
    _branch_manager: BranchManager | None = None

    def __init__(self, **data) -> None:
        """Initialize modular graph with components."""
        super().__init__(**data)
        self._initialize_components()

    def _initialize_components(self) -> None:
        """Initialize all graph components."""
        # Create component registry
        self._component_registry = ComponentRegistry()

        # Create and register components
        self._node_manager = NodeManager(self)
        self._edge_manager = EdgeManager(self)
        self._branch_manager = BranchManager(self)

        self._component_registry.register("nodes", self._node_manager)
        self._component_registry.register("edges", self._edge_manager)
        self._component_registry.register("branches", self._branch_manager)

        # Initialize all components
        self._component_registry.initialize_all()

        logger.debug(
            f"Initialized ModularBaseGraph '{self.name}' with {
                len(self._component_registry.get_all())
            } components"
        )

    # =================================================================
    # NODE MANAGEMENT - Delegate to NodeManager
    # =================================================================

    def add_node(
        self,
        node_or_name: Any | dict[str, Any] | str,
        node_like: Any | None = None,
        **kwargs,
    ) -> "ModularBaseGraph":
        """Add a node to the graph.

        Args:
            node_or_name: Node identifier or node object
            node_like: Optional node function/config when first arg is string
            **kwargs: Additional node properties

        Returns:
            Self for method chaining
        """
        self._node_manager.add_node(node_or_name, node_like, **kwargs)
        return self

    def remove_node(
        self, node_name: str, cleanup_edges: bool = True
    ) -> "ModularBaseGraph":
        """Remove a node from the graph.

        Args:
            node_name: Name of the node to remove
            cleanup_edges: Whether to remove associated edges

        Returns:
            Self for method chaining
        """
        self._node_manager.remove_node(node_name, cleanup_edges)
        return self

    def get_node(self, node_name: str) -> Any | None:
        """Get a node by name."""
        return self._node_manager.get_node(node_name)

    def update_node(self, node_name: str, **updates) -> "ModularBaseGraph":
        """Update properties of an existing node."""
        self._node_manager.update_node(node_name, **updates)
        return self

    def get_nodes_by_type(self, node_type: NodeType) -> list[str]:
        """Get all nodes of a specific type."""
        return self._node_manager.get_nodes_by_type(node_type)

    def get_node_count(self) -> int:
        """Get total number of nodes in the graph."""
        return self._node_manager.get_node_count()

    # =================================================================
    # EDGE MANAGEMENT - Delegate to EdgeManager
    # =================================================================

    def add_edge(
        self, source: str, target: str, validate_nodes: bool = True
    ) -> "ModularBaseGraph":
        """Add a direct edge between two nodes.

        Args:
            source: Name of the source node
            target: Name of the target node
            validate_nodes: Whether to validate that nodes exist

        Returns:
            Self for method chaining
        """
        self._edge_manager.add_edge(source, target, validate_nodes)
        return self

    def remove_edge(self, source: str, target: str | None = None) -> "ModularBaseGraph":
        """Remove edge(s) from the graph.

        Args:
            source: Name of the source node
            target: Name of the target node. If None, removes all edges from source

        Returns:
            Self for method chaining
        """
        self._edge_manager.remove_edge(source, target)
        return self

    def get_edges(
        self, source: str | None = None, target: str | None = None
    ) -> list[tuple[str, str]]:
        """Get edges matching the specified criteria."""
        return self._edge_manager.get_edges(source, target)

    def has_edge(self, source: str, target: str) -> bool:
        """Check if an edge exists between two nodes."""
        return self._edge_manager.has_edge(source, target)

    def get_edge_count(self) -> int:
        """Get total number of edges in the graph."""
        return self._edge_manager.get_edge_count()

    def find_dangling_edges(self) -> list[tuple[str, str]]:
        """Find edges that reference non-existent nodes."""
        return self._edge_manager.find_dangling_edges()

    # =================================================================
    # BRANCH MANAGEMENT - Delegate to BranchManager
    # =================================================================

    def add_conditional_edges(
        self,
        source_node: str,
        condition: Any | Callable | Any,
        destinations: str | list[str] | dict[bool | str | int, str] | None = None,
        default: str | Literal["END"] | None = "END",
        create_missing_nodes: bool = False,
    ) -> "ModularBaseGraph":
        """Add conditional edges from a source node.

        Args:
            source_node: Name of the source node
            condition: Condition function or Branch object
            destinations: Mapping of condition results to target nodes
            default: Default destination if condition doesn't match
            create_missing_nodes: Whether to create missing destination nodes

        Returns:
            Self for method chaining
        """
        self._branch_manager.add_conditional_edges(
            source_node, condition, destinations, default, create_missing_nodes
        )
        return self

    def add_function_branch(
        self,
        source_node: str,
        function: Callable,
        default_destination: str | Literal["END"] = "END",
    ) -> "ModularBaseGraph":
        """Add a function-based branch."""
        self._branch_manager.add_function_branch(
            source_node, function, default_destination
        )
        return self

    def add_key_value_branch(
        self,
        source_node: str,
        key: str,
        value_map: dict[Any, str],
        default_destination: str | Literal["END"] = "END",
    ) -> "ModularBaseGraph":
        """Add a key-value conditional branch."""
        self._branch_manager.add_key_value_branch(
            source_node, key, value_map, default_destination
        )
        return self

    def remove_branch(self, branch_id: str) -> "ModularBaseGraph":
        """Remove a branch from the graph."""
        self._branch_manager.remove_branch(branch_id)
        return self

    def get_branches_for_node(self, node_name: str) -> list[Any]:
        """Get all branches originating from a specific node."""
        return self._branch_manager.get_branches_for_node(node_name)

    def get_branch_count(self) -> int:
        """Get total number of branches in the graph."""
        return self._branch_manager.get_branch_count()

    # =================================================================
    # ENTRY/EXIT POINT MANAGEMENT
    # =================================================================

    def set_entry_point(self, node_name: str) -> "ModularBaseGraph":
        """Set the main entry point for the graph.

        Args:
            node_name: Name of the entry node

        Returns:
            Self for method chaining

        Raises:
            ValueError: If node doesn't exist
        """
        if node_name not in self.nodes:
            raise ValueError(f"Entry point node '{node_name}' not found in graph")

        self.entry_point = node_name
        self.updated_at = datetime.now()

        logger.debug(f"Set entry point to '{node_name}' in graph '{self.name}'")

        return self

    def set_finish_point(self, node_name: str) -> "ModularBaseGraph":
        """Set the main finish point for the graph.

        Args:
            node_name: Name of the finish node

        Returns:
            Self for method chaining

        Raises:
            ValueError: If node doesn't exist
        """
        if node_name not in self.nodes and node_name != "END":
            raise ValueError(f"Finish point node '{node_name}' not found in graph")

        self.finish_point = node_name
        self.updated_at = datetime.now()

        logger.debug(f"Set finish point to '{node_name}' in graph '{self.name}'")

        return self

    def has_entry_point(self) -> bool:
        """Check if graph has an entry point."""
        return self.entry_point is not None

    # =================================================================
    # VALIDATION AND ANALYSIS
    # =================================================================

    def validate_graph(self) -> list[str]:
        """Validate the entire graph structure.

        Returns:
            List of validation error messages. Empty list if valid.
        """
        errors = []

        # Validate all components
        component_errors = self._component_registry.validate_all()
        for component_name, component_errors_list in component_errors.items():
            for error in component_errors_list:
                errors.append(f"{component_name}: {error}")

        # Graph-level validation
        if not self.nodes:
            errors.append("Graph has no nodes")

        if not self.has_entry_point():
            errors.append("Graph has no entry point")

        # Check connectivity
        if self.nodes and not self.edges and not self.branches:
            errors.append("Graph nodes are not connected")

        return errors

    def is_valid(self) -> bool:
        """Check if graph is valid."""
        return len(self.validate_graph()) == 0

    # =================================================================
    # COMPILATION AND EXECUTION
    # =================================================================

    def compile(self, raise_on_validation_error: bool = False) -> Any:
        """Compile the graph into an executable form.

        Args:
            raise_on_validation_error: Whether to raise on validation errors

        Returns:
            Compiled graph ready for execution

        Raises:
            ValueError: If validation fails and raise_on_validation_error=True
        """
        # Validate graph before compilation
        validation_errors = self.validate_graph()
        if validation_errors:
            error_msg = f"Graph validation failed: {'; '.join(validation_errors)}"
            if raise_on_validation_error:
                raise ValueError(error_msg)
            logger.warning(error_msg)

        # TODO: Implement actual compilation to LangGraph
        # This would integrate with the existing LangGraph compilation logic
        logger.info(
            f"Compiling graph '{self.name}' with {self.get_node_count()} nodes and {
                self.get_edge_count()
            } edges"
        )

        # Placeholder - return self for now
        return self

    # =================================================================
    # UTILITY METHODS
    # =================================================================

    def get_graph_summary(self) -> dict[str, Any]:
        """Get comprehensive graph statistics and information.

        Returns:
            Dictionary containing graph metadata and statistics
        """
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "node_count": self.get_node_count(),
            "edge_count": self.get_edge_count(),
            "branch_count": self.get_branch_count(),
            "entry_point": self.entry_point,
            "finish_point": self.finish_point,
            "has_conditional_entries": len(self.conditional_entries) > 0,
            "has_conditional_exits": len(self.conditional_exits) > 0,
            "is_valid": self.is_valid(),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "components": self._component_registry.get_registry_info(),
        }

    def cleanup(self) -> None:
        """Clean up graph resources."""
        if self._component_registry:
            self._component_registry.cleanup_all()

        logger.debug(f"Cleaned up ModularBaseGraph '{self.name}'")

    def __del__(self):
        """Cleanup when graph is destroyed."""
        try:
            self.cleanup()
        except Exception:
            pass  # Ignore cleanup errors during destruction
