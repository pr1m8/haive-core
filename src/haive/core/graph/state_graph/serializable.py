"""
Serialization support for BaseGraph.

This module provides serialization and deserialization utilities for BaseGraph,
enabling graphs to be stored, loaded, and exchanged with minimal information loss.
"""

import importlib
import inspect
import json
import logging
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from haive.core.graph.branches import Branch
from haive.core.graph.branches.types import BranchMode, ComparisonType
from haive.core.graph.common.references import CallableReference

# Get logger
logger = logging.getLogger(__name__)


# Type references
class TypeReference(BaseModel):
    """Reference to a type that can be serialized."""

    module_path: Optional[str] = None
    name: str
    is_generic: bool = False
    generic_params: Optional[List["TypeReference"]] = None

    model_config = ConfigDict(extra="allow")

    @classmethod
    def from_type(cls, type_obj):
        if type_obj is None:
            return None

        ref = cls(
            name=getattr(type_obj, "__name__", str(type_obj)),
            module_path=getattr(type_obj, "__module__", None),
        )

        # TODO: Add support for complex generic types
        return ref

    def resolve(self):
        """Resolve the reference back to a type."""
        if not self.module_path or not self.name:
            return None

        try:
            module = importlib.import_module(self.module_path)
            return getattr(module, self.name)
        except (ImportError, AttributeError) as e:
            logger.warning(f"Error resolving type: {e}")
            return None


# Node serialization
class SerializableNode(BaseModel):
    """Serializable representation of a Node."""

    id: str
    name: str
    node_type: str

    # Configuration
    input_mapping: Optional[Dict[str, str]] = None
    output_mapping: Optional[Dict[str, str]] = None
    command_goto: Optional[Union[str, List[str]]] = None
    retry_policy: Optional[Dict[str, Any]] = None

    # Metadata
    description: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


# Edge serialization
class SerializableConditionalEdge(BaseModel):
    """Serializable representation of a ConditionalEdge."""

    source: str
    branch_id: str
    targets: Dict[str, str]

    model_config = ConfigDict(arbitrary_types_allowed=True)


# Branch serialization
class SerializableBranch(BaseModel):
    """Serializable representation of a Branch."""

    id: str
    name: str
    branch_type: str
    source_node: str

    # Core branch fields
    key: Optional[str] = None
    value: Optional[Any] = None
    comparison: Optional[str] = None
    default_route: Optional[str] = None
    allow_none: Optional[bool] = None
    message_key: Optional[str] = None
    mode: Optional[str] = None

    # Function references
    function_ref: Optional[CallableReference] = None
    condition_ref: Optional[CallableReference] = None

    # Routing information
    destinations: Dict[str, str] = Field(default_factory=dict)

    # Advanced branch features
    send_mappings: Optional[List[Dict[str, Any]]] = None
    send_generators: Optional[List[Dict[str, Any]]] = None
    dynamic_mapping: Optional[Dict[str, Any]] = None

    # Chain branches - these need special handling
    chain_branches_ids: Optional[List[str]] = None
    true_branch_id: Optional[str] = None
    false_branch_id: Optional[str] = None

    # Metadata
    description: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class SerializableGraph(BaseModel):
    """Serializable representation of the BaseGraph."""

    # Basic identification
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: Optional[str] = None

    # Graph structure
    nodes: Dict[str, SerializableNode] = Field(default_factory=dict)

    # Edges stored as lists for easier serialization
    direct_edges: List[Tuple[str, str]] = Field(default_factory=list)
    conditional_edges: List[Dict[str, Any]] = Field(default_factory=list)

    # Branches
    branches: Dict[str, SerializableBranch] = Field(default_factory=dict)

    # Schema reference
    state_schema: Optional[TypeReference] = None

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: str
    updated_at: str

    # Configuration
    config: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_graph(cls, graph):
        """Create a serializable representation from a BaseGraph."""
        from haive.core.graph.state_graph.base_graph import (
            BaseGraph,
            ConditionalEdge,
            Edge,
        )

        if not isinstance(graph, BaseGraph):
            raise TypeError("Expected a BaseGraph instance")

        # Initialize basic fields
        serializable = cls(
            id=graph.id,
            name=graph.name,
            description=graph.description,
            metadata=graph.metadata.copy(),
            created_at=graph.created_at.isoformat(),
            updated_at=graph.updated_at.isoformat(),
        )

        # Convert schema if present
        if graph.state_schema:
            serializable.state_schema = TypeReference.from_type(graph.state_schema)

        # Convert nodes
        for name, node in graph.nodes.items():
            serializable.nodes[name] = SerializableNode(
                id=node.id,
                name=node.name,
                node_type=node.node_type.value,
                input_mapping=node.input_mapping,
                output_mapping=node.output_mapping,
                command_goto=node.command_goto,
                retry_policy=node.retry_policy._asdict() if node.retry_policy else None,
                description=node.description,
                metadata=node.metadata.copy(),
                created_at=node.created_at.isoformat() if node.created_at else None,
            )

        # Convert edges
        for edge in graph.edges:
            if isinstance(edge, tuple):
                # Direct edge
                serializable.direct_edges.append(edge)
            elif isinstance(edge, ConditionalEdge):
                # Conditional edge
                serializable.conditional_edges.append(
                    {
                        "source": edge.source,
                        "branch_id": edge.branch_id,
                        "targets": edge.targets,
                    }
                )

        # Convert branches
        branch_ids_map = {}  # To track branch IDs for references

        for branch_id, branch in graph.branches.items():
            # Store ID mapping
            branch_ids_map[id(branch)] = branch_id

            # Basic properties all branches have
            branch_data = {
                "id": branch.id,
                "name": branch.name,
                "source_node": branch.source_node,
                "default_route": getattr(branch, "default", None),
                "description": getattr(branch, "description", None),
                "metadata": getattr(branch, "metadata", {}).copy(),
            }

            # Get branch type
            branch_type = None
            if hasattr(branch, "branch_type"):
                branch_type = getattr(branch, "branch_type", None)
                branch_data["branch_type"] = (
                    branch_type.value
                    if hasattr(branch_type, "value")
                    else str(branch_type)
                )
            else:
                # Default to FUNCTION if has function
                branch_data["branch_type"] = (
                    "function" if hasattr(branch, "function") else "key_value"
                )

            # Add standard fields that most branches have
            if hasattr(branch, "key"):
                branch_data["key"] = branch.key
            if hasattr(branch, "value"):
                branch_data["value"] = branch.value
            if hasattr(branch, "comparison"):
                comp = branch.comparison
                branch_data["comparison"] = (
                    comp.value if hasattr(comp, "value") else str(comp)
                )
            if hasattr(branch, "allow_none"):
                branch_data["allow_none"] = branch.allow_none
            if hasattr(branch, "message_key"):
                branch_data["message_key"] = branch.message_key

            # Handle mode
            if hasattr(branch, "mode"):
                mode = branch.mode
                branch_data["mode"] = (
                    mode.value if hasattr(mode, "value") else str(mode)
                )

            # Handle destinations/routes
            if hasattr(branch, "destinations"):
                # Convert any non-string keys to strings
                destinations = {}
                for k, v in branch.destinations.items():
                    destinations[str(k)] = v
                branch_data["destinations"] = destinations
            elif hasattr(branch, "routes"):
                branch_data["destinations"] = branch.routes

            # Handle function references
            if hasattr(branch, "function_ref"):
                branch_data["function_ref"] = branch.function_ref
            elif hasattr(branch, "function") and callable(branch.function):
                branch_data["function_ref"] = CallableReference.from_callable(
                    branch.function
                )

            if hasattr(branch, "condition_ref"):
                branch_data["condition_ref"] = branch.condition_ref
            elif hasattr(branch, "condition") and callable(branch.condition):
                branch_data["condition_ref"] = CallableReference.from_callable(
                    branch.condition
                )

            # Handle advanced branch features
            if hasattr(branch, "send_mappings") and branch.send_mappings:
                # Simplified serialization of send mappings
                branch_data["send_mappings"] = [
                    {
                        "condition": m.condition if hasattr(m, "condition") else None,
                        "node": m.node if hasattr(m, "node") else None,
                        "fields": m.fields if hasattr(m, "fields") else {},
                    }
                    for m in branch.send_mappings
                ]

            if hasattr(branch, "send_generators") and branch.send_generators:
                # Simplified serialization of send generators
                branch_data["send_generators"] = [
                    {
                        "collection_field": (
                            g.collection_field
                            if hasattr(g, "collection_field")
                            else None
                        ),
                        "target_node": (
                            g.target_node if hasattr(g, "target_node") else None
                        ),
                        "item_field": (
                            g.item_field if hasattr(g, "item_field") else None
                        ),
                    }
                    for g in branch.send_generators
                ]

            if hasattr(branch, "dynamic_mapping") and branch.dynamic_mapping:
                # Serialize dynamic mapping
                mapping = branch.dynamic_mapping
                branch_data["dynamic_mapping"] = {
                    "field": mapping.field if hasattr(mapping, "field") else None,
                    "mappings": (
                        mapping.mappings if hasattr(mapping, "mappings") else {}
                    ),
                }

            # Handle chain branches
            if hasattr(branch, "chain_branches") and branch.chain_branches:
                # Store IDs instead of actual objects
                branch_data["chain_branches_ids"] = []
                for chain_branch in branch.chain_branches:
                    # Use ID if it exists in our branches
                    found = False
                    for bid, b in graph.branches.items():
                        if b is chain_branch:
                            branch_data["chain_branches_ids"].append(bid)
                            found = True
                            break

                    # If not found by direct reference, try by object ID
                    if not found and id(chain_branch) in branch_ids_map:
                        branch_data["chain_branches_ids"].append(
                            branch_ids_map[id(chain_branch)]
                        )

            # Handle true/false branches
            if hasattr(branch, "true_branch"):
                true_branch = branch.true_branch
                if isinstance(true_branch, str):
                    branch_data["true_branch_id"] = true_branch  # Direct node name
                elif id(true_branch) in branch_ids_map:
                    branch_data["true_branch_id"] = branch_ids_map[id(true_branch)]

            if hasattr(branch, "false_branch"):
                false_branch = branch.false_branch
                if isinstance(false_branch, str):
                    branch_data["false_branch_id"] = false_branch  # Direct node name
                elif id(false_branch) in branch_ids_map:
                    branch_data["false_branch_id"] = branch_ids_map[id(false_branch)]

            # Create serializable branch
            serializable.branches[branch_id] = SerializableBranch(**branch_data)

        return serializable

    def to_graph(self):
        """Convert serializable representation back to a BaseGraph."""
        from haive.core.graph.branches import Branch, BranchMode, ComparisonType
        from haive.core.graph.branches.dynamic import DynamicMapping
        from haive.core.graph.branches.send_mapping import SendGenerator, SendMapping
        from haive.core.graph.state_graph.base_graph import (
            BaseGraph,
            BranchType,
            ConditionalEdge,
            Edge,
            Node,
            NodeType,
        )

        # Create basic graph
        graph = BaseGraph(
            id=self.id,
            name=self.name,
            description=self.description,
            metadata=self.metadata.copy(),
        )

        # Set timestamps
        try:
            graph.created_at = datetime.fromisoformat(self.created_at)
            graph.updated_at = datetime.fromisoformat(self.updated_at)
        except (ValueError, TypeError):
            # Use current time if iso format parsing fails
            now = datetime.now()
            graph.created_at = now
            graph.updated_at = now

        # Resolve schema if present
        if self.state_schema:
            graph.state_schema = self.state_schema.resolve()

        # Add nodes
        for name, node_data in self.nodes.items():
            try:
                # Convert string node_type to enum
                node_type = NodeType(node_data.node_type)

                # Create the node
                node = Node(
                    id=node_data.id,
                    name=node_data.name,
                    node_type=node_type,
                    input_mapping=node_data.input_mapping,
                    output_mapping=node_data.output_mapping,
                    command_goto=node_data.command_goto,
                    retry_policy=node_data.retry_policy,  # TODO: Convert dict to RetryPolicy
                    description=node_data.description,
                    metadata=node_data.metadata.copy(),
                )

                # Parse created_at
                if node_data.created_at:
                    try:
                        node.created_at = datetime.fromisoformat(node_data.created_at)
                    except (ValueError, TypeError):
                        node.created_at = datetime.now()

                # Add to graph
                graph.nodes[name] = node

            except Exception as e:
                logger.error(f"Error reconstructing node {name}: {e}")

        # Add direct edges
        for source, target in self.direct_edges:
            graph.edges.append((source, target))

        # Add conditional edges
        for edge_data in self.conditional_edges:
            try:
                edge = ConditionalEdge(
                    source=edge_data["source"],
                    branch_id=edge_data["branch_id"],
                    targets=edge_data["targets"],
                )
                graph.edges.append(edge)
            except Exception as e:
                logger.error(f"Error reconstructing conditional edge: {e}")

        # First pass to create all branches (without references to other branches)
        temp_branches = {}
        for branch_id, branch_data in self.branches.items():
            try:
                # Resolve function reference if available
                function = None
                if branch_data.function_ref:
                    function = branch_data.function_ref.resolve()

                # Resolve condition reference if available
                condition = None
                if branch_data.condition_ref:
                    condition = branch_data.condition_ref.resolve()

                # Convert comparison enum
                comparison = branch_data.comparison
                if comparison and not isinstance(comparison, ComparisonType):
                    try:
                        comparison = ComparisonType(comparison)
                    except (ValueError, TypeError):
                        pass  # Keep as string if not an enum value

                # Convert mode enum
                mode = branch_data.mode
                if mode and not isinstance(mode, BranchMode):
                    try:
                        mode = BranchMode(mode)
                    except (ValueError, TypeError):
                        pass  # Keep as string if not an enum value

                # Create dynamic mapping if provided
                dynamic_mapping = None
                if branch_data.dynamic_mapping:
                    dynamic_mapping = DynamicMapping(
                        field=branch_data.dynamic_mapping.get("field", ""),
                        mappings=branch_data.dynamic_mapping.get("mappings", {}),
                    )

                # Create send mappings if provided
                send_mappings = []
                if branch_data.send_mappings:
                    for mapping_data in branch_data.send_mappings:
                        mapping = SendMapping(
                            condition=mapping_data.get("condition"),
                            node=mapping_data.get("node"),
                            fields=mapping_data.get("fields", {}),
                        )
                        send_mappings.append(mapping)

                # Create send generators if provided
                send_generators = []
                if branch_data.send_generators:
                    for generator_data in branch_data.send_generators:
                        generator = SendGenerator(
                            collection_field=generator_data.get("collection_field"),
                            target_node=generator_data.get("target_node"),
                            item_field=generator_data.get("item_field"),
                        )
                        send_generators.append(generator)

                # Create branch with available data
                branch = Branch(
                    id=branch_data.id,
                    name=branch_data.name,
                    source_node=branch_data.source_node,
                    key=branch_data.key,
                    value=branch_data.value,
                    comparison=comparison,
                    function=function,
                    function_ref=branch_data.function_ref,
                    destinations=branch_data.destinations,
                    default=branch_data.default_route or "END",
                    allow_none=branch_data.allow_none or False,
                    message_key=branch_data.message_key or "messages",
                    mode=mode or BranchMode.DIRECT,
                    # Dynamic mapping
                    dynamic_mapping=dynamic_mapping,
                    # Send mapping
                    send_mappings=send_mappings,
                    send_generators=send_generators,
                    # For condition branches
                    condition_ref=branch_data.condition_ref,
                )

                # Store for second pass linking
                temp_branches[branch_id] = branch

                # Add to graph
                graph.branches[branch_id] = branch

            except Exception as e:
                logger.error(f"Error reconstructing branch {branch_id}: {e}")

        # Second pass to link branches to each other
        for branch_id, branch_data in self.branches.items():
            if branch_id not in temp_branches:
                continue

            branch = temp_branches[branch_id]

            # Link chain branches
            if branch_data.chain_branches_ids:
                branch.chain_branches = []
                for chain_id in branch_data.chain_branches_ids:
                    if chain_id in temp_branches:
                        branch.chain_branches.append(temp_branches[chain_id])

            # Link true/false branches
            if branch_data.true_branch_id:
                if branch_data.true_branch_id in temp_branches:
                    branch.true_branch = temp_branches[branch_data.true_branch_id]
                else:
                    # Might be a node name
                    branch.true_branch = branch_data.true_branch_id

            if branch_data.false_branch_id:
                if branch_data.false_branch_id in temp_branches:
                    branch.false_branch = temp_branches[branch_data.false_branch_id]
                else:
                    # Might be a node name
                    branch.false_branch = branch_data.false_branch_id

        return graph

    def to_dict(self):
        """Convert to a dictionary for serialization."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data):
        """Create from a dictionary."""
        return cls.model_validate(data)

    def to_json(self, **kwargs):
        """Convert to JSON string."""
        return self.model_dump_json(**kwargs)

    @classmethod
    def from_json(cls, json_str):
        """Create from JSON string."""
        return cls.model_validate_json(json_str)
