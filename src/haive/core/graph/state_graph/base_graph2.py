"""Base graph implementation for the Haive framework.

Provides a comprehensive system for building, manipulating, and executing
graphs with consistent interfaces, serialization support, and dynamic composition.
"""
import inspect
import logging
import uuid
from collections.abc import Callable
from datetime import datetime
from enum import Enum
from typing import Any, Generic, Literal, Self, Union

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START
from langgraph.types import Command, RetryPolicy, Send
from pydantic import BaseModel, ConfigDict, Field, model_validator

from haive.core.graph.branches.branch import Branch
from haive.core.graph.branches.types import BranchMode, ComparisonType
from haive.core.graph.common.references import CallableReference
from haive.core.graph.common.types import ConfigLike, NodeOutput, NodeType, StateLike
from haive.core.graph.node.agent_node_v3 import create_agent_node_v3
from haive.core.graph.node.config import NodeConfig
from haive.core.graph.state_graph.graph_path import GraphPath
from haive.core.graph.state_graph.validation_mixin import ValidationMixin

BranchResultType = Union[str, bool, list[str], list[Send], Send, Command, None]
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

class BranchType(str, Enum):
    """Types of branches for conditional routing."""
    FUNCTION = 'function'
    KEY_VALUE = 'key_value'
    SEND = 'send'
    COMMAND = 'command'

class EdgeType(str, Enum):
    """Types of edges in a graph."""
    DIRECT = 'direct'
    CONDITIONAL = 'conditional'
    DYNAMIC = 'dynamic'
SimpleEdge = tuple[str, str]
Edge = SimpleEdge

class Node(BaseModel, Generic[StateLike, ConfigLike, NodeOutput]):
    """Base node in a graph system."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    node_type: NodeType
    input_mapping: dict[str, str] | None = None
    output_mapping: dict[str, str] | None = None
    command_goto: str | list[str] | None = None
    retry_policy: RetryPolicy | None = None
    description: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    model_config = {'arbitrary_types_allowed': True}

    def process(self, state: StateLike, config: ConfigLike | None=None) -> NodeOutput:
        """Process state and return output."""
        raise NotImplementedError('Subclasses must implement process method')

    @property
    def display_name(self) -> str:
        """Get a human-readable display name."""
        return self.name

    def to_dict(self) -> dict[str, Any]:
        """Convert to serializable dictionary."""
        return {'id': self.id, 'name': self.name, 'node_type': self.node_type, 'input_mapping': self.input_mapping, 'output_mapping': self.output_mapping, 'command_goto': self.command_goto, 'description': self.description, 'metadata': self.metadata, 'created_at': self.created_at.isoformat() if self.created_at else None}

class BaseGraph(BaseModel, ValidationMixin):
    """Base class for graph management in the Haive framework.

    Provides comprehensive graph management capabilities including:
    - Node management (add, remove, update)
    - Edge management (direct and branch-based)
    - Branch management
    - Graph validation
    - Serialization
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    nodes: dict[str, Node | NodeConfig | Any | None] = Field(default_factory=dict)
    edges: list[Edge] = Field(default_factory=list)
    branches: dict[str, Branch] = Field(default_factory=dict)
    entry_points: list[str] = Field(default_factory=list)
    finish_points: list[str] = Field(default_factory=list)
    conditional_entries: dict[str, dict[str, Any]] = Field(default_factory=dict)
    conditional_exits: dict[str, dict[str, Any]] = Field(default_factory=dict)
    entry_point: str | None = Field(default=None, description='Deprecated: Use entry_points instead')
    finish_point: str | None = Field(default=None, description='Deprecated: Use finish_points instead')
    state_schema: Any | None = None
    default_config: RunnableConfig | None = None
    subgraphs: dict[str, Any] = Field(default_factory=dict)
    node_types: dict[str, NodeType] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    needs_recompile_flag: bool = Field(default=False, exclude=True)
    last_compiled_at: datetime | None = Field(default=None, exclude=True)
    compilation_state_hash: str | None = Field(default=None, exclude=True)
    last_input_schema: Any | None = Field(default=None, exclude=True)
    last_output_schema: Any | None = Field(default=None, exclude=True)
    last_config_schema: Any | None = Field(default=None, exclude=True)
    last_interrupt_before: list | None = Field(default=None, exclude=True)
    last_interrupt_after: list | None = Field(default=None, exclude=True)
    last_compile_kwargs: dict[str, Any] | None = Field(default=None, exclude=True)
    model_config = {'arbitrary_types_allowed': True}
    allow_cycles: bool = Field(default=False, description='Whether to allow cycles in the graph')
    require_end_path: bool = Field(default=True, description='Whether all nodes must have a path to END')

    @model_validator(mode='after')
    def validate_graph(self) -> Self:
        """Validate the graph structure."""
        non_none_nodes = {k: v for k, v in self.nodes.items() if v is not None}
        node_names = set()
        for name, node in non_none_nodes.items():
            if hasattr(node, 'name'):
                node_names.add(node.name)
            else:
                node_names.add(name)
        if len(node_names) != len(non_none_nodes):
            raise ValueError('Node names must be unique')
        if self.entry_points and self.entry_points[0] not in self.nodes:
            raise ValueError(f"Entry point '{self.entry_points[0]}' not found in nodes")
        if self.finish_points and self.finish_points[0] not in self.nodes:
            raise ValueError(f"Finish point '{self.finish_points[0]}' not found in nodes")
        if not hasattr(self, 'subgraphs'):
            self.subgraphs = {}
        if not hasattr(self, 'node_types'):
            self.node_types = {}
        if not hasattr(self, 'conditional_entries'):
            self.conditional_entries = {}
        if not hasattr(self, 'conditional_exits'):
            self.conditional_exits = {}
        for name, node in non_none_nodes.items():
            if name not in self.node_types:
                node_type = self._infer_node_type(node)
                self._track_node_type(name, node_type, getattr(node, 'metadata', {}))
        for source, target in self.edges:
            if source != START and source not in self.nodes:
                raise ValueError(f"Edge source '{source}' not found in nodes")
            if target != END and target not in self.nodes:
                raise ValueError(f"Edge target '{target}' not found in nodes")
        for _branch_id, branch in self.branches.items():
            if branch.source_node != START and branch.source_node not in self.nodes:
                raise ValueError(f"Branch source '{branch.source_node}' not found in nodes")
            for target in branch.destinations.values():
                if target != END and target not in self.nodes:
                    raise ValueError(f"Branch target '{target}' not found in nodes")
        return self

    def _mark_needs_recompile(self, reason: str='') -> None:
        """Mark the graph as needing recompilation.

        Args:
            reason: Optional reason for needing recompilation
        """
        self.needs_recompile_flag = True
        if reason:
            logger.debug(f"Graph '{self.name}' marked for recompilation: {reason}")

    def needs_recompile(self) -> bool:
        """Check if the graph needs recompilation.

        Returns:
            True if the graph has been modified since last compilation
        """
        return self.needs_recompile_flag

    def mark_compiled(self, input_schema: Any | None=None, output_schema: Any | None=None, config_schema: Any | None=None, interrupt_before: list | None=None, interrupt_after: list | None=None, **compile_kwargs) -> None:
        """Mark the graph as compiled and reset the recompilation flag.

        Args:
            input_schema: Input schema used for compilation
            output_schema: Output schema used for compilation
            config_schema: Config schema used for compilation
            interrupt_before: Interrupt before nodes used for compilation
            interrupt_after: Interrupt after nodes used for compilation
            **compile_kwargs: Additional compilation parameters
        """
        self.needs_recompile_flag = False
        self.last_compiled_at = datetime.now()
        self.compilation_state_hash = self._compute_state_hash()
        self.last_input_schema = input_schema
        self.last_output_schema = output_schema
        self.last_config_schema = config_schema
        self.last_interrupt_before = interrupt_before.copy() if interrupt_before else None
        self.last_interrupt_after = interrupt_after.copy() if interrupt_after else None
        self.last_compile_kwargs = compile_kwargs.copy() if compile_kwargs else {}
        logger.debug(f"Graph '{self.name}' marked as compiled at {self.last_compiled_at}")

    def get_compilation_info(self) -> dict[str, Any]:
        """Get information about the compilation state.

        Returns:
            Dictionary with compilation state information
        """
        return {'needs_recompile': self.needs_recompile_flag, 'last_compiled_at': self.last_compiled_at.isoformat() if self.last_compiled_at else None, 'compilation_state_hash': self.compilation_state_hash, 'current_state_hash': self._compute_state_hash(), 'state_matches': self.compilation_state_hash == self._compute_state_hash() if self.compilation_state_hash else None, 'schemas': {'input_schema': str(self.last_input_schema) if self.last_input_schema else None, 'output_schema': str(self.last_output_schema) if self.last_output_schema else None, 'config_schema': str(self.last_config_schema) if self.last_config_schema else None, 'state_schema': str(self.state_schema) if self.state_schema else None}, 'interrupts': {'interrupt_before': self.last_interrupt_before, 'interrupt_after': self.last_interrupt_after}, 'compile_kwargs': self.last_compile_kwargs or {}}

    def _compute_state_hash(self) -> str:
        """Compute a hash of the current graph state for comparison.

        Returns:
            Hash string representing the current graph structure
        """
        import hashlib
        import json
        state_repr = {'nodes': sorted(self.nodes.keys()), 'edges': sorted([f'{s}->{t}' for s, t in self.edges]), 'branches': sorted(self.branches.keys()), 'entry_points': sorted(self.entry_points), 'finish_points': sorted(self.finish_points), 'state_schema': str(self.state_schema) if self.state_schema else None}
        state_json = json.dumps(state_repr, sort_keys=True)
        return hashlib.sha256(state_json.encode()).hexdigest()[:16]

    def needs_recompile_for_schemas(self, input_schema: Any | None=None, output_schema: Any | None=None, config_schema: Any | None=None) -> bool:
        """Check if recompilation is needed due to schema changes.

        Args:
            input_schema: New input schema to compare
            output_schema: New output schema to compare
            config_schema: New config schema to compare

        Returns:
            True if any schema has changed since last compilation
        """
        if self.needs_recompile_flag:
            return True
        if input_schema != self.last_input_schema:
            return True
        if output_schema != self.last_output_schema:
            return True
        return config_schema != self.last_config_schema

    def needs_recompile_for_interrupts(self, interrupt_before: list | None=None, interrupt_after: list | None=None) -> bool:
        """Check if recompilation is needed due to interrupt changes.

        Args:
            interrupt_before: New interrupt_before list to compare
            interrupt_after: New interrupt_after list to compare

        Returns:
            True if interrupt configuration has changed since last compilation
        """
        if self.needs_recompile_flag:
            return True
        if interrupt_before != self.last_interrupt_before:
            return True
        return interrupt_after != self.last_interrupt_after

    def check_full_recompilation_needed(self, input_schema: Any | None=None, output_schema: Any | None=None, config_schema: Any | None=None, interrupt_before: list | None=None, interrupt_after: list | None=None, **compile_kwargs) -> dict[str, Any]:
        """Check if recompilation is needed for any reason and provide details.

        Args:
            input_schema: Input schema to check
            output_schema: Output schema to check
            config_schema: Config schema to check
            interrupt_before: Interrupt before configuration to check
            interrupt_after: Interrupt after configuration to check
            **compile_kwargs: Additional compilation parameters to check

        Returns:
            Dictionary with recompilation status and reasons
        """
        reasons = []
        if self.needs_recompile_flag:
            reasons.append('Graph structure has changed')
        if input_schema != self.last_input_schema:
            reasons.append(f'Input schema changed: {self.last_input_schema} -> {input_schema}')
        if output_schema != self.last_output_schema:
            reasons.append(f'Output schema changed: {self.last_output_schema} -> {output_schema}')
        if config_schema != self.last_config_schema:
            reasons.append(f'Config schema changed: {self.last_config_schema} -> {config_schema}')
        if interrupt_before != self.last_interrupt_before:
            reasons.append(f'Interrupt before changed: {self.last_interrupt_before} -> {interrupt_before}')
        if interrupt_after != self.last_interrupt_after:
            reasons.append(f'Interrupt after changed: {self.last_interrupt_after} -> {interrupt_after}')
        last_kwargs = self.last_compile_kwargs or {}
        if compile_kwargs != last_kwargs:
            reasons.append(f'Compile kwargs changed: {last_kwargs} -> {compile_kwargs}')
        return {'needs_recompile': len(reasons) > 0, 'reasons': reasons, 'structural_changes': self.needs_recompile_flag, 'schema_changes': input_schema != self.last_input_schema or output_schema != self.last_output_schema or config_schema != self.last_config_schema, 'interrupt_changes': interrupt_before != self.last_interrupt_before or interrupt_after != self.last_interrupt_after, 'kwargs_changes': compile_kwargs != last_kwargs}

    def set_state_schema(self, schema: type[BaseModel] | None) -> 'BaseGraph':
        """Set the state schema and mark the graph as needing recompilation.

        Args:
            schema: New state schema to use

        Returns:
            Self for method chaining
        """
        old_schema = self.state_schema
        self.state_schema = schema
        if old_schema != schema:
            self._mark_needs_recompile(f'Changed state schema from {old_schema} to {schema}')
        return self

    def _infer_node_type(self, node: Any) -> NodeType:
        """Infer the node type from a node object.

        Args:
            node: Node object to infer type from

        Returns:
            Inferred NodeType
        """
        if hasattr(node, 'node_type'):
            return node.node_type
        if hasattr(node, '__class__') and 'NodeConfig' in node.__class__.__name__:
            if 'EngineNodeConfig' in node.__class__.__name__:
                return NodeType.ENGINE
            if 'ToolNodeConfig' in node.__class__.__name__:
                return NodeType.TOOL
            if 'ValidationNodeConfig' in node.__class__.__name__:
                return NodeType.VALIDATION
            return NodeType.CALLABLE
        if hasattr(node, 'engine_type') and hasattr(node, 'create_runnable'):
            return NodeType.ENGINE
        if isinstance(node, BaseGraph):
            return NodeType.SUBGRAPH
        if callable(node):
            return NodeType.CALLABLE
        return NodeType.CALLABLE

    def add_node(self, node_or_name: Node | dict[str, Any] | str | NodeConfig, node_like: Any | None=None, **kwargs) -> 'BaseGraph':
        """Add a node to the graph with flexible input options.

        Args:
            node_or_name: Node object, dictionary, node name, or NodeConfig
            node_like: If node_or_name is a string, this is the node object, callable, or engine
            **kwargs: Additional properties when creating a node from name

        Returns:
            Self for method chaining
        """
        node_obj = None
        if isinstance(node_or_name, str):
            name = node_or_name
            if node_like is None and (not kwargs):
                self.nodes[name] = None
                self._mark_needs_recompile(f"Added placeholder node '{name}'")
                return self
            node_data = {'name': name}
            if node_like is None:
                node_data.update(kwargs)
                node_data.setdefault('node_type', NodeType.CALLABLE)
            elif hasattr(node_like, '__class__') and 'NodeConfig' in node_like.__class__.__name__:
                self.nodes[name] = node_like
                node_type = self._infer_node_type(node_like)
                self._track_node_type(name, node_type, getattr(node_like, 'metadata', {}))
                self.updated_at = datetime.now()
                self._mark_needs_recompile(f"Added NodeConfig node '{name}'")
                return self
            elif isinstance(node_like, Node):
                node_data = node_like.model_dump()
                node_data['name'] = name
            elif callable(node_like):
                node_data.update({'node_type': NodeType.CALLABLE, 'metadata': {'callable': node_like}, **kwargs})
            elif hasattr(node_like, 'engine_type') and hasattr(node_like, 'create_runnable'):
                node_data.update({'node_type': NodeType.ENGINE, 'metadata': {'engine': node_like, 'callable': node_like}, **kwargs})
            elif isinstance(node_like, BaseGraph):
                return self.add_subgraph(name, node_like, **kwargs)
            else:
                node_data.update({'node_type': NodeType.CALLABLE if callable(node_like) else NodeType.ENGINE, 'metadata': {'object': node_like, 'callable': node_like if callable(node_like) else None}, **kwargs})
            node_obj = Node(**node_data)
        elif isinstance(node_or_name, dict):
            if 'name' not in node_or_name:
                raise ValueError("Node dictionary must have a 'name' field")
            node_data = {**node_or_name, **kwargs}
            node_obj = Node(**node_data)
        elif isinstance(node_or_name, Node):
            node_obj = node_or_name
            if kwargs:
                for key, value in kwargs.items():
                    if hasattr(node_obj, key):
                        setattr(node_obj, key, value)
        elif hasattr(node_or_name, '__class__') and 'NodeConfig' in node_or_name.__class__.__name__:
            config = node_or_name
            name = getattr(config, 'name', f'node_{uuid.uuid4().hex[:8]}')
            self.nodes[name] = config
            node_type = self._infer_node_type(config)
            self._track_node_type(name, node_type, getattr(config, 'metadata', {}))
            logger.debug(f"Added NodeConfig '{name}' to graph '{self.name}'")
            self.updated_at = datetime.now()
            self._mark_needs_recompile(f"Added NodeConfig '{name}'")
            return self
        else:
            raise TypeError(f'Unsupported node type: {type(node_or_name)}')
        if node_obj.name in self.nodes:
            raise ValueError(f"Node '{node_obj.name}' already exists in the graph")
        self.nodes[node_obj.name] = node_obj
        self._track_node_type(node_obj.name, node_obj.node_type, node_obj.metadata)
        logger.debug(f"Added node '{node_obj.name}' to graph '{self.name}'")
        self.updated_at = datetime.now()
        self._mark_needs_recompile(f"Added node '{node_obj.name}'")
        return self

    def _track_node_type(self, node_name: str, node_type: NodeType | None=None, metadata: dict[str, Any] | None=None) -> None:
        """Track node type for a node.

        Args:
            node_name: Name of the node
            node_type: Type of node (if known)
            metadata: Node metadata (used to infer type if not provided)
        """
        determined_type = node_type
        if not determined_type and metadata:
            if 'engine' in metadata:
                determined_type = NodeType.ENGINE
            elif 'tools' in metadata:
                determined_type = NodeType.TOOL
            elif 'validation' in metadata:
                determined_type = NodeType.VALIDATION
            elif 'subgraph' in metadata:
                determined_type = NodeType.SUBGRAPH
            elif 'callable' in metadata:
                determined_type = NodeType.CALLABLE
        if not determined_type:
            determined_type = NodeType.CALLABLE
        self.node_types[node_name] = determined_type

    def set_entry_point(self, node_name: str) -> 'BaseGraph':
        """Set an entry point of the graph.

        Args:
            node_name: Name of the node to set as an entry point

        Returns:
            Self for method chaining
        """
        if node_name not in self.nodes:
            raise ValueError(f"Node '{node_name}' not found in graph")
        if node_name not in self.entry_points:
            self.entry_points.append(node_name)
        self.entry_point = node_name
        if not any((src == START and dst == node_name for src, dst in self.edges)):
            self.add_edge(START, node_name)
        logger.debug(f"Set entry point to '{node_name}' in graph '{self.name}'")
        self.updated_at = datetime.now()
        return self

    def set_finish_point(self, node_name: str) -> 'BaseGraph':
        """Set a finish point of the graph.

        Args:
            node_name: Name of the node to set as a finish point

        Returns:
            Self for method chaining
        """
        if node_name not in self.nodes:
            raise ValueError(f"Node '{node_name}' not found in graph")
        if node_name not in self.finish_points:
            self.finish_points.append(node_name)
        self.finish_point = node_name
        if not any((src == node_name and dst == END for src, dst in self.edges)):
            self.add_edge(node_name, END)
        logger.debug(f"Set finish point to '{node_name}' in graph '{self.name}'")
        self.updated_at = datetime.now()
        return self

    def set_end_point(self, node_name: str) -> 'BaseGraph':
        """Deprecated: Use set_finish_point instead.

        Set a finish point of the graph.

        Args:
            node_name: Name of the node to set as a finish point

        Returns:
            Self for method chaining
        """
        return self.set_finish_point(node_name)

    def set_conditional_entry(self, condition: Callable[[StateLike, ConfigLike | None], bool], entry_node: str, default_entry: str | None=None) -> 'BaseGraph':
        """Set a conditional entry point for the graph.

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
        entry_id = str(uuid.uuid4())
        self.conditional_entries[entry_id] = {'condition': condition, 'true_entry': entry_node, 'false_entry': default_entry or self.entry_point, 'function_ref': CallableReference.from_callable(condition)}
        if not any((src == START and dst == entry_node for src, dst in self.edges)):
            self.add_edge(START, entry_node)
        false_node = default_entry or self.entry_point
        if false_node and (not any((src == START and dst == false_node for src, dst in self.edges))):
            self.add_edge(START, false_node)
        logger.debug(f"Added conditional entry point to '{entry_node}' in graph '{self.name}'")
        self.updated_at = datetime.now()
        return self

    def set_conditional_exit(self, node_name: str, condition: Callable[[StateLike, ConfigLike | None], bool], exit_if_true: bool=True) -> 'BaseGraph':
        """Set a conditional exit point for the graph.

        Args:
            node_name: Name of the node to set as conditional exit
            condition: Function that takes state and config, returns boolean
            exit_if_true: Whether to exit when condition is True (default) or False

        Returns:
            Self for method chaining
        """
        if node_name not in self.nodes:
            raise ValueError(f"Node '{node_name}' not found in graph")
        exit_id = str(uuid.uuid4())
        self.conditional_exits[exit_id] = {'node': node_name, 'condition': condition, 'exit_if_true': exit_if_true, 'function_ref': CallableReference.from_callable(condition)}
        if not any((src == node_name and dst == END for src, dst in self.edges)):
            self.add_edge(node_name, END)
        logger.debug(f"Added conditional exit point at '{node_name}' in graph '{self.name}'")
        self.updated_at = datetime.now()
        return self

    def remove_conditional_entry(self, entry_id: str) -> 'BaseGraph':
        """Remove a conditional entry point.

        Args:
            entry_id: ID of the conditional entry to remove

        Returns:
            Self for method chaining
        """
        if entry_id not in self.conditional_entries:
            raise ValueError(f"Conditional entry '{entry_id}' not found")
        del self.conditional_entries[entry_id]
        logger.debug(f"Removed conditional entry point '{entry_id}' from graph '{self.name}'")
        self.updated_at = datetime.now()
        return self

    def remove_conditional_exit(self, exit_id: str) -> 'BaseGraph':
        """Remove a conditional exit point.

        Args:
            exit_id: ID of the conditional exit to remove

        Returns:
            Self for method chaining
        """
        if exit_id not in self.conditional_exits:
            raise ValueError(f"Conditional exit '{exit_id}' not found")
        del self.conditional_exits[exit_id]
        logger.debug(f"Removed conditional exit point '{exit_id}' from graph '{self.name}'")
        self.updated_at = datetime.now()
        return self

    def get_conditional_entries(self) -> dict[str, dict[str, Any]]:
        """Get all conditional entry points.

        Returns:
            Dictionary of conditional entries indexed by ID
        """
        return self.conditional_entries

    def get_conditional_exits(self) -> dict[str, dict[str, Any]]:
        """Get all conditional exit points.

        Returns:
            Dictionary of conditional exits indexed by ID
        """
        return self.conditional_exits

    @property
    def all_entry_points(self) -> dict[str, Any]:
        """Property that returns all entry points (regular and conditional).

        Returns:
            Dictionary containing all entry points information
        """
        result = {'primary': self.entry_points, 'conditional': self.conditional_entries}
        start_connections = []
        for src, dst in self.edges:
            if src == START and dst not in start_connections:
                start_connections.append(dst)
        result['all_start_connections'] = start_connections
        return result

    @property
    def all_finish_points(self) -> dict[str, Any]:
        """Property that returns all finish points (regular and conditional).

        Returns:
            Dictionary containing all finish points information
        """
        result = {'primary': self.finish_points, 'conditional': self.conditional_exits}
        end_connections = []
        for src, dst in self.edges:
            if dst == END and src not in end_connections:
                end_connections.append(src)
        result['all_end_connections'] = end_connections
        return result

    @property
    def entry_points_data(self) -> dict[str, Any]:
        """Deprecated: Use all_entry_points instead."""
        return self.all_entry_points

    @property
    def exit_points(self) -> dict[str, Any]:
        """Deprecated: Use all_finish_points instead."""
        return self.all_finish_points

    @property
    def all_exit_points(self) -> dict[str, Any]:
        """Deprecated: Use all_finish_points instead."""
        return self.all_finish_points

    def add_tool_node(self, node_name: str, node_type: NodeType=NodeType.TOOL, **kwargs) -> 'BaseGraph':
        """Add a tool node to the graph.

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

    def add_subgraph(self, name: str, subgraph: 'BaseGraph', **kwargs) -> 'BaseGraph':
        """Add a subgraph as a node.

        Args:
            name: Name for the subgraph node
            subgraph: Subgraph object to add
            **kwargs: Additional node properties

        Returns:
            Self for method chaining
        """
        if name in self.nodes:
            raise ValueError(f"Node '{name}' already exists in graph")
        self.subgraphs[name] = subgraph
        node_obj = Node(name=name, node_type=NodeType.SUBGRAPH, metadata={'subgraph': subgraph.name, 'subgraph_id': subgraph.id}, **kwargs)
        self.nodes[name] = node_obj
        self.node_types[name] = NodeType.SUBGRAPH
        logger.debug(f"Added subgraph '{subgraph.name}' as node '{name}'")
        self.updated_at = datetime.now()
        return self

    def remove_node(self, node_name: str) -> 'BaseGraph':
        """Remove a node from the graph.

        Args:
            node_name: Name of the node to remove

        Returns:
            Self for method chaining
        """
        if node_name not in self.nodes:
            raise ValueError(f"Node '{node_name}' not found in graph")
        del self.nodes[node_name]
        if node_name in self.node_types:
            del self.node_types[node_name]
        if node_name in self.subgraphs:
            del self.subgraphs[node_name]
        self.edges = [edge for edge in self.edges if edge[0] != node_name and edge[1] != node_name]
        branch_ids_to_remove = []
        for branch_id, branch in self.branches.items():
            if branch.source_node == node_name:
                branch_ids_to_remove.append(branch_id)
        for branch_id in branch_ids_to_remove:
            del self.branches[branch_id]
        for branch in self.branches.values():
            for condition, target in list(branch.destinations.items()):
                if target == node_name:
                    del branch.destinations[condition]
            if branch.default == node_name:
                branch.default = END
        logger.debug(f"Removed node '{node_name}' from graph '{self.name}'")
        self.updated_at = datetime.now()
        self._mark_needs_recompile(f"Removed node '{node_name}'")
        return self

    def get_node(self, node_name: str) -> Any | None:
        """Get a node by name.

        Args:
            node_name: Name of the node to retrieve

        Returns:
            Node object if found, None otherwise
        """
        return self.nodes.get(node_name)

    def update_node(self, node_name: str, **updates) -> 'BaseGraph':
        """Update a node's properties.

        Args:
            node_name: Name of the node to update
            **updates: Properties to update

        Returns:
            Self for method chaining
        """
        if node_name not in self.nodes:
            raise ValueError(f"Node '{node_name}' not found in graph")
        node = self.nodes[node_name]
        if node is None:
            raise ValueError(f"Cannot update None node '{node_name}'")
        for key, value in updates.items():
            if hasattr(node, key):
                setattr(node, key, value)
        if 'node_type' in updates and node_name in self.node_types:
            self.node_types[node_name] = updates['node_type']
        self.updated_at = datetime.now()
        return self

    def replace_node(self, node_name: str, new_node: Node | dict | Any, preserve_connections: bool=True) -> 'BaseGraph':
        """Replace a node while optionally preserving its connections.

        Args:
            node_name: Name of the node to replace
            new_node: New node to insert
            preserve_connections: Whether to preserve existing connections

        Returns:
            Self for method chaining
        """
        if node_name not in self.nodes:
            raise ValueError(f"Node '{node_name}' not found in graph")
        incoming_edges = []
        outgoing_edges = []
        source_branches = []
        if preserve_connections:
            for source, target in self.edges:
                if source == node_name:
                    outgoing_edges.append((source, target))
                elif target == node_name:
                    incoming_edges.append((source, target))
            source_branches = [branch for branch in self.branches.values() if branch.source_node == node_name]
        self.remove_node(node_name)
        if isinstance(new_node, Node):
            if new_node.name != node_name:
                new_node_copy = new_node.model_copy(deep=True)
                new_node_copy.name = node_name
                self.add_node(new_node_copy)
            else:
                self.add_node(new_node)
        elif hasattr(new_node, '__class__') and 'NodeConfig' in new_node.__class__.__name__:
            if hasattr(new_node, 'name') and new_node.name != node_name:
                if hasattr(new_node, 'model_copy'):
                    new_node_copy = new_node.model_copy(deep=True)
                    new_node_copy.name = node_name
                    self.nodes[node_name] = new_node_copy
                else:
                    self.nodes[node_name] = new_node
            else:
                self.nodes[node_name] = new_node
        elif callable(new_node):
            self.add_node(node_name, new_node)
            if hasattr(self.nodes[node_name], 'metadata'):
                self.nodes[node_name].metadata['callable'] = new_node
        else:
            self.add_node(node_name, new_node)
        if preserve_connections:
            for source, target in incoming_edges:
                self.add_edge(source, node_name)
            for source, target in outgoing_edges:
                self.add_edge(node_name, target)
            for branch in source_branches:
                branch.source_node = node_name
                self.add_branch(branch)
        logger.debug(f"Replaced node '{node_name}' in graph '{self.name}'")
        return self

    def insert_node_after(self, target_node: str, new_node: str | Node | dict | Any, new_node_obj: Any | None=None, **kwargs) -> 'BaseGraph':
        """Insert a new node after an existing node, redirecting all outgoing connections.

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
        outgoing_edges = []
        for source, target in self.edges:
            if source == target_node:
                outgoing_edges.append((source, target))
        if isinstance(new_node, str):
            self.add_node(new_node, new_node_obj, **kwargs)
            new_node_name = new_node
        elif isinstance(new_node, Node):
            self.add_node(new_node)
            new_node_name = new_node.name
        elif isinstance(new_node, dict):
            if 'name' not in new_node:
                raise ValueError("Node dictionary must have a 'name' field")
            self.add_node(new_node)
            new_node_name = new_node['name']
        elif hasattr(new_node, '__class__') and 'NodeConfig' in new_node.__class__.__name__:
            new_node_name = getattr(new_node, 'name', f'{target_node}_after_{uuid.uuid4().hex[:6]}')
            if not hasattr(new_node, 'name') or not new_node.name:
                if hasattr(new_node, 'model_copy'):
                    node_copy = new_node.model_copy(deep=True)
                    node_copy.name = new_node_name
                    self.nodes[new_node_name] = node_copy
                else:
                    self.nodes[new_node_name] = new_node
            else:
                self.nodes[new_node_name] = new_node
            node_type = self._infer_node_type(new_node)
            self._track_node_type(new_node_name, node_type, getattr(new_node, 'metadata', {}))
        else:
            new_node_name = f'{target_node}_after_{uuid.uuid4().hex[:6]}'
            self.add_node(new_node_name, new_node, **kwargs)
        branches_to_update = []
        for _branch_id, branch in self.branches.items():
            if branch.source_node == target_node:
                branches_to_update.append(branch)
        for edge in outgoing_edges:
            self.remove_edge(edge[0], edge[1])
        self.add_edge(target_node, new_node_name)
        for _, target in outgoing_edges:
            self.add_edge(new_node_name, target)
        for branch in branches_to_update:
            self.remove_branch(branch.id)
            new_branch = branch.model_copy(deep=True)
            new_branch.id = str(uuid.uuid4())
            new_branch.source_node = new_node_name
            self.add_branch(new_branch)
        logger.debug(f"Inserted node '{new_node_name}' after '{target_node}' in graph '{self.name}'")
        return self

    def insert_node_before(self, target_node: str, new_node: str | Node | dict | Any, new_node_obj: Any | None=None, **kwargs) -> 'BaseGraph':
        """Insert a new node before an existing node, redirecting all incoming connections.

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
        incoming_edges = []
        for source, target in self.edges:
            if target == target_node:
                incoming_edges.append((source, target))
        incoming_branches = []
        for _branch_id, branch in self.branches.items():
            for condition, dest in branch.destinations.items():
                if dest == target_node:
                    incoming_branches.append((branch, condition))
            if branch.default == target_node:
                incoming_branches.append((branch, 'default'))
        if isinstance(new_node, str):
            self.add_node(new_node, new_node_obj, **kwargs)
            new_node_name = new_node
        elif isinstance(new_node, Node):
            self.add_node(new_node)
            new_node_name = new_node.name
        elif isinstance(new_node, dict):
            if 'name' not in new_node:
                raise ValueError("Node dictionary must have a 'name' field")
            self.add_node(new_node)
            new_node_name = new_node['name']
        elif hasattr(new_node, '__class__') and 'NodeConfig' in new_node.__class__.__name__:
            new_node_name = getattr(new_node, 'name', f'{target_node}_before_{uuid.uuid4().hex[:6]}')
            if not hasattr(new_node, 'name') or not new_node.name:
                if hasattr(new_node, 'model_copy'):
                    node_copy = new_node.model_copy(deep=True)
                    node_copy.name = new_node_name
                    self.nodes[new_node_name] = node_copy
                else:
                    self.nodes[new_node_name] = new_node
            else:
                self.nodes[new_node_name] = new_node
            node_type = self._infer_node_type(new_node)
            self._track_node_type(new_node_name, node_type, getattr(new_node, 'metadata', {}))
        else:
            new_node_name = f'{target_node}_before_{uuid.uuid4().hex[:6]}'
            self.add_node(new_node_name, new_node, **kwargs)
        for edge in incoming_edges:
            self.remove_edge(edge[0], edge[1])
        for source, _ in incoming_edges:
            self.add_edge(source, new_node_name)
        self.add_edge(new_node_name, target_node)
        for branch, condition in incoming_branches:
            if condition == 'default':
                branch.default = new_node_name
            else:
                branch.destinations[condition] = new_node_name
        logger.debug(f"Inserted node '{new_node_name}' before '{target_node}' in graph '{self.name}'")
        return self

    def add_prelude_node(self, prelude_node: str | Node | dict | Any, node_obj: Any | None=None, **kwargs) -> 'BaseGraph':
        """Add a node at the beginning of the graph (after START).

        Args:
            prelude_node: Node to add at the start
            node_obj: If prelude_node is a string, this is the node object/callable
            **kwargs: Additional properties if creating from name

        Returns:
            Self for method chaining
        """
        start_edges = [(source, target) for source, target in self.edges if source == START]
        start_branches = [branch for branch in self.branches.values() if branch.source_node == START]
        if isinstance(prelude_node, str):
            self.add_node(prelude_node, node_obj, **kwargs)
            prelude_name = prelude_node
        elif isinstance(prelude_node, Node):
            self.add_node(prelude_node)
            prelude_name = prelude_node.name
        elif isinstance(prelude_node, dict):
            if 'name' not in prelude_node:
                raise ValueError("Node dictionary must have a 'name' field")
            self.add_node(prelude_node)
            prelude_name = prelude_node['name']
        elif hasattr(prelude_node, '__class__') and 'NodeConfig' in prelude_node.__class__.__name__:
            prelude_name = getattr(prelude_node, 'name', f'prelude_{uuid.uuid4().hex[:6]}')
            self.nodes[prelude_name] = prelude_node
            node_type = self._infer_node_type(prelude_node)
            self._track_node_type(prelude_name, node_type, getattr(prelude_node, 'metadata', {}))
        else:
            prelude_name = f'prelude_{uuid.uuid4().hex[:6]}'
            self.add_node(prelude_name, prelude_node, **kwargs)
        for edge in start_edges:
            self.remove_edge(START, edge[1])
        self.add_edge(START, prelude_name)
        for _, target in start_edges:
            self.add_edge(prelude_name, target)
        for branch in start_branches:
            new_branch = branch.model_copy(deep=True)
            new_branch.id = str(uuid.uuid4())
            new_branch.source_node = prelude_name
            self.add_branch(new_branch)
            self.remove_branch(branch.id)
        logger.debug(f"Added prelude node '{prelude_name}' to graph '{self.name}'")
        return self

    def add_postlude_node(self, postlude_node: str | Node | dict | Any, node_obj: Any | None=None, **kwargs) -> 'BaseGraph':
        """Add a node at the end of the graph (before END).

        Args:
            postlude_node: Node to add at the end
            node_obj: If postlude_node is a string, this is the node object/callable
            **kwargs: Additional properties if creating from name

        Returns:
            Self for method chaining
        """
        end_edges = [(source, target) for source, target in self.edges if target == END]
        end_branch_destinations = []
        for branch in self.branches.values():
            for condition, target in branch.destinations.items():
                if target == END:
                    end_branch_destinations.append((branch, condition))
            if branch.default == END:
                end_branch_destinations.append((branch, 'default'))
        if isinstance(postlude_node, str):
            self.add_node(postlude_node, node_obj, **kwargs)
            postlude_name = postlude_node
        elif isinstance(postlude_node, Node):
            self.add_node(postlude_node)
            postlude_name = postlude_node.name
        elif isinstance(postlude_node, dict):
            if 'name' not in postlude_node:
                raise ValueError("Node dictionary must have a 'name' field")
            self.add_node(postlude_node)
            postlude_name = postlude_node['name']
        elif hasattr(postlude_node, '__class__') and 'NodeConfig' in postlude_node.__class__.__name__:
            postlude_name = getattr(postlude_node, 'name', f'postlude_{uuid.uuid4().hex[:6]}')
            self.nodes[postlude_name] = postlude_node
            node_type = self._infer_node_type(postlude_node)
            self._track_node_type(postlude_name, node_type, getattr(postlude_node, 'metadata', {}))
        else:
            postlude_name = f'postlude_{uuid.uuid4().hex[:6]}'
            self.add_node(postlude_name, postlude_node, **kwargs)
        for edge in end_edges:
            self.remove_edge(edge[0], END)
        for source, _ in end_edges:
            self.add_edge(source, postlude_name)
        self.add_edge(postlude_name, END)
        for branch, condition in end_branch_destinations:
            if condition == 'default':
                branch.default = postlude_name
            else:
                branch.destinations[condition] = postlude_name
        logger.debug(f"Added postlude node '{postlude_name}' to graph '{self.name}'")
        return self

    def add_sequence(self, nodes: list[str | Node | dict | Any], node_objects: list[Any] | None=None, connect_start: bool=False, connect_end: bool=False, **kwargs) -> 'BaseGraph':
        """Add a sequence of nodes and connect them in order.

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
        if node_objects and len(node_objects) < len(nodes):
            node_objects.extend([None] * (len(nodes) - len(node_objects)))
        elif not node_objects:
            node_objects = [None] * len(nodes)
        node_names = []
        for i, node in enumerate(nodes):
            node_obj = node_objects[i] if node_objects else None
            if isinstance(node, str):
                self.add_node(node, node_obj, **kwargs)
                node_names.append(node)
            elif isinstance(node, Node):
                self.add_node(node, **kwargs)
                node_names.append(node.name)
            elif isinstance(node, dict):
                if 'name' not in node:
                    node['name'] = f'seq_node_{i}_{uuid.uuid4().hex[:6]}'
                self.add_node(node, **kwargs)
                node_names.append(node['name'])
            elif hasattr(node, '__class__') and 'NodeConfig' in node.__class__.__name__:
                node_name = getattr(node, 'name', f'seq_node_{i}_{uuid.uuid4().hex[:6]}')
                self.nodes[node_name] = node
                node_type = self._infer_node_type(node)
                self._track_node_type(node_name, node_type, getattr(node, 'metadata', {}))
                node_names.append(node_name)
            else:
                node_name = f'seq_node_{i}_{uuid.uuid4().hex[:6]}'
                self.add_node(node_name, node, **kwargs)
                node_names.append(node_name)
        if connect_start and node_names:
            self.add_edge(START, node_names[0])
        for i in range(len(node_names) - 1):
            self.add_edge(node_names[i], node_names[i + 1])
        if connect_end and node_names:
            self.add_edge(node_names[-1], END)
        logger.debug(f"Added sequence of {len(node_names)} nodes to graph '{self.name}'")
        return self

    def add_parallel_branches(self, source_node: str, branches: list[list[str] | list[Node] | list[dict] | list[Any]], branch_names: list[str] | None=None, join_node: str | Node | dict | Any | None=None, join_node_obj: Any | None=None, **kwargs) -> 'BaseGraph':
        """Add parallel branches from a source node, optionally joining at a common node.

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
        if not branch_names:
            branch_names = [f'branch_{i}_{uuid.uuid4().hex[:6]}' for i in range(len(branches))]
        elif len(branch_names) < len(branches):
            branch_names.extend([f'branch_{i}_{uuid.uuid4().hex[:6]}' for i in range(len(branch_names), len(branches))])
        branch_ends = []
        for i, branch_sequence in enumerate(branches):
            self.add_sequence(branch_sequence, **kwargs)
            if branch_sequence:
                first_node = None
                last_node = None
                if isinstance(branch_sequence[0], str):
                    first_node = branch_sequence[0]
                elif isinstance(branch_sequence[0], Node):
                    first_node = branch_sequence[0].name
                elif isinstance(branch_sequence[0], dict) and 'name' in branch_sequence[0]:
                    first_node = branch_sequence[0]['name']
                elif hasattr(branch_sequence[0], '__class__') and 'NodeConfig' in branch_sequence[0].__class__.__name__:
                    first_node = getattr(branch_sequence[0], 'name', f'branch_{i}_node_0')
                if isinstance(branch_sequence[-1], str):
                    last_node = branch_sequence[-1]
                elif isinstance(branch_sequence[-1], Node):
                    last_node = branch_sequence[-1].name
                elif isinstance(branch_sequence[-1], dict) and 'name' in branch_sequence[-1]:
                    last_node = branch_sequence[-1]['name']
                elif hasattr(branch_sequence[-1], '__class__') and 'NodeConfig' in branch_sequence[-1].__class__.__name__:
                    last_node = getattr(branch_sequence[-1], 'name', f'branch_{i}_node_last')
                if first_node and first_node in self.nodes:
                    self.add_edge(source_node, first_node)
                if last_node and last_node in self.nodes:
                    branch_ends.append(last_node)
        if join_node is not None:
            if isinstance(join_node, str):
                if join_node not in self.nodes:
                    self.add_node(join_node, join_node_obj, **kwargs)
                join_name = join_node
            elif isinstance(join_node, Node):
                self.add_node(join_node, **kwargs)
                join_name = join_node.name
            elif isinstance(join_node, dict):
                if 'name' not in join_node:
                    join_node['name'] = f'join_{uuid.uuid4().hex[:6]}'
                self.add_node(join_node, **kwargs)
                join_name = join_node['name']
            elif hasattr(join_node, '__class__') and 'NodeConfig' in join_node.__class__.__name__:
                join_name = getattr(join_node, 'name', f'join_{uuid.uuid4().hex[:6]}')
                self.nodes[join_name] = join_node
                node_type = self._infer_node_type(join_node)
                self._track_node_type(join_name, node_type, getattr(join_node, 'metadata', {}))
            else:
                join_name = f'join_{uuid.uuid4().hex[:6]}'
                self.add_node(join_name, join_node, **kwargs)
            for end_node in branch_ends:
                self.add_edge(end_node, join_name)
        logger.debug(f"Added {len(branches)} parallel branches to graph '{self.name}'")
        return self

    def add_edge(self, source: str, target: str) -> 'BaseGraph':
        """Add a direct edge to the graph.

        Args:
            source: Source node name
            target: Target node name

        Returns:
            Self for method chaining
        """
        if source != START and source not in self.nodes:
            raise ValueError(f"Source node '{source}' not found in graph")
        if target != END and target not in self.nodes:
            raise ValueError(f"Target node '{target}' not found in graph")
        edge = (source, target)
        if edge in self.edges:
            logger.warning(f'Edge {source} -> {target} already exists in graph')
            return self
        self.edges.append(edge)
        logger.debug(f"Added edge {source} -> {target} to graph '{self.name}'")
        self.updated_at = datetime.now()
        self._mark_needs_recompile(f'Added edge {source} -> {target}')
        return self

    def remove_edge(self, source: str, target: str | None=None) -> 'BaseGraph':
        """Remove an edge from the graph.

        Args:
            source: Source node name
            target: Target node name (if None, removes all edges from source)

        Returns:
            Self for method chaining
        """
        if target:
            self.edges = [edge for edge in self.edges if not (edge[0] == source and edge[1] == target)]
            logger.debug(f"Removed edge {source} -> {target} from graph '{self.name}'")
        else:
            self.edges = [edge for edge in self.edges if edge[0] != source]
            logger.debug(f"Removed all edges from {source} in graph '{self.name}'")
        self.updated_at = datetime.now()
        self._mark_needs_recompile(f'Removed edges from {source}')
        return self

    def get_edges(self, source: str | None=None, target: str | None=None, include_branches: bool=True) -> list[tuple[str, str]]:
        """Get edges matching criteria.

        Args:
            source: Filter by source node
            target: Filter by target node
            include_branches: Include edges from branches

        Returns:
            List of matching edges as (source, target) tuples
        """
        result = []
        for edge_source, edge_target in self.edges:
            source_match = source is None or edge_source == source
            target_match = target is None or edge_target == target
            if source_match and target_match:
                result.append((edge_source, edge_target))
        if include_branches:
            for branch in self.branches.values():
                if source is not None and branch.source_node != source:
                    continue
                for dest in branch.destinations.values():
                    if target is None or dest == target:
                        result.append((branch.source_node, dest))
                if branch.default and (target is None or branch.default == target):
                    result.append((branch.source_node, branch.default))
        return result

    def find_all_paths(self, start_node=START, end_node=END, max_depth=100, include_loops=False, debug=False):
        """Find all possible paths between two nodes.

        Args:
            start_node: Starting node (defaults to START)
            end_node: Ending node (defaults to END)
            max_depth: Maximum path depth to prevent infinite loops
            include_loops: Whether to include paths with loops/cycles
            debug: Whether to show detailed debug logging

        Returns:
            List of GraphPath objects
        """
        stats = {'nodes_visited': 0, 'paths_found': 0, 'max_depth_reached': 0, 'branches_explored': 0}
        if debug:
            logger.info(f'Finding paths from {start_node} to {end_node} (max_depth={max_depth}, include_loops={include_loops})')
        paths = []

        def dfs(current, path_obj, visited=None, depth=0, parent_branch=None) -> None:
            """Depth-first search to find all paths."""
            if visited is None:
                visited = set()
            stats['nodes_visited'] += 1
            stats['max_depth_reached'] = max(stats['max_depth_reached'], depth)
            if depth > max_depth:
                if debug:
                    logger.debug(f'Max depth reached at node {current} (depth={depth})')
                return
            if debug:
                prefix = '│   ' * depth
                if parent_branch:
                    logger.debug(f'{prefix}├── Exploring node: {current} (via {parent_branch})')
                else:
                    logger.debug(f'{prefix}├── Exploring node: {current}')
                logger.debug(f'{prefix}│   Path so far: {' → '.join(path_obj.nodes)}')
                logger.debug(f'{prefix}│   Depth: {depth}, Visited: {visited}')
            target_found = current == end_node
            if target_found:
                new_path = new_path = path_obj.append(current, is_conditional=False, is_end=True)
                paths.append(new_path)
                stats['paths_found'] += 1
                if debug:
                    logger.debug(f'{prefix}│   ✓ Found path to target!')
                    logger.debug(f'{prefix}│   Path: {' → '.join(new_path.nodes)}')
                if not include_loops or end_node == END:
                    if debug:
                        logger.debug(f'{prefix}│   Stopping exploration of this branch (target found)')
                    return
                if debug:
                    logger.debug(f'{prefix}│   Continuing exploration past target to find loops')
            if current in visited and (not include_loops):
                if debug:
                    logger.debug(f'{prefix}│   Already visited {current}, skipping (loops disabled)')
                return
            new_visited = visited.copy()
            new_visited.add(current)
            for src, dst in self.edges:
                if src == current:
                    if dst in visited and (not include_loops):
                        if debug:
                            logger.debug(f'{prefix}│   Skipping edge to {dst} (already visited)')
                        continue
                    if debug:
                        logger.debug(f'{prefix}│   → Following direct edge to {dst}')
                    is_end = dst == end_node
                    new_path = path_obj.append(dst, is_conditional=False, is_end=is_end)
                    dfs(dst, new_path, new_visited, depth + 1, f'direct edge from {src}')
            for _branch_id, branch in self.branches.items():
                if branch.source_node == current:
                    stats['branches_explored'] += 1
                    if debug:
                        logger.debug(f'{prefix}│   Exploring branch: {branch.name}')
                    for condition, target in branch.destinations.items():
                        if target in visited and (not include_loops):
                            if debug:
                                logger.debug(f'{prefix}│   Skipping branch to {target} (already visited)')
                            continue
                        if debug:
                            logger.debug(f'{prefix}│   → Following branch ({condition}) to {target}')
                        is_end = target == end_node
                        new_path = path_obj.append(target, is_conditional=True, is_end=is_end)
                        dfs(target, new_path, new_visited, depth + 1, f"branch condition '{condition}'")
                    if branch.default and branch.default != END:
                        if branch.default in visited and (not include_loops):
                            if debug:
                                logger.debug(f'{prefix}│   Skipping default branch to {branch.default} (already visited)')
                            continue
                        if debug:
                            logger.debug(f'{prefix}│   → Following default branch to {branch.default}')
                        is_end = branch.default == end_node
                        new_path = path_obj.append(branch.default, is_conditional=True, is_end=is_end)
                        dfs(branch.default, new_path, new_visited, depth + 1, 'default branch')
                    if branch.default == END:
                        if debug:
                            logger.debug(f'{prefix}│   → Following default branch to END')
                        new_path = path_obj.append(END, is_conditional=True, is_end=True)
                        paths.append(new_path)
                        stats['paths_found'] += 1
        initial_path = GraphPath(nodes=[start_node])
        dfs(start_node, initial_path)
        result_paths = []
        if end_node != END:
            for path in paths:
                if debug:
                    logger.debug(f'Checking path: {' → '.join(path.nodes)}')
                if end_node in path.nodes:
                    result_paths.append(path)
                    if debug:
                        logger.debug(f'✓ Path contains target {end_node}')
                    if path.nodes.count(end_node) > 1 and debug:
                        logger.debug(f'  This path contains multiple occurrences of {end_node} (loop)')
                elif debug:
                    logger.debug(f'✗ Path does not contain target {end_node}')
        else:
            result_paths = [path for path in paths if path.reaches_end]
        if debug:
            logger.info(f'Found {len(result_paths)} paths from {start_node} to {end_node}')
        return result_paths

    def check_graph_validity(self) -> Any:
        """Validate the graph structure.

        Returns:
            List of validation issues (empty if graph is valid)
        """
        issues = []
        if not self.get_edges(source=START):
            issues.append('No connections from START node')
        if not self.get_edges(target=END):
            issues.append('No connections to END node')
        orphaned = self.find_unreachable_nodes()
        if orphaned:
            issues.append(f'Orphaned nodes: {orphaned}')
        no_end_path = self.find_nodes_without_end_path()
        if no_end_path:
            issues.append(f'Nodes without path to END: {no_end_path}')
        return issues

    def find_unreachable_nodes(self) -> Any | None:
        """Find nodes that can't be reached from START.

        Returns:
            List of unreachable node names
        """
        reachable = set()

        def dfs(node) -> None:
            if node in reachable:
                return
            reachable.add(node)
            for _src, dst in self.get_edges(source=node):
                dfs(dst)
            for branch in self.branches.values():
                if branch.source_node == node:
                    for dst in branch.destinations.values():
                        dfs(dst)
                    if branch.default:
                        dfs(branch.default)
        dfs(START)
        return [node for node in self.nodes if node not in reachable and self.nodes[node] is not None]

    def find_nodes_without_end_path(self) -> Any | None:
        """Find nodes that can't reach END.

        Returns:
            List of node names that can't reach END
        """
        no_end_path = []
        for node_name in self.nodes:
            if self.nodes[node_name] is None:
                continue
            if not self.has_path(node_name, END):
                no_end_path.append(node_name)
        return no_end_path

    def find_nodes_without_finish_path(self) -> Any | None:
        """Find nodes that can't reach a finish point.
        Alias for find_nodes_without_end_path for API consistency.

        Returns:
            List of node names that can't reach a finish point
        """
        return self.find_nodes_without_end_path()

    def get_source_nodes(self) -> Any | None:
        """Get nodes that have no incoming edges (other than START).

        Returns:
            List of source node names
        """
        has_incoming = set()
        for _, dst in self.edges:
            has_incoming.add(dst)
        for branch in self.branches.values():
            for dst in branch.destinations.values():
                has_incoming.add(dst)
            if branch.default:
                has_incoming.add(branch.default)
        return [node for node in self.nodes if node not in has_incoming and node != START and (self.nodes[node] is not None)]

    def get_sink_nodes(self) -> Any | None:
        """Get nodes that have no outgoing edges (other than to END).

        Returns:
            List of sink node names
        """
        has_outgoing = set()
        for src, dst in self.edges:
            if dst != END:
                has_outgoing.add(src)
        for branch in self.branches.values():
            has_outgoing.add(branch.source_node)
        return [node for node in self.nodes if node not in has_outgoing and self.nodes[node] is not None]

    def add_branch(self, branch_or_name: Branch | str, source_node: str | None=None, condition: Any | None=None, routes: dict[bool | str, str] | None=None, branch_type: BranchType | None=None, **kwargs) -> 'BaseGraph':
        """Add a branch to the graph with flexible input options.

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
            branch = branch_or_name
            if branch.source_node is None:
                if source_node is None:
                    raise ValueError('Branch must have a source_node')
                branch.source_node = source_node
            if branch.source_node != START and branch.source_node not in self.nodes:
                raise ValueError(f"Source node '{branch.source_node}' not found in graph")
            for dest in branch.destinations.values():
                if dest != END and dest not in self.nodes:
                    raise ValueError(f"Destination node '{dest}' not found in graph")
            if branch.default != END and branch.default not in self.nodes:
                raise ValueError(f"Default node '{branch.default}' not found in graph")
            self.branches[branch.id] = branch
        else:
            branch_name = branch_or_name
            if not source_node:
                raise ValueError('source_node is required when adding a branch by name')
            if source_node != START and source_node not in self.nodes:
                raise ValueError(f"Source node '{source_node}' not found in graph")
            branch_data = {'name': branch_name, 'source_node': source_node, **kwargs}
            if callable(condition):
                branch_data['function'] = condition
                branch_data['function_ref'] = CallableReference.from_callable(condition)
                branch_data['mode'] = BranchMode.FUNCTION
            elif isinstance(condition, tuple) and len(condition) >= 2:
                branch_data['key'] = condition[0]
                branch_data['value'] = condition[1]
                if len(condition) >= 3:
                    branch_data['comparison'] = condition[2]
                else:
                    branch_data['comparison'] = ComparisonType.EQUALS
            elif condition:
                branch_data['key'] = kwargs.get('key')
                branch_data['value'] = condition
                branch_data['comparison'] = kwargs.get('comparison', ComparisonType.EQUALS)
            if routes:
                branch_data['destinations'] = routes
            if 'destinations' in branch_data:
                for dest in branch_data['destinations'].values():
                    if dest != END and dest not in self.nodes:
                        raise ValueError(f"Destination node '{dest}' not found in graph")
            branch = Branch(**branch_data)
            self.branches[branch.id] = branch
        logger.debug(f"Added branch '{branch.name}' from node '{branch.source_node}' to graph '{self.name}'")
        self.updated_at = datetime.now()
        return self

    def add_conditional_edges(self, source_node: str, condition: Branch | Callable[[StateLike, ConfigLike | None], BranchResultType] | Any, destinations: str | list[str] | dict[bool | str | int, str] | None=None, default: str | Literal['END'] | None=END, create_missing_nodes: bool=False) -> 'BaseGraph':
        """Add conditional edges from a source node based on a condition.

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
        if source_node != START and source_node not in self.nodes:
            raise ValueError(f"Source node '{source_node}' not found in graph")
        if isinstance(condition, Branch):
            branch = condition
            branch.source_node = source_node
            if destinations is not None:
                if isinstance(destinations, str):
                    branch.destinations = {True: destinations}
                elif isinstance(destinations, list):
                    if len(destinations) >= 2:
                        branch.destinations = {True: destinations[0], False: destinations[1]}
                    elif len(destinations) == 1:
                        branch.destinations = {True: destinations[0], False: END}
                    else:
                        branch.destinations = {True: 'continue', False: END}
                elif isinstance(destinations, dict):
                    branch.destinations = destinations
                    branch.default = None
            else:
                branch.destinations = {True: 'continue', False: END}
                branch.default = default
            if not create_missing_nodes:
                for dest_name in branch.destinations.values():
                    if dest_name != END and dest_name not in self.nodes:
                        raise ValueError(f"Destination node '{dest_name}' not found in graph. Use create_missing_nodes=True to create it automatically.")
            self.branches[branch.id] = branch
            logger.debug(f'Added branch with ID: {branch.id}')
            return self
        if hasattr(condition, '__class__') and 'NodeConfig' in condition.__class__.__name__:
            condition_name = getattr(condition, 'name', f'branch_condition_{uuid.uuid4().hex[:8]}')
            logger.debug(f"Using NodeConfig '{condition_name}' as branch controller")
        else:
            condition_name = getattr(condition, '__name__', f'branch_condition_{uuid.uuid4().hex[:8]}')
            logger.debug(f"Using function '{condition_name}' as branch controller")
        destination_map = {}
        if isinstance(destinations, dict):
            use_default = None
        else:
            use_default = default
        if isinstance(destinations, str):
            destination_map = {True: destinations, False: default}
        elif isinstance(destinations, list):
            if len(destinations) >= 2:
                destination_map = {True: destinations[0], False: destinations[1]}
            elif destinations:
                destination_map = {True: destinations[0], False: default}
            else:
                destination_map = {True: 'continue', False: default}
        elif isinstance(destinations, dict):
            destination_map = destinations
        elif destinations is None:
            destination_map = {True: 'continue', False: default}
        if not create_missing_nodes:
            for dest_key, dest_name in destination_map.items():
                if dest_name not in (END, 'continue') and dest_name not in self.nodes:
                    raise ValueError(f"Destination node '{dest_name}' for condition '{dest_key}' not found in graph. Use create_missing_nodes=True to create it automatically.")
        branch_id = str(uuid.uuid4())
        branch_name = f'branch_{branch_id[:8]}'
        metadata = {'condition_object': condition} if not callable(condition) else {}
        if callable(condition):
            function_to_use = self._create_branch_wrapper(condition, destination_map, use_default)
            function_ref = CallableReference.from_callable(function_to_use)
        else:
            function_to_use = condition
            function_ref = None
        if callable(condition) and hasattr(condition, '__name__'):
            logger.info(f'Using branch function: {condition.__name__}')
        elif hasattr(condition, '__class__'):
            logger.info(f'Using condition of type: {condition.__class__.__name__}')
        if hasattr(condition, '__class__') and 'ValidationNodeConfig' in condition.__class__.__name__:
            logger.info('Detected ValidationNodeConfig - ensuring correct routing')
            metadata['is_validation_node'] = True
            function_to_use = self._create_validation_wrapper(condition, destination_map)
            function_ref = CallableReference.from_callable(function_to_use)
        branch = Branch(id=branch_id, name=branch_name, source_node=source_node, function=function_to_use, function_ref=function_ref, metadata=metadata, mode=BranchMode.FUNCTION if callable(condition) else BranchMode.DIRECT, destinations=destination_map, default=use_default)
        logger.debug(f'Branch configuration: source={source_node}, destinations={destination_map}, default={use_default}')
        self.branches[branch_id] = branch
        logger.debug(f"Branch '{branch_name}' added successfully!")
        self.updated_at = datetime.now()
        self._mark_needs_recompile(f"Added conditional edges from '{source_node}'")
        return self

    def _create_validation_wrapper(self, validation_config, destination_map):
        """Special wrapper for ValidationNodeConfig to ensure correct routing.

        ValidationNodeConfig is a common source of routing issues because it can return
        complex results that don't directly match routing keys.
        """
        from langgraph.types import Send
        valid_keys = list(destination_map.keys())
        logger.info(f'Valid routing keys: {valid_keys}')

        def validation_wrapper(state: dict[str, Any], config: dict[str, Any] | None=None):
            try:
                if callable(validation_config):
                    result = validation_config(state, config)
                    logger.info(f'ValidationNodeConfig result: {type(result).__name__}')
                    if isinstance(result, list) and all((isinstance(item, Send) for item in result)):
                        logger.info(f'ValidationNodeConfig returned {len(result)} Send objects')
                        return result
                    if isinstance(result, Send):
                        logger.info(f'ValidationNodeConfig returned single Send object to {result.node}')
                        return result
                    if hasattr(result, '__class__') and 'Command' in result.__class__.__name__:
                        logger.info('ValidationNodeConfig returned Command object')
                        return result
                    if isinstance(result, str) and result in destination_map:
                        logger.info(f'ValidationNodeConfig returned string key: {result}')
                        return result
                    if isinstance(result, str) and result == 'no_tool_calls':
                        for key, dest in destination_map.items():
                            if dest == 'END' or str(dest).upper() == 'END':
                                logger.info(f"Converting 'no_tool_calls' to routing key: {key}")
                                return key
                        logger.info('ValidationNodeConfig returned no_tool_calls, returning as-is')
                        return result
                    if isinstance(result, dict):
                        for key in ['has_errors', 'has_tools', 'parse_output']:
                            if key in result and result[key] and (key in destination_map):
                                logger.info(f'Found validation key: {key}')
                                return key
                        for key, value in result.items():
                            if value is True and key in destination_map:
                                logger.info(f'Found True key: {key}')
                                return key
                    if 'has_errors' in destination_map:
                        logger.info('Defaulting to has_errors')
                        return 'has_errors'
                    if valid_keys:
                        logger.warning(f'No routing match found - using first key: {valid_keys[0]}')
                        return valid_keys[0]
                    return False
                if valid_keys:
                    logger.warning("ValidationNodeConfig doesn't have __call__ method")
                    return valid_keys[0]
                return False
            except Exception as e:
                logger.exception(f'Error in validation function: {e}')
                import traceback
                traceback.print_exc()
                if 'has_errors' in destination_map:
                    return 'has_errors'
                return False
        return validation_wrapper

    def _create_branch_wrapper(self, func, destination_map, default_dest):
        """Wrapper for branch functions that handles boolean to string conversion."""
        param_count = len(inspect.signature(func).parameters)
        has_boolean_keys = any((isinstance(k, bool) for k in destination_map))
        has_string_keys = any((isinstance(k, str) for k in destination_map))

        def wrapper(state: dict[str, Any], config: dict[str, Any] | None=None):
            try:
                result = func(state) if param_count == 1 else func(state, config)
                logger.debug(f'Branch function returned: {result} (type: {type(result).__name__})')
                if hasattr(func, '__name__') and 'has_tool_calls' in func.__name__:
                    logger.info('=== DEBUGGING has_tool_calls function ===')
                    logger.info(f'Function result: {result} (type: {type(result)})')
                    logger.info(f'Available routing keys: {list(destination_map.keys())}')
                    logger.info(f'State type: {type(state)}')
                    if hasattr(state, 'messages'):
                        messages = state.messages
                        logger.info(f'State has {len(messages)} messages')
                        if messages:
                            last_msg = messages[-1]
                            logger.info(f'Last message type: {type(last_msg)}')
                            logger.info(f'Last message: {last_msg}')
                            if hasattr(last_msg, 'tool_calls'):
                                tool_calls = getattr(last_msg, 'tool_calls', None)
                                logger.info(f'tool_calls attribute: {tool_calls}')
                            if hasattr(last_msg, 'additional_kwargs'):
                                additional_kwargs = getattr(last_msg, 'additional_kwargs', {})
                                logger.info(f'additional_kwargs: {additional_kwargs}')
                                if 'tool_calls' in additional_kwargs:
                                    logger.info(f'tool_calls in additional_kwargs: {additional_kwargs['tool_calls']}')
                    logger.info('=== END DEBUGGING ===')
                if isinstance(result, bool) and has_string_keys and (not has_boolean_keys):
                    if result is True:
                        for key in destination_map:
                            if key in ['has_tool_calls', 'has_tools', 'true', 'yes', 'continue']:
                                logger.debug(f'Converting True to string key: {key}')
                                return key
                        first_key = next(iter(destination_map.keys()))
                        logger.debug(f'Converting True to first key: {first_key}')
                        return first_key
                    for key in destination_map:
                        if key in ['no_tool_calls', 'no_tools', 'false', 'no', 'end']:
                            logger.debug(f'Converting False to string key: {key}')
                            return key
                    if default_dest and default_dest in destination_map.values():
                        for k, v in destination_map.items():
                            if v == default_dest:
                                logger.debug(f'Converting False to default key: {k}')
                                return k
                    last_key = list(destination_map.keys())[-1]
                    logger.debug(f'Converting False to last key: {last_key}')
                    return last_key
                return result
            except Exception as e:
                logger.exception(f'Error in branch function: {e}')
                import traceback
                traceback.print_exc()
                return False
        return wrapper

    @property
    def conditional_edges(self) -> Any:
        """Property for accessing branches as conditional edges (compatibility).

        Returns:
            Dictionary of branches indexed by ID
        """
        return self.branches

    def add_function_branch(self, source_node: str, condition: Callable[[Any], Any], routes: dict[bool | str, str], default_route: str=END, name: str | None=None) -> 'BaseGraph':
        """Add a function-based branch.

        Args:
            source_node: Source node name
            condition: Condition function
            routes: Mapping of condition results to target nodes
            default_route: Default destination
            name: Optional branch name

        Returns:
            Self for method chaining
        """
        if source_node != START and source_node not in self.nodes:
            raise ValueError(f"Source node '{source_node}' not found in graph")
        for dest in routes.values():
            if dest != END and dest not in self.nodes:
                raise ValueError(f"Destination node '{dest}' not found in graph")
        if default_route != END and default_route not in self.nodes:
            raise ValueError(f"Default node '{default_route}' not found in graph")
        branch = Branch(id=str(uuid.uuid4()), name=name or f'func_branch_{uuid.uuid4().hex[:6]}', source_node=source_node, function=condition, function_ref=CallableReference.from_callable(condition), mode=BranchMode.FUNCTION, destinations=routes, default=default_route)
        self.branches[branch.id] = branch
        logger.debug(f"Added function branch '{branch.name}' from {source_node} to graph '{self.name}'")
        return self

    def add_key_value_branch(self, source_node: str, key: str, value: Any, comparison: ComparisonType | str=ComparisonType.EQUALS, true_dest: str='continue', false_dest: str=END, name: str | None=None) -> 'BaseGraph':
        """Add a key-value comparison branch.

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
        if source_node != START and source_node not in self.nodes:
            raise ValueError(f"Source node '{source_node}' not found in graph")
        if true_dest != END and true_dest not in self.nodes:
            raise ValueError(f"True destination node '{true_dest}' not found in graph")
        if false_dest != END and false_dest not in self.nodes:
            raise ValueError(f"False destination node '{false_dest}' not found in graph")
        branch = Branch(id=str(uuid.uuid4()), name=name or f'kv_branch_{uuid.uuid4().hex[:6]}', source_node=source_node, key=key, value=value, comparison=comparison, destinations={True: true_dest, False: false_dest}, default=false_dest, mode=BranchMode.DIRECT)
        self.branches[branch.id] = branch
        logger.debug(f"Added key-value branch '{branch.name}' from {source_node} to graph '{self.name}'")
        return self

    def remove_branch(self, branch_id: str) -> 'BaseGraph':
        """Remove a branch from the graph.

        Args:
            branch_id: ID of the branch to remove

        Returns:
            Self for method chaining
        """
        if branch_id not in self.branches:
            raise ValueError(f"Branch '{branch_id}' not found in graph")
        del self.branches[branch_id]
        logger.debug(f"Removed branch '{branch_id}' from graph '{self.name}'")
        self.updated_at = datetime.now()
        return self

    def update_branch(self, branch_id: str, **updates) -> 'BaseGraph':
        """Update a branch's properties.

        Args:
            branch_id: ID of the branch to update
            **updates: Properties to update

        Returns:
            Self for method chaining
        """
        if branch_id not in self.branches:
            raise ValueError(f"Branch '{branch_id}' not found in graph")
        branch = self.branches[branch_id]
        for key, value in updates.items():
            if hasattr(branch, key):
                setattr(branch, key, value)
        if 'source_node' in updates:
            source = updates['source_node']
            if source != START and source not in self.nodes:
                raise ValueError(f"Updated source node '{source}' not found in graph")
        if 'destinations' in updates:
            for dest in updates['destinations'].values():
                if dest != END and dest not in self.nodes:
                    raise ValueError(f"Updated destination node '{dest}' not found in graph")
        if 'default' in updates:
            default = updates['default']
            if default != END and default not in self.nodes:
                raise ValueError(f"Updated default node '{default}' not found in graph")
        self.updated_at = datetime.now()
        return self

    def replace_branch(self, branch_id: str, new_branch: Branch) -> 'BaseGraph':
        """Replace a branch with a new one.

        Args:
            branch_id: ID of the branch to replace
            new_branch: New branch to insert

        Returns:
            Self for method chaining
        """
        if branch_id not in self.branches:
            raise ValueError(f"Branch '{branch_id}' not found in graph")
        if new_branch.source_node != START and new_branch.source_node not in self.nodes:
            raise ValueError(f"Source node '{new_branch.source_node}' not found in graph")
        for dest in new_branch.destinations.values():
            if dest != END and dest not in self.nodes:
                raise ValueError(f"Destination node '{dest}' not found in graph")
        if new_branch.default != END and new_branch.default not in self.nodes:
            raise ValueError(f"Default node '{new_branch.default}' not found in graph")
        if new_branch.id != branch_id:
            new_branch_copy = new_branch.model_copy(deep=True)
            new_branch_copy.id = branch_id
            self.branches[branch_id] = new_branch_copy
        else:
            self.branches[branch_id] = new_branch
        logger.debug(f"Replaced branch '{branch_id}' in graph '{self.name}'")
        self.updated_at = datetime.now()
        return self

    def get_branches_for_node(self, node_name: str) -> list[Branch]:
        """Get all branches with a given source node.

        Args:
            node_name: Name of the source node

        Returns:
            List of branch objects
        """
        return [branch for branch in self.branches.values() if branch.source_node == node_name]

    def get_branch(self, branch_id: str) -> Branch | None:
        """Get a branch by ID.

        Args:
            branch_id: ID of the branch to retrieve

        Returns:
            Branch object if found, None otherwise
        """
        return self.branches.get(branch_id)

    def get_branch_by_name(self, name: str) -> Branch | None:
        """Get a branch by name.

        Args:
            name: Name of the branch to retrieve

        Returns:
            First matching branch or None if not found
        """
        for branch in self.branches.values():
            if branch.name == name:
                return branch
        return None

    def has_path(self, source: str, target: str) -> bool:
        """Check if there is a path between source and target nodes.

        Args:
            source: Source node name
            target: Target node name

        Returns:
            True if path exists, False otherwise
        """
        visited = set()
        queue = [source]
        while queue:
            current = queue.pop(0)
            if current == target:
                return True
            if current in visited:
                continue
            visited.add(current)
            for _src, dest in self.get_edges(source=current, include_branches=True):
                if dest not in visited:
                    queue.append(dest)
        return False

    def get_node_pattern(self, pattern: str) -> list[Node]:
        """Get nodes matching a pattern.

        Args:
            pattern: Pattern to match (supports * wildcard)

        Returns:
            List of matching nodes
        """
        import fnmatch
        return [node for name, node in self.nodes.items() if node is not None and fnmatch.fnmatch(name, pattern)]

    def extend_from(self, other_graph, prefix='') -> Any:
        """Extend this graph with nodes and edges from another graph.

        Args:
            other_graph: Graph to extend from
            prefix: Optional prefix for imported node names

        Returns:
            Self for method chaining
        """
        for name, node in other_graph.nodes.items():
            if node is None:
                new_name = f'{prefix}_{name}' if prefix else name
                self.nodes[new_name] = None
                continue
            new_name = f'{prefix}_{name}' if prefix else name
            if new_name not in self.nodes:
                if hasattr(node, 'model_copy'):
                    new_node = node.model_copy(deep=True)
                    new_node.name = new_name
                    self.nodes[new_name] = new_node
                elif hasattr(node, '__class__') and 'NodeConfig' in node.__class__.__name__:
                    self.nodes[new_name] = node
                else:
                    new_node = Node(name=new_name, node_type=getattr(node, 'node_type', NodeType.CALLABLE), metadata=getattr(node, 'metadata', {}).copy(), input_mapping=getattr(node, 'input_mapping', None), output_mapping=getattr(node, 'output_mapping', None), command_goto=getattr(node, 'command_goto', None), retry_policy=getattr(node, 'retry_policy', None), description=getattr(node, 'description', None))
                    self.nodes[new_name] = new_node
                if hasattr(other_graph, 'node_types') and name in other_graph.node_types:
                    self._track_node_type(new_name, other_graph.node_types[name])
                if hasattr(other_graph, 'subgraphs') and name in other_graph.subgraphs:
                    if not hasattr(self, 'subgraphs'):
                        self.subgraphs = {}
                    self.subgraphs[new_name] = other_graph.subgraphs[name]
        for src, dst in other_graph.edges:
            new_src = f'{prefix}_{src}' if prefix and src not in (START, END) else src
            new_dst = f'{prefix}_{dst}' if prefix and dst not in (START, END) else dst
            if (new_src == START or new_src in self.nodes) and (new_dst == END or new_dst in self.nodes):
                self.add_edge(new_src, new_dst)
        if hasattr(other_graph, 'branches'):
            for _branch_id, branch in other_graph.branches.items():
                new_src = f'{prefix}_{branch.source_node}' if prefix and branch.source_node != START else branch.source_node
                if new_src != START and new_src not in self.nodes:
                    continue
                new_destinations = {}
                for cond, dest in branch.destinations.items():
                    new_dest = f'{prefix}_{dest}' if prefix and dest != END else dest
                    if new_dest == END or new_dest in self.nodes:
                        new_destinations[cond] = new_dest
                new_default = f'{prefix}_{branch.default}' if prefix and branch.default != END else branch.default
                if new_default != END and new_default not in self.nodes:
                    new_default = END
                if hasattr(branch, 'model_copy'):
                    new_branch = branch.model_copy(deep=True)
                    new_branch.id = str(uuid.uuid4())
                    new_branch.source_node = new_src
                    new_branch.destinations = new_destinations
                    new_branch.default = new_default
                else:
                    new_branch = Branch(id=str(uuid.uuid4()), name=f'{prefix}_{branch.name}' if prefix else branch.name, source_node=new_src, function=getattr(branch, 'function', None), key=getattr(branch, 'key', None), value=getattr(branch, 'value', None), comparison=getattr(branch, 'comparison', None), destinations=new_destinations, default=new_default, mode=getattr(branch, 'mode', 'FUNCTION'))
                self.branches[new_branch.id] = new_branch
        logger.info(f'Extended graph with {len(other_graph.nodes)} nodes from {other_graph.name}')
        self.updated_at = datetime.now()
        return self

    def to_langgraph(self, state_schema: type[BaseModel] | None=None, input_schema: type[BaseModel] | None=None, output_schema: type[BaseModel] | None=None, config_schema: type[BaseModel] | None=None, **kwargs) -> Any:
        """Convert to LangGraph StateGraph with proper schema handling.

        Schema Resolution Logic:
        1. If state_schema provided: use it, default input/output to state_schema
        2. If input_schema and output_schema provided: use them, create PassThroughState for state_schema
        3. If only input_schema provided: use it for both input and state, output defaults to state
        4. If only output_schema provided: use it for both output and state, input defaults to state
        5. If none provided: use self.state_schema or dict

        Note: This method marks the graph as compiled after successful conversion.
        """
        if self.needs_recompile():
            logger.info(f"Graph '{self.name}' needs recompilation")
        try:
            from langgraph.graph import StateGraph
            logger.info('Converting to LangGraph StateGraph')
            resolved_state_schema = None
            resolved_input_schema = None
            resolved_output_schema = None
            resolved_config_schema = config_schema
            if state_schema is not None:
                resolved_state_schema = state_schema
                resolved_input_schema = input_schema or state_schema
                resolved_output_schema = output_schema or state_schema
                logger.info(f'Using provided state_schema: {state_schema.__name__}')
            elif input_schema is not None and output_schema is not None:
                resolved_input_schema = input_schema
                resolved_output_schema = output_schema

                class PassThroughState(BaseModel):
                    model_config = ConfigDict(arbitrary_types_allowed=True)
                resolved_state_schema = PassThroughState
                logger.info(f'Created PassThroughState, input: {input_schema.__name__}, output: {output_schema.__name__}')
            elif input_schema is not None:
                resolved_input_schema = input_schema
                resolved_state_schema = input_schema
                resolved_output_schema = output_schema or input_schema
                logger.info(f'Using input_schema as state: {input_schema.__name__}')
            elif output_schema is not None:
                resolved_output_schema = output_schema
                resolved_state_schema = output_schema
                resolved_input_schema = input_schema or output_schema
                logger.info(f'Using output_schema as state: {output_schema.__name__}')
            else:
                resolved_state_schema = getattr(self, 'state_schema', dict)
                resolved_input_schema = resolved_state_schema
                resolved_output_schema = resolved_state_schema
                schema_name = resolved_state_schema.__name__ if hasattr(resolved_state_schema, '__name__') else str(resolved_state_schema)
                logger.info(f'Using default schema: {schema_name}')
            schema_info = {'State': resolved_state_schema.__name__ if hasattr(resolved_state_schema, '__name__') else str(resolved_state_schema), 'Input': resolved_input_schema.__name__ if hasattr(resolved_input_schema, '__name__') else str(resolved_input_schema), 'Output': resolved_output_schema.__name__ if hasattr(resolved_output_schema, '__name__') else str(resolved_output_schema)}
            if resolved_config_schema:
                schema_info['Config'] = resolved_config_schema.__name__
            logger.info(f'Final Schemas: {schema_info}')
            graph_builder = StateGraph(resolved_state_schema)
            logger.info('Created StateGraph')

            def log_function_call(func, name: str):
                import inspect
                sig = inspect.signature(func)
                param_count = len(sig.parameters)
                logger.debug(f'Node [yellow]{name}[/yellow]: Function accepts {param_count} parameter(s)')

                def inner(state: dict[str, Any], config: dict[str, Any] | None=None):
                    try:
                        if param_count == 1:
                            logger.debug(f'Calling {name} with 1 parameter (state only)')
                            result = func(state)
                        else:
                            logger.debug(f'Calling {name} with 2 parameters (state and config)')
                            result = func(state, config)
                        logger.debug(f'Node {name} returned: {type(result).__name__}')
                        from langgraph.types import Command
                        if isinstance(result, Command):
                            command_info = {'Type': type(result).__name__, 'Update': str(getattr(result, 'update', None)), 'Branch': str(getattr(result, 'branch', None)), 'Raw': str(result)}
                            logger.debug(f'Command Details - {name}', command_info)
                        return result
                    except Exception as e:
                        logger.exception(f'Error in {name}: {e!s}')
                        raise
                return inner
            logger.info('Adding Nodes')
            for node_name, node in self.nodes.items():
                if node_name in [START, END] or node is None:
                    continue
                action = None
                if callable(node):
                    action = node
                    logger.info(f'Node [yellow]{node_name}[/yellow]: Using direct callable')
                elif hasattr(node, 'metadata') and 'callable' in node.metadata and callable(node.metadata['callable']):
                    action = node.metadata['callable']
                    logger.info(f'Node [yellow]{node_name}[/yellow]: Using metadata callable')
                elif callable(node) and callable(node.__call__):
                    action = node
                    logger.info(f'Node [yellow]{node_name}[/yellow]: Using __call__ method')
                else:
                    logger.warning(f'Node [yellow]{node_name}[/yellow]: No callable found, using pass-through')

                    def action(state: dict[str, Any], config: dict[str, Any] | None=None):
                        return state
                import os
                if os.environ.get('PRODUCTION') != 'true':
                    action = log_function_call(action, node_name)
                graph_builder.add_node(node_name, action)
            logger.info('Adding Edges')
            for source, target in self.edges:
                graph_builder.add_edge(source, target)
                logger.debug(f'{source} → {target}')
            logger.info('Adding Branches')
            for _branch_id, branch in self.branches.items():
                source = branch.source_node
                destinations = {}
                for key, value in branch.destinations.items():
                    destinations[key] = value
                logger.info(f'Branch from [yellow]{source}[/yellow] with conditions: {list(destinations.keys())}')
                logger.debug(f'Destinations dict: {destinations}')
                if branch.mode == BranchMode.FUNCTION and branch.function:
                    import inspect
                    try:
                        sig = inspect.signature(branch.function)
                        param_count = len(sig.parameters)
                        is_validation_node = getattr(branch, 'metadata', {}).get('is_validation_node', False)
                        if is_validation_node:
                            logger.info(f"[bold magenta]Special handling for ValidationNodeConfig in '{branch.name}'[/bold magenta]")

                        def branch_wrapper(branch_func, param_count, branch_name, dest_dict):

                            def wrapper(state: dict[str, Any], config: dict[str, Any] | None=None):
                                try:
                                    if param_count == 1:
                                        result = branch_func(state)
                                    else:
                                        result = branch_func(state, config)
                                    if isinstance(result, list) and all((isinstance(item, Send) for item in result)):
                                        logger.info(f'Branch returning list of {len(result)} Send objects')
                                        return result
                                    if isinstance(result, Send):
                                        logger.info(f'Branch returning Send object to {result.target}')
                                        return result
                                    if is_validation_node and isinstance(result, dict):
                                        for key in ['has_errors', 'has_tools', 'parse_output']:
                                            if key in result and result[key] and (key in dest_dict):
                                                logger.info(f'Found validation key: {key}')
                                                return key
                                        for key, value in result.items():
                                            if value is True and key in dest_dict:
                                                logger.info(f'Found True key: {key}')
                                                return key
                                        if 'has_errors' in dest_dict:
                                            logger.info('Using has_errors for validation result')
                                            return 'has_errors'
                                    return result
                                except Exception as e:
                                    logger.exception(f'Error in branch: {e!s}')
                                    return False
                            return wrapper
                        branch_func = branch_wrapper(branch.function, param_count, branch.name, destinations)
                        try:
                            logger.debug(f'Adding conditional edges: {source} -> {destinations}')
                            graph_builder.add_conditional_edges(source, branch_func, destinations)
                            logger.info(f'✓ Successfully added conditional edges for {source}')
                        except Exception as e:
                            logger.exception(f'Error adding conditional edges for {source}: {e}')
                            logger.exception(f'  Destinations: {destinations}')
                            logger.exception(f'  Function: {branch_func}')
                            raise
                    except Exception as e:
                        logger.warning(f'Could not inspect branch function: {e!s}')
                        logger.warning('Falling back to original function')
                        try:
                            graph_builder.add_conditional_edges(source, branch.function, destinations)
                            logger.info(f'✓ Successfully added conditional edges for {source} (fallback)')
                        except Exception as fallback_error:
                            logger.exception(f'Fallback also failed for {source}: {fallback_error}')
                            logger.exception(f'  Destinations: {destinations}')
                            logger.exception(f'  Original function: {branch.function}')
                            raise
                else:
                    logger.info(f'Using branch object directly for {branch.name}')
                    try:
                        graph_builder.add_conditional_edges(source, branch, destinations)
                        logger.info(f'✓ Successfully added conditional edges for {source} (branch object)')
                    except Exception as e:
                        logger.exception(f'Error using branch object for {source}: {e}')
                        logger.exception(f'  Destinations: {destinations}')
                        logger.exception(f'  Branch: {branch}')
                        raise
            logger.info('LangGraph conversion complete!')
            self.mark_compiled()
            return graph_builder
        except ImportError:
            raise ImportError('LangGraph not installed. Install with: pip install langgraph')

    @classmethod
    def from_langgraph(cls, state_graph: Any, name: str | None=None) -> 'BaseGraph':
        """Create a BaseGraph from a LangGraph StateGraph.

        Args:
            state_graph: LangGraph StateGraph instance
            name: Optional name for the created graph

        Returns:
            BaseGraph instance
        """
        try:
            if not hasattr(state_graph, 'nodes') or not hasattr(state_graph, 'edges'):
                raise ValueError('Input must be a LangGraph StateGraph')
            graph = cls(name=name or getattr(state_graph, 'name', f'graph_{uuid.uuid4().hex[:8]}'), state_schema=getattr(state_graph, 'schema', None))
            for node_name, node_spec in state_graph.nodes.items():
                if node_name in [START, END]:
                    continue
                action = getattr(node_spec, 'action', None)
                metadata = getattr(node_spec, 'metadata', {})
                retry_policy = getattr(node_spec, 'retry_policy', None)
                node = Node(name=node_name, node_type=NodeType.CALLABLE if callable(action) else NodeType.ENGINE, metadata=metadata or {}, retry_policy=retry_policy)
                if action:
                    node.metadata['callable'] = action
                graph.add_node(node)
            if hasattr(state_graph, 'edges'):
                for source, target in state_graph.edges:
                    graph.add_edge(source, target)
            if hasattr(state_graph, 'branches'):
                for source_node, conditions in state_graph.branches.items():
                    for condition_name, branch_obj in conditions.items():
                        destinations = {}
                        if hasattr(branch_obj, 'ends'):
                            for condition, target in branch_obj.ends.items():
                                destinations[condition] = target
                        branch_func = None
                        if hasattr(branch_obj, 'condition'):
                            branch_func = branch_obj.condition
                        branch_id = str(uuid.uuid4())
                        branch = Branch(id=branch_id, name=f'branch_{condition_name}_{uuid.uuid4().hex[:6]}', source_node=source_node, function=branch_func, function_ref=CallableReference.from_callable(branch_func) if branch_func else None, destinations=destinations, default=END, mode=BranchMode.FUNCTION if branch_func else BranchMode.DIRECT)
                        graph.branches[branch_id] = branch
            return graph
        except ImportError:
            logger.exception('LangGraph not installed or not found')
            raise ImportError('LangGraph must be installed to convert from StateGraph')

    def to_dict(self) -> dict[str, Any]:
        """Convert the graph to a serializable dictionary.

        Returns:
            Dictionary representation of the graph
        """
        from haive.core.graph.state_graph.serializable import SerializableGraph
        serializable = SerializableGraph.from_graph(self)
        return serializable.to_dict()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'BaseGraph':
        """Create a graph from a dictionary.

        Args:
            data: Dictionary representation of the graph

        Returns:
            Instantiated graph
        """
        from haive.core.graph.state_graph.serializable import SerializableGraph
        serializable = SerializableGraph.from_dict(data)
        return serializable.to_graph()

    def to_json(self, **kwargs) -> str:
        """Convert graph to JSON string.

        Args:
            **kwargs: Additional parameters for JSON serialization

        Returns:
            JSON string representation of the graph
        """
        from haive.core.graph.state_graph.serializable import SerializableGraph
        serializable = SerializableGraph.from_graph(self)
        return serializable.to_json(**kwargs)

    @classmethod
    def from_json(cls, json_str: str) -> 'BaseGraph':
        """Create a graph from a JSON string.

        Args:
            json_str: JSON string representation of the graph

        Returns:
            Instantiated graph
        """
        from haive.core.graph.state_graph.serializable import SerializableGraph
        serializable = SerializableGraph.from_json(json_str)
        return serializable.to_graph()

    def visualize(self, output_path: str | None=None, include_subgraphs: bool=True, highlight_nodes: list[str] | None=None, highlight_paths: list[list[str]] | None=None, save_png: bool=True, width: str='100%', theme: str='default', subgraph_mode: str='cluster', show_default_branches: bool=False, debug: bool=False) -> str:
        """Generate and display a visualization of the graph.

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
            subgraph_mode: How to render subgraphs ("cluster", "inline", or "separate") - DEPRECATED
            show_default_branches: Whether to show default branches
            debug: Whether to enable debug output

        Returns:
            The generated Mermaid code
        """
        from haive.core.graph.state_graph.graph_visualizer import GraphVisualizer
        from haive.core.utils.mermaid_utils import Environment, detect_environment
        if debug:
            debug_info = {'Graph': self.name, 'Type': type(self).__name__, 'Has nodes': hasattr(self, 'nodes'), 'Has edges': hasattr(self, 'edges'), 'Has subgraphs': hasattr(self, 'subgraphs')}
            if hasattr(self, 'nodes'):
                nodes_count = len(self.nodes) if self.nodes is not None else 0
                debug_info['Nodes count'] = nodes_count
                if self.nodes:
                    debug_info['Node names'] = list(self.nodes.keys())
                    for name, node in self.nodes.items():
                        logger.debug(f"Node '{name}' - Type: {(type(node).__name__ if node else 'None')}, Has graph: {(hasattr(node, 'graph') if node else False)}, Graph type: {(type(node.graph).__name__ if node and hasattr(node, 'graph') and node.graph else None)}")
            if hasattr(self, 'edges'):
                edges_count = len(self.edges) if self.edges is not None else 0
                debug_info['Edges count'] = edges_count
                if self.edges:
                    debug_info['Edges'] = str(list(self.edges))
            if hasattr(self, 'subgraphs'):
                subgraphs_count = len(self.subgraphs) if self.subgraphs is not None else 0
                debug_info['Subgraphs count'] = subgraphs_count
                if self.subgraphs:
                    debug_info['Subgraph names'] = list(self.subgraphs.keys())
            logger.debug(f'Graph Structure: {debug_info}')
        logger.debug(f'Visualizing graph: {self.name} (nodes: {(len(self.nodes) if self.nodes else 0)}, edges: {(len(self.edges) if self.edges else 0)})')
        if include_subgraphs and hasattr(self, 'subgraphs') and self.subgraphs:
            subgraph_info = ', '.join([f'{name} ({(len(sg.nodes) if hasattr(sg, 'nodes') and sg.nodes else 0)} nodes)' for name, sg in self.subgraphs.items()])
            logger.debug(f'Including {len(self.subgraphs)} subgraphs: {subgraph_info}')
            if debug:
                logger.info(f'Subgraph info: {subgraph_info}')
        all_highlight_nodes = highlight_nodes or []
        if highlight_paths:
            for path in highlight_paths:
                all_highlight_nodes.extend(path)
            logger.debug(f'Highlighting {len(all_highlight_nodes)} nodes')
        if debug:
            logger.info('Running structure analysis...')
            try:
                debug_info = GraphVisualizer.debug_graph_structure(self)
                logger.debug(f'Structure Analysis: {debug_info}')
            except Exception as e:
                logger.exception(f'Error in structure analysis: {e}')
                if debug:
                    import traceback
                    traceback.print_exc()
        try:
            if debug:
                logger.info('Attempting to generate Mermaid code...')
            mermaid_code = GraphVisualizer.generate_mermaid(self, include_subgraphs=include_subgraphs, highlight_nodes=all_highlight_nodes if all_highlight_nodes else None, theme=theme, show_branch_labels=show_default_branches, direction='TD', compact=False, max_depth=3, debug=debug)
            if debug:
                logger.info(f'Successfully generated Mermaid code: {len(mermaid_code)} characters')
            logger.debug(f'Generated Mermaid code: {len(mermaid_code)} characters')
        except Exception as e:
            if debug:
                logger.exception(f'Error generating Mermaid code: {e}')
                import traceback
                traceback.print_exc()
            logger.exception(f'Error generating Mermaid code: {e!s}')
            if include_subgraphs and hasattr(self, 'subgraphs') and self.subgraphs:
                logger.warning('Retrying visualization with subgraphs disabled')
                if debug:
                    logger.info('Retrying with subgraphs disabled...')
                try:
                    mermaid_code = GraphVisualizer.generate_mermaid(self, include_subgraphs=False, highlight_nodes=all_highlight_nodes if all_highlight_nodes else None, theme=theme, show_branch_labels=show_default_branches, direction='TD', compact=False, max_depth=3, debug=debug)
                    if debug:
                        logger.info('Successfully generated Mermaid code without subgraphs')
                except Exception as e2:
                    if debug:
                        logger.exception(f'Failed even without subgraphs: {e2}')
                        import traceback
                        traceback.print_exc()
                    logger.exception(f'Failed to generate Mermaid code even without subgraphs: {e2!s}')
                    return f'Error generating graph visualization: {e!s}'
            else:
                if debug:
                    logger.exception('No subgraphs to disable, returning error')
                return f'Error generating graph visualization: {e!s}'
        try:
            if debug:
                logger.info('Attempting to display graph...')
            from haive.core.utils.mermaid_utils import display_mermaid
            display_mermaid(mermaid_code, output_path=output_path, save_png=save_png, width=width)
            if output_path and save_png:
                logger.info(f'Graph visualization saved to: {output_path}')
                if debug:
                    logger.info(f'Graph saved to: {output_path}')
        except Exception as e:
            try:
                GraphVisualizer.display_graph(self, output_path=output_path, include_subgraphs=include_subgraphs, highlight_nodes=highlight_nodes, highlight_paths=highlight_paths, save_png=save_png, width=width, theme=theme, title=None, debug=debug)
            except Exception as e2:
                env = detect_environment()
                if debug:
                    logger.exception(f'Error displaying graph: {e}')
                    logger.exception(f'Error with alternative display: {e2}')
                    import traceback
                    traceback.print_exc()
                logger.exception(f'Error displaying graph: {e}')
                logger.exception(f'Detected environment: {env}')
                if env == Environment.JUPYTER_LAB:
                    suggestions = ['Install JupyterLab Mermaid extension: jupyter labextension install @jupyterlab/mermaid', 'Run this in a cell to display using HTML:', '   from IPython.display import HTML', '   HTML(f\'\'\'<div class="mermaid">{mermaid_code}</div>', '   <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>', "   <script>mermaid.initialize({startOnLoad:true});</script>''')"]
                    logger.info('\n'.join(suggestions))
                elif env == Environment.JUPYTER_NOTEBOOK:
                    suggestions = ['Run this in a cell to display using HTML:', '   from IPython.display import HTML', f"   mermaid_code = '''{mermaid_code}'''", '   HTML(f\'\'\'<div class="mermaid">{mermaid_code}</div>', '   <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>', "   <script>mermaid.initialize({startOnLoad:true});</script>''')"]
                    logger.info('\n'.join(suggestions))
                elif env == Environment.VSCODE_NOTEBOOK:
                    suggestions = ['Install Mermaid Preview extension for VSCode', "Save the diagram and view manually: my_graph.save_visualization('diagram.png')"]
                    logger.info('\n'.join(suggestions))
        return mermaid_code

    def to_mermaid(self, include_subgraphs: bool=True, theme: str='default', subgraph_mode: str='cluster', show_default_branches: bool=False) -> str:
        """Generate a Mermaid graph diagram string.

        Args:
            include_subgraphs: Whether to visualize subgraphs as clusters
            theme: Mermaid theme name (default, forest, dark, neutral)
            subgraph_mode: How to render subgraphs ("cluster", "inline", or "separate") - DEPRECATED
            show_default_branches: Whether to show default branches

        Returns:
            Mermaid diagram as string
        """
        from haive.core.graph.state_graph.graph_visualizer import GraphVisualizer
        return GraphVisualizer.generate_mermaid(self, include_subgraphs=include_subgraphs, highlight_nodes=None, theme=theme, show_branch_labels=show_default_branches, direction='TD', compact=False, max_depth=3, debug=False)

    def analyze_cycles(self) -> list[list[str]]:
        """Find all cycles in the graph."""
        cycles = []
        visited = set()
        path = []
        path_set = set()

        def dfs(node) -> None:
            if node in path_set:
                cycle_start = path.index(node)
                cycles.append(path[cycle_start:] + [node])
                return
            if node in visited:
                return
            visited.add(node)
            path.append(node)
            path_set.add(node)
            for src, dst in self.edges:
                if src == node and dst != END:
                    dfs(dst)
            for branch in self.branches.values():
                if branch.source_node == node:
                    for dest in branch.destinations.values():
                        if dest != END:
                            dfs(dest)
                    if branch.default and branch.default != END:
                        dfs(branch.default)
            path.pop()
            path_set.remove(node)
        for node in self.nodes:
            if node not in visited:
                dfs(node)
        return cycles

    def find_orphan_nodes(self) -> list[str]:
        """Find nodes with no incoming or outgoing edges."""
        orphans = []
        for node_name in self.nodes:
            if node_name in [START, END] or self.nodes[node_name] is None:
                continue
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
            has_outgoing = False
            for src, _ in self.edges:
                if src == node_name:
                    has_outgoing = True
                    break
            for branch in self.branches.values():
                if branch.source_node == node_name:
                    has_outgoing = True
                    break
            if not has_incoming and (not has_outgoing):
                orphans.append(node_name)
        return orphans

    def find_dangling_edges(self) -> list[tuple[str, str]]:
        """Find edges pointing to non-existent nodes."""
        dangling = []
        for src, dst in self.edges:
            if src != START and src not in self.nodes:
                dangling.append((src, dst))
            if dst != END and dst not in self.nodes:
                dangling.append((src, dst))
        for branch in self.branches.values():
            src = branch.source_node
            if src != START and src not in self.nodes:
                for dest in branch.destinations.values():
                    dangling.append((src, dest))
            else:
                for dest in branch.destinations.values():
                    if dest != END and dest not in self.nodes:
                        dangling.append((src, dest))
                if branch.default and branch.default != END and (branch.default not in self.nodes):
                    dangling.append((src, branch.default))
        return dangling

    def has_entry_point(self) -> bool:
        """Check if the graph has an entry point."""
        for src, _ in self.edges:
            if src == START:
                return True
        return any((branch.source_node == START for branch in self.branches.values()))

    def compile(self, raise_on_validation_error: bool=False) -> Any:
        """Validate and compile the graph to a runnable LangGraph StateGraph.

        Args:
            raise_on_validation_error: Whether to raise an exception on validation errors

        Returns:
            Compiled LangGraph StateGraph

        Raises:
            ValueError: If validation fails and raise_on_validation_error is True
        """
        issues = self.validate_graph()
        if issues:
            logger.error('Graph Validation Issues:\n' + '\n'.join([f'- {issue}' for issue in issues]))
            if raise_on_validation_error:
                raise ValueError(f'Graph validation failed with {len(issues)} issues')
            logger.warning('Proceeding with compilation despite validation issues')
        graph = self.to_langgraph()
        compiled_graph = graph.compile()
        return compiled_graph

    def add_boolean_conditional_edges(self, source_node: str, condition: Callable[[Any], bool], true_destination: str, false_destination: str=END, also_accept_strings: bool=True) -> 'BaseGraph':
        """Add conditional edges that explicitly handle boolean results.

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
        destinations = {True: true_destination, False: false_destination}
        if also_accept_strings:
            if 'tool' in true_destination.lower() or 'validation' in true_destination.lower():
                destinations['has_tool_calls'] = true_destination
                destinations['no_tool_calls'] = false_destination
                destinations['has_tools'] = true_destination
                destinations['no_tools'] = false_destination
            destinations['yes'] = true_destination
            destinations['no'] = false_destination
            destinations['true'] = true_destination
            destinations['false'] = false_destination
        logger.debug(f'Boolean conditional edges destinations: {destinations}')
        return self.add_conditional_edges(source_node=source_node, condition=condition, destinations=destinations, default=None)

    def debug_conditional_routing(self, source_node: str) -> None:
        """Debug conditional routing for a specific node.

        This shows how boolean and string results will be routed for debugging purposes.

        Args:
            source_node: The node to debug routing for
        """
        node_branches = self.get_branches_for_node(source_node)
        if not node_branches:
            logger.warning(f"No conditional routing found for node '{source_node}'")
            return
        logger.info(f"Conditional Routing Debug for '{source_node}'")
        for i, branch in enumerate(node_branches):
            logger.info(f'\n[cyan]Branch {i + 1}: {branch.name}[/cyan]')
            routing_data = {}
            for condition_result, destination in branch.destinations.items():
                result_type = 'Boolean' if isinstance(condition_result, bool) else 'String'
                routing_data[f'{condition_result} ({result_type})'] = destination
            if branch.default:
                routing_data['(default)'] = branch.default
            logger.table(f'Routing Map - {branch.name}', routing_data)
            if branch.function:
                func_name = getattr(branch.function, '__name__', 'Unknown')
                logger.info(f'[dim]Condition function: {func_name}[/dim]')
        tips = ['• Functions returning True/False will use boolean keys (True, False)', '• String-based keys automatically get boolean fallbacks', '• Use graph.add_boolean_conditional_edges() for explicit True/False routing', '• Enable debug logging: logger.setLevel(logging.DEBUG)']
        logger.info('\n'.join(tips), title='💡 Routing Tips', style='blue')

    def add_intelligent_agent_routing(self, agents: dict[str, Any], execution_mode: str='infer', branches: dict[str, dict[str, Any]] | None=None, prefix: str='agent_') -> 'BaseGraph':
        """Add intelligent agent routing with sequence inference and branching.

        Args:
            agents: Dictionary of agent name -> agent instance
            execution_mode: Execution mode (infer, sequential, parallel, branch, conditional)
            branches: Branch configurations for conditional routing
            prefix: Prefix for agent node names

        Returns:
            Self for method chaining
        """
        if not agents:
            logger.warning('No agents provided for intelligent routing')
            return self
        for agent_name, agent in agents.items():
            node_name = f'{prefix}{agent_name}'
            agent_node = create_agent_node_v3(agent_name=agent_name, agent=agent, name=node_name)
            self.add_node(node_name, agent_node)
        if execution_mode == 'infer':
            self._add_inferred_routing(agents, prefix)
        elif execution_mode == 'sequential':
            self._add_sequential_routing(agents, prefix)
        elif execution_mode == 'parallel':
            self._add_parallel_routing(agents, prefix)
        elif execution_mode == 'branch':
            self._add_branch_routing(agents, branches or {}, prefix)
        elif execution_mode == 'conditional':
            self._add_conditional_routing(agents, prefix)
        self._mark_needs_recompile(f'Added intelligent agent routing with {len(agents)} agents')
        return self

    def _add_inferred_routing(self, agents: dict[str, Any], prefix: str):
        """Add routing with inferred sequence."""
        sequence = self._infer_agent_sequence(agents)
        logger.info(f'Inferred agent sequence: {sequence}')
        self._add_sequential_routing_with_sequence(sequence, prefix)

    def _add_sequential_routing(self, agents: dict[str, Any], prefix: str):
        """Add sequential routing in dict order."""
        agent_names = list(agents.keys())
        self._add_sequential_routing_with_sequence(agent_names, prefix)

    def _add_sequential_routing_with_sequence(self, sequence: list[str], prefix: str):
        """Add sequential routing with specific sequence."""
        if not sequence:
            return
        self.add_edge(START, f'{prefix}{sequence[0]}')
        for i in range(len(sequence) - 1):
            current = f'{prefix}{sequence[i]}'
            next_agent = f'{prefix}{sequence[i + 1]}'
            self.add_edge(current, next_agent)
        self.add_edge(f'{prefix}{sequence[-1]}', END)

    def _add_parallel_routing(self, agents: dict[str, Any], prefix: str):
        """Add parallel routing."""
        agent_names = list(agents.keys())
        for agent_name in agent_names:
            node_name = f'{prefix}{agent_name}'
            self.add_edge(START, node_name)
            self.add_edge(node_name, END)

    def _add_branch_routing(self, agents: dict[str, Any], branches: dict[str, dict[str, Any]], prefix: str):
        """Add branch routing with conditions."""
        agent_names = list(agents.keys())
        if not agent_names:
            return
        self.add_edge(START, f'{prefix}{agent_names[0]}')
        for i, agent_name in enumerate(agent_names):
            current_node = f'{prefix}{agent_name}'
            if agent_name in branches:
                branch_config = branches[agent_name]
                self._add_agent_branch(current_node, branch_config, prefix)
            elif i < len(agent_names) - 1:
                next_node = f'{prefix}{agent_names[i + 1]}'
                self.add_edge(current_node, next_node)
            else:
                self.add_edge(current_node, END)

    def _add_conditional_routing(self, agents: dict[str, Any], prefix: str):
        """Add conditional routing with decision points."""
        agent_names = list(agents.keys())
        if not agent_names:
            return
        self.add_edge(START, f'{prefix}{agent_names[0]}')
        for i in range(len(agent_names) - 1):
            current_node = f'{prefix}{agent_names[i]}'
            next_node = f'{prefix}{agent_names[i + 1]}'

            def make_condition(current=agent_names[i], next=agent_names[i + 1]) -> Any:

                def condition(_____state: dict[str, Any]):
                    return f'{prefix}{next}'
                return condition
            condition_node = f'condition_{i}'
            self.add_node(condition_node, make_condition())
            self.add_edge(current_node, condition_node)
            self.add_edge(condition_node, next_node)
        self.add_edge(f'{prefix}{agent_names[-1]}', END)

    def _add_agent_branch(self, source_node: str, branch_config: dict[str, Any], prefix: str):
        """Add branch logic for an agent."""
        targets = branch_config.get('targets', [])
        if not targets:
            self.add_edge(source_node, END)
            return

        def branch_condition(_____state: dict[str, Any]):
            condition = branch_config.get('condition', 'default')
            if condition == 'default':
                return f'{prefix}{targets[0]}' if targets else END
            return f'{prefix}{targets[0]}' if targets else END
        branch_node = f'branch_{source_node}'
        self.add_node(branch_node, branch_condition)
        self.add_edge(source_node, branch_node)
        for target in targets:
            target_node = f'{prefix}{target}'
            self.add_edge(branch_node, target_node)

    def _infer_agent_sequence(self, agents: dict[str, Any]) -> list[str]:
        """Infer optimal agent execution sequence."""
        agent_names = list(agents.keys())
        if len(agent_names) <= 1:
            return agent_names
        sequence = self._infer_from_naming_patterns(agent_names)
        if sequence:
            return sequence
        sequence = self._infer_from_agent_types(agent_names, agents)
        if sequence:
            return sequence
        sequence = self._infer_from_prompt_dependencies(agent_names, agents)
        if sequence:
            return sequence
        return agent_names

    def _infer_from_naming_patterns(self, agent_names: list[str]) -> list[str]:
        """Infer sequence from naming patterns."""
        patterns = ['planner', 'plan', 'planning', 'analyzer', 'analysis', 'analyze', 'researcher', 'research', 'search', 'executor', 'execute', 'execution', 'worker', 'validator', 'validate', 'validation', 'reviewer', 'review', 'critique', 'replanner', 'replan', 'replanning', 'formatter', 'format', 'output', 'summary', 'summarize', 'summarizer']
        agent_scores = {}
        for agent_name in agent_names:
            score = len(patterns)
            for i, pattern in enumerate(patterns):
                if pattern in agent_name.lower():
                    score = i
                    break
            agent_scores[agent_name] = score
        sorted_agents = sorted(agent_names, key=lambda x: agent_scores[x])
        if len(set(agent_scores.values())) > 1:
            return sorted_agents
        return []

    def _infer_from_agent_types(self, agent_names: list[str], agents: dict[str, Any]) -> list[str]:
        """Infer sequence from agent types."""
        type_priority = {'ReactAgent': 1, 'SimpleAgent': 2, 'RAGAgent': 3, 'ToolAgent': 4}
        agent_scores = {}
        for agent_name in agent_names:
            agent = agents[agent_name]
            agent_type = type(agent).__name__
            agent_scores[agent_name] = type_priority.get(agent_type, 5)
        sorted_agents = sorted(agent_names, key=lambda x: agent_scores[x])
        if len(set(agent_scores.values())) > 1:
            return sorted_agents
        return []

    def _infer_from_prompt_dependencies(self, agent_names: list[str], agents: dict[str, Any]) -> list[str]:
        """Infer sequence from prompt dependencies."""
        dependencies = {}
        for agent_name in agent_names:
            agent = agents[agent_name]
            dependencies[agent_name] = set()
            if hasattr(agent, 'engine') and hasattr(agent.engine, 'prompt_template'):
                prompt = str(agent.engine.prompt_template)
                for other_agent in agent_names:
                    if other_agent != agent_name and any((field in prompt.lower() for field in [f'{other_agent}_result', f'{other_agent}_output', f'result_from_{other_agent}', f'output_from_{other_agent}'])):
                        dependencies[agent_name].add(other_agent)
        sequence = []
        remaining = set(agent_names)
        while remaining:
            ready = []
            for agent_name in remaining:
                if not dependencies[agent_name] & remaining:
                    ready.append(agent_name)
            if not ready:
                ready = [next(iter(remaining))]
            for agent_name in ready:
                sequence.append(agent_name)
                remaining.remove(agent_name)
        return sequence if len(sequence) > 1 else []

def has_tool_calls_fixed(state: dict[str, Any]) -> bool:
    """FIXED VERSION: Check if the last AI message has tool calls.

    This function properly checks for tool calls in various message formats
    and handles edge cases that the original function missed.

    Args:
        state: The state object containing messages

    Returns:
        bool: True if the last AI message has tool calls, False otherwise
    """
    from langchain_core.messages import AIMessage
    messages = None
    if hasattr(state, 'messages'):
        messages = state.messages
    elif isinstance(state, dict) and 'messages' in state:
        messages = state['messages']
    else:
        logger.debug('No messages found in state')
        return False
    if not messages:
        logger.debug('Messages list is empty')
        return False
    last_msg = messages[-1]
    logger.debug(f'Last message type: {type(last_msg)}')
    if not isinstance(last_msg, AIMessage):
        logger.debug(f"Last message is not AIMessage, it's {type(last_msg)}")
        return False
    if hasattr(last_msg, 'tool_calls'):
        tool_calls = getattr(last_msg, 'tool_calls', None)
        logger.debug(f'tool_calls attribute: {tool_calls}')
        if tool_calls:
            logger.debug(f'Found {len(tool_calls)} tool calls')
            return True
        logger.debug('tool_calls attribute exists but is empty/None')
    if hasattr(last_msg, 'additional_kwargs'):
        additional_kwargs = getattr(last_msg, 'additional_kwargs', {})
        if isinstance(additional_kwargs, dict) and 'tool_calls' in additional_kwargs:
            tool_calls_in_kwargs = additional_kwargs['tool_calls']
            logger.debug(f'tool_calls in additional_kwargs: {tool_calls_in_kwargs}')
            if tool_calls_in_kwargs:
                logger.debug(f'Found {len(tool_calls_in_kwargs)} tool calls in additional_kwargs')
                return True
            logger.debug('tool_calls in additional_kwargs is empty/None')
    logger.debug('No tool calls found')
    return False

def create_debug_has_tool_calls(original_func) -> Any:
    """Create a debug wrapper around a has_tool_calls function to help diagnose issues.

    Args:
        original_func: The original has_tool_calls function

    Returns:
        A wrapped function with detailed debugging
    """

    def debug_wrapper(_____state: dict[str, Any]):
        from langchain_core.messages import AIMessage
        logger.debug('Starting has_tool_calls debug')
        logger.info('[bold]DEBUGGING has_tool_calls[/bold]', style='cyan')
        debug_info = {'State type': str(type(state))}
        if hasattr(state, 'messages'):
            messages = state.messages
            debug_info['Messages location'] = 'state.messages'
            debug_info['Messages count'] = len(messages)
        elif isinstance(state, dict) and 'messages' in state:
            messages = state['messages']
            debug_info['Messages location'] = "state['messages']"
            debug_info['Messages count'] = len(messages)
        else:
            logger.error('No messages found in state!')
            return False
        if not messages:
            logger.error('Messages list is empty!')
            return False
        last_msg = messages[-1]
        debug_info['Last message type'] = str(type(last_msg))
        debug_info['Last message preview'] = str(last_msg)[:200] + '...'
        logger.table('State Info', debug_info)
        if isinstance(last_msg, AIMessage):
            logger.info('✓ Last message is AIMessage')
            message_details = {}
            if hasattr(last_msg, 'tool_calls'):
                tool_calls = getattr(last_msg, 'tool_calls', None)
                message_details['tool_calls attribute'] = str(tool_calls)
                message_details['tool_calls type'] = str(type(tool_calls))
                message_details['tool_calls bool'] = str(bool(tool_calls))
                if tool_calls:
                    logger.info(f'✓ Found {len(tool_calls)} tool calls')
                    for i, call in enumerate(tool_calls):
                        logger.info(f'  Tool call {i}: {call}')
                else:
                    logger.error('✗ tool_calls is empty/None')
            else:
                logger.error('✗ No tool_calls attribute')
            if hasattr(last_msg, 'additional_kwargs'):
                additional_kwargs = getattr(last_msg, 'additional_kwargs', {})
                message_details['additional_kwargs'] = str(additional_kwargs)
                if 'tool_calls' in additional_kwargs:
                    tool_calls_kwargs = additional_kwargs['tool_calls']
                    message_details['tool_calls in additional_kwargs'] = str(tool_calls_kwargs)
                    message_details['tool_calls_kwargs bool'] = str(bool(tool_calls_kwargs))
                else:
                    logger.error('✗ No tool_calls in additional_kwargs')
            else:
                logger.error('✗ No additional_kwargs')
            logger.table('Message Details', message_details)
        else:
            logger.error(f'✗ Last message is not AIMessage: {type(last_msg)}')
        original_result = original_func(state)
        fixed_result = has_tool_calls_fixed(state)
        results = {'Original function': original_result, 'Fixed function': fixed_result, 'Match': original_result == fixed_result}
        logger.table('Results', results)
        if original_result != fixed_result:
            logger.warning(f'⚠️  MISMATCH! Original: {original_result}, Fixed: {fixed_result}')
        else:
            logger.info(f'✓ Results match: {original_result}')
        return original_result
    return debug_wrapper

def debug_wrapper(original_func) -> None:
    """Debug wrapper function for external use."""