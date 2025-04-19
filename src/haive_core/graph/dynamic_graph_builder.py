# src/haive/core/graph/DynamicGraph.py

import json
import logging
import os
import uuid
from collections.abc import Callable
from typing import Any, Literal, Union

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field

from haive_core.config.runnable import RunnableConfigManager
from haive_core.engine.base import Engine, EngineRegistry
from haive_core.graph.node.config import NodeConfig
from haive_core.graph.node.factory import NodeFactory
from haive_core.schema.schema_composer import SchemaComposer

logger = logging.getLogger(__name__)

class ComponentRef(BaseModel):
    """Reference to an engine or component."""
    name: str
    id: str | None = None
    type: str | None = None

    model_config = {
        "arbitrary_types_allowed": True
    }

class DynamicGraphNode(BaseModel):
    """Node in a dynamic graph."""
    name: str = Field(description="Node name")
    config: NodeConfig = Field(description="Node configuration")
    function: Any | None = Field(default=None, exclude=True, description="Node function")

    model_config = {
        "arbitrary_types_allowed": True
    }

class DynamicGraphEdge(BaseModel):
    """Edge in a dynamic graph."""
    source: str = Field(description="Source node name")
    target: str = Field(description="Target node name")

class DynamicGraphBranch(BaseModel):
    """Conditional branch in a dynamic graph."""
    source: str = Field(description="Source node name")
    condition_name: str = Field(description="Name of condition in registry or reference to function")
    routes: dict[str, str] = Field(description="Mapping of condition values to target nodes")
    default_route: str | None = Field(default=None, description="Default route if no condition matches")

class DynamicGraph(BaseModel):
    """A dynamic graph builder that automatically derives state schema from components
    and supports runnable_config throughout the graph.
    
    Features:
    - Auto-derives state schema from components
    - Supports input/output schema separation
    - Handles configuration schemas
    - Compatible with langgraph.graph.StateGraph
    - Supports engine ID tracking
    - Serialization and visualization
    """

    name: str = Field(default_factory=lambda: f"graph_{uuid.uuid4().hex[:8]}")
    description: str | None = None

    # Components and schema
    components: list[Engine | str | dict[str, Any] | ComponentRef] = Field(default_factory=list)
    state_schema: type[BaseModel] | dict[str, Any] | None = None
    input_schema: type[BaseModel] | dict[str, Any] | None = None
    output_schema: type[BaseModel] | dict[str, Any] | None = None
    config_schema: type[BaseModel] | dict[str, Any] | None = None

    # Runtime configuration
    default_runnable_config: RunnableConfig | None = None

    # Override this parameter to always be False by default and document that it should never be True
    include_runnable_config: bool = Field(
        default=False,
        description="Whether to include runnable_config in state schema (should always be False)"
    )

    # Visualization
    visualize: bool = Field(default=True)
    visualization_dir: str = Field(default="graphs")

    # Node and graph structure tracking
    nodes: dict[str, DynamicGraphNode] = Field(default_factory=dict)
    edges: list[DynamicGraphEdge] = Field(default_factory=list)
    branches: list[DynamicGraphBranch] = Field(default_factory=list)

    # Applied patterns for tracking
    applied_patterns: list[str] = Field(default_factory=list)

    # Internal state (excluded from serialization)
    schema_composer: SchemaComposer | None = Field(default=None, exclude=True)
    state_model: type[BaseModel] | None = Field(default=None, exclude=True)
    input_model: type[BaseModel] | None = Field(default=None, exclude=True)
    output_model: type[BaseModel] | None = Field(default=None, exclude=True)
    config_model: type[BaseModel] | None = Field(default=None, exclude=True)
    graph: StateGraph | None = Field(default=None, exclude=True)
    graph_manager: Any | None = Field(default=None, exclude=True)
    engines: dict[str, Engine] = Field(default_factory=dict, exclude=True)
    engines_by_id: dict[str, Engine] = Field(default_factory=dict, exclude=True)
    compiled_app: Any | None = Field(default=None, exclude=True)

    model_config = {
        "arbitrary_types_allowed": True
    }

    def __init__(self, **data):
        """Initialize the dynamic graph with component-derived schema."""
        super().__init__(**data)

        # Create visualization directory if needed
        if self.visualize and self.visualization_dir:
            os.makedirs(self.visualization_dir, exist_ok=True)

        # Process components and initialize engines
        self._process_components()

        # Initialize schemas
        self._initialize_schemas()

        # Initialize graph
        self._initialize_graph()

    def apply_pattern(self, pattern_name: str | None = None, **pattern_params) -> "DynamicGraph":
        """Apply a registered workflow pattern to the graph.
        
        Args:
            pattern_name: Name of the pattern to apply, or None to use self.pattern
            **pattern_params: Pattern parameters
            
        Returns:
            Self for chaining
        """
        # Use provided pattern or default
        pattern = pattern_name or self.pattern

        if not pattern:
            logger.warning("No pattern specified to apply")
            return self

        # Merge provided params with existing ones
        params = {**self.pattern_params, **pattern_params}

        # Try to find pattern in registry
        from haive_core.graph.graph_pattern_registry import GraphPatternRegistry
        registry = GraphPatternRegistry.get_instance()

        pattern_obj = registry.get_pattern(pattern)
        if pattern_obj:
            # Apply the pattern using registry
            pattern_obj.apply(self, **params)
            logger.info(f"Applied registered pattern '{pattern}' to graph")
        else:
            # Try to apply a built-in pattern
            method_name = f"_apply_{pattern}_pattern"
            if hasattr(self, method_name) and callable(getattr(self, method_name)):
                getattr(self, method_name)(**params)
                logger.info(f"Applied built-in pattern '{pattern}' to graph")
            else:
                logger.warning(f"Pattern '{pattern}' not found in registry or built-in patterns")

        return self

    def _apply_simple_pattern(self, node_name: str = "process", **kwargs) -> None:
        """Apply a simple single-node workflow pattern.
        
        This creates a graph with a single node that processes input.
        
        Args:
            node_name: Name for the processing node
            **kwargs: Additional parameters
        """
        from langgraph.graph import END

        # Check if we have components
        if not self.components:
            raise ValueError("Cannot apply simple pattern without components")

        # Use the first component as the main engine
        main_engine = self.components[0]

        # Add the main engine node if not already present
        if node_name not in self.nodes:
            self.add_node(
                name=node_name,
                config=main_engine,
                command_goto=END
            )

        # Add edge from START to the node if not already present
        # Check if edge exists
        edge_exists = False
        for edge in self.edges:
            if edge.source == "START" and edge.target == node_name:
                edge_exists = True
                break

        if not edge_exists:
            self.add_edge(START, node_name)

    def _process_components(self) -> None:
        """Process and resolve components to engines."""
        processed_components = []

        for component in self.components:
            if isinstance(component, Engine):
                processed_components.append(component)
                self.engines[component.name] = component

                # Track by ID if available
                if hasattr(component, "id") and component.id:
                    self.engines_by_id[component.id] = component

            elif isinstance(component, str):
                # Look up in registry
                engine = self._lookup_engine(component)
                if engine:
                    processed_components.append(engine)
                    self.engines[engine.name] = engine

                    # Track by ID if available
                    if hasattr(engine, "id") and engine.id:
                        self.engines_by_id[engine.id] = engine

            elif isinstance(component, dict):
                # Try to convert to ComponentRef
                try:
                    ref = ComponentRef(**component)
                    engine = self._lookup_engine(ref.name, ref.type, ref.id)
                    if engine:
                        processed_components.append(engine)
                        self.engines[engine.name] = engine

                        # Track by ID if available
                        if hasattr(engine, "id") and engine.id:
                            self.engines_by_id[engine.id] = engine
                except Exception as e:
                    logger.warning(f"Could not process component dict: {component}, error: {e}")
            elif isinstance(component, ComponentRef):
                engine = self._lookup_engine(component.name, component.type, component.id)
                if engine:
                    processed_components.append(engine)
                    self.engines[engine.name] = engine

                    # Track by ID if available
                    if hasattr(engine, "id") and engine.id:
                        self.engines_by_id[engine.id] = engine

        # Replace components with processed list
        self.components = processed_components

    def _lookup_engine(self, name: str, engine_type: str | None = None, engine_id: str | None = None) -> Engine | None:
        """Look up an engine by name, type, or ID in the registry.
        
        Args:
            name: Name of the engine to lookup
            engine_type: Optional type of the engine
            engine_id: Optional unique ID of the engine
            
        Returns:
            Engine instance if found, None otherwise
        """
        registry = EngineRegistry.get_instance()

        # First try by ID if provided
        if engine_id:
            # Check if registry supports find by ID
            if hasattr(registry, "find"):
                engine = registry.find(engine_id)
                if engine:
                    return engine

        # If engine type is specified, look up directly
        if engine_type:
            try:
                from haive_core.engine.base import EngineType
                enum_type = EngineType(engine_type.lower())
                return registry.get(enum_type, name)
            except (ValueError, AttributeError) as e:
                logger.warning(f"Invalid engine type '{engine_type}': {e}")

        # Otherwise try all engine types
        for enum_type in registry.engines:
            engine = registry.get(enum_type, name)
            if engine:
                return engine

        logger.warning(f"Engine not found: {name}")
        return None

    def _initialize_schemas(self) -> None:
        """Initialize state, input, output and config schemas."""
        # Create schema composer if needed
        if self.schema_composer is None:
            self.schema_composer = SchemaComposer(name=f"{self.name.replace('-', '_').title()}State")

            # Add component schemas
            self.schema_composer.compose_from_components(self.components)

            # NEVER add runnable_config field
            # We explicitly override any include_runnable_config setting

        # Process state schema
        if self.state_schema is None:
            # Always pass include_runnable_config=False when deriving from components
            self.state_model = self.schema_composer.build()
        elif isinstance(self.state_schema, type) and issubclass(self.state_schema, BaseModel):
            # Use provided class directly
            self.state_model = self.state_schema
        else:
            # Create from dict or other format - always pass include_runnable_config=False
            self.state_model = SchemaComposer.compose(
                [self.state_schema],
                name=f"{self.name}State",
                include_runnable_config=False
            )

        # Process input schema - similarly ensure include_runnable_config=False
        if self.input_schema is None:
            # Default to state schema
            self.input_model = self.state_model
        elif isinstance(self.input_schema, type) and issubclass(self.input_schema, BaseModel):
            # Use provided class directly
            self.input_model = self.input_schema
        else:
            # Create from dict or other format - always pass include_runnable_config=False
            self.input_model = SchemaComposer.compose(
                [self.input_schema],
                name=f"{self.name}Input",
                include_runnable_config=False
            )

        # Process output schema - similarly ensure include_runnable_config=False
        if self.output_schema is None:
            # Default to state schema
            self.output_model = self.state_model
        elif isinstance(self.output_schema, type) and issubclass(self.output_schema, BaseModel):
            # Use provided class directly
            self.output_model = self.output_schema
        else:
            # Create from dict or other format - always pass include_runnable_config=False
            self.output_model = SchemaComposer.compose(
                [self.output_schema],
                name=f"{self.name}Output",
                include_runnable_config=False
            )

        # Process config schema
        if self.config_schema is not None:
            if isinstance(self.config_schema, type) and issubclass(self.config_schema, BaseModel):
                # Use provided class directly
                self.config_model = self.config_schema
            else:
                # Create from dict or other format - always pass include_runnable_config=False
                self.config_model = SchemaComposer.compose(
                    [self.config_schema],
                    name=f"{self.name}Config",
                    include_runnable_config=False
                )

    def _initialize_graph(self) -> None:
        """Initialize the StateGraph instance."""
        # Import langgraph components
        from langgraph.graph import StateGraph

        # Create state graph with the derived schema
        try:
            if not self.state_model:
                self._initialize_schemas()

            # Create with or without schema depending on what we have
            if self.state_model:
                self.graph = StateGraph(self.state_model)
            else:
                # Create without schema - langgraph will use dict
                self.graph = StateGraph()

            logger.info(f"Created StateGraph for {self.name}")
        except Exception as e:
            logger.error(f"Error creating StateGraph: {e}")
            # Fall back to dict-based graph without schema
            self.graph = StateGraph()
            logger.info("Created fallback StateGraph without schema")

    def add_node(
        self,
        name: str,
        config: NodeConfig | Engine | Callable | str,
        command_goto: str | Literal["END"] | Send | list[Send | str] | None = None,
        input_mapping: dict[str, str] | None = None,
        output_mapping: dict[str, str] | None = None,
        config_overrides: dict[str, Any] | None = None,
        runnable_config: RunnableConfig | None = None,
        metadata: dict[str, Any] | None = None
    ) -> "DynamicGraph":
        """Add a node to the graph.
        
        Args:
            name: Name of the node
            config: NodeConfig, Engine, engine name, or callable function
            command_goto: Optional next node to go to
            input_mapping: Optional mapping from state keys to engine input keys
            output_mapping: Optional mapping from engine output keys to state keys
            config_overrides: Optional engine configuration overrides specific to this node
            runnable_config: Optional default runtime configuration
            metadata: Optional metadata for the node
            
        Returns:
            Self for chaining
        """
        # Convert to NodeConfig if not already
        if not isinstance(config, NodeConfig):
            # If it's a string, try to resolve engine
            if isinstance(config, str):
                resolved_config = self._lookup_engine(config)
                if resolved_config is None:
                    logger.warning(f"Engine '{config}' not found, using as-is")
                    resolved_config = config
            else:
                resolved_config = config

            # Create NodeConfig
            node_config = NodeConfig(
                name=name,
                engine=resolved_config,
                command_goto=command_goto,
                input_mapping=input_mapping,
                output_mapping=output_mapping,
                config_overrides=config_overrides or {},
                runnable_config=runnable_config or self.default_runnable_config,
                metadata=metadata or {}
            )
        else:
            node_config = config

        # Resolve engine and engine ID
        engine, engine_id = node_config.resolve_engine()

        # Register engine if it's an Engine
        if isinstance(engine, Engine):
            self.engines[name] = engine

            # Track by ID if available
            if hasattr(engine, "id") and engine.id:
                self.engines_by_id[engine.id] = engine

        # Create node function
        node_function = NodeFactory.create_node_function(node_config)

        # Store in our node structure
        dynamic_node = DynamicGraphNode(
            name=name,
            config=node_config,
            function=node_function
        )
        self.nodes[name] = dynamic_node

        # Add to graph
        if self.graph:
            self.graph.add_node(name, node_function)
            logger.info(f"Added node '{name}' to graph")
        else:
            logger.warning(f"Graph not initialized, cannot add node '{name}'")

        return self

    def add_subgraph(
        self,
        name: str,
        subgraph: Union["DynamicGraph", Engine, Any],
        input_mapping: dict[str, str] | None = None,
        output_mapping: dict[str, str] | None = None,
        runnable_config: RunnableConfig | None = None
    ) -> "DynamicGraph":
        """Add a subgraph node to the graph.
        
        Args:
            name: Name for the subgraph node
            subgraph: Subgraph to add (DynamicGraph, compiled graph, or agent engine)
            input_mapping: Optional mapping from parent state to subgraph input
            output_mapping: Optional mapping from subgraph output to parent state
            runnable_config: Optional runtime configuration
            
        Returns:
            Self for chaining
        """
        # Handle different subgraph types
        if isinstance(subgraph, DynamicGraph):
            # Build the subgraph
            compiled_subgraph = subgraph.build()

            # Create metadata about the subgraph
            metadata = {
                "type": "subgraph",
                "original_name": subgraph.name,
                "num_nodes": len(subgraph.nodes),
                "components": [
                    getattr(comp, "name", str(comp)) for comp in subgraph.components
                ]
            }

            # Add to graph
            if self.graph:
                self.graph.add_node(name, compiled_subgraph)
                logger.info(f"Added subgraph '{name}' to graph")

                # Create NodeConfig for tracking
                node_config = NodeConfig(
                    name=name,
                    engine=None,  # No direct engine for subgraphs
                    input_mapping=input_mapping,
                    output_mapping=output_mapping,
                    runnable_config=runnable_config,
                    metadata=metadata
                )

                # Store in our node structure
                dynamic_node = DynamicGraphNode(
                    name=name,
                    config=node_config,
                    function=compiled_subgraph
                )
                self.nodes[name] = dynamic_node

        elif hasattr(subgraph, "app") or hasattr(subgraph, "create_runnable"):
            # It's an Agent or Engine that can create a runnable
            if hasattr(subgraph, "app"):
                # Use agent.app directly
                app = subgraph.app

                # Extract metadata
                metadata = {
                    "type": "agent",
                    "agent_name": getattr(subgraph, "name", name),
                    "agent_type": getattr(subgraph, "agent_type", "unknown")
                }

            else:
                # Use create_runnable
                app = subgraph.create_runnable(runnable_config)

                # Extract metadata
                metadata = {
                    "type": "engine",
                    "engine_name": getattr(subgraph, "name", name),
                    "engine_type": getattr(subgraph, "engine_type", "unknown")
                }

                # Track engine
                if isinstance(subgraph, Engine):
                    self.engines[name] = subgraph
                    if hasattr(subgraph, "id") and subgraph.id:
                        self.engines_by_id[subgraph.id] = subgraph

            # Add to graph
            if self.graph:
                self.graph.add_node(name, app)
                logger.info(f"Added agent/engine subgraph '{name}' to graph")

                # Create NodeConfig for tracking
                node_config = NodeConfig(
                    name=name,
                    engine=subgraph if isinstance(subgraph, Engine) else None,
                    input_mapping=input_mapping,
                    output_mapping=output_mapping,
                    runnable_config=runnable_config,
                    metadata=metadata
                )

                # Store in our node structure
                dynamic_node = DynamicGraphNode(
                    name=name,
                    config=node_config,
                    function=app
                )
                self.nodes[name] = dynamic_node

        # Assume it's already a compiled graph-like object
        elif self.graph:
            self.graph.add_node(name, subgraph)
            logger.info(f"Added custom subgraph '{name}' to graph")

            # Create NodeConfig for tracking
            node_config = NodeConfig(
                name=name,
                engine=None,
                input_mapping=input_mapping,
                output_mapping=output_mapping,
                runnable_config=runnable_config,
                metadata={"type": "custom"}
            )

            # Store in our node structure
            dynamic_node = DynamicGraphNode(
                name=name,
                config=node_config,
                function=subgraph
            )
            self.nodes[name] = dynamic_node

        return self

    def add_edge(self, from_node: str | list[str], to_node: str | Literal["END"]) -> "DynamicGraph":
        """Add an edge between nodes.
        
        Args:
            from_node: Source node name or list of source node names
            to_node: Target node name or END
            
        Returns:
            Self for chaining
        """
        if not self.graph:
            logger.warning("Graph not initialized, cannot add edge")
            return self

        # Handle array of source nodes
        if isinstance(from_node, list):
            for source in from_node:
                self.graph.add_edge(source, to_node)
                logger.info(f"Added edge from '{source}' to '{to_node}'")

                # Track the edge
                edge = DynamicGraphEdge(
                    source=source,
                    target="END" if to_node is END else to_node
                )
                self.edges.append(edge)

        else:
            # Single source node
            self.graph.add_edge(from_node, to_node)
            logger.info(f"Added edge from '{from_node}' to '{to_node}'")

            # Track the edge
            edge = DynamicGraphEdge(
                source=from_node,
                target="END" if to_node is END else to_node
            )
            self.edges.append(edge)

        return self

    def add_conditional_edges(
        self,
        from_node: str,
        condition: Callable,
        routes: dict[Any, str | Literal["END"]]
    ) -> "DynamicGraph":
        """Add conditional edges based on a condition function.
        
        Args:
            from_node: Source node name
            condition: Function that evaluates state and returns a key
            routes: Mapping from condition results to target nodes
            
        Returns:
            Self for chaining
        """
        if not self.graph:
            logger.warning("Graph not initialized, cannot add conditional edges")
            return self

        # Normalize routes for serialization
        normalized_routes = {}
        for key, value in routes.items():
            if value is END:
                normalized_routes[str(key)] = "END"
            else:
                normalized_routes[str(key)] = value

        # Add conditional edges
        self.graph.add_conditional_edges(from_node, condition, routes)
        logger.info(f"Added conditional edges from '{from_node}'")

        # Track the branch
        condition_name = getattr(condition, "__name__", "conditional_branch")
        branch = DynamicGraphBranch(
            source=from_node,
            condition_name=condition_name,
            routes=normalized_routes
        )
        self.branches.append(branch)

        return self

    def set_entry_point(self, node_name: str) -> "DynamicGraph":
        """Set the entry point for the graph.
        
        Args:
            node_name: Name of the entry point node
            
        Returns:
            Self for chaining
        """
        if not self.graph:
            logger.warning("Graph not initialized, cannot set entry point")
            return self

        # Check if node exists
        if node_name not in self.nodes:
            logger.warning(f"Node '{node_name}' not found, cannot set as entry point")
            return self

        try:
            # Set entry point in StateGraph
            self.graph.set_entry_point(node_name)
            logger.info(f"Set entry point to '{node_name}' in StateGraph")

            # Also update our edge list to stay in sync
            from langgraph.graph import START

            # First check if we have any START edges already
            has_start_edge = False
            for i, edge in enumerate(self.edges):
                # Check for DynamicGraphEdge or tuple format
                if hasattr(edge, "source") and (edge.source == "START" or edge.source == "__start__"):
                    if hasattr(edge, "target") and edge.target != node_name:
                        # Update target
                        edge.target = node_name
                        has_start_edge = True
                        logger.debug(f"Updated existing START edge to point to {node_name}")
                    elif hasattr(edge, "target") and edge.target == node_name:
                        # Already points to the right node
                        has_start_edge = True
                        logger.debug(f"START edge already points to {node_name}")
                elif isinstance(edge, tuple) and len(edge) >= 2:
                    if edge[0] == START or edge[0] == "__start__":
                        # Replace tuple with proper DynamicGraphEdge
                        self.edges[i] = DynamicGraphEdge(source="START", target=node_name)
                        has_start_edge = True
                        logger.debug(f"Replaced tuple START edge with DynamicGraphEdge to {node_name}")

            # Add a new START edge if none exists
            if not has_start_edge:
                new_edge = DynamicGraphEdge(source="START", target=node_name)
                self.edges.append(new_edge)
                logger.debug(f"Added new START -> {node_name} edge to edge list")

            # Check validation after setting entry point
            if self.validate_structure():
                logger.debug("Graph validation passed after setting entry point")
            else:
                logger.warning("Graph validation still failing after setting entry point")

        except Exception as e:
            logger.error(f"Error setting entry point: {e}")

        return self

    def with_runnable_config(self, runnable_config: RunnableConfig) -> "DynamicGraph":
        """Create a new graph with the specified runnable_config as default.
        
        Args:
            runnable_config: The configuration to use as default
            
        Returns:
            A new DynamicGraph instance with the specified config
        """
        # Create a new graph with the same schema and components
        new_graph = DynamicGraph(
            name=self.name,
            description=self.description,
            components=self.components,
            state_schema=self.state_model,
            input_schema=self.input_model,
            output_schema=self.output_model,
            config_schema=self.config_model,
            default_runnable_config=runnable_config,
            visualize=self.visualize
        )

        return new_graph

    def set_default_runnable_config(self, runnable_config: RunnableConfig) -> "DynamicGraph":
        """Set the default runnable_config for this graph.
        
        Args:
            runnable_config: The configuration to use as default
            
        Returns:
            Self for chaining
        """
        self.default_runnable_config = runnable_config
        return self

    def update_default_runnable_config(self, **kwargs) -> "DynamicGraph":
        """Update the default runnable_config with additional parameters.
        
        Args:
            **kwargs: Key-value pairs to add to the config
            
        Returns:
            Self for chaining
        """
        if self.default_runnable_config is None:
            self.default_runnable_config = RunnableConfigManager.create(**kwargs)
        else:
            self.default_runnable_config = RunnableConfigManager.merge(
                self.default_runnable_config,
                RunnableConfigManager.create(**kwargs)
            )
        return self

    def get_manager(self):
        """Get a StateGraphManager for this graph.
        
        Returns:
            StateGraphManager instance
        """
        # Import here to avoid circular imports
        from haive_core.graph.state_graph_manager import StateGraphManager

        if self.graph is None:
            self._initialize_graph()

        if self.graph_manager is None:
            self.graph_manager = StateGraphManager(self.graph)

        return self.graph_manager

    def insert_node(
        self,
        node_name: str,
        between: tuple[str, str],
        node_config: NodeConfig | Engine | Callable | str,
        input_mapping: dict[str, str] | None = None,
        output_mapping: dict[str, str] | None = None,
        config_overrides: dict[str, Any] | None = None,
        runnable_config: RunnableConfig | None = None
    ) -> "DynamicGraph":
        """Insert a node between two existing nodes.
        
        Args:
            node_name: Name for the new node
            between: Tuple (source_node, target_node) to insert between
            node_config: Node configuration, engine, or callable
            input_mapping: Optional mapping from state keys to engine input keys
            output_mapping: Optional mapping from engine output keys to state keys
            config_overrides: Optional engine configuration overrides
            runnable_config: Optional runtime configuration
            
        Returns:
            Self for chaining
        """
        manager = self.get_manager()

        # If node_config is not a NodeConfig, create one
        if not isinstance(node_config, NodeConfig):
            # If it's a string, try to resolve engine
            if isinstance(node_config, str):
                resolved_config = self._lookup_engine(node_config)
                if resolved_config is None:
                    logger.warning(f"Engine '{node_config}' not found, using as-is")
                    resolved_config = node_config
            else:
                resolved_config = node_config

            # Create node config
            final_config = NodeConfig(
                name=node_name,
                engine=resolved_config,
                input_mapping=input_mapping,
                output_mapping=output_mapping,
                config_overrides=config_overrides or {},
                runnable_config=runnable_config or self.default_runnable_config
            )
        else:
            final_config = node_config

        # Create node function
        node_function = NodeFactory.create_node_function(final_config)

        # Resolve engine and engine ID
        engine, engine_id = final_config.resolve_engine()

        # Register engine if applicable
        if isinstance(engine, Engine):
            self.engines[node_name] = engine

            # Track by ID if available
            if hasattr(engine, "id") and engine.id:
                self.engines_by_id[engine.id] = engine

        # Use the manager to insert the node
        manager.insert_node(node_name, between, node_function)

        # Track in our node structure
        dynamic_node = DynamicGraphNode(
            name=node_name,
            config=final_config,
            function=node_function
        )
        self.nodes[node_name] = dynamic_node

        # Update edges
        source, target = between

        # Remove existing edge
        self.edges = [e for e in self.edges if not (e.source == source and e.target == target)]

        # Add new edges
        self.edges.append(DynamicGraphEdge(source=source, target=node_name))
        self.edges.append(DynamicGraphEdge(source=node_name, target=target))

        return self

    def apply_pattern(self, pattern_name: str, **kwargs) -> "DynamicGraph":
        """Apply a registered pattern to the graph.
        
        Args:
            pattern_name: Name of the pattern in registry
            **kwargs: Pattern parameters
            
        Returns:
            Self for chaining
        """
        # Import registry avoiding circular imports
        try:
            from haive_core.graph.graph_pattern_registry import GraphPatternRegistry
            registry = GraphPatternRegistry.get_instance()

            # Get the pattern
            pattern = registry.get_pattern(pattern_name)
            if not pattern:
                logger.warning(f"Pattern '{pattern_name}' not found in registry")
                return self

            # Apply the pattern (implementation will depend on pattern type)
            # This is a placeholder - actual implementation will vary by pattern
            logger.info(f"Applying pattern '{pattern_name}' to graph")

            # Record that this pattern was applied
            self.applied_patterns.append(pattern_name)

        except ImportError:
            logger.warning("GraphPatternRegistry not available, cannot apply pattern")

        return self

    def build(self, checkpointer=None, **kwargs) -> Any:
        """Build and return the graph.
        
        Args:
            checkpointer: Optional checkpoint saver
            **kwargs: Additional arguments for future extensions
            
        Returns:
            Built graph (not compiled)
        """
        # Initialize graph if needed
        if self.graph is None:
            self._initialize_graph()

        return self.graph

    def validate_structure(self) -> bool:
        """Validate the graph structure before compilation.
        Checks for critical issues that would prevent compilation like missing START edges.
        
        Returns:
            True if structure is valid, False otherwise
        """
        # Empty graphs are trivially valid - we'll handle them during compilation
        if not self.nodes:
            logger.debug("Empty graph with no nodes found")
            return True

        # Find any entry points - check both our edges list and the StateGraph
        has_edge_entry = any(
            (hasattr(edge, "source") and (edge.source == "START" or edge.source == "__start__"))
            or (isinstance(edge, tuple) and len(edge) >= 2 and (edge[0] == START or edge[0] == "__start__"))
            for edge in self.edges
        )

        # Check StateGraph entry point
        has_graph_entry = False
        if self.graph and hasattr(self.graph, "_entry_point"):
            entry = getattr(self.graph, "_entry_point", None)
            has_graph_entry = entry is not None

        # Log findings
        if has_edge_entry:
            logger.debug("START edge found in edge list")
        if has_graph_entry:
            logger.debug("Entry point found in StateGraph")

        # Graph is valid if it has either type of entry point
        has_entry = has_edge_entry or has_graph_entry

        if self.nodes and not has_entry:
            # We have nodes but no entry point
            node_names = ", ".join(list(self.nodes.keys()))
            logger.warning(
                f"Graph has {len(self.nodes)} nodes but no entry point - compilation will fail. "
                f"Available nodes: {node_names}"
            )
            return False

        return True

    def set_auto_entry_point(self) -> bool:
        """Automatically set an entry point if one doesn't exist.
        Uses the first node in the graph as the entry point.
        
        Returns:
            True if an entry point was set, False if no suitable node was found
        """
        if not self.nodes:
            logger.debug("No nodes to set as entry point")
            return False

        # Check if we already have a START edge
        if any(edge.source == "START" or edge.source == "__start__"
              for edge in self.edges if hasattr(edge, "source")):
            logger.debug("Entry point already exists")
            return True

        # Use the first node as entry point
        first_node = next(iter(self.nodes.keys()))
        logger.info(f"Automatically setting {first_node} as graph entry point")

        # Set entry point using the proper method
        try:
            self.set_entry_point(first_node)
            return True
        except Exception as e:
            logger.error(f"Failed to set auto entry point: {e}")
            return False

    def debug_graph(self) -> None:
        """Print detailed debug information about the graph structure.
        Includes nodes, edges, entry points, and connectivity.
        """
        logger.info(f"--- Graph Debug: {self.name} ---")

        # Nodes
        if not self.nodes:
            logger.info("No nodes in graph")
        else:
            node_names = list(self.nodes.keys())
            logger.info(f"Nodes ({len(node_names)}): {', '.join(node_names)}")

        # Edges
        if not self.edges:
            logger.info("No edges in graph")
        else:
            edge_strings = []
            for edge in self.edges:
                if hasattr(edge, "source") and hasattr(edge, "target"):
                    edge_strings.append(f"{edge.source} -> {edge.target}")
                elif isinstance(edge, tuple) and len(edge) >= 2:
                    # Handle tuple representation
                    edge_strings.append(f"{edge[0]} -> {edge[1]}")
                else:
                    edge_strings.append(f"(Unknown edge format: {type(edge)})")

            if edge_strings:
                logger.info(f"Edges ({len(self.edges)}): {', '.join(edge_strings)}")
            else:
                logger.info(f"Edges exist ({len(self.edges)}) but couldn't be formatted")

        # Check for entry point in edges
        entry_points = []
        for edge in self.edges:
            if hasattr(edge, "source") and (edge.source == "START" or edge.source == "__start__"):
                if hasattr(edge, "target"):
                    entry_points.append(edge.target)
            elif isinstance(edge, tuple) and len(edge) >= 2:
                if edge[0] == START or edge[0] == "__start__":
                    entry_points.append(edge[1])

        if entry_points:
            logger.info(f"Entry points from edges: {', '.join(entry_points)}")
        else:
            logger.info("No entry points found in edges")

        # Check for explicit entry point in StateGraph if available
        if hasattr(self.graph, "_entry_point"):
            entry = getattr(self.graph, "_entry_point", None)
            if entry:
                logger.info(f"StateGraph entry point: {entry}")
            else:
                logger.info("StateGraph has no explicit entry point")

        # END points
        end_targets = []
        for edge in self.edges:
            if hasattr(edge, "target") and (edge.target == "END" or edge.target == "__end__"):
                if hasattr(edge, "source"):
                    end_targets.append(edge.source)
            elif isinstance(edge, tuple) and len(edge) >= 2:
                if edge[1] == END or edge[1] == "__end__":
                    end_targets.append(edge[0])

        if end_targets:
            logger.info(f"END edges from: {', '.join(end_targets)}")
        else:
            logger.info("No explicit END edges found")

        # Validation status
        if self.validate_structure():
            logger.info("✅ Graph structure validation: PASSED")
        else:
            logger.warning("❌ Graph structure validation: FAILED")

        logger.info("--- End Graph Debug ---")

    def ensure_start_edge(self) -> bool:
        """Ensure the graph can be properly compiled.
        If no START edge exists, tries to set an auto entry point.
        
        Returns:
            True if graph is valid for compilation, False otherwise
        """
        # Check if structure is already valid
        if self.validate_structure():
            return True

        # Try to set an auto entry point
        return self.set_auto_entry_point()

    def compile(self, checkpointer=None, **kwargs) -> Any:
        """Compile the graph to a runnable application.
        
        Args:
            checkpointer: Optional checkpointer for persistence
            **kwargs: Additional arguments for compilation
            
        Returns:
            Compiled application
        """
        # Initialize graph if needed
        if self.graph is None:
            self._initialize_graph()

        # Debug the graph structure before compilation
        self.debug_graph()

        # Validate the graph or try to automatically fix it
        if not self.ensure_start_edge():
            if not self.nodes:
                # Special handling for empty graphs - add a simple identity processor
                logger.info("Compiling empty graph as pass-through processor")
                def identity_processor(state):
                    return state

                # Add a simple processor that passes through state
                self.add_node("identity", identity_processor)
                self.add_edge(START, "identity")
                logger.info("Added identity processor with START edge for empty graph")
            else:
                # Graph has nodes but no valid START edge - provide detailed error
                available_nodes = ", ".join(self.nodes.keys())
                raise ValueError(
                    f"Graph validation failed: must have at least one edge from START. "
                    f"Available nodes: [{available_nodes}]. "
                    f"Add a START edge using graph.add_edge(START, '<node_name>') "
                    f"or set an entry point using graph.set_entry_point('<node_name>')."
                )

        # Compile the graph
        try:
            # Build the graph
            graph = self.build(checkpointer)

            # Compile with checkpointer and other args
            self.compiled_app = graph.compile(checkpointer=checkpointer,)

            # Visualize if enabled
            if self.visualize:
                logger.info(f"Generating visualization for {self.name}")
                self.visualize_graph()

            return self.compiled_app
        except Exception as e:
            logger.error(f"Error compiling graph: {e}")
            raise

    def _sanitize_secret_values(self, data):
        """Sanitize sensitive values to ensure they can be serialized.
        
        Args:
            data: Data to sanitize
            
        Returns:
            Sanitized data
        """
        from pydantic import SecretStr

        if data is None:
            return None

        # Handle SecretStr objects
        if isinstance(data, SecretStr):
            return "**********"  # Return redacted string instead of actual value

        # Handle dictionaries recursively
        if isinstance(data, dict):
            return {k: self._sanitize_secret_values(v) for k, v in data.items()}

        # Handle lists recursively
        if isinstance(data, list):
            return [self._sanitize_secret_values(item) for item in data]

        # Handle tuples recursively
        if isinstance(data, tuple):
            return tuple(self._sanitize_secret_values(item) for item in data)

        # Return primitive types as is
        return data

    def to_dict(self) -> dict[str, Any]:
        """Convert to a serializable dictionary.
        
        Returns:
            Dictionary representation
        """
        # Use model_dump for serialization
        if hasattr(self, "model_dump"):
            # Exclude fields that would cause serialization issues
            exclude = {"schema_composer", "state_model", "input_model", "output_model",
                      "config_model", "graph", "graph_manager", "engines", "engines_by_id",
                      "compiled_app"}
            result = self.model_dump(exclude=exclude)
        else:
            # Pydantic v1 fallback
            exclude = {"schema_composer", "state_model", "input_model", "output_model",
                      "config_model", "graph", "graph_manager", "engines", "engines_by_id",
                      "compiled_app"}
            result = {k: v for k, v in self.dict(exclude=exclude).items()}

        # Convert components to serializable form
        result["components"] = []
        for component in self.components:
            if isinstance(component, Engine):
                result["components"].append({
                    "name": component.name,
                    "id": getattr(component, "id", None),
                    "type": str(component.engine_type.value) if hasattr(component, "engine_type") else None
                })
            elif isinstance(component, str):
                result["components"].append(component)
            elif isinstance(component, ComponentRef):
                if hasattr(component, "model_dump"):
                    result["components"].append(component.model_dump())
                else:
                    result["components"].append(component.dict())

        # Convert schemas if present
        if self.state_model:
            result["state_schema"] = self.state_model.__name__
        if self.input_model and self.input_model != self.state_model:
            result["input_schema"] = self.input_model.__name__
        if self.output_model and self.output_model != self.state_model:
            result["output_schema"] = self.output_model.__name__
        if self.config_model:
            result["config_schema"] = self.config_model.__name__

        # Convert default runnable config if present and sanitize sensitive values
        if self.default_runnable_config:
            result["default_runnable_config"] = self._sanitize_secret_values(self.default_runnable_config)

        # Sanitize any remaining sensitive values
        result = self._sanitize_secret_values(result)

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DynamicGraph":
        """Create from a serialized dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            New DynamicGraph instance
        """
        # Process components
        components = []
        for component in data.get("components", []):
            if isinstance(component, dict):
                components.append(ComponentRef(**component))
            else:
                components.append(component)

        # Create a clean dict without schema fields that might be strings
        clean_data = {
            "name": data.get("name", f"graph_{uuid.uuid4().hex[:8]}"),
            "description": data.get("description"),
            "components": components,
            "default_runnable_config": data.get("default_runnable_config"),
            "visualize": data.get("visualize", True),
            "applied_patterns": data.get("applied_patterns", [])
        }

        # For schema fields, only include if they're not strings
        # (string schema refs are just class names that we can't resolve here)
        if "state_schema" in data and not isinstance(data["state_schema"], str):
            clean_data["state_schema"] = data["state_schema"]
        if "input_schema" in data and not isinstance(data["input_schema"], str):
            clean_data["input_schema"] = data["input_schema"]
        if "output_schema" in data and not isinstance(data["output_schema"], str):
            clean_data["output_schema"] = data["output_schema"]
        if "config_schema" in data and not isinstance(data["config_schema"], str):
            clean_data["config_schema"] = data["config_schema"]

        # Create the graph
        graph = cls(**clean_data)

        # Process nodes
        for name, node_data in data.get("nodes", {}).items():
            # Create NodeConfig
            node_config = NodeConfig.from_dict(node_data.get("config", {}))

            # Add node to graph
            graph.add_node(name, node_config)

        # Process edges
        for edge_data in data.get("edges", []):
            source = edge_data.get("source")
            target = edge_data.get("target")
            if source and target:
                # Convert "END" to END constant
                if target == "END":
                    target = END
                graph.add_edge(source, target)

        # Process branches
        for branch_data in data.get("branches", []):
            # Note: This is a simplified implementation - actual branch loading
            # would require resolving the condition function
            source = branch_data.get("source")
            condition_name = branch_data.get("condition_name")
            routes = branch_data.get("routes", {})

            if source and condition_name and routes:
                # Create a placeholder condition function
                def condition(state):
                    # This is a placeholder - actual implementation would be more complex
                    return next(iter(routes.keys()))

                # Normalize routes for END
                normalized_routes = {}
                for key, value in routes.items():
                    if value == "END":
                        normalized_routes[key] = END
                    else:
                        normalized_routes[key] = value

                # Add to graph
                graph.add_conditional_edges(source, condition, normalized_routes)

        return graph

    def save(self, filename: str) -> None:
        """Save the graph configuration to a file.
        
        Args:
            filename: Path to save the configuration
        """
        # Convert to dictionary
        data = self.to_dict()

        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)

        # Save to file
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Graph configuration saved to {filename}")

    @classmethod
    def load(cls, filename: str) -> "DynamicGraph":
        """Load a graph configuration from a file.
        
        Args:
            filename: Path to load the configuration from
            
        Returns:
            Loaded DynamicGraph instance
        """
        # Load from file
        with open(filename) as f:
            data = json.load(f)

        # Create from dictionary
        graph = cls.from_dict(data)

        logger.info(f"Graph configuration loaded from {filename}")
        return graph

    def visualize_graph(self, filename: str | None = None) -> None:
        """Visualize the graph structure using the haive_core utility function.
        
        Args:
            filename: Optional filename to save the visualization
        """
        if not self.visualize:
            return

        # Default filename if not provided
        if filename is None:
            filename = os.path.join(self.visualization_dir, f"{self.name}.png")

        # Initialize graph if needed
        if self.graph is None:
            self._initialize_graph()

        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)

        try:
            # Use the haive_core visualization utility
            from haive_core.utils.visualize_graph_utils import render_and_display_graph

            # Make sure we have a built graph
            built_graph = self.build()

            # Use the utility function
            render_and_display_graph(
                compiled_graph=built_graph,
                output_dir=os.path.dirname(filename),
                output_name=os.path.basename(filename)
            )
            logger.info(f"Graph visualization saved to {filename}")
        except ImportError as e:
            logger.warning(f"Cannot visualize graph - visualization utils not available: {e}")
        except Exception as e:
            logger.error(f"Error visualizing graph: {e}")

            # Try fallback method using StateGraphManager
            try:
                manager = self.get_manager()
                manager.visualize(output_file=filename)
                logger.info(f"Graph visualization saved to {filename} (fallback method)")
            except Exception as fallback_error:
                logger.error(f"Fallback visualization also failed: {fallback_error}")
