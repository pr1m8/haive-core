# src/haive/core/graph/DynamicGraph.py

from typing import Dict, List, Optional, Union, Literal, Callable, Type, Any, get_origin, get_args, cast, Tuple, Set
import logging
import json
import uuid
import os
from pydantic import BaseModel, Field, model_validator

from langgraph.graph import StateGraph, END, START
from langgraph.types import Command, Send
from langchain_core.runnables import RunnableConfig

from haive_core.engine.base import Engine, InvokableEngine, EngineRegistry, EngineType
from haive_core.graph.node.factory import NodeFactory
from haive_core.graph.node.config import NodeConfig 
from haive_core.schema.schema_composer import SchemaComposer
from haive_core.config.runnable import RunnableConfigManager

logger = logging.getLogger(__name__)

class ComponentRef(BaseModel):
    """Reference to an engine or component."""
    name: str
    id: Optional[str] = None
    type: Optional[str] = None
    
    model_config = {
        "arbitrary_types_allowed": True
    }

class DynamicGraphNode(BaseModel):
    """Node in a dynamic graph."""
    name: str = Field(description="Node name")
    config: NodeConfig = Field(description="Node configuration")
    function: Optional[Any] = Field(default=None, exclude=True, description="Node function")
    
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
    routes: Dict[str, str] = Field(description="Mapping of condition values to target nodes")
    default_route: Optional[str] = Field(default=None, description="Default route if no condition matches")

class DynamicGraph(BaseModel):
    """
    A dynamic graph builder that automatically derives state schema from components
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
    description: Optional[str] = None
    
    # Components and schema
    components: List[Union[Engine, str, Dict[str, Any], ComponentRef]] = Field(default_factory=list)
    state_schema: Optional[Union[Type[BaseModel], Dict[str, Any]]] = None
    input_schema: Optional[Union[Type[BaseModel], Dict[str, Any]]] = None
    output_schema: Optional[Union[Type[BaseModel], Dict[str, Any]]] = None
    config_schema: Optional[Union[Type[BaseModel], Dict[str, Any]]] = None
    
    # Runtime configuration
    default_runnable_config: Optional[RunnableConfig] = None
    
    # Visualization
    visualize: bool = Field(default=True)
    visualization_dir: str = Field(default="graphs")
    
    # Node and graph structure tracking
    nodes: Dict[str, DynamicGraphNode] = Field(default_factory=dict)
    edges: List[DynamicGraphEdge] = Field(default_factory=list)
    branches: List[DynamicGraphBranch] = Field(default_factory=list)
    
    # Applied patterns for tracking
    applied_patterns: List[str] = Field(default_factory=list)
    
    # Internal state (excluded from serialization)
    schema_composer: Optional[SchemaComposer] = Field(default=None, exclude=True)
    state_model: Optional[Type[BaseModel]] = Field(default=None, exclude=True)
    input_model: Optional[Type[BaseModel]] = Field(default=None, exclude=True)
    output_model: Optional[Type[BaseModel]] = Field(default=None, exclude=True)
    config_model: Optional[Type[BaseModel]] = Field(default=None, exclude=True)
    graph: Optional[StateGraph] = Field(default=None, exclude=True)
    graph_manager: Optional[Any] = Field(default=None, exclude=True)
    engines: Dict[str, Engine] = Field(default_factory=dict, exclude=True)
    engines_by_id: Dict[str, Engine] = Field(default_factory=dict, exclude=True)
    compiled_app: Optional[Any] = Field(default=None, exclude=True)
    
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
    
    def _process_components(self) -> None:
        """Process and resolve components to engines."""
        processed_components = []
        
        for component in self.components:
            if isinstance(component, Engine):
                processed_components.append(component)
                self.engines[component.name] = component
                
                # Track by ID if available
                if hasattr(component, "id") and getattr(component, "id"):
                    self.engines_by_id[getattr(component, "id")] = component
                    
            elif isinstance(component, str):
                # Look up in registry
                engine = self._lookup_engine(component)
                if engine:
                    processed_components.append(engine)
                    self.engines[engine.name] = engine
                    
                    # Track by ID if available
                    if hasattr(engine, "id") and getattr(engine, "id"):
                        self.engines_by_id[getattr(engine, "id")] = engine
                        
            elif isinstance(component, dict):
                # Try to convert to ComponentRef
                try:
                    ref = ComponentRef(**component)
                    engine = self._lookup_engine(ref.name, ref.type, ref.id)
                    if engine:
                        processed_components.append(engine)
                        self.engines[engine.name] = engine
                        
                        # Track by ID if available
                        if hasattr(engine, "id") and getattr(engine, "id"):
                            self.engines_by_id[getattr(engine, "id")] = engine
                except Exception as e:
                    logger.warning(f"Could not process component dict: {component}, error: {e}")
            elif isinstance(component, ComponentRef):
                engine = self._lookup_engine(component.name, component.type, component.id)
                if engine:
                    processed_components.append(engine)
                    self.engines[engine.name] = engine
                    
                    # Track by ID if available
                    if hasattr(engine, "id") and getattr(engine, "id"):
                        self.engines_by_id[getattr(engine, "id")] = engine
        
        # Replace components with processed list
        self.components = processed_components
    
    def _lookup_engine(self, name: str, engine_type: Optional[str] = None, engine_id: Optional[str] = None) -> Optional[Engine]:
        """
        Look up an engine by name, type, or ID in the registry.
        
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
            
            # Always add runnable_config field
            from typing import Dict, Any
            self.schema_composer.add_field(
                name='__runnable_config__',
                field_type=Dict[str, Any],
                default_factory=dict
            )
        
        # Process state schema
        if self.state_schema is None:
            # Derive from components
            self.state_model = self.schema_composer.build()
        elif isinstance(self.state_schema, type) and issubclass(self.state_schema, BaseModel):
            # Use provided class directly
            self.state_model = self.state_schema
        else:
            # Create from dict or other format
            self.state_model = SchemaComposer.compose(
                [self.state_schema], 
                name=f"{self.name}State"
            )
            
        # Process input schema
        if self.input_schema is None:
            # Default to state schema
            self.input_model = self.state_model
        elif isinstance(self.input_schema, type) and issubclass(self.input_schema, BaseModel):
            # Use provided class directly
            self.input_model = self.input_schema
        else:
            # Create from dict or other format
            self.input_model = SchemaComposer.compose(
                [self.input_schema],
                name=f"{self.name}Input"
            )
            
        # Process output schema
        if self.output_schema is None:
            # Default to state schema
            self.output_model = self.state_model
        elif isinstance(self.output_schema, type) and issubclass(self.output_schema, BaseModel):
            # Use provided class directly
            self.output_model = self.output_schema
        else:
            # Create from dict or other format
            self.output_model = SchemaComposer.compose(
                [self.output_schema],
                name=f"{self.name}Output"
            )
            
        # Process config schema
        if self.config_schema is not None:
            if isinstance(self.config_schema, type) and issubclass(self.config_schema, BaseModel):
                # Use provided class directly
                self.config_model = self.config_schema
            else:
                # Create from dict or other format
                self.config_model = SchemaComposer.compose(
                    [self.config_schema],
                    name=f"{self.name}Config"
                )
    
    def _initialize_graph(self) -> None:
        """Initialize the StateGraph."""
        if self.graph is None:
            if self.input_model and self.output_model and self.config_model:
                # Create with all schemas
                self.graph = StateGraph(
                    self.state_model,
                    input=self.input_model,
                    output=self.output_model,
                    config=self.config_model
                )
            elif self.input_model and self.output_model:
                # Create with input and output schemas
                self.graph = StateGraph(
                    self.state_model,
                    input=self.input_model,
                    output=self.output_model
                )
            elif self.config_model:
                # Create with config schema
                self.graph = StateGraph(
                    self.state_model,
                    config=self.config_model
                )
            else:
                # Create with just state schema
                self.graph = StateGraph(self.state_model)
    
    def add_node(
        self, 
        name: str, 
        config: Union[NodeConfig, Engine, Callable, str], 
        command_goto: Optional[Union[str, Literal["END"], Send, List[Union[Send, str]]]] = None,
        input_mapping: Optional[Dict[str, str]] = None,
        output_mapping: Optional[Dict[str, str]] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
        runnable_config: Optional[RunnableConfig] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'DynamicGraph':
        """
        Add a node to the graph.
        
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
            if hasattr(engine, "id") and getattr(engine, "id"):
                self.engines_by_id[getattr(engine, "id")] = engine
        
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
        subgraph: Union['DynamicGraph', Engine, Any],
        input_mapping: Optional[Dict[str, str]] = None,
        output_mapping: Optional[Dict[str, str]] = None,
        runnable_config: Optional[RunnableConfig] = None
    ) -> 'DynamicGraph':
        """
        Add a subgraph node to the graph.
        
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
                    if hasattr(subgraph, "id") and getattr(subgraph, "id"):
                        self.engines_by_id[getattr(subgraph, "id")] = subgraph
            
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
                
        else:
            # Assume it's already a compiled graph-like object
            if self.graph:
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
    
    def add_edge(self, from_node: Union[str, List[str]], to_node: Union[str, Literal["END"]]) -> 'DynamicGraph':
        """
        Add an edge between nodes.
        
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
        routes: Dict[Any, Union[str, Literal["END"]]]
    ) -> 'DynamicGraph':
        """
        Add conditional edges based on a condition function.
        
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
    
    def set_entry_point(self, node_name: str) -> 'DynamicGraph':
        """
        Set the entry point for the graph.
        
        Args:
            node_name: Name of the entry point node
            
        Returns:
            Self for chaining
        """
        if not self.graph:
            logger.warning("Graph not initialized, cannot set entry point")
            return self
            
        self.graph.set_entry_point(node_name)
        logger.info(f"Set entry point to '{node_name}'")
        
        return self
    
    def with_runnable_config(self, runnable_config: RunnableConfig) -> 'DynamicGraph':
        """
        Create a new graph with the specified runnable_config as default.
        
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
    
    def set_default_runnable_config(self, runnable_config: RunnableConfig) -> 'DynamicGraph':
        """
        Set the default runnable_config for this graph.
        
        Args:
            runnable_config: The configuration to use as default
            
        Returns:
            Self for chaining
        """
        self.default_runnable_config = runnable_config
        return self
    
    def update_default_runnable_config(self, **kwargs) -> 'DynamicGraph':
        """
        Update the default runnable_config with additional parameters.
        
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
        """
        Get a StateGraphManager for this graph.
        
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
        between: Tuple[str, str], 
        node_config: Union[NodeConfig, Engine, Callable, str],
        input_mapping: Optional[Dict[str, str]] = None,
        output_mapping: Optional[Dict[str, str]] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
        runnable_config: Optional[RunnableConfig] = None
    ) -> 'DynamicGraph':
        """
        Insert a node between two existing nodes.
        
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
            if hasattr(engine, "id") and getattr(engine, "id"):
                self.engines_by_id[getattr(engine, "id")] = engine
        
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
    
    def apply_pattern(self, pattern_name: str, **kwargs) -> 'DynamicGraph':
        """
        Apply a registered pattern to the graph.
        
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
        """
        Build and return the graph.
        
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
    
    def compile(self, checkpointer=None, **kwargs) -> Any:
        """
        Build and compile the graph.
        
        Args:
            checkpointer: Optional checkpoint saver
            **kwargs: Additional arguments to pass to the compile method
            
        Returns:
            Compiled graph application
        """
        # Add default_runnable_config to kwargs if provided
        if self.default_runnable_config and 'default_config' not in kwargs:
            kwargs['default_config'] = self.default_runnable_config
            
        # Build the graph
        graph = self.build(checkpointer)
        
        # Compile with checkpointer and other args
        self.compiled_app = graph.compile(checkpointer=checkpointer, **kwargs)
        
        # Visualize if enabled
        if self.visualize:
            self.visualize_graph()
        
        return self.compiled_app
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to a serializable dictionary.
        
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
            
        # Convert default runnable config if present
        if self.default_runnable_config:
            result["default_runnable_config"] = self.default_runnable_config
                
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DynamicGraph':
        """
        Create from a serialized dictionary.
        
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
        
        # Create the graph
        graph = cls(
            name=data.get("name", f"graph_{uuid.uuid4().hex[:8]}"),
            description=data.get("description"),
            components=components,
            state_schema=data.get("state_schema"),
            input_schema=data.get("input_schema"),
            output_schema=data.get("output_schema"),
            config_schema=data.get("config_schema"),
            default_runnable_config=data.get("default_runnable_config"),
            visualize=data.get("visualize", True),
            applied_patterns=data.get("applied_patterns", [])
        )
        
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
        """
        Save the graph configuration to a file.
        
        Args:
            filename: Path to save the configuration
        """
        # Convert to dictionary
        data = self.to_dict()
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
            
        logger.info(f"Graph configuration saved to {filename}")
    
    @classmethod
    def load(cls, filename: str) -> 'DynamicGraph':
        """
        Load a graph configuration from a file.
        
        Args:
            filename: Path to load the configuration from
            
        Returns:
            Loaded DynamicGraph instance
        """
        # Load from file
        with open(filename, 'r') as f:
            data = json.load(f)
            
        # Create from dictionary
        graph = cls.from_dict(data)
        
        logger.info(f"Graph configuration loaded from {filename}")
        return graph
    
    def visualize_graph(self, filename: Optional[str] = None) -> None:
        """
        Visualize the graph structure.
        
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
            
        # Get manager and use its visualization method
        try:
            manager = self.get_manager()
            manager.visualize(output_file=filename)
            logger.info(f"Graph visualization saved to {filename}")
        except ImportError as e:
            logger.warning(f"Cannot visualize graph - required modules not available: {e}")
        except Exception as e:
            logger.warning(f"Error visualizing graph: {e}")
            
            # Try fallback visualization
            try:
                from haive_core.utils.visualize_graph_utils import render_and_display_graph
                render_and_display_graph(self.graph, output_name=filename)
                logger.info(f"Graph visualization saved to {filename} (fallback method)")
            except ImportError:
                logger.warning("Cannot visualize graph: visualization utils not available")
            except Exception as e:
                logger.error(f"Fallback visualization failed: {e}")