from typing import Any, Dict, List, Optional, Union, Type, Tuple, Callable, Literal
from pydantic import BaseModel, Field
import logging
import os
import uuid
import json
import inspect
import time
import traceback
from datetime import datetime
from enum import Enum

try:
    import networkx as nx
    import matplotlib.pyplot as plt
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

from langgraph.graph import StateGraph, START, END

from haive.core.engine.base import Engine, EngineType, EngineRegistry
from haive.core.schema.schema_composer import SchemaComposer
from haive.core.graph.node.config import NodeConfig
from haive.core.graph.node.factory import NodeFactory
from haive.core.config.runnable import RunnableConfigManager

# Configure logger
logger = logging.getLogger("DynamicGraph")
logger.setLevel(logging.INFO)

# Debug level enum
class DebugLevel(str, Enum):
    """Debug level for the graph builder."""
    NONE = "none"     # No debugging output except errors
    BASIC = "basic"   # Basic information and warnings
    VERBOSE = "verbose"  # Detailed information about operations
    TRACE = "trace"   # Complete trace of all operations with entry/exit logging
    PERFORMANCE = "performance"  # Include timing information for operations

# Edge representation
class DynamicGraphEdge(BaseModel):
    """Edge representation with enhanced debugging properties."""
    source: str
    target: str
    condition: Optional[Callable] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    
    def __str__(self):
        """String representation for debugging."""
        base = f"{self.source} → {self.target}"
        if self.condition:
            cond_name = getattr(self.condition, "__name__", "condition")
            return f"{base} [conditional: {cond_name}]"
        return base
    
    class Config:
        arbitrary_types_allowed = True

# Component reference model
class ComponentRef(BaseModel):
    """Reference to a component that can be resolved at runtime."""
    name: str
    type: Optional[str] = None
    id: Optional[str] = None
    description: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

# Node status tracking
class NodeStatus(str, Enum):
    """Status of nodes in the graph."""
    ADDED = "added"
    CONNECTED = "connected"
    UNREACHABLE = "unreachable"
    ERROR = "error"

class DynamicGraph:
    """
    Dynamic graph builder with enhanced error tracing.
    
    This class provides a builder interface for creating StateGraph instances
    with improved error diagnostics and tracing.
    """
    
    def __init__(
        self,
        name: str,
        components: List[Any] = None,
        state_schema: Optional[Type[BaseModel]] = None,
        description: Optional[str] = None,
        default_runnable_config: Optional[Dict[str, Any]] = None,
        visualize: bool = False,
        debug_level: Union[str, DebugLevel] = DebugLevel.BASIC,
        **kwargs
    ):
        """
        Initialize a new DynamicGraph builder.
        
        Args:
            name: Unique name for this graph
            components: List of engines, component references, or other components
            state_schema: Optional explicit schema class (otherwise derived from components)
            description: Optional description
            default_runnable_config: Default runtime configuration for nodes
            visualize: Whether to automatically visualize the graph
            debug_level: Level of debugging information
            **kwargs: Additional keyword arguments
        """
        self.id = kwargs.get("id", str(uuid.uuid4()))
        self.name = name
        self.description = description
        self.components = components or []
        self.visualize = visualize
        self.default_runnable_config = default_runnable_config or {}
        self.debug_level = DebugLevel(debug_level) if isinstance(debug_level, str) else debug_level
        
        # State tracking
        self.engines = {}  # name -> engine
        self.engines_by_id = {}  # id -> engine
        self.nodes = {}  # name -> node_config
        self.node_statuses = {}  # name -> NodeStatus
        self.edges = []  # list of DynamicGraphEdge
        self.branches = []  # list of branch conditions
        self.applied_patterns = []  # list of applied pattern names
        
        # Error tracking
        self.errors = []  # track errors with traces
        self.warnings = []  # track warnings
        self.operation_counts = {}  # count of operations by type
        self.created_at = datetime.now()
        self.last_modified = self.created_at
        
        # Configure logger based on debug level
        self._configure_logger()
        
        # Log initialization
        logger.info(f"Initializing DynamicGraph: {self.name} [{self.id}]")
        
        # Process components and initialize schemas
        try:
            self._process_components()
            self._initialize_schemas(state_schema)
            self._initialize_graph()
            logger.info(f"DynamicGraph initialized: {self.name}")
        except Exception as e:
            self._log_error("Initialization failed", e)
            # Re-raise with context
            raise
    
    def _configure_logger(self):
        """Configure logger based on debug level."""
        if self.debug_level == DebugLevel.NONE:
            logger.setLevel(logging.WARNING)
        elif self.debug_level == DebugLevel.BASIC:
            logger.setLevel(logging.INFO)
        elif self.debug_level in [DebugLevel.VERBOSE, DebugLevel.PERFORMANCE]:
            logger.setLevel(logging.DEBUG)
        elif self.debug_level == DebugLevel.TRACE:
            logger.setLevel(logging.DEBUG)
    
    def _log_error(self, message: str, error: Exception, is_warning: bool = False):
        """
        Log error with detailed traceback.
        
        Args:
            message: Error message
            error: Exception that occurred
            is_warning: Whether this is a warning (otherwise error)
        """
        error_message = f"{message}: {error}"
        
        if is_warning:
            logger.warning(error_message)
            self.warnings.append(error_message)
        else:
            logger.error(error_message)
            self.errors.append(error_message)
        
        if self.debug_level in [DebugLevel.VERBOSE, DebugLevel.TRACE]:
            # Get and log full traceback
            tb = traceback.format_exc()
            for line in tb.split('\n'):
                if is_warning:
                    logger.warning(f"  {line}")
                else:
                    logger.error(f"  {line}")
    
    def _process_components(self):
        """Process and resolve component references to engines."""
        logger.info(f"Processing {len(self.components)} components")
        
        registry = EngineRegistry.get_instance()
        
        resolved_components = 0
        
        for i, component in enumerate(self.components):
            try:
                # Handle string references
                if isinstance(component, str):
                    # Try to lookup in registry by name
                    engine = registry.find(component)
                    if engine:
                        logger.info(f"Resolved component string '{component}' to engine: {engine.name}")
                        component = engine
                        resolved_components += 1
                    else:
                        logger.warning(f"Could not resolve component string: {component}")
                        continue
                
                # Handle ComponentRef objects
                elif isinstance(component, ComponentRef):
                    engine = None
                    # Try to lookup by ID first
                    if component.id:
                        engine = registry.find(component.id)
                    
                    # Try to lookup by name
                    if not engine and component.name:
                        engine = registry.find(component.name)
                    
                    if engine:
                        logger.info(f"Resolved ComponentRef '{component.name}' to engine: {engine.name}")
                        component = engine
                        resolved_components += 1
                    else:
                        logger.warning(f"Could not resolve ComponentRef: {component.name}")
                        continue
                
                # If it's an Engine, register it by name and ID
                if isinstance(component, Engine):
                    self.engines[component.name] = component
                    # Register by ID if available
                    engine_id = getattr(component, "id", None)
                    if engine_id:
                        self.engines_by_id[engine_id] = component
                
                # Replace in components list (if changed)
                self.components[i] = component
                
            except Exception as e:
                self._log_error(f"Error processing component {i}", e)
                # Continue with other components
        
        logger.info(f"Processed {len(self.components)} components, resolved {resolved_components}")
    
    def _initialize_schemas(self, state_schema=None):
        """Initialize state and I/O schemas."""
        # Use provided schema if available
        if state_schema:
            self.state_model = state_schema
            logger.info(f"Using provided state schema: {state_schema.__name__}")
        else:
            # Auto-derive schema from components
            schema_name = f"{self.name.replace('-', '_').title()}State"
            logger.info(f"Deriving state schema from components as '{schema_name}'")
            
            try:
                self.schema_composer = SchemaComposer(name=schema_name)
                self.schema_composer.compose_from_components(self.components)
                self.state_model = self.schema_composer.build()
                
                logger.info(f"Successfully derived state schema: {self.state_model.__name__}")
            except Exception as e:
                self._log_error("Error deriving state schema", e, is_warning=True)
                
                # Create minimal backup schema
                logger.warning(f"Creating backup minimal schema due to derivation failure")
                self.schema_composer = SchemaComposer(name=schema_name)
                self.schema_composer.add_field("messages", list, default_factory=list)
                self.schema_composer.add_field("input", str, default="")
                self.schema_composer.add_field("output", str, default="")
                self.state_model = self.schema_composer.build()
    
    def _initialize_graph(self):
        """Initialize the underlying StateGraph."""
        try:
            self.graph_builder = StateGraph(self.state_model)
            logger.info(f"Initialized StateGraph with schema: {self.state_model.__name__}")
        except Exception as e:
            self._log_error("Error initializing StateGraph", e)
            # Analyze common initialization errors
            if "required positional argument" in str(e):
                logger.error("StateGraph initialization failed due to missing required arguments")
            elif "not callable" in str(e):
                logger.error("StateGraph initialization failed due to non-callable schema")
            
            # Re-raise with context
            raise ValueError(f"Failed to initialize StateGraph: {e}")
    
    def with_runnable_config(self, config: Dict[str, Any]) -> 'DynamicGraph':
        """
        Create a new DynamicGraph with the specified runnable config.
        
        Args:
            config: Runnable configuration to use
            
        Returns:
            New DynamicGraph instance with the specified config
        """
        # Create a new graph with the same settings but different config
        new_graph = DynamicGraph(
            name=self.name,
            components=self.components,
            state_schema=self.state_model,
            description=self.description,
            default_runnable_config=config,
            visualize=self.visualize,
            debug_level=self.debug_level,
            id=str(uuid.uuid4())  # New ID for new instance
        )
        
        # Copy existing nodes, edges, and branches
        new_graph.nodes = self.nodes.copy()
        new_graph.node_statuses = self.node_statuses.copy()
        new_graph.edges = self.edges.copy()
        new_graph.branches = self.branches.copy()
        new_graph.applied_patterns = self.applied_patterns.copy()
        
        logger.info(f"Created new graph with custom runnable config: {new_graph.id}")
        
        return new_graph
    
    def set_default_runnable_config(self, config: Dict[str, Any]) -> 'DynamicGraph':
        """
        Set the default runnable config for this graph.
        
        Args:
            config: Runnable configuration to set as default
            
        Returns:
            Self for chaining
        """
        self.default_runnable_config = config
        logger.info("Updated default runnable config")
        return self
    
    def update_default_runnable_config(self, **kwargs) -> 'DynamicGraph':
        """
        Update the default runnable config with new values.
        
        Args:
            **kwargs: Key-value pairs to update in the config
            
        Returns:
            Self for chaining
        """
        # Convert to RunnableConfig if not already
        if not self.default_runnable_config:
            self.default_runnable_config = RunnableConfigManager.create(**kwargs)
        else:
            if "configurable" not in self.default_runnable_config:
                self.default_runnable_config = {"configurable": {}}
            
            # Add each kwarg to the configurable section
            for key, value in kwargs.items():
                self.default_runnable_config["configurable"][key] = value
        
        logger.info(f"Updated default runnable config with: {', '.join(kwargs.keys())}")
        return self
    
    def add_node(
        self,
        name: str,
        config: Union[NodeConfig, Engine, str, Callable, Any],
        command_goto: Optional[Union[str, Literal["END"]]] = None,
        input_mapping: Optional[Dict[str, str]] = None,
        output_mapping: Optional[Dict[str, str]] = None,
        runnable_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> 'DynamicGraph':
        """
        Add a node to the graph with enhanced error tracking.
        
        Args:
            name: Name of the node
            config: NodeConfig, Engine, engine name, or callable function
            command_goto: Optional next node to go to
            input_mapping: Optional mapping from state keys to engine input keys
            output_mapping: Optional mapping from engine output keys to state keys
            runnable_config: Optional runtime configuration
            **kwargs: Additional node configuration parameters
            
        Returns:
            Self for chaining
        """
        try:
            # Check if node already exists
            if name in self.nodes:
                logger.warning(f"Node '{name}' already exists. Overwriting.")
            
            # Handle END constant conversion
            if command_goto == "END":
                command_goto = END
            
            # Convert to NodeConfig if not already
            if not isinstance(config, NodeConfig):
                node_config = NodeConfig(
                    name=name,
                    engine=config,
                    command_goto=command_goto,
                    input_mapping=input_mapping,
                    output_mapping=output_mapping,
                    runnable_config=runnable_config,
                    **kwargs
                )
            else:
                node_config = config
                # If this is already a NodeConfig, ensure END is properly set
                if getattr(node_config, "command_goto", None) == "END":
                    node_config.command_goto = END
            
            # Resolve engine reference
            engine, engine_id = node_config.resolve_engine()
            
            # Create node function with detailed error tracking
            try:
                # Debug log the engine type
                if isinstance(engine, Engine):
                    logger.info(f"Creating node function for {name} using engine type: {type(engine).__name__}")
                elif callable(engine):
                    logger.info(f"Creating node function for {name} using callable: {getattr(engine, '__name__', 'anonymous')}")
                else:
                    logger.info(f"Creating node function for {name} using: {type(engine).__name__}")
                
                node_function = NodeFactory.create_node_function(node_config)
            except Exception as nf_error:
                self._log_error(f"Error creating node function for '{name}'", nf_error)
                # Re-raise with more context
                raise ValueError(f"Failed to create node function: {nf_error}")
            
            # Add to StateGraph
            try:
                self.graph_builder.add_node(name, node_function)
            except Exception as sg_error:
                self._log_error(f"Error adding node '{name}' to StateGraph", sg_error)
                # Re-raise with more context
                raise ValueError(f"Failed to add node to StateGraph: {sg_error}")
            
            # Store node config for debugging
            self.nodes[name] = node_config
            self.node_statuses[name] = NodeStatus.ADDED
            
            logger.info(f"Added node: {name}")
            
            # If node has command_goto set to END, explicitly add an edge to END
            if node_config.command_goto is END:
                logger.debug(f"Node '{name}' has command_goto=END, adding explicit edge")
                self.add_edge(name, END)
            
            return self
            
        except Exception as e:
            self._log_error(f"Error adding node '{name}'", e)
            raise
    
    def add_edge(self, from_node: str, to_node: Union[str, Literal["END"]]) -> 'DynamicGraph':
        """
        Add an edge between two nodes with enhanced validation.
        
        Args:
            from_node: Source node name
            to_node: Target node name
            
        Returns:
            Self for chaining
        """
        try:
            # Convert "START" to special constant
            original_from = from_node
            if from_node == "START":
                from_node = START
            
            # Convert "END" to special constant
            original_to = to_node
            if to_node == "END":
                to_node = END
            
            # Validate nodes exist (except START/END)
            # First, recognize both "END" and "__end__" as valid special nodes
            is_end_node = original_to == "END" or original_to == "__end__"
            is_start_node = original_from == "START" or original_from == "__start__"
            
            # Check if we have any undefined nodes
            if (not is_start_node and original_from not in self.nodes) or \
               (not is_end_node and original_to not in self.nodes):
                error_nodes = []
                
                if not is_start_node and original_from not in self.nodes:
                    error_nodes.append(f"'{original_from}' (source)")
                    
                if not is_end_node and original_to not in self.nodes:
                    error_nodes.append(f"'{original_to}' (target)")
                
                # Only log warning if we have actual undefined nodes
                if error_nodes:
                    logger.warning(f"Cannot add edge between undefined nodes: {', '.join(error_nodes)}")
                    logger.warning("Edge will be added, but graph may not compile")
            
            # Add to StateGraph
            self.graph_builder.add_edge(from_node, to_node)
            
            # Update node statuses
            if original_from != "START" and original_from in self.nodes:
                self.node_statuses[original_from] = NodeStatus.CONNECTED
            
            if original_to != "END" and original_to in self.nodes:
                self.node_statuses[original_to] = NodeStatus.CONNECTED
            
            # Store edge for debugging
            edge = DynamicGraphEdge(
                source=str(original_from), 
                target=str(original_to)
            )
            self.edges.append(edge)
            
            logger.info(f"Added edge: {original_from} → {original_to}")
            
            return self
            
        except Exception as e:
            self._log_error(f"Error adding edge {from_node} → {to_node}", e)
            
            # Additional error analysis
            if "from_node" in str(e) and "is not in graph" in str(e):
                logger.error(f"Source node '{from_node}' is not in the graph")
                logger.error("Make sure to add the node first with add_node()")
            elif "to_node" in str(e) and "is not in graph" in str(e):
                logger.error(f"Target node '{to_node}' is not in the graph")
                logger.error("Make sure to add the node first with add_node()")
            
            raise
    
    def add_conditional_edges(
        self,
        from_node: str,
        condition: Callable,
        routes: Dict[str, str]
    ) -> 'DynamicGraph':
        """
        Add conditional routing based on a condition function.
        
        Args:
            from_node: Source node name
            condition: Function that takes state and returns a key
            routes: Mapping from condition results to target nodes
            
        Returns:
            Self for chaining
        """
        try:
            # Get condition name for better logging
            condition_name = getattr(condition, "__name__", "anonymous_condition")
            
            # Validate source node
            if from_node not in self.nodes:
                logger.warning(f"Conditional source node '{from_node}' doesn't exist yet")
            
            # Validate target nodes
            missing_targets = []
            for key, target in routes.items():
                if target != "END" and target not in self.nodes:
                    missing_targets.append(target)
            
            if missing_targets:
                logger.warning(f"Conditional targets don't exist yet: {', '.join(missing_targets)}")
            
            # Validate condition function takes a state parameter
            try:
                sig = inspect.signature(condition)
                if len(sig.parameters) < 1:
                    logger.warning(f"Condition function '{condition_name}' doesn't accept state parameter")
            except Exception:
                # Skip validation if we can't inspect the function
                pass
            
            # Add to StateGraph
            self.graph_builder.add_conditional_edges(
                from_node,
                condition,
                routes
            )
            
            # Store branch info for debugging
            self.branches.append({
                "source": from_node,
                "condition": condition,
                "condition_name": condition_name,
                "routes": routes,
                "created_at": datetime.now()
            })
            
            # Add edges with special formatting for conditional edges
            for key, target in routes.items():
                self.edges.append(
                    DynamicGraphEdge(
                        source=from_node,
                        target=target,
                        condition=condition,
                        metadata={
                            "condition_key": key,
                            "condition_name": condition_name
                        }
                    )
                )
                
                # Update node statuses
                if from_node in self.nodes:
                    self.node_statuses[from_node] = NodeStatus.CONNECTED
                
                if target != "END" and target in self.nodes:
                    self.node_statuses[target] = NodeStatus.CONNECTED
            
            logger.info(f"Added conditional edges from: {from_node} using '{condition_name}'")
            
            return self
            
        except Exception as e:
            self._log_error(f"Error adding conditional edges from {from_node}", e)
            
            # Enhanced error analysis
            if "not in graph" in str(e):
                logger.error(f"Node '{from_node}' is not in the graph")
                logger.error("Add the source node first with add_node()")
            elif "callable" in str(e):
                logger.error("Condition must be a callable function")
                logger.error(f"Got {type(condition).__name__} instead")
            
            raise
    
    def set_entry_point(self, node_name: str) -> 'DynamicGraph':
        """
        Set the entry point for the graph.
        
        Args:
            node_name: Name of the node to set as entry point
            
        Returns:
            Self for chaining
        """
        try:
            # Validate node exists
            if node_name not in self.nodes:
                logger.warning(f"Entry point node '{node_name}' doesn't exist yet")
            
            # Check for existing START edges
            existing_start_edges = []
            for edge in self.edges:
                if edge.source in ["START", "__start__", str(START)]:
                    existing_start_edges.append(edge)
            
            # Log warning if overriding existing entry points
            if existing_start_edges:
                target_list = ", ".join(edge.target for edge in existing_start_edges)
                logger.warning(f"Overriding existing entry point(s): {target_list}")
            
            # Add edge from START to the node
            self.add_edge(START, node_name)
            
            logger.info(f"Set entry point to: {node_name}")
            return self
            
        except Exception as e:
            self._log_error(f"Error setting entry point to {node_name}", e)
            raise
    
    def apply_pattern(self, pattern_name: str, **kwargs) -> 'DynamicGraph':
        """
        Apply a registered workflow pattern.
        
        Args:
            pattern_name: Name of the pattern to apply
            **kwargs: Pattern-specific parameters
            
        Returns:
            Self for chaining
        """
        try:
            # Import here to avoid circular imports
            from haive.core.graph.graph_pattern_registry import GraphPatternRegistry
            
            # Get pattern registry
            registry = GraphPatternRegistry.get_instance()
            
            logger.info(f"Applying pattern: {pattern_name}")
            
            # Get pattern
            pattern = registry.get_pattern(pattern_name)
            if not pattern:
                logger.warning(f"Pattern not found: {pattern_name}")
                # List available patterns as a suggestion
                try:
                    available_patterns = registry.list_patterns()
                    if available_patterns:
                        logger.warning(f"Available patterns: {', '.join(available_patterns)}")
                except Exception:
                    pass
                
                # Exit early but don't raise exception
                return self
            
            # Track initial state for comparison
            initial_node_count = len(self.nodes)
            initial_edge_count = len(self.edges)
            
            # Apply the pattern
            pattern.apply(self, **kwargs)
            
            # Track changes for detailed logging
            new_nodes = len(self.nodes) - initial_node_count
            new_edges = len(self.edges) - initial_edge_count
            
            # Track applied pattern
            self.applied_patterns.append(pattern_name)
            
            logger.info(f"Applied pattern '{pattern_name}'")
            logger.info(f"Added {new_nodes} nodes and {new_edges} edges")
            
            return self
            
        except Exception as e:
            self._log_error(f"Error applying pattern {pattern_name}", e)
            
            # Enhanced error analysis
            if "has no attribute 'apply'" in str(e):
                logger.error("Pattern doesn't have 'apply' method")
                logger.error("Pattern may be incorrectly implemented")
            elif "TypeError" in str(e) and "apply() got an unexpected keyword argument" in str(e):
                logger.error("Pattern doesn't accept the provided parameters")
                # Try to extract the invalid parameter
                import re
                param_match = re.search(r"unexpected keyword argument '(.*?)'", str(e))
                if param_match:
                    invalid_param = param_match.group(1)
                    logger.error(f"Invalid parameter: '{invalid_param}'")
            
            raise
    
    def _validate_graph(self) -> List[str]:
        """
        Perform validation of graph structure before compilation.
        
        Returns:
            List of validation issues found
        """
        logger.info("Validating graph structure...")
        
        validation_issues = []
        
        # 1. Check for START edges
        start_edges = []
        for edge in self.edges:
            if edge.source in ["START", "__start__", str(START)]:
                start_edges.append(edge)
        
        if not start_edges:
            warning_msg = "Graph has no START edges - compilation will likely fail"
            logger.warning(warning_msg)
            validation_issues.append(warning_msg)
            
            # Try to suggest a fix
            if len(self.nodes) > 0:
                first_node = next(iter(self.nodes.keys()))
                suggestion = f"Add entry point with 'graph.add_edge(START, \"{first_node}\")'"
                logger.warning(f"Suggestion: {suggestion}")
        else:
            logger.info(f"Found {len(start_edges)} START edge(s)")
        
        # 2. Check for unreachable nodes
        reachable_nodes = set()
        
        # Start with nodes directly connected to START
        for edge in self.edges:
            if edge.source in ["START", "__start__", str(START)]:
                reachable_nodes.add(edge.target)
        
        # Iteratively add nodes reachable from current set until no more are found
        new_nodes_found = True
        iteration = 0
        max_iterations = 100  # Prevent infinite loop
        
        while new_nodes_found and iteration < max_iterations:
            iteration += 1
            new_nodes_found = False
            
            for edge in self.edges:
                if edge.source in reachable_nodes and edge.target not in ["END", "__end__", str(END)]:
                    if edge.target not in reachable_nodes:
                        reachable_nodes.add(edge.target)
                        new_nodes_found = True
        
        # Find unreachable nodes
        unreachable_nodes = set(self.nodes.keys()) - reachable_nodes
        if unreachable_nodes:
            warning_msg = f"Found {len(unreachable_nodes)} unreachable node(s)"
            logger.warning(warning_msg)
            validation_issues.append(warning_msg)
            
            # Show the list of unreachable nodes
            logger.warning(f"Unreachable nodes: {', '.join(unreachable_nodes)}")
            
            # Try to suggest a fix
            for node in unreachable_nodes:
                # Update node status for visualization
                self.node_statuses[node] = NodeStatus.UNREACHABLE
        else:
            logger.info("All nodes are reachable from START")
        
        return validation_issues
    
    def build(self, checkpointer=None, **kwargs) -> Any:
        """
        Build the graph (not compiled) with validation.
        
        Args:
            checkpointer: Optional checkpointer for state persistence
            **kwargs: Additional build parameters
            
        Returns:
            Built but not compiled graph
        """
        try:
            # Run validation before building
            validation_issues = self._validate_graph()
            
            # Log build statistics
            logger.info(f"Built graph: {self.name}")
            logger.info(f"Nodes: {len(self.nodes)}")
            logger.info(f"Edges: {len(self.edges)}")
            logger.info(f"Branches: {len(self.branches)}")
            
            if validation_issues:
                logger.warning(f"{len(validation_issues)} validation issues detected")
            
            # Return built graph
            return self.graph_builder
            
        except Exception as e:
            self._log_error(f"Error building graph", e)
            
            # Enhanced error analysis
            self._analyze_build_error(e)
            
            raise
    
    def _analyze_build_error(self, error: Exception) -> None:
        """Analyze build error and provide diagnostic information."""
        error_str = str(error)
        
        # Check for common build errors
        if "Graph must have an entrypoint" in error_str:
            logger.error("Graph is missing an entry point (START edge)")
            logger.error("Add at least one edge from START to a node")
            
            # Suggest a fix if possible
            if self.nodes:
                first_node = next(iter(self.nodes.keys()))
                logger.error(f"Try adding: graph.add_edge(START, \"{first_node}\")")
        elif "No transitions defined" in error_str:
            logger.error("Graph has no transitions (edges) defined")
            logger.error("Add edges between nodes")
        elif "Node" in error_str and "is not in graph" in error_str:
            # Extract node name if possible
            import re
            node_match = re.search(r"Node ['\"](.*?)['\"] is not in graph", error_str)
            if node_match:
                node_name = node_match.group(1)
                logger.error(f"Node '{node_name}' referenced in edges but not added to graph")
                logger.error(f"Add this node first with graph.add_node()")
    
    def compile(self, checkpointer=None, **kwargs) -> Any:
        """
        Build and compile the graph with validation and diagnostics.
        
        Args:
            checkpointer: Optional checkpointer for state persistence
            **kwargs: Additional compile parameters
            
        Returns:
            Compiled graph
        """
        try:
            # Perform pre-compilation checks
            validation_issues = self._validate_graph()
            
            # Log detailed pre-compilation state
            self.debug_graph()
            
            # Compilation banner
            logger.info("COMPILING GRAPH WORKFLOW")
            
            # Handle validation issues
            if validation_issues:
                logger.warning(f"Proceeding with compilation despite {len(validation_issues)} validation issues")
            
            # Compile the graph with detailed error trapping
            logger.info(f"Compiling graph: {self.name}")
            
            try:
                compiled_graph = self.graph_builder.compile(checkpointer=checkpointer, **kwargs)
                
                # Store the compiled graph for future use
                self.compiled_graph = compiled_graph
                
                # Log success
                logger.info("COMPILATION SUCCESSFUL")
                logger.info(f"Compiled graph: {self.name}")
                logger.info(f"Nodes: {len(self.nodes)}")
                logger.info(f"Edges: {len(self.edges)}")
                logger.info(f"Branches: {len(self.branches)}")
                
                return compiled_graph
                
            except Exception as compile_error:
                # Detailed compile error handling
                self._log_error("Compilation error", compile_error)
                
                # Analyze the most common compilation issues with enhanced diagnostics
                self._analyze_compilation_error(compile_error)
                
                raise compile_error
            
        except Exception as e:
            self._log_error("Error during graph compilation process", e)
            
            # Debug detailed graph state for troubleshooting
            logger.error("Graph state at compilation failure:")
            self.debug_graph()
            
            raise
    
    def _analyze_compilation_error(self, error: Exception) -> None:
        """
        Analyze compilation error and provide targeted debugging guidance.
        
        Args:
            error: The exception that occurred
        """
        error_str = str(error)
        
        # Format issue header
        logger.error("COMPILATION DIAGNOSTIC")
        
        # Handle specific known error patterns with enhanced diagnosis
        if "Graph must have an entrypoint" in error_str:
            logger.error("CRITICAL: Graph is missing an entry point (START edge)")
            logger.error("SOLUTION: Add at least one edge from START to a node:")
            
            if self.nodes:
                first_node = next(iter(self.nodes.keys()))
                logger.error(f"    graph.add_edge(START, \"{first_node}\")")
                
                # Suggest entry point
                logger.error(f"SUGGESTION: Consider using '{first_node}' as entry point")
            else:
                logger.error("    No nodes found. Add nodes before setting entry point.")
                
        elif "Node" in error_str and "is unreachable" in error_str:
            # Extract node name from error message
            import re
            node_match = re.search(r"Node ['\"](.*?)['\"] is unreachable", error_str)
            if node_match:
                unreachable_node = node_match.group(1)
                logger.error(f"CRITICAL: Node '{unreachable_node}' is unreachable")
                logger.error("SOLUTION: Ensure there is a path from START to this node:")
                
                # Attempt to find closest connected node to suggest a fix
                connected_nodes = set()
                for edge in self.edges:
                    if edge.source in ["START", "__start__", str(START)]:
                        connected_nodes.add(edge.target)
                
                if connected_nodes:
                    closest_node = next(iter(connected_nodes))
                    logger.error(f"    graph.add_edge(\"{closest_node}\", \"{unreachable_node}\")")
                    logger.error(f"SUGGESTION: Consider connecting '{unreachable_node}' to '{closest_node}'")
                else:
                    logger.error(f"    graph.add_edge(START, \"{unreachable_node}\")")
            else:
                logger.error("CRITICAL: Some node is unreachable")
        
        # General advice
        logger.error("GENERAL DEBUGGING STEPS")
        logger.error("1. Ensure all nodes are properly connected (check START and END edges)")
        logger.error("2. Validate that all conditional functions return expected values")
        logger.error("3. Check for typos in node names throughout your code")
        logger.error("4. Verify your state schema is consistent and supports all required fields")
        logger.error("5. Try adding nodes incrementally and compile after each addition to isolate issues")
    
    def get_manager(self) -> Any:
        """
        Get a StateGraphManager for this graph.
        
        Returns:
            StateGraphManager instance
        """
        try:
            from haive.core.graph.state_graph_manager import StateGraphManager
            return StateGraphManager(self.graph_builder)
        except Exception as e:
            self._log_error("Error creating StateGraphManager", e)
            raise
    
    def debug_graph(self) -> None:
        """Print detailed debug information about the graph."""
        logger.info(f"GRAPH DEBUG: {self.name}")
        
        # Graph info
        logger.info("GRAPH INFO:")
        logger.info(f"  • Name: {self.name}")
        logger.info(f"  • ID: {self.id}")
        logger.info(f"  • Description: {self.description}")
        logger.info(f"  • State Schema: {self.state_model.__name__}")
        logger.info(f"  • Debug Level: {self.debug_level.value}")
        logger.info(f"  • Created: {self.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Components
        logger.info("COMPONENTS:")
        if not self.components:
            logger.info("  • None defined")
        else:
            for i, component in enumerate(self.components):
                component_type = type(component).__name__
                name = getattr(component, "name", f"component_{i}")
                engine_type = getattr(component, "engine_type", None)
                engine_id = getattr(component, "id", None)
                
                logger.info(f"  • [{i+1}/{len(self.components)}] {name} ({component_type})")
                if engine_type:
                    logger.info(f"    → Type: {engine_type}")
                if engine_id:
                    logger.info(f"    → ID: {engine_id}")
        
        # Nodes with status
        logger.info("NODES:")
        if not self.nodes:
            logger.info("  • None defined")
        else:
            for name, node_config in self.nodes.items():
                # Get status
                status = self.node_statuses.get(name, NodeStatus.ADDED)
                
                # Get engine info
                engine_info = ""
                if hasattr(node_config, "engine"):
                    engine = node_config.engine
                    if hasattr(engine, "name"):
                        engine_info = f" [Engine: {engine.name}]"
                    elif isinstance(engine, str):
                        engine_info = f" [Engine: {engine}]"
                    elif callable(engine) and hasattr(engine, "__name__"):
                        engine_info = f" [Function: {engine.__name__}]"
                
                logger.info(f"  • {name}{engine_info} - {status.value}")
        
        # Edges
        logger.info("EDGES:")
        if not self.edges:
            logger.info("  • None defined")
        else:
            # Group edges for better visibility
            standard_edges = []
            conditional_edges = []
            start_edges = []
            end_edges = []
            
            for edge in self.edges:
                if edge.source in ["START", "__start__", str(START)]:
                    start_edges.append(edge)
                elif edge.target in ["END", "__end__", str(END)]:
                    end_edges.append(edge)
                elif edge.condition:
                    conditional_edges.append(edge)
                else:
                    standard_edges.append(edge)
            
            # Print entry points first
            if start_edges:
                logger.info("  • Entry Points:")
                for edge in start_edges:
                    logger.info(f"    → START → {edge.target}")
            
            # Print standard edges
            if standard_edges:
                logger.info("  • Standard Edges:")
                for edge in standard_edges:
                    logger.info(f"    → {edge.source} → {edge.target}")
            
            # Print conditional edges
            if conditional_edges:
                logger.info("  • Conditional Edges:")
                for edge in conditional_edges:
                    condition_name = getattr(edge.condition, "__name__", "conditional")
                    logger.info(f"    → {edge.source} → {edge.target} [{condition_name}]")
            
            # Print exit points
            if end_edges:
                logger.info("  • Exit Points:")
                for edge in end_edges:
                    logger.info(f"    → {edge.source} → END")
        
        # Errors and warnings
        if self.errors:
            logger.info("ERRORS:")
            for i, error in enumerate(self.errors):
                logger.error(f"  • Error {i+1}: {error}")
        
        if self.warnings:
            logger.info("WARNINGS:")
            for i, warning in enumerate(self.warnings):
                logger.warning(f"  • Warning {i+1}: {warning}")
    
    def visualize_graph(self, output_file: Optional[str] = None) -> None:
        """
        Visualize the graph using NetworkX.
        
        Args:
            output_file: Optional path to save the visualization (defaults to graph name)
        """
        if not VISUALIZATION_AVAILABLE:
            logger.warning("Visualization requires NetworkX and Matplotlib - skipping")
            return
        
        try:
            # Set default output file if not provided
            if output_file is None:
                # Generate a default filename with timestamp for uniqueness
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = f"{self.name.replace(' ', '_').lower()}_graph_{timestamp}.png"
            
            logger.info(f"Visualizing graph to: {output_file}")
            
            # Create a new graph
            G = nx.DiGraph()
            
            # Add START and END nodes
            start_node = "START"
            end_node = "END"
            G.add_node(start_node)
            G.add_node(end_node)
            
            # Add regular nodes
            for name in self.nodes:
                G.add_node(name)
            
            # Add all edges
            for edge in self.edges:
                # Handle START/END conversion
                source = start_node if edge.source in ["START", "__start__", str(START)] else edge.source
                target = end_node if edge.target in ["END", "__end__", str(END)] else edge.target
                
                # Add edge to graph
                G.add_edge(source, target)
            
            # Create the layout
            pos = nx.spring_layout(G, seed=42)
            
            # Draw nodes
            nx.draw_networkx_nodes(
                G, pos,
                node_color="lightblue",
                node_size=1500,
                edgecolors='black'
            )
            
            # Draw edges
            nx.draw_networkx_edges(
                G, pos,
                edge_color="gray",
                width=1.5,
                arrowsize=15,
                arrows=True
            )
            
            # Draw node labels
            nx.draw_networkx_labels(
                G, pos,
                font_size=10,
                font_weight='bold'
            )
            
            # Add title
            plt.title(f"{self.name} Graph")
            
            # Remove axis
            plt.axis('off')
            
            # Make sure directory exists
            import os
            os.makedirs(os.path.dirname(os.path.abspath(output_file)) or '.', exist_ok=True)
            
            # Save the visualization
            plt.tight_layout()
            plt.savefig(output_file, format="png", bbox_inches='tight')
            plt.close()
            
            logger.info(f"Graph visualization saved to: {output_file}")
            
        except Exception as e:
            self._log_error(f"Error in visualization", e)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to a serializable dictionary.
        
        Returns:
            Dictionary representation of the graph
        """
        try:
            # Basic info
            result = {
                "id": self.id,
                "name": self.name,
                "description": self.description,
                "visualize": self.visualize,
                "debug_level": self.debug_level.value,
                "created_at": self.created_at.isoformat(),
                "nodes": {},
                "edges": [],
                "applied_patterns": self.applied_patterns,
                "stats": {
                    "node_count": len(self.nodes),
                    "edge_count": len(self.edges),
                    "branch_count": len(self.branches),
                    "component_count": len(self.components),
                    "error_count": len(self.errors),
                    "warning_count": len(self.warnings)
                }
            }
            
            # Add components (with special handling)
            component_refs = []
            for component in self.components:
                if isinstance(component, Engine):
                    component_refs.append({
                        "type": "engine",
                        "name": component.name,
                        "id": getattr(component, "id", None),
                        "engine_type": str(getattr(component, "engine_type", None))
                    })
                elif isinstance(component, str):
                    component_refs.append({
                        "type": "reference",
                        "name": component
                    })
                elif hasattr(component, "to_dict"):
                    component_refs.append(component.to_dict())
                else:
                    component_refs.append({
                        "type": "unknown",
                        "class": type(component).__name__
                    })
            
            result["components"] = component_refs
            
            # Add schema info
            result["schema"] = {
                "name": self.state_model.__name__,
                "fields": list(getattr(self.state_model, "__annotations__", {}).keys())
            }
            
            # Add nodes with status
            for name, node_config in self.nodes.items():
                if hasattr(node_config, "to_dict"):
                    node_data = node_config.to_dict()
                    # Add node status
                    node_data["status"] = self.node_statuses.get(name, NodeStatus.ADDED).value
                    result["nodes"][name] = node_data
                else:
                    result["nodes"][name] = {
                        "type": type(node_config).__name__,
                        "status": self.node_statuses.get(name, NodeStatus.ADDED).value
                    }
            
            # Add edges
            for edge in self.edges:
                edge_dict = {
                    "source": edge.source,
                    "target": edge.target,
                    "conditional": edge.condition is not None,
                }
                
                # Add condition name if available
                if edge.condition and hasattr(edge.condition, "__name__"):
                    edge_dict["condition_name"] = edge.condition.__name__
                    
                if edge.metadata:
                    edge_dict["metadata"] = edge.metadata
                    
                result["edges"].append(edge_dict)
            
            # Add default config
            result["default_runnable_config"] = self._sanitize_secret_values(self.default_runnable_config)
            
            return result
            
        except Exception as e:
            self._log_error(f"Error serializing graph", e)
            raise
    
    def _sanitize_secret_values(self, data: Any) -> Any:
        """
        Sanitize sensitive values in data structure.
        
        Args:
            data: Data structure to sanitize
            
        Returns:
            Sanitized data structure
        """
        # List of patterns that indicate sensitive information
        sensitive_patterns = [
            "api_key", "apikey", "key", "token", "secret", "password", "credential",
            "auth", "access", "private"
        ]
        
        if isinstance(data, dict):
            result = {}
            for key, value in data.items():
                # Check for sensitive keys
                is_sensitive = any(pattern in key.lower() for pattern in sensitive_patterns)
                
                if is_sensitive and isinstance(value, str):
                    # Mask string values
                    if value:
                        result[key] = "********"
                    else:
                        result[key] = ""
                else:
                    result[key] = self._sanitize_secret_values(value)
            return result
        elif isinstance(data, list):
            return [self._sanitize_secret_values(item) for item in data]
        elif isinstance(data, tuple):
            return tuple(self._sanitize_secret_values(item) for item in data)
        elif isinstance(data, set):
            return {self._sanitize_secret_values(item) for item in data}
        else:
            return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DynamicGraph':
        """
        Create from a serialized dictionary.
        
        Args:
            data: Dictionary representation of a graph
            
        Returns:
            Instantiated DynamicGraph
        """
        # Configure logger for class method
        logger.info(f"Loading graph from dictionary: {data.get('name')}")
        
        # Extract basic info
        name = data.get("name", "loaded_graph")
        description = data.get("description")
        visualize = data.get("visualize", False)
        debug_level = data.get("debug_level", DebugLevel.BASIC)
        
        # Create empty graph
        graph = cls(
            name=name,
            description=description,
            visualize=visualize,
            debug_level=debug_level,
            id=data.get("id", str(uuid.uuid4()))
        )
        
        # Resolve components if provided
        registry = EngineRegistry.get_instance()
        components = []
        
        for comp_ref in data.get("components", []):
            try:
                # Handle different component reference formats
                if isinstance(comp_ref, dict):
                    comp_type = comp_ref.get("type")
                    
                    if comp_type == "engine" and comp_ref.get("name"):
                        # Try to find engine by ID first, then by name
                        engine = None
                        if comp_ref.get("id"):
                            engine = registry.find(comp_ref["id"])
                        
                        if not engine and comp_ref.get("name"):
                            engine = registry.find(comp_ref["name"])
                        
                        if engine:
                            components.append(engine)
                        else:
                            # Add as component reference
                            components.append(ComponentRef(
                                name=comp_ref["name"],
                                id=comp_ref.get("id"),
                                type=str(comp_ref.get("engine_type"))
                            ))
                    elif comp_type == "reference" and comp_ref.get("name"):
                        # Simple string reference
                        components.append(comp_ref["name"])
                elif isinstance(comp_ref, str):
                    # Direct string reference
                    components.append(comp_ref)
            except Exception as e:
                logger.error(f"Error processing component reference: {e}")
        
        graph.components = components
        graph._process_components()
        graph._initialize_schemas()
        
        # Add nodes if provided
        for name, node_data in data.get("nodes", {}).items():
            try:
                # Find engine for this node
                engine = None
                engine_ref = node_data.get("engine_ref")
                
                if isinstance(engine_ref, dict) and engine_ref.get("name"):
                    # Try by ID first
                    if engine_ref.get("id"):
                        engine = registry.find(engine_ref["id"])
                    
                    # Then by name
                    if not engine and engine_ref.get("name"):
                        engine = registry.find(engine_ref["name"])
                elif isinstance(engine_ref, str):
                    engine = registry.find(engine_ref)
                
                if engine:
                    # Create node config
                    node_config = NodeConfig(
                        name=name,
                        engine=engine,
                        command_goto=node_data.get("command_goto"),
                        input_mapping=node_data.get("input_mapping"),
                        output_mapping=node_data.get("output_mapping")
                    )
                    
                    # Add to graph
                    graph.add_node(name, node_config)
                    
                    # Set node status if provided
                    if node_data.get("status"):
                        try:
                            graph.node_statuses[name] = NodeStatus(node_data["status"])
                        except:
                            # Use default status if invalid
                            graph.node_statuses[name] = NodeStatus.ADDED
                else:
                    logger.warning(f"Could not resolve engine for node: {name}")
            except Exception as e:
                logger.warning(f"Error adding node {name} from serialized data: {e}")
        
        # Add edges if provided
        for edge_data in data.get("edges", []):
            try:
                source = edge_data.get("source")
                target = edge_data.get("target")
                
                if not source or not target:
                    logger.warning(f"Invalid edge data: missing source or target")
                    continue
                
                # Handle only non-conditional edges with the simple add_edge method
                if not edge_data.get("conditional", False):
                    graph.add_edge(source, target)
                else:
                    # Log that conditional edges require special handling
                    logger.warning(f"Conditional edge from {source} to {target} needs manual re-creation")
            except Exception as e:
                logger.warning(f"Error processing edge data: {e}")
        
        # Set default config if provided
        default_config = data.get("default_runnable_config")
        if default_config:
            graph.default_runnable_config = default_config
        
        # Final restoration summary
        logger.info(f"Loaded graph: {name} with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
        return graph
    
    def save(self, filename: str) -> None:
        """
        Save the graph configuration to a file.
        
        Args:
            filename: Path to save the graph
        """
        try:
            # Convert to dictionary
            data = self.to_dict()
            
            # Create directory if needed
            import os
            os.makedirs(os.path.dirname(os.path.abspath(filename)) or '.', exist_ok=True)
            
            # Save to file with pretty formatting
            with open(filename, "w") as f:
                json.dump(data, f, indent=2, default=str)  # Use str for non-serializable objects
            
            logger.info(f"Graph saved to: {filename}")
            
        except Exception as e:
            self._log_error(f"Error saving graph", e)
            raise
    
    @classmethod
    def load(cls, filename: str) -> 'DynamicGraph':
        """
        Load a graph configuration from a file.
        
        Args:
            filename: Path to the graph configuration file
            
        Returns:
            Instantiated DynamicGraph
        """
        logger.info(f"Loading graph from file: {filename}")
        
        try:
            # Check if file exists
            import os
            if not os.path.exists(filename):
                raise FileNotFoundError(f"Graph configuration file not found: {filename}")
            
            # Load from file
            try:
                with open(filename, "r") as f:
                    data = json.load(f)
            except json.JSONDecodeError as json_err:
                logger.error(f"JSON parsing error: {json_err}")
                raise
            
            # Create from dictionary
            graph = cls.from_dict(data)
            
            logger.info(f"Graph loaded from file")
            return graph
            
        except Exception as e:
            logger.error(f"Error loading graph: {e}")
            
            # Enhanced error analysis
            if isinstance(e, FileNotFoundError):
                logger.error(f"The file '{filename}' does not exist")
            elif isinstance(e, PermissionError):
                logger.error(f"Permission denied when trying to read '{filename}'")
            elif isinstance(e, json.JSONDecodeError):
                logger.error(f"Invalid JSON format in '{filename}'")
            
            raise