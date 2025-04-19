"""
Agent base class for the Haive framework.

This module provides the Agent base class that implements the agent architecture
based on the configuration. It handles graph building, pattern application,
and execution of the agent workflow.
"""

import os
import json
import uuid
import logging
from datetime import datetime
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Type, Generic, TypeVar, Generator, AsyncGenerator, Tuple

from pydantic import BaseModel
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore
from langgraph.types import Command, Send
from langchain_core.runnables import RunnableConfig

from haive_core.engine.base import Engine, EngineType
from haive_core.graph.dynamic_graph_builder import DynamicGraph
from haive_core.config.runnable import RunnableConfigManager

# Import persistence-related functionality
from haive_core.engine.agent.persistence.types import CheckpointerType
from haive_core.engine.agent.persistence.base import CheckpointerConfig
from haive_core.engine.agent.persistence.handlers import (
    setup_checkpointer, 
    ensure_pool_open, 
    close_pool_if_needed, 
    register_thread_if_needed
)

# Import state handling functionality - check this 
from haive_core.engine.agent.utils.state_handling import (
    process_input, 
    prepare_merged_input
)
from haive_core.engine.agent.pattern import PatternManager
from haive_core.engine.agent.config import AgentConfig
# Import pattern system - handle import outside class to avoid circular imports
try:
    from haive_core.graph.patterns.registry import GraphPatternRegistry
    from haive_core.graph.patterns.integration import (
        apply_pattern_to_graph,
        register_integrations as register_pattern_integrations
    )
    PATTERN_SYSTEM_AVAILABLE = True
    
    # Register integrations
    register_pattern_integrations()
except ImportError:
    PATTERN_SYSTEM_AVAILABLE = False

# Set up logging
logger = logging.getLogger(__name__)

# Type variables for generics
TConfig = TypeVar('TConfig', bound='AgentConfig')

# -----------------------------------------------------
# Agent Registry - Maps config classes to agent classes
# -----------------------------------------------------
AGENT_REGISTRY: Dict[Type, Type] = {}

def register_agent(config_class: Type):
    """
    Register an agent class with its configuration class.
    
    Args:
        config_class: The agent config class to register
        
    Returns:
        A decorator function that registers the agent class
    """
    def decorator(agent_class: Type):
        AGENT_REGISTRY[config_class] = agent_class
        logger.debug(f"Registered agent {agent_class.__name__} for config {config_class.__name__}")
        return agent_class
    return decorator


class Agent(Generic[TConfig], ABC):
    """
    Base agent architecture class.
    
    Defines how an agent works - its implementation and behavior through a workflow graph.
    Each Agent is paired with an AgentConfig class that defines its configuration.
    """
    def __init__(self, config: TConfig):
        """Initialize the agent with its configuration."""
        self.config = config
        
        # Ensure output directory exists
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Set up checkpointer
        self.checkpointer = setup_checkpointer(self.config)
        
        # Initialize runnable_config from config
        self.runnable_config = getattr(self.config, 'runnable_config', 
            RunnableConfigManager.create(thread_id=str(uuid.uuid4())))
        
        # Add store if configured 
        if getattr(self.config, 'add_store', False):
            self.store = BaseStore()
        else:
            self.store = None
        
        # Initialize engines
        self._initialize_engines()
        
        # Initialize state schemas
        self._initialize_schemas()
        
        # Set up output paths
        self._setup_output_paths()
        
        # Create graph builder
        self._create_graph_builder()
        
        # Set up workflow
        self.setup_workflow()
        
        # Apply node configurations if available
        self._apply_node_configs()
        
        # Apply patterns
        self._apply_patterns()
        
        # Compile the graph
        self.app = self.compile()
        
        # Generate visualization if requested
        if getattr(self.config, 'visualize', True) and self.graph:
            self.visualize_graph()
    
    def _initialize_engines(self) -> None:
        """Initialize all engines (LLMs, retrievers, etc.)."""
        self.engines = {}
        
        # Initialize main engine if present
        if self.config.engine:
            self.engine = self._build_engine(self.config.engine)
            self.engines["main"] = self.engine
        
        # Initialize additional engines
        for name, engine_config in getattr(self.config, 'engines', {}).items():
            self.engines[name] = self._build_engine(engine_config)
    
    def _build_engine(self, config):
        """Build an engine from configuration."""
        if isinstance(config, str):
            # Try to resolve from registry
            from haive_core.engine.base import EngineRegistry, EngineType
            registry = EngineRegistry.get_instance()
            for engine_type in EngineType:
                engine = registry.get(engine_type, config)
                if engine:
                    config = engine
                    break
            if isinstance(config, str):
                raise ValueError(f"Engine '{config}' not found in registry")
        
        # Build based on engine type
        if hasattr(config, "create_runnable"):
            return config.create_runnable()
        return config  # Return as-is if already built
    
    def _initialize_schemas(self) -> None:
        """Initialize state, input, output, and config schemas."""
        # Generate schemas from config
        self.state_schema = self.config.derive_schema()
        self.input_schema = self.config.derive_input_schema()
        self.output_schema = self.config.derive_output_schema()
    
    def _setup_output_paths(self) -> None:
        """Set up paths for output files (state history, graph images)."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Set up state history directory and file
        self.state_history_dir = os.path.join(self.config.output_dir, "state_history")
        os.makedirs(self.state_history_dir, exist_ok=True)
        self.state_filename = os.path.join(
            self.state_history_dir, 
            f"{self.config.name}_{timestamp}.json"
        )
        
        # Set up graphs directory and file
        self.graphs_dir = os.path.join(self.config.output_dir, "graphs")
        os.makedirs(self.graphs_dir, exist_ok=True)
        self.graph_image_path = os.path.join(
            self.graphs_dir, 
            f"{self.config.name}_{timestamp}.png"
        )
    
    def _create_graph_builder(self) -> None:
        """Create the DynamicGraph builder for defining the workflow."""
        # Create graph builder with schemas
        self.graph = DynamicGraph(
            name=self.config.name,
            description=getattr(self.config, 'description', None),
            components=list(self.engines.values()),
            state_schema=self.state_schema,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            default_runnable_config=self.runnable_config,
            visualize=self.config.visualize,
            visualization_dir=self.graphs_dir
        )
    
    @abstractmethod
    def setup_workflow(self) -> None:
        """
        Set up the workflow graph for this agent.
        
        This method must be implemented by concrete agent classes to define
        the agent's workflow structure.
        """
        pass
    
    def _apply_node_configs(self) -> None:
        """Apply node configurations from the agent config."""
        if not hasattr(self.config, 'node_configs') or not self.config.node_configs:
            return
            
        for node_name, node_config in self.config.node_configs.items():
            if node_name not in self.graph.nodes:
                # Add node if not already added
                self.graph.add_node(name=node_name, config=node_config)
                logger.info(f"Added node '{node_name}' from config")
    
    def _apply_patterns(self) -> None:
        """Apply patterns based on configuration."""
        if not PATTERN_SYSTEM_AVAILABLE:
            logger.warning("Pattern system not available, skipping pattern application")
            return
        
        if not hasattr(self.config, '_pattern_manager') or not self.config._pattern_manager:
            logger.debug("No patterns to apply")
            return
            
        # Get pattern order from manager
        pattern_order = self.config.get_pattern_order()
        
        # Apply patterns in order
        registry = GraphPatternRegistry.get_instance()
        for pattern_name in pattern_order:
            # Get combined parameters
            params = self.config.get_pattern_parameters(pattern_name)
            
            # Apply pattern
            logger.info(f"Applying pattern '{pattern_name}' to agent")
            try:
                apply_pattern_to_graph(self.graph, pattern_name, **params)
                
                # Mark as applied
                self.config.mark_pattern_applied(pattern_name)
                
            except Exception as e:
                logger.error(f"Error applying pattern '{pattern_name}': {e}")
    
    def compile(self) -> Any:
        """
        Compile the workflow graph into an executable app.
        
        Returns:
            Compiled graph application
        """
        # Ensure graph is built
        if not self.graph:
            raise RuntimeError("Graph not set up")
        
        # Compile with checkpointer and proper runnable config
        compiled = self.graph.compile(
            checkpointer=self.checkpointer,
            default_config=self.runnable_config
        )
        
        return compiled
    
    def visualize_graph(self) -> None:
        """Generate and save a visualization of the graph."""
        if not hasattr(self, 'graph') or not self.graph:
            logger.warning("Cannot visualize: no graph available")
            return
            
        try:
            self.graph.visualize_graph(self.graph_image_path)
        except Exception as e:
            logger.error(f"Error during graph visualization: {e}")
    
    def run(
        self,
        input_data: Union[str, List[str], Dict[str, Any], BaseModel]={},
        thread_id: Optional[str] = None,
        config: Optional[RunnableConfig] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run the agent with the given input and return the final state.
        
        Args:
            input_data: Input data in various formats
            thread_id: Optional thread ID for persistence
            config: Optional runtime configuration
            **kwargs: Additional runtime configuration
            
        Returns:
            Final state or output
        """
        if not self.app:
            self.app = self.compile()
            
        # Prepare runtime configuration
        runtime_config = self._prepare_runnable_config(thread_id, config, **kwargs)
        current_thread_id = runtime_config["configurable"]["thread_id"]
        
        # Register thread in PostgreSQL if needed
        register_thread_if_needed(self.checkpointer, current_thread_id)
        
        # Ensure PostgreSQL connection is ready
        opened_pool = ensure_pool_open(self.checkpointer)
        
        try:
            # Try to retrieve previous state
            try:
                previous_state = self.app.get_state(runtime_config)
                if previous_state:
                    logger.info(f"Found previous state for thread {current_thread_id}")
            except Exception as e:
                logger.debug(f"No previous state found: {e}")
                previous_state = None
            
            # Process and merge input with previous state if it exists
            processed_input = prepare_merged_input(
                input_data, 
                previous_state, 
                runtime_config, 
                self.input_schema, 
                self.state_schema
            )
            
            # Run the agent
            logger.info(f"Running agent {self.config.name} with thread_id: {current_thread_id}")
            result = self.app.invoke(
                processed_input,
                config=runtime_config,
                debug=getattr(self.config, 'debug', False)
            )
            
            # Save state history if requested
            if getattr(self.config, 'save_history', True):
                self.save_state_history(runtime_config)
            
            return result
        
        except Exception as e:
            logger.error(f"Error running agent: {e}")
            raise e
        
        finally:
            # Close pool if opened
            if opened_pool:
                close_pool_if_needed(self.checkpointer, opened_pool)
    
    def stream(
        self, 
        input_data: Union[str, List[str], Dict[str, Any], BaseModel], 
        thread_id: Optional[str] = None,
        stream_mode: str = "values", 
        config: Optional[RunnableConfig] = None,
        **kwargs
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Stream the agent execution.
        
        Args:
            input_data: Input data in various formats
            thread_id: Optional thread ID for persistence
            stream_mode: Stream mode (values, updates, debug, etc.)
            config: Optional runtime configuration
            **kwargs: Additional runtime configuration
            
        Yields:
            State updates during execution
        """
        if not self.app:
            self.app = self.compile()
            
        # Prepare runtime configuration
        runtime_config = self._prepare_runnable_config(thread_id, config, **kwargs)
        current_thread_id = runtime_config["configurable"]["thread_id"]
        
        # Register thread in PostgreSQL if needed
        register_thread_if_needed(self.checkpointer, current_thread_id)
        
        # Ensure PostgreSQL connection is ready
        opened_pool = ensure_pool_open(self.checkpointer)
        
        try:
            # Try to retrieve previous state
            try:
                previous_state = self.app.get_state(runtime_config)
                if previous_state:
                    logger.info(f"Found previous state for thread {current_thread_id}")
            except Exception as e:
                logger.debug(f"No previous state found: {e}")
                previous_state = None
            
            # Process and merge input with previous state if it exists
            processed_input = prepare_merged_input(
                input_data, 
                previous_state, 
                runtime_config, 
                self.input_schema, 
                self.state_schema
            )
            
            # Stream the execution
            logger.info(f"Streaming agent {self.config.name}")
            for output in self.app.stream(
                processed_input,
                stream_mode=stream_mode,
                config=runtime_config,
                debug=getattr(self.config, 'debug', False)
            ):
                yield output
                
            # Save state history if requested
            if getattr(self.config, 'save_history', True):
                self.save_state_history(runtime_config)
        
        finally:
            # Close pool if opened
            if opened_pool:
                close_pool_if_needed(self.checkpointer, opened_pool)
    
    async def arun(
        self,
        input_data: Union[str, List[str], Dict[str, Any], BaseModel],
        thread_id: Optional[str] = None,
        config: Optional[RunnableConfig] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run the agent asynchronously with the given input.
        
        Args:
            input_data: Input data in various formats
            thread_id: Optional thread ID for persistence
            config: Optional runtime configuration
            **kwargs: Additional runtime configuration
            
        Returns:
            Final state or output
        """
        if not self.app:
            self.app = self.compile()
            
        # Prepare runtime configuration
        runtime_config = self._prepare_runnable_config(thread_id, config, **kwargs)
        current_thread_id = runtime_config["configurable"]["thread_id"]
        
        # Register thread in PostgreSQL if needed - use sync version for now
        register_thread_if_needed(self.checkpointer, current_thread_id)
        
        # Ensure PostgreSQL connection is ready
        opened_pool = ensure_pool_open(self.checkpointer)
        
        try:
            # Try to retrieve previous state
            try:
                previous_state = await self.app.aget_state(runtime_config)
                if previous_state:
                    logger.info(f"Found previous state for thread {current_thread_id}")
            except Exception as e:
                logger.debug(f"No previous state found: {e}")
                previous_state = None
            
            # Process and merge input with previous state if it exists
            processed_input = prepare_merged_input(
                input_data, 
                previous_state, 
                runtime_config, 
                self.input_schema, 
                self.state_schema
            )
            
            # Run the agent asynchronously
            logger.info(f"Running agent {self.config.name} asynchronously with thread_id: {current_thread_id}")
            result = await self.app.ainvoke(
                processed_input,
                config=runtime_config,
                debug=getattr(self.config, 'debug', False)
            )
            
            # Save state history if requested
            if getattr(self.config, 'save_history', True):
                self.save_state_history(runtime_config)
                
            return result
        except Exception as e:
            logger.error(f"Error running agent asynchronously: {e}")
            raise e
        finally:
            # Close pool if opened
            if opened_pool:
                close_pool_if_needed(self.checkpointer, opened_pool)
            
    async def astream(
        self, 
        input_data: Union[str, List[str], Dict[str, Any], BaseModel], 
        thread_id: Optional[str] = None,
        stream_mode: str = "values", 
        config: Optional[RunnableConfig] = None,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream the agent execution asynchronously.
        
        Args:
            input_data: Input data in various formats
            thread_id: Optional thread ID for persistence
            stream_mode: Stream mode (values, updates, debug, etc.)
            config: Optional runtime configuration
            **kwargs: Additional runtime configuration
            
        Yields:
            State updates during execution
        """
        if not self.app:
            self.app = self.compile()
            
        # Prepare runtime configuration
        runtime_config = self._prepare_runnable_config(thread_id, config, **kwargs)
        current_thread_id = runtime_config["configurable"]["thread_id"]
        
        # Register thread in PostgreSQL if needed - use sync version for now
        register_thread_if_needed(self.checkpointer, current_thread_id)
        
        # Ensure PostgreSQL connection is ready
        opened_pool = ensure_pool_open(self.checkpointer)
        
        try:
            # Try to retrieve previous state
            try:
                previous_state = await self.app.aget_state(runtime_config)
                if previous_state:
                    logger.info(f"Found previous state for thread {current_thread_id}")
            except Exception as e:
                logger.debug(f"No previous state found: {e}")
                previous_state = None
            
            # Process and merge input with previous state if it exists
            processed_input = prepare_merged_input(
                input_data, 
                previous_state, 
                runtime_config, 
                self.input_schema, 
                self.state_schema
            )
            
            # Stream the execution asynchronously
            logger.info(f"Streaming agent {self.config.name} asynchronously")
            async for output in self.app.astream(
                processed_input,
                stream_mode=stream_mode,
                config=runtime_config,
                debug=getattr(self.config, 'debug', False)
            ):
                yield output
                
            # Save state history if requested
            if getattr(self.config, 'save_history', True):
                self.save_state_history(runtime_config)
        finally:
            # Close pool if opened
            if opened_pool:
                close_pool_if_needed(self.checkpointer, opened_pool)
    
    def save_state_history(self, runnable_config=None) -> None:
        """
        Save the current agent state to a JSON file.
        
        Args:
            runnable_config: Optional runnable configuration
        """
        if not self.app:
            logger.error(f"Cannot save state history: Workflow graph not compiled")
            return

        # Use provided runnable config or default
        runnable_config = runnable_config or self.runnable_config
            
        try:
            # Get state from app
            state_json = self.app.get_state(runnable_config)
            
            if not state_json:
                logger.warning(f"No state history available for {self.config.name}")
                return
            
            # Ensure state is JSON serializable
            from haive_core.utils.pydantic_utils import ensure_json_serializable
            state_json = ensure_json_serializable(state_json)
            
            # Save to file
            with open(self.state_filename, "w", encoding="utf-8") as f:
                json.dump(state_json, f, indent=4)
            
            logger.info(f"State history saved to: {self.state_filename}")
        except Exception as e:
            logger.error(f"Error saving state history: {str(e)}")
    
    def _prepare_runnable_config(self, thread_id=None, config=None, **kwargs) -> RunnableConfig:
        """
        Prepare runnable config with thread ID and other parameters.
        
        Args:
            thread_id: Optional thread ID for persistence
            config: Optional base configuration
            **kwargs: Additional parameters
            
        Returns:
            Prepared RunnableConfig
        """
        # Start with base config from the agent
        base_config = getattr(self, 'runnable_config', {})
        
        # If explicit config provided, start with that instead
        if config:
            # Use RunnableConfigManager to merge configs
            runtime_config = RunnableConfigManager.merge(base_config, config)
        else:
            # Start with a copy of the base config
            runtime_config = json.loads(json.dumps(base_config))
        
        # Ensure configurable section exists
        if "configurable" not in runtime_config:
            runtime_config["configurable"] = {}
        
        # Use provided thread_id or keep existing one
        if thread_id:
            runtime_config["configurable"]["thread_id"] = thread_id
        elif "thread_id" not in runtime_config["configurable"]:
            # Generate a new thread ID if none exists
            runtime_config["configurable"]["thread_id"] = str(uuid.uuid4())
        
        # Add other kwargs
        for key, value in kwargs.items():
            # If it's a configurable param, add to configurable section
            if key.startswith("configurable_"):
                param_name = key.replace("configurable_", "")
                runtime_config["configurable"][param_name] = value
            elif key == "configurable" and isinstance(value, dict):
                # Merge configurable section
                for k, v in value.items():
                    runtime_config["configurable"][k] = v
            else:
                # Otherwise add to top level
                runtime_config[key] = value
        
        return runtime_config
