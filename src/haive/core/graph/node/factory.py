# src/haive/core/graph/node/factory.py

from typing import Any, Dict, List, Optional, Callable, Union, Awaitable
import logging
import asyncio
import traceback
from datetime import datetime
from langgraph.types import Command, Send
from langgraph.graph import END

from haive.core.graph.node.config import NodeConfig
from haive.core.graph.node.registry import NodeTypeRegistry
from haive.core.registry.manager import RegistryManager
from haive.core.config.runnable import RunnableConfigManager
from haive.core.engine.base import Engine, EngineType

# Setup logging
logger = logging.getLogger(__name__)

class NodeFactory:
    """
    Factory for creating node functions with comprehensive engine support.
    
    Handles creation of node functions from different engine types and configurations,
    ensuring proper Command usage for control flow and engine ID targeting.
    """
    
    # Class-level registry reference
    _registry = None
    
    @classmethod
    def get_registry(cls) -> NodeTypeRegistry:
        """Get the node type registry."""
        if cls._registry is None:
            # Try to get from RegistryManager first
            try:
                manager = RegistryManager.get_instance()
                registry = manager.get_registry("node")
                if isinstance(registry, NodeTypeRegistry):
                    cls._registry = registry
                else:
                    cls._registry = NodeTypeRegistry.get_instance()
            except (ImportError, AttributeError):
                cls._registry = NodeTypeRegistry.get_instance()
                
            # Register processors if needed
            if not cls._registry.node_processors:
                cls._registry.register_default_processors()
                
        return cls._registry
    
    @classmethod
    def set_registry(cls, registry: Any) -> None:
        """Set the node type registry."""
        cls._registry = registry
    
    @classmethod
    def create_node_function(
        cls,
        config: Union[NodeConfig, Any],
        command_goto: Optional[str] = None,
        input_mapping: Optional[Dict[str, str]] = None,
        output_mapping: Optional[Dict[str, str]] = None,
        runnable_config: Optional[Dict[str, Any]] = None,
        debug: bool = False,
        async_mode: Optional[bool] = None
    ) -> Callable:
        """
        Create a node function with proper engine handling.
        
        Args:
            config: NodeConfig, Engine, or callable function
            command_goto: Optional next node to go to (ignored if NodeConfig provided)
            input_mapping: Optional mapping from state keys to engine input keys
            output_mapping: Optional mapping from engine output keys to state keys
            runnable_config: Optional default runtime configuration
            debug: Enable debug logging
            async_mode: Whether to create an asynchronous node function
            
        Returns:
            A node function compatible with LangGraph
        """
        # Enable verbose logging if debug mode
        if debug:
            logger.setLevel(logging.DEBUG)
            logger.debug("Debug mode enabled for node function creation")
        
        # Convert to NodeConfig if not already
        if not isinstance(config, NodeConfig):
            node_config = NodeConfig(
                name=getattr(config, "name", "unnamed_node"),
                engine=config,
                command_goto=command_goto,
                input_mapping=input_mapping,
                output_mapping=output_mapping,
                runnable_config=runnable_config,
                debug=debug,
                async_mode=async_mode  # May be None
            )
        else:
            node_config = config
            if debug:
                node_config.debug = True
            # Only override async_mode if explicitly set
            if async_mode is not None:
                node_config.async_mode = async_mode
                
        logger.debug(f"Creating node function for {node_config.name}, async_mode={node_config.async_mode}")
        
        # Get registry
        registry = cls.get_registry()
        node_config.registry = registry
            
        # Resolve engine reference if needed
        try:
            engine, engine_id = node_config.resolve_engine(registry)
            logger.debug(f"Resolved engine {type(engine).__name__}, id={engine_id}")
            
            # Auto-detect async_mode if not specified
            if node_config.async_mode is None:
                # Check if engine supports async operations
                has_ainvoke = hasattr(engine, "ainvoke") and callable(getattr(engine, "ainvoke"))
                is_async_func = asyncio.iscoroutinefunction(engine) if callable(engine) else False
                node_config.async_mode = has_ainvoke or is_async_func
                logger.debug(f"Auto-detected async_mode={node_config.async_mode}")
            
            # Ensure engine has been converted to a runnable when appropriate
            if isinstance(engine, Engine):
                # Create runnable in advance to detect any initialization issues
                logger.debug(f"Pre-creating runnable for engine {engine.__class__.__name__}")
                try:
                    runnable = engine.create_runnable(node_config.runnable_config)
                    logger.debug(f"Successfully created runnable: {type(runnable).__name__}")
                except Exception as e:
                    logger.error(f"Error creating runnable: {e}")
                    logger.error(traceback.format_exc())
            
        except Exception as e:
            logger.error(f"Error resolving engine: {e}")
            logger.error(traceback.format_exc())
            engine, engine_id = node_config.engine, None
        
        # Determine node type
        if node_config.async_mode:
            # For async mode, prefer async node types
            if hasattr(engine, "ainvoke") and callable(getattr(engine, "ainvoke")):
                node_type = "async_invokable"
                logger.debug(f"Using async_invokable processor type")
            elif asyncio.iscoroutinefunction(engine):
                node_type = "async"
                logger.debug(f"Using async processor type")
            else:
                # Fallback to regular node type detection
                node_type = node_config.determine_node_type()
                logger.debug(f"Fell back to node type: {node_type}")
        else:
            node_type = node_config.determine_node_type()
            logger.debug(f"Determined node type: {node_type}")
            
        logger.debug(f"Using node type: {node_type}")
        
        # Get processor for this node type
        processor = registry.get_node_processor(node_type)
        
        # If no processor found, try to find by capability
        if processor is None:
            logger.debug(f"No processor for {node_type}, searching by capability")
            processor = registry.find_processor_for_engine(engine)
            
        # If still no processor, use generic
        if processor is None:
            logger.debug("No processor found, using generic")
            processor = registry.get_node_processor("generic")
            if processor is None:
                raise ValueError(f"No processor found for {node_type}")
        
        # Create node function
        try:
            node_func = processor.create_node_function(engine, node_config)
            logger.debug(f"Created node function with processor {processor.__class__.__name__}")
        except Exception as e:
            logger.error(f"Error creating node function: {e}")
            logger.error(traceback.format_exc())
            
            # Create fallback error function
            def error_func(state, runtime_config=None):
                logger.error(f"Using fallback error function for {node_config.name}")
                error_data = {
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "traceback": traceback.format_exc()
                }
                
                if node_config.command_goto:
                    return Command(update={"error": error_data}, goto=node_config.command_goto)
                return {"error": error_data}
                
            node_func = error_func
        
        # Add debug wrapper if needed
        if node_config.debug:
            # Use appropriate debug wrapper for sync or async
            if node_config.async_mode:
                node_func = cls._create_async_debug_wrapper(node_func, node_config)
            else:
                node_func = cls._create_debug_wrapper(node_func, node_config)
        
        # Add metadata
        node_func.__node_config__ = node_config
        node_func.__engine_id__ = engine_id
        node_func.__node_type__ = node_type
        node_func.__async_mode__ = node_config.async_mode
        
        return node_func
    
    @staticmethod
    def _create_debug_wrapper(node_func, config):
        """Create a debug wrapper for synchronous node functions."""
        def debug_wrapper(state, runtime_config=None):
            logger = logging.getLogger(f"node.{config.name}")
            logger.debug(f"Node {config.name} called with state type: {type(state).__name__}")
            if isinstance(state, dict):
                logger.debug(f"State keys: {list(state.keys())}")
            
            try:
                # Simple pass-through to the node function
                result = node_func(state, runtime_config)
                return result
                    
            except Exception as e:
                logger.error(f"Error in node {config.name}: {e}")
                logger.error(traceback.format_exc())
                
                # Create error result
                error_data = {
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "timestamp": datetime.now().isoformat()
                }
                
                if config.command_goto:
                    return Command(update={"error": error_data}, goto=config.command_goto)
                return {"error": error_data}
                    
        return debug_wrapper

    @staticmethod
    def _create_async_debug_wrapper(node_func, config):
        """Create a debug wrapper for asynchronous node functions."""
        # We're going to create a SYNCHRONOUS function that handles async
        # internally - this is key for langgraph compatibility
        def async_wrapper(state, runtime_config=None):
            logger = logging.getLogger(f"node.{config.name}")
            logger.debug(f"Async node {config.name} called with state type: {type(state).__name__}")
            if isinstance(state, dict):
                logger.debug(f"State keys: {list(state.keys())}")
            
            try:
                # Call the node function directly - should internally handle async
                result = node_func(state, runtime_config)
                
                # No awaiting needed - node_func should handle that internally
                return result
                    
            except Exception as e:
                logger.error(f"Error in async node {config.name}: {e}")
                logger.error(traceback.format_exc())
                
                # Create error result
                error_data = {
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "timestamp": datetime.now().isoformat()
                }
                
                if config.command_goto:
                    return Command(update={"error": error_data}, goto=config.command_goto)
                return {"error": error_data}
                    
        return async_wrapper
    
    @classmethod
    def create_mapping_node(cls, 
                          item_provider: str,
                          target_node: str,
                          item_key: str = "item",
                          name: str = "mapping_node") -> Callable:
        """
        Create a node function that maps items to parallel processing.
        
        Args:
            item_provider: State key containing the items list
            target_node: Target node to send items to
            item_key: Key to use for each item in the target
            name: Name for the node
            
        Returns:
            Node function that creates Send objects
        """
        logger.debug(f"Creating mapping node: {name}")
        logger.debug(f"Items from: {item_provider}, Target: {target_node}, Item key: {item_key}")
        
        # For mapping nodes, we use a synchronous function that returns Send objects
        def map_items(state):
            """Map items from the state to Send objects."""
            logger.debug(f"Mapping node called with state type: {type(state).__name__}")
            
            # Extract items from state
            if isinstance(state, dict):
                items = state.get(item_provider, [])
            elif hasattr(state, "get"):
                items = state.get(item_provider, [])
            elif hasattr(state, item_provider):
                items = getattr(state, item_provider)
            else:
                items = []
                
            logger.debug(f"Found {len(items)} items to map")
            
            if not items:
                logger.debug("No items to map, returning empty list")
                return []
                
            # Create Send objects for each item
            result = [Send(target_node, {item_key: item}) for item in items]
            logger.debug(f"Created {len(result)} Send objects")
            return result
        
        # Mark as mapping node for detection
        map_items.__mapping_node__ = True
        
        # Create node config
        node_config = NodeConfig(
            name=name,
            engine=map_items,
            node_type="mapping",
            async_mode=False  # Use synchronous mapping
        )
        
        # Create the node function
        return cls.create_node_function(node_config)

    @classmethod
    def create_conditional_node(cls,
                            condition_func: Callable,
                            routes: Dict[Any, str],
                            default_route: Optional[str] = None,
                            name: str = "conditional_node") -> Callable:
        """
        Create a node function that routes based on a condition.
        
        Args:
            condition_func: Function that evaluates state and returns a key
            routes: Mapping from condition values to node names
            default_route: Default route if no match (or condition_func returns None)
            name: Name for the node
            
        Returns:
            Node function that routes based on condition
        """
        logger.debug(f"Creating conditional node: {name}")
        logger.debug(f"Routes: {routes}, Default: {default_route}")
        
        def route_by_condition(state):
            """Route based on condition function result."""
            logger.debug(f"Condition node called with state type: {type(state).__name__}")
            
            try:
                # Evaluate the condition
                result = condition_func(state)
                logger.debug(f"Condition function returned: {result}")
                
                # Find matching route
                if result in routes:
                    logger.debug(f"Found matching route: {routes[result]}")
                    return Command(goto=routes[result])
                
                # Fall back to default
                if default_route:
                    logger.debug(f"Using default route: {default_route}")
                    return Command(goto=default_route)
                    
                # No route found
                logger.warning(f"No route found for condition result: {result}")
                return {"error": f"No route found for condition result: {result}"}
            except Exception as e:
                logger.error(f"Error in condition function: {str(e)}")
                logger.error(traceback.format_exc())
                return {"error": f"Error in condition function: {str(e)}"}
        
        # Create node config
        node_config = NodeConfig(
            name=name,
            engine=route_by_condition
        )
        
        # Create the node function
        return cls.create_node_function(node_config)

    @classmethod
    def create_error_handler_node(cls,
                                fallback_node: str = "END",
                                name: str = "error_handler") -> Callable:
        """
        Create a node function that handles errors.
        
        Args:
            fallback_node: Node to route to after handling error
            name: Name for the node
            
        Returns:
            Node function that handles errors
        """
        logger.debug(f"Creating error handler node: {name}")
        logger.debug(f"Fallback node: {fallback_node}")
        
        def handle_error(state):
            """Handle error state and route to fallback."""
            logger.debug(f"Error handler called with state type: {type(state).__name__}")
            
            # Mark as handled
            result = state.copy() if isinstance(state, dict) else {"state": state}
            result["error_handled"] = True
            logger.debug(f"Marked error as handled")
            
            # Route to fallback
            return Command(update=result, goto=fallback_node)
        
        # Create node config
        node_config = NodeConfig(
            name=name,
            engine=handle_error
        )
        
        # Create the node function
        return cls.create_node_function(node_config)