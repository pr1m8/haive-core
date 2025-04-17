# src/haive/core/graph/NodeFactory.py

from typing import Dict, Any, Optional, Union, Callable, Type, List, Tuple, Protocol, runtime_checkable
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import BaseMessage
from langgraph.graph import END
from langgraph.types import Command, Send
import inspect
import logging
from functools import wraps
from typing import Literal
from haive_core.engine.base import Engine, InvokableEngine, NonInvokableEngine, EngineType
from haive_core.graph.node.config import NodeConfig
from haive_core.config.runnable import RunnableConfigManager

logger = logging.getLogger(__name__)

@runtime_checkable
class NodeFunction(Protocol):
    """Protocol for node functions."""
    def __call__(self, state: Any, config: Optional[Dict[str, Any]] = None) -> Any:
        ...

class NodeFactoryResult:
    """Result of creating a node function."""
    def __init__(
        self,
        function: Callable,
        config: NodeConfig,
        engine_id: Optional[str] = None
    ):
        self.function = function
        self.config = config
        self.engine_id = engine_id

class NodeFactory:
    """
    Factory for creating node functions with comprehensive engine support.
    
    Handles creation of node functions from different engine types and configurations,
    ensuring proper Command usage for control flow and engine ID targeting.
    """
    
    @classmethod
    def create_node_function(
        cls,
        config: Union[NodeConfig, Engine, Callable],
        command_goto: Optional[str] = None,
        input_mapping: Optional[Dict[str, str]] = None,
        output_mapping: Optional[Dict[str, str]] = None,
        runnable_config: Optional[RunnableConfig] = None
    ) -> Callable:
        """
        Create a node function with proper engine handling.
        
        Args:
            config: NodeConfig, Engine, or callable function
            command_goto: Optional next node to go to (ignored if NodeConfig provided)
            input_mapping: Optional mapping from state keys to engine input keys (ignored if NodeConfig provided)
            output_mapping: Optional mapping from engine output keys to state keys (ignored if NodeConfig provided)
            runnable_config: Optional default runtime configuration (ignored if NodeConfig provided)
            
        Returns:
            A node function compatible with LangGraph
        """
        # Convert to NodeConfig if not already
        if not isinstance(config, NodeConfig):
            node_config = NodeConfig(
                name=getattr(config, "name", "unnamed_node"),
                engine=config,
                command_goto=command_goto,
                input_mapping=input_mapping,
                output_mapping=output_mapping,
                runnable_config=runnable_config
            )
        else:
            node_config = config
            
        # Resolve engine reference if needed
        engine, engine_id = node_config.resolve_engine()
        
        # Create appropriate node function based on engine type
        if isinstance(engine, InvokableEngine):
            node_func = cls._create_invokable_engine_node(engine, node_config)
        elif isinstance(engine, NonInvokableEngine):
            node_func = cls._create_non_invokable_engine_node(engine, node_config)
        elif callable(engine):
            node_func = cls._ensure_command_wrapper(engine, node_config)
        else:
            logger.warning(f"Unknown engine type: {type(engine)}. Creating generic node.")
            node_func = cls._create_generic_node(engine, node_config)
        
        # Add metadata to the function for serialization and node tracking
        node_func.__node_config__ = node_config
        node_func.__engine_id__ = engine_id
        
        return node_func
    
    @classmethod
    def _create_invokable_engine_node(cls, engine: InvokableEngine, config: NodeConfig) -> NodeFunction:
        """
        Create a node function for an invokable engine.
        
        Args:
            engine: The invokable engine
            config: Node configuration
            
        Returns:
            Node function
        """
        def node_function(state, runtime_config=None):
            """Node function that uses engine's invoke method."""
            # Extract input from state
            input_data = cls._extract_input(state, config.input_mapping)
            
            # Merge runtime configs
            merged_config = cls._merge_configs(config.runnable_config, runtime_config)
            
            # Apply engine ID targeting
            if engine_id := getattr(engine, "id", None):
                cls._ensure_engine_id_targeting(merged_config, engine_id)
                
            # Apply node-specific config overrides
            if config.config_overrides and merged_config:
                cls._apply_config_overrides(merged_config, engine_id, config.config_overrides)
            
            try:
                # Invoke the engine
                result = engine.invoke(input_data, merged_config)
                
                # Handle different result types for Command/Send pattern
                return cls._handle_result(result, config.command_goto, config.output_mapping)
            except Exception as e:
                logger.error(f"Error invoking engine {engine.name}: {e}")
                return Command(update={"error": str(e)}, goto=config.command_goto)
        
        return node_function
    
    @classmethod
    def _create_non_invokable_engine_node(cls, engine: NonInvokableEngine, config: NodeConfig) -> NodeFunction:
        """
        Create a node function for a non-invokable engine.
        
        Args:
            engine: The non-invokable engine
            config: Node configuration
            
        Returns:
            Node function
        """
        def node_function(state, runtime_config=None):
            """Node function that instantiates the engine."""
            # Extract input from state
            input_data = cls._extract_input(state, config.input_mapping)
            
            # Merge runtime configs
            merged_config = cls._merge_configs(config.runnable_config, runtime_config)
            
            # Apply engine ID targeting
            if engine_id := getattr(engine, "id", None):
                cls._ensure_engine_id_targeting(merged_config, engine_id)
                
            # Apply node-specific config overrides
            if config.config_overrides and merged_config:
                cls._apply_config_overrides(merged_config, engine_id, config.config_overrides)
            
            try:
                # Instantiate the engine
                instance = engine.instantiate(merged_config)
                
                # Return the instance with Command for control flow
                return Command(update={"instance": instance}, goto=config.command_goto)
            except Exception as e:
                logger.error(f"Error instantiating engine {engine.name}: {e}")
                return Command(update={"error": str(e)}, goto=config.command_goto)
        
        return node_function
    
    @classmethod
    def _ensure_command_wrapper(cls, func: Callable, config: NodeConfig) -> NodeFunction:
        """
        Ensure a callable returns a Command object.
        
        Args:
            func: The callable function
            config: Node configuration
            
        Returns:
            Wrapped function that ensures Command return
        """
        # Check if function already accepts config
        sig = inspect.signature(func)
        accepts_config = "config" in sig.parameters
        
        @wraps(func)
        def wrapper(state, runtime_config=None):
            """Wrapper that ensures Command output."""
            try:
                # Call original function with appropriate arguments
                if accepts_config:
                    # Merge configs
                    merged_config = cls._merge_configs(config.runnable_config, runtime_config)
                    result = func(state, merged_config)
                else:
                    # Just pass state
                    result = func(state)
                
                # Handle different result types for Command/Send pattern
                return cls._handle_result(result, config.command_goto, config.output_mapping)
            except Exception as e:
                logger.error(f"Error in node function: {e}")
                return Command(update={"error": str(e)}, goto=config.command_goto)
        
        return wrapper
    
    @classmethod
    def _create_generic_node(cls, obj: Any, config: NodeConfig) -> NodeFunction:
        """
        Create a generic node function for any object.
        
        Args:
            obj: Any object
            config: Node configuration
            
        Returns:
            Node function
        """
        def node_function(state, runtime_config=None):
            """Generic node function that uses object as-is."""
            # For input/output objects, try to use them if callable
            if hasattr(obj, "__call__"):
                try:
                    if "state" in inspect.signature(obj.__call__).parameters:
                        result = obj(state)
                        return cls._handle_result(result, config.command_goto, config.output_mapping)
                except Exception:
                    pass
            
            # Otherwise, just pass through the object
            return Command(update={"result": obj}, goto=config.command_goto)
        
        return node_function
    
    @classmethod
    def _extract_input(cls, state: Dict[str, Any], input_mapping: Optional[Dict[str, str]]) -> Any:
        """Extract input from state based on mapping, excluding runnable_config."""
        # Filter out runnable_config initially, unless it's the ONLY key mapped
        filtered_state = {k: v for k, v in state.items() if k != "runnable_config"}

        if not input_mapping:
            # If no mapping, return the state without runnable_config
            return filtered_state

        # If mapping exists, process it
        mapped_input = {}
        for state_key, input_key in input_mapping.items():
            if state_key == "runnable_config":
                # If runnable_config is explicitly mapped, include it
                if state_key in state:
                     mapped_input[input_key] = state[state_key]
            elif state_key in filtered_state: # Check against the filtered state
                mapped_input[input_key] = filtered_state[state_key]

        # If only one key was mapped, return the value directly
        # Check original mapping length to decide this, not mapped_input length
        if len(input_mapping) == 1 and len(mapped_input) == 1:
            return list(mapped_input.values())[0]
        
        # Return the dictionary of mapped inputs
        return mapped_input
    
    @classmethod
    def _handle_result(
        cls, 
        result: Any, 
        command_goto: Optional[Union[str, Literal["END"], Send, List[Union[Send, str]]]], 
        output_mapping: Optional[Dict[str, str]]
    ) -> Any:
        """
        Handle different result types to ensure proper Command/Send pattern support.
        
        Args:
            result: Result from engine or function
            command_goto: Optional goto directive
            output_mapping: Optional output mapping
            
        Returns:
            Result wrapped in Command if needed
        """
        # Handle already Command/Send results
        if isinstance(result, Command):
            # If already Command, ensure goto is set if not already
            if result.goto is None and command_goto is not None:
                return Command(
                    update=result.update, 
                    goto=command_goto,
                    resume=result.resume,
                    graph=result.graph
                )
            return result
        elif isinstance(result, Send):
            # Return Send as-is
            return result
        elif isinstance(result, list) and all(isinstance(item, Send) for item in result):
            # Return list of Send objects as-is
            return result
        else:
            # Process output if it's not already a Command or Send
            processed_output = cls._process_output(result, output_mapping)
            
            # Wrap in Command if goto is specified
            if command_goto is not None:
                return Command(update=processed_output, goto=command_goto)
            else:
                return processed_output
    
    @classmethod
    def _process_output(cls, output: Any, output_mapping: Optional[Dict[str, str]]) -> Dict[str, Any]:
        """
        Process output according to mapping.
        
        Args:
            output: Output from engine or function
            output_mapping: Mapping from output keys to state keys
            
        Returns:
            Processed output
        """
        # Handle non-dict output
        if not isinstance(output, dict):
            return {"result": output}
            
        # Return as-is if no mapping
        if not output_mapping:
            return output
            
        # Apply mapping
        result = {}
        for output_key, state_key in output_mapping.items():
            if output_key in output:
                result[state_key] = output[output_key]
                
        # Return original output if no mapped keys were found
        return result if result else output
    
    @classmethod
    def _merge_configs(cls, base_config: Optional[RunnableConfig], override_config: Optional[RunnableConfig]) -> Optional[RunnableConfig]:
        """
        Merge two RunnableConfigs.
        
        Args:
            base_config: Base configuration
            override_config: Configuration to override with
            
        Returns:
            Merged configuration or None if both inputs are None
        """
        # Handle None cases
        if base_config is None and override_config is None:
            return None
        elif base_config is None:
            return override_config
        elif override_config is None:
            return base_config
        
        # Use RunnableConfigManager for merging
        return RunnableConfigManager.merge(base_config, override_config)
    
    @classmethod
    def _ensure_engine_id_targeting(cls, config: Optional[Dict[str, Any]], engine_id: str) -> Dict[str, Any]:
        """
        Ensure a config includes targeting for a specific engine ID.
        
        Args:
            config: Config dictionary (may be None)
            engine_id: Engine ID to target
            
        Returns:
            Updated config
        """
        if not config:
            config = {}
            
        if "configurable" not in config:
            config["configurable"] = {}
            
        if "engine_configs" not in config["configurable"]:
            config["configurable"]["engine_configs"] = {}
            
        if engine_id not in config["configurable"]["engine_configs"]:
            config["configurable"]["engine_configs"][engine_id] = {}
            
        return config
    
    @classmethod
    def _apply_config_overrides(cls, config: Dict[str, Any], engine_id: Optional[str], overrides: Dict[str, Any]) -> None:
        """
        Apply node-specific configuration overrides to the config.
        
        Args:
            config: Config dictionary to modify
            engine_id: Optional engine ID to target
            overrides: Configuration overrides to apply
        """
        if not engine_id:
            # Apply to top-level configurable if no engine ID
            for key, value in overrides.items():
                config["configurable"][key] = value
        else:
            # Apply to engine-specific section
            if engine_id not in config["configurable"]["engine_configs"]:
                config["configurable"]["engine_configs"][engine_id] = {}
                
            config["configurable"]["engine_configs"][engine_id].update(overrides)