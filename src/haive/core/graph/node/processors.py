# src/haive/core/graph/node/processors.py

import asyncio
import inspect
import logging
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable
from pydantic import BaseModel
from langchain_core.messages import BaseMessage
from langgraph.types import Command, Send
from haive.core.engine.base import EngineType
from haive.core.config.runnable import RunnableConfigManager
from haive.core.graph.node.config import NodeConfig
from haive.core.graph.node.registry import register_node_processor
from haive.core.graph.node.protocols import NodeProcessor

# Setup detailed logging
logger = logging.getLogger(__name__)

# Common utility functions used by processors
def process_state(state: Any) -> Dict[str, Any]:
    """Process state into a standardized dictionary format."""
    logger.debug(f"Processing state of type: {type(state).__name__}")
    
    # Handle different state types
    if isinstance(state, dict):
        return state.copy()  # Make a copy to avoid modifying the original
    elif hasattr(state, "model_dump"):  # Pydantic v2
        return state.model_dump()
    elif hasattr(state, "dict"):  # Pydantic v1
        return state.dict()
    elif hasattr(state, "__dict__"):  # Object with __dict__
        # Filter out private attributes
        return {k: v for k, v in state.__dict__.items() 
                if not k.startswith('_')}
    else:
        # Unknown state type - wrap as value
        logger.debug(f"Unknown state type {type(state).__name__}, wrapping as 'value'")
        return {"value": state}

def merge_configs(base_config: Optional[Dict[str, Any]], override_config: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Merge two configs with smart handling."""
    logger.debug(f"Merging configs: base={base_config is not None}, override={override_config is not None}")
    
    # Handle None cases
    if base_config is None and override_config is None:
        return None
    elif base_config is None:
        return override_config
    elif override_config is None:
        return base_config
    
    # Use RunnableConfigManager for proper merging
    merged = RunnableConfigManager.merge(base_config, override_config)
    logger.debug(f"Merged config keys: {list(merged.keys() if isinstance(merged, dict) else [])}")
    return merged

def ensure_engine_id_targeting(config: Dict[str, Any], engine_id: str) -> Dict[str, Any]:
    """Ensure config includes targeting for a specific engine ID."""
    logger.debug(f"Ensuring engine ID targeting for: {engine_id}")
    
    if not config:
        config = {}
        
    if "configurable" not in config:
        config["configurable"] = {}
        
    if "engine_configs" not in config["configurable"]:
        config["configurable"]["engine_configs"] = {}
        
    if engine_id not in config["configurable"]["engine_configs"]:
        config["configurable"]["engine_configs"][engine_id] = {}
        
    return config

def apply_config_overrides(config: Dict[str, Any], engine_id: Optional[str], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Apply node-specific configuration overrides."""
    logger.debug(f"Applying config overrides: {list(overrides.keys())}")
    
    # Make a copy to avoid modifying the original
    config = config.copy()
    
    if "configurable" not in config:
        config["configurable"] = {}
    
    if engine_id:
        # Target engine-specific config
        logger.debug(f"Targeting engine ID: {engine_id}")
        if "engine_configs" not in config["configurable"]:
            config["configurable"]["engine_configs"] = {}
            
        if engine_id not in config["configurable"]["engine_configs"]:
            config["configurable"]["engine_configs"][engine_id] = {}
            
        # Apply overrides to this engine's config
        for key, value in overrides.items():
            config["configurable"]["engine_configs"][engine_id][key] = value
    else:
        # Apply to top-level configurable
        logger.debug("Applying overrides to top-level configurable")
        for key, value in overrides.items():
            config["configurable"][key] = value
            
    return config
def extract_input(state: Dict[str, Any], config: NodeConfig) -> Any:
    """
    Extract input based on configuration with improved debugging.
    
    Args:
        state: The current state
        config: Node configuration
        
    Returns:
        Extracted input data
    """
    logger = logging.getLogger("input_extraction")
    logger.debug(f"Extracting input for node: {config.name}")
    logger.debug(f"State keys: {list(state.keys())}")
    logger.debug(f"Input mapping: {config.input_mapping}")
    logger.debug(f"Use direct messages: {config.use_direct_messages}")
    
    # Ensure state is a dictionary
    if not isinstance(state, dict):
        logger.debug(f"Converting state of type {type(state).__name__} to dictionary")
        if hasattr(state, "model_dump"):
            state = state.model_dump()
        elif hasattr(state, "dict"):
            state = state.dict()
        else:
            state = {"value": state}
        logger.debug(f"Converted state keys: {list(state.keys())}")
    
    # Apply mapping if it exists
    if config.input_mapping:
        logger.debug(f"Applying input mapping: {config.input_mapping}")
        mapped_input = {}
        for state_key, input_key in config.input_mapping.items():
            if state_key in state:
                mapped_input[input_key] = state[state_key]
                logger.debug(f"Mapped {state_key} → {input_key}: {type(state[state_key]).__name__}")
            else:
                logger.warning(f"State key '{state_key}' not found in state")
        
        # Return mapped dict if we have any values
        if mapped_input:
            logger.debug(f"Returning mapped input with keys: {list(mapped_input.keys())}")
            return mapped_input
        
        # If mapping exists but resulted in empty dict, log warning
        if not mapped_input:
            logger.warning(f"Mapping resulted in empty input dictionary")
    
    # If using direct messages and they exist, return them
    if config.use_direct_messages and "messages" in state:
        logger.debug(f"Using direct messages (count: {len(state['messages'])})")
        return state["messages"]
    
    # If we get here with a mapping but no matches, log a clear error
    if config.input_mapping and len(config.input_mapping) > 0:
        msg = f"No input fields could be extracted using mapping: {config.input_mapping}"
        logger.error(msg)
        
        # Check for common errors - missing state keys
        missing_keys = [k for k in config.input_mapping.keys() if k not in state]
        if missing_keys:
            logger.error(f"State is missing these keys defined in mapping: {missing_keys}")
            logger.error(f"Available state keys: {list(state.keys())}")
    
    # Default to returning full state
    logger.debug(f"Returning full state with keys: {list(state.keys())}")
    return state

def process_output(result: Any, config: NodeConfig, original_state: Dict[str, Any]) -> Dict[str, Any]:
    """Process output according to configuration."""
    logger = logging.getLogger("output_processing")
    logger.debug(f"Processing output for node: {config.name}")
    logger.debug(f"Result type: {type(result).__name__}")
    logger.debug(f"Output mapping: {config.output_mapping}")
    logger.debug(f"Preserve state: {config.preserve_state}")
    
    # Get output processor from registry
    registry = config.registry
    if registry:
        # Use structured processor for BaseModel results, otherwise standard
        processor_type = "structured" if isinstance(result, BaseModel) else "standard"
        processor = registry.get_output_processor(processor_type)
        if processor:
            logger.debug(f"Using processor: {processor_type}")
            processed = processor.process_output(result, config, original_state)
            logger.debug(f"Processed output: {processed}")
            return processed
    
    # Fallback implementation if no processor or registry
    logger.debug(f"Using fallback implementation")
    updates = {}
    
    # Start with original state if preserving
    if config.preserve_state:
        logger.debug(f"Preserving original state")
        updates = original_state.copy()
    
    # Handle different result types
    if isinstance(result, dict):
        logger.debug(f"Result is dictionary with keys: {list(result.keys())}")
        
        # Apply output mapping if exists
        if config.output_mapping:
            logger.debug(f"Applying output mapping")
            for output_key, state_key in config.output_mapping.items():
                if output_key in result:
                    updates[state_key] = result[output_key]
                    logger.debug(f"Mapped {output_key} → {state_key}")
                else:
                    logger.warning(f"Output key '{output_key}' not found in result")
        else:
            # No mapping - update with all result keys
            logger.debug(f"No mapping - updating with all result keys")
            updates.update(result)
    elif isinstance(result, BaseModel):
        logger.debug(f"Result is BaseModel: {result.__class__.__name__}")
        
        # Convert to dict
        if hasattr(result, "model_dump"):
            result_dict = result.model_dump()
        else:
            result_dict = result.dict()
            
        logger.debug(f"Model keys: {list(result_dict.keys())}")
            
        # Apply output mapping if exists
        if config.output_mapping:
            logger.debug(f"Applying output mapping")
            for output_key, state_key in config.output_mapping.items():
                if output_key in result_dict:
                    updates[state_key] = result_dict[output_key]
                    logger.debug(f"Mapped {output_key} → {state_key}")
                else:
                    logger.warning(f"Output key '{output_key}' not found in model")
        else:
            # No mapping - add the model under its class name
            model_name = result.__class__.__name__.lower()
            updates[model_name] = result
            logger.debug(f"Added model as {model_name}")
    elif isinstance(result, BaseMessage):
        logger.debug(f"Result is BaseMessage: {result.__class__.__name__}")
        
        # If we have messages field in state, append the message
        if "messages" in updates and isinstance(updates["messages"], list):
            updates["messages"].append(result)
            logger.debug(f"Added message to existing messages list (now {len(updates['messages'])})")
        else:
            # Create a new messages list
            updates["messages"] = [result]
            logger.debug(f"Created new messages list with message")
        
        # Extract content if needed
        if config.extract_content:
            updates["content"] = result.content
            logger.debug(f"Extracted content from message")
    else:
        # Non-dict, non-model result - store as result
        logger.debug(f"Result is not dict or model, storing as 'result'")
        updates["result"] = result
    
    logger.debug(f"Final updates: {updates}")
    return updates

def handle_command_pattern(result: Any, config: NodeConfig) -> Any:
    """Handle Command/Send pattern for results."""
    logger = logging.getLogger("CommandHandler")
    logger.debug(f"Handling command pattern for result type: {type(result).__name__}")
    
    # Check if result is already a Command or Send
    if isinstance(result, Command):
        logger.debug(f"Result is already a Command: goto={result.goto}")
        # Only override goto if not set but config has one
        if result.goto is None and config.command_goto is not None:
            logger.debug(f"Overriding Command goto: {config.command_goto}")
            
            # Get update data safely
            if hasattr(result, "update"):
                # Check if update is callable
                if callable(result.update):
                    try:
                        # Call update() to get actual data
                        update_data = result.update()
                        logger.debug(f"Called callable update, got type: {type(update_data).__name__}")
                    except Exception as e:
                        logger.error(f"Error calling update(): {e}")
                        update_data = {"error": str(e)}
                else:
                    # Use as attribute
                    update_data = result.update
                    logger.debug(f"Used update attribute, type: {type(update_data).__name__}")
            else:
                update_data = {}
                logger.debug("No update data found, using empty dict")
                
            # Create new Command with correct goto
            new_command = Command(
                update=update_data,
                goto=config.command_goto,
                resume=getattr(result, "resume", None),
                graph=getattr(result, "graph", None)
            )
            logger.debug(f"Created new Command: {new_command}")
            return new_command
        return result
    
    # Handle Send objects
    elif isinstance(result, Send):
        logger.debug(f"Result is a Send object: node={result.node}")
        return result
    elif isinstance(result, list) and all(isinstance(item, Send) for item in result):
        logger.debug(f"Result is a list of Send objects: count={len(result)}")
        return result
    
    # Not Command/Send - apply command_goto if specified
    if config.command_goto is not None:
        logger.debug(f"Creating Command with goto: {config.command_goto}")
        
        # Ensure result is dict-like
        if not isinstance(result, dict) and hasattr(result, "model_dump"):
            result_dict = result.model_dump()
            logger.debug(f"Converted BaseModel to dict: {list(result_dict.keys())}")
        elif not isinstance(result, dict) and hasattr(result, "dict"):
            result_dict = result.dict()
            logger.debug(f"Converted dict-like to dict: {list(result_dict.keys())}")
        elif not isinstance(result, dict):
            result_dict = {"result": result}
            logger.debug(f"Wrapped non-dict in 'result' key")
        else:
            result_dict = result
            logger.debug(f"Using result as-is: {list(result_dict.keys())}")
            
        new_command = Command(update=result_dict, goto=config.command_goto)
        logger.debug(f"Created Command: {new_command}")
        return new_command
    
    # Return as-is
    logger.debug(f"Returning result as-is: {type(result).__name__}")
    return result

def create_error_result(e: Exception, config: NodeConfig) -> Any:
    """Create standardized error result."""
    logger = logging.getLogger("error_handling")
    logger.debug(f"Creating error result for: {type(e).__name__}: {str(e)}")
    
    # Create error data
    error_data = {
        "error": str(e),
        "error_type": type(e).__name__,
        "timestamp": datetime.now().isoformat()
    }
    
    # Add traceback if debugging
    if config.debug:
        error_data["traceback"] = traceback.format_exc()
        logger.debug(f"Added traceback to error data")
    
    # Apply Command pattern if needed
    if config.command_goto is not None:
        logger.debug(f"Creating Command with error and goto: {config.command_goto}")
        return Command(update={"error": error_data}, goto=config.command_goto)
    
    # Return as dict
    logger.debug(f"Returning error data as dict")
    return {"error": error_data}

# Node processor implementations

@register_node_processor("invokable")
class InvokableNodeProcessor:
    """Processor for invokable engines."""
    
    def can_process(self, engine: Any) -> bool:
        """Check if this processor can handle the engine."""
        can_invoke = hasattr(engine, "invoke") and callable(getattr(engine, "invoke"))
        logger.debug(f"InvokableNodeProcessor.can_process: {can_invoke}")
        return can_invoke
    
    def create_node_function(self, engine: Any, config: NodeConfig) -> Callable:
        """Create a node function for an invokable engine."""
        logger.debug(f"Creating node function for invokable engine: {config.name}")
        
        def node_function(state, runtime_config=None):
            """Node function for invokable engines."""
            try:
                # Process state
                processed_state = process_state(state)
                
                # Merge configs
                merged_config = merge_configs(config.runnable_config, runtime_config)
                
                # Apply engine ID targeting
                if engine_id := getattr(engine, "id", None):
                    merged_config = ensure_engine_id_targeting(merged_config, engine_id)
                
                # Apply config overrides
                if config.config_overrides and merged_config:
                    merged_config = apply_config_overrides(
                        merged_config, engine_id, config.config_overrides)
                
                # Extract input based on mapping
                input_data = extract_input(processed_state, config)
                
                # Invoke engine
                logger.debug(f"Invoking engine: {engine.__class__.__name__}")
                result = engine.invoke(input_data, merged_config)
                logger.debug(f"Engine returned result of type: {type(result).__name__}")
                
                # Process output
                processed_output = process_output(result, config, processed_state)
                
                # Handle command pattern
                return handle_command_pattern(processed_output, config)
                
            except Exception as e:
                logger.error(f"Error in invokable node {config.name}: {str(e)}")
                logger.error(traceback.format_exc())
                
                # Create error result
                return create_error_result(e, config)
                
        return node_function

@register_node_processor("async_invokable")
class AsyncInvokableNodeProcessor:
    """Processor for async invokable engines."""
    
    def can_process(self, engine: Any) -> bool:
        """Check if this processor can handle the engine."""
        can_ainvoke = hasattr(engine, "ainvoke") and callable(getattr(engine, "ainvoke"))
        logger.debug(f"AsyncInvokableNodeProcessor.can_process: {can_ainvoke}")
        return can_ainvoke
    
    def create_node_function(self, engine: Any, config: NodeConfig) -> Callable:
        """Create a node function for an async invokable engine."""
        logger.debug(f"Creating async node function for engine: {config.name}")
        
        def node_function(state, runtime_config=None):
            """Node function that internally handles async execution."""
            logger.debug(f"Called async invokable node: {config.name}")
            
            try:
                # Process state
                processed_state = process_state(state)
                
                # Merge configs
                merged_config = merge_configs(config.runnable_config, runtime_config)
                
                # Apply engine ID targeting
                if engine_id := getattr(engine, "id", None):
                    merged_config = ensure_engine_id_targeting(merged_config, engine_id)
                
                # Apply config overrides
                if config.config_overrides and merged_config:
                    merged_config = apply_config_overrides(
                        merged_config, engine_id, config.config_overrides)
                logger.debug(f"Processed state: {processed_state}")
                # Extract input based on mapping
                input_data = extract_input(processed_state, config)
                logger.debug(f"Input data: {input_data}")
                print(f"Input data: {input_data}")  
                print(f"Processed state: {processed_state}")
                # For async invokable engines, we'll use the synchronous invoke method
                # instead of trying to run ainvoke in a sync context
                logger.debug(f"Using invoke() instead of ainvoke() for async engine")
                result = engine.invoke(input_data, merged_config)
                logger.debug(f"Engine invoke() returned: {type(result).__name__}")
                
                # Process output
                processed_output = process_output(result, config, processed_state)
                
                # Handle command pattern
                return handle_command_pattern(processed_output, config)
                
            except Exception as e:
                logger.error(f"Error in async invokable node {config.name}: {str(e)}")
                logger.error(traceback.format_exc())
                
                # Create error result
                return create_error_result(e, config)
                
        return node_function

@register_node_processor("callable")
class CallableNodeProcessor:
    """Processor for callable functions."""
    
    def can_process(self, engine: Any) -> bool:
        """Check if this processor can handle the engine."""
        is_callable = callable(engine) and not asyncio.iscoroutinefunction(engine)
        logger.debug(f"CallableNodeProcessor.can_process: {is_callable}")
        return is_callable
    
    def create_node_function(self, engine: Any, config: NodeConfig) -> Callable:
        """Create a node function from a callable function."""
        logger.debug(f"Creating callable node function: {config.name}")
        
        def node_function(state, runtime_config=None):
            """Node function for callable."""
            logger.debug(f"Called callable node: {config.name}")
            
            try:
                # Process state
                processed_state = process_state(state)
                
                # Merge configs
                merged_config = merge_configs(config.runnable_config, runtime_config)
                
                # Detect function signature
                sig = inspect.signature(engine)
                accepts_config = "config" in sig.parameters
                logger.debug(f"Function accepts config: {accepts_config}")
                
                # Call function with appropriate arguments
                if accepts_config:
                    logger.debug(f"Calling with state and config")
                    result = engine(processed_state, merged_config)
                else:
                    logger.debug(f"Calling with state only")
                    logger.debug(f"Processed state: {processed_state}")
                    logger.debug(f"Merged config: {merged_config}")
                    logger.debug(f"Engine: {engine}")
                    logger.debug(f"Engine type: {type(engine)}")    
                    logger.debug(f"Engine signature: {sig}")
                    result = engine(processed_state)
                    
                logger.debug(f"Function returned: {type(result).__name__}")
                
                # For callable functions, we assume they might directly return Command/Send
                # so we don't process the output further unless it's a dict
                if isinstance(result, (Command, Send)) or (
                        isinstance(result, list) and all(isinstance(x, Send) for x in result)):
                    logger.debug(f"Returning Command/Send directly")
                    return result
                
                # Process output for other result types
                processed_output = process_output(result, config, processed_state)
                
                # Handle command pattern
                return handle_command_pattern(processed_output, config)
                
            except Exception as e:
                logger.error(f"Error in callable node {config.name}: {str(e)}")
                logger.error(traceback.format_exc())
                
                # Create error result
                return create_error_result(e, config)
                
        return node_function

@register_node_processor("async")
class AsyncNodeProcessor:
    """Processor for async functions."""
    
    def can_process(self, engine: Any) -> bool:
        """Check if this processor can handle the engine."""
        is_async = asyncio.iscoroutinefunction(engine)
        logger.debug(f"AsyncNodeProcessor.can_process: {is_async}")
        return is_async
    
    def create_node_function(self, engine: Any, config: NodeConfig) -> Callable:
        """Create a node function for an async function."""
        logger.debug(f"Creating async function node: {config.name}")
        
        def node_function(state, runtime_config=None):
            """Node function for async callable."""
            logger.debug(f"Called async node: {config.name}")
            
            try:
                # Process state
                processed_state = process_state(state)
                
                # Merge configs
                merged_config = merge_configs(config.runnable_config, runtime_config)
                
                # Detect function signature
                sig = inspect.signature(engine)
                accepts_config = "config" in sig.parameters
                logger.debug(f"Async function accepts config: {accepts_config}")
                
                # Setup asyncio event loop
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    logger.debug(f"Creating new event loop")
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                # Call function with appropriate arguments and run in event loop
                if accepts_config:
                    logger.debug(f"Calling async function with state and config")
                    coro = engine(processed_state, merged_config)
                else:
                    logger.debug(f"Calling async function with state only")
                    coro = engine(processed_state)
                
                # Run coroutine in event loop
                logger.debug(f"Running coroutine in event loop")
                result = loop.run_until_complete(coro)
                logger.debug(f"Async function returned: {type(result).__name__}")
                
                # Handle Command/Send directly
                if isinstance(result, (Command, Send)) or (
                        isinstance(result, list) and all(isinstance(x, Send) for x in result)):
                    logger.debug(f"Returning Command/Send from async function directly")
                    return result
                
                # Process output
                processed_output = process_output(result, config, processed_state)
                
                # Handle command pattern
                return handle_command_pattern(processed_output, config)
                
            except Exception as e:
                logger.error(f"Error in async node {config.name}: {str(e)}")
                logger.error(traceback.format_exc())
                
                # Create error result
                return create_error_result(e, config)
                
        return node_function

@register_node_processor("mapping")
class MappingNodeProcessor:
    """Processor for mapping functions that return Send objects."""
    
    def can_process(self, engine: Any) -> bool:
        """Check if this processor can handle the engine."""
        # Check for functions with Send return annotation or explicit marker
        if callable(engine):
            if hasattr(engine, "__mapping_node__") and engine.__mapping_node__:
                return True
                
            if hasattr(engine, "__annotations__"):
                if "return" in engine.__annotations__:
                    return_type = str(engine.__annotations__["return"])
                    has_send = "List[Send]" in return_type or "list[Send]" in return_type
                    logger.debug(f"MappingNodeProcessor.can_process (annotations): {has_send}")
                    return has_send
        return False
    
    def create_node_function(self, engine: Any, config: NodeConfig) -> Callable:
        """Create a node function for a mapping function."""
        logger.debug(f"Creating mapping node function: {config.name}")
        
        def node_function(state, runtime_config=None):
            """Node function for mapping."""
            logger.debug(f"Called mapping node: {config.name}")
            
            try:
                # Process state
                processed_state = process_state(state)
                
                # Handle async mapping functions
                if asyncio.iscoroutinefunction(engine):
                    logger.debug(f"Handling async mapping function")
                    try:
                        loop = asyncio.get_event_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                    
                    # Run coroutine
                    logger.debug(f"Running async mapping function in event loop")
                    result = loop.run_until_complete(engine(processed_state))
                else:
                    # For mapping functions, we don't apply normal input/output processing
                    # since they are expected to return Send objects directly
                    logger.debug(f"Calling sync mapping function")
                    result = engine(processed_state)
                
                logger.debug(f"Mapping function returned: {type(result).__name__}")
                
                # Return Send objects as-is
                return result
                
            except Exception as e:
                logger.error(f"Error in mapping node {config.name}: {str(e)}")
                logger.error(traceback.format_exc())
                
                # For mapping nodes, return empty list on error
                logger.debug(f"Returning empty list due to error")
                return []
                
        return node_function

@register_node_processor("generic")
class GenericNodeProcessor:
    """Fallback processor for any engine type."""
    
    def can_process(self, engine: Any) -> bool:
        """This processor can handle any engine."""
        return True
    
    def create_node_function(self, engine: Any, config: NodeConfig) -> Callable:
        """Create a node function for a generic object."""
        logger.debug(f"Creating generic node function: {config.name}")
        
        def node_function(state, runtime_config=None):
            """Node function for generic engine."""
            logger.debug(f"Called generic node: {config.name}")
            
            try:
                # Process state
                processed_state = process_state(state)
                
                # Return the engine as the result
                result = {"result": engine}
                logger.debug(f"Created result with engine as 'result' key")
                
                # Handle command pattern
                return handle_command_pattern(result, config)
                
            except Exception as e:
                logger.error(f"Error in generic node {config.name}: {str(e)}")
                logger.error(traceback.format_exc())
                
                # Create error result
                return create_error_result(e, config)
                
        return node_function