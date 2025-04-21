# src/haive_core/graph/node/factory.py

import inspect
import logging
import traceback
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command, Send

from haive.core.config.runnable import RunnableConfigManager
from haive.core.engine.base import Engine, InvokableEngine, NonInvokableEngine
from haive.core.graph.node.config import NodeConfig
from pydantic import BaseModel
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NodeFactory")

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
        runnable_config: Optional[RunnableConfig] = None,
        debug: bool = False
    ) -> Callable:
        """
        Create a node function with proper engine handling.
        
        Args:
            config: NodeConfig, Engine, or callable function
            command_goto: Optional next node to go to (ignored if NodeConfig provided)
            input_mapping: Optional mapping from state keys to engine input keys (ignored if NodeConfig provided)
            output_mapping: Optional mapping from engine output keys to state keys (ignored if NodeConfig provided)
            runnable_config: Optional default runtime configuration (ignored if NodeConfig provided)
            debug: Enable debug logging
            
        Returns:
            A node function compatible with LangGraph
        """
        # Convert to NodeConfig if not already
        if not isinstance(config, NodeConfig):
            node_name = getattr(config, "name", "unnamed_node")
            node_config = NodeConfig(
                name=node_name,
                engine=config,
                command_goto=command_goto,
                input_mapping=input_mapping,
                output_mapping=output_mapping,
                runnable_config=runnable_config,
                debug=debug
            )
        else:
            node_config = config
            
        # Resolve engine reference if needed
        engine, engine_id = node_config.resolve_engine()
        logger.debug(f"Creating node function for {node_config.name} using engine: {getattr(engine, 'name', str(engine))}")
        
        # Create appropriate node function based on engine type
        if isinstance(engine, InvokableEngine):
            node_func = cls._create_invokable_engine_node(engine, node_config)
        elif isinstance(engine, NonInvokableEngine):
            node_func = cls._create_non_invokable_engine_node(engine, node_config)
        elif callable(engine):
            node_func = cls._create_callable_node(engine, node_config)
        else:
            logger.warning(f"Unknown engine type: {type(engine).__name__}. Creating generic node.")
            node_func = cls._create_generic_node(engine, node_config)
        
        # Add metadata to the function for serialization and node tracking
        node_func.__node_config__ = node_config
        node_func.__engine_id__ = engine_id
        
        return node_func
    
    @classmethod
    def _create_invokable_engine_node(cls, engine: InvokableEngine, config: NodeConfig) -> Callable:
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
            try:
                # Log debug info if enabled
                if config.debug:
                    logger.info(f"Node {config.name} processing state: {type(state).__name__}")
                
                # Process the state object into a dict if needed
                processed_state = cls._preprocess_state(state)
                
                # Merge runtime configs
                merged_config = cls._merge_configs(config.runnable_config, runtime_config)
                
                # Apply engine ID targeting
                if engine_id := getattr(engine, "id", None):
                    merged_config = cls._ensure_engine_id_targeting(merged_config, engine_id)
                    
                # Apply node-specific config overrides
                if config.config_overrides and merged_config:
                    merged_config = cls._apply_config_overrides(merged_config, engine_id, config.config_overrides)
                
                # Check if this is a messages-based input and should use direct message passing
                if config.use_direct_messages and "messages" in processed_state:
                    # Get messages from state
                    messages = cls._prepare_messages(processed_state["messages"])
                    
                    # Log messages if in debug mode
                    if config.debug:
                        logger.info(f"Invoking with direct messages: {len(messages)} messages")
                    
                    # Invoke engine with messages
                    result = engine.invoke(messages, merged_config)
                    
                    # Process result based on its type
                    return cls._handle_message_result(result, processed_state, messages, config)
                
                # Not using direct messages - apply input mapping
                input_data = cls._extract_input(processed_state, config.input_mapping)
                
                # Log input data if in debug mode
                if config.debug:
                    logger.info(f"Invoking with input: {type(input_data).__name__}")
                
                # Invoke the engine
                result = engine.invoke(input_data, merged_config)
                
                # Handle the result
                return cls._handle_result(result, config.command_goto, config.output_mapping, processed_state)
                
            except Exception as e:
                logger.error(f"Error in node {config.name}: {str(e)}")
                logger.error(traceback.format_exc())
                
                # Return error with command goto if specified
                error_data = {
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "timestamp": datetime.now().isoformat()
                }
                
                return Command(update={"error": error_data}, goto=config.command_goto)
        
        return node_function
    
    @classmethod
    def _create_non_invokable_engine_node(cls, engine: NonInvokableEngine, config: NodeConfig) -> Callable:
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
            try:
                # Process the state object into a dict if needed
                processed_state = cls._preprocess_state(state)
                
                # Merge runtime configs
                merged_config = cls._merge_configs(config.runnable_config, runtime_config)
                
                # Apply engine ID targeting
                if engine_id := getattr(engine, "id", None):
                    merged_config = cls._ensure_engine_id_targeting(merged_config, engine_id)
                    
                # Apply node-specific config overrides
                if config.config_overrides and merged_config:
                    merged_config = cls._apply_config_overrides(merged_config, engine_id, config.config_overrides)
                
                # Instantiate the engine
                instance = engine.instantiate(merged_config)
                
                # Return the instance with Command for control flow
                return Command(update={"instance": instance}, goto=config.command_goto)
                
            except Exception as e:
                logger.error(f"Error instantiating engine in node {config.name}: {str(e)}")
                logger.error(traceback.format_exc())
                
                # Return error with command goto if specified
                error_data = {
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "timestamp": datetime.now().isoformat()
                }
                
                return Command(update={"error": error_data}, goto=config.command_goto)
        
        return node_function
    
    @classmethod
    def _create_callable_node(cls, func: Callable, config: NodeConfig) -> Callable:
        """
        Create a node function from a callable.
        
        Args:
            func: The callable function
            config: Node configuration
            
        Returns:
            Node function
        """
        # Check if function accepts config
        sig = inspect.signature(func)
        accepts_config = "config" in sig.parameters
        
        def node_function(state, runtime_config=None):
            """Wrapped node function."""
            try:
                # Process the state object into a dict if needed
                processed_state = cls._preprocess_state(state)
                
                # Merge runtime configs
                merged_config = cls._merge_configs(config.runnable_config, runtime_config)
                
                # Call original function with appropriate arguments
                if accepts_config:
                    result = func(processed_state, merged_config)
                else:
                    result = func(processed_state)
                
                # Handle the result for output mapping and command
                return cls._handle_result(result, config.command_goto, config.output_mapping, processed_state)
                
            except Exception as e:
                logger.error(f"Error in callable node {config.name}: {str(e)}")
                logger.error(traceback.format_exc())
                
                # Return error with command goto if specified
                error_data = {
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "timestamp": datetime.now().isoformat()
                }
                
                return Command(update={"error": error_data}, goto=config.command_goto)
        
        return node_function
    
    @classmethod
    def _create_generic_node(cls, obj: Any, config: NodeConfig) -> Callable:
        """
        Create a node function for a non-engine, non-callable object.
        
        Args:
            obj: Any object
            config: Node configuration
            
        Returns:
            Node function
        """
        def node_function(state, runtime_config=None):
            """Generic node function that returns the object as-is."""
            try:
                # Return the object with Command for control flow
                return Command(update={"result": obj}, goto=config.command_goto)
                
            except Exception as e:
                logger.error(f"Error in generic node {config.name}: {str(e)}")
                logger.error(traceback.format_exc())
                
                # Return error with command goto if specified
                error_data = {
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "timestamp": datetime.now().isoformat()
                }
                
                return Command(update={"error": error_data}, goto=config.command_goto)
        
        return node_function
    
    @classmethod
    def _preprocess_state(cls, state: Any) -> Dict[str, Any]:
        """
        Preprocess state object into a dictionary.
        
        Args:
            state: State object (dict, BaseModel, etc.)
            
        Returns:
            Dictionary representation of state
        """
        # Handle different state types
        if isinstance(state, dict):
            # Already a dictionary
            return state
        elif hasattr(state, "model_dump"):
            # Pydantic v2 model
            return state.model_dump()
        elif hasattr(state, "dict"):
            # Pydantic v1 model
            return state.dict()
        elif hasattr(state, "__dict__"):
            # Object with __dict__ attribute
            return state.__dict__
        else:
            # Fallback - just return empty dict
            return {}
    
    @classmethod
    def _prepare_messages(cls, messages_data: Any) -> List[BaseMessage]:
        """
        Prepare messages data into a list of BaseMessage objects.
        
        Args:
            messages_data: Messages in various formats
            
        Returns:
            List of BaseMessage objects
        """
        if not messages_data:
            return []
            
        # Handle different message formats
        if isinstance(messages_data, list):
            normalized_messages = []
            
            for msg in messages_data:
                if isinstance(msg, BaseMessage):
                    # Already a proper message
                    normalized_messages.append(msg)
                elif isinstance(msg, tuple) and len(msg) >= 2:
                    # Process tuple messages: (role, content)
                    role, content = msg[0], msg[1]
                    
                    if role == "human" or role == "user":
                        normalized_messages.append(HumanMessage(content=content))
                    elif role in ["ai", "assistant"]:
                        normalized_messages.append(AIMessage(content=content))
                    elif role == "system":
                        normalized_messages.append(SystemMessage(content=content))
                    else:
                        # Use ChatMessage for custom roles
                        from langchain_core.messages import ChatMessage
                        normalized_messages.append(ChatMessage(role=role, content=content))
                elif isinstance(msg, dict) and "content" in msg:
                    # Dict-format message with at least content
                    role = msg.get("role", "human")
                    content = msg["content"]
                    
                    if role == "human" or role == "user":
                        normalized_messages.append(HumanMessage(content=content))
                    elif role in ["ai", "assistant"]:
                        normalized_messages.append(AIMessage(content=content))
                    elif role == "system":
                        normalized_messages.append(SystemMessage(content=content))
                    else:
                        # Use ChatMessage for custom roles
                        from langchain_core.messages import ChatMessage
                        normalized_messages.append(ChatMessage(role=role, content=content))
                elif isinstance(msg, str):
                    # String message - treated as human message
                    normalized_messages.append(HumanMessage(content=msg))
                else:
                    # Unknown format - convert to string and use as human message
                    normalized_messages.append(HumanMessage(content=str(msg)))
            
            return normalized_messages
        elif isinstance(messages_data, str):
            # Single string - treat as human message
            return [HumanMessage(content=messages_data)]
        elif isinstance(messages_data, BaseMessage):
            # Single message object
            return [messages_data]
        else:
            # Unknown format - convert to string and use as human message
            return [HumanMessage(content=str(messages_data))]
    
    @classmethod
    def _extract_input(cls, state: Dict[str, Any], input_mapping: Optional[Dict[str, str]]) -> Any:
        """
        Extract input from state based on mapping.
        
        Args:
            state: State dictionary
            input_mapping: Mapping from state keys to engine input keys
            
        Returns:
            Input data for engine
        """
        # If no mapping, return state as-is
        if not input_mapping:
            return state
        
        # Apply mapping
        mapped_input = {}
        for state_key, input_key in input_mapping.items():
            if state_key in state:
                mapped_input[input_key] = state[state_key]
            elif state_key == "messages" and "messages" not in state:
                # Special handling for messages field - many engines expect this
                # If asked for messages but none in state, provide empty list
                mapped_input[input_key] = []
        
        # If only one key was mapped and we have that value, return it directly
        if len(input_mapping) == 1 and len(mapped_input) == 1:
            return list(mapped_input.values())[0]
        
        # Return the mapped dictionary
        return mapped_input
    
    @classmethod
    def _handle_message_result(
        cls, 
        result: Any, 
        state: Dict[str, Any],
        input_messages: List[BaseMessage],
        config: NodeConfig
    ) -> Any:
        """
        Handle result specific to message-based interaction.
        
        Args:
            result: Engine result
            state: Original state
            input_messages: Input messages sent to engine
            config: Node configuration
            
        Returns:
            Processed result as Command or dict
        """
        # Start with state updates
        updates = {}
        
        # Handle different result types
        if isinstance(result, BaseMessage):
            # Single message result - update messages with input + new message
            all_messages = input_messages + [result]
            updates["messages"] = all_messages
            
            # Extract content if needed
            if config.extract_content:
                updates["content"] = result.content
                
        elif isinstance(result, list) and all(isinstance(msg, BaseMessage) for msg in result):
            # List of messages - use as the new messages
            all_messages = input_messages + result
            updates["messages"] = all_messages
            
            # Extract content from last message if needed
            if config.extract_content and result:
                updates["content"] = result[-1].content
                
        elif isinstance(result, dict):
            # Dictionary result
            
            # If it has a 'generations' key, extract messages
            if "generations" in result:
                try:
                    message = result["generations"][0][0].message
                    all_messages = input_messages + [message]
                    updates["messages"] = all_messages
                    
                    # Extract content if needed
                    if config.extract_content:
                        updates["content"] = message.content
                        
                except (IndexError, KeyError, AttributeError):
                    # Fallback to full result
                    updates["result"] = result
            
            # If it has a structured output model field
            elif hasattr(config.engine, "structured_output_model") and config.engine.structured_output_model:
                model_name = config.engine.structured_output_model.__name__.lower()
                if model_name in result:
                    # Result has a field matching the model name
                    updates[model_name] = result[model_name]
                    
                # Always include the full result
                updates["output"] = result
                
                # Try to find an AI message in the result
                if "message" in result and isinstance(result["message"], BaseMessage):
                    # Update messages with the message
                    all_messages = input_messages + [result["message"]]
                    updates["messages"] = all_messages
                
            else:
                # Apply output mapping if needed
                if config.output_mapping:
                    for result_key, state_key in config.output_mapping.items():
                        if result_key in result:
                            updates[state_key] = result[result_key]
                else:
                    # No output mapping - use entire result
                    updates.update(result)
                    
                # Keep messages in state
                if "messages" in state:
                    updates["messages"] = state["messages"]
                
                # Check if result has content that should be extracted
                if config.extract_content and "content" in result:
                    updates["content"] = result["content"]
        
        else:
            # Other result types - store as result
            updates["result"] = result
            
            # Maintain existing messages
            if "messages" in state:
                updates["messages"] = state["messages"]
        
        # Return as Command if command_goto is specified
        if config.command_goto is not None:
            return Command(update=updates, goto=config.command_goto)
        else:
            return updates
    
    @classmethod
    def _handle_result(
        cls, 
        result: Any, 
        command_goto: Optional[str], 
        output_mapping: Optional[Dict[str, str]],
        original_state: Dict[str, Any] = None
    ) -> Any:
        """
        Handle different result types to ensure proper Command/Send pattern support.
        
        Args:
            result: Result from engine or function
            command_goto: Optional goto directive
            output_mapping: Optional output mapping
            original_state: Original state for preserving data
            
        Returns:
            Result wrapped in Command if needed
        """
        # Handle already Command/Send results
        if isinstance(result, Command):
            # If already Command, ensure goto is set if not already and one is provided
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
        
        # Special handling for Pydantic models
        if isinstance(result, BaseModel):
            # If the result is directly a model instance, process it specially
            processed_output = {}
            model_name = result.__class__.__name__.lower()
            processed_output[model_name] = result
            
            # Apply output mapping if needed
            if output_mapping:
                for output_key, state_key in output_mapping.items():
                    if output_key.startswith(f"{model_name}."):
                        field = output_key.split(".", 1)[1]
                        if hasattr(result, field):
                            processed_output[state_key] = getattr(result, field)
        else:
            # Process output if it's not already a Command, Send, or Pydantic model
            processed_output = cls._process_output(result, output_mapping, original_state)
        
        # Wrap in Command if goto is specified
        if command_goto is not None:
            return Command(update=processed_output, goto=command_goto)
        else:
            return processed_output
    @classmethod
    def _process_output(
        cls, 
        output: Any, 
        output_mapping: Optional[Dict[str, str]],
        original_state: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Process output according to mapping.
        
        Args:
            output: Output from engine or function
            output_mapping: Mapping from output keys to state keys
            original_state: Original state for preserving data
            
        Returns:
            Processed output
        """
        # Handle non-dict output
        if not isinstance(output, dict):
            # Special case: If output is a BaseMessage, extract content
            if isinstance(output, BaseMessage):
                result = {"content": output.content}
                
                # If we had messages in the original state, append this message
                if original_state and "messages" in original_state:
                    messages = original_state["messages"]
                    if isinstance(messages, list):
                        result["messages"] = messages + [output]
                
                return result
            
            # If output is a string and we have an output mapping with 'text' key,
            # assume this is from a StrOutputParser
            if isinstance(output, str) and output_mapping and 'text' in output_mapping:
                result = {output_mapping['text']: output}
                return result
                
            # Default handling - store as result
            return {"result": output}
        
        # Handle dictionary output
        result = {}
        
        # First check for structured output models in the result
        # Look for keys that match model class names in lowercase
        model_keys = [k for k in output.keys() if k.endswith("model") or k[0].islower() and k[0:1].isupper()]
        
        # If output has a structured model field, we need special handling
        if any(isinstance(output.get(k), BaseModel) for k in output):
            for k, v in output.items():
                if isinstance(v, BaseModel):
                    # Add the model directly under its name
                    result[k] = v
                    
                    # Also add individual fields if in output_mapping
                    if output_mapping:
                        model_name = k.lower()
                        for model_field in v.__dict__:
                            mapping_key = f"{model_name}.{model_field}"
                            if mapping_key in output_mapping:
                                result[output_mapping[mapping_key]] = getattr(v, model_field)
        
        # Apply standard output mapping if provided
        if output_mapping:
            for output_key, state_key in output_mapping.items():
                # Check for direct fields
                if output_key in output:
                    result[state_key] = output[output_key]
                # Check for nested fields (using dot notation)
                elif "." in output_key:
                    parts = output_key.split(".")
                    obj = output
                    valid_path = True
                    
                    # Navigate through the nested structure
                    for part in parts[:-1]:
                        if part in obj and isinstance(obj, dict):
                            obj = obj[part]
                        elif hasattr(obj, part):
                            obj = getattr(obj, part)
                        else:
                            valid_path = False
                            break
                    
                    # Set the value if we found a valid path
                    if valid_path:
                        last_part = parts[-1]
                        if isinstance(obj, dict) and last_part in obj:
                            result[state_key] = obj[last_part]
                        elif hasattr(obj, last_part):
                            result[state_key] = getattr(obj, last_part)
        
        # If we applied mappings but found nothing, or no mapping provided
        # include the original output
        if not result or not output_mapping:
            # For Pydantic models, convert to dict
            if isinstance(output, BaseModel):
                if hasattr(output, "model_dump"):
                    # Pydantic v2
                    result.update(output.model_dump())
                else:
                    # Pydantic v1
                    result.update(output.dict())
            else:
                # Regular dict
                result.update(output)
        
        return result
    
    
    @classmethod
    def _merge_configs(
        cls, 
        base_config: Optional[RunnableConfig], 
        override_config: Optional[RunnableConfig]
    ) -> Optional[RunnableConfig]:
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
        from haive.core.config.runnable import RunnableConfigManager
        return RunnableConfigManager.merge(base_config, override_config)
    
    @classmethod
    def _ensure_engine_id_targeting(
        cls, 
        config: Optional[Dict[str, Any]], 
        engine_id: str
    ) -> Dict[str, Any]:
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
    def _apply_config_overrides(
        cls, 
        config: Dict[str, Any], 
        engine_id: Optional[str], 
        overrides: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply node-specific configuration overrides to the config.
        
        Args:
            config: Config dictionary to modify
            engine_id: Optional engine ID to target
            overrides: Configuration overrides to apply
            
        Returns:
            Updated config
        """
        # Make a copy to avoid modifying the original
        config = config.copy()
        
        if "configurable" not in config:
            config["configurable"] = {}
        
        if engine_id:
            # Apply to engine-specific configuration
            if "engine_configs" not in config["configurable"]:
                config["configurable"]["engine_configs"] = {}
                
            if engine_id not in config["configurable"]["engine_configs"]:
                config["configurable"]["engine_configs"][engine_id] = {}
                
            # Update with overrides
            for key, value in overrides.items():
                config["configurable"]["engine_configs"][engine_id][key] = value
        else:
            # Apply to top-level configurable
            for key, value in overrides.items():
                config["configurable"][key] = value
                
        return config
    

