# src/haive_core/graph/node/factory.py

import inspect
import logging
import traceback
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command, Send

from haive_core.config.runnable import RunnableConfigManager
from haive_core.engine.base import Engine, InvokableEngine, NonInvokableEngine
from haive_core.graph.node.config import NodeConfig

# Basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NodeFactory")

class NodeFactory:
    """Factory for creating node functions with better error handling and debugging."""
    
    @classmethod
    def create_node_function(
        cls,
        config: Union[NodeConfig, Engine, Callable],
        command_goto: Optional[str] = None,
        input_mapping: Optional[Dict[str, str]] = None,
        output_mapping: Optional[Dict[str, str]] = None,
        runnable_config: Optional[RunnableConfig] = None,
        debug: bool = True
    ) -> Callable:
        """Create a node function with proper engine handling."""
        
        # Convert to NodeConfig if not already
        if not isinstance(config, NodeConfig):
            node_name = getattr(config, "name", "unnamed_node")
            node_config = NodeConfig(
                name=node_name,
                engine=config,
                command_goto=command_goto,
                input_mapping=input_mapping,
                output_mapping=output_mapping,
                runnable_config=runnable_config
            )
        else:
            node_config = config
        
        # Get engine
        engine, engine_id = node_config.resolve_engine()
        logger.info(f"Creating node function for {node_config.name} using engine type: {type(engine).__name__}")
        
        def node_function(state, runtime_config=None):
            """Node function with robust error handling and response handling."""
            try:
                # Log input state type
                logger.info(f"Node {node_config.name} received state type: {type(state)}")
                
                # Handle Pydantic model state
                if hasattr(state, "__dict__") and not isinstance(state, dict):
                    # Extract state dictionary from Pydantic model
                    if hasattr(state, "model_dump"):
                        # Pydantic v2
                        processed_state = state.model_dump()
                    elif hasattr(state, "dict"):
                        # Pydantic v1
                        processed_state = state.dict()
                    else:
                        # Fallback to __dict__
                        processed_state = state.__dict__
                else:
                    processed_state = state
                
                # Extract and process messages
                messages = []
                
                # If state has a messages field, extract and process
                if isinstance(processed_state, dict) and "messages" in processed_state:
                    raw_messages = processed_state["messages"]
                    
                    # Process raw messages into proper format
                    for msg in raw_messages:
                        if isinstance(msg, BaseMessage):
                            # Already a proper message
                            messages.append(msg)
                        elif isinstance(msg, tuple) and len(msg) >= 2:
                            # Process tuple messages
                            role, content = msg[0], msg[1]
                            
                            if role == "human":
                                messages.append(HumanMessage(content=content))
                            elif role in ["ai", "assistant"]:
                                messages.append(AIMessage(content=content))
                            elif role == "system":
                                messages.append(SystemMessage(content=content))
                            else:
                                from langchain_core.messages import ChatMessage
                                messages.append(ChatMessage(role=role, content=content))
                        elif isinstance(msg, str):
                            # String message
                            messages.append(HumanMessage(content=msg))
                        elif isinstance(msg, dict) and "content" in msg:
                            # Dict-format message
                            role = msg.get("role", "human")
                            content = msg["content"]
                            
                            if role == "human":
                                messages.append(HumanMessage(content=content))
                            elif role in ["ai", "assistant"]:
                                messages.append(AIMessage(content=content))
                            elif role == "system":
                                messages.append(SystemMessage(content=content))
                            else:
                                from langchain_core.messages import ChatMessage
                                messages.append(ChatMessage(role=role, content=content))
                        else:
                            # Unknown format
                            messages.append(HumanMessage(content=str(msg)))
                
                # Log engine info for debugging
                if debug and hasattr(engine, "__dict__"):
                    logger.info(f"Engine: {engine.__dict__}")
                
                # CRITICAL FIX: Pass messages list directly to the engine when it's an LLM
                if messages:
                    logger.info(f"Input data: {{'messages': {messages}}}")
                    logger.info(f"State: messages={messages}")
                    
                    # Pass messages list directly to the engine
                    response = engine.invoke(messages)
                    
                    # Handle response based on its type
                    if isinstance(response, BaseMessage):
                        # Single message response - update messages list to include it
                        all_messages = messages + [response]
                        return Command(update={"messages": all_messages}, goto=node_config.command_goto)
                    elif isinstance(response, List) and all(isinstance(msg, BaseMessage) for msg in response):
                        # List of messages response - update messages list with all returned messages
                        all_messages = messages + response
                        return Command(update={"messages": all_messages}, goto=node_config.command_goto) 
                    elif isinstance(response, Dict) and "generations" in response:
                        # Handle generation-style response
                        try:
                            # Extract the generated message and add it to the messages
                            ai_message = response["generations"][0][0].message
                            all_messages = messages + [ai_message]
                            return Command(update={"messages": all_messages}, goto=node_config.command_goto)
                        except (IndexError, KeyError, AttributeError):
                            # Fallback if extraction fails
                            logger.warning(f"Could not extract AI message from generations")
                            return Command(update={"result": response}, goto=node_config.command_goto)
                    else:
                        # For other response types, just return as result
                        return Command(update={"result": response, "messages": messages}, goto=node_config.command_goto)
                else:
                    # If no messages, use process normally with mapped input
                    # Apply input mapping if specified
                    if isinstance(processed_state, dict) and node_config.input_mapping:
                        mapped_input = {}
                        for state_key, input_key in node_config.input_mapping.items():
                            if state_key in processed_state:
                                mapped_input[input_key] = processed_state[state_key]
                        
                        # Use value directly if only one item mapped
                        if len(node_config.input_mapping) == 1 and len(mapped_input) == 1:
                            input_data = list(mapped_input.values())[0]
                        else:
                            input_data = mapped_input
                    else:
                        input_data = processed_state
                    
                    logger.info(f"Calling function with input type: {type(input_data)}")
                    result = engine.invoke(input_data)
                    
                    # Apply output mapping if needed
                    if isinstance(result, dict) and node_config.output_mapping:
                        output = {}
                        for result_key, state_key in node_config.output_mapping.items():
                            if result_key in result:
                                output[state_key] = result[result_key]
                        
                        if output:
                            result = output
                    
                    return Command(update=result, goto=node_config.command_goto)
                
            except Exception as e:
                error_msg = f"Error in node {node_config.name}: {str(e)}"
                logger.error(error_msg)
                logger.error(f"State type: {type(state)}")
                logger.error("Traceback:")
                logger.error(traceback.format_exc())
                
                error_data = {
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "timestamp": datetime.now().isoformat()
                }
                
                if node_config.command_goto is not None:
                    return Command(update={"error": error_data}, goto=node_config.command_goto)
                else:
                    return {"error": error_data}
        
        # Add metadata to function
        node_function.__node_config__ = node_config
        node_function.__engine_id__ = engine_id
        
        return node_function