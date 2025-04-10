from typing import Dict, Optional, Union, Literal, Any, Callable
import logging
from pydantic import BaseModel
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.graph import END
from langgraph.types import Command
from src.haive.core.engine.aug_llm import AugLLMConfig
from langgraph.prebuilt import create_react_agent
logger = logging.getLogger(__name__)

def create_node_function(
    config: AugLLMConfig,
    command_goto: Optional[Union[str, Literal["END"]]] = None,
    input_mapping: Optional[Dict[str, str]] = None,
    output_mapping: Optional[Dict[str, str]] = None,
    async_mode: bool = False
):
    """
    Creates a reliable node function from an AugLLMConfig.
    """
    runnable = config.create_runnable()
    # Ensure input_mapping is initialized
    if not input_mapping:
        input_mapping = {}  # ✅ Fix: Ensure `input_mapping` is always set
        logger.debug(f"Initialized empty input_mapping: {input_mapping}")

    # Ensure output_mapping is initialized
    if not output_mapping:
        output_mapping = {"output": "output"}  # ✅ Fix: Ensure default output mapping

    logger.debug(f"Using input_mapping: {input_mapping}")
    logger.debug(f"Using output_mapping: {output_mapping}")

    # Determine next node
    goto = command_goto if command_goto is not None else END

    # Create the node function
    def node_function(state):
        logger.debug(f"Node function called with state: {state}")

        # Convert state to dictionary if needed
        state_dict = state.model_dump() if hasattr(state, "model_dump") else state
        logger.debug(f"State dictionary: {state_dict}")

        # Extract input dynamically
        engine_input = {
            engine_key: state_dict[state_key]
            for state_key, engine_key in input_mapping.items()
            if state_key in state_dict
        }
        if engine_input == {}:
            if 'messages' in state_dict:
                engine_input = {'messages': state_dict['messages']}
        # Debugging
        logger.debug(f"Processed engine_input: {engine_input}")

        # Handle missing input
        if not engine_input:
            logger.warning("No valid input extracted from state")
            if "messages" in state_dict:
                error_msg = AIMessage(content="Error: Missing required input.")
                return Command(update={"messages": state_dict["messages"] + [error_msg]}, goto=goto)
            return Command(update=state_dict, goto=goto)

        # Invoke the engine safely
        try:
            logger.debug(f"Invoking engine with input: {engine_input}")
            result = runnable.invoke(input=engine_input)
            logger.debug(f"Engine result: {result}")
        except Exception as e:
            logger.error(f"Error invoking engine: {e}")
            if "messages" in state_dict:
                error_msg = AIMessage(content=f"Error encountered: {str(e)}")
                return Command(update={"messages": state_dict["messages"] + [error_msg]}, goto=goto)
            return Command(update=state_dict, goto=goto)

        # Process result and update state
        state_update = process_result(result, state_dict, output_mapping)

        return Command(update=state_update, goto=goto)

    return node_function


def process_result(result, state_dict, output_mapping):
    """
    Process result from engine and update state.

    Args:
        result: The result from the engine
        state_dict: The current state
        output_mapping: Output mapping configuration

    Returns:
        Updated state dictionary
    """
    state_update = dict(state_dict)
    logger.debug(f"Processing result: {result}")
    # Handle AIMessage results
    if isinstance(result, AIMessage):
        logger.debug(f"Processing AIMessage with content: {result.content[:50]}...")
        for output_key, state_key in output_mapping.items():
            if output_key == "content":
                state_update[state_key] = result.content
        if "messages" in state_dict:
            state_update["messages"] = state_dict["messages"] + [result]

    # Handle tuple responses
    elif isinstance(result, tuple) and len(result) == 2:
        logger.debug(f"Processing tuple response: ({result[0]}, {result[1][:50]}...)")
        for output_key, state_key in output_mapping.items():
            if output_key == "content":
                state_update[state_key] = result[1]
        if "messages" in state_dict:
            state_update["messages"] = state_dict["messages"] + [result]

    # Handle dictionary results
    elif isinstance(result, dict):
        logger.debug(f"Processing dict result with keys: {list(result.keys())}")
        for output_key, state_key in output_mapping.items():
            if output_key in result:
                state_update[state_key] = result[output_key]
        if "messages" in result and "messages" in state_dict:
            if isinstance(result["messages"], list):
                state_update["messages"] = state_dict["messages"] + result["messages"]

    # Handle other data types (string, int, etc.)
    else:
        logger.debug(f"Processing other result type: {type(result).__name__}")
        ai_msg = AIMessage(content=str(result))
        for output_key, state_key in output_mapping.items():
            state_update[state_key] = str(result)
        if "messages" in state_dict:
            state_update["messages"] = state_dict["messages"] + [ai_msg]

    return state_update


def create_tool_node(
    tools,
    post_processor=None,
    command_goto=None
):
    """
    Create a tool execution node.
    
    Args:
        tools: List of tools to use
        post_processor: Optional function to process results
        command_goto: Next node to go to
        
    Returns:
        A node function for tool execution
    """
    # Import here to avoid circular imports
    from langgraph.prebuilt.tool_node import ToolNode
    
    # Create tool node
    tool_node = ToolNode(tools)
    
    # Create wrapper for correct state handling and routing
    def node_function(state):
        logger.debug("Tool node called")
        
        try:
            # Process with tool node
            result = tool_node.invoke(state)
            logger.debug("Tool executed successfully")
            
            # Post-process if needed
            if post_processor:
                result = post_processor(result)
                logger.debug("Post-processed tool result")
        except Exception as e:
            logger.error(f"Error in tool node: {e}")
            # Handle error gracefully
            state_dict = state.model_dump() if hasattr(state, "model_dump") else state
            if "messages" in state_dict:
                from langchain_core.messages import AIMessage
                error_msg = AIMessage(content=f"Error executing tool: {str(e)}")
                result = {**state_dict, "messages": state_dict["messages"] + [error_msg]}
            else:
                result = state_dict
        
        # Add routing if specified
        goto = command_goto or END
        return Command(update=result, goto=goto)
    
    return node_function