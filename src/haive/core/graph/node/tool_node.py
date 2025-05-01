import copy
from typing import Any, Dict, List, Optional, Union

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, BaseToolkit, StructuredTool, Tool
from langgraph.types import Command
from pydantic import BaseModel

from haive.core.schema.state_schema import StateSchema


def create_tool_node(
    tools: Union[List[Union[BaseTool, Tool, StructuredTool, BaseModel]], BaseToolkit],
    command_goto: Optional[Any] = None,
    handle_errors: bool = True,
    parallel: bool = True,
    max_workers: int = 4,
) -> Callable:
    """
    Create a node function for tool execution.

    Args:
        tools: Tools to use (various types supported)
        command_goto: Optional destination for command routing
        handle_errors: Whether to handle errors during tool execution
        parallel: Whether to execute tools in parallel
        max_workers: Maximum number of parallel workers

    Returns:
        Tool node function
    """
    # Process different tool input types
    processed_tools = []

    # Handle BaseToolkit
    if isinstance(tools, BaseToolkit):
        processed_tools.extend(tools.get_tools())
    # Handle list of tools
    elif isinstance(tools, list):
        for tool in tools:
            # Handle BaseModel -> convert to StructuredTool
            if isinstance(tool, BaseModel) and not isinstance(tool, BaseTool):
                from langchain_core.tools import StructuredTool

                structured_tool = StructuredTool.from_function(
                    func=tool.execute if hasattr(tool, "execute") else None,
                    name=getattr(tool, "name", tool.__class__.__name__),
                    description=getattr(tool, "description", ""),
                    args_schema=tool.__class__,
                )
                processed_tools.append(structured_tool)
            else:
                processed_tools.append(tool)
    else:
        raise ValueError(f"Unsupported tool type: {type(tools)}")

    # Create tool map for lookup
    tool_map = {tool.name: tool for tool in processed_tools}

    def tool_node(state: StateSchema, config: RunnableConfig) -> Any:
        # Extract messages and tool calls
        messages = getattr(state, "messages", [])
        if not messages:
            # No messages to process
            if command_goto:
                return Command(update=state, goto=command_goto)
            return state

        # Get last message
        last_message = messages[-1]

        # Extract tool calls (handle different formats)
        tool_calls = []
        if hasattr(last_message, "tool_calls"):
            tool_calls = last_message.tool_calls
        elif (
            hasattr(last_message, "additional_kwargs")
            and "tool_calls" in last_message.additional_kwargs
        ):
            tool_calls = last_message.additional_kwargs["tool_calls"]

        if not tool_calls:
            # No tool calls to process
            if command_goto:
                return Command(update=state, goto=command_goto)
            return state

        # Execute tools
        results = execute_tools(
            tool_calls, tool_map, parallel, max_workers, handle_errors, config
        )

        # Create tool messages
        tool_messages = []
        for result in results:
            tool_message = ToolMessage(
                content=result["content"],
                tool_call_id=result["tool_call_id"],
                name=result["tool_name"],
            )
            tool_messages.append(tool_message)

        # Update messages
        updated_messages = list(messages)
        updated_messages.extend(tool_messages)

        # Create updated state
        updated_state = copy.deepcopy(state)
        setattr(updated_state, "messages", updated_messages)

        # Return with command_goto if specified
        if command_goto:
            return Command(update=updated_state, goto=command_goto)
        return updated_state

    return tool_node
