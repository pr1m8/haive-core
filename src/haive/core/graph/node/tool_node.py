# node/tool_node.py
from typing import Any, List, Optional, Union

from langchain.agents.tools import BaseToolkit, Tool
from langchain.tools import BaseTool, StructuredTool, Tool
from langgraph.prebuilt import ToolNode
from langgraph.types import Command
from pydantic import BaseModel

from haive.core.graph.node.protocols import NodeFunction


def create_tool_node(
    tools: List[Union[BaseTool, Tool, StructuredTool, BaseModel, BaseToolkit]],
    command_goto: Optional[Any] = None,
    handle_errors: bool = True,
    parallel: bool = True,
    max_workers: int = 4,
    node_config: Optional[dict] = None,
    retry_policy: Optional[Any] = None,
) -> NodeFunction:
    """
    Create a tool execution node.

    Args:
        tools: List of tools or toolkits
        command_goto: Optional destination for Command routing
        handle_errors: Whether to handle tool errors
        parallel: Whether to execute tools in parallel
        max_workers: Maximum number of parallel workers
        node_config: Optional node configuration
        retry_policy: Optional retry policy

    Returns:
        Tool node function
    """
    # Process toolkits to extract tools
    processed_tools = []

    for tool in tools:
        if isinstance(tool, BaseToolkit):
            # Extract tools from toolkit
            processed_tools.extend(tool.get_tools())
        else:
            processed_tools.append(tool)

    # Create LangGraph tool node with all params
    tool_node = ToolNode(
        tools=processed_tools,
        handle_tool_error=handle_errors,
        parallel=parallel,
        max_workers=max_workers,
    )

    # Wrap the tool node to handle configuration and command_goto
    def wrapped_tool_node(state, config=None):
        # Use node_config as fallback
        effective_config = config or node_config

        # Call the LangGraph tool node
        result = tool_node(state, effective_config)

        # Add command_goto if needed
        if command_goto is not None and not isinstance(result, Command):
            return Command(update=result, goto=command_goto)

        return result

    # Apply retry policy if specified
    if retry_policy:
        from langgraph.prebuilt import RetryNode

        wrapped_tool_node = RetryNode(wrapped_tool_node, retry_policy)

    # Add metadata
    wrapped_tool_node.__tools__ = processed_tools
    wrapped_tool_node.__node_config__ = node_config
    wrapped_tool_node.__command_goto__ = command_goto

    return wrapped_tool_node
