# src/haive/core/graph/tool_config.py

from typing import Any, Dict, List, Optional, Union, Callable,Tuple
from pydantic import BaseModel, Field,create_model

from langchain_core.tools import BaseTool, StructuredTool
from langgraph.types import RetryPolicy


class ToolConfig(BaseModel):
    """Configuration for a tool with routing and retry behavior."""
    tool: Union[BaseTool, StructuredTool, Callable] = Field(
        ..., description="The tool implementation"
    )
    route_to: Optional[str] = Field(
        default=None, description="Where to route after this tool is used (node name or 'END')"
    )
    return_direct: bool = Field(
        default=False, 
        description="Whether to return directly to user without further LLM processing"
    )
    timeout: Optional[float] = Field(
        default=None, description="Timeout in seconds for this tool"
    )
    retry_policy: Optional[RetryPolicy] = Field(
        default=None, description="Retry policy for this tool"
    )
    fallback_response: Optional[str] = Field(
        default=None, description="Fallback response if tool fails"
    )


class NodeConfig(BaseModel):
    """Configuration for a custom node in the agent graph."""
    node_function: Callable = Field(..., description="The node function")
    node_name: str = Field(..., description="Name for this node")
    routes_to: Union[str, Dict[str, str], Any] = Field(
        ..., description="Where this node routes to (direct, mapping, or router config)"
    )


# Helper functions for tool configuration
def configure_tool(
    tool: Union[BaseTool, StructuredTool, Callable],
    route_to: Optional[str] = None,
    return_direct: bool = False,
    timeout: Optional[float] = None,
    retry_policy: Optional[RetryPolicy] = None,
    fallback_response: Optional[str] = None
) -> ToolConfig:
    """
    Configure a tool with routing, retry, and other settings.
    
    Args:
        tool: The tool implementation (BaseTool, StructuredTool, or callable)
        route_to: Node name to route to after this tool is used
        return_direct: Whether to return tool output directly to user
        timeout: Timeout in seconds for this tool
        retry_policy: Retry policy for this tool
        fallback_response: Fallback message if tool execution fails
        
    Returns:
        ToolConfig instance
    """
    return ToolConfig(
        tool=tool,
        route_to=route_to,
        return_direct=return_direct,
        timeout=timeout,
        retry_policy=retry_policy,
        fallback_response=fallback_response
    )


def create_node_config(
    node_function: Callable,
    node_name: str,
    routes_to: Union[str, Dict[str, str], Any]
) -> NodeConfig:
    """
    Create a configuration for a custom node.
    
    Args:
        node_function: Function that implements the node
        node_name: Name for this node
        routes_to: Where this node routes to (string, mapping, or router)
        
    Returns:
        NodeConfig instance
    """
    return NodeConfig(
        node_function=node_function,
        node_name=node_name,
        routes_to=routes_to
    )


def process_tools(tools: List[Union[BaseTool, StructuredTool, Callable, ToolConfig]]) -> Tuple[List[Any], Dict[str, ToolConfig]]:
    """
    Process a list of tools, extracting configurations.
    
    Args:
        tools: List of tools and tool configs
        
    Returns:
        Tuple of (processed_tools, tool_configs)
    """
    processed_tools = []
    tool_configs = {}
    
    for item in tools:
        if isinstance(item, ToolConfig):
            # Extract the tool and keep the config
            tool = item.tool
            tool_configs[getattr(tool, "name", str(id(tool)))] = item
            processed_tools.append(tool)
        else:
            # Keep as is
            processed_tools.append(item)
    
    return processed_tools, tool_configs