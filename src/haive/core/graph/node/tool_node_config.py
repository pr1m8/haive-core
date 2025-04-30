# src/haive/core/graph/tool_node_config.py

from typing import Dict, List, Any, Optional, Union, Callable, Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool, StructuredTool, Tool
from langgraph.prebuilt import ToolNode
from langgraph.types import Send

from haive.core.graph.node.config import NodeConfig

class ToolNodeConfig(NodeConfig):
    """
    NodeConfig for tool execution with rich debugging support.
    
    This extends NodeConfig with additional capabilities for defining,
    configuring, and inspecting tools in a graph.
    """
    # Tool configurations
    tools: List[Union[BaseTool, Dict[str, Any]]] = Field(
        default_factory=list,
        description="Tools to be used in this node"
    )
    
    single_tool_mode: bool = Field(
        default=False,
        description="Whether this node handles only a specific tool"
    )
    
    filter_by_tool_name: Optional[str] = Field(
        default=None,
        description="Only execute tools with this name"
    )
    
    skip_parsing: bool = Field(
        default=False,
        description="Skip automatic input parsing"
    )
    
    parallel_execution: bool = Field(
        default=True,
        description="Execute multiple tool calls in parallel"
    )
    
    tool_error_handling: Optional[str] = Field(
        default=None,
        description="How to handle tool errors: 'ignore', 'return', or 'raise'"
    )
    
    # Override model config
    model_config = {
        "arbitrary_types_allowed": True
    }
    
    def create_runnable(self, runnable_config=None) -> Any:
        """
        Create a ToolNode runnable for this configuration.
        
        Args:
            runnable_config: Optional runtime configuration
            
        Returns:
            Configured ToolNode
        """
        # Resolve tools if needed
        tools = self._resolve_tools()
        
        # Create ToolNode with appropriate options
        if self.filter_by_tool_name:
            # Filter tools by name
            filtered_tools = [t for t in tools if getattr(t, "name", None) == self.filter_by_tool_name]
            tool_node = ToolNode(tools=filtered_tools)
            self.debug_log(f"Created ToolNode with filtered tools: {[getattr(t, 'name', 'unnamed') for t in filtered_tools]}")
        else:
            # Use all tools
            tool_node = ToolNode(tools=tools)
            self.debug_log(f"Created ToolNode with all tools: {[getattr(t, 'name', 'unnamed') for t in tools]}")
        
        return tool_node
    
    def _resolve_tools(self) -> List[BaseTool]:
        """
        Resolve tool references to actual tool objects.
        
        Returns:
            List of resolved tool objects
        """
        resolved_tools = []
        
        for tool in self.tools:
            if isinstance(tool, (BaseTool, StructuredTool, Tool)):
                # Already a tool object
                resolved_tools.append(tool)
                continue
                
            if isinstance(tool, dict):
                # Tool configuration dictionary
                try:
                    if "name" in tool and "func" in tool:
                        # Create Tool from function and config
                        resolved_tool = Tool(
                            name=tool["name"],
                            description=tool.get("description", ""),
                            func=tool["func"]
                        )
                        resolved_tools.append(resolved_tool)
                        self.debug_log(f"Created Tool from config: {tool['name']}")
                    else:
                        self.debug_log(f"Skipping invalid tool config - missing required fields: {tool}", level="warning")
                except Exception as e:
                    self.debug_log(f"Error creating tool from config: {e}", level="error")
                    continue
            elif callable(tool):
                # Callable function
                try:
                    # Try to get tool name from function name
                    name = getattr(tool, "__name__", "tool")
                    # Try to get docstring as description
                    description = getattr(tool, "__doc__", "") or f"Tool function {name}"
                    
                    # Create Tool
                    resolved_tool = Tool(
                        name=name,
                        description=description,
                        func=tool
                    )
                    resolved_tools.append(resolved_tool)
                    self.debug_log(f"Created Tool from callable: {name}")
                except Exception as e:
                    self.debug_log(f"Error creating tool from callable: {e}", level="error")
                    continue
            else:
                self.debug_log(f"Unsupported tool type: {type(tool)}", level="warning")
        
        if not resolved_tools:
            self.debug_log("No tools were resolved", level="warning")
            
        return resolved_tools
    
    @classmethod
    def from_tools(cls, 
                 tools: List[Union[BaseTool, Dict[str, Any], Callable]],
                 name: Optional[str] = None,
                 filter_by_tool_name: Optional[str] = None,
                 parallel_execution: bool = True,
                 **kwargs) -> 'ToolNodeConfig':
        """
        Create a ToolNodeConfig from a list of tools.
        
        Args:
            tools: List of tools or tool configurations
            name: Optional name for the node
            filter_by_tool_name: Optional tool name filter
            parallel_execution: Whether to execute tools in parallel
            **kwargs: Additional NodeConfig parameters
            
        Returns:
            Configured ToolNodeConfig
        """
        return cls(
            name=name or "tool_node",
            tools=tools,
            filter_by_tool_name=filter_by_tool_name,
            parallel_execution=parallel_execution,
            **kwargs
        )

    @classmethod
    def for_single_tool(cls,
                      tool: Union[BaseTool, Dict[str, Any], Callable],
                      name: Optional[str] = None,
                      **kwargs) -> 'ToolNodeConfig':
        """
        Create a ToolNodeConfig for a single tool.
        
        Args:
            tool: The tool to use
            name: Optional name for the node
            **kwargs: Additional NodeConfig parameters
            
        Returns:
            Configured ToolNodeConfig for single tool
        """
        # Extract tool name for the node name if not provided
        if not name and isinstance(tool, BaseTool):
            name = f"{tool.name}_node"
        elif not name and isinstance(tool, dict) and "name" in tool:
            name = f"{tool['name']}_node"
        elif not name and callable(tool):
            name = f"{getattr(tool, '__name__', 'tool')}_node"
        else:
            name = name or "single_tool_node"
        
        # Create config
        return cls(
            name=name,
            tools=[tool],
            single_tool_mode=True,
            filter_by_tool_name=getattr(tool, "name", None) if isinstance(tool, BaseTool) else None,
            **kwargs
        )