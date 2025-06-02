from typing import Callable, List, Optional, Sequence, Union

from langchain_core.tools import BaseTool, BaseToolkit, StructuredTool, Tool
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field, model_validator

from haive.core.graph.common.types import ConfigLike, NodeType, StateLike
from haive.core.graph.node.config import NodeConfig


class ToolNodeConfig(NodeConfig):
    node_type: NodeType = Field(default=NodeType.TOOL, description="The type of node")
    name: str = Field(default="tools", description="The name of the node")
    tools: Sequence[
        Union[BaseTool, Callable, StructuredTool, Tool, BaseToolkit, BaseModel]
    ] = Field(default_factory=list, description="The tools to use for the node")
    resolved_tools: List[BaseTool] = Field(
        default_factory=list,
        description="The resolved tools after processing",
        exclude=True,
    )
    tags: Optional[List[str]] = Field(
        default=None, description="Optional tags for the tools"
    )
    handle_tool_errors: Union[
        bool, str, Callable[..., str], tuple[type[Exception], ...]
    ] = Field(default=True, description="How to handle tool errors")
    messages_key: str = Field(
        default="messages", description="The key to use for messages in the state"
    )

    @model_validator(mode="after")
    def resolve_tools(self):
        """Validate and transform the tools into resolved_tools."""
        resolved_tools = []

        for tool_item in self.tools:
            if isinstance(tool_item, BaseToolkit):
                # Extract tools from a toolkit
                toolkit_tools = tool_item.get_tools()
                resolved_tools.extend(toolkit_tools)
            elif isinstance(tool_item, (BaseTool, StructuredTool, Tool)):
                # Add individual LangChain tool
                resolved_tools.append(tool_item)
            elif isinstance(tool_item, BaseModel) and hasattr(tool_item, "to_tool"):
                # Handle Pydantic models with to_tool method
                resolved_tools.append(tool_item.to_tool())
            elif callable(tool_item):
                # Convert callable to a tool
                try:
                    from langchain_core.tools import tool

                    resolved_tools.append(tool(tool_item))
                except Exception as e:
                    raise ValueError(f"Failed to convert callable to tool: {e}")
            else:
                raise ValueError(f"Unsupported tool type: {type(tool_item)}")

        self.resolved_tools = resolved_tools
        return self

    def __call__(self, state: StateLike, config: Optional[ConfigLike] = None):
        """Create and invoke a ToolNode with the current configuration."""
        return ToolNode(
            tools=self.resolved_tools,
            name=self.name,
            tags=self.tags,
            handle_tool_errors=self.handle_tool_errors,
            messages_key=self.messages_key,
        )(state)

    @classmethod
    def from_tools(
        cls,
        tools: Sequence[
            Union[BaseTool, Callable, StructuredTool, Tool, BaseToolkit, BaseModel]
        ],
        **kwargs,
    ):
        config = cls(tools=tools, **kwargs)
        config.resolve_tools()
        return config
