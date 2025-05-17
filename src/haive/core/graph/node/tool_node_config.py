# src/haive/core/graph/node/tool.py

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool, BaseToolkit, StructuredTool, Tool
from langgraph.prebuilt import ToolNode
from langgraph.types import Command
from pydantic import BaseModel, Field, field_validator, model_validator

from haive.core.graph.node.base_config import NodeConfig
from haive.core.graph.node.types import CommandGoto, NodeType


class ToolNodeConfig(NodeConfig):
    """
    Configuration for a tool node in a graph.

    Tool nodes execute LangChain tools and handle tool calls from LLM messages.
    """

    # Override node_type from base class
    node_type: NodeType = Field(default=NodeType.TOOL, description="The type of node")

    # Tool-specific fields
    tools: Sequence[
        Union[Type[BaseTool], Type[BaseModel], Callable, StructuredTool, BaseModel]
    ] = Field(
        default_factory=list,
        description="The schemas to use for validation (BaseTool, Type[BaseModel], or Callable)",
    )
    tags: Optional[List[str]] = Field(
        default=None, description="Optional tags for the tool node"
    )
    handle_tool_errors: Union[
        bool, str, Callable[..., str], Tuple[Type[Exception], ...]
    ] = Field(default=True, description="How to handle tool errors")
    messages_key: str = Field(
        default="messages", description="The key to use for the messages field"
    )

    def __call__(
        self, state: Dict[str, Any], config: Optional[Dict[str, Any]] = None
    ) -> Command:
        """
        Execute the tool node with the given state and configuration.

        Args:
            state: The current state of the graph
            config: Optional runtime configuration

        Returns:
            A Command with state update including tool execution results
        """
        # Create the tool node
        tool_node = ToolNode(
            tools=self.tools,
            name=self.name,
            tags=self.tags,
            handle_tool_errors=self.handle_tool_errors,
            messages_key=self.messages_key,
        )

        # Execute the tool node
        result = tool_node.invoke(state)
        result_messages = result.get(self.messages_key, [])

        # Create the update with the new messages
        update = {self.messages_key: result_messages}

        # Return a Command with the update and next node
        return Command(update=update, goto=self.command_goto)

    @classmethod
    def from_tools(
        cls,
        tools: Sequence[
            Union[
                BaseTool, Callable, StructuredTool, Tool, BaseToolkit, Type[BaseModel]
            ]
        ],
        **kwargs
    ):
        """
        Create a tool node configuration from a list of tools.

        Args:
            tools: List of tools to use in this node
            **kwargs: Additional configuration parameters

        Returns:
            Configured ToolNodeConfig
        """
        return cls(tools=tools, **kwargs)
