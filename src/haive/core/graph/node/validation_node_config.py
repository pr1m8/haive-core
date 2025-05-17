from typing import Any, Callable, Dict, List, Optional, Sequence, Type, Union

from langchain_core.messages import BaseMessage, ToolCall, ToolMessage
from langchain_core.tools import BaseTool, BaseToolkit, StructuredTool, Tool
from langgraph.prebuilt import ValidationNode
from langgraph.types import Command
from pydantic import BaseModel, Field, model_validator

from haive.core.graph.common.types import ConfigLike, NodeType, StateLike
from haive.core.graph.node.config import NodeConfig
from haive.core.utils.message_utils import has_tool_error


class ValidationNodeConfig(NodeConfig):
    """
    Configuration for a validation node in a graph.
    """

    node_type: NodeType = Field(
        default=NodeType.VALIDATION, description="The type of node"
    )
    name: str = Field(default="validation", description="The name of the node")
    messages_key: str = Field(
        default="messages", description="The key to use for the messages field"
    )
    schemas: Sequence[Union[BaseTool, Type[BaseModel], Callable]] = Field(
        default_factory=list, description="The schemas to use for the node"
    )
    tags: Optional[List[str]] = Field(
        default=None, description="Optional tags for the validation node"
    )
    format_error: Optional[
        Callable[[BaseException, ToolCall, Type[BaseModel]], str]
    ] = Field(default=None, description="Custom formatter for validation errors")
    validation_status_key: str = Field(
        default="validated", description="Key in state to store validation status"
    )

    @model_validator(mode="after")
    def validate_validation_node(self) -> "ValidationNodeConfig":
        """Validate that validation schemas are specified properly."""
        if not self.schemas and not getattr(self, "validation_schemas", None):
            raise ValueError("Schemas must be specified for a ValidationNodeConfig")

        # Use validation_schemas if present (backward compatibility)
        if not self.schemas and getattr(self, "validation_schemas", None):
            object.__setattr__(self, "schemas", self.validation_schemas)

        # Override node_type without triggering recursive validation
        object.__setattr__(self, "node_type", NodeType.VALIDATION)

        return self

    def __call__(
        self, state: StateLike, config: Optional[ConfigLike] = None
    ) -> Command:
        """
        Execute the validation node, remapping `messages_key` to 'messages' as needed.
        """
        # Step 1: Pull the messages from the state using the configured key
        messages = state.get(self.messages_key)
        if messages is None:
            raise ValueError(f"No messages found under '{self.messages_key}'")

        # Step 2: Create a dict with 'messages' key expected by ValidationNode
        validation_input = {"messages": messages}

        # Step 3: Create and run the validation node
        validation_node = ValidationNode(
            schemas=self.schemas,
            format_error=self.format_error,
            name=self.name,
            tags=self.tags,
        )
        print("----------------------------")
        result_messages = validation_node.invoke(validation_input)["messages"]
        # print('----------------------------')
        # print(result_messages)
        # print('----------------------------')
        # Step 4: Check for tool errors
        has_errors = any(
            isinstance(msg, ToolMessage) and has_tool_error(msg)
            for msg in result_messages
        )
        print(has_errors)
        # Step 5: Build update dict
        update = {self.validation_status_key: not has_errors}

        if has_errors:
            update[self.messages_key] = result_messages

        return Command(update=update, goto=self.command_goto)
