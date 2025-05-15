# src/haive/core/graph/node/validation.py

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Type, Union

from langchain_core.messages import BaseMessage, ToolCall, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.prebuilt import ValidationNode
from langgraph.types import Command
from pydantic import BaseModel, Field, field_validator

from haive.core.graph.node.base_config import NodeConfig
from haive.core.graph.node.types import NodeType
from haive.core.utils.message_utils import has_tool_error

# Define logger
logger = logging.getLogger(__name__)


class ValidationNodeConfig(NodeConfig):
    """
    Configuration for a validation node in a graph.

    Validation nodes process tool calls and validate their inputs against schemas.
    """

    # Override node_type from base class
    node_type: NodeType = Field(
        default=NodeType.VALIDATION, description="The type of node"
    )

    # Validation-specific fields
    messages_key: str = Field(
        default="messages", description="The key to use for the messages field"
    )
    # Use Any for the runtime type to avoid validation issues, but document the expected types
    schemas: Sequence[
        Union[Type[BaseTool], Type[BaseModel], Callable, StructuredTool]
    ] = Field(
        default_factory=list,
        description="The schemas to use for validation (BaseTool, Type[BaseModel], or Callable)",
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

    def __call__(
        self, state: Dict[str, Any], config: Optional[Dict[str, Any]] = None
    ) -> Command:
        """
        Execute the validation node, remapping `messages_key` to 'messages' as needed.

        Args:
            state: The current state of the graph
            config: Optional runtime configuration

        Returns:
            A Command with validation results and next node
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

        result = validation_node.invoke(validation_input)
        result_messages = result.get("messages", [])

        # Step 4: Check for tool errors
        has_errors = any(
            isinstance(msg, ToolMessage) and has_tool_error(msg)
            for msg in result_messages
        )

        # Step 5: Build update dict
        update = {self.validation_status_key: not has_errors}

        # Only update messages if there are errors
        if has_errors:
            update[self.messages_key] = result_messages

        return Command(update=update, goto=self.command_goto)

    @classmethod
    def from_schemas(cls, schemas: List[Any], **kwargs):
        """
        Create a validation node configuration from a list of schemas.

        Args:
            schemas: List of schemas (BaseTools, Pydantic models, or callables)
            **kwargs: Additional configuration parameters

        Returns:
            Configured ValidationNodeConfig
        """
        return cls(schemas=schemas, **kwargs)
