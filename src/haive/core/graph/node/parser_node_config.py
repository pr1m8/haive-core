import logging
from typing import Optional

from langchain_core.messages import ToolMessage
from langgraph.types import Command
from pydantic import BaseModel, Field
from rich.console import Console

from haive.core.graph.common.types import ConfigLike, NodeType, StateLike
from haive.core.graph.node.base_config import NodeConfig

logger = logging.getLogger(__name__)
console = Console()


class ParserNodeConfig(NodeConfig):
    """Configuration for a node that parses tool outputs into Pydantic models."""

    node_type: NodeType = Field(default=NodeType.PARSER)
    messages_key: str = Field(default="messages")
    agent_node: str = Field(
        default="agent", description="Node to return to after parsing"
    )

    def __call__(
        self, state: StateLike, config: Optional[ConfigLike] = None
    ) -> Command:
        """Parse the tool message into a Pydantic model."""
        # Ensure we have a valid command_goto
        goto_node = self.command_goto or self.agent_node

        # Extract tool information from state
        # Handle both dict-like and object-like state access
        tool_name = (
            getattr(state, "tool_name", state.get("tool_name", None))
            if hasattr(state, "get")
            else getattr(state, "tool_name", None)
        )
        tool = (
            getattr(state, "tool", state.get("tool", None))
            if hasattr(state, "get")
            else getattr(state, "tool", None)
        )
        tool_call = (
            getattr(state, "tool_call", state.get("tool_call", None))
            if hasattr(state, "get")
            else getattr(state, "tool_call", None)
        )
        tool_message = (
            getattr(state, "tool_message", state.get("tool_message", None))
            if hasattr(state, "get")
            else getattr(state, "tool_message", None)
        )
        messages = (
            getattr(state, self.messages_key, state.get(self.messages_key, []))
            if hasattr(state, "get")
            else getattr(state, self.messages_key, [])
        )

        # Log what we received for debugging
        logger.debug(
            f"ParserNode received - Tool: {tool_name}, Message: {tool_message}"
        )

        if not tool or not tool_name:
            logger.error("Missing required tool information")
            return Command(update={"error": "Missing tool information"}, goto=goto_node)

        try:
            # Get content from tool message
            if tool_message and hasattr(tool_message, "content"):
                content = tool_message.content

                # Try to parse as JSON
                try:
                    import json

                    json_data = json.loads(content)

                    # Create model instance from JSON
                    if isinstance(tool, type) and issubclass(tool, BaseModel):
                        model_instance = tool.model_validate(json_data)
                    else:
                        model_instance = json_data
                except json.JSONDecodeError:
                    # Not valid JSON, try using PydanticOutputParser
                    if isinstance(tool, type) and issubclass(tool, BaseModel):
                        from langchain.output_parsers import PydanticOutputParser

                        parser = PydanticOutputParser(pydantic_object=tool)
                        model_instance = parser.parse(content)
                    else:
                        model_instance = {"content": content}

            # Fallback to tool_call args if no tool message or parsing failed
            elif tool_call and (
                hasattr(tool_call, "args")
                or isinstance(tool_call, dict)
                and "args" in tool_call
            ):
                args = (
                    tool_call.args if hasattr(tool_call, "args") else tool_call["args"]
                )

                if isinstance(tool, type) and issubclass(tool, BaseModel):
                    model_instance = tool.model_validate(args)
                else:
                    model_instance = args

            else:
                return Command(
                    update={"error": f"No valid content found for parsing {tool_name}"},
                    goto=goto_node,
                )

            # Determine field name for the parsed model
            field_name = (
                tool.__name__.lower()
                if hasattr(tool, "__name__")
                else tool_name.lower()
            )

            # Create update dictionary with parsed model
            update_dict = {field_name: model_instance}

            # Add result message to messages if needed
            if messages:
                new_messages = list(messages)

                # Use existing tool message or create a new one
                if not tool_message:
                    tool_id = (
                        getattr(tool_call, "id", f"call_{tool_name}_autogen")
                        if hasattr(tool_call, "id")
                        else tool_call.get("id", f"call_{tool_name}_autogen")
                    )
                    tool_message = ToolMessage(
                        content=str(model_instance),
                        name=tool_name,
                        tool_call_id=tool_id,
                    )

                new_messages.append(tool_message)
                update_dict[self.messages_key] = new_messages

            logger.info(f"Successfully parsed {tool_name}")
            return Command(update=update_dict, goto=goto_node)

        except Exception as e:
            logger.exception(f"Error parsing {tool_name}: {str(e)}")
            return Command(
                update={"error": f"Failed to parse {tool_name}: {str(e)}"},
                goto=goto_node,
            )
