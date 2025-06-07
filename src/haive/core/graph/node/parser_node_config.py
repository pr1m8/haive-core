import logging
from typing import Any, Optional

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
    # Removed tool field - it will come from state

    def __call__(
        self, state: StateLike, config: Optional[ConfigLike] = None
    ) -> Command:
        """Parse the tool message into a Pydantic model."""
        logger.debug("=" * 60)
        logger.debug("ParserNodeConfig.__call__ starting")
        logger.debug(f"State type: {type(state)}")
        logger.debug(
            f"State attributes: {dir(state) if hasattr(state, '__dict__') else 'N/A'}"
        )
        logger.debug(f"Config: {config}")

        # Ensure we have a valid command_goto
        goto_node = self.command_goto or self.agent_node
        logger.debug(f"Goto node: {goto_node}")

        # Debug state structure
        if hasattr(state, "__dict__"):
            logger.debug(f"State.__dict__: {state.__dict__}")
        if hasattr(state, "get"):
            logger.debug("State has .get() method (dict-like)")
        else:
            logger.debug("State does not have .get() method (object-like)")

        # Extract tool information from state with detailed logging
        logger.debug("Extracting tool_name...")
        tool_name = (
            getattr(state, "tool_name", state.get("tool_name", None))
            if hasattr(state, "get")
            else getattr(state, "tool_name", None)
        )
        logger.debug(f"Extracted tool_name: {tool_name}")

        logger.debug("Extracting tool from state...")
        tool = (
            getattr(state, "tool", state.get("tool", None))
            if hasattr(state, "get")
            else getattr(state, "tool", None)
        )
        logger.debug(f"Extracted tool: {tool}")
        logger.debug(f"Tool type: {type(tool)}")

        logger.debug("Extracting tool_call...")
        tool_call = (
            getattr(state, "tool_call", state.get("tool_call", None))
            if hasattr(state, "get")
            else getattr(state, "tool_call", None)
        )
        logger.debug(f"Extracted tool_call: {tool_call}")
        logger.debug(f"Tool_call type: {type(tool_call)}")

        logger.debug("Extracting tool_message...")
        tool_message = (
            getattr(state, "tool_message", state.get("tool_message", None))
            if hasattr(state, "get")
            else getattr(state, "tool_message", None)
        )
        logger.debug(f"Extracted tool_message: {tool_message}")
        logger.debug(f"Tool_message type: {type(tool_message)}")

        logger.debug(f"Extracting messages using key: {self.messages_key}")
        messages = (
            getattr(state, self.messages_key, state.get(self.messages_key, []))
            if hasattr(state, "get")
            else getattr(state, self.messages_key, [])
        )
        logger.debug(f"Extracted messages: {len(messages) if messages else 0} messages")

        # Debug messages in detail
        if messages:
            logger.debug("Messages breakdown:")
            for i, msg in enumerate(messages):
                logger.debug(
                    f"  Message {i}: type={type(msg)}, content_preview={str(msg)[:100]}..."
                )
                if hasattr(msg, "__class__"):
                    logger.debug(f"    Class: {msg.__class__.__name__}")
                if hasattr(msg, "content"):
                    logger.debug(f"    Content length: {len(str(msg.content))}")
                if isinstance(msg, ToolMessage):
                    logger.debug(
                        f"    ✓ Is ToolMessage - name: {getattr(msg, 'name', 'N/A')}, tool_call_id: {getattr(msg, 'tool_call_id', 'N/A')}"
                    )

            # Check last message specifically
            last_msg = messages[-1]
            logger.debug(f"Last message type: {type(last_msg)}")
            logger.debug(
                f"Last message is ToolMessage: {isinstance(last_msg, ToolMessage)}"
            )
            if isinstance(last_msg, ToolMessage):
                logger.debug(
                    f"Last ToolMessage name: {getattr(last_msg, 'name', 'N/A')}"
                )
                logger.debug(
                    f"Last ToolMessage content: {getattr(last_msg, 'content', 'N/A')}"
                )
        else:
            logger.debug("No messages found!")

        # Log what we received for debugging
        logger.debug("SUMMARY of extracted values:")
        logger.debug(f"  Tool name: {tool_name} (type: {type(tool_name)})")
        logger.debug(f"  Tool object: {tool} (type: {type(tool)})")
        logger.debug(f"  Tool call: {tool_call} (type: {type(tool_call)})")
        logger.debug(f"  Tool message: {tool_message} (type: {type(tool_message)})")
        logger.debug(f"  Messages count: {len(messages) if messages else 0}")

        # Handle missing tool information
        if not tool:
            logger.error("No tool provided in state!")
            logger.debug("State keys available:")
            if hasattr(state, "keys"):
                logger.debug(f"  Keys: {list(state.keys())}")
            elif hasattr(state, "__dict__"):
                logger.debug(f"  Attributes: {list(state.__dict__.keys())}")

            # Try to provide helpful error message
            error_msg = (
                f"Missing tool for {tool_name}"
                if tool_name
                else "Missing tool information"
            )
            return Command(update={"error": error_msg}, goto=goto_node)

        # If we don't have tool_name but have tool, try to get name from tool
        if not tool_name and tool:
            if hasattr(tool, "__name__"):
                tool_name = tool.__name__
                logger.debug(f"Retrieved tool_name from tool.__name__: {tool_name}")
            elif hasattr(tool, "name"):
                tool_name = tool.name
                logger.debug(f"Retrieved tool_name from tool.name: {tool_name}")
            else:
                logger.warning("Could not determine tool_name from tool object")
                tool_name = "unknown_tool"

        logger.debug(f"Final tool_name for processing: {tool_name}")

        try:
            # Get content from tool message with detailed logging
            logger.debug("Attempting to get tool message content...")

            # Check if we need to get tool_message from last message
            if not tool_message and messages:
                logger.debug("No tool_message provided, checking last message...")
                last_message = messages[-1]
                logger.debug(
                    f"Last message type check: {type(last_message)} == ToolMessage? {isinstance(last_message, ToolMessage)}"
                )

                if isinstance(last_message, ToolMessage):
                    tool_message = last_message
                    logger.debug(
                        f"✓ Using last message as tool_message: {tool_message}"
                    )
                    logger.debug(
                        f"  Tool message name: {getattr(tool_message, 'name', 'N/A')}"
                    )
                    logger.debug(
                        f"  Tool message content preview: {str(getattr(tool_message, 'content', 'N/A'))[:200]}..."
                    )
                else:
                    logger.debug(
                        f"✗ Last message is not ToolMessage, it's {type(last_message)}"
                    )
                    # Try to find the last ToolMessage in the list
                    logger.debug("Searching for ToolMessage in messages...")
                    for i, msg in enumerate(reversed(messages)):
                        if isinstance(msg, ToolMessage):
                            tool_message = msg
                            logger.debug(
                                f"✓ Found ToolMessage at position {len(messages)-1-i}: {tool_message}"
                            )
                            break
                    if not tool_message:
                        logger.debug("✗ No ToolMessage found in any message")

            content = None
            if tool_message and hasattr(tool_message, "content"):
                content = tool_message.content
                logger.debug(
                    f"Got content from tool_message: {content[:200]}..."
                    if len(str(content)) > 200
                    else f"Got content from tool_message: {content}"
                )
            else:
                logger.debug(
                    f"No content available from tool_message. tool_message: {tool_message}, has content: {hasattr(tool_message, 'content') if tool_message else False}"
                )

            model_instance = None

            if content:
                logger.debug("Attempting to parse content...")
                # Try to parse as JSON
                try:
                    import json

                    logger.debug("Trying JSON parsing...")
                    json_data = json.loads(content)
                    logger.debug(f"JSON parsing successful: {json_data}")

                    # Create model instance from JSON
                    if isinstance(tool, type) and issubclass(tool, BaseModel):
                        logger.debug(
                            f"Creating {tool.__name__} instance from JSON data"
                        )
                        model_instance = tool.model_validate(json_data)
                        logger.debug(
                            f"Model instance created successfully: {model_instance}"
                        )
                    else:
                        logger.debug(
                            "Tool is not a BaseModel subclass, using JSON data directly"
                        )
                        model_instance = json_data
                except json.JSONDecodeError as e:
                    logger.debug(f"JSON parsing failed: {e}")
                    # Not valid JSON, try using PydanticOutputParser
                    if isinstance(tool, type) and issubclass(tool, BaseModel):
                        logger.debug("Trying PydanticOutputParser...")
                        try:
                            from langchain.output_parsers import PydanticOutputParser

                            parser = PydanticOutputParser(pydantic_object=tool)
                            model_instance = parser.parse(content)
                            logger.debug(
                                f"PydanticOutputParser successful: {model_instance}"
                            )
                        except Exception as parser_error:
                            logger.debug(f"PydanticOutputParser failed: {parser_error}")
                            model_instance = {"content": content}
                    else:
                        logger.debug("Using content as-is in dictionary")
                        model_instance = {"content": content}

            # Fallback to tool_call args if no tool message or parsing failed
            elif tool_call and (
                hasattr(tool_call, "args")
                or isinstance(tool_call, dict)
                and "args" in tool_call
            ):
                logger.debug("Falling back to tool_call args...")
                args = (
                    tool_call.args if hasattr(tool_call, "args") else tool_call["args"]
                )
                logger.debug(f"Tool call args: {args}")

                if isinstance(tool, type) and issubclass(tool, BaseModel):
                    logger.debug(
                        f"Creating {tool.__name__} instance from tool_call args"
                    )
                    model_instance = tool.model_validate(args)
                    logger.debug(f"Model instance from args: {model_instance}")
                else:
                    logger.debug("Using args directly")
                    model_instance = args

            else:
                logger.error(f"No valid content found for parsing {tool_name}")
                logger.debug("Debug info for failed parsing:")
                logger.debug(f"  tool_message: {tool_message}")
                logger.debug(f"  tool_call: {tool_call}")
                logger.debug(f"  content: {content}")
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
            logger.debug(f"Using field_name: {field_name}")

            # Create update dictionary with parsed model
            update_dict = {field_name: model_instance}
            logger.debug(f"Created update_dict: {update_dict}")

            # Add result message to messages if needed
            if messages:
                logger.debug("Adding result to messages...")
                new_messages = list(messages)

                # Use existing tool message or create a new one
                if not tool_message:
                    logger.debug("Creating new ToolMessage...")
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
                    logger.debug(f"Created new ToolMessage: {tool_message}")

                new_messages.append(tool_message)
                update_dict[self.messages_key] = new_messages
                logger.debug(f"Added {len(new_messages)} messages to update_dict")

            logger.info(f"✓ Successfully parsed {tool_name}")
            logger.debug(f"Final update_dict keys: {list(update_dict.keys())}")
            logger.debug("=" * 60)
            return Command(update=update_dict, goto=goto_node)

        except Exception as e:
            logger.exception(f"Error parsing {tool_name}: {str(e)}")
            logger.debug(f"Exception details: {type(e).__name__}: {str(e)}")
            logger.debug("=" * 60)
            return Command(
                update={"error": f"Failed to parse {tool_name}: {str(e)}"},
                goto=goto_node,
            )
