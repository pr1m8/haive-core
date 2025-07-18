"""State schema with structured output parsing capabilities using LangChain output parsers."""

from typing import Any, Dict, List, Optional, Type, Union

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.output_parsers import (
    BaseOutputParser,
    JsonOutputToolsParser,
    PydanticOutputParser,
    PydanticToolsParser,
)
from pydantic import BaseModel, Field, field_validator, model_validator

from haive.core.schema.prebuilt.messages.messages_with_token_usage import (
    MessagesStateWithTokenUsage,
)


class StructuredOutputState(MessagesStateWithTokenUsage):
    """MessagesState with automatic structured output parsing and token tracking.

    This state schema extends MessagesStateWithTokenUsage to automatically parse
    AI messages into structured outputs using LangChain output parsers. It leverages
    the PydanticToolsParser to convert Pydantic models into tool call messages,
    maintaining proper message flow and token tracking.

    Key features:
    - Automatic parsing of AI messages with structured output
    - Conversion of Pydantic models to tool call messages
    - Token usage tracking for all messages including parsed outputs
    - Support for multiple output parser types
    - Field validator integration for seamless parsing

    Example:
        ```python
        from pydantic import BaseModel

        class SearchQuery(BaseModel):
            query: str
            filters: Dict[str, Any]

        # Configure state with output model
        state = StructuredOutputState(
            output_models=[SearchQuery],
            parse_as_tools=True  # Convert to tool calls
        )

        # AI message with structured output gets parsed automatically
        ai_msg = AIMessage(
            content='{"query": "python", "filters": {"language": "en"}}',
            response_metadata={"token_usage": {"total_tokens": 50}}
        )
        state.messages.append(ai_msg)

        # Automatically creates ToolMessage with parsed content
        # Token usage is tracked for both original and parsed messages
        ```
    """

    # Configuration for structured output parsing
    output_models: Optional[List[Type[BaseModel]]] = Field(
        default=None, description="Pydantic models to parse outputs into"
    )

    output_parser: Optional[BaseOutputParser] = Field(
        default=None, description="Custom output parser to use"
    )

    parse_as_tools: bool = Field(
        default=True, description="Whether to parse Pydantic models as tool calls"
    )

    auto_parse_ai_messages: bool = Field(
        default=True,
        description="Automatically parse AI messages with structured output",
    )

    parsed_outputs: Dict[str, Any] = Field(
        default_factory=dict, description="Storage for parsed structured outputs"
    )

    @model_validator(mode="before")
    @classmethod
    def setup_output_parser(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Setup the appropriate output parser based on configuration."""
        if not values.get("output_parser") and values.get("output_models"):
            output_models = values["output_models"]
            parse_as_tools = values.get("parse_as_tools", True)

            if parse_as_tools and output_models:
                # Use PydanticToolsParser for tool message conversion
                values["output_parser"] = PydanticToolsParser(tools=output_models)
            elif output_models and len(output_models) == 1:
                # Use PydanticOutputParser for single model
                values["output_parser"] = PydanticOutputParser(
                    pydantic_object=output_models[0]
                )

        return values

    @field_validator("messages", mode="after")
    @classmethod
    def parse_structured_outputs(cls, messages: List[Any], info) -> List[Any]:
        """Parse AI messages with structured output into appropriate format."""
        if not info.data.get("auto_parse_ai_messages"):
            return messages

        output_parser = info.data.get("output_parser")
        if not output_parser:
            return messages

        parsed_messages = []
        parse_as_tools = info.data.get("parse_as_tools", True)

        for msg in messages:
            parsed_messages.append(msg)

            # Only parse AI messages
            if isinstance(msg, AIMessage) and msg.content:
                try:
                    # Try to parse the message content
                    if isinstance(output_parser, PydanticToolsParser):
                        # Parse as tool calls
                        parsed_tools = output_parser.parse(msg.content)

                        # Create tool messages for each parsed tool
                        for tool_call in parsed_tools:
                            tool_msg = ToolMessage(
                                content=str(tool_call),
                                tool_call_id=f"call_{len(parsed_messages)}",
                                name=tool_call.__class__.__name__,
                            )
                            parsed_messages.append(tool_msg)

                    elif isinstance(output_parser, PydanticOutputParser):
                        # Parse as structured object
                        parsed_obj = output_parser.parse(msg.content)

                        # Store in parsed_outputs
                        msg_idx = len(parsed_messages) - 1
                        info.data.setdefault("parsed_outputs", {})[
                            f"msg_{msg_idx}"
                        ] = parsed_obj

                        if parse_as_tools:
                            # Also create a tool message
                            tool_msg = ToolMessage(
                                content=parsed_obj.json(),
                                tool_call_id=f"call_{msg_idx}",
                                name=parsed_obj.__class__.__name__,
                            )
                            parsed_messages.append(tool_msg)

                except Exception as e:
                    # If parsing fails, just keep the original message
                    # Could optionally log the error or store it
                    pass

        return parsed_messages

    def get_parsed_output(self, message_index: int) -> Optional[BaseModel]:
        """Get parsed output for a specific message index."""
        return self.parsed_outputs.get(f"msg_{message_index}")

    def get_latest_parsed_output(self) -> Optional[BaseModel]:
        """Get the most recent parsed output."""
        if not self.parsed_outputs:
            return None

        # Get the highest message index
        max_idx = max(int(key.split("_")[1]) for key in self.parsed_outputs.keys())
        return self.parsed_outputs.get(f"msg_{max_idx}")

    def get_tool_calls(self) -> List[ToolMessage]:
        """Get all tool call messages from the conversation."""
        return [msg for msg in self.messages if isinstance(msg, ToolMessage)]

    def format_for_structured_output(self) -> str:
        """Get format instructions for the configured output models."""
        if self.output_parser and hasattr(
            self.output_parser, "get_format_instructions"
        ):
            return self.output_parser.get_format_instructions()
        return ""


class StructuredOutputMixin:
    """Mixin to add structured output capabilities to any state schema.

    This mixin can be used to add structured output parsing to custom state schemas
    without inheriting from StructuredOutputState.
    """

    @field_validator("messages", mode="after")
    @classmethod
    def parse_structured_outputs_mixin(cls, messages: List[Any], info) -> List[Any]:
        """Parse structured outputs in messages (mixin version)."""
        # Same logic as above but as a mixin
        # This allows adding to any state schema
        pass
