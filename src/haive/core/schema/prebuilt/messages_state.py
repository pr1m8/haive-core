from typing import Annotated, Any, ClassVar, Dict, List, Optional, Sequence, Type, Union

import tiktoken
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    convert_to_messages,
    filter_messages,
    get_buffer_string,
)
from langchain_core.messages.utils import (
    convert_to_openai_messages,
    messages_from_dict,
)
from langgraph.graph import add_messages
from langgraph.types import Send
from pydantic import Field, model_validator

from haive.core.schema.state_schema import StateSchema


class MessagesState(StateSchema):
    """
    State schema for conversation management with LangChain integration.

    Provides message handling, filtering, and support for tools and system messages.
    Designed to work seamlessly with LangGraph for agent workflows.
    """

    # Core messages field with reducer annotation
    messages: Annotated[List[BaseMessage], add_messages] = Field(
        default_factory=list, description="Conversation messages"
    )

    # Configuration for LangGraph compatibility
    __shared_fields__ = ["messages"]
    __serializable_reducers__ = {"messages": "add_messages"}
    __reducer_fields__ = {"messages": add_messages}

    # Proper class variable (not a model field)
    tokenizer: ClassVar = tiktoken.get_encoding("cl100k_base")

    @model_validator(mode="before")
    def validate_message_format(cls, data: Any) -> Any:
        """Automatically convert message dicts to proper Message objects"""
        if isinstance(data, dict) and "messages" in data:
            data["messages"] = convert_to_messages(data["messages"])
        return data

    # Basic message handling

    def add_message(self, message: Union[BaseMessage, Dict]) -> None:
        """Add a message to the conversation."""
        if isinstance(message, dict):
            message = messages_from_dict([message])[0]
        self.messages.append(message)

    def get_last_message(self) -> Optional[BaseMessage]:
        """Get the last message in the conversation."""
        if not self.messages:
            return None
        return self.messages[-1]

    # System message handling

    def add_system_message(
        self, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a system message at the beginning of the conversation.
        If a system message already exists, it will be replaced.
        """
        # Create system message
        kwargs = {"content": content}
        if metadata:
            kwargs["additional_kwargs"] = metadata

        system_msg = SystemMessage(**kwargs)

        # Check if there are existing system messages
        system_msgs = self.get_filtered_messages(include_types=[SystemMessage])

        if system_msgs:
            # Remove all existing system messages
            for msg in system_msgs:
                if msg in self.messages:
                    self.messages.remove(msg)

        # Add at the beginning
        self.messages.insert(0, system_msg)

    def get_system_message(self) -> Optional[SystemMessage]:
        """Get the first system message if one exists."""
        system_msgs = self.get_filtered_messages(include_types=[SystemMessage])
        return system_msgs[0] if system_msgs else None

    # Message filtering

    def get_filtered_messages(self, **filter_kwargs) -> List[BaseMessage]:
        """
        Filter messages using LangChain's built-in filter_messages utility.

        Args:
            **filter_kwargs: Arguments for LangChain's filter_messages
                - include_types: List of message types to include
                - exclude_types: List of message types to exclude
                - include_names: List of message names to include
                - exclude_names: List of message names to exclude
                - include_ids: List of message IDs to include
                - exclude_ids: List of message IDs to exclude
                - exclude_tool_calls: Tool call IDs to exclude
        """
        # Extract limit parameter if provided (not supported by filter_messages)
        limit = filter_kwargs.pop("limit", None)

        # Apply filter_messages with supported parameters
        filtered_messages = filter_messages(self.messages, **filter_kwargs)

        # Apply limit manually if specified
        if limit is not None and limit > 0:
            return filtered_messages[-limit:]

        return filtered_messages

    # Type-specific message getters

    def get_last_human_message(self) -> Optional[HumanMessage]:
        """Get the last human message."""
        human_msgs = self.get_filtered_messages(include_types=[HumanMessage])
        return human_msgs[-1] if human_msgs else None

    def get_last_ai_message(self) -> Optional[AIMessage]:
        """Get the last AI message."""
        ai_msgs = self.get_filtered_messages(include_types=[AIMessage])
        return ai_msgs[-1] if ai_msgs else None

    def get_last_tool_message(self) -> Optional[ToolMessage]:
        """Get the last tool message."""
        tool_msgs = self.get_filtered_messages(include_types=[ToolMessage])
        return tool_msgs[-1] if tool_msgs else None

    # Simple message type checks

    def is_last_message_from_ai(self) -> bool:
        """Check if the last message is from the AI."""
        last_msg = self.get_last_message()
        return last_msg is not None and last_msg.type == "ai"

    def is_last_message_from_human(self) -> bool:
        """Check if the last message is from a human."""
        last_msg = self.get_last_message()
        return last_msg is not None and last_msg.type == "human"

    def is_last_message_from_tool(self) -> bool:
        """Check if the last message is from a tool."""
        last_msg = self.get_last_message()
        return last_msg is not None and last_msg.type == "tool"

    # Tool-related utilities

    def has_tool_calls(self) -> bool:
        """Check if the last AI message has tool calls."""
        last_ai = self.get_last_ai_message()
        if not last_ai:
            return False

        tool_calls = getattr(last_ai, "tool_calls", None)
        if tool_calls:
            return bool(tool_calls)

        return bool(getattr(last_ai, "additional_kwargs", {}).get("tool_calls"))

    def get_tool_calls(self, message: Optional[AIMessage] = None) -> List[Dict]:
        """
        Get tool calls from an AI message.

        Args:
            message: The AI message to extract tool calls from (defaults to last AI message)

        Returns:
            List of tool call dictionaries
        """
        msg = message or self.get_last_ai_message()
        if not msg:
            return []

        # Check direct tool_calls attribute
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            return msg.tool_calls

        # Check in additional_kwargs
        if hasattr(msg, "additional_kwargs") and msg.additional_kwargs.get(
            "tool_calls"
        ):
            return msg.additional_kwargs["tool_calls"]

        return []

    def inject_state_into_tool_calls(
        self, tool_calls: List[Dict], keys: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Inject state data into tool call arguments.

        Args:
            tool_calls: List of tool call dictionaries
            keys: Optional list of state keys to inject (defaults to all)

        Returns:
            Modified tool calls with injected state
        """
        if not tool_calls:
            return []

        # Prepare state dict with only specified keys
        if keys is not None:
            state_dict = {k: getattr(self, k) for k in keys if hasattr(self, k)}
        else:
            # Default to a dictionary of the state
            state_dict = self.model_dump()

        # Inject state into each tool call
        injected_calls = []
        for call in tool_calls:
            # Make a copy to avoid modifying the original
            call_copy = call.copy()

            # Ensure args is a dictionary
            if "args" not in call_copy or not isinstance(call_copy["args"], dict):
                call_copy["args"] = {}

            # Add state key if it doesn't exist
            if "_state" not in call_copy["args"]:
                call_copy["args"]["_state"] = state_dict

            injected_calls.append(call_copy)

        return injected_calls

    def send_tool_calls(self, node_name: str = "tools") -> Union[str, List[Send]]:
        """
        Convert tool calls from the last AI message into Send objects for LangGraph routing.

        Args:
            node_name: The name of the node to send tool calls to

        Returns:
            Either a string (if no tool calls) or a list of Send objects
        """
        tool_calls = self.get_tool_calls()
        if not tool_calls:
            return "END"

        # Inject state into tool calls
        injected_calls = self.inject_state_into_tool_calls(tool_calls)

        # Create a Send object for each tool call
        return [Send(node_name, tool_call) for tool_call in injected_calls]

    def decide_next_node(self) -> Union[str, List[Send]]:
        """
        Decide which node to go to next based on the last message.

        Returns:
            Either a string node name or a list of Send objects for parallel tool execution
        """
        last_msg = self.get_last_message()

        if not last_msg:
            return "START"

        # Check for AI message with tool calls
        if last_msg.type == "ai":
            tool_calls = self.get_tool_calls(last_msg)
            if tool_calls:
                # Send each tool call to the tools node
                return self.send_tool_calls("tools")
            # No tool calls, we're done
            return "END"

        # Check for tool message with error
        if last_msg.type == "tool" and getattr(last_msg, "additional_kwargs", {}).get(
            "is_error"
        ):
            return "handle_error"

        # Check for human message
        if last_msg.type == "human":
            return "process_input"

        # Default case
        return "continue"

    # Format conversion

    def to_openai_format(self) -> List[Dict]:
        """Convert messages to OpenAI API format."""
        return convert_to_openai_messages(self.messages)

    def to_langchain_prompt(self) -> str:
        """Convert message history to LangChain compatible prompt string."""
        return get_buffer_string(
            self.messages, human_prefix="User", ai_prefix="Assistant"
        )

    # Static constructors

    @classmethod
    def from_dict(cls, data: Union[List[Dict], Dict[str, Any]]) -> "MessagesState":
        """Create instance from dictionary format."""
        if isinstance(data, dict) and "messages" in data:
            messages = convert_to_messages(data["messages"])
            return cls(messages=messages)
        else:
            messages = convert_to_messages(data)
            return cls(messages=messages)

    @classmethod
    def with_system_message(cls, system_content: str) -> "MessagesState":
        """Create a new state with a system message."""
        state = cls()
        state.add_system_message(system_content)
        return state
