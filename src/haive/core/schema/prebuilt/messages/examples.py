"""From typing import Any
Examples demonstrating the use of the enhanced MessagesState system.

This file contains complete examples of common usage patterns, demonstrating
both the classic API and the enhanced features.
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

# The examples below use try/except to handle the case where the enhanced
# features are not available, making them safe to run in any environment


def basic_usage_example() -> Any:
    """Demonstrates basic usage of MessagesState."""
    from haive.core.schema.prebuilt.messages import MessagesState

    # Create a new state
    state = MessagesState()

    # Add a system message
    state.add_system_message("You are a helpful assistant.")

    # Add conversation messages
    state.add_message(HumanMessage(content="Hello, can you help me?"))
    state.add_message(AIMessage(content="Of course! What can I help you with?"))
    state.add_message(HumanMessage(content="I need help with Python."))
    state.add_message(
        AIMessage(
            content="I'd be happy to help with Python. What specific question do you have?"
        )
    )

    # Access messages

    # Format conversion
    state.to_openai_format()

    return state


def tool_usage_example() -> Any:
    """Demonstrates tool usage with MessagesState."""
    from haive.core.schema.prebuilt.messages import MessagesState

    # Create a new state
    state = MessagesState()

    # Add a system message
    state.add_system_message("You are a helpful assistant with tool access.")

    # Add conversation with tool usage
    state.add_message(HumanMessage(content="What's the weather in New York?"))

    # AI message with tool calls
    ai_msg = AIMessage(
        content="I'll check the weather for you.",
        tool_calls=[
            {
                "id": "call_abc123",
                "name": "get_weather",
                "args": {"location": "New York"},
            }
        ],
    )
    state.add_message(ai_msg)

    # Tool response
    tool_msg = ToolMessage(
        content='{"temperature": 72, "condition": "sunny"}', tool_call_id="call_abc123"
    )
    state.add_message(tool_msg)

    # AI response after tool
    state.add_message(
        AIMessage(content="The weather in New York is currently 72°F and sunny.")
    )

    # Check for tool calls

    # Get tool calls from a specific message
    state.get_tool_calls(ai_msg)

    return state


def enhanced_features_example() -> Any:
    """Demonstrates enhanced features of MessagesState."""
    # Create state with conversation including tools
    state = tool_usage_example()

    try:
        # Get conversation rounds
        rounds = state.get_conversation_rounds()
        for _i, _round_info in enumerate(rounds):
            pass

        # Get completed tool calls
        completed_calls = state.get_completed_tool_calls()
        for _call in completed_calls:
            pass

        # Check if messages are from real humans
        for msg in state.messages:
            if isinstance(msg, HumanMessage):
                state.is_real_human_message(msg)

    except NotImplementedError:
        pass

    return state


def agent_handoff_example() -> Any:
    """Demonstrates agent-to-agent handoff with message transformation."""
    from haive.core.schema.prebuilt.messages import MessagesState

    # First agent's state
    agent1 = MessagesState()
    agent1.add_system_message("You are Agent 1, an expert in data analysis.")
    agent1.add_message(HumanMessage(content="Analyze this dataset"))
    agent1.add_message(
        AIMessage(
            content="I found these patterns in the data: high correlation between X and Y"
        )
    )

    for msg in agent1.messages:
        pass

    # Second agent's state
    agent2 = MessagesState()
    agent2.add_system_message("You are Agent 2, an expert in visualization.")

    # Transform and transfer messages
    try:
        # Transform AI messages to human messages for handoff
        agent1.transform_ai_to_human(
            preserve_metadata=True, engine_id="agent1", engine_name="DataAnalysisAgent"
        )

        for msg in agent1.messages:
            pass

        # Copy transformed messages to agent2 (skipping system message)
        for msg in agent1.messages:
            if not isinstance(msg, SystemMessage):
                agent2.add_message(msg)

        for msg in agent2.messages:

            # Check metadata on transferred messages
            if hasattr(msg, "additional_kwargs") and msg.additional_kwargs:
                pass

    except NotImplementedError:
        pass

    return agent1, agent2


def enhanced_implementation_example() -> Any:
    """Demonstrates using the enhanced implementation directly."""
    try:
        from haive.core.schema.prebuilt.messages import EnhancedMessagesState

        # Create new state with the enhanced implementation
        state = EnhancedMessagesState()

        # Add messages
        state.add_system_message("You are a helpful assistant.")
        state.add_message(HumanMessage(content="Hello!"))
        state.add_message(AIMessage(content="Hi there! How can I help you?"))
        state.add_message(HumanMessage(content="Tell me about Python."))
        state.add_message(
            AIMessage(
                content="Python is a programming language known for its readability."
            )
        )

        # Access computed properties

        # Use the message list directly

        # Use filtering
        state.messages.filter_by_type(HumanMessage)

        # Use transformation
        state.transform_for_reflection()
        for _msg in state.messages:
            pass

        return state

    except (ImportError, AttributeError):
        return None


if __name__ == "__main__":
    basic_usage_example()

    tool_usage_example()

    enhanced_features_example()

    agent_handoff_example()

    enhanced_implementation_example()
