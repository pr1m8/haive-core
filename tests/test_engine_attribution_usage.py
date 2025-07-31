"""Practical usage examples of engine attribution functionality.

This demonstrates how to use engine attribution in real-world scenarios
like debugging, auditing, and routing based on engine source.
"""

import logging
from datetime import datetime
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import Field

from haive.core.schema import StateSchema

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConversationState(StateSchema):
    """State with conversation tracking."""

    messages: list[BaseMessage] = Field(default_factory=list)
    engine_usage: dict[str, int] = Field(
        default_factory=dict, description="Track engine usage counts"
    )
    last_engine: str | None = Field(
        default=None, description="Last engine that responded"
    )


def track_engine_usage(state: ConversationState) -> dict[str, Any]:
    """Analyze engine usage from message attribution."""

    engine_stats = {
        "total_ai_messages": 0,
        "messages_by_engine": {},
        "response_times": {},
        "token_usage": {},
    }

    for msg in state.messages:
        if isinstance(msg, AIMessage):
            engine_stats["total_ai_messages"] += 1

            # Get engine attribution
            engine_name = msg.additional_kwargs.get("engine_name", "unknown")

            # Count messages per engine
            if engine_name not in engine_stats["messages_by_engine"]:
                engine_stats["messages_by_engine"][engine_name] = 0
            engine_stats["messages_by_engine"][engine_name] += 1

            # Track response time if available
            if "response_time" in msg.additional_kwargs:
                if engine_name not in engine_stats["response_times"]:
                    engine_stats["response_times"][engine_name] = []
                engine_stats["response_times"][engine_name].append(
                    msg.additional_kwargs["response_time"]
                )

            # Track token usage if available
            if "token_count" in msg.additional_kwargs:
                if engine_name not in engine_stats["token_usage"]:
                    engine_stats["token_usage"][engine_name] = 0
                engine_stats["token_usage"][engine_name] += msg.additional_kwargs[
                    "token_count"
                ]

    for _engine, _count in engine_stats["messages_by_engine"].items():
        pass

    if engine_stats["response_times"]:
        for _engine, times in engine_stats["response_times"].items():
            sum(times) / len(times)

    if engine_stats["token_usage"]:
        for _engine, _tokens in engine_stats["token_usage"].items():
            pass

    return engine_stats


def route_based_on_engine(messages: list[BaseMessage]) -> str:
    """Route to different handlers based on source engine."""

    # Find the last AI message
    last_ai_msg = None
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            last_ai_msg = msg
            break

    if not last_ai_msg:
        return "default"

    engine_name = last_ai_msg.additional_kwargs.get("engine_name", "unknown")

    # Define routing rules
    routing_rules = {
        "gpt4_engine": "premium_handler",
        "gpt35_engine": "standard_handler",
        "local_llm": "edge_handler",
        "analyzer_engine": "analysis_post_processor",
        "summarizer_engine": "summary_formatter",
    }

    route = routing_rules.get(engine_name, "default")

    return route


def create_audit_log(state: ConversationState) -> list[dict[str, Any]]:
    """Create an audit log with engine attribution."""

    audit_entries = []

    for i, msg in enumerate(state.messages):
        entry = {
            "index": i,
            "timestamp": datetime.now().isoformat(),
            "message_type": type(msg).__name__,
            "content_preview": (
                msg.content[:50] + "..." if len(msg.content) > 50 else msg.content
            ),
        }

        if isinstance(msg, AIMessage):
            # Add engine attribution
            entry["engine_name"] = msg.additional_kwargs.get("engine_name", "unknown")
            entry["engine_id"] = msg.additional_kwargs.get("engine_id")

            # Add any performance metrics
            if "response_time" in msg.additional_kwargs:
                entry["response_time_ms"] = msg.additional_kwargs["response_time"]

            if "token_count" in msg.additional_kwargs:
                entry["tokens_used"] = msg.additional_kwargs["token_count"]

        audit_entries.append(entry)

    for entry in audit_entries:
        if entry["message_type"] == "AIMessage":
            if "response_time_ms" in entry:
                pass
            if "tokens_used" in entry:
                pass
        else:
            pass

    return audit_entries


def filter_messages_by_engine(
    messages: list[BaseMessage], engine_name: str
) -> list[BaseMessage]:
    """Filter messages to only show those from a specific engine."""

    filtered = []

    for msg in messages:
        # Keep all non-AI messages
        if (
            not isinstance(msg, AIMessage)
            or msg.additional_kwargs.get("engine_name") == engine_name
        ):
            filtered.append(msg)

    return filtered


def debug_engine_behavior(state: ConversationState):
    """Debug helper to analyze engine behavior."""

    # Group messages by engine
    engines_data = {}

    for msg in state.messages:
        if isinstance(msg, AIMessage):
            engine = msg.additional_kwargs.get("engine_name", "unknown")
            if engine not in engines_data:
                engines_data[engine] = []
            engines_data[engine].append(msg)

    for engine, messages in engines_data.items():

        # Analyze response patterns
        response_lengths = [len(msg.content) for msg in messages]
        (sum(response_lengths) / len(response_lengths) if response_lengths else 0)

        # Check for any errors or special conditions
        errors = [msg for msg in messages if "error" in msg.additional_kwargs]
        if errors:
            pass


def main():
    """Demonstrate practical usage of engine attribution."""

    # Create a sample conversation with multiple engines
    state = ConversationState(
        messages=[
            HumanMessage(content="What is machine learning?"),
            AIMessage(
                content="Machine learning is a subset of artificial intelligence...",
                additional_kwargs={
                    "engine_name": "gpt4_engine",
                    "engine_id": "eng_001",
                    "response_time": 1250,
                    "token_count": 45,
                },
            ),
            HumanMessage(content="Can you summarize that?"),
            AIMessage(
                content="ML is AI that learns from data to make predictions.",
                additional_kwargs={
                    "engine_name": "summarizer_engine",
                    "engine_id": "eng_002",
                    "response_time": 350,
                    "token_count": 12,
                },
            ),
            HumanMessage(content="Give me a detailed analysis"),
            AIMessage(
                content="Machine learning encompasses various algorithms and techniques...",
                additional_kwargs={
                    "engine_name": "analyzer_engine",
                    "engine_id": "eng_003",
                    "response_time": 2100,
                    "token_count": 156,
                },
            ),
            HumanMessage(content="One more question"),
            AIMessage(
                content="I'd be happy to help with your question...",
                additional_kwargs={
                    "engine_name": "gpt4_engine",
                    "engine_id": "eng_001",
                    "response_time": 890,
                    "token_count": 28,
                },
            ),
        ]
    )

    # Run all examples
    track_engine_usage(state)

    route_based_on_engine(state.messages)

    create_audit_log(state)

    filter_messages_by_engine(state.messages, "gpt4_engine")

    debug_engine_behavior(state)


if __name__ == "__main__":
    main()
