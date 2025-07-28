"""Demonstrate schema composition from nodes and callable wrapping.

This shows how to:
1. Compose schemas from multiple nodes
2. Wrap simple functions as nodes
3. Handle generic state extraction
"""

import logging
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import Field, create_model

from haive.core.graph.node.callable_node import CallableNodeConfig
from haive.core.schema import StateSchema

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Example state schemas
class MessagesState(StateSchema):
    """Basic state with messages."""

    messages: list[BaseMessage] = Field(default_factory=list)


class TokenTrackingState(MessagesState):
    """State that tracks tokens."""

    token_count: int = Field(default=0)
    max_tokens: int = Field(default=4000)


class AdvancedState(TokenTrackingState):
    """State with additional tracking."""

    total_cost: float = Field(default=0.0)
    summary_count: int = Field(default=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


# Example threshold check functions
def check_summarization_threshold(
    messages: list[BaseMessage], threshold: int = 1000
) -> bool:
    """Check if messages exceed length threshold."""
    total_length = sum(len(msg.content) for msg in messages)
    logger.info(
        f"Message length check: {total_length} > {threshold} = {
            total_length > threshold}"
    )
    return total_length > threshold


def check_token_threshold(token_count: int, max_tokens: int) -> bool:
    """Check if approaching token limit."""
    ratio = token_count / max_tokens if max_tokens > 0 else 0
    logger.info(f"Token check: {token_count}/{max_tokens} = {ratio:.1%}")
    return ratio > 0.8


def check_cost_threshold(total_cost: float, cost_limit: float = 5.0) -> bool:
    """Check if cost exceeds limit."""
    logger.info(
        f"Cost check: ${
            total_cost:.2f} > ${
            cost_limit:.2f} = {
            total_cost > cost_limit}"
    )
    return total_cost > cost_limit


# Import the proper CallableNodeConfig instead of creating custom classes


def demonstrate_schema_composition():
    """Show how to compose schemas from node requirements."""
    # Define nodes with their parameter requirements
    nodes = [
        {
            "name": "length_check",
            "func": check_summarization_threshold,
            "params": {"messages": "messages", "threshold": "summary_threshold"},
            "required_fields": [
                ("messages", list[BaseMessage], Field(default_factory=list)),
                ("summary_threshold", int, Field(default=1000)),
            ],
        },
        {
            "name": "token_check",
            "func": check_token_threshold,
            "params": {"token_count": "token_count", "max_tokens": "max_tokens"},
            "required_fields": [
                ("token_count", int, Field(default=0)),
                ("max_tokens", int, Field(default=4000)),
            ],
        },
        {
            "name": "cost_check",
            "func": check_cost_threshold,
            "params": {"total_cost": "total_cost", "cost_limit": "cost_limit"},
            "required_fields": [
                ("total_cost", float, Field(default=0.0)),
                ("cost_limit", float, Field(default=5.0)),
            ],
        },
    ]

    # Collect all required fields
    all_fields = {}
    for node_def in nodes:
        for field_name, field_type, field_default in node_def["required_fields"]:
            if field_name not in all_fields:
                all_fields[field_name] = (field_type, field_default)

    # Create composite schema dynamically
    CompositeState = create_model(
        "CompositeState",
        __base__=StateSchema,
        **all_fields)

    for _name, _field_info in CompositeState.model_fields.items():
        pass

    # Test the composite state
    state = CompositeState(
        messages=[
            HumanMessage(content="Hello " * 100),
            AIMessage(content="Response " * 150),
        ],
        token_count=3500,
        total_cost=2.5,
    )

    # Create and test nodes
    for node_def in nodes:
        node = CallableNodeConfig(
            name=node_def["name"],
            callable_func=node_def["func"],
            parameter_mapping=node_def["params"],
            goto_on_true="handle_threshold_exceeded",
            goto_on_false="continue_normal",
        )

        node(state)


def demonstrate_generic_extraction():
    """Show generic parameter extraction from different state types."""

    # Generic function that works with any state having messages
    def message_density_check(
        messages: list[BaseMessage], message_threshold: int = 10
    ) -> str:
        """Categorize based on message density."""
        if not messages:
            return "empty"
        if len(messages) < message_threshold:
            return "sparse"
        avg_length = sum(len(m.content) for m in messages) / len(messages)
        if avg_length < 50:
            return "brief"
        if avg_length > 200:
            return "verbose"
        return "normal"

    # Create wrapper that works with different state types
    density_node = CallableNodeConfig(
        name="density_check",
        callable_func=message_density_check,
        parameter_mapping={
            "messages": "messages",
            # Use default parameter instead of nested
            "message_threshold": "message_threshold",
        },
        result_key="conversation_density",
    )

    # Test with different state types
    state1 = MessagesState(
        messages=[HumanMessage(content="Hi"), AIMessage(content="Hello!")]
    )
    density_node(state1)

    state2 = AdvancedState(
        messages=[HumanMessage(content="Long message " * 50)] * 15,
        metadata={"message_threshold": 5},
    )
    density_node(state2)


def demonstrate_conditional_routing():
    """Show complex conditional routing based on multiple factors."""

    # Multi-factor routing function
    def determine_next_action(
        token_count: int,
        messages: list[BaseMessage],
        total_cost: float,
        summary_count: int = 0,
    ) -> str:
        """Determine next action based on multiple factors."""
        # Complex routing logic
        if token_count > 3500:
            return "summarize_urgent"
        if total_cost > 10.0:
            return "switch_to_cheaper_model"
        if len(messages) > 50 and summary_count == 0:
            return "summarize_recommended"
        if len(messages) < 3:
            return "gather_more_context"
        return "continue_conversation"

    # Create routing node
    router_node = CallableNodeConfig(
        name="smart_routef",
        callable_func=determine_next_action,
        parameter_mapping={
            "token_count": "token_count",
            "messages": "messages",
            "total_cost": "total_cost",
            "summary_count": "summary_count",
        },
        result_key="next_action",
    )

    # Test different scenarios
    scenarios = [
        (
            "High tokens",
            AdvancedState(
                token_count=4000, messages=[HumanMessage("Hi")] * 10, total_cost=2.0
            ),
        ),
        (
            "High cost",
            AdvancedState(
                token_count=1000, messages=[HumanMessage("Hi")] * 10, total_cost=15.0
            ),
        ),
        (
            "Many messages",
            AdvancedState(
                token_count=1000, messages=[HumanMessage("Hi")] * 60, total_cost=2.0
            ),
        ),
        (
            "Few messages",
            AdvancedState(
                token_count=100, messages=[HumanMessage("Hi")], total_cost=0.1
            ),
        ),
    ]

    for _scenario_name, state in scenarios:
        router_node(state)


def demonstrate_state_agnostic_wrapper():
    """Show how to make truly state-agnostic callable nodes."""

    # Function that works with any state having required fields
    def universal_threshold_check(value: float, threshold: float) -> bool:
        """Universal threshold checker."""
        return value > threshold

    # Create reusable nodes for different fields using CallableNodeConfig
    token_threshold = CallableNodeConfig(
        name="token_threshold",
        callable_func=universal_threshold_check,
        parameter_mapping={
            "value": "token_count",
            "threshold": "limits.max_tokens"},
        extraction_paths={
            "threshold": "limits.max_tokens"  # Use advanced path extraction
        },
        goto_on_true="threshold_exceeded",
        goto_on_false="continue",
        result_key="check_result",
    )

    cost_threshold = CallableNodeConfig(
        name="cost_threshold",
        callable_func=universal_threshold_check,
        parameter_mapping={
            "value": "total_cost",
            "threshold": "limits.max_cost"},
        extraction_paths={
            "threshold": "limits.max_cost"  # Use advanced path extraction
        },
        goto_on_true="threshold_exceeded",
        goto_on_false="continue",
        result_key="check_result",
    )

    # Test with different state structures
    class StateWithLimits(StateSchema):
        token_count: int = Field(default=0)
        total_cost: float = Field(default=0.0)
        limits: dict[str, float] = Field(
            default_factory=lambda: {"max_tokens": 4000, "max_cost": 10.0}
        )

    state = StateWithLimits(token_count=3500, total_cost=12.5)

    token_threshold(state)

    cost_threshold(state)


if __name__ == "__main__":

    demonstrate_schema_composition()
    demonstrate_generic_extraction()
    demonstrate_conditional_routing()
    demonstrate_state_agnostic_wrapper()
