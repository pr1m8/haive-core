"""Examples demonstrating callable nodes and schema composition.

This shows:
1. How to wrap simple functions as nodes
2. How to compose schemas from multiple nodes
3. How to handle different state types generically
"""

import logging
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import StateGraph
from pydantic import Field

from haive.core.graph.node.callable_node import (
    CallableNodeConfig,
    as_node,
    wrap_callable,
)
from haive.core.schema import StateSchema

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Example 1: Simple threshold check function
def check_summarization_needed(
    messages: list[BaseMessage], threshold: int = 1000
) -> bool:
    """Check if total message length exceeds threshold."""
    total_length = sum(len(msg.content) for msg in messages)
    logger.info(f"Total message length: {total_length}, threshold: {threshold}")
    return total_length > threshold


# Example 2: Token-aware check
def check_token_limit(token_count: int, max_tokens: int = 4000) -> bool:
    """Check if token count is approaching limit."""
    utilization = token_count / max_tokens
    logger.info(
        f"Token utilization: {
            utilization:.1%} ({token_count}/{max_tokens})"
    )
    return utilization > 0.8  # 80% threshold


# Example 3: Complex state function
class MessagesWithTokenTracking(StateSchema):
    """State with message and token tracking."""

    messages: list[BaseMessage] = Field(default_factory=list)
    token_count: int = Field(default=0)
    total_cost: float = Field(default=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


def needs_summarization(state: MessagesWithTokenTracking) -> bool:
    """Check if state indicates summarization is needed."""
    # Multiple conditions
    conditions = [
        state.token_count > 3000,  # Token threshold
        len(state.messages) > 20,  # Message count threshold
        state.total_cost > 1.0,  # Cost threshold
    ]

    result = any(conditions)
    logger.info(
        f"Summarization check: tokens={
            state.token_count}, messages={
            len(
                state.messages)}, cost=${
                state.total_cost:.2f} -> {result}"
    )
    return result


# Example 4: Generic field extractor
def check_field_threshold(
    value: float, threshold: float, field_name: str = "unknown"
) -> bool:
    """Generic threshold checker for any numeric field."""
    result = value > threshold
    logger.info(f"Field '{field_name}' check: {value} > {threshold} = {result}")
    return result


# Example 5: Multi-value categorizer
def categorize_conversation(messages: list[BaseMessage]) -> str:
    """Categorize conversation based on content."""
    if not messages:
        return "empty"

    # Simple categorization logic
    total_length = sum(len(msg.content) for msg in messages)
    ai_messages = sum(1 for msg in messages if isinstance(msg, AIMessage))

    if total_length < 100:
        return "brief"
    if ai_messages > 10:
        return "extended"
    if total_length > 5000:
        return "verbose"
    return "normal"


# Example 6: Using the decorator
@as_node(
    goto_on_true="summarize_node",
    goto_on_false="continue_conversation",
    result_key="needs_summary",
)
def smart_summary_check(
    messages: list[BaseMessage], token_count: int | None = None
) -> bool:
    """Smart check combining multiple factors."""
    # Length-based check
    total_length = sum(len(msg.content) for msg in messages)

    # Token-based check if available
    if token_count:
        return token_count > 3000 or total_length > 5000

    # Fallback to length only
    return total_length > 5000


def demonstrate_callable_nodes():
    """Demonstrate various callable node patterns."""
    # 1. Simple threshold check
    threshold_node = wrap_callable(
        check_summarization_needed,
        goto_on_true="summarize",
        goto_on_false="continue",
        parameter_mapping={"threshold": "summary_threshold"},
    )

    # Test state
    class SimpleState(StateSchema):
        messages: list[BaseMessage] = Field(default_factory=list)
        summary_threshold: int = Field(default=1000)

    state1 = SimpleState(
        messages=[
            HumanMessage(content="Hello " * 100),
            AIMessage(content="Hi there " * 150),
        ],
        summary_threshold=500,
    )

    threshold_node(state1)

    # 2. Token-based check
    token_node = CallableNodeConfig(
        name="check_tokens",
        callable_func=check_token_limit,
        goto_on_true="reduce_context",
        goto_on_false="continue",
        parameter_mapping={
            "token_count": "current_tokens",
            "max_tokens": "token_limit",
        },
    )

    class TokenState(StateSchema):
        current_tokens: int = Field(default=0)
        token_limit: int = Field(default=4000)

    state2 = TokenState(current_tokens=3500, token_limit=4000)
    token_node(state2)

    # 3. Full state function
    state_node = CallableNodeConfig(
        name="check_summary_needed",
        callable_func=needs_summarization,
        extract_full_state=True,
        goto_on_true="create_summary",
        goto_on_false="continue",
        result_key="summary_needed",
    )

    state3 = MessagesWithTokenTracking(
        messages=[HumanMessage(content="Test")] * 25, token_count=3500, total_cost=0.5
    )

    state_node(state3)

    # 4. Generic field extraction
    cost_check_node = CallableNodeConfig(
        name="check_cost",
        callable_func=check_field_threshold,
        goto_on_true="expensive_path",
        goto_on_false="normal_path",
        parameter_mapping={"value": "total_cost", "threshold": "cost_limit"},
        # Advanced path
        extraction_paths={"field_name": "metadata.cost_field_name"},
    )

    class CostState(StateSchema):
        total_cost: float = Field(default=0.0)
        cost_limit: float = Field(default=5.0)
        metadata: dict[str, Any] = Field(default_factory=dict)

    state4 = CostState(
        total_cost=6.5, cost_limit=5.0, metadata={"cost_field_name": "api_costs"}
    )

    cost_check_node(state4)

    # 5. Multi-value routing
    categorizer_node = CallableNodeConfig(
        name="categorize",
        callable_func=categorize_conversation,
        goto_mapping={
            "empty": "handle_empty",
            "brief": "quick_response",
            "extended": "detailed_analysis",
            "verbose": "summarize_first",
            "normal": "standard_flow",
        },
        default_goto="standard_flow",
        result_key="conversation_type",
    )

    state5 = SimpleState(
        messages=[
            HumanMessage(content="Hello"),
            AIMessage(content="Hi! How can I help you today?"),
        ]
    )

    categorizer_node(state5)


def demonstrate_schema_composition():
    """Demonstrate composing schemas from nodes."""
    # Define multiple nodes with different requirements
    nodes = [
        wrap_callable(
            check_summarization_needed,
            name="length_check",
            goto_on_true="summarize",
            goto_on_false="continue",
        ),
        wrap_callable(
            check_token_limit,
            name="token_check",
            goto_on_true="reduce",
            goto_on_false="continue",
        ),
        CallableNodeConfig(
            name="cost_check",
            callable_func=check_field_threshold,
            goto_on_true="expensive",
            goto_on_false="continue",
            parameter_mapping={"value": "total_cost", "threshold": "cost_limit"},
        ),
    ]

    # Compose required fields from all nodes
    all_fields = set()
    field_sources = {}

    for node in nodes:
        fields = node.get_default_input_fields()
        for field in fields:
            all_fields.add(field.name)
            if field.name not in field_sources:
                field_sources[field.name] = []
            field_sources[field.name].append(node.name)

    for field_name in sorted(all_fields):
        field_sources[field_name]

    # Create composite schema

    # Dynamic schema creation
    from pydantic import create_model

    field_definitions = {
        "messages": (list[BaseMessage], Field(default_factory=list)),
        "threshold": (int, Field(default=1000)),
        "token_count": (int, Field(default=0)),
        "max_tokens": (int, Field(default=4000)),
        "total_cost": (float, Field(default=0.0)),
        "cost_limit": (float, Field(default=5.0)),
    }

    CompositeState = create_model(
        "CompositeState", __base__=StateSchema, **field_definitions
    )

    for _name, field in CompositeState.model_fields.items():
        pass

    # Test with composite state
    composite_state = CompositeState(
        messages=[HumanMessage(content="Test " * 200)], token_count=3500, total_cost=2.5
    )

    for node in nodes:
        node(composite_state)


def create_example_graph():
    """Create a graph using callable nodes."""

    # Define state
    class ConversationState(StateSchema):
        messages: list[BaseMessage] = Field(default_factory=list)
        token_count: int = Field(default=0)
        summary_count: int = Field(default=0)
        should_summarize: bool = Field(default=False)

    # Create graph
    graph = StateGraph(ConversationState)

    # Add callable nodes
    graph.add_node(
        "check_summary",
        wrap_callable(
            check_summarization_needed,
            goto_on_true="summarize",
            goto_on_false="respond",
            result_key="should_summarize",
        ),
    )

    # Mock nodes for complete flow
    graph.add_node("summarize", lambda s: {"summary_count": s.summary_count + 1})
    graph.add_node("respond", lambda s: {"token_count": s.token_count + 100})

    # Define flow
    graph.set_entry_point("check_summary")
    graph.add_edge("summarize", "__end__")
    graph.add_edge("respond", "__end__")

    # Compile
    app = graph.compile()

    return app


if __name__ == "__main__":

    # Run demonstrations
    demonstrate_callable_nodes()
    demonstrate_schema_composition()
    create_example_graph()
