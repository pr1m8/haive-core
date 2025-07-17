"""
Live demonstration of engine attribution in the Haive node system.

This demo shows how engine_name is automatically added to AIMessage.additional_kwargs
when messages are processed through EngineNode.
"""

import logging
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import StateGraph
from pydantic import Field

from haive.core.graph.node.engine_node import EngineNodeConfig
from haive.core.graph.node.message_transformation import (
    MessageTransformationNodeConfig,
    TransformationType,
)
from haive.core.schema import StateSchema

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DemoState(StateSchema):
    """State for our demo with multiple message lists."""

    messages: List[BaseMessage] = Field(
        default_factory=list, description="Main conversation"
    )
    analysis_messages: List[BaseMessage] = Field(
        default_factory=list, description="Analysis results"
    )
    engines: Dict[str, Any] = Field(
        default_factory=dict, description="Engine instances"
    )


def create_demo_graph():
    """Create a graph that demonstrates engine attribution."""

    # Create the graph
    graph = StateGraph(DemoState)

    # Create two different engines (mocked for demo)
    main_engine = type(
        "MockEngine",
        (),
        {
            "name": "main_llm",
            "invoke": lambda self, input_data, config=None: AIMessage(
                content=f"Main LLM response to: {input_data.get('messages', [''])[0].content if isinstance(input_data, dict) and 'messages' in input_data else input_data}"
            ),
        },
    )()

    analyzer_engine = type(
        "MockEngine",
        (),
        {
            "name": "analyzer_llm",
            "invoke": lambda self, input_data, config=None: AIMessage(
                content=f"Analysis of: {input_data.get('messages', [''])[0].content if isinstance(input_data, dict) and 'messages' in input_data else input_data}"
            ),
        },
    )()

    # Create engine nodes
    main_node = EngineNodeConfig(
        name="main_engine_node",
        engine=main_engine,
        input_fields=["messages"],
        output_fields={"messages": "messages"},
    )

    analyzer_node = EngineNodeConfig(
        name="analyzer_node",
        engine=analyzer_engine,
        input_fields=["messages"],
        output_fields={"analysis_messages": "messages"},
    )

    # Create message transformation node
    transformer = MessageTransformationNodeConfig(
        name="transformer",
        transformation_type=TransformationType.AI_TO_HUMAN,
        messages_key="analysis_messages",
        output_key="analysis_messages",
        preserve_metadata=True,  # This ensures engine_name is preserved
    )

    # Add nodes to graph
    graph.add_node("main_engine", main_node)
    graph.add_node("analyzer", analyzer_node)
    graph.add_node("transform", transformer)

    # Define the flow
    graph.set_entry_point("main_engine")
    graph.add_edge("main_engine", "analyzer")
    graph.add_edge("analyzer", "transform")
    graph.add_edge("transform", "__end__")

    return graph.compile()


def analyze_engine_attribution(state: DemoState):
    """Analyze and display engine attribution in messages."""
    print("\n" + "=" * 80)
    print("ENGINE ATTRIBUTION ANALYSIS")
    print("=" * 80)

    # Analyze main messages
    print("\n📨 Main Messages:")
    for i, msg in enumerate(state.messages):
        print(f"\n[{i}] {type(msg).__name__}:")
        print(f"    Content: {msg.content[:100]}...")
        if isinstance(msg, AIMessage) and hasattr(msg, "additional_kwargs"):
            print(f"    Additional kwargs: {msg.additional_kwargs}")
            if "engine_name" in msg.additional_kwargs:
                print(f"    ✅ ENGINE: {msg.additional_kwargs['engine_name']}")

    # Analyze analysis messages
    print("\n📊 Analysis Messages:")
    for i, msg in enumerate(state.analysis_messages):
        print(f"\n[{i}] {type(msg).__name__}:")
        print(f"    Content: {msg.content[:100]}...")
        if hasattr(msg, "additional_kwargs"):
            print(f"    Additional kwargs: {msg.additional_kwargs}")
            if "engine_name" in msg.additional_kwargs:
                print(f"    ✅ ENGINE: {msg.additional_kwargs['engine_name']}")


async def run_demo():
    """Run the engine attribution demo."""
    print("\n" + "🚀" * 40)
    print("HAIVE ENGINE ATTRIBUTION DEMO")
    print("🚀" * 40)

    # Create the graph
    app = create_demo_graph()

    # Create initial state
    initial_state = DemoState(
        messages=[HumanMessage(content="What is the capital of France?")]
    )

    print("\n📝 Initial State:")
    print(f"Messages: {[msg.content for msg in initial_state.messages]}")

    # Run the graph
    print("\n⚙️  Running the graph...")
    try:
        final_state = await app.ainvoke(initial_state)

        # Analyze the results
        analyze_engine_attribution(final_state)

        # Show the attribution in action
        print("\n" + "=" * 80)
        print("DEMONSTRATION RESULTS:")
        print("=" * 80)

        # Count messages by engine
        engine_count = {}
        for msg in final_state.messages + final_state.analysis_messages:
            if isinstance(msg, AIMessage) and hasattr(msg, "additional_kwargs"):
                engine_name = msg.additional_kwargs.get(
                    "engine_name", "unknown")
                engine_count[engine_name] = engine_count.get(
                    engine_name, 0) + 1

        print("\n📊 Message Count by Engine:")
        for engine, count in engine_count.items():
            print(f"   - {engine}: {count} messages")

        print("\n✅ Demo Complete!")

    except Exception as e:
        print(f"\n❌ Error running demo: {e}")
        import traceback

        traceback.print_exc()


# Direct test of the attribution mechanism
def test_direct_attribution():
    """Direct test of the _add_engine_attribution_to_message method."""
    print("\n" + "=" * 80)
    print("DIRECT ATTRIBUTION TEST")
    print("=" * 80)

    from haive.core.graph.node.engine_node import EngineNodeConfig

    # Create an engine node
    node = EngineNodeConfig(name="test_node")

    # Mock engine
    node.engine = type("MockEngine", (), {"name": "test_engine_v1"})()

    # Create a test message
    original_msg = AIMessage(
        content="Test response", additional_kwargs={"custom_field": "value"}
    )

    print("\n📝 Original Message:"e: ")
    print(f"   Content: {original_msg.content}")
    print(f"   Additional kwargs: {original_msg.additional_kwargs}")

    # Add attribution
    attributed_msg = node._add_engine_attribution_to_message(original_msg)

    print("\n✅ After Attribution:":")
    print(f"   Content: {attributed_msg.content}")
    print(f"   Additional kwargs: {attributed_msg.additional_kwargs}")
    print(
        f"   Engine name: {attributed_msg.additional_kwargs.get('engine_name', 'NOT FOUND')}"
    )

    # Verify
    assert attributed_msg.additional_kwargs["engine_name"] == "test_engine_v1"
    assert attributed_msg.additional_kwargs["custom_field"] == "value"
    print("\n✅ Attribution test passed!")


if __name__ == "__main__":
    print("\nRunning Engine Attribution Demo...")

    # First run the direct test
    test_direct_attribution()

    # Then run the full demo
    # asyncio.run(run_demo())

    print("\n✅ All demonstrations complete!")
