"""Simplified test for engine attribution functionality.

This test demonstrates the engine attribution mechanism by directly
testing the relevant methods without requiring a full LLM execution.
"""

import logging
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.types import Command
from pydantic import BaseModel, Field

from haive.core.schema import StateSchema

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestState(StateSchema):
    """Test state with messages field."""

    messages: List[BaseMessage] = Field(
        default_factory=list, description="Conversation messages"
    )
    transformed_messages: List[BaseMessage] = Field(
        default_factory=list, description="Transformed messages"
    )


def test_engine_attribution_mechanism():
    """Test the core engine attribution mechanism."""
    print("\n" + "=" * 80)
    print("TEST: Engine Attribution Mechanism")
    print("=" * 80)

    # Import the engine node config
    from haive.core.graph.node.engine_node import EngineNodeConfig

    # Create a mock engine node config
    engine_node = EngineNodeConfig(name="test_node", engine_name="test_engine")

    # Test the _add_engine_attribution_to_message method directly
    print("\n1. Testing _add_engine_attribution_to_message method")

    # Create a test AI message
    ai_message = AIMessage(
        content="This is a test response", additional_kwargs={"existing_field": "value"}
    )

    print(f"   Original message additional_kwargs: {ai_message.additional_kwargs}")

    # Mock the engine attribute on the node
    class MockEngine:
        name = "test_llm_engine"

    engine_node.engine = MockEngine()

    # Call the attribution method
    attributed_message = engine_node._add_engine_attribution_to_message(ai_message)

    print(
        f"   After attribution additional_kwargs: {attributed_message.additional_kwargs}"
    )

    # Verify attribution was added
    assert isinstance(attributed_message, AIMessage)
    assert "engine_name" in attributed_message.additional_kwargs
    assert attributed_message.additional_kwargs["engine_name"] == "test_llm_engine"
    assert (
        attributed_message.additional_kwargs["existing_field"] == "value"
    )  # Original data preserved

    print(f"\n✅ SUCCESS: Engine attribution added correctly")
    print(f"   engine_name = '{attributed_message.additional_kwargs['engine_name']}'")


def test_message_transformation_preservation():
    """Test that message transformation preserves engine attribution."""
    print("\n" + "=" * 80)
    print("TEST: Message Transformation Preserves Attribution")
    print("=" * 80)

    from haive.core.graph.node.message_transformation import (
        MessageTransformationNodeConfig,
        TransformationType,
    )

    # Create a message with engine attribution
    ai_message = AIMessage(
        content="AI response content",
        additional_kwargs={"engine_name": "source_engine", "custom_data": "preserved"},
    )

    # Create test state
    state = TestState(messages=[HumanMessage(content="User question"), ai_message])

    print(
        f"\n1. Original AI message has engine_name: '{ai_message.additional_kwargs.get('engine_name')}'"
    )

    # Test the _transform_ai_to_human method directly
    transformer = MessageTransformationNodeConfig(
        name="test_transformer",
        transformation_type=TransformationType.AI_TO_HUMAN,
        preserve_metadata=True,
    )

    # Transform the messages
    transformed = transformer._transform_ai_to_human(state.messages)

    print(f"2. Transformed {len(transformed)} messages")

    # Check the transformed AI message (now Human)
    transformed_msg = transformed[1]
    print(f"3. Transformed message type: {type(transformed_msg).__name__}")
    print(
        f"4. Transformed message additional_kwargs: {transformed_msg.additional_kwargs}"
    )

    # Verify preservation
    assert isinstance(transformed_msg, HumanMessage)
    assert transformed_msg.additional_kwargs.get("engine_name") == "source_engine"
    assert transformed_msg.additional_kwargs.get("custom_data") == "preserved"

    print(f"\n✅ SUCCESS: Attribution preserved during transformation")


def test_complete_attribution_flow():
    """Test the complete attribution flow."""
    print("\n" + "=" * 80)
    print("TEST: Complete Attribution Flow")
    print("=" * 80)

    from haive.core.graph.node.engine_node import EngineNodeConfig
    from haive.core.graph.node.message_transformation import (
        MessageTransformationNodeConfig,
        TransformationType,
    )

    # Step 1: Simulate engine node adding attribution
    print("\n1. Simulating engine node attribution")

    # Create an AI message as if from an engine
    ai_message = AIMessage(content="Engine response")

    # Create engine node
    engine_node = EngineNodeConfig(name="engine_node")

    # Mock engine
    class MockEngine:
        name = "gpt4_engine"

    engine_node.engine = MockEngine()

    # Add attribution
    attributed_msg = engine_node._add_engine_attribution_to_message(ai_message)

    print(
        f"   Added engine_name: {attributed_msg.additional_kwargs.get('engine_name')}"
    )

    # Step 2: Add engine ID through transformation
    print("\n2. Adding engine_id through transformation")

    state = TestState(messages=[HumanMessage(content="Query"), attributed_msg])

    id_transformer = MessageTransformationNodeConfig(
        name="id_adder",
        transformation_type=TransformationType.ADD_ENGINE_ID,
        engine_id="engine_123",
        engine_name="gpt4_engine",  # Can reinforce or override
    )

    # Apply transformation
    messages_with_id = id_transformer._add_engine_id(state.messages)
    final_msg = messages_with_id[-1]

    print(f"   Final additional_kwargs: {final_msg.additional_kwargs}")

    # Verify both attributions
    assert final_msg.additional_kwargs.get("engine_name") == "gpt4_engine"
    assert final_msg.additional_kwargs.get("engine_id") == "engine_123"

    # Step 3: Test reflection preserves all metadata
    print("\n3. Testing reflection transformation")

    reflector = MessageTransformationNodeConfig(
        name="reflector",
        transformation_type=TransformationType.REFLECTION,
        preserve_metadata=True,
        preserve_first_message=False,
    )

    reflected = reflector._transform_reflection(messages_with_id)
    reflected_msg = reflected[-1]

    print(f"   Reflected message type: {type(reflected_msg).__name__}")
    print(f"   Reflected additional_kwargs: {reflected_msg.additional_kwargs}")

    # Verify metadata preserved through reflection
    assert isinstance(reflected_msg, HumanMessage)  # AI → Human in reflection
    assert reflected_msg.additional_kwargs.get("engine_name") == "gpt4_engine"
    assert reflected_msg.additional_kwargs.get("engine_id") == "engine_123"

    print(f"\n✅ SUCCESS: Complete attribution flow works correctly")


def test_multiple_engine_attribution():
    """Test attribution from multiple engines."""
    print("\n" + "=" * 80)
    print("TEST: Multiple Engine Attribution")
    print("=" * 80)

    from haive.core.graph.node.engine_node import EngineNodeConfig

    # Create two engine nodes with different engines
    class MockEngine1:
        name = "analyzer_engine"

    class MockEngine2:
        name = "summarizer_engine"

    node1 = EngineNodeConfig(name="analyzer")
    node1.engine = MockEngine1()

    node2 = EngineNodeConfig(name="summarizer")
    node2.engine = MockEngine2()

    # Create messages from each engine
    msg1 = AIMessage(content="Analysis result")
    msg2 = AIMessage(content="Summary result")

    # Add attribution from each engine
    attributed1 = node1._add_engine_attribution_to_message(msg1)
    attributed2 = node2._add_engine_attribution_to_message(msg2)

    print(f"\n1. First engine attribution: {attributed1.additional_kwargs}")
    print(f"2. Second engine attribution: {attributed2.additional_kwargs}")

    # Verify different attributions
    assert attributed1.additional_kwargs["engine_name"] == "analyzer_engine"
    assert attributed2.additional_kwargs["engine_name"] == "summarizer_engine"

    # Create a conversation with both
    messages = [
        HumanMessage(content="Analyze this"),
        attributed1,
        HumanMessage(content="Now summarize"),
        attributed2,
    ]

    # Count by engine
    engine_counts = {}
    for msg in messages:
        if isinstance(msg, AIMessage) and "engine_name" in msg.additional_kwargs:
            engine = msg.additional_kwargs["engine_name"]
            engine_counts[engine] = engine_counts.get(engine, 0) + 1

    print(f"\n3. Engine message counts: {engine_counts}")

    assert engine_counts["analyzer_engine"] == 1
    assert engine_counts["summarizer_engine"] == 1

    print(f"\n✅ SUCCESS: Multiple engines properly attributed")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("RUNNING SIMPLIFIED ENGINE ATTRIBUTION TESTS")
    print("=" * 80)

    try:
        test_engine_attribution_mechanism()
        test_message_transformation_preservation()
        test_complete_attribution_flow()
        test_multiple_engine_attribution()

        print("\n" + "=" * 80)
        print("ALL TESTS PASSED! ✅")
        print("=" * 80)

        print("\nSUMMARY:")
        print(
            "- EngineNode._add_engine_attribution_to_message() adds engine_name to AIMessage.additional_kwargs"
        )
        print(
            "- MessageTransformationNode preserves additional_kwargs when preserve_metadata=True"
        )
        print(
            "- Attribution survives through various transformations (AI→Human, reflection, etc.)"
        )
        print(
            "- Multiple engines can be distinguished by their engine_name attribution"
        )
        print(
            "- The complete flow enables tracking which engine generated each message"
        )

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        raise
