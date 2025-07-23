"""Test engine attribution functionality in the node system.

This test demonstrates:
1. Creating an EngineNode with an AugLLMConfig engine
2. Processing a message through the node
3. Verifying that the resulting AIMessage has engine_name in additional_kwargs
4. Showing how MessageTransformationNode preserves this attribution
5. Demonstrating the complete flow
"""

import logging
from contextlib import contextmanager
from typing import List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.types import Command
from pydantic import Field

from haive.core.engine.aug_llm import AugLLMConfig
from haive.core.graph.node.engine_node import EngineNodeConfig
from haive.core.graph.node.message_transformation import (
    MessageTransformationNodeConfig,
    TransformationType,
)
from haive.core.schema import StateSchema

# Configure logging for better visibility
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Patch logger.track_time to work around missing method
@contextmanager
def track_time_noop(self, msg):
    """No-op context manager for track_time."""
    yield


def log_exception_noop(self, exception, msg):
    """No-op method for log_exception."""
    self.exception(f"{msg}: {exception}")


# Monkey-patch the missing methods
logging.Logger.track_time = track_time_noop
logging.Logger.log_exception = log_exception_noop


class TestState(StateSchema):
    """Test state with messages field."""

    messages: List[BaseMessage] = Field(
        default_factory=list, description="Conversation messages"
    )
    transformed_messages: List[BaseMessage] = Field(
        default_factory=list, description="Transformed messages"
    )


class TestEngineAttribution:
    """Test suite for engine attribution functionality."""

    def test_engine_node_adds_attribution(self):
        """Test that EngineNode adds engine_name to AIMessage additional_kwargs."""
        print("\n" + "=" * 80)
        print("TEST: Engine Node Attribution")
        print("=" * 80)

        # Create a test engine with a specific name
        engine_name = "test_llm_engine"
        engine = AugLLMConfig(
            name=engine_name,
            system_message="You are a helpful assistant.",
            temperature=0.1,  # Low temperature for consistent responses
        )

        # Create an EngineNode with the engine
        engine_node = EngineNodeConfig(
            name="test_engine_node", engine=engine, command_goto="next"
        )

        # Create test state with a user message
        state = TestState(
            messages=[HumanMessage(
                content="Hello, please respond with a greeting.")]
        )

        print(f"\n1. Created engine with name: '{engine_name}'")
        print(f"2. Initial state has {len(state.messages)} message(s)")

        # Execute the engine node
        print("\n3. Executing engine node...")
        result = engine_node(state)

        # Verify the result is a Command
        assert isinstance(
            result, Command), f"Expected Command, got {type(result)}"
        assert "messages" in result.update, "Expected 'messages' in update"

        # Get the updated messages
        updated_messages = result.update["messages"]
        assert len(updated_messages) > len(
            state.messages
        ), "Expected new message to be added"

        # Get the last message (should be the AI response)
        last_message = updated_messages[-1]
        print(f"\n4. Result contains {len(updated_messages)} messages")
        print(f"5. Last message type: {type(last_message).__name__}")

        # Verify it's an AIMessage with engine attribution
        assert isinstance(
            last_message, AIMessage
        ), f"Expected AIMessage, got {type(last_message)}"

        # Check for engine_name in additional_kwargs
        assert hasattr(
            last_message, "additional_kwargs"
        ), "AIMessage should have additional_kwargs"
        assert (
            "engine_name" in last_message.additional_kwargs
        ), "Expected 'engine_name' in additional_kwargs"
        assert (
            last_message.additional_kwargs["engine_name"] == engine_name
        ), f"Expected engine_name to be '{engine_name}', got '{last_message.additional_kwargs['engine_name']}'"

        print(
            f"\n✅ SUCCESS: AIMessage has engine_name = '{
    last_message.additional_kwargs['engine_name']}'"
        )
        print(f"   Message content: {last_message.content[:100]}...")
        print(f"   Additional kwargs: {last_message.additional_kwargs}")

    def test_message_transformation_preserves_attribution(self):
        """Test that MessageTransformationNode preserves engine attribution."""
        print("\n" + "=" * 80)
        print("TEST: Message Transformation Preserves Attribution")
        print("=" * 80)

        # Create test state with an AI message that has engine attribution
        engine_name = "source_engine"
        ai_message = AIMessage(
            content="This is a response from the AI.",
            additional_kwargs={
                "engine_name": engine_name,
                "custom_field": "test_value",
            },
        )

        state = TestState(
            messages=[HumanMessage(content="Initial request"), ai_message]
        )

        print(
            f"\n1. Initial AI message has engine_name: '{
    ai_message.additional_kwargs.get('engine_name')}'"
        )
        print(f"2. Initial additional_kwargs: {ai_message.additional_kwargs}")

        # Create a message transformation node (AI to Human)
        transformer = MessageTransformationNodeConfig(
            name="test_transformer",
            transformation_type=TransformationType.AI_TO_HUMAN,
            messages_key="messages",
            output_key="transformed_messages",
            preserve_metadata=True,  # This should preserve additional_kwargs
            command_goto="next",
        )

        # Execute the transformation
        print("\n3. Executing message transformation (AI → Human)...")
        result = transformer(state)

        # Verify the result
        assert isinstance(
            result, Command), f"Expected Command, got {type(result)}"
        assert (
            "transformed_messages" in result.update
        ), "Expected 'transformed_messages' in update"

        # Get the transformed messages
        transformed_messages = result.update["transformed_messages"]
        assert (
            len(transformed_messages) == 2
        ), f"Expected 2 messages, got {len(transformed_messages)}"

        # Check the transformed message (should be the second one)
        transformed_msg = transformed_messages[1]
        print(
            f"\n4. Transformed message type: {type(transformed_msg).__name__}")

        # Verify it's now a HumanMessage but retains attribution
        assert isinstance(
            transformed_msg, HumanMessage
        ), f"Expected HumanMessage, got {type(transformed_msg)}"
        assert hasattr(
            transformed_msg, "additional_kwargs"
        ), "Transformed message should have additional_kwargs"
        assert (
            "engine_name" in transformed_msg.additional_kwargs
        ), "Engine attribution should be preserved"
        assert (
            transformed_msg.additional_kwargs["engine_name"] == engine_name
        ), f"Engine name should be '{engine_name}', got '{transformed_msg.additional_kwargs['engine_name']}'"
        assert (
            transformed_msg.additional_kwargs["custom_field"] == "test_value"
        ), "Other metadata should also be preserved"

        print("\n✅ SUCCESS: Transformation preserved engine attributionn")
        print(f"   Transformed to: {type(transformed_msg).__name__}")
        print(
            f"   Engine name preserved: '{transformed_msg.additional_kwargs['engine_name']}'"
        )
        print(f"   All metadata preserved: {transformed_msg.additional_kwargs}")

    def test_complete_flow_with_attribution(self):
        """Test complete flow: Engine → Transformation with attribution."""
        print("\n" + "=" * 80)
        print("TEST: Complete Flow with Engine Attribution")
        print("=" * 80)

        # Step 1: Create engine and process message
        engine_name = "gpt4_engine"
        engine = AugLLMConfig(
            name=engine_name,
            system_message="You are a helpful assistant. Keep responses brief.",
            temperature=0.1,
        )

        engine_node = EngineNodeConfig(name="engine_processor", engine=engine)

        # Initial state
        state = TestState(
            messages=[
                SystemMessage(content="You are a helpful assistant."),
                HumanMessage(content="What is 2+2?"),
            ]
        )

        print(f"\n1. Starting with {len(state.messages)} messages")
        print(f"2. Processing through engine '{engine_name}'...")

        # Process through engine
        engine_result = engine_node(state)
        state.messages = engine_result.update["messages"]

        # Verify AI message has attribution
        ai_msg = state.messages[-1]
        assert isinstance(ai_msg, AIMessage)
        assert ai_msg.additional_kwargs.get("engine_name") == engine_name

        print(
            f"3. Engine added AI message with attribution: engine_name='{ai_msg.additional_kwargs['engine_name']}'"
        )
        print(f"   Response: {ai_msg.content[:100]}...")

        # Step 2: Add engine ID through transformation
        engine_id = "engine_123"
        id_transformer = MessageTransformationNodeConfig(
            name="id_adder",
            transformation_type=TransformationType.ADD_ENGINE_ID,
            engine_id=engine_id,
            engine_name=engine_name,  # This can override or supplement
            messages_key="messages",
            output_key="messages",
        )

        print(f"\n4. Adding engine_id '{engine_id}' through transformation...")

        # Transform messages
        transform_result = id_transformer(state)
        state.messages = transform_result.update["messages"]

        # Verify the last message now has both engine_name and engine_id
        final_ai_msg = state.messages[-1]
        assert isinstance(final_ai_msg, AIMessage)
        assert final_ai_msg.additional_kwargs.get("engine_name") == engine_name
        assert final_ai_msg.additional_kwargs.get("engine_id") == engine_id

        print("5. Transformation added engine_id while preserving engine_name")
        print(f"   Final additional_kwargs: {final_ai_msg.additional_kwargs}")

        # Step 3: Reflection transformation (swap roles but preserve metadata)
        reflection_transformer = MessageTransformationNodeConfig(
            name="reflector",
            transformation_type=TransformationType.REFLECTION,
            preserve_first_message=True,
            preserve_metadata=True,
            messages_key="messages",
            output_key="reflected_messages",
        )

        print("\n6. Applying reflection transformation...")

        reflection_result = reflection_transformer(state)
        reflected_messages = reflection_result.update["reflected_messages"]

        # Check that the AI message became Human but kept attribution
        # (First message is preserved, so AI message should be at index -1 after transformation)
        reflected_msg = reflected_messages[-1]
        assert isinstance(
            reflected_msg, HumanMessage
        ), "AI should be transformed to Human in reflection"
        assert reflected_msg.additional_kwargs.get("engine_name") == engine_name
        assert reflected_msg.additional_kwargs.get("engine_id") == engine_id

        print("7. Reflection changed AI → Human but preserved attributionn")
        print(f"   Message type: {type(reflected_msg).__name__}")
        print(f"   Attribution preserved: {reflected_msg.additional_kwargs}")

        print("\n✅ COMPLETE FLOW SUCCESS::")
        print("   - Engine node added engine_name attribution")
        print("   - Transformation node added engine_id")
        print("   - Reflection preserved all metadata")
        print("   - Final message has complete attribution chain")

    def test_multiple_engines_with_attribution(self):
        """Test multiple engines each adding their own attribution."""
        print("\n" + "=" * 80)
        print("TEST: Multiple Engines with Attribution")
        print("=" * 80)

        # Create two engines with different names
        engine1 = AugLLMConfig(
            name="analyzer_engine",
            system_message="You are an analyzer. Analyze the input briefly.",
            temperature=0.1,
        )

        engine2 = AugLLMConfig(
            name="summarizer_engine",
            system_message="You are a summarizer. Summarize the conversation briefly.",
            temperature=0.1,
        )

        # Create nodes for each engine
        analyzer_node = EngineNodeConfig(name="analyzer", engine=engine1)

        summarizer_node = EngineNodeConfig(name="summarizer", engine=engine2)

        # Initial state
        state = TestState(
            messages=[
                HumanMessage(content="Explain quantum computing in simple terms.")
            ]
        )

        print("\n1. Starting with user query")

        # Process through first engine
        print(f"2. Processing through '{engine1.name}'...")
        result1 = analyzer_node(state)
        state.messages = result1.update["messages"]

        # Verify first attribution
        analyzer_msg = state.messages[-1]
        assert isinstance(analyzer_msg, AIMessage)
        assert analyzer_msg.additional_kwargs.get("engine_name") == "analyzer_engine"

        print(
            f"   Added message with engine_name='{analyzer_msg.additional_kwargs['engine_name']}'"
        )

        # Process through second engine
        print(f"3. Processing through '{engine2.name}'...")
        result2 = summarizer_node(state)
        state.messages = result2.update["messages"]

        # Verify second attribution
        summarizer_msg = state.messages[-1]
        assert isinstance(summarizer_msg, AIMessage)
        assert (
            summarizer_msg.additional_kwargs.get("engine_name") == "summarizer_engine"
        )

        print(
            f"   Added message with engine_name='{summarizer_msg.additional_kwargs['engine_name']}'"
        )

        # Verify we can distinguish messages by engine
        print(f"\n4. Final conversation has {len(state.messages)} messages:")
        for i, msg in enumerate(state.messages):
            if isinstance(msg, AIMessage):
                engine_name = msg.additional_kwargs.get("engine_name", "unknown")
                print(f"   [{i}] AI from '{engine_name}': {msg.content[:60]}...")
            else:
                print(f"   [{i}] {type(msg).__name__}: {msg.content[:60]}...")

        # Count messages by engine
        engine_counts = {}
        for msg in state.messages:
            if isinstance(msg, AIMessage) and "engine_name" in msg.additional_kwargs:
                engine = msg.additional_kwargs["engine_name"]
                engine_counts[engine] = engine_counts.get(engine, 0) + 1

        print("\n✅ SUCCESS: Multiple engines with proper attributionn")
        print(f"   Engine message counts: {engine_counts}")
        print("   Each AI message is properly attributed to its source engine")


if __name__ == "__main__":
    # Run tests directly
    test = TestEngineAttribution()

    print("\n" + "=" * 80)
    print("RUNNING ENGINE ATTRIBUTION TESTS")
    print("=" * 80)

    try:
        test.test_engine_node_adds_attribution()
        test.test_message_transformation_preserves_attribution()
        test.test_complete_flow_with_attribution()
        test.test_multiple_engines_with_attribution()

        print("\n" + "=" * 80)
        print("ALL TESTS PASSED! ✅")
        print("=" * 80)

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        raise
