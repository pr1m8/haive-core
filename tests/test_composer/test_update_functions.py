"""Tests for update functions library - using real components only.

Tests all update function patterns identified from node analysis,
ensuring they work with real Haive components.
"""

import json
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

from haive.core.graph.node.composer.update_functions import (
    UpdateFunctions,
    update_conditional,
    update_hierarchical,
    update_messages_append,
    update_multi_field,
    update_simple_field,
    update_type_aware,
    update_with_path,
    update_with_transform,
)
from haive.core.schema.prebuilt.messages_state import MessagesState


class AgentConfig(BaseModel):
    """Sample agent configuration for testing."""

    name: str = Field(default="test_agent")
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=1000)
    save_history: bool = Field(default=True)


class ComplexState(BaseModel):
    """Complex state for testing various update patterns."""

    messages: list[str] = Field(default_factory=list)
    config: AgentConfig = Field(default_factory=AgentConfig)
    current_agent: dict[str, Any] = Field(default_factory=dict)
    ai_response: str = Field(default="")
    response_confidence: float = Field(default=0.0)
    token_usage: int = Field(default=0)
    iteration_count: int = Field(default=0)
    conversation_history: list[str] = Field(default_factory=list)
    current_response: str = Field(default="")
    title: str = Field(default="")
    data: dict[str, Any] = Field(default_factory=dict)


class TestUpdateFunctions:
    """Test update function library with real components."""

    @pytest.fixture
    def update_lib(self):
        """Create UpdateFunctions instance."""
        return UpdateFunctions()

    @pytest.fixture
    def complex_state(self):
        """Create complex test state."""
        return ComplexState(
            messages=["Hello", "How are you?"],
            config=AgentConfig(name="test", temperature=0.8, save_history=True),
            current_agent={"status": "idle", "last_action": "wait"},
            ai_response="Previous response",
            iteration_count=3,
        )

    @pytest.fixture
    def real_messages_state(self):
        """Create real MessagesState with LangChain messages."""
        return MessagesState(
            messages=[HumanMessage(content="Hello"), AIMessage(content="Hi there!")]
        )

    def test_update_simple_field_replace(self, update_lib, complex_state):
        """Test simple field update with replace mode."""
        # Create update function
        update_response = update_lib.update_simple_field("ai_response")

        # Test update
        updates = update_response("New response", complex_state, {})
        assert updates == {"ai_response": "New response"}

    def test_update_simple_field_append(self, update_lib, complex_state):
        """Test simple field update with append mode."""
        # Create update function for list
        update_messages = update_lib.update_simple_field("messages", "append")

        # Test append
        updates = update_messages("Goodbye", complex_state, {})
        expected_messages = ["Hello", "How are you?", "Goodbye"]
        assert updates == {"messages": expected_messages}

    def test_update_simple_field_merge(self, update_lib, complex_state):
        """Test simple field update with merge mode."""
        # Create update function for dict
        update_agent = update_lib.update_simple_field("current_agent", "merge")

        # Test merge
        new_data = {"status": "active", "new_field": "value"}
        updates = update_agent(new_data, complex_state, {})

        expected = {
            "status": "active",  # Updated
            "last_action": "wait",  # Preserved
            "new_field": "value",  # Added
        }
        assert updates == {"current_agent": expected}

    def test_update_simple_field_module_function(self, complex_state):
        """Test module-level update_simple_field function."""
        # Test module function
        update_count = update_simple_field("iteration_count")
        updates = update_count(10, complex_state, {})
        assert updates == {"iteration_count": 10}

        # Test append mode
        update_messages = update_simple_field("messages", "append")
        updates = update_messages("New message", complex_state, {})
        expected = ["Hello", "How are you?", "New message"]
        assert updates == {"messages": expected}

    def test_update_with_path_simple(self, update_lib, complex_state):
        """Test path update with simple field."""
        # Update simple field through path
        update_response = update_lib.update_with_path("ai_response")
        updates = update_response("Path response", complex_state, {})
        assert updates == {"ai_response": "Path response"}

    def test_update_with_path_nested(self, update_lib, complex_state):
        """Test path update with nested field."""
        # Update nested config value
        update_temp = update_lib.update_with_path("config.temperature")
        updates = update_temp(0.9, complex_state, {})

        expected = {"config": {"temperature": 0.9}}
        assert updates == expected

    def test_update_messages_append_real_state(self, update_lib, real_messages_state):
        """Test message append with real MessagesState."""
        # Create new message
        new_message = HumanMessage(content="How are you?")

        # Update function
        update_msgs = update_lib.update_messages_append()
        updates = update_msgs(new_message, real_messages_state, {})

        # Verify update
        assert "messages" in updates
        updated_messages = updates["messages"]
        assert len(updated_messages) == 3
        assert updated_messages[0].content == "Hello"
        assert updated_messages[1].content == "Hi there!"
        assert updated_messages[2].content == "How are you?"

    def test_update_messages_append_custom_field(self, update_lib, complex_state):
        """Test message append with custom field name."""
        # Update conversation history instead of messages
        update_history = update_lib.update_messages_append("conversation_history")
        updates = update_history("New entry", complex_state, {})

        assert updates == {"conversation_history": ["New entry"]}

    def test_update_type_aware_valid_type(self, update_lib, complex_state):
        """Test type-aware update with valid type."""
        # Update with correct type
        update_count = update_lib.update_type_aware("iteration_count", int)
        updates = update_count(15, complex_state, {})
        assert updates == {"iteration_count": 15}

        # Update with convertible type
        updates = update_count("20", complex_state, {})
        assert updates == {"iteration_count": 20}

    def test_update_type_aware_invalid_type(self, update_lib, complex_state):
        """Test type-aware update with invalid type."""
        # Try to update int field with unconvertible value
        update_count = update_lib.update_type_aware("iteration_count", int)
        updates = update_count("not_a_numbef", complex_state, {})
        assert updates == {}  # Should skip update

        # Try to update with wrong type
        update_temp = update_lib.update_type_aware("temperature", float)
        updates = update_temp({"not": "float"}, complex_state, {})
        assert updates == {}  # Should skip update

    def test_update_conditional_true_case(self, update_lib, complex_state):
        """Test conditional update - true case."""
        # When save_history is True, update conversation_history
        update_output = update_lib.update_conditional(
            "config.save_history", "conversation_history", "current_response"
        )

        updates = update_output("Test response", complex_state, {})
        assert updates == {"conversation_history": "Test response"}

    def test_update_conditional_false_case(self, update_lib):
        """Test conditional update - false case."""
        # Create state with save_history = False
        state = ComplexState(config=AgentConfig(save_history=False))

        update_output = update_lib.update_conditional(
            "config.save_history", "conversation_history", "current_response"
        )

        updates = update_output("Test response", state, {})
        assert updates == {"current_response": "Test response"}

    def test_update_multi_field_basic(self, update_lib, complex_state):
        """Test multi-field update."""
        # Split result into multiple fields
        update_multi = update_lib.update_multi_field(
            {
                "response": "ai_response",
                "confidence": "response_confidence",
                "tokens": "token_usage",
            }
        )

        result = {"response": "Hello there!", "confidence": 0.95, "tokens": 15}

        updates = update_multi(result, complex_state, {})
        expected = {
            "ai_response": "Hello there!",
            "response_confidence": 0.95,
            "token_usage": 15,
        }
        assert updates == expected

    def test_update_multi_field_partial_result(self, update_lib, complex_state):
        """Test multi-field update with partial result."""
        update_multi = update_lib.update_multi_field(
            {
                "response": "ai_response",
                "confidence": "response_confidence",
                "missing": "some_field",
            }
        )

        result = {"response": "Partial result"}  # Missing other fields

        updates = update_multi(result, complex_state, {})
        assert updates == {"ai_response": "Partial result"}

    def test_update_multi_field_non_dict(self, update_lib, complex_state):
        """Test multi-field update with non-dict result."""
        update_multi = update_lib.update_multi_field({"key": "field"})

        # Try with string instead of dict
        updates = update_multi("not a dict", complex_state, {})
        assert updates == {}  # Should return empty

    def test_update_with_transform_basic(self, update_lib, complex_state):
        """Test update with transformation."""
        # Transform to uppercase
        update_upper = update_lib.update_with_transform("title", str.upper)
        updates = update_upper("hello world", complex_state, {})
        assert updates == {"title": "HELLO WORLD"}

        # Transform with JSON parsing
        update_json = update_lib.update_with_transform("data", json.loads)
        updates = update_json('{"key": "value"}', complex_state, {})
        assert updates == {"data": {"key": "value"}}

    def test_update_with_transform_failure(self, update_lib, complex_state):
        """Test update with transformation failure."""
        # Try to parse invalid JSON
        update_json = update_lib.update_with_transform("data", json.loads)
        updates = update_json("invalid json", complex_state, {})
        assert updates == {}  # Should skip update on error

    def test_update_hierarchical_basic(self, update_lib, complex_state):
        """Test hierarchical update."""
        # Update agent with merge
        update_agent = update_lib.update_hierarchical("current_agent")

        new_data = {"status": "active", "new_action": "search"}

        updates = update_agent(new_data, complex_state, {})
        expected = {
            "current_agent": {
                "status": "active",  # Updated
                "last_action": "wait",  # Preserved
                "new_action": "search",  # Added
            }
        }
        assert updates == expected

    def test_update_hierarchical_with_projection(self, update_lib, complex_state):
        """Test hierarchical update with field projection."""
        # Update only specific fields
        update_agent = update_lib.update_hierarchical("current_agent", ["status"])

        new_data = {
            "status": "active",
            "new_action": "search",  # Should be ignored
            "ignored": "value",  # Should be ignored
        }

        updates = update_agent(new_data, complex_state, {})
        expected = {
            "current_agent": {
                "status": "active",  # Updated (in projection)
                "last_action": "wait",  # Preserved (existing)
                # new_action and ignored should not be included
            }
        }
        assert updates == expected

    def test_update_hierarchical_non_dict(self, update_lib, complex_state):
        """Test hierarchical update with non-dict result."""
        update_agent = update_lib.update_hierarchical("current_agent")
        updates = update_agent("simple string", complex_state, {})
        assert updates == {"current_agent": "simple string"}

    def test_all_module_functions_work(self, complex_state, real_messages_state):
        """Test that all module-level functions work correctly."""
        # Test all module functions
        updates1 = update_simple_field("ai_response")("Module test", complex_state, {})
        assert updates1 == {"ai_response": "Module test"}

        updates2 = update_with_path("config.temperature")(0.95, complex_state, {})
        assert updates2 == {"config": {"temperature": 0.95}}

        new_msg = HumanMessage(content="Test message")
        updates3 = update_messages_append()(new_msg, real_messages_state, {})
        assert len(updates3["messages"]) == 3

        updates4 = update_type_aware("iteration_count", int)("25", complex_state, {})
        assert updates4 == {"iteration_count": 25}

        updates5 = update_conditional("config.save_history", "hist", "curr")(
            "test", complex_state, {}
        )
        assert updates5 == {"hist": "test"}

        result_dict = {"a": 1, "b": 2}
        updates6 = update_multi_field({"a": "field_a", "b": "field_b"})(
            result_dict, complex_state, {}
        )
        assert updates6 == {"field_a": 1, "field_b": 2}

        updates7 = update_with_transform("title", str.lower)("UPPER", complex_state, {})
        assert updates7 == {"title": "uppef"}

        agent_data = {"status": "new"}
        updates8 = update_hierarchical("current_agent")(agent_data, complex_state, {})
        expected_agent = {"status": "new", "last_action": "wait"}
        assert updates8 == {"current_agent": expected_agent}

    def test_update_functions_with_real_components_integration(
        self, real_messages_state
    ):
        """Test update functions with real Haive components."""
        # Test with actual MessagesState and LangChain messages

        # Add AI response to conversation
        ai_response = AIMessage(content="I'm doing well, thank you!")
        update_msgs = update_messages_append()
        updates = update_msgs(ai_response, real_messages_state, {})

        # Verify the update
        assert "messages" in updates
        updated_messages = updates["messages"]
        assert len(updated_messages) == 3
        assert updated_messages[0].content == "Hello"
        assert updated_messages[1].content == "Hi there!"
        assert updated_messages[2].content == "I'm doing well, thank you!"
        assert isinstance(updated_messages[2], AIMessage)

        # Test multi-field update with real message data
        message_result = {"content": "How can I help you?", "type": "ai", "tokens": 6}

        update_multi = update_multi_field(
            {"content": "last_response", "tokens": "response_tokens"}
        )

        updates = update_multi(message_result, real_messages_state, {})
        assert updates == {"last_response": "How can I help you?", "response_tokens": 6}

    def test_update_functions_error_handling(self, complex_state):
        """Test error handling in update functions."""
        # Test append to non-list field
        update_append = update_simple_field("ai_response", "append")
        updates = update_append("new", complex_state, {})
        # Should convert to list and append
        assert updates == {"ai_response": ["Previous response", "new"]}

        # Test merge with non-dict field
        update_merge = update_simple_field("ai_response", "merge")
        non_dict_result = "not a dict"
        updates = update_merge(non_dict_result, complex_state, {})
        # Should replace since result is not dict
        assert updates == {"ai_response": "not a dict"}
