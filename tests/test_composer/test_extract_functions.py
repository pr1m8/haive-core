"""Tests for extract functions library - using real components only.

Tests all extract function patterns identified from node analysis,
ensuring they work with real Haive components.
"""

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

from haive.core.graph.node.composer.extract_functions import (
    ExtractFunctions,
    extract_conditional,
    extract_messages_content,
    extract_multi_field,
    extract_simple_field,
    extract_typed,
    extract_with_path,
    extract_with_projection,
)
from haive.core.schema.prebuilt.messages_state import MessagesState


class AgentConfig(BaseModel):
    """Sample agent configuration for testing."""

    name: str = Field(default="test_agent")
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=1000)
    use_history: bool = Field(default=True)


class ComplexState(BaseModel):
    """Complex state for testing various extract patterns."""

    messages: list[str] = Field(default_factory=list)
    config: AgentConfig = Field(default_factory=AgentConfig)
    agents: list[dict[str, Any]] = Field(default_factory=list)
    current_query: str = Field(default="")
    full_conversation: str = Field(default="")
    current_message: str = Field(default="")
    iteration_count: int = Field(default=0)


class TestExtractFunctions:
    """Test extract function library with real components."""

    @pytest.fixture
    def extract_lib(self):
        """Create ExtractFunctions instance."""
        return ExtractFunctions()

    @pytest.fixture
    def complex_state(self):
        """Create complex test state."""
        return ComplexState(
            messages=["Hello", "How are you?", "Goodbye"],
            config=AgentConfig(name="test", temperature=0.8, use_history=True),
            agents=[
                {"name": "agent1", "status": "active", "priority": "high"},
                {"name": "agent2", "status": "idle", "priority": "low"},
            ],
            current_query="What is the weather?",
            full_conversation="Hello. How are you? What is the weather?",
            current_message="What is the weather?",
            iteration_count=5,
        )

    @pytest.fixture
    def real_messages_state(self):
        """Create real MessagesState with LangChain messages."""
        return MessagesState(
            messages=[
                HumanMessage(content="What is AI?"),
                AIMessage(content="AI stands for Artificial Intelligence."),
                HumanMessage(content="How does it work?"),
                AIMessage(
                    content="AI works through machine learning algorithms."),
            ]
        )

    def test_extract_simple_field_basic(self, extract_lib, complex_state):
        """Test simple field extraction."""
        # Create extract function
        extract_messages = extract_lib.extract_simple_field("messages")

        # Test extraction
        result = extract_messages(complex_state, {})
        assert result == ["Hello", "How are you?", "Goodbye"]

        # Test with default
        extract_missing = extract_lib.extract_simple_field(
            "missing_field", "default")
        result = extract_missing(complex_state, {})
        assert result == "default"

    def test_extract_simple_field_module_function(self, complex_state):
        """Test module-level extract_simple_field function."""
        # Test module function
        extract_query = extract_simple_field("current_query")
        result = extract_query(complex_state, {})
        assert result == "What is the weather?"

        # Test with default
        extract_missing = extract_simple_field("missing", "not_found")
        result = extract_missing(complex_state, {})
        assert result == "not_found"

    def test_extract_with_path_dot_notation(self, extract_lib, complex_state):
        """Test path extraction with dot notation."""
        # Extract nested config value
        extract_temp = extract_lib.extract_with_path("config.temperature")
        result = extract_temp(complex_state, {})
        assert result == 0.8

        # Extract nested config name
        extract_name = extract_lib.extract_with_path("config.name", "unknown")
        result = extract_name(complex_state, {})
        assert result == "test"

        # Test missing path with default
        extract_missing = extract_lib.extract_with_path(
            "config.missing.field", "default"
        )
        result = extract_missing(complex_state, {})
        assert result == "default"

    def test_extract_with_path_array_access(self, extract_lib, complex_state):
        """Test path extraction with array access."""
        # Extract first message
        extract_first = extract_lib.extract_with_path("messages[0]")
        result = extract_first(complex_state, {})
        assert result == "Hello"

        # Extract last message
        extract_last = extract_lib.extract_with_path("messages[-1]")
        result = extract_last(complex_state, {})
        assert result == "Goodbye"

        # Extract agent name
        extract_agent_name = extract_lib.extract_with_path("agents[0].name")
        result = extract_agent_name(complex_state, {})
        assert result == "agent1"

    def test_extract_with_path_module_function(self, complex_state):
        """Test module-level extract_with_path function."""
        # Test complex path
        extract_temp = extract_with_path("config.temperature", 0.5)
        result = extract_temp(complex_state, {})
        assert result == 0.8

        # Test array access
        extract_last_msg = extract_with_path("messages[-1]")
        result = extract_last_msg(complex_state, {})
        assert result == "Goodbye"

    def test_extract_with_projection(self, extract_lib, complex_state):
        """Test extraction with field projection."""
        # Project specific fields from first agent
        extract_projected = extract_lib.extract_with_projection(
            "agents[0]", ["name", "status"]
        )
        result = extract_projected(complex_state, {})

        assert result == {"name": "agent1", "status": "active"}
        # Should not include "priority" field
        assert "priority" not in result

        # Test with missing source
        extract_missing = extract_lib.extract_with_projection(
            "missing_agents", ["name"]
        )
        result = extract_missing(complex_state, {})
        assert result == {}

    def test_extract_messages_content_real_state(
        self, extract_lib, real_messages_state
    ):
        """Test message content extraction with real MessagesState."""
        # Extract content from real LangChain messages
        extract_content = extract_lib.extract_messages_content()
        result = extract_content(real_messages_state, {})

        expected = [
            "What is AI?",
            "AI stands for Artificial Intelligence.",
            "How does it work?",
            "AI works through machine learning algorithms.",
        ]
        assert result == expected

    def test_extract_messages_content_custom_field(
            self, extract_lib, complex_state):
        """Test message content extraction with custom field name."""
        # Extract from regular string list (simulating message content)
        extract_content = extract_lib.extract_messages_content("messages")
        result = extract_content(complex_state, {})

        # Should convert strings to content
        assert result == ["Hello", "How are you?", "Goodbye"]

        # Test with missing field
        extract_missing = extract_lib.extract_messages_content(
            "missing_messages")
        result = extract_missing(complex_state, {})
        assert result == []

    def test_extract_conditional_true_case(self, extract_lib, complex_state):
        """Test conditional extraction - true case."""
        # When use_history is True, extract full_conversation
        extract_input = extract_lib.extract_conditional(
            "config.use_history", "full_conversation", "current_message"
        )

        result = extract_input(complex_state, {})
        assert result == "Hello. How are you? What is the weather?"

    def test_extract_conditional_false_case(self, extract_lib):
        """Test conditional extraction - false case."""
        # Create state with use_history = False
        state = ComplexState(
            config=AgentConfig(use_history=False),
            full_conversation="Full conversation text",
            current_message="Just current message",
        )

        extract_input = extract_lib.extract_conditional(
            "config.use_history", "full_conversation", "current_message"
        )

        result = extract_input(state, {})
        assert result == "Just current message"

    def test_extract_conditional_with_defaults(
            self, extract_lib, complex_state):
        """Test conditional extraction with defaults."""
        extract_input = extract_lib.extract_conditional(
            "config.use_history",
            "missing_field",
            "also_missing",
            true_default="default_true",
            false_default="default_false",
        )

        result = extract_input(complex_state, {})
        assert result == "default_true"  # use_history is True

    def test_extract_multi_field_basic(self, extract_lib, complex_state):
        """Test multi-field extraction."""
        # Extract multiple fields
        extract_multi = extract_lib.extract_multi_field(
            {
                "query": "current_query",
                "temperature": "config.temperature",
                "agent_name": "config.name",
            }
        )

        result = extract_multi(complex_state, {})
        expected = {
            "query": "What is the weather?",
            "temperature": 0.8,
            "agent_name": "test",
        }
        assert result == expected

    def test_extract_multi_field_with_defaults(
            self, extract_lib, complex_state):
        """Test multi-field extraction with defaults."""
        extract_multi = extract_lib.extract_multi_field(
            {
                "query": "current_query",
                "missing": "missing_field",
                "temperature": "config.temperature",
            },
            defaults={"missing": "default_value", "temperature": 0.5},
        )

        result = extract_multi(complex_state, {})
        expected = {
            "query": "What is the weather?",
            "missing": "default_value",  # Used default
            "temperature": 0.8,  # Found actual value
        }
        assert result == expected

    def test_extract_typed_valid_type(self, extract_lib, complex_state):
        """Test typed extraction with valid type."""
        # Extract integer field
        extract_count = extract_lib.extract_typed("iteration_count", int, 0)
        result = extract_count(complex_state, {})
        assert result == 5
        assert isinstance(result, int)

        # Extract float field
        extract_temp = extract_lib.extract_typed(
            "config.temperature", float, 0.0)
        result = extract_temp(complex_state, {})
        assert result == 0.8
        assert isinstance(result, float)

    def test_extract_typed_invalid_type(self, extract_lib, complex_state):
        """Test typed extraction with type mismatch."""
        # Try to extract string as int
        extract_bad = extract_lib.extract_typed("current_query", int, 999)
        result = extract_bad(complex_state, {})
        assert result == 999  # Should return default

        # Try to extract int as string
        extract_bad2 = extract_lib.extract_typed(
            "iteration_count", str, "default")
        result = extract_bad2(complex_state, {})
        assert result == "default"  # Should return default

    def test_extract_typed_missing_field(self, extract_lib, complex_state):
        """Test typed extraction with missing field."""
        extract_missing = extract_lib.extract_typed(
            "missing_field", str, "not_found")
        result = extract_missing(complex_state, {})
        assert result == "not_found"

    def test_all_module_functions_work(
            self, complex_state, real_messages_state):
        """Test that all module-level functions work correctly."""
        # Test all module functions
        result1 = extract_simple_field("current_query")(complex_state, {})
        assert result1 == "What is the weather?"

        result2 = extract_with_path("config.name")(complex_state, {})
        assert result2 == "test"

        result3 = extract_with_projection(
            "agents[0]", ["name"])(
            complex_state, {})
        assert result3 == {"name": "agent1"}

        result4 = extract_messages_content()(real_messages_state, {})
        assert len(result4) == 4

        result5 = extract_conditional(
            "config.use_history", "full_conversation", "current_message"
        )(complex_state, {})
        assert result5 == "Hello. How are you? What is the weather?"

        result6 = extract_multi_field(
            {"query": "current_query"})(complex_state, {})
        assert result6 == {"query": "What is the weather?"}

        result7 = extract_typed("iteration_count", int, 0)(complex_state, {})
        assert result7 == 5

    def test_extract_functions_with_real_components_integration(
        self, real_messages_state
    ):
        """Test extract functions with real Haive components."""
        # Test with actual MessagesState and LangChain messages

        # Extract last message content
        extract_last = extract_with_path("messages[-1].content")
        last_content = extract_last(real_messages_state, {})
        assert last_content == "AI works through machine learning algorithms."

        # Extract first message type (should be HumanMessage)
        extract_first_msg = extract_with_path("messages[0]")
        first_msg = extract_first_msg(real_messages_state, {})
        assert isinstance(first_msg, HumanMessage)
        assert first_msg.content == "What is AI?"

        # Extract all message contents
        extract_all_content = extract_messages_content()
        all_content = extract_all_content(real_messages_state, {})
        assert len(all_content) == 4
        assert all_content[0] == "What is AI?"
        assert all_content[-1] == "AI works through machine learning algorithms."
