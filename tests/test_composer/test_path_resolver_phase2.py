"""Tests for PathResolver Phase 2 features - using real components only.

Tests for dot notation and array access features added in Phase 2.
"""

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

from haive.core.graph.node.composer import PathResolver
from haive.core.schema.prebuilt.messages_state import MessagesState


class NestedConfig(BaseModel):
    """Nested configuration for testing dot notation."""

    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=1000)
    model_name: str = Field(default="gpt-4")


class ComplexState(BaseModel):
    """Complex state for testing Phase 2 features."""

    messages: list[str] = Field(default_factory=list)
    config: NestedConfig = Field(default_factory=NestedConfig)
    metadata: dict[str, Any] = Field(default_factory=dict)
    agents: list[dict[str, Any]] = Field(default_factory=list)


class TestPathResolverPhase2:
    """Test Phase 2 enhancements: dot notation and array access."""

    @pytest.fixture
    def resolver(self):
        """Create a PathResolver instance."""
        return PathResolver()

    @pytest.fixture
    def nested_dict(self):
        """Create nested dict structure for testing."""
        return {
            "config": {
                "temperature": 0.8,
                "model": {"name": "claude", "version": "3.0"},
            },
            "messages": ["hello", "world", "test"],
            "agents": [
                {"name": "agent1", "status": "active"},
                {"name": "agent2", "status": "idle"},
            ],
        }

    @pytest.fixture
    def nested_pydantic(self):
        """Create nested Pydantic model for testing."""
        return ComplexState(
            messages=["first", "second", "third"],
            config=NestedConfig(temperature=0.5, model_name="gpt-4-turbo"),
            agents=[{"id": "a1", "role": "worker"}, {"id": "a2", "role": "manager"}],
        )

    @pytest.fixture
    def real_messages_state(self):
        """Create real MessagesState with nested message structure."""
        return MessagesState(
            messages=[
                HumanMessage(content="First question"),
                AIMessage(content="First answer"),
                HumanMessage(content="Second question"),
            ]
        )

    def test_dot_notation_with_dict(self, resolver, nested_dict):
        """Test dot notation access with nested dictionaries."""
        # Test simple dot notation
        result = resolver.extract_value(nested_dict, "config.temperature")
        assert result == 0.8

        # Test deeper nesting
        result = resolver.extract_value(nested_dict, "config.model.name")
        assert result == "claude"

        result = resolver.extract_value(nested_dict, "config.model.version")
        assert result == "3.0"

        # Test non-existent path
        result = resolver.extract_value(nested_dict, "config.missing.field", default="not_found")
        assert result == "not_found"

    def test_dot_notation_with_pydantic(self, resolver, nested_pydantic):
        """Test dot notation access with nested Pydantic models."""
        # Test accessing nested model fields
        result = resolver.extract_value(nested_pydantic, "config.temperature")
        assert result == 0.5

        result = resolver.extract_value(nested_pydantic, "config.model_name")
        assert result == "gpt-4-turbo"

        result = resolver.extract_value(nested_pydantic, "config.max_tokens")
        assert result == 1000

        # Test non-existent path
        result = resolver.extract_value(nested_pydantic, "config.missing", default="default")
        assert result == "default"

    def test_array_access_with_dict(self, resolver, nested_dict):
        """Test array access with dictionaries."""
        # Test positive indices
        result = resolver.extract_value(nested_dict, "messages[0]")
        assert result == "hello"

        result = resolver.extract_value(nested_dict, "messages[1]")
        assert result == "world"

        result = resolver.extract_value(nested_dict, "messages[2]")
        assert result == "test"

        # Test negative indices
        result = resolver.extract_value(nested_dict, "messages[-1]")
        assert result == "test"

        result = resolver.extract_value(nested_dict, "messages[-2]")
        assert result == "world"

        # Test out of bounds
        result = resolver.extract_value(nested_dict, "messages[99]", default="oob")
        assert result == "oob"

        result = resolver.extract_value(nested_dict, "messages[-99]", default="oob")
        assert result == "oob"

    def test_array_access_with_pydantic(self, resolver, nested_pydantic):
        """Test array access with Pydantic models."""
        # Test accessing list fields
        result = resolver.extract_value(nested_pydantic, "messages[0]")
        assert result == "first"

        result = resolver.extract_value(nested_pydantic, "messages[-1]")
        assert result == "third"

        # Test accessing nested dict arrays
        result = resolver.extract_value(nested_pydantic, "agents[0]")
        assert result == {"id": "a1", "role": "worker"}

        result = resolver.extract_value(nested_pydantic, "agents[1]")
        assert result == {"id": "a2", "role": "manager"}

    def test_combined_dot_and_array_access(self, resolver, nested_dict):
        """Test combining dot notation and array access."""
        # Test array access followed by dict access
        result = resolver.extract_value(nested_dict, "agents[0].name")
        assert result == "agent1"

        result = resolver.extract_value(nested_dict, "agents[1].status")
        assert result == "idle"

        result = resolver.extract_value(nested_dict, "agents[-1].name")
        assert result == "agent2"

        # Test non-existent combined paths
        result = resolver.extract_value(nested_dict, "agents[0].missing", default="none")
        assert result == "none"

        result = resolver.extract_value(nested_dict, "agents[99].name", default="none")
        assert result == "none"

    def test_real_messages_state_complex_paths(self, resolver, real_messages_state):
        """Test complex paths with real MessagesState."""
        # Test array access on real messages
        result = resolver.extract_value(real_messages_state, "messages[0]")
        assert isinstance(result, HumanMessage)
        assert result.content == "First question"

        result = resolver.extract_value(real_messages_state, "messages[1]")
        assert isinstance(result, AIMessage)
        assert result.content == "First answer"

        result = resolver.extract_value(real_messages_state, "messages[-1]")
        assert isinstance(result, HumanMessage)
        assert result.content == "Second question"

        # Test accessing message content directly
        # Note: This tests the path parsing but content access depends on
        # message structure
        result = resolver.extract_value(real_messages_state, "messages[0].content")
        assert result == "First question"

        result = resolver.extract_value(real_messages_state, "messages[-1].content")
        assert result == "Second question"

    def test_path_segment_parsing(self, resolver):
        """Test the internal path segment parsing logic."""
        # Test simple dot notation
        segments = resolver._parse_path_segments("config.temperature")
        assert segments == ["config", "temperature"]

        # Test array notation
        segments = resolver._parse_path_segments("messages[0]")
        assert segments == ["messages[0]"]

        # Test combined notation
        segments = resolver._parse_path_segments("agents[0].name")
        assert segments == ["agents[0]", "name"]

        # Test complex nesting
        segments = resolver._parse_path_segments("data.items[-1].value")
        assert segments == ["data", "items[-1]", "value"]

        # Test multiple arrays
        segments = resolver._parse_path_segments("matrix[0][1].field")
        assert segments == ["matrix[0]", "[1]", "field"]

    def test_array_access_parsing(self, resolver):
        """Test the internal array access parsing logic."""
        # Create test object
        test_obj = {"items": ["a", "b", "c"]}

        # Test array access parsing
        result = resolver._extract_array_access(test_obj, "items[0]")
        assert result == "a"

        result = resolver._extract_array_access(test_obj, "items[-1]")
        assert result == "c"

        # Test invalid cases
        result = resolver._extract_array_access(test_obj, "items[abc]", default="invalid")
        assert result == "invalid"

        result = resolver._extract_array_access(test_obj, "missing[0]", default="missing")
        assert result == "missing"

    def test_backward_compatibility_with_phase1(self, resolver):
        """Test that Phase 2 enhancements don't break Phase 1 functionality."""
        # Create simple test data
        simple_state = {
            "messages": ["hello"],
            "temperature": 0.7,
            "config": {"nested": "value"},
        }

        # Test that simple field access still works
        result = resolver.extract_value(simple_state, "messages")
        assert result == ["hello"]

        result = resolver.extract_value(simple_state, "temperature")
        assert result == 0.7

        result = resolver.extract_value(simple_state, "config")
        assert result == {"nested": "value"}

        # Test error handling still works
        result = resolver.extract_value(simple_state, "missing", default="default")
        assert result == "default"

    def test_error_handling_complex_paths(self, resolver):
        """Test error handling with complex paths."""
        test_data = {"config": {"temp": 0.7}, "messages": ["hello", "world"]}

        # Test missing intermediate objects
        result = resolver.extract_value(test_data, "missing.field", default="error")
        assert result == "error"

        # Test array access on non-arrays
        result = resolver.extract_value(test_data, "config[0]", default="error")
        assert result == "error"

        # Test field access on arrays
        result = resolver.extract_value(test_data, "messages.missing", default="error")
        assert result == "error"

        # Test deeply nested missing paths
        result = resolver.extract_value(test_data, "a.b.c.d.e", default="deep_error")
        assert result == "deep_error"

    def test_edge_cases(self, resolver):
        """Test edge cases and unusual inputs."""
        test_data = {
            "empty_list": [],
            "empty_dict": {},
            "null_value": None,
            "nested": {"empty": [], "null": None},
        }

        # Test empty containers
        result = resolver.extract_value(test_data, "empty_list[0]", default="empty")
        assert result == "empty"

        result = resolver.extract_value(test_data, "empty_dict.field", default="empty")
        assert result == "empty"

        # Test null values in path
        result = resolver.extract_value(test_data, "null_value.field", default="null")
        assert result == "null"

        result = resolver.extract_value(test_data, "nested.null.field", default="nested_null")
        assert result == "nested_null"
