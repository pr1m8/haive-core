"""Tests for PathResolver - using real components only."""

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

from haive.core.graph.node.composer import PathResolver
from haive.core.schema.prebuilt.messages_state import MessagesState


class SampleConfig(BaseModel):
    """Sample configuration model for testing."""

    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=1000)
    model: str = Field(default="gpt-4")


class SampleState(BaseModel):
    """Sample state model with nested structure for testing."""

    messages: list[str] = Field(default_factory=list)
    config: SampleConfig = Field(default_factory=SampleConfig)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TestPathResolver:
    """Test PathResolver with real Pydantic models and state objects."""

    @pytest.fixture
    def resolver(self):
        """Create a PathResolver instance."""
        return PathResolver()

    @pytest.fixture
    def dict_state(self):
        """Create test state as dict."""
        return {
            "messages": ["Hello", "World"],
            "temperature": 0.7,
            "config": {"model": "gpt-4", "temperature": 0.5},
        }

    @pytest.fixture
    def pydantic_state(self):
        """Create test state as Pydantic model."""
        return SampleState(
            messages=["Hello", "World"],
            config=SampleConfig(temperature=0.5, model="claude"),
            metadata={"session": "test123"},
        )

    @pytest.fixture
    def real_messages_state(self):
        """Create real MessagesState with LangChain messages."""
        return MessagesState(
            messages=[
                HumanMessage(
                    content="Hello"), AIMessage(
                    content="Hi there!")]
        )

    def test_simple_field_from_dict(self, resolver, dict_state):
        """Test extracting simple fields from dict."""
        # Extract existing field
        messages = resolver.extract_value(dict_state, "messages")
        assert messages == ["Hello", "World"]

        # Extract another field
        temp = resolver.extract_value(dict_state, "temperature")
        assert temp == 0.7

        # Missing field with default
        missing = resolver.extract_value(
            dict_state, "missing", default="default_value")
        assert missing == "default_value"

        # Missing field without default
        missing_none = resolver.extract_value(dict_state, "missing")
        assert missing_none is None

    def test_simple_field_from_pydantic(self, resolver, pydantic_state):
        """Test extracting simple fields from Pydantic model."""
        # Extract list field
        messages = resolver.extract_value(pydantic_state, "messages")
        assert messages == ["Hello", "World"]

        # Extract nested model field
        config = resolver.extract_value(pydantic_state, "config")
        assert isinstance(config, SampleConfig)
        assert config.temperature == 0.5
        assert config.model == "claude"

        # Extract dict field
        metadata = resolver.extract_value(pydantic_state, "metadata")
        assert metadata == {"session": "test123"}

        # Missing field
        missing = resolver.extract_value(
            pydantic_state, "nonexistent", default=[])
        assert missing == []

    def test_real_messages_state(self, resolver, real_messages_state):
        """Test with real MessagesState from haive."""
        # Extract messages field
        messages = resolver.extract_value(real_messages_state, "messages")
        assert len(messages) == 2
        assert isinstance(messages[0], HumanMessage)
        assert isinstance(messages[1], AIMessage)
        assert messages[0].content == "Hello"
        assert messages[1].content == "Hi there!"

        # Try to extract non-existent field
        missing = resolver.extract_value(
            real_messages_state, "agents", default=[])
        assert missing == []

    def test_none_and_empty_handling(self, resolver):
        """Test handling of None and empty objects."""
        # None object
        result = resolver.extract_value(None, "field", default="default")
        assert result == "default"

        # Empty dict
        result = resolver.extract_value({}, "field", default="default")
        assert result == "default"

        # Empty path
        result = resolver.extract_value(
            {"field": "value"}, "", default="default")
        assert result == "default"

        # None path
        result = resolver.extract_value(
            {"field": "value"}, None, default="default")
        assert result == "default"

    def test_different_object_types(self, resolver):
        """Test extraction from different object types."""

        # Plain object with attributes
        class PlainObject:
            def __init__(self):
                self.value = 42
                self.name = "test"

        obj = PlainObject()
        assert resolver.extract_value(obj, "value") == 42
        assert resolver.extract_value(obj, "name") == "test"
        assert resolver.extract_value(obj, "missing", default=0) == 0

        # List (has __getitem__ but not get)
        lst = ["a", "b", "c"]
        # For now, simple paths don't support index access
        assert resolver.extract_value(lst, "0", default="default") == "default"

        # String (immutable)
        s = "hello"
        assert resolver.extract_value(
            s, "field", default="default") == "default"

    def test_error_handling(self, resolver):
        """Test error handling in extraction."""

        # Object that raises on attribute access
        class ErrorObject:
            def __getattr__(self, name):
                raise RuntimeError(f"Cannot access {name}")

        obj = ErrorObject()
        result = resolver.extract_value(obj, "field", default="safe")
        assert result == "safe"

        # Dict that raises on key access
        class ErrorDict(dict):
            def __getitem__(self, key):
                raise KeyError(f"Key error: {key}")

            def get(self, key, default=None):
                raise KeyError(f"Key error: {key}")

        d = ErrorDict()
        result = resolver.extract_value(d, "field", default="safe")
        assert result == "safe"

    def test_complex_state_extraction(self, resolver):
        """Test extraction from complex nested state."""
        # Create a complex state similar to what nodes use

        from pydantic import BaseModel, Field

        class AgentState(BaseModel):
            agent_id: str
            local_data: dict[str, Any] = Field(default_factory=dict)
            status: str = "active"

        class MultiAgentState(BaseModel):
            messages: list[str] = Field(default_factory=list)
            agent_states: dict[str, AgentState] = Field(default_factory=dict)
            active_agent: str | None = None
            global_config: SampleConfig = Field(default_factory=SampleConfig)

        state = MultiAgentState(
            messages=["Start", "Processing"],
            agent_states={
                "agent1": AgentState(
                    agent_id="agent1", local_data={"task": "analyze"}, status="busy"
                ),
                "agent2": AgentState(
                    agent_id="agent2", local_data={"task": "summarize"}, status="idle"
                ),
            },
            active_agent="agent1",
            global_config=SampleConfig(temperature=0.3),
        )

        # Extract various fields
        messages = resolver.extract_value(state, "messages")
        assert messages == ["Start", "Processing"]

        agent_states = resolver.extract_value(state, "agent_states")
        assert isinstance(agent_states, dict)
        assert "agent1" in agent_states
        assert "agent2" in agent_states

        active = resolver.extract_value(state, "active_agent")
        assert active == "agent1"

        config = resolver.extract_value(state, "global_config")
        assert isinstance(config, SampleConfig)
        assert config.temperature == 0.3
