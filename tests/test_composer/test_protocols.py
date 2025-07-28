"""Tests for composer protocols - using real components only."""

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

from haive.core.graph.node.composer.protocols import (
    ExtractFunction,
    TransformFunction,
    UpdateFunction,
)
from haive.core.schema.prebuilt.messages_state import MessagesState


class SampleState(BaseModel):
    """Sample state for testing protocols."""

    messages: list[str] = Field(default_factory=list)
    temperature: float = Field(default=0.7)


class TestProtocols:
    """Test protocol implementations work correctly."""

    def test_extract_function_protocol(self):
        """Test that extract function protocol works with real implementations."""

        def simple_extract(state: SampleState, config: dict[str, Any]) -> list[str]:
            """Simple extract function implementation."""
            field_name = config.get("field_name", "messages")
            return getattr(state, field_name, [])

        # Test protocol compliance
        def use_extract_function(
            func: ExtractFunction, state: SampleState
        ) -> list[str]:
            return func(state, {"field_name": "messages"})

        # Create test state
        test_state = SampleState(messages=["hello", "world"])

        # Use the function through protocol
        result = use_extract_function(simple_extract, test_state)
        assert result == ["hello", "world"]

    def test_update_function_protocol(self):
        """Test that update function protocol works with real implementations."""

        def simple_update(
            result: str, state: SampleState, config: dict[str, Any]
        ) -> dict[str, Any]:
            """Simple update function implementation."""
            field_name = config.get("field_name", "messages")
            current = getattr(state, field_name, [])
            updated = [*list(current), result]
            return {field_name: updated}

        # Test protocol compliance
        def use_update_function(
            func: UpdateFunction, result: str, state: SampleState
        ) -> dict[str, Any]:
            return func(result, state, {"field_name": "messages"})

        # Create test state
        test_state = SampleState(messages=["initial"])

        # Use the function through protocol
        update = use_update_function(simple_update, "new_message", test_state)
        assert update == {"messages": ["initial", "new_message"]}

    def test_transform_function_protocol(self):
        """Test that transform function protocol works with real implementations."""

        def uppercase_transform(value: Any) -> str:
            """Simple transform function implementation."""
            return str(value).upper()

        def parse_number(value: Any) -> float:
            """Number parsing transform."""
            return float(value)

        # Test protocol compliance
        def use_transform_function(func: TransformFunction, value: Any) -> Any:
            return func(value)

        # Test string transform
        result = use_transform_function(uppercase_transform, "hello")
        assert result == "HELLO"

        # Test number transform
        result = use_transform_function(parse_number, "3.14")
        assert result == 3.14

    def test_real_state_extract_function(self):
        """Test extract function with real MessagesState."""

        def extract_messages(state: MessagesState, config: dict[str, Any]) -> list[Any]:
            """Extract messages from real MessagesState."""
            messages = getattr(state, "messages", [])
            return list(messages)

        # Create real MessagesState
        state = MessagesState(
            messages=[HumanMessage(content="Hello"), AIMessage(content="Hi there!")]
        )

        # Extract through protocol
        def use_extract(func: ExtractFunction) -> list[Any]:
            return func(state, {})

        messages = use_extract(extract_messages)
        assert len(messages) == 2
        assert messages[0].content == "Hello"
        assert messages[1].content == "Hi there!"

    def test_real_state_update_function(self):
        """Test update function with real MessagesState."""

        def update_with_message(
            result: AIMessage, state: MessagesState, config: dict[str, Any]
        ) -> dict[str, Any]:
            """Update messages with new AIMessage."""
            current_messages = list(getattr(state, "messages", []))
            current_messages.append(result)
            return {"messages": current_messages}

        # Create real state and result
        state = MessagesState(messages=[HumanMessage(content="Question")])
        new_message = AIMessage(content="Answef")

        # Update through protocol
        def use_update(func: UpdateFunction) -> dict[str, Any]:
            return func(new_message, state, {})

        update = use_update(update_with_message)
        assert "messages" in update
        assert len(update["messages"]) == 2
        assert update["messages"][0].content == "Question"
        assert update["messages"][1].content == "Answer"

    def test_transform_pipeline(self):
        """Test chaining multiple transform functions."""

        def strip_transform(value: Any) -> str:
            """Strip whitespace."""
            return str(value).strip()

        def lowercase_transform(value: Any) -> str:
            """Convert to lowercase."""
            return str(value).lower()

        def add_prefix(value: Any) -> str:
            """Add prefix."""
            return f"processed: {value}"

        # Chain transforms through protocol
        def apply_transforms(value: Any, transforms: list[TransformFunction]) -> Any:
            result = value
            for transform in transforms:
                result = transform(result)
            return result

        # Test transform pipeline
        input_value = "  HELLO WORLD  "
        transforms = [strip_transform, lowercase_transform, add_prefix]

        result = apply_transforms(input_value, transforms)
        assert result == "processed: hello world"

    def test_protocol_type_checking(self):
        """Test that protocols provide proper type checking."""

        # Valid extract function
        def valid_extract(state: SampleState, config: dict[str, Any]) -> str:
            return "extracted"

        # Valid update function
        def valid_update(
            result: str, state: SampleState, config: dict[str, Any]
        ) -> dict[str, Any]:
            return {"field": result}

        # Valid transform function
        def valid_transform(value: Any) -> Any:
            return value

        # These should work without type errors
        extract_func: ExtractFunction = valid_extract
        update_func: UpdateFunction = valid_update
        transform_func: TransformFunction = valid_transform

        # Test they can be called
        state = SampleState()
        config = {}

        extracted = extract_func(state, config)
        assert extracted == "extracted"

        updated = update_func("result", state, config)
        assert updated == {"field": "result"}

        transformed = transform_func("value")
        assert transformed == "value"
