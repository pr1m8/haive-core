"""
Tests for composing state schemas from input and output schemas.

These tests verify that the SchemaComposer can correctly create a state schema
by combining existing input and output schemas, rather than deriving them from a state schema.
"""

import logging
from typing import Any, Dict, List

import pytest
from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel, Field

from haive.core.schema.schema_composer import SchemaComposer
from haive.core.schema.state_schema import StateSchema

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SampleInputSchema(BaseModel):
    """Input schema for testing."""

    messages: List[BaseMessage] = Field(default_factory=list)
    query: str = Field(default="")
    context: Dict[str, Any] = Field(default_factory=dict)


class SampleOutputSchema(BaseModel):
    """Output schema for testing."""

    messages: List[BaseMessage] = Field(default_factory=list)
    response: str = Field(default="")
    sources: List[Dict[str, Any]] = Field(default_factory=list)


def test_compose_state_from_io_schemas():
    """Test creating a state schema from input and output schemas."""
    # Create a state schema from input and output schemas
    logger.info("Creating state schema from input and output schemas")
    state_schema = SchemaComposer.create_state_from_io_schemas(
        SampleInputSchema, SampleOutputSchema, name="ComposedTestSchema"
    )

    # Print debug info
    logger.info(f"Schema name: {state_schema.__name__}")
    logger.info(f"Schema bases: {state_schema.__bases__}")
    logger.info(f"Schema fields: {list(state_schema.model_fields.keys())}")

    # Verify the result is a proper state schema
    assert issubclass(state_schema, StateSchema)
    assert issubclass(state_schema, SampleInputSchema)
    assert issubclass(state_schema, SampleOutputSchema)

    # Verify schema has the right name
    assert state_schema.__name__ == "ComposedTestSchema"

    # Verify all expected fields exist
    assert "messages" in state_schema.model_fields
    assert "query" in state_schema.model_fields
    assert "context" in state_schema.model_fields
    assert "response" in state_schema.model_fields
    assert "sources" in state_schema.model_fields

    # Debug the contents of __engine_io_mappings__
    logger.info(
        f"Engine I/O mappings: {getattr(state_schema, '__engine_io_mappings__', {})}"
    )

    # Verify engine I/O mappings have been properly assigned
    assert hasattr(state_schema, "__engine_io_mappings__")

    # Debug engine I/O mappings just before assertion
    mappings = state_schema.__engine_io_mappings__
    logger.info(f"Engine I/O mappings keys: {list(mappings.keys())}")

    # Check for default engine in mappings and messages field is properly configured
    if "default" not in state_schema.__engine_io_mappings__:
        # If test would fail, print detailed debug info then fail
        logger.error(
            f"'default' not in engine_io_mappings! Available keys: {list(state_schema.__engine_io_mappings__.keys())}"
        )
        logger.error(
            f"Full engine_io_mappings content: {state_schema.__engine_io_mappings__}"
        )
        assert "default" in state_schema.__engine_io_mappings__

    # Continue with the test
    default_mapping = state_schema.__engine_io_mappings__["default"]
    logger.info(f"Default engine mapping content: {default_mapping}")

    assert "inputs" in default_mapping
    assert "outputs" in default_mapping
    assert "messages" in default_mapping["inputs"]
    assert "messages" in default_mapping["outputs"]


def test_state_instance_creation():
    """Test creating and using a state instance from composed schema."""
    # Create a state schema from input and output schemas
    logger.info("Creating state schema for instance creation test")
    state_schema = SchemaComposer.create_state_from_io_schemas(
        SampleInputSchema, SampleOutputSchema, name="ComposedTestSchema"
    )

    # Create a state instance with some initial values
    logger.info("Creating state instance with initial values")
    state = state_schema(
        messages=[HumanMessage(content="Hello")],
        query="What is the weather?",
        context={"location": "New York"},
    )

    # Verify the instance has the expected values
    assert len(state.messages) == 1
    assert state.messages[0].content == "Hello"
    assert state.query == "What is the weather?"
    assert state.context["location"] == "New York"
    assert state.response == ""  # Default value
    assert len(state.sources) == 0  # Default empty list

    # Test updating the state
    logger.info("Updating state instance")
    state.response = "The weather is sunny"
    state.sources = [{"url": "weather.com", "content": "Sunny and 75°F"}]

    assert state.response == "The weather is sunny"
    assert len(state.sources) == 1
    assert state.sources[0]["url"] == "weather.com"


def test_compose_state_directly():
    """Test using the compose_state_from_io method directly."""
    # Create a composer
    logger.info("Creating composer for direct composition test")
    composer = SchemaComposer(name="DirectComposedSchema")

    # Add some field definitions to the composer
    from typing import List

    from langchain_core.messages import BaseMessage

    # Configure messages field with reducer
    try:
        from langgraph.graph import add_messages

        reducer = add_messages
    except ImportError:
        # Fallback
        def reducer(a, b):
            return (a or []) + (b or [])

    # Add messages field with reducer
    logger.info("Adding messages field with reducer")
    composer.add_field(
        name="messages",
        field_type=List[BaseMessage],
        default_factory=list,
        description="Messages for conversation",
        reducer=reducer,
        shared=True,
        input_for=["llm"],
        output_from=["llm"],
    )

    # Log engine_io_mappings before composing
    logger.info(f"Engine I/O mappings before composing: {composer.engine_io_mappings}")

    # Directly compose a state schema
    logger.info("Directly composing state schema")
    state_schema = composer.compose_state_from_io(SampleInputSchema, SampleOutputSchema)

    # Log engine_io_mappings after composing
    logger.info(
        f"Engine I/O mappings after composing: {state_schema.__engine_io_mappings__}"
    )

    # Verify it inherits from both input and output schemas
    assert issubclass(state_schema, StateSchema)
    assert issubclass(state_schema, SampleInputSchema)
    assert issubclass(state_schema, SampleOutputSchema)

    # Verify shared fields were properly transferred
    assert "messages" in state_schema.__shared_fields__

    # Verify reducer was configured
    assert "messages" in state_schema.__serializable_reducers__
    assert "messages" in state_schema.__reducer_fields__

    # Verify engine I/O mapping was maintained
    assert "llm" in state_schema.__engine_io_mappings__
    assert "messages" in state_schema.__engine_io_mappings__["llm"]["inputs"]
    assert "messages" in state_schema.__engine_io_mappings__["llm"]["outputs"]
