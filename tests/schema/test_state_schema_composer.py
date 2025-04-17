# tests/core/schema/test_state_schema_composer.py

import pytest
import logging
from typing import Dict, List, Optional, Any, Sequence, Annotated
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages

from haive_core.schema.state_schema import StateSchema
from haive_core.schema.schema_composer import SchemaComposer
from haive_core.engine.base import Engine, InvokableEngine, EngineType

# Set up logging for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Mock engine for testing
class MockEngine(InvokableEngine):
    # Proper type annotation for engine_type field override
    engine_type: EngineType = Field(default=EngineType.LLM)
    name: str = "mock_engine"
    
    def create_runnable(self, runnable_config=None):
        return lambda x: x
    
    def derive_input_schema(self):
        from pydantic import create_model
        return create_model(
            "MockInputSchema",
            query=(str, ""),
            temperature=(float, 0.7)
        )
        
    def derive_output_schema(self):
        from pydantic import create_model
        return create_model(
            "MockOutputSchema",
            result=(str, ""),
            confidence=(float, 0.0)
        )
        
    def get_schema_fields(self):
        return {
            "query": (str, ""),
            "result": (str, ""),
            "messages": (List[BaseMessage], Field(default_factory=list))
        }

# Test fixtures
@pytest.fixture
def mock_engine():
    """Create a mock engine for testing."""
    return MockEngine(name="test_engine")

@pytest.fixture
def sample_model():
    """Create a sample model class for testing."""
    class SampleModel(BaseModel):
        name: str
        age: int = 0
        active: bool = True
        notes: Optional[str] = Field(default=None, description="Additional notes")
    
    return SampleModel

# Tests
def test_schema_from_engines(mock_engine):
    """Test creating a schema from engines."""
    logger.info("Testing schema creation from engines")
    
    # Use compose_schema method
    schema_cls = SchemaComposer.compose_schema([mock_engine], name="EngineSchema")
    logger.debug(f"Created schema class: {schema_cls.__name__}")
    
    # Verify schema has expected fields
    assert schema_cls.__name__ == "EngineSchema"
    assert issubclass(schema_cls, StateSchema)
    
    # Check fields from engine
    assert "query" in schema_cls.model_fields
    assert "result" in schema_cls.model_fields
    assert "messages" in schema_cls.model_fields
    
    # Create instance and check defaults
    instance = schema_cls()
    logger.debug(f"Instance fields: {instance.model_dump()}")
    
    assert instance.query == ""
    assert instance.result == ""
    assert isinstance(instance.messages, list)
    assert len(instance.messages) == 0

def test_schema_from_model(sample_model):
    """Test creating a schema from a model."""
    logger.info("Testing schema creation from model")
    
    # The required name field should be made optional
    schema_cls = SchemaComposer.compose_schema([sample_model], name="ModelSchema")
    logger.debug(f"Created schema class: {schema_cls.__name__}")
    
    # Verify schema has expected fields
    assert "name" in schema_cls.model_fields
    assert "age" in schema_cls.model_fields
    assert "active" in schema_cls.model_fields
    assert "notes" in schema_cls.model_fields
    
    # Create instance and check defaults
    instance = schema_cls()
    logger.debug(f"Instance fields: {instance.model_dump()}")
    
    # The name field should be None because it was made optional
    assert instance.name is None
    assert instance.age == 0
    assert instance.active is True
    assert instance.notes is None

def test_derived_input_schema(mock_engine):
    """Test creating an input schema from engines."""
    logger.info("Testing input schema creation")
    
    input_schema = SchemaComposer.compose_input_schema([mock_engine], name="InputSchema")
    logger.debug(f"Created input schema: {input_schema.__name__}")
    
    # Verify schema has expected fields
    assert input_schema.__name__ == "InputSchema"
    assert "query" in input_schema.model_fields
    assert "temperature" in input_schema.model_fields
    
    # Output fields should not be present
    assert "result" not in input_schema.model_fields
    assert "confidence" not in input_schema.model_fields

def test_derived_output_schema(mock_engine):
    """Test creating an output schema from engines."""
    logger.info("Testing output schema creation")
    
    output_schema = SchemaComposer.compose_output_schema([mock_engine], name="OutputSchema")
    logger.debug(f"Created output schema: {output_schema.__name__}")
    
    # Verify schema has expected fields
    assert output_schema.__name__ == "OutputSchema"
    assert "result" in output_schema.model_fields
    assert "confidence" in output_schema.model_fields
    
    # Input fields should not be present
    assert "query" not in output_schema.model_fields
    assert "temperature" not in output_schema.model_fields

def test_create_schema_for_components(mock_engine, sample_model):
    """Test creating a schema manager for components."""
    logger.info("Testing schema manager creation")
    
    schema_manager = SchemaComposer.create_schema_for_components(
        [mock_engine, sample_model],
        name="CombinedSchema"
    )
    logger.debug(f"Created schema manager with fields: {list(schema_manager.fields.keys())}")
    
    # Verify schema manager has expected fields
    assert schema_manager.name == "CombinedSchema"
    
    # Fields from engine
    assert "query" in schema_manager.fields
    assert "result" in schema_manager.fields
    
    # Fields from model
    assert "name" in schema_manager.fields
    assert "age" in schema_manager.fields
    assert "active" in schema_manager.fields
    assert "notes" in schema_manager.fields
    
    # Always added
    assert "messages" in schema_manager.fields

def test_add_field_method():
    """Test adding fields to a SchemaComposer."""
    composer = SchemaComposer("FieldTestSchema")
    
    # Add primitive fields
    composer.add_field("text_field", str, default="default text")
    composer.add_field("int_field", int, default=42)
    composer.add_field("bool_field", bool, default=True)
    
    # Add a field with default_factory
    composer.add_field("list_field", List[str], default_factory=list)
    
    # Add an optional field
    composer.add_field("optional_field", Optional[str], default=None)
    
    # Add a field with a reducer
    composer.add_field(
        "messages",
        Annotated[Sequence[BaseMessage], add_messages],
        default_factory=list,
        reducer=add_messages
    )
    
    # Build the schema and verify fields
    schema_cls = composer.build()
    instance = schema_cls()
    
    assert instance.text_field == "default text"
    assert instance.int_field == 42
    assert instance.bool_field is True
    assert isinstance(instance.list_field, list)
    assert instance.optional_field is None
    assert isinstance(instance.messages, list)

def test_add_fields_from_model():
    """Test adding fields from a Pydantic model."""
    class TestModel(BaseModel):
        required_field: str
        optional_field: Optional[str] = None
        default_field: int = 100
        list_field: List[str] = Field(default_factory=list)
    
    composer = SchemaComposer("ModelFieldsSchema")
    composer.add_fields_from_model(TestModel)
    
    # Build and verify
    schema_cls = composer.build()
    instance = schema_cls()
    
    # Required field should be made optional with None default
    assert instance.required_field is None
    assert instance.optional_field is None
    assert instance.default_field == 100
    assert isinstance(instance.list_field, list)

def test_message_state_creation():
    """Test creating a message state schema."""
    schema_cls = SchemaComposer.create_message_state(
        additional_fields={
            "query": (str, ""),
            "result": (str, "")
        },
        name="TestMessageState"
    )
    
    # Verify schema
    assert schema_cls.__name__ == "TestMessageState"
    assert issubclass(schema_cls, StateSchema)
    
    # Check fields
    instance = schema_cls()
    assert "messages" in schema_cls.model_fields
    assert "query" in schema_cls.model_fields
    assert "result" in schema_cls.model_fields
    
    # Check defaults
    assert isinstance(instance.messages, list)
    assert instance.query == ""
    assert instance.result == ""
    
    # Check reducer is set up
    assert "messages" in schema_cls.__reducer_fields__

def test_compose_as_state_schema():
    """Test the compose_as_state_schema method."""
    class InputModel(BaseModel):
        query: str
        temperature: float = 0.7
    
    class OutputModel(BaseModel):
        result: str
        confidence: float = 0.0
    
    schema_cls = SchemaComposer.compose_as_state_schema(
        [InputModel, OutputModel],
        name="CombinedStateSchema",
        include_messages=True,
        include_runnable_config=True
    )
    
    # Verify schema
    assert schema_cls.__name__ == "CombinedStateSchema"
    assert issubclass(schema_cls, StateSchema)
    
    # Check all fields are present
    instance = schema_cls()
    assert instance.query is None  # Required field made optional
    assert instance.temperature == 0.7
    assert instance.result is None  # Required field made optional
    assert instance.confidence == 0.0
    assert isinstance(instance.messages, list)
    assert isinstance(instance.runnable_config, dict)

def test_schema_with_reducers():
    """Test creating a schema with reducers."""
    # Create a simple reducer function
    def concat_reducer(a: str, b: str) -> str:
        return (a or "") + (b or "")
    
    composer = SchemaComposer("ReducerSchema")
    composer.add_field("text", str, default="", reducer=concat_reducer)
    composer.add_field(
        "messages",
        Annotated[Sequence[BaseMessage], add_messages],
        default_factory=list,
        reducer=add_messages
    )
    
    # Build schema
    schema_cls = composer.build()
    
    # Verify reducers
    assert "text" in schema_cls.__reducer_fields__
    assert schema_cls.__reducer_fields__["text"] == concat_reducer
    assert "messages" in schema_cls.__reducer_fields__
    assert schema_cls.__reducer_fields__["messages"] == add_messages

def test_shared_fields():
    """Test schema with shared fields."""
    composer = SchemaComposer("SharedFieldSchema")
    composer.add_field("private_field", str, default="private")
    composer.add_field("shared_field", int, default=0, shared=True)
    
    # Build schema
    schema_cls = composer.build()
    
    # Verify shared fields
    assert "shared_field" in schema_cls.__shared_fields__
    assert "private_field" not in schema_cls.__shared_fields__