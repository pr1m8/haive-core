"""
Tests for SchemaComposer focusing on field extraction, engine component integration,
and schema composition with logging and debugging.
"""
import logging
import operator
import pytest
from typing import List, Dict, Any, Optional, Annotated, Union
from pydantic import BaseModel, Field
from rich.console import Console

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage

from haive.core.schema.schema_composer import SchemaComposer
from haive.core.schema.state_schema import StateSchema
from haive.core.schema.ui import SchemaUI

from haive.core.engine.aug_llm import AugLLMConfig
from haive.core.models.llm.base import AzureLLMConfig


# Set up logging for tests
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("schema_composer_tests")
console = Console()


# Sample models for testing
class SimpleModel(BaseModel):
    """Simple model for testing."""
    name: str = Field(description="Name field")
    value: int = Field(default=0, description="Value field")


class SearchResult(BaseModel):
    """Search result model for testing."""
    answer: str = Field(description="Answer to the query")
    sources: List[str] = Field(default_factory=list, description="Source documents")
    confidence: float = Field(default=0.0, description="Confidence score")


class ChatState(StateSchema):
    """Chat state with messages and reducers."""
    messages: Annotated[List[BaseMessage], operator.add] = Field(
        default_factory=list, 
        description="Conversation messages"
    )
    context: List[str] = Field(default_factory=list, description="Context documents")
    query: str = Field(default="", description="User query")


# Simple test for SchemaComposer basics
def test_schema_composer_basics():
    """Test basic SchemaComposer functionality."""
    # Create a composer
    composer = SchemaComposer(name="TestSchema")
    
    # Add fields
    composer.add_field(
        name="name",
        field_type=str,
        description="Name field"
    )
    
    composer.add_field(
        name="value",
        field_type=int,
        default=0,
        description="Value field"
    )
    
    composer.add_field(
        name="active",
        field_type=bool,
        default=False,
        description="Active status"
    )
    
    # Build schema
    schema = composer.build()
    
    # Log schema details
    logger.info(f"Created schema: {schema.__name__}")
    logger.info(f"Schema fields: {list(schema.model_fields.keys())}")
    
    # Verify fields
    assert hasattr(schema, "model_fields")
    assert "name" in schema.model_fields
    assert "value" in schema.model_fields
    assert "active" in schema.model_fields
    
    # Create instance
    instance = schema(name="Test")
    assert instance.name == "Test"
    assert instance.value == 0
    assert instance.active == False
    
    # Display with rich UI
    SchemaUI.display_schema(schema, "Basic Schema")
    SchemaUI.display_schema_code(schema, "Schema Code")


def test_schema_composer_from_model():
    """Test SchemaComposer with Pydantic model."""
    # Add fields from model
    composer = SchemaComposer(name="ModelSchema")
    composer.add_fields_from_model(SimpleModel)
    
    # Log fields
    logger.info(f"Fields after adding model: {list(composer.fields.keys())}")
    
    # Build schema
    schema = composer.build()
    
    # Log schema details
    logger.info(f"Created schema: {schema.__name__}")
    logger.info(f"Schema fields: {list(schema.model_fields.keys())}")
    
    # Verify fields from model
    assert hasattr(schema, "model_fields")
    assert "name" in schema.model_fields
    assert "value" in schema.model_fields
    
    # Display with rich UI
    SchemaUI.display_schema(schema, "Schema From Model")


def test_schema_composer_with_reducers():
    """Test SchemaComposer with reducer functions."""
    # Create composer
    composer = SchemaComposer(name="ReducerSchema")
    
    # Add field with reducer
    composer.add_field(
        name="messages",
        field_type=List[BaseMessage],
        default_factory=list,
        reducer=operator.add,
        description="Messages with add reducer"
    )
    
    composer.add_field(
        name="counter",
        field_type=int,
        default=0,
        reducer=operator.add,
        description="Counter with add reducer"
    )
    
    # Add normal field
    composer.add_field(
        name="query",
        field_type=str,
        default="",
        description="User query"
    )
    
    # Build schema
    schema = composer.build()
    
    # Log schema details
    logger.info(f"Created schema: {schema.__name__}")
    logger.info(f"Schema fields: {list(schema.model_fields.keys())}")
    logger.info(f"Reducers: {schema.__serializable_reducers__}")
    
    # Verify reducers
    assert hasattr(schema, "__serializable_reducers__")
    assert "messages" in schema.__serializable_reducers__
    assert "counter" in schema.__serializable_reducers__
    
    # Create instance
    instance = schema(
        messages=[HumanMessage(content="Hello")],
        counter=5
    )
    
    # Test reducer functionality
    updated = instance.apply_reducers({
        "messages": [AIMessage(content="Hi there")],
        "counter": 3
    })
    
    assert len(updated.messages) == 2
    assert updated.counter == 8
    
    # Display with rich UI
    SchemaUI.display_schema(schema, "Schema With Reducers")


def test_schema_composer_from_components():
    """Test SchemaComposer.from_components class method."""
    # Create components
    components = [
        SimpleModel,
        SearchResult,
        ChatState
    ]
    
    # Create schema from components
    schema = SchemaComposer.from_components(components, name="ComponentsSchema")
    
    # Log schema details
    logger.info(f"Created schema: {schema.__name__}")
    logger.info(f"Schema fields: {list(schema.model_fields.keys())}")
    
    if hasattr(schema, "__reducer_fields__"):
        logger.info(f"Reducer fields: {list(schema.__reducer_fields__.keys())}")
    
    if hasattr(schema, "__shared_fields__"):
        logger.info(f"Shared fields: {schema.__shared_fields__}")
    
    # Verify fields from all components
    assert hasattr(schema, "model_fields")
    
    # Fields from SimpleModel
    assert "name" in schema.model_fields
    assert "value" in schema.model_fields
    
    # Fields from SearchResult
    assert "answer" in schema.model_fields
    assert "sources" in schema.model_fields
    assert "confidence" in schema.model_fields
    
    # Fields from ChatState
    assert "messages" in schema.model_fields
    assert "context" in schema.model_fields
    assert "query" in schema.model_fields
    
    # Verify reducers were preserved
    assert hasattr(schema, "__serializable_reducers__")
    assert "messages" in schema.__serializable_reducers__
    
    # Display with rich UI
    SchemaUI.display_schema(schema, "Schema From Components")
    SchemaUI.display_schema_code(schema, "Schema Code")


def test_schema_composer_with_shared_fields():
    """Test SchemaComposer with shared fields."""
    # Create composer
    composer = SchemaComposer(name="SharedFieldsSchema")
    
    # Add shared fields
    composer.add_field(
        name="messages",
        field_type=List[BaseMessage],
        default_factory=list,
        shared=True,
        description="Shared messages field"
    )
    
    composer.add_field(
        name="context",
        field_type=List[str],
        default_factory=list,
        shared=True,
        description="Shared context field"
    )
    
    # Add non-shared field
    composer.add_field(
        name="local_data",
        field_type=Dict[str, Any],
        default_factory=dict,
        description="Local non-shared data"
    )
    
    # Build schema
    schema = composer.build()
    
    # Log schema details
    logger.info(f"Created schema: {schema.__name__}")
    logger.info(f"Schema fields: {list(schema.model_fields.keys())}")
    logger.info(f"Shared fields: {schema.__shared_fields__}")
    
    # Verify shared fields
    assert hasattr(schema, "__shared_fields__")
    assert "messages" in schema.__shared_fields__
    assert "context" in schema.__shared_fields__
    assert "local_data" not in schema.__shared_fields__
    
    # Display with rich UI
    SchemaUI.display_schema(schema, "Schema With Shared Fields")


def test_schema_composer_create_message_state():
    """Test SchemaComposer.create_message_state convenience method."""
    # Create schema with messages field and additional fields
    schema = SchemaComposer.create_message_state(
        additional_fields={
            "query": (str, ""),
            "results": (List[str], [])
        },
        name="MessageState"
    )
    
    # Log schema details
    logger.info(f"Created schema: {schema.__name__}")
    logger.info(f"Schema fields: {list(schema.model_fields.keys())}")
    
    if hasattr(schema, "__serializable_reducers__"):
        logger.info(f"Reducers: {schema.__serializable_reducers__}")
    
    # Verify fields
    assert hasattr(schema, "model_fields")
    assert "messages" in schema.model_fields
    assert "query" in schema.model_fields
    assert "results" in schema.model_fields
    
    # Verify messages field has reducer
    assert hasattr(schema, "__serializable_reducers__")
    assert "messages" in schema.__serializable_reducers__
    
    # Create instance
    instance = schema(query="Test query")
    assert instance.query == "Test query"
    assert isinstance(instance.messages, list)
    assert isinstance(instance.results, list)
    
    # Display with rich UI
    SchemaUI.display_schema(schema, "Message State Schema")


def test_schema_composer_with_engine():
    """Test SchemaComposer with AugLLMConfig engine."""
    # Create AugLLMConfig
    aug_llm = AugLLMConfig(
        name="test_llm",
        system_message="You are a helpful assistant.",
        structured_output_model=SearchResult,
        llm_config=AzureLLMConfig(model="gpt-4o")
    )
    
    # Create schema from engine
    composer = SchemaComposer(name="EngineSchema")
    composer.add_fields_from_engine(aug_llm)
    
    # Log extracted fields
    logger.info(f"Extracted fields: {list(composer.fields.keys())}")
    logger.info(f"Input fields: {composer.input_fields}")
    logger.info(f"Output fields: {composer.output_fields}")
    
    # Build schema
    schema = composer.build()
    
    # Log schema details
    logger.info(f"Created schema: {schema.__name__}")
    logger.info(f"Schema fields: {list(schema.model_fields.keys())}")
    
    # Log engine I/O mappings if present
    if hasattr(schema, "__engine_io_mappings__"):
        logger.info(f"Engine I/O mappings: {schema.__engine_io_mappings__}")
    
    # Verify schema has fields
    assert hasattr(schema, "model_fields")
    
    # Always verify messages field is present (standard field)
    assert "messages" in schema.model_fields
    
    # Display with rich UI
    SchemaUI.display_schema(schema, "Schema From AugLLM Engine")


def test_schema_composer_engine_io_tracking():
    """Test engine I/O field tracking in SchemaComposer."""
    # Create composer
    composer = SchemaComposer(name="IOTrackingSchema")
    
    # Add fields
    composer.add_field(
        name="query",
        field_type=str,
        default="",
        description="User query"
    )
    
    composer.add_field(
        name="context",
        field_type=List[str],
        default_factory=list,
        description="Context documents"
    )
    
    composer.add_field(
        name="response",
        field_type=str,
        default="",
        description="LLM response"
    )
    
    # Mark fields as inputs/outputs for specific engines
    composer.mark_as_input_field("query", "retriever")
    composer.mark_as_output_field("context", "retriever")
    
    composer.mark_as_input_field("query", "llm")
    composer.mark_as_input_field("context", "llm")
    composer.mark_as_output_field("response", "llm")
    
    # Build schema
    schema = composer.build()
    
    # Log schema details
    logger.info(f"Created schema: {schema.__name__}")
    logger.info(f"Schema fields: {list(schema.model_fields.keys())}")
    
    # Log engine I/O mappings
    logger.info(f"Engine I/O mappings: {schema.__engine_io_mappings__}")
    logger.info(f"Input fields: {schema.__input_fields__}")
    logger.info(f"Output fields: {schema.__output_fields__}")
    
    # Verify engine I/O mappings
    assert hasattr(schema, "__engine_io_mappings__")
    assert "retriever" in schema.__engine_io_mappings__
    assert "llm" in schema.__engine_io_mappings__
    
    # Verify input fields
    assert hasattr(schema, "__input_fields__")
    assert "retriever" in schema.__input_fields__
    assert "query" in schema.__input_fields__["retriever"]
    
    assert "llm" in schema.__input_fields__
    assert "query" in schema.__input_fields__["llm"]
    assert "context" in schema.__input_fields__["llm"]
    
    # Verify output fields
    assert hasattr(schema, "__output_fields__")
    assert "retriever" in schema.__output_fields__
    assert "context" in schema.__output_fields__["retriever"]
    
    assert "llm" in schema.__output_fields__
    assert "response" in schema.__output_fields__["llm"]
    
    # Display with rich UI
    SchemaUI.display_schema(schema, "Schema With Engine I/O Tracking")


def test_schema_composer_merge():
    """Test SchemaComposer merge method."""
    # Create first composer
    composer1 = SchemaComposer(name="FirstSchema")
    composer1.add_field(
        name="field1",
        field_type=str,
        description="Field from first composer"
    )
    composer1.add_field(
        name="shared_field",
        field_type=int,
        default=0,
        description="Shared field (from first)"
    )
    
    # Create second composer
    composer2 = SchemaComposer(name="SecondSchema")
    composer2.add_field(
        name="field2",
        field_type=str,
        description="Field from second composer"
    )
    composer2.add_field(
        name="shared_field",
        field_type=int,
        default=100,
        description="Shared field (from second)"
    )
    
    # Merge composers
    merged = composer1.merge(composer2)
    
    # Log merged fields
    logger.info(f"Merged composer fields: {list(merged.fields.keys())}")
    
    # Build schema
    schema = merged.build()
    
    # Log schema details
    logger.info(f"Created schema: {schema.__name__}")
    logger.info(f"Schema fields: {list(schema.model_fields.keys())}")
    
    # Verify fields from both composers
    assert hasattr(schema, "model_fields")
    assert "field1" in schema.model_fields
    assert "field2" in schema.model_fields
    assert "shared_field" in schema.model_fields
    
    # Verify shared field has first composer's definition (priority)
    instance = schema(field1="test", field2="test2")
    assert instance.shared_field == 0  # First composer's default
    
    # Display with rich UI
    SchemaUI.display_schema(schema, "Merged Schema")


def test_schema_composer_input_output_schema():
    """Test SchemaComposer for input/output schema derivation."""
    # Create AugLLMConfig with input/output schema
    aug_llm = AugLLMConfig(
        name="io_llm",
        system_message="You are a helpful assistant.",
        structured_output_model=SearchResult,
        llm_config=AzureLLMConfig(model="gpt-4o")
    )
    
    # Create input schema
    input_schema = SchemaComposer.compose_input_schema(
        components=[aug_llm],
        name="InputSchema"
    )
    
    # Log input schema details
    logger.info(f"Input schema: {input_schema.__name__}")
    logger.info(f"Input schema fields: {list(input_schema.model_fields.keys())}")
    
    # Create output schema
    output_schema = SchemaComposer.compose_output_schema(
        components=[aug_llm],
        name="OutputSchema"
    )
    
    # Log output schema details
    logger.info(f"Output schema: {output_schema.__name__}")
    logger.info(f"Output schema fields: {list(output_schema.model_fields.keys())}")
    
    # Verify schemas have fields
    assert hasattr(input_schema, "model_fields")
    assert hasattr(output_schema, "model_fields")
    
    # Basic assertions on expected fields
    assert "messages" in input_schema.model_fields  # Standard input field
    assert "content" in output_schema.model_fields  # Standard output field
    
    # Display with rich UI
    SchemaUI.display_schema(input_schema, "Input Schema")
    SchemaUI.display_schema(output_schema, "Output Schema")
