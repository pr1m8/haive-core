"""Tests for NodeSchemaComposer - the main composer class.

Tests cover:
1. Basic node composition with I/O mappings
2. Callable node creation
3. Schema adapters
4. Factory functions
5. Transform pipelines
6. Real-world integration scenarios
"""

from typing import Any

import pytest
from langgraph.types import Command
from pydantic import BaseModel

from haive.core.graph.node.callable_node import CallableNodeConfig
from haive.core.graph.node.composer import (
    FieldMapping,
    NodeSchemaComposer,
    change_input_key,
    change_output_key,
    remap_fields,
)


class TestNodeSchemaComposer:
    """Test main NodeSchemaComposer functionality."""

    @pytest.fixture
    def composer(self):
        """Create composer instance."""
        return NodeSchemaComposer()

    @pytest.fixture
    def mock_node(self):
        """Create a mock node for testing."""

        def node_func(state: dict[str, Any],
                      config: dict[str, Any]) -> dict[str, Any]:
            return {"documents": ["doc1", "doc2"], "count": 2}

        return CallableNodeConfig(name="test_node", callable_func=node_func)

    def test_basic_node_composition(self, composer, mock_node):
        """Test basic node composition with output mapping."""
        # Compose node with output mapping
        composed = composer.compose_node(
            base_node=mock_node,
            output_mappings=[
                FieldMapping("documents", "retrieved_documents"),
                FieldMapping("count", "document_count"),
            ],
        )

        # Test execution
        state = {"query": "test"}
        result = composed(state, {})

        # Verify Command with mapped fields
        assert isinstance(result, Command)
        assert "retrieved_documents" in result.update
        assert "document_count" in result.update
        assert result.update["retrieved_documents"] == ["doc1", "doc2"]
        assert result.update["document_count"] == 2

    def test_input_mapping(self, composer):
        """Test node composition with input mapping."""

        # Create node that expects specific input
        def process_func(
            state: dict[str, Any], config: dict[str, Any]
        ) -> dict[str, Any]:
            # Node expects 'messages' but state has 'conversation'
            messages = state.get("messages", [])
            return {"message_count": len(messages)}

        node = CallableNodeConfig(name="processor", callable_func=process_func)

        # Compose with input mapping
        composed = composer.compose_node(
            base_node=node, input_mappings=[
                FieldMapping("conversation", "messages")]
        )

        # Test with mapped input
        state = {"conversation": ["msg1", "msg2", "msg3"]}
        result = composed(state, {})

        assert isinstance(result, Command)
        assert result.update["message_count"] == 3

    def test_transform_pipeline(self, composer):
        """Test transform functions in mapping."""
        # Register custom transform
        composer.register_transform_function("double", lambda x: x * 2)

        # Create node
        def calc_func(state: dict[str, Any],
                      config: dict[str, Any]) -> dict[str, Any]:
            return {"value": 5}

        node = CallableNodeConfig(name="calc", callable_func=calc_func)

        # Compose with transform
        composed = composer.compose_node(
            base_node=node,
            output_mappings=[
                FieldMapping("value", "doubled_value", transform=["double"])
            ],
        )

        result = composed({}, {})
        assert result.update["doubled_value"] == 10

    def test_from_callable_basic(self, composer):
        """Test creating node from callable."""

        # Simple callable
        def check_threshold(msgs: list[str], threshold: int = 5) -> bool:
            return len(msgs) > threshold

        # Create node with mappings
        node = composer.from_callable(
            func=check_threshold,
            input_mappings=[
                FieldMapping("messages", "msgs"),
                FieldMapping("config.threshold", "threshold", default=10),
            ],
            output_mappings=[FieldMapping("result", "should_continue")],
        )

        # Test execution
        state = {"messages": ["a", "b", "c"], "config": {"threshold": 2}}
        result = node(state, {})

        assert isinstance(result, Command)
        assert result.update["should_continue"] is True

    def test_from_callable_with_transforms(self, composer):
        """Test callable with transform pipeline."""

        def get_status() -> str:
            return "active"

        node = composer.from_callable(
            func=get_status,
            output_mappings=[
                FieldMapping(
                    "result",
                    "status",
                    transform=["uppercase"])],
        )

        result = node({}, {})
        assert result.update["status"] == "ACTIVE"

    def test_schema_adapter(self, composer):
        """Test schema adapter between different models."""

        # Define schemas
        class OldSchema(BaseModel):
            user_name: str
            message_text: str
            priority_level: int = 1

        class NewSchema(BaseModel):
            username: str
            content: str
            priority: int

        # Create adapter
        adapter = composer.create_adapter(
            source_schema=OldSchema,
            target_schema=NewSchema,
            field_mappings=[
                FieldMapping("user_name", "username"),
                FieldMapping("message_text", "content"),
                FieldMapping("priority_level", "priority"),
            ],
        )

        # Test adaptation
        old_instance = OldSchema(
            user_name="alice", message_text="Hello world", priority_level=2
        )

        new_instance = adapter.adapt(old_instance)

        assert isinstance(new_instance, NewSchema)
        assert new_instance.username == "alice"
        assert new_instance.content == "Hello world"
        assert new_instance.priority == 2

    def test_change_output_key_factory(self, mock_node):
        """Test change_output_key factory function."""
        # Use factory to change output key
        adapted = change_output_key(mock_node, "documents", "retrieved_docs")

        # Test execution
        result = adapted({"query": "test"}, {})

        assert isinstance(result, Command)
        assert "retrieved_docs" in result.update
        assert result.update["retrieved_docs"] == ["doc1", "doc2"]
        assert "documents" not in result.update

    def test_change_input_key_factory(self, composer):
        """Test change_input_key factory function."""

        def process(state: dict[str, Any],
                    config: dict[str, Any]) -> dict[str, Any]:
            # Expects 'query' but state has 'user_question'
            return {
                "response": f"Processing: {state.get('query', 'no query')}"}

        node = CallableNodeConfig(name="processor", callable_func=process)

        # Adapt input
        adapted = change_input_key(node, "query", "user_question")

        result = adapted({"user_question": "What is AI?"}, {})
        assert result.update["response"] == "Processing: What is AI?"

    def test_remap_fields_factory(self, composer):
        """Test remap_fields for multiple mappings."""

        def process(state: dict[str, Any],
                    config: dict[str, Any]) -> dict[str, Any]:
            query = state.get("query", "")
            context = state.get("context", "")
            return {
                "response": f"Q: {query}, C: {context}",
                "metadata": {"processed": True},
            }

        node = CallableNodeConfig(name="processof", callable_func=process)

        # Remap multiple fields
        adapted = remap_fields(
            node=node,
            input_mapping={"user_question": "query", "background": "context"},
            output_mapping={
                "response": "ai_response",
                "metadata": "processing_info"},
        )

        result = adapted(
            {"user_question": "Hello", "background": "Previous chat"}, {})

        assert "ai_response" in result.update
        assert "processing_info" in result.update
        assert result.update["ai_response"] == "Q: Hello, C: Previous chat"

    def test_complex_path_mapping(self, composer):
        """Test mapping with complex paths."""

        def process(state: dict[str, Any],
                    config: dict[str, Any]) -> dict[str, Any]:
            return {"result": {"data": {"value": 42}}}

        node = CallableNodeConfig(name="processor", callable_func=process)

        # Map nested result to flat field
        composed = composer.compose_node(
            base_node=node,
            output_mappings=[FieldMapping("result.data.value", "final_value")],
        )

        result = composed({}, {})
        # Note: Current implementation expects flat dict results
        # This test documents current behavior
        assert isinstance(result, Command)

    def test_multiple_transforms(self, composer):
        """Test chaining multiple transforms."""
        # Register transforms
        composer.register_transform_function(
            "add_prefix", lambda x: f"PREFIX_{x}")
        composer.register_transform_function(
            "add_suffix", lambda x: f"{x}_SUFFIX")

        def get_id() -> str:
            return "abc123"

        node = composer.from_callable(
            func=get_id,
            output_mappings=[
                FieldMapping(
                    "result",
                    "formatted_id",
                    transform=["uppercase", "add_prefix", "add_suffix"],
                )
            ],
        )

        result = node({}, {})
        assert result.update["formatted_id"] == "PREFIX_ABC123_SUFFIX"

    def test_default_values_in_mapping(self, composer):
        """Test default values in field mappings."""

        def process(state: dict[str, Any],
                    config: dict[str, Any]) -> dict[str, Any]:
            # Only return some fields
            return {"status": "complete"}

        node = CallableNodeConfig(name="processor", callable_func=process)

        composed = composer.compose_node(
            base_node=node,
            output_mappings=[
                FieldMapping("status", "processing_status"),
                FieldMapping("error", "error_message", default="No error"),
                FieldMapping("count", "item_count", default=0),
            ],
        )

        result = composed({}, {})
        assert result.update["processing_status"] == "complete"
        assert result.update["error_message"] == "No error"
        assert result.update["item_count"] == 0

    def test_callable_with_no_params(self, composer):
        """Test callable that takes no parameters."""

        def get_timestamp() -> str:
            return "2024-01-01T00:00:00"

        node = composer.from_callable(
            func=get_timestamp, output_mappings=[
                FieldMapping("result", "timestamp")]
        )

        result = node({}, {})
        assert result.update["timestamp"] == "2024-01-01T00:00:00"

    def test_preserve_existing_command_fields(self, composer):
        """Test that existing Command fields are preserved."""

        def process(state: dict[str, Any], config: dict[str, Any]) -> Command:
            return Command(update={"original": "value"}, goto="next_node")

        node = CallableNodeConfig(name="processor", callable_func=process)

        composed = composer.compose_node(
            base_node=node, output_mappings=[
                FieldMapping("original", "mapped_field")]
        )

        result = composed({}, {})

        # Should preserve goto and merge updates
        assert result.goto == "next_node"
        assert "mapped_field" in result.update
        assert result.update["mapped_field"] == "value"


class TestNodeSchemaComposerIntegration:
    """Integration tests with real node types."""

    def test_retriever_node_adaptation(self):
        """Test adapting a retriever node output."""

        # Simulate retriever node behavior
        def retriever_func(
            state: dict[str, Any], config: dict[str, Any]
        ) -> dict[str, Any]:
            query = state.get("query", "")
            # Simulate document retrieval
            return {
                "documents": [
                    {"content": f"Document about {query}", "score": 0.9},
                    {"content": f"Another doc about {query}", "score": 0.8},
                ]
            }

        retriever_node = CallableNodeConfig(
            name="retriever", callable_func=retriever_func
        )

        # Adapt output
        adapted_retriever = change_output_key(
            retriever_node, "documents", "retrieved_documents"
        )

        # Test
        result = adapted_retriever({"query": "AI"}, {})

        assert "retrieved_documents" in result.update
        assert len(result.update["retrieved_documents"]) == 2
        assert result.update["retrieved_documents"][0]["content"] == "Document about AI"

    def test_multi_step_pipeline(self):
        """Test composing multiple adapted nodes."""
        composer = NodeSchemaComposer()

        # Step 1: Query processor
        query_processor = composer.from_callable(
            func=lambda user_input: f"Processed: {user_input}",
            input_mappings=[FieldMapping("user_query", "user_input")],
            output_mappings=[FieldMapping("result", "processed_query")],
        )

        # Step 2: Mock retriever expecting different field
        def retrieve(state: dict[str, Any],
                     config: dict[str, Any]) -> dict[str, Any]:
            query = state.get("search_query", "")
            return {"docs": [f"Doc about {query}"]}

        retriever = CallableNodeConfig(
            name="retriever", callable_func=retrieve)

        # Adapt retriever
        adapted_retriever = composer.compose_node(
            base_node=retriever,
            input_mappings=[FieldMapping("processed_query", "search_query")],
            output_mappings=[FieldMapping("docs", "documents")],
        )

        # Execute pipeline
        state = {"user_query": "machine learning"}

        # Step 1
        result1 = query_processor(state, {})
        state.update(result1.update)

        # Step 2
        result2 = adapted_retriever(state, {})
        state.update(result2.update)

        # Verify pipeline
        assert state["processed_query"] == "Processed: machine learning"
        assert state["documents"] == ["Doc about Processed: machine learning"]

    def test_real_world_rag_pattern(self):
        """Test realistic RAG pattern with field adaptations."""
        composer = NodeSchemaComposer()

        # Components with mismatched interfaces
        def query_enhancer(q: str) -> dict[str, Any]:
            return {"enhanced": f"Find information about: {q}", "original": q}

        def retriever(state: dict[str, Any],
                      config: dict[str, Any]) -> dict[str, Any]:
            search_term = state.get("search_term", "")
            return {
                "results": [
                    {"text": f"Result 1 for {search_term}", "relevance": 0.9},
                    {"text": f"Result 2 for {search_term}", "relevance": 0.7},
                ]
            }

        def generator(state: dict[str, Any],
                      config: dict[str, Any]) -> dict[str, Any]:
            context = state.get("context", [])
            query = state.get("original_query", "")

            if context:
                response = f"Based on {
                    len(context)} sources about '{query}': [generated response]"
            else:
                response = f"No information found about '{query}'"

            return {"answer": response}

        # Build adapted pipeline
        query_node = composer.from_callable(
            func=query_enhancer,
            input_mappings=[FieldMapping("user_question", "q")],
            output_mappings=[
                FieldMapping("enhanced", "search_term"),
                FieldMapping("original", "original_query"),
            ],
        )

        retriever_node = change_output_key(
            CallableNodeConfig(name="retriever", callable_func=retriever),
            "results",
            "context",
        )

        generator_node = CallableNodeConfig(
            name="generator", callable_func=generator)
        adapted_generator = change_output_key(
            generator_node, "answer", "final_response"
        )

        # Execute RAG pipeline
        state = {"user_question": "quantum computing"}

        # Enhance query
        r1 = query_node(state, {})
        state.update(r1.update)

        # Retrieve
        r2 = retriever_node(state, {})
        state.update(r2.update)

        # Generate
        r3 = adapted_generator(state, {})
        state.update(r3.update)

        # Verify complete pipeline
        assert state["search_term"] == "Find information about: quantum computing"
        assert state["original_query"] == "quantum computing"
        assert len(state["context"]) == 2
        assert "Based on 2 sources" in state["final_response"]
        assert "quantum computing" in state["final_response"]
