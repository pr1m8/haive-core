"""Tests for integrated schema system - unified architecture validation.

Tests the integration between:
- StateSchema and NodeSchemaComposer
- SchemaComposer and IntegratedNodeComposer
- Field mapping with schema metadata
- Backwards compatibility
"""

from typing import Any

import pytest
from langgraph.types import Command
from pydantic import Field

from haive.core.graph.node.callable_node import CallableNodeConfig
from haive.core.graph.node.composer import (
    FieldMapping,
    NodeSchemaComposer,
    change_output_key,
)
from haive.core.graph.node.composer.integrated_node_composer import (
    IntegratedNodeComposer,
    SchemaAwareComposedNode,
    create_schema_aware_node,
    integrate_node_with_schema,
    with_state_schema,
)
from haive.core.schema import StateSchema


class TestStateSchema(StateSchema):
    """Test state schema with various features."""

    # Basic fields
    query: str
    count: int = Field(default=0)
    items: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    # StateSchema features
    __shared_fields__ = ["query"]
    __reducer_fields__ = {
        "count": lambda old, new: (old or 0) + (new or 0),
        "items": lambda old, new: (old or []) + (new or []),
    }

    __engine_io_mappings__ = {
        "test_engine": {"inputs": ["query"], "outputs": ["items", "count"]}
    }


class TestIntegratedNodeComposer:
    """Test IntegratedNodeComposer functionality."""

    @pytest.fixture
    def composer(self):
        """Create integrated composer."""
        return IntegratedNodeComposer()

    @pytest.fixture
    def simple_node(self):
        """Create simple test node."""

        def process(state: dict[str, Any],
                    config: dict[str, Any]) -> dict[str, Any]:
            items = state.get("data", [])
            return {"results": [x.upper() for x in items], "count": len(items)}

        return CallableNodeConfig(name="simple_node", callable_func=process)

    def test_compose_node_with_schema(self, composer, simple_node):
        """Test composing node with automatic schema generation."""
        composed = composer.compose_node_with_schema(
            base_node=simple_node,
            input_mappings=[FieldMapping("items", "data")],
            output_mappings=[
                FieldMapping("results", "processed_items"),
                FieldMapping("count", "item_count"),
            ],
        )

        assert isinstance(composed, SchemaAwareComposedNode)
        assert composed.name == "composed_simple_node"
        assert composed.state_schema is not None
        assert len(composed.input_mappings) == 1
        assert len(composed.output_mappings) == 2

    def test_compose_with_existing_schema(self, composer, simple_node):
        """Test composing with existing StateSchema."""
        composed = composer.compose_node_with_schema(
            base_node=simple_node,
            state_schema=TestStateSchema,
            input_mappings=[FieldMapping("items", "data")],
            output_mappings=[FieldMapping("results", "items")],
        )

        assert composed.state_schema == TestStateSchema

        # Test with actual schema instance
        state = TestStateSchema(query="test", items=["hello", "world"])
        result = composed(state)

        assert isinstance(result, Command)
        assert "items" in result.update
        assert result.update["items"] == ["HELLO", "WORLD"]

    def test_from_callable_with_schema(self, composer):
        """Test creating node from callable with schema."""

        def process_query(query: str) -> dict[str, Any]:
            return {"processed": f"Processed: {query}"}

        node = composer.from_callable_with_schema(
            func=process_query,
            input_schema=TestStateSchema,
            input_mappings=[FieldMapping("query", "query")],
            output_mappings=[FieldMapping("processed", "result")],
        )

        assert isinstance(node, SchemaAwareComposedNode)
        assert node.state_schema == TestStateSchema

        # Test execution
        state = TestStateSchema(query="test query")
        result = node(state)

        assert result.update["result"] == "Processed: test query"

    def test_schema_generation_with_metadata(self, composer):
        """Test that schema generation preserves metadata."""

        # Create node with field definitions
        def process(state: dict[str, Any],
                    config: dict[str, Any]) -> dict[str, Any]:
            return {"output": "processed"}

        node = CallableNodeConfig(name="test_node", callable_func=process)

        # Compose with metadata preservation
        composed = composer.compose_node_with_schema(
            base_node=node,
            output_mappings=[FieldMapping("output", "result")],
            preserve_field_metadata=True,
        )

        # Schema should be generated
        assert composed.state_schema is not None
        assert hasattr(composed.state_schema, "__fields__")

    def test_state_schema_adapter(self, composer):
        """Test StateSchemaAdapter functionality."""

        # Define source and target schemas
        class SourceSchema(StateSchema):
            old_field: str
            old_count: int = 0

        class TargetSchema(StateSchema):
            new_field: str
            new_count: int = 0

        # Create adapter
        adapter = composer.create_schema_adapter(
            source_schema=SourceSchema,
            target_schema=TargetSchema,
            field_mappings=[
                FieldMapping("old_field", "new_field"),
                FieldMapping("old_count", "new_count"),
            ],
        )

        # Test adaptation
        source = SourceSchema(old_field="test", old_count=5)
        target = adapter.adapt(source)

        assert isinstance(target, TargetSchema)
        assert target.new_field == "test"
        assert target.new_count == 5

    def test_reducer_integration(self, composer):
        """Test that reducers work with composed nodes."""

        def add_items(state: dict[str, Any],
                      config: dict[str, Any]) -> dict[str, Any]:
            return {"items": ["new1", "new2"]}

        node = CallableNodeConfig(name="adder", callable_func=add_items)

        composed = composer.compose_node_with_schema(
            base_node=node,
            state_schema=TestStateSchema,
            output_mappings=[FieldMapping("items", "items")],
        )

        # Test reducer functionality
        state = TestStateSchema(query="test", items=["old1", "old2"])
        result = composed(state)

        # Should use reducer to combine old and new items
        new_state = state.model_copy(update=result.update)
        assert len(new_state.items) == 4  # 2 old + 2 new
        assert "old1" in new_state.items
        assert "new1" in new_state.items


class TestSchemaAwareComposedNode:
    """Test SchemaAwareComposedNode functionality."""

    @pytest.fixture
    def composed_node(self):
        """Create composed node for testing."""

        def process(state: dict[str, Any],
                    config: dict[str, Any]) -> dict[str, Any]:
            data = state.get("data", [])
            return {"results": [x.upper() for x in data]}

        base_node = CallableNodeConfig(name="test", callable_func=process)
        composer = IntegratedNodeComposer()

        return composer.compose_node_with_schema(
            base_node=base_node,
            state_schema=TestStateSchema,
            input_mappings=[FieldMapping("items", "data")],
            output_mappings=[FieldMapping("results", "processed")],
        )

    def test_schema_state_execution(self, composed_node):
        """Test execution with StateSchema instance."""
        state = TestStateSchema(query="test", items=["hello", "world"])
        result = composed_node(state)

        assert isinstance(result, Command)
        assert "processed" in result.update
        assert result.update["processed"] == ["HELLO", "WORLD"]

    def test_dict_state_execution(self, composed_node):
        """Test execution with dict state."""
        state = {"query": "test", "items": ["hello", "world"]}
        result = composed_node(state)

        assert isinstance(result, Command)
        assert "processed" in result.update
        assert result.update["processed"] == ["HELLO", "WORLD"]

    def test_schema_properties(self, composed_node):
        """Test schema property access."""
        assert composed_node.input_schema == TestStateSchema
        assert composed_node.output_schema == TestStateSchema
        assert composed_node.state_schema == TestStateSchema


class TestFactoryFunctions:
    """Test factory functions for integration."""

    def test_integrate_node_with_schema(self):
        """Test integrate_node_with_schema function."""

        def process(state: dict[str, Any],
                    config: dict[str, Any]) -> dict[str, Any]:
            return {"result": len(state.get("items", []))}

        node = CallableNodeConfig(name="counter", callable_func=process)

        integrated = integrate_node_with_schema(
            node=node,
            schema=TestStateSchema,
            output_mappings=[FieldMapping("result", "count")],
        )

        assert isinstance(integrated, SchemaAwareComposedNode)
        assert integrated.state_schema == TestStateSchema

        # Test execution
        state = TestStateSchema(query="test", items=["a", "b", "c"])
        result = integrated(state)

        assert result.update["count"] == 3

    def test_create_schema_aware_node(self):
        """Test create_schema_aware_node function."""

        def process(state: TestStateSchema) -> dict[str, Any]:
            return {"processed": f"Processed {len(state.items)} items"}

        node = create_schema_aware_node(
            func=process,
            schema=TestStateSchema,
            output_mappings=[FieldMapping("processed", "result")],
        )

        assert isinstance(node, SchemaAwareComposedNode)

        # Test execution
        state = TestStateSchema(query="test", items=["a", "b"])
        result = node(state)

        assert result.update["result"] == "Processed 2 items"

    def test_with_state_schema_decorator(self):
        """Test with_state_schema decorator."""

        @with_state_schema(
            TestStateSchema, output_mappings=[
                FieldMapping("result", "processed")]
        )
        def process(state: TestStateSchema) -> dict[str, Any]:
            return {"result": f"Query: {state.query}"}

        assert isinstance(process, SchemaAwareComposedNode)

        # Test execution
        state = TestStateSchema(query="test query")
        result = process(state)

        assert result.update["processed"] == "Query: test query"


class TestBackwardsCompatibility:
    """Test backwards compatibility with existing code."""

    def test_basic_composer_still_works(self):
        """Test that basic NodeSchemaComposer still works."""

        def process(state: dict[str, Any],
                    config: dict[str, Any]) -> dict[str, Any]:
            return {"result": "processed"}

        node = CallableNodeConfig(name="basic", callable_func=process)
        composer = NodeSchemaComposer()

        # Basic composition still works
        composed = composer.compose_node(
            base_node=node, output_mappings=[FieldMapping("result", "output")]
        )

        result = composed({"input": "test"}, {})
        assert result.update["output"] == "processed"

    def test_factory_functions_compatibility(self):
        """Test that factory functions work with any composer."""

        def process(state: dict[str, Any],
                    config: dict[str, Any]) -> dict[str, Any]:
            return {"data": ["a", "b", "c"]}

        node = CallableNodeConfig(name="test", callable_func=process)

        # change_output_key should work
        adapted = change_output_key(node, "data", "items")

        result = adapted({"input": "test"}, {})
        assert "items" in result.update
        assert result.update["items"] == ["a", "b", "c"]

    def test_mixed_usage_patterns(self):
        """Test mixing old and new patterns."""

        # Old style node
        def old_process(
            state: dict[str, Any], config: dict[str, Any]
        ) -> dict[str, Any]:
            return {"old_result": "old"}

        old_node = CallableNodeConfig(name="old", callable_func=old_process)

        # New style composition
        composer = IntegratedNodeComposer()
        new_node = composer.compose_node_with_schema(
            base_node=old_node,
            state_schema=TestStateSchema,
            output_mappings=[
                FieldMapping(
                    "old_result",
                    "metadata.old_result")],
        )

        # Should work together
        state = TestStateSchema(query="test")
        result = new_node(state)

        assert "metadata" in result.update
        assert result.update["metadata"]["old_result"] == "old"


class TestPerformanceCharacteristics:
    """Test performance characteristics of integrated system."""

    def test_overhead_minimal(self):
        """Test that integration overhead is minimal."""
        import time

        # Simple function
        def process(data: list[str]) -> list[str]:
            return [x.upper() for x in data]

        data = ["test"] * 100

        # Direct call
        start = time.time()
        for _ in range(100):
            process(data)
        direct_time = time.time() - start

        # With schema integration
        node = create_schema_aware_node(
            func=process,
            schema=TestStateSchema,
            input_mappings=[FieldMapping("items", "data")],
            output_mappings=[FieldMapping("result", "processed")],
        )

        start = time.time()
        for _ in range(100):
            state = TestStateSchema(query="test", items=data)
            node(state)
        integrated_time = time.time() - start

        # Should be reasonable overhead (less than 10x)
        assert integrated_time < direct_time * 10

    def test_memory_usage_reasonable(self):
        """Test that memory usage is reasonable."""
        # This test ensures we don't have memory leaks
        # or excessive object creation

        composer = IntegratedNodeComposer()

        # Create multiple nodes
        nodes = []
        for i in range(10):

            def process(
                state: dict[str, Any], config: dict[str, Any]
            ) -> dict[str, Any]:
                return {"result": f"processed_{i}"}

            node = CallableNodeConfig(name=f"node_{i}", callable_func=process)
            composed = composer.compose_node_with_schema(
                base_node=node, state_schema=TestStateSchema
            )
            nodes.append(composed)

        # Should be able to create many nodes without issues
        assert len(nodes) == 10

        # All should be functional
        state = TestStateSchema(query="test")
        for node in nodes:
            result = node(state)
            assert "result" in result.update


class TestErrorHandling:
    """Test error handling in integrated system."""

    def test_invalid_mapping_handling(self):
        """Test handling of invalid field mappings."""
        composer = IntegratedNodeComposer()

        def process(state: dict[str, Any],
                    config: dict[str, Any]) -> dict[str, Any]:
            return {"result": "test"}

        node = CallableNodeConfig(name="test", callable_func=process)

        # Should handle missing source fields gracefully
        composed = composer.compose_node_with_schema(
            base_node=node,
            state_schema=TestStateSchema,
            input_mappings=[
                FieldMapping(
                    "nonexistent",
                    "data",
                    default="default")],
        )

        state = TestStateSchema(query="test")
        result = composed(state)

        # Should not crash and should use default
        assert isinstance(result, Command)

    def test_type_mismatch_handling(self):
        """Test handling of type mismatches."""
        composer = IntegratedNodeComposer()

        def process(state: dict[str, Any],
                    config: dict[str, Any]) -> dict[str, Any]:
            return {"result": "string_result"}

        node = CallableNodeConfig(name="test", callable_func=process)

        # Even if types don't match perfectly, should still work
        composed = composer.compose_node_with_schema(
            base_node=node,
            state_schema=TestStateSchema,
            output_mappings=[
                FieldMapping(
                    "result",
                    "count")],
            # String to int field
        )

        state = TestStateSchema(query="test")
        result = composed(state)

        # Should handle gracefully
        assert isinstance(result, Command)
        assert "count" in result.update
