"""Tests for AdvancedNodeComposer - extended node logic and callable patterns.

Tests cover:
1. Automatic signature detection
2. Different callable signatures
3. Extended extract/update logic
4. Type-safe nodes
5. Decorator patterns
6. Command/Send handling
"""

from typing import Any

import pytest
from langgraph.types import Command, Send
from pydantic import BaseModel

from haive.core.graph.node.composer import FieldMapping
from haive.core.graph.node.composer.advanced_node_composer import (
    AdvancedNodeComposer,
    as_node,
    callable_to_node,
    node_with_custom_logic,
)


class MessagesState(BaseModel):
    """Test state with messages."""

    messages: list[str] = []
    metadata: dict[str, Any] = {}


class TestAdvancedNodeComposer:
    """Test AdvancedNodeComposer functionality."""

    @pytest.fixture
    def composer(self):
        """Create advanced composer instance."""
        return AdvancedNodeComposer()

    def test_signature_detection_state_only(self, composer):
        """Test detecting function with state parameter only."""

        def process(state):
            return {"count": len(state.get("messages", []))}

        sig_info = composer._analyze_callable_signature(process)

        assert sig_info["has_state"] is True
        assert sig_info["has_config"] is False
        assert sig_info["state_param"] == "state"

    def test_signature_detection_state_and_config(self, composer):
        """Test detecting function with state and config."""

        def process(state, config):
            return {"mode": config.get("mode", "default")}

        sig_info = composer._analyze_callable_signature(process)

        assert sig_info["has_state"] is True
        assert sig_info["has_config"] is True
        assert sig_info["state_param"] == "state"
        assert sig_info["config_param"] == "config"

    def test_signature_detection_typed(self, composer):
        """Test detecting typed function signature."""

        def process(state: MessagesState, config: dict[str, Any]) -> Command:
            return Command(update={"typed": True})

        sig_info = composer._analyze_callable_signature(process)

        assert sig_info["has_state"] is True
        assert sig_info["has_config"] is True
        assert sig_info["return_type"] == Command

    def test_signature_detection_no_params(self, composer):
        """Test detecting function with no parameters."""

        def get_value():
            return {"value": 42}

        sig_info = composer._analyze_callable_signature(get_value)

        assert sig_info["has_state"] is False
        assert sig_info["has_config"] is False
        assert len(sig_info["params"]) == 0

    def test_from_callable_advanced_simple(self, composer):
        """Test creating node from simple callable."""

        def process(state):
            return {"processed": True}

        node = composer.from_callable_advanced(process)

        # Test execution
        result = node({"messages": []}, {})

        assert isinstance(result, Command)
        assert result.update["processed"] is True

    def test_from_callable_advanced_with_config(self, composer):
        """Test callable that uses config."""

        def process(state, config):
            threshold = config.get("threshold", 5)
            return {"over_threshold": len(state.get("messages", [])) > threshold}

        node = composer.from_callable_advanced(process)

        # Test with config
        result = node({"messages": ["a", "b", "c"]}, {"threshold": 2})

        assert result.update["over_threshold"] is True

    def test_from_callable_advanced_no_params(self, composer):
        """Test callable with no parameters."""
        counter = {"value": 0}

        def increment():
            counter["value"] += 1
            return {"count": counter["value"]}

        node = composer.from_callable_advanced(increment)

        # Execute multiple times
        r1 = node({}, {})
        r2 = node({}, {})

        assert r1.update["count"] == 1
        assert r2.update["count"] == 2

    def test_command_auto_wrapping(self, composer):
        """Test automatic Command wrapping."""

        def process(state):
            # Return plain dict
            return {"result": "done"}

        node = composer.from_callable_advanced(process, handle_command=True)

        result = node({}, {})

        # Should be wrapped in Command
        assert isinstance(result, Command)
        assert result.update["result"] == "done"

    def test_command_preservation(self, composer):
        """Test that explicit Commands are preserved."""

        def process(state) -> Command:
            return Command(update={"processed": True}, goto="next_step")

        node = composer.from_callable_advanced(process, handle_command=True)

        result = node({}, {})

        # Should preserve Command fields
        assert isinstance(result, Command)
        assert result.update["processed"] is True
        assert result.goto == "next_step"

    def test_send_handling(self, composer):
        """Test Send command handling."""

        def split_work(state) -> list[Send]:
            return [Send("worker1", {"task": "A"}), Send("worker2", {"task": "B"})]

        node = composer.from_callable_advanced(split_work, handle_command=False)

        result = node({}, {})

        # Should return Send list unchanged
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(s, Send) for s in result)

    def test_extended_extract_logic(self, composer):
        """Test custom extraction logic."""

        def custom_extract(state, config):
            # Complex extraction
            messages = state.get("messages", [])
            return {
                "recent": messages[-3:],
                "total": len(messages),
                "has_data": bool(messages),
            }

        def process(extracted):
            return {
                "summary": f"Processing {extracted['total']} messages",
                "status": "has_data" if extracted["has_data"] else "empty",
            }

        node = composer.from_callable_advanced(
            func=process, extract_logic=custom_extract
        )

        result = node({"messages": ["a", "b", "c", "d", "e"]}, {})

        assert result.update["summary"] == "Processing 5 messages"
        assert result.update["status"] == "has_data"

    def test_extended_update_logic(self, composer):
        """Test custom update logic."""

        def process(state):
            return {"count": len(state.get("messages", []))}

        def custom_update(result, state, config):
            # Complex update logic
            count = result.get("count", 0)
            return {
                "message_count": count,
                "status": "many" if count > 5 else "few",
                "timestamp": "2024-01-01",  # Fixed for testing
                "metadata": {**state.get("metadata", {}), "last_count": count},
            }

        node = composer.from_callable_advanced(func=process, update_logic=custom_update)

        result = node({"messages": ["a", "b", "c"]}, {})

        assert result.update["message_count"] == 3
        assert result.update["status"] == "few"
        assert result.update["timestamp"] == "2024-01-01"
        assert result.update["metadata"]["last_count"] == 3

    def test_typed_callable_node(self, composer):
        """Test type-safe callable node."""

        class InputState(BaseModel):
            text: str
            options: dict[str, Any] = {}

        class OutputResult(BaseModel):
            processed: str
            length: int

        def process_typed(state: InputState, config: dict[str, Any]) -> OutputResult:
            return OutputResult(processed=state.text.upper(), length=len(state.text))

        node = composer.create_typed_callable_node(
            func=process_typed,
            state_type=InputState,
            result_type=OutputResult,
            output_mappings=[
                FieldMapping("processed", "result_text"),
                FieldMapping("length", "text_length"),
            ],
            validate_types=True,
        )

        # Test with correct types
        state = InputState(text="hello world")
        result = node.node(state, {})

        assert result.update["result_text"] == "HELLO WORLD"
        assert result.update["text_length"] == 11

    def test_extract_update_node(self, composer):
        """Test creating node with extract/process/update pipeline."""

        def extract_messages(state, config):
            messages = state.get("messages", [])
            window = config.get("window", 5)
            return messages[-window:]

        def count_messages(messages):
            return {"count": len(messages), "empty": len(messages) == 0}

        def update_counts(result, state, config):
            return {
                "recent_count": result["count"],
                "is_empty": result["empty"],
                "checked_at": "2024-01-01",
            }

        node = composer.create_extract_update_node(
            extract_func=extract_messages,
            process_func=count_messages,
            update_func=update_counts,
            name="message_counter",
        )

        result = node({"messages": ["a", "b", "c", "d", "e", "f"]}, {"window": 3})

        assert result.update["recent_count"] == 3
        assert result.update["is_empty"] is False


class TestDecoratorPatterns:
    """Test decorator-based node creation."""

    def test_as_node_simple(self):
        """Test simple @as_node decorator."""

        @as_node()
        def process(state):
            return {"done": True}

        result = process({}, {})

        assert isinstance(result, Command)
        assert result.update["done"] is True

    def test_as_node_with_mappings(self):
        """Test @as_node with field mappings."""

        @as_node(
            input_mappings=[FieldMapping("user_messages", "messages")],
            output_mappings=[FieldMapping("result", "processed")],
        )
        def count_messages(messages):
            return {"result": len(messages)}

        result = count_messages({"user_messages": ["a", "b", "c"]}, {})

        assert result.update["processed"] == 3

    def test_as_node_with_extract_logic(self):
        """Test @as_node with custom extraction."""

        def extract_last_message(state, config):
            messages = state.get("messages", [])
            return messages[-1] if messages else None

        @as_node(extract_logic=extract_last_message)
        def process_message(message):
            if message:
                return {"found": True, "content": message}
            return {"found": False}

        result = process_message({"messages": ["first", "last"]}, {})

        assert result.update["found"] is True
        assert result.update["content"] == "last"

    def test_callable_to_node_decorator(self):
        """Test callable_to_node as decorator."""

        @callable_to_node
        def my_processor(state):
            return {"processed_by": "decorator"}

        result = my_processor({}, {})

        assert result.update["processed_by"] == "decorator"

    def test_node_with_custom_logic_helper(self):
        """Test node_with_custom_logic factory."""

        def extract(state, config):
            return state.get("data", [])

        def process(data):
            return {"count": len(data)}

        def update(result, state, config):
            return {"data_count": result["count"], "has_data": result["count"] > 0}

        node = node_with_custom_logic(
            name="data_processor", extract=extract, process=process, update=update
        )

        result = node({"data": [1, 2, 3]}, {})

        assert result.update["data_count"] == 3
        assert result.update["has_data"] is True


class TestRealWorldScenarios:
    """Test real-world usage patterns."""

    def test_message_processor_pattern(self):
        """Test realistic message processing pattern."""

        @as_node(
            output_mappings=[
                FieldMapping("should_continue", "continue_conversation"),
                FieldMapping("reason", "stop_reason"),
            ]
        )
        def check_conversation_end(state):
            messages = state.get("messages", [])

            if len(messages) > 20:
                return {"should_continue": False, "reason": "max_length_reached"}

            last_message = messages[-1] if messages else ""
            if "goodbye" in last_message.lower():
                return {"should_continue": False, "reason": "user_goodbye"}

            return {"should_continue": True, "reason": None}

        # Test max length
        result1 = check_conversation_end({"messages": ["msg"] * 25}, {})
        assert result1.update["continue_conversation"] is False
        assert result1.update["stop_reason"] == "max_length_reached"

        # Test goodbye
        result2 = check_conversation_end({"messages": ["hello", "goodbye"]}, {})
        assert result2.update["continue_conversation"] is False
        assert result2.update["stop_reason"] == "user_goodbye"

        # Test continue
        result3 = check_conversation_end({"messages": ["hello", "how are you?"]}, {})
        assert result3.update["continue_conversation"] is True
        assert result3.update["stop_reason"] is None

    def test_document_processing_pattern(self):
        """Test document processing with complex logic."""
        composer = AdvancedNodeComposer()

        def extract_documents(state, config):
            """Extract documents meeting criteria."""
            docs = state.get("documents", [])
            min_length = config.get("min_length", 50)

            valid_docs = [
                doc for doc in docs if len(doc.get("content", "")) >= min_length
            ]

            return {
                "docs": valid_docs,
                "total": len(docs),
                "filtered": len(docs) - len(valid_docs),
            }

        def summarize_docs(extracted):
            """Create summaries."""
            summaries = []
            for doc in extracted["docs"]:
                content = doc["content"]
                summary = content[:30] + "..." if len(content) > 30 else content
                summaries.append({"id": doc.get("id", "unknown"), "summary": summary})

            return {
                "summaries": summaries,
                "processed": len(summaries),
                "skipped": extracted["filtered"],
            }

        def update_state(result, state, config):
            """Update with processing results."""
            return {
                "document_summaries": result["summaries"],
                "processing_stats": {
                    "processed": result["processed"],
                    "skipped": result["skipped"],
                    "timestamp": "2024-01-01",
                },
                "status": "completed",
            }

        processor = composer.from_callable_advanced(
            func=summarize_docs,
            extract_logic=extract_documents,
            update_logic=update_state,
            name="document_processor",
        )

        # Test with documents
        state = {
            "documents": [
                {"id": 1, "content": "Short"},
                {
                    "id": 2,
                    "content": "This is a much longer document that should be processed",
                },
                {
                    "id": 3,
                    "content": "Another long document with lots of interesting content",
                },
            ]
        }

        result = processor(state, {"min_length": 20})

        assert len(result.update["document_summaries"]) == 2
        assert result.update["processing_stats"]["processed"] == 2
        assert result.update["processing_stats"]["skipped"] == 1
        assert result.update["status"] == "completed"
