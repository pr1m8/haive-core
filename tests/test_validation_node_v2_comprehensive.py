"""Comprehensive test suite for ValidationNodeV2 with edge cases and bad tool calls.

This test suite validates:
1. Correct routing of pydantic_model vs langchain_tool
2. Proper ToolMessage creation for Pydantic models
3. AIMessage injection for langchain tools
4. Error handling for bad tool calls
5. Unknown tool handling
6. Dynamic engine attribution
7. Multiple tool calls with mixed types
"""

import json
import logging
import pytest
from typing import List, Optional

from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.types import Command
from pydantic import BaseModel, Field, ValidationError

from haive.core.graph.node.validation_node_v2 import ValidationNodeV2

logger = logging.getLogger(__name__)


# Test Pydantic models
class TaskAnalysis(BaseModel):
    """Structured task analysis model."""
    task_type: str = Field(description="Type of task")
    complexity: int = Field(ge=1, le=10, description="Complexity score")
    requirements: List[str] = Field(description="Task requirements")


class InvalidModel(BaseModel):
    """Model that will fail validation."""
    required_field: str = Field(description="This is required")
    must_be_positive: int = Field(gt=0, description="Must be positive")


# Test tools
@tool
def calculator(expression: str) -> str:
    """Calculate mathematical expressions."""
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def search_tool(query: str, max_results: int = 5) -> str:
    """Search for information."""
    return f"Found {max_results} results for '{query}'"


class TestValidationNodeV2Comprehensive:
    """Comprehensive test suite for ValidationNodeV2."""

    @pytest.fixture
    def mock_engine(self):
        """Create a mock engine with tools and structured output."""
        # Create a simple mock object without importing AugLLMConfig
        engine = type('Engine', (), {
            'name': 'test_engine',
            'tool_routes': {
                "TaskAnalysis": "pydantic_model",
                "InvalidModel": "pydantic_model",
                "calculator": "langchain_tool",
                "search_tool": "langchain_tool",
                "unknown_tool": "unknown"
            },
            'structured_output_model': TaskAnalysis,
            'schemas': [TaskAnalysis, InvalidModel]
        })()
        
        return engine

    @pytest.fixture
    def validation_node(self):
        """Create ValidationNodeV2 instance."""
        return ValidationNodeV2(
            name="test_validation",
            router_node="validation_router"
        )

    @pytest.fixture
    def base_state(self, mock_engine):
        """Create base state with engine."""
        # Create a simple state object without importing MessagesState
        state = type('State', (), {
            'messages': [
                HumanMessage(content="Test the validation")
            ],
            'engines': {"test_engine": mock_engine},
            'tool_routes': mock_engine.tool_routes,
            'engine_name': 'test_engine'
        })()
        return state

    def test_pydantic_model_validation_success(self, validation_node, base_state):
        """Test successful Pydantic model validation creates proper ToolMessage."""
        # Create AIMessage with tool call for TaskAnalysis
        ai_message = AIMessage(
            content="I'll analyze this task",
            tool_calls=[{
                "id": "call_123",
                "name": "TaskAnalysis",
                "args": {
                    "task_type": "coding",
                    "complexity": 7,
                    "requirements": ["Python", "Testing", "Documentation"]
                }
            }],
            additional_kwargs={"engine_name": "test_engine"}
        )
        base_state.messages = base_state.messages + [ai_message]

        # Execute validation node
        result = validation_node(base_state)

        # Verify Command structure
        assert isinstance(result, Command)
        assert result.goto == "validation_router"
        assert "messages" in result.update

        # Check the ToolMessage was created
        updated_messages = result.update["messages"]
        tool_message = updated_messages[-1]
        
        assert isinstance(tool_message, ToolMessage)
        assert tool_message.tool_call_id == "call_123"
        assert tool_message.name == "TaskAnalysis"
        
        # Parse content
        content = json.loads(tool_message.content)
        assert content["success"] is True
        assert content["model"] == "TaskAnalysis"
        assert content["validated"] is True
        assert content["data"]["task_type"] == "coding"
        assert content["data"]["complexity"] == 7
        
        # Check additional kwargs
        assert tool_message.additional_kwargs["is_error"] is False
        assert tool_message.additional_kwargs["validation_passed"] is True
        assert tool_message.additional_kwargs["model_type"] == "pydantic"

    def test_pydantic_model_validation_failure(self, validation_node, base_state):
        """Test Pydantic model validation failure creates error ToolMessage."""
        # Create AIMessage with invalid data
        ai_message = AIMessage(
            content="Testing invalid data",
            tool_calls=[{
                "id": "call_456",
                "name": "TaskAnalysis",
                "args": {
                    "task_type": "coding",
                    "complexity": 15,  # Invalid: > 10
                    "requirements": []  # This is ok, empty list is valid
                }
            }]
        )
        base_state.messages = base_state.messages + [ai_message]

        # Execute validation node
        result = validation_node(base_state)

        # Check error ToolMessage
        tool_message = result.update["messages"][-1]
        assert isinstance(tool_message, ToolMessage)
        
        content = json.loads(tool_message.content)
        assert content["success"] is False
        assert content["error"] == "ValidationError"
        assert "errors" in content
        
        # Check additional kwargs
        assert tool_message.additional_kwargs["is_error"] is True
        assert tool_message.additional_kwargs["validation_passed"] is False

    def test_langchain_tool_routing_no_toolmessage(self, validation_node, base_state):
        """Test that langchain tools don't get ToolMessages, just route to tool_node."""
        # Create AIMessage with calculator tool call
        ai_message = AIMessage(
            content="Let me calculate that",
            tool_calls=[{
                "id": "call_789",
                "name": "calculator",
                "args": {"expression": "15 * 23"}
            }]
        )
        base_state.messages = base_state.messages + [ai_message]

        # Execute validation node
        result = validation_node(base_state)

        # Should NOT create ToolMessage for langchain_tool
        assert result.goto == "validation_router"
        if "messages" in result.update:
            # No new messages should be added for langchain tools
            assert len(result.update["messages"]) == len(base_state.messages)

    def test_unknown_tool_error_handling(self, validation_node, base_state):
        """Test unknown tool creates error ToolMessage."""
        # Create AIMessage with unknown tool
        ai_message = AIMessage(
            content="Using unknown tool",
            tool_calls=[{
                "id": "call_unknown",
                "name": "nonexistent_tool",
                "args": {"data": "test"}
            }]
        )
        base_state.messages = base_state.messages + [ai_message]

        # Execute validation node
        result = validation_node(base_state)

        # Should create error ToolMessage
        tool_message = result.update["messages"][-1]
        assert isinstance(tool_message, ToolMessage)
        
        content = json.loads(tool_message.content)
        assert content["success"] is False
        assert "Unknown tool" in content["error"] or "unknown" in content["error"].lower()

    def test_multiple_mixed_tool_calls(self, validation_node, base_state):
        """Test handling multiple tool calls of different types in one message."""
        # Create AIMessage with mixed tool calls
        ai_message = AIMessage(
            content="Multiple operations",
            tool_calls=[
                {
                    "id": "call_1",
                    "name": "TaskAnalysis",
                    "args": {
                        "task_type": "mixed",
                        "complexity": 5,
                        "requirements": ["Multi-tool"]
                    }
                },
                {
                    "id": "call_2",
                    "name": "calculator",
                    "args": {"expression": "10 + 20"}
                },
                {
                    "id": "call_3",
                    "name": "unknown_tool",
                    "args": {"param": "value"}
                }
            ]
        )
        base_state.messages = base_state.messages + [ai_message]

        # Execute validation node
        result = validation_node(base_state)

        # Count new ToolMessages
        new_messages = result.update.get("messages", [])
        new_tool_messages = [m for m in new_messages[len(base_state.messages):] 
                            if isinstance(m, ToolMessage)]

        # Should have 2 ToolMessages: TaskAnalysis and unknown_tool
        # calculator should NOT get a ToolMessage
        assert len(new_tool_messages) == 2

        # Check first is TaskAnalysis success
        task_msg = next(m for m in new_tool_messages if m.tool_call_id == "call_1")
        content = json.loads(task_msg.content)
        assert content["success"] is True
        assert content["model"] == "TaskAnalysis"

        # Check second is unknown tool error
        unknown_msg = next(m for m in new_tool_messages if m.tool_call_id == "call_3")
        content = json.loads(unknown_msg.content)
        assert content["success"] is False

    def test_dynamic_engine_attribution(self, validation_node, base_state):
        """Test that engine_name is extracted from AIMessage attribution."""
        # Create another engine
        other_engine = AugLLMConfig(
            model="gpt-3.5-turbo",
            structured_output_model=InvalidModel
        )
        other_engine.tool_routes = {"InvalidModel": "pydantic_model"}
        other_engine.schemas = [InvalidModel]
        
        # Add to state
        base_state.engines["other_engine"] = other_engine

        # Create AIMessage with different engine attribution
        ai_message = AIMessage(
            content="Using different engine",
            tool_calls=[{
                "id": "call_other",
                "name": "InvalidModel",
                "args": {
                    "required_field": "test",
                    "must_be_positive": 5
                }
            }],
            additional_kwargs={"engine_name": "other_engine"}  # Different engine!
        )
        base_state.messages = base_state.messages + [ai_message]

        # Execute validation node
        result = validation_node(base_state)

        # Should use the other_engine for validation
        tool_message = result.update["messages"][-1]
        content = json.loads(tool_message.content)
        assert content["success"] is True
        assert content["model"] == "InvalidModel"

    def test_missing_tool_id_handling(self, validation_node, base_state):
        """Test handling of tool calls with missing IDs."""
        # Create AIMessage with missing tool ID
        ai_message = AIMessage(
            content="Bad tool call",
            tool_calls=[{
                # Missing 'id' field
                "name": "TaskAnalysis",
                "args": {"task_type": "test", "complexity": 1, "requirements": []}
            }]
        )
        base_state.messages = base_state.messages + [ai_message]

        # Execute validation node
        result = validation_node(base_state)

        # Should handle gracefully - no new messages
        assert result.goto == "validation_router"
        new_messages = result.update.get("messages", [])
        assert len(new_messages) == len(base_state.messages)

    def test_empty_tool_calls(self, validation_node, base_state):
        """Test AIMessage with empty tool_calls list."""
        ai_message = AIMessage(
            content="No tools needed",
            tool_calls=[]
        )
        base_state.messages = base_state.messages + [ai_message]

        # Execute validation node
        result = validation_node(base_state)

        # Should route to router without changes
        assert result.goto == "validation_router"
        assert len(result.update.get("messages", [])) == len(base_state.messages)

    def test_structured_output_model_as_tool(self, validation_node, base_state):
        """Test that structured_output_model from engine is recognized as a tool."""
        # The engine already has TaskAnalysis as structured_output_model
        # Verify it's in tool_routes as pydantic_model
        assert base_state.tool_routes["TaskAnalysis"] == "pydantic_model"

        # Call it as a tool
        ai_message = AIMessage(
            content="Using structured output",
            tool_calls=[{
                "id": "struct_out_123",
                "name": "TaskAnalysis",  # Same name as structured_output_model
                "args": {
                    "task_type": "structured",
                    "complexity": 8,
                    "requirements": ["output", "validation"]
                }
            }]
        )
        base_state.messages = base_state.messages + [ai_message]

        # Execute validation node
        result = validation_node(base_state)

        # Should create ToolMessage with validated structured output
        tool_message = result.update["messages"][-1]
        content = json.loads(tool_message.content)
        
        assert content["success"] is True
        assert content["model"] == "TaskAnalysis"
        assert content["data"]["task_type"] == "structured"
        assert tool_message.additional_kwargs["model_type"] == "pydantic"

    def test_duplicate_tool_message_prevention(self, validation_node, base_state):
        """Test that duplicate ToolMessages aren't created for same tool_call_id."""
        # Create AIMessage with tool call
        ai_message = AIMessage(
            content="First call",
            tool_calls=[{
                "id": "duplicate_123",
                "name": "TaskAnalysis",
                "args": {"task_type": "test", "complexity": 5, "requirements": []}
            }]
        )
        base_state.messages = base_state.messages + [ai_message]

        # Add existing ToolMessage for same tool_call_id
        existing_tool_msg = ToolMessage(
            content='{"success": true}',
            tool_call_id="duplicate_123",
            name="TaskAnalysis"
        )
        base_state.messages.append(existing_tool_msg)

        # Execute validation node
        result = validation_node(base_state)

        # Should NOT create another ToolMessage
        new_messages = result.update.get("messages", [])
        assert len(new_messages) == len(base_state.messages)  # No new messages


if __name__ == "__main__":
    pytest.main([__file__, "-v"])