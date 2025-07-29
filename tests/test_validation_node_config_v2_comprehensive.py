"""Comprehensive test for ValidationNodeConfigV2 to demonstrate proper tool validation."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.types import Command
from pydantic import BaseModel, Field

from haive.core.graph.node.validation_node_config_v2 import ValidationNodeConfigV2


# Test Pydantic models
class UserInfo(BaseModel):
    """User information model."""
    name: str = Field(..., description="User's name")
    age: int = Field(..., ge=0, le=150, description="User's age")
    email: str = Field(..., pattern=r'^[\w\.-]+@[\w\.-]+\.\w+$', description="Email address")


class AnalysisResult(BaseModel):
    """Analysis result model."""
    summary: str = Field(..., min_length=10, description="Summary of analysis")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    tags: list[str] = Field(default_factory=list, description="Tags")


# Test tools
@tool
def calculate(expression: str) -> float:
    """Calculate a mathematical expression.
    
    Args:
        expression: Mathematical expression to evaluate
        
    Returns:
        The result of the calculation
    """
    return eval(expression, {"__builtins__": {}}, {})


@tool  
def search(query: str, max_results: int = 10) -> str:
    """Search for information.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
        
    Returns:
        Search results as a string
    """
    return f"Found {max_results} results for: {query}"


class MockEngine:
    """Mock engine for testing."""
    
    def __init__(self, name="test_engine"):
        self.name = name
        self.tools = [calculate, search]
        self.schemas = [UserInfo, AnalysisResult]
        self.pydantic_tools = []
        self.structured_output_model = None
        
    def get_tool_routes(self):
        """Get tool routes."""
        routes = {}
        # Add tool routes
        for tool in self.tools:
            routes[tool.name] = "langchain_tool"
        # Add schema routes  
        for schema in self.schemas:
            routes[schema.__name__] = "pydantic_model"
        return routes


class TestValidationNodeConfigV2Comprehensive:
    """Comprehensive tests for ValidationNodeConfigV2."""
    
    def test_basic_validation_node_creation(self):
        """Test basic node creation."""
        node = ValidationNodeConfigV2(
            name="test_validation",
            engine_name="test_engine",
            tool_node="tool_node",
            parser_node="parse_output"
        )
        
        assert node.engine_name == "test_engine"
        assert node.tool_node == "tool_node" 
        assert node.parser_node == "parse_output"
        
    def test_valid_pydantic_model_validation(self):
        """Test validation of valid Pydantic model data."""
        engine = MockEngine()
        node = ValidationNodeConfigV2(name="test_validation", engine_name="test_engine")
        
        # Create state with valid tool call
        state = {
            "messages": [
                HumanMessage(content="Create user"),
                AIMessage(
                    content="Creating user",
                    tool_calls=[{
                        "name": "UserInfo",
                        "args": {
                            "name": "John Doe",
                            "age": 30,
                            "email": "john@example.com"
                        },
                        "id": "call_123"
                    }]
                )
            ],
            "engines": {"test_engine": engine},
            "tool_routes": engine.get_tool_routes()
        }
        
        # Execute validation
        result = node(state)
        
        # Should route to parse_output for valid Pydantic model
        assert isinstance(result, Command)
        assert result.goto == "parse_output"
        assert "messages" in result.update
        
        # Check that ToolMessage was added
        tool_messages = [m for m in result.update["messages"] if isinstance(m, ToolMessage)]
        assert len(tool_messages) == 1
        assert tool_messages[0].tool_call_id == "call_123"
        assert not tool_messages[0].additional_kwargs.get("is_error", False)
        
    def test_invalid_pydantic_model_validation(self):
        """Test validation of invalid Pydantic model data."""
        engine = MockEngine()
        node = ValidationNodeConfigV2(name="test_validation", engine_name="test_engine")
        
        # Create state with invalid tool call (bad email format)
        state = {
            "messages": [
                HumanMessage(content="Create user"),
                AIMessage(
                    content="Creating user",
                    tool_calls=[{
                        "name": "UserInfo",
                        "args": {
                            "name": "John Doe",
                            "age": 200,  # Invalid age
                            "email": "not-an-email"  # Invalid email
                        },
                        "id": "call_456"
                    }]
                )
            ],
            "engines": {"test_engine": engine},
            "tool_routes": engine.get_tool_routes()
        }
        
        # Execute validation
        result = node(state)
        
        # Should route back to agent_node for validation errors
        assert isinstance(result, Command)
        assert result.goto == "agent_node"
        assert "messages" in result.update
        
        # Check that error ToolMessage was added
        tool_messages = [m for m in result.update["messages"] if isinstance(m, ToolMessage)]
        assert len(tool_messages) == 1
        assert tool_messages[0].tool_call_id == "call_456"
        assert tool_messages[0].additional_kwargs.get("is_error", False) is True
        assert "validation error" in tool_messages[0].content.lower()
        
    def test_tool_validation_with_args_schema(self):
        """Test validation of tools with args_schema."""
        engine = MockEngine()
        node = ValidationNodeConfigV2(name="test_validation", engine_name="test_engine")
        
        # Create state with tool call
        state = {
            "messages": [
                HumanMessage(content="Calculate something"),
                AIMessage(
                    content="Calculating",
                    tool_calls=[{
                        "name": "calculate",
                        "args": {"expression": "2 + 2"},
                        "id": "call_789"
                    }]
                )
            ],
            "engines": {"test_engine": engine},
            "tool_routes": engine.get_tool_routes()
        }
        
        # Execute validation
        result = node(state)
        
        # Should route to tool_node for valid tool call
        assert isinstance(result, Command)
        assert result.goto == "tool_node"
        
        # Check that AIMessage with validated tool calls is injected
        assert "messages" in result.update
        messages = result.update["messages"]
        
        # Should have original ToolMessage from validation + updated AIMessage
        ai_messages = [m for m in messages if isinstance(m, AIMessage)]
        assert len(ai_messages) >= 1
        
        # Find the AIMessage with tool calls
        ai_with_tools = None
        for msg in ai_messages:
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                ai_with_tools = msg
                break
                
        assert ai_with_tools is not None
        assert len(ai_with_tools.tool_calls) == 1
        assert ai_with_tools.tool_calls[0]["name"] == "calculate"
        assert ai_with_tools.tool_calls[0]["id"] == "call_789"
        
    def test_multiple_tool_calls_mixed_validation(self):
        """Test validation with multiple tool calls of different types."""
        engine = MockEngine()
        node = ValidationNodeConfigV2(name="test_validation", engine_name="test_engine")
        
        # Create state with multiple tool calls
        state = {
            "messages": [
                HumanMessage(content="Multiple operations"),
                AIMessage(
                    content="Processing",
                    tool_calls=[
                        {
                            "name": "UserInfo",
                            "args": {
                                "name": "Alice",
                                "age": 25,
                                "email": "alice@example.com"
                            },
                            "id": "call_001"
                        },
                        {
                            "name": "calculate",
                            "args": {"expression": "10 * 5"},
                            "id": "call_002"
                        },
                        {
                            "name": "AnalysisResult",
                            "args": {
                                "summary": "This is a comprehensive analysis",
                                "confidence": 0.95,
                                "tags": ["important", "verified"]
                            },
                            "id": "call_003"
                        }
                    ]
                )
            ],
            "engines": {"test_engine": engine},
            "tool_routes": engine.get_tool_routes()
        }
        
        # Execute validation
        result = node(state)
        
        # With mixed types, should prioritize based on implementation
        assert isinstance(result, Command)
        assert result.goto in ["tool_node", "parse_output"]  # Depends on prioritization logic
        
    def test_unknown_tool_routing(self):
        """Test handling of unknown tools."""
        engine = MockEngine()
        node = ValidationNodeConfigV2(name="test_validation", engine_name="test_engine")
        
        # Create state with unknown tool
        state = {
            "messages": [
                HumanMessage(content="Use unknown tool"),
                AIMessage(
                    content="Using unknown",
                    tool_calls=[{
                        "name": "unknown_tool",
                        "args": {"data": "test"},
                        "id": "call_unknown"
                    }]
                )
            ],
            "engines": {"test_engine": engine},
            "tool_routes": engine.get_tool_routes()
        }
        
        # Execute validation
        result = node(state)
        
        # Should route to agent_node for unknown tools
        assert isinstance(result, Command)
        assert result.goto == "agent_node"
        
    def test_engine_not_found_handling(self):
        """Test handling when engine is not found."""
        node = ValidationNodeConfigV2(name="test_validation", engine_name="missing_engine")
        
        # Create state without engine
        state = {
            "messages": [
                HumanMessage(content="Test"),
                AIMessage(
                    content="Testing",
                    tool_calls=[{
                        "name": "UserInfo",
                        "args": {"name": "Test", "age": 30, "email": "test@example.com"},
                        "id": "call_test"
                    }]
                )
            ],
            "engines": {},
            "tool_routes": {}
        }
        
        # Execute validation
        result = node(state)
        
        # Should handle gracefully and route to agent_node
        assert isinstance(result, Command)
        assert result.goto == "agent_node"
        
    def test_state_like_dict_access(self):
        """Test that node works with dict-like state access."""
        engine = MockEngine()
        node = ValidationNodeConfigV2(name="test_validation", engine_name="test_engine")
        
        # Use dict state
        state = {
            "messages": [
                HumanMessage(content="Test"),
                AIMessage(
                    content="Testing",
                    tool_calls=[{
                        "name": "search",
                        "args": {"query": "test query"},
                        "id": "call_search"
                    }]
                )
            ],
            "engines": {"test_engine": engine},
            "tool_routes": engine.get_tool_routes()
        }
        
        result = node(state)
        assert isinstance(result, Command)
        
    def test_state_like_object_access(self):
        """Test that node works with object-like state access."""
        engine = MockEngine()
        node = ValidationNodeConfigV2(name="test_validation", engine_name="test_engine")
        
        # Create object-like state
        class StateObject:
            def __init__(self):
                self.messages = [
                    HumanMessage(content="Test"),
                    AIMessage(
                        content="Testing",
                        tool_calls=[{
                            "name": "calculate",
                            "args": {"expression": "5 + 3"},
                            "id": "call_calc"
                        }]
                    )
                ]
                self.engines = {"test_engine": engine}
                self.tool_routes = engine.get_tool_routes()
                
        state = StateObject()
        result = node(state)
        assert isinstance(result, Command)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])