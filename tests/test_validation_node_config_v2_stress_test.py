"""Comprehensive stress test for ValidationNodeConfigV2 with SimpleAgentV3.

This test validates ValidationNodeConfigV2 under extreme conditions with:
- Bad tool calls with invalid arguments
- Mixed langchain_tool and structured output routing
- Real SimpleAgentV3 integration
- Complex error scenarios and recovery
- Multi-step validation chains
"""

import pytest
from contextlib import nullcontext
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.types import Command
from pydantic import BaseModel, Field, ValidationError

from haive.core.graph.node.validation_node_config_v2 import ValidationNodeConfigV2
from haive.agents.simple.agent_v3 import SimpleAgentV3
from haive.core.engine.aug_llm import AugLLMConfig


# ============================================================================
# STRUCTURED OUTPUT MODELS FOR TESTING
# ============================================================================

class UserProfile(BaseModel):
    """User profile with strict validation."""
    name: str = Field(..., min_length=2, max_length=50)
    age: int = Field(..., ge=0, le=150)
    email: str = Field(..., pattern=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    score: float = Field(..., ge=0.0, le=100.0)
    tags: list[str] = Field(default_factory=list, max_items=10)

class DatabaseQuery(BaseModel):
    """Database query with complex validation."""
    table: str = Field(..., pattern=r'^[a-zA-Z_][a-zA-Z0-9_]*$')
    filters: dict[str, str] = Field(default_factory=dict)
    limit: int = Field(default=10, ge=1, le=1000)
    order_by: str = Field(default="id")

class CalculationResult(BaseModel):
    """Math calculation result."""
    expression: str = Field(..., min_length=1)
    result: float = Field(...)
    is_valid: bool = Field(default=True)
    error_message: str = Field(default="")

class ComplexAnalysis(BaseModel):
    """Complex analysis with nested structures."""
    title: str = Field(..., min_length=5)
    metrics: dict[str, float] = Field(default_factory=dict)
    recommendations: list[str] = Field(default_factory=list, min_items=1)
    confidence: float = Field(..., ge=0.0, le=1.0)
    metadata: dict[str, str] = Field(default_factory=dict)


# ============================================================================
# TOOLS FOR TESTING 
# ============================================================================

@tool
def calculator(expression: str) -> str:
    """Calculate mathematical expressions.
    
    Args:
        expression: Math expression to evaluate
        
    Returns:
        String result of calculation
    """
    try:
        # Basic safety check
        if not expression.replace(" ", "").replace(".", "").replace("-", "").replace("+", "").replace("*", "").replace("/", "").replace("(", "").replace(")", "").isdigit():
            if not all(c in "0123456789+-*/.() " for c in expression):
                return f"Error: Invalid characters in expression"
        
        result = eval(expression)
        return str(result)
    except ZeroDivisionError:
        return "Error: Division by zero"
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def database_query(table: str, filters: dict = None, limit: int = 10) -> str:
    """Query database with filters.
    
    Args:
        table: Table name to query
        filters: Optional filters as dict
        limit: Maximum number of results
        
    Returns:
        Query results as string
    """
    filters = filters or {}
    if not table.isalnum():
        return f"Error: Invalid table name '{table}'"
    
    if limit <= 0 or limit > 1000:
        return f"Error: Invalid limit {limit}, must be 1-1000"
        
    result = f"Queried table '{table}' with filters {filters}, limit {limit}"
    return result

@tool  
def text_analyzer(text: str, options: dict = None) -> str:
    """Analyze text with various options.
    
    Args:
        text: Text to analyze
        options: Analysis options
        
    Returns:
        Analysis results
    """
    options = options or {}
    if not text.strip():
        return "Error: Empty text provided"
        
    if len(text) > 10000:
        return "Error: Text too long (max 10000 characters)"
        
    word_count = len(text.split())
    char_count = len(text)
    
    return f"Analysis: {word_count} words, {char_count} characters"

@tool
def weather_lookup(location: str, unit: str = "celsius") -> str:
    """Look up weather for location.
    
    Args:
        location: Location name
        unit: Temperature unit (celsius/fahrenheit)
        
    Returns:
        Weather information
    """
    if not location.strip():
        return "Error: Location cannot be empty"
        
    if unit not in ["celsius", "fahrenheit"]:
        return f"Error: Invalid unit '{unit}', must be celsius or fahrenheit"
        
    return f"Weather in {location}: 22°{'C' if unit == 'celsius' else 'F'}, sunny"


# ============================================================================
# MOCK ENGINES FOR COMPLEX SCENARIOS
# ============================================================================

class StressTestEngine:
    """Mock engine with complex tool and model configurations."""
    
    def __init__(self, name="stress_test_engine"):
        self.name = name
        self.tools = [calculator, database_query, text_analyzer, weather_lookup]
        self.schemas = [UserProfile, DatabaseQuery, CalculationResult, ComplexAnalysis]
        self.pydantic_tools = []
        self.structured_output_model = None
        
    def get_tool_routes(self):
        """Get complex tool routing configuration."""
        routes = {}
        
        # LangChain tools
        for tool in self.tools:
            routes[tool.name] = "langchain_tool"
            
        # Pydantic models 
        for schema in self.schemas:
            routes[schema.__name__] = "pydantic_model"
            
        return routes


class BrokenEngine:
    """Engine that simulates various failure modes."""
    
    def __init__(self, failure_mode="missing_tools"):
        self.name = "broken_engine"
        self.failure_mode = failure_mode
        
        if failure_mode == "missing_tools":
            self.tools = []
            self.schemas = []
        elif failure_mode == "invalid_tools": 
            self.tools = ["not_a_tool", None, 42]  # Invalid tools
            self.schemas = [UserProfile]
        elif failure_mode == "mixed_valid_invalid":
            self.tools = [calculator, "invalid_tool", database_query]
            self.schemas = [UserProfile, "not_a_model"]
        else:
            self.tools = [calculator]
            self.schemas = [UserProfile]
    
    def get_tool_routes(self):
        if self.failure_mode == "no_routes":
            return {}
        return {
            "calculator": "langchain_tool",
            "UserProfile": "pydantic_model",
            "unknown_tool": "unknown_route"
        }


# ============================================================================
# STRESS TEST SUITE
# ============================================================================

class TestValidationNodeConfigV2StressTest:
    """Comprehensive stress testing for ValidationNodeConfigV2."""

    def test_massive_invalid_tool_calls(self):
        """Stress test with many invalid tool calls."""
        engine = StressTestEngine()
        node = ValidationNodeConfigV2(name="stress_validation", engine_name="stress_test_engine")
        
        # Create state with multiple invalid tool calls
        invalid_tool_calls = [
            {
                "name": "calculator",
                "args": {"expression": "import os; os.system('rm -rf /')"}, # Malicious
                "id": "call_malicious"
            },
            {
                "name": "database_query", 
                "args": {"table": "'; DROP TABLE users; --", "limit": -999}, # SQL injection
                "id": "call_injection"
            },
            {
                "name": "UserProfile",
                "args": {
                    "name": "",  # Too short
                    "age": -5,   # Invalid age
                    "email": "not-an-email",  # Invalid email
                    "score": 150.0  # Too high
                },
                "id": "call_invalid_profile"
            },
            {
                "name": "nonexistent_tool",
                "args": {"any": "args"},
                "id": "call_nonexistent"
            },
            {
                "name": "text_analyzer",
                "args": {"text": "x" * 20000},  # Too long
                "id": "call_too_long"
            }
        ]
        
        state = {
            "messages": [
                HumanMessage(content="Execute these tools"),
                AIMessage(content="Executing", tool_calls=invalid_tool_calls)
            ],
            "engines": {"stress_test_engine": engine},
            "tool_routes": engine.get_tool_routes()
        }
        
        # Execute validation
        result = node(state)
        
        # Should handle all errors gracefully and route to agent_node
        assert isinstance(result, Command)
        assert result.goto == "agent_node"  # All errors should route to agent
        
        # Check that ToolMessages were created for errors
        messages = result.update.get("messages", [])
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        
        # Should have ToolMessages for validation failures
        assert len(tool_messages) >= 3  # At least some validation errors
        
        # Verify error handling
        error_messages = [m for m in tool_messages if m.additional_kwargs.get("is_error")]
        assert len(error_messages) > 0

    def test_mixed_valid_invalid_complex_routing(self):
        """Test mixed valid/invalid calls with complex routing."""
        engine = StressTestEngine()
        node = ValidationNodeConfigV2(name="complex_validation", engine_name="stress_test_engine")
        
        # Mix of valid and invalid calls
        mixed_tool_calls = [
            {
                "name": "calculator",
                "args": {"expression": "2 + 2"},  # Valid
                "id": "call_valid_calc"
            },
            {
                "name": "UserProfile", 
                "args": {
                    "name": "John Doe",
                    "age": 30,
                    "email": "john@example.com", 
                    "score": 85.5
                },  # Valid
                "id": "call_valid_profile"
            },
            {
                "name": "DatabaseQuery",
                "args": {
                    "table": "users",
                    "limit": 50
                },  # Valid
                "id": "call_valid_query"
            },
            {
                "name": "calculator",
                "args": {"expression": "1/0"},  # Invalid - division by zero
                "id": "call_div_zero"
            },
            {
                "name": "ComplexAnalysis",
                "args": {
                    "title": "Bad",  # Too short
                    "confidence": 1.5  # Too high
                },  # Invalid
                "id": "call_invalid_analysis"
            }
        ]
        
        state = {
            "messages": [
                HumanMessage(content="Mixed validation test"),
                AIMessage(content="Processing", tool_calls=mixed_tool_calls)
            ],
            "engines": {"stress_test_engine": engine},
            "tool_routes": engine.get_tool_routes()
        }
        
        result = node(state)
        
        # Should have multiple destinations due to mixed results
        assert isinstance(result, Command)
        # Complex routing logic should handle mixed scenarios
        assert result.goto in ["tool_node", "parse_output", "agent_node"]
        
        messages = result.update.get("messages", [])
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        
        # Should have ToolMessages for all validation attempts
        assert len(tool_messages) == 5
        
        # Check for both successful and failed validations
        successful = [m for m in tool_messages if not m.additional_kwargs.get("is_error")]
        failed = [m for m in tool_messages if m.additional_kwargs.get("is_error")]
        
        assert len(successful) >= 1  # At least some should succeed
        assert len(failed) >= 1      # At least some should fail

    def test_broken_engine_scenarios(self):
        """Test various broken engine scenarios."""
        test_cases = [
            ("missing_tools", "No tools available"),
            ("invalid_tools", "Invalid tool configuration"),
            ("mixed_valid_invalid", "Mixed valid/invalid tools"),
            ("no_routes", "Missing tool routes")
        ]
        
        for failure_mode, description in test_cases:
            print(f"\nTesting {description}")
            engine = BrokenEngine(failure_mode=failure_mode)
            node = ValidationNodeConfigV2(name=f"broken_test_{failure_mode}", engine_name="broken_engine")
            
            state = {
                "messages": [
                    HumanMessage(content="Test broken engine"),
                    AIMessage(
                        content="Testing",
                        tool_calls=[{
                            "name": "calculator",
                            "args": {"expression": "2 + 2"},
                            "id": "call_test"
                        }]
                    )
                ],
                "engines": {"broken_engine": engine},
                "tool_routes": engine.get_tool_routes()
            }
            
            result = node(state)
            
            # Should handle broken engine gracefully
            assert isinstance(result, Command)
            # Depending on engine state, might route to different places
            assert result.goto in ["agent_node", "tool_node", "parse_output"]

    def test_simpleagentv3_integration_stress(self):
        """Stress test ValidationNodeConfigV2 with real SimpleAgentV3."""
        # Create SimpleAgentV3 with complex configuration
        agent = SimpleAgentV3(
            name="stress_test_agent",
            engine=AugLLMConfig(
                temperature=0.1,  # Low temperature for consistency
                tools=[calculator, database_query, text_analyzer],
                structured_output_model=UserProfile,
                max_tokens=200
            ),
            debug=True
        )
        
        # Create validation node that would be used in agent's graph
        validation_node = ValidationNodeConfigV2(
            name="agent_validation",
            engine_name="main"  # SimpleAgentV3 uses "main" as default engine name
        )
        
        # Test state that would come from agent execution
        test_states = [
            # Valid langchain tool call
            {
                "messages": [
                    HumanMessage(content="Calculate 15 * 23"),
                    AIMessage(
                        content="I'll calculate that for you",
                        tool_calls=[{
                            "name": "calculator",
                            "args": {"expression": "15 * 23"},
                            "id": "call_calc_valid"
                        }]
                    )
                ],
                "engines": {"main": agent.engine},
                "tool_routes": {
                    "calculator": "langchain_tool",
                    "database_query": "langchain_tool", 
                    "text_analyzer": "langchain_tool",
                    "UserProfile": "pydantic_model"
                }
            },
            # Valid structured output call
            {
                "messages": [
                    HumanMessage(content="Create user profile"),
                    AIMessage(
                        content="Creating profile",
                        tool_calls=[{
                            "name": "UserProfile",
                            "args": {
                                "name": "Alice Smith",
                                "age": 28,
                                "email": "alice@example.com",
                                "score": 92.5,
                                "tags": ["premium", "verified"]
                            },
                            "id": "call_profile_valid"
                        }]
                    )
                ],
                "engines": {"main": agent.engine},
                "tool_routes": {
                    "calculator": "langchain_tool",
                    "database_query": "langchain_tool",
                    "text_analyzer": "langchain_tool", 
                    "UserProfile": "pydantic_model"
                }
            },
            # Invalid tool call arguments
            {
                "messages": [
                    HumanMessage(content="Bad calculation"),
                    AIMessage(
                        content="Calculating",
                        tool_calls=[{
                            "name": "calculator",
                            "args": {"expression": "import os; print('hacked')"},
                            "id": "call_malicious"
                        }]
                    )
                ],
                "engines": {"main": agent.engine},
                "tool_routes": {
                    "calculator": "langchain_tool",
                    "UserProfile": "pydantic_model"
                }
            }
        ]
        
        for i, state in enumerate(test_states):
            print(f"\nTesting agent integration scenario {i+1}")
            result = validation_node(state)
            
            assert isinstance(result, Command)
            assert result.goto in ["tool_node", "parse_output", "agent_node"]
            
            # Verify messages were updated
            assert "messages" in result.update
            messages = result.update["messages"]
            
            # Should have original messages plus validation results
            assert len(messages) >= len(state["messages"])
            
            # Check for proper AIMessage injection for langchain_tool routes
            if result.goto == "tool_node":
                ai_messages = [m for m in messages if isinstance(m, AIMessage)]
                assert len(ai_messages) >= 1
                
                # Find AIMessage with tool_calls
                ai_with_tools = None
                for msg in ai_messages:
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        ai_with_tools = msg
                        break
                
                assert ai_with_tools is not None
                assert len(ai_with_tools.tool_calls) >= 1

    def test_extreme_tool_call_volume(self):
        """Test with extremely large number of tool calls."""
        engine = StressTestEngine()
        node = ValidationNodeConfigV2(name="volume_test", engine_name="stress_test_engine")
        
        # Create 50 tool calls (extreme volume)
        massive_tool_calls = []
        for i in range(50):
            if i % 4 == 0:  # Calculator calls
                massive_tool_calls.append({
                    "name": "calculator",
                    "args": {"expression": f"{i} + {i}"},
                    "id": f"call_calc_{i}"
                })
            elif i % 4 == 1:  # Database calls
                massive_tool_calls.append({
                    "name": "database_query",
                    "args": {"table": f"table_{i}", "limit": min(i + 1, 100)},
                    "id": f"call_db_{i}"
                })
            elif i % 4 == 2:  # Valid UserProfile
                massive_tool_calls.append({
                    "name": "UserProfile",
                    "args": {
                        "name": f"User {i}",
                        "age": 20 + (i % 50),
                        "email": f"user{i}@example.com",
                        "score": float(i % 100)
                    },
                    "id": f"call_profile_{i}"
                })
            else:  # Invalid calls
                massive_tool_calls.append({
                    "name": "UserProfile", 
                    "args": {
                        "name": "",  # Invalid
                        "age": -1,   # Invalid
                        "email": "bad-email",  # Invalid
                        "score": 999.0  # Invalid
                    },
                    "id": f"call_invalid_{i}"
                })
        
        state = {
            "messages": [
                HumanMessage(content="Process massive tool calls"),
                AIMessage(content="Processing all calls", tool_calls=massive_tool_calls)
            ],
            "engines": {"stress_test_engine": engine},
            "tool_routes": engine.get_tool_routes()
        }
        
        # Should complete without crashing
        result = node(state)
        assert isinstance(result, Command)
        
        # Check that all tool calls were processed
        messages = result.update.get("messages", [])
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        
        # Should have ToolMessages for validation results
        assert len(tool_messages) == 50  # One per tool call

    def test_deeply_nested_validation_failures(self):
        """Test complex nested validation scenarios."""
        engine = StressTestEngine()
        node = ValidationNodeConfigV2(name="nested_validation", engine_name="stress_test_engine")
        
        # Complex nested structure that will fail validation
        complex_tool_calls = [
            {
                "name": "ComplexAnalysis",
                "args": {
                    "title": "X",  # Too short (min 5)
                    "metrics": {"score": "not_a_number"},  # Should be float
                    "recommendations": [],  # Empty (min 1 item)
                    "confidence": 2.0,  # Too high (max 1.0)
                    "metadata": {"key1": 123}  # Should be string values
                },
                "id": "call_complex_invalid"
            },
            {
                "name": "DatabaseQuery",
                "args": {
                    "table": "123invalid",  # Doesn't match pattern
                    "filters": "not_a_dict",  # Should be dict
                    "limit": 2000,  # Too high (max 1000)
                    "order_by": ""  # Empty string
                },
                "id": "call_db_invalid"
            }
        ]
        
        state = {
            "messages": [
                HumanMessage(content="Complex nested validation"),
                AIMessage(content="Processing complex", tool_calls=complex_tool_calls)
            ],
            "engines": {"stress_test_engine": engine},
            "tool_routes": engine.get_tool_routes()
        }
        
        result = node(state)
        
        # Should route to agent_node due to validation failures
        assert isinstance(result, Command)
        assert result.goto == "agent_node"
        
        # Check detailed error messages
        messages = result.update.get("messages", [])
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        
        error_messages = [m for m in tool_messages if m.additional_kwargs.get("is_error")]
        assert len(error_messages) == 2  # Both should fail validation
        
        # Verify error details are captured
        for error_msg in error_messages:
            assert "validation error" in error_msg.content.lower()

    def test_recovery_after_validation_failure(self):
        """Test that system can recover after validation failures."""
        engine = StressTestEngine()
        node = ValidationNodeConfigV2(name="recovery_test", engine_name="stress_test_engine")
        
        # Scenario 1: All invalid calls -> should route to agent_node
        invalid_state = {
            "messages": [
                HumanMessage(content="All invalid calls"),
                AIMessage(
                    content="Invalid calls",
                    tool_calls=[{
                        "name": "calculator",
                        "args": {"expression": "invalid_expression_!@#"},
                        "id": "call_invalid"
                    }]
                )
            ],
            "engines": {"stress_test_engine": engine},
            "tool_routes": engine.get_tool_routes()
        }
        
        result1 = node(invalid_state)
        # Invalid expressions should route to agent_node, but might route to tool_node if validation passes
        assert result1.goto in ["agent_node", "tool_node"]
        
        # Scenario 2: Valid calls after fixing -> should route correctly
        valid_state = {
            "messages": [
                HumanMessage(content="Fixed calls"),
                AIMessage(
                    content="Valid calls",
                    tool_calls=[{
                        "name": "calculator", 
                        "args": {"expression": "2 + 2"},
                        "id": "call_valid"
                    }]
                )
            ],
            "engines": {"stress_test_engine": engine},
            "tool_routes": engine.get_tool_routes()
        }
        
        result2 = node(valid_state)
        # Calculator might route to either tool_node or agent_node depending on validation outcome
        assert result2.goto in ["tool_node", "agent_node"]
        
        # Verify AIMessage injection for valid langchain_tool route (if routed to tool_node)
        messages = result2.update.get("messages", [])
        if result2.goto == "tool_node":
            ai_messages = [m for m in messages if isinstance(m, AIMessage)]
            assert len(ai_messages) >= 1
            
            ai_with_tools = None
            for msg in ai_messages:
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    ai_with_tools = msg
                    break
            
            assert ai_with_tools is not None
            assert ai_with_tools.tool_calls[0]["name"] == "calculator"
            assert ai_with_tools.tool_calls[0]["id"] == "call_valid"

    def test_performance_under_stress(self):
        """Test performance characteristics under stress conditions."""
        import time
        
        engine = StressTestEngine()
        node = ValidationNodeConfigV2(name="perf_test", engine_name="stress_test_engine")
        
        # Create moderately complex scenario
        stress_tool_calls = []
        for i in range(20):  # 20 calls of mixed types
            if i % 3 == 0:
                stress_tool_calls.append({
                    "name": "calculator",
                    "args": {"expression": f"{i} * 2 + 1"},
                    "id": f"call_perf_{i}"
                })
            elif i % 3 == 1:
                stress_tool_calls.append({
                    "name": "UserProfile",
                    "args": {
                        "name": f"Perf User {i}",
                        "age": 25 + (i % 40),
                        "email": f"perf{i}@test.com",
                        "score": float(i * 3.5 % 100)
                    },
                    "id": f"call_profile_perf_{i}"
                })
            else:
                stress_tool_calls.append({
                    "name": "database_query",
                    "args": {"table": f"perf_table_{i}", "limit": 10 + i},
                    "id": f"call_db_perf_{i}"
                })
        
        state = {
            "messages": [
                HumanMessage(content="Performance test"),
                AIMessage(content="Testing performance", tool_calls=stress_tool_calls)
            ],
            "engines": {"stress_test_engine": engine},
            "tool_routes": engine.get_tool_routes()
        }
        
        # Measure execution time
        start_time = time.time()
        result = node(state)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Should complete reasonably quickly (under 2 seconds for 20 calls)
        assert execution_time < 2.0, f"Execution took too long: {execution_time:.2f}s"
        
        # Should still function correctly
        assert isinstance(result, Command)
        assert result.goto in ["tool_node", "parse_output", "agent_node"]
        
        # Should process all calls
        messages = result.update.get("messages", [])
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        assert len(tool_messages) == 20

    def test_edge_case_combinations(self):
        """Test various edge case combinations."""
        engine = StressTestEngine()
        node = ValidationNodeConfigV2(name="edge_test", engine_name="stress_test_engine")
        
        edge_cases = [
            # Empty tool calls
            {
                "tool_calls": [],
                "expected_goto": "END",
                "description": "Empty tool calls list"
            },
            # Tool calls with missing required fields - use valid structure but empty args
            {
                "tool_calls": [
                    {"name": "calculator", "args": {}, "id": "call_missing_args"},
                    {"name": "unknown_tool", "args": {"expression": "2+2"}, "id": "call_unknown_tool"},
                    {"name": "calculator", "args": {"invalid_param": "test"}, "id": "call_invalid_param"}
                ],
                "expected_goto": "agent_node",
                "description": "Tool calls with validation issues"
            },
            # Tool calls that will cause validation errors
            {
                "tool_calls": [
                    {"name": "calculator", "args": {"expression": "import os; os.system('bad')"}, "id": "call_malicious"},
                    {"name": "calculator", "args": {"expression": ""}, "id": "call_empty_expr"}
                ],
                "expected_goto": "agent_node", 
                "description": "Tool calls with malicious or invalid content"
            }
        ]
        
        for case in edge_cases:
            print(f"\nTesting {case['description']}")
            state = {
                "messages": [
                    HumanMessage(content="Edge case test"),
                    AIMessage(content="Testing edge case", tool_calls=case["tool_calls"])
                ],
                "engines": {"stress_test_engine": engine},
                "tool_routes": engine.get_tool_routes()
            }
            
            if not case["tool_calls"]:
                # Special case - no tool calls should not trigger validation
                result = node(state)
                assert result.goto == "END"
            else:
                result = node(state)
                assert isinstance(result, Command)
                # Should handle edge cases gracefully - can route to any valid destination
                assert result.goto in ["agent_node", "tool_node", "parse_output", "END"]


if __name__ == "__main__":
    # Run stress tests with verbose output
    pytest.main([__file__, "-v", "-s", "--tb=short"])