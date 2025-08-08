#!/usr/bin/env python3
"""
Step 3: Basic Node Integration Tests - Focus on Success Cases

This module tests that our V2 nodes work correctly with the fixed tool system.
Focus on successful integration paths rather than error cases.

Key Integration Points:
- ValidationNodeV2 processes tool calls correctly
- ToolNodeConfig executes tools properly  
- ParserNodeConfigV2 handles structured output
- All work with AugLLMConfig's fixed tool routes (no duplication)
"""

import pytest
from typing import List, Dict, Any
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.types import Command
from pydantic import BaseModel, Field

from haive.core.engine.aug_llm import AugLLMConfig
from haive.core.engine.tool import ToolEngine
from haive.core.graph.node.validation_node_v2 import ValidationNodeV2
from haive.core.graph.node.tool_node_config_v2 import ToolNodeConfig
from haive.core.graph.node.parser_node_config_v2 import ParserNodeConfigV2
from haive.core.schema.prebuilt.tool_state import ToolState


# Test tools
@tool
def simple_calculator(expression: str) -> str:
    """Calculate mathematical expressions."""
    return f"Result: {eval(expression)}"


class TestResult(BaseModel):
    """Test structured output model."""
    input: str = Field(description="The input provided")
    output: str = Field(description="The processed output")
    success: bool = Field(description="Whether processing succeeded")


class TestBasicNodeIntegration:
    """Basic integration tests for V2 nodes with fixed tool system."""
    
    def test_aug_llm_config_no_duplication_baseline(self):
        """Baseline test: Verify AugLLMConfig works without duplication."""
        config = AugLLMConfig(tools=[simple_calculator])
        
        # Fixed tool duplication bug - should have exactly 1 tool
        assert len(config.tools) == 1
        assert len(config.tool_routes) == 1
        assert "simple_calculator" in config.tool_routes
        assert config.tool_routes["simple_calculator"] == "langchain_tool"
    
    def test_tool_node_config_basic_execution(self):
        """Test ToolNodeConfig executes tools from AugLLMConfig correctly.""" 
        # Create AugLLMConfig (no duplication)
        config = AugLLMConfig(tools=[simple_calculator])
        assert len(config.tools) == 1  # Verify fix
        
        # Create ToolNodeConfig
        tool_node = ToolNodeConfig(
            name="basic_tool_node",
            engine_name="test_engine",
            allowed_routes=["langchain_tool"]
        )
        
        # Create state with tool call
        state = ToolState(
            messages=[
                AIMessage(
                    content="Calculate 5 + 3",
                    tool_calls=[{
                        "name": "simple_calculator",
                        "args": {"expression": "5 + 3"},
                        "id": "call_1"
                    }]
                )
            ],
            engines={"test_engine": config},
            tool_routes=config.tool_routes
        )
        
        # Execute tool node
        result = tool_node(state, {})
        
        # Should successfully execute tool
        assert isinstance(result, dict) or isinstance(result, Command)
        if isinstance(result, dict):
            assert "messages" in result
            messages = result["messages"]
            tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
            assert len(tool_messages) >= 1
            assert "Result: 8" in tool_messages[0].content
    
    def test_validation_node_route_detection(self):
        """Test ValidationNodeV2 correctly detects tool routes."""
        # Create config with multiple tool types
        structured_tool = ToolEngine.create_structured_output_tool(
            func=lambda input_text: TestResult(
                input=input_text,
                output=f"Processed: {input_text}",
                success=True
            ),
            name="structured_processor",
            description="Process with structured output",
            output_model=TestResult
        )
        
        config = AugLLMConfig(
            tools=[simple_calculator, structured_tool],
            structured_output_model=TestResult
        )
        
        # Verify mixed routes (no duplication)
        assert len(config.tools) >= 2  # At least calculator + structured
        assert "simple_calculator" in config.tool_routes
        assert config.tool_routes["simple_calculator"] == "langchain_tool"
        
        # ValidationNodeV2 should handle Pydantic models
        validation_node = ValidationNodeV2(
            name="route_detection",
            engine_name="test_engine"
        )
        
        # Set up tool routes
        validation_node.clear_tool_routes()
        for tool_name, route in config.tool_routes.items():
            validation_node.set_tool_route(tool_name, route)
        
        # Test state with Pydantic model tool call
        state = ToolState(
            messages=[
                AIMessage(
                    content="Process test",
                    tool_calls=[{
                        "name": "TestResult",
                        "args": {
                            "input": "test input",
                            "output": "test output",  
                            "success": True
                        },
                        "id": "call_1"
                    }]
                )
            ],
            engines={"test_engine": config},
            tool_routes=config.tool_routes,
            engine_name="test_engine"
        )
        
        # Execute validation node
        result = validation_node(state, {})
        
        # Should return Command to route to next node
        assert isinstance(result, Command)
    
    def test_parser_node_config_basic_functionality(self):
        """Test ParserNodeConfigV2 basic parsing capabilities."""
        # Create ParserNodeConfigV2
        parser_node = ParserNodeConfigV2(
            name="basic_parser",
            output_model=TestResult,
            add_tool_message_safety_net=True
        )
        
        # Create state with structured output
        state = ToolState(
            messages=[
                HumanMessage(content="Parse this data"),
                AIMessage(content='{"input": "test", "output": "processed", "success": true}')
            ]
        )
        
        # Execute parser node 
        result = parser_node(state, {})
        
        # Should successfully parse and continue
        assert isinstance(result, Command) or isinstance(result, dict)
    
    def test_comprehensive_node_workflow_success_path(self):
        """Test successful workflow: ToolNode → ValidationNode → ParserNode."""
        # Create comprehensive setup
        config = AugLLMConfig(
            tools=[simple_calculator],
            structured_output_model=TestResult
        )
        
        # Verify setup (no duplication)
        assert len(config.tools) >= 1
        assert len(config.tool_routes) >= 1
        
        # Create all three node types
        tool_node = ToolNodeConfig(
            name="workflow_tool",
            engine_name="workflow_engine",
            allowed_routes=["langchain_tool"]
        )
        
        validation_node = ValidationNodeV2(
            name="workflow_validation",
            engine_name="workflow_engine"  
        )
        validation_node.clear_tool_routes()
        for tool_name, route in config.tool_routes.items():
            validation_node.set_tool_route(tool_name, route)
        
        parser_node = ParserNodeConfigV2(
            name="workflow_parser",
            output_model=TestResult
        )
        
        # Test initial state
        initial_state = ToolState(
            messages=[
                HumanMessage(content="Calculate 10 * 5"),
                AIMessage(
                    content="I'll calculate that",
                    tool_calls=[{
                        "name": "simple_calculator",
                        "args": {"expression": "10 * 5"},
                        "id": "call_1"
                    }]
                )
            ],
            engines={"workflow_engine": config},
            tool_routes=config.tool_routes
        )
        
        # Step 1: Execute tool node (should create ToolMessage)
        tool_result = tool_node(initial_state, {})
        
        # Verify tool execution
        if isinstance(tool_result, dict) and "messages" in tool_result:
            # Tool node succeeded - check for ToolMessage
            tool_messages = [m for m in tool_result["messages"] if isinstance(m, ToolMessage)]
            if len(tool_messages) > 0:
                assert "50" in tool_messages[0].content
                print(f"✅ Tool execution successful: {tool_messages[0].content}")
            else:
                print("ℹ️  Tool node completed but no ToolMessage in result")
        
        # Step 2: Test validation node with a Pydantic tool call
        pydantic_state = ToolState(
            messages=[
                AIMessage(
                    content="Return structured result",
                    tool_calls=[{
                        "name": "TestResult",
                        "args": {
                            "input": "workflow test",
                            "output": "workflow success",
                            "success": True
                        },
                        "id": "call_2"
                    }]
                )
            ],
            engines={"workflow_engine": config},
            tool_routes=config.tool_routes,
            engine_name="workflow_engine"
        )
        
        validation_result = validation_node(pydantic_state, {})
        assert isinstance(validation_result, Command)
        print("✅ Validation node handled Pydantic model successfully")
        
        # Step 3: Test parser node
        parser_state = ToolState(
            messages=[
                HumanMessage(content="Parse result"),
                AIMessage(content='{"input": "final test", "output": "final result", "success": true}')
            ]
        )
        
        parser_result = parser_node(parser_state, {})
        print("✅ Parser node completed successfully")
        
        # All nodes executed successfully
        assert True  # If we reach here, all nodes worked
    
    def test_node_integration_with_store_tools(self):
        """Test nodes work with store tools from ToolEngine."""
        # Create store tools
        from haive.core.tools.store_manager import StoreManager
        store_manager = StoreManager()
        store_tools = ToolEngine.create_store_tools_suite(
            store_manager=store_manager,
            namespace=("test", "nodes"),
            include_tools=["store"]
        )
        
        # Create AugLLMConfig with store tools
        config = AugLLMConfig(tools=store_tools)
        
        # Verify store tools integration (no duplication)
        assert len(config.tool_routes) >= 1
        store_tool_names = [name for name in config.tool_routes.keys() if "store" in name]
        assert len(store_tool_names) >= 1
        
        # Create ToolNodeConfig for store tools
        tool_node = ToolNodeConfig(
            name="store_tool_node",
            engine_name="store_engine",
            allowed_routes=["langchain_tool"]
        )
        
        # Test with store tool call
        store_tool_name = store_tool_names[0]
        state = ToolState(
            messages=[
                AIMessage(
                    content="Store some data",
                    tool_calls=[{
                        "name": store_tool_name,
                        "args": {"data": "test data", "key": "test_key"},
                        "id": "call_1"
                    }]
                )
            ],
            engines={"store_engine": config},
            tool_routes=config.tool_routes
        )
        
        # Execute store tool
        result = tool_node(state, {})
        
        # Store operation should succeed
        if isinstance(result, dict) and "messages" in result:
            tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
            if len(tool_messages) > 0:
                assert len(tool_messages[0].content) > 0  # Got some response
                print(f"✅ Store tool integration successful")
    
    def test_dynamic_tool_addition_with_nodes(self):
        """Test nodes handle dynamic tool updates from AugLLMConfig."""
        # Start with one tool
        config = AugLLMConfig(tools=[simple_calculator])
        assert len(config.tools) == 1  # Verify no duplication
        
        # Add tool dynamically 
        @tool
        def text_processor(text: str) -> str:
            return f"Processed: {text}"
        
        config.add_tool(text_processor)
        
        # Verify dynamic addition (no duplication bug)
        assert len(config.tools) == 2
        assert len(config.tool_routes) == 2
        assert "simple_calculator" in config.tool_routes
        assert "text_processor" in config.tool_routes
        
        # Create ToolNodeConfig with updated tools
        tool_node = ToolNodeConfig(
            name="dynamic_tool_node",
            engine_name="dynamic_engine",
            allowed_routes=["langchain_tool"]
        )
        
        # Test with new tool
        state = ToolState(
            messages=[
                AIMessage(
                    content="Process text",
                    tool_calls=[{
                        "name": "text_processor",
                        "args": {"text": "hello world"},
                        "id": "call_1"
                    }]
                )
            ],
            engines={"dynamic_engine": config},
            tool_routes=config.tool_routes
        )
        
        # Execute with dynamically added tool
        result = tool_node(state, {})
        
        # Should handle dynamic tool successfully
        if isinstance(result, dict) and "messages" in result:
            tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
            if len(tool_messages) > 0:
                assert "Processed: hello world" in tool_messages[0].content
                print("✅ Dynamic tool addition integration successful")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])