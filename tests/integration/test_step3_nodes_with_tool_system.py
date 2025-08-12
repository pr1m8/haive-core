#!/usr/bin/env python3
"""
Step 3: Node Integration Tests - ValidationNodeV2 + ToolNodeConfig + Fixed Tool System

This module tests that our node layer (ValidationNodeV2, ToolNodeConfig) works properly
with the fixed tool system (ToolRouteMixin, ToolEngine, AugLLMConfig) from Steps 1-2.

Key Integration Points:
- ValidationNodeV2 uses ToolRouteMixin.tool_routes for routing decisions  
- ToolNodeConfig filters tools by routes from engines
- Both nodes work with state that contains tool_routes from AugLLMConfig
- No tool duplication issues (fixed in Step 2)
- Proper state updates with ToolMessages
"""


import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.types import Command
from pydantic import BaseModel, Field

from haive.core.engine.aug_llm import AugLLMConfig
from haive.core.engine.tool import ToolEngine
from haive.core.graph.node.tool_node_config_v2 import ToolNodeConfig
from haive.core.graph.node.validation_node_v2 import ValidationNodeV2
from haive.core.schema.prebuilt.tool_state import ToolState
from haive.core.tools.store_manager import StoreManager


# Test tools
@tool
def calculator(expression: str) -> str:
    """Calculate mathematical expressions."""
    return f"Result: {eval(expression)}"


@tool
def text_processor(text: str) -> str:
    """Process text."""
    return f"Processed: {text}"


class SearchResults(BaseModel):
    """Search results Pydantic model."""
    query: str = Field(description="The search query")
    results: list[str] = Field(description="Search results")
    count: int = Field(description="Number of results")


class TestNodesWithToolSystem:
    """Test ValidationNodeV2 and ToolNodeConfig with the fixed tool system."""

    def test_validation_node_v2_with_aug_llm_config_routes(self):
        """Test ValidationNodeV2 uses tool routes from AugLLMConfig correctly."""
        # Create AugLLMConfig with tools (fixed - no duplication)
        config = AugLLMConfig(
            tools=[calculator, text_processor]
        )

        # Verify no duplication (from Step 2 fix)
        assert len(config.tools) == 2
        assert len(config.tool_routes) == 2

        # Create ValidationNodeV2
        validation_node = ValidationNodeV2(
            name="test_validation",
            messages_key="messages",
            engine_name="test_engine"
        )

        # Create state with tool routes from AugLLMConfig
        test_state = ToolState(
            messages=[
                AIMessage(
                    content="I'll calculate 15 * 23",
                    tool_calls=[{
                        "name": "calculator",
                        "args": {"expression": "15 * 23"},
                        "id": "call_1"
                    }]
                )
            ],
            tool_routes=config.tool_routes,  # Routes from fixed AugLLMConfig
            engines={"test_engine": config},  # Provide engine in state
            engine_name="test_engine"
        )

        # Set the engine on validation node (simulates graph context)
        validation_node.clear_tool_routes()
        for tool_name, route in config.tool_routes.items():
            validation_node.set_tool_route(tool_name, route)

        # Execute validation node
        result = validation_node(test_state, {})

        # Should return Command to continue to router
        assert isinstance(result, Command)
        assert "validation_router" in result.graph

        # Check that ToolMessages were added to state
        updated_state = result.update
        assert "messages" in updated_state
        messages = updated_state["messages"]

        # Should have original AIMessage + new ToolMessage
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        assert len(tool_messages) == 1
        assert tool_messages[0].name == "calculator"
        assert "Result: 345" in tool_messages[0].content

    def test_tool_node_config_with_engine_tool_routes(self):
        """Test ToolNodeConfig properly filters tools using engine routes."""
        # Create AugLLMConfig with mixed tools
        calculator_tool = calculator
        processor_tool = text_processor

        config = AugLLMConfig(
            tools=[calculator_tool, processor_tool]
        )

        # Verify routes are set correctly (no duplication)
        assert len(config.tool_routes) == 2
        assert "calculator" in config.tool_routes
        assert "text_processor" in config.tool_routes

        # Create ToolNodeConfig that allows langchain_tool routes
        tool_node = ToolNodeConfig(
            name="test_tool_node",
            engine_name="test_engine",
            allowed_routes=["langchain_tool", "pydantic_model"]
        )

        # Create state with engine and tool routes
        test_state = ToolState(
            messages=[
                AIMessage(
                    content="I need to calculate something",
                    tool_calls=[{
                        "name": "calculator",
                        "args": {"expression": "10 + 5"},
                        "id": "call_1"
                    }]
                )
            ],
            engines={"test_engine": config},  # Engine with tool routes
            tool_routes=config.tool_routes
        )

        # Execute tool node
        result = tool_node(test_state, {})

        # Should return state update with ToolMessages
        assert isinstance(result, dict)
        assert "messages" in result

        messages = result["messages"]
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        assert len(tool_messages) == 1
        assert "Result: 15" in tool_messages[0].content

    def test_validation_node_with_pydantic_model_routing(self):
        """Test ValidationNodeV2 handles Pydantic model routing correctly."""
        # Create ToolEngine with structured output tool
        search_tool = ToolEngine.create_structured_output_tool(
            func=lambda query: SearchResults(
                query=query,
                results=[f"Result 1 for {query}", f"Result 2 for {query}"],
                count=2
            ),
            name="structured_search",
            description="Search with structured output",
            output_model=SearchResults
        )

        # Create AugLLMConfig with structured tool
        config = AugLLMConfig(
            tools=[search_tool, calculator],
            structured_output_model=SearchResults
        )

        # Verify routing includes pydantic_model
        assert "SearchResults" in config.tool_routes
        assert config.tool_routes["SearchResults"] == "pydantic_model"
        assert config.tool_routes["calculator"] == "langchain_tool"

        # Create ValidationNodeV2
        validation_node = ValidationNodeV2(name="pydantic_validation")

        # Set tool routes from config
        validation_node.clear_tool_routes()
        for tool_name, route in config.tool_routes.items():
            validation_node.set_tool_route(tool_name, route)

        # Create state with Pydantic model tool call
        test_state = ToolState(
            messages=[
                AIMessage(
                    content="Search results",
                    tool_calls=[{
                        "name": "SearchResults",
                        "args": {
                            "query": "python tutorial",
                            "results": ["Result 1", "Result 2"],
                            "count": 2
                        },
                        "id": "call_1"
                    }]
                )
            ],
            tool_routes=config.tool_routes,
            engine_name="test_engine"
        )

        # Execute validation
        result = validation_node(test_state, {})

        # Should create ToolMessage for Pydantic model
        assert isinstance(result, Command)
        updated_state = result.update
        messages = updated_state["messages"]

        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        assert len(tool_messages) == 1
        assert tool_messages[0].name == "SearchResults"

        # Should contain the Pydantic model data
        tool_content = tool_messages[0].content
        assert "python tutorial" in tool_content
        assert "Result 1" in tool_content

    def test_nodes_with_store_tools_integration(self):
        """Test nodes work with store tools from ToolEngine."""
        # Create store tools suite (tests memory tools integration)
        store_manager = StoreManager()
        store_tools = ToolEngine.create_store_tools_suite(
            store_manager=store_manager,
            namespace=("test", "nodes"),
            include_tools=["store", "search"]
        )

        # Create AugLLMConfig with store tools
        config = AugLLMConfig(tools=store_tools)

        # Verify store tools have correct routes
        assert len(config.tool_routes) == 2
        store_names = [name for name in config.tool_routes.keys() if "store" in name]
        search_names = [name for name in config.tool_routes.keys() if "search" in name]
        assert len(store_names) == 1
        assert len(search_names) == 1

        # Create ToolNodeConfig for store tools
        tool_node = ToolNodeConfig(
            name="store_tool_node",
            engine_name="store_engine",
            allowed_routes=["langchain_tool"]
        )

        # Create state with store tool call
        store_tool_name = store_names[0]
        test_state = ToolState(
            messages=[
                AIMessage(
                    content="I'll store some data",
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

        # Execute tool node with store tools
        result = tool_node(test_state, {})

        # Should successfully execute store tool
        assert isinstance(result, dict)
        assert "messages" in result
        messages = result["messages"]
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        assert len(tool_messages) == 1
        # Store operation should succeed
        assert "stored" in tool_messages[0].content.lower() or "success" in tool_messages[0].content.lower()

    def test_dynamic_tool_updates_with_nodes(self):
        """Test that nodes handle dynamic tool updates correctly."""
        # Start with basic AugLLMConfig
        config = AugLLMConfig(tools=[calculator])

        # Verify initial state (no duplication bug)
        assert len(config.tools) == 1
        assert len(config.tool_routes) == 1

        # Create ValidationNodeV2 with initial routes
        validation_node = ValidationNodeV2(name="dynamic_validation")
        validation_node.clear_tool_routes()
        for tool_name, route in config.tool_routes.items():
            validation_node.set_tool_route(tool_name, route)

        # Add tool dynamically
        config.add_tool(text_processor)

        # Verify dynamic addition (no duplication bug)
        assert len(config.tools) == 2
        assert len(config.tool_routes) == 2

        # Update validation node routes
        validation_node.clear_tool_routes()
        for tool_name, route in config.tool_routes.items():
            validation_node.set_tool_route(tool_name, route)

        # Test with both tools
        test_state = ToolState(
            messages=[
                AIMessage(
                    content="Process text and calculate",
                    tool_calls=[
                        {
                            "name": "calculator",
                            "args": {"expression": "5 + 5"},
                            "id": "call_1"
                        },
                        {
                            "name": "text_processor",
                            "args": {"text": "hello world"},
                            "id": "call_2"
                        }
                    ]
                )
            ],
            tool_routes=config.tool_routes,
            engine_name="dynamic_engine"
        )

        # Execute validation with both tools
        result = validation_node(test_state, {})

        # Should handle both tool calls
        assert isinstance(result, Command)
        updated_state = result.update
        messages = updated_state["messages"]
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]

        # Should have 2 ToolMessages (one for each tool)
        assert len(tool_messages) == 2
        tool_names = {msg.name for msg in tool_messages}
        assert "calculator" in tool_names
        assert "text_processor" in tool_names

    def test_comprehensive_node_tool_workflow(self):
        """Test complete workflow: AugLLMConfig → ToolNode → ValidationNode."""
        # Create comprehensive tool setup
        enhanced_tool = ToolEngine.create_state_tool(
            lambda x: f"Enhanced: {x}",
            name="enhanced_processor",
            reads_state=True,
            state_keys=["messages"]
        )

        config = AugLLMConfig(
            tools=[calculator, enhanced_tool, text_processor]
        )

        # Verify comprehensive setup (no duplication)
        assert len(config.tools) == 3
        assert len(config.tool_routes) == 3

        # Create both node types
        tool_node = ToolNodeConfig(
            name="workflow_tool_node",
            engine_name="workflow_engine",
            allowed_routes=["langchain_tool"]
        )

        validation_node = ValidationNodeV2(name="workflow_validation")
        validation_node.clear_tool_routes()
        for tool_name, route in config.tool_routes.items():
            validation_node.set_tool_route(tool_name, route)

        # Create comprehensive test state
        test_state = ToolState(
            messages=[
                HumanMessage(content="Please calculate 20 * 3"),
                AIMessage(
                    content="I'll calculate that for you",
                    tool_calls=[{
                        "name": "calculator",
                        "args": {"expression": "20 * 3"},
                        "id": "call_1"
                    }]
                )
            ],
            engines={"workflow_engine": config},
            tool_routes=config.tool_routes
        )

        # Step 1: Execute tool node (should add ToolMessage)
        tool_result = tool_node(test_state, {})
        assert isinstance(tool_result, dict)

        # Step 2: Update state with tool node results
        updated_state = ToolState(
            messages=tool_result["messages"],
            tool_routes=test_state.tool_routes,
            engine_name="workflow_engine"
        )

        # Step 3: Execute validation node (should handle ToolMessages)
        validation_result = validation_node(updated_state, {})

        # Should complete the workflow successfully
        assert isinstance(validation_result, Command)
        final_state = validation_result.update
        final_messages = final_state["messages"]

        # Should have: HumanMessage + AIMessage + ToolMessage + validation ToolMessage
        assert len(final_messages) >= 3
        tool_messages = [m for m in final_messages if isinstance(m, ToolMessage)]
        assert len(tool_messages) >= 1

        # Verify calculation was performed
        calc_results = [msg.content for msg in tool_messages if "60" in msg.content]
        assert len(calc_results) > 0

    def test_error_handling_in_node_tool_integration(self):
        """Test error handling when nodes encounter tool issues."""
        # Create config with valid and invalid scenarios
        config = AugLLMConfig(tools=[calculator])

        # Create ValidationNodeV2
        validation_node = ValidationNodeV2(name="error_validation")
        validation_node.clear_tool_routes()
        for tool_name, route in config.tool_routes.items():
            validation_node.set_tool_route(tool_name, route)

        # Test with invalid tool call (nonexistent tool)
        error_state = ToolState(
            messages=[
                AIMessage(
                    content="I'll use a nonexistent tool",
                    tool_calls=[{
                        "name": "nonexistent_tool",
                        "args": {"param": "value"},
                        "id": "call_1"
                    }]
                )
            ],
            tool_routes=config.tool_routes,
            engine_name="error_engine"
        )

        # Execute validation node with error
        result = validation_node(error_state, {})

        # Should handle error gracefully and create error ToolMessage
        assert isinstance(result, Command)
        updated_state = result.update
        messages = updated_state["messages"]
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]

        # Should have error ToolMessage
        assert len(tool_messages) == 1
        error_msg = tool_messages[0]
        assert error_msg.name == "nonexistent_tool"
        # Should contain error information
        assert "error" in error_msg.content.lower() or "unknown" in error_msg.content.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
