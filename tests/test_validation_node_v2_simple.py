"""Simple direct test of ValidationNodeV2 functionality."""

import json
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from haive.core.graph.node.validation_node_v2 import ValidationNodeV2


def test_validation_node_basic():
    """Test basic functionality of ValidationNodeV2."""
    # Create validation node
    validation_node = ValidationNodeV2(
        name="test_validation",
        engine_name="test_engine",
        router_node="validation_router"
    )
    
    # Create mock state with messages
    state = type('State', (), {
        'messages': [
            HumanMessage(content="Test"),
            AIMessage(
                content="I'll analyze this",
                tool_calls=[{
                    "id": "call_123",
                    "name": "TestModel",
                    "args": {"field": "value"}
                }]
            )
        ],
        'engines': {
            'test_engine': type('Engine', (), {
                'tool_routes': {
                    'TestModel': 'pydantic_model',
                    'calculator': 'langchain_tool'
                },
                'structured_output_model': type('TestModel', (), {
                    '__name__': 'TestModel'
                })
            })()
        }
    })()
    
    # Execute
    result = validation_node(state)
    
    print(f"Result type: {type(result)}")
    print(f"Result goto: {result.goto}")
    print(f"Result update: {result.update}")
    
    # Check if we're looking at tool routes
    print(f"\nChecking _get_tool_routes_from_engine...")
    engine = validation_node._get_engine_from_state(state)
    print(f"Engine found: {engine}")
    if engine:
        tool_routes = validation_node._get_tool_routes_from_engine(engine)
        print(f"Tool routes: {tool_routes}")


if __name__ == "__main__":
    test_validation_node_basic()