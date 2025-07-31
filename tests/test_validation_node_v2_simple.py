"""Simple direct test of ValidationNodeV2 functionality."""

from langchain_core.messages import AIMessage, HumanMessage

from haive.core.graph.node.validation_node_v2 import ValidationNodeV2


def test_validation_node_basic():
    """Test basic functionality of ValidationNodeV2."""
    # Create validation node
    validation_node = ValidationNodeV2(
        name="test_validation",
        engine_name="test_engine",
        router_node="validation_router",
    )

    # Create mock state with messages
    state = type(
        "State",
        (),
        {
            "messages": [
                HumanMessage(content="Test"),
                AIMessage(
                    content="I'll analyze this",
                    tool_calls=[
                        {
                            "id": "call_123",
                            "name": "TestModel",
                            "args": {"field": "value"},
                        }
                    ],
                ),
            ],
            "engines": {
                "test_engine": type(
                    "Engine",
                    (),
                    {
                        "tool_routes": {
                            "TestModel": "pydantic_model",
                            "calculator": "langchain_tool",
                        },
                        "structured_output_model": type(
                            "TestModel", (), {"__name__": "TestModel"}
                        ),
                    },
                )()
            },
        },
    )()

    # Execute
    validation_node(state)

    # Check if we're looking at tool routes
    engine = validation_node._get_engine_from_state(state)
    if engine:
        validation_node._get_tool_routes_from_engine(engine)


if __name__ == "__main__":
    test_validation_node_basic()
