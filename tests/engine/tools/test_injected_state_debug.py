"""Debug test for InjectedState detection."""
import inspect
from typing import Annotated, get_type_hints

from langchain_core.tools import tool
from langgraph.prebuilt.tool_node import InjectedState

from haive.core.engine.tool import ToolEngine


@tool
def state_aware_tool(
    message: str,
    state: Annotated[dict, InjectedState]
) -> str:
    """Tool that accesses injected state."""
    msg_count = len(state.get("messages", []))
    return f"Processed '{message}' with {msg_count} messages in state"


def test_debug_injected_state():
    """Debug why InjectedState isn't being detected."""
    print(f"\nTool type: {type(state_aware_tool)}")
    print(f"Tool name: {state_aware_tool.name}")

    # Check the actual function
    if hasattr(state_aware_tool, "func"):
        actual_func = state_aware_tool.func
        print(f"\nActual function: {actual_func}")
    else:
        actual_func = state_aware_tool

    # Try to get type hints different ways
    try:
        hints1 = get_type_hints(actual_func)
        print(f"\nType hints (basic): {hints1}")
    except Exception as e:
        print(f"Error getting basic hints: {e}")

    try:
        hints2 = get_type_hints(actual_func, include_extras=True)
        print(f"\nType hints (with extras): {hints2}")
    except Exception as e:
        print(f"Error getting hints with extras: {e}")

    # Check signature
    try:
        sig = inspect.signature(actual_func)
        print(f"\nSignature: {sig}")
        for param_name, param in sig.parameters.items():
            print(f"  {param_name}: {param.annotation}")
    except Exception as e:
        print(f"Error getting signature: {e}")

    # Test with engine
    engine = ToolEngine(tools=[state_aware_tool], enable_analysis=True)
    props = engine.get_tool_properties("state_aware_tool")
    print(f"\nDetected capabilities: {props.capabilities}")
    print(f"Is state tool: {props.is_state_tool}")
    print(f"From state tool: {props.from_state_tool}")

    # Test the analyzer directly
    analyzer = engine._analyzer
    uses_injected = analyzer._uses_injected_state(state_aware_tool)
    print(f"\n_uses_injected_state result: {uses_injected}")


if __name__ == "__main__":
    test_debug_injected_state()
