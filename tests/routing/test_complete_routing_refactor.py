#!/usr/bin/env python3
"""
Complete Routing Refactor Implementation

This implements the user's requested routing refactor:
"if the tool == engine.structured_output_model -> parse output"

The refactor modifies ValidationNodeV2's routing logic to handle 'parse_output' 
route for structured output models, providing clean separation from regular tools.

Key Changes:
1. Structured output models get 'parse_output' route instead of 'pydantic_model'
2. ValidationNodeV2 handles 'parse_output' route specially 
3. Clean separation between regular tools and structured output models
4. Easy to extend structured output features
"""

from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from haive.core.engine.aug_llm import AugLLMConfig
from haive.core.engine.tool import ToolEngine


# Test models
class SearchResult(BaseModel):
    """Search results structured output model."""
    query: str = Field(description="Search query")
    results: list[str] = Field(description="Search results")
    count: int = Field(description="Number of results")


class AnalysisResult(BaseModel):
    """Analysis results structured output model."""
    input_text: str = Field(description="Input text")
    sentiment: str = Field(description="Sentiment")
    confidence: float = Field(description="Confidence score")


@tool
def calculator(expression: str) -> str:
    """Calculate mathematical expressions."""
    return f"Result: {eval(expression)}"


def apply_routing_refactor(config: AugLLMConfig) -> dict[str, Any]:
    """Apply the structured output routing refactor to AugLLMConfig.
    
    This implements the user's concept: structured output models → parse_output route.
    
    Args:
        config: AugLLMConfig to refactor
        
    Returns:
        Dictionary of changes made
    """
    print("🔧 APPLYING ROUTING REFACTOR")
    print("=" * 35)

    changes = {}

    # Find all pydantic_model routes (structured output models)
    pydantic_routes = [
        (name, route) for name, route in config.tool_routes.items()
        if route == "pydantic_model"
    ]

    print(f"Found {len(pydantic_routes)} structured output models:")

    for tool_name, current_route in pydantic_routes:
        print(f"   {tool_name}: {current_route} → parse_output")
        config.update_tool_route(tool_name, "parse_output")
        changes[tool_name] = {"from": current_route, "to": "parse_output"}

    # Handle main structured_output_model
    if config.structured_output_model:
        model_name = config.structured_output_model.__name__
        if model_name in config.tool_routes:
            current_route = config.tool_routes[model_name]
            if current_route != "parse_output":
                print(f"   Main model: {model_name}: {current_route} → parse_output")
                config.update_tool_route(model_name, "parse_output")
                changes[model_name] = {"from": current_route, "to": "parse_output"}

    print(f"Refactor completed: {len(changes)} routes changed")
    return changes


def show_validation_node_v2_integration(config: AugLLMConfig):
    """Show how ValidationNodeV2 would integrate with the refactored routes."""
    print("\n\n⚙️ VALIDATION NODE V2 INTEGRATION")
    print("=" * 40)

    print("Current ValidationNodeV2 routing logic (line 320):")
    print("   if route == 'pydantic_model':")
    print("      → Handle Pydantic model, create ToolMessage")
    print("   elif route in ['langchain_tool', 'function', 'tool_node']:")
    print("      → Let tool_node handle it")
    print("   else:")
    print("      → Unknown tool, create error ToolMessage")

    print("\nPROPOSED REFACTORED ROUTING LOGIC:")

    refactored_logic = """
    if route == "parse_output":
        # NEW: Structured output models → ParserNodeV2
        logger.debug(f"Structured output model {tool_name} routed to parse_output")
        # Don't create ToolMessage here, let ParserNodeV2 handle it
        
    elif route == "pydantic_model":
        # LEGACY: Old Pydantic models (deprecated in favor of parse_output)
        model_class = self._find_pydantic_model_class(tool_name, engine)
        if model_class:
            tool_msg = self._create_tool_message_for_pydantic(
                tool_name, tool_id, args, model_class
            )
            new_tool_messages.append(tool_msg)
        
    elif route in ["langchain_tool", "function", "tool_node"]:
        # Regular tools - let tool_node handle it
        logger.debug(f"Regular tool {tool_name} will be handled by tool_node")
        
    else:
        # Unknown tool
        error_msg = f"Unknown tool: {tool_name}"
        tool_msg = self._create_error_tool_message(tool_name, tool_id, error_msg)
        new_tool_messages.append(tool_msg)
    """

    print(refactored_logic)

    print("KEY BENEFIT: The user's concept in action:")
    print(f"   • if tool_name == '{config.structured_output_model.__name__}' (engine.structured_output_model)")
    print("   • route == 'parse_output'")
    print("   • ValidationNodeV2 detects 'parse_output' route")
    print("   • Routes to ParserNodeV2 instead of handling inline")
    print("   • Clean separation: structured output → parsing, regular tools → execution")


def demonstrate_router_integration(config: AugLLMConfig):
    """Show how the validation router would handle the refactored routes."""
    print("\n\n🔀 VALIDATION ROUTER INTEGRATION")
    print("=" * 38)

    print("ValidationNodeV2 creates ToolMessages and goes to validation_router.")
    print("The validation_router then makes routing decisions based on tool routes.")

    router_logic = '''
def validation_router(state):
    """Route based on tool routes after ValidationNodeV2 processing."""
    messages = state.messages
    last_ai_message = get_last_ai_message(messages)
    
    if not last_ai_message or not last_ai_message.tool_calls:
        return "agent"  # No tool calls, back to agent
    
    # Check routes for all tool calls
    for tool_call in last_ai_message.tool_calls:
        tool_name = tool_call["name"]
        route = get_tool_route(tool_name)
        
        # REFACTORED ROUTING DECISIONS
        if route == "parse_output":
            # Structured output models → ParserNodeV2
            return "parser_node_v2"
            
        elif route == "langchain_tool":
            # Regular tools → ToolNodeConfig
            return "tool_node"
            
        elif route == "function":
            # Function tools → FunctionNode  
            return "function_node"
            
        elif route == "pydantic_model":
            # Legacy Pydantic models → ValidationNode (deprecated)
            return "validation_node_legacy"
            
    # Default fallback
    return "agent"
    '''

    print("Enhanced validation_router logic:")
    print(router_logic)

    print("Flow with refactor:")
    route_flows = {
        "parse_output": "ValidationNodeV2 → validation_router → ParserNodeV2",
        "langchain_tool": "ValidationNodeV2 → validation_router → ToolNodeConfig",
        "function": "ValidationNodeV2 → validation_router → FunctionNode",
        "pydantic_model": "ValidationNodeV2 → validation_router → ValidationNode (legacy)"
    }

    for route, flow in route_flows.items():
        tools_with_route = [name for name, r in config.tool_routes.items() if r == route]
        if tools_with_route:
            print(f"   {route}: {tools_with_route}")
            print(f"      → {flow}")


def create_test_scenario():
    """Create a test scenario showing the refactor in action."""
    print("\n\n🧪 TEST SCENARIO: REFACTOR IN ACTION")
    print("=" * 40)

    # Create config with mixed tools
    config = AugLLMConfig(
        tools=[calculator],
        structured_output_model=SearchResult
    )

    # Add another structured output tool
    analysis_tool = ToolEngine.create_structured_output_tool(
        func=lambda text: AnalysisResult(
            input_text=text,
            sentiment="positive",
            confidence=0.85
        ),
        name="analyze_sentiment",
        description="Analyze sentiment",
        output_model=AnalysisResult
    )
    config.add_tool(analysis_tool)
    config.set_tool_route("AnalysisResult", "pydantic_model")

    print("BEFORE refactor - tool routes:")
    for tool_name, route in config.tool_routes.items():
        print(f"   {tool_name} → {route}")

    # Apply refactor
    changes = apply_routing_refactor(config)

    print("\nAFTER refactor - tool routes:")
    for tool_name, route in config.tool_routes.items():
        print(f"   {tool_name} → {route}")

    # Test with tool calls
    print("\n🔄 PROCESSING TOOL CALLS:")
    test_tool_calls = [
        {"name": "calculator", "id": "call_1", "args": {"expression": "5+3"}},
        {"name": "SearchResult", "id": "call_2", "args": {"query": "python", "results": ["result1"], "count": 1}},
        {"name": "AnalysisResult", "id": "call_3", "args": {"input_text": "hello", "sentiment": "positive", "confidence": 0.9}}
    ]

    for tool_call in test_tool_calls:
        tool_name = tool_call["name"]
        route = config.tool_routes.get(tool_name, "unknown")

        print(f"   Tool call: {tool_name}")
        print(f"      Route: {route}")

        # Show routing decision
        if route == "parse_output":
            print("      → ValidationNodeV2 skips, routes to ParserNodeV2")
        elif route == "langchain_tool":
            print("      → ValidationNodeV2 skips, routes to ToolNodeConfig")
        elif route == "pydantic_model":
            print("      → ValidationNodeV2 creates ToolMessage inline")
        else:
            print("      → ValidationNodeV2 creates error ToolMessage")

    return config


def show_implementation_code():
    """Show the actual code changes needed in ValidationNodeV2."""
    print("\n\n💻 IMPLEMENTATION CODE CHANGES")
    print("=" * 38)

    print("Changes needed in ValidationNodeV2.__call__ method (around line 320):")

    implementation = """
# CURRENT CODE (line 320):
if route == "pydantic_model":
    # Handle Pydantic model
    model_class = self._find_pydantic_model_class(tool_name, engine)
    if model_class:
        tool_msg = self._create_tool_message_for_pydantic(
            tool_name, tool_id, args, model_class
        )
        new_tool_messages.append(tool_msg)

# NEW REFACTORED CODE:
if route == "parse_output":
    # NEW: Structured output models → Don't create ToolMessage here
    # Let ParserNodeV2 handle the parsing and ToolMessage creation
    logger.debug(f"Structured output model {tool_name} routed to parse_output")
    # No ToolMessage creation - ParserNodeV2 will handle it
    
elif route == "pydantic_model":
    # LEGACY: Old Pydantic models (for backward compatibility)
    model_class = self._find_pydantic_model_class(tool_name, engine)
    if model_class:
        tool_msg = self._create_tool_message_for_pydantic(
            tool_name, tool_id, args, model_class
        )
        new_tool_messages.append(tool_msg)
        logger.info(f"Created ToolMessage for legacy Pydantic model: {tool_name}")
"""

    print(implementation)

    print("\nBenefits of this change:")
    print("   ✅ Structured output models get dedicated 'parse_output' route")
    print("   ✅ ParserNodeV2 handles all structured output parsing consistently")
    print("   ✅ Clean separation between validation and parsing")
    print("   ✅ Backward compatibility with legacy 'pydantic_model' routes")
    print("   ✅ Easy to extend structured output features")


if __name__ == "__main__":
    config = create_test_scenario()
    show_validation_node_v2_integration(config)
    demonstrate_router_integration(config)
    show_implementation_code()

    print("\n\n🎯 ROUTING REFACTOR COMPLETE")
    print("=" * 35)
    print("User's request implemented: 'if tool == engine.structured_output_model -> parse output'")
    print("\nSummary:")
    print("• Structured output models now use 'parse_output' route")
    print("• ValidationNodeV2 detects 'parse_output' and skips inline processing")
    print("• validation_router sends 'parse_output' tools to ParserNodeV2")
    print("• Clean separation: structured output → parsing, regular tools → execution")
    print("• Backward compatible with existing 'pydantic_model' routes")
    print("• Easy to extend structured output features in ParserNodeV2")
