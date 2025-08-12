#!/usr/bin/env python3
"""
Routing Refactor Implementation - Structured Output Models → Parse Output Route

This implements the user's concept: "if the tool == engine.structured_output_model -> parse output"

The refactor creates a clear routing distinction:
1. Regular tools → existing routes (langchain_tool, function, etc.)
2. Structured output models → dedicated "parse_output" route
3. ValidationNodeV2 can handle "parse_output" route specially

This provides clean separation and makes it easy to extend structured output features.
"""


from langchain_core.tools import tool
from pydantic import BaseModel, Field

from haive.core.engine.aug_llm import AugLLMConfig
from haive.core.engine.tool import ToolEngine


# Test models and tools
class SearchResult(BaseModel):
    """Search results model."""
    query: str = Field(description="Search query")
    results: list[str] = Field(description="Search results")
    count: int = Field(description="Number of results")


class AnalysisResult(BaseModel):
    """Analysis results model."""
    input_text: str = Field(description="Input text")
    sentiment: str = Field(description="Sentiment")
    confidence: float = Field(description="Confidence score")


@tool
def calculator(expression: str) -> str:
    """Calculate mathematical expressions."""
    return f"Result: {eval(expression)}"


@tool
def text_processor(text: str) -> str:
    """Process text."""
    return f"Processed: {text}"


class StructuredOutputRoutingRefactor:
    """Implements the structured output routing refactor."""

    def __init__(self, config: AugLLMConfig):
        self.config = config
        self.original_routes = dict(config.tool_routes)

    def apply_refactor(self) -> dict[str, str]:
        """Apply the routing refactor: structured output models → parse_output route."""
        print("🔄 APPLYING ROUTING REFACTOR")
        print("=" * 35)

        changes_made = {}

        # Find all pydantic_model routes (these are structured output models)
        pydantic_routes = [
            (name, route) for name, route in self.config.tool_routes.items()
            if route == "pydantic_model"
        ]

        print(f"Found {len(pydantic_routes)} structured output models to refactor:")

        for tool_name, current_route in pydantic_routes:
            print(f"   {tool_name}: {current_route} → parse_output")

            # Apply the refactor
            self.config.update_tool_route(tool_name, "parse_output")
            changes_made[tool_name] = {"from": current_route, "to": "parse_output"}

        # Also handle the main structured_output_model
        if self.config.structured_output_model:
            model_name = self.config.structured_output_model.__name__
            if model_name in self.config.tool_routes:
                current_route = self.config.tool_routes[model_name]
                if current_route != "parse_output":
                    print(f"   Main structured output: {model_name}: {current_route} → parse_output")
                    self.config.update_tool_route(model_name, "parse_output")
                    changes_made[model_name] = {"from": current_route, "to": "parse_output"}

        print(f"\nRefactor applied: {len(changes_made)} routes changed")
        return changes_made

    def show_routing_groups(self):
        """Show tools grouped by their routes after refactor."""
        print("\n📊 ROUTING GROUPS AFTER REFACTOR")
        print("=" * 38)

        route_groups = {}
        for tool_name, route in self.config.tool_routes.items():
            if route not in route_groups:
                route_groups[route] = []
            route_groups[route].append(tool_name)

        for route, tools in route_groups.items():
            print(f"   {route}:")
            for tool in tools:
                # Check if this is the main structured output model
                is_main = (
                    self.config.structured_output_model and
                    tool == self.config.structured_output_model.__name__
                )
                main_marker = " (MAIN)" if is_main else ""
                print(f"      • {tool}{main_marker}")

    def demonstrate_validation_node_routing(self):
        """Show how ValidationNodeV2 would handle the refactored routes."""
        print("\n⚙️ VALIDATION NODE ROUTING WITH REFACTOR")
        print("=" * 45)

        # Routing logic for ValidationNodeV2
        route_handlers = {
            "langchain_tool": "ToolNodeConfig → Execute LangChain tool",
            "function": "FunctionNode → Direct function call",
            "parse_output": "ParserNodeConfigV2 → Parse structured output",
            "pydantic_model": "ValidationNode → Validate Pydantic model",
            "structured_output": "StructuredOutputNode → Handle structured output"
        }

        print("ValidationNodeV2 routing decision logic:")
        for route, handler in route_handlers.items():
            tools_with_route = [
                name for name, r in self.config.tool_routes.items() if r == route
            ]
            if tools_with_route:
                print(f"   if route == '{route}':")
                print(f"      → {handler}")
                print(f"      Tools: {tools_with_route}")

        print("\n🎯 KEY BENEFIT:")
        print("   • Clear separation: structured output models go to parse_output")
        print("   • Regular tools continue with existing routing")
        print("   • ParserNodeConfigV2 handles all structured output parsing")
        print("   • Easy to add new structured output features")


def demonstrate_before_after_comparison():
    """Show before/after comparison of the routing refactor."""
    print("🔄 BEFORE/AFTER ROUTING COMPARISON")
    print("=" * 40)

    # Create config with mixed tools
    config = AugLLMConfig(
        tools=[calculator, text_processor],
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
        description="Analyze sentiment of text",
        output_model=AnalysisResult
    )
    config.add_tool(analysis_tool)

    # Manually add AnalysisResult as pydantic_model (simulating mixed scenario)
    config.set_tool_route("AnalysisResult", "pydantic_model")

    print("BEFORE refactor:")
    for tool_name, route in config.tool_routes.items():
        print(f"   {tool_name} → {route}")

    # Apply refactor
    refactor = StructuredOutputRoutingRefactor(config)
    changes = refactor.apply_refactor()

    print("\nAFTER refactor:")
    for tool_name, route in config.tool_routes.items():
        print(f"   {tool_name} → {route}")

    print(f"\nChanges made: {len(changes)}")
    for tool_name, change in changes.items():
        print(f"   {tool_name}: {change['from']} → {change['to']}")

    return refactor


def show_implementation_in_validation_node():
    """Show how this would be implemented in ValidationNodeV2."""
    print("\n\n💻 IMPLEMENTATION IN VALIDATION NODE V2")
    print("=" * 45)

    print("The ValidationNodeV2 routing logic would look like:")

    routing_code = '''
def _determine_route_and_next_node(self, tool_call) -> tuple[str, str]:
    """Determine routing based on tool call."""
    tool_name = tool_call["name"]
    route = self.get_tool_route(tool_name)
    
    # REFACTORED ROUTING LOGIC
    if route == "parse_output":
        # Structured output models → ParserNodeV2
        return route, "parser_node_v2"
    elif route == "langchain_tool":
        # Regular tools → ToolNodeConfig
        return route, "tool_node" 
    elif route == "function":
        # Function tools → FunctionNode
        return route, "function_node"
    elif route == "pydantic_model":
        # Legacy Pydantic models → ValidationNode (deprecated)
        return route, "validation_node"
    else:
        # Unknown tools → Error handling
        return "unknown", "error_node"
'''

    print(routing_code)

    print("\nKey benefits of this implementation:")
    print("   • if tool_name == engine.structured_output_model.__name__:")
    print("       → route will be 'parse_output'")
    print("       → ValidationNodeV2 routes to ParserNodeV2")
    print("   • Clean separation between structured output and regular tools")
    print("   • ParserNodeV2 handles all structured output parsing consistently")
    print("   • Easy to extend with new structured output features")


def create_helper_functions():
    """Create helper functions for applying this refactor pattern."""
    print("\n\n🛠️ HELPER FUNCTIONS FOR ROUTING REFACTOR")
    print("=" * 45)

    helper_code = '''
def apply_structured_output_routing_refactor(config: AugLLMConfig) -> Dict[str, Any]:
    """Apply the structured output routing refactor to AugLLMConfig.
    
    Changes all pydantic_model routes to parse_output for structured output models.
    
    Args:
        config: AugLLMConfig to refactor
        
    Returns:
        Dictionary of changes made
    """
    changes = {}
    
    # Find all structured output models (pydantic_model routes)
    for tool_name, route in list(config.tool_routes.items()):
        if route == "pydantic_model":
            config.update_tool_route(tool_name, "parse_output")
            changes[tool_name] = {"from": route, "to": "parse_output"}
    
    return changes


def is_structured_output_tool(tool_name: str, config: AugLLMConfig) -> bool:
    """Check if a tool is a structured output model.
    
    Args:
        tool_name: Name of the tool to check
        config: AugLLMConfig to check in
        
    Returns:
        True if tool is structured output model
    """
    route = config.get_tool_route(tool_name)
    return route == "parse_output"


def get_structured_output_tools(config: AugLLMConfig) -> List[str]:
    """Get all structured output model tool names.
    
    Args:
        config: AugLLMConfig to search
        
    Returns:
        List of structured output tool names
    """
    return [
        name for name, route in config.tool_routes.items() 
        if route == "parse_output"
    ]
'''

    print("Helper functions to implement:")
    print(helper_code)


if __name__ == "__main__":
    # Run the demonstration
    refactor = demonstrate_before_after_comparison()
    refactor.show_routing_groups()
    refactor.demonstrate_validation_node_routing()
    show_implementation_in_validation_node()
    create_helper_functions()

    print("\n\n✅ ROUTING REFACTOR SUMMARY")
    print("=" * 35)
    print("The user's concept 'if tool == engine.structured_output_model -> parse output' means:")
    print("• Structured output models get 'parse_output' route instead of 'pydantic_model'")
    print("• ValidationNodeV2 detects 'parse_output' and routes to ParserNodeV2")
    print("• Clean separation between regular tools and structured output models")
    print("• Enables specialized handling and easier feature extension")
    print("• Regular tools continue with existing routing (langchain_tool, function, etc.)")
