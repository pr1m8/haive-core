#!/usr/bin/env python3
"""
Demo: How Tool Routes Work and Are Changeable

This shows you exactly how tool routes are stored, assigned, and can be changed.
"""

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from haive.core.engine.aug_llm import AugLLMConfig
from haive.core.engine.tool import ToolEngine


@tool
def calculator(expression: str) -> str:
    """Calculate mathematical expressions."""
    return f"Result: {eval(expression)}"


class SearchResult(BaseModel):
    """Pydantic model for structured output."""
    query: str = Field(description="The search query")
    results: list[str] = Field(description="Search results")


def demo_tool_routes():
    print("🔧 TOOL ROUTES DEMO")
    print("=" * 50)

    # 1. Create AugLLMConfig with tools
    print("\n1. Creating AugLLMConfig with tools...")
    config = AugLLMConfig(tools=[calculator])

    print(f"   Tools: {len(config.tools)}")
    print(f"   Tool routes: {config.tool_routes}")
    print(f"   Tool metadata: {config.tool_metadata}")

    # 2. Show automatic route assignment
    print("\n2. Automatic route assignment:")
    for tool_name, route in config.tool_routes.items():
        metadata = config.get_tool_metadata(tool_name)
        print(f"   {tool_name} → {route}")
        if metadata:
            print(f"      Metadata: {metadata}")

    # 3. Add structured output model
    print("\n3. Adding structured output model...")
    config.structured_output_model = SearchResult

    print(f"   Updated tool routes: {config.tool_routes}")
    print(f"   New routes added: {[k for k in config.tool_routes.keys() if k != 'calculator']}")

    # 4. Manually change a route
    print("\n4. Manually changing tool route...")
    print(f"   Original calculator route: {config.get_tool_route('calculator')}")

    # Change the route
    config.update_tool_route("calculator", "custom_route")

    print(f"   New calculator route: {config.get_tool_route('calculator')}")
    print(f"   Updated metadata: {config.get_tool_metadata('calculator')}")

    # 5. Add tool with explicit route
    print("\n5. Adding tool with explicit route...")

    @tool
    def text_processor(text: str) -> str:
        """Process text."""
        return f"Processed: {text}"

    # Method 1: Add with automatic route detection
    config.add_tool(text_processor)
    print(f"   text_processor auto route: {config.get_tool_route('text_processor')}")

    # Method 2: Add with explicit route
    config.set_tool_route("text_processor", "special_processor", {"custom": True})
    print(f"   text_processor explicit route: {config.get_tool_route('text_processor')}")

    # 6. Show all routes
    print("\n6. Final tool routes summary:")
    config.debug_tool_routes()

    # 7. Change routes dynamically
    print("\n7. Dynamic route changes...")

    # Get tools by route type
    langchain_tools = config.get_tools_by_route("langchain_tool")
    custom_tools = config.get_tools_by_route("custom_route")
    special_tools = config.get_tools_by_route("special_processor")

    print(f"   LangChain tools: {[t for t in langchain_tools if hasattr(t, 'name')]}")
    print(f"   Custom route tools: {len(custom_tools)} tools")
    print(f"   Special processor tools: {len(special_tools)} tools")

    # 8. Route manipulation methods
    print("\n8. Route manipulation methods available:")
    methods = [
        "set_tool_route(name, route, metadata)",
        "get_tool_route(name)",
        "update_tool_route(name, new_route)",
        "remove_tool_route(name)",
        "get_tools_by_route(route)",
        "clear_tool_routes()",
        "update_tool_routes({name: route, ...})",
        "set_tool_route_for_existing(identifier, route)"
    ]

    for method in methods:
        print(f"   • config.{method}")


def demo_tool_engine_routes():
    print("\n\n🏭 TOOL ENGINE ROUTES DEMO")
    print("=" * 50)

    # ToolEngine also has routing capabilities
    enhanced_tool = ToolEngine.create_state_tool(
        lambda x: f"Enhanced: {x}",
        name="enhanced_processor",
        reads_state=True
    )

    config = AugLLMConfig(tools=[enhanced_tool])
    print(f"Enhanced tool route: {config.tool_routes}")

    # Tool routes are stored in the tool_routes dict
    print(f"Route details: {config.get_tool_metadata('enhanced_processor')}")


if __name__ == "__main__":
    demo_tool_routes()
    demo_tool_engine_routes()
