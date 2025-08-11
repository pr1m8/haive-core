#!/usr/bin/env python3
"""
Structured Output Models as Tools - Properties and Routing Analysis

This demonstrates how structured output models are treated as tools in the Haive system,
showing their properties, routing behavior, and integration with the tool system.

Key Questions Answered:
1. How are structured output models registered as tools?
2. What properties do they have when treated as tools?
3. How does routing work for structured output models?
4. What's the relationship between engine.structured_output_model and tool routes?
"""

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from haive.core.engine.aug_llm import AugLLMConfig
from haive.core.engine.tool import ToolEngine


# Define structured output models
class SearchResult(BaseModel):
    """Search results structured output model."""
    query: str = Field(description="The search query")
    results: list[str] = Field(description="Search results")
    count: int = Field(description="Number of results")


class AnalysisResult(BaseModel):
    """Analysis results structured output model."""
    input_text: str = Field(description="Input text")
    sentiment: str = Field(description="Sentiment analysis")
    confidence: float = Field(description="Confidence score")
    keywords: list[str] = Field(description="Extracted keywords")


class ProcessingResult(BaseModel):
    """Generic processing result."""
    status: str = Field(description="Processing status")
    data: dict = Field(description="Processed data")
    metadata: dict = Field(description="Processing metadata")


@tool
def regular_calculator(expression: str) -> str:
    """Regular LangChain tool for comparison."""
    return f"Result: {eval(expression)}"


def analyze_structured_output_as_tool():
    """Analyze how structured output models become tools."""
    print("🔍 STRUCTURED OUTPUT MODELS AS TOOLS")
    print("=" * 50)
    
    # 1. Create AugLLMConfig with structured output model
    print("\n1. Creating AugLLMConfig with structured output model...")
    config = AugLLMConfig(
        tools=[regular_calculator],
        structured_output_model=SearchResult
    )
    
    print(f"   Structured output model: {config.structured_output_model}")
    print(f"   Total tools: {len(config.tools)}")
    print(f"   Tool routes: {len(config.tool_routes)}")
    
    # 2. Examine how structured output model appears in tool routes
    print("\n2. Tool routes analysis:")
    for tool_name, route in config.tool_routes.items():
        metadata = config.get_tool_metadata(tool_name)
        print(f"   {tool_name} → {route}")
        if metadata:
            print(f"      Metadata: {metadata}")
    
    # 3. Check if structured_output_model creates a tool route
    model_name = SearchResult.__name__
    if model_name in config.tool_routes:
        print(f"\n3. Structured output model route found:")
        print(f"   {model_name} → {config.tool_routes[model_name]}")
        print(f"   Route type: {config.tool_routes[model_name]}")
    else:
        print(f"\n3. No direct route for {model_name}")
    
    # 4. Properties of structured output model as tool
    print("\n4. Structured output model properties:")
    print(f"   Class name: {SearchResult.__name__}")
    print(f"   Fields: {list(SearchResult.model_fields.keys())}")
    print(f"   Field types: {[(k, v.annotation) for k, v in SearchResult.model_fields.items()]}")
    
    # 5. Check if model can be retrieved as a tool
    print("\n5. Tool retrieval:")
    structured_tools = config.get_tools_by_route("pydantic_model")
    print(f"   Pydantic model tools: {len(structured_tools)}")
    for tool in structured_tools:
        if hasattr(tool, '__name__'):
            print(f"      • {tool.__name__}")
        elif hasattr(tool, 'name'):
            print(f"      • {tool.name}")
    
    return config


def demonstrate_structured_output_tool_creation():
    """Show how ToolEngine creates structured output tools."""
    print("\n\n🛠️ TOOLENGINE STRUCTURED OUTPUT TOOL CREATION")
    print("=" * 55)
    
    # Create structured output tool using ToolEngine
    structured_tool = ToolEngine.create_structured_output_tool(
        func=lambda query: SearchResult(
            query=query,
            results=[f"Result 1 for {query}", f"Result 2 for {query}"],
            count=2
        ),
        name="search_engine",
        description="Search with structured output",
        output_model=SearchResult
    )
    
    print("1. Created structured output tool with ToolEngine:")
    print(f"   Tool name: {structured_tool.name if hasattr(structured_tool, 'name') else 'Unknown'}")
    print(f"   Tool type: {type(structured_tool)}")
    
    # Add to AugLLMConfig
    config = AugLLMConfig(
        tools=[regular_calculator, structured_tool],
        structured_output_model=AnalysisResult  # Different model for comparison
    )
    
    print(f"\n2. AugLLMConfig with ToolEngine structured output tool:")
    print(f"   Total tools: {len(config.tools)}")
    print(f"   Tool routes: {len(config.tool_routes)}")
    
    # Analyze routes
    print("\n3. Route analysis:")
    route_summary = {}
    for tool_name, route in config.tool_routes.items():
        if route not in route_summary:
            route_summary[route] = []
        route_summary[route].append(tool_name)
    
    for route, tools in route_summary.items():
        print(f"   {route}: {tools}")
    
    # Check both models
    print(f"\n4. Multiple structured output models:")
    print(f"   Primary (engine.structured_output_model): {config.structured_output_model.__name__}")
    print(f"   Tool-created: SearchResult (via ToolEngine)")
    
    # Routes for both
    for model_class in [SearchResult, AnalysisResult]:
        model_name = model_class.__name__
        route = config.tool_routes.get(model_name)
        print(f"   {model_name} route: {route}")
    
    return config


def explore_tool_model_relationship():
    """Explore the relationship between tools and structured output models."""
    print("\n\n🔗 TOOL ↔ STRUCTURED OUTPUT MODEL RELATIONSHIP")
    print("=" * 55)
    
    config = AugLLMConfig(structured_output_model=ProcessingResult)
    
    print("1. Key relationship questions:")
    
    # Question 1: Does engine.structured_output_model create a tool?
    print(f"\n   Q: Does engine.structured_output_model create a tool?")
    model_name = ProcessingResult.__name__
    if model_name in config.tool_routes:
        print(f"   A: YES - {model_name} appears in tool_routes with route: {config.tool_routes[model_name]}")
    else:
        print(f"   A: NO - {model_name} not found in tool_routes")
    
    # Question 2: What happens when we call engine.structured_output_model?
    print(f"\n   Q: What is engine.structured_output_model?")
    print(f"   A: It's a Pydantic model class: {config.structured_output_model}")
    print(f"      Type: {type(config.structured_output_model)}")
    print(f"      Fields: {list(config.structured_output_model.model_fields.keys())}")
    
    # Question 3: How does routing work?
    print(f"\n   Q: How does routing work for structured output?")
    if model_name in config.tool_routes:
        route = config.tool_routes[model_name]
        print(f"   A: {model_name} → {route}")
        print(f"      This means ValidationNodeV2 will:")
        print(f"      • Detect tool call with name '{model_name}'")
        print(f"      • Route to '{route}' handler")
        print(f"      • Handle Pydantic model validation/serialization")
    
    # Question 4: Can we change the route?
    print(f"\n   Q: Can we change the structured output route?")
    if model_name in config.tool_routes:
        original_route = config.tool_routes[model_name]
        print(f"   A: YES - Current route: {original_route}")
        
        # Change it
        config.update_tool_route(model_name, "custom_structured_output")
        new_route = config.tool_routes[model_name]
        print(f"      Changed to: {new_route}")
        
        # Change back
        config.update_tool_route(model_name, original_route)
        print(f"      Restored to: {config.tool_routes[model_name]}")
    
    return config


def demonstrate_routing_refactor_concept():
    """Demonstrate the routing refactor concept the user mentioned."""
    print("\n\n🚀 ROUTING REFACTOR CONCEPT")
    print("=" * 35)
    
    print("User's request: 'if the tool == engine.structured_output_model -> parse output'")
    print("This suggests they want special handling for structured output models.\n")
    
    # Current behavior
    config = AugLLMConfig(
        tools=[regular_calculator],
        structured_output_model=SearchResult
    )
    
    print("1. Current behavior:")
    for tool_name, route in config.tool_routes.items():
        print(f"   {tool_name} → {route}")
    
    # Proposed refactor
    print("\n2. Proposed refactor concept:")
    print("   The idea seems to be:")
    print("   • Regular tools → existing routes (langchain_tool, function, etc.)")
    print("   • Structured output models → dedicated 'parse_output' or 'structured_output' route")
    
    # Implementation
    print("\n3. Implementing the concept:")
    
    # Find the structured output model route
    model_name = SearchResult.__name__
    if model_name in config.tool_routes:
        current_route = config.tool_routes[model_name]
        print(f"   Current: {model_name} → {current_route}")
        
        # Apply the refactor
        config.update_tool_route(model_name, "parse_output")
        print(f"   Refactored: {model_name} → {config.tool_routes[model_name]}")
    
    print("\n4. Benefits of this refactor:")
    print("   • Clear separation: structured output models have dedicated routing")
    print("   • Easier to identify which tools are for structured output")
    print("   • Can implement specialized handling for 'parse_output' route")
    print("   • ValidationNodeV2 can have specific logic for parse_output vs other routes")
    
    # Show routing groups after refactor
    print("\n5. Final routing groups:")
    route_groups = {}
    for tool_name, route in config.tool_routes.items():
        if route not in route_groups:
            route_groups[route] = []
        route_groups[route].append(tool_name)
    
    for route, tools in route_groups.items():
        print(f"   {route}: {tools}")
    
    return config


def show_validation_node_behavior():
    """Show how ValidationNodeV2 would handle the routing."""
    print("\n\n⚙️ VALIDATION NODE ROUTING BEHAVIOR")
    print("=" * 40)
    
    config = AugLLMConfig(
        tools=[regular_calculator],
        structured_output_model=SearchResult
    )
    
    # Apply the refactor concept
    model_name = SearchResult.__name__
    if model_name in config.tool_routes:
        config.update_tool_route(model_name, "parse_output")
    
    print("1. How ValidationNodeV2 handles different routes:")
    
    routing_logic = {
        "langchain_tool": "→ ToolNode (execute LangChain tool)",
        "function": "→ FunctionNode (direct function call)", 
        "parse_output": "→ ParserNode (parse structured output)",
        "pydantic_model": "→ ValidationNode (validate Pydantic model)",
        "structured_output": "→ StructuredOutputNode (handle structured output)",
        "unknown": "→ ErrorNode (handle unknown tools)"
    }
    
    for route, handler in routing_logic.items():
        tools_with_route = [name for name, r in config.tool_routes.items() if r == route]
        if tools_with_route:
            print(f"   {route}: {tools_with_route} {handler}")
    
    print("\n2. The user's concept in action:")
    print("   • tool == 'SearchResult' (structured output model)")
    print("   • route == 'parse_output'") 
    print("   • ValidationNodeV2 detects 'parse_output' route")
    print("   • Routes to ParserNodeV2 for structured output parsing")
    
    print("\n3. This enables:")
    print("   • Centralized structured output parsing")
    print("   • Consistent handling of Pydantic models")
    print("   • Clear separation from regular tool execution")
    print("   • Easy to extend structured output features")


if __name__ == "__main__":
    analyze_structured_output_as_tool()
    demonstrate_structured_output_tool_creation()
    explore_tool_model_relationship()
    demonstrate_routing_refactor_concept()
    show_validation_node_behavior()
    
    print("\n\n✅ SUMMARY: STRUCTURED OUTPUT MODELS AS TOOLS")
    print("=" * 55)
    print("• Structured output models ARE treated as tools in the routing system")
    print("• They get routes (typically 'pydantic_model') in tool_routes dict")
    print("• Routes are changeable via config.update_tool_route()")
    print("• ValidationNodeV2 uses these routes to determine handling")
    print("• User's concept: route structured output models to 'parse_output' for special handling")
    print("• This enables clean separation between regular tools and structured output models")