#!/usr/bin/env python3
"""
Refactored Routing: Structured Output Models → Dedicated Route

This shows a refactored routing system where:
1. Structured output models go to a dedicated route
2. Regular tools keep their automatic routes
3. Clean separation of concerns
"""

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from haive.core.engine.aug_llm import AugLLMConfig


@tool
def calculator(expression: str) -> str:
    """Calculate mathematical expressions."""
    return f"Result: {eval(expression)}"


@tool
def text_processor(text: str) -> str:
    """Process text."""
    return f"Processed: {text}"


class SearchResult(BaseModel):
    """Structured output model."""
    query: str = Field(description="The search query")
    results: list[str] = Field(description="Search results")
    count: int = Field(description="Number of results")


class AnalysisResult(BaseModel):
    """Another structured output model."""
    input_text: str = Field(description="Input text")
    sentiment: str = Field(description="Sentiment analysis")
    confidence: float = Field(description="Confidence score")


def demo_current_behavior():
    """Show current routing behavior."""
    print("🔄 CURRENT ROUTING BEHAVIOR")
    print("=" * 40)
    
    config = AugLLMConfig(
        tools=[calculator, text_processor],
        structured_output_model=SearchResult
    )
    
    print("Current routes:")
    for tool_name, route in config.tool_routes.items():
        print(f"   {tool_name} → {route}")
    
    print(f"\nStructured output model: {config.structured_output_model}")
    
    return config


def demo_proposed_refactor():
    """Show proposed refactored routing."""
    print("\n\n🚀 PROPOSED REFACTORED ROUTING")
    print("=" * 40)
    
    config = AugLLMConfig(
        tools=[calculator, text_processor],
        structured_output_model=SearchResult
    )
    
    # The refactor: Move structured output models to dedicated route
    print("1. Moving structured output to dedicated route...")
    
    # Find any pydantic_model routes and change them to structured_output
    structured_models = []
    for tool_name, route in list(config.tool_routes.items()):
        if route == "pydantic_model":
            # This is a structured output model
            config.update_tool_route(tool_name, "structured_output")
            structured_models.append(tool_name)
            print(f"   Moved {tool_name}: pydantic_model → structured_output")
    
    # Also handle the structured_output_model field
    if config.structured_output_model:
        model_name = config.structured_output_model.__name__
        config.set_tool_route(model_name, "structured_output", {
            "is_main_structured_output": True,
            "model_class": config.structured_output_model
        })
        print(f"   Set main structured output: {model_name} → structured_output")
    
    print("\n2. Final routing after refactor:")
    route_groups = {}
    for tool_name, route in config.tool_routes.items():
        if route not in route_groups:
            route_groups[route] = []
        route_groups[route].append(tool_name)
    
    for route, tools in route_groups.items():
        print(f"   {route}:")
        for tool in tools:
            metadata = config.get_tool_metadata(tool)
            if metadata and metadata.get("is_main_structured_output"):
                print(f"      • {tool} (main structured output)")
            else:
                print(f"      • {tool}")
    
    return config


def demo_enhanced_structured_routing():
    """Show enhanced routing with multiple structured outputs."""
    print("\n\n⚡ ENHANCED STRUCTURED OUTPUT ROUTING")
    print("=" * 45)
    
    config = AugLLMConfig(
        tools=[calculator, text_processor]
    )
    
    # Add multiple structured output models
    structured_models = [SearchResult, AnalysisResult]
    
    print("1. Adding multiple structured output models...")
    for model in structured_models:
        config.set_tool_route(model.__name__, "structured_output", {
            "model_class": model,
            "fields": list(model.model_fields.keys()),
            "is_structured_output": True
        })
        print(f"   Added {model.__name__} → structured_output")
    
    # Set primary structured output
    config.structured_output_model = SearchResult
    config.update_tool_route("SearchResult", "structured_output", {
        "model_class": SearchResult,
        "is_primary_structured_output": True,
        "fields": list(SearchResult.model_fields.keys())
    })
    
    print("\n2. Structured output routing summary:")
    structured_tools = config.get_tools_by_route("structured_output")
    regular_tools = config.get_tools_by_route("langchain_tool") 
    
    print(f"   Structured output models: {len(structured_models)}")
    for tool_name, route in config.tool_routes.items():
        if route == "structured_output":
            metadata = config.get_tool_metadata(tool_name)
            is_primary = metadata.get("is_primary_structured_output", False)
            primary_str = " (PRIMARY)" if is_primary else ""
            fields = metadata.get("fields", [])
            print(f"      • {tool_name}{primary_str} - {len(fields)} fields")
    
    print(f"   Regular tools: {len(config.get_tools_by_route('langchain_tool'))}")
    for tool_name, route in config.tool_routes.items():
        if route == "langchain_tool":
            print(f"      • {tool_name}")
    
    return config


def demo_routing_handlers():
    """Show how different routes would be handled."""
    print("\n\n🎯 ROUTING HANDLERS")
    print("=" * 25)
    
    config = demo_enhanced_structured_routing()
    
    print("\n3. How different routes would be handled:")
    
    handlers = {
        "structured_output": "StructuredOutputNode - handles Pydantic models, validation, serialization",
        "langchain_tool": "ToolNode - executes LangChain tools normally", 
        "function": "FunctionNode - direct function calls",
        "unknown": "ErrorNode - handles unknown tool types"
    }
    
    for route, handler in handlers.items():
        tools_with_route = config.get_tools_by_route(route)
        count = len([t for t in config.tool_routes.values() if t == route])
        if count > 0:
            print(f"   {route} ({count} tools) → {handler}")
    
    print("\n4. Structured output processing flow:")
    print("   Input → StructuredOutputNode → Validate → Serialize → ToolMessage")
    print("   Benefits:")
    print("      • Centralized Pydantic validation")
    print("      • Consistent structured output handling") 
    print("      • Easier to add new structured output features")
    print("      • Clean separation from regular tool execution")


def create_refactored_config_helper():
    """Helper function to create configs with refactored routing."""
    print("\n\n🛠️ REFACTORED CONFIG HELPER")
    print("=" * 35)
    
    def create_structured_output_config(tools, structured_models=None, primary_model=None):
        """Create AugLLMConfig with refactored structured output routing."""
        config = AugLLMConfig(tools=tools)
        
        # Add structured output models
        if structured_models:
            for model in structured_models:
                is_primary = model == primary_model
                config.set_tool_route(model.__name__, "structured_output", {
                    "model_class": model,
                    "fields": list(model.model_fields.keys()),
                    "is_structured_output": True,
                    "is_primary": is_primary
                })
        
        # Set primary structured output
        if primary_model:
            config.structured_output_model = primary_model
            
        return config
    
    # Example usage
    print("Example usage:")
    config = create_structured_output_config(
        tools=[calculator, text_processor],
        structured_models=[SearchResult, AnalysisResult], 
        primary_model=SearchResult
    )
    
    print("   Routes created:")
    for tool_name, route in config.tool_routes.items():
        metadata = config.get_tool_metadata(tool_name)
        if route == "structured_output":
            is_primary = metadata.get("is_primary", False)
            primary_str = " (PRIMARY)" if is_primary else ""
            print(f"      {tool_name} → {route}{primary_str}")
        else:
            print(f"      {tool_name} → {route}")
    
    return create_structured_output_config


if __name__ == "__main__":
    demo_current_behavior()
    demo_proposed_refactor() 
    demo_enhanced_structured_routing()
    demo_routing_handlers()
    create_refactored_config_helper()
    
    print("\n\n✅ REFACTOR SUMMARY:")
    print("   • Structured output models → 'structured_output' route")
    print("   • Regular tools → keep existing routes")  
    print("   • Clean separation of concerns")
    print("   • Easier to extend structured output features")