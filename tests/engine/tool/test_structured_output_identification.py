#!/usr/bin/env python3
"""
Structured Output Model Identification - Using ToolEngine Capabilities

After reviewing the ToolEngine capabilities system, this shows how to properly
identify which tools are structured output models vs regular Pydantic models.

Key Insight: We have a comprehensive capability system with:
- ToolCapability.STRUCTURED_OUTPUT
- Tool metadata: __tool_capabilities__
- ToolProperties.has_structured_output()
- Enhanced naming utilities for complex types

The routing refactor should use these capabilities, not just route names.
"""

from langchain_core.tools import tool, StructuredTool
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from haive.core.engine.aug_llm import AugLLMConfig
from haive.core.engine.tool import ToolEngine
from haive.core.engine.tool.types import ToolCapability, ToolProperties


# Test models
class RegularValidationModel(BaseModel):
    """A regular Pydantic model for validation (NOT structured output)."""
    name: str = Field(description="User name")
    age: int = Field(ge=0, le=150, description="User age")


class StructuredOutputModel(BaseModel):
    """A structured output model (IS structured output).""" 
    query: str = Field(description="Search query")
    results: List[str] = Field(description="Search results")
    metadata: Dict[str, Any] = Field(description="Additional metadata")


@tool
def regular_calculator(expression: str) -> str:
    """Regular LangChain tool (no structured output)."""
    return f"Result: {eval(expression)}"


def analyze_tool_capabilities():
    """Analyze how tools get capabilities assigned."""
    print("🔧 TOOL CAPABILITIES ANALYSIS")
    print("=" * 35)
    
    # 1. Regular LangChain tool
    regular = regular_calculator
    print(f"1. Regular tool: {regular.name}")
    print(f"   Capabilities: {getattr(regular, '__tool_capabilities__', 'None')}")
    
    # 2. ToolEngine structured output tool
    structured_tool = ToolEngine.create_structured_output_tool(
        func=lambda query: StructuredOutputModel(
            query=query,
            results=[f"Result 1", f"Result 2"], 
            metadata={"count": 2}
        ),
        name="search_structured",
        description="Search with structured output",
        output_model=StructuredOutputModel
    )
    
    print(f"\n2. ToolEngine structured output tool: {structured_tool.name}")
    print(f"   Capabilities: {getattr(structured_tool, '__tool_capabilities__', 'None')}")
    print(f"   Has STRUCTURED_OUTPUT: {ToolCapability.STRUCTURED_OUTPUT in getattr(structured_tool, '__tool_capabilities__', set())}")
    
    # 3. Enhanced tool with structured output
    enhanced_tool = ToolEngine.augment_tool(
        regular_calculator,
        structured_output_model=StructuredOutputModel,
        name="enhanced_calculator"
    )
    
    print(f"\n3. Enhanced tool with structured output: {enhanced_tool.name}")
    print(f"   Capabilities: {getattr(enhanced_tool, '__tool_capabilities__', 'None')}")
    print(f"   Has STRUCTURED_OUTPUT: {ToolCapability.STRUCTURED_OUTPUT in getattr(enhanced_tool, '__tool_capabilities__', set())}")
    
    return {
        "regular": regular,
        "structured": structured_tool,
        "enhanced": enhanced_tool
    }


def analyze_pydantic_model_distinction():
    """Analyze the distinction between structured output and validation Pydantic models."""
    print("\n\n📋 PYDANTIC MODEL DISTINCTION")
    print("=" * 37)
    
    print("Key insight: NOT ALL Pydantic models are structured output!")
    print("\n1. Regular Pydantic models (validation only):")
    print("   • Used for tool argument validation")
    print("   • Used for configuration, data structures")
    print("   • Should NOT go to parse_output")
    print("   • Example: RegularValidationModel for user input validation")
    
    print("\n2. Structured output Pydantic models:")
    print("   • Intended for LLM output generation")
    print("   • Have ToolCapability.STRUCTURED_OUTPUT")
    print("   • SHOULD go to parse_output")
    print("   • Example: StructuredOutputModel for search results")
    
    # Create tools with different Pydantic models
    config = AugLLMConfig(structured_output_model=StructuredOutputModel)
    
    print(f"\n3. AugLLMConfig analysis:")
    print(f"   structured_output_model: {config.structured_output_model}")
    print(f"   This model SHOULD go to parse_output route")
    
    print(f"\n4. Tool routes in AugLLMConfig:")
    for tool_name, route in config.tool_routes.items():
        print(f"   {tool_name} → {route}")
    
    return config


def demonstrate_capability_based_routing():
    """Demonstrate how to route based on capabilities, not just names."""
    print("\n\n⚙️ CAPABILITY-BASED ROUTING")
    print("=" * 35)
    
    tools = analyze_tool_capabilities()
    config = AugLLMConfig(
        tools=[tools["regular"], tools["structured"], tools["enhanced"]],
        structured_output_model=StructuredOutputModel
    )
    
    print("Proposed routing logic based on capabilities:")
    print("\nfor tool_name, route in tool_routes.items():")
    print("    tool = find_tool_by_name(tool_name)")
    print("    capabilities = getattr(tool, '__tool_capabilities__', set())")
    print("    ")
    print("    if ToolCapability.STRUCTURED_OUTPUT in capabilities:")
    print("        # This tool produces structured output")
    print("        route_to = 'parse_output'")
    print("    elif route == 'pydantic_model':")
    print("        # Regular Pydantic validation (not structured output)")
    print("        route_to = 'validation_node'")
    print("    elif route == 'langchain_tool':")
    print("        # Regular tool execution")
    print("        route_to = 'tool_node'")
    
    print("\nApplying this logic to our tools:")
    for tool_name, route in config.tool_routes.items():
        # Find the actual tool
        tool = None
        for t in config.tools:
            if hasattr(t, 'name') and t.name == tool_name:
                tool = t
                break
        
        if tool:
            capabilities = getattr(tool, '__tool_capabilities__', set())
            has_structured_output = ToolCapability.STRUCTURED_OUTPUT in capabilities
            
            if has_structured_output:
                new_route = "parse_output"
                reason = "has STRUCTURED_OUTPUT capability"
            elif route == "pydantic_model":
                new_route = "validation_node" 
                reason = "regular Pydantic validation"
            else:
                new_route = route
                reason = "existing route"
            
            print(f"   {tool_name}: {route} → {new_route} ({reason})")
        else:
            # Handle engine.structured_output_model
            if tool_name == config.structured_output_model.__name__:
                print(f"   {tool_name}: {route} → parse_output (engine.structured_output_model)")
            else:
                print(f"   {tool_name}: {route} → {route} (tool not found)")


def show_augllmconfig_structured_output_detection():
    """Show how AugLLMConfig should detect structured output models."""
    print("\n\n🎯 AUGLLMCONFIG STRUCTURED OUTPUT DETECTION")
    print("=" * 50)
    
    print("Current issue: AugLLMConfig assigns 'pydantic_model' route to structured_output_model")
    print("Better approach: Assign 'parse_output' route specifically to structured output models")
    
    config = AugLLMConfig(structured_output_model=StructuredOutputModel)
    
    print(f"\nCurrent behavior:")
    for tool_name, route in config.tool_routes.items():
        print(f"   {tool_name} → {route}")
    
    print(f"\nProposed behavior:")
    print(f"   {StructuredOutputModel.__name__} → parse_output (because it's engine.structured_output_model)")
    
    print(f"\nImplementation in AugLLMConfig:")
    print(f"   if self.structured_output_model:")
    print(f"       model_name = self.structured_output_model.__name__")
    print(f"       self.set_tool_route(model_name, 'parse_output')")
    print(f"       # Instead of 'pydantic_model'")


def demonstrate_validation_node_integration():
    """Show how ValidationNodeConfigV2 should integrate with capabilities."""
    print("\n\n🔗 VALIDATION NODE INTEGRATION")
    print("=" * 35)
    
    print("ValidationNodeConfigV2 routing logic should be:")
    print("\nfor tool_call in tool_calls:")
    print("    tool_name = tool_call['name']")
    print("    route = tool_routes.get(tool_name)")
    print("    ")
    print("    # Check if this is a structured output model")
    print("    if route == 'parse_output':")
    print("        # Structured output model - validate and route to ParserNodeV2")
    print("        destinations.add('parse_output')")
    print("    elif route == 'pydantic_model':")
    print("        # Regular Pydantic validation - handle normally")
    print("        destinations.add('validation_node')")
    print("    elif route in ['langchain_tool', 'function']:")
    print("        # Regular tools - route to tool execution") 
    print("        destinations.add('tool_node')")
    
    print("\nKey insight: ValidationNodeConfigV2 already has the right structure!")
    print("We just need to:")
    print("1. AugLLMConfig assigns 'parse_output' to structured output models")
    print("2. ValidationNodeConfigV2 routes 'parse_output' to ParserNodeV2")
    print("3. Keep 'pydantic_model' for other Pydantic validation")


def show_enhanced_naming_integration():
    """Show how enhanced naming works with structured output models."""
    print("\n\n🏷️ ENHANCED NAMING INTEGRATION")
    print("=" * 35)
    
    print("Enhanced naming utilities handle complex structured output models:")
    print("Example: List[Dict[str, SearchResult]] → list_dict_str_searchresult_nested_generic")
    
    from haive.core.utils.enhanced_naming import enhanced_sanitize_tool_name
    
    complex_names = [
        "List[Dict[str, StructuredOutputModel]]",
        "Union[StructuredOutputModel, RegularValidationModel]",
        "Optional[List[StructuredOutputModel]]"
    ]
    
    for name in complex_names:
        try:
            sanitized, transform = enhanced_sanitize_tool_name(name, include_metadata=True)
            print(f"   {name}")
            print(f"   → {sanitized}")
            if transform:
                print(f"   → Description: {transform.description}")
        except Exception as e:
            print(f"   {name} → Error: {e}")


if __name__ == "__main__":
    analyze_tool_capabilities()
    analyze_pydantic_model_distinction() 
    demonstrate_capability_based_routing()
    show_augllmconfig_structured_output_detection()
    demonstrate_validation_node_integration()
    show_enhanced_naming_integration()
    
    print("\n\n🎯 KEY CONCLUSIONS")
    print("=" * 20)
    print("• ToolEngine has comprehensive capability system")
    print("• ToolCapability.STRUCTURED_OUTPUT identifies structured output tools")
    print("• NOT all Pydantic models should go to parse_output")
    print("• AugLLMConfig should assign 'parse_output' to structured_output_model")
    print("• ValidationNodeConfigV2 routing logic is already correct")
    print("• Enhanced naming handles complex generic types")
    
    print("\n💡 Routing refactor should:")
    print("• Use capabilities to identify structured output models")
    print("• Assign 'parse_output' route to engine.structured_output_model")  
    print("• Keep 'pydantic_model' for regular Pydantic validation")
    print("• Route based on intent: structured output vs validation")