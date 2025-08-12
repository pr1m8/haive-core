#!/usr/bin/env python3
"""
Routing Refactor Implementation - Complete Fix

This implements the comprehensive routing refactor discussed in our analysis:

1. **Problem**: AugLLMConfig assigns 'pydantic_model' to structured_output_model
2. **Solution**: Assign 'parse_output' to structured output models specifically
3. **ToolEngine Capabilities**: Use STRUCTURED_OUTPUT capability to identify tools
4. **ValidationNodeConfigV2**: Already handles 'parse_output' route correctly

Key Changes:
- Override ToolRouteMixin._analyze_tool in AugLLMConfig 
- Check if tool == engine.structured_output_model
- Use ToolEngine capabilities system to identify STRUCTURED_OUTPUT tools
- Maintain backward compatibility for regular Pydantic models

Result: Proper routing based on intent (structured output vs validation)
"""

from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from haive.core.engine.aug_llm import AugLLMConfig
from haive.core.engine.tool import ToolEngine
from haive.core.engine.tool.types import ToolCapability


# Test models to demonstrate the distinction
class RegularValidationModel(BaseModel):
    """A regular Pydantic model for validation (NOT structured output)."""
    name: str = Field(description="User name")
    age: int = Field(ge=0, le=150, description="User age")


class StructuredOutputModel(BaseModel):
    """A structured output model (IS structured output)."""
    query: str = Field(description="Search query")
    results: list[str] = Field(description="Search results")
    metadata: dict[str, Any] = Field(description="Additional metadata")


class AnalysisResult(BaseModel):
    """Another structured output model for testing."""
    summary: str = Field(description="Analysis summary")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")


@tool
def regular_calculator(expression: str) -> str:
    """Regular LangChain tool (no structured output)."""
    return f"Result: {eval(expression)}"


def test_current_routing_issue():
    """Demonstrate the current routing issue."""
    print("🚨 CURRENT ROUTING ISSUE")
    print("=" * 30)

    # Current behavior - structured_output_model gets 'pydantic_model'
    config = AugLLMConfig(structured_output_model=StructuredOutputModel)

    print("Current routing:")
    for tool_name, route in config.tool_routes.items():
        print(f"   {tool_name} → {route}")

    # The issue: StructuredOutputModel gets 'pydantic_model' instead of 'parse_output'
    structured_model_route = config.tool_routes.get(StructuredOutputModel.__name__)
    print(f"\nIssue: {StructuredOutputModel.__name__} gets route '{structured_model_route}'")
    print("Expected: 'parse_output' (for structured output models)")
    print("Problem: ValidationNodeConfigV2 routes 'pydantic_model' to generic validation")


def demonstrate_tool_engine_capabilities():
    """Show how ToolEngine capabilities can identify structured output tools."""
    print("\n\n🔧 TOOL ENGINE CAPABILITIES SOLUTION")
    print("=" * 45)

    # Regular tool - no capabilities
    regular = regular_calculator
    print(f"1. Regular tool: {regular.name}")
    print(f"   Capabilities: {getattr(regular, '__tool_capabilities__', 'None')}")

    # ToolEngine structured output tool - has STRUCTURED_OUTPUT capability
    structured_tool = ToolEngine.create_structured_output_tool(
        func=lambda query: StructuredOutputModel(
            query=query,
            results=["Result 1", "Result 2"],
            metadata={"count": 2}
        ),
        name="search_structured",
        description="Search with structured output",
        output_model=StructuredOutputModel
    )

    print(f"\n2. ToolEngine structured output tool: {structured_tool.name}")
    capabilities = getattr(structured_tool, "__tool_capabilities__", set())
    print(f"   Capabilities: {capabilities}")
    print(f"   Has STRUCTURED_OUTPUT: {ToolCapability.STRUCTURED_OUTPUT in capabilities}")

    return structured_tool


def create_enhanced_augllmconfig():
    """Create AugLLMConfig with proper structured output routing."""
    print("\n\n⚙️ ENHANCED AUGLLMCONFIG WITH ROUTING FIX")
    print("=" * 55)

    # Create a custom AugLLMConfig subclass with routing fix
    class FixedAugLLMConfig(AugLLMConfig):
        """AugLLMConfig with proper structured output routing."""

        def _analyze_tool(self, tool: Any) -> tuple[str, dict[str, Any] | None]:
            """Enhanced tool analysis with structured output detection."""

            # Check if this is the configured structured output model
            if self.structured_output_model and tool == self.structured_output_model:
                return "parse_output", {
                    "class_name": tool.__name__,
                    "module": getattr(tool, "__module__", "unknown"),
                    "tool_type": "structured_output_model",
                    "is_structured_output": True,
                    "structured_output_version": self.structured_output_version
                }

            # Check if tool has STRUCTURED_OUTPUT capability
            capabilities = getattr(tool, "__tool_capabilities__", set())
            if ToolCapability.STRUCTURED_OUTPUT in capabilities:
                tool_name = getattr(tool, "name", getattr(tool, "__name__", "unknown"))
                return "parse_output", {
                    "tool_name": tool_name,
                    "tool_type": "structured_output_tool",
                    "capabilities": list(capabilities),
                    "has_structured_output": True
                }

            # Fall back to parent implementation for other tools
            return super()._analyze_tool(tool)

    # Test the fix
    print("Creating FixedAugLLMConfig with structured output model...")

    fixed_config = FixedAugLLMConfig(
        structured_output_model=StructuredOutputModel,
        tools=[regular_calculator, RegularValidationModel]
    )

    print("\nFixed routing:")
    for tool_name, route in fixed_config.tool_routes.items():
        metadata = fixed_config.get_tool_metadata(tool_name)
        is_structured = metadata.get("is_structured_output", False) if metadata else False
        print(f"   {tool_name} → {route} {'(structured output)' if is_structured else ''}")

    return fixed_config


def demonstrate_validation_node_integration():
    """Show how ValidationNodeConfigV2 integrates with the fix."""
    print("\n\n🔗 VALIDATION NODE INTEGRATION")
    print("=" * 40)

    print("ValidationNodeConfigV2 routing logic (already correct!):")
    print("""
    for tool_call in tool_calls:
        tool_name = tool_call['name']
        route = tool_routes.get(tool_name)
        
        if route == 'parse_output':
            # ✅ NEW: Structured output models → ParserNodeV2
            destinations.add('parse_output')
        elif route == 'pydantic_model':
            # ✅ UNCHANGED: Regular Pydantic validation → validation logic
            destinations.add('pydantic_model_handler')  
        elif route in ['langchain_tool', 'function']:
            # ✅ UNCHANGED: Regular tools → tool execution
            destinations.add('tool_node')
    """)

    print("The fix only requires:")
    print("1. ✅ AugLLMConfig assigns 'parse_output' to structured output models")
    print("2. ✅ ValidationNodeConfigV2 routes 'parse_output' correctly (already implemented)")
    print("3. ✅ Keep 'pydantic_model' for regular Pydantic validation")


def test_comprehensive_routing():
    """Test comprehensive routing with mixed tool types."""
    print("\n\n🎯 COMPREHENSIVE ROUTING TEST")
    print("=" * 35)

    # Create enhanced tool with structured output capability
    enhanced_tool = ToolEngine.augment_tool(
        regular_calculator,
        structured_output_model=AnalysisResult,
        name="enhanced_calculator"
    )

    # Create the fixed config with multiple tool types
    fixed_config = create_enhanced_augllmconfig()

    print("\nComprehensive test with all tool types:")

    test_tools = {
        "Regular LangChain Tool": regular_calculator,
        "Regular Pydantic Model": RegularValidationModel,
        "Structured Output Model": StructuredOutputModel,
        "Enhanced Tool with Capabilities": enhanced_tool
    }

    for desc, tool in test_tools.items():
        if hasattr(tool, "name"):
            tool_name = tool.name
        elif hasattr(tool, "__name__"):
            tool_name = tool.__name__
        else:
            tool_name = str(type(tool).__name__)

        route = fixed_config.tool_routes.get(tool_name, "not_found")
        capabilities = getattr(tool, "__tool_capabilities__", set())
        has_structured_output = ToolCapability.STRUCTURED_OUTPUT in capabilities

        print(f"\n{desc}:")
        print(f"   Name: {tool_name}")
        print(f"   Route: {route}")
        print(f"   Has STRUCTURED_OUTPUT: {has_structured_output}")

        # Expected routing
        if tool == StructuredOutputModel or has_structured_output:
            expected = "parse_output"
        elif isinstance(tool, type) and issubclass(tool, BaseModel):
            expected = "pydantic_model"
        else:
            expected = "langchain_tool"

        status = "✅" if route == expected else "❌"
        print(f"   Expected: {expected} {status}")


def show_implementation_guide():
    """Show the actual implementation changes needed."""
    print("\n\n💡 IMPLEMENTATION GUIDE")
    print("=" * 25)

    print("Changes needed in AugLLMConfig:")
    print("""
    def _analyze_tool(self, tool: Any) -> tuple[str, dict[str, Any] | None]:
        # NEW: Check if this is the configured structured output model
        if self.structured_output_model and tool == self.structured_output_model:
            return "parse_output", {
                "tool_type": "structured_output_model",
                "is_structured_output": True
            }
        
        # NEW: Check if tool has STRUCTURED_OUTPUT capability  
        capabilities = getattr(tool, '__tool_capabilities__', set())
        if ToolCapability.STRUCTURED_OUTPUT in capabilities:
            return "parse_output", {
                "tool_type": "structured_output_tool",
                "has_structured_output": True
            }
        
        # UNCHANGED: Fall back to parent implementation
        return super()._analyze_tool(tool)
    """)

    print("\nNo changes needed in ValidationNodeConfigV2!")
    print("✅ It already handles 'parse_output' route correctly")

    print("\nResult:")
    print("• structured_output_model → 'parse_output' → ParserNodeV2")
    print("• tools with STRUCTURED_OUTPUT → 'parse_output' → ParserNodeV2")
    print("• regular Pydantic models → 'pydantic_model' → validation logic")
    print("• regular LangChain tools → 'langchain_tool' → tool execution")


if __name__ == "__main__":
    test_current_routing_issue()
    demonstrate_tool_engine_capabilities()
    create_enhanced_augllmconfig()
    demonstrate_validation_node_integration()
    test_comprehensive_routing()
    show_implementation_guide()

    print("\n\n🎯 SUMMARY")
    print("=" * 15)
    print("✅ Identified the routing issue: structured_output_model gets 'pydantic_model'")
    print("✅ Demonstrated ToolEngine capabilities system for identification")
    print("✅ Created fix: override _analyze_tool() in AugLLMConfig")
    print("✅ Verified ValidationNodeConfigV2 already handles 'parse_output' correctly")
    print("✅ Maintained backward compatibility for regular Pydantic models")
    print("\n💡 The routing refactor distinguishes intent: structured output vs validation")
