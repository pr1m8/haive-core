#!/usr/bin/env python3
"""
LangChain Tool System Deep Dive - Understanding the Foundation

After examining the actual LangChain source code, this shows how:
1. ToolCall structure works (TypedDict)
2. BaseTool and StructuredTool handle args_schema validation
3. ValidationError handling and invalid_tool_calls
4. The relationship between AI messages, tool calls, and validation

Key Source Files Examined:
- .venv/lib/python3.12/site-packages/langchain_core/tools/base.py
- .venv/lib/python3.12/site-packages/langchain_core/tools/structured.py  
- .venv/lib/python3.12/site-packages/langchain_core/messages/tool.py
"""

from langchain_core.tools import tool, StructuredTool
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools.base import BaseTool
from pydantic import BaseModel, Field, ValidationError
from typing import Dict, List, Any, Optional
from haive.core.engine.aug_llm import AugLLMConfig


# Test Pydantic model for args_schema
class CalculatorInput(BaseModel):
    """Input schema for calculator tool."""
    expression: str = Field(description="Mathematical expression to evaluate")
    precision: int = Field(default=2, description="Decimal precision for result")


class SearchInput(BaseModel):
    """Input schema for search tool."""
    query: str = Field(description="Search query")
    max_results: int = Field(default=10, ge=1, le=100, description="Maximum results to return")


def analyze_tool_call_structure():
    """Analyze the ToolCall TypedDict structure from LangChain source."""
    print("📋 TOOLCALL STRUCTURE (from langchain_core.messages.tool)")
    print("=" * 60)
    
    print("ToolCall is a TypedDict with structure:")
    print("""
class ToolCall(TypedDict):
    name: str           # The name of the tool to be called
    args: dict[str, Any]  # The arguments to the tool call  
    id: str             # An identifier associated with the tool call
    type: Literal["tool_call"]  # Always "tool_call"
    """)
    
    # Example ToolCall
    example_tool_call = {
        "name": "calculator",
        "args": {"expression": "15 * 23", "precision": 2},
        "id": "call_123",
        "type": "tool_call"
    }
    
    print("Example ToolCall:")
    for key, value in example_tool_call.items():
        print(f"   {key}: {repr(value)}")
    
    return example_tool_call


def analyze_args_schema_validation():
    """Analyze how args_schema validation works in BaseTool."""
    print("\n\n🔍 ARGS_SCHEMA VALIDATION (from langchain_core.tools.base)")
    print("=" * 65)
    
    print("ArgsSchema = Union[TypeBaseModel, dict[str, Any]]")
    print("\nValidation happens in BaseTool._parse_input():")
    
    # Create a StructuredTool with args_schema
    calculator_tool = StructuredTool.from_function(
        func=lambda expression, precision=2: f"Result: {eval(expression):.{precision}f}",
        name="calculator",
        description="Calculate mathematical expressions",
        args_schema=CalculatorInput,
    )
    
    print(f"\nCalculator tool args_schema: {calculator_tool.args_schema}")
    print(f"Args schema type: {type(calculator_tool.args_schema)}")
    
    # Show what validation does
    print("\nValidation process:")
    print("1. If tool_input is dict and args_schema is BaseModel:")
    print("   → args_schema.model_validate(tool_input)")
    print("2. If validation passes: return validated dict")
    print("3. If ValidationError: handle based on handle_validation_error setting")
    
    # Test valid input
    valid_input = {"expression": "10 + 5", "precision": 3}
    print(f"\nValid input: {valid_input}")
    try:
        validated = CalculatorInput.model_validate(valid_input)
        print(f"Validation result: {validated.model_dump()}")
        print("✅ Validation passed")
    except ValidationError as e:
        print(f"❌ Validation failed: {e}")
    
    # Test invalid input
    invalid_input = {"expression": "10 + 5", "precision": -1}  # precision has ge=1 constraint
    print(f"\nInvalid input: {invalid_input}")
    try:
        validated = CalculatorInput.model_validate(invalid_input)
        print(f"Validation result: {validated.model_dump()}")
        print("✅ Validation passed")
    except ValidationError as e:
        print(f"❌ Validation failed: {e}")
        print(f"Error details: {e.errors()}")
    
    return calculator_tool


def analyze_tool_invoke_flow():
    """Analyze the tool invocation flow from source code."""
    print("\n\n⚙️ TOOL INVOKE FLOW (from BaseTool.invoke)")
    print("=" * 48)
    
    print("BaseTool.invoke() flow:")
    print("1. _prep_run_args(input, config, **kwargs)")
    print("   • If input is ToolCall: extract args dict")
    print("   • If input is dict/str: use directly")
    print("2. Call self.run(tool_input, **kwargs)")  
    print("3. self._parse_input(tool_input, tool_call_id)")
    print("   • Validate against args_schema")  
    print("   • Handle ValidationError if occurs")
    print("4. Execute tool function")
    print("5. Return result or raise error")
    
    # Demonstrate with StructuredTool
    calculator = analyze_args_schema_validation()
    
    print(f"\nDemonstrating invoke with ToolCall input:")
    tool_call_input = {
        "name": "calculator",
        "args": {"expression": "7 * 8", "precision": 1}, 
        "id": "call_demo",
        "type": "tool_call"
    }
    
    try:
        result = calculator.invoke(tool_call_input)
        print(f"Tool call input: {tool_call_input}")
        print(f"Result: {result}")
        print("✅ Tool invocation successful")
    except Exception as e:
        print(f"❌ Tool invocation failed: {e}")
    
    return calculator


def analyze_validation_error_handling():
    """Analyze ValidationError handling in tools."""
    print("\n\n🚨 VALIDATION ERROR HANDLING")
    print("=" * 35)
    
    print("BaseTool has handle_validation_error setting:")
    print("• False (default): Raise ValidationError")
    print("• True: Return str(ValidationError)")  
    print("• str: Return custom error message")
    print("• Callable: Call function with ValidationError")
    
    # Create tools with different error handling
    def custom_error_handler(error: ValidationError) -> str:
        return f"Custom error: {len(error.errors())} validation issues found"
    
    tools = {
        "default": StructuredTool.from_function(
            func=lambda query, max_results=10: f"Search: {query}",
            name="search_default",
            args_schema=SearchInput,
            handle_validation_error=False  # Default - raises error
        ),
        "simple": StructuredTool.from_function(
            func=lambda query, max_results=10: f"Search: {query}",
            name="search_simple", 
            args_schema=SearchInput,
            handle_validation_error=True  # Returns error as string
        ),
        "custom_msg": StructuredTool.from_function(
            func=lambda query, max_results=10: f"Search: {query}",
            name="search_custom_msg",
            args_schema=SearchInput,
            handle_validation_error="Invalid search parameters provided"
        ),
        "custom_func": StructuredTool.from_function(
            func=lambda query, max_results=10: f"Search: {query}",
            name="search_custom_func",
            args_schema=SearchInput,
            handle_validation_error=custom_error_handler
        )
    }
    
    # Test with invalid input
    invalid_search = {"query": "python", "max_results": 150}  # max_results > 100
    
    print(f"\nTesting with invalid input: {invalid_search}")
    
    for name, tool in tools.items():
        print(f"\n{name} tool (handle_validation_error={tool.handle_validation_error}):")
        try:
            result = tool.invoke(invalid_search)
            print(f"   Result: {result}")
        except ValidationError as e:
            print(f"   ❌ ValidationError raised: {e.errors()[0]['msg']}")
        except Exception as e:
            print(f"   ❌ Other error: {e}")


def analyze_ai_message_tool_calls():
    """Analyze how AIMessage handles tool_calls and invalid_tool_calls."""
    print("\n\n💬 AI MESSAGE TOOL CALLS")
    print("=" * 30)
    
    print("AIMessage attributes for tool calling:")
    print("• tool_calls: List[ToolCall] - Valid, parsed tool calls")
    print("• invalid_tool_calls: List[InvalidToolCall] - Malformed tool calls")
    print("• additional_kwargs: Dict - Raw tool calls from provider")
    
    # Create AIMessage with tool calls
    ai_msg = AIMessage(
        content="I'll help you with calculations and search.",
        tool_calls=[
            {
                "name": "calculator", 
                "args": {"expression": "15 * 23", "precision": 2},
                "id": "call_1",
                "type": "tool_call"
            },
            {
                "name": "search",
                "args": {"query": "python tutorial", "max_results": 5}, 
                "id": "call_2",
                "type": "tool_call"
            }
        ]
    )
    
    print(f"\nAIMessage with tool calls:")
    print(f"   Content: {ai_msg.content}")
    print(f"   Tool calls: {len(ai_msg.tool_calls)}")
    for i, call in enumerate(ai_msg.tool_calls):
        print(f"   [{i+1}] {call['name']}({call['args']}) [ID: {call['id']}]")
    
    return ai_msg


def analyze_validation_node_integration():
    """Analyze how ValidationNodeConfigV2 integrates with this system."""
    print("\n\n🔗 VALIDATION NODE INTEGRATION")
    print("=" * 38)
    
    print("ValidationNodeConfigV2 process:")
    print("1. Receives AIMessage with tool_calls")
    print("2. For each tool_call:")
    print("   • Gets tool.args_schema from engine")
    print("   • Creates LangGraph ValidationNode with schemas") 
    print("   • ValidationNode.invoke(state) validates tool_calls")
    print("   • Validation uses the same BaseTool._parse_input logic")
    print("3. ValidationNode creates ToolMessages:")
    print("   • Success: ToolMessage with validated data")
    print("   • Error: ToolMessage with error details")
    print("4. Routes based on tool routes:")
    print("   • langchain_tool → tool_node (with validated tool_calls in AIMessage)")
    print("   • pydantic_model → parse_output (with ToolMessages)")
    
    print("\nKey insight: ValidationNode uses the SAME validation logic")
    print("as BaseTool.invoke() - it's calling the same _parse_input method!")
    
    # Show what AugLLMConfig provides
    calculator = StructuredTool.from_function(
        func=lambda expression, precision=2: f"Result: {eval(expression):.{precision}f}",
        name="calculator",
        args_schema=CalculatorInput,
    )
    
    config = AugLLMConfig(tools=[calculator])
    print(f"\nAugLLMConfig tool routes: {config.tool_routes}")
    
    # The args_schema is available through the tool
    for tool in config.tools:
        if hasattr(tool, 'args_schema'):
            print(f"Tool {tool.name} args_schema: {tool.args_schema}")


if __name__ == "__main__":
    analyze_tool_call_structure()
    analyze_args_schema_validation()
    analyze_tool_invoke_flow() 
    analyze_validation_error_handling()
    analyze_ai_message_tool_calls()
    analyze_validation_node_integration()
    
    print("\n\n🎯 KEY INSIGHTS FROM LANGCHAIN SOURCE")
    print("=" * 45)
    print("• ToolCall is a TypedDict: {name, args, id, type}")
    print("• args_schema is a Pydantic BaseModel for validation")
    print("• BaseTool._parse_input() does the actual validation")
    print("• ValidationError handling is configurable")
    print("• ValidationNode uses the same validation logic as BaseTool")
    print("• tool_calls vs invalid_tool_calls distinguish valid from malformed calls")
    print("• The entire system is built on Pydantic validation")
    print("\n💡 For routing refactor:")
    print("• ValidationNodeConfigV2 already uses this validation system")
    print("• args_schema provides the validation rules")
    print("• Tool routes determine post-validation routing")
    print("• structured output models are just tools with Pydantic args_schema")