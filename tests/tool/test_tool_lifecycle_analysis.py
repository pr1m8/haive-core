#!/usr/bin/env python3
"""
Tool Lifecycle Analysis - Understanding the Full Tool Flow

This analyzes the complete lifecycle:
1. ToolEngine creates tools
2. AI messages get tool calls  
3. ValidationNode validates tool calls
4. ToolMessages are created
5. Tools are executed
6. Results flow back

Key relationships to understand:
- How do tool calls get into AI messages?
- What makes a tool call valid vs invalid?
- How do ToolMessages relate to tool calls?
- What's the execution flow?
"""

from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel, Field
from typing import Dict, List, Any
from haive.core.engine.aug_llm import AugLLMConfig
from haive.core.engine.tool import ToolEngine


# Test models and tools
class SearchResult(BaseModel):
    """Structured output model."""
    query: str = Field(description="Search query")
    results: List[str] = Field(description="Search results")


@tool
def calculator(expression: str) -> str:
    """Calculate mathematical expressions."""
    return f"Result: {eval(expression)}"


def analyze_tool_engine_capabilities():
    """Analyze what types of tools ToolEngine can create."""
    print("🔧 TOOL ENGINE CAPABILITIES")
    print("=" * 35)
    
    # 1. Regular LangChain tool
    regular_tool = calculator
    
    # 2. Structured output tool 
    structured_tool = ToolEngine.create_structured_output_tool(
        func=lambda query: SearchResult(
            query=query,
            results=[f"Result 1 for {query}", f"Result 2 for {query}"]
        ),
        name="search_engine",
        description="Search with structured output",
        output_model=SearchResult
    )
    
    # 3. State-aware tool
    state_tool = ToolEngine.create_state_tool(
        lambda state: f"State has {len(state.get('messages', []))} messages",
        name="state_reader",
        description="Read state information",
        reads_state=True,
        state_keys=["messages"]
    )
    
    # 4. Store tools
    from haive.core.tools.store_manager import StoreManager
    store_manager = StoreManager()
    store_tools = ToolEngine.create_store_tools_suite(
        store_manager=store_manager,
        namespace=("test", "analysis"),
        include_tools=["store"]
    )
    
    print("Tool types ToolEngine can create:")
    print(f"   • Regular LangChain tool: {regular_tool.name}")
    print(f"   • Structured output tool: {structured_tool.name}")
    print(f"   • State-aware tool: {state_tool.name}")
    print(f"   • Store tools: {[t.name for t in store_tools]}")
    
    return {
        "regular": regular_tool,
        "structured": structured_tool, 
        "state": state_tool,
        "store": store_tools[0] if store_tools else None
    }


def analyze_ai_message_tool_calls():
    """Analyze how tool calls get into AI messages."""
    print("\n\n💬 AI MESSAGE TOOL CALLS")
    print("=" * 30)
    
    # Simulate what an LLM would generate
    ai_message_with_tools = AIMessage(
        content="I'll help you calculate that and search for information.",
        tool_calls=[
            {
                "name": "calculator",
                "args": {"expression": "15 * 23"},
                "id": "call_1",
                "type": "function"
            },
            {
                "name": "SearchResult", 
                "args": {
                    "query": "python tutorial",
                    "results": ["Tutorial 1", "Tutorial 2"]
                },
                "id": "call_2",
                "type": "function"
            }
        ]
    )
    
    print("AI message with tool calls:")
    print(f"   Content: {ai_message_with_tools.content}")
    print(f"   Tool calls: {len(ai_message_with_tools.tool_calls)}")
    
    for i, call in enumerate(ai_message_with_tools.tool_calls):
        print(f"   [{i+1}] {call['name']}({call['args']}) [ID: {call['id']}]")
    
    return ai_message_with_tools


def analyze_tool_validation_process():
    """Analyze how tool call validation works."""
    print("\n\n✅ TOOL VALIDATION PROCESS")  
    print("=" * 32)
    
    # Create AugLLMConfig with tools
    tools = analyze_tool_engine_capabilities()
    config = AugLLMConfig(
        tools=[tools["regular"], tools["structured"]],
        structured_output_model=SearchResult
    )
    
    print("Tool routes in AugLLMConfig:")
    for tool_name, route in config.tool_routes.items():
        print(f"   {tool_name} → {route}")
    
    # Simulate validation process
    ai_message = analyze_ai_message_tool_calls()
    
    print("\nValidation process for each tool call:")
    for call in ai_message.tool_calls:
        tool_name = call["name"]
        tool_id = call["id"]
        args = call["args"]
        route = config.tool_routes.get(tool_name, "unknown")
        
        print(f"\n   Tool call: {tool_name} [ID: {tool_id}]")
        print(f"      Route: {route}")
        print(f"      Args: {args}")
        
        # Analyze what validation would do
        if route == "langchain_tool":
            print(f"      Validation: Check args against tool schema")
            print(f"      If valid: Include in validated_tool_calls")
            print(f"      If invalid: Mark as invalid, create error ToolMessage")
            
        elif route == "pydantic_model":
            print(f"      Validation: Validate args against Pydantic model")
            print(f"      If valid: Create success ToolMessage with validated data")
            print(f"      If invalid: Create error ToolMessage with validation errors")
            
        elif route == "unknown":
            print(f"      Validation: Tool not found in engine")
            print(f"      Result: Create error ToolMessage")


def analyze_tool_message_creation():
    """Analyze how ToolMessages are created."""
    print("\n\n📨 TOOL MESSAGE CREATION")
    print("=" * 30)
    
    print("Different types of ToolMessages created:")
    
    # 1. Successful Pydantic validation
    success_tool_msg = ToolMessage(
        content='{"success": true, "data": {"query": "test", "results": ["r1", "r2"]}, "validated": true}',
        tool_call_id="call_1",
        name="SearchResult",
        additional_kwargs={
            "is_error": False,
            "validation_passed": True,
            "model_type": "pydantic"
        }
    )
    
    # 2. Failed Pydantic validation  
    error_tool_msg = ToolMessage(
        content='{"success": false, "error": "ValidationError", "details": "Field required"}',
        tool_call_id="call_2", 
        name="SearchResult",
        additional_kwargs={
            "is_error": True,
            "validation_passed": False,
            "error_type": "validation_error"
        }
    )
    
    # 3. Tool execution result (from ToolNode)
    execution_tool_msg = ToolMessage(
        content="Result: 345",
        tool_call_id="call_3",
        name="calculator",
        additional_kwargs={}
    )
    
    print("   1. Successful Pydantic validation:")
    print(f"      Content: {success_tool_msg.content[:50]}...")
    print(f"      Additional kwargs: {success_tool_msg.additional_kwargs}")
    
    print("   2. Failed Pydantic validation:")
    print(f"      Content: {error_tool_msg.content[:50]}...")
    print(f"      Additional kwargs: {error_tool_msg.additional_kwargs}")
    
    print("   3. Tool execution result:")
    print(f"      Content: {execution_tool_msg.content}")
    print(f"      Additional kwargs: {execution_tool_msg.additional_kwargs}")


def analyze_complete_tool_flow():
    """Analyze the complete tool execution flow."""
    print("\n\n🔄 COMPLETE TOOL FLOW")
    print("=" * 25)
    
    flow_steps = [
        "1. LLM generates AIMessage with tool_calls",
        "2. ValidationNodeConfigV2 receives messages",
        "3. For each tool call:",
        "   a. Look up route in tool_routes",
        "   b. If langchain_tool: Validate & collect for injection",
        "   c. If pydantic_model: Validate & create ToolMessage",
        "   d. If unknown: Create error ToolMessage",
        "4. For langchain_tools: Inject validated calls into new AIMessage",
        "5. Route based on tool types:",
        "   • langchain_tool → tool_node (with injected AIMessage)",
        "   • pydantic_model → parse_output (with ToolMessages)",
        "6. Tool execution:",
        "   • tool_node: Executes tools, creates ToolMessages",
        "   • parse_output: Parses existing ToolMessages",
        "7. Results added to conversation history"
    ]
    
    for step in flow_steps:
        print(f"   {step}")


def analyze_valid_vs_invalid_tool_calls():
    """Analyze what makes tool calls valid vs invalid."""
    print("\n\n⚖️ VALID VS INVALID TOOL CALLS")
    print("=" * 35)
    
    print("A tool call is VALID if:")
    print("   ✅ Tool name exists in tool_routes")
    print("   ✅ Tool arguments match the tool's schema")
    print("   ✅ Required arguments are provided")
    print("   ✅ Argument types are correct")
    print("   ✅ For Pydantic models: Data passes model validation")
    
    print("\nA tool call is INVALID if:")
    print("   ❌ Tool name not found in engine")  
    print("   ❌ Missing required arguments")
    print("   ❌ Wrong argument types")
    print("   ❌ For Pydantic models: Validation errors")
    print("   ❌ Tool call structure is malformed")
    
    print("\nValidation outcomes:")
    print("   • Valid langchain_tool → Include in validated_tool_calls → Inject into AIMessage → Execute")
    print("   • Valid pydantic_model → Create success ToolMessage → Route to parse_output")
    print("   • Invalid tool_call → Create error ToolMessage → Route to agent")


if __name__ == "__main__":
    analyze_tool_engine_capabilities()
    analyze_ai_message_tool_calls()
    analyze_tool_validation_process()
    analyze_tool_message_creation()
    analyze_complete_tool_flow()
    analyze_valid_vs_invalid_tool_calls()
    
    print("\n\n🎯 KEY INSIGHTS")
    print("=" * 20)
    print("• ToolEngine creates different types of tools with different capabilities")
    print("• AI messages contain tool_calls that reference these tools")
    print("• ValidationNodeConfigV2 validates tool calls and creates ToolMessages")  
    print("• Valid langchain tools get injected back into AIMessage for execution")
    print("• Valid pydantic models get ToolMessages created immediately")
    print("• Invalid tools get error ToolMessages")
    print("• The route determines how ValidationNodeConfigV2 handles each tool call")
    print("• Tool execution happens in tool_node (langchain) or parse_output (pydantic)")