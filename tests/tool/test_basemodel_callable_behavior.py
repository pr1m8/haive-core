#!/usr/bin/env python3
"""Test what happens with callable BaseModel in different scenarios."""

from pydantic import BaseModel, Field
from haive.core.engine.tool import ToolEngine
from haive.core.engine.aug_llm import AugLLMConfig
from langchain_core.tools import tool
import json

class ConfigurableTool(BaseModel):
    """A BaseModel with both fields AND __call__."""
    api_key: str = Field(description="API key for service")
    endpoint: str = Field(default="https://api.example.com", description="API endpoint")
    timeout: int = Field(default=30, description="Request timeout")
    
    def __call__(self, query: str, max_results: int = 10) -> str:
        """Execute search query."""
        return f"Searching '{query}' at {self.endpoint} (max: {max_results})"

print("="*80)
print("CALLABLE BASEMODEL BEHAVIOR ANALYSIS")
print("="*80)

# Test 1: What schema does the BaseModel have?
print("\n1️⃣ BaseModel Schema (all fields)")
print("-"*40)
schema = ConfigurableTool.model_json_schema()
print("BaseModel fields:", list(schema['properties'].keys()))
print("Full schema:", json.dumps(schema, indent=2))

# Test 2: What happens when we convert it to a tool?
print("\n\n2️⃣ Tool Conversion")
print("-"*40)

# Create an instance
instance = ConfigurableTool(api_key="secret123")
print(f"Instance state: api_key='{instance.api_key}', endpoint='{instance.endpoint}'")

# Convert using ToolEngine
engine = ToolEngine()
converted = engine._convert_model_to_tool(instance)

if converted:
    print(f"\nConverted tool type: {type(converted)}")
    print(f"Tool name: {converted.name}")
    
    # Check the tool's input schema
    if hasattr(converted, 'args_schema'):
        tool_schema = converted.args_schema.model_json_schema()
        print(f"Tool input fields: {list(tool_schema['properties'].keys())}")
        print(f"Tool schema: {json.dumps(tool_schema, indent=2)}")
    
    # Test execution
    print(f"\nExecution test:")
    result = converted.invoke({"query": "test", "max_results": 5})
    print(f"Result: {result}")

# Test 3: What if we want BOTH schemas?
print("\n\n3️⃣ Manual Tool Creation Options")
print("-"*40)

# Option A: Tool that only takes __call__ params
@tool
def search_tool_a(query: str, max_results: int = 10) -> str:
    """Search using pre-configured instance."""
    return instance(query, max_results)

print("Option A - Only __call__ params:")
print(f"  Schema: {list(search_tool_a.args_schema.model_json_schema()['properties'].keys())}")

# Option B: Tool that takes ALL BaseModel fields + __call__ params
@tool
def search_tool_b(api_key: str, endpoint: str, timeout: int, query: str, max_results: int = 10) -> str:
    """Search with full configuration."""
    tool = ConfigurableTool(api_key=api_key, endpoint=endpoint, timeout=timeout)
    return tool(query, max_results)

print("\nOption B - All fields:")
print(f"  Schema: {list(search_tool_b.args_schema.model_json_schema()['properties'].keys())}")

# Test 4: What does AugLLMConfig do?
print("\n\n4️⃣ AugLLMConfig Behavior")
print("-"*40)

# With class
config1 = AugLLMConfig(tools=[ConfigurableTool])
print(f"Class as tool:")
print(f"  Routes: {config1.tool_routes}")
print(f"  Tool count: {len(config1.get_tools())}")

# What about with instance?
try:
    config2 = AugLLMConfig(tools=[instance])
    print(f"\nInstance as tool:")
    print(f"  Routes: {config2.tool_routes}")
except Exception as e:
    print(f"\nInstance as tool: ERROR - {e}")

# Test 5: The real question - control over behavior
print("\n\n5️⃣ Control Over Behavior")
print("-"*40)
print("Current behavior: BaseModel with __call__ loses its field schema")
print("Only __call__ parameters are exposed to the tool")
print("\nOptions for control:")
print("1. Use instance with state → Only __call__ params exposed")
print("2. Create wrapper tool → Can expose any schema you want")
print("3. Different route could preserve BaseModel schema?")

print("\n" + "="*80)
print("KEY INSIGHTS:")
print("="*80)
print("1. BaseModel schema ≠ Tool schema")
print("2. Tool conversion only looks at __call__ signature")
print("3. BaseModel fields become internal state, not tool inputs")
print("4. You lose direct control over schema when using automatic conversion")
print("5. Manual @tool wrapper gives full control over schema")