#!/usr/bin/env python3
"""Detailed trace of BaseModel tool conversion."""


from pydantic import BaseModel, Field

from haive.core.engine.aug_llm import AugLLMConfig
from haive.core.engine.tool.engine import ToolEngine


class StatefulTool(BaseModel):
    """A tool with internal state."""
    name: str = Field(default="statefultool", description="Tool name")
    multiplier: int = Field(default=2, description="Multiplication factor")

    def __call__(self, value: int) -> int:
        """Multiply value by configured multiplier."""
        return value * self.multiplier

print("="*80)
print("BASEMODEL TOOL CONVERSION ANALYSIS")
print("="*80)

# Test 1: Check if CLASS can be converted
print("\n1️⃣ Testing BaseModel CLASS as tool")
print("-"*40)
engine = ToolEngine()

# Check the conversion method directly
print("Can convert class to tool?")
try:
    # The engine expects an instance, not a class
    result = engine._convert_model_to_tool(StatefulTool)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {e}")

# Test 2: Check INSTANCE conversion
print("\n\n2️⃣ Testing BaseModel INSTANCE as tool")
print("-"*40)
instance1 = StatefulTool(name="tool1", multiplier=3)
instance2 = StatefulTool(name="tool2", multiplier=5)

print(f"Instance1: multiplier={instance1.multiplier}")
print(f"Instance2: multiplier={instance2.multiplier}")
print(f"Instance1 callable: {callable(instance1)}")

# Convert instances
converted1 = engine._convert_model_to_tool(instance1)
converted2 = engine._convert_model_to_tool(instance2)

if converted1:
    print("\nConverted tool 1:")
    print(f"  Type: {type(converted1)}")
    print(f"  Name: {converted1.name}")
    print(f"  Description: {converted1.description}")
    if hasattr(converted1, "args_schema"):
        print(f"  Args schema: {converted1.args_schema.model_json_schema()}")

    # Test execution
    print("\n  Testing execution:")
    result1 = converted1.invoke({"value": 10})
    print(f"  10 * 3 = {result1}")

if converted2:
    print("\nConverted tool 2:")
    result2 = converted2.invoke({"value": 10})
    print(f"  10 * 5 = {result2}")

# Test 3: Check AugLLMConfig handling
print("\n\n3️⃣ Testing in AugLLMConfig")
print("-"*40)

# With instance
config = AugLLMConfig(tools=[instance1])
print(f"Tool routes: {config.tool_routes}")
print(f"Pydantic tools stored: {len(config.pydantic_tools)}")

# Get processed tools
processed = config.get_tools()
print(f"Processed tools: {len(processed)}")
if processed:
    tool = processed[0]
    print(f"First tool type: {type(tool)}")
    print(f"First tool name: {tool.name}")

# Test 4: State preservation
print("\n\n4️⃣ Testing state preservation")
print("-"*40)
instance3 = StatefulTool(multiplier=7)
converted3 = engine._convert_model_to_tool(instance3)

if converted3:
    # Test with different inputs
    for val in [1, 5, 10]:
        result = converted3.invoke({"value": val})
        print(f"  {val} * 7 = {result}")

print("\n\n5️⃣ Key insights:")
print("-"*40)
print("- BaseModel CLASS cannot be converted (not callable)")
print("- BaseModel INSTANCE can be converted if it has __call__")
print("- Each instance preserves its own state (multiplier)")
print("- Converted tool schema only includes __call__ parameters")
print("- BaseModel fields are NOT part of the tool input schema")
