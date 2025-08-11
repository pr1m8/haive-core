#!/usr/bin/env python3
"""Complete end-to-end test of the routing system."""

from pydantic import BaseModel, Field
from haive.core.engine.aug_llm import AugLLMConfig

# Test models
class ResponseModel(BaseModel):
    """Structured output model."""
    answer: str = Field(description="The answer")
    confidence: float = Field(description="Confidence score")

class NonExecutableModel(BaseModel):
    """BaseModel without __call__ (should get pydantic_model route)."""
    name: str = "test"

class ExecutableModel(BaseModel):
    """BaseModel with __call__ (should get pydantic_tool route)."""
    multiplier: int = 2
    
    def __call__(self, value: int) -> int:
        return value * self.multiplier

print("="*60)
print("COMPLETE ROUTING WORKFLOW TEST")
print("="*60)

# Test 1: Structured output model
print("\n1️⃣ Testing structured output model")
config1 = AugLLMConfig(structured_output_model=ResponseModel)
print(f"   Route: {config1.tool_routes.get('ResponseModel', 'NOT_FOUND')}")
assert config1.tool_routes.get('ResponseModel') == 'parse_output', "Should be parse_output"

# Test 2: Non-executable BaseModel as tool (error case)
print("\n2️⃣ Testing non-executable BaseModel as tool")
config2 = AugLLMConfig(tools=[NonExecutableModel])
print(f"   Route: {config2.tool_routes.get('NonExecutableModel', 'NOT_FOUND')}")
assert config2.tool_routes.get('NonExecutableModel') == 'pydantic_model', "Should be pydantic_model"

# Test 3: Executable BaseModel as tool
print("\n3️⃣ Testing executable BaseModel as tool")
config3 = AugLLMConfig(tools=[ExecutableModel])
print(f"   Route: {config3.tool_routes.get('ExecutableModel', 'NOT_FOUND')}")
assert config3.tool_routes.get('ExecutableModel') == 'pydantic_tool', "Should be pydantic_tool"

# Test 4: Mixed tools
print("\n4️⃣ Testing mixed tools")
instance = ExecutableModel(multiplier=5)
config4 = AugLLMConfig(tools=[ExecutableModel, instance, NonExecutableModel])
print(f"   Routes: {config4.tool_routes}")

# Expected:
# - ExecutableModel (class): pydantic_tool
# - tool_1 (instance): function  (instance is at index 1)
# - NonExecutableModel (class): pydantic_model
expected_routes = {
    'ExecutableModel': 'pydantic_tool',
    'tool_1': 'function',  # Instance gets function route (index 1)
    'NonExecutableModel': 'pydantic_model'
}

for tool_name, expected_route in expected_routes.items():
    actual_route = config4.tool_routes.get(tool_name, 'NOT_FOUND')
    print(f"   {tool_name}: {actual_route} (expected: {expected_route})")
    assert actual_route == expected_route, f"Route mismatch for {tool_name}"

# Test 5: with_structured_output method
print("\n5️⃣ Testing with_structured_output method")
config5 = AugLLMConfig()
config5.with_structured_output(ResponseModel, version="v2")
print(f"   Route: {config5.tool_routes.get('ResponseModel', 'NOT_FOUND')}")
assert config5.tool_routes.get('ResponseModel') == 'parse_output', "Should be parse_output after with_structured_output"

print("\n" + "="*60)
print("✅ ALL ROUTING TESTS PASSED!")
print("="*60)