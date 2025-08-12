#!/usr/bin/env python3
"""Test enhanced tool detection in AugLLMConfig."""

from pydantic import BaseModel, Field

from haive.core.engine.aug_llm import AugLLMConfig


class PlainModel(BaseModel):
    """BaseModel without __call__ - not executable."""
    name: str
    value: int = 42

class ExecutableModel(BaseModel):
    """BaseModel with __call__ - executable."""
    multiplier: int = Field(default=2)

    def __call__(self, value: int) -> int:
        return value * self.multiplier

print("="*60)
print("ENHANCED DETECTION TEST")
print("="*60)

# Test 1: Class detection
print("\n1️⃣ Class Detection")
print("-"*40)

config1 = AugLLMConfig(tools=[PlainModel])
plain_route = config1.tool_routes.get("PlainModel")
plain_meta = config1.get_tool_metadata("PlainModel")

print("PlainModel (no __call__):")
print(f"  Route: {plain_route}")
print(f"  Metadata: {plain_meta}")

config2 = AugLLMConfig(tools=[ExecutableModel])
exec_route = config2.tool_routes.get("ExecutableModel")
exec_meta = config2.get_tool_metadata("ExecutableModel")

print("\nExecutableModel (has __call__):")
print(f"  Route: {exec_route}")
print(f"  Metadata: {exec_meta}")

# Test 2: Instance detection
print("\n\n2️⃣ Instance Detection")
print("-"*40)

instance = ExecutableModel(multiplier=5)
try:
    config3 = AugLLMConfig(tools=[instance])
    instance_route = config3.tool_routes.get("ExecutableModel")
    instance_meta = config3.get_tool_metadata("ExecutableModel")

    print("ExecutableModel instance:")
    print(f"  Route: {instance_route}")
    print(f"  Metadata: {instance_meta}")
except Exception as e:
    print(f"ExecutableModel instance: ERROR - {e}")

# Test 3: Structured output override
print("\n\n3️⃣ Structured Output Override")
print("-"*40)

config4 = AugLLMConfig(structured_output_model=PlainModel)
struct_route = config4.tool_routes.get("PlainModel")
struct_meta = config4.get_tool_metadata("PlainModel")

print("PlainModel as structured_output:")
print(f"  Route: {struct_route}")
print(f"  Metadata: {struct_meta}")

print("\n" + "="*60)
print("EXPECTED RESULTS:")
print("="*60)
print("✅ PlainModel (class) → 'pydantic_model' (not executable)")
print("✅ ExecutableModel (class) → 'pydantic_tool' (executable)")
print("✅ ExecutableModel (instance) → 'function' (callable)")
print("✅ structured_output_model → 'parse_output' (override)")
