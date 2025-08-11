#!/usr/bin/env python3
"""Test BaseModel routing behavior."""

from pydantic import BaseModel, Field
from haive.core.engine.aug_llm import AugLLMConfig
from haive.core.common.mixins.tool_route_mixin import ToolRouteMixin

class StatelessTool(BaseModel):
    """BaseModel without __call__ - cannot be a tool."""
    name: str = "stateless"
    value: int = 42

class StatefulTool(BaseModel):
    """BaseModel with __call__ - can be a tool."""
    multiplier: int = Field(default=2)
    
    def __call__(self, value: int) -> int:
        return value * self.multiplier

print("="*60)
print("BASEMODEL ROUTING ANALYSIS")
print("="*60)

# Test routing detection
mixin = ToolRouteMixin()

print("\n1️⃣ Stateless BaseModel (no __call__):")
route1, metadata1 = mixin._analyze_tool(StatelessTool)
print(f"Route: '{route1}'")
print(f"Metadata: {metadata1}")

print("\n2️⃣ Stateful BaseModel CLASS (has __call__):")
route2, metadata2 = mixin._analyze_tool(StatefulTool)
print(f"Route: '{route2}'")
print(f"Metadata: {metadata2}")

print("\n3️⃣ Stateful BaseModel INSTANCE:")
instance = StatefulTool(multiplier=5)
route3, metadata3 = mixin._analyze_tool(instance)
print(f"Route: '{route3}'")
print(f"Metadata: {metadata3}")

print("\n4️⃣ In AugLLMConfig:")
# Test with class
config1 = AugLLMConfig(tools=[StatelessTool])
print(f"\nStateless class routes: {config1.tool_routes}")

# Test with stateful class  
config2 = AugLLMConfig(tools=[StatefulTool])
print(f"Stateful class routes: {config2.tool_routes}")

# Test with instance
config3 = AugLLMConfig(tools=[instance])
print(f"Instance routes: {config3.tool_routes}")

# Test structured output
config4 = AugLLMConfig(structured_output_model=StatelessTool)
print(f"Structured output routes: {config4.tool_routes}")

print("\n" + "="*60)
print("ROUTING SUMMARY:")
print("="*60)
print("1. BaseModel without __call__ → 'pydantic_model'")
print("2. BaseModel CLASS with __call__ → 'pydantic_tool' (has explicit __call__)")
print("3. BaseModel INSTANCE → depends on instance type check")
print("4. structured_output_model → 'parse_output' (AugLLMConfig override)")
print("\nThe 'pydantic_tool' route identifies BaseModel classes that CAN be tools")