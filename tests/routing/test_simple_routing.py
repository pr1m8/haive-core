#!/usr/bin/env python3
"""Simple test to understand BaseModel routing."""

from pydantic import BaseModel, Field
from haive.core.engine.aug_llm import AugLLMConfig
from haive.agents.simple.agent import SimpleAgent

# Test different scenarios
class MyModel(BaseModel):
    """Test model."""
    name: str
    value: int

class CallableModel(BaseModel):
    """Model with __call__."""
    query: str
    
    def __call__(self):
        return f"Result: {self.query}"

print("1. BaseModel as TOOL:")
config1 = AugLLMConfig(tools=[MyModel])
print(f"   Routes: {config1.tool_routes}")
print(f"   Pydantic tools: {config1.pydantic_tools}")

print("\n2. CallableModel as TOOL:")
config2 = AugLLMConfig(tools=[CallableModel])
print(f"   Routes: {config2.tool_routes}")
print(f"   Pydantic tools: {config2.pydantic_tools}")

print("\n3. BaseModel as STRUCTURED OUTPUT:")
config3 = AugLLMConfig(structured_output_model=MyModel)
print(f"   Routes: {config3.tool_routes}")

print("\n4. Both tool AND structured output:")
config4 = AugLLMConfig(tools=[CallableModel], structured_output_model=MyModel)
print(f"   Routes: {config4.tool_routes}")

# Key question: What SHOULD the routes be?
print("\n" + "="*50)
print("EXPECTED BEHAVIOR:")
print("- BaseModel as tool (no __call__) → ???")
print("- BaseModel as tool (with __call__) → ???") 
print("- BaseModel as structured_output → 'parse_output'")
print("\nACTUAL: All BaseModel get 'pydantic_model' by default")