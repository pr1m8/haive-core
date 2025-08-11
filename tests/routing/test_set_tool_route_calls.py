#!/usr/bin/env python3
"""Test if set_tool_route is being called in structured output mixin."""

from pydantic import BaseModel, Field
from haive.core.engine.aug_llm import AugLLMConfig

class ResponseModel(BaseModel):
    """Test response model."""
    answer: str = Field(description="The answer")
    confidence: float = Field(description="Confidence score")

print("="*60)
print("SET_TOOL_ROUTE CALL TEST")
print("="*60)

# Monkey patch set_tool_route to see if it gets called
original_set_tool_route = None

class TrackedConfig(AugLLMConfig):
    def set_tool_route(self, name, route, metadata=None):
        print(f"🚨 set_tool_route called: name='{name}', route='{route}', metadata={metadata}")
        return super().set_tool_route(name, route, metadata)

print("\n1️⃣ Testing with tracked config")
config = TrackedConfig()

print(f"   Calling with_structured_output...")
config.with_structured_output(ResponseModel, version="v2")

print(f"\n   Tool routes after with_structured_output: {config.tool_routes}")

print(f"\n   Manually calling _sync_tool_routes...")
config._sync_tool_routes()

print(f"   Tool routes after _sync_tool_routes: {config.tool_routes}")

# Also test the _mark_structured_output_tools method
print(f"\n2️⃣ Testing _mark_structured_output_tools explicitly")
config2 = TrackedConfig()
config2.structured_output_model = ResponseModel
config2.structured_output_version = "v2"
config2.tools = [ResponseModel]

print(f"   Calling _mark_structured_output_tools...")
config2._mark_structured_output_tools()

print(f"   Tool routes after _mark_structured_output_tools: {config2.tool_routes}")

print("\n" + "="*60)
print("SET_TOOL_ROUTE TEST COMPLETE")  
print("="*60)