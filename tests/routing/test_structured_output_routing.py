#!/usr/bin/env python3
"""Test structured output routing patterns."""

from pydantic import BaseModel, Field

from haive.core.engine.aug_llm import AugLLMConfig


class ResponseModel(BaseModel):
    """Test response model."""
    answer: str = Field(description="The answer")
    confidence: float = Field(description="Confidence score")

print("="*60)
print("STRUCTURED OUTPUT ROUTING TEST")
print("="*60)

# Test structured output model routing
print("\n1️⃣ Testing structured output model")
try:
    config = AugLLMConfig(structured_output_model=ResponseModel)
    print("✅ SUCCESS! Config created")
    print(f"   Tool routes: {config.tool_routes}")

    # Check what route the structured output model gets
    for tool_name, route in config.tool_routes.items():
        if "ResponseModel" in tool_name or "response" in tool_name.lower():
            print(f"   ResponseModel route: {route}")

except Exception as e:
    print(f"❌ Failed: {e}")
    import traceback
    traceback.print_exc()

# Test the structured output mixin
print("\n2️⃣ Testing with_structured_output method")
try:
    config2 = AugLLMConfig()
    config2.with_structured_output(ResponseModel, version="v2")
    print("✅ SUCCESS! with_structured_output worked")
    print(f"   Tool routes: {config2.tool_routes}")

    # Check routes
    for tool_name, route in config2.tool_routes.items():
        print(f"   {tool_name}: {route}")

except Exception as e:
    print(f"❌ Failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("ROUTING ANALYSIS COMPLETE")
print("="*60)
