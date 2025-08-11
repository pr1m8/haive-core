#!/usr/bin/env python3
"""Detailed test of structured output mixin routing."""

from pydantic import BaseModel, Field
from haive.core.engine.aug_llm import AugLLMConfig

class ResponseModel(BaseModel):
    """Test response model."""
    answer: str = Field(description="The answer")
    confidence: float = Field(description="Confidence score")

print("="*60)
print("DETAILED STRUCTURED OUTPUT MIXIN TEST")
print("="*60)

# Test the structured output mixin in detail
print("\n1️⃣ Testing with_structured_output method step by step")
config = AugLLMConfig()

print(f"   Initial state:")
print(f"   - tools: {config.tools}")
print(f"   - tool_routes: {config.tool_routes}")
print(f"   - structured_output_model: {config.structured_output_model}")

print(f"\n   Calling with_structured_output...")
config.with_structured_output(ResponseModel, version="v2")

print(f"   After with_structured_output:")
print(f"   - tools: {config.tools}")
print(f"   - tool_routes: {config.tool_routes}")
print(f"   - structured_output_model: {config.structured_output_model}")
print(f"   - force_tool_use: {config.force_tool_use}")
print(f"   - force_tool_choice: {config.force_tool_choice}")

# Check if set_tool_route was called
print(f"\n   Checking _tool_name_mapping: {config._tool_name_mapping}")

# Let's see what routes it would set if we call _sync_tool_routes manually
print(f"\n   Manually calling _sync_tool_routes...")
config._sync_tool_routes()
print(f"   After _sync_tool_routes:")
print(f"   - tool_routes: {config.tool_routes}")

# Check if the ResponseModel is in tools
print(f"\n   Checking if ResponseModel is in tools:")
for i, tool in enumerate(config.tools):
    print(f"   - Tool {i}: {tool} (type: {type(tool)})")
    if hasattr(tool, '__name__'):
        print(f"     - __name__: {tool.__name__}")

print("\n" + "="*60)
print("DETAILED TEST COMPLETE")
print("="*60)