#!/usr/bin/env python3
"""Debug with_structured_output method."""

from pydantic import BaseModel, Field
from haive.core.engine.aug_llm import AugLLMConfig

class ResponseModel(BaseModel):
    """Structured output model."""
    answer: str = Field(description="The answer")
    confidence: float = Field(description="Confidence score")

print("="*60)
print("WITH_STRUCTURED_OUTPUT DEBUG")
print("="*60)

config = AugLLMConfig()
print(f"Initial routes: {config.tool_routes}")

config.with_structured_output(ResponseModel, version="v2")
print(f"After with_structured_output: {config.tool_routes}")

# The issue might be that _sync_tool_routes isn't called automatically
# Let's call it manually
config._sync_tool_routes()
print(f"After manual _sync_tool_routes: {config.tool_routes}")

print("\n" + "="*60)
print("WITH_STRUCTURED_OUTPUT DEBUG COMPLETE")
print("="*60)