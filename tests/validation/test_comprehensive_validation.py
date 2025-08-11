#!/usr/bin/env python3
"""Test comprehensive validation and setup."""

from pydantic import BaseModel, Field
from haive.core.engine.aug_llm import AugLLMConfig

class ResponseModel(BaseModel):
    """Structured output model."""
    answer: str = Field(description="The answer")
    confidence: float = Field(description="Confidence score")

print("="*60)
print("COMPREHENSIVE VALIDATION TEST")
print("="*60)

config = AugLLMConfig()
print(f"Initial routes: {config.tool_routes}")

config.with_structured_output(ResponseModel, version="v2")
print(f"After with_structured_output: {config.tool_routes}")

# Check if comprehensive_validation_and_setup was called automatically
print(f"Tools in config: {config.tools}")
print(f"structured_output_model: {config.structured_output_model}")

# Try calling it manually
print(f"\nCalling comprehensive_validation_and_setup manually...")
config.comprehensive_validation_and_setup()
print(f"After comprehensive_validation_and_setup: {config.tool_routes}")

print("\n" + "="*60)
print("COMPREHENSIVE VALIDATION TEST COMPLETE")
print("="*60)