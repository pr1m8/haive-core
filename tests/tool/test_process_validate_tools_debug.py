#!/usr/bin/env python3
"""Debug _process_and_validate_tools method."""

from pydantic import BaseModel, Field

from haive.core.engine.aug_llm import AugLLMConfig


class ResponseModel(BaseModel):
    """Structured output model."""
    answer: str = Field(description="The answer")
    confidence: float = Field(description="Confidence score")

class TrackedConfig(AugLLMConfig):
    def _process_and_validate_tools(self):
        print("🔧 _process_and_validate_tools called")
        print(f"   tools: {self.tools}")
        result = super()._process_and_validate_tools()
        print(f"   tool_routes after: {self.tool_routes}")
        return result

    def add_tool(self, tool, name=None, route=None, **metadata):
        print(f"➕ add_tool called: {tool}")
        result = super().add_tool(tool, name, route, **metadata)
        print(f"   tool_routes after add_tool: {self.tool_routes}")
        return result

print("="*60)
print("PROCESS AND VALIDATE TOOLS DEBUG")
print("="*60)

config = TrackedConfig()
config.with_structured_output(ResponseModel, version="v2")

print("\nCalling comprehensive_validation_and_setup...")
config.comprehensive_validation_and_setup()

print("\n" + "="*60)
print("PROCESS AND VALIDATE TOOLS DEBUG COMPLETE")
print("="*60)
