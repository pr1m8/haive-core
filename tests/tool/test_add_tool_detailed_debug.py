#!/usr/bin/env python3
"""Detailed debug of add_tool method."""

from pydantic import BaseModel, Field
from haive.core.engine.aug_llm import AugLLMConfig

class ResponseModel(BaseModel):
    """Structured output model."""
    answer: str = Field(description="The answer")
    confidence: float = Field(description="Confidence score")

class TrackedConfig(AugLLMConfig):
    def add_tool(self, tool, name=None, route=None):
        print(f"\n🔧 add_tool called:")
        print(f"   tool: {tool}")
        print(f"   name: {name}")
        print(f"   route: {route}")
        print(f"   tool in self.tools: {tool in self.tools}")
        print(f"   current tools: {self.tools}")
        print(f"   current tool_routes: {self.tool_routes}")
        
        result = super().add_tool(tool, name, route)
        
        print(f"   After add_tool:")
        print(f"   tools: {self.tools}")
        print(f"   tool_routes: {self.tool_routes}")
        return result
        
    def _sync_tool_routes(self):
        print(f"🔄 _sync_tool_routes called")
        print(f"   tools before: {len(self.tools)} tools")
        result = super()._sync_tool_routes()
        print(f"   tool_routes after: {self.tool_routes}")
        return result

print("="*60)
print("DETAILED ADD_TOOL DEBUG")
print("="*60)

config = TrackedConfig()
config.with_structured_output(ResponseModel, version="v2")

print("\n" + "="*60)
print("DETAILED ADD_TOOL DEBUG COMPLETE")
print("="*60)