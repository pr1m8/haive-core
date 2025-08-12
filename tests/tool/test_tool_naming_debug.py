#!/usr/bin/env python3
"""Debug tool naming in mixed tools."""

from pydantic import BaseModel

from haive.core.engine.aug_llm import AugLLMConfig


class ExecutableModel(BaseModel):
    """BaseModel with __call__."""
    multiplier: int = 2

    def __call__(self, value: int) -> int:
        return value * self.multiplier

class NonExecutableModel(BaseModel):
    """BaseModel without __call__."""
    name: str = "test"

print("="*60)
print("TOOL NAMING DEBUG")
print("="*60)

# Test mixed tools
instance = ExecutableModel(multiplier=5)
config = AugLLMConfig(tools=[ExecutableModel, instance, NonExecutableModel])

print("All tools in config.tools:")
for i, tool in enumerate(config.tools):
    print(f"  {i}: {tool} (type: {type(tool)})")
    if hasattr(tool, "__name__"):
        print(f"     __name__: {tool.__name__}")

print("\nAll tool routes:")
for name, route in config.tool_routes.items():
    print(f"  {name}: {route}")

print("\n" + "="*60)
print("NAMING DEBUG COMPLETE")
print("="*60)
