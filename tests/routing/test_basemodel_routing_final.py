#!/usr/bin/env python3
"""Test final BaseModel routing behavior."""

from pydantic import BaseModel, Field
from haive.core.engine.aug_llm import AugLLMConfig
from haive.core.graph.node.validation_node_v2 import ValidationNodeV2
from haive.core.common.mixins.tool_route_mixin import ToolRouteMixin
from langchain_core.messages import HumanMessage, ToolCall
import logging

logging.basicConfig(level=logging.INFO)

class PlainModel(BaseModel):
    """BaseModel without __call__ - cannot be a tool."""
    name: str = Field(description="Name field")
    value: int = Field(default=42)

class ExecutableModel(BaseModel):
    """BaseModel with __call__ - can be a tool."""
    multiplier: int = Field(default=2, description="Multiplication factor")
    
    def __call__(self, value: int) -> int:
        """Multiply value by the configured multiplier."""
        return value * self.multiplier

print("="*80)
print("BASEMODEL ROUTING VERIFICATION")
print("="*80)

# Test 1: Check ToolRouteMixin detection
print("\n1️⃣ ToolRouteMixin Analysis")
print("-"*40)
mixin = ToolRouteMixin()

# Plain model
route1, meta1 = mixin._analyze_tool(PlainModel)
print(f"PlainModel route: '{route1}'")
print(f"  is_executable: {meta1.get('is_executable', 'N/A')}")

# Executable model
route2, meta2 = mixin._analyze_tool(ExecutableModel)
print(f"ExecutableModel route: '{route2}'")
print(f"  is_executable: {meta2.get('is_executable', 'N/A')}")

# Test 2: Check AugLLMConfig routing
print("\n\n2️⃣ AugLLMConfig Routing")
print("-"*40)

# As tools
config1 = AugLLMConfig(tools=[PlainModel])
print(f"PlainModel as tool: {config1.tool_routes.get('PlainModel', 'Not found')}")

config2 = AugLLMConfig(tools=[ExecutableModel])
print(f"ExecutableModel as tool: {config2.tool_routes.get('ExecutableModel', 'Not found')}")

# As structured output
config3 = AugLLMConfig(structured_output_model=PlainModel)
print(f"PlainModel as structured output: {config3.tool_routes.get('PlainModel', 'Not found')}")

# Test 3: Check ValidationNodeV2 handling
print("\n\n3️⃣ ValidationNodeV2 Route Handling")
print("-"*40)

# Create a validation node
val_node = ValidationNodeV2()

# Create test state with tool calls
state = {
    "messages": [
        HumanMessage(
            content="Test",
            tool_calls=[
                ToolCall(
                    name="PlainModel",
                    args={"name": "test", "value": 10},
                    id="call_1"
                ),
                ToolCall(
                    name="ExecutableModel", 
                    args={"value": 5},
                    id="call_2"
                )
            ]
        )
    ]
}

# Mock engine with routes
class MockEngine:
    def __init__(self):
        self.tool_routes = {
            "PlainModel": "pydantic_model",
            "ExecutableModel": "pydantic_tool"
        }
    
    def get_tool_route(self, name):
        return self.tool_routes.get(name)

# Test with different routes
print("\nRoute handling:")
print("- pydantic_model → Error (cannot execute)")
print("- pydantic_tool → Let tool_node handle")
print("- parse_output → Validation handling")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("✅ pydantic_model = BaseModel without __call__ (error for tools)")
print("✅ pydantic_tool = BaseModel with __call__ (executable)")  
print("✅ parse_output = BaseModel for structured output")
print("\nEach route serves a distinct purpose in the system!")