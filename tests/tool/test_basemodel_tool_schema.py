#!/usr/bin/env python3
"""Trace BaseModel tool schema creation."""

from pydantic import BaseModel, Field
from haive.core.engine.tool import ToolEngine
from langchain_core.tools import tool
import json

class MyTool(BaseModel):
    """A configurable tool."""
    config: str = Field(default="default", description="Tool configuration")
    threshold: float = Field(default=0.5, description="Threshold value")
    
    def __call__(self, query: str) -> str:
        """Process query with configuration."""
        return f"{self.config}: {query} (threshold={self.threshold})"

print("="*60)
print("TRACING TOOL SCHEMA CREATION")
print("="*60)

# Test 1: BaseModel CLASS
print("\n1️⃣ BaseModel as CLASS")
try:
    engine1 = ToolEngine(tools=[MyTool])
    tools1 = engine1.get_tools()
    print(f"Number of tools: {len(tools1)}")
    if tools1:
        tool1 = tools1[0]
        print(f"Tool type: {type(tool1)}")
        print(f"Tool name: {tool1.name}")
        print(f"Input schema: {tool1.args_schema.schema() if hasattr(tool1, 'args_schema') else 'No schema'}")
except Exception as e:
    print(f"Error with class: {e}")

# Test 2: BaseModel INSTANCE
print("\n\n2️⃣ BaseModel as INSTANCE")
try:
    instance = MyTool(config="custom", threshold=0.8)
    print(f"Instance callable: {callable(instance)}")
    print(f"Instance __call__ exists: {hasattr(instance, '__call__')}")
    
    engine2 = ToolEngine(tools=[instance])
    tools2 = engine2.get_tools()
    print(f"Number of tools: {len(tools2)}")
    if tools2:
        tool2 = tools2[0]
        print(f"Tool type: {type(tool2)}")
        print(f"Tool name: {tool2.name}")
        print(f"Input schema: {tool2.args_schema.schema() if hasattr(tool2, 'args_schema') else 'No schema'}")
        
        # Try to call it
        result = tool2.invoke({"query": "test input"})
        print(f"Tool result: {result}")
except Exception as e:
    print(f"Error with instance: {e}")

# Test 3: Compare manual @tool conversion
print("\n\n3️⃣ Manual @tool conversion")
@tool
def manual_tool(query: str) -> str:
    """Manual tool from instance."""
    return instance(query)

print(f"Manual tool name: {manual_tool.name}")
print(f"Manual tool schema: {manual_tool.args_schema.schema()}")

# Test 4: What about the BaseModel schema itself?
print("\n\n4️⃣ BaseModel schemas")
print(f"MyTool model schema (fields): {json.dumps(MyTool.model_json_schema(), indent=2)}")
print(f"\nInstance values: config='{instance.config}', threshold={instance.threshold}")

# Test 5: Direct __call__ signature
print("\n\n5️⃣ __call__ signature analysis")
import inspect
sig = inspect.signature(MyTool.__call__)
print(f"__call__ signature: {sig}")
print(f"__call__ parameters: {list(sig.parameters.keys())}")