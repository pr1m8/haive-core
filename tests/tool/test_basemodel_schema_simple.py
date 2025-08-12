#!/usr/bin/env python3
"""Simple test to show BaseModel tool schema behavior."""

import inspect

from langchain_core.tools import tool
from pydantic import BaseModel, Field


class ConfigurableTool(BaseModel):
    """A tool with configuration state."""
    prefix: str = Field(default="Result", description="Output prefix")
    suffix: str = Field(default="!", description="Output suffix")

    def __call__(self, query: str) -> str:
        """Process query with configuration."""
        return f"{self.prefix}: {query}{self.suffix}"

print("="*60)
print("BASEMODEL TOOL SCHEMA ANALYSIS")
print("="*60)

# Create two instances with different configs
tool1 = ConfigurableTool(prefix="Answer", suffix=".")
tool2 = ConfigurableTool(prefix="Response", suffix="!!!")

print("\n1️⃣ BaseModel Schema (configuration fields):")
print(f"Fields: {list(ConfigurableTool.model_fields.keys())}")
print("Schema includes: prefix, suffix")

print("\n2️⃣ __call__ Method Signature (tool input):")
sig = inspect.signature(ConfigurableTool.__call__)
params = [p for p in sig.parameters.keys() if p != "self"]
print(f"Parameters: {params}")
print("Schema includes: query")

print("\n3️⃣ Instance State:")
print(f"tool1.prefix = '{tool1.prefix}', tool1.suffix = '{tool1.suffix}'")
print(f"tool2.prefix = '{tool2.prefix}', tool2.suffix = '{tool2.suffix}'")

print("\n4️⃣ Execution Results:")
print(f"tool1('Hello') = '{tool1('Hello')}'")
print(f"tool2('Hello') = '{tool2('Hello')}'")

print("\n5️⃣ Manual Tool Conversion:")
# Convert to LangChain tools manually
@tool
def wrapped_tool1(query: str) -> str:
    """Tool 1 with custom config."""
    return tool1(query)

@tool
def wrapped_tool2(query: str) -> str:
    """Tool 2 with custom config."""
    return tool2(query)

print(f"\nwrapped_tool1 schema: {wrapped_tool1.args_schema.model_fields.keys()}")
print(f"wrapped_tool1('Test') = '{wrapped_tool1.invoke({'query': 'Test'})}'")
print(f"wrapped_tool2('Test') = '{wrapped_tool2.invoke({'query': 'Test'})}'")

print("\n" + "="*60)
print("KEY INSIGHTS:")
print("="*60)
print("1. BaseModel fields (prefix, suffix) = CONFIGURATION/STATE")
print("2. __call__ parameters (query) = TOOL INPUT SCHEMA")
print("3. Each instance preserves its own state")
print("4. Tool schema only includes __call__ params, not BaseModel fields")
print("5. This is why 'pydantic_tool' route might be useful - to track stateful tools")
